"""Dataset de PyTorch, aumentos y muestreador equilibrado.

- `CXRDataset`: devuelve un dict con image (float32 3 x R x R; canal p1 en [0, 255], canal homogeneo en
  [0, 1]), label, source (índice estable de src/data/manifest.py), patient, image_id e index. Lee de la caché si
  existe la entrada y si no preprocesa al vuelo.
- `Paper1Augment`: réplica del aumento del artículo 1 (covid_official.ipynb, celda 1, build_train_dataset):
  RandomFlip("horizontal") -> RandomRotation(0.03) -> RandomZoom(height_factor=0.1) por imagen, con relleno
  "reflect" e interpolación bilineal como ImageProjectiveTransformV3; después, sobre el lote, brillo aditivo
  uniforme en [-0.1, 0.1] (unidades de la entrada 0..255) y contraste en [0.9, 1.1] (un único valor por lote,
  porque tf.image.random_brightness y random_contrast se aplicaban al tensor del lote), cutout 24 x 24 por imagen
  con la fórmula exacta de `cutout` y mixup con un único lambda ~ Beta(0.2, 0.2) por lote y permutación del lote.
  Se aplica en el dispositivo, por lotes, con un generador sembrado por (semilla, paso): reproducible y reanudable.
- `NewAugment`: mismo aumento geométrico y fotométrico, sin mixup (rompe la etiqueta de fuente), con brillo y
  contraste por imagen y en unidades del rango [0, 1]; transforma igual una máscara opcional.
- `BalancedInfiniteBatchSampler`: flujo infinito con probabilidad 0.5 por clase en cada elemento (como
  tf.data.Dataset.sample_from_datasets(weights=[0.5, 0.5])) o exactamente 50/50 por lote (modo "exact"). Cada
  clase recorre permutaciones completas sucesivas (el artículo 1 usaba un búfer de barajado de 8 192, que es una
  permutación parcial; la diferencia es solo de orden). El lote b es función determinista de (semilla, b), así que
  la reanudación desde el paso b es exacta aunque el DataLoader haya precargado lotes.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .cache import decode_file, read_index
from .paths import resolve_data
from .preprocess import preprocess_homogeneo, preprocess_p1


class CXRDataset(Dataset):
    def __init__(self, df, channel: str = "p1", res: int = 224, cache_dir=None, require_cache: bool = False):
        self.df = df.reset_index(drop=True)
        self.channel, self.res = channel, res
        self.paths = self.df["image_path"].tolist()
        self.ids = self.df["image_id"].tolist()
        self.labels = self.df["label"].astype("int64").to_numpy()
        self.sources = self.df["source_idx"].astype("int64").to_numpy()
        self.patients = self.df["patient_id"].astype(str).tolist()
        self.cache = {}
        if cache_dir is not None:
            try:
                self.cache = read_index(cache_dir)
            except FileNotFoundError:
                self.cache = {}
        if require_cache:
            miss = [i for i in self.ids if i not in self.cache]
            if miss:
                raise FileNotFoundError("faltan %d imágenes en la caché %s (p. ej. %s)" % (len(miss), cache_dir, miss[0]))

    def __len__(self):
        return len(self.df)

    def load_image(self, i) -> np.ndarray:
        iid = self.ids[i]
        if iid in self.cache:
            return decode_file(self.cache[iid], self.channel)
        p = resolve_data(self.paths[i])
        if self.channel == "p1":
            return preprocess_p1(p, self.res)
        y = preprocess_homogeneo(p, self.res)
        y = np.rint(y * 255.0) / 255.0      # igual que la caché PNG de 8 bits (P6)
        return np.repeat(y[None].astype(np.float32), 3, axis=0)

    def __getitem__(self, i):
        return {"image": torch.from_numpy(np.ascontiguousarray(self.load_image(i), dtype=np.float32)),
                "label": torch.tensor(float(self.labels[i])), "source": torch.tensor(int(self.sources[i])),
                "patient": self.patients[i], "image_id": self.ids[i], "index": torch.tensor(i)}


# ====================================================================== muestreador
class BalancedInfiniteBatchSampler(Sampler):
    def __init__(self, labels, batch_size: int, seed: int = 0, mode: str = "bernoulli", start_batch: int = 0):
        labels = np.asarray(labels).astype(int)
        self.idx = [np.nonzero(labels == c)[0] for c in (0, 1)]
        if any(len(v) == 0 for v in self.idx):
            raise ValueError("el muestreador equilibrado necesita las dos clases")
        if mode not in ("bernoulli", "exact"):
            raise ValueError(mode)
        self.B, self.seed, self.mode = int(batch_size), int(seed), mode
        self.start_batch = int(start_batch)
        self._perm_cache = {}

    def _bits(self, b):
        """Clase de cada posición del lote b (1 = positiva)."""
        if self.mode == "exact":
            half = self.B // 2
            bits = np.r_[np.ones(self.B - half, dtype=int), np.zeros(half, dtype=int)]
            return np.random.default_rng([self.seed, 7, b]).permutation(bits)
        return (np.random.default_rng([self.seed, 1, b]).random(self.B) < 0.5).astype(int)

    def _elem(self, c, k):
        n = len(self.idx[c])
        cyc, pos = divmod(int(k), n)
        key = (c, cyc)
        if key not in self._perm_cache:
            if len(self._perm_cache) > 8:
                self._perm_cache.clear()
            self._perm_cache[key] = np.random.default_rng([self.seed, 2, c, cyc]).permutation(n)
        return int(self.idx[c][self._perm_cache[key][pos]])

    def consumed_before(self, b):
        """Elementos de cada clase usados en los lotes 0..b-1."""
        if self.mode == "exact":
            n1 = (self.B - self.B // 2) * b
            return [self.B * b - n1, n1]
        n1 = 0
        for j in range(b):
            n1 += int(self._bits(j).sum())
        return [self.B * b - n1, n1]

    def batch(self, b):
        bits = self._bits(b)
        cons = self.consumed_before(b) if b != getattr(self, "_next_b", None) else self._cons
        out, k = [], list(cons)
        for c in bits:
            out.append(self._elem(c, k[c])); k[c] += 1
        self._next_b, self._cons = b + 1, k
        return out

    def set_start(self, b):
        self.start_batch = int(b)

    def __iter__(self):
        b = self.start_batch
        while True:
            yield self.batch(b)
            b += 1

    def __len__(self):           # flujo infinito: la longitud la fija steps_per_epoch del entrenador
        return 2 ** 31 - 1


# ====================================================================== aumentos
def _reflect_index(i, n):
    """Relleno 'reflect' de ImageProjectiveTransform (d c b a | a b c d | d c b a) sobre índices enteros."""
    p = 2 * n
    i = torch.remainder(i, p)
    return torch.where(i >= n, p - 1 - i, i)


def _sample_bilinear_reflect(x, xin, yin):
    """x: (B, C, H, W); xin, yin: (B, H, W) coordenadas de entrada (centros de píxel enteros)."""
    B, C, H, W = x.shape
    x0, y0 = torch.floor(xin), torch.floor(yin)
    wx, wy = (xin - x0).unsqueeze(1), (yin - y0).unsqueeze(1)
    x0, y0 = x0.long(), y0.long()
    flat = x.reshape(B, C, H * W)

    def g(yy, xx):
        yy, xx = _reflect_index(yy, H), _reflect_index(xx, W)
        lin = (yy * W + xx).reshape(B, 1, H * W).expand(B, C, H * W)
        return torch.gather(flat, 2, lin).reshape(B, C, H, W)

    v00, v01, v10, v11 = g(y0, x0), g(y0, x0 + 1), g(y0 + 1, x0), g(y0 + 1, x0 + 1)
    top = v00 + (v01 - v00) * wx
    bot = v10 + (v11 - v10) * wx
    return top + (bot - top) * wy


def _grid(B, H, W, device):
    ys = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1).expand(B, H, W)
    xs = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W).expand(B, H, W)
    return xs, ys


def rotate_batch(x, angles):
    """Rotación por imagen alrededor del centro ((W-1)/2, (H-1)/2), como get_rotation_matrix de Keras."""
    B, C, H, W = x.shape
    xs, ys = _grid(B, H, W, x.device)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    c, s = torch.cos(angles).view(B, 1, 1), torch.sin(angles).view(B, 1, 1)
    xin = c * (xs - cx) - s * (ys - cy) + cx
    yin = s * (xs - cx) + c * (ys - cy) + cy
    return _sample_bilinear_reflect(x, xin, yin)


def zoom_batch(x, zx, zy):
    """Zoom por imagen (get_zoom_matrix de Keras): x_in = zx * x + (W - 1) / 2 * (1 - zx)."""
    B, C, H, W = x.shape
    xs, ys = _grid(B, H, W, x.device)
    zx, zy = zx.view(B, 1, 1), zy.view(B, 1, 1)
    xin = zx * xs + (W - 1) / 2.0 * (1 - zx)
    yin = zy * ys + (H - 1) / 2.0 * (1 - zy)
    return _sample_bilinear_reflect(x, xin, yin)


def cutout_batch(x, cy, cx, mask_size=24):
    """Fórmula exacta de `cutout` del artículo 1: se anulan los píxeles con |y - cy| <= s/2 y |x - cx| <= s/2."""
    B, C, H, W = x.shape
    yy = torch.arange(H, device=x.device, dtype=torch.float32).view(1, 1, H, 1)
    xx = torch.arange(W, device=x.device, dtype=torch.float32).view(1, 1, 1, W)
    half = mask_size / 2.0
    keep = ((yy - cy.view(B, 1, 1, 1)).abs() > half) | ((xx - cx.view(B, 1, 1, 1)).abs() > half)
    return x * keep.to(x.dtype)


def _beta(gen, a, device):
    g1 = torch._standard_gamma(torch.tensor([a], device=device), generator=gen)
    g2 = torch._standard_gamma(torch.tensor([a], device=device), generator=gen)
    return (g1 / (g1 + g2)).squeeze(0)


def make_generator(seed: int, step: int, device="cpu"):
    g = torch.Generator(device=device)
    g.manual_seed((int(seed) * 1_000_003 + int(step) * 7919 + 12345) % (2 ** 63 - 1))
    return g


class Paper1Augment:
    def __init__(self, rotation=0.03, zoom=0.1, brightness=0.1, contrast=(0.9, 1.1), cutout=24, mixup_alpha=0.2,
                 flip=True):
        self.rotation, self.zoom, self.brightness = rotation, zoom, brightness
        self.contrast, self.cutout, self.mixup_alpha, self.flip = contrast, cutout, mixup_alpha, flip

    def geometric(self, x, gen, mask=None):
        B, C, H, W = x.shape
        dev = x.device
        if self.flip:
            f = torch.rand(B, generator=gen, device=dev) < 0.5
            x = torch.where(f.view(B, 1, 1, 1), x.flip(3), x)
            if mask is not None:
                mask = torch.where(f.view(B, 1, 1, 1), mask.flip(3), mask)
        if self.rotation:
            ang = (torch.rand(B, generator=gen, device=dev) * 2 - 1) * self.rotation * 2 * math.pi
            x = rotate_batch(x, ang)
            if mask is not None:
                mask = rotate_batch(mask, ang)
        if self.zoom:
            z = 1.0 + (torch.rand(B, generator=gen, device=dev) * 2 - 1) * self.zoom
            x = zoom_batch(x, z, z)
            if mask is not None:
                mask = zoom_batch(mask, z, z)
        return x, mask

    def photometric(self, x, gen, per_image=False):
        B = x.shape[0]
        n = B if per_image else 1
        dev = x.device
        if self.brightness:
            d = (torch.rand(n, generator=gen, device=dev) * 2 - 1) * self.brightness
            x = x + d.view(n, 1, 1, 1)
        if self.contrast:
            lo, hi = self.contrast
            f = lo + torch.rand(n, generator=gen, device=dev) * (hi - lo)
            m = x.mean(dim=(2, 3), keepdim=True)
            x = (x - m) * f.view(n, 1, 1, 1) + m
        return x

    def apply_cutout(self, x, gen):
        if not self.cutout:
            return x
        B, C, H, W = x.shape
        cy = torch.rand(B, generator=gen, device=x.device) * H
        cx = torch.rand(B, generator=gen, device=x.device) * W
        return cutout_batch(x, cy, cx, self.cutout)

    def mixup(self, x, y, gen):
        if not self.mixup_alpha:
            return x, y
        lam = _beta(gen, float(self.mixup_alpha), x.device)
        idx = torch.randperm(x.shape[0], generator=gen, device=x.device)
        return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]

    def __call__(self, x, y, gen, mask=None):
        x, mask = self.geometric(x.float(), gen, mask)
        x = self.photometric(x, gen, per_image=False)
        x = self.apply_cutout(x, gen)
        x, y = self.mixup(x, y.float(), gen)
        return (x, y) if mask is None else (x, y, mask)


class NewAugment(Paper1Augment):
    """Aumento del protocolo nuevo: sin mixup, brillo y contraste por imagen; entradas en [0, 1]."""

    def __init__(self, rotation=0.03, zoom=0.1, brightness=0.1, contrast=(0.9, 1.1), cutout=24, flip=True):
        super().__init__(rotation, zoom, brightness, contrast, cutout, mixup_alpha=0.0, flip=flip)

    def __call__(self, x, y, gen, mask=None):
        x, mask = self.geometric(x.float(), gen, mask)
        x = self.photometric(x, gen, per_image=True)
        x = self.apply_cutout(x, gen)
        return (x, y.float()) if mask is None else (x, y.float(), mask)


def build_augment(name: str, **kw):
    if name == "p1":
        return Paper1Augment(**kw)
    if name == "nuevo":
        kw.pop("mixup_alpha", None)
        return NewAugment(**kw)
    if name in ("none", None, ""):
        return None
    raise ValueError("aumento desconocido: %s" % name)
