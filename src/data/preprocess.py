"""Preprocesado de imágenes: canal "p1" (réplica del artículo 1) y canal "homogeneo" (P1 a P6).

Canal "p1" (docs/00_inventario.md, Section 4.1; covid_official.ipynb, celda 1, `_decode`):
    tf.io.read_file -> tf.image.decode_image(channels=3, expand_animations=False) -> tf.image.resize((224, 224))
    bilineal, sin antialias y con half_pixel_centers (comportamiento de TF2), sin conservar la relación de aspecto,
    salida float32 en [0, 255]. `convnext.preprocess_input` es la identidad: la normalización de ImageNet la hace
    la capa PreStem dentro del modelo (aquí, Paper1ConvNeXt con input_scale=255).
    Equivalencias con TF de la decodificación con PIL:
      - gris (L) y gris con alfa (LA): se replica la luminancia en 3 canales (libpng gray_to_rgb, alfa descartado);
      - RGBA: se descarta el alfa sin componer (TF decode_png con channels=3 hace strip_alpha);
      - paleta (P): se expande a RGB;
      - PNG de 16 bits (I;16, I): TF devuelve uint8 con png_set_strip_16, es decir, el byte alto (v >> 8);
      - JPEG: TF usa libjpeg-turbo con DCT entera rápida (JDCT_IFAST) por defecto y PIL la DCT entera precisa
        (JDCT_ISLOW). Diferencias de 1 o 2 niveles en algunos píxeles. No afecta al test oficial (todo PNG de 8 bits:
        bimcv, rsna y ricord, comprobado), sí a parte del train (stonybrook, cohen, fig1). Documentado como
        discrepancia residual conocida.
    El redimensionado se hace con torch.nn.functional.interpolate(mode="bilinear", align_corners=False,
    antialias=False), que usa la misma fórmula de coordenadas que TF (centro de píxel en +0.5).

Canal "homogeneo" (docs/03_arquitectura.md, Section 2.1):
    P1 luminancia en [0, 1] según la profundidad de bits (RGB a L con ITU-R 601; RGBA compuesto sobre negro);
    P2 recorte iterativo de filas y columnas exteriores con desviación típica < 2/255;
    P3 recorte cuadrado centrado en la caja de la máscara pulmonar (lado = mayor lado de la caja + 10 % por cada
       lado, limitado a la imagen) si hay máscara y pasa el control de calidad; si no, recorte central cuadrado
       de lado min(H, W) (respaldo documentado sin máscara);
    P4 reducción con antialias (bicúbica de PIL, que ensancha el núcleo al reducir) a R_c y después a R
       (R = 224 -> R_c = 256; R = 384 -> R_c = 384);
    P5 recorte a los percentiles 0.5 y 99.5 de la propia imagen y reescalado a [0, 1], sin CLAHE;
    P6 almacenamiento en PNG de 8 bits sin pérdida (lo hace cache.py). La réplica a 3 canales y la
       normalización de ImageNet se aplican al cargar (datasets.py y el modelo).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

P1_SIZE = 224


# ====================================================================== canal p1
def decode_like_tf(path) -> np.ndarray:
    """Devuelve uint8 H x W x 3 como tf.image.decode_image(channels=3) para PNG y JPEG de 8 o 16 bits."""
    with Image.open(path) as im:
        im.load()
        mode = im.mode
        if mode in ("I;16", "I;16B", "I;16L", "I"):
            a = np.asarray(im)
            if a.dtype != np.uint16:
                a = np.clip(a, 0, 65535).astype(np.uint16)
            g = (a >> 8).astype(np.uint8)
            return np.repeat(g[..., None], 3, axis=2)
        if mode == "L":
            g = np.asarray(im, dtype=np.uint8)
            return np.repeat(g[..., None], 3, axis=2)
        if mode == "LA":
            g = np.asarray(im.getchannel(0), dtype=np.uint8)
            return np.repeat(g[..., None], 3, axis=2)
        if mode == "1":
            g = np.asarray(im.convert("L"), dtype=np.uint8)
            return np.repeat(g[..., None], 3, axis=2)
        if mode == "RGBA":
            return np.asarray(im, dtype=np.uint8)[..., :3].copy()
        if mode == "RGB":
            return np.asarray(im, dtype=np.uint8)
        if mode in ("P", "PA"):
            return np.asarray(im.convert("RGBA"), dtype=np.uint8)[..., :3].copy()
        if mode == "CMYK":      # libjpeg convierte CMYK a RGB; PIL hace lo mismo con convert
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def resize_bilinear_tf(img: np.ndarray, size: int = P1_SIZE) -> np.ndarray:
    """tf.image.resize(img, (size, size)) bilineal (antialias=False). Entrada H x W x C; salida float32 C x size x size."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    if t.shape[-2:] == (size, size):
        return t[0].numpy()
    out = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False, antialias=False)
    return out[0].numpy()


def preprocess_p1(path, size: int = P1_SIZE) -> np.ndarray:
    """Canal p1: float32 3 x size x size en [0, 255], igual que _decode del artículo 1."""
    return resize_bilinear_tf(decode_like_tf(path), size)


# ====================================================================== canal homogeneo
def load_luminance(path) -> np.ndarray:
    """P1: luminancia float64 en [0, 1] según la profundidad de bits."""
    with Image.open(path) as im:
        im.load()
        mode = im.mode
        if mode in ("I;16", "I;16B", "I;16L"):
            return np.asarray(im, dtype=np.float64) / 65535.0
        if mode == "I":
            a = np.asarray(im, dtype=np.float64)
            return a / (65535.0 if a.max() > 255 else 255.0)
        if mode == "F":
            a = np.asarray(im, dtype=np.float64)
            return (a - a.min()) / max(a.max() - a.min(), 1e-12)
        if mode == "L":
            return np.asarray(im, dtype=np.float64) / 255.0
        if mode == "1":
            return np.asarray(im.convert("L"), dtype=np.float64) / 255.0
        if mode in ("LA", "PA", "P", "RGBA", "RGB", "CMYK", "YCbCr"):
            if mode in ("LA",):
                a = np.asarray(im, dtype=np.float64) / 255.0
                return a[..., 0] * a[..., 1]
            rgba = np.asarray(im.convert("RGBA"), dtype=np.float64) / 255.0 if mode != "RGB" else None
            if rgba is None:
                rgb = np.asarray(im, dtype=np.float64) / 255.0
                alpha = 1.0
            else:
                rgb, alpha = rgba[..., :3], rgba[..., 3]
            lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            return lum * alpha
        return np.asarray(im.convert("L"), dtype=np.float64) / 255.0


def trim_borders(x: np.ndarray, std_thr: float = 2.0 / 255.0, min_side: int = 32):
    """P2: quita filas y columnas exteriores con std < umbral, repitiendo hasta que ninguna lo cumpla.
    Devuelve (imagen recortada, (top, bottom, left, right)) con bottom y right exclusivos."""
    t, b, l, r = 0, x.shape[0], 0, x.shape[1]
    changed = True
    while changed:
        changed = False
        if b - t > min_side:
            if x[t, l:r].std() < std_thr:
                t += 1; changed = True
            if b - t > min_side and x[b - 1, l:r].std() < std_thr:
                b -= 1; changed = True
        if r - l > min_side:
            if x[t:b, l].std() < std_thr:
                l += 1; changed = True
            if r - l > min_side and x[t:b, r - 1].std() < std_thr:
                r -= 1; changed = True
    return x[t:b, l:r], (t, b, l, r)


def mask_qc(mask: np.ndarray | None) -> bool:
    """Control de calidad mínimo de la máscara pulmonar: área entre el 5 y el 80 % de la imagen."""
    if mask is None:
        return False
    m = np.asarray(mask) > 0
    if m.ndim != 2 or not m.any():
        return False
    frac = m.mean()
    return 0.05 <= frac <= 0.80


def square_crop_box(shape, mask: np.ndarray | None = None, margin: float = 0.10):
    """P3: caja cuadrada (top, left, side). Con máscara válida, centrada en su caja envolvente; si no, central."""
    h, w = shape
    if mask is not None and mask_qc(mask):
        ys, xs = np.nonzero(np.asarray(mask) > 0)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        side = int(round(max(y1 - y0, x1 - x0) * (1.0 + 2.0 * margin)))
        side = max(1, min(side, h, w))
        cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
        top = int(round(cy - side / 2.0)); left = int(round(cx - side / 2.0))
        top = min(max(top, 0), h - side); left = min(max(left, 0), w - side)
        return top, left, side, "mascara"
    side = min(h, w)
    return (h - side) // 2, (w - side) // 2, side, "central"


def resize_antialias(x: np.ndarray, size: int) -> np.ndarray:
    """P4: bicúbica de PIL en modo F (con filtro previo de antialias al reducir)."""
    if x.shape == (size, size):
        return x.astype(np.float32)
    im = Image.fromarray(x.astype(np.float32), mode="F")
    return np.asarray(im.resize((size, size), resample=Image.BICUBIC), dtype=np.float32)


def intermediate_side(res: int) -> int:
    return 256 if res == 224 else res


def percentile_rescale(x: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    """P5: recorte a percentiles de la propia imagen y reescalado a [0, 1]."""
    a, b = np.percentile(x, [lo, hi])
    if b - a < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - a) / (b - a), 0.0, 1.0).astype(np.float32)


def preprocess_homogeneo(path, res: int = 224, mask: np.ndarray | None = None, return_meta: bool = False):
    """P1 a P5. Devuelve float32 res x res en [0, 1] (y metadatos si return_meta). La máscara, si se da, debe
    estar en la rejilla de la imagen original."""
    x = load_luminance(path)
    h0, w0 = x.shape
    xt, (t, b, l, r) = trim_borders(x)
    m = None
    if mask is not None:
        m = np.asarray(mask)
        if m.shape != (h0, w0):
            m = np.asarray(Image.fromarray((m > 0).astype(np.uint8) * 255).resize((w0, h0), Image.NEAREST))
        m = m[t:b, l:r]
    top, left, side, how = square_crop_box(xt.shape, m)
    xc = xt[top:top + side, left:left + side]
    rc = intermediate_side(res)
    y = resize_antialias(resize_antialias(xc, rc), res) if rc != res else resize_antialias(xc, res)
    y = percentile_rescale(y)
    if not return_meta:
        return y
    meta = {"orig_h": h0, "orig_w": w0, "trim_top": t, "trim_bottom": h0 - b, "trim_left": l, "trim_right": w0 - r,
            "trim_frac": 1.0 - (b - t) * (r - l) / float(h0 * w0), "crop_side": side, "crop_mode": how,
            "orig_aspect": w0 / float(h0)}
    return y, meta


def to_uint8_png_array(y: np.ndarray) -> np.ndarray:
    """P6: cuantización a 8 bits para PNG sin pérdida."""
    return np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8)


CHANNELS = ("p1", "homogeneo")


def preprocess(path, channel: str, res: int = 224, mask=None):
    if channel == "p1":
        return preprocess_p1(path, res)
    if channel == "homogeneo":
        return preprocess_homogeneo(path, res, mask)
    raise ValueError("canal desconocido: %s" % channel)
