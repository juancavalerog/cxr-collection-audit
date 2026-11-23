"""Capas de CSD-Net.

Contiene el frente fijo de alta frecuencia (wavelet de Haar), la construcción de la entrada del
codificador de adquisición con la región torácica enmascarada, el codificador de adquisición, la capa
de re-firma, la compuerta anatómica, el bloque Transformer con sesgo de atención y la cabeza de
segmentación. Las ecuaciones citadas remiten a docs/03_arquitectura.md.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------------
# Frente fijo de alta frecuencia
# ---------------------------------------------------------------------------------------------
class HaarHighPass(nn.Module):
    """Transformada de Haar 2D fija de dos niveles, sin parámetros entrenables (ecuación 14).

    Entrada: imagen en gris (B, 1, H, W) con H y W múltiplos de 4.
    Salida: (detalle, aproximación), con detalle (B, 6, H/2, W/2) formado por LH, HL y HH de los
    niveles 1 y 2 (el nivel 2 se repite por vecino más próximo) y aproximación LL1 (B, 1, H/2, W/2).
    """

    def __init__(self) -> None:
        super().__init__()
        ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) / 2.0
        lh = torch.tensor([[1.0, 1.0], [-1.0, -1.0]]) / 2.0
        hl = torch.tensor([[1.0, -1.0], [1.0, -1.0]]) / 2.0
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) / 2.0
        self.register_buffer("kernel", torch.stack([ll, lh, hl, hh]).unsqueeze(1), persistent=False)

    def _dwt(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel.to(dtype=x.dtype), stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c1 = self._dwt(x)
        ll1, d1 = c1[:, :1], c1[:, 1:]
        c2 = self._dwt(ll1)
        d2 = F.interpolate(c2[:, 1:], scale_factor=2.0, mode="nearest")
        return torch.cat([d1, d2], dim=1), ll1


def thoracic_exclusion(
    lung_mask: torch.Tensor,
    mode: str = "box",
    margin: float = 0.05,
    dilate_frac: float = 0.08,
    fallback: Tuple[float, float] = (0.10, 0.90),
) -> torch.Tensor:
    """Región que el codificador de adquisición no puede ver (1 = excluida), ecuación (15).

    mode="box": caja envolvente de la máscara pulmonar binarizada, ampliada en `margin` del lado.
    mode="dilate": máscara pulmonar dilatada con un núcleo de `dilate_frac` del lado.
    Si una imagen no tiene píxeles pulmonares se excluye una caja central fija (`fallback`).
    """
    b, _, h, w = lung_mask.shape
    m = (lung_mask > 0.5).to(lung_mask.dtype)
    if mode == "dilate":
        r = max(1, int(round(dilate_frac * h)))
        out = F.max_pool2d(m, kernel_size=2 * r + 1, stride=1, padding=r)
        empty = m.flatten(1).amax(1) == 0
        if bool(empty.any()):
            fb = torch.zeros_like(m[:1])
            y0, y1 = int(fallback[0] * h), int(fallback[1] * h)
            x0, x1 = int(fallback[0] * w), int(fallback[1] * w)
            fb[..., y0:y1, x0:x1] = 1.0
            out = torch.where(empty.view(b, 1, 1, 1), fb.expand_as(out), out)
        return out
    if mode != "box":
        raise ValueError(f"modo de exclusión desconocido: {mode}")
    rows = m.amax(dim=3).squeeze(1) > 0  # (B, H)
    cols = m.amax(dim=2).squeeze(1) > 0  # (B, W)
    iy = torch.arange(h, device=m.device).expand(b, h)
    ix = torch.arange(w, device=m.device).expand(b, w)
    y0 = torch.where(rows, iy, torch.full_like(iy, h)).amin(1)
    y1 = torch.where(rows, iy, torch.full_like(iy, -1)).amax(1)
    x0 = torch.where(cols, ix, torch.full_like(ix, w)).amin(1)
    x1 = torch.where(cols, ix, torch.full_like(ix, -1)).amax(1)
    empty = y1 < y0
    y0 = torch.where(empty, torch.full_like(y0, int(fallback[0] * h)), y0)
    y1 = torch.where(empty, torch.full_like(y1, int(fallback[1] * h) - 1), y1)
    x0 = torch.where(empty, torch.full_like(x0, int(fallback[0] * w)), x0)
    x1 = torch.where(empty, torch.full_like(x1, int(fallback[1] * w) - 1), x1)
    my, mx = int(round(margin * h)), int(round(margin * w))
    y0, y1 = (y0 - my).clamp(min=0), (y1 + my).clamp(max=h - 1)
    x0, x1 = (x0 - mx).clamp(min=0), (x1 + mx).clamp(max=w - 1)
    yy = torch.arange(h, device=m.device).view(1, h, 1)
    xx = torch.arange(w, device=m.device).view(1, 1, w)
    box = (yy >= y0.view(b, 1, 1)) & (yy <= y1.view(b, 1, 1)) & (xx >= x0.view(b, 1, 1)) & (xx <= x1.view(b, 1, 1))
    return box.unsqueeze(1).to(lung_mask.dtype)


def periphery_band(h: int, w: int, frac: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Banda periférica (1 en el anillo exterior de anchura `frac` del lado), forma (1, 1, h, w)."""
    by, bx = max(1, int(round(frac * h))), max(1, int(round(frac * w)))
    p = torch.ones(1, 1, h, w, device=device, dtype=dtype)
    p[..., by : h - by, bx : w - bx] = 0.0
    return p


class AcquisitionInput(nn.Module):
    """Entrada del codificador de adquisición, ecuación (16).

    x_acq = [ D(x) * (1 - E) , LL1(x) * P * (1 - E) ], donde D son las seis sub-bandas de detalle,
    P la banda periférica y E la región torácica excluida, todas a resolución H/2.
    Devuelve (x_acq, keep) con keep = 1 - E a resolución H/2.
    """

    def __init__(self, periphery_frac: float = 0.125, mode: str = "box", margin: float = 0.05,
                 dilate_frac: float = 0.08, exclude: bool = True, highpass: bool = True) -> None:
        super().__init__()
        self.haar = HaarHighPass()
        self.exclude = exclude
        self.highpass = highpass
        self.out_channels = 7 if highpass else 2
        self.periphery_frac = periphery_frac
        self.mode = mode
        self.margin = margin
        self.dilate_frac = dilate_frac

    def forward(self, gray: torch.Tensor, lung_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if lung_mask.shape[-2:] != gray.shape[-2:]:
            lung_mask = F.interpolate(lung_mask.float(), size=gray.shape[-2:], mode="nearest")
        detail, ll1 = self.haar(gray)
        if self.exclude:
            excl = thoracic_exclusion(lung_mask.to(gray.dtype), self.mode, self.margin, self.dilate_frac)
            keep = 1.0 - F.max_pool2d(excl, kernel_size=2, stride=2)  # un píxel torácico excluye su bloque 2 x 2
        else:
            keep = torch.ones_like(ll1)
        p = periphery_band(ll1.shape[-2], ll1.shape[-1], self.periphery_frac, gray.device, gray.dtype)
        if self.highpass:
            x_acq = torch.cat([detail * keep, ll1 * p * keep], dim=1)
        else:  # ablación sin alta frecuencia: intensidad fuera del tórax e intensidad periférica
            x_acq = torch.cat([ll1 * keep, ll1 * p * keep], dim=1)
        return x_acq, keep


class AcquisitionEncoder(nn.Module):
    """Codificador ligero de adquisición: cuatro etapas conv con GroupNorm y agregado de media y
    desviación restringido a la zona visible; produce la firma a de dimensión `sig_dim`, ecuación (17)."""

    def __init__(self, in_ch: int = 7, widths: Tuple[int, ...] = (32, 64, 128, 128), sig_dim: int = 64) -> None:
        super().__init__()
        layers = []
        c = in_ch
        for wd in widths:
            layers += [
                nn.Conv2d(c, wd, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, wd), nn.GELU(),
                nn.Conv2d(wd, wd, 3, padding=1, bias=False), nn.GroupNorm(8, wd), nn.GELU(),
            ]
            c = wd
        self.body = nn.Sequential(*layers)
        self.proj = nn.Linear(2 * c, sig_dim)
        self.norm = nn.LayerNorm(sig_dim)

    def forward(self, x_acq: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        f = self.body(x_acq)
        k = F.adaptive_avg_pool2d(keep.to(f.dtype), f.shape[-2:])
        mass = k.sum(dim=(2, 3)).clamp_min(1e-6)
        mean = (f * k).sum(dim=(2, 3)) / mass
        var = ((f - mean[..., None, None]) ** 2 * k).sum(dim=(2, 3)) / mass
        std = torch.sqrt(var + 1e-6)
        return self.norm(self.proj(torch.cat([mean, std], dim=1)))


# ---------------------------------------------------------------------------------------------
# Re-firma
# ---------------------------------------------------------------------------------------------
def instance_stats(feat: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Media y desviación por canal y por instancia, en float32, ecuación (18)."""
    f = feat.float()
    mu = f.mean(dim=(2, 3), keepdim=True)
    sd = torch.sqrt(f.var(dim=(2, 3), keepdim=True, unbiased=False) + eps)
    return mu, sd


class SignatureModulator(nn.Module):
    """MLP que predice desde la firma los estadísticos de canal de la etapa de inserción:
    (log-desviación, media) = (g(a), b(a)), ecuación (19)."""

    def __init__(self, sig_dim: int, channels: int, hidden: int = 256) -> None:
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(nn.Linear(sig_dim, hidden), nn.GELU(), nn.Linear(hidden, 2 * channels))

    def forward(self, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(a).float()
        return out[:, : self.channels], out[:, self.channels :]


class ReSignature(nn.Module):
    """Capa de re-firma, ecuaciones (20) a (22).

    mode="relative" (por defecto):
        F~ = s_i * exp(clamp(g_j - g_i)) * IN(F_i) + mu_i + (b_j - b_i)
        Con la firma propia (g_j = g_i, b_j = b_i) devuelve F_i salvo redondeo.
    mode="absolute" (ablación, AdaIN condicionado a la firma):
        F~ = exp(g_j) * IN(F_i) + b_j
    mode="mixstyle" (ablación sin codificador de adquisición; g y b son log-desviación y media reales):
        F~ = exp(l g_i + (1-l) g_j) * IN(F_i) + l b_i + (1-l) b_j, con l ~ Beta(alpha, alpha) en
        entrenamiento y l = 0 en evaluación.
    """

    def __init__(self, mode: str = "relative", log_scale_clamp: float = 2.0, eps: float = 1e-5,
                 mix_alpha: float = 0.1) -> None:
        super().__init__()
        if mode not in ("relative", "absolute", "mixstyle"):
            raise ValueError(f"modo de re-firma desconocido: {mode}")
        self.mode = mode
        self.clamp = log_scale_clamp
        self.eps = eps
        self.mix_alpha = mix_alpha

    def forward(self, feat: torch.Tensor, g_src: torch.Tensor, b_src: torch.Tensor, g_dst: torch.Tensor,
                b_dst: torch.Tensor) -> torch.Tensor:
        mu, sd = instance_stats(feat, self.eps)
        normed = (feat.float() - mu) / sd
        gs, bs = g_src[..., None, None], b_src[..., None, None]
        gd, bd = g_dst[..., None, None], b_dst[..., None, None]
        if self.mode == "relative":
            scale = torch.exp((gd - gs).clamp(-self.clamp, self.clamp))
            out = sd * scale * normed + mu + (bd - bs)
        elif self.mode == "absolute":
            out = torch.exp(gd.clamp(-self.clamp * 4, self.clamp * 4)) * normed + bd
        else:
            if self.training:
                lam = torch.distributions.Beta(self.mix_alpha, self.mix_alpha).sample((feat.shape[0], 1, 1, 1)).to(feat.device)
            else:
                lam = torch.zeros(feat.shape[0], 1, 1, 1, device=feat.device)
            out = torch.exp(lam * gs + (1 - lam) * gd) * normed + lam * bs + (1 - lam) * bd
        return out.to(feat.dtype)


# ---------------------------------------------------------------------------------------------
# Compuerta anatómica y Transformer
# ---------------------------------------------------------------------------------------------
class AnatomicalGate(nn.Module):
    """Compuerta anatómica, ecuaciones (23) a (25).

    w_t = clamp((m_t - tau) / (1 - tau), 0, 1), con m_t la probabilidad pulmonar media en la celda del
    token t. El agregado es z = sum_t w_t h_t / max(sum_t w_t, kappa): un token con w_t = 0 no
    contribuye a z. El sesgo de atención es log(max(w_t, floor)) sobre las claves.
    """

    def __init__(self, threshold: float = 0.1, attn_floor: float = 1e-4, min_mass: float = 1e-6) -> None:
        super().__init__()
        self.threshold = threshold
        self.attn_floor = attn_floor
        self.min_mass = min_mass

    def token_weights(self, mask_prob: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
        m = F.adaptive_avg_pool2d(mask_prob.float(), grid).flatten(1)
        return ((m - self.threshold) / (1.0 - self.threshold)).clamp(0.0, 1.0)

    def attn_bias(self, w: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return torch.log(w.clamp_min(self.attn_floor)).to(dtype)[:, None, None, :]

    def pool(self, tokens: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        wf = w.to(tokens.dtype).unsqueeze(-1)
        return (tokens * wf).sum(dim=1) / wf.sum(dim=1).clamp_min(self.min_mass)


class DropPath(nn.Module):
    """Profundidad estocástica por muestra."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class LayerScale(nn.Module):
    """Escala por canal aprendible."""

    def __init__(self, dim: int, init: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Attention(nn.Module):
    """Atención multicabeza con sesgo aditivo sobre las claves (nombres de parámetros como timm ViT)."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if bias is not None:
            bias = bias.to(q.dtype)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.proj(y.transpose(1, 2).reshape(b, n, c))


class Mlp(nn.Module):
    """MLP del bloque Transformer."""

    def __init__(self, dim: int, ratio: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim * ratio))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(dim * ratio), dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Bloque Transformer pre-norma con LayerScale; compatible por nombres con timm `vit_small`."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, drop_path: float = 0.0,
                 ls_init: float = 1.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.ls1 = LayerScale(dim, ls_init)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, mlp_ratio)
        self.ls2 = LayerScale(dim, ls_init)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), bias)))
        return x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))


class TransformerStage(nn.Module):
    """Etapa de atención global a paso 16: codificación posicional condicional (conv 3x3 en profundidad)
    y `depth` bloques; la máscara anatómica entra como sesgo de atención, ecuación (24)."""

    def __init__(self, dim: int = 384, depth: int = 4, heads: int = 6, mlp_ratio: float = 4.0,
                 drop_path: float = 0.1, ls_init: float = 1.0) -> None:
        super().__init__()
        self.peg = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        dpr = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio, dpr[i], ls_init) for i in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, feat: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = feat + self.peg(feat)
        x = feat.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x, bias)
        return self.norm(x)


class SegHead(nn.Module):
    """Decodificador ligero tipo FPN sobre las etapas conv (pasos 4, 8 y 16); logits a paso 4, un canal por
    estructura (0 pulmones, 1 corazón)."""

    def __init__(self, in_chs: Tuple[int, int, int] = (96, 192, 384), width: int = 64, out_ch: int = 2) -> None:
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, width, 1) for c in in_chs])
        self.fuse = nn.Sequential(nn.Conv2d(width, width, 3, padding=1, bias=False), nn.GroupNorm(8, width), nn.GELU())
        self.out = nn.Conv2d(width, out_ch, 1)

    def forward(self, f4: torch.Tensor, f8: torch.Tensor, f16: torch.Tensor) -> torch.Tensor:
        p = self.lat[2](f16)
        p = F.interpolate(p, size=f8.shape[-2:], mode="bilinear", align_corners=False) + self.lat[1](f8)
        p = F.interpolate(p, size=f4.shape[-2:], mode="bilinear", align_corners=False) + self.lat[0](f4)
        return self.out(self.fuse(p))
