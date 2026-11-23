"""Utilidades compartidas por las líneas base: inversión de gradiente, elección de parejas dentro del lote,
tabla clase por fuente, pérdida GCE, cabezas y acceso a rasgos del modelo común."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_SOURCES_MAX = 8          # fuentes declaradas de COVIDx CXR-4 (src/data/manifest.py)


# ====================================================================== inversión de gradiente
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = float(coeff)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.coeff * g, None


def grad_reverse(x, coeff: float = 1.0):
    """Identidad hacia delante; multiplica el gradiente por -coeff hacia atrás [ganin2016domain]."""
    return _GradReverse.apply(x, coeff)


class _GradReverseTo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff, target):
        ctx.coeff, ctx.target = float(coeff), float(target)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        n = g.float().norm()
        return (-ctx.coeff * ctx.target / (float(n) + 1e-12)) * g, None, None


def grad_reverse_to(x, coeff: float, target_norm: float):
    """Inversión de gradiente con norma fijada: hacia atrás devuelve el gradiente invertido reescalado para que su
    norma sea coeff * target_norm. Con target_norm igual a la norma del gradiente de la pérdida de clasificación en
    el mismo tensor, coeff es el peso relativo del adversario frente a la clasificación, con independencia de la
    escala de cada pérdida."""
    return _GradReverseTo.apply(x, coeff, target_norm)


# ====================================================================== parejas dentro del lote
def pick_partners(allowed: torch.Tensor, gen: torch.Generator):
    """allowed[i, j] indica si j puede ser pareja de i. Elige j uniforme entre los permitidos; si no hay
    ninguno, devuelve i (sin mezcla). Devuelve (índices, máscara de filas con pareja)."""
    B = allowed.shape[0]
    allowed = allowed.clone()
    allowed.fill_diagonal_(False)
    r = torch.rand(B, B, generator=gen, device=allowed.device)
    r = torch.where(allowed, r, torch.full_like(r, -1.0))
    j = r.argmax(dim=1)
    has = allowed.any(dim=1)
    j = torch.where(has, j, torch.arange(B, device=allowed.device))
    return j, has


def cross_source(s):
    return s.view(-1, 1) != s.view(1, -1)


def hard_labels(y):
    return (y.float() > 0.5).long()


# ====================================================================== tabla clase por fuente
def cell_table(frame, n_sources: int = N_SOURCES_MAX) -> np.ndarray:
    """n[y, s] del DataFrame de entrenamiento."""
    t = np.zeros((2, n_sources), dtype=np.int64)
    for (yy, ss), n in frame.groupby(["label", "source_idx"]).size().items():
        t[int(yy), int(ss)] = int(n)
    return t


def empty_cells(table: np.ndarray):
    """Celdas vacías (y, s) entre las fuentes presentes en el entrenamiento."""
    present = table.sum(0) > 0
    return [(y, s) for s in range(table.shape[1]) if present[s] for y in (0, 1) if table[y, s] == 0]


# ====================================================================== pérdidas
def gce_loss(logits, target, q: float = 0.7, reduction="none"):
    """Entropía cruzada generalizada multiclase: (1 - p_y^q) / q."""
    p = F.softmax(logits.float(), dim=1).gather(1, target.view(-1, 1)).squeeze(1).clamp_min(1e-8)
    loss = (1.0 - p.pow(q)) / q
    return loss.mean() if reduction == "mean" else (loss.sum() if reduction == "sum" else loss)


def gce_binary(logits, y, q: float = 0.7, reduction="none"):
    p = torch.sigmoid(logits.float().reshape(-1))
    yy = y.float().reshape(-1)
    pt = (yy * p + (1 - yy) * (1 - p)).clamp_min(1e-8)
    loss = (1.0 - pt.pow(q)) / q
    return loss.mean() if reduction == "mean" else (loss.sum() if reduction == "sum" else loss)


# ====================================================================== módulos
class MLP(nn.Module):
    """MLP de DomainBed (networks.MLP): entrada, mlp_depth - 2 capas ocultas, salida."""

    def __init__(self, n_in, n_out, width=256, depth=3, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(n_in, width), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(max(0, depth - 2)):
            layers += [nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(width, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    """Cabeza con la estructura del artículo 1 (Dense 128 relu, LayerNorm 1e-3, Dropout, Dense n_out)."""

    def __init__(self, n_in, n_out=1, hidden=128, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(n_in, hidden)
        self.ln = nn.LayerNorm(hidden, eps=1e-3)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, n_out)
        for lin in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)

    def forward(self, f):
        out = self.fc2(self.drop(self.ln(F.relu(self.fc1(f)))))
        return out.squeeze(-1) if out.shape[-1] == 1 else out


def head_of(net, f):
    """Cabeza del modelo común (Paper1ConvNeXt o TinyNet) aplicada a rasgos agregados."""
    return net.fc2(net.drop(net.ln(F.relu(net.fc1(f))))).squeeze(-1)


def penultimate(net, f):
    """Representación de entrada de la última capa (fc2), sin dropout en evaluación."""
    return net.drop(net.ln(F.relu(net.fc1(f))))


def n_features(net):
    return net.fc1.in_features


def early_stage_modules(net, layers):
    """Módulos tras los que se inserta MixStyle: etapas de ConvNeXt (timm) o capas del TinyNet de pruebas."""
    bb = net.backbone
    if hasattr(bb, "stages"):
        return [bb.stages[i] for i in layers]
    seq = [m for m in bb if not isinstance(m, nn.Identity)]
    return [seq[min(i + 1, len(seq) - 1)] for i in layers]


def sample_beta(alpha: float, n: int, gen, device):
    a = torch.full((n,), float(alpha), device=device)
    g1 = torch._standard_gamma(a, generator=gen)
    g2 = torch._standard_gamma(a, generator=gen)
    return g1 / (g1 + g2)
