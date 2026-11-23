"""Muestreo dirigido por la tabla clase por fuente (docs/03_arquitectura.md, Sections 1.5 y 2.8).

- ClassSourceTable: tabla de recuentos, tipos de celda (vacía, escasa, completa) y plan de
  enrutamiento que hace la fuente efectiva independiente de la clase (Proposiciones 1 y 2).
- TableGuidedBatchSampler: lotes con cuota de donantes por fuente y de ambas clases en las fuentes mixtas,
  compatible con entrenamiento distribuido (cada proceso recibe lotes distintos).
- sample_swap_pairs: elige para cada muestra una fuente destino distinta de la suya y un donante de
  firma de esa fuente dentro del lote (vista cruzada).
- sample_within_pairs: elige para cada muestra un donante de su misma fuente, distinto de ella si el
  lote lo permite (vista dentro de la fuente, Section 2.4).
Dependencias: numpy y torch.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch


@dataclass
class RoutingPlan:
    """Plan de enrutamiento: q (fuente efectiva objetivo), pi (destino agregado por clase),
    routing[y, s, s'] = P(destino s' | clase y, origen s), con diagonal nula."""

    q: np.ndarray
    pi: np.ndarray
    routing: np.ndarray
    z: float
    lambda_swap: float
    feasible: bool
    residual: float


@dataclass
class ClassSourceTable:
    """Tabla de recuentos por clase (fila 0 negativa, fila 1 positiva) y fuente (columnas)."""

    sources: List[str]
    counts: np.ndarray

    @classmethod
    def from_csv(cls, path: str, split: str = "train") -> "ClassSourceTable":
        """Lee data/manifests/covidx_cxr4_clase_fuente.csv (columnas split, source, label, n_images)."""
        rows = [r for r in csv.DictReader(open(path, newline="")) if r["split"] == split]
        sources = sorted({r["source"] for r in rows})
        counts = np.zeros((2, len(sources)), dtype=np.float64)
        for r in rows:
            counts[1 if r["label"] == "positive" else 0, sources.index(r["source"])] += float(r["n_images"])
        return cls(sources, counts)

    @classmethod
    def from_arrays(cls, labels: Sequence[int], sources: Sequence[int], num_sources: int,
                    names: Optional[List[str]] = None) -> "ClassSourceTable":
        """Construye la tabla a partir de etiquetas y fuentes enteras."""
        counts = np.zeros((2, num_sources))
        for y, s in zip(labels, sources):
            counts[int(y), int(s)] += 1
        return cls(names or [str(i) for i in range(num_sources)], counts)

    @property
    def num_sources(self) -> int:
        return len(self.sources)

    def index(self, name: str) -> int:
        return self.sources.index(name)

    def conditional(self) -> np.ndarray:
        """P(s | y), forma (2, K), ecuación (5)."""
        return self.counts / self.counts.sum(axis=1, keepdims=True)

    def mixed_sources(self) -> List[int]:
        """Fuentes con ambas clases presentes."""
        return [k for k in range(self.num_sources) if (self.counts[:, k] > 0).all()]

    def cell_types(self, scarce_ratio: float = 0.1, scarce_abs: int = 50) -> np.ndarray:
        """Tipo de cada celda: 'vacia' (n = 0), 'escasa' (0 < n < scarce_abs o n < scarce_ratio por la
        otra clase de la misma fuente) o 'completa', ecuación (4)."""
        t = np.empty(self.counts.shape, dtype=object)
        for y in (0, 1):
            for k in range(self.num_sources):
                n, other = self.counts[y, k], self.counts[1 - y, k]
                if n == 0:
                    t[y, k] = "vacia"
                elif n < scarce_abs or n < scarce_ratio * other:
                    t[y, k] = "escasa"
                else:
                    t[y, k] = "completa"
        return t

    def routing(self, lambda_swap: float = 1.0, iters: int = 5000, tol: float = 1e-10) -> RoutingPlan:
        """Plan que iguala la distribución de fuente efectiva entre clases, ecuaciones (10) a (12) y Proposición 1.

        q(s) = max_y P(s|y) / Z, Z = sum_s max_y P(s|y); pi(s|y) = ((1 + lambda) q(s) - P(s|y)) / lambda.
        El plan por origen se obtiene con Sinkhorn sobre el núcleo 1 - I (sin quedarse en la propia fuente).
        """
        p = self.conditional()
        m = p.max(axis=0)
        z = float(m.sum())
        q = m / z
        lam = float(lambda_swap)
        pi = ((1.0 + lam) * q[None, :] - p) / lam
        feasible = bool((pi >= -1e-12).all() and ((p + pi) <= 1.0 + 1e-9).all())
        pi = np.clip(pi, 0.0, None)
        pi = pi / pi.sum(axis=1, keepdims=True)
        kk = self.num_sources
        kern = 1.0 - np.eye(kk)
        routing = np.zeros((2, kk, kk))
        residual = 0.0
        for y in (0, 1):
            a, b = p[y], pi[y]
            u, v = np.ones(kk), np.ones(kk)
            for _ in range(iters):
                u = np.where(a > 0, a / np.maximum(kern @ v, 1e-300), 0.0)
                v = np.where(b > 0, b / np.maximum(kern.T @ u, 1e-300), 0.0)
                t = u[:, None] * kern * v[None, :]
                err = max(abs(t.sum(1) - a).max(), abs(t.sum(0) - b).max())
                if err < tol:
                    break
            residual = max(residual, float(err))
            for s in range(kk):
                if a[s] > 0:
                    routing[y, s] = t[s] / a[s]
                else:  # origen sin muestras: destino pi(.|y) sin la propia fuente
                    r = b.copy()
                    r[s] = 0.0
                    routing[y, s] = r / r.sum() if r.sum() > 0 else kern[s] / kern[s].sum()
        return RoutingPlan(q=q, pi=pi, routing=routing, z=z, lambda_swap=lam, feasible=feasible, residual=residual)


@dataclass
class SwapPairs:
    """Resultado del emparejamiento: índice del donante, fuente destino, válido y si hubo reserva."""

    donor_index: torch.Tensor
    target_source: torch.Tensor
    valid: torch.Tensor
    fallback: torch.Tensor


def sample_swap_pairs(sources: torch.Tensor, labels: torch.Tensor, routing: torch.Tensor,
                      generator: Optional[torch.Generator] = None) -> SwapPairs:
    """Empareja cada muestra i con un donante j de otra fuente presente en el lote.

    La fuente destino se elige con probabilidad routing[y_i, s_i, s'] restringida a las fuentes presentes
    y distintas de s_i; si esa masa es nula se elige al azar entre las fuentes presentes distintas
    (fallback). Si no hay otra fuente en el lote, la pareja es inválida y j = i.
    """
    dev = sources.device
    s = sources.long().cpu()
    y = labels.long().cpu()
    r = routing.detach().float().cpu() if isinstance(routing, torch.Tensor) else torch.as_tensor(routing).float()
    kk = r.shape[-1]
    b = s.shape[0]
    present = torch.zeros(kk)
    present[s.unique()] = 1.0
    probs = r[y, s] * present[None, :]
    probs[torch.arange(b), s] = 0.0
    other = present[None, :].repeat(b, 1)
    other[torch.arange(b), s] = 0.0
    has_other = other.sum(1) > 0
    use_fb = (probs.sum(1) <= 0) & has_other
    probs = torch.where(use_fb[:, None], other, probs)
    safe = torch.where(has_other[:, None], probs, torch.ones_like(probs))
    tgt = torch.multinomial(safe, 1, generator=generator).squeeze(1)
    tgt = torch.where(has_other, tgt, s)
    donor = torch.arange(b)
    for i in range(b):
        if has_other[i]:
            cand = torch.nonzero(s == tgt[i]).squeeze(1)
            donor[i] = cand[torch.randint(len(cand), (1,), generator=generator)]
    return SwapPairs(donor.to(dev), tgt.to(dev), has_other.to(dev), use_fb.to(dev))


@dataclass
class WithinPairs:
    """Donante de la misma fuente y si es distinto de la propia muestra."""

    donor_index: torch.Tensor
    distinct: torch.Tensor


def sample_within_pairs(sources: torch.Tensor, generator: Optional[torch.Generator] = None) -> WithinPairs:
    """Para cada muestra i elige al azar un donante j con s_j = s_i y j != i; si i es la única muestra de
    su fuente en el lote, j = i (la vista coincide entonces con la real por la reconstrucción exacta)."""
    dev = sources.device
    s = sources.long().cpu()
    b = s.shape[0]
    donor = torch.arange(b)
    distinct = torch.zeros(b, dtype=torch.bool)
    for i in range(b):
        cand = torch.nonzero((s == s[i]) & (torch.arange(b) != i)).squeeze(1)
        if len(cand) > 0:
            donor[i] = cand[torch.randint(len(cand), (1,), generator=generator)]
            distinct[i] = True
    return WithinPairs(donor.to(dev), distinct.to(dev))


class TableGuidedBatchSampler:
    """Muestreador de lotes para el intercambio dirigido (Section 2.8, algoritmo 1).

    Cada lote contiene: `min_per_source` donantes de cada fuente (con reemplazo si la fuente es pequeña),
    `min_per_class_mixed` imágenes de cada clase en cada fuente mixta (para L_ind) y el resto equilibrado
    50/50 por clase, uniforme dentro de cada clase. Con rank y world_size cada proceso recibe lotes
    distintos y deterministas dada (seed, epoch).
    """

    def __init__(self, labels: Sequence[int], sources: Sequence[int], batch_size: int, num_batches: int,
                 min_per_source: int = 2, min_per_class_mixed: int = 4, seed: int = 0, rank: int = 0,
                 world_size: int = 1) -> None:
        self.labels = np.asarray(labels, dtype=np.int64)
        self.sources = np.asarray(sources, dtype=np.int64)
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.min_per_source = min_per_source
        self.min_per_class_mixed = min_per_class_mixed
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self.by_source: Dict[int, np.ndarray] = {int(k): np.nonzero(self.sources == k)[0] for k in np.unique(self.sources)}
        self.by_class: Dict[int, np.ndarray] = {c: np.nonzero(self.labels == c)[0] for c in (0, 1)}
        self.mixed: List[int] = [k for k, idx in self.by_source.items() if len(np.unique(self.labels[idx])) == 2]
        quota = min_per_source * len(self.by_source) + 2 * min_per_class_mixed * len(self.mixed)
        if quota >= batch_size:
            raise ValueError(f"la cuota fija ({quota}) no cabe en el lote ({batch_size})")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng([self.seed, self.epoch, self.rank, self.world_size])
        for _ in range(self.num_batches):
            batch: List[int] = []
            for k, idx in self.by_source.items():
                batch += rng.choice(idx, self.min_per_source, replace=len(idx) < self.min_per_source).tolist()
            for k in self.mixed:
                idx = self.by_source[k]
                for c in (0, 1):
                    pool = idx[self.labels[idx] == c]
                    batch += rng.choice(pool, self.min_per_class_mixed, replace=len(pool) < self.min_per_class_mixed).tolist()
            rest = self.batch_size - len(batch)
            n_pos = rest // 2
            batch += rng.choice(self.by_class[1], n_pos, replace=False).tolist()
            batch += rng.choice(self.by_class[0], rest - n_pos, replace=False).tolist()
            rng.shuffle(batch)
            yield [int(i) for i in batch]
