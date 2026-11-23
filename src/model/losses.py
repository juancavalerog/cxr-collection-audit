"""Pérdidas de CSD-Net. Cada función lleva el nombre de su ecuación en docs/03_arquitectura.md
(Section 2.7): L_cls_real (27), L_cls_swap (28), L_cons (29), L_src (30), L_seg (31), L_sig (32),
L_ind (33) y L_total (34).

Enmienda del 13-11-2026 (docs/10_correccion_independencia.md): la penalización de independencia tiene
modos (`IndConfig.mode`). "mmd_sesgada" es la del primer intento del piloto (estimador sesgado de la MMD sobre las
firmas del lote; colapsa la firma). El modo por defecto, "mmd_banco_tope", usa el estimador insesgado (estadístico U)
sobre las firmas del lote más un banco de firmas recientes por fuente mixta y clase, recorta el valor en cero y limita
la norma de su gradiente sobre la firma a `beta` veces la del gradiente de lambda_3 L_src (ecuaciones 33b y 33c)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    """Pesos de la pérdida total, ecuación (34). Valores por defecto de la Section 2.8."""

    lambda_swap: float = 1.0
    lambda_cons: float = 1.0
    lambda_src: float = 0.5
    lambda_seg: float = 0.5
    lambda_sig: float = 0.1
    lambda_ind: float = 10.0   # enmienda de 13-11-2026: 10 con el modo mmd_banco_tope (1.0 en el primer intento)
    rho_within: float = 0.5
    focal_alpha: float = 0.5
    focal_gamma: float = 2.0


def focal_bce(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.5, gamma: float = 2.0,
              weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Entropía cruzada binaria focal con balanceo alpha, ecuación (27). Media sobre las muestras con peso."""
    logits, targets = logits.float(), targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = targets * p + (1 - targets) * (1 - p)
    a_t = targets * alpha + (1 - targets) * (1 - alpha)
    loss = a_t * (1 - p_t).pow(gamma) * bce
    if weight is None:
        return loss.mean()
    weight = weight.float()
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


def L_cls_real(out: Dict[str, torch.Tensor], y: torch.Tensor, w: LossWeights) -> torch.Tensor:
    """Clasificación sobre las vistas con fuente efectiva propia, ecuación (27): la vista real con peso
    1 - rho y, si existe, la vista re-firmada dentro de la fuente con peso rho."""
    real = focal_bce(out["logit"], y, w.focal_alpha, w.focal_gamma)
    if "logit_within" not in out:
        return real
    within = focal_bce(out["logit_within"], y, w.focal_alpha, w.focal_gamma)
    return (1 - w.rho_within) * real + w.rho_within * within


def L_cls_swap(out: Dict[str, torch.Tensor], y: torch.Tensor, valid: torch.Tensor, w: LossWeights) -> torch.Tensor:
    """Clasificación sobre las muestras re-firmadas con la etiqueta de la muestra de contenido, ecuación (28)."""
    return focal_bce(out["logit_swap"], y, w.focal_alpha, w.focal_gamma, weight=valid.float())


def _bernoulli_js(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p, q = p.clamp(eps, 1 - eps), q.clamp(eps, 1 - eps)
    m = 0.5 * (p + q)

    def kl(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return a * torch.log(a / c) + (1 - a) * torch.log((1 - a) / (1 - c))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def L_cons(out: Dict[str, torch.Tensor], valid: torch.Tensor) -> torch.Tensor:
    """Consistencia: divergencia de Jensen-Shannon entre la predicción real y la de cada vista re-firmada
    (cruzada y, si existe, dentro de la fuente), promediada sobre las vistas, ecuación (29)."""
    p = torch.sigmoid(out["logit"].float())
    v = valid.float()
    terms = [(_bernoulli_js(p, torch.sigmoid(out["logit_swap"].float())) * v).sum() / v.sum().clamp_min(1.0)]
    if "logit_within" in out:
        terms.append(_bernoulli_js(p, torch.sigmoid(out["logit_within"].float())).mean())
    return sum(terms) / len(terms)


def L_src(out: Dict[str, torch.Tensor], s: torch.Tensor) -> torch.Tensor:
    """Reconocimiento de la fuente desde la firma, ecuación (30)."""
    return F.cross_entropy(out["src_logits"].float(), s.long())


def L_seg(out: Dict[str, torch.Tensor], mask: torch.Tensor, quality: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Segmentación pulmonar contra pseudoetiquetas del profesor: BCE + Dice, ponderadas por la calidad
    de la pseudoetiqueta y promediadas por canal (pulmones, corazón), ecuación (31)."""
    c = min(out["seg_logits"].shape[1], mask.shape[1])  # canales disponibles en la pseudoetiqueta
    logits = out["seg_logits"][:, :c].float()
    tgt = F.interpolate(mask[:, :c].float(), size=logits.shape[-2:], mode="area")
    bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none").mean(dim=(1, 2, 3))
    p = torch.sigmoid(logits)
    inter = (p * tgt).sum(dim=(2, 3))
    dice = (1 - (2 * inter + 1.0) / (p.sum(dim=(2, 3)) + tgt.sum(dim=(2, 3)) + 1.0)).mean(dim=1)
    per = bce + dice
    if quality is None:
        return per.mean()
    q = quality.float()
    return (per * q).sum() / q.sum().clamp_min(1e-6)


def L_sig(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Los estadísticos de canal de la etapa de re-firma, explicados desde la firma, ecuación (32).
    La desviación se compara en escala logarítmica y la media se normaliza por la desviación propia."""
    pred, tgt = out["sig_pred"].float(), out["stats_real"].float()
    c = tgt.shape[1] // 2
    g, b = pred[:, :c], pred[:, c:]
    log_sd, mu = tgt[:, :c], tgt[:, c:]
    sd = torch.exp(log_sd)
    return ((g - log_sd) ** 2).mean() + (((b - mu) / sd) ** 2).mean()


def _rbf_mmd2(x: torch.Tensor, y: torch.Tensor, scales: Sequence[float] = (0.5, 1.0, 2.0)) -> torch.Tensor:
    """MMD^2 sesgado con suma de núcleos gaussianos; anchura por la mediana de las distancias."""
    z = torch.cat([x, y], dim=0)
    d2 = torch.cdist(z, z).pow(2)
    med = d2.detach()[d2.detach() > 0].median() if bool((d2.detach() > 0).any()) else d2.new_tensor(1.0)
    k = sum(torch.exp(-d2 / (s * med + 1e-8)) for s in scales)
    n = x.shape[0]
    kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    return kxx.mean() + kyy.mean() - 2 * kxy.mean()


def _rbf_mmd2_unbiased(x: torch.Tensor, y: torch.Tensor, scales: Sequence[float] = (0.5, 1.0, 2.0)) -> torch.Tensor:
    """MMD^2 insesgado (estadístico U, sin los términos diagonales de k(x, x) y k(y, y)) con la misma suma de núcleos
    gaussianos; anchura por la mediana de las distancias al cuadrado fuera de la diagonal. Puede ser negativo."""
    n, m = x.shape[0], y.shape[0]
    z = torch.cat([x, y], dim=0)
    d2 = torch.cdist(z, z).pow(2)
    dd = d2.detach()
    off = dd[~torch.eye(n + m, dtype=torch.bool, device=z.device)]
    med = off.median() if bool((off > 0).any()) else d2.new_tensor(1.0)
    k = sum(torch.exp(-d2 / (sc * med + 1e-8)) for sc in scales)
    kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    return ((kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1)) + (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1))
            - 2 * kxy.mean())


IND_MODES = ("ninguna", "mmd_sesgada", "mmd_insesgada", "mmd_banco_tope")


@dataclass
class IndConfig:
    """Modo de la penalización de independencia (`csdnet.ind` en la configuración; Tabla 6, enmienda de 13-11-2026).

    mode: "mmd_banco_tope" (por defecto), "mmd_sesgada" (primer intento del piloto), "mmd_insesgada" (sin banco ni
    tope) o "ninguna". bank_size: firmas recientes guardadas por fuente mixta y clase. beta: tope de la razón entre la
    norma del gradiente de lambda_6 L_ind y la de lambda_3 L_src sobre la firma. desde: fase en la que empieza L_ind
    ("A1" o "A2"; con "A2" la fase A1 entrena la firma solo con L_src)."""

    mode: str = "mmd_banco_tope"
    bank_size: int = 256
    beta: float = 0.5
    desde: str = "A1"
    min_per_class: int = 2

    def __post_init__(self) -> None:
        if self.mode not in IND_MODES:
            raise ValueError(f"modo de independencia desconocido: {self.mode} (válidos: {IND_MODES})")
        if self.desde not in ("A1", "A2"):
            raise ValueError("ind.desde debe ser 'A1' o 'A2'")


class SignatureBank:
    """Banco FIFO de firmas recientes (separadas del grafo) por fuente mixta y clase, ecuación (33b)."""

    def __init__(self, size: int = 256) -> None:
        self.size = int(size)
        self.mem: Dict[Tuple[int, int], torch.Tensor] = {}

    @torch.no_grad()
    def update(self, signature: torch.Tensor, s: torch.Tensor, y: torch.Tensor, mixed_sources: Sequence[int]) -> None:
        a = signature.detach().float()
        for src in mixed_sources:
            for c in (0, 1):
                q = (s == src) & (y.long() == c)
                if bool(q.any()):
                    old = self.mem.get((int(src), c))
                    new = a[q] if old is None else torch.cat([a[q], old.to(a.device)], 0)
                    self.mem[(int(src), c)] = new[: self.size]

    def get(self, key: Tuple[int, int]) -> Optional[torch.Tensor]:
        return self.mem.get(key)

    def state_dict(self) -> dict:
        return {"size": self.size, "mem": {"%d_%d" % k: v.cpu() for k, v in self.mem.items()}}

    def load_state_dict(self, d: dict) -> None:
        self.size = int(d.get("size", self.size))
        self.mem = {tuple(int(t) for t in k.split("_")): v for k, v in (d.get("mem") or {}).items()}


def L_ind(signature: torch.Tensor, s: torch.Tensor, y: torch.Tensor, mixed_sources: Sequence[int],
          min_per_class: int = 2, estimator: str = "sesgado", bank: Optional[SignatureBank] = None) -> torch.Tensor:
    """Independencia firma-clase dentro de las fuentes con ambas clases (MMD entre clases), ecuación (33).

    estimator: "sesgado" (ecuación 33 original, primer intento) o "insesgado" (estadístico U). Con `bank`, a las firmas
    de cada clase del lote se añaden las del banco de esa fuente y clase, sin gradiente (ecuación 33b); el peso de cada
    fuente sigue siendo su número de muestras en el lote."""
    f = _rbf_mmd2 if estimator == "sesgado" else _rbf_mmd2_unbiased
    a = signature.float()
    total = a.new_zeros(())
    mass = 0.0
    for src in mixed_sources:
        pos = (s == src) & (y == 1)
        neg = (s == src) & (y == 0)
        n_pos, n_neg = int(pos.sum()), int(neg.sum())
        xp, xq = a[pos], a[neg]
        if bank is not None:
            bp, bq = bank.get((int(src), 1)), bank.get((int(src), 0))
            if bp is not None and len(bp):
                xp = torch.cat([xp, bp.to(a.device)], 0)
            if bq is not None and len(bq):
                xq = torch.cat([xq, bq.to(a.device)], 0)
            ok = len(xp) >= min_per_class and len(xq) >= min_per_class and n_pos + n_neg > 0
        else:
            ok = n_pos >= min_per_class and n_neg >= min_per_class
        if ok:
            wgt = float(n_pos + n_neg)
            total = total + wgt * f(xp, xq)
            mass += wgt
    return total / mass if mass > 0 else total


def source_entropy(s: torch.Tensor, num_sources: Optional[int] = None) -> float:
    """Entropía (nats) de la distribución de fuentes del lote: la L_src de una cabeza que solo predice las
    frecuencias. Si L_src no baja de este valor, la firma no ha aprendido la fuente."""
    c = torch.bincount(s.long().flatten(), minlength=int(num_sources or 0)).double()
    p = c[c > 0] / c.sum()
    return float(-(p * p.log()).sum())


def _ind_term(out: Dict[str, torch.Tensor], s: torch.Tensor, y: torch.Tensor, w: "LossWeights", ind: IndConfig,
              mixed_sources: Sequence[int], bank: Optional[SignatureBank], l_src: torch.Tensor):
    """Término de independencia de la pérdida total según el modo. Devuelve (sumando, valor de L_ind, registro)."""
    a = out["signature"]
    logs: Dict[str, float] = {}
    if ind.mode in ("mmd_sesgada", "mmd_insesgada"):
        li = L_ind(a, s, y, mixed_sources, ind.min_per_class, "sesgado" if ind.mode == "mmd_sesgada" else "insesgado")
        return w.lambda_ind * li, li, logs
    # mmd_banco_tope: estimador insesgado con banco, recortado en cero, y tope relativo de la norma del gradiente
    li = L_ind(a, s, y, mixed_sources, ind.min_per_class, "insesgado", bank).clamp_min(0.0)
    if not (torch.is_grad_enabled() and a.requires_grad and li.requires_grad):
        return w.lambda_ind * li.detach(), li, logs
    g_src = torch.autograd.grad(w.lambda_src * l_src, a, retain_graph=True, allow_unused=True)[0]
    g_ind = torch.autograd.grad(w.lambda_ind * li, a, retain_graph=True, allow_unused=True)[0]
    if g_ind is None or g_src is None:
        return w.lambda_ind * li.detach(), li, logs
    n_src, n_ind = g_src.float().norm(), g_ind.float().norm()
    c = torch.clamp(ind.beta * n_src / n_ind.clamp_min(1e-12), max=1.0).detach()
    g = (c * g_ind).detach().to(a.dtype)
    surrogate = (a * g).sum() - (a.detach() * g).sum()   # valor nulo; su gradiente sobre la firma es c * d(lambda_6 L_ind)/da
    logs["ind_razon_grad"] = float(c * n_ind / n_src.clamp_min(1e-12))
    logs["ind_tope_activo"] = float(c < 1.0)
    return surrogate + w.lambda_ind * li.detach(), li, logs


def L_total(out: Dict[str, torch.Tensor], y: torch.Tensor, s: torch.Tensor, w: LossWeights,
            valid: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
            mask_quality: Optional[torch.Tensor] = None, mixed_sources: Sequence[int] = (),
            ramp: float = 1.0, ind: Optional[IndConfig] = None,
            bank: Optional[SignatureBank] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Pérdida total, ecuación (34). `ramp` en [0, 1] escala los términos de intercambio (calendario).
    `ind` elige el modo de L_ind (por defecto IndConfig(), "mmd_banco_tope"); `bank` es su banco de firmas.
    Registra además la entropía de las fuentes del lote (H_src_lote), L_src menos esa entropía y la dispersión de la
    firma (desv_firma: media sobre las dimensiones de la desviación típica en el lote)."""
    ind = ind if ind is not None else IndConfig()
    extra: Dict[str, float] = {}
    parts: Dict[str, torch.Tensor] = {"L_cls_real": L_cls_real(out, y, w)}
    total = parts["L_cls_real"] + out["ddp_anchor"]
    if "logit_swap" in out and valid is not None:
        parts["L_cls_swap"] = L_cls_swap(out, y, valid, w)
        parts["L_cons"] = L_cons(out, valid)
        total = total + ramp * (w.lambda_swap * parts["L_cls_swap"] + w.lambda_cons * parts["L_cons"])
    if "src_logits" in out:
        parts["L_src"] = L_src(out, s)
        total = total + w.lambda_src * parts["L_src"]
        if w.lambda_sig > 0:
            parts["L_sig"] = L_sig(out)
            total = total + w.lambda_sig * parts["L_sig"]
        h = source_entropy(s, out["src_logits"].shape[1])
        extra["H_src_lote"] = h
        extra["L_src_menos_H"] = float(parts["L_src"].detach()) - h
        if "signature" in out and out["signature"].shape[0] > 1:
            extra["desv_firma"] = float(out["signature"].detach().float().std(0).mean())
        if w.lambda_ind > 0 and len(mixed_sources) > 0 and ind.mode != "ninguna":
            term, parts["L_ind"], lg = _ind_term(out, s, y, w, ind, mixed_sources, bank, parts["L_src"])
            extra.update(lg)
            total = total + term
    if mask is not None and w.lambda_seg > 0:
        parts["L_seg"] = L_seg(out, mask, mask_quality)
        total = total + w.lambda_seg * parts["L_seg"]
    logs = {k: float(v.detach()) for k, v in parts.items()}
    logs["L_total"] = float(total.detach())
    logs.update(extra)
    return total, logs
