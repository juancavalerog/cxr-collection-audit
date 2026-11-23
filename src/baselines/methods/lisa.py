"""LISA [yao2022improving]: aumento selectivo con mixup en dos modos, elegido por lote con probabilidad mix_ratio.

- Dentro de etiqueta (intra-label): pareja con la misma clase y otra fuente; la etiqueta mezclada es la misma.
- Dentro de dominio (intra-domain): pareja de la misma fuente y la otra clase; etiqueta lambda y_i + (1 - lambda) y_j.
lambda ~ Beta(mix_alpha, mix_alpha). Las parejas se buscan dentro del lote (el muestreador común es equilibrado por
clase). Parejas que no existen por las celdas vacías: en el modo dentro de dominio, ningún ejemplo de una fuente de
una sola clase (en COVIDx limpio: fig1, ricord, sirm y stonybrook solo positivos; rsna solo negativos) tiene pareja,
y esos ejemplos se usan sin mezclar; solo actúan actmed, bimcv y cohen. En el modo dentro de etiqueta, los
negativos solo pueden mezclarse entre actmed, bimcv, cohen y rsna. La proporción de ejemplos sin pareja se registra.
"""
from __future__ import annotations

import torch

from src.baselines.common import hard_labels, pick_partners, sample_beta
from src.baselines.registry import Method, register


def lisa_pairs(y, s, mode, gen):
    yy = hard_labels(y)
    same_y = yy.view(-1, 1) == yy.view(1, -1)
    same_s = s.view(-1, 1) == s.view(1, -1)
    allowed = (same_y & ~same_s) if mode == "intra_label" else (same_s & ~same_y)
    return pick_partners(allowed, gen)


def lisa_mix(x, y, s, mode, gen, alpha=2.0):
    B = x.shape[0]
    j, has = lisa_pairs(y, s, mode, gen)
    lam = sample_beta(alpha, B, gen, x.device)
    lam = torch.where(has, lam, torch.ones_like(lam))
    xm = lam.view(B, 1, 1, 1) * x + (1 - lam.view(B, 1, 1, 1)) * x[j]
    ym = lam * y + (1 - lam) * y[j]
    return xm, ym, j, has


@register
class LISA(Method):
    name = "lisa"
    title = "LISA (mixup selectivo dentro de etiqueta y dentro de fuente)"
    ref = "yao2022improving"
    labels = ("clase", "fuente")
    empty_cells = "pierde las parejas dentro de fuente en las fuentes de una sola clase; esos ejemplos se usan sin mezclar"
    defaults = {"mix_alpha": 2.0, "mix_ratio": 0.5}
    search = {"mix_alpha": ("choice", [0.5, 1.0, 2.0, 4.0]), "mix_ratio": ("choice", [0.25, 0.5, 0.75])}
    hp_source = "repositorio de LISA (mix_alpha 2, mix_ratio 0.5); rango propio acotado"
    cost = 1.0
    degenerado_sin_mixtas = "sin fuentes mixtas no hay parejas dentro de fuente: el modo dentro de dominio no mezcla"

    def loss(self, tr, net, x, y, s, gen, B):
        intra_label = bool(torch.rand((), generator=gen, device=x.device) < float(self.p["mix_ratio"]))
        mode = "intra_label" if intra_label else "intra_domain"
        xm, ym, j, has = lisa_mix(x, y, s, mode, gen, float(self.p["mix_alpha"]))
        self.last = {"mode": mode, "partner": j, "has": has}
        z = net(xm).float()
        return tr.loss_fn(z, ym, reduction="sum") / B, {"lisa_sin_pareja": float((~has).float().mean()),
                                                       "lisa_intra_label": float(intra_label)}
