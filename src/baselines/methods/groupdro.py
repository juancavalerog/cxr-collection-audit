"""GroupDRO sobre las celdas clase por fuente no vacías [sagawa2020distributionally].

Grupo g = 2 s + y. Los pesos q se definen solo sobre los grupos presentes en el entrenamiento: en las celdas
vacías (p. ej. negativos de stonybrook o positivos de rsna) el riesgo del grupo no existe y el método queda
indefinido; esta implementación fija q = 0 en esas celdas y nunca las actualiza, de modo que el objetivo es el
peor caso sobre las celdas presentes. En cada paso se actualizan solo los grupos con ejemplos en el lote
(q_g <- q_g exp(eta L_g), como el algoritmo 1 del artículo y DomainBed) y se renormaliza sobre los grupos
presentes; la pérdida es la media de L_g ponderada por q sobre los grupos del lote. q es un buffer del modelo:
se guarda en el checkpoint y la reanudación lo recupera.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.baselines.common import N_SOURCES_MAX
from src.baselines.registry import Method, register


class _Q(nn.Module):
    def __init__(self, valid):
        super().__init__()
        v = torch.as_tensor(valid, dtype=torch.bool)
        self.register_buffer("valid", v)
        self.register_buffer("q", v.float() / v.float().sum())


@register
class GroupDRO(Method):
    name = "groupdro"
    title = "GroupDRO (celdas clase por fuente)"
    ref = "sagawa2020distributionally"
    labels = ("clase", "fuente")
    empty_cells = "indefinido en las celdas vacías: q = 0 fijo, peor caso solo sobre las celdas presentes"
    defaults = {"eta": 1e-2}
    search = {"eta": ("log10", -3.0, -1.0)}
    hp_source = "DomainBed (groupdro_eta 1e-2, 10^U(-3, -1)); SubpopBench usa el mismo"
    cost = 1.0
    degenerado_sin_mixtas = "sin fuentes mixtas los grupos presentes coinciden con las clases: equivale a DRO por clase"

    def build(self, tr, net):
        t = tr.cells                                  # (2, 8)
        valid = [t[g % 2, g // 2] > 0 for g in range(2 * N_SOURCES_MAX)]
        net.aux["groupdro"] = _Q(valid)

    def loss(self, tr, net, x, y, s, gen, B):
        z = net(x)
        per = tr.loss_fn(z.float(), y, reduction="none")
        g = 2 * s + (y > 0.5).long()
        m = net.aux["groupdro"]
        present = torch.unique(g)
        present = present[m.valid[present]]
        losses = torch.stack([per[g == k].mean() for k in present])
        with torch.no_grad():
            m.q[present] *= torch.exp(float(self.p["eta"]) * losses.detach().float())
            m.q /= m.q[m.valid].sum()
            m.q[~m.valid] = 0.0
        w = m.q[present] / m.q[present].sum()
        robust = (w.detach() * losses).sum()
        return robust * (x.shape[0] / B), {"groupdro_max_q": float(m.q.max())}
