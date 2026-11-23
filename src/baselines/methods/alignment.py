"""CORAL [sun2016deep] e IRM [arjovsky2019invariant] con la fuente como entorno (formas de DomainBed).

CORAL: penalización media de |mu_i - mu_j|^2 + |C_i - C_j|^2 (medias de los elementos) entre todas las parejas
de fuentes del lote con al menos dos ejemplos, sobre los rasgos agregados del tronco.
IRM: penalización de IRMv1 con la estimación insesgada de DomainBed (producto de los gradientes respecto a una
escala ficticia en las mitades pares e impares de cada entorno), peso 1 durante los primeros anneal pasos y
lambda después. DomainBed reinicia Adam al terminar el recocido; el entrenador común no lo hace (diferencia
declarada). Celdas vacías: los entornos de una sola clase dan riesgos que no comparten la clase ausente; la
invarianza se impone igualmente, sin comprobar que el clasificador óptimo sea común.
"""
from __future__ import annotations

import torch

from src.baselines.common import head_of
from src.baselines.registry import Method, register


def coral_penalty(f, s, min_n=2):
    srcs = [k for k in torch.unique(s).tolist() if int((s == k).sum()) >= min_n]
    if len(srcs) < 2:
        return f.sum() * 0.0, 0
    stats = []
    for k in srcs:
        fk = f[s == k].float()
        mu = fk.mean(0, keepdim=True)
        c = fk - mu
        stats.append((mu, c.t() @ c / (len(fk) - 1)))
    pen, n = 0.0, 0
    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            pen = pen + (stats[i][0] - stats[j][0]).pow(2).mean() + (stats[i][1] - stats[j][1]).pow(2).mean()
            n += 1
    return pen / n, n


@register
class CORAL(Method):
    name = "coral"
    title = "CORAL (alineación de medias y covarianzas entre fuentes)"
    ref = "sun2016deep"
    labels = ("clase", "fuente")
    empty_cells = "alinea distribuciones marginales de fuentes con clases distintas: empuja a borrar la clase en las fuentes de una sola clase"
    defaults = {"mmd_gamma": 1.0}
    search = {"mmd_gamma": ("log10", -1.0, 1.0)}
    hp_source = "DomainBed (mmd_gamma 1, 10^U(-1, 1))"
    cost = 1.05
    degenerado_sin_mixtas = "sin fuentes mixtas alinear las fuentes es alinear las clases"

    def loss(self, tr, net, x, y, s, gen, B):
        f = net.pooled(x)
        z = head_of(net.net, f)
        cls = tr.loss_fn(z.float(), y, reduction="sum") / B
        pen, n = coral_penalty(f, s)
        return cls + float(self.p["mmd_gamma"]) * pen * (x.shape[0] / B), {"coral_pen": float(pen.detach()) if torch.is_tensor(pen) else float(pen), "coral_pairs": n}


@register
class IRM(Method):
    name = "irm"
    title = "IRM (IRMv1, fuente como entorno)"
    ref = "arjovsky2019invariant"
    labels = ("clase", "fuente")
    empty_cells = "los entornos de una sola clase tienen un riesgo trivialmente minimizable por la constante; la penalización no ve las celdas vacías"
    defaults = {"irm_lambda": 1e2, "irm_penalty_anneal_iters": 500}
    search = {"irm_lambda": ("log10", -1.0, 5.0), "irm_penalty_anneal_iters": ("log10", 0.0, 4.0)}
    hp_source = "DomainBed (irm_lambda 1e2, 10^U(-1, 5); anneal 500, 10^U(0, 4))"
    cost = 1.05

    def loss(self, tr, net, x, y, s, gen, B):
        z = net(x).float()
        nll, pen, n = 0.0, 0.0, 0
        for k in torch.unique(s).tolist():
            idx = (s == k).nonzero().view(-1)
            if len(idx) < 2:
                nll = nll + tr.loss_fn(z[idx], y[idx], reduction="mean")
                n += 1
                continue
            zk, yk = z[idx], y[idx]
            scale = torch.ones((), device=z.device, requires_grad=True)
            l1 = tr.loss_fn(zk[0::2] * scale, yk[0::2], reduction="mean")
            l2 = tr.loss_fn(zk[1::2] * scale, yk[1::2], reduction="mean")
            g1 = torch.autograd.grad(l1, [scale], create_graph=True)[0]
            g2 = torch.autograd.grad(l2, [scale], create_graph=True)[0]
            pen = pen + g1 * g2
            nll = nll + tr.loss_fn(zk, yk, reduction="mean")
            n += 1
        nll, pen = nll / n, pen / n
        wpen = float(self.p["irm_lambda"]) if tr.state["global_step"] >= int(self.p["irm_penalty_anneal_iters"]) else 1.0
        return (nll + wpen * pen) * (x.shape[0] / B), {"irm_pen": float(pen.detach()) if torch.is_tensor(pen) else float(pen)}
