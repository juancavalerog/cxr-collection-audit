"""DANN [ganin2016domain] y CDANN [li2018deep] con la fuente como dominio.

Discriminador MLP de DomainBed (ancho 256, profundidad 3) sobre los rasgos agregados del tronco, con una capa de
inversión de gradiente (GRL): el discriminador minimiza la entropía cruzada de la fuente y el tronco la maximiza
con peso lambda. DomainBed optimiza lo mismo con pasos alternos y dos optimizadores; aquí se usa la GRL original
de Ganin et al. para conservar el optimizador único del entrenador común. Las clases del discriminador son solo
las fuentes presentes en el entrenamiento. CDANN suma al rasgo una incrustación de la clase (como DomainBed) y
pondera la pérdida del discriminador por el inverso de la frecuencia de clase en el lote. Celdas vacías: el
discriminador condicionado a y = 0 nunca ve las fuentes solo positivas (y viceversa), de modo que en CDANN la
invarianza solo se impone entre las fuentes que comparten clase.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.baselines.common import MLP, N_SOURCES_MAX, grad_reverse, grad_reverse_to, head_of, n_features
from src.baselines.registry import Method, register


class _Disc(nn.Module):
    def __init__(self, nf, present, width, depth, dropout, conditional):
        super().__init__()
        present = list(present)
        lut = torch.full((N_SOURCES_MAX,), -1, dtype=torch.long)
        for i, sidx in enumerate(present):
            lut[sidx] = i
        self.register_buffer("lut", lut)
        self.mlp = MLP(nf, len(present), width, depth, dropout)
        self.emb = nn.Embedding(2, nf) if conditional else None


@register
class DANN(Method):
    name = "dann"
    title = "DANN (adversario de fuente)"
    ref = "ganin2016domain"
    labels = ("clase", "fuente")
    empty_cells = "no depende de las celdas; la invarianza marginal a la fuente choca con la clase cuando fuente y clase coinciden"
    conditional = False
    # Valores por defecto fijados con la prueba de humo en GPU del 13-11-2026 (docs/06_lineas_base.md, Section 3.4),
    # sin mirar el test: con la configuración literal de DomainBed (lambda 1, discriminador a la tasa del tronco,
    # inversión sin normalizar) la pérdida del discriminador se dispara al descongelar el tronco y la clasificación
    # cae al azar, porque la pérdida focal común tiene gradientes mucho menores que los del discriminador.
    defaults = {"lambda": 0.3, "mlp_width": 256, "mlp_depth": 3, "mlp_dropout": 0.0, "schedule": "constante",
                "disc_lr_mult": 10.0, "grad_penalty": 1.0, "grl_norm": "clase"}
    search = {"lambda": ("log10", -2.0, 0.0), "mlp_width": ("choice", [64, 128, 256, 512, 1024]),
              "mlp_depth": ("choice", [3, 4, 5]), "mlp_dropout": ("choice", [0.0, 0.1, 0.5]),
              "disc_lr_mult": ("choice", [3.0, 10.0, 30.0]), "grad_penalty": ("log10", -2.0, 1.0)}
    hp_source = ("DomainBed para el discriminador (mlp_width 256, mlp_depth 3, mlp_dropout 0) y para la penalización "
                 "de gradiente (10^U(-2, 1)); propios, fijados con la prueba de humo en GPU: lambda como peso relativo "
                 "al gradiente de clasificación (0.3; 10^U(-2, 0)) y discriminador 10 veces más rápido que el tronco")
    cost = 1.05
    degenerado_sin_mixtas = "sin fuentes mixtas la fuente determina la clase: el adversario de fuente es un adversario de clase"

    def build(self, tr, net):
        present = [s for s in range(N_SOURCES_MAX) if tr.cells[:, s].sum() > 0]
        if len(present) < 2:
            raise ValueError("%s necesita al menos dos fuentes en el entrenamiento" % self.name)
        p = self.p
        net.aux["disc"] = _Disc(n_features(net.net), present, int(p["mlp_width"]), int(p["mlp_depth"]),
                                float(p["mlp_dropout"]), self.conditional)
        # El discriminador lleva su propio grupo de tasa (disc_lr_mult veces la del tronco): debe ser más rápido que
        # el tronco para seguir a los rasgos; a la tasa del tronco la pérdida adversaria se dispara en la fase B.
        net._aux_lr.append((net.aux["disc"], float(p.get("disc_lr_mult", 1.0))))

    def coeff(self, tr):
        """Peso de la inversión de gradiente. Con `schedule: ganin` sigue 2 / (1 + exp(-10 p)) - 1 [ganin2016domain],
        con p el progreso contado desde el primer paso en que el tronco se entrena: en las fases con el tronco
        congelado la inversión no actúa sobre ningún parámetro y no deben consumir la rampa."""
        lam = float(self.p["lambda"])
        sch = self.p["schedule"]
        if sch == "ganin":
            phases = tr.cfg["train"]["phases"]
            spe = int(tr.steps_per_epoch)
            tot = sum(int(ph["epochs"]) for ph in phases) * spe
            start = 0
            for ph in phases:
                if ph.get("train_backbone", True):
                    break
                start += int(ph["epochs"]) * spe
            prog = (tr.state["global_step"] - start) / max(1, tot - start)
            prog = min(1.0, max(0.0, prog))
            lam *= 2.0 / (1.0 + math.exp(-10 * prog)) - 1.0
        elif sch != "constante":
            raise ValueError("%s: schedule desconocido %r (admitidos: constante, ganin)" % (self.name, sch))
        return lam

    def loss(self, tr, net, x, y, s, gen, B):
        f = net.pooled(x)
        z = head_of(net.net, f)
        cls = tr.loss_fn(z.float(), y, reduction="sum") / B
        d = net.aux["disc"]
        coef = self.coeff(tr)
        modo = self.p.get("grl_norm", "ninguna")
        if modo == "clase" and f.requires_grad and coef > 0:
            # Norma del gradiente de la clasificación en los rasgos: el gradiente invertido se reescala a
            # coef veces esa norma (una pasada hacia atrás adicional solo por la cabeza).
            gc = torch.autograd.grad(cls, f, retain_graph=True)[0]
            fd = grad_reverse_to(f, coef, float(gc.float().norm()))
        elif modo in ("ninguna", "clase"):
            fd = grad_reverse(f, coef)
        else:
            raise ValueError("%s: grl_norm desconocido %r (admitidos: ninguna, clase)" % (self.name, modo))
        yy = (y > 0.5).long()
        if d.emb is not None:
            fd = fd + d.emb(yy)
        out = d.mlp(fd).float()
        tgt = d.lut[s]
        if self.conditional:
            cnt = torch.bincount(yy, minlength=2).float()
            w = 1.0 / (cnt[yy] * (cnt > 0).sum())
            dl = (F.cross_entropy(out, tgt, reduction="none") * w).sum()
        else:
            dl = F.cross_entropy(out, tgt)
        logs = {"cls_loss": float(cls.detach()), "disc_loss": float(dl.detach()) * (x.shape[0] / B)}
        gp_w = float(self.p.get("grad_penalty", 0.0))
        if gp_w > 0:
            # Penalización de gradiente del discriminador (DomainBed, DANN): cuadrado del gradiente de la
            # probabilidad de la fuente verdadera respecto de la entrada del discriminador. La entrada se separa
            # del grafo, de modo que el término solo regulariza al discriminador y no llega al tronco.
            with torch.autocast(device_type=fd.device.type, enabled=False):
                fin = fd.detach().float().requires_grad_(True)
                pr = F.softmax(d.mlp(fin).float(), dim=1).gather(1, tgt[:, None]).sum()
                g = torch.autograd.grad(pr, fin, create_graph=True)[0]
                gp = (g ** 2).sum(dim=1).mean()
            dl = dl + gp_w * gp
            logs["grad_pen"] = float(gp.detach()) * (x.shape[0] / B)
        return cls + dl * (x.shape[0] / B), logs


@register
class CDANN(DANN):
    name = "cdann"
    title = "CDANN (adversario de fuente condicionado a la clase)"
    ref = "li2018deep"
    empty_cells = "el discriminador de cada clase solo ve las fuentes con esa clase; invarianza solo entre fuentes compartidas"
    conditional = True
    hp_source = "DomainBed (mismos rangos que DANN; class_balance activado)"
