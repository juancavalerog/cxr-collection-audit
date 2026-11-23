"""DFA con etiquetas de fuente [lee2021learning] y reimplementación aproximada de Lai et al. 2026
[lai2026mitigating] con la fuente como atributo de sesgo.

Estructura de DFA: dos codificadores con el mismo tronco (contenido E_i = tronco común; sesgo E_b = segunda
copia, que cuenta como tronco para la tasa y la congelación de la fase A), rasgos z_i y z_b, y dos clasificadores
sobre la concatenación [z_i, z_b]: C_i predice la clase y C_b predice la fuente. Cada clasificador recibe sin
gradiente la mitad que no es suya. En inferencia el logit es C_i([z_i, z_b]).

Diferencia con DFA original: DFA infiere el sesgo entrenando C_b con GCE sobre la clase y pondera los ejemplos con
la dificultad relativa. Aquí la fuente es conocida y se usa como etiqueta de sesgo: C_b se entrena con entropía
cruzada sobre la fuente, y el peso de la pérdida de contenido es w(y, s) = 1 / P(y | s) de la tabla de
entrenamiento, recortado a peso_max y normalizado a media 1 en el train (los ejemplos de celdas escasas pesan más).
Desde swap_start_epoch (en la fase B) se intercambian los vectores de sesgo dentro del lote: z_swap = [z_i,
z_b[pi]], con pareja pi de otra fuente cuando existe. El objetivo de C_i sobre z_swap es la clase del contenido
(y_i, no la de la pareja) y el de C_b es la fuente de la pareja s[pi]. Pérdida:
L = L_cont + lambda_dis_align L_sesgo + lambda_swap (L_cont_swap + lambda_swap_align L_sesgo_swap).

Variante Lai et al. (`lai2026`): la misma estructura con la rama de sesgo entrenada con GCE sobre el atributo
(aquí la fuente), permutación aleatoria del lote para el intercambio y sin pesos por celda. Es una
reimplementación a partir de la descripción del resumen y de la lectura del código (sin copiar código, el
repositorio no tiene licencia); queda pendiente de contrastar con el texto completo del artículo.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from src.baselines.common import N_SOURCES_MAX, Head, cross_source, gce_loss, n_features, pick_partners
from src.baselines.registry import Method, register


def swap_bias(zi, zb, y, s, gen, cross=True):
    """Intercambio de DFA. Devuelve (z_swap, objetivo de clase, objetivo de fuente, pareja)."""
    B = zi.shape[0]
    if cross:
        j, _ = pick_partners(cross_source(s), gen)
    else:
        j = torch.randperm(B, generator=gen, device=zi.device)
    return torch.cat([zi, zb[j]], dim=1), y, s[j], j


@register
class DFAFuente(Method):
    name = "dfa_fuente"
    title = "DFA con la fuente como etiqueta de sesgo"
    ref = "lee2021learning"
    labels = ("clase", "fuente (etiqueta de sesgo)")
    empty_cells = "el intercambio crea combinaciones contenido por firma no dirigidas (pareja de otra fuente al azar); no usa la tabla para elegir el destino"
    defaults = {"lambda_dis_align": 2.0, "lambda_swap_align": 2.0, "lambda_swap": 0.1, "swap_start_epoch": 3,
                "peso": "inverso_p_y_dado_s", "peso_max": 10.0, "bias_loss": "ce", "gce_q": 0.7, "swap": "cross_source"}
    search = {"lambda_dis_align": ("choice", [1.0, 2.0, 5.0, 10.0]), "lambda_swap_align": ("choice", [1.0, 2.0, 5.0, 10.0]),
              "lambda_swap": ("choice", [0.1, 0.5, 1.0]), "swap_start_epoch": ("choice", [1, 3, 5, 10]),
              "peso": ("choice", ["inverso_p_y_dado_s", "ninguno"])}
    hp_source = "repositorio de DFA (bFFHQ: lambda_swap 0.1, lambda_dis_align 2, lambda_swap_align 2; CIFAR10C 1 a 5; curr_step 10 000 pasos, aquí en épocas de la fase B)"
    cost = 2.1
    disk = 2.0

    def build(self, tr, net):
        nf = n_features(net.net)
        bias = copy.deepcopy(net.net)
        for m in bias.modules():
            if hasattr(m, "_forward_hooks"):
                m._forward_hooks.clear()
        net.aux["bias_net"] = bias
        net._extra_backbone = [bias.backbone]
        drop = float(tr.cfg["model"].get("dropout", 0.5))
        net.aux["c_i"] = Head(2 * nf, 1, dropout=drop)
        net.aux["c_b"] = Head(2 * nf, N_SOURCES_MAX, dropout=drop)
        t = tr.cells.astype(np.float64)
        p_y_s = t / np.maximum(t.sum(0, keepdims=True), 1)
        w = np.where(t > 0, 1.0 / np.maximum(p_y_s, 1e-12), 0.0)
        w = np.minimum(w, float(self.p["peso_max"]))
        mean_w = (w * t).sum() / t.sum()
        w = w / mean_w if self.p["peso"] == "inverso_p_y_dado_s" else (t > 0).astype(np.float64)
        net.aux["c_i"].register_buffer("w_cell", torch.tensor(w, dtype=torch.float32))
        net._fwd = lambda m, x: m.aux["c_i"](torch.cat([m.net.pooled(x), m.aux["bias_net"].pooled(x)], dim=1))

    def swap_active(self, tr):
        ph = tr.cfg["train"]["phases"]
        last = len(ph) - 1
        return tr.state["phase"] == last and tr.state["epoch"] >= int(self.p["swap_start_epoch"]) or tr.state["phase"] > last

    def _bias_loss(self, logits, s):
        if self.p["bias_loss"] == "gce":
            return gce_loss(logits, s, float(self.p["gce_q"]), reduction="none")
        return F.cross_entropy(logits.float(), s, reduction="none")

    def loss(self, tr, net, x, y, s, gen, B):
        zi = net.net.pooled(x)
        zb = net.aux["bias_net"].pooled(x)
        ci, cb = net.aux["c_i"], net.aux["c_b"]
        yy = (y > 0.5).long()
        w = ci.w_cell[yy, s]
        n = x.shape[0]
        lc = (w * tr.loss_fn(ci(torch.cat([zi, zb.detach()], 1)).float(), y, reduction="none")).sum() / B
        lb = self._bias_loss(cb(torch.cat([zi.detach(), zb], 1)), s).sum() / B
        loss = lc + float(self.p["lambda_dis_align"]) * lb
        logs = {"dfa_cont": float(lc.detach()) * B / n, "dfa_sesgo": float(lb.detach()) * B / n}
        if self.swap_active(tr):
            zs, ty, ts, j = swap_bias(zi, zb.detach(), y, s, gen, self.p["swap"] == "cross_source")
            lcs = (w * tr.loss_fn(ci(zs).float(), ty, reduction="none")).sum() / B
            zs_b = torch.cat([zi.detach(), zb[j]], 1)
            lbs = self._bias_loss(cb(zs_b), ts).sum() / B
            loss = loss + float(self.p["lambda_swap"]) * (lcs + float(self.p["lambda_swap_align"]) * lbs)
            logs["dfa_swap"] = float(lcs.detach()) * B / n
        return loss, logs


@register
class Lai2026(DFAFuente):
    name = "lai2026"
    title = "Lai et al. 2026 reimplementado (fuente como atributo de sesgo)"
    ref = "lai2026mitigating"
    status = "aproximacion"
    defaults = dict(DFAFuente.defaults, bias_loss="gce", swap="random", peso="ninguno")
    hp_source = "valores de DFA a falta del texto completo (lambda_swap y swap_epoch del main.py del repositorio por verificar)"
