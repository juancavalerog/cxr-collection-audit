"""MixStyle [zhou2021domain] y mezcla de amplitud de Fourier al estilo FACT [xu2021fourier], ambas solo entre
imágenes de fuentes distintas.

MixStyle: tras las etapas tempranas del tronco (por defecto las etapas 0 y 1 de ConvNeXt, equivalentes a res1 y
res2 de ResNet en el artículo original), con probabilidad p por lote, se normaliza cada mapa con su media y
desviación por canal y se reescala con la mezcla lambda (mu_i, sigma_i) + (1 - lambda)(mu_j, sigma_j),
lambda ~ Beta(alpha, alpha), estadísticos sin gradiente. La pareja j se elige al azar entre los ejemplos del lote
de otra fuente; si no hay ninguno, el ejemplo no se mezcla. Solo actúa en entrenamiento.

Fourier (FACT simplificado): para cada imagen se elige una pareja de otra fuente y se mezcla la amplitud del
espectro, A = (1 - lambda) A_i + lambda A_j con lambda ~ U(0, eta), conservando la fase de i, en la región central
de lado ratio del espectro desplazado (ratio 1: todo el espectro, como el valor por defecto de FACT). La pérdida es
L(x) + L(x_mezcla) + w_c KL simétrica con temperatura T entre las predicciones de las dos vistas. FACT usa un
profesor con media móvil para la consistencia; aquí la referencia es la propia predicción sin gradiente
(simplificación declarada).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.baselines.common import cross_source, early_stage_modules, pick_partners, sample_beta
from src.baselines.registry import Method, register


def mixstyle(x, s, gen, alpha=0.1, eps=1e-6):
    """Devuelve (x mezclado, índice de la pareja, filas con pareja)."""
    B, dt = x.shape[0], x.dtype
    x = x.float()                      # estadísticos en float32 también con autocast bf16
    j, has = pick_partners(cross_source(s), gen)
    mu = x.mean(dim=(2, 3), keepdim=True).detach()
    sig = (x.var(dim=(2, 3), keepdim=True) + eps).sqrt().detach()
    xn = (x - mu) / sig
    lam = sample_beta(alpha, B, gen, x.device).view(B, 1, 1, 1)
    lam = torch.where(has.view(B, 1, 1, 1), lam, torch.ones_like(lam))
    mu2, sig2 = mu[j], sig[j]
    return (xn * (lam * sig + (1 - lam) * sig2) + (lam * mu + (1 - lam) * mu2)).to(dt), j, has


@register
class MixStyle(Method):
    name = "mixstyle"
    title = "MixStyle entre fuentes"
    ref = "zhou2021domain"
    labels = ("clase", "fuente (para elegir parejas)")
    empty_cells = "mezcla estilos entre fuentes sin mirar la clase; no crea combinaciones clase por fuente dirigidas"
    defaults = {"p": 0.5, "alpha": 0.1, "layers": [0, 1]}
    search = {"p": ("choice", [0.3, 0.5, 0.7]), "alpha": ("choice", [0.1, 0.2, 0.3])}
    hp_source = "artículo original (p 0.5, alpha 0.1, tras res1 y res2); rango propio acotado"
    cost = 1.05

    def build(self, tr, net):
        mods = early_stage_modules(net.net, list(self.p["layers"]))

        def hook(module, inp, out):
            ctx = net._ctx
            if ctx is None or not net.training or not ctx.get("apply"):
                return out
            y, ctx["partner"], ctx["has"] = mixstyle(out, ctx["s"], ctx["gen"], float(self.p["alpha"]))
            return y
        self._handles = [m.register_forward_hook(hook) for m in mods]

    def loss(self, tr, net, x, y, s, gen, B):
        apply = bool(torch.rand((), generator=gen, device=x.device) < float(self.p["p"]))
        net._ctx = {"s": s, "gen": gen, "apply": apply}
        try:
            z = net(x)
        finally:
            ctx, net._ctx = net._ctx, None
        self.last_ctx = ctx
        return tr.loss_fn(z.float(), y, reduction="sum") / B, {"mixstyle_apply": float(apply)}


def amplitude_mix(x, s, gen, eta=1.0, ratio=1.0):
    B, C, H, W = x.shape
    j, has = pick_partners(cross_source(s), gen)
    Fx = torch.fft.fft2(x.float())
    amp, pha = Fx.abs(), torch.angle(Fx)
    amp_s = torch.fft.fftshift(amp, dim=(-2, -1))
    lam = (torch.rand(B, generator=gen, device=x.device) * float(eta)).view(B, 1, 1, 1)
    lam = torch.where(has.view(B, 1, 1, 1), lam, torch.zeros_like(lam))
    mixed = (1 - lam) * amp_s + lam * amp_s[j]
    if ratio < 1.0:
        h, w = max(1, int(H * ratio)), max(1, int(W * ratio))
        m = torch.zeros(1, 1, H, W, device=x.device)
        h0, w0 = (H - h) // 2, (W - w) // 2
        m[..., h0:h0 + h, w0:w0 + w] = 1
        mixed = m * mixed + (1 - m) * amp_s
    amp2 = torch.fft.ifftshift(mixed, dim=(-2, -1))
    out = torch.fft.ifft2(amp2 * torch.exp(1j * pha)).real
    return out.to(x.dtype), j, has


def _bin_kl(p_logit, q_logit, T):
    """KL(p || q) entre bernoullis con temperatura; p sin gradiente."""
    p = torch.sigmoid(p_logit / T)
    lq = F.logsigmoid(q_logit / T); lq1 = F.logsigmoid(-q_logit / T)
    lp = F.logsigmoid(p_logit / T); lp1 = F.logsigmoid(-p_logit / T)
    return (p * (lp - lq) + (1 - p) * (lp1 - lq1))


@register
class FourierMix(Method):
    name = "fourier"
    title = "Mezcla de amplitud de Fourier entre fuentes (FACT sin profesor)"
    ref = "xu2021fourier"
    labels = ("clase", "fuente (para elegir parejas)")
    empty_cells = "transfiere el espectro de amplitud de otra fuente sin mirar la clase; no dirigido por la tabla"
    defaults = {"eta": 1.0, "ratio": 1.0, "cons_weight": 2.0, "T": 10.0}
    search = {"eta": ("choice", [0.5, 1.0]), "ratio": ("choice", [0.5, 1.0]), "cons_weight": ("choice", [0.5, 1.0, 2.0, 4.0])}
    hp_source = "FACT (eta 1, ratio 1, peso de consistencia 2, T 10); rango propio acotado"
    cost = 2.0

    def loss(self, tr, net, x, y, s, gen, B):
        xa, j, has = amplitude_mix(x, s, gen, float(self.p["eta"]), float(self.p["ratio"]))
        self.last_partner, self.last_has = j, has
        z = net(x).float()
        za = net(xa).float()
        n = x.shape[0]
        cls = (tr.loss_fn(z, y, reduction="sum") + tr.loss_fn(za, y, reduction="sum")) / B
        T = float(self.p["T"])
        cons = 0.5 * (_bin_kl(z.detach(), za, T) + _bin_kl(za.detach(), z, T)).mean()
        return cls + float(self.p["cons_weight"]) * cons * (n / B), {"cons": float(cons.detach())}
