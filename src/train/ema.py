"""Media móvil exponencial de los pesos.

Modo "tfkeras_exacta" (tf_keras 2.16 con use_ema=True y sin ema_overwrite_frequency; docs/00_inventario.md,
Section 4.3): la media se crea al construir el optimizador de la fase B con el valor de los pesos entrenables en
ese momento; tras cada paso del optimizador, avg <- m * avg + (1 - m) * w con m fijo (0.9999), sin corrección de
sesgo ni calentamiento del momento. Al terminar, finalize_variable_values sobrescribe los pesos entrenables con la
media. Tras T pasos, el estado inicial conserva un peso m^T en la media.

Modo "estandar": inicializada igual, con momento efectivo min(m, (1 + t) / (10 + t)) (calentamiento habitual en
PyTorch, p. ej. timm ModelEmaV3 con warmup), sobre todos los parámetros de coma flotante y los buffers.
"""
from __future__ import annotations

import torch


class WeightEMA:
    def __init__(self, model: torch.nn.Module, momentum: float = 0.9999, mode: str = "tfkeras_exacta",
                 params: list | None = None):
        if mode not in ("tfkeras_exacta", "estandar"):
            raise ValueError("modo de EMA desconocido: %s" % mode)
        self.mode, self.momentum, self.t = mode, float(momentum), 0
        named = dict(model.named_parameters())
        if params is None:
            if mode == "tfkeras_exacta":
                self.names = [n for n, p in named.items() if p.requires_grad]
            else:
                self.names = [n for n, p in named.items() if p.dtype.is_floating_point]
        else:
            ids = {id(p) for p in params}
            self.names = [n for n, p in named.items() if id(p) in ids]
        self.shadow = {n: named[n].detach().clone().float() for n in self.names}

    def current_momentum(self):
        if self.mode == "estandar":
            return min(self.momentum, (1.0 + self.t) / (10.0 + self.t))
        return self.momentum

    @torch.no_grad()
    def update(self, model):
        m = self.current_momentum()
        named = dict(model.named_parameters())
        for n in self.names:
            self.shadow[n].mul_(m).add_(named[n].detach().float(), alpha=1.0 - m)
        self.t += 1

    def ema_state_dict(self, model) -> dict:
        """state_dict del modelo con los parámetros seguidos sustituidos por la media (el resto sin cambios)."""
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for n in self.names:
            sd[n] = self.shadow[n].to(sd[n].dtype).clone()
        return sd

    @torch.no_grad()
    def copy_to(self, model):
        named = dict(model.named_parameters())
        for n in self.names:
            named[n].copy_(self.shadow[n].to(named[n].dtype))

    def state_dict(self):
        return {"mode": self.mode, "momentum": self.momentum, "t": self.t, "names": list(self.names),
                "shadow": {k: v.clone() for k, v in self.shadow.items()}}

    def load_state_dict(self, st):
        self.mode, self.momentum, self.t = st["mode"], st["momentum"], st["t"]
        self.names = list(st["names"])
        self.shadow = {k: v.clone() for k, v in st["shadow"].items()}
