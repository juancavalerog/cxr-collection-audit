"""Optimizadores y programación de la tasa.

`TFAdamW` reproduce tf_keras 2.16 AdamW (optimizers/adamw.py y optimizer.py):
    1) decaimiento desacoplado antes del gradiente: w <- w - w * wd * lr
    2) m <- m + (g - m)(1 - b1);  v <- v + (g^2 - v)(1 - b2)
    3) alpha = lr * sqrt(1 - b2^t) / (1 - b1^t);  w <- w - alpha * m / (sqrt(v) + eps), eps = 1e-7
(PyTorch AdamW suma eps a sqrt(v_hat), no a sqrt(v), y usa eps 1e-8 por defecto.)

`warmup_cosine` reproduce WarmupCosine del artículo 1: en el paso t (contador del optimizador antes de
incrementarse, empezando en 0) lr = peak * t / warmup si t < warmup; si no, coseno desde peak hasta min_lr sobre
total - warmup pasos. En el paso 0 la tasa es 0, como en Keras.
"""
from __future__ import annotations

import math

import torch


def warmup_cosine(step: int, peak: float, total: int, warmup: int, min_lr: float = 1e-7) -> float:
    total = max(int(total), 1)
    warmup = max(int(warmup), 1)
    if step < warmup:
        return peak * (step / warmup)
    prog = (step - warmup) / max(1.0, float(total - warmup))
    prog = min(max(prog, 0.0), 1.0)
    return min_lr + 0.5 * (peak - min_lr) * (1.0 + math.cos(math.pi * prog))


class TFAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-7, weight_decay=0.004):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            lr, (b1, b2), eps, wd = g["lr"], g["betas"], g["eps"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if wd:
                    p.sub_(p * (wd * lr))
                st["step"] += 1
                t = st["step"]
                grad = p.grad
                st["m"].add_((grad - st["m"]) * (1 - b1))
                st["v"].add_((grad * grad - st["v"]) * (1 - b2))
                alpha = lr * math.sqrt(1 - b2 ** t) / (1 - b1 ** t)
                p.sub_(st["m"] * alpha / (st["v"].sqrt() + eps))
        return loss


def build_optimizer(groups, cfg: dict, base_lr: float):
    """groups: lista de dicts {'params', 'name', 'lr_mult'}."""
    name = cfg.get("name", "tf_adamw")
    wd = float(cfg.get("weight_decay", 1e-5))
    betas = tuple(cfg.get("betas", (0.9, 0.999)))
    pg = [{"params": g["params"], "name": g["name"], "lr_mult": float(g.get("lr_mult", 1.0)),
           "lr": base_lr * float(g.get("lr_mult", 1.0)),
           "weight_decay": float(g.get("weight_decay", wd))} for g in groups if g["params"]]
    if name == "tf_adamw":
        return TFAdamW(pg, lr=base_lr, betas=betas, eps=float(cfg.get("eps", 1e-7)), weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(pg, lr=base_lr, betas=betas, eps=float(cfg.get("eps", 1e-8)), weight_decay=wd)
    raise ValueError("optimizador desconocido: %s" % name)


def set_lr(opt, lr: float):
    for g in opt.param_groups:
        g["lr"] = lr * g.get("lr_mult", 1.0)
