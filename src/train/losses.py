"""Pérdida focal binaria igual a la del artículo 1.

tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, alpha=0.5, gamma=2.0, label_smoothing=0.0)
sobre la salida sigmoide de la última Dense. tf_keras recupera el logit de la sigmoide (_get_logits), así que la
entropía cruzada se calcula con logits (sigmoid_cross_entropy_with_logits) y el factor focal con la probabilidad:

    p = sigmoid(z),  p_t = y p + (1 - y)(1 - p),  w = y alpha + (1 - y)(1 - alpha)
    L = mean_i  w (1 - p_t)^gamma  BCE_logits(z, y)

Con alpha = 0.5 el peso es 0.5 constante. Admite etiquetas blandas (mixup).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss_keras(logits, target, alpha: float = 0.5, gamma: float = 2.0, apply_class_balancing: bool = True,
                     label_smoothing: float = 0.0, reduction: str = "mean"):
    z = logits.float().reshape(-1)
    y = target.float().reshape(-1)
    if label_smoothing:
        y = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
    p = torch.sigmoid(z)
    p_t = y * p + (1 - y) * (1 - p)
    bce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    loss = torch.pow(1.0 - p_t, gamma) * bce
    if apply_class_balancing:
        loss = (y * alpha + (1 - y) * (1 - alpha)) * loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def build_loss(cfg: dict):
    name = cfg.get("name", "focal")
    if name == "focal":
        kw = dict(alpha=cfg.get("alpha", 0.5), gamma=cfg.get("gamma", 2.0),
                  apply_class_balancing=cfg.get("apply_class_balancing", True),
                  label_smoothing=cfg.get("label_smoothing", 0.0))
        return lambda z, y, reduction="mean": focal_loss_keras(z, y, reduction=reduction, **kw)
    if name == "bce":
        return lambda z, y, reduction="mean": F.binary_cross_entropy_with_logits(z.float().reshape(-1), y.float().reshape(-1),
                                                                                reduction=reduction)
    raise ValueError("pérdida desconocida: %s" % name)
