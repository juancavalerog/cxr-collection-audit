"""Métricas: AUC exactas (scikit-learn) y réplica de tf.keras.metrics.AUC (200 umbrales, interpolación), que es la
que miraba la parada temprana del artículo 1 (val_auc_pr)."""
from __future__ import annotations

import numpy as np


def keras_auc(y, p, curve: str = "PR", num_thresholds: int = 200) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    eps = 1e-7
    thr = np.array([0.0 - eps] + [(i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)] + [1.0 + eps])
    pred = p[None, :] > thr[:, None]
    pos = y[None, :]
    tp = (pred * pos).sum(1)
    fp = (pred * (1 - pos)).sum(1)
    fn = ((~pred) * pos).sum(1)
    tn = ((~pred) * (1 - pos)).sum(1)
    if curve == "PR":
        dtp = tp[:-1] - tp[1:]
        pp = tp + fp
        dp = pp[:-1] - pp[1:]
        prec_slope = np.divide(dtp, np.maximum(dp, 0), out=np.zeros_like(dtp), where=np.maximum(dp, 0) != 0)
        intercept = tp[1:] - prec_slope * pp[1:]
        ok = (pp[:-1] > 0) & (pp[1:] > 0)
        ratio = np.where(ok, np.divide(pp[:-1], np.maximum(pp[1:], 0), out=np.ones_like(dp),
                                       where=np.maximum(pp[1:], 0) != 0), 1.0)
        inc = prec_slope * (dtp + intercept * np.log(ratio))
        den = np.maximum(tp[1:] + fn[1:], 0)
        return float(np.sum(np.divide(inc, den, out=np.zeros_like(inc), where=den != 0)))
    tpr = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    fpr = np.divide(fp, fp + tn, out=np.zeros_like(fp), where=(fp + tn) != 0)
    return float(np.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + tpr[1:]) / 2.0))


def exact_metric(name: str, y, p) -> float:
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y).astype(int)
    if len(set(y.tolist())) < 2:
        return float("nan")
    if name == "auc_roc":
        return float(roc_auc_score(y, p))
    if name == "auc_pr":
        return float(average_precision_score(y, p))
    raise ValueError(name)


def selection_metric(name: str, impl: str, y, p) -> float:
    if impl == "keras200":
        return keras_auc(y, p, "PR" if name == "auc_pr" else "ROC")
    return exact_metric(name, y, p)
