"""Datos sintéticos diminutos para las pruebas del bucle de CSD-Net en CPU (tests/test_csdnet_loop.py).

Cada imagen (3 x R x R en [0, 1]) tiene dos "pulmones" elípticos y un "corazón"; la enfermedad añade una
opacidad dentro de un pulmón y la fuente añade una firma de adquisición (ganancia global y ruido de alta
frecuencia en la periferia). Devuelve también la máscara del profesor (2 canales) y su calidad. La tabla clase
por fuente se fija con `spec` e incluye celdas vacías, como COVIDx.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# fuente: (negativas, positivas) en train y val
DEFAULT_SPEC = {
    "train": {"s0": (14, 14), "s1": (20, 0), "s2": (0, 20), "s3": (5, 5)},
    "val": {"s0": (14, 14), "s1": (12, 0), "s2": (0, 12), "s3": (3, 3)},
}


def synthetic_csd_frame(split: str, spec=None, seed: int = 0) -> pd.DataFrame:
    spec = (spec or DEFAULT_SPEC)[split]
    rows = []
    for si, name in enumerate(sorted(DEFAULT_SPEC["train"].keys() | spec.keys())):
        n0, n1 = spec.get(name, (0, 0))
        for lab, n in ((0, n0), (1, n1)):
            for k in range(n):
                rows.append({"image_id": "%s_%s_%d_%03d" % (split, name, lab, k), "image_path": "", "label": lab,
                             "source": name, "source_idx": si, "split": split,
                             "patient_id": "%s_%s_p%d_%d" % (split, name, lab, k // 2)})
    df = pd.DataFrame(rows)
    df["val_robust"] = (df["source"] == "s0").astype(int)
    return df.reset_index(drop=True)


def _ellipse(res, cx, cy, rx, ry):
    yy, xx = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res), indexing="ij")
    return (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1


class SyntheticCSD(Dataset):
    def __init__(self, df: pd.DataFrame, res: int = 64, seed: int = 0, signal: float = 0.35):
        self.df = df.reset_index(drop=True)
        self.res, self.seed, self.signal = res, seed, signal
        self.labels = self.df["label"].astype(int).to_numpy()
        self.sources = self.df["source_idx"].astype(int).to_numpy()
        self.patients = self.df["patient_id"].astype(str).tolist()
        self.ids = self.df["image_id"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        h = int(hashlib.md5(("%s|%s" % (self.seed, r.image_id)).encode()).hexdigest()[:8], 16)
        g = np.random.default_rng(h)
        R = self.res
        jx, jy = g.normal(0, 0.03, 2)
        lungs = _ellipse(R, -0.38 + jx, jy, 0.24, 0.5) | _ellipse(R, 0.38 + jx, jy, 0.24, 0.5)
        heart = _ellipse(R, 0.05 + jx, 0.25 + jy, 0.12, 0.22) & ~lungs
        x = 0.25 + 0.05 * g.standard_normal((R, R))
        x = x + 0.35 * lungs + 0.2 * heart
        if r.label == 1:
            blob = _ellipse(R, -0.38 + jx + g.normal(0, 0.05), 0.15 + jy, 0.14, 0.18)
            x = x + self.signal * blob
        s = int(r.source_idx)
        gain = 1.0 + 0.15 * (s - 1.5)
        x = x * gain
        per = np.ones((R, R), bool)
        b = max(2, R // 8)
        per[b:-b, b:-b] = False
        x = x + per * (0.04 * (s + 1)) * g.standard_normal((R, R))
        x = np.clip(x, 0, 1).astype(np.float32)
        mask = np.stack([lungs, heart]).astype(np.float32)
        return {"image": torch.from_numpy(np.repeat(x[None], 3, 0)), "mask": torch.from_numpy(mask),
                "quality": torch.tensor(1.0), "label": torch.tensor(float(r.label)), "source": torch.tensor(s),
                "patient": str(r.patient_id), "image_id": str(r.image_id), "index": torch.tensor(i)}
