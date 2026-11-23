"""Datos y modelo diminutos para las pruebas del entrenador en CPU."""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def synthetic_frame(n, n_sources=3, seed=0, split="train"):
    rng = np.random.default_rng([seed, zlib.crc32(split.encode()) % 1000])   # no hash(): cambia con PYTHONHASHSEED
    y = rng.integers(0, 2, n)
    y[:2] = [0, 1]
    s = rng.integers(0, n_sources, n)
    return pd.DataFrame({"image_id": ["%s_%05d" % (split, i) for i in range(n)], "image_path": "",
                         "label": y, "source_idx": s, "source": ["s%d" % v for v in s],
                         "patient_id": ["p%d" % (i // 2) for i in range(n)], "split": split})


class SyntheticCXR(Dataset):
    """Imágenes 3 x R x R en [0, scale]: ruido más un parche central cuya intensidad depende de la clase."""

    def __init__(self, df, res=32, scale=255.0, seed=0):
        self.df, self.res, self.scale, self.seed = df.reset_index(drop=True), res, scale, seed
        self.labels = self.df["label"].to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        g = np.random.default_rng([self.seed, i, len(self.df)])
        x = g.normal(0.5, 0.15, (1, self.res, self.res)).astype(np.float32)
        c = self.res // 4
        x[:, c:3 * c, c:3 * c] += 0.25 * (1 if r.label else -1)
        x = np.clip(np.repeat(x, 3, 0), 0, 1) * self.scale
        return {"image": torch.from_numpy(x), "label": torch.tensor(float(r.label)),
                "source": torch.tensor(int(r.source_idx)), "patient": str(r.patient_id),
                "image_id": str(r.image_id), "index": torch.tensor(i)}


class TinyNet(nn.Module):
    """Misma interfaz que Paper1ConvNeXt (backbone, cabeza fc1-ln-drop-fc2, l2_penalty, backbone_norm_params)."""

    def __init__(self, input_scale=255.0, width=8, hidden=16, dropout=0.5, l2_hidden=1e-2):
        super().__init__()
        self.input_scale = input_scale
        self.backbone = nn.Sequential(nn.Conv2d(3, width, 3, stride=2, padding=1), nn.GELU(),
                                      nn.Conv2d(width, 2 * width, 3, stride=2, padding=1))
        self.bnorm = nn.LayerNorm(2 * width)
        self.backbone.add_module("norm_marker", nn.Identity())
        self.fc1 = nn.Linear(2 * width, hidden)
        self.ln = nn.LayerNorm(hidden, eps=1e-3)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)
        self.l2_hidden = l2_hidden

    def pooled(self, x):
        f = self.backbone(x / self.input_scale).mean(dim=(2, 3))
        return self.bnorm(f)

    def forward(self, x):
        h = F.relu(self.fc1(self.pooled(x)))
        return self.fc2(self.drop(self.ln(h))).squeeze(-1)

    def backbone_norm_params(self):
        return list(self.bnorm.named_parameters())

    def l2_penalty(self):
        return self.l2_hidden * self.fc1.weight.pow(2).sum()

    def backbone_parameters(self):
        return list(self.backbone.parameters()) + list(self.bnorm.parameters())
