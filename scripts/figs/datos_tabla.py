"""Datos comunes de las figuras 1 y 3: tabla clase por fuente del train limpio, composición de los lotes y plan."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FUENTES = ["actmed", "bimcv", "cohen", "fig1", "ricord", "rsna", "sirm", "stonybrook"]
NOMBRE = {"actmed": "Actualmed", "bimcv": "BIMCV", "cohen": "Cohen", "fig1": "Figure 1", "ricord": "RICORD",
          "rsna": "RSNA", "sirm": "SIRM", "stonybrook": "Stony Brook"}
MANIFIESTO = ROOT / "data" / "manifests" / "covidx_cxr4_clean.csv"


def tabla_train(manifiesto=MANIFIESTO):
    """Devuelve (tabla 2 x K de imágenes, tabla 2 x K de pacientes, etiquetas, fuentes) del train limpio."""
    d = pd.read_csv(manifiesto, low_memory=False)
    d = d[d["split_clean"] == "train"]
    idx = {s: i for i, s in enumerate(FUENTES)}
    desconocidas = set(d["source"].unique()) - set(idx)
    if desconocidas:
        raise ValueError("fuentes no previstas en el manifiesto: %s" % sorted(desconocidas))
    y = d["label"].astype(int).to_numpy()
    s = d["source"].map(idx).astype(int).to_numpy()
    n = np.zeros((2, len(FUENTES)), dtype=np.int64)
    np.add.at(n, (y, s), 1)
    pac = np.zeros_like(n)
    for (yy, ss), g in d.assign(_y=y, _s=s).groupby(["_y", "_s"]):
        pac[yy, ss] = g["patient_id"].astype(str).nunique()
    return n, pac, y, s


def plan_vigente(y, s, lote=64, cuota_fuente=2, cuota_mixta=4, lam=1.0):
    """Tabla de composición de los lotes y plan de enrutamiento vigente (docs/03_arquitectura.md, Tabla 2b)."""
    from src.train.csdnet_loop import batch_composition_table
    t = batch_composition_table(y, s, len(FUENTES), FUENTES, lote, cuota_fuente, cuota_mixta, seed=0, n_batches=2000)
    return t, t.routing(lam)
