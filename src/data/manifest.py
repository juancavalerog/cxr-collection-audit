"""Carga de data/manifests/covidx_cxr4_clean.csv en dos modos.

- "oficial": partición oficial de COVIDx CXR-4 (columna `split`; train = dataset menos val y test), la del
  artículo 1. Recuentos 67 863 / 8 473 / 8 482.
- "limpia": columna `split_clean` sin las filas `excluded` (reglas de data/manifests/README.md). Recuentos
  67 932 / 7 664 / 8 479.

Además expone la validación robusta (`val_robust == 1`, 3 407 imágenes BIMCV) y los subconjuntos del test oficial
(`test_subset`: bimcv, ricord, rsna_sin_fuga, rsna_con_fuga). Los índices de fuente son estables: orden alfabético
de las 8 fuentes declaradas en COVIDx, independiente de la partición y del modo.

Orden de filas: dentro de cada partición se conserva el orden del CSV. Para comparar con el artículo 1 (probs.npz,
en el orden de test.txt) use `order_like_txt`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .paths import data_root, project_path

COVIDX_SOURCES = ("actmed", "bimcv", "cohen", "fig1", "ricord", "rsna", "sirm", "stonybrook")
SOURCE_INDEX = {s: i for i, s in enumerate(COVIDX_SOURCES)}
MODES = ("oficial", "limpia")
DEFAULT_CSV = project_path("data", "manifests", "covidx_cxr4_clean.csv")


def load_manifest(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DEFAULT_CSV, dtype={"patient_id": str, "nih_patient_id": str, "test_subset": str,
                                                  "exclude_reason": str})
    df["file"] = df["image_path"].str.split("/").str[-1]
    df["image_id"] = df["file"]
    df["source_idx"] = df["source"].map(SOURCE_INDEX).astype("int64")
    df["label"] = df["label"].astype("int64")
    df["val_robust"] = df["val_robust"].fillna(0).astype("int64")
    df["test_subset"] = df["test_subset"].fillna("")
    df["exclude_reason"] = df["exclude_reason"].fillna("")
    df["nih_patient_id"] = df["nih_patient_id"].fillna("")
    return df


def split_column(mode: str) -> str:
    if mode not in MODES:
        raise ValueError("modo desconocido: %s (use %s)" % (mode, MODES))
    return "split" if mode == "oficial" else "split_clean"


def get_splits(mode: str = "oficial", df: pd.DataFrame | None = None, train_sources=None) -> dict:
    """Devuelve {'train', 'val', 'test'} como DataFrames (índice reiniciado). `train_sources` restringe el train
    (p. ej. ['bimcv'] = régimen controlado); val y test siempre completos, como en el artículo 1."""
    df = load_manifest() if df is None else df
    col = split_column(mode)
    out = {}
    for s in ("train", "val", "test"):
        d = df[df[col] == s]
        if s == "train" and train_sources:
            d = d[d["source"].isin(list(train_sources))]
        out[s] = d.reset_index(drop=True)
    return out


def val_robust(df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = load_manifest() if df is None else df
    return df[df["val_robust"] == 1].reset_index(drop=True)


def test_subsets(df: pd.DataFrame | None = None) -> dict:
    """{nombre: DataFrame} de los subconjuntos del test oficial."""
    df = load_manifest() if df is None else df
    t = df[df["split"] == "test"]
    return {k: g.reset_index(drop=True) for k, g in t.groupby("test_subset") if k}


def get_eval_set(name: str, mode: str = "oficial", df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Conjuntos de evaluación por nombre: val, test, val_robust, test_<subconjunto>."""
    df = load_manifest() if df is None else df
    if name in ("val", "test", "train"):
        return get_splits(mode, df)[name]
    if name == "val_robust":
        return val_robust(df)
    if name.startswith("test_"):
        return test_subsets(df)[name[5:]]
    raise ValueError("conjunto desconocido: %s" % name)


def order_like_txt(d: pd.DataFrame, txt: str = "test.txt") -> pd.DataFrame:
    """Reordena un DataFrame según el orden de data/<txt> de COVIDx (el de probs.npz del artículo 1)."""
    names = [ln.split()[1] for ln in open(data_root() / "data" / txt, encoding="utf-8") if ln.strip()]
    pos = {n: i for i, n in enumerate(names)}
    d = d.copy()
    d["_o"] = d["file"].map(pos)
    if d["_o"].isna().any():
        raise ValueError("hay ficheros que no están en %s" % txt)
    return d.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def counts(mode: str = "oficial", df: pd.DataFrame | None = None) -> dict:
    s = get_splits(mode, df)
    return {k: len(v) for k, v in s.items()}


def summary(df: pd.DataFrame | None = None) -> str:
    df = load_manifest() if df is None else df
    lines = []
    for mode in MODES:
        c = counts(mode, df)
        lines.append("%s: train %d, val %d, test %d" % (mode, c["train"], c["val"], c["test"]))
    lines.append("val_robust: %d" % len(val_robust(df)))
    lines.append("test_subset: " + ", ".join("%s %d" % (k, len(v)) for k, v in test_subsets(df).items()))
    return "\n".join(lines)


def image_paths(d: pd.DataFrame) -> np.ndarray:
    root = data_root()
    return np.array([str(root / p) for p in d["image_path"].values])
