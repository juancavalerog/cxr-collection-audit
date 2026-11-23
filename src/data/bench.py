"""Manifiestos de los benchmarks del plan B (B2, B3a y COVIDGR): contrato de datos de PLAN.md Section 13.7.

Cada manifiesto es un CSV en data/manifests/bench/<id>/... con un JSON del mismo nombre (extensión .json). Columnas
obligatorias del CSV:

- `dataset`: conjunto de datos de origen de la imagen (nih_cxr14, chexpert_small, tb_rahman, tbx11k, ...).
- `source`: fuente en el sentido del benchmark (su posición en `meta["fuentes"]` es el índice de fuente).
- `image_path`: ruta relativa a la raíz indicada en `raiz`.
- `raiz`: `datos` (relativa a la raíz de datos, src/data/paths.py:data_root) o `proyecto` (relativa a paper_new).
- `patient_id`, `label` (0 o 1), `split` (`train`, `val`, `test` o `fuera`) e `image_id` (único en el manifiesto).
- una columna 0/1 por cada subconjunto de evaluación de `meta["subconjuntos"]`.

Columnas opcionales que los constructores añaden para la auditoría: `reserva` (reserva de pacientes de la que sale la
fila), `view` (AP o PA) y otras descriptivas.

Claves obligatorias del JSON: `benchmark`, `fuentes` (lista ordenada), `fuentes_mixtas`, `rho` (número o null),
`semilla` (entero o null), `subconjuntos` (lista) y `seleccion` (nombre de la partición o del subconjunto que se usa
para seleccionar modelo; `val` o uno de `subconjuntos`).

`load_bench(ruta)` devuelve `(frames, meta)`. `frames` tiene `train`, `val`, `test` (filas con ese `split`) y un
DataFrame por subconjunto, todos con índice reiniciado y con las columnas que usa el entrenador (`image_id`,
`patient_id`, `label`, `source`, `source_idx`, `image_path`) más `raiz`, `image_path_rel` (la ruta del CSV) y
`abs_path`. Por defecto `image_path` se sustituye por la ruta absoluta resuelta en la máquina que llama (VM de
trabajo, contenedor o anfitrión), de modo que CXRDataset, la caché y las máscaras la abren sin más; con
`absolute_paths=False` se conserva la relativa. Las filas `fuera` solo aparecen en los subconjuntos que las marcan.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .paths import PROJECT_ROOT, data_root

REQUIRED_COLS = ("dataset", "source", "image_path", "raiz", "patient_id", "label", "split", "image_id")
REQUIRED_META = ("benchmark", "fuentes", "fuentes_mixtas", "rho", "semilla", "subconjuntos", "seleccion")
SPLITS = ("train", "val", "test", "fuera")
RAICES = ("datos", "proyecto")
BENCH_DIR = PROJECT_ROOT / "data" / "manifests" / "bench"


class ContratoError(ValueError):
    """El manifiesto o su JSON no cumplen el contrato de PLAN.md Section 13.7."""


def meta_path(csv_path) -> Path:
    return Path(csv_path).with_suffix(".json")


def read_meta(csv_path) -> dict:
    p = meta_path(csv_path)
    if not p.exists():
        raise ContratoError("falta el JSON del manifiesto: %s" % p)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def read_csv(csv_path) -> pd.DataFrame:
    return pd.read_csv(csv_path, dtype={"patient_id": str, "image_id": str, "image_path": str, "source": str,
                                        "dataset": str, "split": str, "raiz": str}, keep_default_na=False)


def resolve_row_path(rel: str, raiz: str) -> Path:
    if raiz == "datos":
        return data_root() / rel
    if raiz == "proyecto":
        return PROJECT_ROOT / rel
    raise ContratoError("raiz desconocida: %r" % raiz)


def abs_paths(df: pd.DataFrame) -> list[str]:
    droot, proot = data_root(), PROJECT_ROOT
    return [str((droot if r == "datos" else proot) / p) for p, r in zip(df["image_path"], df["raiz"])]


def validate(df: pd.DataFrame, meta: dict, check_files: bool = False) -> list[str]:
    """Lista de incumplimientos del contrato (vacía si lo cumple). Con `check_files` comprueba además que cada
    imagen existe en disco."""
    err = []
    for k in REQUIRED_META:
        if k not in meta:
            err.append("JSON sin la clave %s" % k)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            err.append("CSV sin la columna %s" % c)
    if err:
        return err
    fuentes = list(meta["fuentes"])
    if len(set(fuentes)) != len(fuentes) or not fuentes:
        err.append("lista de fuentes vacía o con repetidos")
    if not set(meta["fuentes_mixtas"]) <= set(fuentes):
        err.append("fuentes_mixtas fuera de la lista de fuentes")
    extra = sorted(set(df["source"]) - set(fuentes))
    if extra:
        err.append("fuentes del CSV que no están en el JSON: %s" % extra)
    if df["image_id"].duplicated().any():
        err.append("image_id repetido (%d)" % int(df["image_id"].duplicated().sum()))
    if (df["image_id"] == "").any() or (df["patient_id"] == "").any() or (df["image_path"] == "").any():
        err.append("image_id, patient_id o image_path vacíos")
    bad = sorted(set(df["split"]) - set(SPLITS))
    if bad:
        err.append("valores de split no admitidos: %s" % bad)
    bad = sorted(set(df["raiz"]) - set(RAICES))
    if bad:
        err.append("valores de raiz no admitidos: %s" % bad)
    lab = pd.to_numeric(df["label"], errors="coerce")
    if lab.isna().any() or not set(lab.astype(int)) <= {0, 1}:
        err.append("label fuera de {0, 1}")
    for s in meta["subconjuntos"]:
        if s in ("train", "val", "test", "fuera"):
            err.append("subconjunto con nombre reservado: %s" % s)
            continue
        if s not in df.columns:
            err.append("falta la columna del subconjunto %s" % s)
            continue
        v = pd.to_numeric(df[s], errors="coerce")
        if v.isna().any() or not set(v.astype(int)) <= {0, 1}:
            err.append("columna %s no es 0/1" % s)
        elif int(v.sum()) == 0:
            err.append("subconjunto %s vacío" % s)
    if meta["seleccion"] not in ("val",) + tuple(meta["subconjuntos"]):
        err.append("seleccion %r no es val ni un subconjunto" % meta["seleccion"])
    if meta["rho"] is not None and not (0.0 <= float(meta["rho"]) <= 1.0):
        err.append("rho fuera de [0, 1]")
    if (df["split"] == "train").any() and df.loc[df["split"] == "train", "label"].astype(int).nunique() < 2:
        err.append("el entrenamiento no tiene las dos clases")
    if check_files and not err:
        miss = [p for p in abs_paths(df) if not Path(p).is_file()]
        if miss:
            err.append("%d imágenes inexistentes (p. ej. %s)" % (len(miss), miss[0]))
    return err


def load_bench(path, absolute_paths: bool = True, check: bool = True, check_files: bool = False):
    """Carga un manifiesto del plan B y su JSON. Devuelve (frames, meta)."""
    path = Path(path)
    df = read_csv(path)
    meta = read_meta(path)
    if check:
        err = validate(df, meta, check_files=check_files)
        if err:
            raise ContratoError("%s: %s" % (path, "; ".join(err)))
    index = {s: i for i, s in enumerate(meta["fuentes"])}
    df["label"] = df["label"].astype("int64")
    df["source_idx"] = df["source"].map(index).astype("int64")
    for s in meta["subconjuntos"]:
        df[s] = df[s].astype("int64")
    df["image_path_rel"] = df["image_path"]
    df["abs_path"] = abs_paths(df)
    if absolute_paths:
        df["image_path"] = df["abs_path"]
    frames = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    for s in meta["subconjuntos"]:
        frames[s] = df[df[s] == 1].reset_index(drop=True)
    meta = dict(meta)
    meta["ruta"] = str(path)
    return frames, meta


def list_manifests(root=None, bench: str | None = None) -> list[Path]:
    """Todos los manifiestos (CSV con JSON hermano) bajo data/manifests/bench[/<bench>]."""
    root = Path(root) if root else BENCH_DIR
    if bench:
        root = root / bench
    return sorted(p for p in root.rglob("*.csv") if meta_path(p).exists())


def write_bench(df: pd.DataFrame, meta: dict, csv_path, check: bool = True) -> Path:
    """Escribe CSV y JSON tras validar el contrato (las columnas obligatorias van primero)."""
    csv_path = Path(csv_path)
    if check:
        err = validate(df, meta)
        if err:
            raise ContratoError("%s: %s" % (csv_path, "; ".join(err)))
    cols = list(REQUIRED_COLS) + [s for s in meta["subconjuntos"]]
    cols += [c for c in df.columns if c not in cols]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(csv_path, index=False)
    with open(meta_path(csv_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return csv_path


def bench_images(path) -> pd.DataFrame:
    """Imágenes distintas de un manifiesto o de todos los manifiestos bajo un directorio (una fila por image_id),
    con image_path absoluto resuelto en esta máquina, image_path_rel, raiz, dataset y source. Es la lista que usan la
    caché y las máscaras: cada imagen se procesa una sola vez aunque aparezca en varios niveles y semillas."""
    path = Path(path)
    mans = [path] if path.is_file() else list_manifests(path)
    if not mans:
        raise FileNotFoundError("no hay manifiestos en %s" % path)
    cols = ["image_id", "image_path", "raiz", "dataset", "source"]
    d = pd.concat([read_csv(p)[cols] for p in mans], ignore_index=True)
    d = d.drop_duplicates(["image_id", "image_path", "raiz"])
    if d.image_id.duplicated().any():
        raise ContratoError("image_id con rutas distintas entre manifiestos de %s" % path)
    d["image_path_rel"] = d["image_path"]
    d["image_path"] = abs_paths(d)
    return d.sort_values("image_id").reset_index(drop=True)
