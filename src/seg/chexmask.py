"""Lectura de CheXmask 1.0.0 [gaggion2024chexmask] y emparejamiento con NIH ChestX-ray14 y CheXpert en disco.

Formato real (repositorio ngaggion/CheXmask-Database: Example.ipynb, DataPostprocessing/utils.py y las tablas de
TechnicalValidation/PhysicianAnnotations/DataSubsets; comprobado el 12-11-2026):
- Un CSV por dataset en OriginalResolution/ (ChestX-Ray8.csv, CheXpert.csv, VinDr-CXR.csv, ...), posiblemente
  comprimido (.csv.gz). Columnas: identificador (cambia por dataset: "Image Index" en ChestX-Ray8, p. ej.
  00028162_002.png; "Path" en CheXpert, p. ej. train/patient18591/study2/view1_frontal.jpg; "image_id" en
  VinDr-CXR), "Dice RCA (Mean)", "Dice RCA (Max)", "Landmarks", "Left Lung", "Right Lung", "Heart", "Height",
  "Width".
- RLE: pares "inicio longitud" separados por espacios, con inicio contado desde 1, sobre la máscara aplanada en
  orden de filas (C) de tamaño Height x Width (get_mask_from_RLE: start -= 1; mask[start:start+length] = 255).
- Calidad: los autores recomiendan Dice RCA (Mean) >= 0.7 (docs/02_datos.md, Section 1.4).
- Las máscaras de CheXpert están a la resolución original (p. ej. 2320 x 2828); en disco está CheXpert-small
  (lado corto de unos 320), así que la máscara se reduce al tamaño de la imagen con vecino más próximo y se
  comprueba que la relación de aspecto coincide (tolerancia del 3 %).

Funciones principales: rle_decode, rle_encode, read_chexmask (por bloques), match_manifest, decode_row.
Lo que no se ha podido comprobar en la VM: los CSV completos (aún no descargados); en particular el prefijo
exacto de "Path" en el CSV de resolución original de CheXpert, que `chexpert_key` normaliza en cualquier caso
desde "train/" o "valid/".
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
import pandas as pd

ID_COLUMNS = {"chestx-ray8": "Image Index", "chexpert": "Path", "vindr-cxr": "image_id"}
RLE_COLUMNS = ("Left Lung", "Right Lung", "Heart")
DICE_MEAN = "Dice RCA (Mean)"


def rle_decode(rle, height: int, width: int) -> np.ndarray:
    """RLE de CheXmask a máscara uint8 {0, 1} de forma (height, width). Cadena vacía o NaN: máscara vacía."""
    out = np.zeros(int(height) * int(width), dtype=np.uint8)
    if rle is None or (isinstance(rle, float) and np.isnan(rle)):
        return out.reshape(int(height), int(width))
    runs = np.asarray(str(rle).split(), dtype=np.int64)
    if runs.size % 2:
        raise ValueError("RLE con número impar de valores")
    starts, lengths = runs[0::2] - 1, runs[1::2]
    if runs.size and (starts.min() < 0 or (starts + lengths).max() > out.size):
        raise ValueError("RLE fuera de la máscara")
    for s, l in zip(starts, lengths):
        out[s:s + l] = 1
    return out.reshape(int(height), int(width))


def rle_encode(mask: np.ndarray) -> str:
    """Inversa de rle_decode (misma convención que get_RLE_from_mask de CheXmask)."""
    p = np.concatenate([[0], (np.asarray(mask).ravel() > 0).astype(np.int8), [0]])
    runs = np.nonzero(p[1:] != p[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(int(x)) for x in runs)


def detect_id_column(columns) -> str:
    for c in ("Image Index", "Path", "image_id", "ImageID", "dicom_id"):
        if c in columns:
            return c
    raise KeyError("no se encuentra la columna de identificador en %s" % list(columns))


def nih_key(s: str) -> str:
    return os.path.basename(str(s))


def chexpert_key(s: str) -> str:
    """Normaliza rutas de CheXpert a 'train/patientX/studyY/viewZ.jpg' (o 'valid/...')."""
    s = str(s).replace("\\", "/")
    for tag in ("train/", "valid/"):
        i = s.find(tag)
        if i >= 0:
            return s[i:]
    return s


KEY_FUNCS = {"nih_cxr14": nih_key, "chestx-ray8": nih_key, "chexpert_small": chexpert_key, "chexpert": chexpert_key}


def read_chexmask(path, chunksize: int = 20000, usecols=None) -> Iterator[pd.DataFrame]:
    """Lee el CSV de CheXmask por bloques (los ficheros completos ocupan hasta 8.8 GB)."""
    yield from pd.read_csv(path, chunksize=chunksize, usecols=usecols)


def match_manifest(chexmask_csv, dataset: str, manifest: pd.DataFrame, min_dice: float = 0.7,
                   n_max: Optional[int] = None, one_per_patient: bool = True, seed: int = 0,
                   exclude_patients=(), frontal_only: bool = True, chunksize: int = 20000) -> pd.DataFrame:
    """Selecciona filas de CheXmask emparejadas con un manifiesto (nih_cxr14.csv o chexpert_small.csv) y con
    Dice RCA medio >= min_dice. Dos pasadas: la primera solo lee identificador y calidad; la segunda lee las RLE
    de las filas elegidas. Devuelve un DataFrame con image_path, patient_id, dataset, key, las RLE, Height,
    Width y la calidad."""
    kf = KEY_FUNCS[dataset]
    man = manifest.copy()
    if frontal_only and "view" in man.columns and dataset.startswith("chexpert"):
        man = man[man["view"] == "Frontal"]
    if exclude_patients:
        man = man[~man["patient_id"].isin(set(exclude_patients))]
    man["key"] = man["image_path"].map(kf)
    keymap = man.set_index("key")[["image_path", "patient_id"]]
    first = next(read_chexmask(chexmask_csv, chunksize=5))
    idc = detect_id_column(first.columns)
    cand = []
    for ch in read_chexmask(chexmask_csv, chunksize, usecols=[idc, DICE_MEAN]):
        k = ch[idc].map(kf)
        ok = k.isin(keymap.index) & (ch[DICE_MEAN] >= min_dice)
        cand.append(pd.DataFrame({"key": k[ok].values, "dice_rca_mean": ch.loc[ok, DICE_MEAN].values}))
    cand = pd.concat(cand, ignore_index=True) if cand else pd.DataFrame(columns=["key", "dice_rca_mean"])
    cand = cand.drop_duplicates("key").join(keymap, on="key")
    rng = np.random.default_rng(seed)
    cand = cand.iloc[rng.permutation(len(cand))]
    if one_per_patient:
        cand = cand.drop_duplicates("patient_id")
    if n_max:
        cand = cand.head(int(n_max))
    chosen = set(cand["key"])
    rows = []
    for ch in read_chexmask(chexmask_csv, chunksize):
        k = ch[idc].map(kf)
        sel = ch[k.isin(chosen)].copy()
        if len(sel):
            sel["key"] = k[sel.index]
            rows.append(sel[["key", *RLE_COLUMNS, "Height", "Width"]])
    full = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["key", *RLE_COLUMNS, "Height", "Width"])
    out = cand.merge(full.drop_duplicates("key"), on="key", how="inner")
    out["dataset"] = dataset
    return out.reset_index(drop=True)


def decode_row(row, out_hw=None, aspect_tol: float = 0.03) -> np.ndarray:
    """Máscaras (2, H, W) uint8: canal 0 pulmones (izquierdo o derecho), canal 1 corazón. Con out_hw se reducen
    con vecino más próximo al tamaño de la imagen en disco; si la relación de aspecto no coincide, ValueError."""
    h, w = int(row["Height"]), int(row["Width"])
    lungs = rle_decode(row["Left Lung"], h, w) | rle_decode(row["Right Lung"], h, w)
    heart = rle_decode(row["Heart"], h, w)
    m = np.stack([lungs, heart])
    if out_hw is not None and tuple(out_hw) != (h, w):
        oh, ow = out_hw
        if abs((w / h) / (ow / oh) - 1.0) > aspect_tol:
            raise ValueError("relación de aspecto distinta: máscara %dx%d, imagen %dx%d" % (h, w, oh, ow))
        from PIL import Image
        m = np.stack([np.asarray(Image.fromarray(c * 255).resize((ow, oh), Image.NEAREST)) > 127 for c in m]).astype(np.uint8)
    return m


def nih_patients_in_covidx(map_csv, splits=None) -> set:
    """Pacientes NIH con imágenes RSNA en COVIDx (data/manifests/covidx_rsna_to_nih_map.csv). Con splits=None
    devuelve los de todas las particiones; con una lista (p. ej. ["val", "test"]), solo los de esas.

    Por defecto el profesor excluye a TODOS (enmienda del 12-11-2026): las 13 756 imágenes RSNA de
    COVIDx son copias de NIH y todas son negativas. Si el profesor se entrenase con ellas, sus pseudomáscaras
    serían mejores en esa fuente (vistas en el entrenamiento) que en bimcv o stonybrook (nunca vistas), y la
    calidad de la máscara quedaría ligada a la fuente y, por construcción, a la clase."""
    d = pd.read_csv(map_csv, dtype=str)
    if splits is not None:
        d = d[d["split"].isin(list(splits))]
    return set(d["nih_patient_id"].dropna())


def nih_patients_in_covidx_eval(map_csv) -> set:
    """Pacientes NIH con imágenes RSNA en val o test de COVIDx (subconjunto del anterior)."""
    return nih_patients_in_covidx(map_csv, ["val", "test"])
