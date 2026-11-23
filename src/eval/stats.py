"""Estadística de evaluación de CSD-Net y de las líneas base (docs/08_evaluacion.md).

Biblioteca sin dependencias nuevas (numpy, pandas, scipy y scikit-learn). Reutiliza de `src/eval/probes.py`
`cluster_bootstrap` (remuestreo de pacientes), `youden_threshold` y `v6_specificity_gap`, importándolas.

Contenido:
- Lectura robusta de predicciones (`read_predictions`), de corridas de los tres entrenadores (`read_run`) y de
  los informes de verificaciones (`read_probe_report`); bloqueo de lectura de conjuntos test (`assert_not_test`).
- Alineación estricta de dos ficheros de predicciones (`align`): mismos image_id y mismas etiquetas o error.
- Métricas por conjunto (`point_metrics`): AUC-ROC, AUC-PR, sensibilidad, especificidad y exactitud balanceada a
  un umbral fijado en otro conjunto, Brier y ECE de 15 bins; IC95 % por bootstrap de pacientes
  (`bootstrap_metrics`), con B y semilla devueltos en la salida.
- Comparación pareada de dos modelos sobre las mismas imágenes (`paired_auc_test`): diferencia de AUC con
  bootstrap pareado de pacientes (IC95 % de percentiles y p unilateral = fracción de remuestreos con
  diferencia <= 0) y DeLong pareado con el algoritmo rápido de Sun y Xu (`delong_paired`) como comprobación
  secundaria; no inferioridad con margen (`noninferiority`).
- Semillas: métrica por semilla, media y desviación (`seed_summary`), conjunto de probabilidades medias
  (`ensemble`), Welch y prueba pareada entre métodos (`compare_seed_values`).
- Holm (`holm`).
- Métricas por fuente y por subconjunto (`per_source_metrics`), brecha de especificidad (`specificity_gap`).

Convenciones: la sensibilidad es la fracción de positivos con p >= umbral y la especificidad la de negativos con
p < umbral, como en `v6_specificity_gap`. El umbral nunca se calcula en el conjunto evaluado salvo que se pida
expresamente (`threshold_from`). La unidad de remuestreo es el paciente; `patient_map_from_manifest` permite
sustituir el UUID por imagen de RSNA por el paciente NIH enlazado por hash (docs/02_datos.md, Section 4.1).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from src.eval.probes import cluster_bootstrap, v6_specificity_gap, youden_threshold

PRED_COLUMNS = ("image_id", "label", "source", "patient", "prob")
KNOWN_STATES = ("dfr_last", "dfr_best", "dfr_ema", "last", "best", "ema",
                "dfr_bestdc", "bestdc")  # bestdc: estado elegido con val_desconf (B3a, docs/08 versión 2)
COVIDX_SOURCES = ("actmed", "bimcv", "cohen", "fig1", "ricord", "rsna", "sirm", "stonybrook")
DEFAULT_B = 2000
DEFAULT_BOOT_SEED = 2026
ECE_BINS = 15

__all__ = [
    "AlignmentError", "TestSetForbidden", "PredictionFileError", "assert_not_test", "is_test_set",
    "split_pred_name", "read_predictions", "align", "patient_map_from_manifest", "apply_patient_map",
    "auc_roc", "auc_pr", "brier", "ece", "reliability_bins", "threshold_metrics", "point_metrics",
    "bootstrap_draws", "bootstrap_metrics", "bootstrap_auc", "paired_auc_test", "delong_paired",
    "noninferiority", "ensemble", "seed_summary", "compare_seed_values", "holm", "per_source_metrics",
    "specificity_gap", "read_run", "find_seed_dirs", "read_probe_report", "sha256_file", "youden_threshold",
    "cluster_bootstrap", "roc_band",
]


# ===================================================================================== errores y guardas
class AlignmentError(ValueError):
    """Dos ficheros de predicciones no cubren exactamente las mismas imágenes con las mismas etiquetas."""


class TestSetForbidden(PermissionError):
    """Se intentó leer un conjunto test o test_* donde está prohibido (piloto, Puerta 3)."""


class PredictionFileError(ValueError):
    """Fichero de predicciones mal formado."""


def is_test_set(name: str) -> bool:
    n = str(name).strip().lower()
    return n == "test" or n.startswith("test_")


def assert_not_test(name: str, context: str = "") -> None:
    if is_test_set(name):
        raise TestSetForbidden("conjunto '%s' prohibido%s: el test está bloqueado hasta después de la Puerta 3 "
                               "(docs/07_piloto.md, Section 3)" % (name, (" en " + context) if context else ""))


def split_pred_name(path) -> tuple:
    """`predictions/<estado>_<conjunto>.csv` -> (estado, conjunto). Los estados pueden llevar guion bajo."""
    stem = Path(path).name
    if stem.endswith(".csv"):
        stem = stem[:-4]
    for st in KNOWN_STATES:
        if stem.startswith(st + "_"):
            return st, stem[len(st) + 1:]
    if "_" in stem:
        a, b = stem.split("_", 1)
        return a, b
    raise PredictionFileError("nombre de predicciones no reconocido: %s" % path)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ===================================================================================== lectura
def read_predictions(path, forbid_test: bool = False, context: str = "") -> pd.DataFrame:
    """Lee un CSV de predicciones (image_id, label, source, patient, prob) y lo valida.

    Con `forbid_test=True` se niega a abrir un conjunto test o test_* (por el nombre del fichero) antes de
    tocar el disco. Comprueba columnas, etiquetas binarias, probabilidades finitas en [0, 1] e image_id únicos.
    """
    path = Path(path)
    st, name = split_pred_name(path)
    if forbid_test:
        assert_not_test(name, context or str(path))
    if not path.exists():
        raise FileNotFoundError("no existe el fichero de predicciones %s" % path)
    df = pd.read_csv(path, dtype={"image_id": str, "patient": str})
    miss = [c for c in PRED_COLUMNS if c not in df.columns]
    if miss:
        raise PredictionFileError("%s: faltan columnas %s" % (path, miss))
    df = df[list(PRED_COLUMNS)].copy()
    if df.empty:
        raise PredictionFileError("%s: fichero vacío" % path)
    if df["image_id"].isna().any() or df["image_id"].duplicated().any():
        raise PredictionFileError("%s: image_id vacíos o duplicados" % path)
    lab = pd.to_numeric(df["label"], errors="coerce")
    if lab.isna().any() or not set(np.unique(lab)).issubset({0, 1}):
        raise PredictionFileError("%s: etiquetas no binarias" % path)
    df["label"] = lab.astype(int)
    pr = pd.to_numeric(df["prob"], errors="coerce").astype(float)
    if (~np.isfinite(pr)).any() or (pr < 0).any() or (pr > 1).any():
        raise PredictionFileError("%s: probabilidades no finitas o fuera de [0, 1]" % path)
    df["prob"] = pr
    df["patient"] = df["patient"].fillna("").astype(str)
    if (df["patient"] == "").any():
        raise PredictionFileError("%s: hay imágenes sin paciente (el remuestreo es por paciente)" % path)
    df.attrs.update({"path": str(path), "state": st, "set": name})
    return df.reset_index(drop=True)


def align(a: pd.DataFrame, b: pd.DataFrame, name_a: str = "A", name_b: str = "B") -> pd.DataFrame:
    """Devuelve `b` reordenado como `a`. Error si los image_id no son exactamente los mismos o si alguna
    etiqueta difiere. El paciente de `a` manda (y se avisa con error si no coincide)."""
    ia, ib = set(a["image_id"]), set(b["image_id"])
    if ia != ib:
        only_a, only_b = sorted(ia - ib), sorted(ib - ia)
        raise AlignmentError("%s y %s no tienen los mismos image_id: %d solo en %s (p. ej. %s), %d solo en %s (p. ej. "
                             "%s)" % (name_a, name_b, len(only_a), name_a, only_a[:3], len(only_b), name_b, only_b[:3]))
    bb = b.set_index("image_id").loc[a["image_id"]].reset_index()
    if not np.array_equal(bb["label"].to_numpy(), a["label"].to_numpy()):
        n = int((bb["label"].to_numpy() != a["label"].to_numpy()).sum())
        raise AlignmentError("%s y %s tienen %d etiquetas distintas para los mismos image_id" % (name_a, name_b, n))
    if not np.array_equal(bb["patient"].astype(str).to_numpy(), a["patient"].astype(str).to_numpy()):
        raise AlignmentError("%s y %s asignan pacientes distintos a las mismas imágenes" % (name_a, name_b))
    bb.attrs.update(b.attrs)
    return bb


def patient_map_from_manifest(manifest_csv) -> Dict[str, str]:
    """image_id -> unidad de remuestreo: paciente NIH enlazado por hash si existe (RSNA), si no patient_id."""
    m = pd.read_csv(manifest_csv, dtype=str, usecols=lambda c: c in ("image_path", "patient_id", "nih_patient_id"))
    iid = m["image_path"].str.split("/").str[-1]
    nih = m["nih_patient_id"].fillna("") if "nih_patient_id" in m else pd.Series([""] * len(m))
    unit = np.where(nih != "", nih, m["patient_id"].fillna(""))
    return dict(zip(iid, unit))


def apply_patient_map(df: pd.DataFrame, pmap: Optional[Dict[str, str]]) -> pd.DataFrame:
    """Sustituye la columna patient por la unidad del mapa (las imágenes ausentes del mapa conservan la suya)."""
    if not pmap:
        return df
    out = df.copy()
    out["patient"] = [pmap.get(i, p) or p for i, p in zip(out["image_id"], out["patient"])]
    out.attrs.update(df.attrs)
    return out


# ===================================================================================== métricas puntuales
def auc_roc(y, p) -> float:
    """AUC-ROC exacta (estadístico de Mann-Whitney con rangos medios para empates)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = sps.rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auc_pr(y, p) -> float:
    from sklearn.metrics import average_precision_score
    y = np.asarray(y).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, p))


def brier(y, p) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def _bin_index(p, n_bins):
    return np.minimum((np.asarray(p, float) * n_bins).astype(int), n_bins - 1)


def ece(y, p, n_bins: int = ECE_BINS) -> float:
    """ECE con n_bins bins de igual anchura en [0, 1] (el último incluye 1.0): suma de |frecuencia observada -
    probabilidad media| ponderada por la fracción de imágenes del bin."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if len(y) == 0:
        return float("nan")
    b = _bin_index(p, n_bins)
    tot = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            tot += m.sum() / len(y) * abs(y[m].mean() - p[m].mean())
    return float(tot)


def reliability_bins(y, p, n_bins: int = ECE_BINS) -> pd.DataFrame:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    b = _bin_index(p, n_bins)
    rows = []
    for k in range(n_bins):
        m = b == k
        rows.append({"bin": k, "lo": k / n_bins, "hi": (k + 1) / n_bins, "n": int(m.sum()),
                     "p_media": float(p[m].mean()) if m.any() else float("nan"),
                     "frecuencia": float(y[m].mean()) if m.any() else float("nan")})
    return pd.DataFrame(rows)


def threshold_metrics(y, p, thr: float) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    pos, neg = y == 1, y == 0
    sens = float((p[pos] >= thr).mean()) if pos.any() else float("nan")
    spec = float((p[neg] < thr).mean()) if neg.any() else float("nan")
    ba = 0.5 * (sens + spec) if pos.any() and neg.any() else float("nan")
    return {"sensibilidad": sens, "especificidad": spec, "exactitud_balanceada": ba}


METRIC_NAMES = ("auc_roc", "auc_pr", "sensibilidad", "especificidad", "exactitud_balanceada", "brier", "ece")


def point_metrics(y, p, thr: Optional[float] = None, n_bins: int = ECE_BINS) -> dict:
    """Todas las métricas puntuales. Las de umbral solo si se da `thr` (fijado en otro conjunto)."""
    out = {"n": int(len(y)), "n_pos": int(np.sum(y)), "n_neg": int(len(y) - np.sum(y)),
           "auc_roc": auc_roc(y, p), "auc_pr": auc_pr(y, p), "brier": brier(y, p), "ece": ece(y, p, n_bins)}
    if thr is not None:
        out.update(threshold_metrics(y, p, thr))
        out["umbral"] = float(thr)
    return out


# ===================================================================================== bootstrap de pacientes
def bootstrap_draws(stat: Callable[[np.ndarray], Sequence[float]], groups, B: int = DEFAULT_B,
                    seed: int = DEFAULT_BOOT_SEED) -> np.ndarray:
    """Distribución bootstrap de un estadístico vectorial remuestreando pacientes con reemplazo.

    Usa `cluster_bootstrap` de src/eval/probes.py (mismo generador y mismos remuestreos para la misma semilla):
    `stat(idx)` recibe los índices de las imágenes de los pacientes elegidos (cada paciente entra con todas sus
    imágenes tantas veces como se elija) y devuelve un vector. Devuelve una matriz B x k (filas con NaN si el
    estadístico no está definido en ese remuestreo)."""
    rec: List[np.ndarray] = []

    def f(idx):
        try:
            v = np.atleast_1d(np.asarray(stat(idx), dtype=float))
        except ValueError:
            v = None                                          # estadístico no definido en este remuestreo
        rec.append(v)
        return 0.0                                            # el valor devuelto no se usa

    cluster_bootstrap(f, groups, n_boot=int(B), seed=int(seed))
    if len(rec) != int(B):
        raise RuntimeError("el bootstrap devolvió %d remuestreos de %d" % (len(rec), B))
    width = max((len(v) for v in rec if v is not None), default=1)
    return np.vstack([v if v is not None else np.full(width, np.nan) for v in rec])


def _pct(vals, alpha=0.05):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))


def bootstrap_metrics(y, p, groups, thr: Optional[float] = None, B: int = DEFAULT_B,
                      seed: int = DEFAULT_BOOT_SEED, alpha: float = 0.05, n_bins: int = ECE_BINS) -> dict:
    """Métricas puntuales con IC por bootstrap de pacientes (percentiles). El umbral queda fijo (no se
    recalcula en cada remuestreo: viene de otro conjunto)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    est = point_metrics(y, p, thr, n_bins)
    names = [m for m in METRIC_NAMES if m in est]

    def stat(i):
        pm = point_metrics(y[i], p[i], thr, n_bins)
        return [pm[m] for m in names]

    D = bootstrap_draws(stat, groups, B, seed)
    out = {"estimacion": est, "ic95": {}, "B": int(B), "B_validos": {}, "semilla_bootstrap": int(seed),
           "unidad_remuestreo": "paciente", "n_pacientes": int(len(np.unique(np.asarray(groups).astype(str)))),
           "alpha": alpha}
    for k, m in enumerate(names):
        out["ic95"][m] = _pct(D[:, k], alpha)
        out["B_validos"][m] = int(np.isfinite(D[:, k]).sum())
    return out


def bootstrap_auc(y, p, groups, B: int = DEFAULT_B, seed: int = DEFAULT_BOOT_SEED, alpha: float = 0.05) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    D = bootstrap_draws(lambda i: [auc_roc(y[i], p[i])], groups, B, seed)[:, 0]
    lo, hi = _pct(D, alpha)
    return {"auc": auc_roc(y, p), "ic95": (lo, hi), "B": int(B), "B_validos": int(np.isfinite(D).sum()),
            "semilla_bootstrap": int(seed), "unidad_remuestreo": "paciente"}


# ===================================================================================== DeLong (Sun y Xu)
def _midrank(x):
    """Rangos medios (1..n) con empates, como en Sun y Xu (2014)."""
    return sps.rankdata(x)


def delong_paired(y, p1, p2) -> dict:
    """DeLong pareado con el algoritmo rápido de Sun y Xu: AUC de cada modelo, matriz de covarianza 2 x 2 de las
    AUC, varianza de la diferencia, z y valores p (bilateral y unilateral para AUC1 > AUC2). No tiene en cuenta
    los conglomerados por paciente: es la comprobación secundaria; la principal es el bootstrap de pacientes."""
    y = np.asarray(y).astype(int)
    P = np.vstack([np.asarray(p1, float), np.asarray(p2, float)])
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    if m < 2 or n < 2:
        raise ValueError("DeLong necesita al menos dos positivos y dos negativos")
    X, Yn = P[:, pos], P[:, neg]
    k = P.shape[0]
    tx = np.vstack([_midrank(X[r]) for r in range(k)])
    ty = np.vstack([_midrank(Yn[r]) for r in range(k)])
    tz = np.vstack([_midrank(np.r_[X[r], Yn[r]]) for r in range(k)])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n          # componentes estructurales de los positivos
    v10 = 1.0 - (tz[:, m:] - ty) / m    # componentes de los negativos
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    S = sx / m + sy / n
    L = np.array([1.0, -1.0])
    var = float(L @ S @ L)
    d = float(aucs[0] - aucs[1])
    if var <= 0:
        z = float("inf") if d > 0 else (float("-inf") if d < 0 else 0.0)
    else:
        z = d / math.sqrt(var)
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "delta": d, "cov": S.tolist(), "var_delta": var,
            "se_delta": math.sqrt(var) if var > 0 else 0.0, "z": z,
            "p_bilateral": float(2 * sps.norm.sf(abs(z))), "p_unilateral": float(sps.norm.sf(z)),
            "ic95_normal": (d - 1.959963984540054 * math.sqrt(max(var, 0)), d + 1.959963984540054 * math.sqrt(max(var, 0))),
            "metodo": "DeLong pareado, algoritmo rápido de Sun y Xu (2014)"}


# ===================================================================================== comparación pareada
def paired_auc_test(y, p1, p2, groups, B: int = DEFAULT_B, seed: int = DEFAULT_BOOT_SEED, alpha: float = 0.05,
                    margin: Optional[float] = None, delong: bool = True) -> dict:
    """Δ = AUC(p1) - AUC(p2) sobre las mismas imágenes, con bootstrap pareado de pacientes: en cada remuestreo se
    recalculan las dos AUC con los mismos pacientes. IC de percentiles; p unilateral = (1 + número de remuestreos
    válidos con Δ* <= 0) / (B válidos + 1), de modo que nunca vale cero (enmienda del 12-11-2026). Si se da `margin`, añade la no inferioridad (extremo inferior > -margin) y la fracción
    con Δ* <= -margin."""
    y = np.asarray(y).astype(int)
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    if not (len(y) == len(p1) == len(p2) == len(groups)):
        raise AlignmentError("longitudes distintas en la comparación pareada")
    a1, a2 = auc_roc(y, p1), auc_roc(y, p2)

    def stat(i):
        return [auc_roc(y[i], p1[i]) - auc_roc(y[i], p2[i])]

    D = bootstrap_draws(stat, groups, B, seed)[:, 0]
    ok = np.isfinite(D)
    lo, hi = _pct(D, alpha)
    out = {"auc1": a1, "auc2": a2, "delta": a1 - a2, "ic95": (lo, hi), "p_unilateral": float(((D[ok] <= 0).sum() + 1) / (ok.sum() + 1)),
           "B": int(B), "B_validos": int(ok.sum()), "semilla_bootstrap": int(seed), "unidad_remuestreo": "paciente",
           "n_imagenes": int(len(y)), "n_pacientes": int(len(np.unique(np.asarray(groups).astype(str)))),
           "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
    if margin is not None:
        out["no_inferioridad"] = noninferiority(lo, margin)
        out["no_inferioridad"]["p_unilateral_margen"] = float(((D[ok] <= -float(margin)).sum() + 1) / (ok.sum() + 1))
    if delong:
        try:
            out["delong"] = delong_paired(y, p1, p2)
        except ValueError as e:
            out["delong"] = {"error": str(e)}
    return out


def paired_auc_test_seedmean(y, P1, P2, groups, B: int = DEFAULT_B, seed: int = DEFAULT_BOOT_SEED,
                             alpha: float = 0.05, margin: Optional[float] = None,
                             seeds: Optional[Sequence] = None) -> dict:
    """Comparación primaria del artículo (decisión del autor, 12-11-2026): media por semilla.

    P1 y P2 son matrices (S, N) con las probabilidades de las S semillas de cada método sobre las mismas N
    imágenes y en el mismo orden; la fila s de P1 se empareja con la fila s de P2 (misma semilla de entrenamiento).
    El estadístico es Δ = media_s [AUC(P1_s) - AUC(P2_s)], es decir, la diferencia de AUC de un modelo individual
    promediada sobre las semillas, y no la del conjunto de modelos. El IC95 % es el de percentiles del bootstrap de
    pacientes: en cada remuestreo se recalculan las 2 S AUC con los mismos pacientes y se promedian las S
    diferencias. p unilateral = (1 + número de remuestreos válidos con Δ* <= 0) / (B válidos + 1). El intervalo
    recoge la incertidumbre del muestreo de pacientes con estas S semillas; la variabilidad entre semillas se
    informa aparte (diferencia por semilla, su desviación típica y en cuántas semillas gana el primer método)."""
    y = np.asarray(y).astype(int)
    P1 = np.atleast_2d(np.asarray(P1, float))
    P2 = np.atleast_2d(np.asarray(P2, float))
    if P1.shape != P2.shape or P1.shape[1] != len(y) or len(groups) != len(y):
        raise AlignmentError("matrices de semillas con formas distintas o no alineadas con las etiquetas: %s, %s, %d"
                             % (P1.shape, P2.shape, len(y)))
    S_ = P1.shape[0]
    names = [str(k) for k in (seeds if seeds is not None else range(S_))]
    if len(names) != S_:
        raise AlignmentError("el número de semillas no coincide con las filas de las matrices")
    a1 = np.array([auc_roc(y, P1[k]) for k in range(S_)])
    a2 = np.array([auc_roc(y, P2[k]) for k in range(S_)])
    d = a1 - a2

    def stat(i):
        yi = y[i]
        return [float(np.mean([auc_roc(yi, P1[k][i]) - auc_roc(yi, P2[k][i]) for k in range(S_)]))]

    D = bootstrap_draws(stat, groups, B, seed)[:, 0]
    ok = np.isfinite(D)
    lo, hi = _pct(D, alpha)
    out = {"estadistico": "media por semilla de la diferencia de AUC-ROC (modelo individual)",
           "delta": float(d.mean()), "ic95": (lo, hi),
           "p_unilateral": float(((D[ok] <= 0).sum() + 1) / (ok.sum() + 1)),
           "n_semillas": int(S_), "semillas": names,
           "delta_por_semilla": {names[k]: float(d[k]) for k in range(S_)},
           "dt_delta_entre_semillas": float(d.std(ddof=1)) if S_ > 1 else float("nan"),
           "semillas_a_favor": int((d > 0).sum()),
           "auc1_por_semilla": {names[k]: float(a1[k]) for k in range(S_)},
           "auc2_por_semilla": {names[k]: float(a2[k]) for k in range(S_)},
           "auc1": float(a1.mean()), "auc2": float(a2.mean()),
           "B": int(B), "B_validos": int(ok.sum()), "semilla_bootstrap": int(seed), "unidad_remuestreo": "paciente",
           "n_imagenes": int(len(y)), "n_pacientes": int(len(np.unique(np.asarray(groups).astype(str)))),
           "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
    if margin is not None:
        out["no_inferioridad"] = noninferiority(lo, margin)
        out["no_inferioridad"]["p_unilateral_margen"] = float(((D[ok] <= -float(margin)).sum() + 1) / (ok.sum() + 1))
    return out


def noninferiority(ci_lower: float, margin: float) -> dict:
    """No inferioridad con margen: se declara si el extremo inferior del IC95 % de Δ es mayor que -margin."""
    return {"margen": float(margin), "ic95_inferior": float(ci_lower),
            "no_inferior": bool(np.isfinite(ci_lower) and ci_lower > -float(margin))}


# ===================================================================================== semillas
def ensemble(dfs: Sequence[pd.DataFrame], names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Probabilidad media entre semillas, con alineación estricta frente a la primera."""
    if not dfs:
        raise ValueError("sin predicciones para el conjunto")
    names = list(names or ["semilla%d" % i for i in range(len(dfs))])
    base = dfs[0]
    probs = [base["prob"].to_numpy()]
    for d, nm in zip(dfs[1:], names[1:]):
        probs.append(align(base, d, names[0], nm)["prob"].to_numpy())
    out = base[["image_id", "label", "source", "patient"]].copy()
    out["prob"] = np.mean(np.vstack(probs), axis=0)
    out.attrs.update({"miembros": names})
    return out


def seed_summary(values: Dict[str, float]) -> dict:
    """Valores por semilla, media y desviación típica muestral (ddof = 1)."""
    v = np.array([values[k] for k in values], float)
    f = v[np.isfinite(v)]
    return {"por_semilla": {str(k): float(values[k]) for k in values}, "n_semillas": int(len(f)),
            "media": float(f.mean()) if len(f) else float("nan"),
            "dt": float(f.std(ddof=1)) if len(f) > 1 else float("nan")}


def compare_seed_values(a: Dict[str, float], b: Dict[str, float], alternative: str = "greater") -> dict:
    """Comparación entre métodos con las semillas como réplicas: Welch siempre y, si las semillas coinciden,
    prueba t pareada por semilla. `alternative='greater'` contrasta media(a) > media(b)."""
    va = np.array(list(a.values()), float)
    vb = np.array(list(b.values()), float)
    out = {"media_a": float(np.mean(va)), "media_b": float(np.mean(vb)), "diferencia": float(np.mean(va) - np.mean(vb)),
           "n_a": int(len(va)), "n_b": int(len(vb)), "alternativa": alternative}
    if len(va) >= 2 and len(vb) >= 2:
        w = sps.ttest_ind(va, vb, equal_var=False, alternative=alternative)
        sa, sb = va.var(ddof=1) / len(va), vb.var(ddof=1) / len(vb)
        df = (sa + sb) ** 2 / (sa ** 2 / (len(va) - 1) + sb ** 2 / (len(vb) - 1)) if (sa + sb) > 0 else float("nan")
        out["welch"] = {"t": float(w.statistic), "gl": float(df), "p": float(w.pvalue)}
    else:
        out["welch"] = {"t": float("nan"), "gl": float("nan"), "p": float("nan")}
    common = sorted(set(a) & set(b))
    if set(a) == set(b) and len(common) >= 2:
        da = np.array([a[k] for k in common], float)
        db = np.array([b[k] for k in common], float)
        t = sps.ttest_rel(da, db, alternative=alternative)
        out["pareada"] = {"semillas": common, "t": float(t.statistic), "gl": len(common) - 1, "p": float(t.pvalue),
                          "diferencia_media": float(np.mean(da - db)), "dt_diferencia": float(np.std(da - db, ddof=1))}
    else:
        out["pareada"] = None
    return out


# ===================================================================================== Holm
def holm(pvals: Dict[str, float] | Sequence[float], alpha: float = 0.05):
    """Corrección de Holm. Devuelve valores p ajustados (monótonos, recortados a 1) y la decisión al nivel alpha.
    Acepta un dict (nombre -> p) o una secuencia; los NaN se excluyen de la familia y se devuelven como NaN."""
    if isinstance(pvals, dict):
        keys = list(pvals.keys())
        p = np.array([pvals[k] for k in keys], float)
    else:
        keys = None
        p = np.asarray(list(pvals), float)
    adj = np.full(len(p), np.nan)
    ok = np.nonzero(np.isfinite(p))[0]
    m = len(ok)
    if m:
        order = ok[np.argsort(p[ok], kind="mergesort")]
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (m - rank) * p[i]))
            adj[i] = running
    reject = np.where(np.isfinite(adj), adj <= alpha, False)
    if keys is not None:
        return {k: {"p": float(p[i]), "p_holm": float(adj[i]), "rechaza": bool(reject[i])} for i, k in enumerate(keys)}
    return adj, reject


# ===================================================================================== por fuente
def per_source_metrics(df: pd.DataFrame, thr: float, source_names: Sequence[str] = COVIDX_SOURCES) -> pd.DataFrame:
    rows = []
    for s in sorted(df["source"].unique()):
        d = df[df["source"] == s]
        nm = source_names[int(s)] if str(s).lstrip("-").isdigit() and 0 <= int(s) < len(source_names) else str(s)
        y, p = d["label"].to_numpy(), d["prob"].to_numpy()
        tm = threshold_metrics(y, p, thr)
        rows.append({"fuente": nm, "n": len(d), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
                     "auc_roc": auc_roc(y, p), **tm})
    return pd.DataFrame(rows)


def specificity_gap(df: pd.DataFrame, thr: float, src_a="rsna", src_b="bimcv", B: int = DEFAULT_B,
                    seed: int = DEFAULT_BOOT_SEED, source_names: Sequence[str] = COVIDX_SOURCES) -> dict:
    """Brecha de especificidad src_a menos src_b en los negativos al umbral dado, con IC95 % por bootstrap de
    pacientes (v6_specificity_gap de src/eval/probes.py)."""
    ia = source_names.index(src_a) if isinstance(src_a, str) else int(src_a)
    ib = source_names.index(src_b) if isinstance(src_b, str) else int(src_b)
    r = v6_specificity_gap(df["prob"].to_numpy(), df["label"].to_numpy(), df["source"].to_numpy().astype(int),
                           df["patient"].astype(str).to_numpy(), float(thr), ia, ib, n_boot=int(B), seed=int(seed))
    m = r.get("metricas", {})
    return {"brecha": m.get("brecha"), "ic95": (m.get("ci_lo"), m.get("ci_hi")), "esp_a": m.get("esp_a"),
            "esp_b": m.get("esp_b"), "umbral": float(thr), "fuente_a": src_a, "fuente_b": src_b, "B": int(B),
            "semilla_bootstrap": int(seed), "veredicto_probes": r.get("veredicto"),
            "indicador": (r.get("detalle") or {}).get("indicador"),
            "n_neg_a": int(((df["label"] == 0) & (df["source"].astype(int) == ia)).sum()),
            "n_neg_b": int(((df["label"] == 0) & (df["source"].astype(int) == ib)).sum())}


# ===================================================================================== curvas ROC con banda
def roc_band(y, p, groups, B: int = DEFAULT_B, seed: int = DEFAULT_BOOT_SEED, grid: int = 101, alpha: float = 0.05):
    """Curva ROC y banda puntual de percentiles de la TPR sobre una rejilla de FPR (bootstrap de pacientes)."""
    from sklearn.metrics import roc_curve
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    fg = np.linspace(0, 1, grid)

    def tpr_at(i):
        yy, pp = y[i], p[i]
        if yy.min() == yy.max():
            return np.full(grid, np.nan)
        f, t, _ = roc_curve(yy, pp)
        return np.interp(fg, f, t)

    D = bootstrap_draws(tpr_at, groups, B, seed)
    lo = np.nanpercentile(D, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(D, 100 * (1 - alpha / 2), axis=0)
    f, t, _ = roc_curve(y, p)
    return {"fpr": f, "tpr": t, "fpr_rejilla": fg, "tpr_lo": lo, "tpr_hi": hi, "B": int(B), "semilla_bootstrap": seed}


# ===================================================================================== corridas e informes
def find_seed_dirs(results_root, experiment: str) -> Dict[int, Path]:
    """results/train/<experimento>/seed<S>/ -> {S: ruta}."""
    base = Path(results_root) / experiment
    out = {}
    if base.is_dir():
        for d in sorted(base.iterdir()):
            mm = re.fullmatch(r"seed(\d+)", d.name)
            if mm and d.is_dir():
                out[int(mm.group(1))] = d
    return out


def _load_json(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        raise ValueError("JSON ilegible en %s: %s" % (p, e))


def read_run(run_dir) -> dict:
    """Lee una corrida de cualquiera de los tres entrenadores sin abrir las predicciones.

    Devuelve: ruta, experimento, semilla, tipo ('csdnet' si summary.json tiene probes o v1; 'linea_base' si
    tiene 'seleccion'; 'comun' en otro caso), summary, config (config_resolved.json), predicciones disponibles
    {(estado, conjunto): ruta}, informes de verificaciones {nombre: ruta}, protocolo (test_bloqueado, congelado,
    caché y selección) y no_comparables."""
    d = Path(run_dir)
    summ = _load_json(d / "summary.json")
    cfg = _load_json(d / "config_resolved.json") or {}
    preds = {}
    if (d / "predictions").is_dir():
        for f in sorted((d / "predictions").glob("*.csv")):
            try:
                preds[split_pred_name(f)] = f
            except PredictionFileError:
                continue
    probes = {}
    if (d / "probes").is_dir():
        for f in sorted((d / "probes").glob("*.json")):
            probes[f.stem] = f
    if summ is not None and ("probes" in summ or "v1" in summ):
        kind = "csdnet"
    elif summ is not None and "seleccion" in summ:
        kind = "linea_base"
    else:
        kind = "comun"
    prot = cfg.get("protocol") or {}
    data = cfg.get("data") or {}
    sel = cfg.get("selection") or {}
    mm = re.fullmatch(r"seed(\d+)", d.name)
    return {"ruta": str(d), "experimento": (summ or {}).get("experiment") or cfg.get("experiment") or d.parent.name,
            "semilla": int((summ or {}).get("seed", mm.group(1) if mm else -1)), "tipo": kind, "summary": summ,
            "config": cfg, "predicciones": preds, "informes_verificaciones": probes,
            "protocolo": {"test_bloqueado": prot.get("test_bloqueado"), "congelado": prot.get("congelado"),
                          "fase": prot.get("fase"), "cache": data.get("cache"), "seleccion": sel.get("subset"),
                          "metrica_seleccion": sel.get("metric")},
            "no_comparables": (summ or {}).get("no_comparables", {}), "terminada": bool(summ is not None)}


def read_probe_report(path) -> dict:
    """Informe de verificaciones de src/eval/probes.py (write_report): {'cabecera', 'verificaciones'} ->
    {'cabecera', 'veredictos': {Vk: veredicto}, 'metricas': {Vk: {...}}, 'ruta'}."""
    js = _load_json(Path(path))
    if js is None:
        raise FileNotFoundError("no existe el informe de verificaciones %s" % path)
    ver = js.get("verificaciones")
    if not isinstance(ver, dict):
        raise ValueError("informe de verificaciones sin el campo 'verificaciones': %s" % path)
    return {"ruta": str(path), "cabecera": js.get("cabecera", {}),
            "veredictos": {k: (v or {}).get("veredicto") for k, v in ver.items()},
            "metricas": {k: (v or {}).get("metricas", {}) for k, v in ver.items()},
            "detalle": {k: (v or {}).get("detalle", {}) for k, v in ver.items()}}


def jsonable(o):
    """Convierte a tipos JSON (NaN -> None, tuplas -> listas, numpy -> nativos)."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not math.isfinite(v) else v
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o
