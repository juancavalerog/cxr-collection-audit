"""Auditoría de COVIDx CXR-4 (aporte C1; docs/08_evaluacion.md, versión 2, Section V2.3).

Versión de producción de los prototipos exploratorios `scripts/diag/expl_lectura_reserva.py` y
`scripts/diag/expl_colecciones.py`. Trabaja sobre las características ya extraídas por
`scripts/diag/expl_lectura_features.py` (pares `<tag>.npz` con `pooled`, `penult` y `logit`, y `<tag>_meta.csv` con
image_id, patient_id, label, source y conjunto) y sobre las predicciones de los entrenadores. Solo CPU; no lee el test.

Definiciones registradas (Section V2.3.1):
- Colección de bimcv: el identificador de sujeto `sub-S<dígitos>_` tiene cinco cifras en la colección antigua y seis
  en la nueva (docs/07_piloto.md, Tabla E1). Solo se usa para positivos; los negativos son una colección aparte.
- Proyección: `vp-ap` o `vp-pa` en el nombre de fichero.
- Detector de colección sin información de clase: regresión logística L2 (C = 0.05, pesos de clase equilibrados,
  lbfgs, 4 000 iteraciones) sobre las características agrupadas (`pooled`, 768) estandarizadas con media y
  desviación del propio conjunto de ajuste, que contiene solo positivos de bimcv de entrenamiento: diana 1 para la
  colección antigua y 0 para la nueva. La etiqueta de clase solo interviene para restringir el ajuste a positivos;
  el detector no ve ningún negativo ni contrasta clases. Si el fichero trae positivos reservados (que el tronco no
  vio: conjuntos `res_pos_antigua` y `res_pos_nueva`), el ajuste usa esos; si no, la muestra `tr_bimcv_pos`. La
  puntuación es la función de decisión (mayor = más parecido a la colección antigua, que es la de los positivos de
  validación y de test).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from src.eval import stats as S
from src.eval.dosis import auc_remuestreo, _ic, _p_mayor

C_DETECTOR = 0.05
FEAT_DETECTOR = "pooled"
SEMILLA_MUESTRA = 20261113
MARGEN_NI = 0.02                 # margen de no inferioridad del detector frente a cada método (Section V2.3.2)
ID_RE = r"^sub-S(\d+)_"


def coleccion_bimcv(image_ids) -> np.ndarray:
    """'antigua' (cinco cifras), 'nueva' (seis cifras) o '' si el nombre no sigue el patrón de bimcv."""
    s = pd.Series(list(image_ids), dtype=str).str.extract(ID_RE)[0]
    n = s.str.len()
    return np.where(n == 5, "antigua", np.where(n == 6, "nueva", "")).astype(object)


def proyeccion(image_ids) -> np.ndarray:
    s = pd.Series(list(image_ids), dtype=str).str.lower()
    return np.where(s.str.contains("vp-ap"), "AP", np.where(s.str.contains("vp-pa"), "PA", "")).astype(object)


def mitad(pid) -> int:
    """Igual que src/baselines/methods/dfr.py: 0 = val_robust_fit, 1 = val_robust_sel."""
    return int(hashlib.sha1(str(pid).encode("utf-8")).hexdigest(), 16) % 2


def cargar(d, tag: str):
    """Características y metadatos de un tag, con las columnas derivadas `coleccion`, `proyeccion` y
    `val_robust` (bimcv de validación). Se niega a cargar filas de un conjunto test."""
    d = Path(d)
    z = np.load(d / ("%s.npz" % tag))
    m = pd.read_csv(d / ("%s_meta.csv" % tag), dtype={"image_id": str, "patient_id": str})
    bad = [c for c in m["conjunto"].unique() if S.is_test_set(c)]
    if bad:
        raise S.TestSetForbidden("el fichero %s trae conjuntos test %s" % (tag, bad))
    m["coleccion"] = coleccion_bimcv(m["image_id"])
    m["proyeccion"] = proyeccion(m["image_id"])
    m["val_robust"] = (m["conjunto"] == "val") & (m["source"] == "bimcv")
    feats = {k: z[k].astype(np.float64) for k in z.files}
    if any(len(v) != len(m) for v in feats.values()):
        raise S.AlignmentError("%s: características y metadatos con longitudes distintas" % tag)
    return feats, m


def conjunto_ajuste(m: pd.DataFrame):
    """Máscara del conjunto de ajuste del detector, diana (1 = antigua) y origen."""
    cj = m["conjunto"].to_numpy()
    res = np.isin(cj, ["res_pos_antigua", "res_pos_nueva"])
    if res.any():
        mk, origen = res, "positivos reservados (no vistos por el tronco)"
    else:
        mk, origen = (cj == "tr_bimcv_pos"), "muestra de positivos de bimcv de entrenamiento"
    col = m["coleccion"].to_numpy()
    if not mk.any():
        raise ValueError("no hay positivos de bimcv de entrenamiento en el fichero")
    if (m.loc[mk, "label"] != 1).any() or (col[mk] == "").any():
        raise ValueError("el conjunto de ajuste del detector debe tener solo positivos de bimcv con colección conocida")
    t = (col == "antigua").astype(int)
    if len(set(t[mk].tolist())) < 2:
        raise ValueError("el conjunto de ajuste necesita las dos colecciones de positivos")
    return mk, t, origen


def _lr(C):
    return LogisticRegression(penalty="l2", C=C, max_iter=4000, class_weight="balanced")


def detector(feats: Dict[str, np.ndarray], m: pd.DataFrame, C: float = C_DETECTOR, feat: str = FEAT_DETECTOR):
    """Puntuación del detector de colección en todas las filas y descripción del ajuste."""
    X = feats[feat]
    mk, t, origen = conjunto_ajuste(m)
    mu, sd = X[mk].mean(0), X[mk].std(0) + 1e-8
    Z = (X - mu) / sd
    f = _lr(C).fit(Z[mk], t[mk])
    s = f.decision_function(Z)
    info = {"caracteristicas": feat, "C": C, "penalizacion": "l2", "pesos_clase": "equilibrados",
            "estandarizacion": "conjunto de ajuste", "origen_ajuste": origen, "n_ajuste": int(mk.sum()),
            "n_antigua": int(t[mk].sum()), "n_nueva": int(mk.sum() - t[mk].sum()),
            "negativos_en_ajuste": 0, "usa_clase": "solo para restringir el ajuste a positivos"}
    return s, info


def auc_ic(y, s, g, B=S.DEFAULT_B, seed=S.DEFAULT_BOOT_SEED) -> dict:
    y = np.asarray(y).astype(int)
    D = auc_remuestreo(y, s, g, B, seed)[:, 0]
    lo, hi = _ic(D)
    return {"auc": S.auc_roc(y, s), "ic95_inf": lo, "ic95_sup": hi, "n": int(len(y)), "n_pos": int(y.sum()),
            "n_neg": int(len(y) - y.sum()), "n_pacientes": int(len(np.unique(np.asarray(g).astype(str)))),
            "B_validos": int(np.isfinite(D).sum())}


def diferencia(y, P1, P2, g, B=S.DEFAULT_B, seed=S.DEFAULT_BOOT_SEED, margen: Optional[float] = None,
               semillas: Optional[Sequence] = None) -> dict:
    """Δ = media_s [AUC(P1_s) - AUC(P2_s)] con P1, P2 de forma (S, n), emparejadas por fila (semilla) y por imágenes;
    bootstrap de pacientes. Si P1 tiene una sola fila se replica (referencia única frente a S semillas). Con
    `margen`, contraste de no inferioridad H0: Δ <= -margen frente a H1: Δ > -margen."""
    y = np.asarray(y).astype(int)
    P1, P2 = np.atleast_2d(np.asarray(P1, float)), np.atleast_2d(np.asarray(P2, float))
    if P1.shape[0] == 1 and P2.shape[0] > 1:
        P1 = np.repeat(P1, P2.shape[0], axis=0)
    if P1.shape != P2.shape:
        raise S.AlignmentError("formas distintas: %s y %s" % (P1.shape, P2.shape))
    k = P1.shape[0]
    D = auc_remuestreo(y, np.vstack([P1, P2]), g, B, seed)
    dd = np.nanmean(D[:, :k] - D[:, k:], axis=1)
    a1 = np.array([S.auc_roc(y, P1[i]) for i in range(k)])
    a2 = np.array([S.auc_roc(y, P2[i]) for i in range(k)])
    lo, hi = _ic(dd)
    out = {"auc1": float(a1.mean()), "auc2": float(a2.mean()), "delta": float((a1 - a2).mean()), "ic95_inf": lo,
           "ic95_sup": hi, "p_superioridad": _p_mayor(dd), "n_semillas": int(k),
           "delta_por_semilla": [float(v) for v in a1 - a2], "semillas": [str(s) for s in (semillas or range(k))]}
    if margen is not None:
        out.update({"margen": float(margen), "p_no_inferioridad": _p_mayor(dd, -float(margen)),
                    "no_inferior": bool(np.isfinite(lo) and lo > -float(margen))})
    return out


def validacion_cruzada(Z, m: pd.DataFrame, mk, t, C=C_DETECTOR, k=5) -> float:
    """AUC fuera de pliegue de un contraste, con pliegues agrupados por paciente (GroupKFold, como el prototipo)."""
    Xm, tm, gm = Z[mk], t[mk], m["patient_id"].astype(str).to_numpy()[mk]
    s = np.zeros(int(mk.sum()))
    for tr, te in GroupKFold(k).split(Xm, tm, gm):
        s[te] = _lr(C).fit(Xm[tr], tm[tr]).decision_function(Xm[te])
    return S.auc_roc(tm, s)


def contrastes_reservado(feats, m: pd.DataFrame, C=C_DETECTOR, feat=FEAT_DETECTOR, B=S.DEFAULT_B,
                         seed=S.DEFAULT_BOOT_SEED) -> Optional[dict]:
    """Análisis (ii) de la Section V2.3.1: validación cruzada de los tres contrastes dentro del conjunto reservado y
    transferencia de una lectura entre colecciones de positivos. Solo si el fichero trae conjunto reservado. La
    estandarización usa todo el conjunto reservado, como el prototipo (no interviene la clase)."""
    cj = m["conjunto"].to_numpy()
    neg, ant, nue = cj == "res_neg", cj == "res_pos_antigua", cj == "res_pos_nueva"
    if not (neg.any() and ant.any() and nue.any()):
        return None
    res = neg | ant | nue
    X = feats[feat]
    Z = (X - X[res].mean(0)) / (X[res].std(0) + 1e-8)
    y = m["label"].to_numpy().astype(int)
    out = {"cv_neg_vs_antiguos": validacion_cruzada(Z, m, neg | ant, y, C),
           "cv_neg_vs_nuevos": validacion_cruzada(Z, m, neg | nue, y, C),
           "cv_antiguos_vs_nuevos": validacion_cruzada(Z, m, ant | nue, nue.astype(int), C)}
    h = m["patient_id"].astype(str).map(lambda p: int(hashlib.sha1(p.encode()).hexdigest(), 16) % 2).to_numpy()
    n0, n1 = neg & (h == 0), neg & (h == 1)
    f_ant = _lr(C).fit(Z[n0 | ant], y[n0 | ant])
    f_nue = _lr(C).fit(Z[n0 | nue], y[n0 | nue])
    vr = m["val_robust"].to_numpy()
    g = m["patient_id"].astype(str).to_numpy()
    ev = lambda f, mk: S.auc_roc(y[mk], f.decision_function(Z[mk]))
    for nombre, f, mk in (("transfer_antiguos_a_nuevos", f_ant, n1 | nue), ("transfer_nuevos_a_antiguos", f_nue, n1 | ant),
                          ("lectura_nuevos_en_val_robust", f_nue, vr)):
        r = auc_ic(y[mk], f.decision_function(Z[mk]), g[mk], B, seed)
        out[nombre + "_ic95_inf"], out[nombre + "_ic95_sup"] = r["ic95_inf"], r["ic95_sup"]
    out.update({"transfer_antiguos_a_nuevos": ev(f_ant, n1 | nue), "transfer_nuevos_a_antiguos": ev(f_nue, n1 | ant),
                "lectura_nuevos_en_val_robust": ev(f_nue, vr), "lectura_antiguos_en_val_robust": ev(f_ant, vr),
                "n": {"res_neg": int(neg.sum()), "res_pos_antigua": int(ant.sum()), "res_pos_nueva": int(nue.sum()),
                      "neg_ajuste": int(n0.sum()), "neg_retenidos": int(n1.sum())}})
    return out


def referencia_proyeccion(m: pd.DataFrame, mk) -> dict:
    """Clasificador de solo proyección (AP = 1) en un subconjunto: cuánta AUC da la proyección por sí sola."""
    y = m["label"].to_numpy()[mk]
    pr = m["proyeccion"].to_numpy()[mk]
    ok = pr != ""
    return {"auc_solo_proyeccion_AP": S.auc_roc(y[ok], (pr[ok] == "AP").astype(float)),
            "frac_AP_negativos": float((pr[ok][y[ok] == 0] == "AP").mean()),
            "frac_AP_positivos": float((pr[ok][y[ok] == 1] == "AP").mean()), "n_sin_proyeccion": int((~ok).sum())}


# ===================================================================================== evaluación por colección
COLECCIONES_NEG = ("bimcv", "rsna", "cohen", "actmed")
COLECCIONES_POS = ("bimcv_antigua", "bimcv_nueva", "stonybrook", "ricord", "sirm", "cohen", "actmed")
N_MIN = 20


def etiquetar_coleccion(df: pd.DataFrame, nombres_fuente: Sequence[str] = S.COVIDX_SOURCES) -> np.ndarray:
    """Colección de cada imagen de unas predicciones: fuente, y en los positivos de bimcv, colección antigua o nueva."""
    src = df["source"]
    if np.issubdtype(np.asarray(src).dtype, np.number):
        src = pd.Series([nombres_fuente[int(i)] for i in src])
    src = np.asarray(src).astype(object)
    col = coleccion_bimcv(df["image_id"])
    out = src.copy()
    pos_b = (src == "bimcv") & (df["label"].to_numpy() == 1)
    out[pos_b] = np.array(["bimcv_" + c if c else "bimcv_desconocida" for c in col[pos_b]], dtype=object)
    return out


def tabla_por_coleccion(df: pd.DataFrame, score: np.ndarray, B=S.DEFAULT_B, seed=S.DEFAULT_BOOT_SEED,
                        n_min: int = N_MIN) -> pd.DataFrame:
    """AUC de cada pareja (colección de negativos, colección de positivos) con al menos `n_min` imágenes por lado."""
    col = etiquetar_coleccion(df)
    y = df["label"].to_numpy().astype(int)
    g = df["patient"].astype(str).to_numpy()
    filas = []
    for cn in COLECCIONES_NEG:
        mn = (col == cn) & (y == 0)
        for cp in COLECCIONES_POS:
            mp = (col == cp) & (y == 1)
            if mn.sum() < n_min or mp.sum() < n_min:
                continue
            mk = mn | mp
            r = auc_ic(y[mk], np.asarray(score)[mk], g[mk], B, seed)
            filas.append({"negativos": cn, "positivos": cp, **r})
    return pd.DataFrame(filas)


# ===================================================================================== C1 completo (docs/14)
# Añadido el 17-11-2026. Nada de lo anterior cambia: `cargar`, `detector`, `diferencia`,
# `contrastes_reservado` y `tabla_por_coleccion` siguen dando las cifras de results/audit/. Lo que sigue amplía la
# auditoría a los siete análisis registrados (V2.3.1 a V2.3.7) en validación y, tras la congelación del protocolo, en el test.
MARGEN_NI_SENS = 0.01            # M3 del revisor: sensibilidad registrada del margen
FUENTES_SOLO_POS = ("stonybrook", "ricord", "sirm")
ALPHA_HOLM = 0.025


def _es_conj_test(c) -> bool:
    from src.eval.desbloqueo import es_bloqueado
    return es_bloqueado(c)


def cargar_c1(d, tag: str, d_test=None, conjuntos_test: Sequence[str] = (), registro: Optional[dict] = None):
    """Características de un tronco: el fichero de validación `<tag>.npz` y, si se piden y hay congelación válida,
    los de los conjuntos bloqueados `<d_test>/<tag>__<conjunto>.npz`. Añade `coleccion`, `proyeccion`, `val_robust`
    (bimcv de validación) y `test_bimcv` (bimcv del test limpio). Solo se conservan los arrays comunes."""
    if conjuntos_test and (not registro or not registro.get("sha256_congelacion")):
        from src.eval.desbloqueo import TestBloqueadoError
        raise TestBloqueadoError("cargar_c1: los conjuntos %s exigen la congelación" % list(conjuntos_test))
    d = Path(d)
    z = np.load(d / ("%s.npz" % tag))
    m = pd.read_csv(d / ("%s_meta.csv" % tag), dtype={"image_id": str, "patient_id": str})
    bad = [c for c in m["conjunto"].unique() if _es_conj_test(c)]
    if bad:
        raise S.TestSetForbidden("el fichero de validación %s trae conjuntos bloqueados %s" % (tag, bad))
    feats = {k: z[k].astype(np.float64) for k in z.files}
    metas = [m]
    if conjuntos_test:
        for c in conjuntos_test:
            zt = np.load(Path(d_test) / ("%s__%s.npz" % (tag, c)))
            mt = pd.read_csv(Path(d_test) / ("%s__%s_meta.csv" % (tag, c)), dtype={"image_id": str, "patient_id": str})
            comunes = [k for k in feats if k in zt.files]
            feats = {k: np.concatenate([feats[k], zt[k].astype(np.float64)]) for k in comunes}
            metas.append(mt)
    m = pd.concat(metas, ignore_index=True)
    m["coleccion"] = coleccion_bimcv(m["image_id"])
    m["proyeccion"] = proyeccion(m["image_id"])
    m["val_robust"] = (m["conjunto"] == "val") & (m["source"] == "bimcv")
    m["test_bimcv"] = (m["conjunto"] == "test") & (m["source"] == "bimcv")
    if any(len(v) != len(m) for v in feats.values()):
        raise S.AlignmentError("%s: características y metadatos con longitudes distintas" % tag)
    return feats, m


def detector_c1(feats, m: pd.DataFrame, C: float = C_DETECTOR, feat: str = FEAT_DETECTOR, tta: str = "no"):
    """Detector de colección de V2.3.1. Con tta='no' es idéntico a `detector`. Con tta='espejo' (decisión m9 previa a la
    congelación) el ajuste y la estandarización son los mismos y la puntuación es la media de la función de decisión
    sobre la imagen y su espejo (`<feat>_espejo`)."""
    X = feats[feat]
    mk, t, origen = conjunto_ajuste(m)
    mu, sd = X[mk].mean(0), X[mk].std(0) + 1e-8
    f = _lr(C).fit(((X - mu) / sd)[mk], t[mk])
    s = f.decision_function((X - mu) / sd)
    if tta == "espejo":
        s = 0.5 * (s + f.decision_function((feats[feat + "_espejo"] - mu) / sd))
    elif tta not in ("no", None, False):
        raise ValueError("tta del detector: 'no' o 'espejo'")
    info = {"caracteristicas": feat, "C": C, "tta": "espejo" if tta == "espejo" else "no", "origen_ajuste": origen,
            "n_ajuste": int(mk.sum()), "n_antigua": int(t[mk].sum()), "n_nueva": int(mk.sum() - t[mk].sum()),
            "negativos_en_ajuste": 0}
    return s, info


def alinear(ids_ref, ids, valores) -> np.ndarray:
    s = pd.Series(np.asarray(valores, float), index=pd.Index([str(i) for i in ids]))
    falt = [i for i in ids_ref if str(i) not in s.index]
    if falt:
        raise S.AlignmentError("faltan %d imágenes al alinear (p. ej. %s)" % (len(falt), falt[0]))
    return s.loc[[str(i) for i in ids_ref]].to_numpy()


def auc_media(y, P, g, B=S.DEFAULT_B, seed=S.DEFAULT_BOOT_SEED) -> dict:
    """AUC media por semilla (ecuación 8) de P (S, n) con IC95 por remuestreo de pacientes que recalcula las S AUC."""
    y = np.asarray(y).astype(int)
    P = np.atleast_2d(np.asarray(P, float))
    D = auc_remuestreo(y, P, g, B, seed)
    a = np.array([S.auc_roc(y, p) for p in P])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # remuestreos sin una de las clases (NaN)
        lo, hi = _ic(np.nanmean(D, axis=1))
    return {"auc": float(a.mean()), "ic95_inf": lo, "ic95_sup": hi, "desv_semillas": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "auc_por_semilla": [float(v) for v in a], "n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
            "n_pacientes": int(len(np.unique(np.asarray(g).astype(str)))), "B_validos": int(np.isfinite(D).all(1).sum())}


def familia_no_inferioridad(y, g, ref, metodos: Dict[str, np.ndarray], semillas: Dict[str, Sequence] = None,
                            margenes=(MARGEN_NI, MARGEN_NI_SENS), comparables: Optional[Dict[str, bool]] = None,
                            B=S.DEFAULT_B, seed=S.DEFAULT_BOOT_SEED, emparejar: bool = False,
                            semillas_ref: Optional[Sequence] = None) -> pd.DataFrame:
    """F-C1 (ref única, p. ej. D_IN, de forma (1, n)) o F-C1b (`emparejar=True`: ref de forma (S_ref, n) con las
    semillas `semillas_ref`, emparejada con las del método). Para cada método: AUC media, Δ, IC95, p de superioridad
    (descriptivo) y, para cada margen, p de no inferioridad sin corregir y de Holm (al 0.025 unilateral, sobre los
    métodos comparables) y la decisión. El primer margen es el registrado (0.02); el segundo, la sensibilidad (0.01)."""
    filas = []
    ref = np.atleast_2d(np.asarray(ref, float))
    for met, P in metodos.items():
        P = np.atleast_2d(np.asarray(P, float))
        sem = list(semillas.get(met, range(P.shape[0]))) if semillas else list(range(P.shape[0]))
        if emparejar:
            idx = {str(s): i for i, s in enumerate(semillas_ref)}
            comunes = [i for i, s in enumerate(sem) if str(s) in idx]
            if not comunes:
                continue
            R = ref[[idx[str(sem[i])] for i in comunes]]
            P, sem = P[comunes], [sem[i] for i in comunes]
        else:
            R = ref[:1]
        r = diferencia(y, R, P, g, B, seed, margen=margenes[0], semillas=sem)
        fila = {"metodo": met, "n_semillas": r["n_semillas"], "semillas": ",".join(map(str, sem)),
                "auc_referencia": r["auc1"], "auc_metodo": r["auc2"], "delta": r["delta"], "ic95_inf": r["ic95_inf"],
                "ic95_sup": r["ic95_sup"], "p_superioridad": r["p_superioridad"],
                "comparable": True if comparables is None else bool(comparables.get(met, True)), "n": int(len(y))}
        D = auc_remuestreo(y, np.vstack([R if R.shape[0] == P.shape[0] else np.repeat(R, P.shape[0], 0), P]), g, B, seed)
        dd = np.nanmean(D[:, :P.shape[0]] - D[:, P.shape[0]:], axis=1)
        for mg in margenes:
            fila["p_ni_%.2f" % mg] = _p_mayor(dd, -float(mg))
        filas.append(fila)
    t = pd.DataFrame(filas)
    if not len(t):
        return t
    for mg in margenes:
        c = "p_ni_%.2f" % mg
        t["p_holm_%.2f" % mg] = np.nan
        t["no_inferior_%.2f" % mg] = False
        sub = t[t["comparable"]]
        if len(sub):
            h = S.holm({i: p for i, p in zip(sub.index, sub[c])}, alpha=ALPHA_HOLM)
            for i in sub.index:
                t.loc[i, "p_holm_%.2f" % mg] = h[i]["p_holm"]
                t.loc[i, "no_inferior_%.2f" % mg] = bool(h[i]["rechaza"])
    return t


def contrastes_resumen(df: pd.DataFrame, P: np.ndarray, B=S.DEFAULT_B, seed=S.DEFAULT_BOOT_SEED) -> list:
    """C1-iii, contrastes destacados: (a) negativos de bimcv frente a positivos de bimcv y (b) negativos de bimcv
    frente a positivos de stonybrook, ricord y sirm. P (S, n) alineada con df; AUC media por semilla con IC95."""
    col = etiquetar_coleccion(df)
    y = df["label"].to_numpy().astype(int)
    g = df["patient"].astype(str).to_numpy()
    neg = (col == "bimcv") & (y == 0)
    out = []
    for nombre, pos in (("a_bimcv_neg_vs_bimcv_pos", (np.char.startswith(col.astype(str), "bimcv_")) & (y == 1)),
                        ("b_bimcv_neg_vs_otras_fuentes_pos", np.isin(col, FUENTES_SOLO_POS) & (y == 1))):
        mk = neg | pos
        if neg.sum() < N_MIN or pos.sum() < N_MIN:
            out.append({"contraste": nombre, "auc": np.nan, "n_neg": int(neg.sum()), "n_pos": int(pos.sum())})
            continue
        r = auc_media(y[mk], np.atleast_2d(P)[:, mk], g[mk], B, seed)
        out.append({"contraste": nombre, **{k: v for k, v in r.items() if k != "auc_por_semilla"}})
    if len(out) == 2 and np.isfinite(out[0].get("auc", np.nan)) and np.isfinite(out[1].get("auc", np.nan)):
        out.append({"contraste": "b_menos_a", "auc": out[1]["auc"] - out[0]["auc"]})
    return out


def tabla_por_coleccion_media(df: pd.DataFrame, P: np.ndarray, B=S.DEFAULT_B, seed=S.DEFAULT_BOOT_SEED,
                              n_min: int = N_MIN) -> pd.DataFrame:
    """Como `tabla_por_coleccion`, pero con la AUC media por semilla de P (S, n) y su IC95 (C1-iii registrado)."""
    col = etiquetar_coleccion(df)
    y = df["label"].to_numpy().astype(int)
    g = df["patient"].astype(str).to_numpy()
    P = np.atleast_2d(P)
    filas = []
    for cn in COLECCIONES_NEG:
        mn = (col == cn) & (y == 0)
        for cp in COLECCIONES_POS:
            mp = (col == cp) & (y == 1)
            if mn.sum() < n_min or mp.sum() < n_min:
                continue
            mk = mn | mp
            r = auc_media(y[mk], P[:, mk], g[mk], B, seed)
            filas.append({"negativos": cn, "positivos": cp, **{k: v for k, v in r.items() if k != "auc_por_semilla"}})
    return pd.DataFrame(filas)


def solo_adquisicion(y, g, p_a, p_b, P_erm: Optional[np.ndarray] = None, B=S.DEFAULT_B, seed=S.DEFAULT_BOOT_SEED) -> list:
    """C1-iv: AUC de las variantes (a) y (b) con IC95 y, si se da P_erm (S, n), Δ frente a ERM (media por semilla)."""
    out = []
    for nombre, p in (("a_ajustada_en_entrenamiento", p_a), ("b_validacion_cruzada_en_el_conjunto", p_b)):
        if p is None:
            continue
        r = auc_ic(y, p, g, B, seed)
        fila = {"variante": nombre, **r}
        if P_erm is not None:
            dd = diferencia(y, np.asarray(p)[None], P_erm, g, B, seed)
            fila.update({"auc_erm_media": dd["auc2"], "delta_vs_erm": dd["delta"], "delta_ic95_inf": dd["ic95_inf"],
                         "delta_ic95_sup": dd["ic95_sup"]})
        out.append(fila)
    return out


def youden_por_semilla(y_val, P_val) -> list:
    """Umbral de Youden de cada semilla en su validación robusta (src/eval/probes.youden_threshold)."""
    return [float(S.youden_threshold(np.asarray(y_val).astype(int), np.asarray(p))) for p in np.atleast_2d(P_val)]


def externa(y_int, P_int, g_int, y_ext, P_ext, g_ext, umbrales: Sequence[float], gravedad=None,
            B=S.DEFAULT_B, seed=S.DEFAULT_BOOT_SEED) -> dict:
    """C1-vi: AUC media por semilla en la cohorte externa con IC95, sensibilidad y especificidad al umbral de Youden
    de la validación de cada semilla (medias), caída de la ecuación (11) con remuestreos independientes en cada
    conjunto (mismo B) y, si se da la gravedad de las positivas, la sensibilidad por grado."""
    P_int, P_ext = np.atleast_2d(P_int), np.atleast_2d(P_ext)
    y_int, y_ext = np.asarray(y_int).astype(int), np.asarray(y_ext).astype(int)
    if P_int.shape[0] != P_ext.shape[0] or len(umbrales) != P_ext.shape[0]:
        raise S.AlignmentError("semillas distintas entre el conjunto interno, el externo y los umbrales")
    ri, re = auc_media(y_int, P_int, g_int, B, seed), auc_media(y_ext, P_ext, g_ext, B, seed)
    Di = np.nanmean(auc_remuestreo(y_int, P_int, g_int, B, seed), axis=1)
    De = np.nanmean(auc_remuestreo(y_ext, P_ext, g_ext, B, seed), axis=1)
    lo, hi = _ic(Di - De)
    sens = [float((P_ext[i][y_ext == 1] >= u).mean()) for i, u in enumerate(umbrales)]
    esp = [float((P_ext[i][y_ext == 0] < u).mean()) for i, u in enumerate(umbrales)]
    out = {"auc_interna": ri["auc"], "auc_externa": re["auc"], "auc_externa_ic95_inf": re["ic95_inf"],
           "auc_externa_ic95_sup": re["ic95_sup"], "auc_externa_por_semilla": re["auc_por_semilla"],
           "caida": ri["auc"] - re["auc"], "caida_ic95_inf": lo, "caida_ic95_sup": hi,
           "sensibilidad_youden": float(np.mean(sens)), "especificidad_youden": float(np.mean(esp)),
           "umbrales": [float(u) for u in umbrales], "n_externa": int(len(y_ext)), "n_interna": int(len(y_int))}
    if gravedad is not None:
        gv = np.asarray(gravedad).astype(object)
        por = {}
        for k in sorted({str(x) for x in gv[y_ext == 1] if isinstance(x, str) and x}):
            mk = (y_ext == 1) & (gv == k)
            por[k] = {"n": int(mk.sum()), "sensibilidad": float(np.mean([(P_ext[i][mk] >= u).mean() for i, u in enumerate(umbrales)]))}
        out["sensibilidad_por_gravedad"] = por
    return out


def estratos_proyeccion(y, g, proy, ref, metodos: Dict[str, np.ndarray], B=S.DEFAULT_B, seed=S.DEFAULT_BOOT_SEED) -> pd.DataFrame:
    """C1-vii: dentro de AP y de PA, AUC del detector (ref, forma (1, n)) y Δ frente a cada método (media por
    semilla), descriptivo y sin Holm; y la AUC del clasificador de solo proyección en el conjunto completo."""
    y = np.asarray(y).astype(int)
    proy = np.asarray(proy).astype(object)
    filas = []
    for pr in ("AP", "PA"):
        mk = proy == pr
        if mk.sum() < 2 * N_MIN or len(set(y[mk].tolist())) < 2:
            continue
        rr = auc_ic(y[mk], np.atleast_2d(ref)[0][mk], np.asarray(g)[mk], B, seed)
        filas.append({"estrato": pr, "comparacion": "detector", "auc": rr["auc"], "ic95_inf": rr["ic95_inf"],
                      "ic95_sup": rr["ic95_sup"], "n": rr["n"], "n_neg": rr["n_neg"]})
        for met, P in metodos.items():
            r = diferencia(y[mk], np.atleast_2d(ref)[:, mk], np.atleast_2d(P)[:, mk], np.asarray(g)[mk], B, seed)
            filas.append({"estrato": pr, "comparacion": "detector menos " + met, "auc": r["auc2"], "delta": r["delta"],
                          "ic95_inf": r["ic95_inf"], "ic95_sup": r["ic95_sup"], "n": int(mk.sum()), "n_neg": int((y[mk] == 0).sum())})
    ok = np.isin(proy, ["AP", "PA"])
    if ok.any() and len(set(y[ok].tolist())) == 2:
        filas.append({"estrato": "todos", "comparacion": "solo proyeccion (AP = 1)",
                      "auc": S.auc_roc(y[ok], (proy[ok] == "AP").astype(float)), "n": int(ok.sum()),
                      "n_neg": int((y[ok] == 0).sum()), "frac_AP_neg": float((proy[ok][y[ok] == 0] == "AP").mean()),
                      "frac_AP_pos": float((proy[ok][y[ok] == 1] == "AP").mean())})
    return pd.DataFrame(filas)
