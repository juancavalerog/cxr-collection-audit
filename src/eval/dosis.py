"""Curva dosis-respuesta del benchmark de confusión controlada B3a (docs/08_evaluacion.md, versión 2, Section V2.4).

Lee las predicciones que escribe el entrenador, organizadas como
`results/train/<experimento>/seed<S>/predictions/<estado>_<conjunto>.csv` (columnas image_id, label, source, patient,
prob), y calcula para cada método la AUC frente a rho, las medidas resumen registradas y las comparaciones con ERM
pareadas por semilla y por imágenes, con remuestreo de pacientes.

Convención de nombres de experimento (contrato provisional, PLAN.md Section 13.8):
`<bench>_<orient>_<metodo>_rho<RRR>`, por ejemplo `b3a_A_erm_rho095`, donde RRR es round(100 rho) con tres cifras
(050, 060, ..., 095, 099, 100) y orient es A (A positiva, principal) o B (comprobación). Si la corrida trae
`rho` en `config_resolved.json` (claves `bench.rho` o `data.rho`), manda ese valor y se comprueba que coincide con el
nombre.

Remuestreo. `auc_remuestreo` genera los mismos remuestreos de pacientes que `src.eval.probes.cluster_bootstrap` y
`src.eval.stats.bootstrap_draws` con la misma semilla (mismo generador y mismo orden de llamadas), pero calcula todas
las AUC de una vez con pesos enteros por imagen: un paciente elegido k veces aporta sus imágenes con peso k. El
resultado coincide con el de recalcular la AUC sobre los índices repetidos (prueba en tests/test_eval_plan_b.py) y
es unas cien veces más rápido, lo que permite remuestrear a la vez todos los niveles de rho, semillas y métodos que
comparten imágenes. Así las medidas resumen que combinan niveles (área bajo la curva, caída) se remuestrean con los
mismos pacientes en todos los niveles.

Nunca abre un conjunto test si no se pide con `permitir_test=True`, que solo puede usarse tras la congelación del protocolo.
"""
from __future__ import annotations

import json
import re
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.eval import stats as S

RHOS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
# el área primaria se calcula sobre 0.5 a 0.99, donde todos los métodos están definidos (DFR no lo está en 1.0)
RHOS_AREA = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
METODOS_B3 = ("erm", "control_fuente", "groupdro", "dann", "coral", "mixstyle", "fourier", "lisa", "dfa_fuente", "dfr",
              "erm_mascara", "erm_cuotas")
REFERENCIA = "erm"
NOMBRE_RE = re.compile(r"^(?P<bench>b3a|b3b|b2)_(?P<orient>[AB])_(?P<metodo>[a-z0-9_]+?)_rho(?P<rho>\d{3})$")
# conjuntos con nombre (PLAN.md Section 13.7); el de validación con la distribución de entrenamiento es "val"
CONJ_SELECCION = "val"
CONJ_VAL_DESCONF = "val_desconf"
CONJ_TEST = "test_desconf"
CONJ_INVERTIDO = "test_invertido"
MARGEN_COSTE = 0.01          # no inferioridad del coste en rho = 0.5 (Section V2.4.3)
UMBRAL_P6 = 0.05             # caída mínima de ERM entre rho 0.5 y 1.0 en val_desconf (PLAN.md Section 13.9)


# ===================================================================================== remuestreo rápido
def pesos_remuestreo(groups, B: int, seed: int, inicio: int = 0, fin: Optional[int] = None) -> np.ndarray:
    """Matriz (b, n) de pesos por imagen de los remuestreos [inicio, fin) de pacientes con reemplazo.

    Reproduce la secuencia de `cluster_bootstrap`: una llamada `rng.integers(0, G, G)` por remuestreo con
    `np.random.default_rng(seed)` y los pacientes ordenados como `np.unique` de las cadenas."""
    g = np.asarray(groups).astype(str)
    _, inv = np.unique(g, return_inverse=True)
    G = int(inv.max()) + 1 if len(inv) else 0
    fin = int(B) if fin is None else int(fin)
    rng = np.random.default_rng(int(seed))
    out = np.empty((fin - inicio, len(g)), dtype=np.float64)
    for b in range(fin):
        pick = rng.integers(0, G, G)
        if b >= inicio:
            out[b - inicio] = np.bincount(pick, minlength=G)[inv]
    return out


def _auc_pesos(y: np.ndarray, p: np.ndarray, W: np.ndarray) -> np.ndarray:
    """AUC de Mann y Whitney con rangos medios para empates, ponderada por fila de W (b, n). NaN si falta una clase."""
    order = np.argsort(p, kind="mergesort")
    ps, ys = p[order], y[order]
    Ws = W[:, order]
    starts = np.r_[0, np.nonzero(np.diff(ps))[0] + 1]           # bloques de empates contiguos
    neg = np.add.reduceat(Ws * (ys == 0), starts, axis=1)
    pos = np.add.reduceat(Ws * (ys == 1), starts, axis=1)
    below = np.cumsum(neg, axis=1) - neg
    n0, n1 = neg.sum(1), pos.sum(1)
    num = (pos * (below + 0.5 * neg)).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = num / (n0 * n1)
    a[(n0 == 0) | (n1 == 0)] = np.nan
    return a


def auc_remuestreo(y, P, groups, B: int = S.DEFAULT_B, seed: int = S.DEFAULT_BOOT_SEED, bloque: int = 250) -> np.ndarray:
    """AUC remuestreadas por paciente de varias series sobre las mismas imágenes.

    `P` es una matriz (k, n) (o un vector) de puntuaciones. Devuelve una matriz (B, k) con la AUC de cada serie en
    cada remuestreo, con los mismos pacientes para todas las series."""
    y = np.asarray(y).astype(int)
    P = np.atleast_2d(np.asarray(P, float))
    if P.shape[1] != len(y) or len(groups) != len(y):
        raise S.AlignmentError("puntuaciones, etiquetas y pacientes con longitudes distintas")
    out = np.empty((int(B), P.shape[0]))
    for a in range(0, int(B), int(bloque)):
        z = min(int(B), a + int(bloque))
        W = pesos_remuestreo(groups, B, seed, a, z)
        for k in range(P.shape[0]):
            out[a:z, k] = _auc_pesos(y, P[k], W)
    return out


def _ic(v, alpha=0.05):
    return S._pct(v, alpha)


def _p_mayor(D, umbral=0.0):
    """p unilateral de remuestreo para H1: estadístico > umbral (unidad en numerador y denominador)."""
    D = np.asarray(D, float)
    ok = np.isfinite(D)
    return float(((D[ok] <= umbral).sum() + 1) / (ok.sum() + 1))


# ===================================================================================== lectura del árbol
def parse_nombre(exp: str) -> Optional[dict]:
    m = NOMBRE_RE.match(exp)
    if not m:
        return None
    d = m.groupdict()
    d["rho"] = int(d["rho"]) / 100.0
    return d


def _rho_config(run_dir: Path) -> Optional[float]:
    p = run_dir / "config_resolved.json"
    if not p.exists():
        return None
    cfg = json.loads(p.read_text(encoding="utf-8"))
    for a, b in (("bench", "rho"), ("data", "rho")):
        v = (cfg.get(a) or {}).get(b) if isinstance(cfg.get(a), dict) else None
        if v is not None:
            return float(v)
    return None


def descubrir(root, bench: str = "b3a", orient: str = "A", estado: str = "best",
              conjuntos: Sequence[str] = (CONJ_SELECCION, CONJ_VAL_DESCONF), permitir_test: bool = False) -> pd.DataFrame:
    """Inventario de corridas: una fila por (método, rho, semilla, conjunto) con la ruta del fichero de predicciones.

    Para DFR el estado es `dfr_<estado>`. Error si se pide un conjunto test sin `permitir_test`."""
    for c in conjuntos:
        if ("test" in str(c).lower() or S.is_test_set(c)) and not permitir_test:
            S.assert_not_test(c, "dosis-respuesta antes de la Puerta 7")
    rows = []
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        info = parse_nombre(exp_dir.name)
        if not info or info["bench"] != bench or info["orient"] != orient:
            continue
        for sd in S.find_seed_dirs(root, exp_dir.name).items():
            seed, run = sd
            rc = _rho_config(run)
            if rc is not None and abs(rc - info["rho"]) > 1e-9:
                raise ValueError("%s: rho del nombre (%.2f) distinto del de la configuración (%.2f)"
                                 % (run, info["rho"], rc))
            st = ("dfr_" + estado) if info["metodo"] == "dfr" else estado
            for c in conjuntos:
                f = run / "predictions" / ("%s_%s.csv" % (st, c))
                rows.append({"experimento": exp_dir.name, "metodo": info["metodo"], "rho": info["rho"], "semilla": int(seed),
                             "conjunto": c, "estado": st, "ruta": str(f), "existe": f.exists()})
    return pd.DataFrame(rows, columns=["experimento", "metodo", "rho", "semilla", "conjunto", "estado", "ruta", "existe"])


def cargar(inv: pd.DataFrame, conjunto: str, permitir_test: bool = False, verificador=None) -> Dict[tuple, pd.DataFrame]:
    """(método, rho, semilla) -> predicciones validadas del conjunto pedido.

    Desde el 17-11-2026 (hallazgo M1 y enmienda E7 de docs/08), abrir un test con `permitir_test=True` exige además la
    congelación válida del protocolo: se llama a `verificador(motivo)` si se da (pruebas, o un llamante que ya la
    verificó) o, si no, a `src.eval.desbloqueo.exigir_congelacion`."""
    if permitir_test and (S.is_test_set(conjunto) or "test" in str(conjunto).lower()):
        motivo = "dosis.cargar %s" % conjunto
        if verificador is not None:
            verificador(motivo)
        else:
            from src.eval.desbloqueo import exigir_congelacion
            exigir_congelacion(motivo)
    out = {}
    sub = inv[(inv["conjunto"] == conjunto) & inv["existe"]]
    for r in sub.itertuples():
        out[(r.metodo, float(r.rho), int(r.semilla))] = S.read_predictions(r.ruta, forbid_test=not permitir_test,
                                                                            context="dosis-respuesta")
    return out


# ===================================================================================== curva y resumen
def _matriz(preds: Dict[tuple, pd.DataFrame], metodo: str, rhos: Sequence[float], semillas: Sequence[int], ref: pd.DataFrame):
    """Matriz (len(rhos) * len(semillas), n) alineada con `ref`; filas en orden rho mayor, semilla menor."""
    filas = []
    for r in rhos:
        for s in semillas:
            d = preds[(metodo, r, s)]
            filas.append(S.align(ref, d, "referencia", "%s rho=%.2f semilla %d" % (metodo, r, s))["prob"].to_numpy())
    return np.vstack(filas)


def niveles_y_semillas(preds: Dict[tuple, pd.DataFrame], metodo: str):
    rh = sorted({k[1] for k in preds if k[0] == metodo})
    se = sorted({k[2] for k in preds if k[0] == metodo})
    return rh, se


def _semillas_comunes(preds, metodo, rhos):
    se = None
    for r in rhos:
        x = {k[2] for k in preds if k[0] == metodo and abs(k[1] - r) < 1e-9}
        se = x if se is None else se & x
    return sorted(se or [])


def area_dosis(rhos: Sequence[float], aucs: np.ndarray) -> np.ndarray:
    """Área normalizada bajo la curva dosis-respuesta (regla del trapecio sobre rho, dividida por el rango de rho):
    AUC media en el intervalo de rho estudiado. `aucs` (..., len(rhos))."""
    r = np.asarray(rhos, float)
    a = np.asarray(aucs, float)
    if len(r) < 2:
        return a[..., 0]
    tr = (a[..., 1:] + a[..., :-1]) / 2.0 * np.diff(r)
    return tr.sum(-1) / (r[-1] - r[0])


def medidas(rhos: Sequence[float], aucs: np.ndarray, diseno: Sequence[float] = RHOS_AREA) -> Dict[str, np.ndarray]:
    """Medidas resumen registradas a partir de AUC (..., len(rhos)) medias por semilla. El área se calcula sobre los
    niveles de `diseno` (por defecto 0.5 a 0.99) y solo si están todos (una curva incompleta no es comparable)."""
    r = [round(float(x), 4) for x in rhos]
    a = np.asarray(aucs, float)
    out = {}
    for v, nombre in ((1.0, "auc_rho100"), (0.99, "auc_rho099"), (0.5, "auc_rho050")):
        if v in r:
            out[nombre] = a[..., r.index(v)]
    if 0.5 in r and 1.0 in r:
        out["caida_050_100"] = a[..., r.index(0.5)] - a[..., r.index(1.0)]
    if set(round(float(x), 4) for x in diseno) <= set(r):
        idx = [r.index(round(float(x), 4)) for x in diseno]
        out["area_dosis"] = area_dosis(diseno, a[..., idx])
    return out


def _por_metodo(preds, m, ref, B, seed):
    rh, _ = niveles_y_semillas(preds, m)
    se = _semillas_comunes(preds, m, rh)
    if not se:
        raise ValueError("%s: ninguna semilla presente en todos sus niveles %s" % (m, rh))
    M = _matriz(preds, m, rh, se, ref)
    y, g = ref["label"].to_numpy(), ref["patient"].to_numpy()
    R, K = len(rh), len(se)
    punt = np.array([S.auc_roc(y, M[i]) for i in range(R * K)]).reshape(R, K)
    dr = auc_remuestreo(y, M, g, B, seed).reshape(int(B), R, K)
    return {"rhos": rh, "semillas": se, "puntual": punt, "draws": dr}


def _sub(d, rhos, semillas):
    i = [d["rhos"].index(r) for r in rhos]
    j = [d["semillas"].index(s) for s in semillas]
    return d["puntual"][np.ix_(i, j)], d["draws"][:, i][:, :, j]


def curva(preds: Dict[tuple, pd.DataFrame], metodos: Optional[Sequence[str]] = None, B: int = S.DEFAULT_B,
          seed: int = S.DEFAULT_BOOT_SEED, referencia: str = REFERENCIA, margen_coste: float = MARGEN_COSTE,
          diseno: Sequence[float] = RHOS_AREA):
    """Curva dosis-respuesta, medidas resumen y comparación con la referencia (ERM).

    Cada método usa sus propios niveles (DFR puede no estar definido en rho = 1.0, Section V2.4.6) y las semillas
    presentes en todos ellos; la comparación con la referencia usa los niveles y semillas comunes a los dos. Exige que
    todas las predicciones del conjunto tengan las mismas imágenes (test desconfundido común a todos los niveles y
    semillas, docs/02_datos.md Section 3.3, punto 6), de modo que los remuestreos de pacientes son los mismos para
    todos los métodos, niveles y semillas. Devuelve (tabla_curva, tabla_resumen, tabla_comparacion, info)."""
    metodos = list(metodos or sorted({k[0] for k in preds}))
    if referencia not in metodos:
        raise ValueError("falta la referencia %s" % referencia)
    ref = preds[sorted(k for k in preds if k[0] == referencia)[0]]
    y, g = ref["label"].to_numpy(), ref["patient"].to_numpy()
    res = {m: _por_metodo(preds, m, ref, B, seed) for m in metodos}
    filas_c, filas_r, filas_k = [], [], []
    for m in metodos:
        d = res[m]
        K = len(d["semillas"])
        media = d["puntual"].mean(1)
        dmed = np.nanmean(d["draws"], axis=2)
        for i, r in enumerate(d["rhos"]):
            lo, hi = _ic(dmed[:, i])
            filas_c.append({"metodo": m, "rho": r, "auc_media_semillas": float(media[i]),
                            "dt_semillas": float(d["puntual"][i].std(ddof=1)) if K > 1 else float("nan"),
                            "ic95_inf": lo, "ic95_sup": hi, "n_semillas": K,
                            "semillas": " ".join(map(str, d["semillas"])),
                            "aucs_por_semilla": " ".join("%.4f" % v for v in d["puntual"][i])})
        mp, mb = medidas(d["rhos"], media, diseno), medidas(d["rhos"], dmed, diseno)
        for k in mp:
            lo, hi = _ic(mb[k])
            filas_r.append({"metodo": m, "medida": k, "estimacion": float(mp[k]), "ic95_inf": lo, "ic95_sup": hi})
        if "area_dosis" not in mp:
            filas_r.append({"metodo": m, "medida": "area_dosis", "estimacion": float("nan"), "ic95_inf": float("nan"),
                            "ic95_sup": float("nan"), "nota": "curva incompleta: niveles %s" % d["rhos"]})
    dref = res[referencia]
    for m in metodos:
        if m == referencia:
            continue
        rh = [r for r in res[m]["rhos"] if r in dref["rhos"]]
        se = [s for s in res[m]["semillas"] if s in dref["semillas"]]
        if not rh or not se:
            continue
        pm, dm = _sub(res[m], rh, se)
        pr, drf = _sub(dref, rh, se)
        K = len(se)
        semi = pm - pr
        dif_p = semi.mean(1)
        dif_b = np.nanmean(dm - drf, axis=2)
        for i, r in enumerate(rh):
            lo, hi = _ic(dif_b[:, i])
            filas_k.append({"metodo": m, "medida": "delta_auc_rho%03d" % round(100 * r), "rho": r, "delta": float(dif_p[i]),
                            "ic95_inf": lo, "ic95_sup": hi, "p_unilateral": _p_mayor(dif_b[:, i]),
                            "semillas_a_favor": int((semi[i] > 0).sum()), "n_semillas": K})
        mp = medidas(rh, pm.mean(1) - pr.mean(1), diseno)
        mb = medidas(rh, dif_b, diseno)
        for k in ("area_dosis", "caida_050_100"):
            if k in mp:
                lo, hi = _ic(mb[k])
                # una caída menor es mejor: H1 es delta de caída < 0
                p = _p_mayor(-mb[k]) if k.startswith("caida") else _p_mayor(mb[k])
                filas_k.append({"metodo": m, "medida": "delta_" + k, "rho": float("nan"), "delta": float(mp[k]),
                                "ic95_inf": lo, "ic95_sup": hi, "p_unilateral": p, "semillas_a_favor": np.nan,
                                "n_semillas": K})
        if 0.5 in [round(x, 4) for x in rh]:
            i = [round(x, 4) for x in rh].index(0.5)
            lo, hi = _ic(dif_b[:, i])
            filas_k.append({"metodo": m, "medida": "coste_rho050_no_inferioridad", "rho": 0.5, "delta": float(dif_p[i]),
                            "ic95_inf": lo, "ic95_sup": hi, "p_unilateral": _p_mayor(dif_b[:, i], -float(margen_coste)),
                            "semillas_a_favor": int((semi[i] > 0).sum()), "n_semillas": K, "margen": float(margen_coste),
                            "no_inferior": bool(np.isfinite(lo) and lo > -margen_coste)})
    info = {"metodos": metodos, "referencia": referencia, "niveles_por_metodo": {m: res[m]["rhos"] for m in metodos},
            "semillas_por_metodo": {m: res[m]["semillas"] for m in metodos}, "diseno_rho": list(diseno), "B": int(B),
            "semilla_bootstrap": int(seed), "unidad_remuestreo": "paciente", "n_imagenes": int(len(y)),
            "n_pacientes": int(len(np.unique(g.astype(str)))), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
    return pd.DataFrame(filas_c), pd.DataFrame(filas_r), pd.DataFrame(filas_k), info


def familias_holm(comp: pd.DataFrame, alpha: float = 0.025) -> pd.DataFrame:
    """Añade el p de Holm por familia registrada (Section V2.2.4): F-C2-area (confirmatoria), F-C2-099, F-C2-100 y
    F-C2-coste (secundarias). Corregido el 17-11-2026 (enmienda E5 de docs/08, hallazgo m1): solo F-C2-area es
    confirmatoria y la caída no forma una familia registrada (es descriptiva). El resto de medidas es descriptivo; el
    p de Holm que se les calcula dentro de su medida es orientativo y no sostiene ninguna afirmación (columna papel).
    El análisis principal tras la enmienda E1 es src.eval.c2.curva_jerarquica con src.eval.c2.familias_c2."""
    comp = comp.copy()
    comp["familia"] = ""
    fam = {"delta_auc_rho100": "F-C2-100", "delta_auc_rho099": "F-C2-099", "delta_area_dosis": "F-C2-area",
           "coste_rho050_no_inferioridad": "F-C2-coste"}
    for medida, f in fam.items():
        comp.loc[comp["medida"] == medida, "familia"] = f
    otros = comp["familia"] == ""
    comp.loc[otros, "familia"] = "descriptiva_" + comp.loc[otros, "medida"]
    comp["p_holm"] = np.nan
    comp["rechaza_holm"] = False
    for f, sub in comp.groupby("familia"):
        h = S.holm({i: p for i, p in zip(sub.index, sub["p_unilateral"])}, alpha=alpha)
        for i in sub.index:
            comp.loc[i, "p_holm"] = h[i]["p_holm"]
            comp.loc[i, "rechaza_holm"] = h[i]["rechaza"]
    comp["confirmatoria"] = comp["familia"] == "F-C2-area"
    comp["papel"] = np.where(comp["confirmatoria"], "confirmatoria",
                             np.where(comp["familia"].str.startswith("F-C2-"), "secundaria", "descriptiva"))
    return comp


# ===================================================================================== piloto de B3a
def puerta6(preds_val_desconf: Dict[tuple, pd.DataFrame], B: int = S.DEFAULT_B, seed: int = S.DEFAULT_BOOT_SEED,
            umbral: float = UMBRAL_P6, metodo: str = REFERENCIA, rho_bajo: float = 0.5, rho_alto: float = 1.0) -> dict:
    """Criterio registrado del piloto de B3a (Section V2.6.1). Caída de ERM = media por semilla de
    AUC(rho_bajo) - AUC(rho_alto) en val_desconf. Informativo si la caída puntual es >= umbral, el extremo inferior
    del IC95 por pacientes es > 0 y la caída es positiva en cada semilla; no informativo si el extremo superior del
    IC95 es < umbral; indeterminado en cualquier otro caso (decide el autor, Section V2.6.1)."""
    claves = [k for k in preds_val_desconf if k[0] == metodo]
    rh = sorted({k[1] for k in claves})
    if rho_bajo not in rh or rho_alto not in rh:
        return {"veredicto": "no evaluable", "motivo": "faltan los niveles %.2f o %.2f de %s" % (rho_bajo, rho_alto, metodo),
                "niveles_presentes": rh}
    se = sorted({k[2] for k in claves if k[1] == rho_bajo} & {k[2] for k in claves if k[1] == rho_alto})
    if len(se) < 2:
        return {"veredicto": "no evaluable", "motivo": "se necesitan dos semillas comunes", "semillas": se}
    ref = preds_val_desconf[(metodo, rho_bajo, se[0])]
    try:
        M = _matriz(preds_val_desconf, metodo, [rho_bajo, rho_alto], se, ref)
    except S.AlignmentError as e:
        return {"veredicto": "no evaluable", "motivo": "val_desconf no tiene las mismas imágenes en los dos niveles: %s" % e}
    y, g = ref["label"].to_numpy(), ref["patient"].to_numpy()
    K = len(se)
    a = np.array([S.auc_roc(y, M[i]) for i in range(2 * K)]).reshape(2, K)
    D = auc_remuestreo(y, M, g, B, seed).reshape(int(B), 2, K)
    caida_b = np.nanmean(D[:, 0, :] - D[:, 1, :], axis=1)
    caida = float((a[0] - a[1]).mean())
    lo, hi = _ic(caida_b)
    por_semilla = a[0] - a[1]
    if caida >= umbral and lo > 0 and bool((por_semilla > 0).all()):
        ver = "informativo"
    elif hi < umbral:
        ver = "no informativo"
    else:
        ver = "indeterminado"
    return {"veredicto": ver, "metodo": metodo, "conjunto": CONJ_VAL_DESCONF, "rho_bajo": rho_bajo, "rho_alto": rho_alto,
            "semillas": se, "auc_rho_bajo_por_semilla": dict(zip(map(str, se), a[0].tolist())),
            "auc_rho_alto_por_semilla": dict(zip(map(str, se), a[1].tolist())), "caida": caida, "ic95": [lo, hi],
            "umbral": umbral, "caida_por_semilla": dict(zip(map(str, se), (a[0] - a[1]).tolist())),
            "semillas_con_caida_>=_umbral": int(((a[0] - a[1]) >= umbral).sum()), "B": int(B),
            "semilla_bootstrap": int(seed), "unidad_remuestreo": "paciente", "n_imagenes": int(len(y))}


def a_markdown(df: pd.DataFrame, cifras: int = 3) -> str:
    """Tabla Markdown sin dependencias (tabulate no está garantizado)."""
    if df is None or df.empty:
        return "(sin filas)\n"
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, (bool, np.bool_)):
            return "sí" if v else "no"
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else ("%.*f" % (cifras, v))
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


# ===================================================================================== árbol sintético (solo pruebas)
ESCENARIOS = ("informativo", "no_informativo", "uno_mejor", "ninguno")


def _poblacion_b3(n_por_celda: int, prev_a: float, prev_b: float, rng, prefijo: str) -> pd.DataFrame:
    """Imágenes sintéticas de dos fuentes (0 = A, 1 = B), una por paciente, como en B3a."""
    filas = []
    for src, prev in ((0, prev_a), (1, prev_b)):
        n = 2 * n_por_celda
        npos = int(round(prev * n))
        for k in range(n):
            pid = "%s_s%d_p%05d" % (prefijo, src, k)
            filas.append({"image_id": pid + ".png", "label": int(k < npos), "source": src, "patient": pid})
    return pd.DataFrame(filas)


def _prob(pop, senal, atajo, rng, orient="A"):
    """sigmoide(senal (2y - 1) + atajo (fuente con más positivos en entrenamiento) + ruido)."""
    pos_src = 0 if orient == "A" else 1
    z = senal * (2 * pop["label"].to_numpy() - 1) + atajo * np.where(pop["source"].to_numpy() == pos_src, 1.0, -1.0)
    z = z + rng.normal(0, 1.0, len(pop))
    return 1.0 / (1.0 + np.exp(-z))


def parametros_escenario(escenario: str, metodo: str, rho: float):
    """(señal, atajo) del modelo sintético. El atajo crece con rho y la señal baja un poco cuando rho = 1."""
    x = (rho - 0.5) / 0.5                                   # 0 en rho = 0.5, 1 en rho = 1
    senal = 1.6 - 0.3 * x ** 3
    if escenario == "no_informativo":
        return senal + 0.3 * x ** 3, 0.0
    atajo = 2.2 * x ** 4
    if escenario == "uno_mejor" and metodo == "groupdro":
        atajo *= 0.25
        senal -= 0.05                                        # pequeño coste sin confusión
    return senal, atajo


def arbol_sintetico_b3(root, escenario: str = "informativo", metodos: Sequence[str] = ("erm", "groupdro", "dfr"),
                       rhos: Sequence[float] = RHOS, semillas: Sequence[int] = (0, 1), n_test: int = 150,
                       n_val: int = 60, orient: str = "A", seed: int = 0, dfr_sin_rho100: bool = True,
                       estados: Sequence[str] = ("best",)) -> Path:
    """Árbol `results/train`-like con predicciones SINTÉTICAS de B3a (ningún dato real). Escribe, por método, rho y
    semilla, `val`, `val_desconf`, `test_desconf` y `test_invertido`. El test desconfundido y la validación
    desconfundida son los mismos para todos los niveles y semillas, como en el diseño. Con `dfr_sin_rho100`, DFR no
    tiene corrida en rho = 1.0 (no hay celdas minoritarias, Section V2.4.6)."""
    if escenario not in ESCENARIOS:
        raise ValueError(escenario)
    root = Path(root)
    rng0 = np.random.default_rng(seed)
    test = _poblacion_b3(n_test, 0.5, 0.5, rng0, "test")
    vdc = _poblacion_b3(n_val, 0.5, 0.5, rng0, "vdc")
    for m in metodos:
        for rho in rhos:
            if m == "dfr" and dfr_sin_rho100 and rho >= 1.0:
                continue
            pa, pb = (rho, 1 - rho) if orient == "A" else (1 - rho, rho)
            for sd in semillas:
                rng = np.random.default_rng(zlib.crc32(repr((m, round(rho, 3), sd, escenario, seed)).encode()))
                senal, atajo = parametros_escenario(escenario, m, rho)
                senal += rng.normal(0, 0.03)
                exp = "b3a_%s_%s_rho%03d" % (orient, m, round(100 * rho))
                d = root / exp / ("seed%d" % sd)
                (d / "predictions").mkdir(parents=True, exist_ok=True)
                cfg = {"experiment": exp, "seed": sd, "bench": {"id": "b3a", "rho": rho, "orient": orient},
                       "protocol": {"test_bloqueado": True, "congelado": None}, "sintetico": True}
                (d / "config_resolved.json").write_text(json.dumps(cfg), encoding="utf-8")
                val = _poblacion_b3(n_val, pa, pb, rng, "val_r%03d" % round(100 * rho))
                inv = _poblacion_b3(n_test, pb, pa, np.random.default_rng(7), "inv_r%03d" % round(100 * rho))
                st0 = "dfr_" if m == "dfr" else ""
                for st in estados:
                    for nombre, pop in (("val", val), ("val_desconf", vdc), ("test_desconf", test), ("test_invertido", inv)):
                        out = pop.copy()
                        out["prob"] = _prob(pop, senal, atajo, rng, orient)
                        out.to_csv(d / "predictions" / ("%s%s_%s.csv" % (st0, st, nombre)), index=False, float_format="%.8f")
    return root
