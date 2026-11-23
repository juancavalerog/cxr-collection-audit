"""Remuestreo jerárquico de semillas y pacientes para las comparaciones de C2, C2-R y C3 (docs/08_evaluacion.md,
"Enmiendas previas a la congelación (17-11-2026)", enmienda E1; hallazgo B2 del revisor independiente).

Problema. `src.eval.dosis.curva` remuestrea solo pacientes con los modelos fijos: el intervalo mide la incertidumbre
del muestreo de imágenes, no la de las semillas, que en B3a fija a la vez la submuestra de entrenamiento y la
inicialización y es la fuente de variación dominante (review/comprobaciones/d_varianza_semillas.out).

Esquema registrado. El diseño es cruzado: todas las semillas se evalúan sobre las mismas imágenes. La diferencia
por semilla en un nivel se escribe Δ_s = μ + a_s + b + c_s, donde a_s es el efecto de la semilla (submuestra e
inicialización), b el del muestreo de pacientes común a todas las semillas y c_s la interacción semilla por paciente.
La varianza que debe recoger el intervalo es la de la ecuación (15) de docs/08 (enmienda E1):

    Var(media) = σ_S² / K + V_P,   con V_P = Var(b) + Var(c) / K.

1. V_P se estima con el remuestreo de pacientes de siempre (los mismos pesos por imagen para todos los métodos,
   niveles y semillas; `dosis.auc_remuestreo` con la semilla 2026).
2. σ_S² no es la varianza observada entre semillas, s², porque esta incluye la interacción: s² estima
   σ_S² + Var(c). Var(c) se obtiene de la diferencia entre la varianza de pacientes de una semilla sola, V_1, y la
   de la media, V_P: Var(c) = K (V_1 - V_P) / (K - 1). Entonces σ_S² = max(0, s² - Var(c)).
3. Cada remuestreo b suma a la media remuestreada por pacientes un término de semillas: la media, con las semillas
   remuestreadas con reemplazo, de los efectos centrados e_s = Δ_s - media, multiplicados por
   λ = sqrt(σ_S² / s²) · sqrt(K / (K - 1)). Así la varianza del término de semillas es σ_S² / K (corrige el factor
   (K - 1) / K del remuestreo de K semillas) y la interacción no se cuenta dos veces.

Las semillas se remuestrean como conglomerados: el mismo identificador para todos los métodos (la semilla fija la
misma submuestra para todos: comparación pareada) y para todos los niveles de rho en que existe (las submuestras de
una misma semilla en niveles contiguos comparten el 95 % de las imágenes de entrenamiento, frente al 57 % entre
semillas distintas). Para admitir cualquier número de semillas por nivel, se agrupan en estratos por su patrón de
presencia y se remuestrean dentro de cada estrato con su tamaño. El generador de semillas (`SEMILLA_SEMILLAS`) es
independiente del de pacientes, de modo que los remuestreos de pacientes siguen siendo los de `cluster_bootstrap`.

Con una sola semilla en un nivel no hay término de semillas y el intervalo es el de pacientes; la salida lo marca.
Sensibilidades registradas: (i) el remuestreo doble sin corrección (`modo="doble"`: semillas y pacientes a la vez,
que cuenta dos veces la interacción y es conservador) y (ii) la prueba t pareada entre semillas (K - 1 grados de
libertad). Las tres versiones se comprueban con casos sintéticos de varianza conocida (tests/test_eval_jerarquico.py).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy import stats as _st

SEMILLA_SEMILLAS = 20261117


def estratos(presencia: np.ndarray):
    """Lista de arrays de índices de semilla (columnas) con el mismo patrón de presencia por niveles (filas)."""
    P = np.asarray(presencia, bool)
    pats = {}
    for k in range(P.shape[1]):
        if P[:, k].any():
            pats.setdefault(P[:, k].tobytes(), []).append(k)
    return [np.array(v) for _, v in sorted(pats.items(), key=lambda kv: kv[1][0])]


def multiplicidades(presencia: np.ndarray, B: int, seed: int = SEMILLA_SEMILLAS) -> np.ndarray:
    """Matriz (B, K) con el número de veces que entra cada semilla en cada remuestreo (estratificado por patrón)."""
    P = np.asarray(presencia, bool)
    K = P.shape[1]
    rng = np.random.default_rng(int(seed))
    C = np.zeros((int(B), K))
    for idx in estratos(P):
        n = len(idx)
        pick = rng.integers(0, n, size=(int(B), n))
        for j in range(n):
            C[np.arange(int(B)), idx[pick[:, j]]] += 1
    return C


def media_semillas(X: np.ndarray, presencia: np.ndarray, C: Optional[np.ndarray] = None) -> np.ndarray:
    """Media por semilla de X (B, R, K) en cada remuestreo y nivel, con multiplicidades C (B, K) o, sin C, con todas
    las semillas presentes (peso 1). Las celdas ausentes (presencia False) no cuentan. Devuelve (B, R)."""
    X = np.asarray(X, float)
    P = np.asarray(presencia, bool)
    if C is None:
        C = np.ones((X.shape[0], X.shape[2]))
    Xz = np.where(P[None, :, :], X, 0.0)
    num = np.einsum("brk,bk->br", Xz, C)
    den = C @ P.T.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


def componentes(X: np.ndarray, punt: np.ndarray, presencia: np.ndarray) -> dict:
    """Componentes de varianza por nivel (ecuación (15)): V_P, V_1 medio, s², Var(c), σ_S² y el factor λ."""
    X = np.asarray(X, float)
    P = np.asarray(presencia, bool)
    R = P.shape[0]
    out = {k: np.full(R, np.nan) for k in ("V_P", "V_1", "s2", "var_interaccion", "sigma2_S", "lambda", "K")}
    hp = media_semillas(X, P)
    for r in range(R):
        ks = np.nonzero(P[r])[0]
        K = len(ks)
        out["K"][r] = K
        if K == 0:
            continue
        out["V_P"][r] = np.nanvar(hp[:, r])
        out["V_1"][r] = np.nanmean([np.nanvar(X[:, r, k]) for k in ks])
        if K < 2:
            out["s2"][r] = out["var_interaccion"][r] = out["sigma2_S"][r] = out["lambda"][r] = 0.0
            continue
        s2 = float(np.var(punt[r, ks], ddof=1))
        vi = max(0.0, K * (out["V_1"][r] - out["V_P"][r]) / (K - 1))
        s2S = max(0.0, s2 - vi)
        out["s2"][r], out["var_interaccion"][r], out["sigma2_S"][r] = s2, vi, s2S
        out["lambda"][r] = np.sqrt(s2S / s2) * np.sqrt(K / (K - 1)) if s2 > 0 else 0.0
    return out


def extraer(X: np.ndarray, punt: np.ndarray, presencia: np.ndarray, C: np.ndarray, modo: str = "corregido"):
    """Remuestreos (B, R) con multiplicidades de semillas C (B, K) ya generadas (para compartirlas entre métodos)."""
    P = np.asarray(presencia, bool)
    if modo == "doble":
        return media_semillas(X, P, C)
    if modo != "corregido":
        raise ValueError(modo)
    comp = componentes(X, punt, P)
    hp = media_semillas(X, P)
    medias = np.array([np.nanmean(np.where(P[r], punt[r], np.nan)) if P[r].any() else np.nan for r in range(P.shape[0])])
    E = np.where(P, np.asarray(punt, float) - medias[:, None], 0.0)            # efectos centrados (R, K)
    e = media_semillas(np.broadcast_to(E[None], (C.shape[0],) + E.shape), P, C)
    return hp + np.nan_to_num(comp["lambda"])[None, :] * np.nan_to_num(e)


def remuestreo_jerarquico(X: np.ndarray, presencia: np.ndarray, seed: int = SEMILLA_SEMILLAS, punt=None,
                          modo: str = "corregido") -> np.ndarray:
    """X (B, R, K): estadístico por remuestreo de pacientes, nivel y semilla (AUC o diferencia de AUC pareada);
    `punt` (R, K): el mismo estadístico con todas las imágenes. Devuelve (B, R). Sin `punt` solo puede usarse el modo
    doble."""
    C = multiplicidades(presencia, X.shape[0], seed)
    if punt is None:
        if modo != "doble":
            raise ValueError("el modo corregido necesita los valores puntuales por semilla (punt)")
        return media_semillas(X, presencia, C)
    return extraer(X, punt, presencia, C, modo)


def t_pareada(valores_por_semilla: Sequence[float], umbral: float = 0.0) -> dict:
    """Prueba t de una muestra (H1: media > umbral) sobre las diferencias por semilla; IC95 bilateral de t."""
    v = np.asarray([x for x in valores_por_semilla if np.isfinite(x)], float)
    K = len(v)
    if K < 2:
        return {"t_media": float(v.mean()) if K else float("nan"), "t_ic95_inf": float("nan"),
                "t_ic95_sup": float("nan"), "t_p_unilateral": float("nan"), "t_gl": K - 1}
    m, se = float(v.mean()), float(v.std(ddof=1) / np.sqrt(K))
    q = _st.t.ppf(0.975, K - 1)
    if se == 0:
        p = 0.0 if m > umbral else 1.0
    else:
        p = float(_st.t.sf((m - umbral) / se, K - 1))
    return {"t_media": m, "t_ic95_inf": m - q * se, "t_ic95_sup": m + q * se, "t_p_unilateral": p, "t_gl": K - 1}


def ic(v, alpha: float = 0.05):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan"), float("nan")
    return float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))


def p_mayor(D, umbral: float = 0.0) -> float:
    """p unilateral de remuestreo para H1: estadístico > umbral, ecuación (3) de la versión 1."""
    D = np.asarray(D, float)
    ok = np.isfinite(D)
    return float(((D[ok] <= umbral).sum() + 1) / (ok.sum() + 1))
