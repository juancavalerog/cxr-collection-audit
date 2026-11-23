"""Corrección del factor de calibración del término de semillas de la ecuación (17) de docs/08_evaluacion.md
(ecuación (6) del manuscrito). Código posterior a la apertura de los tests; no modifica ningún fichero congelado:
importa `src.eval.jerarquico` y, solo dentro del contexto `parche("corregido")`, sustituye en tiempo de ejecución
dos funciones del módulo (`multiplicidades`, que además anota los estratos con que generó cada matriz, y `extraer`,
que calcula el factor corregido). Fuera del contexto el módulo queda exactamente como estaba.

Problema. `jerarquico.multiplicidades` remuestrea las semillas dentro de estratos (patrón de presencia por niveles)
con el tamaño de cada estrato fijo. En un nivel r con K_r semillas repartidas en H_r estratos, la media remuestreada
de los efectos centrados e_s tiene varianza exacta

    V*_r = sum_h sum_{s en h} (e_s - ebar_h)^2 / K_r^2,

y no (K_r - 1) s_r^2 / K_r^2, que es la del remuestreo sin estratos con que se dedujo el factor original
lambda_r = sqrt(sigma2_S / s2) sqrt(K / (K - 1)). Con el factor original la varianza del término es
(sigma2_S / K) * SC_dentro / SC_total.

Factor corregido (definición por casos):
    lambda_r = 0                                    si K_r < 2 o sigma2_S,r = 0 (incluye s2_r = 0);
    lambda_r = sqrt(sigma2_S/s2) sqrt(K/(K-1))      si las semillas del nivel forman un solo estrato (coincide
                                                    algebraicamente con la fórmula general);
    lambda_r = sqrt((sigma2_S,r / K_r) / V*_r)      si hay dos o más estratos y V*_r > 0;
    lambda_r = 0 y aviso                            si V*_r = 0 con sigma2_S,r > 0 (el remuestreo estratificado
                                                    no genera variación: el nivel no puede calibrarse).
Con ello la varianza del término de semillas es exactamente sigma2_S,r / K_r en todos los casos calibrables.
"""
from __future__ import annotations

import contextlib
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import jerarquico as J  # noqa: E402

_ORIG = {"multiplicidades": J.multiplicidades, "extraer": J.extraer}
_ESTRATOS_DE_C = {}            # id(C) -> (C, lista de estratos) para las matrices generadas dentro del contexto
AVISOS = []                    # niveles con V* = 0 y sigma2_S > 0
REGISTRO = []                  # una entrada por llamada a extraer corregido: lambdas original y corregido por nivel
MODO = {"actual": "original"}


def _multiplicidades_anotadas(presencia, B, seed=J.SEMILLA_SEMILLAS):
    C = _ORIG["multiplicidades"](presencia, B, seed)
    _ESTRATOS_DE_C[id(C)] = (C, J.estratos(np.asarray(presencia, bool)))
    return C


def estratos_de(C, presencia):
    """Estratos con que se generó C. Si C no se generó dentro del contexto, los de `presencia`, comprobando que la
    suma de multiplicidades de cada estrato es constante e igual a su tamaño en todos los remuestreos."""
    ent = _ESTRATOS_DE_C.get(id(C))
    if ent is not None and ent[0] is C:
        return ent[1]
    est = J.estratos(np.asarray(presencia, bool))
    for idx in est:
        if not np.all(C[:, idx].sum(1) == len(idx)):
            raise RuntimeError("las multiplicidades no corresponden a los estratos de la presencia")
    return est


def efectos_centrados(punt, P):
    P = np.asarray(P, bool)
    medias = np.array([np.nanmean(np.where(P[r], punt[r], np.nan)) if P[r].any() else np.nan
                       for r in range(P.shape[0])])
    return np.where(P, np.asarray(punt, float) - medias[:, None], 0.0)


def varianza_estratificada(E, P, est):
    """V*_r exacta de la media estratificada de los efectos centrados en cada nivel; y número de estratos."""
    P = np.asarray(P, bool)
    R = P.shape[0]
    V = np.full(R, np.nan)
    H = np.zeros(R, int)
    for r in range(R):
        K = int(P[r].sum())
        if K == 0:
            continue
        ss = 0.0
        for idx in est:
            ks = idx[P[r, idx]]
            if len(ks) == 0:
                continue
            if len(ks) != len(idx):
                raise RuntimeError("un estrato aparece incompleto en un nivel")
            H[r] += 1
            v = E[r, ks]
            ss += float(((v - v.mean()) ** 2).sum())
        V[r] = ss / K ** 2
    return V, H


def lambdas(X, punt, P, est):
    """(lambda original, lambda corregido, V*, número de estratos, componentes) por nivel."""
    P = np.asarray(P, bool)
    comp = J.componentes(X, punt, P)
    E = efectos_centrados(punt, P)
    V, H = varianza_estratificada(E, P, est)
    lo = np.nan_to_num(comp["lambda"])
    lc = lo.copy()
    for r in range(P.shape[0]):
        K = int(P[r].sum())
        s2S = comp["sigma2_S"][r]
        if K < 2 or not (s2S > 0):
            lc[r] = 0.0
        elif H[r] <= 1:
            lc[r] = lo[r]                                 # un solo estrato: el factor original es el exacto
        elif V[r] > 0:
            lc[r] = float(np.sqrt((s2S / K) / V[r]))
        else:
            lc[r] = 0.0
            AVISOS.append({"nivel": r, "K": K, "sigma2_S": float(s2S), "V_estratificada": float(V[r])})
            warnings.warn("V* = 0 con sigma2_S > 0 en el nivel %d: el término de semillas no puede calibrarse" % r)
    return lo, lc, V, H, comp


def _extraer_corregido(X, punt, presencia, C, modo="corregido"):
    """Igual que `jerarquico.extraer` salvo el factor del término de semillas."""
    P = np.asarray(presencia, bool)
    if modo != "corregido":
        return _ORIG["extraer"](X, punt, presencia, C, modo)
    est = estratos_de(C, P)
    lo, lc, V, H, comp = lambdas(X, punt, P, est)
    hp = J.media_semillas(X, P)
    E = efectos_centrados(punt, P)
    e = J.media_semillas(np.broadcast_to(E[None], (C.shape[0],) + E.shape), P, C)
    REGISTRO.append({"K": comp["K"].tolist(), "H": H.tolist(), "lambda_original": lo.tolist(),
                     "lambda_corregido": lc.tolist(), "V_estratificada": V.tolist(),
                     "sigma2_S": np.nan_to_num(comp["sigma2_S"]).tolist(), "s2": np.nan_to_num(comp["s2"]).tolist()})
    return hp + lc[None, :] * np.nan_to_num(e)


@contextlib.contextmanager
def parche(modo: str):
    """modo 'original': no cambia nada (el control). modo 'corregido': factor corregido."""
    if modo not in ("original", "corregido"):
        raise ValueError(modo)
    if modo == "original":
        MODO["actual"] = "original"
        yield
        return
    J.multiplicidades = _multiplicidades_anotadas
    J.extraer = _extraer_corregido
    MODO["actual"] = "corregido"
    try:
        yield
    finally:
        J.multiplicidades = _ORIG["multiplicidades"]
        J.extraer = _ORIG["extraer"]
        _ESTRATOS_DE_C.clear()
        MODO["actual"] = "original"


def intacto() -> bool:
    return J.multiplicidades is _ORIG["multiplicidades"] and J.extraer is _ORIG["extraer"]
