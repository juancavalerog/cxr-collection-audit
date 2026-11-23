"""Validación por simulación del factor del término de semillas (ecuación (17) de docs/08, ecuación (6) del
manuscrito) con la estructura del benchmark de derrame: ocho niveles de rho, cinco semillas en todos y cinco más solo
en rho 0.9, 0.95 y 0.99. Sin datos clínicos: el estadístico por semilla es lineal en los pacientes.

Modelo de cada réplica (nivel r, semilla s, paciente i = 1..n):
    Delta_{s,r} = mu_r + a_{s,r} + (1/n) sum_i (u_{i,r} + v_{i,r,s}),
a_{s,.} ~ N(0, Sigma_a) (efecto de semilla, correlacionado entre niveles), u_{i,.} ~ N(0, Sigma_u) (pacientes, común a
las semillas), v_{i,.,s} ~ N(0, Sigma_v) (interacción). Remuestreo de pacientes: los mismos pesos multinomiales para
todos los niveles y semillas. Sobre cada réplica se aplican, con las funciones congeladas de src/eval/jerarquico.py:
el estimador original, el corregido (scripts/correccion/ec6.py) y la alternativa sin estratos (todas las semillas
remuestreadas juntas, factor original). Se mide la razón entre la varianza de remuestreo y la verdadera (analítica)
por nivel y para el área trapezoidal de rho 0.5 a 0.99, la cobertura del IC95 de percentiles y el error de tipo I de la
prueba unilateral al 0.025 (ecuación (3)) bajo mu = 0.

    python scripts/correccion/simulacion.py correr realista --reps 4000 --presupuesto-s 170   # por tandas, reanudable
    python scripts/correccion/simulacion.py resumir
    python scripts/correccion/simulacion.py revisores        # casos deterministas de los revisores
    python scripts/correccion/simulacion.py todo --dir logs/correccion_ec6/sim_contenedor   # todo de una vez

Las réplicas se generan con semillas deterministas (escenario y número de réplica), de modo que una ejecución desde
cero en otro directorio reproduce las mismas réplicas.
"""
from __future__ import annotations

import argparse
import zlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.correccion import ec6  # noqa: E402
from src.eval import dosis as D  # noqa: E402
from src.eval import jerarquico as J  # noqa: E402

RHOS = np.array(D.RHOS)
R, K = 8, 10
PRES = np.zeros((R, K), bool)
PRES[:, :5] = True
PRES[4:7, 5:] = True
W_AREA = None
SAL = ROOT / "logs" / "correccion_ec6" / "sim"
OUT = ROOT / "results" / "correccion_ec6" / "simulacion"
B = 2000
N_PAC = 400
ALPHA = 0.025

# magnitudes del orden de las observadas en results/b3/test_A_best/comparacion_erm.csv (medianas por nivel de la
# desviación de semillas sigma_S, de la de pacientes de la media V_P y de la interacción de una semilla)
SIGMA_OBS = np.array([0.0009, 0.0026, 0.0046, 0.0162, 0.0205, 0.0294, 0.0245, 0.0182])
SDB_OBS = np.array([0.0008, 0.0009, 0.0011, 0.0016, 0.0014, 0.0015, 0.0011, 0.0025])   # Var(b) del común
TAU_OBS = np.array([0.0021, 0.0023, 0.0027, 0.0033, 0.0036, 0.0040, 0.0034, 0.0038])   # interacción por semilla

ESCENARIOS = {
    "realista": dict(sigma=SIGMA_OBS, sdb=SDB_OBS, tau=TAU_OBS, rho_a=0.6, rho_u=0.8, rho_v=0.5,
                     texto="magnitudes observadas; efecto de semilla con correlación 0.6 entre niveles contiguos"),
    "realista_corr_alta": dict(sigma=SIGMA_OBS, sdb=SDB_OBS, tau=TAU_OBS, rho_a=0.9, rho_u=0.8, rho_v=0.5,
                               texto="como el realista con correlación 0.9 del efecto de semilla entre niveles"),
    "interaccion_grande": dict(sigma=SIGMA_OBS, sdb=SDB_OBS, tau=SIGMA_OBS, rho_a=0.9, rho_u=0.8, rho_v=0.0,
                               texto="interacción tan grande como el efecto de semilla y sin correlación entre "
                                     "niveles (estructura no proporcional)"),
    "sin_semillas": dict(sigma=np.zeros(R), sdb=SDB_OBS, tau=TAU_OBS, rho_a=0.6, rho_u=0.8, rho_v=0.5,
                         texto="sin efecto de semilla (solo pacientes e interacción)"),
}


def _ar(sd, rho):
    idx = np.arange(len(sd))
    return np.outer(sd, sd) * rho ** np.abs(idx[:, None] - idx[None, :])


def pesos_area():
    w = np.zeros(R)
    x = RHOS[:7]
    for k in range(6):
        h = (x[k + 1] - x[k]) / 2
        w[k] += h
        w[k + 1] += h
    return w / (x[-1] - x[0])


def verdad(e):
    """Covarianza analítica (R, R) de la media por semilla de cada nivel."""
    Sa, Sb, Sc = _ar(e["sigma"], e["rho_a"]), _ar(e["sdb"], e["rho_u"]), _ar(e["tau"], e["rho_v"])
    Kr = PRES.sum(1)
    comun = (PRES[:, None, :] & PRES[None, :, :]).sum(2)              # semillas compartidas entre niveles
    return (Sa + Sc) * comun / np.outer(Kr, Kr) + Sb


def multiplicidades_sin_estratos(Bn, seed):
    rng = np.random.default_rng(int(seed))
    pick = rng.integers(0, K, size=(Bn, K))
    C = np.zeros((Bn, K))
    for j in range(K):
        np.add.at(C, (np.arange(Bn), pick[:, j]), 1)
    return C


def replica(e, rng, seed_sem):
    La = np.linalg.cholesky(_ar(e["sigma"], e["rho_a"]) + 1e-18 * np.eye(R))
    Lu = np.linalg.cholesky(_ar(e["sdb"] * np.sqrt(N_PAC), e["rho_u"]) + 1e-18 * np.eye(R))
    Lv = np.linalg.cholesky(_ar(e["tau"] * np.sqrt(N_PAC), e["rho_v"]) + 1e-18 * np.eye(R))
    a = rng.standard_normal((K, R)) @ La.T                                # (K, R)
    u = rng.standard_normal((N_PAC, R)) @ Lu.T                            # (n, R)
    v = rng.standard_normal((N_PAC, K, R)) @ Lv.T                         # (n, K, R)
    Y = (u[:, None, :] + v).reshape(N_PAC, K * R)                         # contribución por paciente
    pick = rng.integers(0, N_PAC, size=(B, N_PAC))
    Wt = np.zeros((B, N_PAC))
    np.add.at(Wt, (np.repeat(np.arange(B), N_PAC), pick.ravel()), 1)
    Xb = (Wt @ Y / N_PAC).reshape(B, K, R).transpose(0, 2, 1) + a.T[None]  # (B, R, K)
    punt = (Y.mean(0).reshape(K, R) + a).T                                 # (R, K)
    Xb = np.where(PRES[None], Xb, np.nan)
    punt = np.where(PRES, punt, np.nan)
    C = J.multiplicidades(PRES, B, seed_sem)
    out = {}
    out["original"] = J.extraer(Xb, punt, PRES, C)
    with ec6.parche("corregido"):
        C2 = J.multiplicidades(PRES, B, seed_sem)
        out["corregido"] = J.extraer(Xb, punt, PRES, C2)
    assert np.array_equal(C, C2)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["sin_estratos"] = J.extraer(Xb, punt, PRES, multiplicidades_sin_estratos(B, seed_sem + 7))
    out["pacientes"] = J.media_semillas(Xb, PRES)
    est = J.media_semillas(punt[None], PRES)[0]
    return est, out


def ficheros(nombre):
    import re
    return sorted(f for f in SAL.glob("%s_*.npz" % nombre) if re.fullmatch(re.escape(nombre) + r"_\d{6}\.npz", f.name))


def correr(nombre, reps, presupuesto, semilla0=0):
    e = ESCENARIOS[nombre]
    SAL.mkdir(parents=True, exist_ok=True)
    w = pesos_area()
    t0 = time.time()
    hechos = ficheros(nombre)
    n_hecho = sum(int(np.load(f)["n"]) for f in hechos)
    tanda = []
    i = n_hecho
    while i < reps and time.time() - t0 < presupuesto:
        rng = np.random.default_rng([semilla0, zlib.crc32(nombre.encode()), i])
        est, out = replica(e, rng, 1000003 + i)
        fila = {"est": est, "area": float(w @ np.nan_to_num(est))}
        for k, T in out.items():
            ok = np.isfinite(T)
            Ta = np.nansum(T * w[None], 1)
            Ta[~np.all(np.isfinite(T[:, :7]), 1)] = np.nan
            fila[k] = np.stack([np.nanvar(T, 0), np.nanpercentile(T, 2.5, 0), np.nanpercentile(T, 97.5, 0),
                                ((T <= 0) & ok).sum(0) + 1, ok.sum(0) + 1])                 # (5, R)
            fa = np.isfinite(Ta)
            fila[k + "_area"] = np.array([np.nanvar(Ta), np.nanpercentile(Ta, 2.5), np.nanpercentile(Ta, 97.5),
                                          ((Ta <= 0) & fa).sum() + 1, fa.sum() + 1])
        tanda.append(fila)
        i += 1
    if tanda:
        claves = tanda[0].keys()
        np.savez(SAL / ("%s_%06d.npz" % (nombre, n_hecho)), n=len(tanda),
                 **{k: np.array([f[k] for f in tanda]) for k in claves})
    print("%s: %d de %d réplicas (%.0f s)" % (nombre, i, reps, time.time() - t0))
    return i


def resumir():
    w = pesos_area()
    filas = []
    for nombre, e in ESCENARIOS.items():
        fs = ficheros(nombre)
        if not fs:
            continue
        Z = [np.load(f) for f in fs]
        n = sum(int(z["n"]) for z in Z)
        Sig = verdad(e)
        vt = np.diag(Sig)
        va = float(w @ Sig @ w)
        est = np.concatenate([z["est"] for z in Z])
        area = np.concatenate([z["area"] for z in Z])
        for k in ("original", "corregido", "sin_estratos", "pacientes"):
            M = np.concatenate([z[k] for z in Z])                    # (n, 5, R)
            Ma = np.concatenate([z[k + "_area"] for z in Z])         # (n, 5)
            for r in list(range(R)) + ["area"]:
                if r == "area":
                    var, lo, hi, num, den, truth, e_ = Ma[:, 0], Ma[:, 1], Ma[:, 2], Ma[:, 3], Ma[:, 4], va, area
                    nivel = "área 0.5 a 0.99"
                else:
                    var, lo, hi, num, den, truth, e_ = M[:, 0, r], M[:, 1, r], M[:, 2, r], M[:, 3, r], M[:, 4, r], vt[r], est[:, r]
                    nivel = "%.2f" % RHOS[r]
                p = num / den
                cob = (lo <= 0) & (hi >= 0)
                rej = p <= ALPHA
                filas.append({"escenario": nombre, "estimador": k, "nivel": nivel, "replicas": n,
                              "K": 10 if (r != "area" and 4 <= r <= 6) else (5 if r != "area" else np.nan),
                              "var_verdadera": truth, "var_empirica_estimador": float(np.var(e_)),
                              "razon_varianza": float(np.mean(var) / truth),
                              "razon_varianza_ee_mc": float(np.std(var) / np.sqrt(n) / truth),
                              "cobertura_ic95": float(cob.mean()), "cobertura_ee_mc": float(np.sqrt(cob.mean() * (1 - cob.mean()) / n)),
                              "error_tipo_I_0.025": float(rej.mean()), "tipo_I_ee_mc": float(np.sqrt(rej.mean() * (1 - rej.mean()) / n)),
                              "remuestreos_no_validos_medio": float(B + 1 - np.mean(den))})
    t = pd.DataFrame(filas)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "simulacion.csv", index=False)
    clave = t[t.nivel.isin(["0.90", "0.95", "0.99", "área 0.5 a 0.99", "1.00", "0.80"])]
    md = ["# Validación por simulación del término de semillas", "",
          "Estructura del benchmark de derrame (ocho niveles, cinco semillas en todos y cinco más en rho 0.9, 0.95 y "
          "0.99), %d pacientes por réplica, B = %d, nivel 0.025 unilateral. Escenarios: %s." % (
              N_PAC, B, "; ".join("%s: %s" % (k, v["texto"]) for k, v in ESCENARIOS.items())), "",
          D.a_markdown(clave[["escenario", "estimador", "nivel", "replicas", "razon_varianza", "cobertura_ic95",
                              "cobertura_ee_mc", "error_tipo_I_0.025", "tipo_I_ee_mc"]], 4)]
    (OUT / "simulacion.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(clave[["escenario", "estimador", "nivel", "replicas", "razon_varianza", "cobertura_ic95",
                 "error_tipo_I_0.025"]].to_string())


def revisores():
    """Casos deterministas de la revisión (sin pacientes): varianza exacta del término de semillas en un nivel con dos
    estratos de cinco, objetivo s2 / K, y comprobación por remuestreo con B = 20 000."""
    casos = {"cinco_0_y_cinco_0.1": np.r_[np.zeros(5), np.full(5, 0.1)],
             "0_a_0.09": np.arange(10) * 0.01,
             "medias_de_bloque_distintas_con_variacion_interna": np.r_[0.015 + np.linspace(-0.01, 0.01, 5),
                                                                       0.040 + np.linspace(-0.01, 0.01, 5)]}
    P = np.zeros((2, K), bool)
    P[0, :5] = True
    P[1, :] = True
    filas = []
    for nombre, v in casos.items():
        punt = np.vstack([v, v])
        X = np.broadcast_to(punt[None], (20000, 2, K)).copy()
        objetivo = v.var(ddof=1) / K
        E = ec6.efectos_centrados(punt, P)
        Vs, H = ec6.varianza_estratificada(E, P, J.estratos(P))
        ss_d = Vs[1] * K ** 2
        ss_t = ((v - v.mean()) ** 2).sum()
        res = {"caso": nombre, "objetivo_sigma2_S_sobre_K": objetivo, "SC_dentro_sobre_SC_total": ss_d / ss_t}
        for modo in ("original", "corregido"):
            ec6.AVISOS.clear()
            with ec6.parche(modo):
                h = J.remuestreo_jerarquico(X, P, seed=1, punt=punt)[:, 1]
            res["var_remuestreo_%s" % modo] = float(h.var())
            res["razon_%s" % modo] = float(h.var() / objetivo)
            if modo == "corregido":
                res["aviso_V0"] = bool(ec6.AVISOS)
        res["var_teorica_original"] = objetivo * ss_d / ss_t
        h1 = J.remuestreo_jerarquico(X[:, 1:], np.ones((1, K), bool), seed=1, punt=punt[1:])[:, 0]
        res["var_un_estrato"] = float(h1.var())
        filas.append(res)
    t = pd.DataFrame(filas)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "casos_revisores.csv", index=False)
    print(t.T.to_string())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("orden", choices=["correr", "resumir", "revisores", "todo"])
    ap.add_argument("--dir", default=None, help="directorio de las réplicas (por defecto logs/correccion_ec6/sim)")
    ap.add_argument("escenario", nargs="?", default="realista")
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--presupuesto-s", type=float, default=160)
    a = ap.parse_args(argv)
    global SAL
    if a.dir:
        SAL = (ROOT / a.dir) if not Path(a.dir).is_absolute() else Path(a.dir)
    if a.orden == "todo":
        for e in ESCENARIOS:
            while correr(e, a.reps, 1e9) < a.reps:
                pass
        revisores()
        resumir()
        return 0
    if a.orden == "correr":
        correr(a.escenario, a.reps, a.presupuesto_s)
    elif a.orden == "resumir":
        resumir()
    else:
        revisores()
    return 0


if __name__ == "__main__":
    sys.exit(main())
