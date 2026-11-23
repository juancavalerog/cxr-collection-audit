"""Precisión esperada de C2 con el remuestreo jerárquico (docs/08_evaluacion.md, enmienda E1, corrección de V2.6).
Solo usa predicciones de VALIDACIÓN ya existentes: el piloto de B3a (dos semillas; results/piloto_b3a/) y los
ensayos elegidos de la búsqueda (una semilla; results/train/b3a_A_*_t*/seed0/). No lee ningún test.

    python scripts/eval/precision_b3.py                    # salida en logs/eval_r2/precision/

Componentes (ecuación (15) de la enmienda E1):
- σ_P: error típico por pacientes de la diferencia método - ERM con una semilla, medido en val_desconf (800
  imágenes) y reescalado al tamaño de test_desconf (2 000) con la raíz de 800 / 2 000. Se usa como cota superior de
  V_P (la media de K semillas reduce la parte de interacción).
- σ_S: desviación entre semillas del efecto de semilla en la diferencia, descontada la interacción semilla por
  paciente (enmienda E1, paso 2), con las dos semillas del piloto (un grado de libertad por método y nivel).
En DFR se quitan de val_desconf las imágenes que también están en val (contienen su reajuste o comparten
imágenes con él; enmienda E5) y la diferencia se calcula frente a ERM en las mismas imágenes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import dosis as D  # noqa: E402
from src.eval import jerarquico as J  # noqa: E402
from src.eval import stats as S  # noqa: E402

METODOS = ("erm_cuotas", "erm_mascara", "groupdro", "dann", "coral", "mixstyle", "fourier", "lisa", "dfa_fuente", "dfr")
RHOS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
N_VAL, N_TEST = 800, 2000
Z_POT = norm.ppf(0.80)
Z_UNA = norm.ppf(1 - 0.025)
Z_HOLM = norm.ppf(1 - 0.025 / 10)


def _leer(run, metodo, conj="val_desconf", quitar_val=True):
    st = "dfr_best" if metodo == "dfr" else "best"
    d = S.read_predictions(Path(run) / "predictions" / ("%s_%s.csv" % (st, conj)), forbid_test=True)
    if metodo == "dfr" and quitar_val:
        v = pd.read_csv(Path(run) / "predictions" / ("%s_val.csv" % st), usecols=["image_id"], dtype=str)
        d = d[~d.image_id.isin(set(v.image_id))].reset_index(drop=True)
    return d


def _par(a, e, B, seed):
    e = e[e.image_id.isin(set(a.image_id))].reset_index(drop=True)
    a = S.align(e, a)
    y, g = e["label"].to_numpy(), e["patient"].to_numpy()
    Db = D.auc_remuestreo(y, np.vstack([a["prob"], e["prob"]]), g, B, seed)
    d = S.auc_roc(y, a["prob"].to_numpy()) - S.auc_roc(y, e["prob"].to_numpy())
    return d, Db[:, 0] - Db[:, 1], len(y), float(np.corrcoef(a["prob"], e["prob"])[0, 1])


def componente_pacientes(B, seed):
    eleg = json.load(open(ROOT / "results" / "b3a" / "congelados" / "elegidos.json", encoding="utf-8"))
    run = lambda m, r: ROOT / "results" / "train" / ("b3a_A_%s_rho%03d_t%d" % (m, round(100 * r), eleg["b3a_A_%s_rho%03d" % (m, round(100 * r))]["trial"])) / "seed0"
    filas, areas = [], {}
    for r in RHOS:
        e = _leer(run("erm", r), "erm")
        for m in METODOS:
            a = _leer(run(m, r), m)
            d, Db, n, c = _par(a, e, B, seed)
            filas.append({"metodo": m, "rho": r, "delta_val": d, "de_pacientes_val": float(np.nanstd(Db)), "n": n,
                          "de_pacientes_test": float(np.nanstd(Db) * np.sqrt(n / N_TEST)), "corr_prob": c})
            areas.setdefault(m, {})[r] = (Db, n)
    t = pd.DataFrame(filas)
    fa = []
    for m, d in areas.items():
        if len({v[1] for v in d.values()}) != 1:   # DFR: tamaños distintos por nivel, se omite del área
            continue
        Dm = np.vstack([d[r][0] for r in RHOS]).T
        a = D.area_dosis(RHOS, Dm)
        fa.append({"metodo": m, "de_pacientes_area_test": float(np.nanstd(a) * np.sqrt(N_VAL / N_TEST))})
    return t, pd.DataFrame(fa)


def componente_semillas(B, seed):
    base = ROOT / "results" / "piloto_b3a"
    filas = []
    for r in (0.5, 0.9, 1.0):
        for m, quitar in (("groupdro", True), ("dfr", False), ("dfr", True)):
            ex = base / ("b3a_A_%s_rho%03d" % (m, round(100 * r)))
            if not ex.exists() or not (ex / "seed0" / "predictions").exists():
                continue
            try:
                vals, Xs = [], []
                for s in (0, 1):
                    a = _leer(ex / ("seed%d" % s), m, quitar_val=quitar)
                    e = _leer(base / ("b3a_A_erm_rho%03d" % round(100 * r)) / ("seed%d" % s), "erm")
                    d, Db, n, _ = _par(a, e, B, seed)
                    vals.append(d)
                    Xs.append(Db)
            except FileNotFoundError:
                continue
            X = np.stack(Xs, axis=1)[:, None, :]
            comp = J.componentes(X, np.array(vals)[None], np.ones((1, 2), bool))
            filas.append({"metodo": m if m != "dfr" else ("dfr, sin las imágenes de val (subconjunto sesgado)" if quitar
                                                         else "dfr, val_desconf completo (contiene su reajuste)"),
                          "usar": m != "dfr" or not quitar, "rho": r, "delta_semilla0": vals[0], "delta_semilla1": vals[1],
                          "de_entre_semillas": float(np.std(vals, ddof=1)), "n_val": n,
                          "var_interaccion": float(comp["var_interaccion"][0]),
                          "sigma_S": float(np.sqrt(comp["sigma2_S"][0]))})
    return pd.DataFrame(filas)


def potencia(sp, ss, sa):
    """Semianchura y diferencia mínima detectable (potencia del 80 %) frente a ERM, con K = 5 y K = 10."""
    sP = sp.groupby("rho")["de_pacientes_test"].median()
    obs = ss[(ss.rho == 0.9) & ss.usar]["sigma_S"]
    s09 = float(np.sqrt(np.mean(obs ** 2))) if len(obs) else np.nan
    filas = []
    # área: pesos del trapecio de la ecuación (12) y σ_S por nivel lineal entre 0 (rho 0.5, observado ≈ 0) y σ_S(0.9),
    # igual a f σ_S(0.9) en 0.95 y 0.99, con correlación perfecta entre niveles (cota superior)
    w = np.diff(RHOS)
    pesos = (np.r_[w, 0] + np.r_[0, w]) / 2 / (RHOS[-1] - RHOS[0])
    sPa = float(sa["de_pacientes_area_test"].median()) if len(sa) else np.nan
    for etiqueta, f in (("observada en 0.9", 1.0), ("1.5 veces", 1.5), ("2 veces", 2.0)):
        perfil = np.array([0, 0.25, 0.5, 0.75, 1.0, f, f]) * s09
        sS = float((pesos * perfil).sum())
        for K in (5, 10):
            se = np.sqrt(sS ** 2 / K + sPa ** 2)
            filas.append({"rho": "área 0.5 a 0.99", "sigma_S_supuesta": etiqueta, "sigma_S": sS, "sigma_P_test": sPa,
                          "K": K, "error_tipico": se, "semianchura_IC95": Z_UNA * se,
                          "dmd_sin_correccion": (Z_UNA + Z_POT) * se, "dmd_holm_10": (Z_HOLM + Z_POT) * se})
    for r in (0.9, 0.95, 0.99):
        for etiqueta, f in (("observada en 0.9", 1.0), ("1.5 veces", 1.5), ("2 veces", 2.0)):
            sS = s09 * f
            for K in (5, 10):
                se = np.sqrt(sS ** 2 / K + sP[r] ** 2)
                filas.append({"rho": r, "sigma_S_supuesta": etiqueta, "sigma_S": sS, "sigma_P_test": float(sP[r]),
                              "K": K, "error_tipico": se, "semianchura_IC95": Z_UNA * se,
                              "dmd_sin_correccion": (Z_UNA + Z_POT) * se, "dmd_holm_10": (Z_HOLM + Z_POT) * se})
    return pd.DataFrame(filas), s09


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "logs" / "eval_r2" / "precision"))
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=S.DEFAULT_BOOT_SEED)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sp, sa = componente_pacientes(a.B, a.seed)
    ss = componente_semillas(a.B, a.seed)
    pot, s09 = potencia(sp, ss, sa)
    sp.to_csv(out / "componente_pacientes.csv", index=False)
    sa.to_csv(out / "componente_pacientes_area.csv", index=False)
    ss.to_csv(out / "componente_semillas_piloto.csv", index=False)
    pot.to_csv(out / "potencia_K5_K10.csv", index=False)
    res = sp.groupby("rho").agg(de_pacientes_test_min=("de_pacientes_test", "min"),
                                de_pacientes_test_mediana=("de_pacientes_test", "median"),
                                de_pacientes_test_max=("de_pacientes_test", "max"),
                                corr_min=("corr_prob", "min"), corr_max=("corr_prob", "max")).reset_index()
    res.to_csv(out / "resumen_pacientes.csv", index=False)
    md = ["# Precisión esperada de C2 (`python scripts/eval/precision_b3.py`, B = %d)" % a.B, "",
          "Cifras de planificación calculadas con validación; no son resultados.", "",
          "## Error típico por pacientes de Δ frente a ERM, una semilla, reescalado a test_desconf", "",
          D.a_markdown(res, 4), "## Área (0.5 a 0.99), una semilla", "", D.a_markdown(sa, 4),
          "## Componente de semillas (piloto, dos semillas)", "", D.a_markdown(ss, 4),
          "σ_S combinada en rho 0.9 (GroupDRO y DFR con val_desconf completo, media cuadrática): %.4f" % s09, "", "## Semianchura y diferencia mínima detectable", "",
          D.a_markdown(pot, 4)]
    (out / "precision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
