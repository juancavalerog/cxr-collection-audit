"""Tabla de impacto de la corrección del factor de la ecuación (17) de docs/08 (ecuación (6) del manuscrito):
compara los CSV publicados con los recalculados en results/correccion_ec6/ y escribe antes_despues.csv y .md.
No lee ninguna predicción: solo tablas ya calculadas.

    python scripts/correccion/impacto.py

Decisiones. En las familias (F-C2-area, F-C2-099, F-C2-100, F-C2-coste y F-C3, y sus versiones con la selección de
oráculo) la decisión es el rechazo con Holm al 0.025 unilateral (en F-C2-coste, la no inferioridad con margen 0.01).
En los descriptivos no hay decisión; se informa aparte si el IC95 excluye el cero antes y después, como lectura.
Valores p en el suelo de resolución: con B = 2 000 el menor valor p posible es 1/2001; se marca 'suelo'.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import dosis as D  # noqa: E402

PUB = ROOT / "results"
COR = ROOT / "results" / "correccion_ec6"
SUELO = 1.0 / 2001
TOL = 1e-12


def _p(p, n=1):
    if not np.isfinite(p):
        return ""
    if abs(p - SUELO) < 1e-12:
        return "1/2001 (suelo)"
    if n > 1 and abs(p - n * SUELO) < 1e-12:
        return "%d/2001 (suelo de Holm)" % n
    return "%.4f" % p


def _excluye(lo, hi):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return ""
    return "por encima" if lo > 0 else ("por debajo" if hi < 0 else "contiene el cero")


def comparacion(tabla, sub):
    a = pd.read_csv(PUB / sub / "comparacion_erm.csv")
    b = pd.read_csv(COR / sub / "comparacion_erm.csv")
    assert list(a.metodo) == list(b.metodo) and list(a.medida) == list(b.medida)
    filas = []
    for i in range(len(a)):
        x, y = a.iloc[i], b.iloc[i]
        cambia_ic = abs(x.ic95_inf - y.ic95_inf) > TOL or abs(x.ic95_sup - y.ic95_sup) > TOL
        fam = x.familia if isinstance(x.familia, str) else ""
        n = int(x.n_familia) if fam else 1
        es_ni = x.medida.startswith("coste")
        f = {"tabla": tabla, "metodo": x.metodo, "medida": x.medida, "familia": fam, "papel": x.papel,
             "estimacion": x.delta, "ic95_inf_antes": x.ic95_inf, "ic95_sup_antes": x.ic95_sup,
             "ic95_inf_despues": y.ic95_inf, "ic95_sup_despues": y.ic95_sup,
             "p_antes": x.p_unilateral, "p_despues": y.p_unilateral,
             "p_holm_antes": x.p_holm if fam else np.nan, "p_holm_despues": y.p_holm if fam else np.nan,
             "razon_anchuras": (y.ic95_sup - y.ic95_inf) / (x.ic95_sup - x.ic95_inf) if x.ic95_sup > x.ic95_inf else np.nan,
             "n_familia": n, "cambia_intervalo": bool(cambia_ic)}
        if fam:
            f["decision_antes"] = ("no inferior" if x.rechaza_holm else "no se declara no inferior") if es_ni else \
                ("rechaza H0" if x.rechaza_holm else "no rechaza")
            f["decision_despues"] = ("no inferior" if y.rechaza_holm else "no se declara no inferior") if es_ni else \
                ("rechaza H0" if y.rechaza_holm else "no rechaza")
            f["cambia_decision"] = bool(x.rechaza_holm != y.rechaza_holm)
        else:
            f["decision_antes"] = f["decision_despues"] = "descriptiva"
            f["cambia_decision"] = False
        f["ic_antes"] = _excluye(x.ic95_inf, x.ic95_sup)
        f["ic_despues"] = _excluye(y.ic95_inf, y.ic95_sup)
        f["cambia_lectura_ic"] = f["ic_antes"] != f["ic_despues"]
        f["p_antes_txt"], f["p_despues_txt"] = _p(x.p_unilateral), _p(y.p_unilateral)
        f["p_holm_antes_txt"] = _p(x.p_holm, n) if fam else ""
        f["p_holm_despues_txt"] = _p(y.p_holm, n) if fam else ""
        filas.append(f)
    return filas


def c3():
    a = pd.read_csv(PUB / "c2c3" / "c3-epoca.csv")
    b = pd.read_csv(COR / "c2c3" / "c3-epoca.csv")
    filas = []
    for i in range(len(a)):
        x, y = a.iloc[i], b.iloc[i]
        fam = x.familia if isinstance(x.familia, str) else ""
        f = {"tabla": "c3-epoca (media de los once métodos)", "metodo": "media de once", "medida": "delta_sel_rho%03d" % round(100 * x.rho),
             "familia": fam, "papel": x.papel, "estimacion": x.delta_sel_media_metodos,
             "ic95_inf_antes": x.ic95_inf, "ic95_sup_antes": x.ic95_sup, "ic95_inf_despues": y.ic95_inf,
             "ic95_sup_despues": y.ic95_sup, "p_antes": x.p_unilateral, "p_despues": y.p_unilateral,
             "p_holm_antes": x.p_holm, "p_holm_despues": y.p_holm,
             "razon_anchuras": (y.ic95_sup - y.ic95_inf) / (x.ic95_sup - x.ic95_inf), "n_familia": 3 if fam else 1,
             "cambia_intervalo": bool(abs(x.ic95_inf - y.ic95_inf) > TOL or abs(x.ic95_sup - y.ic95_sup) > TOL),
             "decision_antes": ("rechaza H0" if x.rechaza_holm else "no rechaza") if fam else "descriptiva",
             "decision_despues": ("rechaza H0" if y.rechaza_holm else "no rechaza") if fam else "descriptiva",
             "cambia_decision": bool(fam and x.rechaza_holm != y.rechaza_holm),
             "ic_antes": _excluye(x.ic95_inf, x.ic95_sup), "ic_despues": _excluye(y.ic95_inf, y.ic95_sup)}
        f["cambia_lectura_ic"] = f["ic_antes"] != f["ic_despues"]
        f["p_antes_txt"], f["p_despues_txt"] = _p(x.p_unilateral), _p(y.p_unilateral)
        f["p_holm_antes_txt"] = _p(x.p_holm, 3) if fam else ""
        f["p_holm_despues_txt"] = _p(y.p_holm, 3) if fam else ""
        filas.append(f)
    a = pd.read_csv(PUB / "c2c3" / "c3_epoca_por_metodo.csv")
    b = pd.read_csv(COR / "c2c3" / "c3_epoca_por_metodo.csv")
    for i in range(len(a)):
        x, y = a.iloc[i], b.iloc[i]
        f = {"tabla": "c3_epoca_por_metodo", "metodo": x.metodo, "medida": "delta_sel_rho%03d" % round(100 * x.rho),
             "familia": "", "papel": "descriptiva", "estimacion": x.delta_sel, "ic95_inf_antes": x.ic95_inf,
             "ic95_sup_antes": x.ic95_sup, "ic95_inf_despues": y.ic95_inf, "ic95_sup_despues": y.ic95_sup,
             "p_antes": np.nan, "p_despues": np.nan, "p_holm_antes": np.nan, "p_holm_despues": np.nan,
             "razon_anchuras": (y.ic95_sup - y.ic95_inf) / (x.ic95_sup - x.ic95_inf) if x.ic95_sup > x.ic95_inf else np.nan,
             "n_familia": 1, "cambia_intervalo": bool(abs(x.ic95_inf - y.ic95_inf) > TOL or abs(x.ic95_sup - y.ic95_sup) > TOL),
             "decision_antes": "descriptiva", "decision_despues": "descriptiva", "cambia_decision": False,
             "ic_antes": _excluye(x.ic95_inf, x.ic95_sup), "ic_despues": _excluye(y.ic95_inf, y.ic95_sup)}
        f["cambia_lectura_ic"] = f["ic_antes"] != f["ic_despues"]
        filas.append(f)
    return filas


def intervalos(tabla, fa, fb, claves, est, lo="ic95_inf", hi="ic95_sup", medida=None):
    a, b = pd.read_csv(fa), pd.read_csv(fb)
    assert a[claves].astype(str).equals(b[claves].astype(str))
    filas = []
    for i in range(len(a)):
        x, y = a.iloc[i], b.iloc[i]
        if not (np.isfinite(x[lo]) and np.isfinite(y[lo])):
            continue
        f = {"tabla": tabla, "metodo": " ".join(str(x[c]) for c in claves if c != "rho" and c != "medida"),
             "medida": medida(x) if medida else str(x.get("medida", "")), "familia": "", "papel": "descriptiva",
             "estimacion": x[est], "ic95_inf_antes": x[lo], "ic95_sup_antes": x[hi], "ic95_inf_despues": y[lo],
             "ic95_sup_despues": y[hi], "p_antes": np.nan, "p_despues": np.nan, "p_holm_antes": np.nan,
             "p_holm_despues": np.nan,
             "razon_anchuras": (y[hi] - y[lo]) / (x[hi] - x[lo]) if x[hi] > x[lo] else np.nan, "n_familia": 1,
             "cambia_intervalo": bool(abs(x[lo] - y[lo]) > TOL or abs(x[hi] - y[hi]) > TOL),
             "decision_antes": "descriptiva", "decision_despues": "descriptiva", "cambia_decision": False,
             "ic_antes": _excluye(x[lo], x[hi]), "ic_despues": _excluye(y[lo], y[hi])}
        f["cambia_lectura_ic"] = f["ic_antes"] != f["ic_despues"]
        filas.append(f)
    return filas


def orientacion():
    a = pd.read_csv(PUB / "c2c3" / "orientacion.csv").iloc[0]
    b = pd.read_csv(COR / "c2c3" / "orientacion.csv").iloc[0]
    la, ha = map(float, str(a.ic95).split())
    lb, hb = map(float, str(b.ic95).split())
    f = {"tabla": "orientacion", "metodo": "erm", "medida": "caida_A_menos_caida_B", "familia": "",
         "papel": "regla registrada sobre la estimación puntual", "estimacion": a.diferencia_A_menos_B,
         "ic95_inf_antes": la, "ic95_sup_antes": ha, "ic95_inf_despues": lb, "ic95_sup_despues": hb,
         "p_antes": np.nan, "p_despues": np.nan, "p_holm_antes": np.nan, "p_holm_despues": np.nan,
         "razon_anchuras": (hb - lb) / (ha - la), "n_familia": 1, "cambia_intervalo": True,
         "decision_antes": a.veredicto, "decision_despues": b.veredicto, "cambia_decision": bool(a.veredicto != b.veredicto),
         "ic_antes": _excluye(la, ha), "ic_despues": _excluye(lb, hb)}
    f["cambia_lectura_ic"] = f["ic_antes"] != f["ic_despues"]
    return [f]


def main():
    filas = []
    filas += comparacion("B3a selección principal, comparación con ERM", "b3/test_A_best")
    filas += comparacion("B3a selección de oráculo, comparación con ERM", "b3/test_A_bestdc")
    filas += c3()
    for sub, nom in (("b3/test_A_best", "principal"), ("b3/test_A_bestdc", "oráculo")):
        filas += intervalos("B3a curva, selección %s" % nom, PUB / sub / "curva.csv", COR / sub / "curva.csv",
                            ["metodo", "rho"], "auc_media_semillas", medida=lambda x: "auc_rho%03d" % round(100 * x.rho))
        filas += intervalos("B3a medidas resumen, selección %s" % nom, PUB / sub / "resumen.csv", COR / sub / "resumen.csv",
                            ["metodo", "medida"], "estimacion")
    filas += intervalos("AUC por fuente", PUB / "c2c3" / "fuente.csv", COR / "c2c3" / "fuente.csv",
                        ["metodo", "fuente", "rho"], "auc_media_semillas", medida=lambda x: "auc_rho%03d" % round(100 * x.rho))
    filas += orientacion()
    for ap in ("tronco", "etiqueta"):
        filas += intervalos("concordancia %s, columnas del análisis principal" % ap,
                            PUB / "sensibilidad" / ap / "concordancia.csv", COR / "sensibilidad" / ap / "concordancia.csv",
                            ["medida", "metodo", "rho"], "estimacion_principal", "ic95_inf_principal", "ic95_sup_principal",
                            medida=lambda x: "%s_rho%s" % (x.medida, "" if not np.isfinite(x.rho) else "%03d" % round(100 * x.rho)))
    t = pd.DataFrame(filas)
    todas = t.copy()
    t = t[t.cambia_intervalo].reset_index(drop=True)
    t.to_csv(COR / "antes_despues.csv", index=False)
    todas.to_csv(COR / "antes_despues_todas_las_filas.csv", index=False)
    fam = t[t.familia != ""]
    cambios = fam[fam.cambia_decision]
    lect = t[t.cambia_lectura_ic & (t.familia == "")]
    sin_cambio = todas[~todas.cambia_intervalo]
    md = ["# Impacto de la corrección del factor del término de semillas", "",
          "Filas cuyo IC95 cambia (%d de %d filas revisadas). Las %d filas restantes son idénticas bit a bit: niveles con "
          "un solo estrato de semillas (rho 0.5 a 0.8 y 1.0), la caída entre rho 0.5 y 1.0 y el coste en rho 0.5." % (
              len(t), len(todas), len(sin_cambio)), "",
          "## Decisiones de familia que cambian: %d" % len(cambios), ""]
    md += ["- %s, %s, %s: %s -> %s" % (r.tabla, r.metodo, r.medida, r.decision_antes, r.decision_despues)
           for r in cambios.itertuples()] or ["Ninguna."]
    md += ["", "## Comparaciones de familia con IC95 distinto", "",
           D.a_markdown(fam[["tabla", "familia", "metodo", "medida", "estimacion", "ic95_inf_antes", "ic95_sup_antes",
                             "ic95_inf_despues", "ic95_sup_despues", "p_antes_txt", "p_despues_txt", "p_holm_antes_txt",
                             "p_holm_despues_txt", "razon_anchuras", "decision_antes", "decision_despues",
                             "cambia_decision"]], 4),
           "## Lecturas descriptivas del IC95 que cambian (sin prueba de hipótesis): %d" % len(lect), "",
           D.a_markdown(lect[["tabla", "metodo", "medida", "estimacion", "ic95_inf_antes", "ic95_sup_antes",
                              "ic95_inf_despues", "ic95_sup_despues", "ic_antes", "ic_despues"]], 4) if len(lect) else "Ninguna.",
           "## Resumen de la razón de anchuras (después / antes) por tabla", "",
           D.a_markdown(t.groupby("tabla").razon_anchuras.describe()[["count", "min", "50%", "max"]].reset_index(), 3)]
    (COR / "antes_despues.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:12]))
    print("filas con cambio de IC: %d; decisiones de familia que cambian: %d; lecturas de IC que cambian: %d" % (
        len(t), len(cambios), len(lect)))


if __name__ == "__main__":
    main()
    from scripts.correccion import cifras        # cifras del manuscrito que cambian (results/correccion_ec6/cifras_manuscrito.csv)
    cifras.main()
