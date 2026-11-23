#!/usr/bin/env python3
"""Comprobación de que el suplemento no conserva intervalos ni valores p calculados con el factor original.

    python3 scripts/suplemento/comprobar_correccion.py

Lee results/suplemento/trazas.json y los CSV tabla_SNN.csv que escribe tablas.py, y
results/correccion_ec6/antes_despues_todas_las_filas.csv (786 filas, con el intervalo y los valores p con el factor
original y con el corregido). Para cada celda del suplemento cuya traza apunta a una fila de esa tabla comprueba que
el IC95 impreso (y los valores p de la comparación con ERM) coinciden con el valor corregido al redondeo impreso y, si
el valor original redondea distinto, que no es el original. La tabla de impacto de la corrección, que da a propósito
los dos intervalos, no se comprueba aquí. Comprueba además que ninguna traza apunta a un fichero original de results/
que tenga versión corregida. Termina con 0 si no hay fallos.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RES = RAIZ / "results"
SUP = RES / "suplemento"
CORR = "correccion_ec6/"
AD = RES / CORR / "antes_despues_todas_las_filas.csv"
EXCLUIDAS = {"impacto de la corrección del factor de semillas"}

TABLA_AD = {"b3/test_A_best/comparacion_erm.csv": "B3a selección principal, comparación con ERM",
            "b3/test_A_bestdc/comparacion_erm.csv": "B3a selección de oráculo, comparación con ERM",
            "b3/test_A_best/curva.csv": "B3a curva, selección principal",
            "b3/test_A_bestdc/curva.csv": "B3a curva, selección oráculo",
            "c2c3/c3_epoca_por_metodo.csv": "c3_epoca_por_metodo",
            "c2c3/c3-epoca.csv": "c3-epoca (media de los once métodos)",
            "c2c3/fuente.csv": "AUC por fuente",
            "c2c3/orientacion.csv": "orientacion"}


def med(r):
    return "rho%03d" % int(round(float(r) * 100))


def clave_ad(rel, f):
    """(tabla, metodo, medida) de antes_despues_todas_las_filas.csv para una celda, o None."""
    t = TABLA_AD.get(rel)
    if t is None:
        return None
    if "comparacion_erm" in rel:
        return (t, f["metodo"], f["medida"])
    if rel.endswith("curva.csv"):
        return (t, f["metodo"], "auc_" + med(f["rho"]))
    if rel.endswith("c3_epoca_por_metodo.csv"):
        return (t, f["metodo"], "delta_sel_" + med(f["rho"]))
    if rel.endswith("c3-epoca.csv"):
        return (t, "media de once", "delta_sel_" + med(f["rho"]))
    if rel.endswith("fuente.csv"):
        return (t, "%s %s" % (f["metodo"], f["fuente"]), "auc_" + med(f["rho"]))
    if rel.endswith("orientacion.csv"):
        return (t, f["metodo"], "caida_A_menos_caida_B")
    return None


def numeros(texto):
    t = texto.replace("−", "-")
    t = re.sub(r"(?<=\d) (?=\d{3}\b)", "", t)
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", t)]


def r(x, d):
    return round(float(x) + 0.0, d)


def igual(a, b, d):
    return abs(a - b) < 0.5 * 10 ** (-d) + 1e-9


def main():
    ad = pd.read_csv(AD)
    ad["clave"] = list(zip(ad.tabla, ad.metodo, ad.medida))
    ad["cambia"] = (ad.cambia_intervalo.astype(bool) | ((ad.p_antes - ad.p_despues).abs() > 0)
                    | ((ad.p_holm_antes - ad.p_holm_despues).abs() > 0))
    por_clave = {k: row for k, row in zip(ad.clave, ad.to_dict("records"))}
    traz = json.loads((SUP / "trazas.json").read_text(encoding="utf-8"))
    fallos, vistas_cambian, comprobadas, distintas_al_redondeo = [], set(), 0, 0
    fuentes_orig = set()
    for tid, t in traz.items():
        csv = pd.read_csv(SUP / ("tabla_S%02d.csv" % int(tid[1:])), dtype=str, keep_default_na=False)
        celdas = [(c, csv.iat[c["fila"], c["col"]]) for c in t["celdas"]] + [(c, c["texto"]) for c in t["leyenda"]]
        for c, txt in celdas:
            sp = c["spec"]
            if not isinstance(sp, dict):
                continue
            srcs = [sp["src"]] if "src" in sp else [x for x in sp.get("a", []) if isinstance(x, str) and x.endswith(".csv")]
            for s in srcs:
                if not s.startswith(CORR) and (RES / CORR / s).is_file():
                    fuentes_orig.add(s)
            if t["titulo"] in EXCLUIDAS:
                continue
            if sp.get("k") == "agg" and sp.get("n") == "icstr":
                src, f = sp["a"][0], sp["a"][1]
                cols, d = ["ic95_inf", "ic95_sup"], sp["a"][4]
                vals = numeros(txt)[1:3]
            elif sp.get("k") == "ic" and sp["c"][1:] in (["ic95_inf", "ic95_sup"],):
                src, f, d = sp["src"], sp["f"], sp["d"]
                cols, vals = ["ic95_inf", "ic95_sup"], numeros(txt)[1:3]
            elif sp.get("k") == "v" and sp["c"] in ("p_unilateral", "p_holm") and "comparacion_erm" in sp["src"]:
                src, f = sp["src"], sp["f"]
                cols, vals = [sp["c"]], numeros(txt)[:1]
                d = None
            else:
                continue
            rel = src[len(CORR):] if src.startswith(CORR) else src
            k = clave_ad(rel, f)
            if k is None or k not in por_clave:
                continue
            fila = por_clave[k]
            if not fila["cambia"]:
                continue
            vistas_cambian.add(k)
            if cols == ["ic95_inf", "ic95_sup"]:
                desp = [fila["ic95_inf_despues"], fila["ic95_sup_despues"]]
                ant = [fila["ic95_inf_antes"], fila["ic95_sup_antes"]]
            else:
                nombre = "p_despues" if cols[0] == "p_unilateral" else "p_holm_despues"
                desp, ant = [fila[nombre]], [fila[nombre.replace("despues", "antes")]]
                if pd.isna(desp[0]):
                    continue
                d = 3 if float(desp[0]) >= 0.005 else 4
                if float(desp[0]) < 0.0001:
                    continue
            comprobadas += 1
            if len(vals) < len(desp):
                fallos.append("%s %s: faltan cifras en '%s'" % (tid, k, txt))
                continue
            if not all(igual(v, x, d) for v, x in zip(vals, desp)):
                fallos.append("%s %s: '%s' no coincide con el valor corregido %s" % (tid, k, txt, desp))
            if any(r(a, d) != r(x, d) for a, x in zip(ant, desp)):
                distintas_al_redondeo += 1
                if all(igual(v, a, d) for v, a in zip(vals, ant)):
                    fallos.append("%s %s: '%s' es el valor con el factor original" % (tid, k, txt))
    cambian = ad[ad.cambia]
    no_vistas = cambian[~cambian.clave.isin(vistas_cambian)]
    print("filas de antes_despues_todas_las_filas.csv: %d; con intervalo o valor p distinto: %d" % (len(ad), len(cambian)))
    print("filas que cambian y aparecen en el suplemento: %d; celdas comprobadas: %d; de ellas con el valor original "
          "distinto al redondeo impreso: %d" % (len(vistas_cambian), comprobadas, distintas_al_redondeo))
    print("filas que cambian y no figuran en el suplemento, por tabla:")
    for tb, n in no_vistas.tabla.value_counts().items():
        print("   %-60s %d" % (tb, n))
    for s in sorted(fuentes_orig):
        fallos.append("la traza apunta al fichero original %s, que tiene versión corregida" % s)
    if fallos:
        print("FALLOS (%d):" % len(fallos))
        for f in fallos:
            print("  " + f)
        return 1
    print("0 fallos: ningún intervalo ni valor p del suplemento es el del factor original en las filas que cambian")
    return 0


if __name__ == "__main__":
    sys.exit(main())
