#!/usr/bin/env python3
"""Tabla de validación de las corridas finales de B1 (COVIDx CXR-4): media y desviación típica entre semillas.

Lee solo summary.json de results/train/baseline_<metodo>_final/seed<S> (validación; no lee ningún test). Estado best
(dfr_best en DFR). Escribe results/b1/finales_validacion.csv y muestra la tabla en Markdown.

    python scripts/eval/tabla_b1_validacion.py
"""
import glob
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORDEN = ["erm", "erm_cuotas", "erm_mascara", "control_fuente", "groupdro", "dann", "coral", "mixstyle", "fourier", "lisa",
         "dfa_fuente", "dfr"]
SEMILLAS = [42, 123, 2024, 7, 999]


def main():
    filas, detalle = [], []
    for m in ORDEN:
        estado = "dfr_best" if m == "dfr" else "best"
        vals = {"val_robust_sel": [], "val_robust": [], "val": []}
        for s in SEMILLAS:
            f = os.path.join(ROOT, "results", "train", "baseline_%s_final" % m, "seed%d" % s, "summary.json")
            if not os.path.exists(f):
                sys.exit("falta %s" % f)
            r = json.load(open(f, encoding="utf-8"))["results"]
            for c in vals:
                v = r.get("%s/%s" % (estado, c), {}).get("auc_roc")
                if v is not None:
                    vals[c].append(v)
                detalle.append((m, s, estado, c, "" if v is None else "%.6f" % v))
        fila = {"metodo": m, "estado": estado}
        for c, x in vals.items():
            fila[c + "_n"] = len(x)
            fila[c + "_media"] = st.mean(x) if x else None
            fila[c + "_dt"] = st.stdev(x) if len(x) > 1 else None
            fila[c + "_min"] = min(x) if x else None
            fila[c + "_max"] = max(x) if x else None
        filas.append(fila)
    out = os.path.join(ROOT, "results", "b1", "finales_validacion.csv")
    cols = list(filas[0])
    with open(out, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in filas:
            f.write(",".join("" if r[c] is None else ("%.6f" % r[c] if isinstance(r[c], float) else str(r[c])) for c in cols) + "\n")
    with open(out.replace(".csv", "_detalle.csv"), "w", encoding="utf-8") as f:
        f.write("metodo,semilla,estado,conjunto,auc_roc\n")
        for d in detalle:
            f.write(",".join(str(x) for x in d) + "\n")

    def fm(r, c):
        return "no comparable" if r[c + "_media"] is None else "%.3f ± %.3f" % (r[c + "_media"], r[c + "_dt"])
    print("| Método | val_robust_sel | val_robust | Intervalo en val_robust_sel (mín. a máx.) |")
    print("|---|---|---|---|")
    for r in filas:
        print("| %s | %s | %s | %.3f a %.3f |" % (r["metodo"], fm(r, "val_robust_sel"), fm(r, "val_robust"),
                                                 r["val_robust_sel_min"], r["val_robust_sel_max"]))
    print("escrito %s" % out)


if __name__ == "__main__":
    main()
