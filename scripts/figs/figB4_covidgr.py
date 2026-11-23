"""Figura B4 del plan B: rendimiento interno (parte bimcv del test) frente a la cohorte externa COVIDGR (C1-vi).

Una fila por método: AUC media de cinco semillas en la parte bimcv del test de COVIDx CXR-4 (marcador relleno) y en
COVIDGR (marcador hueco), con su IC95, unidas por una línea. Debajo, en gris, las dos referencias sin información de
clase. Línea vertical en 0.5 (azar).

Datos: results/c1/test/c1_vi_covidgr.csv y c1_iii_resumen.csv (python scripts/eval/auditoria_covidx.py --c1 test).

Uso: python scripts/figs/figB4_covidgr.py [--out results/figs/figB4_covidgr]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.viz import estilo as E  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import comun as C  # noqa: E402

NOMBRE = {"erm": "ERM", "erm_cuotas": "ERM con cuotas", "erm_mascara": "ERM con máscara", "control_fuente": "Control de fuente",
          "groupdro": "GroupDRO", "dann": "DANN", "coral": "CORAL", "mixstyle": "MixStyle", "fourier": "Fourier",
          "lisa": "LISA", "dfa_fuente": "DFA con fuente", "dfr": "DFR", "D_IN": "Sin clase (ImageNet)",
          "D_ERM": "Sin clase (ERM)"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "figs" / "figB4_covidgr"))
    a = ap.parse_args(argv)
    E.aplicar()
    c = C.leer("c1/test/c1_vi_covidgr.csv")
    r = C.leer("c1/test/c1_iii_resumen.csv")
    r = r[(r.estrato_bimcv == "todos") & (r.contraste == "a_bimcv_neg_vs_bimcv_pos")].set_index("metodo")
    c = c.set_index("metodo")
    assert len(c) == 14 and all(abs(c.loc[m, "auc_interna"] - r.loc[m, "auc"]) < 5e-4 for m in c.index)
    refs = ["D_ERM", "D_IN"]
    metodos = list(c.drop(index=refs).sort_values("auc_externa", ascending=False).index)
    filas = metodos + refs
    y = {m: len(filas) - 1 - i - (0.0 if m not in refs else 0.6) for i, m in enumerate(filas)}
    fig = E.figura(90, 140)
    ax = fig.add_axes([0.36, 0.175, 0.60, 0.80])
    ax.axvline(0.5, color=E.GRIS["tenue"], lw=0.8, ls=(0, (4, 2)), zorder=1)
    datos = []
    for m in filas:
        col = E.GRIS["tenue"] if m in refs else E.COLOR_METODO[m]
        mk = "o" if m in refs else E.MARCA_METODO[m]
        ai, ae = r.loc[m, "auc"], c.loc[m, "auc_externa"]
        ax.plot([ae, ai], [y[m], y[m]], color=col, lw=0.8, alpha=0.55, zorder=2)
        ax.plot([r.loc[m, "ic95_inf"], r.loc[m, "ic95_sup"]], [y[m], y[m]], color=col, lw=1.6, solid_capstyle="butt", zorder=3)
        ax.plot([c.loc[m, "auc_externa_ic95_inf"], c.loc[m, "auc_externa_ic95_sup"]], [y[m], y[m]], color=col, lw=1.6,
                solid_capstyle="butt", zorder=3)
        ax.plot([ai], [y[m]], marker=mk, ms=5, color=col, mfc=col, mew=1.0, zorder=4)
        ax.plot([ae], [y[m]], marker=mk, ms=C.hueco_ms(mk), color=col, mfc="white", mew=1.0, zorder=4)
        datos.append({"metodo": m, "auc_test_bimcv": ai, "ic95_inf_test_bimcv": r.loc[m, "ic95_inf"], "ic95_sup_test_bimcv": r.loc[m, "ic95_sup"],
                      "auc_covidgr": ae, "ic95_inf_covidgr": c.loc[m, "auc_externa_ic95_inf"], "ic95_sup_covidgr": c.loc[m, "auc_externa_ic95_sup"],
                      "caida": c.loc[m, "caida"]})
    ax.axhline((y[metodos[-1]] + y[refs[0]]) / 2.0, color=E.GRIS["rejilla"], lw=0.6)
    ax.set_yticks([y[m] for m in filas])
    ax.set_yticklabels([NOMBRE[m] for m in filas])
    for t, m in zip(ax.get_yticklabels(), filas):
        if m in refs:
            t.set_color(E.GRIS["tenue"])
    ax.set_ylim(min(y.values()) - 0.7, max(y.values()) + 0.7)
    ax.set_xlim(0.42, 0.90)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.grid(axis="x", color=E.GRIS["rejilla"], lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("AUC")
    ax.text(0.508, max(y.values()) + 0.62, "azar", fontsize=6, color=E.GRIS["tenue"], va="top", ha="left")
    # leyenda del relleno del marcador bajo el eje (solo la clave; la explicación está en el pie)
    from matplotlib.lines import Line2D
    g = E.GRIS["tenue"]
    fig.legend([Line2D([], [], ls="none", marker="o", ms=5, color=g, mfc=g, mew=1.0),
                Line2D([], [], ls="none", marker="o", ms=5, color=g, mfc="white", mew=1.0)],
               ["Parte BIMCV del test de COVIDx CXR-4", "COVIDGR (cohorte externa)"], loc="upper left",
               bbox_to_anchor=(0.04, 0.075), ncol=1, frameon=False, handletextpad=0.3, labelspacing=0.3, borderaxespad=0)
    out = C.guardar(fig, a.out, datos=pd.DataFrame(datos))
    print(out["png"])


if __name__ == "__main__":
    main()
