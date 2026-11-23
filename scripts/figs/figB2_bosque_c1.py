"""Figura B2 del plan B: referencias sin información de clase frente a los doce métodos en el test (familia F-C1).

Gráfico de bosque vertical: diferencia de AUC (referencia menos método, media de cinco semillas) en la parte bimcv del
test de COVIDx CXR-4, con IC95, la línea del cero y la del margen de no inferioridad (-0.02).
(a) D_IN, detector de colección sobre el tronco de ImageNet sin ajuste (referencia principal).
(b) D_ERM, detector sobre los troncos finales de ERM (referencia secundaria).

Datos: results/c1/test/f_c1.csv y f_c1b.csv (python scripts/eval/auditoria_covidx.py --c1 test). No calcula nada.

Uso: python scripts/figs/figB2_bosque_c1.py [--out results/figs/figB2_bosque_c1]
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
          "lisa": "LISA", "dfa_fuente": "DFA con fuente", "dfr": "DFR"}
MARGEN = -0.02


def panel(ax, d, orden, titulo, letra):
    d = d.set_index("metodo").loc[orden]
    for i, (m, r) in enumerate(d.iterrows()):
        y = len(orden) - 1 - i
        c = E.COLOR_METODO.get(m, E.GRIS["texto"])
        ni = bool(r["no_inferior_0.02"])
        ax.plot([r.ic95_inf, r.ic95_sup], [y, y], color=c, lw=1.2, solid_capstyle="butt", zorder=2)
        mk = E.MARCA_METODO.get(m, "o")
        ax.plot([r.delta], [y], marker=mk, ms=5 if ni else C.hueco_ms(mk), color=c, mfc=c if ni else "white", mew=1.0, zorder=3)
        ax.text(1.02, y, "%.3f" % r.auc_metodo, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=6.5,
                color=E.GRIS["texto"])
    ax.axvline(0.0, color=E.GRIS["eje"], lw=0.8, zorder=1)
    ax.axvline(MARGEN, color=E.GRIS["tenue"], lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.set_yticks(range(len(orden)))
    ax.set_yticklabels([NOMBRE[m] for m in orden][::-1])
    ax.set_ylim(-0.7, len(orden) - 0.3)
    ax.set_xlim(-0.06, 0.12)
    ax.set_xticks([-0.04, 0.0, 0.04, 0.08, 0.12])
    ax.grid(axis="x", color=E.GRIS["rejilla"], lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(1.02, len(orden) - 0.25, "AUC", transform=ax.get_yaxis_transform(), va="bottom", ha="left", fontsize=6.5,
            color=E.GRIS["tenue"])
    ax.text(MARGEN - 0.003, 3.0, "margen de no inferioridad (-0.02)", rotation=90, va="center", ha="right", fontsize=6, color=E.GRIS["tenue"])
    t = E.rotulo(ax, letra, dx_mm=-27.0, dy_mm=3.0)
    from matplotlib.transforms import ScaledTranslation
    tr = ax.transAxes + ScaledTranslation(-20.5 * E.MM, 3.3 * E.MM, ax.figure.dpi_scale_trans)
    ax.text(0.0, 1.0, titulo, transform=tr, fontsize=7.5, va="bottom", ha="left", color=E.GRIS["texto"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "figs" / "figB2_bosque_c1"))
    a = ap.parse_args(argv)
    E.aplicar()
    fa = C.leer("c1/test/f_c1.csv")
    fb = C.leer("c1/test/f_c1b.csv")
    fa = fa[fa.referencia == "D_IN"]; fb = fb[fb.referencia == "D_ERM"]
    assert len(fa) == 12 and len(fb) == 12, (len(fa), len(fb))
    orden = list(fa.sort_values("auc_metodo", ascending=False).metodo)        # mismo orden en los dos paneles
    fig = E.figura(90, 170)
    axs = [fig.add_axes([0.31, 0.575, 0.55, 0.37]), fig.add_axes([0.31, 0.125, 0.55, 0.37])]
    panel(axs[0], fa, orden, "Red de ImageNet sin ajuste (AUC %.3f)" % fa.auc_referencia.iloc[0], "a")
    panel(axs[1], fb, orden, "Red entrenada con ERM (AUC %.3f)" % fb.auc_referencia.iloc[0], "b")
    axs[1].set_xlabel("Diferencia de AUC: referencia sin clase menos método")
    from matplotlib.lines import Line2D
    g = E.GRIS["texto"]
    fig.legend([Line2D([], [], ls="none", marker="o", ms=5, color=g, mfc=g, mew=1.0),
                Line2D([], [], ls="none", marker="o", ms=5, color=g, mfc="white", mew=1.0)],
               ["No inferior con Holm (margen 0.02)", "No inferioridad no demostrada"], loc="lower left",
               bbox_to_anchor=(0.04, 0.026), ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.0, borderaxespad=0,
               fontsize=6.5)
    datos = pd.concat([fa.assign(panel="a"), fb.assign(panel="b")])[["panel", "referencia", "metodo", "auc_referencia", "auc_metodo",
                                                                    "delta", "ic95_inf", "ic95_sup", "p_holm_0.02", "no_inferior_0.02"]]
    out = C.guardar(fig, a.out, datos=datos)
    print(out)


if __name__ == "__main__":
    main()
