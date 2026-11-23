"""Figura B6 del plan B: qué método supera a ERM y con qué coste (familias F-C2-area y F-C2-coste).

(a) Diferencia de área bajo la curva dosis-respuesta frente a ERM (rho de 0.5 a 0.99), con IC95 jerárquico y el valor p
    de Holm. Marcador relleno: supera a ERM con Holm.
(b) Diferencia de AUC frente a ERM sin confusión (rho 0.5), con la línea del margen de no inferioridad (-0.01).
    Marcador relleno: no inferior a ERM con Holm.

Datos: results/b3/test_A_best/comparacion_erm.csv. Uso: python scripts/figs/figB6_area_coste.py
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

NOMBRE = {"erm_cuotas": "ERM con cuotas", "erm_mascara": "ERM con máscara", "groupdro": "GroupDRO", "dann": "DANN", "coral": "CORAL",
          "mixstyle": "MixStyle", "fourier": "Fourier", "lisa": "LISA", "dfa_fuente": "DFA con fuente", "dfr": "DFR"}


def panel(ax, d, orden, xlim, xticks, margen=None, col_p=True):
    for i, m in enumerate(orden):
        r = d.loc[m]; y = len(orden) - 1 - i; c = E.COLOR_METODO[m]; ok = bool(r.rechaza_holm)
        ax.plot([r.ic95_inf, r.ic95_sup], [y, y], color=c, lw=1.2, solid_capstyle="butt", zorder=2)
        mk = E.MARCA_METODO[m]
        ax.plot([r.delta], [y], marker=mk, ms=5 if ok else C.hueco_ms(mk), color=c, mfc=c if ok else "white", mew=1.0, zorder=3)
        if col_p:
            p = r.p_holm
            ax.text(1.03, y, C.p_txt(p), transform=ax.get_yaxis_transform(), va="center", ha="left",
                    fontsize=6.5, color=E.GRIS["texto"], fontweight="bold" if ok else "normal")
    ax.axvline(0.0, color=E.GRIS["eje"], lw=0.8, zorder=1)
    if margen is not None:
        ax.axvline(margen, color=E.GRIS["tenue"], lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.set_yticks(range(len(orden)))
    ax.set_yticklabels([NOMBRE[m] for m in orden][::-1])
    ax.set_ylim(-0.7, len(orden) - 0.3)
    ax.set_xlim(*xlim); ax.set_xticks(xticks)
    ax.grid(axis="x", color=E.GRIS["rejilla"], lw=0.5, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def leyenda(ax, relleno, hueco):
    """Leyenda del relleno del marcador, bajo el rótulo del eje x del panel."""
    from matplotlib.lines import Line2D
    from matplotlib.transforms import ScaledTranslation
    g = E.GRIS["texto"]
    tr = ax.transAxes + ScaledTranslation(-20.5 * E.MM, -8.5 * E.MM, ax.figure.dpi_scale_trans)
    ax.legend([Line2D([], [], ls="none", marker="o", ms=5, color=g, mfc=g, mew=1.0),
               Line2D([], [], ls="none", marker="o", ms=5, color=g, mfc="white", mew=1.0)], [relleno, hueco],
              loc="upper left", bbox_to_anchor=(0.0, 0.0), bbox_transform=tr, ncol=2, frameon=False, handletextpad=0.3,
              columnspacing=1.0, borderaxespad=0, fontsize=6.5)


def titulo(ax, letra, texto):
    from matplotlib.transforms import ScaledTranslation
    E.rotulo(ax, letra, dx_mm=-27.0, dy_mm=3.0)
    tr = ax.transAxes + ScaledTranslation(-20.5 * E.MM, 3.3 * E.MM, ax.figure.dpi_scale_trans)
    ax.text(0.0, 1.0, texto, transform=tr, fontsize=7.5, va="bottom", ha="left", color=E.GRIS["texto"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "figs" / "figB6_area_coste"))
    a = ap.parse_args(argv)
    E.aplicar()
    c = C.leer("b3/test_A_best/comparacion_erm.csv")
    ar = c[c.medida == "delta_area_dosis"].set_index("metodo"); co = c[c.medida == "coste_rho050_no_inferioridad"].set_index("metodo")
    assert len(ar) == 10 and len(co) == 10
    orden = list(ar.sort_values("delta", ascending=False).index)
    fig = E.figura(90, 180)
    a1 = fig.add_axes([0.31, 0.605, 0.53, 0.345]); a2 = fig.add_axes([0.31, 0.125, 0.53, 0.345])
    panel(a1, ar, orden, (-0.032, 0.024), [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02])
    a1.set_xlabel("Diferencia de área frente a ERM")
    leyenda(a1, "Supera a ERM con Holm", "Superioridad no demostrada")
    a1.text(1.03, len(orden) - 0.25, "p (Holm)", transform=a1.get_yaxis_transform(), va="bottom", ha="left", fontsize=6.5, color=E.GRIS["tenue"])
    titulo(a1, "a", "Área bajo la curva dosis-respuesta")
    panel(a2, co, orden, (-0.032, 0.024), [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02], margen=-0.01, col_p=False)
    a2.set_xlabel("Diferencia de AUC frente a ERM en ρ = 0.5")
    leyenda(a2, "No inferior con Holm", "No inferioridad no demostrada")
    a2.text(-0.0115, 2.5, "margen de no inferioridad (-0.01)", rotation=90, va="center", ha="right", fontsize=6, color=E.GRIS["tenue"])
    titulo(a2, "b", "Coste sin confusión (ρ = 0.5)")
    datos = pd.concat([ar.assign(panel="a"), co.assign(panel="b")]).reset_index()[["panel", "metodo", "delta", "ic95_inf", "ic95_sup", "p_holm", "rechaza_holm"]]
    out = C.guardar(fig, a.out, datos=datos)
    print(out["png"])


if __name__ == "__main__":
    main()
