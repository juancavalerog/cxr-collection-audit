"""Figura B5 del plan B: curvas dosis-respuesta de los métodos en B3a (C2).

Rejilla vertical de 4 x 3 paneles. En cada panel, AUC en test_desconf frente a la fuerza de la confusión (rho), media
entre semillas con la banda del IC95 jerárquico (semillas y pacientes); ERM se repite en gris como referencia. El
primer panel es ERM. Los niveles de rho se dibujan equiespaciados porque 0.95, 0.99 y 1.0 no se distinguirían en un
eje lineal. Cada panel rotula la diferencia de área bajo la curva frente a ERM (familia confirmatoria F-C2-area) y si
supera a ERM con Holm.

Datos: results/b3/test_A_best/curva.csv y comparacion_erm.csv
(python scripts/eval/informe_b3.py --conjunto test_desconf --permitir-test --out results/b3/test_A_best).

Uso: python scripts/figs/figB5_dosis_respuesta.py [--out results/figs/figB5_dosis_respuesta]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.viz import estilo as E  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import comun as C  # noqa: E402

NOMBRE = {"erm": "ERM", "erm_cuotas": "ERM con cuotas", "erm_mascara": "ERM con máscara", "groupdro": "GroupDRO", "dann": "DANN",
          "coral": "CORAL", "mixstyle": "MixStyle", "fourier": "Fourier", "lisa": "LISA", "dfa_fuente": "DFA con fuente", "dfr": "DFR"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "figs" / "figB5_dosis_respuesta"))
    a = ap.parse_args(argv)
    E.aplicar()
    cur = C.leer("b3/test_A_best/curva.csv")
    cmp_ = C.leer("b3/test_A_best/comparacion_erm.csv")
    area = cmp_[cmp_.medida == "delta_area_dosis"].set_index("metodo")
    niveles = sorted(cur.rho.unique())
    assert len(niveles) == 8 and set(cur.metodo) == set(NOMBRE)
    pos = {r: i for i, r in enumerate(niveles)}
    orden = ["erm"] + list(area.sort_values("delta", ascending=False).index)
    erm = cur[cur.metodo == "erm"].sort_values("rho")
    fig = E.figura(190, 230)
    nf, nc = 4, 3
    x0, x1, y0, y1, hg, vg = 0.075, 0.985, 0.075, 0.955, 0.035, 0.05
    w = (x1 - x0 - (nc - 1) * hg) / nc
    h = (y1 - y0 - (nf - 1) * vg) / nf
    for k, m in enumerate(orden):
        f, c = divmod(k, nc)
        ax = fig.add_axes([x0 + c * (w + hg), y1 - (f + 1) * h - f * vg, w, h])
        d = cur[cur.metodo == m].sort_values("rho").dropna(subset=["auc_media_semillas"])
        ax.axhline(0.5, color=E.GRIS["tenue"], lw=0.6, ls=(0, (4, 2)), zorder=1)
        if m != "erm":
            ax.plot([pos[r] for r in erm.rho], erm.auc_media_semillas, color=E.GRIS["tenue"], lw=1.0, marker="^", ms=3, mfc="white",
                    mew=0.7, zorder=2)
        col = E.COLOR_METODO[m]
        xs = [pos[r] for r in d.rho]
        ax.fill_between(xs, d.ic95_inf, d.ic95_sup, color=col, alpha=0.22, lw=0, zorder=3)
        ax.plot(xs, d.auc_media_semillas, color=col, lw=1.4, marker=E.MARCA_METODO[m], ms=4.5, zorder=4)
        if m == "dfr":
            ax.text(pos[1.0], d.auc_media_semillas.iloc[-1], "no\naplica", fontsize=6, color=E.GRIS["tenue"], ha="center", va="center")
        ax.set_xlim(-0.4, len(niveles) - 0.6)
        ax.set_ylim(0.33, 1.0)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_xticks(range(len(niveles)))
        ax.set_xticklabels([("%.2f" % r).rstrip("0").rstrip(".") if r not in (0.5, 1.0) else ("0.5" if r == 0.5 else "1.0") for r in niveles])
        ax.grid(axis="y", color=E.GRIS["rejilla"], lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if c != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("AUC en el test desconfundido")
        if f == nf - 1 or k + nc >= len(orden):
            ax.set_xlabel(C.EJE_RHO_ES)
        ax.set_title(NOMBRE[m], loc="left", fontsize=7.5, fontweight="bold", color=E.GRIS["texto"], pad=3)
        if m == "erm":
            ax.text(0.03, 0.03, "Referencia\nCaída entre ρ = 0.5 y 1.0: %s" % C.num(d.auc_media_semillas.iloc[0] - d.auc_media_semillas.iloc[-1]),
                    transform=ax.transAxes, fontsize=6.5, color=E.GRIS["texto"], va="bottom", ha="left", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.9))
        else:
            r = area.loc[m]
            txt = "Diferencia de área frente a ERM:\n%s (IC95 %s a %s)\n%s" % (
                C.num(r.delta, 4, True), C.num(r.ic95_inf, 4, True), C.num(r.ic95_sup, 4, True),
                "Supera a ERM (Holm)" if bool(r.rechaza_holm) else "Superioridad no demostrada (Holm)")
            ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=6.5, color=E.GRIS["texto"], va="bottom", ha="left", linespacing=1.3,
                    fontweight="bold" if bool(r.rechaza_holm) else "normal", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.9))
    # duodécima celda: leyenda
    f, c = divmod(len(orden), nc)
    lax = fig.add_axes([x0 + c * (w + hg), y1 - (f + 1) * h - f * vg, w, h]); lax.axis("off")
    lax.plot([0.05, 0.2], [0.86, 0.86], color=E.GRIS["texto"], lw=1.4, marker="o", ms=4.5)
    lax.text(0.25, 0.86, "Método: media entre semillas", fontsize=7, va="center")
    lax.fill_between([0.05, 0.2], [0.70, 0.70], [0.76, 0.76], color=E.GRIS["texto"], alpha=0.22, lw=0)
    lax.text(0.25, 0.73, "IC95 (semillas y pacientes)", fontsize=7, va="center")
    lax.plot([0.05, 0.2], [0.60, 0.60], color=E.GRIS["tenue"], lw=1.0, marker="^", ms=3, mfc="white", mew=0.7)
    lax.text(0.25, 0.60, "ERM, como referencia", fontsize=7, va="center")
    lax.plot([0.05, 0.2], [0.47, 0.47], color=E.GRIS["tenue"], lw=0.6, ls=(0, (4, 2)))
    lax.text(0.25, 0.47, "Azar (0.5)", fontsize=7, va="center")
    lax.set_xlim(0, 1); lax.set_ylim(0, 1)
    datos = cur[["metodo", "rho", "auc_media_semillas", "ic95_inf", "ic95_sup", "n_semillas"]]
    out = C.guardar(fig, a.out, datos=datos)
    print(out["png"])


if __name__ == "__main__":
    main()
