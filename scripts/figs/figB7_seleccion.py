"""Figura B7 del plan B: pérdida por elegir la época con la validación de la distribución de entrenamiento (C3, F-C3).

Diferencia de AUC en test_desconf entre el estado elegido con la validación desconfundida y el elegido con la validación
de la misma rho, frente a rho: media sobre los once métodos con IC95 jerárquico (línea y banda) y cada método como un
punto pequeño. Como referencia, en gris, la mayor diferencia de un método frente a ERM en cada nivel (C2).

Cambios del 22-11-2026: leyenda completa bajo el eje (sustituye a la nota), eje de rho rotulado como niveles equiespaciados y
lectura de los intervalos corregidos (results/correccion_ec6/ cuando existe).

Datos: results/c2c3/c3-epoca.csv, c3_epoca_por_metodo.csv y results/b3/test_A_best/comparacion_erm.csv.
Uso: python scripts/figs/figB7_seleccion.py
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
TEXTO_ES = {"nombre": NOMBRE, "xlabel": C.EJE_RHO_ES, "ylabel": "Pérdida de AUC en el test desconfundido",
            "resumen": ["Media de los once métodos (círculo hueco: nivel descriptivo)",
                        "Media en un nivel confirmatorio, significativa con Holm",
                        "IC95 de la media (semillas y pacientes)",
                        "Mayor ventaja de un método sobre ERM en el mismo nivel"],
            "titulo_metodos": "Cada método (media entre semillas)"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "figs" / "figB7_seleccion"))
    a = ap.parse_args(argv)
    E.aplicar()
    dibujar(Path(a.out), TEXTO_ES)


def dibujar(out_base: Path, T):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ep = C.leer("c2c3/c3-epoca.csv").sort_values("rho")
    pm = C.leer("c2c3/c3_epoca_por_metodo.csv")
    c = C.leer("b3/test_A_best/comparacion_erm.csv")
    niveles = list(ep.rho); pos = {r: i for i, r in enumerate(niveles)}
    ref = []
    for r in niveles:
        x = c[c.medida == "delta_auc_rho%03d" % round(r * 100)]
        ref.append(x.delta.max())
    fig = E.figura(90, 150)
    ax = C.ejes_mm(fig, 15.5, 62.0, 72.0, 85.0)
    ax.axhline(0.0, color=E.GRIS["eje"], lw=0.8, zorder=1)
    rng = np.random.default_rng(7)
    for m, d in pm.groupby("metodo"):
        ax.plot([pos[r] + rng.uniform(-0.16, 0.16) for r in d.rho], d.delta_sel, ls="none", marker=E.MARCA_METODO[m], ms=3,
                color=E.COLOR_METODO[m], alpha=0.75, mew=0, zorder=2)
    xs = [pos[r] for r in niveles]
    ax.plot(xs, ref, color=E.GRIS["tenue"], lw=1.0, ls=(0, (4, 2)), marker="s", ms=3.5, mfc="white", mew=0.8, zorder=3)
    ax.fill_between(xs, ep.ic95_inf, ep.ic95_sup, color="#000000", alpha=0.15, lw=0, zorder=4)
    conf = ep.papel.eq("confirmatoria").values
    ax.plot(xs, ep.delta_sel_media_metodos, color="#000000", lw=1.5, zorder=5)
    ax.plot(np.array(xs)[~conf], ep.delta_sel_media_metodos[~conf], ls="none", marker="o", ms=5, color="#000000", mfc="white", mew=1.1, zorder=6)
    ax.plot(np.array(xs)[conf], ep.delta_sel_media_metodos[conf], ls="none", marker="o", ms=5.5, color="#000000", zorder=6)
    ax.set_xticks(xs); ax.set_xticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "0.95", "0.99", "1.0"])
    ax.set_xlim(-0.5, len(niveles) - 0.5)
    ax.set_ylim(-0.012, 0.13)
    ax.set_yticks([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12])
    ax.grid(axis="y", color=E.GRIS["rejilla"], lw=0.5, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(T["xlabel"])
    ax.set_ylabel(T["ylabel"])
    # leyenda de los elementos comunes (dos columnas) y de los métodos (tres columnas), bajo el eje
    h1 = [Line2D([], [], color="#000000", lw=1.5, marker="o", ms=5, mfc="white", mew=1.1),
          Line2D([], [], color="#000000", lw=0, marker="o", ms=5.5),
          Patch(facecolor="#000000", alpha=0.15, lw=0),
          Line2D([], [], color=E.GRIS["tenue"], lw=1.0, ls=(0, (4, 2)), marker="s", ms=3.5, mfc="white", mew=0.8)]
    fig.legend(h1, T["resumen"], loc="upper left", bbox_to_anchor=(0.02, 48.0 / 150), ncol=1, frameon=False, fontsize=6.5,
               handlelength=2.0, handletextpad=0.4, columnspacing=0.8, labelspacing=0.5, borderaxespad=0)
    orden = list(pm.pivot(index="metodo", columns="rho", values="delta_sel")[[0.9, 0.95, 0.99]].mean(axis=1)
                 .sort_values(ascending=False).index)
    h2 = [Line2D([], [], ls="none", marker=E.MARCA_METODO[m], ms=4, color=E.COLOR_METODO[m], mew=0) for m in orden]
    lg = fig.legend(h2, [T["nombre"][m] for m in orden], loc="upper left", bbox_to_anchor=(0.02, 29.5 / 150), ncol=3,
                    frameon=False, fontsize=6.5, handlelength=1.0, handletextpad=0.3, columnspacing=0.8, labelspacing=0.35,
                    borderaxespad=0, title=T["titulo_metodos"], title_fontsize=6.5, alignment="left")
    datos = ep.assign(mayor_ventaja_sobre_erm=ref)[["rho", "delta_sel_media_metodos", "ic95_inf", "ic95_sup", "papel", "p_holm", "rechaza_holm",
                                                    "mayor_ventaja_sobre_erm"]]
    out = C.guardar(fig, out_base, datos=datos)
    pm.to_csv(out_base.with_name(out_base.name + "_por_metodo.csv"), index=False)
    print(out["png"])
    return out


if __name__ == "__main__":
    main()
