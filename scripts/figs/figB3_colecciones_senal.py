"""Figura B3 del plan B: la señal interna depende de la colección de los positivos (C1-ii y C1-iii).

(a) Por método (media de cinco semillas) y para las dos referencias sin clase: AUC en el test con los negativos de bimcv
    frente a los positivos de bimcv (marcador relleno) y frente a los positivos de las otras fuentes (marcador hueco),
    con IC95.
(b) Lectura lineal con reserva de negativos (cuatro troncos): AUC al ajustarla con positivos de una colección y evaluarla
    en la misma colección (validación cruzada), en la otra colección y en la parte bimcv del test. Media de los cuatro
    troncos y, entre paréntesis, mínimo y máximo.

Cambios del 22-11-2026: leyenda del panel a junto al panel a, barra de color rotulada en el panel b, colecciones nombradas por el
formato del identificador y figura de 205 mm de alto.

Datos: results/c1/test/c1_iii_resumen.csv y c1_ii_reservado.csv. Uso: python scripts/figs/figB3_colecciones_senal.py
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

NOMBRE = {"erm": "ERM", "erm_cuotas": "ERM con cuotas", "erm_mascara": "ERM con máscara", "control_fuente": "Control de fuente",
          "groupdro": "GroupDRO", "dann": "DANN", "coral": "CORAL", "mixstyle": "MixStyle", "fourier": "Fourier",
          "lisa": "LISA", "dfa_fuente": "DFA con fuente", "dfr": "DFR", "D_IN": "Sin clase (ImageNet)", "D_ERM": "Sin clase (ERM)"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "figs" / "figB3_colecciones_senal"))
    a = ap.parse_args(argv)
    E.aplicar()
    dibujar(a.out, TEXTO_ES)


TEXTO_ES = {"nombre": NOMBRE, "xlabel_a": "AUC en el test (negativos de BIMCV)", "titulo_a": "Según el origen de los positivos",
            "leyenda_a": ["Positivos de BIMCV (4 040)", "Positivos de RICORD (200)"], "decimales_b": 2,
            "celda_b": "%s\n(%s a %s)", "cols_b": ["La misma\ncolección", "La otra\ncolección", "Parte BIMCV\ndel test"],
            "filas_b": ["Colección de cinco cifras", "Colección de seis cifras"],
            "titulo_b": "Lectura ajustada con positivos de una colección y evaluada en",
            "barra_b": "AUC media de la lectura (cuatro redes)"}


def dibujar(out_base, T):
    """Dibuja la figura con los textos de T (español o inglés) y la guarda en out_base."""
    from matplotlib.transforms import ScaledTranslation
    from matplotlib.lines import Line2D
    r = C.leer("c1/test/c1_iii_resumen.csv")
    r = r[r.estrato_bimcv == "todos"]
    A = r[r.contraste == "a_bimcv_neg_vs_bimcv_pos"].set_index("metodo"); B = r[r.contraste == "b_bimcv_neg_vs_otras_fuentes_pos"].set_index("metodo")
    assert len(A) == 14 and len(B) == 14
    refs = ["D_ERM", "D_IN"]
    metodos = list(B.drop(index=refs).sort_values("auc", ascending=False).index)
    filas = metodos + refs
    y = {m: len(filas) - 1 - i - (0.0 if m not in refs else 0.6) for i, m in enumerate(filas)}
    fig = E.figura(90, 205)
    ax = C.ejes_mm(fig, 32.4, 99.0, 54.0, 96.5)
    datos = []
    for m in filas:
        col = E.GRIS["tenue"] if m in refs else E.COLOR_METODO[m]; mk = "o" if m in refs else E.MARCA_METODO[m]
        ax.plot([B.loc[m, "auc"], A.loc[m, "auc"]], [y[m], y[m]], color=col, lw=0.8, alpha=0.55, zorder=2)
        for X in (A, B):
            ax.plot([X.loc[m, "ic95_inf"], X.loc[m, "ic95_sup"]], [y[m], y[m]], color=col, lw=1.6, solid_capstyle="butt", zorder=3)
        ax.plot([A.loc[m, "auc"]], [y[m]], marker=mk, ms=5, color=col, mfc=col, mew=1.0, zorder=4)
        ax.plot([B.loc[m, "auc"]], [y[m]], marker=mk, ms=C.hueco_ms(mk), color=col, mfc="white", mew=1.0, zorder=4)
        datos.append({"panel": "a", "fila": m, "auc_positivos_bimcv": A.loc[m, "auc"], "ic95_inf_bimcv": A.loc[m, "ic95_inf"], "ic95_sup_bimcv": A.loc[m, "ic95_sup"],
                      "auc_positivos_otras_fuentes": B.loc[m, "auc"], "ic95_inf_otras": B.loc[m, "ic95_inf"], "ic95_sup_otras": B.loc[m, "ic95_sup"]})
    ax.axhline((y[metodos[-1]] + y[refs[0]]) / 2.0, color=E.GRIS["rejilla"], lw=0.6)
    ax.set_yticks([y[m] for m in filas]); ax.set_yticklabels([T["nombre"][m] for m in filas])
    for t, m in zip(ax.get_yticklabels(), filas):
        if m in refs:
            t.set_color(E.GRIS["tenue"])
    ax.set_ylim(min(y.values()) - 0.7, max(y.values()) + 0.7)
    ax.set_xlim(0.55, 0.90); ax.set_xticks([0.6, 0.7, 0.8, 0.9])
    ax.grid(axis="x", color=E.GRIS["rejilla"], lw=0.5, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(T["xlabel_a"])
    E.rotulo(ax, "a", dx_mm=-31.5, dy_mm=2.0)
    tr = ax.transAxes + ScaledTranslation(-25.0 * E.MM, 2.3 * E.MM, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, T["titulo_a"], transform=tr, fontsize=7.5, va="bottom", ha="left", color=E.GRIS["texto"])
    # leyenda del relleno del marcador, inmediatamente debajo del rótulo del eje del panel a
    g = E.GRIS["tenue"]
    tr = ax.transAxes + ScaledTranslation(-25.0 * E.MM, -9.0 * E.MM, fig.dpi_scale_trans)
    ax.legend([Line2D([], [], ls="none", marker="o", ms=5, color=g, mfc=g, mew=1.0),
               Line2D([], [], ls="none", marker="o", ms=5, color=g, mfc="white", mew=1.0)],
              T["leyenda_a"], loc="upper left", bbox_to_anchor=(0.0, 0.0), bbox_transform=tr,
              ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.2, borderaxespad=0, fontsize=7)
    # ---------------------------------------------------------------- panel b
    c = C.leer("c1/test/c1_ii_reservado.csv")
    cols = [("cv_neg_vs_antiguos", "transfer_antiguos_a_nuevos", "lectura_antiguos_en_test_bimcv"),
            ("cv_neg_vs_nuevos", "transfer_nuevos_a_antiguos", "lectura_nuevos_en_test_bimcv")]
    M = np.array([[c[k].mean() for k in fila] for fila in cols]); Mn = np.array([[c[k].min() for k in fila] for fila in cols])
    Mx = np.array([[c[k].max() for k in fila] for fila in cols])
    bx = C.ejes_mm(fig, 32.4, 36.0, 54.0, 32.0)
    im = C.mapa(bx, M, cmap="Blues", vmin=0.45, vmax=1.0)
    nd = T["decimales_b"]
    for i in range(2):
        for j in range(3):
            bx.text(j, i, T["celda_b"] % (C.num(M[i, j], nd), C.num(Mn[i, j], nd), C.num(Mx[i, j], nd)), ha="center", va="center",
                    fontsize=6.5, color="white" if M[i, j] > 0.78 else E.GRIS["texto"], linespacing=1.3)
            datos.append({"panel": "b", "fila": ("cinco cifras", "seis cifras")[i] + " / " + ("misma coleccion", "otra coleccion", "test bimcv")[j],
                          "auc_media": M[i, j], "auc_min": Mn[i, j], "auc_max": Mx[i, j]})
    bx.set_xticks(range(3)); bx.set_xticklabels(T["cols_b"])
    bx.xaxis.tick_top(); bx.tick_params(length=0)
    bx.set_yticks(range(2)); bx.set_yticklabels(T["filas_b"])
    for s in bx.spines.values():
        s.set_visible(False)
    E.rotulo(bx, "b", dx_mm=-31.5, dy_mm=11.5)
    tr = bx.transAxes + ScaledTranslation(-25.0 * E.MM, 11.8 * E.MM, fig.dpi_scale_trans)
    bx.text(0.0, 1.0, T["titulo_b"], transform=tr, fontsize=7.5, va="bottom", ha="left", color=E.GRIS["texto"])
    cax = C.ejes_mm(fig, 32.4, 27.5, 54.0, 2.2)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    cb.solids.set_rasterized(False); cb.outline.set_linewidth(0.5);   # barra vectorial
    cax.tick_params(labelsize=6.5, length=2, width=0.5)
    cb.set_label(T["barra_b"], fontsize=7, labelpad=2)
    out = C.guardar(fig, out_base, datos=pd.DataFrame(datos))
    print(out["png"])
    return out


if __name__ == "__main__":
    main()
