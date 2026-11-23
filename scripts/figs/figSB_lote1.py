"""Figuras suplementarias SB1 a SB4 del plan B (docs/09_figuras.md, Section 7.2). Borradores en español.

SB1: referencia sin clase (red de ImageNet) frente a cada método por proyección, AP y PA (results/c1/test/c1_vii_estratos.csv).
SB2: clasificador de solo adquisición y ERM con pulmones enmascarados (c1_iv_adquisicion.csv, c1_v_mascara.csv).
SB3: test invertido frente a test desconfundido en B3a (results/c2c3/invertido.csv).
SB4: orientación B positiva frente a A (results/c2c3/orientacion_b_curva.csv y results/b3/test_A_best/curva.csv).

Uso: python scripts/figs/figSB_lote1.py
"""
from __future__ import annotations

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
          "lisa": "LISA", "dfa_fuente": "DFA con fuente", "dfr": "DFR"}
FIGS = ROOT / "results" / "figs"
RHO_ET = {0.5: "0.5", 0.6: "0.6", 0.7: "0.7", 0.8: "0.8", 0.9: "0.9", 0.95: "0.95", 0.99: "0.99", 1.0: "1.0"}


def limpia(ax, eje="x"):
    ax.grid(axis=eje, color=E.GRIS["rejilla"], lw=0.5, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def titulo(ax, letra, texto, dx=-27.0):
    from matplotlib.transforms import ScaledTranslation
    E.rotulo(ax, letra, dx_mm=dx, dy_mm=3.0)
    tr = ax.transAxes + ScaledTranslation((dx + 6.5) * E.MM, 3.3 * E.MM, ax.figure.dpi_scale_trans)
    ax.text(0.0, 1.0, texto, transform=tr, fontsize=7.5, va="bottom", ha="left", color=E.GRIS["texto"])


def sb1():
    d = C.leer("c1/test/c1_vii_estratos.csv")
    d = d[d.referencia == "D_IN"]
    fig = E.figura(90, 170)
    ejes = [fig.add_axes([0.31, 0.575, 0.60, 0.37]), fig.add_axes([0.31, 0.125, 0.60, 0.37])]
    orden = None
    for ax, est, letra in zip(ejes, ("AP", "PA"), "ab"):
        x = d[d.estrato == est]
        det = x[x.comparacion == "detector"].iloc[0]
        x = x[x.comparacion.str.startswith("detector menos")].copy()
        x["metodo"] = x.comparacion.str.replace("detector menos ", "", regex=False)
        x = x.set_index("metodo")
        if orden is None:
            orden = list(x.sort_values("delta").index)
        for i, m in enumerate(orden):
            r = x.loc[m]; y = len(orden) - 1 - i; c = E.COLOR_METODO[m]
            ax.plot([r.ic95_inf, r.ic95_sup], [y, y], color=c, lw=1.2, solid_capstyle="butt", zorder=2)
            ax.plot([r.delta], [y], marker=E.MARCA_METODO[m], ms=5, color=c, zorder=3)
        ax.axvline(0, color=E.GRIS["eje"], lw=0.8); ax.axvline(-0.02, color=E.GRIS["tenue"], lw=0.8, ls=(0, (4, 2)))
        ax.set_yticks(range(len(orden))); ax.set_yticklabels([NOMBRE[m] for m in orden][::-1])
        ax.set_ylim(-0.7, len(orden) - 0.3); ax.set_xlim(-0.09, 0.13); ax.set_xticks([-0.08, -0.04, 0, 0.04, 0.08, 0.12])
        limpia(ax)
        titulo(ax, letra, "Proyección %s: %s imágenes, AUC de la referencia %.3f" % (est, format(int(det.n), ",").replace(",", " "), det.auc))
    ejes[1].set_xlabel("Diferencia de AUC: referencia sin clase menos método")
    return C.guardar(fig, FIGS / "figSB1_estratos", datos=d)


def sb2():
    r = C.leer("c1/test/c1_iii_resumen.csv")
    r = r[(r.estrato_bimcv == "todos") & (r.contraste == "a_bimcv_neg_vs_bimcv_pos")].set_index("metodo")
    aq = C.leer("c1/test/c1_iv_adquisicion.csv"); mk = C.leer("c1/test/c1_v_mascara.csv")
    det = C.leer("c1/test/detectores.csv")
    dm = det[det.detector == "D_MASK"]
    filas = [("ERM", r.loc["erm", "auc"], r.loc["erm", "ic95_inf"], r.loc["erm", "ic95_sup"], E.COLOR_METODO["erm"], "^"),
             ("ERM con pulmones\nenmascarados", r.loc["erm_mascara", "auc"], r.loc["erm_mascara", "ic95_inf"], r.loc["erm_mascara", "ic95_sup"],
              E.COLOR_METODO["erm_mascara"], "*"),
             ("Sin clase (ImageNet)", r.loc["D_IN", "auc"], r.loc["D_IN", "ic95_inf"], r.loc["D_IN", "ic95_sup"], E.GRIS["tenue"], "o"),
             ("Sin clase (red de ERM\ncon máscara)", mk.iloc[1].auc_1, mk.iloc[1].ic95_inf, mk.iloc[1].ic95_sup, E.GRIS["tenue"], "o"),
             ("Solo adquisición,\najustada en el conjunto", aq.iloc[1].auc, aq.iloc[1].ic95_inf, aq.iloc[1].ic95_sup, E.OKABE_ITO["negro"], "s"),
             ("Solo adquisición, ajustada\nen entrenamiento", aq.iloc[0].auc, aq.iloc[0].ic95_inf, aq.iloc[0].ic95_sup, E.OKABE_ITO["negro"], "s")]
    fig = E.figura(90, 100)
    ax = fig.add_axes([0.40, 0.12, 0.55, 0.83])          # sin nota interna: el eje ocupa el alto libre
    for i, (n, v, lo, hi, c, mk_) in enumerate(filas):
        y = len(filas) - 1 - i
        ax.plot([lo, hi], [y, y], color=c, lw=1.4, solid_capstyle="butt", zorder=2)
        hueco = n.startswith("Solo adquisición,\najustada en el conjunto")   # usa la clase del test
        ax.plot([v], [y], marker=mk_, ms=5.5, color=c, mfc="white" if hueco else c, mew=1.0, zorder=3)
        ax.text(v, y + 0.22, C.num(v), fontsize=6.5, ha="center", va="bottom", color=E.GRIS["texto"])
    ax.axvline(0.5, color=E.GRIS["tenue"], lw=0.8, ls=(0, (4, 2)))
    ax.set_yticks(range(len(filas))); ax.set_yticklabels([f[0] for f in filas][::-1])
    ax.set_ylim(-0.6, len(filas) - 0.3); ax.set_xlim(0.45, 0.90); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    limpia(ax); ax.set_xlabel("AUC en la parte BIMCV del test")
    return C.guardar(fig, FIGS / "figSB2_adquisicion_mascara", datos=pd.DataFrame([{"fila": f[0].replace("\n", " "), "auc": f[1], "ic95_inf": f[2], "ic95_sup": f[3]} for f in filas]))


def sb3():
    d = C.leer("c2c3/invertido.csv")
    area = C.leer("b3/test_A_best/comparacion_erm.csv")
    area = area[area.medida == "delta_area_dosis"].sort_values("delta", ascending=False)
    orden_fig5 = ["erm"] + list(area.metodo)
    niveles = sorted(d.rho.unique()); pos = {r: i for i, r in enumerate(niveles)}
    fig = E.figura(90, 120)
    ax = fig.add_axes([0.16, 0.24, 0.80, 0.72])
    for m, x in d.groupby("metodo"):
        x = x.sort_values("rho").dropna(subset=["auc_invertido"])
        if m == "erm":
            continue
        ax.plot([pos[r] for r in x.rho], x.auc_invertido, color=E.COLOR_METODO[m], lw=0.7, marker=E.MARCA_METODO[m], ms=2.5, alpha=0.5,
                zorder=3, label=NOMBRE[m])
    e = d[d.metodo == "erm"].sort_values("rho")
    xs = [pos[r] for r in e.rho]
    ax.plot(xs, e.auc_desconf, color=E.GRIS["tenue"], lw=1.2, ls=(0, (4, 2)), marker="^", ms=3.5, mfc="white", mew=0.8, zorder=2,
            label="ERM, test desconfundido")
    ax.fill_between(xs, e.ic95_inf_invertido, e.ic95_sup_invertido, color=E.COLOR_METODO["erm"], alpha=0.3, lw=0, zorder=4)
    ax.plot(xs, e.auc_invertido, color=E.COLOR_METODO["erm"], lw=1.8, marker="^", ms=5, zorder=5, label="ERM")
    h, l = ax.get_legend_handles_labels()
    orden = [l.index("ERM"), l.index("ERM, test desconfundido")] + [l.index(NOMBRE[m]) for m in orden_fig5 if m != "erm" and NOMBRE[m] in l]
    ax.legend([h[i] for i in orden], [l[i] for i in orden], loc="lower left", bbox_to_anchor=(0.0, 0.03), ncol=1, fontsize=6.3, frameon=False,
              handlelength=2.2, labelspacing=0.32, borderpad=0.2)
    ax.axhline(0.5, color=E.GRIS["eje"], lw=0.6)
    ax.set_xticks(range(len(niveles))); ax.set_xticklabels([RHO_ET[r] for r in niveles]); ax.set_ylim(-0.03, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    limpia(ax, "y"); ax.set_xlabel(C.EJE_RHO_ES); ax.set_ylabel("AUC")
    return C.guardar(fig, FIGS / "figSB3_test_invertido", datos=d)


def sb4():
    b = C.leer("c2c3/orientacion_b_curva.csv"); a = C.leer("b3/test_A_best/curva.csv")
    o = C.leer("c2c3/orientacion.csv").iloc[0]
    niveles = [0.5, 0.8, 1.0]; pos = {r: i for i, r in enumerate(niveles)}
    fig = E.figura(90, 110)
    ax = fig.add_axes([0.16, 0.24, 0.80, 0.72])
    for T, est, lab, col in ((a, "-", "A", E.COLOR_METODO["erm"]), (b, (0, (4, 2)), "B", E.OKABE_ITO["azul"])):
        x = T[(T.metodo == "erm") & (T.rho.isin(niveles))].sort_values("rho")
        xs = [pos[r] for r in x.rho]
        ax.fill_between(xs, x.ic95_inf, x.ic95_sup, color=col, alpha=0.18, lw=0)
        ax.plot(xs, x.auc_media_semillas, color=col, lw=1.6, ls=est, marker="^" if lab == "A" else "v", ms=5)
        ax.text(xs[-1] + 0.06, x.auc_media_semillas.iloc[-1], "Orientación %s" % lab, color=col, fontsize=6.5, va="center", ha="left")
    ax.axhline(0.5, color=E.GRIS["tenue"], lw=0.6, ls=(0, (4, 2)))
    ax.set_xticks(range(3)); ax.set_xticklabels(["0.5", "0.8", "1.0"]); ax.set_xlim(-0.25, 2.75); ax.set_ylim(0.45, 1.0)
    limpia(ax, "y"); ax.set_xlabel(C.EJE_RHO_ES); ax.set_ylabel("AUC de ERM en el test desconfundido")
    lo, hi = [float(v) for v in str(o.ic95).split()]
    # recuadro numérico dentro del eje (solo cifras; la explicación está en el pie)
    ax.text(0.03, 0.03, "Caída entre ρ = 0.5 y 1.0:\n%s (A), %s (B)\nDiferencia: %s (IC95 %s a %s)"
            % (C.num(o.caida_A), C.num(o.caida_B), C.num(o.diferencia_A_menos_B), C.num(lo), C.num(hi)),
            transform=ax.transAxes, fontsize=6.5, color=E.GRIS["texto"], va="bottom", ha="left", linespacing=1.3, zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.9))
    datos = pd.concat([a[(a.metodo == "erm") & (a.rho.isin(niveles))].assign(orientacion="A (principal)"),
                       b.assign(orientacion="B (inversa)")], ignore_index=True)
    datos = datos[["orientacion"] + [k for k in b.columns]]
    return C.guardar(fig, FIGS / "figSB4_orientacion", datos=datos)


if __name__ == "__main__":
    E.aplicar()
    for f in (sb1, sb2, sb3, sb4):
        print(f()["png"])
