"""Versión inglesa de las Figuras 1 a 7 (rótulos en inglés; datos, tamaños, colores, marcas y posiciones idénticos).

Copia de los guiones españoles de scripts/figs/ (figB1_colecciones.py a figB7_seleccion.py) en la que solo cambia el
texto. Los guiones españoles no se modifican y siguen produciendo las figuras aprobadas de results/figs/.
Cambios de texto respecto al español (logs/figs_en/informe.md): rótulos y nombres de método según el glosario
(manuscript/en/GLOSARIO.md), "Primary reference" y "Secondary reference" para las referencias sin clase, BIMCV en
mayúsculas, la letra griega rho, signo menos tipográfico, miles según el glosario, y sin las notas al pie internas
que repiten el pie de figura aprobado (manuscript/es/06_pies_de_figura.md).

Lectura común con scripts/figs/comun.py (intervalos corregidos de results/correccion_ec6/ cuando
existen), guardado con bbox ajustado, TIFF en RGB y metadatos neutros; las Figuras 3 y 7 usan el dibujo de los
guiones españoles con textos en inglés. Lee solo CSV de resultados ya escritos. La Figura 1 toma los recuentos del CSV de datos de la figura española
(results/figs/figB1_colecciones.csv), que comprueba contra la Tabla E1, en lugar de releer el manifiesto.

Uso: python3 scripts/figs/en/figs_principales_en.py [1 2 ... 7]   (sin argumentos, las siete)
Salidas: results/figs/en/Fig1.{pdf,png,tiff,csv} ... Fig7.*
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts" / "figs"))

from src.viz import estilo as E  # noqa: E402
from nombres_en import NOMBRE, NOMBRE2, EJE_RHO, menos, miles  # noqa: E402
import comun as C  # noqa: E402

FIGS = ROOT / "results" / "figs"
OUT = FIGS / "en"
RES = ROOT / "results"


# ---------------------------------------------------------------------------------------------------------- Fig. 1
def fig1():
    sys.path.insert(0, str(ROOT / "scripts" / "figs"))
    import figB1_colecciones as B1  # constantes de la figura española (no dibuja nada al importarse)
    d = pd.read_csv(FIGS / "figB1_colecciones.csv", float_precision="round_trip")
    for (c, p), v in B1.E1.items():
        got = int(d[(d.coleccion == c) & (d.particion == p)]["imagenes"].iloc[0])
        if got != v:
            raise SystemExit("figB1_colecciones.csv da %d en %s / %s y la Tabla E1 %d" % (got, c, p, v))
    COLECCIONES, PARTICIONES, COLOR, TRAMA, MARCA = B1.COLECCIONES, B1.PARTICIONES, B1.COLOR, B1.TRAMA, B1.MARCA
    COL_EN = dict(zip(COLECCIONES, ("Negatives", "Positives, five-digit identifiers", "Positives, six-digit identifiers")))
    EJE_EN = dict(zip(COLECCIONES, ("Negatives", "Positives,\nfive digits", "Positives,\nsix digits")))
    PART_EN = {"Entrenamiento": "Training", "Validación": "Validation", "Test": "Test"}
    E.aplicar()
    fig = E.figura(90, 118)
    ax1 = fig.add_axes([0.24, 0.57, 0.72, 0.29])
    ax2 = fig.add_axes([0.24, 0.10, 0.72, 0.33])
    for i, p in enumerate(PARTICIONES):
        izq = 0.0
        for c in COLECCIONES:
            r = d[(d.coleccion == c) & (d.particion == p)].iloc[0]
            w = r["fraccion_de_la_particion"]
            if w <= 0:
                continue
            ax1.barh(i, w, left=izq, height=0.62, color=COLOR[c], hatch=TRAMA[c], edgecolor="white", linewidth=0.8,
                     clip_on=False)
            if w > 0.09:
                ax1.text(izq + w / 2, i, miles(r["imagenes"]), ha="center", va="center",
                         fontsize=6.5, color="white" if c != COLECCIONES[2] else E.GRIS["texto"],
                         fontweight="bold",
                         bbox=(dict(facecolor="white", edgecolor="none", pad=1.2) if TRAMA[c] else None))
            elif w > 0:
                ax1.text(izq + w + 0.012, i - 0.33, miles(r["imagenes"]), ha="left",
                         va="bottom", fontsize=6.5, color=E.GRIS["texto"])
            izq += w
    ax1.set_yticks(range(len(PARTICIONES)), [PART_EN[p] for p in PARTICIONES])
    ax1.set_ylim(len(PARTICIONES) - 0.5, -0.75)
    ax1.set_xlim(0, 1)
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    ax1.set_xlabel("BIMCV images in the split (%)")
    from matplotlib.patches import Patch
    ax1.legend([Patch(facecolor=COLOR[c], hatch=TRAMA[c], edgecolor="white") for c in COLECCIONES],
               [COL_EN[c] for c in COLECCIONES],
               loc="lower left", bbox_to_anchor=(0.0, 1.03), ncol=1, handlelength=1.6, borderaxespad=0)
    E.rotulo(ax1, "a", dx_mm=-19)
    for j, c in enumerate(COLECCIONES):
        for k, p in enumerate(PARTICIONES):
            v = int(d[(d.coleccion == c) & (d.particion == p)]["imagenes"].iloc[0])
            yy = j + (k - 1) * 0.22
            if v > 0:
                ax2.plot([v], [yy], marker=MARCA[p], color=COLOR[c], markersize=4.5, linestyle="none",
                         markeredgecolor=E.GRIS["texto"], markeredgewidth=0.5)
                ax2.text(v * 1.25, yy, miles(v), va="center", fontsize=6.5)
            else:
                ax2.text(12, yy, "0 in %s" % PART_EN[p].lower(), va="center", fontsize=6.5, color=E.GRIS["tenue"])
    ax2.set_xscale("log")
    ax2.set_xlim(10, 3e5)
    ax2.set_yticks(range(len(COLECCIONES)), [EJE_EN[c] for c in COLECCIONES])
    ax2.set_ylim(len(COLECCIONES) - 0.5, -0.5)
    ax2.set_xlabel("Images (log scale)")
    ax2.grid(axis="x", which="major")
    ax2.legend([ax2.plot([], [], marker=MARCA[p], color=E.GRIS["tenue"], linestyle="none", markersize=4.5)[0]
                for p in PARTICIONES], [PART_EN[p] for p in PARTICIONES], loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=3,
               handletextpad=0.3, columnspacing=0.9, borderaxespad=0)
    E.rotulo(ax2, "b", dx_mm=-19)
    return C.guardar(fig, OUT / "Fig1", datos=d)


# ---------------------------------------------------------------------------------------------------------- Fig. 2
def fig2():
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
        ax.set_yticklabels([NOMBRE2[m] for m in orden][::-1])
        ax.set_ylim(-0.7, len(orden) - 0.3)
        ax.set_xlim(-0.06, 0.12)
        ax.set_xticks([-0.04, 0.0, 0.04, 0.08, 0.12])
        ax.grid(axis="x", color=E.GRIS["rejilla"], lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.text(1.02, len(orden) - 0.25, "AUC", transform=ax.get_yaxis_transform(), va="bottom", ha="left", fontsize=6.5,
                color=E.GRIS["tenue"])
        ax.text(MARGEN - 0.003, 3.0, "non-inferiority margin (-0.02)", rotation=90, va="center", ha="right", fontsize=6, color=E.GRIS["tenue"])
        E.rotulo(ax, letra, dx_mm=-27.0, dy_mm=3.0)
        from matplotlib.transforms import ScaledTranslation
        tr = ax.transAxes + ScaledTranslation(-20.5 * E.MM, 3.3 * E.MM, ax.figure.dpi_scale_trans)
        ax.text(0.0, 1.0, titulo, transform=tr, fontsize=7.5, va="bottom", ha="left", color=E.GRIS["texto"])

    E.aplicar()
    fa = C.leer("c1/test/f_c1.csv")
    fb = C.leer("c1/test/f_c1b.csv")
    fa = fa[fa.referencia == "D_IN"]; fb = fb[fb.referencia == "D_ERM"]
    assert len(fa) == 12 and len(fb) == 12, (len(fa), len(fb))
    orden = list(fa.sort_values("auc_metodo", ascending=False).metodo)
    fig = E.figura(90, 170)
    axs = [fig.add_axes([0.31, 0.575, 0.55, 0.37]), fig.add_axes([0.31, 0.125, 0.55, 0.37])]
    panel(axs[0], fa, orden, "Primary reference, ImageNet network (AUC %.3f)" % fa.auc_referencia.iloc[0], "a")
    panel(axs[1], fb, orden, "Secondary reference, ERM backbones (AUC %.3f)" % fb.auc_referencia.iloc[0], "b")
    axs[1].set_xlabel("AUC difference: class-blind reference minus method")
    from matplotlib.lines import Line2D
    g = E.GRIS["texto"]
    fig.legend([Line2D([], [], ls="none", marker="o", ms=5, color=g, mfc=g, mew=1.0),
                Line2D([], [], ls="none", marker="o", ms=5, color=g, mfc="white", mew=1.0)],
               ["Non-inferior after Holm (margin 0.02)", "Non-inferiority not established"], loc="lower left",
               bbox_to_anchor=(0.04, 0.026), ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.0, borderaxespad=0,
               fontsize=6.5)
    datos = pd.concat([fa.assign(panel="a"), fb.assign(panel="b")])[["panel", "referencia", "metodo", "auc_referencia", "auc_metodo",
                                                                    "delta", "ic95_inf", "ic95_sup", "p_holm_0.02", "no_inferior_0.02"]]
    return C.guardar(fig, OUT / "Fig2", datos=datos)


# ---------------------------------------------------------------------------------------------------------- Fig. 3
def fig3():
    """Mismo dibujo que scripts/figs/figB3_colecciones_senal.py, con los textos en inglés."""
    import figB3_colecciones_senal as B3
    E.aplicar()
    T = {"nombre": NOMBRE, "xlabel_a": "AUC on the test set (BIMCV negatives)", "titulo_a": "By origin of the positives",
         "leyenda_a": ["BIMCV positives (4040)", "RICORD positives (200)"], "decimales_b": 3,
         "celda_b": "%s\n(%s to %s)", "cols_b": ["Same\ncollection", "Other\ncollection", "BIMCV part\nof the test set"],
         "filas_b": ["Five-digit collection", "Six-digit collection"],
         "titulo_b": "Probe fitted with positives from one collection, evaluated on",
         "barra_b": "Mean AUC of the probe (four backbones)"}
    return B3.dibujar(OUT / "Fig3", T)


# ---------------------------------------------------------------------------------------------------------- Fig. 4
def fig4():
    from matplotlib.lines import Line2D
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
    ax.text(0.508, max(y.values()) + 0.62, "chance", fontsize=6, color=E.GRIS["tenue"], va="top", ha="left")
    # leyenda del relleno del marcador, en el hueco que deja la nota interna retirada
    g = E.GRIS["tenue"]
    fig.legend([Line2D([], [], ls="none", marker="o", ms=5, color=g, mfc=g, mew=1.0),
                Line2D([], [], ls="none", marker="o", ms=5, color=g, mfc="white", mew=1.0)],
               ["BIMCV part of the COVIDx CXR-4 test set", "COVIDGR (external cohort)"], loc="upper left",
               bbox_to_anchor=(0.04, 0.075), ncol=1, frameon=False, handletextpad=0.3, labelspacing=0.3, borderaxespad=0)
    return C.guardar(fig, OUT / "Fig4", datos=pd.DataFrame(datos))


# ---------------------------------------------------------------------------------------------------------- Fig. 5
def fig5():
    NOM = {k: v for k, v in NOMBRE.items() if k not in ("control_fuente", "D_IN", "D_ERM")}
    E.aplicar()
    cur = C.leer("b3/test_A_best/curva.csv")
    cmp_ = C.leer("b3/test_A_best/comparacion_erm.csv")
    area = cmp_[cmp_.medida == "delta_area_dosis"].set_index("metodo")
    niveles = sorted(cur.rho.unique())
    assert len(niveles) == 8 and set(cur.metodo) == set(NOM)
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
            ax.text(pos[1.0], d.auc_media_semillas.iloc[-1], "not\napplicable", fontsize=6, color=E.GRIS["tenue"], ha="center", va="center")
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
            ax.set_ylabel("AUC on the deconfounded test set")
        if f == nf - 1 or k + nc >= len(orden):
            ax.set_xlabel(EJE_RHO)
        ax.set_title(NOM[m], loc="left", fontsize=7.5, fontweight="bold", color=E.GRIS["texto"], pad=3)
        if m == "erm":
            ax.text(0.03, 0.03, "Reference\nDrop from ρ = 0.5 to 1.0: %s" % C.num(d.auc_media_semillas.iloc[0] - d.auc_media_semillas.iloc[-1]),
                    transform=ax.transAxes, fontsize=6.5, color=E.GRIS["texto"], va="bottom", ha="left", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.9))
        else:
            r = area.loc[m]
            txt = "Area difference vs. ERM:\n%s (95%% CI %s to %s)\n" % (C.num(r.delta, 4, True), C.num(r.ic95_inf, 4, True),
                                                                           C.num(r.ic95_sup, 4, True)) + \
                ("Exceeds ERM (Holm)" if bool(r.rechaza_holm) else "Superiority not established (Holm)")
            ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=6.5, color=E.GRIS["texto"], va="bottom", ha="left", linespacing=1.3,
                    fontweight="bold" if bool(r.rechaza_holm) else "normal", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.9))
    f, c = divmod(len(orden), nc)
    lax = fig.add_axes([x0 + c * (w + hg), y1 - (f + 1) * h - f * vg, w, h]); lax.axis("off")
    lax.plot([0.05, 0.2], [0.86, 0.86], color=E.GRIS["texto"], lw=1.4, marker="o", ms=4.5)
    lax.text(0.25, 0.86, "Method: mean across seeds", fontsize=7, va="center")
    lax.fill_between([0.05, 0.2], [0.70, 0.70], [0.76, 0.76], color=E.GRIS["texto"], alpha=0.22, lw=0)
    lax.text(0.25, 0.73, "95% CI (seeds and patients)", fontsize=7, va="center")
    lax.plot([0.05, 0.2], [0.60, 0.60], color=E.GRIS["tenue"], lw=1.0, marker="^", ms=3, mfc="white", mew=0.7)
    lax.text(0.25, 0.60, "ERM, as reference", fontsize=7, va="center")
    lax.plot([0.05, 0.2], [0.47, 0.47], color=E.GRIS["tenue"], lw=0.6, ls=(0, (4, 2)))
    lax.text(0.25, 0.47, "Chance (0.5)", fontsize=7, va="center")
    lax.set_xlim(0, 1); lax.set_ylim(0, 1)
    datos = cur[["metodo", "rho", "auc_media_semillas", "ic95_inf", "ic95_sup", "n_semillas"]]
    return C.guardar(fig, OUT / "Fig5", datos=datos)


# ---------------------------------------------------------------------------------------------------------- Fig. 6
def fig6():
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
        ax.set_yticklabels([NOMBRE2[m] for m in orden][::-1])
        ax.set_ylim(-0.7, len(orden) - 0.3)
        ax.set_xlim(*xlim); ax.set_xticks(xticks)
        ax.grid(axis="x", color=E.GRIS["rejilla"], lw=0.5, zorder=0); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def leyenda(ax, relleno, hueco):
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

    E.aplicar()
    c = C.leer("b3/test_A_best/comparacion_erm.csv")
    ar = c[c.medida == "delta_area_dosis"].set_index("metodo"); co = c[c.medida == "coste_rho050_no_inferioridad"].set_index("metodo")
    assert len(ar) == 10 and len(co) == 10
    orden = list(ar.sort_values("delta", ascending=False).index)
    fig = E.figura(90, 180)
    a1 = fig.add_axes([0.31, 0.605, 0.53, 0.345]); a2 = fig.add_axes([0.31, 0.125, 0.53, 0.345])
    panel(a1, ar, orden, (-0.032, 0.024), [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02])
    a1.set_xlabel("Area difference vs. ERM")
    leyenda(a1, "Exceeds ERM after Holm", "Superiority not established")
    a1.text(1.03, len(orden) - 0.25, "p (Holm)", transform=a1.get_yaxis_transform(), va="bottom", ha="left", fontsize=6.5, color=E.GRIS["tenue"])
    titulo(a1, "a", "Area under the dose-response curve")
    panel(a2, co, orden, (-0.032, 0.024), [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02], margen=-0.01, col_p=False)
    a2.set_xlabel("AUC difference vs. ERM at ρ = 0.5")
    leyenda(a2, "Non-inferior after Holm", "Non-inferiority not established")
    a2.text(-0.0115, 2.5, "non-inferiority margin (-0.01)", rotation=90, va="center", ha="right", fontsize=6, color=E.GRIS["tenue"])
    titulo(a2, "b", "No confounding (ρ = 0.5)")
    datos = pd.concat([ar.assign(panel="a"), co.assign(panel="b")]).reset_index()[["panel", "metodo", "delta", "ic95_inf", "ic95_sup", "p_holm", "rechaza_holm"]]
    return C.guardar(fig, OUT / "Fig6", datos=datos)


# ---------------------------------------------------------------------------------------------------------- Fig. 7
def fig7():
    """Mismo dibujo que scripts/figs/figB7_seleccion.py, con los textos en inglés."""
    import figB7_seleccion as B7
    E.aplicar()
    T = {"nombre": NOMBRE2, "xlabel": EJE_RHO, "ylabel": "Epoch-selection loss (AUC, deconfounded test set)",
         "resumen": ["Mean of the eleven methods (open circle: descriptive level)",
                     "Mean at a confirmatory level, significant after Holm",
                     "95% CI of the mean (seeds and patients)",
                     "Largest advantage of a method over ERM at the same level"],
         "titulo_metodos": "Each method (mean across seeds)"}
    return B7.dibujar(OUT / "Fig7", T)


FIGURAS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6, 7: fig7}

if __name__ == "__main__":
    pedidas = [int(a) for a in sys.argv[1:]] or sorted(FIGURAS)
    for n in pedidas:
        print("Fig%d:" % n, FIGURAS[n]()["png"])
