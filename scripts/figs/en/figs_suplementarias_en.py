"""Versión inglesa de las Figuras S1 a S8 (rótulos en inglés; datos, tamaños, colores, marcas y posiciones idénticos).

Copia de scripts/figs/figSB_lote1.py y figSB_lote2.py en la que solo cambia el texto, con la numeración definitiva:
S1 = SB7 (piloto), S2 = SB1 (estratos), S3 = SB2 (adquisición y máscara), S4 = SB3 (test invertido),
S5 = SB4 (orientación), S6 = SB6 (tuberculosis), S7 = SB5 (selección por método), S8 = SB8 (hiperparámetros).
Los guiones españoles no se modifican. Los CSV de datos se escriben con el mismo contenido que los españoles (en la S3
la columna "fila" conserva los rótulos españoles para que el CSV sea idéntico). Lee solo CSV de resultados, con scripts/figs/comun.py (intervalos corregidos de results/correccion_ec6/ cuando existen).
Cambios de texto en logs/figs_en/informe.md.

Uso: python3 scripts/figs/en/figs_suplementarias_en.py [1 ... 8]   (sin argumentos, las ocho)
Salidas: results/figs/en/FigS1.{pdf,png,tiff,csv} ... FigS8.*
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
from nombres_en import NOMBRE, NOMBRE2, RHO_ET, EJE_RHO, menos  # noqa: E402
import comun as C  # noqa: E402

RES = ROOT / "results"
OUT = RES / "figs" / "en"


def limpia(ax, eje="x"):
    ax.grid(axis=eje, color=E.GRIS["rejilla"], lw=0.5, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def titulo(ax, letra, texto, dx=-27.0):
    from matplotlib.transforms import ScaledTranslation
    E.rotulo(ax, letra, dx_mm=dx, dy_mm=3.0)
    tr = ax.transAxes + ScaledTranslation((dx + 6.5) * E.MM, 3.3 * E.MM, ax.figure.dpi_scale_trans)
    ax.text(0.0, 1.0, texto, transform=tr, fontsize=7.5, va="bottom", ha="left", color=E.GRIS["texto"])


# --------------------------------------------------------------------------------------- Fig. S1 (antes SB7, piloto)
def s1():
    d = C.leer("gate6/tabla_" + "super" + "visor.csv")
    niveles = [0.5, 0.9, 1.0]; pos = {r: i for i, r in enumerate(niveles)}
    fig = E.figura(90, 105)
    ax = fig.add_axes([0.16, 0.25, 0.80, 0.70])
    for m in ("erm", "groupdro", "dfr"):
        x = d[d.metodo == m]
        col = "dfr_best_val_desconf" if m == "dfr" else "best_val_desconf"
        dm = {"erm": -0.14, "groupdro": 0.0, "dfr": 0.14}[m]          # separa los métodos que coinciden
        for s, off in ((0, dm - 0.045), (1, dm + 0.045)):
            y = x[x.semilla == s].sort_values("rho")
            v = y[col].where(y[col].notna(), y["best_val_desconf"]) if m != "dfr" else y[col]
            ok = v.notna()
            ax.plot([pos[r] + off for r in y.rho[ok]], v[ok], color=E.COLOR_METODO[m], lw=1.0, marker=E.MARCA_METODO[m], ms=4.5,
                    mfc=E.COLOR_METODO[m] if s == 0 else "white", mew=1.0, label="%s, seed %d" % (NOMBRE[m], s))
    ax.axhline(0.5, color=E.GRIS["tenue"], lw=0.6, ls=(0, (4, 2)))
    ax.set_xticks(range(3)); ax.set_xticklabels(["0.5", "0.9", "1.0"]); ax.set_xlim(-0.3, 2.3); ax.set_ylim(0.45, 1.0)
    limpia(ax, "y"); ax.set_xlabel(EJE_RHO); ax.set_ylabel("AUC on the deconfounded validation set")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.13), fontsize=6.3, frameon=False, handlelength=2.2, labelspacing=0.32)
    # nota interna retirada: su IC95 (0.356 a 0.440) no sale de ningún CSV (solo de results/gate6/puerta6.md)
    return C.guardar(fig, OUT / "FigS1", datos=d)


# --------------------------------------------------------------------------------------- Fig. S2 (antes SB1, estratos)
def s2():
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
        ax.set_yticks(range(len(orden))); ax.set_yticklabels([NOMBRE2[m] for m in orden][::-1])
        ax.set_ylim(-0.7, len(orden) - 0.3); ax.set_xlim(-0.09, 0.13); ax.set_xticks([-0.08, -0.04, 0, 0.04, 0.08, 0.12])
        limpia(ax)
        titulo(ax, letra, "%s projection, %d images (primary reference AUC %.3f)" % (est, int(det.n), det.auc))
    ejes[1].set_xlabel("AUC difference: primary reference minus method")
    return C.guardar(fig, OUT / "FigS2", datos=d)


# --------------------------------------------------------------------------------------- Fig. S3 (antes SB2)
def s3():
    r = C.leer("c1/test/c1_iii_resumen.csv")
    r = r[(r.estrato_bimcv == "todos") & (r.contraste == "a_bimcv_neg_vs_bimcv_pos")].set_index("metodo")
    aq = C.leer("c1/test/c1_iv_adquisicion.csv"); mk = C.leer("c1/test/c1_v_mascara.csv")
    # (rótulo inglés, rótulo español del CSV de datos, valores, color, marca)
    filas = [("ERM", "ERM", r.loc["erm", "auc"], r.loc["erm", "ic95_inf"], r.loc["erm", "ic95_sup"], E.COLOR_METODO["erm"], "^"),
             ("Masked ERM", "ERM con pulmones\nenmascarados", r.loc["erm_mascara", "auc"], r.loc["erm_mascara", "ic95_inf"],
              r.loc["erm_mascara", "ic95_sup"], E.COLOR_METODO["erm_mascara"], "*"),
             ("Primary reference", "Sin clase (ImageNet)", r.loc["D_IN", "auc"], r.loc["D_IN", "ic95_inf"], r.loc["D_IN", "ic95_sup"],
              E.GRIS["tenue"], "o"),
             ("Collection detector,\nMasked ERM backbones", "Sin clase (red de ERM\ncon máscara)", mk.iloc[1].auc_1, mk.iloc[1].ic95_inf,
              mk.iloc[1].ic95_sup, E.GRIS["tenue"], "o"),
             ("Acquisition-only,\ncross-validated on test", "Solo adquisición,\najustada en el conjunto", aq.iloc[1].auc,
              aq.iloc[1].ic95_inf, aq.iloc[1].ic95_sup, E.OKABE_ITO["negro"], "s"),
             ("Acquisition-only,\nfitted on training", "Solo adquisición, ajustada\nen entrenamiento", aq.iloc[0].auc,
              aq.iloc[0].ic95_inf, aq.iloc[0].ic95_sup, E.OKABE_ITO["negro"], "s")]
    fig = E.figura(90, 100)
    ax = fig.add_axes([0.40, 0.12, 0.55, 0.83])          # sin nota interna: el eje ocupa el alto libre
    for i, (n, _, v, lo, hi, c, mk_) in enumerate(filas):
        y = len(filas) - 1 - i
        ax.plot([lo, hi], [y, y], color=c, lw=1.4, solid_capstyle="butt", zorder=2)
        hueco = n.startswith("Acquisition-only,\ncross-validated")   # usa la clase del test
        ax.plot([v], [y], marker=mk_, ms=5.5, color=c, mfc="white" if hueco else c, mew=1.0, zorder=3)
        ax.text(v, y + 0.22, C.num(v), fontsize=6.5, ha="center", va="bottom", color=E.GRIS["texto"])
    ax.axvline(0.5, color=E.GRIS["tenue"], lw=0.8, ls=(0, (4, 2)))
    ax.set_yticks(range(len(filas))); ax.set_yticklabels([f[0] for f in filas][::-1])
    ax.set_ylim(-0.6, len(filas) - 0.3); ax.set_xlim(0.45, 0.90); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    limpia(ax); ax.set_xlabel("AUC on the BIMCV part of the test set")
    datos = pd.DataFrame([{"fila": f[1].replace("\n", " "), "auc": f[2], "ic95_inf": f[3], "ic95_sup": f[4]} for f in filas])
    return C.guardar(fig, OUT / "FigS3", datos=datos)


# --------------------------------------------------------------------------------------- Fig. S4 (antes SB3)
def s4():
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
    LD = "ERM, deconfounded test set"
    ax.plot(xs, e.auc_desconf, color=E.GRIS["tenue"], lw=1.2, ls=(0, (4, 2)), marker="^", ms=3.5, mfc="white", mew=0.8, zorder=2,
            label=LD)
    ax.fill_between(xs, e.ic95_inf_invertido, e.ic95_sup_invertido, color=E.COLOR_METODO["erm"], alpha=0.3, lw=0, zorder=4)
    ax.plot(xs, e.auc_invertido, color=E.COLOR_METODO["erm"], lw=1.8, marker="^", ms=5, zorder=5, label="ERM")
    h, l = ax.get_legend_handles_labels()
    orden = [l.index("ERM"), l.index(LD)] + [l.index(NOMBRE[m]) for m in orden_fig5 if m != "erm" and NOMBRE[m] in l]
    ax.legend([h[i] for i in orden], [l[i] for i in orden], loc="lower left", bbox_to_anchor=(0.0, 0.03), ncol=1, fontsize=6.3, frameon=False,
              handlelength=2.2, labelspacing=0.32, borderpad=0.2)
    ax.axhline(0.5, color=E.GRIS["eje"], lw=0.6)
    ax.set_xticks(range(len(niveles))); ax.set_xticklabels([RHO_ET[r] for r in niveles]); ax.set_ylim(-0.03, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    limpia(ax, "y"); ax.set_xlabel(EJE_RHO); ax.set_ylabel("AUC")
    return C.guardar(fig, OUT / "FigS4", datos=d)


# --------------------------------------------------------------------------------------- Fig. S5 (antes SB4)
def s5():
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
        ax.text(xs[-1] + 0.06, x.auc_media_semillas.iloc[-1], "Primary\norientation" if lab == "A" else "Reverse\norientation",
                color=col, fontsize=6.5, va="center", ha="left")
    ax.axhline(0.5, color=E.GRIS["tenue"], lw=0.6, ls=(0, (4, 2)))
    ax.set_xticks(range(3)); ax.set_xticklabels(["0.5", "0.8", "1.0"]); ax.set_xlim(-0.25, 2.75); ax.set_ylim(0.45, 1.0)
    limpia(ax, "y"); ax.set_xlabel(EJE_RHO); ax.set_ylabel("ERM AUC on the deconfounded test set")
    lo, hi = [float(v) for v in str(o.ic95).split()]
    # recuadro numérico dentro del eje (solo cifras; la explicación está en el pie)
    ax.text(0.03, 0.03, "Drop from ρ = 0.5 to 1.0:\n%s (primary), %s (reverse)\nDifference: %s (95%% CI %s to %s)"
            % (C.num(o.caida_A), C.num(o.caida_B), C.num(o.diferencia_A_menos_B), C.num(lo), C.num(hi)),
            transform=ax.transAxes, fontsize=6.5, color=E.GRIS["texto"], va="bottom", ha="left", linespacing=1.3, zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.9))
    datos = pd.concat([a[(a.metodo == "erm") & (a.rho.isin(niveles))].assign(orientacion="A (principal)"),
                       b.assign(orientacion="B (inversa)")], ignore_index=True)
    datos = datos[["orientacion"] + [k for k in b.columns]]
    return C.guardar(fig, OUT / "FigS5", datos=datos)


# --------------------------------------------------------------------------------------- Fig. S6 (antes SB6)
def s6():
    d = C.leer("b2/informe_test/comparacion_erm.csv")
    cur = C.leer("b2/informe_test/curva.csv")
    fig = E.figura(90, 170)
    ejes = [fig.add_axes([0.31, 0.575, 0.60, 0.37]), fig.add_axes([0.31, 0.125, 0.60, 0.37])]
    orden = list(d[d.conjunto == "test_shenzhen"].sort_values("delta", ascending=False).metodo)
    for ax, cj, nom, letra in zip(ejes, ("test_shenzhen", "test_montgomery"), ("Shenzhen", "Montgomery"), "ab"):
        x = d[d.conjunto == cj].set_index("metodo")
        erm = cur[(cur.conjunto == cj) & (cur.metodo == "erm")].iloc[0]
        for i, m in enumerate(orden):
            r = x.loc[m]; y = len(orden) - 1 - i; c = E.COLOR_METODO[m]; ok = bool(r.rechaza_holm)
            ax.plot([r.ic95_inf, r.ic95_sup], [y, y], color=c, lw=1.2, solid_capstyle="butt", zorder=2)
            mk = E.MARCA_METODO[m]
            ax.plot([r.delta], [y], marker=mk, ms=5 if ok else C.hueco_ms(mk), color=c, mfc=c if ok else "white", mew=1.0, zorder=3)
        ax.axvline(0, color=E.GRIS["eje"], lw=0.8)
        ax.set_yticks(range(len(orden))); ax.set_yticklabels([NOMBRE2[m] for m in orden][::-1])
        ax.set_ylim(-0.7, len(orden) - 0.3); ax.set_xlim(-0.10, 0.14); ax.set_xticks([-0.08, -0.04, 0, 0.04, 0.08, 0.12])
        limpia(ax)
        titulo(ax, letra, "%s: ERM AUC %.3f" % (nom, erm.auc_media_semillas))
    ejes[1].set_xlabel("AUC difference vs. ERM")
    import figSB_lote2 as L2
    L2.leyenda_relleno(fig, ejes[1], "Exceeds ERM after Holm", "Superiority not established")
    return C.guardar(fig, OUT / "FigS6", datos=d)


# --------------------------------------------------------------------------------------- Fig. S7 (antes SB5)
def s7():
    d = C.leer("c2c3/c3_epoca_por_metodo.csv")
    niveles = sorted(d.rho.unique())
    M = d.pivot(index="metodo", columns="rho", values="delta_sel")
    orden = list(M[[0.9, 0.95, 0.99]].mean(axis=1).sort_values(ascending=False).index)
    M = M.loc[orden, niveles]
    fig = E.figura(90, 135)
    ax = C.ejes_mm(fig, 27.0, 34.0, 60.3, 88.8)
    im = C.mapa(ax, M.values, cmap="Oranges", vmin=0.0, vmax=0.12)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            ax.text(j, i, "n/a" if np.isnan(v) else C.num(v), ha="center", va="center", fontsize=6 if np.isnan(v) else 6.3,
                    color=E.GRIS["tenue"] if np.isnan(v) else ("white" if v > 0.075 else E.GRIS["texto"]))
    ax.set_xticks(range(len(niveles))); ax.set_xticklabels([RHO_ET[r] for r in niveles]); ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    ax.set_xlabel(EJE_RHO)
    ax.set_yticks(range(len(orden))); ax.set_yticklabels([NOMBRE2[m] for m in orden]); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = C.ejes_mm(fig, 27.0, 26.0, 60.3, 2.2)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="min", ticks=[0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12])
    cb.solids.set_rasterized(False); cb.outline.set_linewidth(0.5);   # barra vectorial
    cax.tick_params(labelsize=6.5, length=2, width=0.5)
    cb.set_label("AUC loss (negative values take the colour of 0)", fontsize=6.5, labelpad=2)
    return C.guardar(fig, OUT / "FigS7", datos=d)


# --------------------------------------------------------------------------------------- Fig. S8 (antes SB8)
def s8():
    d = C.leer("c2c3/c3-hiper.csv")
    orden_m = ["erm", "lisa", "groupdro", "dfr"]
    filas = [(m, r) for m in orden_m for r in (0.9, 0.95, 0.99)]
    fig = E.figura(90, 130)
    a1 = C.ejes_mm(fig, 29.7, 14.0, 23.4, 99.0); a2 = C.ejes_mm(fig, 63.0, 14.0, 23.4, 99.0)
    for ax, col, lo, hi, tit in ((a1, "ganancia_sobre_azar", "ic95_inf_G", "ic95_sup_G", "Gain G over the\nmean of the\ntrials"),
                                 (a2, "perdida_frente_oraculo", "ic95_inf_L", "ic95_sup_L", "Loss L vs. the\ntrial selected by\nthe deconfounded\nvalidation set")):
        for i, (m, r) in enumerate(filas):
            x = d[(d.metodo == m) & (np.isclose(d.rho, r))].iloc[0]; y = len(filas) - 1 - i - 0.45 * orden_m.index(m); c = E.COLOR_METODO[m]
            ax.plot([x[lo], x[hi]], [y, y], color=c, lw=1.3, solid_capstyle="butt", zorder=2)
            ax.plot([x[col]], [y], marker=E.MARCA_METODO[m], ms=3.5, color=c, zorder=3)
        ax.axvline(0, color=E.GRIS["eje"], lw=0.8)
        ys = [len(filas) - 1 - i - 0.45 * orden_m.index(m) for i, (m, r) in enumerate(filas)]
        ax.set_yticks(ys); ax.set_ylim(min(ys) - 0.7, max(ys) + 0.7)
        ax.set_yticklabels(["%s, ρ = %s" % (NOMBRE[m], RHO_ET[r]) for m, r in filas] if ax is a1 else [])
        ax.set_xlim(-0.03, 0.09); ax.set_xticks([0, 0.04, 0.08])
        limpia(ax); ax.set_title(tit, fontsize=7, loc="center", pad=4)
        ax.set_xlabel("AUC")
    # letras de panel nuevas (el pie dice "izquierda" y "derecha"): por encima del título de dos líneas
    E.rotulo(a1, "a", dx_mm=-27.0, dy_mm=10.5)
    E.rotulo(a2, "b", dx_mm=-8.5, dy_mm=10.5)
    return C.guardar(fig, OUT / "FigS8", datos=d)


FIGURAS = {1: s1, 2: s2, 3: s3, 4: s4, 5: s5, 6: s6, 7: s7, 8: s8}

if __name__ == "__main__":
    E.aplicar()
    pedidas = [int(a) for a in sys.argv[1:]] or sorted(FIGURAS)
    for n in pedidas:
        print("FigS%d:" % n, FIGURAS[n]()["png"])
