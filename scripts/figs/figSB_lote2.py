"""Figuras suplementarias SB5 a SB8 del plan B (docs/09_figuras.md, Section 7.2 y 7.4). Borradores en español.

SB5: pérdida por selección de época por método y nivel (results/c2c3/c3_epoca_por_metodo.csv).
SB6: réplica en tuberculosis, diferencia frente a ERM por cohorte externa (results/b2/informe_test/comparacion_erm.csv).
SB7: piloto de B3a, AUC en la validación desconfundida por semilla (tabla del piloto en results/gate6/).
SB8: selección de hiperparámetros, ganancia sobre la media de los ensayos y pérdida frente al mejor (results/c2c3/c3-hiper.csv).

Uso: python scripts/figs/figSB_lote2.py
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


def leyenda_relleno(fig, ax, relleno, hueco):
    """Leyenda del relleno del marcador, justo bajo el rótulo del eje x de `ax`."""
    from matplotlib.lines import Line2D
    from matplotlib.transforms import ScaledTranslation
    g = E.GRIS["texto"]
    tr = ax.transAxes + ScaledTranslation(-24.0 * E.MM, -8.5 * E.MM, fig.dpi_scale_trans)
    ax.legend([Line2D([], [], ls="none", marker="o", ms=5, color=g, mfc=g, mew=1.0),
               Line2D([], [], ls="none", marker="o", ms=5, color=g, mfc="white", mew=1.0)], [relleno, hueco],
              loc="upper left", bbox_to_anchor=(0.0, 0.0), bbox_transform=tr, ncol=2, frameon=False, handletextpad=0.3,
              columnspacing=1.0, borderaxespad=0, fontsize=6.5)


def sb5():
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
            ax.text(j, i, "no\naplica" if np.isnan(v) else C.num(v), ha="center", va="center", fontsize=6 if np.isnan(v) else 6.3,
                    color=E.GRIS["tenue"] if np.isnan(v) else ("white" if v > 0.075 else E.GRIS["texto"]))
    ax.set_xticks(range(len(niveles))); ax.set_xticklabels([RHO_ET[r] for r in niveles]); ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    ax.set_xlabel(C.EJE_RHO_ES)
    ax.set_yticks(range(len(orden))); ax.set_yticklabels([NOMBRE[m] for m in orden]); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = C.ejes_mm(fig, 27.0, 26.0, 60.3, 2.2)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="min", ticks=[0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12])
    cb.solids.set_rasterized(False); cb.outline.set_linewidth(0.5);   # barra vectorial
    cax.tick_params(labelsize=6.5, length=2, width=0.5)
    cb.set_label("Pérdida de AUC (los negativos, con el color del 0)", fontsize=6.5, labelpad=2)
    return C.guardar(fig, FIGS / "figSB5_seleccion_por_metodo", datos=d)


def sb6():
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
        ax.set_yticks(range(len(orden))); ax.set_yticklabels([NOMBRE[m] for m in orden][::-1])
        ax.set_ylim(-0.7, len(orden) - 0.3); ax.set_xlim(-0.10, 0.14); ax.set_xticks([-0.08, -0.04, 0, 0.04, 0.08, 0.12])
        limpia(ax)
        titulo(ax, letra, "%s: AUC de ERM %.3f" % (nom, erm.auc_media_semillas))
    ejes[1].set_xlabel("Diferencia de AUC frente a ERM")
    leyenda_relleno(fig, ejes[1], "Supera a ERM con Holm", "Superioridad no demostrada")
    return C.guardar(fig, FIGS / "figSB6_tuberculosis", datos=d)


def sb7():
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
                    mfc=E.COLOR_METODO[m] if s == 0 else "white", mew=1.0, label="%s, semilla %d" % (NOMBRE[m], s))
    ax.axhline(0.5, color=E.GRIS["tenue"], lw=0.6, ls=(0, (4, 2)))
    ax.set_xticks(range(3)); ax.set_xticklabels(["0.5", "0.9", "1.0"]); ax.set_xlim(-0.3, 2.3); ax.set_ylim(0.45, 1.0)
    limpia(ax, "y"); ax.set_xlabel(C.EJE_RHO_ES); ax.set_ylabel("AUC en la validación desconfundida")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.13), fontsize=6.3, frameon=False, handlelength=2.2, labelspacing=0.32)
    return C.guardar(fig, FIGS / "figSB7_piloto", datos=d)


def sb8():
    d = C.leer("c2c3/c3-hiper.csv")
    orden_m = ["erm", "lisa", "groupdro", "dfr"]
    filas = [(m, r) for m in orden_m for r in (0.9, 0.95, 0.99)]
    fig = E.figura(90, 130)
    a1 = C.ejes_mm(fig, 29.7, 14.0, 23.4, 99.0); a2 = C.ejes_mm(fig, 63.0, 14.0, 23.4, 99.0)   # sin nota interna
    for ax, col, lo, hi, tit in ((a1, "ganancia_sobre_azar", "ic95_inf_G", "ic95_sup_G", "Ganancia sobre la\nmedia de los\nensayos"),
                                 (a2, "perdida_frente_oraculo", "ic95_inf_L", "ic95_sup_L", "Pérdida frente al ensayo\nelegido con la validación\ndesconfundida")):
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
    return C.guardar(fig, FIGS / "figSB8_hiperparametros", datos=d)


if __name__ == "__main__":
    E.aplicar()
    for f in (sb5, sb6, sb7, sb8):
        print(f()["png"])
