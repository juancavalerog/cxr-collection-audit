"""Figura 3: plan de encaminamiento (docs/09_figuras.md). Borrador en español.

(a) Plan R_y(s -> s') por clase: de qué fuente toma la firma la vista re-firmada de una imagen de cada fuente.
(b) Distribución de la fuente efectiva por clase antes (lotes reales) y después de completar la tabla.

Uso: python scripts/figs/fig03_plan.py [--out results/figs/fig03_plan]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datos_tabla import FUENTES, NOMBRE, plan_vigente, tabla_train  # noqa: E402
from src.viz import estilo as E  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "figs" / "fig03_plan"))
    ap.add_argument("--lam", type=float, default=1.0)
    a = ap.parse_args(argv)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    E.aplicar()

    n, _, y, s = tabla_train()
    tab, plan = plan_vigente(y, s, lam=a.lam)
    P = tab.conditional()                                   # P(s | y) en los lotes
    R = plan.routing                                        # R[y, origen, destino]
    lam = float(a.lam)
    despues = (P + lam * np.einsum("ys,yst->yt", P, R)) / (1.0 + lam)   # fuente efectiva tras completar
    tv_antes = 0.5 * np.abs(P[0] - P[1]).sum()
    tv_despues = 0.5 * np.abs(despues[0] - despues[1]).sum()
    K = len(FUENTES)
    nombres = [NOMBRE[f] for f in FUENTES]
    clases = ["Negativa", "Positiva"]

    W, H = 90.0, 140.0
    fig = E.figura(W, H)

    def caja(x_mm, y_sup_mm, w_mm, h_mm):
        """Eje a partir de milímetros medidos desde la esquina superior izquierda."""
        return fig.add_axes([x_mm / W, 1.0 - (y_sup_mm + h_mm) / H, w_mm / W, h_mm / H])

    cmap = LinearSegmentedColormap.from_list("plan", ["#FFFFFF", "#C6DBEF", "#4292C6", "#08306B"])
    x0, w, gap = 19.0, 27.0, 3.0
    # ---------------------------------------------------------------- (a) plan por clase
    ya, ha = 9.0, 27.0
    ejes_a = []
    for c in (0, 1):
        ax = caja(x0 + c * (w + gap), ya, w, ha)
        M = np.ma.masked_invalid(np.where(P[c][:, None] > 0, R[c], np.nan))
        im = ax.imshow(M, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
        for i in range(K):
            if P[c][i] <= 0:                                  # origen sin imágenes de esta clase
                ax.add_patch(plt.Rectangle((-0.5, i - 0.5), K, 1, facecolor=E.GRIS["vacio"],
                                           edgecolor=E.GRIS["tenue"], hatch="/////", linewidth=0.0, zorder=2))
                continue
            for j in range(K):
                v = R[c, i, j]
                if i == j:
                    ax.text(j, i, "\u00b7", ha="center", va="center", fontsize=6.0, color=E.GRIS["tenue"], zorder=3)
                elif v >= 0.045:
                    ax.text(j, i, "%d" % round(100 * v), ha="center", va="center", fontsize=6.0,
                            color="white" if v > 0.55 else E.GRIS["texto"], zorder=3)
        ax.set_xticks(range(K))
        ax.set_xticklabels(nombres, rotation=90)
        ax.set_yticks(range(K))
        ax.set_yticklabels(nombres if c == 0 else [])
        ax.tick_params(length=0, pad=2)
        ax.set_xticks(np.arange(-0.5, K, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, K, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6, zorder=4)
        ax.tick_params(which="minor", length=0)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.6)
        ax.set_title("Clase %s" % clases[c].lower(), fontsize=E.PT_BASE, pad=3)
        if c == 0:
            ax.set_ylabel("Fuente de la imagen")
        ejes_a.append(ax)
    fig.text((x0 + w + gap / 2) / W, 1.0 - (ya + ha + 16.5) / H, "Fuente que aporta la firma (% de las vistas re-firmadas)",
             ha="center", va="top", fontsize=E.PT_BASE)
    E.rotulo(ejes_a[0], "a", dx_mm=-18.0, dy_mm=3.5)
    cax = caja(x0 + 2 * w + gap + 2.0, ya, 1.8, ha)
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.set_yticklabels(["0", "25", "50", "75", "100"])
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(length=2, width=0.6, pad=1.5)

    # ---------------------------------------------------------------- (b) fuente efectiva antes y después
    yb, hb = 73.0, 46.0
    alto = 0.38
    pos = np.arange(K)
    ejes_b = []
    for k, (D, tit, tv) in enumerate(((P, "Antes", tv_antes), (despues, "Después", tv_despues))):
        ax = caja(x0 + k * (w + gap), yb, w, hb)
        for c in (0, 1):
            ax.barh(pos + (c - 0.5) * alto, D[c], height=alto, color=E.COLOR_CLASE[c], edgecolor="white",
                    linewidth=0.3, label=clases[c], hatch=None if c == 0 else ".....")
        ax.set_yticks(pos)
        ax.set_yticklabels(nombres if k == 0 else [])
        ax.set_ylim(K - 0.5, -0.5)
        ax.set_xlim(0, 0.5)
        ax.set_xticks([0, 0.2, 0.4])
        ax.xaxis.grid(True)
        ax.tick_params(axis="y", length=0)
        ax.set_title(tit, fontsize=E.PT_BASE, pad=3)
        ax.text(0.96, 0.035, "DVT = %.2f" % tv, transform=ax.transAxes, ha="right", va="bottom", fontsize=E.PT_BASE,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2})
        ejes_b.append(ax)
    ejes_b[0].set_ylabel("Fuente efectiva")
    fig.text((x0 + w + gap / 2) / W, 1.0 - (yb + hb + 6.0) / H, "Proporción de las vistas de cada clase",
             ha="center", va="top", fontsize=E.PT_BASE)
    E.rotulo(ejes_b[0], "b", dx_mm=-18.0, dy_mm=3.5)
    h, l = ejes_b[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=((x0 + w + gap / 2) / W, 1.0 - (yb + hb + 10.5) / H), ncol=2,
               handlelength=1.4, columnspacing=1.8, borderaxespad=0.0)

    filas = []
    for c in (0, 1):
        for i in range(K):
            filas.append({"panel": "b", "clase": clases[c], "fuente": FUENTES[i], "destino": "",
                          "P_lotes": P[c, i], "P_efectiva_despues": despues[c, i], "plan": np.nan})
            for j in range(K):
                filas.append({"panel": "a", "clase": clases[c], "fuente": FUENTES[i], "destino": FUENTES[j],
                              "P_lotes": P[c, i], "P_efectiva_despues": np.nan,
                              "plan": R[c, i, j] if P[c, i] > 0 else np.nan})
    out = E.guardar(fig, a.out, datos=pd.DataFrame(filas))
    print("Z = %.4f; lambda = %.2f; factible = %s; DVT antes = %.4f; DVT después = %.2e" %
          (plan.z, lam, plan.feasible, tv_antes, tv_despues))
    for k, v in out.items():
        print(k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
