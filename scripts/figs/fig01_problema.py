"""Figura 1: el problema (docs/09_figuras.md). Borrador en español.

(a) Tabla clase por fuente del train limpio de COVIDx CXR-4, con las celdas vacías señaladas.
(b) Aciertos por fuente de un modelo entrenado con todas las fuentes. BORRADOR: usa las corridas de verificación
    del canal de entrenamiento (results/train/replica_cnx_nih_ingenuo, partición oficial); en la versión final se
    sustituye por la ERM del protocolo definitivo sobre el test limpio, con sus cinco semillas e IC95 %.

Uso: python scripts/figs/fig01_problema.py [--out results/figs/fig01_problema]
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

from datos_tabla import FUENTES, NOMBRE, tabla_train  # noqa: E402
from src.viz import estilo as E  # noqa: E402

CORRIDAS = [ROOT / "results" / "train" / "replica_cnx_nih_ingenuo" / ("seed%d" % s) for s in (42, 123)]


def aciertos(corrida: Path, estado: str = "best"):
    """Especificidad y sensibilidad por fuente en test, con el umbral de Youden de la validación."""
    from sklearn.metrics import roc_curve
    v = pd.read_csv(corrida / "predictions" / ("%s_val.csv" % estado))
    t = pd.read_csv(corrida / "predictions" / ("%s_test.csv" % estado))
    fpr, tpr, th = roc_curve(v["label"], v["prob"])
    thr = float(th[int(np.argmax(tpr - fpr))])
    out = {}
    for nombre, fuente, clase in (("Especificidad en RSNA", "rsna", 0), ("Especificidad en BIMCV", "bimcv", 0),
                                  ("Sensibilidad en BIMCV", "bimcv", 1)):
        g = t[(t["source"] == FUENTES.index(fuente)) & (t["label"] == clase)]
        ok = (g["prob"] < thr) if clase == 0 else (g["prob"] >= thr)
        out[nombre] = (float(ok.mean()), int(len(g)), clase)
    return thr, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "figs" / "fig01_problema"))
    a = ap.parse_args(argv)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    E.aplicar()
    n, pac, _, _ = tabla_train()
    K = len(FUENTES)
    W, H = 90.0, 118.0
    fig = E.figura(W, H)

    def caja(x_mm, y_sup_mm, w_mm, h_mm):
        return fig.add_axes([x_mm / W, 1.0 - (y_sup_mm + h_mm) / H, w_mm / W, h_mm / H])

    # ---------------------------------------------------------------- (a) tabla clase por fuente
    ax = caja(24.0, 9.0, 44.0, 52.0)
    cmap = LinearSegmentedColormap.from_list("tabla", ["#F7FBFF", "#C6DBEF", "#4292C6", "#08306B"])
    M = np.ma.masked_less_equal(n.T.astype(float), 0)
    norm = LogNorm(vmin=10, vmax=5e4)
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    filas = []
    for i in range(K):
        for c in (0, 1):
            v = int(n[c, i])
            filas.append({"panel": "a", "fuente": FUENTES[i], "clase": ["negativa", "positiva"][c], "imagenes": v,
                          "pacientes": int(pac[c, i]), "semilla": "", "metrica": "", "valor": np.nan})
            if v == 0:
                ax.add_patch(plt.Rectangle((c - 0.5, i - 0.5), 1, 1, facecolor=E.GRIS["vacio"],
                                           edgecolor=E.GRIS["tenue"], hatch="/////", linewidth=0.0, zorder=2))
                ax.text(c, i, "0", ha="center", va="center", fontsize=E.PT_BASE, fontweight="bold", zorder=3,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
            else:
                ax.text(c, i, "{:,}".format(v).replace(",", " "), ha="center", va="center", fontsize=E.PT_BASE,
                        color="white" if norm(v) > 0.62 else E.GRIS["texto"], zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Negativa", "Positiva"])
    ax.xaxis.tick_top()
    ax.set_yticks(range(K))
    ax.set_yticklabels([NOMBRE[f] for f in FUENTES])
    ax.tick_params(length=0, pad=2)
    ax.set_xticks([0.5], minor=True)
    ax.set_yticks(np.arange(0.5, K - 1, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8, zorder=4)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)
    ax.set_ylabel("Fuente")
    E.rotulo(ax, "a", dx_mm=-23.0, dy_mm=3.5)
    cax = caja(24.0 + 44.0 + 2.0, 9.0, 1.8, 52.0)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Imágenes de entrenamiento")
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(length=2, width=0.6, pad=1.5, which="both")
    cb.ax.minorticks_off()

    # ---------------------------------------------------------------- (b) aciertos por fuente del modelo sin corregir
    axb = caja(36.0, 76.0, 46.0, 30.0)
    res = [aciertos(c) for c in CORRIDAS if (c / "predictions" / "best_test.csv").exists()]
    if not res:
        raise SystemExit("faltan las corridas de verificación: %s" % CORRIDAS[0].parent)
    nombres = list(res[0][1].keys())
    for k, nom in enumerate(nombres):
        vals = [r[1][nom][0] for r in res]
        clase = res[0][1][nom][2]
        axb.hlines(k, 0, float(np.mean(vals)), color=E.COLOR_CLASE[clase], linewidth=3.0, alpha=0.35, zorder=2)
        axb.plot(vals, [k] * len(vals), linestyle="none", marker="o" if clase == 0 else "s", markersize=4.5,
                 markerfacecolor=E.COLOR_CLASE[clase], markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        axb.text(1.03, k, "%.2f" % float(np.mean(vals)), va="center", ha="left", fontsize=E.PT_BASE,
                 transform=axb.get_yaxis_transform())
        for (thr, r), corrida in zip(res, CORRIDAS):
            filas.append({"panel": "b", "fuente": "", "clase": ["negativa", "positiva"][clase], "imagenes": r[nom][1],
                          "pacientes": "", "semilla": corrida.name, "metrica": nom, "valor": r[nom][0]})
    axb.set_yticks(range(len(nombres)))
    axb.set_yticklabels(nombres)
    axb.set_ylim(len(nombres) - 0.5, -0.5)
    axb.set_xlim(0, 1)
    axb.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axb.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    axb.xaxis.grid(True)
    axb.tick_params(axis="y", length=0)
    axb.set_xlabel("Proporción de imágenes bien clasificadas")
    E.rotulo(axb, "b", dx_mm=-35.0, dy_mm=3.5)

    out = E.guardar(fig, a.out, datos=pd.DataFrame(filas))
    print("tabla (negativa, positiva) por fuente:", {f: (int(n[0, i]), int(n[1, i])) for i, f in enumerate(FUENTES)})
    print("celdas vacías:", int((n == 0).sum()), "| umbrales:", [round(r[0], 3) for r in res])
    for k, v in out.items():
        print(k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
