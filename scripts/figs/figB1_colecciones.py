"""Figura B1 del plan B: composición de bimcv por colecciones (docs/09_figuras.md, Section 7). Borrador en español.

(a) Fracción de las imágenes de bimcv de cada partición que pertenece a cada colección, con su recuento.
(b) Recuento de imágenes por colección y partición, en escala logarítmica.

Datos: Tabla E1 de docs/07_piloto.md. Las cifras de entrenamiento y validación se recalculan aquí desde el manifiesto
de la partición limpia (data/manifests/covidx_cxr4_clean.csv) y el programa se detiene si no coinciden con la tabla.
Las del test se toman de la Tabla E1 tal como están registradas y no se recalculan, porque exigirían leer las
etiquetas del test, que sigue bloqueado.

Uso: python scripts/figs/figB1_colecciones.py [--out results/figs/figB1_colecciones]
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

COLECCIONES = ("Negativos", "Positivos, identificador de cinco cifras", "Positivos, identificador de seis cifras")
PARTICIONES = ("Entrenamiento", "Validación", "Test")
# Tabla E1 de docs/07_piloto.md (imágenes; pacientes entre paréntesis en el texto de la tabla)
_N, _A, _B = COLECCIONES
E1 = {(_N, "Entrenamiento"): 647, (_N, "Validación"): 257, (_N, "Test"): 4039,
      (_A, "Entrenamiento"): 5370, (_A, "Validación"): 3150, (_A, "Test"): 4040,
      (_B, "Entrenamiento"): 37123, (_B, "Validación"): 0, (_B, "Test"): 0}
COLOR = {_N: E.COLOR_CLASE[0], _A: E.OKABE_ITO["bermellon"], _B: E.OKABE_ITO["naranja"]}
TRAMA = {_N: "", _A: "", _B: "////"}
ETIQUETA_EJE = {_N: "Negativos", _A: "Positivos,\ncinco cifras", _B: "Positivos,\nseis cifras"}
MARCA = {"Entrenamiento": "o", "Validación": "s", "Test": "D"}


def recontar(manifiesto: Path) -> pd.DataFrame:
    """Recuento de entrenamiento y validación desde el manifiesto limpio (no lee el test)."""
    m = pd.read_csv(manifiesto, dtype=str, usecols=["source", "image_path", "label", "split_clean"])
    m = m[(m["source"] == "bimcv") & m["split_clean"].isin(["train", "val"])]
    sid = m["image_path"].str.split("/").str[-1].str.extract(r"^sub-S(\d+)_")[0].str.len()
    col = np.where(m["label"] == "0", COLECCIONES[0], np.where(sid == 5, COLECCIONES[1], COLECCIONES[2]))
    part = np.where(m["split_clean"] == "train", "Entrenamiento", "Validación")
    return pd.DataFrame({"coleccion": col, "particion": part}).value_counts().rename("imagenes").reset_index()


def datos(manifiesto: Path) -> pd.DataFrame:
    rc = recontar(manifiesto).set_index(["coleccion", "particion"])["imagenes"]
    filas = []
    for (c, p), v in E1.items():
        if p != "Test":
            got = int(rc.get((c, p), 0))
            if got != v:
                raise SystemExit("el manifiesto da %d imágenes en %s / %s y la Tabla E1 %d" % (got, c, p, v))
        filas.append({"coleccion": c, "particion": p, "imagenes": v,
                      "origen": "Tabla E1" if p == "Test" else "manifiesto limpio, coincide con la Tabla E1"})
    d = pd.DataFrame(filas)
    d["fraccion_de_la_particion"] = d["imagenes"] / d.groupby("particion")["imagenes"].transform("sum")
    return d


def dibujar(d: pd.DataFrame):
    E.aplicar()
    fig = E.figura(90, 118)
    ax1 = fig.add_axes([0.24, 0.57, 0.72, 0.29])
    ax2 = fig.add_axes([0.24, 0.10, 0.72, 0.33])
    # (a) composición de cada partición
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
                ax1.text(izq + w / 2, i, "{:,}".format(int(r["imagenes"])).replace(",", " "), ha="center", va="center",
                         fontsize=6.5, color="white" if c != COLECCIONES[2] else E.GRIS["texto"],
                         fontweight="bold",
                         bbox=(dict(facecolor="white", edgecolor="none", pad=1.2) if TRAMA[c] else None))
            elif w > 0:
                ax1.text(izq + w + 0.012, i - 0.33, "{:,}".format(int(r["imagenes"])).replace(",", " "), ha="left",
                         va="bottom", fontsize=6.5, color=E.GRIS["texto"])
            izq += w
    ax1.set_yticks(range(len(PARTICIONES)), PARTICIONES)
    ax1.set_ylim(len(PARTICIONES) - 0.5, -0.75)
    ax1.set_xlim(0, 1)
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    ax1.set_xlabel("Imágenes de BIMCV en la partición (%)")
    from matplotlib.patches import Patch
    ax1.legend([Patch(facecolor=COLOR[c], hatch=TRAMA[c], edgecolor="white") for c in COLECCIONES], COLECCIONES,
               loc="lower left", bbox_to_anchor=(0.0, 1.03), ncol=1, handlelength=1.6, borderaxespad=0)
    E.rotulo(ax1, "a", dx_mm=-19)
    # (b) recuentos absolutos en escala logarítmica
    for j, c in enumerate(COLECCIONES):
        for k, p in enumerate(PARTICIONES):
            v = int(d[(d.coleccion == c) & (d.particion == p)]["imagenes"].iloc[0])
            yy = j + (k - 1) * 0.22
            if v > 0:
                ax2.plot([v], [yy], marker=MARCA[p], color=COLOR[c], markersize=4.5, linestyle="none",
                         markeredgecolor=E.GRIS["texto"], markeredgewidth=0.5)
                ax2.text(v * 1.25, yy, "{:,}".format(v).replace(",", " "), va="center", fontsize=6.5)
            else:
                ax2.text(12, yy, "0 en %s" % p.lower(), va="center", fontsize=6.5, color=E.GRIS["tenue"])
    ax2.set_xscale("log")
    ax2.set_xlim(10, 3e5)
    ax2.set_yticks(range(len(COLECCIONES)), [ETIQUETA_EJE[c] for c in COLECCIONES])
    ax2.set_ylim(len(COLECCIONES) - 0.5, -0.5)
    ax2.set_xlabel("Imágenes (escala logarítmica)")
    ax2.grid(axis="x", which="major")
    ax2.legend([ax2.plot([], [], marker=MARCA[p], color=E.GRIS["tenue"], linestyle="none", markersize=4.5)[0]
                for p in PARTICIONES], PARTICIONES, loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=3,
               handletextpad=0.3, columnspacing=0.9, borderaxespad=0)
    E.rotulo(ax2, "b", dx_mm=-19)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifiesto", default=str(ROOT / "data/manifests/covidx_cxr4_clean.csv"))
    ap.add_argument("--out", default=str(ROOT / "results/figs/figB1_colecciones"))
    a = ap.parse_args(argv)
    d = datos(Path(a.manifiesto))
    fig = dibujar(d)
    out = C.guardar(fig, a.out, datos=d)
    print("\n".join("%s: %s" % kv for kv in out.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
