"""Figura S1: flujo de datos y partición limpia de COVIDx CXR-4 (docs/09_figuras.md). Borrador en español.

De la partición oficial a los conjuntos del estudio, con cada exclusión y su motivo. Todas las cifras se calculan
del manifiesto data/manifests/covidx_cxr4_clean.csv; el CSV adjunto las recoge.

Uso: python scripts/figs/figS1_flujo_datos.py [--out results/figs/figS1_flujo_datos]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.viz import estilo as E  # noqa: E402

W, H = 140.0, 168.0
AZUL, AZUL_F = "#0072B2", "#DCEAF5"
NAR, NAR_F = "#D55E00", "#FBE7D6"
GRIS_B = "#666666"


def n(v):
    return "{:,}".format(int(v)).replace(",", " ")


def cifras(manifiesto):
    d = pd.read_csv(manifiesto, low_memory=False, dtype={"patient_id": str})
    r = {"total": len(d)}
    for s in ("train", "val", "test"):
        r["oficial_" + s] = int((d["split"] == s).sum())
        g = d[d["split_clean"] == s]
        r["limpia_" + s] = len(g)
        r["limpia_%s_neg" % s] = int((g["label"] == 0).sum())
        r["limpia_%s_pos" % s] = int((g["label"] == 1).sum())
        r["limpia_%s_pacientes" % s] = int((g["source"] + "|" + g["patient_id"].astype(str)).nunique())
    ex = d["exclude_reason"].fillna("")
    r["excluidas"] = int((d["split_clean"] == "excluded").sum())
    r["exc_rsna_test"] = int((ex == "rsna_nih_patient_in_test").sum())
    r["exc_duplicados"] = int((ex == "duplicate_cross_split").sum())
    r["exc_etiquetas"] = int((ex == "label_conflict").sum())
    r["mov_val_a_train"] = int((ex == "nih_patient_regrouped_to_train").sum())
    r["mov_train_a_val"] = int((ex == "nih_patient_regrouped_to_val").sum())
    if r["exc_rsna_test"] + r["exc_duplicados"] + r["exc_etiquetas"] != r["excluidas"]:
        raise ValueError("los motivos de exclusión no suman el total de excluidas")
    vr = d[(d["split_clean"] == "val") & (d["val_robust"] == 1)]
    r["val_robusta"], r["val_robusta_neg"], r["val_robusta_pos"] = len(vr), int((vr["label"] == 0).sum()), int((vr["label"] == 1).sum())
    t = d[d["split_clean"] == "test"]
    tb = t[t["test_subset"] == "bimcv"]
    r["test_bimcv"], r["test_bimcv_neg"], r["test_bimcv_pos"] = len(tb), int((tb["label"] == 0).sum()), int((tb["label"] == 1).sum())
    r["test_ricord"] = int((t["test_subset"] == "ricord").sum())
    r["test_rsna"] = int(t["test_subset"].astype(str).str.startswith("rsna").sum())
    r["limpia_total"] = r["limpia_train"] + r["limpia_val"] + r["limpia_test"]
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "figs" / "figS1_flujo_datos"))
    ap.add_argument("--manifiesto", default=str(ROOT / "data" / "manifests" / "covidx_cxr4_clean.csv"))
    a = ap.parse_args(argv)
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    E.aplicar()
    c = cifras(a.manifiesto)
    fig = E.figura(W, H)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    PT = E.PT_BASE

    def caja(x, y, w, h, titulo, lineas=(), borde=AZUL, fondo=AZUL_F, grosor=0.8, izq=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", linewidth=grosor,
                                    edgecolor=borde, facecolor=fondo, zorder=2))
        paso = 3.5
        k = 1 + len(lineas)
        y0 = y + h / 2 - (k - 1) * paso / 2
        xt, ha = (x + 2.5, "left") if izq else (x + w / 2, "center")
        ax.text(xt, y0, titulo, ha=ha, va="center", fontsize=PT, fontweight="bold", zorder=3)
        for i, t in enumerate(lineas):
            ax.text(xt, y0 + (i + 1) * paso, t, ha=ha, va="center", fontsize=PT - 0.5, zorder=3)

    def flecha(p, q, color="#333333"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, linewidth=0.8, color=color, shrinkA=0,
                                     shrinkB=0, zorder=4))

    def linea(pts, color="#333333"):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=0.8, zorder=4, solid_capstyle="round")

    cx = 44.0                                   # eje del flujo principal
    # 1. partición oficial
    caja(8, 6, 72, 15, "COVIDx CXR-4, partición oficial", ("%s imágenes" % n(c["total"]),
         "entrenamiento %s, validación %s, test %s" % (n(c["oficial_train"]), n(c["oficial_val"]), n(c["oficial_test"]))),
         GRIS_B, "#FFFFFF", 0.6)
    # 2. exclusiones
    flecha((cx, 21), (cx, 58))
    linea([(cx, 33), (82, 33)], color=NAR)
    flecha((82, 33), (84, 33), color=NAR)
    caja(84, 22, 53, 22, "Excluidas: %s imágenes" % n(c["excluidas"]),
         ("%s de RSNA con el paciente en el test" % n(c["exc_rsna_test"]),
          "%s casi duplicadas entre particiones" % n(c["exc_duplicados"]),
          "%s con etiquetas opuestas entre copias" % n(c["exc_etiquetas"])), NAR, NAR_F, izq=True)
    # 3. reagrupación
    linea([(cx, 54), (84, 54)], color=GRIS_B)
    caja(84, 47, 53, 14.5, "Reagrupadas por paciente", ("%s de validación a entrenamiento" % n(c["mov_val_a_train"]),
         "%s de entrenamiento a validación" % n(c["mov_train_a_val"])), GRIS_B, "#FFFFFF", 0.6, izq=True)
    # 4. partición limpia
    caja(8, 58, 72, 11.5, "Partición limpia", ("%s imágenes, sin pacientes compartidos" % n(c["limpia_total"]),))
    xs, w3 = [4.0, 49.0, 94.0], 42.0
    linea([(cx, 69.5), (cx, 75), (xs[0] + w3 / 2, 75)])
    linea([(cx, 75), (xs[2] + w3 / 2, 75)])
    nombres = [("Entrenamiento", "train"), ("Validación", "val"), ("Test", "test")]
    for x, (nom, s) in zip(xs, nombres):
        flecha((x + w3 / 2, 75), (x + w3 / 2, 81))
        caja(x, 81, w3, 19, nom, ("%s imágenes" % n(c["limpia_" + s]),
             "%s negativas y %s positivas" % (n(c["limpia_%s_neg" % s]), n(c["limpia_%s_pos" % s])),
             "%s pacientes" % n(c["limpia_%s_pacientes" % s])))
    # 5. subconjuntos de evaluación
    flecha((xs[1] + w3 / 2, 100), (xs[1] + w3 / 2, 110))
    caja(xs[1], 110, w3, 19, "Validación robusta", ("solo BIMCV: %s imágenes" % n(c["val_robusta"]),
         "%s negativas y %s positivas" % (n(c["val_robusta_neg"]), n(c["val_robusta_pos"])),
         "selección de modelo"), grosor=1.3)
    flecha((xs[2] + w3 / 2, 100), (xs[2] + w3 / 2, 110))
    caja(xs[2], 110, w3, 19, "Evaluación principal", ("BIMCV del test: %s imágenes" % n(c["test_bimcv"]),
         "%s negativas y %s positivas" % (n(c["test_bimcv_neg"]), n(c["test_bimcv_pos"])),
         "comparación entre métodos"), grosor=1.3)
    linea([(xs[2] + w3 / 2, 105), (xs[2] - 1.5, 105), (xs[2] - 1.5, 144.5)])
    flecha((xs[2] - 1.5, 144.5), (xs[2], 144.5))
    caja(xs[2], 137, w3, 15, "Análisis secundarios", ("RICORD: %s positivas" % n(c["test_ricord"]),
         "RSNA: %s negativas" % n(c["test_rsna"])), GRIS_B, "#FFFFFF", 0.6)
    flecha((xs[0] + w3 / 2, 100), (xs[0] + w3 / 2, 110))
    caja(xs[0], 110, w3, 19, "Ajuste de los modelos", ("8 fuentes; 5 celdas", "clase por fuente vacías", "(Figura 1)"),
         GRIS_B, "#FFFFFF", 0.6)
    # leyenda
    yl = 160.0
    for k, (bo, fo, t) in enumerate(((AZUL, AZUL_F, "Conjuntos del estudio"), (NAR, NAR_F, "Imágenes excluidas"),
                                     (GRIS_B, "#FFFFFF", "Origen, movimientos y usos"))):
        x0 = 8.0 + k * 44.0
        ax.add_patch(FancyBboxPatch((x0, yl - 1.8), 6, 3.6, boxstyle="round,pad=0,rounding_size=0.9", linewidth=0.7,
                                    edgecolor=bo, facecolor=fo, zorder=2))
        ax.text(x0 + 7.5, yl, t, ha="left", va="center", fontsize=PT - 0.5)

    datos = pd.DataFrame([{"cifra": k, "valor": v} for k, v in c.items()])
    out = E.guardar(fig, a.out, datos=datos)
    print({k: c[k] for k in ("total", "excluidas", "limpia_total", "val_robusta", "test_bimcv")})
    for k, v in out.items():
        print(k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
