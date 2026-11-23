"""Figura S5: la fuga entre particiones de COVIDx CXR-4 (docs/09_figuras.md). Borrador en español.

(a) Dos pacientes de NIH ChestX-ray14 con imágenes en el test y en el entrenamiento de la partición oficial: la
    imagen del test en COVIDx, su copia en NIH y otro estudio del mismo paciente en el entrenamiento de COVIDx.
(b) Pacientes NIH con imágenes RSNA en COVIDx según las particiones oficiales en las que aparecen.

Las radiografías se dibujan desde los archivos originales (1 024 x 1 024). Datos: data/manifests/
covidx_rsna_to_nih_map.csv. Uso: python scripts/figs/figS5_fuga.py [--pacientes nih_10496 nih_10613]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.paths import resolve_data  # noqa: E402
from src.viz import estilo as E  # noqa: E402

W, H = 140.0, 168.0
CATS = [("train", "Solo entrenamiento"), ("val", "Solo validación"), ("test", "Solo test"),
        ("train+val", "Entrenamiento y validación"), ("test+train", "Test y entrenamiento"),
        ("test+val", "Test y validación"), ("test+train+val", "Las tres particiones")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "figs" / "figS5_fuga"))
    ap.add_argument("--pacientes", nargs=2, default=["nih_10496", "nih_10613"])
    a = ap.parse_args(argv)
    from PIL import Image
    E.aplicar()
    m = pd.read_csv(ROOT / "data" / "manifests" / "covidx_rsna_to_nih_map.csv", dtype=str)
    fig = E.figura(W, H)

    def caja(x_mm, y_sup_mm, w_mm, h_mm):
        return fig.add_axes([x_mm / W, 1.0 - (y_sup_mm + h_mm) / H, w_mm / W, h_mm / H])

    def leer(rel):
        im = Image.open(resolve_data(rel)).convert("L")
        if min(im.size) < 531:
            raise ValueError("%s tiene menos de 531 píxeles de lado" % rel)
        return np.asarray(im)

    # ---------------------------------------------------------------- (a) ejemplos
    lado, sep, x0, y0 = 40.0, 4.0, 10.0, 14.0
    titulos = ["COVIDx, test\n ", "NIH ChestX-ray14\nla misma imagen", "COVIDx, entrenamiento\notro estudio del mismo paciente"]
    filas = []
    for r, pac in enumerate(a.pacientes):
        g = m[m["nih_patient_id"] == pac]
        t, e = g[g["split"] == "test"], g[g["split"] == "train"]
        if t.empty or e.empty:
            raise SystemExit("el paciente %s no tiene imágenes en test y en entrenamiento" % pac)
        t, e = t.iloc[0], e.iloc[0]
        for c, (rel, quien) in enumerate(((t["path_covidx"], "covidx_test"), (t["path_nih"], "nih_copia"),
                                          (e["path_covidx"], "covidx_train"))):
            ax = caja(x0 + c * (lado + sep), y0 + r * (lado + 7.0), lado, lado)
            ax.imshow(leer(rel), cmap="gray", vmin=0, vmax=255, interpolation="none")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_linewidth(0.6)
            if r == 0:
                ax.set_title(titulos[c], fontsize=E.PT_BASE - 0.5, pad=3)
            if c == 0:
                ax.set_ylabel("Paciente %d" % (r + 1))
                if r == 0:
                    E.rotulo(ax, "a", dx_mm=-9.0, dy_mm=5.5)
            filas.append({"panel": "a", "paciente_nih": pac, "papel": quien, "ruta": rel, "categoria": "", "pacientes": ""})
        fig.text((x0 + lado + sep / 2) / W, 1.0 - (y0 + r * (lado + 7.0) + lado / 2) / H, "=", ha="center", va="center",
                 fontsize=11, fontweight="bold")

    # ---------------------------------------------------------------- (b) pacientes por particiones
    ps = m.groupby("nih_patient_id")["split"].agg(lambda s: "+".join(sorted(set(s))))
    cuenta = ps.value_counts().to_dict()
    desconocidas = set(cuenta) - {k for k, _ in CATS}
    if desconocidas:
        raise ValueError("combinaciones no previstas: %s" % desconocidas)
    vals = [int(cuenta.get(k, 0)) for k, _ in CATS]
    ax = caja(50.0, 114.0, 72.0, 40.0)
    pos = np.arange(len(CATS))
    col = [E.GRIS["tenue"] if "+" not in k else E.OKABE_ITO["bermellon"] for k, _ in CATS]
    ax.barh(pos, vals, color=col, height=0.62, edgecolor="white", linewidth=0.3)
    for p_, v in zip(pos, vals):
        ax.text(v * 1.12, p_, "{:,}".format(v).replace(",", " "), va="center", ha="left", fontsize=E.PT_BASE - 0.5)
    ax.set_xscale("log")
    ax.set_xlim(5, 3e4)
    ax.set_yticks(pos)
    ax.set_yticklabels([n for _, n in CATS])
    ax.set_ylim(len(CATS) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True)
    ax.set_xlabel("Pacientes de NIH con imágenes RSNA en COVIDx (escala logarítmica)")
    E.rotulo(ax, "b", dx_mm=-49.0, dy_mm=3.0)
    for (k, nmb), v in zip(CATS, vals):
        filas.append({"panel": "b", "paciente_nih": "", "papel": "", "ruta": "", "categoria": nmb, "pacientes": v})
    comp = sum(v for (k, _), v in zip(CATS, vals) if "+" in k)
    con_test = sum(v for (k, _), v in zip(CATS, vals) if "+" in k and "test" in k)
    t = m[m["split"] == "test"]
    fuera = set(m[m["split"] != "test"]["nih_patient_id"])
    out = E.guardar(fig, a.out, datos=pd.DataFrame(filas))
    print("pacientes NIH: %d; en más de una partición: %d (%.1f %%); entre test y otra: %d" %
          (len(ps), comp, 100.0 * comp / len(ps), con_test))
    print("imágenes RSNA del test: %d; con el paciente también en entrenamiento o validación: %d" %
          (len(t), int(t["nih_patient_id"].isin(fuera).sum())))
    for k, v in out.items():
        print(k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
