#!/usr/bin/env python3
"""Emparejamiento de las imágenes NLM de la base de Rahman con las copias oficiales de Shenzhen y Montgomery.

El pHash no las empareja (la base de Rahman las recorta o reprocesa), así que se compara cada imagen NLM de Rahman
con todas las oficiales mediante la correlación de miniaturas 64 x 64, también con un recorte central de las
oficiales (por si Rahman recortó bordes), y se guarda la mejor pareja. Fases reanudables:

    python3 scripts/data/match_rahman_nlm.py --fase miniaturas   # repetir hasta 'miniaturas completas'
    python3 scripts/data/match_rahman_nlm.py --fase emparejar
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = ROOT / "data" / "manifests" / "bench" / "b2"
CACHE = ROOT / "logs" / "datos_plan_b" / "miniaturas_nlm.npz"
Image.MAX_IMAGE_PIXELS = None


def thumb(p, crop=0.0):
    with Image.open(p) as im:
        im.draft("L", (512, 512))
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            a = np.asarray(im, dtype=np.float64)
            lo, hi = np.percentile(a, [0.5, 99.5])
            im = Image.fromarray(np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8), "L")
        else:
            im = im.convert("L")
        if crop:
            w, h = im.size
            im = im.crop((int(w * crop), int(h * crop), int(w * (1 - crop)), int(h * (1 - crop))))
        x = np.asarray(im.resize((64, 64), Image.BILINEAR), dtype=np.float32).ravel()
    return (x - x.mean()) / (x.std() + 1e-6)


def listas():
    rb = "data/raw/tb_rahman/TB_Chest_Radiography_Database/"
    rah = []
    for nombre in ("Tuberculosis", "Normal"):
        md = pd.read_excel(ROOT / rb / ("%s.metadata.xlsx" % nombre))
        rah += [rb + "%s/%s.png" % (nombre, f) for f, u in zip(md["FILE NAME"], md["URL"]) if "ncbi" in u]
    nlm = ["data/raw/shenzhen/CXR_png/" + p.name for p in sorted((ROOT / "data/raw/shenzhen/CXR_png").glob("*.png"))]
    nlm += ["data/raw/montgomery/MontgomerySet/CXR_png/" + p.name
            for p in sorted((ROOT / "data/raw/montgomery/MontgomerySet/CXR_png").glob("*.png"))]
    return rah, nlm


def fase_miniaturas(max_s=150):
    rah, nlm = listas()
    st = dict(np.load(CACHE, allow_pickle=True)) if CACHE.exists() else {}
    keys = set(st.get("keys", []))
    data = {k: v for k, v in zip(st.get("keys", []), st.get("vals", []))}
    t = time.time()
    trabajo = [(p, 0.0) for p in rah] + [(p, c) for p in nlm for c in (0.0, 0.05, 0.10)]
    for p, c in trabajo:
        k = "%s|%.2f" % (p, c)
        if k in keys:
            continue
        if time.time() - t > max_s:
            break
        data[k] = thumb(ROOT / p, c)
        keys.add(k)
    np.savez(CACHE, keys=np.array(list(data), dtype=object), vals=np.stack(list(data.values())))
    falta = len(trabajo) - len(data)
    print("miniaturas: %d de %d%s" % (len(data), len(trabajo), " (miniaturas completas)" if not falta else ""))
    return 0 if not falta else 3


def fase_emparejar():
    rah, nlm = listas()
    st = np.load(CACHE, allow_pickle=True)
    data = dict(zip(st["keys"], st["vals"]))
    R = np.stack([data["%s|0.00" % p] for p in rah])
    best = np.full(len(rah), -2.0), np.zeros(len(rah), int), np.zeros(len(rah))
    second = np.full(len(rah), -2.0)
    for c in (0.0, 0.05, 0.10):
        N = np.stack([data["%s|%.2f" % (p, c)] for p in nlm])
        C = R @ N.T / R.shape[1]
        for i in range(len(rah)):
            j = int(np.argmax(C[i]))
            if C[i, j] > best[0][i]:
                best[0][i], best[1][i], best[2][i] = C[i, j], j, c
        srt = np.sort(C, axis=1)
        second = np.maximum(second, srt[:, -2])
    d = pd.DataFrame({"image_path": rah, "nlm_path": [nlm[j] for j in best[1]], "thumb_corr": best[0],
                      "recorte": best[2], "segunda_corr": second})
    d["clase_rahman"] = d.image_path.str.contains("/Tuberculosis/").astype(int)
    d["clase_nlm"] = d.nlm_path.str.endswith("_1.png").astype(int)
    d.to_csv(OUT / "rahman_nlm_emparejamiento.csv", index=False)
    print(d.thumb_corr.describe().to_string())
    print("corr >= 0.90: %d; >= 0.95: %d; parejas NLM distintas (>= 0.90): %d; clase coincidente: %d de %d" % (
        (d.thumb_corr >= 0.9).sum(), (d.thumb_corr >= 0.95).sum(), d[d.thumb_corr >= 0.9].nlm_path.nunique(),
        (d.clase_rahman == d.clase_nlm).sum(), len(d)))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fase", choices=["miniaturas", "emparejar"], required=True)
    a = ap.parse_args()
    sys.exit(fase_miniaturas() if a.fase == "miniaturas" else fase_emparejar())
