#!/usr/bin/env python3
"""Pseudomáscaras pulmonares del profesor para las imágenes de un benchmark del plan B.

Envoltorio de src/seg/predict_masks.py (que no se modifica): toma la lista de imágenes distintas de los manifiestos
(src/data/bench.py:bench_images, rutas absolutas resueltas en la máquina que ejecuta) y llama a `predict_masks` y a
`export_orig` con los mismos parámetros que el trabajo p3_03 de COVIDx. Salida en data/masks/bench_<id>/, con
orig/index.csv listo para scripts/build_cache.py --masks.

    python scripts/data/bench_masks.py --teacher results/seg_teacher/states/best.pt \
        --bench data/manifests/bench/b3a --workers 12 --batch 64
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", type=int, default=2048)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    import torch
    from src.data.bench import bench_images
    from src.seg.predict_masks import export_orig, load_teacher, predict_masks
    df = bench_images(a.bench)[["image_id", "image_path"]]
    if a.limit:
        df = df.head(a.limit)
    name = "bench_" + Path(a.bench).resolve().name.replace(".csv", "")
    out = Path(a.out or (ROOT / "data" / "masks" / name))
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, res = load_teacher(a.teacher, dev)
    print(json.dumps(predict_masks(model, res, df, out, dev, a.batch, a.workers, a.shard)))
    print(json.dumps(export_orig(out, a.workers)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
