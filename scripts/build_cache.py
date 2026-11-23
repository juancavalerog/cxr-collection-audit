#!/usr/bin/env python3
"""Construye la caché de imágenes preprocesadas (src/data/cache.py).

Ejemplos:
    python scripts/build_cache.py --channel p1 --res 224 --workers 16
    python scripts/build_cache.py --channel homogeneo --res 224 --workers 16 --masks <csv image_id,mask_path>
    python scripts/build_cache.py --channel p1 --dry-run          # solo estimación de tamaño y tiempo
    python scripts/build_cache.py --channel homogeneo --res 224 --root data/cache_m \
        --bench data/manifests/bench/b3a --masks data/masks/bench_b3a/orig/index.csv --workers 16
        # plan B: todas las imágenes distintas de los manifiestos de un benchmark (src/data/bench.py); la caché
        # se llama bench_<id> salvo que se indique --dataset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.cache import build_cache  # noqa: E402
from src.data.manifest import load_manifest  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="covidx_cxr4")
    ap.add_argument("--channel", choices=["p1", "homogeneo"], required=True)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--root", default=None, help="raíz de la caché (por defecto data/cache)")
    ap.add_argument("--splits", nargs="+", default=None, help="limitar a particiones oficiales (train val test)")
    ap.add_argument("--limit", type=int, default=0, help="solo las N primeras filas (pruebas)")
    ap.add_argument("--masks", default=None, help="CSV con columnas image_id,mask_path (opcional)")
    ap.add_argument("--bench", default=None, help="manifiesto o directorio de data/manifests/bench (plan B)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-space-check", action="store_true")
    a = ap.parse_args(argv)
    if a.bench:
        from src.data.bench import bench_images
        df = bench_images(a.bench)
        if a.dataset == "covidx_cxr4":
            a.dataset = "bench_" + Path(a.bench).resolve().name.replace(".csv", "")
        if a.splits:
            raise SystemExit("--splits no se aplica con --bench")
    elif a.dataset != "covidx_cxr4":
        raise SystemExit("por ahora solo covidx_cxr4 (los manifiestos de NIH y CheXpert tienen otro formato)")
    else:
        df = load_manifest()
    if a.splits:
        df = df[df["split"].isin(a.splits)]
    if a.limit:
        df = df.head(a.limit)
    masks = None
    if a.masks:
        import pandas as pd
        m = pd.read_csv(a.masks)
        masks = dict(zip(m["image_id"], m["mask_path"]))
    r = build_cache(df, a.dataset, a.channel, a.res, root=a.root, workers=a.workers, masks=masks,
                    check_space=not a.no_space_check, dry_run=a.dry_run)
    print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
