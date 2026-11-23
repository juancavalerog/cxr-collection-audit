#!/usr/bin/env python3
"""MD5 del fichero y hash perceptual (pHash DCT de 64 bits) de imagenes, con multiproceso y reanudable.

El pHash sigue el algoritmo habitual (equivalente a imagehash.phash con hash_size=8):
gris, reduccion a 32x32 (LANCZOS), DCT-II 2D, bloque 8x8 de bajas frecuencias, bit = coef > mediana.
Ademas se guarda dHash de 64 bits como segunda huella independiente.

Entrada: un manifiesto CSV con columna image_path (relativa a --root) o un directorio (--dir).
Salida: CSV con image_path,md5,phash,dhash,width,height,mode,bytes,error. Si la salida ya existe,
se omiten las rutas ya procesadas (reanudacion tras corte).

Ejemplos (servidor):
  python3 hash_images.py --manifest data/manifests/nih_cxr14.csv --root /home/administrador/notebooks/covid \
      --out data/hashes/nih_cxr14.csv --workers 32
  python3 hash_images.py --dir data/raw/tbx11k --out data/hashes/tbx11k.csv --workers 32
"""
import argparse
import csv
import hashlib
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}
FIELDS = ["image_path", "md5", "phash", "dhash", "width", "height", "mode", "bytes", "error"]
_N = 32
_DCT = np.array([[np.cos(np.pi * (2 * x + 1) * k / (2 * _N)) for x in range(_N)] for k in range(_N)])


def bits_to_hex(bits):
    v = 0
    for b in bits.flatten():
        v = (v << 1) | int(b)
    return f"{v:016x}"


def phash(gray):
    a = np.asarray(gray.resize((_N, _N), Image.LANCZOS), dtype=np.float64)
    d = _DCT @ a @ _DCT.T
    low = d[:8, :8]
    return bits_to_hex(low > np.median(low))


def dhash(gray):
    a = np.asarray(gray.resize((9, 8), Image.LANCZOS), dtype=np.float64)
    return bits_to_hex(a[:, 1:] > a[:, :-1])


def work(args):
    rel, root = args
    path = os.path.join(root, rel) if root else rel
    out = {"image_path": rel, "md5": "", "phash": "", "dhash": "", "width": "", "height": "",
           "mode": "", "bytes": "", "error": ""}
    try:
        h = hashlib.md5()
        n = 0
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
                n += len(chunk)
        out["md5"], out["bytes"] = h.hexdigest(), n
        with Image.open(path) as im:
            out["width"], out["height"], out["mode"] = im.size[0], im.size[1], im.mode
            im.draft("L", (256, 256))  # acelera JPEG grandes
            if im.mode in ("I;16", "I;16B", "I;16L", "I"):
                arr = np.asarray(im, dtype=np.float64)
                lo, hi = np.percentile(arr, [0.5, 99.5])
                arr = np.clip((arr - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
                gray = Image.fromarray(arr, "L")
            else:
                gray = im.convert("L")
            out["phash"], out["dhash"] = phash(gray), dhash(gray)
    except Exception as e:  # se registra y se sigue
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    return out


def list_inputs(a):
    if a.manifest:
        import pandas as pd
        df = pd.read_csv(a.manifest, usecols=[a.column], dtype=str)
        paths = df[a.column].dropna().drop_duplicates().tolist()
        return paths, a.root
    paths = []
    for dp, _, fns in os.walk(a.dir):
        for f in fns:
            if os.path.splitext(f)[1].lower() in EXTS:
                paths.append(os.path.relpath(os.path.join(dp, f), a.dir))
    return sorted(paths), a.dir


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest")
    src.add_argument("--dir")
    ap.add_argument("--root", default="", help="raiz de image_path cuando se usa --manifest")
    ap.add_argument("--column", default="image_path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0, help="procesar solo las primeras N (prueba)")
    ap.add_argument("--sample", type=int, default=0, help="muestra aleatoria de N rutas (prueba)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunksize", type=int, default=64)
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="detiene limpiamente tras N segundos (reanudable); 0 = sin limite")
    a = ap.parse_args()

    paths, root = list_inputs(a)
    if a.sample:
        rng = np.random.default_rng(a.seed)
        paths = sorted(rng.choice(paths, size=min(a.sample, len(paths)), replace=False).tolist())
    if a.limit:
        paths = paths[:a.limit]
    done = set()
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        with open(a.out, newline="") as fh:
            for r in csv.DictReader(fh):
                if not r.get("error"):
                    done.add(r["image_path"])
    todo = [p for p in paths if p not in done]
    print(f"total {len(paths)}, ya hechos {len(done & set(paths))}, pendientes {len(todo)}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    new = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    t0, k, errs = time.time(), 0, 0
    with open(a.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with Pool(a.workers) as pool:
            for r in pool.imap_unordered(work, ((p, root) for p in todo), chunksize=a.chunksize):
                w.writerow(r)
                k += 1
                errs += bool(r["error"])
                if a.max_seconds and time.time() - t0 > a.max_seconds:
                    print("limite de tiempo alcanzado; relance para continuar", flush=True)
                    pool.terminate()
                    break
                if k % 2000 == 0:
                    fh.flush()
                    rate = k / (time.time() - t0)
                    print(f"{k}/{len(todo)}  {rate:.1f} img/s  errores {errs}", flush=True)
    print(f"terminado: {k} nuevas, {errs} errores, {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
