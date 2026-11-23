"""Caché de imágenes preprocesadas: data/cache/<dataset>/<canal>_<resolución>/.

Formatos:
- canal "p1": .npy uint16 en coma fija (valor x 256, error <= 1/512 sobre la escala 0..255), forma (C, R, R) con
  C = 1 si los tres canales son iguales (gris replicado) y C = 3 si no. La imagen p1 ya está redimensionada a R, así
  que el aumento del artículo 1 (que actúa tras el redimensionado) se aplica igual sobre la caché.
- canal "homogeneo": PNG de 8 bits sin pérdida (P6), una sola banda.

Organización: subcarpetas de dos caracteres (md5 del identificador) e `index.csv` con image_id, rel_path,
channels, bytes y, para "homogeneo", los metadatos del recorte (útiles para el probe trivial de la ablación PRE).
Reanudable: una imagen cuyo fichero ya existe no se recalcula; cada bloque se registra en `parts/` y el índice se
reconstruye al final. Antes de empezar se procesa una muestra, se estima el tamaño total y se comprueba el espacio
libre (con margen) del disco de destino.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .paths import project_path, resolve_data
from .preprocess import preprocess_homogeneo, preprocess_p1, to_uint8_png_array

EXT = {"p1": ".npy", "homogeneo": ".png"}
META_COLS = ["orig_h", "orig_w", "trim_top", "trim_bottom", "trim_left", "trim_right", "trim_frac", "crop_side",
             "crop_mode", "orig_aspect"]


def cache_dir(dataset: str, channel: str, res: int, root: str | Path | None = None) -> Path:
    root = Path(root) if root else project_path("data", "cache")
    return root / dataset / ("%s_%d" % (channel, res))


def rel_path_for(image_id: str, channel: str) -> str:
    h = hashlib.md5(image_id.encode("utf-8")).hexdigest()[:2]
    stem = os.path.splitext(image_id)[0]
    return "%s/%s%s" % (h, stem, EXT[channel])


def _atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".%d.tmp" % os.getpid())
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def encode(arr, channel: str) -> tuple[bytes, int, dict]:
    if channel == "p1":
        x = np.asarray(arr, dtype=np.float32)
        if np.array_equal(x[0], x[1]) and np.array_equal(x[0], x[2]):
            x = x[:1]
        q = np.clip(np.rint(x * 256.0), 0, 65535).astype(np.uint16)
        buf = io.BytesIO()
        np.save(buf, q, allow_pickle=False)
        return buf.getvalue(), int(q.shape[0]), {}
    if channel == "homogeneo":
        y, meta = arr
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(to_uint8_png_array(y), mode="L").save(buf, format="PNG", optimize=False, compress_level=6)
        return buf.getvalue(), 1, meta
    raise ValueError(channel)


def decode_file(path: str | Path, channel: str):
    """Lee una entrada de la caché y devuelve float32 (3, R, R): p1 en [0, 255]; homogeneo en [0, 1]."""
    if channel == "p1":
        q = np.load(path, allow_pickle=False)
        x = q.astype(np.float32) / 256.0
    else:
        from PIL import Image
        with Image.open(path) as im:
            x = np.asarray(im, dtype=np.float32)[None] / 255.0
    if x.shape[0] == 1:
        x = np.repeat(x, 3, axis=0)
    return x


def _process_one(args):
    image_id, image_path, channel, res, out_dir, mask_path = args
    out = Path(out_dir) / rel_path_for(image_id, channel)
    if out.exists():
        return {"image_id": image_id, "rel_path": rel_path_for(image_id, channel), "status": "existente"}
    src = resolve_data(image_path)
    try:
        if channel == "p1":
            arr = preprocess_p1(src, res)
        else:
            mask = None
            if mask_path:
                from PIL import Image
                with Image.open(mask_path) as mi:
                    mask = np.asarray(mi.convert("L")) > 127
            arr = preprocess_homogeneo(src, res, mask=mask, return_meta=True)
        data, ch, meta = encode(arr, channel)
        _atomic_bytes(out, data)
        row = {"image_id": image_id, "rel_path": rel_path_for(image_id, channel), "channels": ch, "bytes": len(data),
               "status": "nuevo"}
        row.update(meta)
        return row
    except Exception as e:      # se registra y se sigue: el índice marca el error
        return {"image_id": image_id, "rel_path": "", "status": "error: %s" % (str(e)[:200])}


def estimate(rows, channel, res, sample=24, seed=0):
    """Procesa una muestra en memoria y devuelve (bytes medios por imagen, segundos por imagen)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), size=min(sample, len(rows)), replace=False)
    t, sizes = time.time(), []
    for i in idx:
        image_id, image_path = rows[i][0], rows[i][1]
        src = resolve_data(image_path)
        arr = preprocess_p1(src, res) if channel == "p1" else preprocess_homogeneo(src, res, return_meta=True)
        sizes.append(len(encode(arr, channel)[0]))
    return float(np.mean(sizes)), (time.time() - t) / max(1, len(idx))


def build_cache(df, dataset: str, channel: str, res: int, root=None, workers: int = 4, chunk: int = 256,
                margin: float = 1.25, check_space: bool = True, masks: dict | None = None, log=print,
                dry_run: bool = False):
    """Construye (o completa) la caché. `df` necesita columnas image_id e image_path. `masks` opcional:
    {image_id: ruta de la máscara pulmonar} para el recorte P3."""
    out_dir = cache_dir(dataset, channel, res, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [(r.image_id, r.image_path, channel, res, str(out_dir), (masks or {}).get(r.image_id, ""))
            for r in df[["image_id", "image_path"]].itertuples(index=False)]
    todo = [r for r in rows if not (out_dir / rel_path_for(r[0], channel)).exists()]
    per_img, sec = estimate(rows, channel, res) if rows else (0.0, 0.0)
    need = per_img * len(todo)
    free = shutil.disk_usage(out_dir).free
    plan = {"dataset": dataset, "channel": channel, "res": res, "n_total": len(rows), "n_pending": len(todo),
            "bytes_per_image": per_img, "estimated_bytes": need, "estimated_gb_total": per_img * len(rows) / 1e9,
            "free_bytes": free, "sec_per_image_1proc": sec,
            "estimated_minutes": sec * len(todo) / max(1, workers) / 60.0, "out_dir": str(out_dir)}
    log("plan de caché: " + json.dumps(plan))
    with open(out_dir / "plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    if check_space and need * margin > free:
        raise OSError("espacio insuficiente: hacen falta %.1f GB (con margen %.2f) y hay %.1f GB libres"
                      % (need * margin / 1e9, margin, free / 1e9))
    if dry_run:
        return plan
    parts = out_dir / "parts"
    parts.mkdir(exist_ok=True)
    t0, done = time.time(), 0
    ctx = get_context("spawn") if workers > 1 else None
    blocks = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
    pool = ctx.Pool(workers) if ctx else None
    try:
        for bi, blk in enumerate(blocks):
            res_rows = list(pool.imap_unordered(_process_one, blk, chunksize=8)) if pool else [_process_one(b) for b in blk]
            name = parts / ("part_%d_%06d.jsonl" % (int(t0), bi))
            _atomic_bytes(name, ("\n".join(json.dumps(r) for r in res_rows) + "\n").encode("utf-8"))
            done += len(blk)
            el = time.time() - t0
            log("caché %s: %d/%d nuevas (%.1f img/s; quedan %.1f min)" % (out_dir.name, done, len(todo), done / max(el, 1e-9),
                (len(todo) - done) / max(done / max(el, 1e-9), 1e-9) / 60.0))
    finally:
        if pool:
            pool.close(); pool.join()
    return write_index(out_dir, rows, channel)


def write_index(out_dir: Path, rows, channel):
    """Reconstruye index.csv a partir de los ficheros presentes y de los metadatos de parts/."""
    meta = {}
    for p in sorted((out_dir / "parts").glob("part_*.jsonl")) if (out_dir / "parts").exists() else []:
        for ln in open(p, encoding="utf-8"):
            if ln.strip():
                r = json.loads(ln)
                if r.get("status", "").startswith("nuevo"):
                    meta[r["image_id"]] = r
    cols = ["image_id", "rel_path", "channels", "bytes"] + (META_COLS if channel == "homogeneo" else []) + ["status"]
    n_ok, n_err = 0, 0
    tmp = out_dir / ".index.csv.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            rel = rel_path_for(r[0], channel)
            p = out_dir / rel
            if p.exists():
                m = dict(meta.get(r[0], {}))
                m.update({"image_id": r[0], "rel_path": rel, "bytes": p.stat().st_size, "status": "ok"})
                if "channels" not in m:
                    m["channels"] = (np.load(p, mmap_mode="r").shape[0] if channel == "p1" else 1)
                w.writerow(m); n_ok += 1
            else:
                w.writerow({"image_id": r[0], "rel_path": "", "status": "falta"}); n_err += 1
    os.replace(tmp, out_dir / "index.csv")
    return {"out_dir": str(out_dir), "ok": n_ok, "missing": n_err}


def read_index(out_dir) -> dict:
    """{image_id: ruta absoluta} de las entradas presentes."""
    import pandas as pd
    out_dir = Path(out_dir)
    d = pd.read_csv(out_dir / "index.csv")
    d = d[d["status"] == "ok"]
    return {i: str(out_dir / r) for i, r in zip(d["image_id"], d["rel_path"])}
