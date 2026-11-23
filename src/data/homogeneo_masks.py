"""Preprocesado homogéneo con máscaras para CSD-Net: lo que falta sobre src/data/preprocess.py y src/data/cache.py,
que no se modifican ni se reimplementan.

Cadena acordada el 12-11-2026, común a CSD-Net y a las líneas base:
1. Pseudomáscaras en la rejilla original: src/seg/predict_masks.py escribe data/masks/<dataset>/orig/*.png y
   orig/index.csv (image_id, mask_path, quality; solo q > 0).
2. Caché común de imágenes con P1 a P6 y recorte P3 centrado en los pulmones (docs/03_arquitectura.md,
   Section 2.1), construida por el código de 5b:
       python scripts/build_cache.py --channel homogeneo --res 224 --root data/cache_m \
           --masks data/masks/covidx_cxr4/orig/index.csv --workers 16
   -> data/cache_m/covidx_cxr4/homogeneo_224/ (PNG de 8 bits e index.csv con trim_*, crop_side y crop_mode).
3. Máscaras alineadas con la caché (este módulo):
       python -m src.data.homogeneo_masks --cache data/cache_m/covidx_cxr4/homogeneo_224 \
           --orig data/masks/covidx_cxr4/orig --out data/masks/covidx_cxr4/homogeneo_224 --workers 16
   Para cada imagen se repite la geometría de la caché: recorte de bordes con los trim_* del índice, recorte
   cuadrado P3 con `square_crop_box` de preprocess.py sobre la misma máscara que leyó la caché (pulmón > 127),
   lo que reproduce exactamente top y left (el índice guarda crop_side y crop_mode, pero no top ni left), y
   reducción al tamaño de la imagen de la caché (bilineal y umbral 0.5). Salida: un PNG de 8 bits por imagen
   con la codificación 0 fondo, 100 solo corazón, 200 solo pulmón y 255 pulmón y corazón (la que lee
   src/baselines/trainer.py, MaskStore: pulmón = valor > 127) e index.csv con image_id, path (relativa al
   directorio), quality, lung_frac, heart_frac, crop_mode y la caja torácica (box_y0, box_y1, box_x0, box_x1,
   caja de la unión de pulmones y corazón en la rejilla de la caché; el margen δ lo añade el modelo, ecuación 15).
   Una imagen sin pseudomáscara o con q = 0 recibe máscara vacía y quality 0.

`CXRMaskDataset` lee imagen y máscara (pulmones y corazón) alineadas, con la calidad, para el bucle de CSD-Net.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.preprocess import square_crop_box  # noqa: E402


def encode_code(lungs, heart) -> np.ndarray:
    l, h = np.asarray(lungs) > 0, np.asarray(heart) > 0
    out = np.zeros(l.shape, dtype=np.uint8)
    out[h & ~l] = 100
    out[l & ~h] = 200
    out[l & h] = 255
    return out


def decode_code(a) -> np.ndarray:
    a = np.asarray(a)
    return np.stack([a > 127, (a == 100) | (a == 255)]).astype(np.uint8)


def _abs(p):
    p = Path(str(p))
    return p if p.is_absolute() else ROOT / p


def masks_cache_dir(dataset: str, res: int) -> Path:
    return ROOT / "data" / "masks" / dataset / ("homogeneo_%d" % res)


def align_one(code_orig: np.ndarray | None, meta: dict, res: int):
    """Lleva una máscara codificada de la rejilla original a la de la caché con los metadatos del índice de la
    caché. Devuelve (código res x res, caja torácica o None, modo de recorte recalculado)."""
    if code_orig is None:
        return np.zeros((res, res), np.uint8), None, None
    h0, w0 = int(meta["orig_h"]), int(meta["orig_w"])
    a = np.asarray(code_orig)
    if a.shape != (h0, w0):
        a = np.asarray(Image.fromarray(a).resize((w0, h0), Image.NEAREST))
    t, l = int(meta["trim_top"]), int(meta["trim_left"])
    b, r = h0 - int(meta["trim_bottom"]), w0 - int(meta["trim_right"])
    at = a[t:b, l:r]
    lung = at > 127                                   # lo mismo que lee src/data/cache.py (_process_one)
    use_mask = str(meta.get("crop_mode")) == "mascara"
    top, left, side, how = square_crop_box(at.shape, lung if use_mask else None)
    if int(side) != int(meta["crop_side"]):
        raise ValueError("el recorte recalculado (%d) no coincide con el de la caché (%s)" % (side, meta["crop_side"]))
    ch = decode_code(at[top:top + side, left:left + side]).astype(np.float32)
    rs = [np.asarray(Image.fromarray(c, mode="F").resize((res, res), Image.BILINEAR)) > 0.5 for c in ch]
    code = encode_code(rs[0], rs[1])
    u = (code > 0)
    box = None
    if u.any():
        ys, xs = np.nonzero(u)
        box = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
    return code, box, how


def _align_task(args):
    iid, meta, orig_path, quality, res, out_dir, rel = args
    out = Path(out_dir) / rel
    if out.exists():
        return {"image_id": iid, "status": "existente"}
    try:
        code = None
        if orig_path and quality > 0:
            with Image.open(_abs(orig_path)) as im:
                code = np.asarray(im.convert("L"))
        c, box, how = align_one(code, meta, res)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name("." + out.name + ".tmp.png")
        Image.fromarray(c, mode="L").save(tmp, compress_level=6)
        os.replace(tmp, out)
        rec = {"image_id": iid, "status": "nuevo", "lung_frac": float((c > 127).mean()),
               "heart_frac": float(((c == 100) | (c == 255)).mean()), "crop_mode_recalc": how}
        if box:
            rec.update({"box_y0": box[0], "box_y1": box[1], "box_x0": box[2], "box_x1": box[3]})
        return rec
    except Exception as e:  # noqa: BLE001
        return {"image_id": iid, "status": "error: %s" % str(e)[:200]}


def align_masks_to_cache(cache_dir, orig_dir, out_dir, workers: int = 8, check_space: bool = True, log=print):
    """Construye (o completa) out_dir con las máscaras alineadas e index.csv."""
    cache_dir, orig_dir, out_dir = _abs(cache_dir), _abs(orig_dir), _abs(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ci = pd.read_csv(cache_dir / "index.csv")
    ci = ci[ci["status"] == "ok"]
    oi_path = orig_dir / "index_todas.csv"
    oi = pd.read_csv(oi_path if oi_path.exists() else orig_dir / "index.csv")
    om = {r.image_id: (r.mask_path, float(r.quality)) for r in oi.itertuples(index=False)}
    with Image.open(cache_dir / ci["rel_path"].iloc[0]) as im:
        res = im.size[0]
    tasks = []
    for r in ci.to_dict("records"):
        mp, q = om.get(r["image_id"], ("", 0.0))
        rel = os.path.splitext(r["rel_path"])[0] + ".png"
        tasks.append((r["image_id"], r, mp, q, res, str(out_dir), rel))
    need = 1500 * len(tasks)
    free = shutil.disk_usage(out_dir).free
    plan = {"n": len(tasks), "res": res, "estimated_gb": need / 1e9, "free_gb": free / 1e9}
    log("plan de máscaras alineadas: " + json.dumps(plan))
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2))
    if check_space and need * 1.25 > free:
        raise OSError("espacio insuficiente para las máscaras alineadas")
    if workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(workers) as pool:
            res_rows = list(pool.imap_unordered(_align_task, tasks, chunksize=32))
    else:
        res_rows = [_align_task(t) for t in tasks]
    by = {r["image_id"]: r for r in res_rows}
    rows = []
    for (iid, meta, mp, q, _, _, rel) in tasks:
        p = out_dir / rel
        rr = by.get(iid, {})
        if not p.exists():
            rows.append({"image_id": iid, "path": "", "quality": 0.0, "status": rr.get("status", "falta")})
            continue
        if "lung_frac" not in rr:
            with Image.open(p) as im:
                c = np.asarray(im)
            rr = {**rr, "lung_frac": float((c > 127).mean()), "heart_frac": float(((c == 100) | (c == 255)).mean())}
        rows.append({"image_id": iid, "path": rel, "quality": q if (mp and q > 0) else 0.0,
                     "lung_frac": rr.get("lung_frac"), "heart_frac": rr.get("heart_frac"),
                     "crop_mode": meta.get("crop_mode"), **{k: rr.get(k) for k in ("box_y0", "box_y1", "box_x0", "box_x1")},
                     "status": "ok"})
    d = pd.DataFrame(rows)
    tmp = out_dir / ".index.csv.tmp"
    d.to_csv(tmp, index=False)
    os.replace(tmp, out_dir / "index.csv")
    errs = [r for r in res_rows if str(r.get("status", "")).startswith("error")]
    return {"n": int(len(d)), "ok": int((d["status"] == "ok").sum()), "errores": len(errs), "out": str(out_dir)}


class CXRMaskDataset(torch.utils.data.Dataset):
    """Imagen de la caché común (canal homogeneo, [0, 1], 3 canales) y máscara alineada (2, R, R) con su calidad.
    require=True exige que todas las imágenes estén en la caché y tengan máscara (índice)."""

    def __init__(self, df, cache_dir, mask_dir, require: bool = True):
        from src.data.cache import read_index
        self.df = df.reset_index(drop=True)
        self.ids = self.df["image_id"].astype(str).tolist()
        self.labels = self.df["label"].astype("int64").to_numpy()
        self.sources = self.df["source_idx"].astype("int64").to_numpy()
        self.patients = self.df["patient_id"].astype(str).tolist()
        self.cache = read_index(_abs(cache_dir))
        md = _abs(mask_dir)
        mi = pd.read_csv(md / "index.csv")
        mi = mi[mi["status"] == "ok"]
        self.masks = {str(a): (md / str(b), float(q)) for a, b, q in zip(mi["image_id"], mi["path"], mi["quality"])}
        if require:
            miss = [i for i in self.ids if i not in self.cache or i not in self.masks]
            if miss:
                raise FileNotFoundError("faltan %d imágenes en la caché %s o en las máscaras %s (p. ej. %s)"
                                        % (len(miss), cache_dir, mask_dir, miss[0]))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        from src.data.cache import decode_file
        iid = self.ids[i]
        x = decode_file(self.cache[iid], "homogeneo")
        mp, q = self.masks.get(iid, (None, 0.0))
        if mp is not None and Path(mp).exists():
            with Image.open(mp) as im:
                m = decode_code(np.asarray(im)).astype(np.float32)
        else:
            m = np.zeros((2,) + x.shape[1:], np.float32)
        return {"image": torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), "mask": torch.from_numpy(m),
                "quality": torch.tensor(float(q)), "label": torch.tensor(float(self.labels[i])),
                "source": torch.tensor(int(self.sources[i])), "patient": self.patients[i], "image_id": iid,
                "index": torch.tensor(i)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache_m/covidx_cxr4/homogeneo_224")
    ap.add_argument("--orig", default="data/masks/covidx_cxr4/orig")
    ap.add_argument("--out", default="data/masks/covidx_cxr4/homogeneo_224")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-space-check", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(align_masks_to_cache(a.cache, a.orig, a.out, a.workers, not a.no_space_check)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
