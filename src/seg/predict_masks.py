"""Pseudomáscaras del profesor (pulmones y corazón) con índice de calidad por imagen.

    python -m src.seg.predict_masks --teacher results/seg_teacher/states/best.pt --dataset covidx_cxr4 \
        --out data/masks/covidx_cxr4 --workers 12 --batch 64
    python -m src.seg.predict_masks ... --dataset otro --csv <manifiesto con image_id,image_path>

Formato compacto en <out>/shards/part_<n>.npz (comprimido): image_id, máscaras de 2 canales en el marco del
profesor (256 x 256) empaquetadas en bits, probabilidad media y geometría para volver a la rejilla original
(src/seg/teacher.py). <out>/index.csv tiene una fila por imagen con la calidad q (docs/03_arquitectura.md,
Section 2.5), las fracciones de área de pulmones y corazón (sobre la zona de imagen, sin relleno), el número de
componentes, la asimetría, la fracción en la mitad inferior y la geometría. Reanudable: las imágenes presentes en
fragmentos ya escritos no se recalculan. Antes de empezar estima el disco con el primer fragmento y lo compara
con el espacio libre. La carga y la transformación de imágenes van en `--workers` procesos (DataLoader); la red,
en el proceso principal (GPU si hay).

Contrato con la caché común y las líneas base (decisión del 12-11-2026): `export_orig` escribe además
<out>/orig/<stem>.png, un PNG de 8 bits y una banda por imagen en la rejilla de la imagen original con la
codificación 0 fondo, 100 solo corazón, 200 solo pulmón y 255 pulmón y corazón (pulmón = valor > 127, que es lo
que leen src/data/cache.py y src/baselines/trainer.py), y <out>/orig/index.csv con image_id, mask_path (relativa a
la raíz del proyecto, que es el directorio de trabajo de los trabajos; build_cache.py la abre tal cual), quality y
las fracciones de área. index.csv solo lista las imágenes con q > 0, para que P3 use el recorte central en las
demás; orig/index_todas.csv las lista todas. index.csv es el CSV que se pasa a scripts/build_cache.py --masks.

`PseudoMaskStore(out)` lee las máscaras: get(image_id) en el marco del profesor y original(image_id) en la rejilla de
la imagen.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.seg.teacher import (GEOM_KEYS, SegTeacher, from_teacher_frame, mask_quality, pack_masks,  # noqa: E402
                             to_teacher_frame, unpack_masks, valid_region)

INDEX_COLS = ["image_id", "shard", "row", "q", "lung_frac", "heart_frac", "n_comp", "n_comp_big", "conf", "asym",
              "lower_frac", "lung_rows", *GEOM_KEYS]


class _ImgDS(torch.utils.data.Dataset):
    def __init__(self, df, res, loader=None):
        self.df, self.res = df.reset_index(drop=True), res
        self.loader = loader

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        from src.data.paths import resolve_data
        from src.data.preprocess import load_luminance
        r = self.df.iloc[i]
        try:
            lum = self.loader(r) if self.loader else load_luminance(resolve_data(r["image_path"]))
            img, geom, _ = to_teacher_frame(lum, self.res)
            return {"image": torch.from_numpy(img)[None], "image_id": r["image_id"], "ok": True,
                    "geom": json.dumps(geom)}
        except Exception as e:
            return {"image": torch.zeros(1, self.res, self.res), "image_id": r["image_id"], "ok": False,
                    "geom": json.dumps({"error": str(e)[:200]})}


def _collate(items):
    """Apila las imágenes válidas del lote dentro del proceso lector. Con un tensor por imagen, cada lote comparte
    decenas de descriptores de fichero entre procesos y, con 12 lectores, se supera el límite de ficheros abiertos
    ("received 0 items of ancdata"); con un único tensor por lote el problema desaparece."""
    ok = [it for it in items if it["ok"]]
    return {"image": torch.stack([it["image"] for it in ok]) if ok else None,
            "ok": [{"image_id": it["image_id"], "geom": it["geom"]} for it in ok],
            "bad": [{"image_id": it["image_id"], "geom": it["geom"]} for it in items if not it["ok"]]}


def load_teacher(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = SegTeacher(cfg["encoder"], pretrained=False, out_ch=cfg["out_ch"], dec_width=cfg["dec_width"])
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), int(cfg.get("res", 256))


def done_ids(out: Path) -> set:
    s = set()
    for p in sorted((out / "shards").glob("part_*.npz")):
        with np.load(p, allow_pickle=False) as z:
            s.update(z["image_id"].tolist())
    return s


def predict_masks(model, res, df, out, device="cpu", batch=32, workers=4, shard=2048, loader=None, log=print,
                  check_space=True, margin=1.25):
    """Pseudomáscaras para df (image_id, image_path). Devuelve el resumen del índice."""
    out = Path(out)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    have = done_ids(out)
    todo = df[~df["image_id"].isin(have)].reset_index(drop=True)
    log("pseudomáscaras: %d imágenes, %d ya hechas, %d pendientes" % (len(df), len(have), len(todo)))
    existing = sorted((out / "shards").glob("part_*.npz"))
    k0 = (max(int(p.stem.split("_")[1]) for p in existing) + 1) if existing else 0
    dev = torch.device(device)
    model = model.to(dev).eval()
    for si, b0 in enumerate(range(0, len(todo), shard)):
        part = todo.iloc[b0:b0 + shard]
        dl = torch.utils.data.DataLoader(_ImgDS(part, res, loader), batch_size=batch, num_workers=workers,
                                         collate_fn=_collate)
        ids, packs, probs, meta = [], [], [], []
        t0 = time.time()
        for lote in dl:
            ok = lote["ok"]
            for it in lote["bad"]:
                log("error en %s: %s" % (it["image_id"], it["geom"]))
            if not ok:
                continue
            x = lote["image"].to(dev)
            with torch.no_grad():
                p = torch.sigmoid(model(x).float()).cpu().numpy()
            for it, pi in zip(ok, p):
                g = json.loads(it["geom"])
                v = valid_region(g)
                pi = pi * v[None]
                qd = mask_quality(pi, v)
                ids.append(it["image_id"]); packs.append(pack_masks(pi > 0.5))
                probs.append(np.round(pi.mean(axis=(1, 2)), 5))
                meta.append({**qd, **{k: g[k] for k in GEOM_KEYS}})
        name = out / "shards" / ("part_%06d.npz" % (k0 + si))
        tmp = out / "shards" / (".part_%06d.tmp.npz" % (k0 + si))
        np.savez_compressed(tmp, image_id=np.array(ids, dtype=str), masks=np.stack(packs) if packs else np.zeros((0, 1), np.uint8),
                            meta=np.array(json.dumps(meta)), channels=np.array(2), res=np.array(res))
        os.replace(tmp, name)
        if si == 0 and len(ids):
            per = name.stat().st_size / len(ids)
            est = {"bytes_per_image": per, "estimated_gb_total": per * len(df) / 1e9,
                   "free_gb": shutil.disk_usage(out).free / 1e9}
            log("estimación de disco: " + json.dumps(est))
            (out / "plan.json").write_text(json.dumps(est, indent=2))
            if check_space and per * (len(todo) - len(ids)) * margin > shutil.disk_usage(out).free:
                raise OSError("espacio insuficiente para las pseudomáscaras")
        log("fragmento %s: %d imágenes en %.1f s" % (name.name, len(ids), time.time() - t0))
    return write_index(out)


def write_index(out) -> dict:
    out = Path(out)
    rows = []
    for p in sorted((out / "shards").glob("part_*.npz")):
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            for j, (iid, m) in enumerate(zip(z["image_id"].tolist(), meta)):
                rows.append({"image_id": iid, "shard": p.name, "row": j, **m})
    d = pd.DataFrame(rows, columns=INDEX_COLS).drop_duplicates("image_id", keep="last")
    tmp = out / ".index.csv.tmp"
    d.to_csv(tmp, index=False)
    os.replace(tmp, out / "index.csv")
    return {"n": int(len(d)), "q0": int((d["q"] == 0).sum()) if len(d) else 0, "out": str(out)}


class PseudoMaskStore:
    """Lectura de las pseudomáscaras de un dataset."""

    def __init__(self, root):
        self.root = Path(root)
        self.index = pd.read_csv(self.root / "index.csv").set_index("image_id")
        self._cache = {}

    def __contains__(self, iid):
        return iid in self.index.index

    def _shard(self, name):
        if name not in self._cache:
            if len(self._cache) > 4:
                self._cache.clear()
            with np.load(self.root / "shards" / name, allow_pickle=False) as z:
                self._cache[name] = (z["masks"].copy(), int(z["channels"]), int(z["res"]))
        return self._cache[name]

    def row(self, iid) -> dict:
        return self.index.loc[iid].to_dict()

    def get(self, iid) -> np.ndarray:
        r = self.index.loc[iid]
        masks, c, res = self._shard(r["shard"])
        return unpack_masks(masks[int(r["row"])], c, res)

    def original(self, iid) -> np.ndarray:
        r = self.row(iid)
        return from_teacher_frame(self.get(iid), {k: r[k] for k in GEOM_KEYS})


from src.data.homogeneo_masks import decode_code, encode_code  # noqa: E402,F401  (codificación común)


def _export_one(args):
    root, iid, rel, geom = args
    from PIL import Image
    p = ROOT / rel
    if p.exists():
        return iid
    m = from_teacher_frame(geom.pop("_mask"), geom)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name("." + p.name + ".tmp.png")
    Image.fromarray(encode_code(m[0], m[1]), mode="L").save(tmp, optimize=False, compress_level=6)
    os.replace(tmp, p)
    return iid


def export_orig(out, workers: int = 8, log=print) -> dict:
    """Escribe <out>/orig/*.png (rejilla original, codificación común) y orig/index.csv (q > 0) e
    orig/index_todas.csv. Reanudable: no reescribe los PNG que ya existen."""
    out = Path(out)
    store = PseudoMaskStore(out)
    od = out / "orig"
    od.mkdir(parents=True, exist_ok=True)
    try:
        rel_root = od.resolve().relative_to(ROOT.resolve())
    except ValueError:
        rel_root = od.resolve()
    jobs, rows = [], []
    for iid, r in store.index.iterrows():
        rel = str(rel_root / (os.path.splitext(str(iid))[0] + ".png"))
        rows.append({"image_id": iid, "mask_path": rel, "quality": float(r["q"]), "lung_frac": float(r["lung_frac"]),
                     "heart_frac": float(r["heart_frac"])})
        if not (ROOT / rel).exists():
            g = {k: r[k] for k in GEOM_KEYS}
            g["_mask"] = store.get(iid)
            jobs.append((str(out), iid, rel, g))
    log("exportación a la rejilla original: %d pendientes de %d" % (len(jobs), len(rows)))
    if workers > 1 and jobs:
        from multiprocessing import get_context
        with get_context("spawn").Pool(workers) as pool:
            for k, _ in enumerate(pool.imap_unordered(_export_one, jobs, chunksize=16)):
                if (k + 1) % 5000 == 0:
                    log("exportadas %d/%d" % (k + 1, len(jobs)))
    else:
        for j in jobs:
            _export_one(j)
    d = pd.DataFrame(rows)
    d.to_csv(od / "index_todas.csv", index=False)
    d[d["quality"] > 0].to_csv(od / "index.csv", index=False)
    return {"n": int(len(d)), "n_q_pos": int((d["quality"] > 0).sum()), "dir": str(od)}


def _frame_for(dataset, csv):
    if csv:
        d = pd.read_csv(csv)
        if "image_id" not in d.columns:
            d["image_id"] = d["image_path"].str.split("/").str[-1]
        return d[["image_id", "image_path"]]
    if dataset == "covidx_cxr4":
        from src.data.manifest import load_manifest
        return load_manifest()[["image_id", "image_path"]]
    raise SystemExit("dataset %s sin --csv" % dataset)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--dataset", default="covidx_cxr4")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", type=int, default=2048)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-export-orig", action="store_true")
    a = ap.parse_args(argv)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, res = load_teacher(a.teacher, dev)
    df = _frame_for(a.dataset, a.csv)
    if a.limit:
        df = df.head(a.limit)
    out = Path(a.out or (ROOT / "data" / "masks" / a.dataset))
    print(json.dumps(predict_masks(model, res, df, out, dev, a.batch, a.workers, a.shard)))
    if not a.no_export_orig:
        print(json.dumps(export_orig(out, a.workers)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
