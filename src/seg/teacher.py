"""Profesor de segmentación de dos canales (pulmones y corazón) para CSD-Net (docs/03_arquitectura.md,
Section 2.5).

- Marco del profesor: P1 (luminancia) y P2 (recorte de bordes) de src/data/preprocess.py, relleno centrado a
  cuadrado, reducción con antialias a R_t (256 por defecto) y P5 (percentiles). Así se ve igual una imagen de
  CheXmask (NIH, CheXpert) y una de COVIDx, sin el recorte anatómico P3, que necesita precisamente la máscara.
  `to_teacher_frame` devuelve la imagen y la geometría; `from_teacher_frame` lleva una máscara de vuelta a la
  rejilla original.
- `SegTeacher`: U-Net con codificador de timm (features_only, in_chans=1; resnet34 por defecto) y decodificador
  con GroupNorm; salida de 2 logits por píxel a resolución completa [ronneberger2015unet].
- Conjunto del profesor: fragmentos .npy sin comprimir (imágenes uint8 R_t x R_t y máscaras empaquetadas en bits)
  con mmap, partición por paciente (90/10 por hash del paciente), construidos por `build_teacher_set` desde las
  filas de CheXmask emparejadas (src/seg/chexmask.py). Multiproceso y reanudable por fragmentos.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.data.preprocess import load_luminance, percentile_rescale, trim_borders

TEACHER_RES = 256
GEOM_KEYS = ("orig_h", "orig_w", "trim_t", "trim_b", "trim_l", "trim_r", "side", "pad_t", "pad_l", "res")


# ====================================================================== geometría
def to_teacher_frame(lum: np.ndarray, res: int = TEACHER_RES, masks: np.ndarray | None = None):
    """lum: luminancia (H, W) en [0, 1]. Devuelve (imagen float32 res x res en [0, 1], geometría, máscaras
    transformadas (C, res, res) uint8 o None)."""
    h0, w0 = lum.shape
    xt, (t, b, l, r) = trim_borders(lum)
    h, w = xt.shape
    side = max(h, w)
    pt, pl = (side - h) // 2, (side - w) // 2
    sq = np.zeros((side, side), dtype=np.float32)
    sq[pt:pt + h, pl:pl + w] = xt
    img = np.asarray(Image.fromarray(sq, mode="F").resize((res, res), Image.BICUBIC), dtype=np.float32)
    valid = np.zeros((side, side), dtype=np.float32)
    valid[pt:pt + h, pl:pl + w] = 1
    vr = np.asarray(Image.fromarray(valid, mode="F").resize((res, res), Image.BILINEAR)) > 0.5
    img = percentile_rescale(img) if vr.sum() < 16 else _rescale_on(img, vr)
    geom = {"orig_h": h0, "orig_w": w0, "trim_t": t, "trim_b": b, "trim_l": l, "trim_r": r, "side": side,
            "pad_t": pt, "pad_l": pl, "res": res}
    mt = None
    if masks is not None:
        mt = []
        for c in masks:
            c = np.asarray(c, dtype=np.float32)[t:b, l:r]
            s2 = np.zeros((side, side), dtype=np.float32)
            s2[pt:pt + h, pl:pl + w] = c
            mt.append(np.asarray(Image.fromarray(s2, mode="F").resize((res, res), Image.BILINEAR)) > 0.5)
        mt = np.stack(mt).astype(np.uint8)
    return img, geom, mt


def _rescale_on(img, valid):
    a, b = np.percentile(img[valid], [0.5, 99.5])
    if b - a < 1e-8:
        return np.zeros_like(img)
    out = np.clip((img - a) / (b - a), 0, 1).astype(np.float32)
    out[~valid] = 0.0
    return out


def from_teacher_frame(mask: np.ndarray, geom: dict) -> np.ndarray:
    """Máscaras (C, res, res) en el marco del profesor a la rejilla original (C, orig_h, orig_w), uint8."""
    side, res = int(geom["side"]), int(geom["res"])
    t, b, l, r = (int(geom[k]) for k in ("trim_t", "trim_b", "trim_l", "trim_r"))
    pt, pl = int(geom["pad_t"]), int(geom["pad_l"])
    h, w = b - t, r - l
    out = np.zeros((mask.shape[0], int(geom["orig_h"]), int(geom["orig_w"])), dtype=np.uint8)
    for i, c in enumerate(mask):
        up = np.asarray(Image.fromarray(np.asarray(c, dtype=np.float32), mode="F").resize((side, side), Image.BILINEAR)) > 0.5
        out[i, t:b, l:r] = up[pt:pt + h, pl:pl + w]
    return out


def valid_region(geom: dict) -> np.ndarray:
    """Zona del marco del profesor que corresponde a imagen (no relleno), res x res bool."""
    side, res = int(geom["side"]), int(geom["res"])
    h, w = int(geom["trim_b"]) - int(geom["trim_t"]), int(geom["trim_r"]) - int(geom["trim_l"])
    v = np.zeros((side, side), dtype=np.float32)
    v[int(geom["pad_t"]):int(geom["pad_t"]) + h, int(geom["pad_l"]):int(geom["pad_l"]) + w] = 1
    return np.asarray(Image.fromarray(v, mode="F").resize((res, res), Image.BILINEAR)) > 0.5


def pack_masks(m: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(m, dtype=np.uint8).ravel())


def unpack_masks(p: np.ndarray, channels: int, res: int) -> np.ndarray:
    return np.unpackbits(np.asarray(p, dtype=np.uint8))[: channels * res * res].reshape(channels, res, res)


# ====================================================================== red
class _Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        g = lambda c: nn.GroupNorm(min(8, c), c)  # noqa: E731
        self.conv = nn.Sequential(nn.Conv2d(cin + cskip, cout, 3, padding=1), g(cout), nn.GELU(),
                                  nn.Conv2d(cout, cout, 3, padding=1), g(cout), nn.GELU())

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class SegTeacher(nn.Module):
    """U-Net con codificador de timm. Entrada (B, 1, R, R) en [0, 1]; salida (B, 2, R, R) logits."""

    def __init__(self, encoder: str = "resnet34", pretrained: bool = True, out_ch: int = 2, dec_width: int = 64):
        super().__init__()
        import timm
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True, in_chans=1)
        chs = self.encoder.feature_info.channels()
        self.cfg = {"encoder": encoder, "out_ch": out_ch, "dec_width": dec_width}
        self.ups = nn.ModuleList()
        c = chs[-1]
        for skip in reversed(chs[:-1]):
            self.ups.append(_Up(c, skip, dec_width))
            c = dec_width
        self.head = nn.Conv2d(dec_width, out_ch, 1)

    def forward(self, x):
        x = (x - 0.5) / 0.25
        feats = self.encoder(x)
        y = feats[-1]
        for up, skip in zip(self.ups, reversed(feats[:-1])):
            y = up(y, skip)
        return F.interpolate(self.head(y), size=x.shape[-2:], mode="bilinear", align_corners=False)


def dice_per_channel(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Dice binario por imagen y canal (umbral 0.5), forma (B, C)."""
    p = (torch.sigmoid(logits.float()) > 0.5).float()
    t = (target > 0.5).float()
    inter = (p * t).flatten(2).sum(-1)
    return (2 * inter + eps) / (p.flatten(2).sum(-1) + t.flatten(2).sum(-1) + eps)


def seg_loss(logits, target):
    """BCE + (1 - Dice suave) promediados por canal."""
    t = target.float()
    bce = F.binary_cross_entropy_with_logits(logits.float(), t)
    p = torch.sigmoid(logits.float())
    inter = (p * t).flatten(2).sum(-1)
    dice = (2 * inter + 1) / (p.flatten(2).sum(-1) + t.flatten(2).sum(-1) + 1)
    return bce + (1 - dice).mean()


# ====================================================================== calidad de la pseudomáscara
def mask_quality(prob: np.ndarray, valid: np.ndarray | None = None, lo: float = 0.10, hi: float = 0.60) -> dict:
    """Calidad q de docs/03_arquitectura.md, Section 2.5: confianza media del profesor dentro de la máscara
    pulmonar si hay dos componentes conexas con fracción de área entre lo y hi, y 0 en otro caso. `prob`:
    (2, R, R) probabilidades; `valid`: zona de imagen (sin relleno)."""
    from scipy import ndimage
    lung = prob[0] > 0.5
    area_ref = float(valid.sum()) if valid is not None else float(lung.size)
    frac = float(lung.sum()) / max(area_ref, 1.0)
    lab, n = ndimage.label(lung)
    sizes = np.bincount(lab.ravel())[1:] if n else np.zeros(0)
    big = int((sizes >= 0.01 * area_ref).sum())
    conf = float(np.where(lung, np.maximum(prob[0], 1 - prob[0]), 0).sum() / max(lung.sum(), 1))
    q = conf if (big == 2 and lo <= frac <= hi) else 0.0
    heart_frac = float((prob[1] > 0.5).sum()) / max(area_ref, 1.0) if prob.shape[0] > 1 else float("nan")
    asym = float(np.sort(sizes)[-2] / np.sort(sizes)[-1]) if len(sizes) >= 2 else 0.0
    rows = np.nonzero(lung.any(1))[0]
    lower = float(lung[lung.shape[0] // 2:].sum() / max(lung.sum(), 1))
    return {"q": q, "lung_frac": frac, "heart_frac": heart_frac, "n_comp": int(n), "n_comp_big": big,
            "conf": conf, "asym": asym, "lower_frac": lower, "lung_rows": int(len(rows))}


# ====================================================================== conjunto del profesor
def patient_split(pid: str, val_frac: float = 0.1) -> str:
    h = int(hashlib.md5(str(pid).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_frac else "train"


def _teacher_item(args):
    from src.data.paths import resolve_data
    from src.seg.chexmask import decode_row
    row, res = args
    try:
        lum = load_luminance(resolve_data(row["image_path"]))
        m = decode_row(row, lum.shape)
        img, geom, mt = to_teacher_frame(lum, res, m)
        return {"ok": True, "key": row["key"], "patient_id": row["patient_id"], "dataset": row["dataset"],
                "img": np.rint(img * 255).astype(np.uint8), "mask": pack_masks(mt), "geom": geom}
    except Exception as e:  # se registra y se sigue
        return {"ok": False, "key": row.get("key"), "error": str(e)[:200]}


def build_teacher_set(rows, out_dir, res: int = TEACHER_RES, workers: int = 8, shard: int = 1024,
                      val_frac: float = 0.1, log=print, check_space: bool = True):
    """Construye (o completa) el conjunto del profesor en out_dir/shards a partir de las filas de
    match_manifest (DataFrame). Reanudable: los fragmentos ya escritos no se recalculan."""
    from multiprocessing import get_context
    out = Path(out_dir)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    recs = rows.to_dict("records")
    blocks = [recs[i:i + shard] for i in range(0, len(recs), shard)]
    per = res * res + (2 * res * res) // 8 + 200
    need = per * len(recs)
    free = shutil.disk_usage(out).free
    plan = {"n": len(recs), "res": res, "bytes_estimated": need, "gb_estimated": need / 1e9, "free_gb": free / 1e9}
    log("plan del conjunto del profesor: " + json.dumps(plan))
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    if check_space and need * 1.25 > free:
        raise OSError("espacio insuficiente para el conjunto del profesor")
    pool = get_context("spawn").Pool(workers) if workers > 1 else None
    try:
        for bi, blk in enumerate(blocks):
            stem = out / "shards" / ("part_%05d" % bi)
            if Path(str(stem) + "_meta.json").exists():
                continue
            it = pool.imap(_teacher_item, [(r, res) for r in blk], chunksize=8) if pool else map(_teacher_item, [(r, res) for r in blk])
            res_ = [r for r in it]
            good = [r for r in res_ if r["ok"]]
            imgs = np.stack([r["img"] for r in good]) if good else np.zeros((0, res, res), np.uint8)
            msk = np.stack([r["mask"] for r in good]) if good else np.zeros((0, (2 * res * res) // 8), np.uint8)
            np.save(str(stem) + "_img.npy", imgs)
            np.save(str(stem) + "_mask.npy", msk)
            meta = [{"key": r["key"], "patient_id": r["patient_id"], "dataset": r["dataset"],
                     "split": patient_split(r["patient_id"], val_frac), **r["geom"]} for r in good]
            tmp = Path(str(stem) + "_meta.json.tmp")
            tmp.write_text(json.dumps({"rows": meta, "errors": [r for r in res_ if not r["ok"]]}))
            os.replace(tmp, Path(str(stem) + "_meta.json"))
            log("conjunto del profesor: fragmento %d/%d (%d válidas)" % (bi + 1, len(blocks), len(good)))
    finally:
        if pool:
            pool.close(); pool.join()
    return out


class TeacherSet(torch.utils.data.Dataset):
    """Lee los fragmentos de build_teacher_set con mmap. split: 'train' o 'val' (por paciente)."""

    def __init__(self, root, split: str = "train", channels: int = 2):
        self.root = Path(root)
        self.items = []
        self.arrays = {}
        self.channels = channels
        for mp in sorted((self.root / "shards").glob("part_*_meta.json")):
            stem = str(mp)[: -len("_meta.json")]
            meta = json.loads(mp.read_text())["rows"]
            for j, r in enumerate(meta):
                if r["split"] == split:
                    self.items.append((stem, j, r))
        self.res = int(self.items[0][2]["res"]) if self.items else TEACHER_RES

    def _arr(self, stem):
        if stem not in self.arrays:
            self.arrays[stem] = (np.load(stem + "_img.npy", mmap_mode="r"), np.load(stem + "_mask.npy", mmap_mode="r"))
        return self.arrays[stem]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        stem, j, r = self.items[i]
        im, mk = self._arr(stem)
        x = torch.from_numpy(np.asarray(im[j], dtype=np.float32) / 255.0)[None]
        m = torch.from_numpy(unpack_masks(mk[j], self.channels, self.res).astype(np.float32))
        return {"image": x, "mask": m, "key": r["key"], "patient": r["patient_id"]}


def write_synthetic_teacher_set(out_dir, n: int = 40, res: int = 64, seed: int = 0, val_frac: float = 0.25):
    """Conjunto del profesor sintético (pruebas): elipses pulmonares más brillantes que el fondo."""
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    imgs, msks, meta = [], [], []
    yy, xx = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res), indexing="ij")
    for i in range(n):
        jx, jy = rng.normal(0, 0.05, 2)
        lungs = (((xx + 0.4 + jx) / 0.25) ** 2 + ((yy - jy) / 0.5) ** 2 <= 1) | (((xx - 0.4 + jx) / 0.25) ** 2 + ((yy - jy) / 0.5) ** 2 <= 1)
        heart = (((xx + jx) / 0.12) ** 2 + ((yy - 0.25 - jy) / 0.25) ** 2 <= 1) & ~lungs
        x = 0.2 + 0.5 * lungs + 0.25 * heart + 0.05 * rng.standard_normal((res, res))
        imgs.append(np.clip(np.rint(x * 255), 0, 255).astype(np.uint8))
        msks.append(pack_masks(np.stack([lungs, heart]).astype(np.uint8)))
        pid = "p%d" % i
        meta.append({"key": "k%d" % i, "patient_id": pid, "dataset": "sint",
                     "split": "val" if i < int(n * val_frac) else "train", "orig_h": res, "orig_w": res, "trim_t": 0,
                     "trim_b": res, "trim_l": 0, "trim_r": res, "side": res, "pad_t": 0, "pad_l": 0, "res": res})
    np.save(str(out / "shards" / "part_00000_img.npy"), np.stack(imgs))
    np.save(str(out / "shards" / "part_00000_mask.npy"), np.stack(msks))
    (out / "shards" / "part_00000_meta.json").write_text(json.dumps({"rows": meta, "errors": []}))
    return out
