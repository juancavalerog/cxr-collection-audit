"""Pruebas del profesor de segmentación, las pseudomáscaras, la auditoría y el preprocesado con máscaras
(src/seg/, src/data/homogeneo_masks.py) en CPU con datos sintéticos. Ejecutar: pytest -q tests/test_seg.py
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.seg import chexmask as CM  # noqa: E402
from src.seg import teacher as T  # noqa: E402


# ------------------------------------------------------------------------------------- RLE de CheXmask
def test_rle_decode_hand_computed():
    # máscara 3 x 4 aplanada por filas; "2 3" marca los índices 1, 2, 3 (inicio desde 1) y "9 2" los 8 y 9
    m = CM.rle_decode("2 3 9 2", 3, 4)
    exp = np.array([[0, 1, 1, 1], [0, 0, 0, 0], [1, 1, 0, 0]], dtype=np.uint8)
    assert np.array_equal(m, exp)
    assert CM.rle_encode(exp) == "2 3 9 2"
    assert CM.rle_decode("", 2, 2).sum() == 0 and CM.rle_decode(float("nan"), 2, 2).sum() == 0
    with pytest.raises(ValueError):
        CM.rle_decode("1 2 3", 2, 2)
    with pytest.raises(ValueError):
        CM.rle_decode("4 5", 2, 2)
    r = np.random.default_rng(0).random((17, 23)) > 0.6
    assert np.array_equal(CM.rle_decode(CM.rle_encode(r), 17, 23), r.astype(np.uint8))


def test_keys_and_match_manifest(tmp_path):
    assert CM.nih_key("nih/images/00000001_000.png") == "00000001_000.png"
    assert CM.chexpert_key("CheXpert/CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg") == \
        "train/patient00001/study1/view1_frontal.jpg"
    assert CM.chexpert_key("CheXpert-v1.0/valid/patient64541/study1/view1_frontal.jpg").startswith("valid/")
    lung = np.zeros((8, 6), np.uint8); lung[2:6, 1:3] = 1
    rows = []
    for i in range(6):
        rows.append({"Path": "train/patient%05d/study1/view1_frontal.jpg" % (i // 2), "Dice RCA (Mean)": 0.9 if i != 1 else 0.5,
                     "Dice RCA (Max)": 0.95, "Landmarks": "", "Left Lung": CM.rle_encode(lung), "Right Lung": "",
                     "Heart": "", "Height": 8, "Width": 6})
    rows[0]["Path"] = "train/patient00000/study1/view1_frontal.jpg"
    csv = tmp_path / "CheXpert.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    man = pd.DataFrame({"image_path": ["CheXpert/CheXpert-v1.0-small/" + r["Path"] for r in rows],
                        "patient_id": ["chx_%d" % (i // 2) for i in range(6)], "view": "Frontal"}).drop_duplicates("image_path")
    sel = CM.match_manifest(csv, "chexpert_small", man, min_dice=0.7, chunksize=2)
    assert len(sel) == 3 and sel["patient_id"].is_unique and (sel["dice_rca_mean"] >= 0.7).all()
    m = CM.decode_row(sel.iloc[0], (16, 12))
    assert m.shape == (2, 16, 12) and m[0].sum() == 4 * 4 * 2 and m[1].sum() == 0
    with pytest.raises(ValueError):
        CM.decode_row(sel.iloc[0], (10, 12))


# ------------------------------------------------------------------------------------- geometría y calidad
def _ellipses(h, w):
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    l = (((xx + 0.4) / 0.25) ** 2 + (yy / 0.55) ** 2 <= 1) | (((xx - 0.4) / 0.25) ** 2 + (yy / 0.55) ** 2 <= 1)
    hh = ((xx / 0.12) ** 2 + ((yy - 0.25) / 0.25) ** 2 <= 1) & ~l
    return l, hh


def test_teacher_frame_roundtrip_and_quality():
    l, h = _ellipses(90, 140)
    img = np.zeros((110, 160)); img[10:100, 10:150] = 0.2 + 0.6 * l + 0.3 * h + 0.05 * np.random.default_rng(0).random((90, 140))
    ml = np.zeros_like(img, np.uint8); ml[10:100, 10:150] = l
    mh = np.zeros_like(img, np.uint8); mh[10:100, 10:150] = h
    x, g, mt = T.to_teacher_frame(img, 64, np.stack([ml, mh]))
    assert x.shape == (64, 64) and mt.shape == (2, 64, 64)
    back = T.from_teacher_frame(mt, g)
    dice = 2 * (back[0] & ml).sum() / (back[0].sum() + ml.sum())
    assert back.shape == (2, 110, 160) and dice > 0.93
    v = T.valid_region(g)
    q2 = T.mask_quality(np.stack([mt[0] * 0.95, mt[1] * 0.95]).astype(float), v)
    assert q2["q"] > 0.9 and q2["n_comp_big"] == 2
    one = np.zeros((2, 64, 64)); one[0, 10:50, 10:50] = 0.9
    assert T.mask_quality(one, v)["q"] == 0.0


# ------------------------------------------------------------------------------------- profesor
def _train(args):
    from src.seg.train_teacher import main
    return main(["train"] + args)


def test_teacher_smoke_dice_improves_and_resume(tmp_path):
    ts = T.write_synthetic_teacher_set(tmp_path / "ts", n=40, res=64)
    torch.manual_seed(0)
    va = T.TeacherSet(ts, "val")
    x = torch.stack([va[i]["image"] for i in range(len(va))]); m = torch.stack([va[i]["mask"] for i in range(len(va))])
    m0 = T.SegTeacher("resnet18", pretrained=False)
    d0 = float(T.dice_per_channel(m0.eval()(x), m).mean())
    common = ["--set", str(ts), "--encoder", "resnet18", "--no-pretrained", "--batch", "8", "--workers", "0",
              "--lr", "3e-3", "--epochs", "4"]
    assert _train(common + ["--out", str(tmp_path / "cut"), "--stop-after-steps", "5"]) == 3
    ck = torch.load(tmp_path / "cut" / "checkpoint.pt", weights_only=False)
    assert ck["state"]["global_step"] == 5 and ck["reason"] == "parada"
    assert _train(common + ["--out", str(tmp_path / "cut")]) == 0                      # reanuda y termina
    s = json.loads((tmp_path / "cut" / "summary.json").read_text())
    last = s["history"][-1]
    assert len(s["history"]) == 4 and last["dice_pulmones"] > max(0.85, d0 + 0.2)
    assert {"dice_pulmones", "dice_corazon"} <= set(last)
    assert (tmp_path / "cut" / "states" / "best.pt").exists()


def test_teacher_sigterm_exit75(tmp_path):
    ts = T.write_synthetic_teacher_set(tmp_path / "ts", n=40, res=64)
    out = tmp_path / "sig"
    env = dict(os.environ); env.pop("PAPER_NEW_FAKE_SMI", None)
    p = subprocess.Popen([sys.executable, "-m", "src.seg.train_teacher", "train", "--set", str(ts), "--out", str(out),
                          "--encoder", "resnet18", "--no-pretrained", "--batch", "4", "--workers", "0",
                          "--epochs", "500"], cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 90 and not ((out / "train_log.jsonl").exists() and
                                         sum(1 for _ in open(out / "train_log.jsonl")) >= 2):
        time.sleep(0.2)
    time.sleep(1.0)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=90) == 75, p.stdout.read().decode()[-1500:]
    assert torch.load(out / "checkpoint.pt", weights_only=False)["reason"] == "parada"


def test_thermal_teacher_exit75(tmp_path, monkeypatch):
    smi = tmp_path / "smi.txt"; smi.write_text("0, 85\n")
    monkeypatch.setenv("PAPER_NEW_FAKE_SMI", str(smi))
    ts = T.write_synthetic_teacher_set(tmp_path / "ts", n=40, res=64)
    rc = _train(["--set", str(ts), "--out", str(tmp_path / "th"), "--encoder", "resnet18", "--no-pretrained",
                 "--batch", "8", "--workers", "0", "--epochs", "2", "--thermal-every", "2"])
    assert rc == 75


# ------------------------------------------------------------------------------------- pseudomáscaras y caché
class BrightTeacher(torch.nn.Module):
    """Profesor de prueba: pulmón donde la imagen es brillante (> 0.6); corazón en la franja media gris."""

    def forward(self, x):
        lung = (x > 0.62).float()
        heart = ((x > 0.4) & (x <= 0.62)).float()
        return (torch.cat([lung, heart], 1) * 2 - 1) * 10


def _write_images(d, n=6, h=120, w=150, small_lungs=()):
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        l, hh = _ellipses(h - 20, w - 20)
        if i in small_lungs:
            l = l & (np.linspace(-1, 1, h - 20)[:, None] < 0.1)
        img = np.zeros((h, w)); img[10:-10, 10:-10] = 0.15 + 0.7 * l + 0.35 * hh
        img += 0.05 * np.random.default_rng(i).random((h, w))
        p = d / ("img_%02d.png" % i)
        Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8)).save(p)
        rows.append({"image_id": p.name, "image_path": str(p)})
    return pd.DataFrame(rows)


def test_predict_masks_store_export_and_resume(tmp_path):
    from src.seg.predict_masks import PseudoMaskStore, export_orig, predict_masks
    df = _write_images(tmp_path / "img")
    out = tmp_path / "masks"
    r = predict_masks(BrightTeacher(), 64, df, out, batch=4, workers=0, shard=4, log=lambda *a: None)
    assert r["n"] == 6 and len(list((out / "shards").glob("part_*.npz"))) == 2
    assert predict_masks(BrightTeacher(), 64, df, out, workers=0, log=lambda *a: None)["n"] == 6   # nada pendiente
    st = PseudoMaskStore(out)
    assert st.get("img_00.png").shape == (2, 64, 64)
    orig = st.original("img_00.png")
    assert orig.shape == (2, 120, 150)
    assert (st.index["q"] > 0).all()
    e = export_orig(out, workers=1, log=lambda *a: None)
    idx = pd.read_csv(out / "orig" / "index.csv")
    assert e["n_q_pos"] == 6 and set(idx.columns) >= {"image_id", "mask_path", "quality"}
    code = np.asarray(Image.open(ROOT / idx["mask_path"].iloc[0]))
    assert set(np.unique(code)) <= {0, 100, 200, 255}
    assert np.array_equal(code > 127, orig[0].astype(bool))


def test_homogeneo_cache_alignment_and_baseline_maskstore(tmp_path):
    from src.baselines.trainer import MaskStore
    from src.data.cache import build_cache, cache_dir
    from src.data.homogeneo_masks import CXRMaskDataset, align_masks_to_cache, decode_code
    from src.seg.predict_masks import export_orig, predict_masks
    df = _write_images(tmp_path / "img", n=4)
    out = tmp_path / "masks"
    predict_masks(BrightTeacher(), 64, df, out, workers=0, log=lambda *a: None)
    export_orig(out, workers=1, log=lambda *a: None)
    mi = pd.read_csv(out / "orig" / "index.csv")
    masks = {a: str(ROOT / b) for a, b in zip(mi["image_id"], mi["mask_path"])}
    build_cache(df, "sint", "homogeneo", 64, root=tmp_path / "cache_m", workers=1, masks=masks, log=lambda *a: None)
    cd = cache_dir("sint", "homogeneo", 64, tmp_path / "cache_m")
    ci = pd.read_csv(cd / "index.csv")
    assert (ci["crop_mode"] == "mascara").all()
    r = align_masks_to_cache(cd, out / "orig", tmp_path / "aligned", workers=1, log=lambda *a: None)
    assert r["ok"] == 4 and r["errores"] == 0
    ai = pd.read_csv(tmp_path / "aligned" / "index.csv")
    assert set(ai.columns) >= {"image_id", "path", "quality", "lung_frac", "heart_frac"}
    ms = MaskStore(tmp_path / "aligned")                                   # lector de las líneas base
    for iid, rel in zip(ai["image_id"], ai["path"]):
        img = np.asarray(Image.open(cd / ci.set_index("image_id").loc[iid, "rel_path"]), dtype=np.float32) / 255
        code = np.asarray(Image.open(tmp_path / "aligned" / rel))
        ours = decode_code(code)
        base = ms.get(iid, (64, 64)).numpy()
        assert np.array_equal(base > 0.5, ours[0].astype(bool))
        bright = img > 0.6
        dice = 2 * (bright & (ours[0] > 0)).sum() / (bright.sum() + ours[0].sum())
        assert dice > 0.85, dice                                           # la máscara cae sobre los pulmones
    man = df.assign(label=[0, 1, 0, 1], source_idx=1, patient_id=["a", "b", "c", "d"])
    ds = CXRMaskDataset(man, cd, tmp_path / "aligned")
    it = ds[0]
    assert it["image"].shape == (3, 64, 64) and it["mask"].shape == (2, 64, 64) and float(it["quality"]) > 0


# ------------------------------------------------------------------------------------- auditoría
def test_audit_detects_undersegmentation(tmp_path):
    from src.seg.audit_masks import audit
    from src.seg.predict_masks import PseudoMaskStore, predict_masks
    df = _write_images(tmp_path / "img", n=12, small_lungs=(1, 3, 5, 7, 9, 11))
    out = tmp_path / "masks"
    predict_masks(BrightTeacher(), 64, df, out, workers=0, log=lambda *a: None)
    frame = df.assign(label=[i % 2 for i in range(12)], source="bimcv", patient_id=["p%d" % i for i in range(12)])
    # referencia tipo COVID-QU-Ex: máscara pulmonar completa y una infección en la base pulmonar
    ref = tmp_path / "quex" / "Infection Segmentation Data" / "Train" / "COVID-19"
    for sub in ("images", "lung masks", "infection masks"):
        (ref / sub).mkdir(parents=True)
    for i in (1, 3, 5):
        l, _ = _ellipses(100, 130)
        full = np.zeros((120, 150), np.uint8); full[10:-10, 10:-10] = l
        inf = np.zeros_like(full); inf[80:100, 20:60] = 1
        inf &= full
        name = "img_%02d.png" % i
        Image.fromarray(full * 255).resize((256, 256), Image.NEAREST).save(ref / "lung masks" / name)
        Image.fromarray(inf * 255).resize((256, 256), Image.NEAREST).save(ref / "infection masks" / name)
        Image.fromarray(full * 255).save(ref / "images" / name)
    s = audit(PseudoMaskStore(out), frame, tmp_path / "audit", qu_ex=tmp_path / "quex", mixed=("bimcv",), n_boot=100,
              visual_per_group=2)
    txt = " ".join(s["banderas"])
    assert "infrasegmentación" in txt and "cobertura" in txt
    assert s["referencia"]["n_comparadas"] == 3
    assert (tmp_path / "audit" / "flags.md").exists() and (tmp_path / "audit" / "area_por_fuente_clase.csv").exists()
    assert len(list((tmp_path / "audit" / "visual").rglob("*.png"))) == 4
    # sin infrasegmentación no hay bandera de áreas
    df2 = _write_images(tmp_path / "img2", n=8)
    out2 = tmp_path / "masks2"
    predict_masks(BrightTeacher(), 64, df2, out2, workers=0, log=lambda *a: None)
    fr2 = df2.assign(label=[i % 2 for i in range(8)], source="bimcv", patient_id=["q%d" % i for i in range(8)])
    s2 = audit(PseudoMaskStore(out2), fr2, tmp_path / "audit2", mixed=("bimcv",), n_boot=100, visual_per_group=0)
    assert not any("infrasegmentación" in f for f in s2["banderas"])
