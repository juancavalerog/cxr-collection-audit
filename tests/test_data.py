"""Pruebas del canal de datos (src/data). Ejecutar desde la raíz de paper_new: pytest -q tests/test_data.py

Las que usan datos reales (manifiesto, imágenes de COVIDx) se saltan si no están accesibles.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import cache as C  # noqa: E402
from src.data import manifest as M  # noqa: E402
from src.data import preprocess as P  # noqa: E402
from src.data.datasets import (BalancedInfiniteBatchSampler, CXRDataset, NewAugment, Paper1Augment,  # noqa: E402
                               cutout_batch, make_generator)
from src.data.paths import data_root  # noqa: E402

HAS_MANIFEST = M.DEFAULT_CSV.exists()
HAS_IMAGES = (data_root() / "data" / "dataset").is_dir()
needs_manifest = pytest.mark.skipif(not HAS_MANIFEST, reason="sin manifiesto")
needs_images = pytest.mark.skipif(not HAS_IMAGES, reason="sin imágenes de COVIDx")


@pytest.fixture(scope="module")
def man():
    return M.load_manifest()


# ------------------------------------------------------------------ manifiestos (datos reales)
@needs_manifest
def test_counts_oficial(man):
    assert M.counts("oficial", man) == {"train": 67863, "val": 8473, "test": 8482}


@needs_manifest
def test_counts_limpia(man):
    assert M.counts("limpia", man) == {"train": 67932, "val": 7664, "test": 8479}
    assert (M.get_splits("limpia", man)["train"]["split_clean"] != "excluded").all()


@needs_manifest
def test_val_robust_and_subsets(man):
    vr = M.val_robust(man)
    assert len(vr) == 3407
    assert set(vr["source"]) == {"bimcv"}
    assert vr["label"].value_counts().to_dict() == {1: 3150, 0: 257}
    ts = {k: len(v) for k, v in M.test_subsets(man).items()}
    assert ts == {"bimcv": 8082, "ricord": 200, "rsna_con_fuga": 129, "rsna_sin_fuga": 71}


@needs_manifest
def test_regimes_and_source_index(man):
    s = M.get_splits("oficial", man)
    assert s["train"]["label"].value_counts().to_dict() == {1: 57199, 0: 10664}
    c = M.get_splits("oficial", man, ["bimcv"])["train"]
    assert c["label"].value_counts().to_dict() == {1: 42495, 0: 647}
    assert len(s["val"]) == 8473 and len(s["test"]) == 8482          # val y test completos en el controlado
    assert M.SOURCE_INDEX == {"actmed": 0, "bimcv": 1, "cohen": 2, "fig1": 3, "ricord": 4, "rsna": 5, "sirm": 6,
                              "stonybrook": 7}
    for d in s.values():
        assert (d["source"].map(M.SOURCE_INDEX) == d["source_idx"]).all()


@needs_manifest
@needs_images
def test_test_order_matches_txt(man):
    t = M.get_splits("oficial", man)["test"]
    o = M.order_like_txt(t)
    assert (o["file"].values == t["file"].values).all()


# ------------------------------------------------------------------ preprocesado
def _png(path, arr, mode=None):
    Image.fromarray(arr, mode=mode).save(path)
    return path


def test_decode_like_tf_modes(tmp_path):
    rng = np.random.default_rng(0)
    g = rng.integers(0, 256, (20, 30), dtype=np.uint8)
    out = P.decode_like_tf(_png(tmp_path / "l.png", g))
    assert out.shape == (20, 30, 3) and (out == g[..., None]).all()
    rgba = rng.integers(0, 256, (20, 30, 4), dtype=np.uint8)
    out = P.decode_like_tf(_png(tmp_path / "rgba.png", rgba))
    assert (out == rgba[..., :3]).all()                                   # alfa descartado, sin componer
    g16 = rng.integers(0, 65536, (20, 30), dtype=np.uint16)
    Image.fromarray(g16).save(tmp_path / "g16.png")
    out = P.decode_like_tf(tmp_path / "g16.png")
    assert (out[..., 0] == (g16 >> 8)).all()                              # png_set_strip_16


def _tf_bilinear_reference(img, size):
    """Fórmula de ResizeBilinear de TF con half_pixel_centers, en float32 como el núcleo de TF
    (escala = float(in) / out; in = (o + 0.5) * escala - 0.5; interpolación top + (bottom - top) * lerp)."""
    f = np.float32
    im = img.astype(np.float32)
    H, W = img.shape[:2]
    sy, sx = f(H) / f(size), f(W) / f(size)
    out = np.zeros((size, size) + img.shape[2:], dtype=np.float32)
    for oy in range(size):
        iy = (f(oy) + f(0.5)) * sy - f(0.5)
        y0 = int(np.floor(iy)); ly = f(iy - f(y0))
        ya, yb = min(max(y0, 0), H - 1), min(max(y0 + 1, 0), H - 1)
        for ox in range(size):
            ix = (f(ox) + f(0.5)) * sx - f(0.5)
            x0 = int(np.floor(ix)); lx = f(ix - f(x0))
            xa, xb = min(max(x0, 0), W - 1), min(max(x0 + 1, 0), W - 1)
            top = im[ya, xa] + (im[ya, xb] - im[ya, xa]) * lx
            bot = im[yb, xa] + (im[yb, xb] - im[yb, xa]) * lx
            out[oy, ox] = top + (bot - top) * ly
    return out


def test_resize_matches_tf_formula():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (97, 61, 3)).astype(np.uint8)
    ours = P.resize_bilinear_tf(img, 24)
    ref = _tf_bilinear_reference(img, 24).transpose(2, 0, 1)
    assert np.abs(ours - ref).max() < 1e-4


def test_p1_deterministic(tmp_path):
    rng = np.random.default_rng(2)
    p = _png(tmp_path / "a.png", rng.integers(0, 256, (300, 260), dtype=np.uint8))
    a, b = P.preprocess_p1(p), P.preprocess_p1(p)
    assert a.shape == (3, 224, 224) and a.dtype == np.float32
    assert np.array_equal(a, b) and 0 <= a.min() and a.max() <= 255


@needs_images
def test_p1_deterministic_real():
    f = [ln.split()[1] for ln in open(data_root() / "data" / "test.txt")][:3]
    for name in f:
        p = data_root() / "data" / "dataset" / name
        assert np.array_equal(P.preprocess_p1(p), P.preprocess_p1(p))
        assert np.array_equal(P.preprocess_homogeneo(p), P.preprocess_homogeneo(p))


def test_homogeneo_steps(tmp_path):
    rng = np.random.default_rng(3)
    core = rng.integers(30, 220, (200, 150), dtype=np.uint8)
    img = np.zeros((240, 190), dtype=np.uint8)
    img[20:220, 20:170] = core                                            # banda negra de 20 px
    p = _png(tmp_path / "b.png", img)
    y, meta = P.preprocess_homogeneo(p, 224, return_meta=True)
    assert y.shape == (224, 224) and y.min() >= 0 and y.max() <= 1
    assert (meta["trim_top"], meta["trim_left"], meta["trim_bottom"], meta["trim_right"]) == (20, 20, 20, 20)
    assert meta["crop_mode"] == "central" and meta["crop_side"] == 150
    assert np.array_equal(y, P.preprocess_homogeneo(p, 224))
    mask = np.zeros((240, 190), dtype=bool); mask[60:160, 50:120] = True
    y2, meta2 = P.preprocess_homogeneo(p, 224, mask=mask, return_meta=True)
    assert meta2["crop_mode"] == "mascara" and meta2["crop_side"] == 120   # 100 x 1.2
    bad = np.zeros((240, 190), dtype=bool); bad[0, 0] = True               # no pasa el control de calidad
    assert P.preprocess_homogeneo(p, 224, mask=bad, return_meta=True)[1]["crop_mode"] == "central"


def test_luminance_rgba_composite(tmp_path):
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., 0] = 255; rgba[..., 3] = 128
    lum = P.load_luminance(_png(tmp_path / "c.png", rgba))
    assert np.allclose(lum, 0.299 * 128 / 255.0, atol=1e-6)


# ------------------------------------------------------------------ caché
def _fake_root(tmp_path, n=6):
    root = tmp_path / "datos"
    (root / "data" / "dataset").mkdir(parents=True)
    rng = np.random.default_rng(4)
    rows = []
    for i in range(n):
        name = "img%02d.png" % i
        _png(root / "data" / "dataset" / name, rng.integers(0, 256, (64 + i, 80), dtype=np.uint8))
        rows.append({"image_id": name, "image_path": "data/dataset/" + name})
    import pandas as pd
    return root, pd.DataFrame(rows)


@pytest.mark.parametrize("channel", ["p1", "homogeneo"])
def test_cache_build_resume_and_roundtrip(tmp_path, monkeypatch, channel):
    root, df = _fake_root(tmp_path)
    monkeypatch.setenv("PAPER_NEW_DATA_ROOT", str(root))
    out = tmp_path / "cache"
    r = C.build_cache(df, "fake", channel, 32, root=out, workers=2, chunk=3, log=lambda *a: None)
    assert r["ok"] == 6 and r["missing"] == 0
    d = C.cache_dir("fake", channel, 32, out)
    assert (d / "index.csv").exists() and (d / "plan.json").exists()
    idx = C.read_index(d)
    x = C.decode_file(idx["img00.png"], channel)
    ref = P.preprocess_p1(root / "data/dataset/img00.png", 32) if channel == "p1" else \
        np.repeat(P.to_uint8_png_array(P.preprocess_homogeneo(root / "data/dataset/img00.png", 32))[None] / 255.0, 3, 0)
    assert x.shape == (3, 32, 32) and np.abs(x - ref).max() <= (1 / 512 + 1e-6 if channel == "p1" else 1e-6)
    os.remove(idx["img03.png"])                                             # reanudación: solo rehace la que falta
    lines = []
    r2 = C.build_cache(df, "fake", channel, 32, root=out, workers=1, log=lines.append)
    assert r2["ok"] == 6 and any("1/1 nuevas" in str(s) for s in lines)


def test_cache_space_check(tmp_path, monkeypatch):
    root, df = _fake_root(tmp_path, 3)
    monkeypatch.setenv("PAPER_NEW_DATA_ROOT", str(root))
    import collections
    monkeypatch.setattr(C.shutil, "disk_usage", lambda p: collections.namedtuple("u", "total used free")(1, 1, 10))
    with pytest.raises(OSError):
        C.build_cache(df, "fake", "p1", 32, root=tmp_path / "c", workers=1, log=lambda *a: None)


@needs_manifest
@needs_images
def test_dataset_item_real(man):
    t = M.get_splits("oficial", man)["test"].head(2)
    it = CXRDataset(t, "p1")[0]
    assert set(it) == {"image", "label", "source", "patient", "image_id", "index"}
    assert it["image"].shape == (3, 224, 224) and it["image_id"] == t["image_id"][0]
    assert int(it["source"]) == M.SOURCE_INDEX[t["source"][0]]


# ------------------------------------------------------------------ muestreador
def test_sampler_balance_and_determinism():
    labels = np.r_[np.ones(900, int), np.zeros(100, int)]
    s = BalancedInfiniteBatchSampler(labels, 64, seed=5)
    batches = [s.batch(b) for b in range(300)]
    frac = np.mean([labels[b].mean() for b in batches])
    assert abs(frac - 0.5) < 0.01                                             # 0.5 en esperanza
    neg = np.concatenate([[i for i in b if labels[i] == 0] for b in batches])
    assert len(set(neg.tolist())) == 100                                      # recorre la clase pequeña entera
    s2 = BalancedInfiniteBatchSampler(labels, 64, seed=5, start_batch=123)
    it = iter(s2)
    assert next(it) == batches[123] and next(it) == batches[124]              # reanudación exacta
    e = BalancedInfiniteBatchSampler(labels, 64, seed=5, mode="exact")
    assert all(labels[e.batch(b)].sum() == 32 for b in range(20))


# ------------------------------------------------------------------ aumentos
def test_augment_p1_deterministic_and_mixup():
    x = torch.rand(8, 3, 32, 32) * 255
    y = torch.tensor([1., 0, 1, 0, 1, 0, 1, 0])
    aug = Paper1Augment()
    a1, y1 = aug(x, y, make_generator(1, 7))
    a2, y2 = aug(x, y, make_generator(1, 7))
    assert torch.equal(a1, a2) and torch.equal(y1, y2)
    a3, _ = aug(x, y, make_generator(1, 8))
    assert not torch.equal(a1, a3)
    lams = []
    for st in range(200):
        _, yy = aug(x[:2], torch.tensor([1., 0]), make_generator(2, st))
        lams.append(float(yy[0]))
    lams = np.array(lams)
    assert ((lams < 0.1) | (lams > 0.9)).mean() > 0.5                         # Beta(0.2, 0.2) en forma de U


def test_augment_new_keeps_hard_labels_and_mask():
    x = torch.rand(4, 3, 32, 32)
    m = torch.zeros(4, 1, 32, 32); m[:, :, 8:24, 8:24] = 1
    xa, ya, ma = NewAugment(cutout=0)(x, torch.tensor([1., 0, 1, 0]), make_generator(0, 0), mask=m)
    assert torch.equal(ya, torch.tensor([1., 0, 1, 0])) and ma.shape == m.shape


def test_cutout_formula():
    x = torch.ones(1, 1, 50, 50)
    out = cutout_batch(x, torch.tensor([25.0]), torch.tensor([25.0]), 24)
    assert int((out == 0).sum()) == 25 * 25                                    # |d| <= 12 a ambos lados


def test_flip_rotation_identity():
    aug = Paper1Augment(rotation=0.0, zoom=0.0, brightness=0.0, contrast=None, cutout=0, mixup_alpha=0.0, flip=False)
    x = torch.rand(2, 3, 16, 16)
    out, _ = aug(x, torch.tensor([1., 0]), make_generator(0, 0))
    assert torch.allclose(out, x)
    from src.data.datasets import rotate_batch, zoom_batch
    assert torch.allclose(rotate_batch(x, torch.zeros(2)), x, atol=1e-5)
    assert torch.allclose(zoom_batch(x, torch.ones(2), torch.ones(2)), x, atol=1e-5)
