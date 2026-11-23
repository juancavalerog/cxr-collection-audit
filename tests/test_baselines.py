"""Pruebas de las líneas base (src/baselines) en CPU con datos sintéticos diminutos: 4 fuentes y dos celdas
clase por fuente vacías (fuente 2 solo positiva, fuente 3 solo negativa; 0 y 1 con ambas clases).

Ejecutar desde la raíz de paper_new: pytest -q tests/test_baselines.py
"""
from __future__ import annotations

import copy
import glob
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.common import cross_source, grad_reverse  # noqa: E402
from src.baselines.methods.dfa import swap_bias  # noqa: E402
from src.baselines.methods.lisa import lisa_mix, lisa_pairs  # noqa: E402
from src.baselines.methods.style import amplitude_mix, mixstyle  # noqa: E402
from src.baselines.registry import all_methods, apply_hparams, sample_hparams  # noqa: E402
from src.baselines.trainer import BaselineTrainer, MaskedDataset, MaskStore, TestLockedError, check_test_lock  # noqa: E402
from src.train.config import resolve  # noqa: E402
from src.train.trainer import EXIT_DEBUG_STOP, Trainer  # noqa: E402

LEVEL1 = ["erm", "control_fuente", "groupdro", "dann", "coral", "mixstyle", "fourier", "lisa", "dfa_fuente", "dfr",
          "erm_mascara", "erm_cuotas"]
LEVEL2_IMPL = ["cdann", "irm", "lai2026", "rad_dino_probe"]
CELLS = {0: [0, 1, 3], 1: [0, 1, 2]}          # (1, 3) y (0, 2) vacías

TINY = {
    "experiment": "tiny_bl",
    "model": {"name": "tiny"},
    "data": {"synthetic": {"n_train": 64, "n_val": 64, "n_test": 24, "res": 16, "n_sources": 4, "cells": CELLS},
             "eval_sets": ["val", "val_robust", "val_robust_sel"]},
    "train": {"batch_size": 16, "micro_batch": 16, "log_every": 1000,
              "phases": [{"name": "A", "epochs": 1, "lr": 1e-2, "warmup_frac": 0.1, "train_backbone": False,
                          "ema": False, "early_stopping": False},
                         {"name": "B", "epochs": 2, "lr": 1e-3, "warmup_frac": 0.1, "train_backbone": True,
                          "ema": True, "early_stopping": True}]},
    "selection": {"metric": "auc_roc", "subset": "val_robust_sel", "impl": "exact", "patience": 15},
    "thermal": {"every": 1000},
    "protocol": {"fase": "piloto", "test_bloqueado": True},
}

EXTRA = {
    "control_fuente": {"data.train_sources": ["s0"]},
    "dfa_fuente": {"baseline.params": {"swap_start_epoch": 0}},
    "lai2026": {"baseline.params": {"swap_start_epoch": 0}},
    "dfr": {"data.eval_sets": ["val", "val_robust_sel"],
            "baseline.params": {"n_reps": 2, "C_grid": [1.0, 0.1]}},
    "mixstyle": {"baseline.params": {"p": 1.0}},
    "erm_cuotas": {"baseline.params": {"min_per_source": 1, "min_per_class_mixed": 2}},   # cuota 12 en lotes de 16
    "rad_dino_probe": {"baseline.params": {"encoder": "timm", "timm_name": "vit_tiny_patch16_224", "input_res": 16,
                                          "timm_kwargs": {"depth": 1, "embed_dim": 48, "num_heads": 2}},
                       "train.augment": "none", "model.input_scale": 255.0,
                       "train.phases": [{"name": "P", "epochs": 2, "lr": 1e-2, "warmup_frac": 0.1,
                                         "train_backbone": False, "ema": False, "early_stopping": True}]},
}


def cfg_for(method, tmp_path=None, **over):
    c = copy.deepcopy(TINY)
    c["experiment"] = "tiny_" + method
    c["baseline"] = {"method": method, "params": {}}
    for k, v in list(EXTRA.get(method, {}).items()) + list(over.items()):
        d = c
        ks = k.split(".")
        for kk in ks[:-1]:
            d = d.setdefault(kk, {})
        if isinstance(v, dict) and isinstance(d.get(ks[-1]), dict):
            d[ks[-1]].update(v)
        else:
            d[ks[-1]] = copy.deepcopy(v)
    if method == "erm_mascara":
        c["baseline"]["params"]["mask_dir"] = str(make_masks(tmp_path / "masks", c))
    return c


def make_masks(d: Path, c, res=16):
    """Máscara sintética: cuadrado central de 8 x 8 a 1, resto 0, para todas las imágenes de todos los conjuntos."""
    from PIL import Image
    from src.train.synthetic import synthetic_frame
    d.mkdir(parents=True, exist_ok=True)
    a = np.zeros((res, res), np.uint8)
    a[4:12, 4:12] = 255
    sc = c["data"]["synthetic"]
    for split, n in (("train", sc["n_train"]), ("val", sc["n_val"]), ("test", sc["n_test"])):
        for i in range(n):
            p = d / ("%s_%05d.png" % (split, i))
            if not p.exists():
                Image.fromarray(a).save(p)
    return d


def run(c, seed, out, **kw):
    tr = BaselineTrainer(c, seed=seed, out_dir=out, echo=False, **kw)
    return tr, tr.fit()


def _sd_equal(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


# ====================================================================== registro y configuraciones
def test_registry_has_all_methods():
    m = all_methods()
    for n in LEVEL1:
        assert n in m and m[n].status == "implementado", n
    for n in ("cdann", "irm", "pbbl", "lai2026", "rad_dino_probe"):
        assert n in m, n
    assert m["pbbl"].status == "pendiente"
    refs = open(ROOT / "refs" / "refs.bib", encoding="utf-8").read()
    for n, cls in m.items():
        for key in cls.ref.split(";"):
            assert "{%s," % key.strip() in refs, (n, key)


def test_search_protocol_same_budget():
    for n in all_methods():
        assert sample_hparams(n, 0) == {"common": {}, "params": {}}
        a, b = sample_hparams(n, 3), sample_hparams(n, 3)
        assert a == b and "lr_B" in a["common"]
        assert set(a["params"]) == set(all_methods()[n].search)
    c = apply_hparams(resolve(cfg_for("groupdro")), sample_hparams("groupdro", 2))
    assert 1e-5 <= c["train"]["phases"][-1]["lr"] <= 10 ** -3.5
    assert 1e-3 <= c["baseline"]["params"]["eta"] <= 1e-1


def test_configs_exist_and_test_locked():
    from src.baselines.run import load_config
    files = sorted(f for f in glob.glob(str(ROOT / "configs" / "baseline_*.yaml")) if not f.endswith("baseline_base.yaml"))
    base = load_config(ROOT / "configs" / "baseline_base.yaml")
    assert base["data"]["cache"] == "data/cache_m/covidx_cxr4/homogeneo_224" and base["selection"]["subset"] == "val_robust_sel"
    names = {Path(f).stem.replace("baseline_", "") for f in files}
    for n in LEVEL1 + ["cdann", "irm", "pbbl", "lai2026", "rad_dino_probe"]:
        assert n in names, n
    for f in files:
        c = load_config(f)
        assert c["baseline"]["method"] in all_methods()
        assert not [s for s in c["data"]["eval_sets"] if s.startswith("test")], f
        assert c["protocol"]["test_bloqueado"] is True, f
        assert check_test_lock(c) is True
        assert c["data"]["mode"] == "limpia" and c["data"]["channel"] == "homogeneo"
        assert c["data"]["cache"] == "data/cache_m/covidx_cxr4/homogeneo_224" and c["data"]["require_cache"] is True
        assert c["selection"]["subset"] == "val_robust_sel" and "val_robust_sel" in c["data"]["eval_sets"]
        if c["baseline"]["method"] == "dfr":
            assert "val_robust" not in c["data"]["eval_sets"]
        else:
            assert c["data"]["eval_sets"] == ["val", "val_robust", "val_robust_sel"]
        assert "base: baseline_base.yaml" in open(f, encoding="utf-8").read()
        if c["baseline"]["method"] != "rad_dino_probe":
            assert c["model"]["name"] == "paper1_convnext" and c["model"]["init"]["type"] == "timm_imagenet"


# ====================================================================== cada método entrena y la pérdida es finita
@pytest.mark.parametrize("method", LEVEL1 + LEVEL2_IMPL)
def test_method_trains_finite(tmp_path, method):
    c = cfg_for(method, tmp_path)
    tr, code = run(c, 0, tmp_path / "r")
    assert code == 0
    losses = [h["loss"] for h in tr.state["history"]]
    assert losses and all(math.isfinite(v) for v in losses), losses
    s = json.load(open(tmp_path / "r" / "summary.json"))
    assert s["results"] and not [k for k in s["results"] if "/test" in k]
    assert (tmp_path / "r" / "predictions").exists()
    assert "test" not in tr.frames
    # selección y comparación comunes en val_robust_sel; mitades de val_robust disjuntas por paciente
    assert tr.cfg["selection"]["subset"] == "val_robust_sel"
    for st in ("last", "best"):
        assert (tmp_path / "r" / "predictions" / ("%s_val_robust_sel.csv" % st)).exists(), st
    assert set(tr.frames["val_robust_fit"]["patient_id"]).isdisjoint(set(tr.frames["val_robust_sel"]["patient_id"]))
    assert len(tr.frames["val_robust_fit"]) + len(tr.frames["val_robust_sel"]) == len(tr.frames["val_robust"])
    assert s["comparacion_entre_metodos"] == "val_robust_sel" and s["seleccion"] == "val_robust_sel"
    if method == "dfr":
        assert not [k for k in s["results"] if k.endswith("/val_robust")]
    else:
        assert s["no_comparables"] == {}
    assert sorted(tr.empty) == ([(0, 2)] if method == "control_fuente" else [(0, 2), (1, 3)]) or method == "control_fuente"


def test_pbbl_is_declared_pending(tmp_path):
    with pytest.raises(NotImplementedError):
        BaselineTrainer(cfg_for("pbbl"), seed=0, out_dir=tmp_path / "p", echo=False)


def test_control_fuente_requires_train_sources(tmp_path):
    c = cfg_for("control_fuente")
    c["data"]["train_sources"] = None
    with pytest.raises(ValueError):
        BaselineTrainer(c, seed=0, out_dir=tmp_path / "c", echo=False)


def test_erm_identical_to_common_trainer(tmp_path):
    c = cfg_for("erm")
    c["data"]["synthetic"] = {k: v for k, v in c["data"]["synthetic"].items() if k != "cells"}
    c["data"]["eval_sets"] = ["val", "val_robust"]          # el común no conoce las mitades de val_robust
    c["selection"]["subset"] = "val_robust"
    ref = Trainer(copy.deepcopy(c), seed=3, out_dir=tmp_path / "a", echo=False)
    assert ref.fit() == 0
    tr, code = run(c, 3, tmp_path / "b")
    assert code == 0
    for st in ("last", "best", "ema"):
        a = torch.load(tmp_path / "a" / "states" / ("%s.pt" % st))
        b = {k[len("net."):]: v for k, v in torch.load(tmp_path / "b" / "states" / ("%s.pt" % st)).items()}
        assert _sd_equal(a, b), st


# ====================================================================== GroupDRO
def test_groupdro_updates_weights_and_empty_cells(tmp_path):
    c = cfg_for("groupdro")
    c["baseline"]["params"]["eta"] = 0.5
    tr = BaselineTrainer(c, seed=0, out_dir=tmp_path / "g", echo=False)
    m = tr.model.aux["groupdro"]
    valid = m.valid.clone()
    expected = torch.zeros(16, dtype=torch.bool)
    for y in (0, 1):
        for s in CELLS[y]:
            expected[2 * s + y] = True
    assert torch.equal(valid, expected)
    q0 = m.q.clone()
    assert torch.allclose(q0[valid], torch.full((int(valid.sum()),), 1.0 / int(valid.sum())))
    assert tr.fit() == 0
    q = tr.model.aux["groupdro"].q
    assert not torch.allclose(q, q0)
    assert torch.all(q[~valid] == 0) and abs(float(q.sum()) - 1.0) < 1e-5
    assert sorted(tr.empty) == [(0, 2), (1, 3)]
    sd = torch.load(tmp_path / "g" / "states" / "last.pt")
    assert torch.equal(sd["aux.groupdro.q"], q)


# ====================================================================== DANN
def test_grad_reverse_flips_sign():
    x = torch.randn(5, 3, requires_grad=True)
    w = torch.randn(5, 3)
    (grad_reverse(x, 0.7) * w).sum().backward()
    assert torch.allclose(x.grad, -0.7 * w)
    x2 = torch.randn(4, requires_grad=True)
    y = grad_reverse(x2, 1.0)
    assert torch.equal(y, x2)


def test_dann_discriminator_learns_and_backbone_gets_reversed_grad(tmp_path):
    c = cfg_for("dann")
    tr = BaselineTrainer(c, seed=0, out_dir=tmp_path / "d", echo=False)
    tr._setup_phase(fresh=True)
    tr.state["phase"] = 1
    tr._setup_phase(fresh=True)
    b = next(iter(tr.train_loader(0)))
    x, y, s = b["image"].float(), b["label"].float(), b["source"].long()
    net = tr.model
    f = net.pooled(x).detach().requires_grad_(True)
    d = net.aux["disc"]
    import torch.nn.functional as F
    out = d.mlp(grad_reverse(f, 1.0))
    F.cross_entropy(out, d.lut[s]).backward()
    g_rev = f.grad.clone()
    f.grad = None
    F.cross_entropy(d.mlp(f), d.lut[s]).backward()
    assert torch.allclose(g_rev, -f.grad, atol=1e-7)
    assert int((d.lut >= 0).sum()) == 4


# ====================================================================== MixStyle y Fourier
def _batch(B=24, res=8, sources=(0, 1, 2, 3)):
    g = torch.Generator().manual_seed(0)
    x = torch.rand(B, 3, res, res, generator=g)
    s = torch.tensor([sources[i % len(sources)] for i in range(B)])
    y = torch.tensor([float(i % 2) for i in range(B)])
    return x, y, s


def test_mixstyle_only_across_sources():
    x, y, s = _batch()
    g = torch.Generator().manual_seed(1)
    out, j, has = mixstyle(x, s, g, alpha=0.1)
    assert has.all() and torch.all(s[j] != s)
    xs, _, ss = _batch(sources=(2,))
    out2, j2, has2 = mixstyle(xs, ss, torch.Generator().manual_seed(2))
    assert not has2.any() and torch.equal(j2, torch.arange(len(ss)))
    assert torch.allclose(out2, xs, atol=1e-4)


def test_mixstyle_hook_in_trainer(tmp_path):
    tr = BaselineTrainer(cfg_for("mixstyle"), seed=0, out_dir=tmp_path / "m", echo=False)
    tr._setup_phase(fresh=True)
    b = next(iter(tr.train_loader(0)))
    tr.train_step(b)
    ctx = tr.method.last_ctx
    s = b["source"]
    assert ctx["apply"] and "partner" in ctx
    j, has = ctx["partner"], ctx["has"]
    assert torch.all(s[j][has] != s[has])
    tr.model.eval()
    assert tr.model._ctx is None


def test_fourier_only_across_sources():
    x, y, s = _batch()
    out, j, has = amplitude_mix(x, s, torch.Generator().manual_seed(3), eta=1.0)
    assert has.all() and torch.all(s[j] != s)
    assert not torch.allclose(out, x, atol=1e-3)
    xs, _, ss = _batch(sources=(1,))
    out2, j2, has2 = amplitude_mix(xs, ss, torch.Generator().manual_seed(4))
    assert not has2.any() and torch.allclose(out2, xs, atol=1e-5)


# ====================================================================== LISA
def test_lisa_two_modes():
    x, _, _ = _batch(B=32)
    s = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3] * 4)
    y = torch.tensor([0., 1., 0., 1., 1., 1., 0., 0.] * 4)          # 2 solo positiva, 3 solo negativa
    g = torch.Generator().manual_seed(5)
    j, has = lisa_pairs(y, s, "intra_label", g)
    assert has.all()
    assert torch.all(y[j] == y) and torch.all(s[j] != s)
    j, has = lisa_pairs(y, s, "intra_domain", g)
    assert torch.all(s[j][has] == s[has]) and torch.all(y[j][has] != y[has])
    assert not has[(s == 2) | (s == 3)].any() and has[(s == 0) | (s == 1)].all()
    xm, ym, j, has = lisa_mix(x, y, s, "intra_domain", g)
    assert torch.equal(xm[~has], x[~has]) and torch.equal(ym[~has], y[~has])
    xm, ym, j, has = lisa_mix(x, y, s, "intra_label", g)
    assert torch.equal(ym, y)


# ====================================================================== DFA
def test_dfa_swap_keeps_content_label():
    zi, zb = torch.randn(12, 5), torch.randn(12, 5)
    s = torch.tensor([0, 1, 2, 3] * 3)
    y = torch.tensor([1., 0., 1., 0.] * 3)
    zs, ty, ts, j = swap_bias(zi, zb, y, s, torch.Generator().manual_seed(6), cross=True)
    assert torch.equal(ty, y)
    assert torch.equal(ts, s[j]) and torch.all(s[j] != s)
    assert torch.equal(zs[:, :5], zi) and torch.equal(zs[:, 5:], zb[j])


def test_dfa_inference_uses_content_classifier(tmp_path):
    tr = BaselineTrainer(cfg_for("dfa_fuente"), seed=0, out_dir=tmp_path / "f", echo=False)
    net = tr.model.eval()
    x = torch.rand(3, 3, 16, 16) * 255
    with torch.no_grad():
        ref = net.aux["c_i"](torch.cat([net.net.pooled(x), net.aux["bias_net"].pooled(x)], 1))
        assert torch.allclose(net(x), ref)
    bb = {id(p) for p in net.backbone_parameters()}
    assert all(id(p) in bb for p in net.aux["bias_net"].backbone.parameters())


# ====================================================================== DFR
def test_dfr_only_last_layer_and_no_test(tmp_path):
    tr, code = run(cfg_for("dfr"), 0, tmp_path / "r")
    assert code == 0
    st = tmp_path / "r" / "states"
    info = json.load(open(st / "dfr_info.json"))
    fit_ids = set(tr.frames["val_robust_fit"]["image_id"])
    sel_ids = set(tr.frames["val_robust_sel"]["image_id"])
    assert not fit_ids & sel_ids
    assert set(tr.frames["val_robust_fit"]["patient_id"]).isdisjoint(set(tr.frames["val_robust_sel"]["patient_id"]))
    from src.train.synthetic import synthetic_frame
    test_ids = set(synthetic_frame(24, 4, 0, "test")["image_id"])
    for s in ("last", "best", "ema"):
        a = torch.load(st / ("%s.pt" % s))
        b = torch.load(st / ("dfr_%s.pt" % s))
        changed = [k for k in a if not torch.equal(a[k], b[k])]
        assert changed and set(changed) <= {"net.fc2.weight", "net.fc2.bias"}, changed
        assert set(info[s]["fit_ids"]) <= fit_ids and not set(info[s]["fit_ids"]) & test_ids
        assert set(info[s]["sel_ids"]) == sel_ids
    s = json.load(open(tmp_path / "r" / "summary.json"))
    assert "dfr_ema/val_robust_sel" in s["results"]
    assert "test" not in tr.frames


# ====================================================================== enmascarado
def test_mask_zeroes_outside_and_missing_fails(tmp_path):
    c = cfg_for("erm_mascara", tmp_path)
    tr = BaselineTrainer(c, seed=0, out_dir=tmp_path / "k", echo=False)
    ds = tr.train_ds
    assert isinstance(ds, MaskedDataset)
    d = ds[0]
    m = d["mask"]
    assert torch.all(d["image"][:, m == 0] == 0) and torch.any(d["image"][:, m == 1] != 0)
    raw = ds.base[0]["image"]
    assert torch.equal(d["image"][:, m == 1], raw[:, m == 1])
    empty = tmp_path / "vacio"
    empty.mkdir()
    c2 = copy.deepcopy(c)
    c2["baseline"]["params"]["mask_dir"] = str(empty)
    with pytest.raises(FileNotFoundError, match="faltan .* máscaras"):
        BaselineTrainer(c2, seed=0, out_dir=tmp_path / "k2", echo=False)
    c3 = copy.deepcopy(c)
    c3["baseline"]["params"]["mask_dir"] = str(tmp_path / "no_existe")
    with pytest.raises(FileNotFoundError, match="enmascarado pulmonar"):
        BaselineTrainer(c3, seed=0, out_dir=tmp_path / "k3", echo=False)
    st = MaskStore(c["baseline"]["params"]["mask_dir"])
    with pytest.raises(ValueError, match="mask_resize"):
        st.get("train_00000", (32, 32))
    assert st.__class__(c["baseline"]["params"]["mask_dir"], resize=True).get("train_00000", (32, 32)).shape == (32, 32)


# ====================================================================== reanudación
@pytest.mark.parametrize("method", LEVEL1 + LEVEL2_IMPL)
def test_resume_each_derived_trainer(tmp_path, method):
    c = cfg_for(method, tmp_path)
    _, code = run(copy.deepcopy(c), 4, tmp_path / "ref")
    assert code == 0
    c1 = copy.deepcopy(c)
    c1["train"]["stop_after_steps"] = 6
    _, code = run(c1, 4, tmp_path / "res")
    assert code == EXIT_DEBUG_STOP
    tr2, code = run(copy.deepcopy(c), 4, tmp_path / "res")
    assert tr2.resumed and code == 0
    for p in sorted((tmp_path / "ref" / "states").glob("*.pt")):
        a = torch.load(p)
        b = torch.load(tmp_path / "res" / "states" / p.name)
        assert _sd_equal(a, b), (method, p.name)


# ====================================================================== bloqueo del test
def test_test_lock(tmp_path):
    c = resolve(cfg_for("erm"))
    c["data"]["eval_sets"] = ["val", "test"]
    with pytest.raises(TestLockedError, match="bloqueado"):
        check_test_lock(c)
    with pytest.raises(TestLockedError):
        BaselineTrainer(c, seed=0, out_dir=tmp_path / "t", echo=False)
    c["protocol"]["test_bloqueado"] = False
    with pytest.raises(TestLockedError, match="congelado"):
        check_test_lock(c)
    fz = tmp_path / "fz.json"
    c["protocol"]["congelado"] = str(fz)
    with pytest.raises(TestLockedError, match="no existe"):
        check_test_lock(c)
    fz.write_text(json.dumps({"method": "groupdro", "params": {}}))
    with pytest.raises(TestLockedError, match="no coincide"):
        check_test_lock(c)
    fz.write_text(json.dumps({"method": "erm", "params": {}, "lr_B": 5e-4}))
    with pytest.raises(TestLockedError, match="lr"):
        check_test_lock(c)
    fz.write_text(json.dumps({"method": "erm", "params": {}, "lr_B": c["train"]["phases"][-1]["lr"]}))
    assert check_test_lock(c) is False
    tr, code = run(c, 0, tmp_path / "u")
    assert code == 0 and "test" in tr.frames
    s = json.load(open(tmp_path / "u" / "summary.json"))
    assert "ema/test" in s["results"]
    c2 = resolve(cfg_for("erm"))
    c2["selection"]["subset"] = "test"
    c2["protocol"]["test_bloqueado"] = False
    with pytest.raises(TestLockedError, match="selección"):
        check_test_lock(c2)
    c3 = resolve(cfg_for("groupdro"))
    c3["data"]["eval_sets"] = ["val", "test_bimcv"]
    with pytest.raises(TestLockedError):
        check_test_lock(c3)


def test_cli_list_and_make_jobs_dry_run(tmp_path):
    r = subprocess.run([sys.executable, "-m", "src.baselines.run", "--list"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "groupdro" in r.stdout
    r = subprocess.run([sys.executable, "scripts/jobs/make_jobs_baselines.py", "--dry-run"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "total" in r.stdout and r.stdout.count("bl_piloto_") == 2 * len(LEVEL1)


# ====================================================================== comparabilidad de val_robust y DFR
def test_dfr_marks_full_val_robust_not_comparable(tmp_path):
    c = cfg_for("dfr", **{"data.eval_sets": ["val", "val_robust", "val_robust_sel"]})
    tr, code = run(c, 1, tmp_path / "d")
    assert code == 0
    s = json.load(open(tmp_path / "d" / "summary.json"))
    nc = s["no_comparables"]
    assert nc and all(k.endswith("/val_robust") for k in nc)
    assert "dfr_ema/val_robust" in nc and "ema/val_robust" in nc
    assert not [k for k in nc if k.endswith("/val_robust_sel")]
    logs = [json.loads(l) for l in open(tmp_path / "d" / "train_log.jsonl", encoding="utf-8")]
    assert any(r["event"] == "baseline" and r["no_comparables"] == ["val_robust"] for r in logs)
    c2 = cfg_for("dfr", **{"selection.subset": "val_robust"})
    with pytest.raises(ValueError, match="val_robust_sel"):
        BaselineTrainer(c2, seed=0, out_dir=tmp_path / "d2", echo=False)


def test_val_robust_halves_by_patient_on_frame():
    import pandas as pd
    from src.baselines.methods.dfr import patient_half, split_val_robust
    df = pd.DataFrame({"patient_id": ["p%d" % (i // 3) for i in range(300)], "label": [i % 2 for i in range(300)]})
    a, b = split_val_robust(df)
    assert len(a) + len(b) == 300 and set(a.patient_id).isdisjoint(set(b.patient_id))
    assert patient_half("p7") == patient_half("p7") and {patient_half("p%d" % i) for i in range(50)} == {0, 1}
    a2, _ = split_val_robust(df.sample(frac=1.0, random_state=3))
    assert set(a2.patient_id) == set(a.patient_id)                   # independiente del orden y de la semilla


# ====================================================================== contrato de máscaras
def test_mask_contract_codification_and_index(tmp_path):
    """data/masks/<dataset>/homogeneo_224/: PNG de 8 bits y una banda, 224 x 224; 0 fondo, 100 solo corazón,
    200 solo pulmón, 255 pulmón y corazón; pulmón = valor > 127. index.csv con image_id, path (relativa),
    quality, lung_frac, heart_frac."""
    import pandas as pd
    from PIL import Image
    d = tmp_path / "masks" / "covidx_cxr4" / "homogeneo_224"
    (d / "sub").mkdir(parents=True)
    a = np.zeros((224, 224), np.uint8)
    a[:, 56:112] = 100
    a[:, 112:168] = 200
    a[:, 168:] = 255
    Image.fromarray(a, mode="L").save(d / "sub" / "img_a.png")
    pd.DataFrame({"image_id": ["img_a.png"], "path": ["sub/img_a.png"], "quality": [1], "lung_frac": [0.5],
                  "heart_frac": [0.5]}).to_csv(d / "index.csv", index=False)
    st = MaskStore(d)
    st.check(["img_a.png"])
    m = st.get("img_a.png", (224, 224))
    assert m.shape == (224, 224) and m.dtype == torch.float32
    assert torch.all(m[:, :112] == 0) and torch.all(m[:, 112:] == 1)
    with pytest.raises(FileNotFoundError, match="faltan 1 de 2"):
        st.check(["img_a.png", "img_b.png"])
