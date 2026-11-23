"""Pruebas del entrenador (src/train, src/eval) en CPU con datos sintéticos y un modelo diminuto.

Ejecutar desde la raíz de paper_new: pytest -q tests/test_train.py
"""
from __future__ import annotations

import copy
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train.ema import WeightEMA  # noqa: E402
from src.train.losses import focal_loss_keras  # noqa: E402
from src.train.metrics import exact_metric, keras_auc  # noqa: E402
from src.train.optim import TFAdamW, warmup_cosine  # noqa: E402
from src.train.trainer import EXIT_DEBUG_STOP, Trainer  # noqa: E402

TINY = {
    "experiment": "tiny",
    "model": {"name": "tiny"},
    "data": {"synthetic": {"n_train": 48, "n_val": 24, "n_test": 24, "res": 16}, "eval_sets": ["val", "test", "val_robust"]},
    "train": {"batch_size": 12, "micro_batch": 6, "log_every": 1000,
              "phases": [{"name": "A", "epochs": 1, "lr": 1e-2, "warmup_frac": 0.1, "train_backbone": False,
                          "ema": False, "early_stopping": False},
                         {"name": "B", "epochs": 2, "lr": 1e-3, "warmup_frac": 0.1, "train_backbone": True,
                          "ema": True, "early_stopping": True}]},
    "selection": {"metric": "auc_pr", "subset": "val", "impl": "keras200", "patience": 15},
    "thermal": {"every": 3},
}


def tiny(**over):
    c = copy.deepcopy(TINY)
    for k, v in over.items():
        d = c
        ks = k.split(".")
        for kk in ks[:-1]:
            d = d.setdefault(kk, {})
        d[ks[-1]] = v
    return c


def _sd_equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


# ------------------------------------------------------------------ pérdida, optimizador, programa, métricas
def test_focal_loss_hand_value():
    z = torch.tensor([2.0, -1.0, 0.5])
    y = torch.tensor([1.0, 0.0, 0.3])
    vals = []
    for zi, yi in zip(z.tolist(), y.tolist()):
        p = 1 / (1 + math.exp(-zi))
        pt = yi * p + (1 - yi) * (1 - p)
        bce = -(yi * math.log(p) + (1 - yi) * math.log(1 - p))
        w = yi * 0.5 + (1 - yi) * 0.5
        vals.append(w * (1 - pt) ** 2 * bce)
    assert abs(float(focal_loss_keras(z, y)) - sum(vals) / 3) < 1e-6
    # caso z = 2, y = 1 a mano: p = 0.880797, (1 - p)^2 = 0.014209, bce = 0.126928 -> 0.5 * 0.0018036 = 0.00090180
    assert abs(float(focal_loss_keras(torch.tensor([2.0]), torch.tensor([1.0]))) - 0.00090180) < 1e-7


def test_tfadamw_formula():
    w = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    opt = TFAdamW([w], lr=0.1, eps=1e-7, weight_decay=0.01)
    ref, m, v = np.array([1.0, -2.0]), np.zeros(2), np.zeros(2)
    for t, g in enumerate([np.array([0.5, -1.0]), np.array([0.2, 0.3])], start=1):
        w.grad = torch.tensor(g, dtype=torch.float32)
        opt.step()
        ref = ref - ref * 0.01 * 0.1
        m = m + (g - m) * 0.1; v = v + (g * g - v) * 0.001
        alpha = 0.1 * math.sqrt(1 - 0.999 ** t) / (1 - 0.9 ** t)
        ref = ref - alpha * m / (np.sqrt(v) + 1e-7)
    assert np.allclose(w.detach().numpy(), ref, atol=1e-6)


def test_warmup_cosine_like_keras():
    assert warmup_cosine(0, 1e-3, 100, 10) == 0.0
    assert abs(warmup_cosine(5, 1e-3, 100, 10) - 5e-4) < 1e-12
    assert abs(warmup_cosine(10, 1e-3, 100, 10) - 1e-3) < 1e-12
    assert abs(warmup_cosine(100, 1e-3, 100, 10, 1e-7) - 1e-7) < 1e-12


def test_keras_auc_close_to_exact():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    p = np.clip(0.35 * y + rng.normal(0.4, 0.2, 4000), 0, 1)
    assert abs(keras_auc(y, p, "ROC") - exact_metric("auc_roc", y, p)) < 2e-3
    assert abs(keras_auc(y, p, "PR") - exact_metric("auc_pr", y, p)) < 1e-2


# ------------------------------------------------------------------ EMA
def test_ema_tfkeras_exacta_weight_of_initial_state():
    lin = torch.nn.Linear(3, 1)
    with torch.no_grad():
        lin.weight.fill_(2.0); lin.bias.fill_(-1.0)
    ema = WeightEMA(lin, momentum=0.9999, mode="tfkeras_exacta")
    T, c = 250, 5.0
    with torch.no_grad():
        lin.weight.fill_(c); lin.bias.fill_(c)
    for _ in range(T):
        ema.update(lin)
    k = 0.9999 ** T
    sd = ema.ema_state_dict(lin)
    assert torch.allclose(sd["weight"], torch.full((1, 3), 2.0 * k + c * (1 - k)), atol=1e-5)
    assert torch.allclose(sd["bias"], torch.tensor([-1.0 * k + c * (1 - k)]), atol=1e-5)
    # sin corrección de sesgo: tras un paso la media es 0.9999 * inicial + 0.0001 * actual
    e2 = WeightEMA(lin, 0.9999, "tfkeras_exacta")
    with torch.no_grad():
        lin.weight.fill_(0.0)
    e2.update(lin)
    assert torch.allclose(e2.shadow["weight"], torch.full((1, 3), 0.9999 * c), atol=1e-6)
    std = WeightEMA(lin, 0.9999, "estandar")
    assert abs(std.current_momentum() - 0.1) < 1e-12                    # calentamiento (1 + t) / (10 + t)


def test_ema_only_trainable_params_and_trainer_weight(tmp_path):
    tr = Trainer(tiny(), seed=0, out_dir=tmp_path / "r", echo=False)
    assert tr.fit() == 0
    s = json.load(open(tmp_path / "r" / "summary.json"))
    T = s["ema"]["t"]
    assert T == 2 * tr.steps_per_epoch                                      # solo los pasos de la fase B
    hist = [h for h in s["history"] if h["phase"] == "B"]
    assert abs(hist[-1]["ema_init_weight"] - 0.9999 ** T) < 1e-12
    # la EMA (tfkeras_exacta) no toca las LayerNorm congeladas del tronco
    ema = torch.load(tmp_path / "r" / "states" / "ema.pt")
    last = torch.load(tmp_path / "r" / "states" / "last.pt")
    assert torch.equal(ema["bnorm.weight"], last["bnorm.weight"])
    assert not torch.equal(ema["fc1.weight"], last["fc1.weight"])


# ------------------------------------------------------------------ humo, salidas y parada temprana
def test_smoke_two_epochs_outputs(tmp_path):
    out = tmp_path / "s"
    tr = Trainer(tiny(), seed=1, out_dir=out, echo=False)
    assert tr.fit() == 0
    for st in ("last", "best", "ema"):
        assert (out / "states" / ("%s.pt" % st)).exists()
        for name in ("val", "test", "val_robust"):
            import pandas as pd
            d = pd.read_csv(out / "predictions" / ("%s_%s.csv" % (st, name)))
            assert list(d.columns) == ["image_id", "label", "source", "patient", "prob"]
            assert d["image_id"].is_unique and d["prob"].between(0, 1).all()
    ev = [json.loads(l) for l in open(out / "train_log.jsonl")]
    assert sum(e["event"] == "epoch" for e in ev) == 3 and ev[-1]["event"] == "end"
    assert Trainer(tiny(), seed=1, out_dir=out, echo=False).fit() == 0      # ya terminado: no repite


def test_early_stopping_keras_semantics(tmp_path):
    c = tiny()
    c["train"]["phases"][1].update({"epochs": 10, "lr": 0.0})
    c["train"]["optimizer"] = {"weight_decay": 0.0}
    c["model"]["dropout"] = 0.0
    c["selection"]["patience"] = 2
    tr = Trainer(c, seed=2, out_dir=tmp_path / "e", echo=False)
    assert tr.fit() == 0
    hb = [h for h in tr.state["history"] if h["phase"] == "B"]
    assert len(hb) == 3                         # época 0 mejora; épocas 1 y 2 sin mejora; espera 2 >= 2 -> parada
    assert tr.state["es"]["stopped"] and tr.state["es"]["best_epoch"] == 0


def test_selection_on_val_robust_auc_roc(tmp_path):
    c = tiny(**{"selection.metric": "auc_roc", "selection.subset": "val_robust", "selection.impl": "exact"})
    tr = Trainer(c, seed=3, out_dir=tmp_path / "v", echo=False)
    assert tr.fit() == 0
    hb = [h for h in tr.state["history"] if h["phase"] == "B"]
    assert all(h["val_n"] == len(tr.frames["val_robust"]) for h in hb)
    assert abs(hb[0]["selection"] - hb[0]["val_auc_roc"]) < 1e-12


# ------------------------------------------------------------------ reanudación exacta
@pytest.mark.parametrize("stop_at", [2, 5])
def test_resume_identical(tmp_path, stop_at):
    ref = Trainer(tiny(), seed=4, out_dir=tmp_path / "ref", echo=False)
    assert ref.fit() == 0
    c = tiny(**{"train.stop_after_steps": stop_at})
    t1 = Trainer(c, seed=4, out_dir=tmp_path / "res", echo=False)
    assert t1.fit() == EXIT_DEBUG_STOP
    ck = torch.load(tmp_path / "res" / "checkpoint.pt", weights_only=False)
    assert ck["state"]["global_step"] == stop_at
    c2 = tiny()
    t2 = Trainer(c2, seed=4, out_dir=tmp_path / "res", echo=False)
    assert t2.resumed
    assert t2.fit() == 0
    for st in ("last", "best", "ema"):
        a = torch.load(tmp_path / "ref" / "states" / ("%s.pt" % st))
        b = torch.load(tmp_path / "res" / "states" / ("%s.pt" % st))
        assert _sd_equal(a, b), st
    ha = [(h["phase"], h["epoch"], h["loss"], h.get("selection")) for h in ref.state["history"]]
    hb = [(h["phase"], h["epoch"], h["loss"], h.get("selection")) for h in t2.state["history"]]
    assert ha == hb


# ------------------------------------------------------------------ doble seguro térmico y SIGTERM
def test_thermal_selfcheck_stops_loop(tmp_path, monkeypatch):
    smi = tmp_path / "smi.txt"
    smi.write_text("0, 85\n")
    monkeypatch.setenv("PAPER_NEW_FAKE_SMI", str(smi))
    tr = Trainer(tiny(), seed=5, out_dir=tmp_path / "t", echo=False)
    assert tr.fit() == 75
    ck = torch.load(tmp_path / "t" / "checkpoint.pt", weights_only=False)
    assert ck["reason"] == "temperatura" and ck["state"]["global_step"] == 3   # every = 3
    smi.write_text("0, 60\n")                                                   # baja la temperatura: reanuda
    t2 = Trainer(tiny(), seed=5, out_dir=tmp_path / "t", echo=False)
    assert t2.resumed and t2.fit() == 0


def test_thermal_below_limit_does_not_stop(tmp_path, monkeypatch):
    smi = tmp_path / "smi.txt"
    smi.write_text("0, 79.9\n")
    monkeypatch.setenv("PAPER_NEW_FAKE_SMI", str(smi))
    assert Trainer(tiny(), seed=6, out_dir=tmp_path / "u", echo=False).fit() == 0


def test_sigterm_checkpoint_and_exit75(tmp_path):
    import yaml
    c = tiny()
    c["train"]["phases"][1]["epochs"] = 5000                  # largo: la señal llega a mitad del entrenamiento
    c["train"]["log_every"] = 1
    c["thermal"]["every"] = 50
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text(yaml.safe_dump(c))
    out = tmp_path / "sig"
    env = dict(os.environ)
    env.pop("PAPER_NEW_FAKE_SMI", None)
    p = subprocess.Popen([sys.executable, "-m", "src.train.run", "--config", str(cfgp), "--seed", "7", "--out", str(out)],
                         cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = out / "train_log.jsonl"
    t0 = time.time()
    while time.time() - t0 < 60:
        if log.exists() and sum(1 for l in open(log) if '"phase": "B"' in l and '"step"' in l) >= 3:
            break
        time.sleep(0.2)
    p.send_signal(signal.SIGTERM)
    rc = p.wait(timeout=60)
    assert rc == 75, p.stdout.read().decode()[-2000:]
    ck = torch.load(out / "checkpoint.pt", weights_only=False)
    assert ck["reason"] == "parada" and ck["state"]["phase"] == 1 and ck["state"]["global_step"] > 0
    ev = [json.loads(l) for l in open(log)]
    assert any(e["event"] == "stop" and "SIGTERM" in e["reason"] for e in ev)


# ------------------------------------------------------------------ modelo del artículo 1
def test_paper1_model_structure_and_frozen_norms(tmp_path):
    from src.baselines.paper1_model import Paper1ConvNeXt
    m = Paper1ConvNeXt(pretrained=False)
    nb = sum(p.numel() for p in m.backbone.parameters())
    nh = sum(p.numel() for n, p in m.named_parameters() if not n.startswith("backbone."))
    assert nb == 27820128 and nh == 98817                                     # tronco de Keras y cabeza 768-128-1
    fz = m.backbone_norm_params()
    assert len(fz) == 2 * 19 and sum(p.numel() for _, p in fz) == 14784       # 18 LN de bloque + LN final
    names = [n for n, _ in fz]
    assert not any(n.startswith("stem.") or ".downsample." in n for n in names)
    x = torch.rand(2, 3, 224, 224) * 255
    m.eval()
    assert m(x).shape == (2,)
