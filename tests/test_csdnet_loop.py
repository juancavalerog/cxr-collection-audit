"""Pruebas del bucle de CSD-Net (src/train/csdnet_loop.py) en CPU con datos sintéticos diminutos.

Tabla sintética con celdas vacías (s1 solo negativas, s2 solo positivas; s0 y s3 mixtas). Ejecutar desde la raíz:
pytest -q tests/test_csdnet_loop.py
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

from src.baselines.trainer import TestLockedError  # noqa: E402
from src.train.csdnet_loop import CSDNetTrainer, is_locked_set  # noqa: E402

SYN_PROBES = {"robust_source": "s0", "v1_extra_sources": ["s3"], "v2_sources": ["s0", "s1", "s2"],
              "v2b_strata": [[1, "s1", "s0", "s2"], [0, "s2", "s0", "s1"]],
              "v5_strata": [[0, "s0", "s1"], [0, "s1", "s0"], [1, "s2", "s0"], [1, "s0", "s2"]],
              "v3_source": "s0", "v3_neg_dest": "s1", "v3_pos_dest": "s2", "v6_sources": ["s1", "s0"],
              "min_n": 10, "folds": 3}

BASE = {
    "experiment": "csd_tiny",
    "data": {"synthetic_csd": {"res": 64}, "eval_sets": ["val", "val_robust", "val_robust_sel"], "eval_batch_size": 32},
    "model": {"csdnet": {"pretrained": False, "transformer_depth": 1, "transformer_init": "random", "backbone": "convnext_atto", "transformer_heads": 4}},
    "train": {"batch_size": 24, "steps_per_epoch": 2, "log_every": 1, "augment": "nuevo",
              "augment_params": {"rotation": 0.03, "zoom": 0.1, "brightness": 0.05, "contrast": [0.9, 1.1], "cutout": 8}},
    "selection": {"metric": "auc_roc", "subset": "val_robust_sel", "impl": "exact", "patience": 10},
    "eval": {"tta": True, "states": ["last", "best", "ema"]},
    "thermal": {"every": 50},
    "protocol": {"fase": "piloto", "test_bloqueado": True, "congelado": None},
    "csdnet": {"epochs": {"A1": 1, "A2": 1, "ramp": 2, "B": 2}, "probes": SYN_PROBES, "probe_n_boot": 30,
               "probe_every": 0, "v1_gate_override": "aprueba",
               "sampler": {"min_per_source": 2, "min_per_class_mixed": 2}},
}


def cfg(**over):
    c = copy.deepcopy(BASE)
    for k, v in over.items():
        d = c
        ks = k.split(".")
        for kk in ks[:-1]:
            d = d.setdefault(kk, {})
        d[ks[-1]] = v
    return c


def events(out):
    return [json.loads(l) for l in open(Path(out) / "train_log.jsonl")]


# ---------------------------------------------------------------------------- tabla y muestreo
def test_synthetic_table_has_empty_cells_and_plan(tmp_path):
    tr = CSDNetTrainer(cfg(), 0, out_dir=tmp_path / "t", echo=False)
    types = tr.table.cell_types()
    assert (types == "vacia").sum() == 2
    assert tr.plan.feasible
    r = tr.routing.numpy()
    for y in (0, 1):
        assert np.allclose(np.diag(r[y]), 0)
    b = tr.batches_for_epoch(0, 0)
    assert len(b) == tr.steps_per_epoch and all(len(x) == 24 for x in b)
    assert tr.batches_for_epoch(0, 1) == b[1:]                 # la reanudación salta los lotes ya vistos


# ---------------------------------------------------------------------------- humo A1, A2 y B con tres vistas
def test_smoke_phases_three_views_ramp_log(tmp_path):
    out = tmp_path / "smoke"
    tr = CSDNetTrainer(cfg(), 1, out_dir=out, echo=False)
    assert tr.fit() == 0
    ev = events(out)
    ep = [e for e in ev if e["event"] == "epoch"]
    assert [e["phase"] for e in ep] == ["A1", "A2", "B", "B"]
    a1, a2, b1, b2 = ep
    assert "L_sig" not in a1["losses"] and "L_sig" in a2["losses"]            # lambda_sig = 0 en A1
    for e in (a1, a2):
        assert e["losses"]["n_views"] == 1 and "L_cls_swap" not in e["losses"]
    for e in (b1, b2):
        assert e["swap"] and e["losses"]["n_views"] == 3
        for k in ("L_cls_real", "L_cls_swap", "L_cons", "L_src", "L_sig", "L_ind", "L_seg", "L_total"):
            assert math.isfinite(e["losses"][k]), k
        assert "dev_Psy_max_abs" in e and "tv_fuente_efectiva_entre_clases" in e and "val_auc_roc" in e
    steps = [e for e in ev if e["event"] == "step" and e["phase"] == "B"]
    ramps = [s["ramp"] for s in steps]
    assert ramps == sorted(ramps) and 0 < ramps[0] < 1 and ramps[-1] == pytest.approx(1.0)  # rampa de 2 épocas
    # el intercambio dirigido acerca la fuente efectiva de las dos clases
    assert b2["tv_fuente_efectiva_entre_clases"] < a1["tv_fuente_efectiva_entre_clases"]
    for st in ("last", "best", "ema"):
        assert (out / "states" / ("%s.pt" % st)).exists()
        for s in ("val", "val_robust", "val_robust_sel"):
            assert (out / "predictions" / ("%s_%s.csv" % (st, s))).exists()
    assert not list((out / "predictions").glob("*test*"))
    assert (out / "probes" / "V1_fin_A2_ep001.json").exists()
    assert (out / "probes" / "control_rampa_ep003.json").exists()            # época E_A1 + E_A2 + E_rampa - 1
    fin = json.loads((out / "probes" / "final_best.json").read_text())["verificaciones"]
    assert set(fin) >= {"V0", "V1", "V2", "V2b", "V3", "V4", "V5", "V6"}
    assert fin["V0"]["veredicto"] == "aprueba"
    summ = json.loads((out / "summary.json").read_text())
    assert summ["test_bloqueado"] is True and "apto_aporte_b" in summ["flags"]


# ---------------------------------------------------------------------------- regla de parada de V1
def test_v1_failure_stops_before_phase_b(tmp_path):
    out = tmp_path / "v1"
    tr = CSDNetTrainer(cfg(**{"csdnet.v1_gate_override": "falla", "csdnet.v1_fail_action": "stop"}), 2,
                       out_dir=out, echo=False)
    assert tr.fit() == 0
    ep = [e for e in events(out) if e["event"] == "epoch"]
    assert [e["phase"] for e in ep] == ["A1", "A2"]
    summ = json.loads((out / "summary.json").read_text())
    assert summ["flags"]["v1_fallo_intercambio_desactivado"] and "V1" in summ["stopped_reason"]
    assert (out / "states" / "best.pt").exists()


def test_v1_failure_continue_without_swap(tmp_path):
    out = tmp_path / "v1c"
    c = cfg(**{"csdnet.v1_gate_override": "falla", "csdnet.v1_fail_action": "continue_no_swap",
               "csdnet.epochs": {"A1": 1, "A2": 1, "ramp": 1, "B": 1}, "eval.states": ["last"],
               "csdnet.final_probe_states": []})
    assert CSDNetTrainer(c, 2, out_dir=out, echo=False).fit() == 0
    ep = [e for e in events(out) if e["event"] == "epoch"]
    assert ep[-1]["phase"] == "B" and not ep[-1]["swap"] and ep[-1]["losses"]["n_views"] == 1


# ---------------------------------------------------------------------------- reanudación y paradas
FAST = {"csdnet.epochs": {"A1": 1, "A2": 1, "ramp": 1, "B": 1}, "eval.states": ["last"], "csdnet.final_probe_states": []}


def test_resume_from_checkpoint_identical(tmp_path):
    ref = CSDNetTrainer(cfg(**FAST), 3, out_dir=tmp_path / "ref", echo=False)
    assert ref.fit() == 0
    c = cfg(**FAST, **{"train.stop_after_steps": 3})
    t1 = CSDNetTrainer(c, 3, out_dir=tmp_path / "cut", echo=False)
    assert t1.fit() == 3                                                      # parada a mitad de la época 1
    ck = torch.load(tmp_path / "cut" / "checkpoint.pt", weights_only=False)
    assert ck["state"]["global_step"] == 3 and ck["state"]["step_in_epoch"] == 1
    t2 = CSDNetTrainer(cfg(**FAST), 3, out_dir=tmp_path / "cut", echo=False)
    assert t2.resumed and t2.fit() == 0
    a = torch.load(tmp_path / "ref" / "states" / "last.pt")
    b = torch.load(tmp_path / "cut" / "states" / "last.pt")
    assert all(torch.equal(a[k], b[k]) for k in a)
    la = [round(h["loss"], 6) for h in ref.state["history"]]
    lb = [round(h["loss"], 6) for h in t2.state["history"]]
    assert la == lb


def test_thermal_selfcheck_exit75_and_resume(tmp_path, monkeypatch):
    smi = tmp_path / "smi.txt"
    smi.write_text("0, 85\n")
    monkeypatch.setenv("PAPER_NEW_FAKE_SMI", str(smi))
    c = cfg(**FAST, **{"thermal.every": 2})
    tr = CSDNetTrainer(c, 4, out_dir=tmp_path / "th", echo=False)
    assert tr.fit() == 75
    ck = torch.load(tmp_path / "th" / "checkpoint.pt", weights_only=False)
    assert ck["reason"] == "temperatura" and ck["state"]["global_step"] == 2


def test_sigterm_checkpoint_and_exit75(tmp_path):
    import yaml
    c = cfg(**{"csdnet.epochs": {"A1": 1, "A2": 1, "ramp": 1, "B": 500}, "train.steps_per_epoch": 50})
    p_cfg = tmp_path / "c.yaml"
    p_cfg.write_text(yaml.safe_dump(c))
    out = tmp_path / "sig"
    env = dict(os.environ)
    env.pop("PAPER_NEW_FAKE_SMI", None)
    p = subprocess.Popen([sys.executable, "-m", "src.train.csdnet_run", "--config", str(p_cfg), "--seed", "7",
                          "--out", str(out)], cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = out / "train_log.jsonl"
    t0 = time.time()
    while time.time() - t0 < 90:
        if log.exists() and sum(1 for l in open(log) if '"event": "step"' in l) >= 3:
            break
        time.sleep(0.2)
    p.send_signal(signal.SIGTERM)
    rc = p.wait(timeout=90)
    assert rc == 75, p.stdout.read().decode()[-2000:]
    ck = torch.load(out / "checkpoint.pt", weights_only=False)
    assert ck["reason"] == "parada" and ck["state"]["global_step"] >= 3
    assert any(e["event"] == "stop" and "SIGTERM" in e["reason"] for e in events(out))


# ---------------------------------------------------------------------------- bloqueo del test
def test_test_set_is_locked(tmp_path):
    assert is_locked_set("test") and is_locked_set("test_bimcv") and not is_locked_set("val_robust")
    with pytest.raises(TestLockedError):
        CSDNetTrainer(cfg(**{"data.eval_sets": ["val", "test"]}), 0, out_dir=tmp_path / "a", echo=False)
    with pytest.raises(TestLockedError):
        CSDNetTrainer(cfg(**{"data.eval_sets": ["val", "test_bimcv"]}), 0, out_dir=tmp_path / "b", echo=False)
    with pytest.raises(TestLockedError):
        CSDNetTrainer(cfg(**{"selection.subset": "test"}), 0, out_dir=tmp_path / "c", echo=False)
    # desbloqueado sin fichero congelado, o con uno que no coincide: tampoco
    with pytest.raises(TestLockedError):
        CSDNetTrainer(cfg(**{"data.eval_sets": ["val", "test"], "protocol.test_bloqueado": False}), 0,
                      out_dir=tmp_path / "e", echo=False)
    fz = tmp_path / "fz.json"
    fz.write_text(json.dumps({"method": "csdnet", "params": {"model.csdnet": {}, "csdnet": {}}}))
    with pytest.raises(TestLockedError):
        CSDNetTrainer(cfg(**{"data.eval_sets": ["val", "test"], "protocol.test_bloqueado": False,
                             "protocol.congelado": str(fz)}), 0, out_dir=tmp_path / "f", echo=False)
    tr = CSDNetTrainer(cfg(), 0, out_dir=tmp_path / "d", echo=False)
    assert tr.test_locked and "test" not in tr.frames
    with pytest.raises(PermissionError):
        tr.eval_loader("test")
    # validación robusta partida por paciente como en DFR
    a, b = set(tr.frames["val_robust_fit"]["patient_id"]), set(tr.frames["val_robust_sel"]["patient_id"])
    assert a and b and not (a & b) and len(tr.frames["val_robust_fit"]) + len(tr.frames["val_robust_sel"]) == len(tr.frames["val_robust"])


def test_quality_zero_uses_central_box():
    m = torch.zeros(2, 2, 20, 20); m[:, 0, 5:15, 2:8] = 1
    q = torch.tensor([1.0, 0.0])
    acq, gate = CSDNetTrainer.masks_for_quality(m, q)
    assert torch.equal(acq[0], m[0]) and acq[1].sum() == 0
    assert gate[1, 0, 2:18, 2:18].min() == 1 and gate[1, 0, 0].sum() == 0


# ---------------------------------------------------------------------------- ablaciones por configuración
def _one_b_step(tmp_path, name, **over):
    tr = CSDNetTrainer(cfg(**over), 5, out_dir=tmp_path / name, echo=False)
    tr.state["swap_enabled"] = True
    e = tr.E_A1 + tr.E_A2
    batch = next(iter(tr.train_loader_epoch(e, 0)))
    logs = tr.train_step(batch, e)
    assert all(math.isfinite(v) for v in logs.values())
    return tr, logs


@pytest.mark.parametrize("name,over,views", [
    ("a15_sin_dentro", {"csdnet.weights": {"rho_within": 0.0}}, 2),
    ("a14_canonica", {"model.csdnet": {"pretrained": False, "transformer_depth": 1, "transformer_init": "random", "backbone": "convnext_atto", "transformer_heads": 4,
                                       "refirma_target": "canonical"}}, 1),
    ("a4_sin_compuerta", {"model.csdnet": {"pretrained": False, "transformer_depth": 1, "transformer_init": "random", "backbone": "convnext_atto", "transformer_heads": 4,
                                           "use_gate": False}}, 3),
    ("a2_aleatorio", {"csdnet.swap_routing": "random"}, 3),
    ("a10_sin_intercambio", {"csdnet.weights": {"lambda_swap": 0.0, "lambda_cons": 0.0}}, 1),
    ("a1_mixstyle", {"model.csdnet": {"pretrained": False, "transformer_depth": 1, "transformer_init": "random", "backbone": "convnext_atto", "transformer_heads": 4,
                                      "use_acquisition": False}}, 3),
])
def test_ablations_by_config(tmp_path, name, over, views):
    tr, logs = _one_b_step(tmp_path, name, **over)
    assert logs["n_views"] == views
    if name == "a2_aleatorio":
        r = tr.routing.numpy()
        assert np.allclose(r[0, 0, 1:], 1 / 3)
    if name == "a1_mixstyle":
        assert "L_src" not in logs
