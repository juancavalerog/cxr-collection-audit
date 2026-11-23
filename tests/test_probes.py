"""Pruebas de las verificaciones V0 a V7 (src/eval/probes.py) con embeddings sintéticos de respuesta conocida:
para cada verificación, un caso que debe aprobar y otro que debe fallar (o, en las que solo se informan, los
dos valores del indicador). Ejecutar desde la raíz: pytest -q tests/test_probes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval import probes as P  # noqa: E402

NB = 100


def rng(s=0):
    return np.random.default_rng(s)


def patients(n, per=2, prefix="p"):
    return np.array(["%s%d" % (prefix, i // per) for i in range(n)])


# ------------------------------------------------------------------------------------- utilidades
def test_group_folds_are_patient_disjoint_and_stratified():
    y = np.r_[np.zeros(60, int), np.ones(60, int)]
    g = patients(120, 3)
    f = P.group_folds(y, g, k=5, seed=1)
    for pid in np.unique(g):
        assert len(np.unique(f[g == pid])) == 1
    for k in range(5):
        assert 0 < y[f == k].mean() < 1
    h = P.group_holdout(y, g, 0.3, 1)
    assert not set(g[h]) & set(g[~h])


# ------------------------------------------------------------------------------------- V0
def test_v0_pass_and_fail():
    a = rng().normal(size=(4, 10))
    assert P.v0_reconstruction(a, a.copy())["veredicto"] == "aprueba"
    assert P.v0_reconstruction(a, a + 1e-3)["veredicto"] == "falla"


def test_v0_from_model_passes():
    from src.model import CSDNet, CSDNetConfig
    torch.manual_seed(0)
    m = CSDNet(CSDNetConfig(pretrained=False, img_size=64, num_sources=4, transformer_depth=1, transformer_heads=4,
                            transformer_init="random", backbone="convnext_atto"))
    x = torch.randn(3, 3, 64, 64)
    mask = torch.zeros(3, 2, 64, 64); mask[:, 0, 16:48, 8:56] = 1
    assert P.v0_from_model(m, x, mask)["veredicto"] == "aprueba"


# ------------------------------------------------------------------------------------- V1
def _v1_data(n=240, leak=0.0, power=2.0, seed=0):
    r = rng(seed)
    y = r.integers(0, 2, n)
    sig = r.normal(size=(n, 8)); sig[:, 0] += leak * (2 * y - 1)
    emb = r.normal(size=(n, 6)); emb[:, 0] += power * (2 * y - 1)
    return sig, y, patients(n, 2), emb


def test_v1_pass_fail_and_inconclusive():
    sig, y, g, emb = _v1_data(leak=0.0)
    r = P.v1_signature_class(sig, y, g, emb, n_boot=NB)
    assert r["veredicto"] == "aprueba", r
    assert r["metricas"]["auc_control_potencia"] >= 0.70
    sig, y, g, emb = _v1_data(leak=1.5)
    assert P.v1_signature_class(sig, y, g, emb, n_boot=NB)["veredicto"] == "falla"
    sig, y, g, emb = _v1_data(leak=0.0, power=0.0)
    assert P.v1_signature_class(sig, y, g, emb, n_boot=NB)["veredicto"] == "no_concluyente"


# ------------------------------------------------------------------------------------- V1b
def test_v1b_indicator_both_ways():
    r = rng(1)
    n = 300
    s = np.repeat([0, 1], n // 2)
    sig = r.normal(size=(n, 8))
    emb_ind = r.normal(size=(n, 12))
    emb_dep = np.c_[sig @ r.normal(size=(8, 12)) * 3 + 0.1 * r.normal(size=(n, 12))]
    a = P.v1b_signature_content(sig, emb_ind, s, patients(n), min_n=100)
    b = P.v1b_signature_content(sig, emb_dep, s, patients(n), min_n=100)
    assert a["veredicto"] == b["veredicto"] == "informa"
    assert a["detalle"]["indicador"] == "por_debajo_del_orientativo"
    assert b["detalle"]["indicador"] == "supera_el_orientativo"


# ------------------------------------------------------------------------------------- V2
def _v2_data(good=True, n_per=150, seed=2):
    r = rng(seed)
    srcs = [0, 1, 2]
    centers = {k: r.normal(scale=4, size=10) for k in srcs}
    s = np.repeat(srcs, n_per)
    X = np.stack([centers[k] for k in s]) + r.normal(size=(len(s), 10))
    swap = {}
    for t in srcs:
        base = centers[t] if good else None
        Z = np.stack([(centers[t] if good else centers[k]) for k in s]) + r.normal(size=(len(s), 10))
        Z[s == t] = np.nan
        swap[t] = Z
    return X, s, patients(len(s)), swap


def test_v2_pass_and_fail():
    X, s, g, sw = _v2_data(True)
    r = P.v2_target_source(X, s, g, sw, [0, 1, 2], n_boot=NB)
    assert r["veredicto"] == "aprueba" and r["metricas"]["td"] > 0.9 and r["metricas"]["permanencia"] < 0.1
    X, s, g, sw = _v2_data(False)
    r = P.v2_target_source(X, s, g, sw, [0, 1, 2], n_boot=NB)
    assert r["veredicto"] == "falla" and r["metricas"]["permanencia"] > 0.9


# ------------------------------------------------------------------------------------- V2b
def _strata(shift, n=200, seed=3):
    r = rng(seed)
    out = []
    for k in range(2):
        lab = np.repeat([0, 1], n // 2)
        z = r.normal(size=(n, 16)); z[:, 0] += shift * (2 * lab - 1)
        out.append({"name": "e%d" % k, "z": z, "label": lab, "groups": patients(n, 2, "e%d_" % k)})
    return out


def test_v2b_pass_and_fail():
    r = P.v2b_origin(_strata(0.0), strata_real=_strata(2.0), n_boot=NB)
    assert r["veredicto"] == "aprueba", r
    assert r["metricas"]["ba_orig_vistas_reales"] > 0.8
    assert P.v2b_origin(_strata(1.5), n_boot=NB)["veredicto"] == "falla"


# ------------------------------------------------------------------------------------- V3
def _v3_data(inject, n=300, seed=4):
    r = rng(seed)
    y = r.integers(0, 2, n)
    d = r.normal(size=12); d /= np.linalg.norm(d)
    X = r.normal(size=(n, 12)) + 1.5 * np.outer(2 * y - 1, d)
    sw = {"rsna": X + 0.01 * r.normal(size=X.shape) - inject * d,
          "stonybrook": X + 0.01 * r.normal(size=X.shape) + inject * d}
    return X, y, patients(n), sw


def test_v3_pass_and_fail():
    X, y, g, sw = _v3_data(0.0)
    r = P.v3_disease(X, y, g, sw, "rsna", "stonybrook", n_boot=NB)
    assert r["veredicto"] == "aprueba", r
    X, y, g, sw = _v3_data(1.5)
    r = P.v3_disease(X, y, g, sw, "rsna", "stonybrook", n_boot=NB)
    assert r["veredicto"] == "falla" and r["metricas"]["delta_pm"] > 0.05


# ------------------------------------------------------------------------------------- V4
def test_v4_levels_and_compare():
    r = rng(5)
    s = np.repeat([0, 1, 2], 150)
    c = r.normal(scale=3, size=(3, 8))
    enc = c[s] + r.normal(size=(450, 8))
    noenc = r.normal(size=(450, 8))
    acc = P.v4_levels({"nivel1": enc, "nivel2": noenc}, s, patients(450))
    assert acc["nivel1"] > 0.9 and acc["nivel2"] < 0.5
    ok = P.v4_compare([0.40, 0.42, 0.41], [0.70, 0.72, 0.71], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1])
    bad = P.v4_compare([0.70, 0.72, 0.71], [0.40, 0.42, 0.41], [0.1, 0.1, 0.1], [0.5, 0.5, 0.5])
    assert ok["veredicto"] == "aprueba" and bad["veredicto"] == "falla"


# ------------------------------------------------------------------------------------- V5
def test_v5_pass_and_fail():
    st = [dict(x, name="u%d" % i) for i, x in enumerate(_strata(0.0, seed=6) + _strata(0.0, seed=7))]
    assert P.v5_view(st, strata_within=_strata(0.0, seed=8), n_boot=NB)["veredicto"] == "aprueba"
    st = [dict(x, name="u%d" % i) for i, x in enumerate(_strata(1.0, seed=6) + _strata(1.0, seed=7))]
    r = P.v5_view(st, n_boot=NB)
    assert r["veredicto"] == "falla" and r["metricas"]["auc_vista"] > 0.65


# ------------------------------------------------------------------------------------- V6
def test_v6_gap_indicator_both_ways():
    n = 200
    y = np.zeros(n, int)
    s = np.repeat([5, 1], n // 2)
    g = patients(n)
    p_gap = np.where(s == 5, 0.1, 0.9)
    r = P.v6_specificity_gap(p_gap, y, s, g, 0.5, 5, 1, n_boot=NB)
    assert r["metricas"]["brecha"] == pytest.approx(1.0) and r["detalle"]["indicador"] == "brecha_distinta_de_cero"
    p_eq = rng(9).uniform(size=n)
    r = P.v6_specificity_gap(p_eq, y, s, g, 0.5, 5, 1, n_boot=NB)
    assert r["detalle"]["indicador"] == "brecha_compatible_con_cero"
    assert P.youden_threshold(np.r_[0, 0, 1, 1], np.r_[0.1, 0.2, 0.8, 0.9]) >= 0.2


# ------------------------------------------------------------------------------------- V7
def test_v7_descriptors_and_reduction():
    img = rng(10).uniform(size=(64, 64))
    d = P.trivial_descriptors(img, {"trim_frac": 0.1, "orig_aspect": 0.8})
    assert d.shape == (16,) and d[0] == pytest.approx(0.1)
    r = rng(11)
    s = np.repeat([0, 1, 2], 150)
    g = patients(450)
    enc = r.normal(scale=3, size=(3, 16))[s] + r.normal(size=(450, 16))
    rnd = r.normal(size=(450, 16))
    a = P.v7_trivial({"PRE-0": (enc, s, g), "P1-P6": (rnd, s, g)})
    b = P.v7_trivial({"PRE-0": (rnd, s, g), "P1-P6": (enc, s, g)})
    assert a["detalle"]["indicador"] == "reduce_la_firma_trivial"
    assert b["detalle"]["indicador"] == "no_reduce_la_firma_trivial"


# ------------------------------------------------------------------------------------- informe
def test_run_probes_and_report(tmp_path):
    r = rng(12)
    n = 240
    s = np.repeat([0, 1, 2], n // 3)
    y = np.where(s == 0, r.integers(0, 2, n), np.where(s == 1, 0, 1))
    g = patients(n)
    data = {"y": y, "s": s, "groups": g, "signature": r.normal(size=(n, 8)),
            "emb_content": r.normal(size=(n, 6)) + np.outer(2 * y - 1, np.ones(6)),
            "emb_tokens": r.normal(size=(n, 6)), "emb_gated": r.normal(size=(n, 6)),
            "emb_decision": r.normal(size=(n, 4)), "stats_real": r.normal(size=(n, 10)),
            "logit": r.normal(size=n), "emb_within_gated": r.normal(size=(n, 6)), "swap": {}}
    for t in (0, 1, 2):
        data["swap"][t] = {"emb_swap_gated": r.normal(size=(n, 6)), "emb_swap_decision": r.normal(size=(n, 4)),
                           "emb_swap_content": data["emb_content"] + 0.01 * r.normal(size=(n, 6)),
                           "stats_swap": r.normal(size=(n, 10)), "logit_swap": r.normal(size=n)}
    cfg = {"robust_source": "a", "v1_extra_sources": [], "v2_sources": ["a", "b", "c"],
           "v2b_strata": [[1, "b", "a", "c"], [0, "c", "a", "b"]],
           "v5_strata": [[0, "a", "b"], [0, "b", "a"], [1, "c", "a"], [1, "a", "c"]],
           "v3_source": "a", "v3_neg_dest": "b", "v3_pos_dest": "c", "v6_sources": ["b", "a"], "n_boot": 50,
           "min_n": 50, "folds": 3}
    res = P.run_probes(data, ["a", "b", "c"], cfg, ("V1", "V1b", "V2", "V2b", "V3", "V4", "V5", "V6"),
                       prob=r.uniform(size=n), threshold=0.5)
    assert set(res) == {"V1", "V1b", "V2", "V2b", "V3", "V4", "V5", "V6"}
    assert res["V2b"]["veredicto"] == "aprueba" and res["V5"]["veredicto"] == "aprueba"
    path = P.write_report(res, tmp_path, "rep", {"corrida": "x"})
    js = json.loads(path.read_text())
    assert js["verificaciones"]["V1"]["umbrales"]["auc_max"] == 0.60
    assert "| V5 |" in (tmp_path / "rep.md").read_text()


def test_thresholds_match_table7():
    t = P.THRESHOLDS
    assert t["V1"] == {"auc_max": 0.60, "ci_upper_max": 0.65, "power_min": 0.70}
    assert (t["V2"]["td_min"], t["V2"]["td_ratio_min"], t["V2"]["permanence_max"]) == (0.70, 0.80, 0.20)
    assert (t["V2b"]["ba_max"], t["V2b"]["ci_upper_max"]) == (0.60, 0.65)
    assert (t["V3"]["dauc_max"], t["V3"]["mean_abs_dp_max"], t["V3"]["dpm_abs_max"]) == (0.02, 0.05, 0.02)
    assert (t["V5"]["auc_mean_max"], t["V5"]["auc_stratum_max"], t["V5"]["ci_upper_max"]) == (0.60, 0.65, 0.65)
    assert t["V0"]["max_abs_err"] == 1e-4 and t["V4"]["alpha"] == 0.05
