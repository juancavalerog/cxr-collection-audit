"""Pruebas de la estadística de evaluación (src/eval/stats.py), de la Puerta 3 (scripts/jobs/gate3_evaluar.py) y del
informe final (scripts/eval/informe_final.py), en CPU y con datos sintéticos (src/eval/synthetic_tree.py).
Ejecutar desde la raíz: pytest -q tests/test_eval_stats.py --basetemp /tmp/tmp_6/pt
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval import stats as S  # noqa: E402
from src.eval import synthetic_tree as T  # noqa: E402
from src.eval.probes import cluster_bootstrap  # noqa: E402


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G3 = _load("scripts/jobs/gate3_evaluar.py", "gate3_evaluar")
FIN = _load("scripts/eval/informe_final.py", "informe_final")


def _df(ids, y, p, pat=None, src=1):
    return pd.DataFrame({"image_id": ids, "label": y, "source": src, "patient": pat if pat is not None else ids,
                         "prob": p})


# ============================================================================== AUC y DeLong
def test_auc_hand_values_and_ties():
    assert S.auc_roc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    # empates: positivos {0.5, 0.9}, negativos {0.5, 0.5}: (0.5 + 0.5 + 1 + 1) / 4
    assert S.auc_roc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.9]) == pytest.approx(0.75)
    assert math.isnan(S.auc_roc([1, 1], [0.2, 0.3]))
    from sklearn.metrics import roc_auc_score
    r = np.random.default_rng(1)
    y = r.integers(0, 2, 500)
    p = np.round(r.random(500) * 0.6 + 0.3 * y, 2)        # con empates
    assert S.auc_roc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)


def _delong_reference(y, p1, p2):
    """DeLong, DeLong y Clarke-Pearson (1988) por definición, O(m n): componentes V10 y V01 con el núcleo psi."""
    y = np.asarray(y)
    out_v10, out_v01, aucs = [], [], []
    for p in (np.asarray(p1), np.asarray(p2)):
        X, Y = p[y == 1], p[y == 0]
        psi = (X[:, None] > Y[None, :]).astype(float) + 0.5 * (X[:, None] == Y[None, :])
        aucs.append(psi.mean())
        out_v10.append(psi.mean(axis=1))
        out_v01.append(psi.mean(axis=0))
    m, n = len(out_v10[0]), len(out_v01[0])
    s10 = np.cov(np.vstack(out_v10))
    s01 = np.cov(np.vstack(out_v01))
    Sm = s10 / m + s01 / n
    var = Sm[0, 0] + Sm[1, 1] - 2 * Sm[0, 1]
    z = (aucs[0] - aucs[1]) / math.sqrt(var)
    return aucs, Sm, var, z


def test_delong_matches_reference_implementation():
    r = np.random.default_rng(3)
    y = r.integers(0, 2, 300)
    p1 = np.round(1 / (1 + np.exp(-(1.2 * (2 * y - 1) + r.normal(0, 1, 300)))), 3)
    p2 = np.round(0.6 * p1 + 0.4 * r.random(300), 3)
    d = S.delong_paired(y, p1, p2)
    aucs, Sm, var, z = _delong_reference(y, p1, p2)
    assert d["auc1"] == pytest.approx(aucs[0], abs=1e-12) and d["auc2"] == pytest.approx(aucs[1], abs=1e-12)
    assert np.allclose(np.array(d["cov"]), Sm, atol=1e-12)
    assert d["var_delta"] == pytest.approx(var, rel=1e-9)
    assert d["z"] == pytest.approx(z, rel=1e-9)
    assert d["p_bilateral"] == pytest.approx(2 * sps.norm.sf(abs(z)), rel=1e-9)


def test_delong_hand_case():
    # 2 positivos y 2 negativos: modelo 1 perfecto (AUC 1), modelo 2 con AUC 0.5
    y = [1, 1, 0, 0]
    d = S.delong_paired(y, [0.9, 0.8, 0.2, 0.1], [0.9, 0.1, 0.2, 0.8])
    assert d["auc1"] == pytest.approx(1.0) and d["auc2"] == pytest.approx(0.5)
    aucs, Sm, var, z = _delong_reference(np.array(y), np.array([0.9, 0.8, 0.2, 0.1]), np.array([0.9, 0.1, 0.2, 0.8]))
    assert d["var_delta"] == pytest.approx(var)
    # a mano: V10 del modelo 2 = (1, 0), V01 = (0.5, 0.5) -> varianzas 0.5 y 0; modelo 1 constante
    assert d["var_delta"] == pytest.approx(0.5 / 2 + 0.0 / 2)


# ============================================================================== bootstrap de pacientes
def test_bootstrap_uses_cluster_bootstrap_draws():
    r = np.random.default_rng(0)
    y = r.integers(0, 2, 120)
    p = r.random(120) + 0.4 * y
    g = np.array(["p%d" % (i // 3) for i in range(120)])
    lo, hi = cluster_bootstrap(lambda i: S.auc_roc(y[i], p[i]), g, 300, 7)
    b = S.bootstrap_auc(y, p, g, 300, 7)
    assert b["ic95"][0] == pytest.approx(lo) and b["ic95"][1] == pytest.approx(hi)
    assert b["B"] == 300 and b["semilla_bootstrap"] == 7 and b["unidad_remuestreo"] == "paciente"


def test_resampling_is_by_patient():
    sizes = [1, 2, 3, 4, 2, 1, 5]
    g = np.concatenate([["p%d" % k] * n for k, n in enumerate(sizes)])
    seen = []
    S.bootstrap_draws(lambda i: (seen.append(np.array(i)), [0.0])[1], g, 50, 1)
    assert len(seen) == 50
    for idx in seen:
        counts = np.bincount(idx, minlength=len(g))
        for k, n in enumerate(sizes):
            members = np.nonzero(g == "p%d" % k)[0]
            c = counts[members]
            assert len(set(c)) == 1          # todas las imágenes del paciente entran o salen juntas
        assert len(np.unique(g[idx])) <= len(sizes)
        # número de pacientes elegidos (con repetición) igual al de pacientes
        npick = sum(counts[np.nonzero(g == "p%d" % k)[0][0]] for k in range(len(sizes)))
        assert npick == len(sizes)


def test_bootstrap_ci_coverage_small_simulation():
    """Binormal con una imagen por paciente: AUC verdadera = Phi(d / sqrt 2). Cobertura del IC95 % percentil."""
    d = 1.0
    true = sps.norm.cdf(d / math.sqrt(2))
    r = np.random.default_rng(11)
    hits, reps = 0, 120
    for k in range(reps):
        y = np.r_[np.ones(60, int), np.zeros(60, int)]
        p = np.r_[r.normal(d, 1, 60), r.normal(0, 1, 60)]
        g = np.array(["q%d_%d" % (k, i) for i in range(120)])
        lo, hi = S.bootstrap_auc(y, p, g, 300, k)["ic95"]
        hits += lo <= true <= hi
    cov = hits / reps
    assert 0.85 <= cov <= 0.995, cov


def test_cluster_bootstrap_wider_than_image_bootstrap_with_correlated_patients():
    r = np.random.default_rng(5)
    pat = np.repeat(np.arange(60), 5)
    y = np.repeat(r.integers(0, 2, 60), 5)
    eff = np.repeat(r.normal(0, 1.5, 60), 5)
    p = 0.8 * y + eff + r.normal(0, 0.3, 300)
    gp = np.array(["p%d" % i for i in pat])
    gi = np.array(["i%d" % i for i in range(300)])
    w_p = np.diff(S.bootstrap_auc(y, p, gp, 400, 1)["ic95"])[0]
    w_i = np.diff(S.bootstrap_auc(y, p, gi, 400, 1)["ic95"])[0]
    assert w_p > 1.3 * w_i


def test_paired_bootstrap_detects_real_difference_and_not_null():
    r = np.random.default_rng(2)
    n = 400
    pat = np.array(["p%d" % (i // 2) for i in range(n)])
    y = np.repeat(r.integers(0, 2, n // 2), 2)
    z = 2 * y - 1
    strong = 1.5 * z + r.normal(0, 1, n)
    weak = 0.6 * z + r.normal(0, 1, n)
    t = S.paired_auc_test(y, strong, weak, pat, 500, 3)
    assert t["ic95"][0] > 0 and t["p_unilateral"] < 0.01 and t["delta"] > 0.1
    assert t["delong"]["p_unilateral"] < 0.01
    assert t["B"] == 500 and t["semilla_bootstrap"] == 3 and t["B_validos"] == 500
    # nula: dos modelos intercambiables (mismo poder, ruido independiente)
    rej = 0
    for k in range(30):
        a = 0.8 * z + r.normal(0, 1, n)
        b = 0.8 * z + r.normal(0, 1, n)
        tt = S.paired_auc_test(y, a, b, pat, 200, k, delong=False)
        rej += tt["p_unilateral"] < 0.05
    assert rej <= 6           # tasa nominal 0.05; margen para 30 réplicas
    t0 = S.paired_auc_test(y, strong, strong + r.normal(0, 1e-3, n), pat, 300, 0)
    assert t0["ic95"][0] <= 0 <= t0["ic95"][1]


def test_noninferiority_margin():
    assert S.noninferiority(-0.005, 0.01)["no_inferior"] is True
    assert S.noninferiority(-0.02, 0.01)["no_inferior"] is False
    r = np.random.default_rng(4)
    y = r.integers(0, 2, 300)
    a = y + r.normal(0, 1, 300)
    t = S.paired_auc_test(y, a, a + r.normal(0, 0.05, 300), np.arange(300).astype(str), 300, 1, margin=0.05)
    assert t["no_inferioridad"]["no_inferior"] and "p_unilateral_margen" in t["no_inferioridad"]


# ============================================================================== Holm, ECE, umbral, semillas
def test_holm_hand_case():
    adj, rej = S.holm([0.01, 0.04, 0.03, 0.005])
    # ordenados: 0.005 x 4 = 0.02; 0.01 x 3 = 0.03; 0.03 x 2 = 0.06; 0.04 x 1 = 0.04 -> monótono 0.06
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])
    assert list(rej) == [True, False, False, True]
    d = S.holm({"a": 0.5, "b": float("nan"), "c": 0.6})
    assert d["a"]["p_holm"] == pytest.approx(1.0) and math.isnan(d["b"]["p_holm"]) and d["c"]["p_holm"] == pytest.approx(1.0)


def test_ece_hand_case():
    # cuatro bins distintos de 15: (0.1 + 0.1 + 0.2 + 0.3) / 4
    assert S.ece([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.3]) == pytest.approx(0.175)
    # un bin compartido (0.61 y 0.62 caen en el bin 9): |0.5 - 0.615|
    assert S.ece([1, 0], [0.61, 0.62]) == pytest.approx(0.115)
    assert S.ece([1], [1.0]) == pytest.approx(0.0)       # 1.0 entra en el último bin
    assert S.brier([0, 1], [0.2, 0.6]) == pytest.approx((0.04 + 0.16) / 2)


def test_threshold_metrics_use_external_threshold():
    m = S.threshold_metrics([1, 1, 0, 0], [0.7, 0.4, 0.3, 0.6], 0.5)
    assert m["sensibilidad"] == 0.5 and m["especificidad"] == 0.5 and m["exactitud_balanceada"] == 0.5
    b = S.bootstrap_metrics(np.array([1, 1, 0, 0] * 10), np.array([0.7, 0.4, 0.3, 0.6] * 10),
                            np.array(["p%d" % i for i in range(40)]), 0.5, 50, 0)
    assert b["estimacion"]["umbral"] == 0.5 and b["B"] == 50 and set(b["ic95"]) >= {"auc_roc", "ece", "sensibilidad"}


def test_seed_aggregation_and_tests():
    a = {"42": 0.80, "123": 0.82, "2024": 0.81, "7": 0.83, "999": 0.80}
    b = {"42": 0.75, "123": 0.78, "2024": 0.77, "7": 0.76, "999": 0.74}
    s = S.seed_summary(a)
    assert s["media"] == pytest.approx(np.mean(list(a.values()))) and s["dt"] == pytest.approx(np.std(list(a.values()), ddof=1))
    c = S.compare_seed_values(a, b)
    assert c["welch"]["p"] == pytest.approx(sps.ttest_ind(list(a.values()), list(b.values()), equal_var=False,
                                                          alternative="greater").pvalue)
    keys = sorted(a)
    assert c["pareada"]["p"] == pytest.approx(sps.ttest_rel([a[k] for k in keys], [b[k] for k in keys],
                                                            alternative="greater").pvalue)
    c2 = S.compare_seed_values(a, {"1": 0.7, "2": 0.71})
    assert c2["pareada"] is None and np.isfinite(c2["welch"]["p"])
    d1 = _df(["a", "b", "c"], [0, 1, 1], [0.2, 0.6, 0.8])
    d2 = _df(["c", "a", "b"], [1, 0, 1], [0.6, 0.4, 0.8])
    e = S.ensemble([d1, d2])
    assert np.allclose(e["prob"], [0.3, 0.7, 0.7])


# ============================================================================== alineación, lectura y guardas
def test_alignment_rejects_different_ids_and_labels(tmp_path):
    a = _df(["x1", "x2", "x3"], [0, 1, 1], [0.1, 0.5, 0.9])
    with pytest.raises(S.AlignmentError, match="image_id"):
        S.align(a, _df(["x1", "x2", "x4"], [0, 1, 1], [0.1, 0.5, 0.9]))
    with pytest.raises(S.AlignmentError, match="etiquetas"):
        S.align(a, _df(["x1", "x2", "x3"], [0, 1, 0], [0.1, 0.5, 0.9]))
    with pytest.raises(S.AlignmentError):
        S.ensemble([a, _df(["x1", "x2"], [0, 1], [0.1, 0.5])])
    ok = S.align(a, _df(["x3", "x1", "x2"], [1, 0, 1], [0.9, 0.1, 0.5]))
    assert list(ok["image_id"]) == ["x1", "x2", "x3"]


def test_read_predictions_validation_and_test_guard(tmp_path):
    d = tmp_path / "predictions"
    d.mkdir()
    good = _df(["a", "b"], [0, 1], [0.2, 0.9])
    good.to_csv(d / "dfr_best_val_robust_sel.csv", index=False)
    r = S.read_predictions(d / "dfr_best_val_robust_sel.csv", forbid_test=True)
    assert r.attrs["state"] == "dfr_best" and r.attrs["set"] == "val_robust_sel"
    _df(["a", "a"], [0, 1], [0.2, 0.9]).to_csv(d / "best_val.csv", index=False)
    with pytest.raises(S.PredictionFileError):
        S.read_predictions(d / "best_val.csv")
    _df(["a", "b"], [0, 1], [0.2, 1.3]).to_csv(d / "best_val.csv", index=False)
    with pytest.raises(S.PredictionFileError):
        S.read_predictions(d / "best_val.csv")
    for nm in ("best_test.csv", "ema_test_bimcv.csv", "dfr_best_test_rsna_con_fuga.csv"):
        with pytest.raises(S.TestSetForbidden):
            S.read_predictions(d / nm, forbid_test=True)       # error antes de mirar si existe
    with pytest.raises(S.TestSetForbidden):
        S.assert_not_test("TEST_ricord")


def test_level1_list_matches_baseline_jobs():
    mj = _load("scripts/jobs/make_jobs_baselines.py", "make_jobs_baselines")
    assert G3.LEVEL1 == mj.LEVEL1


# ============================================================================== Puerta 3
def _gate(tmp_path, scenario, B=200, **kw):
    res = tmp_path / "train"
    out = tmp_path / "gate3"
    T.make_gate3_tree(res, scenario, **kw)
    base = ["--resultados", str(res), "--salida", str(out), "--sin-mapa-pacientes"]
    assert G3.main(["paso1"] + base) == 0
    rc = G3.main(["paso2"] + base + ["--B", str(B)])
    js = json.loads((out / "informe.json").read_text(encoding="utf-8"))
    return rc, js, base, out


def test_gate3_approves(tmp_path):
    rc, js, base, out = _gate(tmp_path, "aprueba")
    assert rc == 0 and js["regla"]["aprobado"] is True and js["rama"] == "1"
    assert js["regla"] == {"D1": True, "D2": True, "D3": True, "verificaciones": True, "V6": True, "aprobado": True}
    assert js["B"] == 200 and js["semilla_bootstrap"] == 2026 and js["desviaciones_protocolo"]
    assert js["discriminacion"]["D1"]["B"] == 200 and js["discriminacion"]["D1"]["conjunto"] == "val_robust"
    e3 = js["enmienda3_val_robust_fit"]
    assert e3["n_imagenes"] > 0 and "dfr" not in e3["lineas_base"] and "control_fuente" in e3["lineas_base"]
    assert set(js["ablaciones_sin_umbral"]) == {"A14", "A15", "A4", "A2"}
    assert not any(G3.S.is_test_set(G3.S.split_pred_name(k)[1]) for k in js["entradas"] if k.endswith(".csv"))
    md = (out / "informe.md").read_text(encoding="utf-8")
    assert "—" not in md and "§" not in md and "Rama de la Section 6" in md
    ml = json.loads((out / "mejor_linea_base.json").read_text(encoding="utf-8"))
    assert ml["mejor"]["metodo"] == "erm_mascara" and not any("csdnet" in k for k in ml["huellas"])
    # el paso 1 no puede rehacerse después del paso 2
    assert G3.main(["paso1"] + base + ["--sobrescribir"]) == 2


def test_gate3_fails_discrimination(tmp_path):
    rc, js, *_ = _gate(tmp_path, "falla_discriminacion")
    assert rc == 0 and js["regla"]["aprobado"] is False and js["regla"]["D1"] is False and js["rama"] == "3"


def test_gate3_fails_verifications(tmp_path):
    rc, js, *_ = _gate(tmp_path, "falla_verificaciones")
    assert js["regla"]["D1"] and js["regla"]["D2"] and js["regla"]["D3"]
    assert js["regla"]["verificaciones"] is False and js["regla"]["aprobado"] is False and js["rama"] == "2"
    assert js["verificaciones"]["por_semilla"]["123"]["final"]["V2b"] == "falla"


def test_gate3_v6_and_v1_A2_branches(tmp_path):
    rc, js, *_ = _gate(tmp_path / "v6", "falla_v6")
    assert js["V6"]["csd"]["cumple"] is False and js["rama"] == "2-V6" and js["regla"]["verificaciones"]
    rc, js, *_ = _gate(tmp_path / "v1", "falla_v1_A2")
    assert js["verificaciones"]["v1_fin_A2_falla"] and js["rama"] == "3" and js["regla"]["D1"]


def test_gate3_dfr_best_compared_on_val_robust_sel(tmp_path):
    rc, js, *_ = _gate(tmp_path, "aprueba", best_method="dfr", best_strength=0.9)
    assert js["mejor_linea_base"]["metodo"] == "dfr"
    assert js["discriminacion"]["D2"]["conjunto"] == "val_robust_sel" and js["discriminacion"]["D2"]["comparador"] == "dfr"


def test_gate3_refuses_step2_without_step1_or_with_changed_inputs(tmp_path):
    res, out = tmp_path / "train", tmp_path / "gate3"
    T.make_gate3_tree(res, "aprueba")
    base = ["--resultados", str(res), "--salida", str(out), "--sin-mapa-pacientes"]
    assert G3.main(["paso2"] + base + ["--B", "50"]) == 2 and not (out / "informe.json").exists()
    assert G3.main(["paso1"] + base) == 0
    assert G3.main(["paso1"] + base) == 2                              # no sobrescribe sin --sobrescribir
    f = res / "baseline_erm" / "seed42" / "predictions" / "best_val_robust.csv"
    df = pd.read_csv(f)
    df.loc[0, "prob"] = 0.123456
    df.to_csv(f, index=False)
    assert G3.main(["paso2"] + base + ["--B", "50"]) == 2 and not (out / "informe.json").exists()


def test_gate3_never_reads_test(tmp_path):
    res, out = tmp_path / "train", tmp_path / "gate3"
    T.make_gate3_tree(res, "aprueba", test_file_in_csd=True)
    base = ["--resultados", str(res), "--salida", str(out), "--sin-mapa-pacientes"]
    assert G3.main(["paso1"] + base) == 0
    assert G3.main(["paso2"] + base + ["--B", "50", "--conjuntos-adicionales", "test_bimcv"]) == 2
    assert not (out / "informe.json").exists()
    assert G3.main(["paso2"] + base + ["--B", "50", "--conjuntos-adicionales", "val"]) == 0
    js = json.loads((out / "informe.json").read_text(encoding="utf-8"))
    assert any("best_test" in x for x in js["alertas"])                  # avisado, nunca abierto
    assert not any("best_test" in k for k in js["entradas"])
    with pytest.raises(S.TestSetForbidden):
        G3.Lector(res).pred(S.read_run(res / "csdnet_piloto" / "seed42"), "best", "test")


def test_gate3_not_evaluable_when_cache_differs(tmp_path):
    res, out = tmp_path / "train", tmp_path / "gate3"
    T.make_gate3_tree(res, "aprueba")
    base = ["--resultados", str(res), "--salida", str(out), "--sin-mapa-pacientes"]
    assert G3.main(["paso1"] + base) == 0
    cfgp = res / "csdnet_piloto" / "seed123" / "config_resolved.json"
    cfg = json.loads(cfgp.read_text())
    cfg["data"]["cache"] = "data/cache/covidx_cxr4/homogeneo_224"
    cfgp.write_text(json.dumps(cfg))
    assert G3.main(["paso2"] + base + ["--B", "50"]) == 3
    js = json.loads((out / "informe.json").read_text(encoding="utf-8"))
    assert js["evaluable"] is False and js["rama"] == "no_evaluable"


def test_gate3_job_is_staged_and_valid():
    sys.path.insert(0, str(ROOT / "runner"))
    from job_runner import validate_job
    for name, paso in (("p3_13_gate3_mejor_linea_base", "paso1"), ("p3_14_gate3_evaluar", "paso2")):
        # el trabajo estuvo en jobs/staged hasta que se ejecutó o se encoló (13-11-2026)
        cands = [c for d in ("staged", "staged/_ya_encolados", "done")
                 for c in sorted((ROOT / "jobs" / d).glob(name + "*.json"))]
        p = cands[0]
        job = json.loads(p.read_text(encoding="utf-8"))
        job = {k: v for k, v in job.items() if k not in ("intentos", "historial")}      # campos que añade el ejecutor
        assert validate_job(job, name) == []
        assert job["comando"][1:3] == ["scripts/jobs/gate3_evaluar.py", paso]     # p3_14 lleva además --exp-csd
        assert not (ROOT / "jobs" / "pending" / (name + ".json")).exists()
        assert not any(S.is_test_set(x) for x in job["comando"])


# ============================================================================== informe final
SPEC = {"metodos": [{"id": "csdnet", "nombre": "CSD-Net", "experimento": "csdnet_final"},
                    {"id": "control_fuente", "nombre": "Control de fuente", "experimento": "baseline_control_fuente_final"},
                    {"id": "erm", "nombre": "ERM", "experimento": "baseline_erm_final"},
                    {"id": "dfr", "nombre": "DFR", "experimento": "baseline_dfr_final"},
                    {"id": "groupdro", "nombre": "GroupDRO", "experimento": "baseline_groupdro_final"},
                    {"id": "lisa", "nombre": "LISA", "experimento": "baseline_lisa_final"}],
        "ablaciones": [{"id": "A2", "experimento": "csdnet_final_a2"}, {"id": "A4", "experimento": "csdnet_final_a4"},
                       {"id": "A7", "experimento": "csdnet_final_a7"}]}


def _final(tmp_path, locked=False, gate=True):
    res = tmp_path / "train"
    T.make_final_tree(res, tmp_path, methods={"csdnet": 1.4, "control_fuente": 0.9, "erm": 0.8, "dfr": 0.85,
                                              "groupdro": 0.95}, ablations={"a2": 1.1, "a4": 1.35}, locked=locked)
    if not gate:
        (tmp_path / "results" / "gate3" / "informe.json").unlink()
    (tmp_path / "spec.json").write_text(json.dumps(SPEC), encoding="utf-8")
    out = tmp_path / "final"
    rc = FIN.main(["--resultados", str(res), "--gate3", str(tmp_path / "results" / "gate3" / "informe.json"),
                   "--salida", str(out), "--spec", str(tmp_path / "spec.json"), "--B", "60",
                   "--raiz-proyecto", str(tmp_path), "--sin-mapa-pacientes"])
    return rc, out


def test_final_report_generates_all_tables_and_figures(tmp_path):
    rc, out = _final(tmp_path)
    assert rc == 0
    for t in ("tabla_principal", "tabla_conjunto_medio", "tabla_semillas", "tabla_calibracion",
              "tabla_brecha_especificidad", "tabla_ablaciones", "tabla_verificaciones",
              "tabla_otro_protocolo_articulo1"):
        for ext in (".csv", ".md", ".tex"):
            assert (out / "tablas" / (t + ext)).stat().st_size > 0, t + ext
    tex = (out / "tablas" / "tabla_principal.tex").read_text(encoding="utf-8")
    assert "\\toprule" in tex and "\\midrule" in tex and "\\bottomrule" in tex
    for f in ("fig_roc_test_bimcv_limpio", "fig_fiabilidad_test", "fig_bosque_delta_auc", "fig_brecha_especificidad"):
        assert (out / "figuras" / (f + ".pdf")).read_bytes()[:4] == b"%PDF"
        from PIL import Image
        im = Image.open(out / "figuras" / (f + ".png"))
        assert round(im.info["dpi"][0]) == 300
    princ = pd.read_csv(out / "tablas" / "tabla_principal.csv")
    assert set(princ["Conjunto"]) >= {"Test limpio", "BIMCV del test limpio", "RICORD", "RSNA sin fuga", "RSNA con fuga",
                                      "no disponible"}
    # las cifras del artículo 1 solo en su tabla aparte: ninguna fila ni texto de esa configuración en la principal
    assert not princ.astype(str).apply(lambda c: c.str.contains("artículo 1|Ingenuo|otro protocolo")).any().any()
    assert set(princ["Método"]) <= {m["nombre"] for m in SPEC["metodos"]}
    assert "artículo 1" not in (out / "tablas" / "tabla_principal.md").read_text(encoding="utf-8")
    otro = (out / "tablas" / "tabla_otro_protocolo_articulo1.md").read_text(encoding="utf-8")
    assert "Otro protocolo" in otro and "0.723" in otro
    js = json.loads((out / "informe_final.json").read_text(encoding="utf-8"))
    assert js["B"] == 60 and js["semillas"] == [42, 123, 2024, 7, 999] and "lisa" in js["no_disponibles"]
    rp = js["resultado_primario"]
    assert rp["se_declara_mejora"] is True and rp["B"] == 60
    # enmienda 2 de docs/08_evaluacion.md: la cifra principal es la media por semilla (modelo individual) y el
    # conjunto medio de las semillas queda como dato secundario
    assert "media por semilla" in rp["estadistico"] and rp["n_semillas"] == 5 and rp["semillas_a_favor"] == 5
    assert abs(sum(rp["delta_por_semilla"].values()) / 5 - rp["delta_csdnet_menos_control"]) < 1e-12
    assert set(rp["secundario_conjunto_medio"]) == {"delta", "ic95", "p_unilateral", "delong_p_unilateral"}
    assert "media ± dt por semilla" in " ".join(princ.columns) and "AUC-ROC del conjunto (IC95 %)" not in princ.columns
    conj = pd.read_csv(out / "tablas" / "tabla_conjunto_medio.csv")
    assert "AUC-ROC del conjunto (IC95 %)" in conj.columns and "p Holm (DeLong)" in conj.columns
    cc = js["comparaciones"]["test_bimcv_limpio"]
    assert set(cc) == {"control_fuente", "erm", "dfr", "groupdro"}
    ps = {k: v["p_unilateral"] for k, v in cc.items()}
    assert {k: v["p_holm"] for k, v in cc.items()} == {k: v["p_holm"] for k, v in S.holm(ps, 0.025).items()}
    assert all(v["alfa_unilateral"] == 0.025 for v in cc.values())
    ab = pd.read_csv(out / "tablas" / "tabla_ablaciones.csv")
    assert list(ab["Id"]) == ["A2", "A4", "A7"] and ab.loc[2, "AUC-ROC, media ± dt por semilla"] == "no disponible"
    assert "Δ frente a A0, media por semilla (IC95 %)" in ab.columns


def test_final_report_requires_unlock(tmp_path):
    rc, out = _final(tmp_path / "bloq", locked=True)
    assert rc == 2 and not (out / "informe_final.json").exists()
    rc, out = _final(tmp_path / "sin_gate", gate=False)
    assert rc == 2 and not (out / "informe_final.json").exists()


def test_seedmean_paired_test_matches_definition_and_single_seed():
    """Enmienda 2 de docs/08: Δ es la media por semilla de la diferencia de AUC de modelos individuales."""
    rng = np.random.default_rng(11)
    n = 900
    pat = np.repeat(np.arange(n // 3), 3).astype(str)
    y = (rng.random(n) < 0.5).astype(int)
    base = rng.normal(size=n) + y
    P1 = np.stack([base + rng.normal(scale=0.6, size=n) for _ in range(5)])
    P2 = np.stack([base + rng.normal(scale=1.3, size=n) for _ in range(5)])
    t = S.paired_auc_test_seedmean(y, P1, P2, pat, B=150, seed=7, seeds=[42, 123, 2024, 7, 999])
    d = [S.auc_roc(y, P1[k]) - S.auc_roc(y, P2[k]) for k in range(5)]
    assert abs(t["delta"] - float(np.mean(d))) < 1e-12 and t["semillas_a_favor"] == 5 and t["n_semillas"] == 5
    assert list(t["delta_por_semilla"]) == ["42", "123", "2024", "7", "999"]
    assert t["ic95"][0] > 0 and t["p_unilateral"] == pytest.approx(1.0 / (t["B_validos"] + 1))
    # no es la diferencia del conjunto medio (promediar probabilidades cambia la AUC)
    ens = S.auc_roc(y, P1.mean(0)) - S.auc_roc(y, P2.mean(0))
    assert abs(ens - t["delta"]) > 1e-4
    # con una sola semilla coincide exactamente con la prueba pareada simple
    a = S.paired_auc_test(y, P1[0], P2[0], pat, B=120, seed=3, delong=False)
    b = S.paired_auc_test_seedmean(y, P1[:1], P2[:1], pat, B=120, seed=3)
    assert a["delta"] == pytest.approx(b["delta"]) and a["ic95"] == pytest.approx(b["ic95"])
    assert a["p_unilateral"] == pytest.approx(b["p_unilateral"])
    with pytest.raises(S.AlignmentError):
        S.paired_auc_test_seedmean(y, P1, P2[:4], pat, B=10)
