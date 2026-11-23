"""Pruebas de las enmiendas del 12-11-2026.

1. El profesor de segmentación excluye por defecto a todos los pacientes NIH presentes en COVIDx, de cualquier
   partición, para que la calidad de las pseudomáscaras no dependa de la fuente.
2. csdnet.v1_gate_override (sustituye el veredicto de V1) solo se admite con datos sintéticos de prueba.
3. El valor p del bootstrap pareado es (1 + k) / (B + 1) y nunca vale cero.
4. El plan de encaminamiento se calcula con la composición real de los lotes, no con la tabla del conjunto de datos.
5. El discriminador de DANN y CDANN se entrena con su propio multiplicador de tasa (1), no con el de la cabeza (20).
Ejecutar: pytest -q tests/test_enmiendas_supervisor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.seg import chexmask as CM  # noqa: E402


def _map(tmp_path):
    p = tmp_path / "map.csv"
    pd.DataFrame({"path_covidx": list("abcdef"), "nih_patient_id": ["nih_1", "nih_2", "nih_3", "nih_1", "nih_4", None],
                  "split": ["train", "val", "test", "val", "train", "train"]}).to_csv(p, index=False)
    return p


def test_exclusion_todas_frente_a_eval(tmp_path):
    p = _map(tmp_path)
    assert CM.nih_patients_in_covidx(p) == {"nih_1", "nih_2", "nih_3", "nih_4"}
    assert CM.nih_patients_in_covidx(p, ["val", "test"]) == {"nih_1", "nih_2", "nih_3"}
    assert CM.nih_patients_in_covidx_eval(p) == {"nih_1", "nih_2", "nih_3"}
    assert CM.nih_patients_in_covidx(p, ["train"]) == {"nih_1", "nih_4"}


def test_profesor_excluye_todas_por_defecto(monkeypatch, capsys):
    from src.seg import train_teacher as TT
    seen = {}

    def fake_prepare(a):
        seen["mode"] = "ninguna" if a.no_exclude_covidx_eval else a.exclude_covidx

    monkeypatch.setattr(TT, "prepare", fake_prepare)
    for argv, expected in ((["prepare", "--chexmask-dir", "x"], "todas"),
                           (["prepare", "--chexmask-dir", "x", "--exclude-covidx", "eval"], "eval"),
                           (["prepare", "--chexmask-dir", "x", "--no-exclude-covidx-eval"], "ninguna")):
        monkeypatch.setattr(sys, "argv", ["train_teacher"] + argv)
        TT.main()
        assert seen["mode"] == expected


def test_mapa_real_de_covidx():
    p = ROOT / "data" / "manifests" / "covidx_rsna_to_nih_map.csv"
    if not p.exists():
        pytest.skip("sin el mapa real")
    todas, ev = CM.nih_patients_in_covidx(p), CM.nih_patients_in_covidx_eval(p)
    assert ev < todas and len(todas) == 8007 and len(ev) == 3505


def test_v1_override_rechazado_fuera_de_datos_sinteticos():
    from src.train.csdnet_loop import CSDNetTrainer
    with pytest.raises(ValueError, match="v1_gate_override"):
        CSDNetTrainer({"csdnet": {"v1_gate_override": "aprueba"}, "data": {"mode": "limpia"}}, seed=1)


def test_p_del_bootstrap_nunca_es_cero():
    import numpy as np
    from src.eval import stats as S
    rng = np.random.default_rng(0)
    n = 300
    y = (rng.random(n) < 0.4).astype(int)
    fuerte = y * 3.0 + rng.normal(size=n)
    nulo = rng.normal(size=n)
    g = np.arange(n).astype(str)
    t = S.paired_auc_test(y, fuerte, nulo, g, B=200, seed=1, margin=0.01)
    assert t["p_unilateral"] == pytest.approx(1.0 / (t["B_validos"] + 1))
    assert t["no_inferioridad"]["p_unilateral_margen"] > 0
    t2 = S.paired_auc_test(y, nulo, fuerte, g, B=200, seed=1)
    assert t2["p_unilateral"] == pytest.approx(1.0)


def _tv_efectiva(y, s, K, routing, batch_size, mps, mpc, lam=1.0, n_batches=300, seed=5):
    """Distancia de variación total entre clases de la fuente efectiva (real más re-firmada) en lotes simulados."""
    import numpy as np
    import torch
    from src.model.swap import TableGuidedBatchSampler, sample_swap_pairs
    smp = TableGuidedBatchSampler(y, s, batch_size, n_batches, mps, mpc, seed=seed)
    smp.set_epoch(0)
    g = torch.Generator().manual_seed(0)
    R = torch.tensor(routing, dtype=torch.float32)
    eff = np.zeros((2, K))
    for b in smp:
        b = np.asarray(b)
        pr = sample_swap_pairs(torch.tensor(s[b]), torch.tensor(y[b]), R, generator=g)
        np.add.at(eff, (y[b], s[b]), 1.0)
        v = pr.valid.numpy().astype(bool)
        np.add.at(eff, (y[b][v], pr.target_source.numpy()[v]), lam)
    e = eff / eff.sum(1, keepdims=True)
    return float(0.5 * np.abs(e[0] - e[1]).sum())


def test_plan_con_la_composicion_real_de_los_lotes():
    """Con cuotas en el lote, el plan de la tabla del conjunto deja la fuente efectiva ligada a la clase; el plan
    de la composición real de los lotes la iguala (Proposición 1 de docs/03_arquitectura.md)."""
    import numpy as np
    from src.model.swap import ClassSourceTable
    from src.train.csdnet_loop import batch_composition_table
    # tabla sintética con la forma de COVIDx: fuentes de una sola clase, mixtas y muy desiguales en tamaño
    cells = {(0, 0): 80, (1, 0): 25, (0, 1): 650, (1, 1): 9000, (0, 2): 220, (1, 2): 250, (1, 3): 25, (1, 4): 800,
             (0, 5): 5000, (1, 6): 870, (1, 7): 4000}
    y = np.concatenate([np.full(n, c[0]) for c, n in cells.items()]).astype(np.int64)
    s = np.concatenate([np.full(n, c[1]) for c, n in cells.items()]).astype(np.int64)
    K, names, B, mps, mpc = 8, ["f%d" % k for k in range(8)], 64, 2, 4
    tab = ClassSourceTable.from_arrays(y, s, K, names)
    tb = batch_composition_table(y, s, K, names, B, mps, mpc, seed=1, n_batches=600)
    assert float(np.abs(tab.conditional() - tb.conditional()).max()) > 0.2      # las cuotas cambian mucho P(s | y)
    assert (tb.counts > 0).tolist() == (tab.counts > 0).tolist()               # las celdas vacías siguen vacías
    plan_d, plan_b = tab.routing(1.0), tb.routing(1.0)
    assert plan_b.feasible and plan_b.z < plan_d.z
    tv_d = _tv_efectiva(y, s, K, plan_d.routing, B, mps, mpc)
    tv_b = _tv_efectiva(y, s, K, plan_b.routing, B, mps, mpc)
    assert tv_d > 0.10, tv_d            # el defecto: con el plan de la tabla del conjunto queda dependencia
    assert tv_b < 0.03, tv_b            # la corrección: la fuente efectiva no depende de la clase
    # determinista dada la semilla
    assert (batch_composition_table(y, s, K, names, B, mps, mpc, seed=1, n_batches=600).counts == tb.counts).all()


def test_entrenador_usa_la_tabla_de_los_lotes_por_defecto():
    import importlib.util
    spec = importlib.util.spec_from_file_location("t_loop", str(ROOT / "tests" / "test_csdnet_loop.py"))
    tl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tl)
    from src.train.csdnet_loop import CSD_DEFAULTS, CSDNetTrainer
    assert CSD_DEFAULTS["plan_table"] == "lotes"
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        tr = CSDNetTrainer(tl.cfg(), 3, resume=False, out_dir=d, echo=False)
        assert tr.plan_table is not tr.table and tr.plan is not None
        assert abs(tr.plan.z - float(tr.plan_table.conditional().max(0).sum())) < 1e-9
        tr2 = CSDNetTrainer(tl.cfg(**{"csdnet.plan_table": "dataset"}), 3, resume=False, out_dir=d + "/b", echo=False)
        assert tr2.plan_table is tr2.table


@pytest.mark.parametrize("metodo", ["dann", "cdann"])
def test_discriminador_con_tasa_propia(metodo, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("t_bl", str(ROOT / "tests" / "test_baselines.py"))
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)
    tr = tb.BaselineTrainer(tb.cfg_for(metodo), seed=0, out_dir=tmp_path / "d", echo=False)
    tr.cfg["train"]["optimizer"]["lr_mult"] = {"backbone": 1.0, "head": 20.0}      # como en configs/nuevo_base.yaml
    grupos = {g["name"]: g for g in tr._param_groups()}
    disc = {id(p) for p in tr.model.aux["disc"].parameters()}
    propios = [g for n, g in grupos.items() if n.startswith("aux_propio")]
    assert propios and all(g["lr_mult"] == 10.0 for g in propios)      # disc_lr_mult por defecto
    assert disc == {id(p) for g in propios for p in g["params"]}
    for n, g in grupos.items():
        if not n.startswith("aux_propio"):
            assert not (disc & {id(p) for p in g["params"]}), n
    assert grupos["head"]["lr_mult"] == 20.0


# ---------------------------------------------------------------------------------------------------------------
# DANN y CDANN: rampa de Ganin contada desde que el tronco se entrena y penalización de gradiente del discriminador
def _tb():
    import importlib.util
    spec = importlib.util.spec_from_file_location("t_bl", str(ROOT / "tests" / "test_baselines.py"))
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)
    return tb


def test_rampa_de_ganin_empieza_cuando_el_tronco_se_entrena(tmp_path):
    tb = _tb()
    c = tb.cfg_for("dann")
    c["baseline"]["params"] = {"lambda": 2.0, "schedule": "ganin"}
    tr = tb.BaselineTrainer(c, seed=0, out_dir=tmp_path / "d", echo=False)
    fases = tr.cfg["train"]["phases"]
    assert fases[0].get("train_backbone", True) is False and fases[-1].get("train_backbone", True) is True
    spe = int(tr.steps_per_epoch)
    ini = sum(int(ph["epochs"]) for ph in fases if not ph.get("train_backbone", True)) * spe
    tot = sum(int(ph["epochs"]) for ph in fases) * spe
    for paso in (0, ini // 2, ini):                 # tronco congelado y primer paso del ajuste: sin inversión
        tr.state["global_step"] = paso
        assert tr.method.coeff(tr) == 0.0, paso
    tr.state["global_step"] = ini + max(1, (tot - ini) // 2)
    medio = tr.method.coeff(tr)
    tr.state["global_step"] = tot
    fin = tr.method.coeff(tr)
    assert 0.0 < medio < fin <= 2.0 and fin > 0.999 * 2.0
    c2 = tb.cfg_for("dann")
    c2["baseline"]["params"] = {"lambda": 2.0}
    tr2 = tb.BaselineTrainer(c2, seed=0, out_dir=tmp_path / "e", echo=False)
    tr2.state["global_step"] = 0
    assert tr2.method.coeff(tr2) == 2.0             # constante por defecto
    c3 = tb.cfg_for("dann")
    c3["baseline"]["params"] = {"schedule": "otra"}
    tr3 = tb.BaselineTrainer(c3, seed=0, out_dir=tmp_path / "f", echo=False)
    with pytest.raises(ValueError):
        tr3.method.coeff(tr3)


@pytest.mark.parametrize("metodo", ["dann", "cdann"])
def test_penalizacion_de_gradiente_solo_regulariza_al_discriminador(metodo, tmp_path):
    import torch
    tb = _tb()

    def grads(gp):
        c = tb.cfg_for(metodo)
        c["baseline"]["params"] = {"grad_penalty": gp}
        tr = tb.BaselineTrainer(c, seed=0, out_dir=tmp_path / ("g%s" % gp), echo=False)
        tr.model.eval()                              # sin dropout ni drop path: las dos pasadas son comparables
        g = torch.Generator().manual_seed(0)
        x = torch.randn(8, *next(iter(tr.train_ds))["image"].shape, generator=g)
        y = torch.tensor([0., 1., 0., 1., 0., 1., 0., 1.])
        present = [i for i in range(tr.cells.shape[1]) if tr.cells[:, i].sum() > 0]
        s = torch.tensor([present[i % len(present)] for i in range(8)])
        for p in tr.model.parameters():
            p.requires_grad_(True)
        loss, logs = tr.method.loss(tr, tr.model, x, y, s, tr.method_generator(), 8)
        tr.model.zero_grad()
        loss.backward()
        disc = {id(p) for p in tr.model.aux["disc"].parameters()}
        gd = torch.cat([p.grad.flatten() for p in tr.model.aux["disc"].mlp.parameters()])
        gb = torch.cat([p.grad.flatten() for p in tr.model.parameters() if id(p) not in disc and p.grad is not None])
        return float(loss), logs, gd, gb

    l0, lg0, gd0, gb0 = grads(0.0)
    l1, lg1, gd1, gb1 = grads(10.0)
    assert "grad_pen" not in lg0 and lg1["grad_pen"] > 0 and {"cls_loss", "disc_loss"} <= set(lg1)
    assert abs((l1 - l0) - 10.0 * lg1["grad_pen"]) < 1e-4 * max(1.0, abs(l1))
    assert torch.allclose(gb0, gb1, rtol=1e-4, atol=1e-6)        # el tronco y la cabeza no reciben la penalización
    assert not torch.allclose(gd0, gd1, rtol=1e-4, atol=1e-7)    # el discriminador sí


# ---------------------------------------------------------------------------------------------------------------
# ERM con el muestreador de cuotas de CSD-Net (control del muestreador)
def test_erm_cuotas_usa_los_mismos_lotes_que_csdnet(tmp_path):
    import numpy as np
    from src.baselines.trainer import QuotaInfiniteBatchSampler
    from src.model.swap import TableGuidedBatchSampler
    tb = _tb()
    tr = tb.BaselineTrainer(tb.cfg_for("erm_cuotas"), seed=7, out_dir=tmp_path / "q", echo=False)
    assert tr.quota_sampler == {"min_per_source": 1, "min_per_class_mixed": 2}
    fr = tr.frames["train"]
    y, s = fr["label"].to_numpy().astype(int), fr["source_idx"].to_numpy().astype(int)
    B, spe = int(tr.cfg["train"]["batch_size"]), int(tr.steps_per_epoch)
    it = iter(tr.train_loader(0).batch_sampler)
    lotes = [next(it) for _ in range(2 * spe + 1)]
    ref = TableGuidedBatchSampler(y, s, B, spe, 1, 2, seed=7)
    for e in (0, 1):                                  # mismos lotes que el muestreador de CSD-Net, época a época
        ref.set_epoch(e)
        assert [list(map(int, b)) for b in ref] == lotes[e * spe:(e + 1) * spe]
    mixtas = [k for k in np.unique(s) if len(np.unique(y[s == k])) == 2]
    for b in lotes:
        b = np.asarray(b)
        assert len(b) == B
        for k in np.unique(s):
            assert (s[b] == k).sum() >= 1             # cuota por fuente
        for k in mixtas:
            for c in (0, 1):
                assert ((s[b] == k) & (y[b] == c)).sum() >= 2   # cuota por clase en las fuentes mixtas
    # reanudación: empezar en el lote b reproduce la secuencia desde b
    q = QuotaInfiniteBatchSampler(y, s, B, spe, 1, 2, seed=7, start_batch=spe + 1)
    it2 = iter(q)
    assert [next(it2) for _ in range(3)] == lotes[spe + 1:spe + 4]
    # las demás líneas base no cambian de muestreador
    tr2 = tb.BaselineTrainer(tb.cfg_for("erm"), seed=7, out_dir=tmp_path / "e", echo=False)
    assert getattr(tr2, "quota_sampler", None) is None
    assert type(tr2.train_loader(0).batch_sampler).__name__ == "BalancedInfiniteBatchSampler"


def test_config_de_erm_cuotas():
    from src.baselines.run import load_config
    c = load_config(ROOT / "configs" / "baseline_erm_cuotas.yaml")
    assert c["baseline"]["method"] == "erm_cuotas" and c["train"]["batch_size"] == 64
    cs = load_config(ROOT / "configs" / "csdnet_base.yaml")
    from src.baselines.registry import get_method
    d = get_method("erm_cuotas").defaults
    assert d == {"min_per_source": int(cs["csdnet"]["sampler"]["min_per_source"]),
                 "min_per_class_mixed": int(cs["csdnet"]["sampler"]["min_per_class_mixed"])}
    assert int(cs["train"]["batch_size"]) == 64


@pytest.mark.parametrize("metodo", ["dann", "cdann"])
def test_inversion_con_norma_relativa_a_la_clasificacion(metodo, tmp_path):
    """Con grl_norm = clase, el gradiente adversario que llega a los rasgos tiene norma lambda veces la del gradiente
    de clasificación, sea cual sea la escala de la pérdida del discriminador."""
    import torch
    tb = _tb()

    def normas(lam, escala_disc):
        c = tb.cfg_for(metodo)
        c["baseline"]["params"] = {"grl_norm": "clase", "lambda": lam}
        tr = tb.BaselineTrainer(c, seed=0, out_dir=tmp_path / ("n%s_%s" % (lam, escala_disc)), echo=False)
        tr.model.eval()
        for p in tr.model.parameters():
            p.requires_grad_(True)
        with torch.no_grad():                         # discriminador más o menos confiado: cambia la escala
            for p in tr.model.aux["disc"].mlp.parameters():
                p.mul_(escala_disc)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(8, *next(iter(tr.train_ds))["image"].shape, generator=g)
        y = torch.tensor([0., 1., 0., 1., 0., 1., 0., 1.])
        present = [i for i in range(tr.cells.shape[1]) if tr.cells[:, i].sum() > 0]
        s = torch.tensor([present[i % len(present)] for i in range(8)])
        capt = {}
        orig = tr.model.pooled

        def pooled(xx):
            f = orig(xx)
            f.retain_grad()
            capt["f"] = f
            return f
        tr.model.pooled = pooled
        loss, _ = tr.method.loss(tr, tr.model, x, y, s, tr.method_generator(), 8)
        f = capt["f"]
        from src.baselines.common import head_of
        cls = tr.loss_fn(head_of(tr.model.net, f).float(), y, reduction="sum") / 8
        g_cls = torch.autograd.grad(cls, f, retain_graph=True)[0]
        tr.model.zero_grad()
        f.grad = None                                 # retain_grad acumula también en las llamadas a autograd.grad
        loss.backward()
        g_adv = f.grad - g_cls                        # lo que llega por el discriminador
        return float(g_cls.norm()), float(g_adv.norm())

    for lam in (0.5, 2.0):
        for esc in (1.0, 5.0):
            nc, na = normas(lam, esc)
            assert nc > 0 and abs(na - lam * nc) < 1e-3 * max(1e-6, lam * nc), (lam, esc, nc, na)
    c = tb.cfg_for(metodo)
    c["baseline"]["params"] = {"grl_norm": "otra"}
    tr = tb.BaselineTrainer(c, seed=0, out_dir=tmp_path / "x", echo=False)
    b = next(iter(tr.train_loader(0)))
    with pytest.raises(ValueError):
        tr.method.loss(tr, tr.model, b["image"].float(), b["label"].float(), b["source"].long(), tr.method_generator(), 16)


# ---------------------------------------------------------------------------------------------------------------
# Pseudomáscaras: un solo tensor por lote entre procesos (fallo "received 0 items of ancdata" en el servidor)
def test_pseudomascaras_un_tensor_por_lote(tmp_path):
    import json as _json
    import numpy as np
    import torch
    from src.seg import predict_masks as PM

    def item(i, ok=True):
        return {"image": torch.full((1, 8, 8), float(i)), "image_id": "im%d" % i, "ok": ok,
                "geom": _json.dumps({"k": i} if ok else {"error": "x"})}
    lote = PM._collate([item(0), item(1, ok=False), item(2)])
    tensores = [v for v in lote.values() if torch.is_tensor(v)]
    assert len(tensores) == 1 and lote["image"].shape == (2, 1, 8, 8)
    assert [it["image_id"] for it in lote["ok"]] == ["im0", "im2"] and [it["image_id"] for it in lote["bad"]] == ["im1"]
    assert float(lote["image"][1].mean()) == 2.0                    # el orden de imágenes y metadatos coincide
    vacio = PM._collate([item(3, ok=False)])
    assert vacio["image"] is None and vacio["ok"] == [] and len(vacio["bad"]) == 1
    import pickle
    pickle.dumps(PM._collate)                                       # función de módulo: sirve con procesos lectores


def test_pseudomascaras_con_procesos_lectores_igual_que_sin_ellos(tmp_path):
    """El camino con procesos lectores (el del servidor) da las mismas máscaras que el camino sin ellos."""
    import importlib.util
    import numpy as np
    spec = importlib.util.spec_from_file_location("t_seg", str(ROOT / "tests" / "test_seg.py"))
    ts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ts)
    from src.seg.predict_masks import predict_masks
    df = ts._write_images(tmp_path / "img", n=10)
    a, b = tmp_path / "m0", tmp_path / "m2"
    r0 = predict_masks(ts.BrightTeacher(), 64, df, a, batch=4, workers=0, shard=8, log=lambda *x: None)
    r2 = predict_masks(ts.BrightTeacher(), 64, df, b, batch=4, workers=2, shard=8, log=lambda *x: None)
    assert r0["n"] == r2["n"] == 10
    for p0 in sorted((a / "shards").glob("part_*.npz")):
        with np.load(p0) as z0, np.load(b / "shards" / p0.name) as z2:
            assert z0["image_id"].tolist() == z2["image_id"].tolist()
            assert np.array_equal(z0["masks"], z2["masks"])


# ---------------------------------------------------------------------------------------------------------------
# Seguir con el intercambio tras un fallo de V1: solo con una enmienda citada y nunca con la firma sin fuente
def _tl():
    import importlib.util
    spec = importlib.util.spec_from_file_location("t_loop", str(ROOT / "tests" / "test_csdnet_loop.py"))
    tl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tl)
    return tl


def test_continue_swap_exige_enmienda(tmp_path):
    from src.train.csdnet_loop import CSD_DEFAULTS, CSDNetTrainer
    tl = _tl()
    assert CSD_DEFAULTS["v1_fail_action"] == "stop" and CSD_DEFAULTS["v1_fail_enmienda"] is None
    with pytest.raises(ValueError, match="enmienda"):
        CSDNetTrainer(tl.cfg(**{"csdnet.v1_fail_action": "continue_swap"}), 2, resume=False, out_dir=tmp_path / "a",
                      echo=False)
    with pytest.raises(ValueError, match="desconocido"):
        CSDNetTrainer(tl.cfg(**{"csdnet.v1_fail_action": "seguir"}), 2, resume=False, out_dir=tmp_path / "b", echo=False)


def test_continue_swap_activa_el_intercambio_y_lo_deja_anotado(tmp_path):
    import json
    from src.train.csdnet_loop import CSDNetTrainer
    tl = _tl()
    over = {"csdnet.v1_gate_override": "falla", "csdnet.v1_fail_action": "continue_swap",
            "csdnet.v1_fail_enmienda": "docs/07_piloto.md, enmienda 8", "csdnet.firma_check": {"activo": False}}
    tr = CSDNetTrainer(tl.cfg(**over), 2, resume=False, out_dir=tmp_path / "c", echo=False)
    tr.fit()
    s = json.load(open(tmp_path / "c" / "summary.json"))
    assert s["v1"] == "falla" and s["flags"]["v1_falla_intercambio_activo"] is True
    assert s["flags"]["v1_enmienda"] == "docs/07_piloto.md, enmienda 8"
    assert "v1_fallo_intercambio_desactivado" not in s["flags"] and not s.get("stopped_reason")
    filas = [json.loads(l) for l in open(tmp_path / "c" / "train_log.jsonl") if l.strip()]
    assert any(r.get("event") == "v1_parada" and r.get("accion") == "continue_swap" for r in filas)
    assert any(r.get("event") == "epoch" and r.get("phase") == "B" and r.get("swap") for r in filas)
    # con la regla registrada (stop) el mismo fallo detiene la corrida
    tr2 = CSDNetTrainer(tl.cfg(**{"csdnet.v1_gate_override": "falla", "csdnet.v1_fail_action": "stop"}), 2,
                        resume=False, out_dir=tmp_path / "d", echo=False)
    tr2.fit()
    s2 = json.load(open(tmp_path / "d" / "summary.json"))
    assert s2["flags"].get("v1_fallo_intercambio_desactivado") is True and s2.get("stopped_reason")


# ---------------------------------------------------------------------------------------------------------------
# Puerta 3 con la enmienda 8: V1 informada como no aprobada y el intercambio activo
def test_puerta3_con_v1_informada_es_rama_2_y_lee_el_segundo_intento(tmp_path):
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location("gate3_evaluar", str(ROOT / "scripts" / "jobs" / "gate3_evaluar.py"))
    G3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G3)
    from src.eval import synthetic_tree as T
    res, out = tmp_path / "train", tmp_path / "gate3"
    T.make_gate3_tree(res, "v1_informada", exp_csd="csdnet_piloto_v1_informado")
    base = ["--resultados", str(res), "--salida", str(out), "--sin-mapa-pacientes"]
    assert G3.main(["paso1"] + base) == 0
    # sin indicar el experimento, la Puerta no encuentra corridas de CSD-Net del primer intento y no evalúa
    assert G3.main(["paso2"] + base + ["--B", "200"]) != 0 and not (out / "informe.json").exists()
    assert G3.main(["paso2"] + base + ["--B", "200", "--exp-csd", "csdnet_piloto_v1_informado"]) == 0
    js = json.loads((out / "informe.json").read_text(encoding="utf-8"))
    assert js["experimento_csdnet"] == "csdnet_piloto_v1_informado"
    assert js["regla"]["D1"] and js["regla"]["D2"] and js["regla"]["D3"]
    assert js["regla"]["verificaciones"] is False and js["regla"]["aprobado"] is False
    assert js["verificaciones"]["v1_fin_A2_falla"] is False and js["rama"] == "2"
    assert all(v["v1_informada_enmienda8"] for v in js["verificaciones"]["por_semilla"].values())
    assert any("enmienda 8" in a for a in js["alertas"])
    md = (out / "informe.md").read_text(encoding="utf-8")
    assert "—" not in md and "§" not in md
