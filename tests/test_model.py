"""Pruebas unitarias de CSD-Net (src/model). Ejecutar desde la raíz de paper_new: pytest -q tests/test_model.py

No descargan pesos (pretrained=False) y usan lote 2 en CPU; en GPU usan la primera disponible.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import (  # noqa: E402
    AcquisitionInput,
    AnatomicalGate,
    ClassSourceTable,
    CSDNet,
    CSDNetConfig,
    LossWeights,
    ReSignature,
    TableGuidedBatchSampler,
    L_cls_real,
    L_cls_swap,
    L_cons,
    L_total,
    sample_swap_pairs,
    sample_within_pairs,
)

TABLE_CSV = ROOT / "data" / "manifests" / "covidx_cxr4_clase_fuente.csv"
DOC = ROOT / "docs" / "03_arquitectura.md"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B = 2


def synthetic_lung_mask(b: int, res: int) -> torch.Tensor:
    """Dos elipses pulmonares sintéticas centradas, forma (b, 1, res, res)."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, res), torch.linspace(-1, 1, res), indexing="ij")
    left = ((xx + 0.35) / 0.25) ** 2 + (yy / 0.5) ** 2 <= 1
    right = ((xx - 0.35) / 0.25) ** 2 + (yy / 0.5) ** 2 <= 1
    return (left | right).float().view(1, 1, res, res).repeat(b, 1, 1, 1)


def synthetic_heart_mask(b: int, res: int) -> torch.Tensor:
    """Silueta cardiaca sintética entre los pulmones, sin solaparlos, forma (b, 1, res, res)."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, res), torch.linspace(-1, 1, res), indexing="ij")
    heart = ((xx / 0.08) ** 2 + ((yy - 0.2) / 0.25) ** 2) <= 1
    return heart.float().view(1, 1, res, res).repeat(b, 1, 1, 1)


def make_model(res: int = 224, seed: int = 0, **kw) -> CSDNet:
    torch.manual_seed(seed)
    return CSDNet(CSDNetConfig(pretrained=False, img_size=res, **kw)).to(DEV)


@pytest.fixture(scope="module")
def model224() -> CSDNet:
    return make_model(224)


def batch(res: int, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, 3, res, res, generator=g).to(DEV)
    y = torch.tensor([1, 0], device=DEV)
    s = torch.tensor([1, 5], device=DEV)  # bimcv positivo, rsna negativo
    mask = torch.cat([synthetic_lung_mask(B, res), synthetic_heart_mask(B, res)], dim=1)
    return x, y, s, mask.to(DEV)


# ------------------------------------------------------------------------------------------ formas
@pytest.mark.parametrize("res", [224, 384])
def test_output_shapes(res: int) -> None:
    m = make_model(res).train()
    x, y, s, mask = batch(res)
    out = m(x, swap_index=torch.tensor([1, 0], device=DEV), within_index=torch.tensor([0, 1], device=DEV),
            lung_mask=mask)
    t = res // 16
    expected = {
        "logit": (B,), "logit_swap": (B,), "src_logits": (B, 8), "seg_logits": (B, 2, res // 4, res // 4),
        "logit_within": (B,), "emb_within_gated": (B, 384), "emb_within_decision": (B, 128), "emb_swap_gated": (B, 384),
        "signature": (B, 64), "emb_content": (B, 384), "emb_tokens": (B, 384), "emb_gated": (B, 384),
        "emb_decision": (B, 128), "emb_swap_content": (B, 384), "emb_swap_decision": (B, 128),
        "stats_real": (B, 768), "stats_swap": (B, 768), "sig_pred": (B, 768), "token_weights": (B, t * t),
    }
    for k, shape in expected.items():
        assert tuple(out[k].shape) == shape, (k, tuple(out[k].shape))
        assert torch.isfinite(out[k]).all(), k


# ------------------------------------------------------------------------------------------ gradiente
def test_gradient_reaches_all_groups(model224: CSDNet) -> None:
    m = model224.train()
    m.zero_grad(set_to_none=True)
    x, y, s, mask = batch(224)
    out = m(x, swap_index=torch.tensor([1, 0], device=DEV), within_index=torch.tensor([0, 1], device=DEV),
            lung_mask=mask)
    valid = torch.ones(B, dtype=torch.bool, device=DEV)
    loss, logs = L_total(out, y, s, LossWeights(), valid=valid, mask=mask, mixed_sources=[1], ramp=1.0)
    loss.backward()
    for name, params in m.param_groups().items():
        norm = sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)
        assert norm > 0, f"sin gradiente en {name}"
    for key in ("L_cls_real", "L_cls_swap", "L_cons", "L_src", "L_seg", "L_sig", "L_total"):
        assert key in logs


def test_gradient_cuts(model224: CSDNet) -> None:
    """L_cls_real, L_cls_swap y L_cons (con las tres vistas) no deben mover la vía de adquisición ni el
    modulador (Section 2.4)."""
    m = model224.train()
    m.zero_grad(set_to_none=True)
    x, y, s, mask = batch(224)
    out = m(x, swap_index=torch.tensor([1, 0], device=DEV), within_index=torch.tensor([0, 1], device=DEV),
            lung_mask=mask)
    valid = torch.ones(B, dtype=torch.bool, device=DEV)
    w = LossWeights()
    (L_cls_swap(out, y, valid, w) + L_cons(out, valid) + L_cls_real(out, y, w)).backward()
    groups = m.param_groups()
    for name in ("acq_encoder", "modulator", "src_head"):
        norm = sum(float(p.grad.abs().sum()) for p in groups[name] if p.grad is not None)
        assert norm == 0.0, f"fuga de gradiente hacia {name}"
    assert sum(float(p.grad.abs().sum()) for p in groups["transformer"] if p.grad is not None) > 0


def test_all_params_used_without_swap(model224: CSDNet) -> None:
    """Sin intercambio (calentamiento) todos los parámetros reciben gradiente: DDP sin find_unused_parameters."""
    m = model224.train()
    m.zero_grad(set_to_none=True)
    x, y, s, mask = batch(224)
    out = m(x, swap_index=None, lung_mask=mask)
    loss, _ = L_total(out, y, s, LossWeights(lambda_sig=0.0), mask=mask)
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing[:5]


# ------------------------------------------------------------------------------------------ re-firma
def test_reconstruction_with_own_signature(model224: CSDNet) -> None:
    layer = ReSignature("relative")
    f = torch.randn(B, 384, 14, 14) * 3 + 1
    g, b = torch.randn(B, 384), torch.randn(B, 384)
    assert torch.allclose(layer(f, g, b, g, b), f, atol=1e-4, rtol=1e-5)
    m = model224.eval()
    x, _, _, mask = batch(224)
    with torch.no_grad():
        out = m(x, swap_index=torch.arange(B, device=DEV), lung_mask=mask)
    assert torch.allclose(out["emb_swap_content"], out["emb_content"], atol=1e-4)
    assert torch.allclose(out["logit_swap"], out["logit"], atol=1e-4)
    assert torch.allclose(out["stats_swap"], out["stats_real"], atol=1e-3)
    with torch.no_grad():
        out_w = m(x, within_index=torch.arange(B, device=DEV), lung_mask=mask)
    assert torch.allclose(out_w["logit_within"], out_w["logit"], atol=1e-4)


def test_swap_changes_statistics(model224: CSDNet) -> None:
    """Con la firma de otra imagen los estadísticos de canal se mueven en la dirección predicha (ecuación 20)."""
    layer = ReSignature("relative", log_scale_clamp=10.0)
    f = torch.randn(B, 8, 6, 6)
    g = torch.zeros(B, 8)
    b = torch.zeros(B, 8)
    g[1], b[1] = 0.5, 2.0
    out = layer(f, g, b, g[[1, 0]], b[[1, 0]])
    mu0, mu1 = f[0].mean(dim=(1, 2)), out[0].mean(dim=(1, 2))
    assert torch.allclose(mu1 - mu0, torch.full_like(mu0, 2.0), atol=1e-4)
    sd0, sd1 = f[0].std(dim=(1, 2), unbiased=False), out[0].std(dim=(1, 2), unbiased=False)
    assert torch.allclose(sd1 / sd0, torch.full_like(sd0, float(np.exp(0.5))), atol=1e-3)


# ------------------------------------------------------------------------------------------ muestreador
def _covidx_batch_arrays(table: ClassSourceTable):
    """Lote sintético con todas las celdas no vacías de la tabla real: 3 imágenes por celda."""
    srcs, labs = [], []
    for y in (0, 1):
        for k in range(table.num_sources):
            if table.counts[y, k] > 0:
                srcs += [k] * 3
                labs += [y] * 3
    return torch.tensor(srcs), torch.tensor(labs)


def test_swap_sampler_never_same_source_and_prioritizes_empty_cells() -> None:
    table = ClassSourceTable.from_csv(str(TABLE_CSV), "train")
    assert table.counts[1, table.index("rsna")] == 0 and table.counts[0, table.index("stonybrook")] == 0
    plan = table.routing(lambda_swap=1.0)
    assert plan.feasible and plan.residual < 1e-8
    p = table.conditional()
    p_eff = (p + plan.lambda_swap * plan.pi) / (1 + plan.lambda_swap)
    assert np.allclose(p_eff[0], p_eff[1], atol=1e-8)  # Proposición 2: fuente efectiva independiente de la clase
    assert np.allclose(np.diagonal(plan.routing, axis1=1, axis2=2), 0.0)
    routing = torch.tensor(plan.routing)
    s, y = _covidx_batch_arrays(table)
    types = table.cell_types()
    gen = torch.Generator().manual_seed(0)
    counts = {"vacia": 0, "escasa": 0, "completa": 0}
    dest_pos_bimcv = np.zeros(table.num_sources)
    for _ in range(300):
        pairs = sample_swap_pairs(s, y, routing, gen)
        assert pairs.valid.all()
        assert (s[pairs.donor_index] != s).all()
        assert (s[pairs.donor_index] == pairs.target_source).all()
        for i in range(len(s)):
            counts[types[int(y[i]), int(pairs.target_source[i])]] += 1
            if int(s[i]) == table.index("bimcv") and int(y[i]) == 1:
                dest_pos_bimcv[int(pairs.target_source[i])] += 1
    assert counts["vacia"] > counts["completa"]
    assert counts["vacia"] + counts["escasa"] > 0.6 * sum(counts.values())
    assert int(np.argmax(dest_pos_bimcv)) == table.index("rsna")


def test_swap_sampler_invalid_when_single_source() -> None:
    routing = torch.full((2, 3, 3), 0.5)
    pairs = sample_swap_pairs(torch.tensor([2, 2]), torch.tensor([0, 1]), routing)
    assert not pairs.valid.any()
    assert (pairs.donor_index == torch.arange(2)).all()


def test_batch_sampler_quotas_and_ranks() -> None:
    table = ClassSourceTable.from_csv(str(TABLE_CSV), "train")
    rng = np.random.default_rng(0)
    labels, sources = [], []
    for y in (0, 1):
        for k in range(table.num_sources):
            n = int(min(table.counts[y, k], 40))
            labels += [y] * n
            sources += [k] * n
    labels, sources = np.array(labels), np.array(sources)
    smp0 = TableGuidedBatchSampler(labels, sources, batch_size=64, num_batches=3, seed=7, rank=0, world_size=2)
    smp1 = TableGuidedBatchSampler(labels, sources, batch_size=64, num_batches=3, seed=7, rank=1, world_size=2)
    b0, b1 = list(smp0), list(smp1)
    assert b0 != b1
    assert b0 == list(TableGuidedBatchSampler(labels, sources, 64, 3, seed=7, rank=0, world_size=2))
    for bt in b0:
        assert len(bt) == 64
        assert set(sources[bt]) == set(range(table.num_sources))
        for k in table.mixed_sources():
            assert set(labels[np.array(bt)[sources[bt] == k]]) == {0, 1}


# ------------------------------------------------------------------------------------------ adquisición
def test_acquisition_input_masks_lungs(model224: CSDNet) -> None:
    res = 224
    mask = synthetic_lung_mask(B, res)
    gray = torch.rand(B, 1, res, res)
    acq = AcquisitionInput()
    x_acq, keep = acq(gray, mask)
    lung_half = torch.nn.functional.max_pool2d(mask, 2)
    assert x_acq.shape == (B, 7, res // 2, res // 2)
    assert (x_acq * lung_half).abs().max() == 0.0
    assert (keep * lung_half).max() == 0.0
    # la firma no cambia si se altera el contenido pulmonar
    m = model224.eval()
    x = torch.randn(B, 3, res, res, device=DEV)
    x2 = x.clone()
    lm = mask.to(DEV).bool().expand(-1, 3, -1, -1)
    x2[lm] = x2[lm] + 5.0 * torch.randn(int(lm.sum()), device=DEV)
    with torch.no_grad():
        a1 = m.signature(x, mask.to(DEV))
        a2 = m.signature(x2, mask.to(DEV))
    assert torch.allclose(a1, a2, atol=1e-5)


# ------------------------------------------------------------------------------------------ compuerta
def test_anatomical_gate_nulls_periphery(model224: CSDNet) -> None:
    gate = AnatomicalGate(threshold=0.1)
    grid = (14, 14)
    m = torch.zeros(B, 1, 224, 224)
    m[..., 48:176, 32:192] = 1.0
    w = gate.token_weights(m, grid)
    wmap = w.view(B, *grid)
    assert wmap[:, 0, :].abs().max() == 0 and wmap[:, :, 0].abs().max() == 0
    assert wmap[:, 7, 7].min() == 1.0
    tok = torch.randn(B, 196, 384)
    tok2 = tok.clone()
    per = (w == 0)
    tok2[per] = tok2[per] + 100.0 * torch.randn(int(per.sum()), 384)
    assert torch.allclose(gate.pool(tok, w), gate.pool(tok2, w), atol=1e-5)
    # en el modelo, la máscara externa fija los pesos por token
    mdl = model224.eval()
    x, _, _, _ = batch(224)
    with torch.no_grad():
        out = mdl(x, gate_mask=m.to(DEV), compute_signature=False)
    assert torch.equal(out["token_weights"].cpu(), w)


# ------------------------------------------------------------------------------------------ determinismo
def test_determinism_with_seed() -> None:
    m1, m2 = make_model(224, seed=123).eval(), make_model(224, seed=123).eval()
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2), n1
    x, _, _, mask = batch(224)
    with torch.no_grad():
        o1 = m1(x, swap_index=torch.tensor([1, 0], device=DEV), lung_mask=mask)
        o2 = m2(x, swap_index=torch.tensor([1, 0], device=DEV), lung_mask=mask)
    for k in ("logit", "logit_swap", "signature", "seg_logits"):
        assert torch.equal(o1[k], o2[k]), k
    table = ClassSourceTable.from_csv(str(TABLE_CSV))
    s, y = _covidx_batch_arrays(table)
    r = torch.tensor(table.routing().routing)
    pa = sample_swap_pairs(s, y, r, torch.Generator().manual_seed(5))
    pb = sample_swap_pairs(s, y, r, torch.Generator().manual_seed(5))
    assert torch.equal(pa.donor_index, pb.donor_index)


# ------------------------------------------------------------------------------------------ AMP
def test_forward_backward_autocast(model224: CSDNet) -> None:
    m = model224.train()
    m.zero_grad(set_to_none=True)
    x, y, s, mask = batch(224)
    dtype = torch.float16 if DEV.type == "cuda" else torch.bfloat16
    with torch.autocast(device_type=DEV.type, dtype=dtype):
        out = m(x, swap_index=torch.tensor([1, 0], device=DEV), lung_mask=mask)
        loss, _ = L_total(out, y, s, LossWeights(), valid=torch.ones(B, dtype=torch.bool, device=DEV), mask=mask,
                          mixed_sources=[1])
    assert torch.isfinite(loss)
    for k in ("logit", "logit_swap", "src_logits", "seg_logits"):
        assert torch.isfinite(out[k].float()).all(), k
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


# ------------------------------------------------------------------------------------------ documento
def _doc_int(label: str) -> int:
    text = DOC.read_text(encoding="utf-8")
    mt = re.search(re.escape(label) + r"[^:\n]*:\s*\**([\d ]+)", text)
    assert mt, f"no se encuentra '{label}' en {DOC}"
    return int(mt.group(1).replace(" ", ""))


def test_param_count_matches_document(model224: CSDNet) -> None:
    total = sum(p.numel() for p in model224.parameters())
    infer = sum(p.numel() for n, p in model224.named_parameters() if not n.startswith(("acq_", "src_head", "modulator")))
    assert total == _doc_int("Parámetros totales de CSD-Net")
    assert infer == _doc_int("Parámetros de la vía de inferencia")


def test_losses_named_in_document() -> None:
    text = DOC.read_text(encoding="utf-8")
    for name in ("L_cls_real", "L_cls_swap", "L_cons", "L_src", "L_seg", "L_sig", "L_ind", "L_total"):
        assert re.search(r"`" + name + r"`[^\n]*\(\d+\)", text), f"{name} sin ecuación numerada en el documento"


# ------------------------------------------------------------------------------------------ vista dentro de la fuente
def test_within_pairs_same_source_distinct() -> None:
    s = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3])
    gen = torch.Generator().manual_seed(0)
    for _ in range(50):
        wp = sample_within_pairs(s, gen)
        assert (s[wp.donor_index] == s).all()
        assert wp.distinct.tolist() == [True, True, True, True, True, False, True, True]
        idx = torch.arange(len(s))
        assert (wp.donor_index[wp.distinct] != idx[wp.distinct]).all()
        assert wp.donor_index[5] == 5


def test_three_views_and_losses(model224: CSDNet) -> None:
    """Las vistas real, dentro de la fuente y cruzada pasan por la misma etapa y la misma cabeza; L_cls_real
    combina real y dentro con pesos 1 - rho y rho (ecuación 27)."""
    m = model224.eval()
    x, y, s, mask = batch(224)
    s_eq = torch.tensor([1, 1], device=DEV)
    wp = sample_within_pairs(s_eq, torch.Generator().manual_seed(0))
    with torch.no_grad():
        out = m(x, swap_index=torch.tensor([1, 0], device=DEV), within_index=wp.donor_index, lung_mask=mask)
    assert not torch.allclose(out["logit_within"], out["logit"])
    w = LossWeights(rho_within=0.25)
    from src.model.losses import focal_bce
    ref = 0.75 * focal_bce(out["logit"], y) + 0.25 * focal_bce(out["logit_within"], y)
    assert torch.allclose(L_cls_real(out, y, w), ref)
    no_within = {k: v for k, v in out.items() if "within" not in k}
    assert torch.allclose(L_cls_real(no_within, y, w), focal_bce(out["logit"], y))


# ------------------------------------------------------------------------------------------ región de la compuerta
def test_gate_region_configurable() -> None:
    res = 224
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, res), torch.linspace(-1, 1, res), indexing="ij")
    lungs = (((xx.abs() - 0.55) / 0.25) ** 2 + (yy / 0.6) ** 2 <= 1).float().view(1, 1, res, res)
    heart = ((xx / 0.2) ** 2 + ((yy - 0.2) / 0.4) ** 2 <= 1).float().view(1, 1, res, res)
    assert float((lungs * heart).sum()) == 0.0
    prob = torch.cat([lungs, heart], dim=1).to(DEV)
    x = torch.randn(1, 3, res, res, device=DEV)
    pool = torch.nn.functional.adaptive_avg_pool2d
    heart_tok = (pool(heart, (14, 14)).flatten(1) > 0.99) & (pool(lungs, (14, 14)).flatten(1) == 0)
    lung_tok = (pool(lungs, (14, 14)).flatten(1) > 0.99) & (pool(heart, (14, 14)).flatten(1) == 0)
    corner = torch.zeros(1, 196, dtype=torch.bool)
    corner[0, 0] = True
    ws = {}
    for region in ("lungs", "lungs_heart", "thorax"):
        mdl = make_model(224, gate_region=region).eval()
        with torch.no_grad():
            ws[region] = mdl(x, gate_mask=prob, compute_signature=False)["token_weights"].cpu()
    assert heart_tok.any() and lung_tok.any()
    assert ws["lungs"][heart_tok].max() == 0.0 and ws["lungs"][lung_tok].min() == 1.0
    assert ws["lungs_heart"][heart_tok].min() == 1.0 and ws["lungs_heart"][lung_tok].min() == 1.0
    assert ws["thorax"][heart_tok].min() == 1.0 and ws["thorax"][corner].max() == 0.0
    with pytest.raises(ValueError):
        CSDNetConfig(pretrained=False, gate_region="lungs_heart", seg_channels=1)


# ------------------------------------------------------------------------------------------ ablación canónica
def test_canonical_refirma_ablation() -> None:
    m = make_model(224, refirma_target="canonical").train()
    x, y, s, mask = batch(224)
    out = m(x, swap_index=torch.tensor([1, 0], device=DEV), lung_mask=mask)
    assert "logit_swap" not in out and bool(m.ref_init)
    ref_g = m.ref_g.clone()
    m.eval()
    with torch.no_grad():
        o1 = m(x, lung_mask=mask)
    assert torch.equal(m.ref_g, ref_g)  # en evaluación la referencia no cambia
    assert torch.isfinite(o1["logit"]).all()
    with pytest.raises(ValueError):
        m(x, lung_mask=mask, compute_signature=False)
