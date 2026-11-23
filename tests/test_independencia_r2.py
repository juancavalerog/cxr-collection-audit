"""Pruebas de la corrección de la penalización de independencia (13-11-2026;
docs/10_correccion_independencia.md). CPU, datos sintéticos, unos 40 s en total.

pytest -q tests/test_independencia_r2.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.losses import (IndConfig, L_ind, L_total, LossWeights, SignatureBank, _rbf_mmd2,  # noqa: E402
                              _rbf_mmd2_unbiased, source_entropy)

MIXED = [0, 1, 2]


# ------------------------------------------------------------------------------------- datos sintéticos
def _data(n, seed, source_scale, assoc):
    """8 fuentes con medias distintas (escala `source_scale`), las 3 primeras mixtas. Con `assoc`, dentro de las
    fuentes mixtas la clase desplaza un rasgo de adquisición (dimensión 0), como un subequipo ligado a la clase."""
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(8, 16, generator=torch.Generator().manual_seed(0)) * source_scale
    s = torch.randint(0, 8, (n,), generator=g)
    y = torch.randint(0, 2, (n,), generator=g)
    x = mu[s] + torch.randn(n, 16, generator=g)
    if assoc:
        x[:, 0] += 2.0 * (y.float() - 0.5) * (s < 3).float()
    return x, s, y


def _batch(s, y, g):
    """Composición del muestreador real a escala: 2 por fuente, 4 por clase en cada fuente mixta, resto al azar."""
    idx = []
    for k in range(8):
        p = torch.nonzero(s == k).flatten()
        idx += p[torch.randint(0, len(p), (2,), generator=g)].tolist()
    for k in MIXED:
        for c in (0, 1):
            p = torch.nonzero((s == k) & (y == c)).flatten()
            idx += p[torch.randint(0, len(p), (4,), generator=g)].tolist()
    idx += torch.randint(0, len(s), (64 - len(idx),), generator=g).tolist()
    return torch.tensor(idx)


def _train(mode, lam, assoc, steps=300, seed=0, source_scale=0.25):
    torch.manual_seed(seed)
    x, s, y = _data(4000, 1, source_scale, assoc)
    xt, st, yt = _data(2000, 2, source_scale, assoc)
    enc = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 32), nn.LayerNorm(32))  # firma con LayerNorm
    head = nn.Linear(32, 8)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), 1e-3, weight_decay=0.05)
    w = LossWeights(lambda_sig=0.0, lambda_seg=0.0, lambda_ind=lam)
    ind = IndConfig(mode=mode)
    bank = SignatureBank(256) if mode == "mmd_banco_tope" else None
    g = torch.Generator().manual_seed(seed)
    ls, hs = [], []
    for _ in range(steps):
        b = _batch(s, y, g)
        a = enc(x[b])
        out = {"logit": torch.zeros(len(b)), "ddp_anchor": torch.zeros(()), "signature": a, "src_logits": head(a)}
        loss, lg = L_total(out, y[b].float(), s[b], w, mixed_sources=MIXED, ind=ind, bank=bank)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if bank is not None:
            bank.update(a, s[b], y[b], MIXED)
        ls.append(lg["L_src"]); hs.append(lg["H_src_lote"])
    with torch.no_grad():
        A = enc(xt).numpy()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    m = (st < 3).numpy()
    Am, ym = A[m], yt.numpy()[m]
    n = len(ym) // 2
    clf = LogisticRegression(max_iter=2000).fit(Am[:n], ym[:n])
    auc = roc_auc_score(ym[n:], clf.predict_proba(Am[n:])[:, 1])
    return {"L_src": float(np.mean(ls[-30:])), "H": float(np.mean(hs[-30:])), "desv": float(A.std(0).mean()),
            "auc": float(auc)}


# ------------------------------------------------------------------------------------- pruebas
def test_modo_antiguo_colapsa_y_modo_nuevo_aprende_la_fuente():
    """El modo del primer intento (MMD sesgada, lambda_ind = 1) reproduce el fallo: firma casi constante y L_src en la
    entropía de las fuentes del lote. El modo nuevo baja L_src con claridad y la firma no colapsa."""
    old = _train("mmd_sesgada", 1.0, assoc=False)
    assert old["desv"] < 0.1, old
    assert abs(old["L_src"] - old["H"]) < 0.05, old
    new = _train("mmd_banco_tope", 10.0, assoc=False)
    assert new["L_src"] < new["H"] - 0.2, new
    assert new["desv"] > 0.5, new


def test_modo_nuevo_reduce_la_clase_en_fuente_mixta():
    """Con la clase ligada a un rasgo de adquisición dentro de las fuentes mixtas, la penalización nueva reduce la AUC
    de clase desde la firma (datos retenidos) respecto de no penalizar, sin dejar de aprender la fuente."""
    none = _train("ninguna", 0.0, assoc=True)
    new = _train("mmd_banco_tope", 10.0, assoc=True)
    assert none["auc"] > 0.75, none
    assert new["auc"] < none["auc"] - 0.05, (none, new)
    assert new["L_src"] < new["H"] - 0.2, new


def test_tope_relativo_del_gradiente_y_valor_del_registro():
    torch.manual_seed(0)
    head = nn.Linear(32, 8)
    s = torch.randint(0, 8, (64,)); s[:24] = torch.tensor(MIXED * 8)
    y = torch.randint(0, 2, (64,))
    a = (torch.randn(64, 32) + 1.5 * y[:, None].float()).requires_grad_()   # clases separadas: L_ind > 0
    w = LossWeights(lambda_sig=0.0, lambda_seg=0.0, lambda_ind=1000.0)
    out = {"logit": torch.zeros(64), "ddp_anchor": torch.zeros(()), "signature": a, "src_logits": head(a)}
    loss, lg = L_total(out, y.float(), s, w, mixed_sources=MIXED, ind=IndConfig(beta=0.5))
    g_tot = torch.autograd.grad(loss, a, retain_graph=True)[0]
    out2 = {"logit": torch.zeros(64), "ddp_anchor": torch.zeros(()), "signature": a, "src_logits": head(a)}
    w0 = LossWeights(lambda_sig=0.0, lambda_seg=0.0, lambda_ind=0.0)
    loss0, _ = L_total(out2, y.float(), s, w0, mixed_sources=MIXED)
    g_src = torch.autograd.grad(loss0, a)[0]
    assert float((g_tot - g_src).norm()) <= 0.5 * float(g_src.norm()) * (1 + 1e-4)
    assert lg["ind_tope_activo"] == 1.0 and lg["ind_razon_grad"] == pytest.approx(0.5, rel=1e-4)
    assert lg["L_total"] == pytest.approx(lg["L_cls_real"] + 0.5 * lg["L_src"] + 1000.0 * lg["L_ind"], rel=1e-5)
    assert lg["H_src_lote"] == pytest.approx(source_entropy(s)) and "desv_firma" in lg and "L_src_menos_H" in lg


def test_modo_mmd_sesgada_es_el_del_primer_intento_y_configuracion():
    torch.manual_seed(1)
    a = torch.randn(64, 16)
    s = torch.randint(0, 8, (64,)); s[:24] = torch.tensor(MIXED * 8)
    y = torch.randint(0, 2, (64,))
    ref, mass = 0.0, 0.0
    for k in MIXED:
        p, q = (s == k) & (y == 1), (s == k) & (y == 0)
        if p.sum() >= 2 and q.sum() >= 2:
            ref += float(p.sum() + q.sum()) * float(_rbf_mmd2(a[p], a[q])); mass += float(p.sum() + q.sum())
    assert float(L_ind(a, s, y, MIXED)) == pytest.approx(ref / mass, rel=1e-6)
    out = {"logit": torch.zeros(64), "ddp_anchor": torch.zeros(()), "signature": a, "src_logits": torch.zeros(64, 8)}
    _, lg = L_total(out, y.float(), s, LossWeights(lambda_sig=0.0, lambda_ind=1.0), mixed_sources=MIXED,
                    ind=IndConfig(mode="mmd_sesgada"))
    assert lg["L_ind"] == pytest.approx(ref / mass, rel=1e-6)
    # el estimador insesgado no tiene el suelo positivo del sesgado con dos muestras iguales en distribución
    x = torch.randn(4, 16); z = torch.randn(4, 16)
    assert float(_rbf_mmd2_unbiased(x, z)) < float(_rbf_mmd2(x, z))
    with pytest.raises(ValueError):
        IndConfig(mode="otro")
    b = SignatureBank(3)
    b.update(torch.randn(10, 4), torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long), [0])
    assert b.get((0, 1)).shape == (3, 4)
    b2 = SignatureBank(1); b2.load_state_dict(b.state_dict())
    assert b2.size == 3 and torch.equal(b2.get((0, 1)), b.get((0, 1)))


# ------------------------------------------------------------------------------------- control de validez
TINY = {
    "experiment": "csd_tiny_firma",
    "data": {"synthetic_csd": {"res": 64}, "eval_sets": ["val", "val_robust", "val_robust_sel"], "eval_batch_size": 32},
    "model": {"csdnet": {"pretrained": False, "transformer_depth": 1, "transformer_init": "random",
                         "backbone": "convnext_atto", "transformer_heads": 4}},
    "train": {"batch_size": 24, "steps_per_epoch": 2, "log_every": 1, "augment": "nuevo",
              "augment_params": {"rotation": 0.03, "zoom": 0.1, "brightness": 0.05, "contrast": [0.9, 1.1], "cutout": 8}},
    "selection": {"metric": "auc_roc", "subset": "val_robust_sel", "impl": "exact", "patience": 10},
    "eval": {"tta": False, "states": ["last"]},
    "thermal": {"every": 50},
    "protocol": {"fase": "piloto", "test_bloqueado": True, "congelado": None},
    "csdnet": {"epochs": {"A1": 1, "A2": 1, "ramp": 1, "B": 1}, "probe_n_boot": 20, "probe_every": 0,
               "final_probe_states": [], "sampler": {"min_per_source": 2, "min_per_class_mixed": 2},
               "probes": {"robust_source": "s0", "v1_extra_sources": ["s3"], "v2_sources": ["s0", "s1", "s2"],
                          "v2b_strata": [[1, "s1", "s0", "s2"], [0, "s2", "s0", "s1"]],
                          "v5_strata": [[0, "s0", "s1"], [0, "s1", "s0"], [1, "s2", "s0"], [1, "s0", "s2"]],
                          "v3_source": "s0", "v3_neg_dest": "s1", "v3_pos_dest": "s2", "v6_sources": ["s1", "s0"],
                          "min_n": 10, "folds": 3}},
}


def test_control_de_validez_deja_v1_no_concluyente(tmp_path):
    """Con dos pasos por época la firma no puede aprender la fuente: el control previo marca "firma sin información
    de fuente", V1 queda no concluyente (no aprobada ni fallida), no se activa la fase B y summary.json lo recoge."""
    from src.train.csdnet_loop import CSDNetTrainer
    c = copy.deepcopy(TINY)
    c["csdnet"]["parar_tras_A2"] = True
    out = tmp_path / "firma"
    tr = CSDNetTrainer(c, 3, out_dir=out, echo=False)
    assert tr.ind_cfg.mode == "mmd_banco_tope" and tr.ind_bank is not None
    assert tr.fit() == 0
    sm = json.loads((out / "summary.json").read_text())
    assert sm["v1"] == "no_concluyente"
    assert sm["flags"]["firma_sin_fuente"] is True and sm["flags"]["v1_motivo"] == "firma sin información de fuente"
    assert sm["flags"]["v1_veredicto_sonda"] in ("aprueba", "falla", "no_concluyente", "no_aplicable")
    assert sm["control_firma"]["valida"] is False and sm["control_firma"]["diferencia_nats"] > -0.05
    assert "firma sin información de fuente" in sm["stopped_reason"]
    assert [h["phase"] for h in sm["history"]] == ["A1", "A2"]                     # no hubo fase B
    ep = [json.loads(l) for l in open(out / "train_log.jsonl") if '"epoch"' in l]
    a2 = [e for e in ep if e.get("event") == "epoch"][-1]
    assert "H_fuente_epoca" in a2 and "L_src_menos_H_epoca" in a2 and "desv_firma" in a2["losses"]
    rep = json.loads((out / "probes" / "V1_fin_A2_ep001.json").read_text())["verificaciones"]
    assert "V1" in rep and "V2" in rep                                              # parar_tras_A2 añade V2
    ck = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ck["ind_bank"] is not None and len(ck["ind_bank"]["mem"]) > 0           # el banco va en el checkpoint


def test_control_de_validez_con_firma_que_reconoce_la_fuente(tmp_path):
    from src.train.csdnet_loop import CSDNetTrainer
    tr = CSDNetTrainer(copy.deepcopy(TINY), 0, out_dir=tmp_path / "v", echo=False)
    ok = tr.signature_validity({"losses": {"L_src": 1.20, "desv_firma": 0.5}, "H_fuente_epoca": 1.77})
    assert ok["valida"] is True and ok["motivo"] is None and ok["diferencia_nats"] == pytest.approx(-0.57)
    bad = tr.signature_validity({"losses": {"L_src": 1.75}, "H_fuente_epoca": 1.77})
    assert bad["valida"] is False and bad["motivo"] == "firma sin información de fuente"
