"""Pruebas del entrenador sobre los benchmarks del plan B (B3a y B2), en CPU (docs/13_entrenamiento_plan_b.md).

Usan manifiestos reducidos que salen de los reales (pocas filas por celda), una caché y unas máscaras sintéticas en
el directorio temporal y el modelo diminuto `tiny` (sin pesos preentrenados, solo admitido con protocol.fase:
prueba). Ningún conjunto de test se carga ni se evalúa.

Ejecutar desde la raíz de paper_new: pytest -q tests/test_train_plan_b.py --basetemp=/tmp/tpb
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "jobs"))
sys.path.insert(0, str(ROOT / "runner"))

from src.baselines.run import load_config  # noqa: E402
from src.baselines.trainer import BaselineTrainer, TestLockedError  # noqa: E402
from src.data import bench as DB  # noqa: E402
from src.train import bench as BM  # noqa: E402
from src.train.config import DEFAULTS, load_yaml, resolve  # noqa: E402

METODOS = ["erm", "control_fuente", "groupdro", "dann", "coral", "mixstyle", "fourier", "lisa", "dfa_fuente", "dfr",
           "erm_mascara", "erm_cuotas"]
RES = 32


# ====================================================================== utilidades
def manifiesto_real(bench="b3a", rho="0.90", seed=0, orient="apos"):
    if bench == "b3a":
        return ROOT / "data/manifests/bench/b3a" / orient / ("rho%s_s%d.csv" % (rho, seed))
    return ROOT / "data/manifests/bench/b2" / ("b2_varB_s%d.csv" % seed)


def make_bench(tmp, bench="b3a", rho="0.90", seed=0, k=4, orient="apos"):
    """Manifiesto reducido: k filas por (fuente, clase) de train, de val y de cada subconjunto (las mismas imágenes de
    val_desconf y test_desconf en todos los niveles y semillas, como en los reales), con caché y máscaras sintéticas."""
    from PIL import Image
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    src = manifiesto_real(bench, rho, seed, orient)
    df = DB.read_csv(src)
    meta = DB.read_meta(src)
    parts, chosen = [], {}
    for sp in ("train", "val"):
        parts.append(df[df["split"] == sp].sort_values("image_id").groupby(["source", "label"]).head(k))
    for s in meta["subconjuntos"]:
        d = df[df[s].astype(int) == 1].sort_values("image_id").groupby(["source", "label"]).head(k)
        chosen[s] = set(d["image_id"])
        parts.append(d)
    red = pd.concat(parts).drop_duplicates("image_id").reset_index(drop=True)
    for s in meta["subconjuntos"]:
        red[s] = red["image_id"].isin(chosen[s]).astype(int)
    tag = "%s_%s_%s_s%d" % (bench, orient, rho, seed)
    out = tmp / (tag + ".csv")
    DB.write_bench(red, meta, out)
    cdir, mdir = tmp / "cache", tmp / "masks"
    cdir.mkdir(exist_ok=True)
    mdir.mkdir(exist_ok=True)
    rows, mrows = [], []
    fuentes = meta["fuentes"]
    for r in red.itertuples():
        rel = hashlib.md5(r.image_id.encode()).hexdigest() + ".png"
        if not (cdir / rel).exists():
            g = np.random.default_rng(int(rel[:6], 16))
            x = g.normal(110, 25, (RES, RES))
            c = RES // 4
            x[c:3 * c, c:3 * c] += 60 if int(r.label) else -60
            x += 15 * fuentes.index(r.source)                 # firma de fuente
            Image.fromarray(np.clip(x, 0, 255).astype(np.uint8)).save(cdir / rel)
            m = np.zeros((RES, RES), np.uint8)
            m[2:-2, 2:-2] = 255
            Image.fromarray(m).save(mdir / rel)
        rows.append({"image_id": r.image_id, "rel_path": rel, "status": "ok"})
        mrows.append({"image_id": r.image_id, "path": rel})
    for f, rr in ((cdir / "index.csv", rows), (mdir / "index.csv", mrows)):
        old = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=list(rr[0]))
        pd.concat([old, pd.DataFrame(rr)]).drop_duplicates("image_id").to_csv(f, index=False)
    return out, cdir, mdir, red, meta


def _set(cfg, k, v):
    d = cfg
    ks = k.split(".")
    for kk in ks[:-1]:
        d = d.setdefault(kk, {})
    d[ks[-1]] = copy.deepcopy(v)


FASES = [{"name": "A", "epochs": 1, "lr": 1e-2, "warmup_frac": 0.1, "train_backbone": False, "ema": False,
          "early_stopping": False},
         {"name": "B", "epochs": 2, "lr": 1e-3, "warmup_frac": 0.1, "train_backbone": True, "ema": True,
          "early_stopping": True}]


def cpu_cfg(method, bench, csv, cdir, mdir, **over):
    cfg = load_config(ROOT / ("configs/%s_%s.yaml" % (bench, method)))
    sets = {"model.name": "tiny", "protocol.fase": "prueba", "data.bench": str(csv), "data.cache": str(cdir),
            "data.num_workers": 0, "data.eval_batch_size": 16, "train.batch_size": 16, "train.micro_batch": 16,
            "train.log_every": 1000, "thermal.every": 1000, "train.augment": "none", "train.phases": FASES}
    if method == "erm_mascara":
        sets["baseline.params"] = {"mask_dir": str(mdir)}
    if method == "erm_cuotas":
        sets["baseline.params"] = {"min_per_source": 1, "min_per_class_mixed": 2}
        sets["train.batch_size"] = sets["train.micro_batch"] = 12     # cuota 10 en rho < 1; resto por clase <= celda
    if method == "dfa_fuente":
        sets["baseline.params"] = {"swap_start_epoch": 0}
    if method == "dfr":
        sets["baseline.params"] = {"states": ["best", "bestdc"], "n_reps": 2, "C_grid": [1.0, 0.1]}
    sets.update(over)
    for k, v in sets.items():
        _set(cfg, k, v)
    return cfg


def run(cfg, out, seed=0):
    tr = BaselineTrainer(cfg, seed=seed, out_dir=out, echo=False)
    return tr, tr.fit()


def preds(out):
    p = Path(out) / "predictions"
    return sorted(os.listdir(p)) if p.exists() else []


@pytest.fixture(scope="module")
def b3a(tmp_path_factory):
    d = tmp_path_factory.mktemp("b3a")
    return {r: make_bench(d, rho=r) for r in ("0.90", "1.00")}


# ====================================================================== COVIDx sin cambios
# Huellas de resolve(load_yaml(f)) calculadas con el código anterior a este cambio (copia en
# logs/entrenamiento_plan_b/backup_pre_12/src/train/config.py): las configuraciones de COVIDx se resuelven igual.
HUELLAS_COVIDX = {
    "baseline_base.yaml": "d967ff0aacbb0fc1", "baseline_cdann.yaml": "11a2baceb18bebdb",
    "baseline_control_fuente.yaml": "1a820d1ce900c87e", "baseline_coral.yaml": "f09e7c633fc1dfc9",
    "baseline_dann.yaml": "7546c51a524e3009", "baseline_dfa_fuente.yaml": "eb737e853f8f6035",
    "baseline_dfr.yaml": "ac2d2f593618b034", "baseline_erm.yaml": "f8e977b3c2f25c06",
    "baseline_erm_cuotas.yaml": "0e196a76f207de5f", "baseline_erm_mascara.yaml": "242a860fadb15d07",
    "baseline_fourier.yaml": "21f9cd28151c33a5", "baseline_groupdro.yaml": "d5e3222367d90e46",
    "baseline_irm.yaml": "5d81343c8b9c96d6", "baseline_lai2026.yaml": "9aea8a08a2d8469b",
    "baseline_lisa.yaml": "7c5fa147ed5669db", "baseline_mixstyle.yaml": "2c67f4d675b1f14e",
    "baseline_pbbl.yaml": "ba9bb07bd25dcf88", "baseline_rad_dino_probe.yaml": "1a909a69e5fd8a0d",
    "csdnet_base.yaml": "48f8b580aadfe647", "csdnet_diag_firma.yaml": "5e6fe3b975066087",
    "csdnet_piloto.yaml": "2543ebb5da706c69", "csdnet_piloto_a14_canonica.yaml": "238abd0c906c1515",
    "csdnet_piloto_a15_sin_dentro.yaml": "cc5e2c65eef09068", "csdnet_piloto_a2_aleatorio.yaml": "4472bcb9364f60a7",
    "csdnet_piloto_a4_sin_compuerta.yaml": "ff685754e2ec0a6c", "csdnet_piloto_r1_mmd_sesgada.yaml": "c768f358706cabe1",
    "csdnet_piloto_v1_informado.yaml": "7636b483509feb74", "nuevo_base.yaml": "a35c71b844dd2dc5",
    "nuevo_control_fuente.yaml": "6b3d303f92cf5577", "nuevo_erm.yaml": "572c03e29ed9b973",
    "replica_cnx_chexpert_controlado.yaml": "79f25cc33335a690", "replica_cnx_chexpert_ingenuo.yaml": "7eba4e42dabd486d",
    "replica_cnx_nih_controlado.yaml": "b0c6d99843bc25f7", "replica_cnx_nih_ingenuo.yaml": "e35376c34284df47",
    "replica_p1_base.yaml": "23dac2fbc3657438"}


def test_covidx_configs_se_resuelven_igual():
    for name, h in HUELLAS_COVIDX.items():
        c = resolve(load_yaml(ROOT / "configs" / name))
        got = hashlib.sha256(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:16]
        assert got == h, name
        assert "bench" not in c["data"] and "storage" not in c and "subset_dc" not in c["selection"], name
    assert "storage" not in DEFAULTS and "bench" not in DEFAULTS["data"] and "subset_dc" not in DEFAULTS["selection"]


def test_datos_sinteticos_no_dependen_de_pythonhashseed():
    """Causa de la prueba intermitente test_dfr_marks_full_val_robust_not_comparable: hash() de la partición cambiaba
    con PYTHONHASHSEED y, para un 1 % de los valores, la mitad val_robust_fit quedaba con una sola clase."""
    code = ("import sys; sys.path.insert(0, %r); from src.train.synthetic import synthetic_frame; "
            "print(synthetic_frame(64, 4, 1, 'val')['label'].tolist())" % str(ROOT))
    outs = {subprocess.run([sys.executable, "-c", code], env=dict(os.environ, PYTHONHASHSEED=str(h)),
                           capture_output=True, text=True, check=True).stdout for h in (0, 174, 9999)}
    assert len(outs) == 1


# ====================================================================== contrato y bloqueo
def test_carga_por_contrato(b3a, tmp_path):
    csv, cd, md, red, meta = b3a["0.90"]
    tr = BaselineTrainer(cpu_cfg("erm", "b3a", csv, cd, md), seed=0, out_dir=tmp_path / "o", echo=False)
    assert set(tr.frames) == {"train", "val", "val_desconf", "val_fit", "val_sel", "val_desconf_fit", "val_desconf_sel"}
    idx = {s: i for i, s in enumerate(meta["fuentes"])}
    for k, f in tr.frames.items():
        assert (f["source_idx"] == f["source"].map(idx)).all(), k
    b = tr.cfg["bench"]
    assert b["rho"] == 0.9 and b["semilla"] == 0 and b["fuentes"] == ["nih", "chexpert"]
    assert sorted(b["fuentes_mixtas"]) == ["chexpert", "nih"] and b["test_bloqueado"] is True
    rc = json.load(open(tmp_path / "o" / "config_resolved.json"))
    assert rc["bench"]["rho"] == 0.9                                     # lo lee src/eval/dosis.py
    for base in ("val", "val_desconf"):
        f, s = tr.frames[base + "_fit"], tr.frames[base + "_sel"]
        assert set(f.patient_id).isdisjoint(s.patient_id) and len(f) + len(s) == len(tr.frames[base])
        cf = f.groupby(["label", "source_idx"]).size()
        cs = s.groupby(["label", "source_idx"]).size()
        assert ((cf - cs).abs() <= 1).all()                              # estratificadas por celda
    # ninguna imagen de test en ningún conjunto
    test_ids = set(red.loc[red["split"] == "test", "image_id"])
    assert test_ids and all(test_ids.isdisjoint(f["image_id"]) for f in tr.frames.values())
    assert tr.cells.shape == (2, 8) and (tr.cells[:, :2] > 0).all()


@pytest.mark.parametrize("bench,conj", [("b3a", "test_desconf"), ("b3a", "test_invertido"), ("b3a", "test"),
                                        ("b2", "test_shenzhen"), ("b2", "test_montgomery"), ("b2", "test_robusto")])
def test_bloqueo_de_todos_los_tests(tmp_path, bench, conj):
    csv, cd, md, _, _ = make_bench(tmp_path, bench=bench, rho="0.90")
    c = cpu_cfg("erm", bench, csv, cd, md)
    c["data"]["eval_sets"] = c["data"]["eval_sets"] + [conj]
    with pytest.raises(TestLockedError):
        BaselineTrainer(c, seed=0, out_dir=tmp_path / "o", echo=False)
    c = cpu_cfg("erm", bench, csv, cd, md, **{"selection.subset_dc": conj})
    with pytest.raises(BM.TestBloqueadoError):
        BaselineTrainer(c, seed=0, out_dir=tmp_path / "o2", echo=False)


def test_bloqueo_covidgr_y_desbloqueo_sin_puerta7(tmp_path, monkeypatch):
    cfg = {"data": {"eval_sets": ["val", "test_covidgr"]}, "selection": {"subset": "val"}, "protocol": {}}
    with pytest.raises(BM.TestBloqueadoError):
        BM.check_test(cfg)
    monkeypatch.setattr(BM, "GATE7", tmp_path / "no_existe.json")
    cfg["protocol"] = {"test_bloqueado": False, "congelado": "x.json"}
    with pytest.raises(BM.TestBloqueadoError, match="Puerta 7"):
        BM.check_test(cfg)


def test_inicializacion_distinta_de_imagenet_es_error(b3a, tmp_path):
    csv, cd, md, _, _ = b3a["0.90"]
    for init in ({"type": "none"}, {"type": "keras_h5", "path": "x.h5"}, {"type": "keras_model", "path": "x"}):
        c = cpu_cfg("erm", "b3a", csv, cd, md, **{"model.name": "paper1_convnext", "model.init": init})
        with pytest.raises(BM.BenchConfigError, match="ImageNet"):
            BaselineTrainer(c, seed=0, out_dir=tmp_path / "o", echo=False)
    c = cpu_cfg("erm", "b3a", csv, cd, md, **{"protocol.fase": "busqueda"})
    with pytest.raises(BM.BenchConfigError, match="tiny"):
        BaselineTrainer(c, seed=0, out_dir=tmp_path / "o", echo=False)
    base = load_config(ROOT / "configs/b3a_erm.yaml")
    assert base["model"]["init"]["type"] == "timm_imagenet" and base["model"]["name"] == "paper1_convnext"
    BM.check_init(base)


def test_semilla_distinta_del_manifiesto_es_error(b3a, tmp_path):
    csv, cd, md, _, _ = b3a["0.90"]
    with pytest.raises(BM.BenchConfigError, match="semilla"):
        BaselineTrainer(cpu_cfg("erm", "b3a", csv, cd, md), seed=1, out_dir=tmp_path / "o", echo=False)


def test_configs_de_benchmark_heredan_el_protocolo_comun():
    for b in ("b3a", "b2"):
        for m in METODOS:
            c = load_config(ROOT / ("configs/%s_%s.yaml" % (b, m)))
            p = load_config(ROOT / ("configs/baseline_%s.yaml" % m))
            for k in ("loss", "optimizer", "phases", "augment", "augment_params", "batch_size", "sampler", "grad_clip"):
                assert c["train"][k] == p["train"][k], (b, m, k)
            assert c["ema"] == p["ema"] and c["selection"]["patience"] == p["selection"]["patience"]
            assert c["train"]["steps_per_epoch"] is None
            assert c["protocol"]["test_bloqueado"] is True and not any(BM.is_test(n) for n in c["data"]["eval_sets"])
            assert c["baseline"]["method"] == m and c["storage"] == {"states": [], "keep_checkpoint": False}


# ====================================================================== doce métodos, aplicabilidad y estados
@pytest.mark.parametrize("metodo", METODOS)
def test_doce_metodos_corrida_minima_b3a(b3a, tmp_path, metodo):
    csv, cd, md, _, _ = b3a["0.90"]
    tr, code = run(cpu_cfg(metodo, "b3a", csv, cd, md, experiment="b3a_A_%s_rho090" % metodo), tmp_path / "o")
    assert code == 0
    s = json.load(open(tmp_path / "o" / "summary.json"))
    p = preds(tmp_path / "o")
    assert not [f for f in p if "test" in f]
    if metodo == "control_fuente":            # rho < 1: todas las fuentes mixtas
        assert s["estado"] == "no_aplicable" and "idéntica a ERM" in s["motivo"] and p == []
        return
    assert s["estado"] == "terminado" and s["aplicabilidad"]["estado"] == "aplicable"
    st = ("dfr_best", "dfr_bestdc") if metodo == "dfr" else ("best", "bestdc")
    for e in st:
        for conj in ("val", "val_desconf"):
            assert "%s_%s.csv" % (e, conj) in p
    assert s["almacenamiento"]["estados_conservados"] == [] and not (tmp_path / "o" / "checkpoint.pt").exists()


@pytest.mark.parametrize("metodo,estado", [("erm", "aplicable"), ("control_fuente", "no_aplicable"),
                                           ("groupdro", "degenerado"), ("dfr", "aplicable"),
                                           ("erm_cuotas", "degenerado"), ("lisa", "degenerado")])
def test_aplicabilidad_rho_1(b3a, tmp_path, metodo, estado):
    csv, cd, md, _, _ = b3a["1.00"]
    over = {"train.batch_size": 8, "train.micro_batch": 8} if metodo == "erm_cuotas" else {}   # celdas de 4 imágenes
    tr, code = run(cpu_cfg(metodo, "b3a", csv, cd, md, **over), tmp_path / "o")
    assert code == 0
    s = json.load(open(tmp_path / "o" / "summary.json"))
    assert s["aplicabilidad"]["estado"] == estado
    p = preds(tmp_path / "o")
    if estado == "no_aplicable":
        assert s["estado"] == "no_aplicable" and p == [] and "no tiene datos" in s["motivo"]
    if metodo == "dfr":                       # V2.4.6: dfr_best no aplicable, dfr_bestdc sí
        assert "dfr_best" in s["no_aplicable"] and not [f for f in p if f.startswith("dfr_best_")]
        assert "dfr_bestdc_val_desconf.csv" in p and "dfr_bestdc/val_desconf" in s["no_comparables"]
        info = json.load(open(tmp_path / "o" / "states" / "dfr_info.json"))
        assert "no_aplicable" in info["best"] and info["bestdc"]["fit"] == "val_desconf_fit"


def test_estados_best_y_bestdc(b3a, tmp_path):
    csv, cd, md, _, _ = b3a["0.90"]
    fases = copy.deepcopy(FASES)
    fases[1]["epochs"] = 4
    tr, code = run(cpu_cfg("erm", "b3a", csv, cd, md, **{"train.phases": fases, "storage": {"states": ["best", "bestdc"],
                                                                                           "keep_checkpoint": False}}),
                   tmp_path / "o")
    assert code == 0
    s = json.load(open(tmp_path / "o" / "summary.json"))
    h = [r for r in s["history"] if "selection" in r]
    assert len(h) == 4 and all("selection_dc" in r for r in h)
    assert s["early_stopping"]["best_epoch"] == int(np.argmax([r["selection"] for r in h]))
    assert s["early_stopping_dc"]["best_epoch"] == int(np.argmax([r["selection_dc"] for r in h]))
    assert s["seleccion"] == "val" and s["seleccion_dc"] == "val_desconf"
    assert sorted(os.listdir(tmp_path / "o" / "states")) == ["best.pt", "bestdc.pt"]
    import torch
    tr.model.load_state_dict(torch.load(tmp_path / "o" / "states" / "bestdc.pt"))
    from src.eval.predict import predict_loader
    p = predict_loader(tr.model, tr.eval_loader("val_desconf"), tr.device, tta=True)
    f = pd.read_csv(tmp_path / "o" / "predictions" / "bestdc_val_desconf.csv")
    assert np.allclose(f["prob"].to_numpy(), p["prob"], atol=1e-6)


def test_dfr_selecciona_en_val_sel_y_ajusta_en_val_fit(b3a, tmp_path):
    csv, cd, md, _, _ = b3a["0.90"]
    tr, code = run(cpu_cfg("dfr", "b3a", csv, cd, md), tmp_path / "o")
    assert code == 0 and tr.cfg["selection"]["subset"] == "val_sel"
    info = json.load(open(tmp_path / "o" / "states" / "dfr_info.json"))
    assert info["best"]["fit"] == "val_fit" and info["best"]["sel"] == "val_sel"
    assert set(info["best"]["fit_ids"]) == set(tr.frames["val_fit"]["image_id"])
    assert info["best"]["inestable"] is True                                     # celdas de 2 imágenes
    c = cpu_cfg("dfr", "b3a", csv, cd, md, **{"selection.subset": "val"})
    with pytest.raises(ValueError, match="val_sel"):
        BaselineTrainer(c, seed=0, out_dir=tmp_path / "o2", echo=False)


def test_b2_carga_y_control_de_fuente(tmp_path):
    csv, cd, md, red, meta = make_bench(tmp_path, bench="b2")
    tr, code = run(cpu_cfg("control_fuente", "b2", csv, cd, md), tmp_path / "o")
    assert code == 0 and set(tr.frames["train"]["source"]) == {"tbx11k"}
    s = json.load(open(tmp_path / "o" / "summary.json"))
    assert s["estado"] == "terminado" and "bestdc_val_robust.csv" in preds(tmp_path / "o")
    tr, code = run(cpu_cfg("dfr", "b2", csv, cd, md), tmp_path / "d")
    assert code == 0 and "dfr_best_val_robust.csv" in preds(tmp_path / "d")
    assert json.load(open(tmp_path / "d" / "states" / "dfr_info.json"))["bestdc"]["fit"] == "val_robust_fit"


# ====================================================================== almacenamiento
def test_almacenamiento_final_conserva_best(b3a, tmp_path):
    csv, cd, md, _, _ = b3a["0.90"]
    st = {"storage": {"states": ["best"], "keep_checkpoint": False}}
    run(cpu_cfg("dfr", "b3a", csv, cd, md, **st), tmp_path / "o")
    assert sorted(f for f in os.listdir(tmp_path / "o" / "states") if f.endswith(".pt")) == ["best.pt", "dfr_best.pt"]
    assert not (tmp_path / "o" / "checkpoint.pt").exists()


def test_checkpoint_no_se_borra_si_no_termina_con_codigo_0(b3a, tmp_path):
    from src.train.trainer import EXIT_DEBUG_STOP
    csv, cd, md, _, _ = b3a["0.90"]
    c = cpu_cfg("erm", "b3a", csv, cd, md, **{"train.stop_after_steps": 3})
    tr, code = run(c, tmp_path / "o")
    assert code == EXIT_DEBUG_STOP and (tmp_path / "o" / "checkpoint.pt").exists()
    assert not (tmp_path / "o" / "summary.json").exists()
    c["train"]["stop_after_steps"] = None                         # reanuda y termina: entonces sí se borra
    tr, code = run(c, tmp_path / "o")
    assert code == 0 and tr.resumed and not (tmp_path / "o" / "checkpoint.pt").exists()
    mt = os.path.getmtime(tmp_path / "o" / "predictions" / "best_val.csv")
    tr, code = run(c, tmp_path / "o")                             # relanzar no reentrena ni sobrescribe
    assert code == 0 and os.path.getmtime(tmp_path / "o" / "predictions" / "best_val.csv") == mt


def test_parada_termica_conserva_checkpoint(b3a, tmp_path, monkeypatch):
    from src.train import trainer as T
    csv, cd, md, _, _ = b3a["0.90"]
    calls = {"n": 0}

    def step(self):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise T.ThermalLimitReached(85.0, 80.0, "prueba de parada térmica")
    monkeypatch.setattr(T.ThermalSelfCheck, "step", step)
    tr, code = run(cpu_cfg("erm", "b3a", csv, cd, md), tmp_path / "o")
    assert code == T.EXIT_THERMAL and (tmp_path / "o" / "checkpoint.pt").exists()


def test_sin_politica_el_comportamiento_no_cambia(tmp_path):
    sys.path.insert(0, str(ROOT / "tests"))
    from test_baselines import cfg_for
    tr, code = run(cfg_for("erm"), tmp_path / "o")
    assert code == 0 and (tmp_path / "o" / "checkpoint.pt").exists()
    assert sorted(os.listdir(tmp_path / "o" / "states")) == ["best.pt", "ema.pt", "last.pt"]
    assert "almacenamiento" not in json.load(open(tmp_path / "o" / "summary.json"))


# ====================================================================== generadores, orden y lectores
def test_orden_del_generador_por_linea_de_ordenes(b3a, tmp_path):
    import bench_jobs as BJ
    csv, cd, md, _, _ = b3a["0.90"]
    c = BJ.corrida("b3a", "A", "erm", 0.9, 0, trial=3, fase="busqueda", prefijo="b3a_busq")
    cmd = [a if not a.startswith("data.bench=") else "data.bench=%s" % csv for a in c["comando"]]
    cmd = [sys.executable if a == "python" else a for a in cmd]
    cmd[cmd.index("--experimento") + 1:cmd.index("--experimento") + 1] = []
    cmd += ["model.name=tiny", "protocol.fase=prueba", "data.cache=%s" % cd, "data.num_workers=0",
            "train.batch_size=16", "train.micro_batch=16", "thermal.every=1000", "train.augment=none",
            "train.phases=%s" % json.dumps(FASES)]
    i = cmd.index("--seed")
    cmd[i:i] = ["--out", str(tmp_path / "o")]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    rc = json.load(open(tmp_path / "o" / "config_resolved.json"))
    assert rc["experiment"] == "b3a_A_erm_rho090_t3" and rc["baseline"]["trial"] == 3
    assert rc["storage"] == {"states": [], "keep_checkpoint": False} and rc["protocol"]["fase"] == "prueba"


def test_puerta6_y_dosis_leen_la_salida(tmp_path):
    from src.eval import dosis as D
    sys.path.insert(0, str(ROOT / "scripts" / "eval"))
    import puerta6
    root = tmp_path / "train"
    fases = copy.deepcopy(FASES)
    fases[1]["epochs"] = 1
    for s in (0, 1):
        for r in ("0.50", "1.00"):
            csv, cd, md, _, _ = make_bench(tmp_path / "m", rho=r, seed=s)
            for m in ("erm", "dfr"):
                exp = "b3a_A_%s_rho%03d" % (m, round(float(r) * 100))
                run(cpu_cfg(m, "b3a", csv, cd, md, experiment=exp, **{"train.phases": fases}),
                    root / exp / ("seed%d" % s), seed=s)
    inv = D.descubrir(root, "b3a", "A", "best", (D.CONJ_VAL_DESCONF,))
    assert len(inv) == 8 and inv[inv.metodo == "erm"]["existe"].all()
    dfr = inv[inv.metodo == "dfr"].set_index("rho")["existe"]
    assert dfr.loc[0.5].all() and not dfr.loc[1.0].any()          # dfr_best no aplicable en rho = 1.0
    inv2 = D.descubrir(root, "b3a", "A", "bestdc", (D.CONJ_VAL_DESCONF,))
    assert inv2["existe"].all()
    res = puerta6.evaluar(root, tmp_path / "g6", B=50)
    assert res["veredicto"] in ("informativo", "no informativo", "indeterminado") and res["semillas"] == [0, 1]
    assert (tmp_path / "g6" / "puerta6.json").exists()


def test_trabajos_preparados_son_validos():
    from job_runner import validate_job
    staged = ROOT / "jobs" / "staged"
    # el piloto y la prueba de humo se encolaron el 14-11-2026: ya no están en staged sino en done
    def _donde(patron):
        return sorted(staged.glob(patron)) or sorted((ROOT / "jobs" / "done").glob(patron))
    def _limpio(j):
        return {k: v for k, v in j.items() if k not in ("intentos", "historial", "interrumpido")}   # los añade el ejecutor
    cad = _donde("b3a_p6_cadena_*.json")
    assert len(cad) == 2
    gpus, pasos = set(), []
    for f in cad:
        j = _limpio(json.load(open(f, encoding="utf-8")))
        assert validate_job(j, f.stem) == [] and j["comando"][1] == "scripts/jobs/cadena.py"
        gpus.add(j["gpu"])
        d = ROOT / j["comando"][j["comando"].index("--staged") + 1]
        for jid in j["comando"][j["comando"].index("--staged") + 2:]:
            pj = json.load(open(d / (jid + ".json"), encoding="utf-8"))
            assert validate_job(pj, jid) == []
            assert not any("test" in a for a in pj["comando"] if a.startswith("data.eval_sets"))
            pasos.append(pj)
    assert gpus == {0, 1} and len(pasos) == 18
    metodos = {p["comando"][p["comando"].index("--config") + 1] for p in pasos}
    assert metodos == {"configs/b3a_erm.yaml", "configs/b3a_dfr.yaml", "configs/b3a_groupdro.yaml"}
    h = _limpio(json.load(open(_donde("humo_b3a.json")[0], encoding="utf-8")))
    assert validate_job(h, "humo_b3a") == [] and "b3a" in h["comando"]
