"""Pruebas de los experimentos adicionales (docs/08_evaluacion.md, enmienda E8; docs/13_entrenamiento_plan_b.md, parte
"Experimentos adicionales (17-11-2026)"): generadores de scripts/jobs/make_jobs_extra.py, segundo tronco (ResNet-50)
en CPU, igualdad numérica de ConvNeXt-Tiny, regla m3 de la elección, marcas de DFR de la enmienda E5 e informe de
sensibilidad sobre árboles sintéticos. Ninguna lee un test real ni usa la GPU; las de ResNet-50 no descargan pesos.

Requieren la serie de parches de logs/entrenamiento_r3/parches/ (después de los de M1):
    python3 -m pytest -q -p no:cacheprovider tests/test_extra_experimentos.py --basetemp=/tmp/pt_extra
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "jobs"))
sys.path.insert(0, str(ROOT / "runner"))
sys.path.insert(0, str(ROOT / "tests"))

import bench_jobs as BJ  # noqa: E402
from job_runner import validate_job  # noqa: E402

from src.eval import desbloqueo as DES  # noqa: E402

HAY_B3A = (ROOT / "data/manifests/bench/b3a/apos/rho0.90_s0.csv").exists()
if not hasattr(BJ, "EnsayoAusente"):       # sin la serie de logs/entrenamiento_r3/parches/ aplicada
    pytest.skip("requiere los parches de logs/entrenamiento_r3/parches/ (LEEME.md)", allow_module_level=True)


def _cargar(rel, nombre):
    spec = importlib.util.spec_from_file_location(nombre, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MX = _cargar("scripts/jobs/make_jobs_extra.py", "make_jobs_extra")
IS = _cargar("scripts/eval/informe_sensibilidad.py", "informe_sensibilidad")


@pytest.fixture
def sin_puerta7(tmp_path, monkeypatch):
    """La congelación apunta a un directorio temporal vacío: ningún desbloqueo es posible."""
    for k, v in (("RUTA_CONGELACION", tmp_path / "g7/congelacion.json"), ("RUTA_STATUS", tmp_path / "STATUS.md"),
                 ("RUTA_REGISTRO", tmp_path / "g7/desbloqueos.jsonl")):
        monkeypatch.setattr(DES, k, v)
    return tmp_path


# ===================================================================================== S-B
def test_estabilidad_identica_a_la_busqueda(tmp_path):
    cs = MX.corridas_estabilidad()
    assert len(cs) == 32
    assert {(c["metodo"], c["rho"]) for c in cs} == {(m, r) for m in ("erm", "lisa") for r in (0.9, 0.99)}
    assert sorted({c["trial"] for c in cs}) == list(range(8))
    for c in cs:
        assert c["semilla"] == 1 and c["salida"].startswith("results/estabilidad_b3a/")
        cmd = c["comando"]
        assert cmd[cmd.index("--experimento") + 1] == BJ.experimento("b3a", "A", c["metodo"], c["rho"], c["trial"])
        assert not any("test" in a for a in cmd)
        assert "storage={states: [], keep_checkpoint: false}" in cmd
    # orden y configuración resuelta: solo cambian manifiesto, semilla y salida (sin árbol real: con las órdenes)
    assert MX.verificar_estabilidad(raiz_busqueda=tmp_path / "nada", staged=tmp_path / "nada") == 0


def test_estabilidad_dos_cadenas_validas_y_equilibradas(tmp_path):
    assert MX.main(["--fase", "estabilidad", "--escribir", "--raiz-trabajos", str(tmp_path)]) == 0
    cad = sorted((tmp_path / "jobs/staged").glob("b3a_estab_cadena_*.json"))
    assert [p.stem for p in cad] == ["b3a_estab_cadena_g0", "b3a_estab_cadena_g1"]
    pasos = sorted((tmp_path / "jobs/staged/_pasos_b3a_estab").glob("*.json"))
    assert len(pasos) == 32
    horas, ids = [], set()
    for p in cad + pasos:
        j = json.loads(p.read_text(encoding="utf-8"))
        assert validate_job(j, p.stem) == []
    for p in cad:
        j = json.loads(p.read_text(encoding="utf-8"))
        assert j["gpu"] == int(p.stem[-1])
        ids |= set(j["comando"][6:])
        horas.append(float(j["notas"].split("estimación: ")[1].split(" h")[0]))
    assert len(ids) == 32 and abs(horas[0] - horas[1]) <= 0.15 * max(horas)


def test_salidas_nuevas_no_se_confunden_con_las_principales(tmp_path):
    """Ningún lector existente (dosis, C2, C3, piloto de B3a, tablas de la búsqueda) recoge estas salidas por error."""
    from src.eval import dosis as D
    for c in MX.corridas_estabilidad() + MX.corridas_tronco() + MX.corridas_etiqueta() + MX.corridas_c3():
        assert not c["salida"].startswith("results/train/")
        exp = c["salida"].split("/")[-2]
        assert D.parse_nombre(exp) is None          # dosis.descubrir no los lee aunque se copien a results/train
    for c in MX.corridas_semillas10(solo_listar=True):
        assert c["salida"].startswith("results/train/b3a_A_") and D.parse_nombre(c["salida"].split("/")[2])


# ===================================================================================== fases tras la congelación del protocolo
def test_conteos_y_formas():
    s10 = MX.corridas_semillas10(solo_listar=True)
    c3 = MX.corridas_c3()
    st, se = MX.corridas_tronco(), MX.corridas_etiqueta()
    assert (len(s10), len(c3), len(st), len(se)) == (165, 96, 45, 45)
    assert {c["semilla"] for c in s10} == {5, 6, 7, 8, 9} and "control_fuente" not in {c["metodo"] for c in s10}
    assert not [c for c in st + se if c["metodo"] == "dfr" and c["rho"] >= 1.0]
    for c in c3:                                # forma que lee src.eval.c3.hiperparametros
        assert c["salida"] == "results/c3_busqueda/b3a_A_%s_rho%03d_t%d/seed0" % (c["metodo"], round(100 * c["rho"]), c["trial"])
    for c in c3 + st + se:
        ev = [a for a in c["comando"] if a.startswith("data.eval_sets=")]
        assert len(ev) == 1 and "test_desconf" in ev[0] and "test_invertido" not in ev[0]
        assert "protocol.test_bloqueado=false" in c["comando"]
    for c in st:
        assert "configs/b3a_tronco_resnet50_" in " ".join(c["comando"]) and c["salida"].startswith("results/sens_tronco/st_r50_b3a_A_")
    for c in se:
        assert "data.bench=data/manifests/bench/b3c/apos/" in " ".join(c["comando"])


@pytest.mark.skipif(not (ROOT / "results/b3a/congelados/elegidos.json").exists(), reason="sin la elección de B3a")
def test_semillas10_igual_que_las_finales():
    fin = _cargar("scripts/jobs/make_jobs_b3a.py", "mjb3a")
    finales = {(c["metodo"], c["rho"], c["semilla"]): c for c in fin.corridas_final(list(BJ.METODOS), [0.9, 0.95, 0.99], True)}
    for c in MX.corridas_semillas10(solo_listar=True):
        ref = finales[(c["metodo"], c["rho"], 4)]["comando"]
        esp = [a.replace("_s4.csv", "_s%d.csv" % c["semilla"]) for a in ref]
        esp[esp.index("--seed") + 1] = str(c["semilla"])
        assert c["comando"] == esp


@pytest.mark.parametrize("fase", ["semillas10", "c3_busqueda", "tronco", "etiqueta"])
def test_fases_tras_la_puerta7_se_niegan_sin_congelacion(sin_puerta7, monkeypatch, fase):
    monkeypatch.setattr(MX, "comprobar_b3c", lambda raiz=None: [])
    monkeypatch.setattr(MX, "comprobar_semillas10", lambda raiz=None: [])
    monkeypatch.setattr(MX, "_faltan_congelados", lambda cs: [])
    monkeypatch.setattr(MX, "corridas_semillas10", lambda solo_listar: MX.corridas_c3()[:3])
    out = sin_puerta7 / "w"
    with pytest.raises(SystemExit, match="Puerta 7"):
        MX.main(["--fase", fase, "--escribir", "--raiz-trabajos", str(out)])
    assert not out.exists()
    assert MX.main(["--fase", fase, "--raiz-trabajos", str(out)]) == 0       # solo lista
    assert not out.exists()


def test_con_congelacion_valida_comprueba_huellas_y_escribe(tmp_path, monkeypatch):
    vistos = []
    monkeypatch.setattr(DES, "exigir_congelacion", lambda motivo, **kw: {"sha256_congelacion": "x", "ficheros": {}})
    monkeypatch.setattr(DES, "huella_congelada", lambda reg, p, raiz=None: vistos.append(str(p)) or "h")
    monkeypatch.setattr(MX, "_faltan_congelados", lambda cs: [])
    assert MX.main(["--fase", "tronco", "--escribir", "--raiz-trabajos", str(tmp_path)]) == 0
    assert any(v.endswith("configs/b3a_tronco_resnet50_erm.yaml") for v in vistos)
    assert any(v.endswith("results/sens_tronco/congelados/st_r50_dfr.json") for v in vistos)
    assert any(v.endswith("rho0.99_s2.csv") for v in vistos) and any(v.endswith("rho0.99_s2.json") for v in vistos)
    cad = sorted((tmp_path / "jobs/staged").glob("st_r50_cadena_*.json"))
    assert len(cad) >= 2 and {json.loads(p.read_text())["gpu"] for p in cad} == {0, 1}
    for p in cad + sorted((tmp_path / "jobs/staged/_pasos_st_r50").glob("*.json")):
        assert validate_job(json.loads(p.read_text(encoding="utf-8")), p.stem) == []


def test_etiqueta_avisa_si_faltan_los_datos_de_b3c(tmp_path, capsys):
    prob = MX.comprobar_b3c(raiz=tmp_path)
    assert prob and "falta el manifiesto data/manifests/bench/b3c/apos/rho0.50_s0.csv" in prob[0]
    assert any("caché de B3c" in p for p in prob)


def test_congelados_coinciden_con_el_protocolo(tmp_path):
    from src.baselines.registry import sample_hparams
    from src.baselines.run import load_config
    from src.baselines.trainer import check_test_lock
    sal = dict(MX.generar_congelados(escribir=False, raiz_busqueda=tmp_path))
    assert len(sal) == 96 + 8
    for (m, r, t) in (("lisa", 0.95, 3), ("groupdro", 0.9, 7), ("dfr", 0.99, 0)):
        d = sal[MX.congelado_c3(m, r, t)]
        hp = sample_hparams(m, t)
        assert d["method"] == m and d["trial"] == t
        for k, v in hp["params"].items():
            assert d["params"][k] == v
        if "lr_B" in hp["common"]:
            assert d["lr_B"] == hp["common"]["lr_B"]
        # la corrida del generador acepta su JSON congelado (datos sintéticos: sin la congelación del protocolo)
        f = tmp_path / ("%s_%d.json" % (m, t))
        f.write_text(json.dumps(d))
        cfg = load_config(ROOT / ("configs/b3a_%s.yaml" % m), t, ["protocol.test_bloqueado=false",
                                                                  "protocol.congelado=%s" % f,
                                                                  "data.eval_sets=[val, test_desconf]",
                                                                  "data.synthetic={n_train: 8}"])
        assert check_test_lock(cfg) is False
    for m in MX.METODOS_SENS:
        assert sal[MX.congelado_tronco(m)]["trial"] == 0 and sal[MX.congelado_etiqueta(m)]["config"] == "configs/b3c_%s.yaml" % m


# ===================================================================================== m3: elegir
def _arbol_busqueda(root, n_trials):
    for t in range(n_trials):
        d = root / ("b3a_A_erm_rho090_t%d" % t) / "seed0"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps({"early_stopping": {"best": 0.9 + 0.001 * (t % 3)}}))
        (d / "config_resolved.json").write_text(json.dumps({"baseline": {"params": {}},
                                                            "train": {"phases": [{"lr": 1e-3}, {"lr": 5e-5}]}}))


def test_elegir_no_salta_ensayos_sin_summary(tmp_path):
    _arbol_busqueda(tmp_path / "t", 7)
    with pytest.raises(BJ.EnsayoAusente, match="permitir-faltantes"):
        BJ.elegir(tmp_path / "t", "b3a", "A", ["erm"], [0.9], salida=tmp_path / "c")
    e = BJ.elegir(tmp_path / "t", "b3a", "A", ["erm"], [0.9], salida=tmp_path / "c", permitir_faltantes=True)
    assert e["b3a_A_erm_rho090"]["trial"] == 2
    assert json.loads((tmp_path / "c" / "b3a_A_erm_rho090.json").read_text())["ensayos_ausentes"] == [7]
    _arbol_busqueda(tmp_path / "u", 8)
    BJ.elegir(tmp_path / "u", "b3a", "A", ["erm"], [0.9], salida=tmp_path / "d")
    assert "ensayos_ausentes" not in json.loads((tmp_path / "d" / "b3a_A_erm_rho090.json").read_text())


# ===================================================================================== tronco ResNet-50
@pytest.fixture
def sin_red(monkeypatch):
    """timm sin descargas: los pesos quedan aleatorios aunque la configuración pida ImageNet."""
    import timm
    orig = timm.create_model
    monkeypatch.setattr(timm, "create_model", lambda name, pretrained=False, **kw: orig(name, pretrained=False, **kw))


def test_bn_del_tronco_congeladas_con_el_tronco():
    import torch
    from src.baselines.paper1_model import Paper1ConvNeXt
    m = Paper1ConvNeXt(backbone="resnet50", pretrained=False, input_scale=1.0)
    bn = [x for x in m.backbone.modules() if isinstance(x, torch.nn.BatchNorm2d)]
    assert len(bn) == m.pesos_info["n_batchnorm"] > 0 and m.pesos_info["num_features"] == 2048
    assert m.fc1.in_features == 2048 and m.pesos_info["etiqueta_por_defecto_timm"] == "resnet50.a1_in1k"
    m.estadisticas_bn_congeladas = True
    m.train()
    antes = bn[0].running_mean.clone()
    m(torch.rand(2, 3, 32, 32))
    assert not any(x.training for x in bn) and torch.equal(antes, bn[0].running_mean) and m.fc1.training
    m.estadisticas_bn_congeladas = False
    m.train()
    m(torch.rand(2, 3, 32, 32))
    assert all(x.training for x in bn) and not torch.equal(antes, bn[0].running_mean)


def test_tronco_no_admitido_en_benchmark():
    from src.train import bench as BM
    with pytest.raises(BM.BenchConfigError, match="tronco"):
        BM.check_init({"model": {"name": "paper1_convnext", "backbone": "vit_base_patch16_224",
                                 "init": {"type": "timm_imagenet"}}, "protocol": {"fase": "busqueda"}})
    BM.check_init({"model": {"name": "paper1_convnext", "backbone": "resnet50", "init": {"type": "timm_imagenet"}},
                   "protocol": {"fase": "busqueda"}})


@pytest.fixture(scope="module")
def b3a_red(tmp_path_factory):
    if not HAY_B3A:
        pytest.skip("sin los manifiestos de B3a")
    from test_train_plan_b import make_bench
    d = tmp_path_factory.mktemp("b3a_r50")
    return {r: make_bench(d, rho=r) for r in ("0.90", "1.00")}


def _cfg_r50(metodo, csv, cdir):
    from src.baselines.run import load_config
    from test_train_plan_b import FASES, _set
    cfg = load_config(ROOT / ("configs/b3a_tronco_resnet50_%s.yaml" % metodo))
    sets = {"protocol.fase": "prueba", "data.bench": str(csv), "data.cache": str(cdir), "data.num_workers": 0,
            "data.eval_batch_size": 16, "train.batch_size": 16, "train.micro_batch": 16, "train.log_every": 1000,
            "thermal.every": 1000, "train.augment": "none", "train.phases": FASES, "device": "cpu"}
    if metodo == "dfr":
        sets["baseline.params"] = {"states": ["best", "bestdc"], "n_reps": 2, "C_grid": [1.0, 0.1]}
    for k, v in sets.items():
        _set(cfg, k, v)
    return cfg


@pytest.mark.parametrize("metodo", ["erm", "groupdro", "lisa", "dfr"])
def test_resnet50_entrena_en_cpu_con_los_cuatro_metodos(b3a_red, sin_red, tmp_path, metodo):
    from src.baselines.trainer import BaselineTrainer
    csv, cd, _, _, _ = b3a_red["0.90"]
    tr = BaselineTrainer(_cfg_r50(metodo, csv, cd), seed=0, out_dir=tmp_path / "o", echo=False)
    assert tr.cfg["model"]["backbone"] == "resnet50" and tr.model.net.fc1.in_features == 2048
    assert tr.fit() == 0
    s = json.loads((tmp_path / "o" / "summary.json").read_text())
    assert s["tronco"]["tronco"] == "resnet50" and s["tronco"]["etiqueta_pesos"] == "resnet50.a1_in1k"
    assert s["tronco"]["preentrenado"] is True and s["estado"] == "terminado"
    preds = sorted(p.name for p in (tmp_path / "o" / "predictions").iterdir())
    assert "best_val_desconf.csv" in preds and not [p for p in preds if "test" in p]
    if metodo == "dfr":
        assert "dfr_best_val_desconf.csv" in preds and "dfr_bestdc_val_desconf.csv" in preds
    lg = [json.loads(x) for x in open(tmp_path / "o" / "train_log.jsonl", encoding="utf-8")]
    assert [r for r in lg if r["event"] == "start"][0]["tronco"]["tronco"] == "resnet50"


def test_resnet50_dfr_no_aplicable_en_rho_1(b3a_red, sin_red, tmp_path):
    from src.baselines.trainer import BaselineTrainer
    csv, cd, _, _, _ = b3a_red["1.00"]
    tr = BaselineTrainer(_cfg_r50("dfr", csv, cd), seed=0, out_dir=tmp_path / "o", echo=False)
    assert tr.fit() == 0
    s = json.loads((tmp_path / "o" / "summary.json").read_text())
    assert "dfr_best" in s["no_aplicable"] and s["tronco"]["tronco"] == "resnet50"


# ===================================================================================== ConvNeXt sin cambios
def test_convnext_construccion_identica_y_sin_consumo_de_azar():
    """La clase parcheada construye ConvNeXt-Tiny con los mismos pesos y el mismo estado del generador que las capas
    equivalentes creadas a mano, y el gancho de BatchNorm no cambia el modo de ningún módulo (no hay BatchNorm)."""
    import timm
    import torch
    from src.baselines.paper1_model import Paper1ConvNeXt
    torch.manual_seed(7)
    m = Paper1ConvNeXt(backbone="convnext_tiny", pretrained=False, input_scale=1.0, drop_path_rate=0.1, l2_hidden=0.0)
    r1 = torch.get_rng_state()
    torch.manual_seed(7)
    bb = timm.create_model("convnext_tiny", pretrained=False, num_classes=0, global_pool="", drop_path_rate=0.1)
    fc1, ln, fc2 = torch.nn.Linear(768, 128), torch.nn.LayerNorm(128, eps=1e-3), torch.nn.Linear(128, 1)
    assert torch.equal(r1, torch.get_rng_state())
    ref = {**{"backbone." + k: v for k, v in bb.state_dict().items()}, **{"fc1." + k: v for k, v in fc1.state_dict().items()},
           **{"ln." + k: v for k, v in ln.state_dict().items()}, **{"fc2." + k: v for k, v in fc2.state_dict().items()}}
    sd = m.state_dict()
    assert set(sd) == set(ref) and all(torch.equal(sd[k], ref[k]) for k in sd)
    assert m.pesos_info["n_batchnorm"] == 0
    m.estadisticas_bn_congeladas = True
    m.train()
    assert all(x.training for x in m.modules())


# Huellas de logs/entrenamiento_r3/verificar_convnext_identico.py calculadas con el código con solo los parches de M1
# (17-11-2026, VM, CPU, fp32, torch 2.14.1 y timm 1.0.31): ERM con ConvNeXt-Tiny, construcción, tres pasos de la fase
# A y tres de la fase B. La serie completa debe dar las mismas.
HUELLAS_M1 = {"init": "e3eeeebe2341", "paso_A": "572d0a28fe90", "paso_B": "27e9612952f1"}


def test_convnext_pasos_identicos_al_codigo_anterior():
    import timm
    import torch
    if not (torch.__version__.startswith("2.14.1") and timm.__version__ == "1.0.31"):
        pytest.skip("huellas de referencia calculadas con torch 2.14.1 y timm 1.0.31")
    v = _cargar("logs/entrenamiento_r3/verificar_convnext_identico.py", "vci")
    r = v.correr(str(ROOT), "convnext_tiny", "erm")
    assert {k: r[k][:12] for k in HUELLAS_M1} == HUELLAS_M1


# ===================================================================================== E5: marcas de DFR
@pytest.mark.skipif(not HAY_B3A, reason="sin los manifiestos de B3a")
def test_dfr_marcas_completas_sin_cambiar_cifras(tmp_path):
    from test_train_plan_b import cpu_cfg, make_bench, run
    csv, cd, md, _, _ = make_bench(tmp_path / "b", rho="0.90")
    tr, code = run(cpu_cfg("dfr", "b3a", csv, cd, md), tmp_path / "o")
    assert code == 0
    s = json.loads((tmp_path / "o" / "summary.json").read_text())
    nc = s["no_comparables"]
    for k in ("dfr_best/val", "dfr_best/val_desconf", "dfr_bestdc/val", "dfr_bestdc/val_desconf"):
        assert k in nc, k
    assert not [k for k in nc if not k.startswith("dfr_")]
    assert "best/val_desconf" in s["results"] and "dfr_best/val_desconf" in s["results"]


# ===================================================================================== trabajos y humo
def test_trabajo_de_humo_del_tronco_valido():
    p = ROOT / "logs/entrenamiento_r3/trabajos_tras_parches/humo_tronco_resnet50.json"
    j = json.loads(p.read_text(encoding="utf-8"))
    assert validate_job(j, p.stem) == []
    out = subprocess.run([sys.executable, str(ROOT / "scripts/jobs/humo_tronco.py"), "--solo-listar"],
                         capture_output=True, text=True, check=True).stdout
    assert out.count("b3a_tronco_resnet50_") == 4 and "configs/b3a_erm.yaml" in out and "test" not in out.replace("protocol", "")


# ===================================================================================== informe de sensibilidad
def test_informe_estabilidad_sintetico(tmp_path):
    rb, re_ = IS._demo_sb(tmp_path)
    df, faltan = IS.estabilidad(rb, re_, B=200)
    assert len(df) == 4 and not faltan
    assert set(df.columns) >= {"coincide", "spearman_metricas_seleccion", "dif_auc_s0", "dif_auc_s1"}
    assert (df.ensayo_semilla0 == 3).all() and (df.ensayo_semilla1 == 5).all() and not df.coincide.any()
    for f in sorted(re_.glob("b3a_A_lisa_rho099_t7/seed1/summary.json")):
        f.unlink()
    with pytest.raises(IS.Incompleto):
        IS.estabilidad(rb, re_, B=50)
    df2, faltan2 = IS.estabilidad(rb, re_, B=50, permitir_incompleto=True)
    assert faltan2 and len(df2) == 4


def test_informe_tronco_y_etiqueta_exigen_congelacion(sin_puerta7):
    raiz = sin_puerta7 / "d"
    r, p = IS._demo_sens(raiz, "tronco")
    with pytest.raises(DES.TestBloqueadoError, match="Puerta 7"):
        IS.sensibilidad("tronco", r, p, B=50)


@pytest.mark.parametrize("apartado", ["tronco", "etiqueta"])
def test_informe_tronco_y_etiqueta_sintetico(tmp_path, apartado):
    r, p = IS._demo_sens(tmp_path, apartado)
    ver = lambda motivo: {"sha256_congelacion": "sintetica"}
    conc, curva, comp, res, info = IS.sensibilidad(apartado, r, p, B=200, verificador=ver)
    assert len(conc) == 7 and info["corridas"] == 45 and info["sha256_congelacion"] == "sintetica"
    caida = conc[conc.medida == "caida_050_100"].iloc[0]
    assert caida["estimacion_%s" % apartado] > 0 and caida["estimacion_principal"] > 0 and caida.mismo_signo
    assert set(conc.metodo) == {"erm", "groupdro", "lisa", "dfr"} and set(conc.rho.dropna()) == {0.9, 0.99}
    assert np.isfinite(conc["ic95_inf_%s" % apartado]).all()
    # sin el tronco registrado en summary.json (corridas sin los parches) se niega
    for f in r.glob("*/seed0/summary.json"):
        f.write_text("{}")
    with pytest.raises(SystemExit, match="tronco"):
        IS.sensibilidad(apartado, r, p, B=50, verificador=ver)


def test_informe_demo_de_extremo_a_extremo(tmp_path):
    assert IS.main(["todos", "--demo", "--B", "100", "--out", str(tmp_path)]) == 0
    for x in ("estabilidad", "tronco", "etiqueta"):
        assert (tmp_path / x / "informe.md").exists() and (tmp_path / x / "informe.json").exists()


# ===================================================================================== diez semillas en C2 y C3
def test_c2_y_c3_admiten_diez_semillas_en_tres_niveles_y_cinco_en_el_resto(tmp_path):
    """Las semillas 5 a 9 de make_jobs_extra --fase semillas10 caen en las mismas rutas que las finales; los análisis
    de C2 (curva jerárquica) y C3 (selección de época) usan en cada nivel todas las semillas que tiene."""
    from src.eval import c2 as C2
    from src.eval import c3 as C3
    from src.eval import dosis as D
    met = ("erm", "groupdro", "dfr")
    D.arbol_sintetico_b3(tmp_path, metodos=met, semillas=(0, 1, 2, 3, 4), estados=("best", "bestdc"))
    D.arbol_sintetico_b3(tmp_path, metodos=met, rhos=(0.9, 0.95, 0.99), semillas=(5, 6, 7, 8, 9), estados=("best", "bestdc"))
    ver = lambda motivo: {"sintetico": motivo}
    preds = D.cargar(D.descubrir(tmp_path, conjuntos=("test_desconf",), permitir_test=True), "test_desconf",
                     permitir_test=True, verificador=ver)
    curva, _, comp, info = C2.curva_jerarquica(preds, list(met), B=100)
    for r, n in ((0.5, 5), (0.8, 5), (0.9, 10), (0.95, 10), (0.99, 10), (1.0, 5)):
        assert int(curva[(curva.metodo == "erm") & np.isclose(curva.rho, r)].n_semillas.iloc[0]) == n
    assert int(comp[(comp.metodo == "groupdro") & (comp.medida == "delta_auc_rho099")].n_semillas.iloc[0]) == 10
    pb, pd_ = C3._estados_epoca(tmp_path, "b3a", "A", "test_desconf", met, ver)
    pm, pn = C3.seleccion_epoca(pb, pd_, list(met), B=100)
    assert int(pm[(pm.metodo == "erm") & np.isclose(pm.rho, 0.95)].n_semillas.iloc[0]) == 10
    assert int(pm[(pm.metodo == "erm") & np.isclose(pm.rho, 0.6)].n_semillas.iloc[0]) == 5
