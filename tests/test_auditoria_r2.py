"""Pruebas de la auditoría C1, del desbloqueo único y de la congelación del protocolo (docs/14).

Solo CPU, datos sintéticos; ninguna lee un test real ni COVIDGR. Ejecutar desde paper_new:
    python3 -m pytest -q -p no:cacheprovider tests/test_auditoria_r2.py --basetemp=/tmp/pt_c1
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from src.eval import audit as A  # noqa: E402
from src.eval import c1_datos as CD  # noqa: E402
from src.eval import desbloqueo as D  # noqa: E402
from src.eval import stats as S  # noqa: E402
from src.train.bench import TestBloqueadoError as BenchBloqueado  # noqa: E402


# ===================================================================================== utilidades
@pytest.fixture
def sin_congelacion(tmp_path, monkeypatch):
    """Apunta el desbloqueo a una raíz temporal sin congelación (el proyecto real nunca interviene)."""
    monkeypatch.setattr(D, "RUTA_CONGELACION", tmp_path / "results" / "gate7" / "congelacion.json")
    monkeypatch.setattr(D, "RUTA_STATUS", tmp_path / "STATUS.md")
    monkeypatch.setattr(D, "RUTA_REGISTRO", tmp_path / "results" / "gate7" / "desbloqueos.jsonl")
    monkeypatch.setattr(D, "PROJECT_ROOT", tmp_path)
    return tmp_path


def congelar_en(raiz: Path, patrones, registrar=True) -> str:
    """Crea una congelación válida en `raiz` con scripts/eval/congelar.py y la anota en STATUS.md."""
    import congelar
    import yaml
    cfg = raiz / "configs" / "gate7_prueba.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(yaml.safe_dump({"grupos": {"todo": {"patrones": list(patrones) + ["configs/*.yaml"], "minimo": 1}}}))
    assert congelar.main(["--config", str(cfg), "--raiz", str(raiz)]) == 0
    sha = D.sha256_fichero(raiz / "results" / "gate7" / "congelacion.json")
    if registrar:
        (raiz / "STATUS.md").write_text("Puerta 7: SHA256 de results/gate7/congelacion.json: %s\n" % sha)
    return sha


# ===================================================================================== desbloqueo
def test_es_bloqueado():
    for n in ("test", "test_bimcv", "test_covidgr", "TEST_desconf", "covidgr"):
        assert D.es_bloqueado(n)
    for n in ("val", "val_robust", "val_robust_sel", "tr_bimcv_pos", "res_neg"):
        assert not D.es_bloqueado(n)
    assert issubclass(D.TestBloqueadoError, BenchBloqueado) and issubclass(D.TestBloqueadoError, RuntimeError)


def test_exigir_sin_fichero_registro_y_huellas(sin_congelacion):
    raiz = sin_congelacion
    with pytest.raises(D.TestBloqueadoError, match="no existe"):
        D.exigir_congelacion("prueba")
    with pytest.raises(D.TestBloqueadoError, match="motivo"):
        D.exigir_congelacion("")
    (raiz / "a.txt").write_text("uno")
    (raiz / "b").mkdir()
    (raiz / "b" / "c.py").write_text("x = 1\n")
    sha = congelar_en(raiz, ["a.txt", "b/*.py"], registrar=False)
    with pytest.raises(D.TestBloqueadoError, match="STATUS.md"):
        D.exigir_congelacion("prueba")
    (raiz / "STATUS.md").write_text("otra línea con %s pero sin el nombre\n" % sha)
    with pytest.raises(D.TestBloqueadoError, match="STATUS.md"):
        D.exigir_congelacion("prueba")
    (raiz / "STATUS.md").write_text("Puerta 7: congelacion.json %s\n" % sha)
    reg = D.exigir_congelacion("prueba de desbloqueo")
    assert reg["sha256_congelacion"] == sha and "a.txt" in reg["ficheros"]
    lineas = (raiz / "results" / "gate7" / "desbloqueos.jsonl").read_text().splitlines()
    assert len(lineas) == 1 and json.loads(lineas[0])["motivo"] == "prueba de desbloqueo"
    assert D.huella_congelada(reg, raiz / "a.txt", raiz) == D.sha256_fichero(raiz / "a.txt")
    with pytest.raises(D.TestBloqueadoError, match="no está en la congelación"):
        (raiz / "nuevo.txt").write_text("z")
        D.huella_congelada(reg, raiz / "nuevo.txt", raiz)
    # una huella cambiada impide el desbloqueo
    (raiz / "b" / "c.py").write_text("x = 2\n")
    with pytest.raises(D.TestBloqueadoError, match="cambiado"):
        D.exigir_congelacion("prueba")
    (raiz / "b" / "c.py").write_text("x = 1\n")
    assert D.exigir_congelacion("otra vez")["sha256_congelacion"] == sha
    # un fichero congelado que desaparece también
    (raiz / "a.txt").rename(raiz / "a2.txt")
    with pytest.raises(D.TestBloqueadoError, match="faltan"):
        D.exigir_congelacion("prueba")
    # un congelacion.json manipulado deja de coincidir con STATUS.md
    (raiz / "a2.txt").rename(raiz / "a.txt")
    p = raiz / "results" / "gate7" / "congelacion.json"
    reg = json.loads(p.read_text())
    reg["ficheros"].pop("a.txt")
    p.write_text(json.dumps(reg))
    with pytest.raises(D.TestBloqueadoError, match="STATUS.md"):
        D.exigir_congelacion("prueba")
    assert D.exigir_si_bloqueado(["val", "val_robust"], "x") is None


# ===================================================================================== congelación
def test_congelar_listar_crear_verificar(tmp_path, capsys):
    import congelar
    import yaml
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text("a = 1\n")
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "m.cpython.pyc").write_bytes(b"\0")
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker" / "requirements.lock.txt").write_text("numpy==1.26.4\n")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"excluir": ["**/__pycache__/**"], "grupos": {
        "codigo": {"patrones": ["src/**/*.py"], "minimo": 1},
        "entorno": {"patrones": ["docker/requirements.lock.txt"], "minimo": 1},
        "estados": {"patrones": ["results/train/*/seed*/states/*.pt"], "minimo": 2}}}))
    assert congelar.main(["--config", str(cfg), "--raiz", str(tmp_path), "--listar"]) == 0
    assert "INCOMPLETO" in capsys.readouterr().out
    assert congelar.main(["--config", str(cfg), "--raiz", str(tmp_path)]) == 2          # falta un grupo
    st = tmp_path / "results" / "train" / "x" / "seed1" / "states"
    st.mkdir(parents=True)
    for k in ("best", "last"):
        (st / ("%s.pt" % k)).write_bytes(k.encode())
    assert congelar.main(["--config", str(cfg), "--raiz", str(tmp_path)]) == 0
    reg = json.loads((tmp_path / "results" / "gate7" / "congelacion.json").read_text())
    assert set(reg["ficheros"]) == {"src/m.py", "docker/requirements.lock.txt", "results/train/x/seed1/states/best.pt",
                                    "results/train/x/seed1/states/last.pt"}
    assert reg["paquetes_congelacion"] and reg["entorno_congelacion"]["python"]
    assert any(d.startswith("numpy") for d in reg["diferencias_con_lock"]) or reg["diferencias_con_lock"] == []
    sums = (tmp_path / "results" / "gate7" / "congelacion.sha256sum").read_text().splitlines()
    assert len(sums) == 4 and all(len(l.split("  ")[0]) == 64 for l in sums)
    assert congelar.main(["--config", str(cfg), "--raiz", str(tmp_path)]) == 3          # no sobrescribe
    assert congelar.main(["--raiz", str(tmp_path), "--verificar"]) == 0
    assert congelar.main(["--raiz", str(tmp_path), "--verificar", "--exigir-status"]) == 1
    (st / "best.pt").write_bytes(b"otro")
    assert congelar.main(["--raiz", str(tmp_path), "--verificar"]) == 1
    assert "CAMBIADO  results/train/x/seed1/states/best.pt" in capsys.readouterr().out
    assert congelar.main(["--config", str(cfg), "--raiz", str(tmp_path), "--reemplazar"]) == 0
    assert list((tmp_path / "results" / "gate7").glob("congelacion_*.json.anterior"))


def test_config_de_congelacion_del_proyecto_es_legible():
    import congelar
    c = congelar.leer_config(ROOT / "configs" / "gate7_congelacion.yaml")
    assert {"plan", "codigo_src", "finales_b1_estados", "caracteristicas_c1", "entorno"} <= set(c["grupos"])
    g = congelar.resolver(c, ROOT)
    assert "src/eval/desbloqueo.py" in g["codigo_src"] and "configs/gate7_congelacion.yaml" in g["configuraciones"]
    assert not any("__pycache__" in f for fs in g.values() for f in fs)


# ===================================================================================== corridas sintéticas
TINY = {
    "experiment": "baseline_erm_final",
    "model": {"name": "tiny"},
    "data": {"synthetic": {"n_train": 48, "n_val": 48, "n_test": 24, "res": 16, "n_sources": 3},
             "eval_sets": ["val", "val_robust", "val_robust_sel"]},
    "train": {"batch_size": 16, "micro_batch": 16, "log_every": 1000,
              "phases": [{"name": "B", "epochs": 1, "lr": 1e-3, "warmup_frac": 0.1, "train_backbone": True,
                          "ema": True, "early_stopping": True}]},
    "selection": {"metric": "auc_roc", "subset": "val_robust_sel", "impl": "exact", "patience": 15},
    "eval": {"tta": True, "states": ["last", "best", "ema"]},
    "thermal": {"every": 1000},
    "protocol": {"fase": "piloto", "test_bloqueado": True},
    "baseline": {"method": "erm", "params": {}},
}


def corrida(raiz: Path, seed: int) -> Path:
    from src.baselines.trainer import BaselineTrainer
    out = raiz / "results" / "train" / "baseline_erm_final" / ("seed%d" % seed)
    tr = BaselineTrainer(copy.deepcopy(TINY), seed, out_dir=str(out), echo=False)
    assert tr.fit() == 0
    return out


@pytest.fixture(scope="module")
def corridas_tiny(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("raiz_c1")
    return raiz, [corrida(raiz, s) for s in (1, 2)]


def test_extraer_lotes_tiny(corridas_tiny):
    from extraer_caracteristicas import extraer_lotes
    _, runs = corridas_tiny
    tr = CD.trainer_de_corrida(runs[0], workers=0, dispositivo="cpu")
    CD.cargar_estado(tr, runs[0] / "states" / "best.pt")
    df = CD.marco(tr, "val")
    r = extraer_lotes(tr.model, CD.cargador(tr, df, "val"), tr.device, cabeza=True, espejo=True)
    assert r["ids"] == df["image_id"].tolist()
    x = torch.stack([tr._make_ds(df, "val")[i]["image"] for i in range(len(df))])
    with torch.no_grad():
        p = tr.model.net.pooled(x)
        z = tr.model.net(x)
        pe = tr.model.net.pooled(torch.flip(x, dims=[3]))
    assert np.allclose(r["arrays"]["pooled"], p.numpy(), atol=1e-6)
    assert np.allclose(r["arrays"]["logit"], z.numpy(), atol=1e-5)
    assert np.allclose(r["arrays"]["pooled_espejo"], pe.numpy(), atol=1e-6)
    r2 = extraer_lotes(tr.model, CD.cargador(tr, df, "val"), tr.device, cabeza=False, espejo=False)
    assert set(r2["arrays"]) == {"pooled"}


def test_extraer_escribe_y_bloquea(corridas_tiny, sin_congelacion, tmp_path):
    from extraer_caracteristicas import extraer, main as extraer_main
    raiz_tiny, runs = corridas_tiny
    tr = CD.trainer_de_corrida(runs[0], workers=0, dispositivo="cpu")
    out, out_t = tmp_path / "car", tmp_path / "car_test"
    r = extraer(tr, "tiny", ["val"], out, out_t, True, True, None, {"tronco": "x"})
    m = pd.read_csv(out / "tiny_meta.csv")
    assert set(m["conjunto"]) == {"val"} and (out / "tiny_info.json").exists() and not out_t.exists()
    with pytest.raises(D.TestBloqueadoError):
        CD.marco(tr, "test")
    with pytest.raises(D.TestBloqueadoError):
        CD.marco(tr, "test_covidgr", {"ficheros": {}})
    with pytest.raises(D.TestBloqueadoError):
        extraer(tr, "tiny", ["test"], out, out_t, True, True, None, {})
    for conj in (["test"], ["val", "test_covidgr"], ["test_bimcv"]):
        with pytest.raises(D.TestBloqueadoError):
            extraer_main(["--runs", str(runs[0]), "--conjuntos"] + conj + ["--out", str(out), "--out-test", str(out_t)])
    assert not out_t.exists()
    # con una congelación válida sí extrae el test (sintético), en su carpeta aparte
    congelar_en(sin_congelacion, ["configs/*.yaml"])
    reg = D.exigir_congelacion("prueba de extracción")
    extraer(tr, "tiny", ["test"], out, out_t, True, True, reg, {})
    mt = pd.read_csv(out_t / "tiny__test_meta.csv")
    assert set(mt["conjunto"]) == {"test"} and len(mt) == 24
    assert set(pd.read_csv(out / "tiny_meta.csv")["conjunto"]) == {"val"}       # el de validación no cambia
    # D_IN: tronco sin estado (con el modelo diminuto no hay pesos que descargar)
    tri = CD.construir_trainer(copy.deepcopy(TINY), 0, workers=0, preentrenado=True, dispositivo="cpu")
    extraer(tri, "din_tiny", ["val"], out, out_t, False, True, None, {"tronco": "D_IN"})
    assert set(np.load(out / "din_tiny.npz").files) == {"pooled", "pooled_espejo"}
    assert len(CD.huella_tronco(tri.model.net)) == 64


def test_generar_trabajos_valida(tmp_path):
    from extraer_caracteristicas import generar_trabajos
    hechos = generar_trabajos(tmp_path, ("erm",), (424242,), din=True)
    assert [Path(h).name for h in hechos] == ["c1_extraer_din_val.json"]
    sys.path.insert(0, str(ROOT / "runner"))
    from job_runner import validate_job
    j = json.loads(Path(hechos[0]).read_text())
    assert validate_job(j, "c1_extraer_din_val") == [] and "--din" in j["comando"]
    assert not any(D.es_bloqueado(x) for x in j["comando"])


def test_evaluar_test_b1_huellas_comprobar_y_test(corridas_tiny, monkeypatch, tmp_path):
    import yaml
    import evaluar_test_b1 as E
    raiz, runs = corridas_tiny
    for k, v in (("RUTA_CONGELACION", raiz / "results/gate7/congelacion.json"), ("RUTA_STATUS", raiz / "STATUS.md"),
                 ("RUTA_REGISTRO", raiz / "results/gate7/desbloqueos.jsonl"), ("PROJECT_ROOT", raiz)):
        monkeypatch.setattr(D, k, v)
    lista = raiz / "configs" / "gate7_lista_c1.yaml"
    lista.parent.mkdir(parents=True, exist_ok=True)
    lista.write_text(yaml.safe_dump({"metodos": ["erm"], "semillas": [1, 2], "experimento": "baseline_{metodo}_final",
                                     "estado_principal": {"defecto": "best"}, "estados_sensibilidad": ["last", "ema"]}))
    base = ["--lista", str(lista), "--raiz", str(raiz), "--workers", "0", "--dispositivo", "cpu"]
    assert E.main(["--modo", "huellas"] + base) == 0
    h = json.loads((raiz / "results/gate7/estados_b1.json").read_text())
    assert h["completo"] and set(h["corridas"][0]["estados"]) == {"best", "last", "ema"}
    with pytest.raises(D.TestBloqueadoError):
        E.main(["--modo", "test"] + base)
    assert not (raiz / "results" / "b1").exists()
    assert E.main(["--modo", "comprobar"] + base) == 0
    c = json.loads((raiz / "results/gate7/comprobacion_estados_b1.json").read_text())
    assert c["todo_cumple"] and c["n_comparaciones"] == 6 and max(x["max_abs_dp"] for x in c["comparaciones"]) < 1e-6
    # un estado distinto del que escribió las predicciones se detecta
    r = E.comparar({"image_id": np.array(["a", "b"]), "label": np.array([0, 1]), "prob": np.array([0.2, 0.9])},
                   pd.DataFrame({"image_id": ["a", "b"], "label": [0, 1], "prob": [0.2, 0.85]}))
    assert not r["cumple"]
    congelar_en(raiz, ["results/train/**/*.json", "results/train/**/*.pt", "results/gate7/estados_b1.json",
                       "results/gate7/comprobacion_estados_b1.json"])
    assert E.main(["--modo", "test"] + base) == 0
    p = raiz / "results/b1/test/baseline_erm_final/seed1/predictions"
    for st in ("best", "last", "ema"):
        t = S.read_predictions(p / ("%s_test.csv" % st))
        b = S.read_predictions(p / ("%s_test_bimcv.csv" % st))
        assert len(t) == 24 and 0 < len(b) < 24 and set(b["image_id"]) <= set(t["image_id"])
    # cambiar un estado después de congelar impide volver a evaluar
    sd = torch.load(runs[1] / "states" / "best.pt")
    k0 = next(iter(sd))
    sd[k0] = sd[k0] + 1
    torch.save(sd, runs[1] / "states" / "best.pt")
    with pytest.raises(D.TestBloqueadoError, match="cambiado"):
        E.main(["--modo", "test"] + base)


def test_guiones_bloqueados_sin_congelacion(sin_congelacion, tmp_path):
    import inferencia_covidgr
    import solo_adquisicion
    import auditoria_covidx
    with pytest.raises(D.TestBloqueadoError):
        inferencia_covidgr.main(["--out", str(tmp_path / "x")])
    with pytest.raises(D.TestBloqueadoError):
        solo_adquisicion.main(["--conjunto", "test_bimcv", "--out", str(tmp_path / "y")])
    with pytest.raises(D.TestBloqueadoError):
        auditoria_covidx.main(["--c1", "test", "--out-c1", str(tmp_path / "z")])
    with pytest.raises(D.TestBloqueadoError):
        A.cargar_c1(tmp_path, "nada", tmp_path, ["test"], None)
    assert not any((tmp_path / d).exists() for d in ("x", "y", "z"))


# ===================================================================================== análisis C1 en sintético
def test_auc_media_y_no_inferioridad_resultado_conocido():
    rng = np.random.default_rng(0)
    n = 600
    y = (np.arange(n) % 2).astype(int)
    g = np.array(["p%d" % (i // 2) for i in range(n)])
    ref = (y * 1.0 + rng.normal(0, 1, n))[None]                               # AUC ~ 0.76
    metodos = {"igual": np.vstack([y * 1.0 + rng.normal(0, 1, n) for _ in range(3)]),
               "mucho_mejor": np.vstack([y * 4.0 + rng.normal(0, 1, n) for _ in range(3)]),
               "peor": np.vstack([y * 0.3 + rng.normal(0, 1, n) for _ in range(3)])}
    r = A.auc_media(y, metodos["igual"], g, B=200)
    assert abs(r["auc"] - np.mean([S.auc_roc(y, p) for p in metodos["igual"]])) < 1e-12
    assert r["ic95_inf"] < r["auc"] < r["ic95_sup"]
    t = A.familia_no_inferioridad(y, g, ref, metodos, B=400).set_index("metodo")
    assert bool(t.loc["peor", "no_inferior_0.02"]) and not bool(t.loc["mucho_mejor", "no_inferior_0.02"])
    assert t.loc["mucho_mejor", "delta"] < -0.15 and t.loc["peor", "delta"] > 0.1
    assert (t["p_holm_0.02"] >= t["p_ni_0.02"]).all() and (t["p_ni_0.01"] >= t["p_ni_0.02"]).all()
    # emparejado por semilla: con la referencia igual al método, Δ = 0 exacta
    t2 = A.familia_no_inferioridad(y, g, metodos["igual"], {"igual": metodos["igual"]}, {"igual": [7, 8, 9]},
                                   B=200, emparejar=True, semillas_ref=[9, 8, 7]).iloc[0]
    assert abs(t2["delta"] - np.mean([S.auc_roc(y, metodos["igual"][i]) for i in (2, 1, 0)]) +
               np.mean([S.auc_roc(y, p) for p in metodos["igual"]])) < 1e-12
    # el comparador no comparable queda fuera de Holm
    t3 = A.familia_no_inferioridad(y, g, ref, metodos, comparables={"peor": False}, B=200).set_index("metodo")
    assert np.isnan(t3.loc["peor", "p_holm_0.02"]) and not t3.loc["peor", "no_inferior_0.02"]


def test_detector_c1_igual_al_registrado_y_tta():
    rng = np.random.default_rng(1)
    n = 300
    ids = ["sub-S%05d_ses-x_vp-ap.png" % i if i % 2 else "sub-S%06d_ses-x_vp-pa.png" % (100000 + i) for i in range(n)]
    m = pd.DataFrame({"image_id": ids, "patient_id": ids, "label": 1, "source": "bimcv", "conjunto": "tr_bimcv_pos"})
    m["coleccion"] = A.coleccion_bimcv(m["image_id"])
    X = rng.normal(0, 1, (n, 5)) + (m["coleccion"] == "antigua").to_numpy()[:, None]
    feats = {"pooled": X, "pooled_espejo": X + rng.normal(0, 0.1, X.shape)}
    s0, _ = A.detector(feats, m)
    s1, info = A.detector_c1(feats, m)
    assert np.array_equal(s0, s1) and info["negativos_en_ajuste"] == 0
    s2, info2 = A.detector_c1(feats, m, tta="espejo")
    assert info2["tta"] == "espejo" and not np.array_equal(s1, s2) and np.corrcoef(s1, s2)[0, 1] > 0.95


def test_externa_estratos_resumen_adquisicion():
    rng = np.random.default_rng(2)
    n = 400
    y = (np.arange(n) % 2).astype(int)
    g = np.arange(n).astype(str)
    Pi = np.vstack([y * 3.0 + rng.normal(0, 1, n) for _ in range(2)])
    Pe = np.vstack([y * 0.5 + rng.normal(0, 1, n) for _ in range(2)])
    grav = np.where(y == 1, np.where(np.arange(n) % 4 == 1, "MILD", "SEVERE"), "")
    r = A.externa(y, Pi, g, y, Pe, g, [0.0, 0.0], grav, B=300)
    assert r["caida"] > 0.2 and r["caida_ic95_inf"] > 0.1 and abs(r["caida"] - (r["auc_interna"] - r["auc_externa"])) < 1e-12
    assert set(r["sensibilidad_por_gravedad"]) == {"MILD", "SEVERE"}
    assert abs(r["sensibilidad_youden"] - np.mean([(p[y == 1] >= 0).mean() for p in Pe])) < 1e-12
    u = A.youden_por_semilla(y, Pi)
    assert len(u) == 2 and all(np.isfinite(u))
    proy = np.where(np.arange(n) % 3 == 0, "AP", "PA")
    t = A.estratos_proyeccion(y, g, proy, Pi[:1], {"m": Pe}, B=200)
    assert set(t["estrato"]) == {"AP", "PA", "todos"} and (t.loc[t["comparacion"] == "detector menos m", "delta"] > 0.2).all()
    # contrastes resumen: (a) bimcv con bimcv y (b) bimcv con fuentes solo positivas
    k = 120
    df = pd.DataFrame({"image_id": ["sub-S%05d_x.png" % i for i in range(k)] + ["sirm_%d.png" % i for i in range(40)],
                       "label": [0] * 60 + [1] * 60 + [1] * 40, "source": ["bimcv"] * k + ["sirm"] * 40,
                       "patient": ["p%d" % i for i in range(k + 40)]})
    P = np.atleast_2d(np.r_[rng.normal(0, 1, 60), rng.normal(2, 1, 60), rng.normal(0, 1, 40)])
    rs = {x["contraste"]: x for x in A.contrastes_resumen(df, P, B=200)}
    assert rs["a_bimcv_neg_vs_bimcv_pos"]["auc"] > 0.8 and abs(rs["b_bimcv_neg_vs_otras_fuentes_pos"]["auc"] - 0.5) < 0.15
    assert rs["b_menos_a"]["auc"] < -0.2
    # solo adquisición: variantes y diferencia frente a ERM
    t = A.solo_adquisicion(y, g, rng.normal(0, 1, n) + y, None, Pi, B=200)
    assert t[0]["delta_vs_erm"] < 0


# ===================================================================================== auditoría completa en sintético
def _ids_bimcv(n, cifras, proy0=0, ini=0):
    return ["sub-S%0*d_ses-E%d_run-1_bp-chest_vp-%s_cr.png" % (cifras, (10 ** (cifras - 1)) + ini + i, i,
                                                                 "ap" if (i + proy0) % 2 else "pa") for i in range(n)]


def arbol_c1(raiz: Path, rng, test: bool = False):
    """Árbol sintético con resultado conocido: el detector separa colecciones (AUC ~ 0.92 en val_robust); 'lisa' es
    mucho mejor que el detector (no no inferior) y 'erm' y 'erm_mascara' son peores o iguales (no inferiores)."""
    car, cart = raiz / "results/c1/caracteristicas", raiz / "results/c1/caracteristicas_test"
    car.mkdir(parents=True, exist_ok=True)
    cart.mkdir(parents=True, exist_ok=True)
    aj = pd.DataFrame({"image_id": _ids_bimcv(150, 5) + _ids_bimcv(300, 6, ini=5000), "label": 1, "source": "bimcv",
                       "conjunto": "tr_bimcv_pos"})

    def poblacion(conj):
        neg = pd.DataFrame({"image_id": [x.replace("sub-S", "sub-S9") if False else x for x in _ids_bimcv(80, 5, 1, ini=60000)],
                            "label": 0, "source": "bimcv"})
        pos = pd.DataFrame({"image_id": _ids_bimcv(160, 5, ini=20000), "label": 1, "source": "bimcv"})
        rs = pd.DataFrame({"image_id": ["rsna_%s_%d.png" % (conj, i) for i in range(40)], "label": 0, "source": "rsna"})
        si = pd.DataFrame({"image_id": ["sirm_%s_%d.png" % (conj, i) for i in range(40)], "label": 1, "source": "sirm"})
        d = pd.concat([neg, pos, rs, si], ignore_index=True)
        d["image_id"] = [conj + "_" + x if not x.startswith("sub-S") else x.replace("ses-", "ses-%s" % conj) for x in d["image_id"]]
        d["conjunto"] = conj
        return d
    val = poblacion("val")
    gr = pd.DataFrame({"image_id": ["covidgr__%s_%d.jpg" % ("P" if i % 2 else "N", i) for i in range(80)],
                       "label": [i % 2 for i in range(80)], "source": "covidgr", "conjunto": "test_covidgr"})
    partes = [aj, val] + ([poblacion("test"), gr] if test else [])
    tod = pd.concat(partes, ignore_index=True)
    tod["patient_id"] = ["q%d" % i for i in range(len(tod))]
    antig = (A.coleccion_bimcv(tod["image_id"]) == "antigua") & (tod["label"] == 1)
    X = rng.normal(0, 1, (len(tod), 6))
    X[:, 0] += 2.0 * antig
    feats = {"pooled": X, "pooled_espejo": X}
    for tag in ["din_imagenet"] + ["baseline_erm_final_seed%d_best" % s for s in (1, 2)]:
        for conj_set, d, nombre in ((("tr_bimcv_pos", "val"), car, tag), (("test",), cart, tag + "__test"),
                                    (("test_covidgr",), cart, tag + "__test_covidgr")):
            mk = tod["conjunto"].isin(conj_set).to_numpy()
            if not mk.any():
                continue
            np.savez(d / ("%s.npz" % nombre), **{k: v[mk] for k, v in feats.items()})
            tod[mk][["image_id", "patient_id", "label", "source", "conjunto"]].to_csv(d / ("%s_meta.csv" % nombre), index=False)
    fuerza = {"erm": 1.0, "lisa": 5.0, "erm_mascara": 0.4}

    def escribir_preds(d0, conj, df, fname_conj, metodo, s):
        y = df["label"].to_numpy()
        sc = y * fuerza[metodo] + np.random.default_rng([s, len(metodo)]).normal(0, 1, len(df))
        p = pd.DataFrame({"image_id": df["image_id"], "label": y,
                          "source": [S.COVIDX_SOURCES.index(x) if x in S.COVIDX_SOURCES else 0 for x in df["source"]],
                          "patient": df["patient_id"], "prob": 1 / (1 + np.exp(-sc))})
        f = d0 / ("baseline_%s_final" % metodo) / ("seed%d" % s) / "predictions" / ("best_%s.csv" % fname_conj)
        f.parent.mkdir(parents=True, exist_ok=True)
        p.to_csv(f, index=False)

    for metodo in fuerza:
        for s in (1, 2):
            v = tod[tod["conjunto"] == "val"]
            vr = v[v["source"] == "bimcv"]
            for nombre, df in (("val", v), ("val_robust", vr), ("val_robust_sel", vr[vr["patient_id"].map(A.mitad) == 1])):
                escribir_preds(raiz / "results/train", None, df, nombre, metodo, s)
            if test:
                t = tod[tod["conjunto"] == "test"]
                escribir_preds(raiz / "results/b1/test", None, t, "test", metodo, s)
                escribir_preds(raiz / "results/b1/test", None, t[t["source"] == "bimcv"], "test_bimcv", metodo, s)
                escribir_preds(raiz / "results/b1/covidgr", None, tod[tod["conjunto"] == "test_covidgr"], "test_covidgr", metodo, s)
    if test:
        gcsv = raiz / "data/manifests/bench/covidgr/covidgr.csv"
        gcsv.parent.mkdir(parents=True, exist_ok=True)
        gr.assign(severidad=["MILD" if i % 4 == 1 else ("SEVERE" if i % 2 else "") for i in range(80)]).to_csv(gcsv, index=False)
    # estadísticos de adquisición (C1-iv): débiles en el ajuste, algo más fuertes en el conjunto evaluado
    for conj, d, df in (("ajuste", raiz / "aq", aj.assign(label=np.arange(len(aj)) % 2, patient_id="a")),
                        ("val_robust", raiz / "aq", tod[(tod["conjunto"] == "val") & (tod["source"] == "bimcv")]),
                        ("test_bimcv", raiz / "aq_test", tod[(tod["conjunto"] == "test") & (tod["source"] == "bimcv")])):
        if not len(df):
            continue
        d.mkdir(parents=True, exist_ok=True)
        X = rng.normal(0, 1, (len(df), 14))
        X[:, 0] += (0.3 if conj == "ajuste" else 1.0) * df["label"].to_numpy()
        np.savez(d / ("%s_estadisticos.npz" % conj), X=X, image_id=df["image_id"].to_numpy().astype(str))
        df.assign(patient_id=df["patient_id"].astype(str) + "_" + df.index.astype(str) if conj == "ajuste" else df["patient_id"])[
            ["image_id", "patient_id", "label", "source"]].to_csv(d / ("%s_meta.csv" % conj), index=False)
    lista = raiz / "configs" / "gate7_lista_c1.yaml"
    lista.parent.mkdir(parents=True, exist_ok=True)
    L = json.loads(json.dumps(__import__("yaml").safe_load((ROOT / "configs" / "gate7_lista_c1.yaml").read_text())))
    L.update({"metodos": list(fuerza), "semillas": [1, 2]})
    L["troncos"]["D_RES"] = []
    lista.write_text(__import__("yaml").safe_dump(L))
    return lista


def test_auditoria_c1_val_y_test_sinteticas(tmp_path, monkeypatch):
    import auditoria_covidx
    rng = np.random.default_rng(5)
    lista = arbol_c1(tmp_path, rng, test=True)
    comunes = ["--lista", str(lista), "--B", "200", "--train", str(tmp_path / "results/train"),
               "--caracteristicas", str(tmp_path / "results/c1/caracteristicas"),
               "--caracteristicas-test", str(tmp_path / "results/c1/caracteristicas_test"),
               "--test-pred", str(tmp_path / "results/b1/test"), "--adquisicion", str(tmp_path / "aq"),
               "--adquisicion-test", str(tmp_path / "aq_test"), "--covidgr-pred", str(tmp_path / "results/b1/covidgr"),
               "--covidgr-csv", str(tmp_path / "data/manifests/bench/covidgr/covidgr.csv")]
    # --din: AUC de D_IN en val_robust con IC95, desde características sintéticas
    assert auditoria_covidx.main(comunes + ["--din", "--out-c1", str(tmp_path / "din")]) == 0
    d = json.loads((tmp_path / "din" / "din_val_robust.json").read_text())
    assert 0.85 < d["auc"] < 0.97 and d["ic95_inf"] < d["auc"] < d["ic95_sup"] and d["n"] == 240 and d["n_neg"] == 80
    assert d["n_ajuste"] == 450 and d["negativos_en_ajuste"] == 0
    # val: F-C1 con el resultado diseñado
    assert auditoria_covidx.main(comunes + ["--c1", "val", "--out-c1", str(tmp_path / "val")]) == 0
    f = pd.read_csv(tmp_path / "val" / "f_c1.csv")
    f = f[f["conjunto"] == "val_robust"].set_index("metodo")
    assert not f.loc["lisa", "no_inferior_0.02"] and f.loc["erm", "no_inferior_0.02"] and f.loc["erm_mascara", "no_inferior_0.02"]
    fb = pd.read_csv(tmp_path / "val" / "f_c1b.csv")
    assert set(fb["referencia"]) == {"D_ERM"} and (fb["n_semillas"] == 2).all()
    r = pd.read_csv(tmp_path / "val" / "c1_iii_resumen.csv")
    lisa = r[(r["metodo"] == "lisa") & (r["estrato_bimcv"] == "todos")].set_index("contraste")
    din = r[(r["metodo"] == "D_IN") & (r["estrato_bimcv"] == "todos")].set_index("contraste")
    assert lisa.loc["b_bimcv_neg_vs_otras_fuentes_pos", "auc"] > 0.9            # el método aprende la clase
    assert abs(din.loc["b_bimcv_neg_vs_otras_fuentes_pos", "auc"] - 0.5) < 0.2   # el detector de colección no
    assert din.loc["a_bimcv_neg_vs_bimcv_pos", "auc"] > 0.85
    e = pd.read_csv(tmp_path / "val" / "c1_vii_estratos.csv")
    assert {"AP", "PA"} <= set(e["estrato"])
    cv = pd.read_csv(tmp_path / "val" / "c1_v_mascara.csv").set_index("estimando")
    assert cv.loc["a_erm_mascara_menos_erm", "delta"] < 0
    # test: bloqueado sin congelación; con ella, mismas tablas sobre test_bimcv
    for k, v in (("RUTA_CONGELACION", tmp_path / "results/gate7/congelacion.json"), ("RUTA_STATUS", tmp_path / "STATUS.md"),
                 ("RUTA_REGISTRO", tmp_path / "results/gate7/desbloqueos.jsonl"), ("PROJECT_ROOT", tmp_path)):
        monkeypatch.setattr(D, k, v)
    with pytest.raises(D.TestBloqueadoError):
        auditoria_covidx.main(comunes + ["--c1", "test", "--out-c1", str(tmp_path / "test")])
    congelar_en(tmp_path, ["results/c1/caracteristicas/*", "configs/*.yaml"])
    assert auditoria_covidx.main(comunes + ["--c1", "test", "--out-c1", str(tmp_path / "test"), "--permitir-incompleto"]) == 0
    ft = pd.read_csv(tmp_path / "test" / "f_c1.csv").set_index("metodo")
    assert set(ft["conjunto"]) == {"test_bimcv"} and not ft.loc["lisa", "no_inferior_0.02"] and ft.loc["erm", "no_inferior_0.02"]
    aq = pd.read_csv(tmp_path / "test" / "c1_iv_adquisicion.csv").set_index("variante")
    assert aq.loc["b_validacion_cruzada_en_el_conjunto", "auc"] > 0.65 and (aq["delta_vs_erm"] < 0.2).all()
    gr = pd.read_csv(tmp_path / "test" / "c1_vi_covidgr.csv").set_index("metodo")
    assert {"erm", "lisa", "erm_mascara", "D_IN", "D_ERM"} <= set(gr.index)
    assert gr.loc["D_IN", "auc_externa"] < 0.7 and gr.loc["lisa", "auc_externa"] > 0.9       # detector cae; lisa no
    assert gr.loc["D_IN", "caida"] > 0.2 and abs(gr.loc["lisa", "caida"]) < 0.1
    gv = pd.read_csv(tmp_path / "test" / "c1_vi_gravedad.csv")
    assert set(gv["gravedad"]) == {"MILD", "SEVERE"}
    meta = json.loads((tmp_path / "test" / "c1.json").read_text())
    assert meta["congelacion"] and meta["conjunto_confirmatorio"] == "test_bimcv"
