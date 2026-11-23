"""Pruebas de los parches de M1 (logs/auditoria_r2/parches/0*.diff). Copiar a tests/ DESPUÉS de aplicar los parches:
    cp logs/auditoria_r2/parches/test_parches_m1.py tests/
    python3 -m pytest -q -p no:cacheprovider tests/test_parches_m1.py --basetemp=/tmp/pt_m1
Antes de aplicarlos fallan (es lo esperado: comprueban que cada camino exige la congelación con sus huellas)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from src.eval import desbloqueo as D  # noqa: E402


@pytest.fixture
def raiz(tmp_path, monkeypatch):
    for k, v in (("RUTA_CONGELACION", tmp_path / "results/gate7/congelacion.json"), ("RUTA_STATUS", tmp_path / "STATUS.md"),
                 ("RUTA_REGISTRO", tmp_path / "results/gate7/desbloqueos.jsonl"), ("PROJECT_ROOT", tmp_path)):
        monkeypatch.setattr(D, k, v)
    return tmp_path


def congelar(raiz: Path):
    import congelar as C
    import yaml
    (raiz / "f.txt").write_text("1")
    cfg = raiz / "c.yaml"
    cfg.write_text(yaml.safe_dump({"grupos": {"g": {"patrones": ["f.txt"], "minimo": 1}}}))
    assert C.main(["--config", str(cfg), "--raiz", str(raiz)]) == 0
    sha = D.sha256_fichero(raiz / "results/gate7/congelacion.json")
    (raiz / "STATUS.md").write_text("Puerta 7: congelacion.json %s\n" % sha)


def test_entrenador_comun_covidx(raiz):
    from src.train.trainer import bloqueo_test_covidx
    cfg = {"experiment": "x", "selection": {"subset": "val_robust"}, "protocol": {"test_bloqueado": True}}
    bloqueo_test_covidx(cfg, {"val", "val_robust"})
    with pytest.raises(D.TestBloqueadoError, match="bloqueado"):
        bloqueo_test_covidx(cfg, {"val", "test"})
    cfg["protocol"]["test_bloqueado"] = False
    with pytest.raises(D.TestBloqueadoError, match="Puerta 7"):
        bloqueo_test_covidx(cfg, {"test_bimcv"})
    with pytest.raises(D.TestBloqueadoError, match="selección"):
        bloqueo_test_covidx({"selection": {"subset": "test"}, "protocol": {"test_bloqueado": False}}, {"val"})
    congelar(raiz)
    bloqueo_test_covidx(cfg, {"test"})
    (raiz / "f.txt").write_text("2")
    with pytest.raises(D.TestBloqueadoError, match="cambiado"):
        bloqueo_test_covidx(cfg, {"test"})


def test_predict_sin_test_por_defecto(raiz):
    from src.eval import predict as P
    with pytest.raises(D.TestBloqueadoError):
        P.main(["--run-dir", str(raiz / "no_existe"), "--sets", "val", "test"])
    with pytest.raises(FileNotFoundError):            # sin --sets no pide test: llega a leer la configuración
        P.main(["--run-dir", str(raiz / "no_existe")])


def _cfg_lineas_base(tmp):
    from src.train.config import resolve
    c = resolve({"experiment": "x", "model": {"name": "tiny"}, "data": {"eval_sets": ["val", "test"]},
                 "baseline": {"method": "erm", "params": {}}, "protocol": {"test_bloqueado": False}})
    fz = tmp / "fz.json"
    fz.write_text(json.dumps({"method": "erm", "params": {}, "lr_B": c["train"]["phases"][-1]["lr"]}))
    c["protocol"]["congelado"] = str(fz)
    return c


def test_check_test_lock_exige_puerta7(raiz):
    from src.baselines.trainer import TestLockedError, check_test_lock
    c = _cfg_lineas_base(raiz)
    with pytest.raises(TestLockedError, match="Puerta 7"):
        check_test_lock(c)
    c["data"]["synthetic"] = {"n_train": 8}           # datos sintéticos de las pruebas: sin cambio
    assert check_test_lock(c) is False
    c["data"]["synthetic"] = None
    congelar(raiz)
    assert check_test_lock(c) is False


def test_bench_exige_huellas(raiz, monkeypatch):
    from src.train import bench as BM
    monkeypatch.setattr(BM, "GATE7", D.RUTA_CONGELACION)
    cfg = {"data": {"eval_sets": ["val", "test_desconf"]}, "selection": {"subset": "val"},
           "protocol": {"test_bloqueado": False, "congelado": "x.json"}}
    with pytest.raises(BM.TestBloqueadoError, match="Puerta 7"):
        BM.check_test(cfg)
    D.RUTA_CONGELACION.parent.mkdir(parents=True, exist_ok=True)
    D.RUTA_CONGELACION.write_text("{}")                # existir ya no basta
    with pytest.raises(BM.TestBloqueadoError, match="estructura"):
        BM.check_test(cfg)
    D.RUTA_CONGELACION.unlink()
    congelar(raiz)
    assert BM.check_test(cfg) is False
    (raiz / "f.txt").write_text("otra")
    with pytest.raises(BM.TestBloqueadoError, match="cambiado"):
        BM.check_test(cfg)


def test_csdnet_exige_puerta7(raiz):
    from src.baselines.trainer import TestLockedError
    from src.train.csdnet_loop import CSDNetTrainer
    fz = raiz / "fz.json"
    fz.write_text(json.dumps({"method": "csdnet", "params": {"model.csdnet": {}, "csdnet": {}}}))
    cfg = {"data": {"eval_sets": ["val", "test"]}, "selection": {"subset": "val"},
           "protocol": {"test_bloqueado": False, "congelado": str(fz)}, "model": {}}
    with pytest.raises(TestLockedError, match="Puerta 7"):
        CSDNetTrainer._check_test_lock(cfg)
    congelar(raiz)
    assert CSDNetTrainer._check_test_lock(cfg) is False
