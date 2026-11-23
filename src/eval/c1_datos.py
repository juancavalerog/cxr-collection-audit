"""Acceso común a corridas, conjuntos y cargadores para la auditoría C1 y la evaluación del test de COVIDx.

Lo usan scripts/eval/extraer_caracteristicas.py, scripts/eval/evaluar_test_b1.py y scripts/eval/inferencia_covidgr.py
(docs/14_auditoria_y_congelacion.md). No modifica el entrenador: reconstruye un `BaselineTrainer` desde el
`config_resolved.json` de una corrida, con el test bloqueado como en el entrenamiento, y le añade los marcos que
hagan falta con el mismo `_make_ds` (mismo preprocesado, caché y enmascarado que el entrenador usó en validación).

Bloqueo: `marco()` solo devuelve un test (test, test_bimcv) o COVIDGR si recibe el registro de la congelación que
devuelve `src.eval.desbloqueo.exigir_congelacion`; sin él lanza TestBloqueadoError antes de leer el manifiesto.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.data.paths import PROJECT_ROOT
from src.eval.desbloqueo import TestBloqueadoError, es_bloqueado

SEMILLA_MUESTRA = 20261113          # la de scripts/diag/expl_lectura_features.py (V2.3.1)
N_AJUSTE = 6000
COVIDGR_CSV = PROJECT_ROOT / "data" / "manifests" / "bench" / "covidgr" / "covidgr.csv"
COVIDGR_CACHE = PROJECT_ROOT / "data" / "cache_m" / "bench_covidgr" / "homogeneo_224"
COVIDGR_MASCARAS = PROJECT_ROOT / "data" / "masks" / "bench_covidgr" / "homogeneo_224"
CONJUNTOS_VALIDACION = ("tr_bimcv_pos", "val", "val_robust", "val_robust_sel", "res_neg", "res_pos_antigua",
                        "res_pos_nueva")
CONJUNTOS_BLOQUEADOS = ("test", "test_bimcv", "test_covidgr")


def rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


def leer_config(run: Path) -> dict:
    return json.load(open(Path(run) / "config_resolved.json", encoding="utf-8"))


def _arreglar_manifiesto(cfg: dict):
    m = cfg["data"].get("manifest")
    if m and not Path(m).exists():
        from src.data.paths import resolve_data
        if not Path(resolve_data(m)).exists():
            alt = PROJECT_ROOT / "data" / "manifests" / "expl" / Path(m).name     # misma copia, otra máquina
            if not alt.exists():
                raise SystemExit("no se encuentra el manifiesto %s" % m)
            cfg["data"]["manifest"] = str(alt)


def construir_trainer(cfg: dict, seed: int, workers: Optional[int] = None, preentrenado: bool = False,
                      dispositivo: Optional[str] = None, out_dir: Optional[Path] = None):
    """BaselineTrainer sin entrenar con la configuración dada. Con preentrenado=False la inicialización se anula
    (los pesos vienen de un estado guardado); con True se conserva `model.init` (D_IN: pesos de ImageNet)."""
    from src.baselines.trainer import BaselineTrainer
    cfg = copy.deepcopy(cfg)
    if not preentrenado:
        cfg["model"]["init"] = {"type": "none", "path": None}
    _arreglar_manifiesto(cfg)
    if workers is not None:
        cfg["data"]["num_workers"] = int(workers)
    if dispositivo:
        cfg["device"] = dispositivo
    od = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="c1_trainer_"))
    tr = BaselineTrainer(cfg, int(seed), resume=False, out_dir=str(od), echo=False)
    if not tr.test_locked:
        raise TestBloqueadoError("la configuración reconstruida no tiene el test bloqueado; este módulo añade el test "
                                 "por su cuenta y solo tras exigir_congelacion")
    if any(es_bloqueado(k) for k in tr.frames):
        raise TestBloqueadoError("el entrenador reconstruido trae conjuntos bloqueados: %s" % sorted(tr.frames))
    return tr


def trainer_de_corrida(run: Path, workers=None, dispositivo=None, out_dir=None):
    cfg = leer_config(run)
    return construir_trainer(cfg, int(cfg.get("seed", 0)), workers=workers, dispositivo=dispositivo, out_dir=out_dir)


def cargar_estado(tr, ruta_estado: Path):
    sd = torch.load(ruta_estado, map_location=tr.device)
    tr.model.load_state_dict(sd)
    tr.model.eval()
    return tr


def huella_tronco(net) -> str:
    """SHA256 de los tensores del tronco (nombres ordenados, bytes en float32): identifica los pesos de D_IN."""
    h = hashlib.sha256()
    sd = net.backbone.state_dict()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


# ===================================================================================== conjuntos
def _manifiesto(tr):
    from src.data.manifest import load_manifest
    from src.data.paths import resolve_data
    m = tr.cfg["data"].get("manifest")
    return load_manifest(resolve_data(m) if m else None)


def marco(tr, nombre: str, registro: Optional[dict] = None, n_ajuste: int = N_AJUSTE):
    """DataFrame del conjunto `nombre` con las columnas que espera el Dataset del entrenador."""
    import pandas as pd
    if es_bloqueado(nombre):
        if not registro or not registro.get("sha256_congelacion"):
            raise TestBloqueadoError("el conjunto %s exige la congelación de la Puerta 7 (exigir_congelacion)" % nombre)
    if nombre == "tr_bimcv_pos":
        fr = tr.frames["train"]
        pos = fr[(fr["source"] == "bimcv") & (fr["label"] == 1)]
        return pos.sample(n=min(int(n_ajuste), len(pos)), random_state=SEMILLA_MUESTRA).reset_index(drop=True)
    if nombre in tr.frames and not es_bloqueado(nombre):
        return tr.frames[nombre].reset_index(drop=True)
    if nombre.startswith("res_"):
        if not tr.cfg["data"].get("manifest"):
            return pd.DataFrame()
        man = _manifiesto(tr)
        r = man[man["split_clean"] == "reserva"]
        nueva = r["image_id"].str.extract(r"^sub-S(\d+)_")[0].str.len() == 6
        return {"res_neg": r[r["label"] == 0], "res_pos_antigua": r[(r["label"] == 1) & ~nueva],
                "res_pos_nueva": r[(r["label"] == 1) & nueva]}[nombre].reset_index(drop=True)
    if nombre in ("test", "test_bimcv"):
        dc = tr.cfg["data"]
        if dc.get("synthetic"):
            from src.train.synthetic import synthetic_frame
            sc = dc["synthetic"]
            t = synthetic_frame(sc.get("n_test", 32), int(sc.get("n_sources", 3)), tr.seed, "test")
            return t if nombre == "test" else t[t["source_idx"] == 0].reset_index(drop=True)
        from src.data.manifest import get_splits
        t = get_splits(dc["mode"], _manifiesto(tr))["test"]
        return t if nombre == "test" else t[t["source"] == "bimcv"].reset_index(drop=True)
    if nombre == "test_covidgr":
        from src.data.bench import load_bench
        frames, _ = load_bench(COVIDGR_CSV)
        return frames["test_covidgr"].reset_index(drop=True)
    raise ValueError("conjunto desconocido: %s" % nombre)


def dataset(tr, df, nombre: str):
    """Dataset con el preprocesado del entrenador. COVIDGR usa su propia caché y, en ERM con máscara, sus máscaras."""
    if nombre != "test_covidgr" or tr.cfg["data"].get("synthetic"):
        return tr._make_ds(df, nombre)
    from src.baselines.trainer import MaskedDataset, MaskStore
    from src.data.datasets import CXRDataset
    dc = tr.cfg["data"]
    ds = CXRDataset(df, dc["channel"], dc["res"], COVIDGR_CACHE, True)
    if getattr(tr.method, "needs_mask", False):
        st = MaskStore(COVIDGR_MASCARAS, tr.method.p.get("mask_resize", False))
        st.check(df["image_id"].astype(str).tolist(), " de COVIDGR")
        ds = MaskedDataset(ds, st)
    return ds


def cargador(tr, df, nombre: str):
    from torch.utils.data import DataLoader
    from src.train.trainer import _collate
    dc = tr.cfg["data"]
    nw = 0 if dc.get("synthetic") else int(dc["num_workers"])
    return DataLoader(dataset(tr, df, nombre), batch_size=int(dc["eval_batch_size"]), shuffle=False, num_workers=nw,
                      collate_fn=_collate, pin_memory=tr.device.type == "cuda")


def predecir(tr, df, nombre: str, max_n: int = 0) -> dict:
    """Probabilidades con el mismo preprocesado, TTA (eval.tta) y precisión que el entrenador en su evaluación final."""
    from src.eval.predict import predict_loader
    if max_n:
        df = df.iloc[:int(max_n)].reset_index(drop=True)
    return predict_loader(tr.model, cargador(tr, df, nombre), tr.device, tta=bool(tr.cfg["eval"]["tta"]),
                          precision=tr.precision)


def predecir_como_entrenador(tr, nombre: str, max_n: int = 0) -> dict:
    """Igual que Trainer.predict(nombre, tta=eval.tta) sobre un conjunto propio del entrenador (validación)."""
    if max_n:
        return predecir(tr, tr.frames[nombre], nombre, max_n)
    return tr.predict(nombre, tta=bool(tr.cfg["eval"]["tta"]))


# ===================================================================================== vigilancia en el servidor
class Parada(Exception):
    """Temperatura por encima del límite o SIGTERM del ejecutor: el guion sale con 75 sin escribir resultados."""


class Vigilante:
    """Doble seguro térmico y SIGTERM para los guiones de inferencia (como el entrenador: runner/thermal.py)."""

    def __init__(self, every: int = 20, log=print):
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "runner"))
        from thermal import GracefulStop, ThermalSelfCheck
        self.stop = GracefulStop().install()
        self.guard = ThermalSelfCheck(every=every, log=lambda m: log(m))

    def step(self):
        from thermal import ThermalLimitReached
        try:
            self.guard.step()
        except ThermalLimitReached as e:
            raise Parada("temperatura: %s" % e)
        if self.stop.requested:
            raise Parada("SIGTERM (señal %s)" % self.stop.signum)

    def cerrar(self):
        self.stop.restore()


EXIT_PARADA = 75


def escribir_atomico(path: Path, texto: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(texto, encoding="utf-8")
    os.replace(tmp, path)
