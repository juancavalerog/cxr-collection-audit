"""Bucle de entrenamiento de CSD-Net: Algoritmo 1 de docs/03_arquitectura.md (Section 2.8) sobre el entrenador
común de src/train/trainer.py (subclase; no modifica el entrenador).

Hereda del entrenador común: directorio de salida y registro JSONL, semillas y determinismo, checkpoint atómico
(`checkpoint.pt`) con reanudación exacta (modelo, optimizador, EMA, contadores, pérdidas parciales y generadores),
SIGTERM y doble seguro térmico cada `thermal.every` pasos con salida 75, y el guardado de probabilidades por
estado y conjunto (`predictions/<estado>_<conjunto>.csv`).

Lo propio de CSD-Net:
- Muestreador guiado por la tabla (src/model/swap.py, `TableGuidedBatchSampler`) con la tabla clase por fuente
  del train, plan de enrutamiento `ClassSourceTable.routing(lambda_swap)` (ecuaciones 9 a 12) o aleatorio
  (ablación A2), y emparejamientos `sample_swap_pairs` y `sample_within_pairs`.
- Tres vistas en la fase B (real, re-firmada cruzada y re-firmada dentro de la fuente) y pérdidas de
  src/model/losses.py con los pesos de la Tabla 6 (`LossWeights`).
- Calendario por épocas globales: A1 (L_cls_real, L_src, L_ind, L_seg; lambda_sig = 0), A2 (se añade L_sig),
  B (vistas cruzada y dentro de la fuente, rampa lineal de E_rampa épocas sobre L_cls_swap y L_cons).
- Grupos de parámetros de `CSDNet.param_groups()` con las tasas de la Tabla 6 (tronco convolucional 5e-5,
  Transformer 1e-4, módulos nuevos 1e-3), AdamW con wd 0.05 sin decaimiento en normas y sesgos, calentamiento
  del 5 % y coseno sobre el total de épocas, recorte de norma 1.0, bf16 con autocast en GPU.
- Máscara de la compuerta: la del profesor mientras el Dice de la cabeza de segmentación contra el profesor en
  val sea < 0.90 o la fase no sea B; después, la predicha.
- Regla de parada de V1 al final de A2 (src/eval/probes.py): si V1 falla no se activa el intercambio
  (`v1_fail_action`: "stop" termina tras A2, como el Algoritmo 1; "continue_no_swap" sigue la fase B sin
  vistas re-firmadas y marca la corrida).
- Punto de control de V2, V2b, V3 y V5 al final de la primera época con la rampa completa; V2b y V5 cada
  `probe_every` épocas de B (solo registro); informe final de V0 a V6 sobre val.
- Parada temprana por AUC-ROC en val_robust (paciencia 10); estados last, best y EMA.
- Registro por época de cada pérdida, de la desviación de P(s | y) de los lotes respecto de la tabla y de la
  distancia entre clases de la distribución de fuente efectiva de las vistas (ecuación 9).
- Bloqueo del test con la misma regla que las líneas base (src/baselines/trainer.py, check_test_lock): claves
  `protocol.test_bloqueado` (por defecto verdadera) y `protocol.congelado`. Con el test bloqueado, ningún conjunto
  test o test_* puede figurar en eval_sets ni en la selección (TestLockedError), no se construye y el cargador de
  evaluación lo rechaza. Para desbloquearlo hace falta `protocol.test_bloqueado: false` y un JSON congelado con
  {"method": "csdnet", "params": {"model.csdnet": ..., "csdnet": ...}} idéntico a la configuración de la corrida.
  (check_test_lock de las líneas base compara con su registro de métodos, que no incluye CSD-Net; por eso se
  replica su semántica aquí y se reutiliza su excepción.)
- Validación robusta partida (src/baselines/methods/dfr.py, split_val_robust): se construyen `val_robust_fit` y
  `val_robust_sel`; por defecto la selección se hace en val_robust_sel, como en las líneas base.

Enmienda del 13-11-2026 (docs/10_correccion_independencia.md):
- `csdnet.ind` elige la penalización de independencia (src/model/losses.py, IndConfig): por defecto
  "mmd_banco_tope" (MMD insesgada con banco de firmas por fuente mixta y clase y tope relativo de su gradiente);
  "mmd_sesgada" con lambda_ind = 1.0 reproduce el primer intento del piloto. El banco se guarda en el checkpoint.
- Registro por paso de H_src_lote (entropía de las fuentes del lote), L_src_menos_H y desv_firma; por época,
  la entropía de la composición de fuentes de la época (H_fuente_epoca) y L_src_menos_H_epoca.
- Control de validez de la firma al final de A2, antes de V1 (`csdnet.firma_check`): si la L_src media de la última
  época no queda al menos `margen_nats` por debajo de H_fuente_epoca, la firma no ha aprendido la fuente y V1 queda
  "no_concluyente" con motivo "firma sin información de fuente" (no aprobada ni fallida); el intercambio no se activa.
  No cambia ni el umbral ni la definición de V1: es una condición previa para que V1 tenga sentido.
- `csdnet.parar_tras_A2` (diagnóstico): tras V1 se ejecuta también V2 y la corrida termina sin fase B.

Ablaciones: las de `CSDNetConfig` (A1 use_acquisition, A3 use_lung_exclusion, A4 use_gate, A6 use_highpass,
A7 refirma_stage, A8 refirma_mode, A12 exclusion_mode, A13 transformer_init, A14 refirma_target) van en
`model.csdnet`; las de pérdida (A5 lambda_cons, A9 lambda_ind, A10 lambda_swap = lambda_cons = 0, A11
lambda_swap = 0.5 con plan recalculado, A15 rho_within = 0) en `csdnet.weights`; A2 en `csdnet.swap_routing`.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.paths import PROJECT_ROOT, resolve_data
from src.train.ema import WeightEMA
from src.train.metrics import exact_metric
from src.train.optim import warmup_cosine
from src.train.trainer import (EXIT_DEBUG_STOP, EXIT_THERMAL, GracefulStop, ThermalLimitReached, ThermalSelfCheck,
                               Trainer, TrainStop, _collate, atomic_torch_save)

CSD_DEFAULTS = {
    "weights": {"lambda_swap": 1.0, "lambda_cons": 1.0, "lambda_src": 0.5, "lambda_seg": 0.5, "lambda_sig": 0.1,
                "lambda_ind": 10.0, "rho_within": 0.5, "focal_alpha": 0.5, "focal_gamma": 2.0},
    "ind": {"mode": "mmd_banco_tope", "bank_size": 256, "beta": 0.5, "desde": "A1"},   # enmienda de 13-11-2026
    "firma_check": {"activo": True, "margen_nats": 0.05},
    "parar_tras_A2": False,
    "epochs": {"A1": 3, "A2": 2, "ramp": 2, "B": 60},
    "lr": {"trunk_conv": 5e-5, "transformer": 1e-4, "new": 1e-3},
    "weight_decay": 0.05, "warmup_frac": 0.05, "min_lr_frac": 1e-3, "grad_clip": 1.0,
    "sampler": {"min_per_source": 2, "min_per_class_mixed": 4},
    "swap_routing": "table",            # table | random (ablación A2)
    "plan_table": "lotes",              # lotes: el plan se calcula con la composición real de los lotes | dataset
    "gate_dice_threshold": 0.90,
    "seg_dice_max_n": 512,
    "v1_fail_action": "stop",           # stop | continue_no_swap | continue_swap (solo con v1_fail_enmienda)
    "v1_fail_enmienda": None,           # identificador de la enmienda de docs/07_piloto.md que autoriza continue_swap
    "v1_inconclusive": "activar",       # activar (y marcar) | no_activar
    "probe_every": 5,
    "probe_rows_cap": 600,
    "probe_n_boot": 1000,
    "probes": {},                       # sustituye DEFAULT_PROBE_CFG (nombres de fuentes y estratos)
    "final_probe_states": ["best"],
    "eval_gate": "same_as_train",       # same_as_train | teacher | predicted
    "v1_gate_override": None,           # solo pruebas: "aprueba" o "falla" sustituye el veredicto de V1 en la puerta
    "ema_mode": "estandar", "ema_momentum": 0.9999,
}
LOCKED_PREFIX = ("test",)


def is_locked_set(name: str) -> bool:
    return name == "test" or name.startswith("test_")


class _ListBatchSampler:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def batch_composition_table(labels, sources, num_sources, names, batch_size, min_per_source, min_per_class_mixed,
                            seed: int = 0, n_batches: int = 2000):
    """Tabla clase por fuente de la composición real de los lotes (enmienda del 13-11-2026).

    El muestreador reserva cuotas por fuente y por clase en las fuentes mixtas (40 de 64 imágenes con los valores
    por defecto), de modo que P(s | y) dentro del lote se aparta mucho de la tabla del conjunto de datos (hasta
    0.46 en COVIDx CXR-4). La Proposición 1 iguala la fuente efectiva entre clases solo si el plan se calcula con
    la distribución con la que de verdad se muestrea: con el plan de la tabla del conjunto queda una distancia de
    variación total de 0.24 entre clases, y con el de la composición de los lotes, de 0.004. Aquí se simula el
    mismo muestreador (época reservada, que no coincide con ningún lote de entrenamiento) y se cuenta."""
    from src.model.swap import ClassSourceTable, TableGuidedBatchSampler
    labels = np.asarray(labels, dtype=np.int64)
    sources = np.asarray(sources, dtype=np.int64)
    smp = TableGuidedBatchSampler(labels, sources, int(batch_size), int(n_batches), int(min_per_source),
                                  int(min_per_class_mixed), seed=int(seed))
    smp.set_epoch(10 ** 6)
    idx = np.concatenate([np.asarray(b, dtype=np.int64) for b in smp])
    return ClassSourceTable.from_arrays(labels[idx], sources[idx], int(num_sources), list(names))


class CSDNetTrainer(Trainer):
    """Entrenador de CSD-Net (Algoritmo 1). Misma interfaz que Trainer: CSDNetTrainer(cfg, seed).fit()."""

    def __init__(self, cfg: dict, seed: int, resume: bool = True, out_dir=None, echo=True):
        cfg = copy.deepcopy(cfg)
        from src.train.config import deep_merge
        cfg["csdnet"] = deep_merge(CSD_DEFAULTS, cfg.get("csdnet") or {})
        cfg.setdefault("model", {})["name"] = "csdnet"
        self.cc = cfg["csdnet"]
        if self.cc["v1_fail_action"] not in ("stop", "continue_no_swap", "continue_swap"):
            raise ValueError("csdnet.v1_fail_action desconocido: %r" % (self.cc["v1_fail_action"],))
        if self.cc["v1_fail_action"] == "continue_swap" and not str(self.cc.get("v1_fail_enmienda") or "").strip():
            raise ValueError("csdnet.v1_fail_action = continue_swap exige csdnet.v1_fail_enmienda: seguir con el "
                             "intercambio tras un fallo de V1 cambia una regla registrada y debe citar la enmienda "
                             "de docs/07_piloto.md que lo autoriza")
        if self.cc.get("v1_gate_override") and (cfg.get("data") or {}).get("synthetic_csd") is None:
            raise ValueError("csdnet.v1_gate_override solo se admite con datos sintéticos de prueba "
                             "(data.synthetic_csd); en una corrida real el veredicto de V1 no se puede sustituir")
        self.test_locked = self._check_test_lock(cfg)
        super().__init__(cfg, seed, resume=False, out_dir=out_dir, echo=echo)
        self.cc = self.cfg["csdnet"]
        from src.model.losses import IndConfig, SignatureBank
        self.ind_cfg = IndConfig(**{k: v for k, v in (self.cc.get("ind") or {}).items()})
        self.ind_bank = SignatureBank(self.ind_cfg.bank_size) if self.ind_cfg.mode == "mmd_banco_tope" else None
        st = self.state
        st.update({"phase_name": "A1", "swap_enabled": None, "gate_pred": False, "v1": None, "flags": {},
                   "probe_reports": [], "stopped_reason": None, "epoch_acc": None})
        self._setup_optimizer()
        if resume and self.ckpt_path.exists():
            self._load_checkpoint()
            self.resumed = True

    # ================================================================== bloqueo del test
    @staticmethod
    def _check_test_lock(cfg):
        """Misma semántica que src/baselines/trainer.check_test_lock; devuelve si el test queda bloqueado."""
        from src.baselines.trainer import TestLockedError
        pc = cfg.get("protocol") or {}
        locked = bool(pc.get("test_bloqueado", True))
        sel = ((cfg.get("selection") or {}).get("subset")) or "val"
        if is_locked_set(str(sel)):
            raise TestLockedError("la selección de modelo no puede hacerse en '%s'" % sel)
        wants = [n for n in ((cfg.get("data") or {}).get("eval_sets") or ["val", "test"]) if is_locked_set(str(n))]
        if not wants:
            return locked
        if locked:
            raise TestLockedError("test bloqueado (protocol.test_bloqueado: true) y eval_sets pide %s" % wants)
        path = pc.get("congelado")
        if not path:
            raise TestLockedError("test desbloqueado sin protocol.congelado (JSON de hiperparámetros congelados)")
        p = Path(path) if os.path.isabs(str(path)) else PROJECT_ROOT / path
        if not p.exists():
            raise TestLockedError("no existe el fichero de hiperparámetros congelados %s" % p)
        fz = json.load(open(p, encoding="utf-8"))
        run = {"model.csdnet": (cfg.get("model") or {}).get("csdnet") or {}, "csdnet": cfg.get("csdnet") or {}}
        if fz.get("method") != "csdnet" or json.dumps(fz.get("params"), sort_keys=True, default=str) != \
                json.dumps(run, sort_keys=True, default=str):
            raise TestLockedError("los hiperparámetros de la corrida no son los congelados en %s" % p)
        if not (cfg.get("data") or {}).get("synthetic"):      # M1: congelación del protocolo con sus huellas
            from src.eval.desbloqueo import TestBloqueadoError, exigir_congelacion
            try:
                exigir_congelacion("CSD-Net: conjuntos %s" % wants)
            except TestBloqueadoError as e:
                raise TestLockedError(str(e)) from e
        return False

    # ================================================================== datos
    def _build_data(self):
        dc = self.cfg["data"]
        names = [n for n in (list(dc["eval_sets"]) + [self.cfg["selection"]["subset"]]) if n != "train"]
        if self.test_locked and any(is_locked_set(n) for n in names):
            raise RuntimeError("test bloqueado")
        if dc.get("synthetic_csd") is not None:
            from src.train.csdnet_synthetic import SyntheticCSD, synthetic_csd_frame
            sc = dc["synthetic_csd"] or {}
            spec, res = sc.get("spec"), int(sc.get("res", 64))
            self.frames = {"train": synthetic_csd_frame("train", spec), "val": synthetic_csd_frame("val", spec)}
            self.frames["val_robust"] = self.frames["val"][self.frames["val"]["val_robust"] == 1].reset_index(drop=True)
            self._split_robust()
            self.source_names = sorted(set(self.frames["train"]["source"].unique().tolist()) |
                                       set(self.frames["val"]["source"].unique().tolist()))
            self._make_ds = lambda df, split: SyntheticCSD(df, res, seed=0 if split == "train" else 1)
        else:
            from src.data.homogeneo_masks import CXRMaskDataset, masks_cache_dir
            from src.data.manifest import COVIDX_SOURCES, get_eval_set, get_splits, load_manifest
            man = load_manifest(resolve_data(dc["manifest"]) if dc.get("manifest") else None)
            sp = get_splits(dc["mode"], man, dc.get("train_sources"))
            self.frames = {"train": sp["train"], "val": sp["val"], "val_robust": get_eval_set("val_robust", dc["mode"], man)}
            self._split_robust()
            for n in set(names) - set(self.frames):
                self.frames[n] = get_eval_set(n, dc["mode"], man)
            cd = dc.get("cache")
            cd = PROJECT_ROOT / "data" / "cache_m" / dc["dataset"] / ("homogeneo_%d" % int(dc["res"])) \
                if cd in (None, "auto") else (Path(cd) if os.path.isabs(cd) else PROJECT_ROOT / cd)
            md = dc.get("masks")
            md = masks_cache_dir(dc["dataset"], int(dc["res"])) if md in (None, "auto") else (
                Path(md) if os.path.isabs(md) else PROJECT_ROOT / md)
            self.cache_dir, self.mask_dir = cd, md
            lim = dc.get("debug_limit")
            if lim:
                import pandas as pd
                self.frames = {k: pd.concat([g.head(int(lim)) for _, g in v.groupby("label")]).reset_index(drop=True)
                               for k, v in self.frames.items()}
            self.source_names = list(COVIDX_SOURCES)
            self._make_ds = lambda df, split: CXRMaskDataset(df, cd, md, require=dc.get("require_cache", True))
        self.train_ds = self._make_ds(self.frames["train"], "train")
        tc = self.cfg["train"]
        self.steps_per_epoch = int(tc["steps_per_epoch"] or math.ceil(len(self.train_ds) / tc["batch_size"]))
        from src.model.swap import ClassSourceTable
        K = len(self.source_names)
        self.table = ClassSourceTable.from_arrays(self.train_ds.labels, self.train_ds.sources, K, list(self.source_names))
        self.mixed = self.table.mixed_sources()
        self.plan_table = self.table
        if self.cc["swap_routing"] != "random" and self.cc.get("plan_table", "lotes") == "lotes":
            self.plan_table = batch_composition_table(self.train_ds.labels, self.train_ds.sources, K,
                                                      list(self.source_names), int(tc["batch_size"]),
                                                      int(self.cc["sampler"]["min_per_source"]),
                                                      int(self.cc["sampler"]["min_per_class_mixed"]), seed=self.seed)
        w = self.cc["weights"]
        lam = float(w["lambda_swap"]) if float(w["lambda_swap"]) > 0 else 1.0
        if self.cc["swap_routing"] == "random":
            r = np.ones((2, K, K)) - np.eye(K)[None]
            r = r / r.sum(-1, keepdims=True)
            self.plan = None
            self.routing = torch.tensor(r, dtype=torch.float32)
        else:
            self.plan = self.plan_table.routing(lam)
            self.routing = torch.tensor(self.plan.routing, dtype=torch.float32)

    def _split_robust(self):
        """val_robust_fit y val_robust_sel: la partición fija por paciente de las líneas base (DFR)."""
        from src.baselines.methods.dfr import split_val_robust
        fit, sel = split_val_robust(self.frames["val_robust"])
        self.frames["val_robust_fit"], self.frames["val_robust_sel"] = fit, sel

    def eval_loader(self, name: str):
        if self.test_locked and is_locked_set(name):
            raise PermissionError("test bloqueado: no se infiere sobre %s en esta corrida" % name)
        dc = self.cfg["data"]
        ds = self._make_ds(self.frames[name], name)
        nw = 0 if dc.get("synthetic_csd") is not None else int(dc["num_workers"])
        g = torch.Generator(); g.manual_seed(self.seed)
        return DataLoader(ds, batch_size=int(dc["eval_batch_size"]), shuffle=False, num_workers=nw,
                          collate_fn=_collate, pin_memory=self.device.type == "cuda", generator=g)

    # ================================================================== modelo y optimizador
    def _build_model(self):
        from src.model import CSDNet, CSDNetConfig
        mc = dict(self.cfg["model"].get("csdnet") or {})
        mc.setdefault("num_sources", len(self.source_names))
        mc.setdefault("img_size", int(self.cfg["data"].get("res", 224)) if self.cfg["data"].get("synthetic_csd") is None
                      else int((self.cfg["data"]["synthetic_csd"] or {}).get("res", 64)))
        for k in ("acq_widths", "mean", "std"):
            if k in mc and isinstance(mc[k], list):
                mc[k] = tuple(mc[k])
        self.model = CSDNet(CSDNetConfig(**mc)).to(self.device)
        self.mcfg = self.model.cfg
        self.uses_swap = (self.mcfg.refirma_target != "canonical") and (
            float(self.cc["weights"]["lambda_swap"]) > 0 or float(self.cc["weights"]["lambda_cons"]) > 0)
        self.uses_within = self.uses_swap and float(self.cc["weights"]["rho_within"]) > 0
        self.register_norm()

    def register_norm(self):
        self._mean = torch.tensor(self.mcfg.mean, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(self.mcfg.std, device=self.device).view(1, 3, 1, 1)

    def normalize(self, x):
        return (x - self._mean) / self._std

    def _setup_optimizer(self):
        groups = self.model.param_groups()
        lr = self.cc["lr"]
        peak = {"trunk_conv": lr["trunk_conv"], "transformer": lr["transformer"]}
        pg = []
        for name, ps in groups.items():
            p_lr = float(peak.get(name, lr["new"]))
            dec = [p for p in ps if p.ndim > 1]
            nod = [p for p in ps if p.ndim <= 1]
            for tag, lst, wd in (("", dec, float(self.cc["weight_decay"])), ("_nd", nod, 0.0)):
                if lst:
                    pg.append({"params": lst, "name": name + tag, "peak_lr": p_lr, "lr": p_lr, "weight_decay": wd})
        seen = {id(p) for g in pg for p in g["params"]}
        rest = [p for p in self.model.parameters() if id(p) not in seen]
        if rest:
            pg.append({"params": rest, "name": "otros", "peak_lr": float(lr["new"]), "lr": float(lr["new"]),
                       "weight_decay": 0.0})
        self.opt = torch.optim.AdamW(pg, betas=(0.9, 0.999), eps=1e-8)
        ep = self.cc["epochs"]
        self.E_A1, self.E_A2, self.E_ramp, self.E_B = int(ep["A1"]), int(ep["A2"]), int(ep["ramp"]), int(ep["B"])
        self.E_total = self.E_A1 + self.E_A2 + self.E_B
        self.total_steps = self.E_total * self.steps_per_epoch
        self.warmup_steps = int(self.total_steps * float(self.cc["warmup_frac"]))
        self.ema = WeightEMA(self.model, float(self.cc["ema_momentum"]), self.cc["ema_mode"])

    def _setup_phase(self, fresh: bool):          # el entrenador común lo llama; aquí no hay fases con optimizador propio
        return None

    def set_lrs(self, step):
        f = warmup_cosine(step, 1.0, self.total_steps, self.warmup_steps, float(self.cc["min_lr_frac"]))
        for g in self.opt.param_groups:
            g["lr"] = g["peak_lr"] * f
        return f

    def save_checkpoint(self, reason="epoch"):
        """El del entrenador común más el banco de firmas de L_ind (reanudación exacta)."""
        obj = {"model": self.model.state_dict(), "optimizer": self.opt.state_dict() if self.opt else None,
               "ema": self.ema.state_dict() if self.ema else None, "state": self.state, "rng": self._rng_state(),
               "reason": reason, "ind_bank": self.ind_bank.state_dict() if self.ind_bank is not None else None}
        atomic_torch_save(obj, self.ckpt_path)
        self.log("checkpoint", reason=reason, path=str(self.ckpt_path), **{k: self.state[k] for k in
                 ("phase", "epoch", "step_in_epoch", "phase_step", "global_step")})

    def signature_validity(self, rec):
        """Control de validez previo a V1 (enmienda de 13-11-2026): ¿ha aprendido la firma la fuente?
        Compara la L_src media de la época con la entropía de la composición de fuentes de esa época."""
        fc = self.cc.get("firma_check") or {}
        ls, h = (rec.get("losses") or {}).get("L_src"), rec.get("H_fuente_epoca")
        res = {"activo": bool(fc.get("activo", True)), "margen_nats": float(fc.get("margen_nats", 0.05)),
               "L_src_epoca": ls, "H_fuente_epoca": h, "desv_firma_epoca": (rec.get("losses") or {}).get("desv_firma")}
        if ls is None or h is None:
            res["valida"] = None
            res["motivo"] = "sin L_src en la época (sin vía de adquisición)"
            return res
        res["diferencia_nats"] = float(ls - h)
        res["valida"] = bool(ls < h - res["margen_nats"])
        res["motivo"] = None if res["valida"] else "firma sin información de fuente"
        return res

    def _load_checkpoint(self):
        ck = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        if ck.get("ind_bank") is not None and getattr(self, "ind_bank", None) is not None:
            self.ind_bank.load_state_dict(ck["ind_bank"])
        if ck["state"].get("config_hash") != self.state["config_hash"]:
            self.log("aviso", msg="la configuración cambió desde el checkpoint; se reanuda igualmente")
        self.model.load_state_dict(ck["model"])
        self.state = ck["state"]
        if ck.get("optimizer") is not None:
            self.opt.load_state_dict(ck["optimizer"])
        if ck.get("ema") is not None:
            self.ema.load_state_dict(ck["ema"])
        self._set_rng_state(ck["rng"])
        self.log("resume", **{k: self.state[k] for k in ("phase", "epoch", "step_in_epoch", "phase_step", "global_step")})

    # ================================================================== calendario
    def phase_of(self, e: int) -> str:
        if e < self.E_A1:
            return "A1"
        if e < self.E_A1 + self.E_A2:
            return "A2"
        return "B"

    def ramp(self, e: int, step: int) -> float:
        if self.E_ramp <= 0:
            return 1.0
        return float(min(1.0, max(0.0, (e - self.E_A1 - self.E_A2 + step / self.steps_per_epoch) / self.E_ramp)))

    def loss_weights(self, phase: str):
        from src.model.losses import LossWeights
        w = LossWeights(**{k: float(v) for k, v in self.cc["weights"].items()})
        if phase == "A1":
            w.lambda_sig = 0.0
            if getattr(self, "ind_cfg", None) is not None and self.ind_cfg.desde == "A2":
                w.lambda_ind = 0.0
        if not self.uses_within:
            w.rho_within = 0.0
        return w

    def batches_for_epoch(self, e: int, start: int):
        from src.model.swap import TableGuidedBatchSampler
        smp = TableGuidedBatchSampler(self.train_ds.labels, self.train_ds.sources, int(self.cfg["train"]["batch_size"]),
                                      self.steps_per_epoch, int(self.cc["sampler"]["min_per_source"]),
                                      int(self.cc["sampler"]["min_per_class_mixed"]), seed=self.seed)
        smp.set_epoch(e)
        return list(smp)[start:]

    def train_loader_epoch(self, e: int, start: int):
        dc = self.cfg["data"]
        nw = 0 if dc.get("synthetic_csd") is not None else int(dc["num_workers"])
        return DataLoader(self.train_ds, batch_sampler=_ListBatchSampler(self.batches_for_epoch(e, start)),
                          num_workers=nw, collate_fn=_collate, pin_memory=self.device.type == "cuda",
                          prefetch_factor=4 if nw else None, generator=torch.Generator().manual_seed(self.seed + e))

    # ================================================================== paso
    def train_step(self, batch, e: int):
        from src.data.datasets import make_generator
        from src.model.losses import L_total
        from src.model.swap import sample_swap_pairs, sample_within_pairs
        phase = self.phase_of(e)
        self.model.train()
        dev = self.device
        x = batch["image"].to(dev, non_blocking=True).float()
        y = batch["label"].to(dev).float()
        s = batch["source"].to(dev).long()
        m = batch["mask"].to(dev).float()
        q = batch["quality"].to(dev).float()
        gen = make_generator(self.seed * 13 + 1, self.state["global_step"], dev)
        if self.augment is not None:
            x, y, m = self.augment(x, y, gen, mask=m)
            m = (m > 0.5).float()
        x = self.normalize(x)
        acq_mask, gate_t = self.masks_for_quality(m, q)
        gate_mask = None if (phase == "B" and self.state["gate_pred"]) else gate_t
        w = self.loss_weights(phase)
        swap = phase == "B" and bool(self.state["swap_enabled"]) and self.uses_swap
        cpu_gen = torch.Generator().manual_seed(int(self.seed * 1_000_003 + self.state["global_step"]))
        valid, ramp, pairs = None, 0.0, None
        if swap:
            pairs = sample_swap_pairs(s, y, self.routing, generator=cpu_gen)
            within = sample_within_pairs(s, generator=cpu_gen) if self.uses_within else None
            ramp = self.ramp(e, self.state["step_in_epoch"] + 1)
            kw = {"swap_index": pairs.donor_index, "within_index": within.donor_index if within is not None else None}
            valid = pairs.valid
        else:
            kw = {}
        f = self.set_lrs(self.state["global_step"])
        self.opt.zero_grad(set_to_none=True)
        with self._autocast():
            out = self.model(x, lung_mask=acq_mask, gate_mask=gate_mask, **kw)
        loss, logs = L_total(out, y, s, w, valid=valid, mask=m, mask_quality=q, mixed_sources=self.mixed, ramp=ramp,
                             ind=self.ind_cfg, bank=self.ind_bank)
        if not torch.isfinite(loss):
            raise FloatingPointError("pérdida no finita en el paso %d: %s" % (self.state["global_step"], logs))
        loss.backward()
        if self.cc.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                                           float(self.cc["grad_clip"]))
        self.opt.step()
        self.ema.update(self.model)
        if self.ind_bank is not None and "signature" in out:
            self.ind_bank.update(out["signature"], s, y, self.mixed)
        logs["ramp"] = ramp
        logs["lr_factor"] = f
        logs["n_views"] = 1 + int("logit_swap" in out) + int("logit_within" in out)
        self._accumulate(logs, y, s, pairs, w, ramp)
        return logs

    @staticmethod
    def masks_for_quality(m, q):
        """Con q = 0 la pseudomáscara no se usa (docs/03_arquitectura.md, Section 2.5): para la entrada de
        adquisición se pasa una máscara vacía (el modelo excluye entonces la caja central [0.1, 0.9], ecuación 15)
        y para la compuerta la misma caja central como región torácica aproximada. L_seg la pondera con q = 0."""
        bad = (q <= 0).view(-1, 1, 1, 1)
        if not bool(bad.any()):
            return m, m
        h, w = m.shape[-2:]
        box = torch.zeros_like(m[:1])
        box[..., int(0.1 * h):int(0.9 * h), int(0.1 * w):int(0.9 * w)] = 1.0
        return torch.where(bad, torch.zeros_like(m), m), torch.where(bad, box.expand_as(m), m)

    def _accumulate(self, logs, y, s, pairs, w, ramp):
        K = len(self.source_names)
        acc = self.state.get("epoch_acc")
        if acc is None:
            acc = {"sums": {}, "n": 0, "counts": np.zeros((2, K)).tolist(), "eff": np.zeros((2, K)).tolist()}
            self.state["epoch_acc"] = acc
        for k, v in logs.items():
            acc["sums"][k] = acc["sums"].get(k, 0.0) + float(v)
        acc["n"] += 1
        cnt, eff = np.asarray(acc["counts"]), np.asarray(acc["eff"])
        yy, ss = y.long().cpu().numpy(), s.long().cpu().numpy()
        np.add.at(cnt, (yy, ss), 1.0)
        np.add.at(eff, (yy, ss), 1.0)
        if pairs is not None:
            v = pairs.valid.cpu().numpy().astype(bool)
            np.add.at(eff, (yy[v], pairs.target_source.cpu().numpy()[v]), float(w.lambda_swap) * max(ramp, 0.0))
        acc["counts"], acc["eff"] = cnt.tolist(), eff.tolist()

    def _epoch_table_stats(self):
        acc = self.state.get("epoch_acc") or {}
        out = {}
        if not acc:
            return out
        cnt, eff = np.asarray(acc["counts"]), np.asarray(acc["eff"])
        P = self.plan_table.conditional()               # la tabla con la que se calculó el plan
        Pb = cnt / np.maximum(cnt.sum(1, keepdims=True), 1e-12)
        out["dev_Psy_max_abs"] = float(np.abs(Pb - P).max())
        out["dev_Psy_dataset_max_abs"] = float(np.abs(Pb - self.table.conditional()).max())
        out["dev_Psy_tv"] = [float(0.5 * np.abs(Pb[c] - P[c]).sum()) for c in (0, 1)]
        Pe = eff / np.maximum(eff.sum(1, keepdims=True), 1e-12)
        out["tv_fuente_efectiva_entre_clases"] = float(0.5 * np.abs(Pe[0] - Pe[1]).sum())
        out["losses"] = {k: v / max(acc["n"], 1) for k, v in acc["sums"].items()}
        ps = cnt.sum(0)
        ps = ps[ps > 0] / max(ps.sum(), 1e-12)
        out["H_fuente_epoca"] = float(-(ps * np.log(ps)).sum())       # L_src de una cabeza que solo da frecuencias
        if "L_src" in out["losses"]:
            out["L_src_menos_H_epoca"] = float(out["losses"]["L_src"] - out["H_fuente_epoca"])
        return out

    # ================================================================== evaluación
    def _gate_for_eval(self, m):
        mode = self.cc["eval_gate"]
        if mode == "teacher":
            return m
        if mode == "predicted":
            return None
        return None if self.state.get("gate_pred") else m

    @torch.no_grad()
    def predict(self, name: str, tta: bool, max_n=None):
        loader = self.eval_loader(name)
        self.model.eval()
        ids, labels, sources, patients, probs = [], [], [], [], []
        n = 0
        for b in loader:
            x = self.normalize(b["image"].to(self.device).float())
            m, gm = self.masks_for_quality(b["mask"].to(self.device).float(), b["quality"].to(self.device).float())
            with self._autocast():
                o = self.model(x, lung_mask=m, gate_mask=self._gate_for_eval(gm), compute_signature=self._need_sig())
                p = torch.sigmoid(o["logit"].float())
                if tta:
                    xf, mf, gf = torch.flip(x, dims=[3]), torch.flip(m, dims=[3]), torch.flip(gm, dims=[3])
                    o2 = self.model(xf, lung_mask=mf, gate_mask=self._gate_for_eval(gf), compute_signature=self._need_sig())
                    p = 0.5 * (p + torch.sigmoid(o2["logit"].float()))
            probs.append(p.cpu().numpy().reshape(-1))
            ids += list(b["image_id"]); patients += list(b["patient"])
            labels.append(b["label"].numpy().reshape(-1)); sources.append(b["source"].numpy().reshape(-1))
            n += len(p)
            if max_n and n >= max_n:
                break
        return {"image_id": np.array(ids), "label": np.concatenate(labels).astype(int),
                "source": np.concatenate(sources).astype(int), "patient": np.array(patients),
                "prob": np.concatenate(probs)}

    def _need_sig(self):
        return self.mcfg.refirma_target == "canonical"

    @torch.no_grad()
    def seg_dice(self, name="val"):
        loader = self.eval_loader(name)
        self.model.eval()
        d, n = [], 0
        for b in loader:
            x = self.normalize(b["image"].to(self.device).float())
            m = b["mask"].to(self.device).float()
            q = b["quality"].numpy().reshape(-1)
            o = self.model(x, lung_mask=m, gate_mask=m, compute_signature=self._need_sig())
            pr = torch.sigmoid(o["seg_logits"].float())
            pr = torch.nn.functional.interpolate(pr, size=m.shape[-2:], mode="bilinear", align_corners=False) > 0.5
            c = min(pr.shape[1], m.shape[1])
            inter = (pr[:, :c] & (m[:, :c] > 0.5)).flatten(2).sum(-1).float()
            den = pr[:, :c].flatten(2).sum(-1).float() + (m[:, :c] > 0.5).flatten(2).sum(-1).float()
            dice = ((2 * inter + 1) / (den + 1)).cpu().numpy()
            d.append(dice[q > 0])
            n += len(q)
            if n >= int(self.cc["seg_dice_max_n"]):
                break
        d = np.concatenate(d) if d else np.zeros((0, 2))
        return {"dice_pulmones": float(d[:, 0].mean()) if len(d) else float("nan"),
                "dice_corazon": float(d[:, 1].mean()) if len(d) and d.shape[1] > 1 else float("nan")}

    def validate(self):
        sc = self.cfg["selection"]
        pred = self.predict(sc["subset"], tta=bool(sc.get("tta", False)))
        y, p = pred["label"], pred["prob"]
        return exact_metric("auc_roc", y, p), {"val_auc_roc": exact_metric("auc_roc", y, p),
                                               "val_auc_pr": exact_metric("auc_pr", y, p), "val_n": int(len(y))}

    # ================================================================== probes
    def probe_cfg(self):
        from src.eval.probes import DEFAULT_PROBE_CFG
        c = dict(DEFAULT_PROBE_CFG)
        c.update(self.cc.get("probes") or {})
        c["n_boot"] = int(self.cc["probe_n_boot"])
        return c

    def run_probe_set(self, which, tag, name="val", state_label="actual", prob=None, threshold=None):
        from src.eval.probes import collect_probe_data, run_probes, v0_from_model, write_report
        pc = self.probe_cfg()
        ds = self._make_ds(self.frames[name], name)
        idx_of = {n: i for i, n in enumerate(self.source_names)}
        s = np.asarray(ds.sources)
        cap = int(self.cc["probe_rows_cap"])
        rng = np.random.default_rng(int(pc["seed"]))
        rows = []
        rob = idx_of.get(pc["robust_source"], -1)
        for src in np.unique(s):
            ii = np.nonzero(s == src)[0]
            if src != rob and len(ii) > cap:
                ii = np.sort(rng.choice(ii, cap, replace=False))
            rows.append(ii)
        rows = np.sort(np.concatenate(rows))
        targets = sorted({idx_of[n] for n in pc["v2_sources"] if n in idx_of} |
                         {idx_of[st[1]] for st in pc["v2b_strata"] if st[1] in idx_of} |
                         {idx_of[st[1]] for st in pc["v5_strata"] if st[1] in idx_of}) if self.uses_swap else []
        data = collect_probe_data(self.model, ds, self.device, int(self.cfg["data"]["eval_batch_size"]), targets, rows,
                                  int(pc["seed"]), normalize=self.normalize, mask_fn=self.masks_for_quality)
        if prob is not None:
            prob = prob[rows]
        res = run_probes(data, self.source_names, pc, which, prob=prob, threshold=threshold)
        if "V0" in which:
            b = [ds[int(i)] for i in rows[:4]]
            x = self.normalize(torch.stack([t["image"] for t in b]).float().to(self.device))
            m, _ = self.masks_for_quality(torch.stack([t["mask"] for t in b]).float().to(self.device),
                                          torch.stack([t["quality"] for t in b]).float().to(self.device))
            res = {"V0": v0_from_model(self.model, x, m), **res}
        path = write_report(res, self.out / "probes", tag, {"epoca": self.state["epoch"], "estado": state_label,
                                                              "conjunto": name, "fase": self.phase_of(
                                                                  min(self.state["epoch"], self.E_total - 1)),
                                                              "n_boot": int(pc["n_boot"]), "semilla_probes": int(pc["seed"]),
                                                              "filas": int(len(rows))})
        self.state["probe_reports"].append(str(path))
        self.log("probes", tag=tag, **{k: v["veredicto"] for k, v in res.items()})
        return res

    # ================================================================== bucle
    def fit(self) -> int:
        if self.state.get("done"):
            self.log("ya_terminado", out=str(self.out))
            return 0
        stop = GracefulStop().install()
        guard = ThermalSelfCheck(every=int(self.cfg["thermal"]["every"]), log=lambda m: self.log("thermal", msg=m))
        tc = self.cfg["train"]
        self.log("start", seed=self.seed, device=str(self.device), precision=self.precision, out=str(self.out),
                 n_train=len(self.train_ds), steps_per_epoch=self.steps_per_epoch, resumed=self.resumed,
                 epochs={"A1": self.E_A1, "A2": self.E_A2, "rampa": self.E_ramp, "B": self.E_B},
                 tabla=self.table.counts.tolist(), fuentes=list(self.source_names),
                 plan_factible=None if self.plan is None else bool(self.plan.feasible),
                 z=None if self.plan is None else float(self.plan.z), tabla_del_plan=self.cc.get("plan_table", "lotes"),
                 z_dataset=float(self.table.conditional().max(0).sum()), swap=self.uses_swap, within=self.uses_within,
                 test_bloqueado=bool(self.test_locked))
        run_steps = 0
        try:
            while self.state["epoch"] < self.E_total and not self.state.get("stopped_reason"):
                e = self.state["epoch"]
                phase = self.phase_of(e)
                self.state["phase_name"] = phase
                self.state["phase"] = {"A1": 0, "A2": 1, "B": 2}[phase]
                t0 = time.time()
                for batch in self.train_loader_epoch(e, self.state["step_in_epoch"]):
                    logs = self.train_step(batch, e)
                    self.state["step_in_epoch"] += 1; self.state["phase_step"] += 1; self.state["global_step"] += 1
                    run_steps += 1
                    if self.state["global_step"] % int(tc["log_every"]) == 0:
                        self.log("step", phase=phase, epoch=e, step=self.state["step_in_epoch"], **logs)
                    guard.step()
                    if stop.requested:
                        raise TrainStop("SIGTERM (señal %s)" % stop.signum)
                    if tc.get("stop_after_steps") and run_steps >= int(tc["stop_after_steps"]):
                        raise TrainStop("stop_after_steps=%s" % tc["stop_after_steps"], EXIT_DEBUG_STOP)
                    if tc.get("checkpoint_every_steps") and self.state["global_step"] % int(tc["checkpoint_every_steps"]) == 0:
                        self.save_checkpoint("pasos")
                self._end_epoch_csd(e, phase, time.time() - t0)
            self._finalize()
            stop.restore()
            return 0
        except ThermalLimitReached as ex:
            self.log("stop", reason="temperatura", detail=str(ex), max_temp=guard.max_seen)
            self.save_checkpoint("temperatura")
            stop.restore()
            return EXIT_THERMAL
        except TrainStop as ex:
            self.log("stop", reason=ex.reason, code=ex.code)
            self.save_checkpoint("parada")
            stop.restore()
            return ex.code

    def _end_epoch_csd(self, e, phase, secs):
        rec = {"phase": phase, "epoch": e, "seconds": round(secs, 1), "swap": bool(phase == "B" and self.state["swap_enabled"]
                                                                                   and self.uses_swap),
               "gate": "predicha" if (phase == "B" and self.state["gate_pred"]) else "profesor"}
        rec.update(self._epoch_table_stats())
        rec["loss"] = rec.get("losses", {}).get("L_total")
        dice = self.seg_dice("val")
        rec.update(dice)
        if dice["dice_pulmones"] == dice["dice_pulmones"]:
            self.state["gate_pred"] = bool(dice["dice_pulmones"] >= float(self.cc["gate_dice_threshold"]))
        stop_now = False
        if phase == "B":
            metric, extra = self.validate()
            rec.update(extra); rec["selection"] = metric
            es = self.state["es"]
            es["wait"] += 1
            if metric == metric and metric > es["best"]:
                es["best"], es["best_epoch"], es["wait"] = metric, e, 0
                atomic_torch_save(self.model.state_dict(), self.out / "states" / "best.pt")
                rec["improved"] = True
            elif es["wait"] >= int(self.cfg["selection"]["patience"]):
                es["stopped"] = True
                stop_now = True
            rec["wait"] = es["wait"]
        # V1 al final de A2: puerta del intercambio
        if e == self.E_A1 + self.E_A2 - 1:
            chk = self.signature_validity(rec)
            rec["control_firma"] = chk
            self.state["control_firma"] = chk
            self.log("control_firma", **chk)
            which = ("V1", "V1b", "V2") if self.cc.get("parar_tras_A2") else ("V1", "V1b")
            r = self.run_probe_set(which, "V1_fin_A2_ep%03d" % e)
            v = r["V1"]["veredicto"]
            sin_fuente = bool(chk["activo"]) and chk.get("valida") is False
            if sin_fuente:
                # la sonda se ejecuta y se guarda, pero su veredicto no vale: la firma no lleva fuente
                self.state["flags"]["firma_sin_fuente"] = True
                self.state["flags"]["v1_veredicto_sonda"] = v
                self.state["flags"]["v1_motivo"] = "firma sin información de fuente"
                v = "no_concluyente"
            if "V2" in r:
                self.state["flags"]["v2_fin_A2"] = r["V2"]["veredicto"]
                rec["v2_fin_A2"] = r["V2"]["veredicto"]
            if self.cc.get("v1_gate_override"):
                v = str(self.cc["v1_gate_override"])
                self.state["flags"]["v1_veredicto_sustituido"] = True
            self.state["v1"] = v
            enabled = v in ("aprueba", "no_aplicable") or (v == "no_concluyente" and self.cc["v1_inconclusive"] == "activar")
            if sin_fuente and not self.cc.get("v1_gate_override"):
                enabled = False                       # sin firma de fuente no hay re-firma que activar
            if v == "no_concluyente":
                self.state["flags"]["v1_no_concluyente"] = True
            self.state["swap_enabled"] = bool(enabled)
            rec["v1"] = v
            seguir = (not enabled and self.cc["v1_fail_action"] == "continue_swap" and v == "falla" and not sin_fuente)
            if seguir:
                # Enmienda autorizada por el autor: V1 se informa como no aprobada y el intercambio sigue activo.
                # Nunca con una firma sin información de fuente: esa corrida no es válida y se detiene.
                self.state["swap_enabled"] = True
                self.state["flags"]["v1_falla_intercambio_activo"] = True
                self.state["flags"]["v1_enmienda"] = str(self.cc["v1_fail_enmienda"])
                self.log("v1_parada", veredicto=v, accion="continue_swap", enmienda=str(self.cc["v1_fail_enmienda"]))
            elif not enabled:
                self.state["flags"]["v1_fallo_intercambio_desactivado"] = True
                if self.cc["v1_fail_action"] in ("stop", "continue_swap"):
                    self.state["stopped_reason"] = (
                        "V1 no concluyente al final de A2: firma sin información de fuente (control de validez previo)"
                        if self.state["flags"].get("firma_sin_fuente") and v == "no_concluyente"
                        else "V1 no superada al final de A2: no se activa la fase B")
                    self.log("v1_parada", veredicto=v, accion="stop")
                else:
                    self.log("v1_parada", veredicto=v, accion="continue_no_swap")
        if self.cc.get("parar_tras_A2") and e == self.E_A1 + self.E_A2 - 1 and not self.state.get("stopped_reason"):
            self.state["stopped_reason"] = "diagnóstico: parada tras A2 (csdnet.parar_tras_A2)"
        if self.E_A1 + self.E_A2 == 0 and self.state["swap_enabled"] is None:
            self.state["swap_enabled"] = True
        if phase == "B" and rec["swap"]:
            eb = e - self.E_A1 - self.E_A2
            if eb == max(self.E_ramp, 1) - 1:
                r = self.run_probe_set(("V2", "V2b", "V3", "V5"), "control_rampa_ep%03d" % e)
                okb = all(r[k]["veredicto"] in ("aprueba", "no_aplicable") for k in ("V2b", "V5"))
                self.state["flags"]["apto_aporte_b"] = bool(okb)
                for k in ("V2", "V3"):
                    self.state["flags"][k.lower() + "_control"] = r[k]["veredicto"]
                rec["control_rampa"] = {k: v["veredicto"] for k, v in r.items()}
            elif int(self.cc["probe_every"]) > 0 and (eb + 1) % int(self.cc["probe_every"]) == 0:
                self.run_probe_set(("V2b", "V5"), "registro_ep%03d" % e)
        self.state["history"].append(rec)
        self.log("epoch", **rec)
        self.state["epoch"] += 1
        self.state["step_in_epoch"] = 0
        self.state["epoch_acc"] = None
        if stop_now:
            self.state["stopped_reason"] = "parada temprana"
            self.log("early_stopping", epoch=e, best_epoch=self.state["es"]["best_epoch"], best=self.state["es"]["best"])
        self.save_checkpoint("epoca")

    def _finalize(self):
        sd_dir = self.out / "states"
        atomic_torch_save(self.model.state_dict(), sd_dir / "last.pt")
        if not (sd_dir / "best.pt").exists():
            atomic_torch_save(self.model.state_dict(), sd_dir / "best.pt")
        atomic_torch_save(self.ema.ema_state_dict(self.model), sd_dir / "ema.pt")
        last = {k: v.clone() for k, v in self.model.state_dict().items()}
        summary = {"seed": self.seed, "experiment": self.cfg["experiment"], "history": self.state["history"],
                   "early_stopping": self.state["es"], "v1": self.state["v1"], "flags": self.state["flags"],
                   "stopped_reason": self.state["stopped_reason"], "test_bloqueado": bool(self.test_locked),
                   "control_firma": self.state.get("control_firma"), "ind": dict(self.cc.get("ind") or {}),
                   "results": {}, "probes": {}}
        from src.eval.predict import save_predictions, summarize
        from src.eval.probes import youden_threshold
        rob_pred = {}
        for st in self.cfg["eval"]["states"]:
            p = sd_dir / ("%s.pt" % st)
            if not p.exists():
                continue
            self.model.load_state_dict(torch.load(p, map_location=self.device))
            for name in self.cfg["data"]["eval_sets"]:
                pred = self.predict(name, tta=bool(self.cfg["eval"]["tta"]))
                save_predictions(pred, self.out / "predictions" / ("%s_%s.csv" % (st, name)))
                summary["results"]["%s/%s" % (st, name)] = summarize(pred)
                self.log("eval", state=st, set=name, **summary["results"]["%s/%s" % (st, name)])
                if name == "val_robust":
                    rob_pred[st] = pred
            if st in self.cc["final_probe_states"]:
                thr = None
                prob = None
                if st in rob_pred:
                    thr = youden_threshold(rob_pred[st]["label"], rob_pred[st]["prob"])
                    pv = self.predict("val", tta=bool(self.cfg["eval"]["tta"]))
                    prob = pv["prob"]
                which = ("V0", "V1", "V1b", "V2", "V2b", "V3", "V4", "V5") + (("V6",) if prob is not None else ())
                r = self.run_probe_set(which, "final_%s" % st, state_label=st, prob=prob, threshold=thr)
                summary["probes"][st] = {k: v["veredicto"] for k, v in r.items()}
        self.model.load_state_dict(last)
        self.state["done"] = True
        with open(self.out / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=float)
        self.save_checkpoint("fin")
        self.log("end", out=str(self.out))
