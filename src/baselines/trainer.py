"""Entrenador de las líneas base, derivado del entrenador común (src/train/trainer.py).

`BaselineTrainer` hereda datos, muestreador, optimizador, programa de la tasa, EMA, parada temprana en
val_robust, checkpoint y reanudación exacta, SIGTERM y autochequeo térmico (código 75) y guardado de
probabilidades. Solo sustituye:

- `_build_model`: envuelve el modelo común en `BaselineNet` (net = el modelo, aux = módulos del método).
- `train_step`: misma secuencia que el común (aumento sembrado por paso, tasa, microlotes con acumulación, L2,
  recorte, paso, EMA) con la pérdida del método. El generador del método se siembra con (semilla, fase, paso),
  así que la reanudación sigue siendo exacta.
- `_build_data`: bloqueo del test, celdas vacías en los datos sintéticos de prueba, conjuntos propios del
  método (mitades de val_robust en DFR) y máscaras pulmonares.
- `_finalize`: gancho del método antes de la evaluación final (DFR).

Bloqueo del test: con `protocol.test_bloqueado: true` (valor por defecto) cualquier conjunto cuyo nombre empiece
por "test" en `data.eval_sets` o en `selection.subset` detiene la construcción con `TestLockedError`, y el
DataFrame del test se descarta. Para desbloquear (solo experimentos finales): `protocol.test_bloqueado: false` y
`protocol.congelado: <ruta a JSON>` con {"method": ..., "params": {...}, "lr_B": ...} que debe coincidir con la
configuración de la corrida (docs/06_lineas_base.md, Section 4) y, con datos reales, la congelación del protocolo
verificada por `src.eval.desbloqueo.exigir_congelacion` (hallazgo M1). La selección nunca puede hacerse en test.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.paths import PROJECT_ROOT
from src.train.bench import TestBloqueadoError
from src.train.config import resolve
from src.train.optim import set_lr, warmup_cosine
from src.train.trainer import Trainer, atomic_torch_save


# Mitades fijas de val_robust por paciente (src/baselines/methods/dfr.py: patient_half, split_val_robust). Se crean
# para todos los métodos: val_robust_sel es el conjunto común de selección, parada temprana y comparación entre
# métodos; val_robust_fit solo lo usa DFR para reajustar fc2. val_robust completa no es comparable para métodos que
# ajustan con val_robust_fit (Method.noncomparable_sets).
VAL_ROBUST_HALVES = ("val_robust_fit", "val_robust_sel")


class TestLockedError(TestBloqueadoError):   # TestBloqueadoError hereda de RuntimeError (sin cambio para COVIDx)
    __test__ = False      # no es una prueba de pytest


def _is_test(name: str) -> bool:
    return str(name) == "test" or str(name).startswith("test_")


def check_test_lock(cfg: dict):
    """Lanza TestLockedError si la configuración evaluaría el test sin desbloqueo válido."""
    pc = cfg.get("protocol") or {}
    locked = pc.get("test_bloqueado", True)
    sel = cfg.get("selection", {}).get("subset", "val")
    if _is_test(sel):
        raise TestLockedError("la selección de modelo no puede hacerse en '%s'" % sel)
    wants = [n for n in cfg.get("data", {}).get("eval_sets", []) if _is_test(n)]
    if not wants:
        return locked
    if locked:
        raise TestLockedError("test bloqueado (protocol.test_bloqueado: true) y eval_sets pide %s. Se desbloquea solo "
                              "con hiperparámetros congelados: ver docs/06_lineas_base.md, Section 4." % wants)
    path = pc.get("congelado")
    if not path:
        raise TestLockedError("test desbloqueado sin protocol.congelado (JSON de hiperparámetros congelados)")
    p = Path(path) if os.path.isabs(str(path)) else PROJECT_ROOT / path
    if not p.exists():
        raise TestLockedError("no existe el fichero de hiperparámetros congelados %s" % p)
    fz = json.load(open(p, encoding="utf-8"))
    bc = cfg.get("baseline") or {}
    if fz.get("method") != bc.get("method", "erm"):
        raise TestLockedError("el método congelado (%s) no coincide con el de la corrida (%s)"
                              % (fz.get("method"), bc.get("method")))
    from src.baselines.registry import get_method
    full = dict(get_method(bc.get("method", "erm")).defaults)
    full.update(bc.get("params") or {})
    fzp = dict(get_method(fz["method"]).defaults)
    fzp.update(fz.get("params") or {})
    if json.dumps(full, sort_keys=True, default=str) != json.dumps(fzp, sort_keys=True, default=str):
        raise TestLockedError("los hiperparámetros de la corrida no son los congelados en %s" % p)
    if "lr_B" in fz and abs(float(fz["lr_B"]) - float(cfg["train"]["phases"][-1]["lr"])) > 1e-15:
        raise TestLockedError("lr de la fase B distinta de la congelada en %s" % p)
    if not (cfg.get("data") or {}).get("synthetic"):      # M1: además, la congelación del protocolo con sus huellas
        from src.eval.desbloqueo import TestBloqueadoError, exigir_congelacion
        try:
            exigir_congelacion("líneas base: %s, conjuntos %s" % (bc.get("method", "erm"), wants))
        except TestBloqueadoError as e:
            raise TestLockedError(str(e)) from e
    return False


class BaselineNet(nn.Module):
    """Modelo común más módulos auxiliares. forward(x) da el logit de la clase (lo usa src/eval/predict.py)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.aux = nn.ModuleDict()
        self._fwd = None            # forward propio del método en inferencia (DFA, sondas)
        self._extra_backbone = []   # módulos auxiliares que cuentan como tronco (tasa y congelación de fase A)
        self._aux_lr = []           # (módulo auxiliar, multiplicador de tasa propio): p. ej. el discriminador de DANN
        self._ctx = None            # contexto del paso de entrenamiento (MixStyle)

    def forward(self, x):
        if self._fwd is not None:
            return self._fwd(self, x)
        return self.net(x)

    def pooled(self, x):
        return self.net.pooled(x)

    def backbone_parameters(self):
        n = self.net
        ps = list(n.backbone_parameters()) if hasattr(n, "backbone_parameters") else list(n.backbone.parameters())
        for m in self._extra_backbone:
            ps += list(m.parameters())
        return ps

    def backbone_norm_params(self):
        return self.net.backbone_norm_params() if hasattr(self.net, "backbone_norm_params") else []

    def l2_penalty(self):
        return self.net.l2_penalty() if hasattr(self.net, "l2_penalty") else None


class MaskedDataset(torch.utils.data.Dataset):
    """Multiplica la imagen por la máscara pulmonar binaria (anula los píxeles fuera de la máscara)."""

    def __init__(self, base, masks: "MaskStore"):
        self.base, self.masks = base, masks
        self.labels = base.labels
        self.df = getattr(base, "df", None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        d = self.base[i]
        m = self.masks.get(d["image_id"], d["image"].shape[-2:])
        d["image"] = d["image"] * m.unsqueeze(0)
        d["mask"] = m
        return d


class MaskStore:
    """Máscaras por imagen en <mask_dir>/<image_id sin extensión>.png (8 bits, > 127 = pulmón) o .npy.
    Si existe <mask_dir>/index.csv (image_id, path) se usa en su lugar."""

    def __init__(self, mask_dir, resize: bool = False):
        d = Path(mask_dir)
        self.dir = d if d.is_absolute() else PROJECT_ROOT / d
        self.resize = bool(resize)
        self.map = {}
        if not self.dir.exists():
            raise FileNotFoundError(self._msg("no existe el directorio de máscaras %s" % self.dir))
        idx = self.dir / "index.csv"
        if idx.exists():
            import pandas as pd
            t = pd.read_csv(idx)
            self.map = {str(a): (self.dir / str(b)) for a, b in zip(t["image_id"], t["path"])}
        else:
            for f in os.listdir(self.dir):
                stem, ext = os.path.splitext(f)
                if ext.lower() in (".png", ".npy"):
                    self.map[stem] = self.dir / f

    @staticmethod
    def _msg(s):
        return ("ERM con enmascarado pulmonar: %s. Las máscaras las produce el profesor de segmentación "
                "en data/masks/; genérelas antes o corrija baseline.params.mask_dir." % s)

    def key(self, image_id):
        return str(image_id) if str(image_id) in self.map else os.path.splitext(str(image_id))[0]

    def check(self, ids, where=""):
        miss = [i for i in ids if self.key(i) not in self.map]
        if miss:
            raise FileNotFoundError(self._msg("faltan %d de %d máscaras%s en %s (p. ej. %s)"
                                              % (len(miss), len(ids), where, self.dir, miss[0])))

    def get(self, image_id, hw):
        k = self.key(image_id)
        if k not in self.map:
            raise FileNotFoundError(self._msg("falta la máscara de %s en %s" % (image_id, self.dir)))
        p = self.map[k]
        if str(p).endswith(".npy"):
            a = np.load(p).astype(np.float32)
            a = (a > 0.5).astype(np.float32)
        else:
            from PIL import Image
            a = (np.asarray(Image.open(p).convert("L")) > 127).astype(np.float32)
        t = torch.from_numpy(a)
        if tuple(t.shape) != tuple(hw):
            if not self.resize:
                raise ValueError(self._msg("la máscara de %s mide %s y la imagen %s (use mask_resize: true para "
                                           "redimensionar por vecino más próximo)" % (image_id, tuple(t.shape), tuple(hw))))
            t = torch.nn.functional.interpolate(t[None, None], size=tuple(hw), mode="nearest")[0, 0]
        return t


def _assign_cells(df, cells: dict, seed: int, split: str):
    """Datos sintéticos de prueba: reasigna la fuente de cada fila entre las permitidas para su clase."""
    import zlib
    rng = np.random.default_rng([seed, zlib.crc32(split.encode())])
    df = df.copy()
    src = []
    for y in df["label"].astype(int):
        allowed = cells[str(y)] if str(y) in cells else cells[y]
        src.append(int(allowed[int(rng.integers(len(allowed)))]))
    df["source_idx"] = src
    df["source"] = ["s%d" % v for v in src]
    return df


class QuotaInfiniteBatchSampler:
    """Lotes del muestreador de cuotas de CSD-Net (`TableGuidedBatchSampler`) como secuencia sin fin y reanudable:
    el lote b es el lote b % steps_per_epoch de la época b // steps_per_epoch de ese muestreador, de modo que la
    secuencia depende solo de la semilla y del índice del lote (igual que en CSD-Net, que lo reinicia cada época)."""

    def __init__(self, labels, sources, batch_size, steps_per_epoch, min_per_source, min_per_class_mixed,
                 seed: int = 0, start_batch: int = 0):
        from src.model.swap import TableGuidedBatchSampler
        self.spe = int(steps_per_epoch)
        self.start_batch = int(start_batch)
        self.smp = TableGuidedBatchSampler(np.asarray(labels).astype(int), np.asarray(sources).astype(int),
                                           int(batch_size), self.spe, int(min_per_source), int(min_per_class_mixed),
                                           seed=int(seed))
        self._epoch, self._batches = None, None

    def batch(self, b: int):
        e, i = divmod(int(b), self.spe)
        if e != self._epoch:
            self.smp.set_epoch(e)
            self._batches = [list(map(int, x)) for x in self.smp]
            self._epoch = e
        return self._batches[i]

    def __iter__(self):
        b = self.start_batch
        while True:
            yield self.batch(b)
            b += 1


class BaselineTrainer(Trainer):
    def __init__(self, cfg: dict, seed: int, resume: bool = True, out_dir=None, echo=True):
        from src.baselines.registry import build_method
        rc = resolve(cfg)
        self.test_locked = check_test_lock(rc)
        self.method = build_method(rc)
        super().__init__(cfg, seed, resume=resume, out_dir=out_dir, echo=echo)
        nc = [n for n in self.cfg["data"]["eval_sets"] if n in set(self.method.noncomparable_sets)]
        self.log("baseline", method=self.method.name, params=self.method.p, status=self.method.status,
                 test_bloqueado=self.test_locked, seleccion=self.cfg["selection"]["subset"], no_comparables=nc)

    # ------------------------------------------------------------------ datos
    def _build_data(self):
        if self.cfg["data"].get("bench"):
            return self._build_bench_data_bl()
        dc, sc = self.cfg["data"], self.cfg["selection"]
        custom = set(VAL_ROBUST_HALVES) | set(self.method.custom_sets)
        orig_eval, orig_sel = list(dc["eval_sets"]), sc["subset"]
        if custom:
            dc["eval_sets"] = [n for n in orig_eval if n not in custom]
            if orig_sel in custom:
                sc["subset"] = "val_robust"
            if "val_robust" not in dc["eval_sets"] and sc["subset"] != "val_robust":
                dc["eval_sets"] = dc["eval_sets"] + ["val_robust"]
        try:
            super()._build_data()
        finally:
            dc["eval_sets"], sc["subset"] = orig_eval, orig_sel
        syn = dc.get("synthetic") or {}
        rebuild = False
        if syn.get("cells"):
            for k in list(self.frames):
                if k == "val_robust":
                    continue
                self.frames[k] = _assign_cells(self.frames[k], syn["cells"], self.seed, k)
            self.frames["val_robust"] = self.frames["val"][self.frames["val"]["source_idx"] == 0].reset_index(drop=True)
            rebuild = True
        if syn and dc.get("train_sources"):
            self.frames["train"] = self.frames["train"][self.frames["train"]["source"].isin(
                list(dc["train_sources"]))].reset_index(drop=True)
            rebuild = True
        if self.test_locked:
            for k in [k for k in self.frames if _is_test(k)]:
                del self.frames[k]
        from src.baselines.methods.dfr import split_val_robust
        fit, sel = split_val_robust(self.frames["val_robust"])
        self.frames["val_robust_fit"], self.frames["val_robust_sel"] = fit, sel
        self.method.prepare_data(self)
        if self.method.needs_mask:
            p = self.method.p
            store = MaskStore(p["mask_dir"], p.get("mask_resize", False))
            for k, f in self.frames.items():
                store.check(f["image_id"].astype(str).tolist(), " del conjunto %s" % k)
            base_make = self._make_ds
            self._make_ds = lambda df, split: MaskedDataset(base_make(df, split), store)
            self.mask_store = store
            rebuild = True
        if rebuild:
            import math
            self.train_ds = self._make_ds(self.frames["train"], "train")
            tc = self.cfg["train"]
            self.steps_per_epoch = int(tc["steps_per_epoch"] or math.ceil(len(self.train_ds) / tc["batch_size"]))
        from src.baselines.common import cell_table, empty_cells
        self.cells = cell_table(self.frames["train"])
        self.empty = empty_cells(self.cells)

    def _build_bench_data_bl(self):
        """Benchmarks del plan B (data.bench). El entrenador común carga el manifiesto, descarta el test bloqueado y
        crea las mitades estratificadas (src/train/bench.py). Aquí: aplicabilidad del método (una corrida no
        aplicable termina con estado explícito y sin predicciones), datos propios del método y máscaras.
        docs/13_entrenamiento_plan_b.md, Sections 2 y 3."""
        import math
        from src.baselines.common import cell_table, empty_cells
        super()._build_data()
        self.no_aplicable_estados = {}
        self.cells = cell_table(self.frames["train"])
        self.empty = empty_cells(self.cells)
        estado, motivo = self.method.aplicabilidad(self)
        self.aplicabilidad = {"estado": estado, "motivo": motivo}
        self.no_aplicable = motivo if estado == "no_aplicable" else None
        if self.no_aplicable:
            return
        self.method.prepare_data(self)
        if self.method.needs_mask:
            p = self.method.p
            store = MaskStore(p["mask_dir"], p.get("mask_resize", False))
            for k, f in self.frames.items():
                store.check(f["image_id"].astype(str).tolist(), " del conjunto %s" % k)
            base_make = self._make_ds
            self._make_ds = lambda df, split: MaskedDataset(base_make(df, split), store)
            self.mask_store = store
            self.train_ds = self._make_ds(self.frames["train"], "train")
            tc = self.cfg["train"]
            self.steps_per_epoch = int(tc["steps_per_epoch"] or math.ceil(len(self.train_ds) / tc["batch_size"]))
        self.cells = cell_table(self.frames["train"])
        self.empty = empty_cells(self.cells)

    # ------------------------------------------------------------------ modelo
    def _build_model(self):
        if getattr(self, "no_aplicable", None):
            self.model = nn.Module()                 # la corrida no entrena (estado no_aplicable)
            return
        net = self.method.build_net(self)
        if net is None:
            super()._build_model()
            net = self.model
        wrapped = BaselineNet(net)
        self.method.build(self, wrapped)
        self.model = wrapped.to(self.device)

    # ------------------------------------------------------------------ paso
    def _param_groups(self):
        """Grupos del entrenador común; los módulos auxiliares declarados en `model._aux_lr` salen del grupo de la
        cabeza y reciben su propio multiplicador de tasa (enmienda del 13-11-2026: con el multiplicador
        de la cabeza, 20 veces la tasa del tronco, el discriminador de DANN y CDANN se volvía demasiado confiado y
        la pérdida adversaria divergía al descongelar el tronco)."""
        groups = super()._param_groups()
        own = {}
        for mod, mult in getattr(self.model, "_aux_lr", []) or []:
            for p in mod.parameters():
                own[id(p)] = float(mult)
        if not own:
            return groups
        out, extra = [], {}
        for g in groups:
            keep = []
            for p in g["params"]:
                if id(p) in own:
                    extra.setdefault((g["name"].endswith("_nd"), own[id(p)]), []).append(p)
                else:
                    keep.append(p)
            out.append(dict(g, params=keep))
        for (nd, mult), ps in sorted(extra.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            d = {"name": "aux_propio" + ("_nd" if nd else ""), "params": ps, "lr_mult": mult}
            if nd:
                d["weight_decay"] = 0.0
            out.append(d)
        return [g for g in out if g["params"]]

    def train_loader(self, start_batch: int):
        q = getattr(self, "quota_sampler", None)
        if not q:
            return super().train_loader(start_batch)
        from torch.utils.data import DataLoader
        from src.train.trainer import _collate
        tc, dc = self.cfg["train"], self.cfg["data"]
        fr = self.frames["train"]                     # mismo orden que self.train_ds
        if len(fr) != len(self.train_ds):
            raise RuntimeError("el marco de train y el conjunto de entrenamiento no tienen el mismo tamaño")
        smp = QuotaInfiniteBatchSampler(fr["label"].to_numpy(), fr["source_idx"].to_numpy(), tc["batch_size"],
                                        self.steps_per_epoch, q["min_per_source"], q["min_per_class_mixed"],
                                        seed=self.seed, start_batch=start_batch)
        nw = 0 if dc.get("synthetic") else int(dc["num_workers"])
        g = torch.Generator(); g.manual_seed(self.seed)
        return DataLoader(self.train_ds, batch_sampler=smp, num_workers=nw, collate_fn=_collate,
                          pin_memory=self.device.type == "cuda", generator=g,
                          prefetch_factor=4 if nw else None, persistent_workers=False)

    def method_generator(self):
        from src.data.datasets import make_generator
        return make_generator(self.seed * 17 + 1000 + self.state["phase"], self.state["phase_step"], self.device)

    def train_step(self, batch):
        from src.data.datasets import make_generator
        tc = self.cfg["train"]
        self.model.train()
        x = batch["image"].to(self.device, non_blocking=True).float()
        y = batch["label"].to(self.device, non_blocking=True).float()
        s = batch["source"].to(self.device, non_blocking=True).long()
        if self.augment is not None:
            gen = make_generator(self.seed * 13 + self.state["phase"], self.state["phase_step"], self.device)
            x, y = self.augment(x, y, gen)
        lr = warmup_cosine(self.state["phase_step"], float(self._phase()["lr"]), self.phase_total, self.phase_warmup,
                           float(self._phase().get("min_lr", 1e-7)))
        set_lr(self.opt, lr)
        self.opt.zero_grad(set_to_none=True)
        B = x.shape[0]
        mb = int(tc.get("micro_batch") or B)
        mgen = self.method_generator()
        self._batch_ids = batch.get("image_id")
        tot = 0.0
        logs = {}
        for i in range(0, B, mb):
            with self._autocast():
                loss, lg = self.method.loss(self, self.model, x[i:i + mb], y[i:i + mb], s[i:i + mb], mgen, B)
            loss.backward()
            tot += float(loss.detach())
            for k, v in (lg or {}).items():
                logs[k] = logs.get(k, 0.0) + float(v)
        reg = self.model.l2_penalty()
        if reg is not None and reg.requires_grad:
            reg.backward()
            tot += float(reg.detach())
        if tc.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], float(tc["grad_clip"]))
        self.opt.step()
        if self.ema is not None:
            self.ema.update(self.model)
        self.method.after_step(self)
        self.last_logs = logs
        if not np.isfinite(tot):
            from src.train.trainer import TrainStop
            raise TrainStop("pérdida no finita (%s) en el paso %d" % (tot, self.state["global_step"]), 1)
        return tot, lr

    # ------------------------------------------------------------------ no aplicable
    def fit(self) -> int:
        if getattr(self, "no_aplicable", None):
            return self._finish_not_applicable()
        return super().fit()

    def _finish_not_applicable(self) -> int:
        """Corrida no aplicable (docs/13_entrenamiento_plan_b.md, Section 3): sin entrenamiento, sin estados y sin
        predicciones; summary.json con estado no_aplicable y el motivo. Código 0, para que una cadena siga."""
        from src.train.bench import write_json
        summ = {"seed": self.seed, "experiment": self.cfg["experiment"], "estado": "no_aplicable",
                "motivo": self.no_aplicable, "metodo": self.method.name, "bench": self.cfg.get("bench"),
                "aplicabilidad": self.aplicabilidad, "results": {}, "no_aplicable": {"*": self.no_aplicable},
                "tronco": {"tronco": self.cfg["model"].get("backbone"), "preentrenado": None, "etiqueta_pesos": None,
                           "nota": "corrida no aplicable: no se construye el modelo"}}
        if self.cfg.get("storage"):
            summ["almacenamiento"] = {"estados_conservados": [], "borrados": [], "politica": self.cfg["storage"]}
        write_json(self.out / "summary.json", summ)
        self.log("no_aplicable", metodo=self.method.name, motivo=self.no_aplicable)
        return 0

    # ------------------------------------------------------------------ final
    def _finalize(self):
        if self.method.before_finalize is not None:
            sd = self.out / "states"
            atomic_torch_save(self.model.state_dict(), sd / "last.pt")
            if not (sd / "best.pt").exists():          # igual que el común: sin mejora, best = last
                atomic_torch_save(self.model.state_dict(), sd / "best.pt")
            if self.cfg["selection"].get("subset_dc") and not (sd / "bestdc.pt").exists():
                atomic_torch_save(self.model.state_dict(), sd / "bestdc.pt")
            self.method.before_finalize(self)
        super()._finalize()
        if self.cfg["data"].get("bench"):
            return self._finalize_bench_summary()
        nc = [n for n in self.cfg["data"]["eval_sets"] if n in set(self.method.noncomparable_sets)]
        sp = self.out / "summary.json"
        summ = json.load(open(sp, encoding="utf-8"))
        summ["seleccion"] = self.cfg["selection"]["subset"]
        summ["comparacion_entre_metodos"] = "val_robust_sel"
        summ["no_comparables"] = {k: "ajustado con val_robust_fit (%s)" % self.method.name
                                  for k in summ["results"] if k.split("/", 1)[1] in nc}
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(summ, f, indent=2, default=float)
        if nc:
            self.log("aviso", msg="conjuntos no comparables entre métodos para %s: %s" % (self.method.name, nc))

    def _finalize_bench_summary(self):
        from src.train.bench import is_test, write_json
        sp = self.out / "summary.json"
        summ = json.load(open(sp, encoding="utf-8"))
        b = self.cfg.get("bench") or {}
        summ["estado"] = "terminado"
        summ["metodo"] = self.method.name
        summ["seleccion"] = self.cfg["selection"]["subset"]
        summ["aplicabilidad"] = self.aplicabilidad
        summ["no_aplicable"] = dict(self.no_aplicable_estados)
        summ["comparacion_entre_metodos"] = [s for s in b.get("subconjuntos", []) if s != summ["seleccion"]]
        nc = {}
        for k in summ["results"]:
            st, name = k.split("/", 1)
            why = self.method.noncomparable_bench(self, st, name)
            if why:
                nc[k] = why
        summ["no_comparables"] = nc
        if any(is_test(k.split("/", 1)[1]) for k in summ["results"]) and b.get("test_bloqueado", True):
            raise RuntimeError("predicciones de test con el test bloqueado")      # no debería ocurrir nunca
        write_json(sp, summ)
        if nc:
            self.log("aviso", msg="estados y conjuntos no comparables para %s: %s" % (self.method.name, sorted(nc)))
