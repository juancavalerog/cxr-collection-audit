"""Entrenador genérico de clasificación binaria con fases (docs/05_entrenamiento.md).

Un proceso por GPU (el ejecutor asigna una GPU a cada trabajo); el lote global se forma en el muestreador y se
divide en microlotes con acumulación de gradiente, así que un lote de 768 en una GPU equivale al de 2 x 384 del
artículo 1 (el modelo no tiene BatchNorm; el mixup se hace sobre el lote global como en tf.data).

Fases (configurables; por defecto las del artículo 1): A con el tronco congelado, B con todo entrenable salvo las
LayerNorm que Keras congelaba. Cada fase crea su optimizador (TFAdamW) y su programa warmup + coseno, igual que
cada model.compile del artículo 1; el muestreador vuelve a empezar en cada fase, como un nuevo model.fit sobre el
mismo tf.data. La EMA ("tfkeras_exacta" o "estandar") se inicializa al empezar la fase con ema: true.

Parada temprana con la semántica de tf_keras EarlyStopping (restore_best_weights=False): espera += 1 al final de
cada época; si mejora estrictamente, se guarda y espera = 0; si espera >= paciencia y la época no es la primera,
se para. La métrica se calcula con los pesos crudos. Se guardan siempre tres estados: last (crudos al terminar),
best (crudos de la mejor época) y ema.

Seguridad: GracefulStop (SIGTERM) y ThermalSelfCheck cada `thermal.every` pasos; al parar por cualquiera de los
dos se guarda checkpoint atómico y fit() devuelve 75 (EXIT_THERMAL). Al arrancar se reanuda desde
<salida>/checkpoint.pt si existe. El lote b de cada fase y el aumento del paso b son funciones deterministas de
(semilla, fase, b), y el estado de los generadores se guarda en el checkpoint: la reanudación es exacta.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.paths import PROJECT_ROOT, resolve_data
from src.train.config import config_hash, resolve
from src.train.ema import WeightEMA
from src.train.losses import build_loss
from src.train.metrics import exact_metric, selection_metric
from src.train.optim import build_optimizer, set_lr, warmup_cosine

sys.path.insert(0, str(PROJECT_ROOT / "runner"))
from thermal import EXIT_THERMAL, GracefulStop, ThermalLimitReached, ThermalSelfCheck  # noqa: E402

EXIT_DEBUG_STOP = 3


def bloqueo_test_covidx(cfg: dict, names) -> None:
    """Hallazgo M1 del revisor (review/informe_puerta7.md): en COVIDx el entrenador común no evalúa ningún test ni
    COVIDGR sin `protocol.test_bloqueado: false` y la congelación del protocolo (src/eval/desbloqueo.py). La selección
    nunca puede hacerse en un test. Los datos sintéticos de las pruebas no pasan por aquí."""
    from src.eval.desbloqueo import TestBloqueadoError, es_bloqueado, exigir_congelacion
    sel = (cfg.get("selection") or {}).get("subset", "val")
    if es_bloqueado(sel):
        raise TestBloqueadoError("la selección de modelo no puede hacerse en '%s'" % sel)
    pedidos = sorted(n for n in names if es_bloqueado(n))
    if not pedidos:
        return
    if (cfg.get("protocol") or {}).get("test_bloqueado", True):
        raise TestBloqueadoError("test bloqueado (protocol.test_bloqueado: true) y la configuración pide %s" % pedidos)
    exigir_congelacion("entrenador común (COVIDx), experimento %s: %s" % (cfg.get("experiment"), pedidos))


class TrainStop(Exception):
    def __init__(self, reason, code=EXIT_THERMAL):
        super().__init__(reason)
        self.reason, self.code = reason, code


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".%d.tmp" % os.getpid())
    torch.save(obj, tmp)
    os.replace(tmp, path)


class JsonlLog:
    def __init__(self, path: Path, echo=True):
        self.path, self.echo = Path(path), echo
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: str, **kw):
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event}
        rec.update(kw)
        line = json.dumps(rec, default=float)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.echo:
            print(line, flush=True)


def _collate(batch):
    out = {}
    for k in batch[0]:
        v = [b[k] for b in batch]
        out[k] = torch.stack(v) if torch.is_tensor(v[0]) else v
    return out


class Trainer:
    def __init__(self, cfg: dict, seed: int, resume: bool = True, out_dir=None, echo=True):
        self.cfg = resolve(cfg)
        self.cfg["seed"] = int(seed)
        self.seed = int(seed)
        od = out_dir or self.cfg["output_dir"].format(experiment=self.cfg["experiment"], seed=self.seed)
        self.out = Path(od) if os.path.isabs(str(od)) else PROJECT_ROOT / od
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = JsonlLog(self.out / "train_log.jsonl", echo=echo)
        dev = self.cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        p = self.cfg["precision"]
        self.precision = ("bf16" if self.device.type == "cuda" else "fp32") if p == "auto" else p
        if self.precision == "bf16" and self.device.type != "cuda":
            self.precision = "fp32"
        set_seed(self.seed)
        if self.cfg["deterministic"]:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
        self._build_data()
        self._build_model()
        self.loss_fn = build_loss(self.cfg["train"]["loss"])
        from src.data.datasets import build_augment
        tc = self.cfg["train"]
        self.augment = build_augment(tc["augment"], **(tc.get("augment_params") or {}))
        self.state = {"phase": 0, "epoch": 0, "step_in_epoch": 0, "phase_step": 0, "global_step": 0,
                      "es": {"best": -math.inf, "wait": 0, "best_epoch": -1, "stopped": False},
                      "history": [], "done": False, "config_hash": config_hash(self.cfg)}
        if self.cfg["selection"].get("subset_dc"):       # estado bestdc (plan B, docs/08 versión 2, V2.4.5)
            self.state["es_dc"] = {"best": -math.inf, "best_epoch": -1}
        self._resume_flag = bool(resume)
        self.opt, self.ema = None, None
        self.ckpt_path = self.out / "checkpoint.pt"
        with open(self.out / "config_resolved.json", "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, indent=2, default=str)
        self.resumed = False
        if resume and self.ckpt_path.exists():
            self._load_checkpoint()
            self.resumed = True

    # ================================================================== datos
    def _build_data(self):
        dc = self.cfg["data"]
        if dc.get("synthetic"):
            from src.train.synthetic import SyntheticCXR, synthetic_frame
            sc = dc["synthetic"]
            res, ns = int(sc.get("res", 32)), int(sc.get("n_sources", 3))
            scale = float(self.cfg["model"]["input_scale"])
            self.frames = {"train": synthetic_frame(sc.get("n_train", 64), ns, self.seed, "train"),
                           "val": synthetic_frame(sc.get("n_val", 32), ns, self.seed, "val"),
                           "test": synthetic_frame(sc.get("n_test", 32), ns, self.seed, "test")}
            self.frames["val_robust"] = self.frames["val"][self.frames["val"]["source_idx"] == 0].reset_index(drop=True)
            # zlib.crc32 y no hash(): hash() de una cadena cambia en cada proceso (PYTHONHASHSEED) y las pruebas
            # dejaban de ser deterministas (docs/13_entrenamiento_plan_b.md, Section 9)
            self._make_ds = lambda df, split: SyntheticCXR(df, res, scale, seed=zlib.crc32(split.encode()) % 997)
        elif dc.get("bench"):
            self._build_bench_data()
        else:
            from src.data.cache import cache_dir
            from src.data.datasets import CXRDataset
            from src.data.manifest import get_eval_set, get_splits, load_manifest
            man = load_manifest(resolve_data(dc["manifest"]) if dc.get("manifest") else None)
            sp = get_splits(dc["mode"], man, dc.get("train_sources"))
            self.frames = {"train": sp["train"]}
            names = set(dc["eval_sets"]) | {self.cfg["selection"]["subset"]}
            bloqueo_test_covidx(self.cfg, names)
            for n in names:
                self.frames[n] = sp[n] if n in ("val", "test") else get_eval_set(n, dc["mode"], man)
            cd = None
            if dc.get("cache") == "auto":
                c = cache_dir(dc["dataset"], dc["channel"], dc["res"])
                cd = c if (c / "index.csv").exists() else None
            elif dc.get("cache"):
                cd = Path(dc["cache"]) if os.path.isabs(dc["cache"]) else PROJECT_ROOT / dc["cache"]
            self.cache_dir = cd
            lim = dc.get("debug_limit")
            if lim:      # solo pruebas: primeras N filas de cada clase en cada conjunto
                import pandas as pd
                self.frames = {k: pd.concat([g.head(int(lim)) for _, g in v.groupby("label")]).reset_index(drop=True)
                               for k, v in self.frames.items()}
            self._make_ds = lambda df, split: CXRDataset(df, dc["channel"], dc["res"], cd, dc.get("require_cache", False))
        self.train_ds = self._make_ds(self.frames["train"], "train")
        tc = self.cfg["train"]
        self.steps_per_epoch = int(tc["steps_per_epoch"] or math.ceil(len(self.train_ds) / tc["batch_size"]))

    def _build_bench_data(self):
        """Benchmarks del plan B (data.bench): src/train/bench.py y docs/13_entrenamiento_plan_b.md."""
        from src.data.datasets import CXRDataset
        from src.train import bench as BM
        dc = self.cfg["data"]
        frames, meta, info = BM.load(self.cfg, self.seed)
        prev = self.cfg.get("bench") or {}
        if prev.get("rho") is not None and meta.get("rho") is not None and abs(float(prev["rho"]) - float(meta["rho"])) > 1e-9:
            raise BM.BenchConfigError("bench.rho de la configuración (%s) distinto del manifiesto (%s)" % (prev["rho"], meta["rho"]))
        self.bench_meta = meta
        self.cfg["bench"] = info
        names = list(dict.fromkeys(BM.requested_sets(self.cfg)))
        missing = [n for n in names if n not in frames]
        if missing:
            raise BM.BenchConfigError("conjuntos que el benchmark no define: %s (disponibles: %s)"
                                      % (missing, sorted(frames)))
        self.frames = frames
        lim = dc.get("debug_limit")
        if lim:      # solo pruebas: primeras N filas de cada clase en cada conjunto
            import pandas as pd
            self.frames = {k: (pd.concat([g.head(int(lim)) for _, g in v.groupby("label")]).reset_index(drop=True)
                               if len(v) else v) for k, v in self.frames.items()}
        cd = None
        if dc.get("cache") == "auto":
            cd = BM.cache_auto(meta, dc["channel"], dc["res"])
        elif dc.get("cache"):
            cd = Path(dc["cache"]) if os.path.isabs(dc["cache"]) else PROJECT_ROOT / dc["cache"]
        self.cache_dir = cd
        if dc.get("require_cache"):
            BM.check_cache(cd, self.frames, ["train"] + names)
        self._make_ds = lambda df, split: CXRDataset(df, dc["channel"], dc["res"], cd, dc.get("require_cache", False))

    def train_loader(self, start_batch: int):
        from src.data.datasets import BalancedInfiniteBatchSampler
        tc, dc = self.cfg["train"], self.cfg["data"]
        smp = BalancedInfiniteBatchSampler(self.train_ds.labels, tc["batch_size"], seed=self.seed,
                                           mode=tc["sampler"], start_batch=start_batch)
        nw = 0 if dc.get("synthetic") else int(dc["num_workers"])
        g = torch.Generator(); g.manual_seed(self.seed)
        return DataLoader(self.train_ds, batch_sampler=smp, num_workers=nw, collate_fn=_collate,
                          pin_memory=self.device.type == "cuda", generator=g,
                          prefetch_factor=4 if nw else None, persistent_workers=False)

    def eval_loader(self, name: str):
        dc = self.cfg["data"]
        ds = self._make_ds(self.frames[name], name)
        nw = 0 if dc.get("synthetic") else int(dc["num_workers"])
        return DataLoader(ds, batch_size=int(dc["eval_batch_size"]), shuffle=False, num_workers=nw,
                          collate_fn=_collate, pin_memory=self.device.type == "cuda")

    # ================================================================== modelo
    def _build_model(self):
        mc = self.cfg["model"]
        if mc["name"] == "tiny":
            from src.train.synthetic import TinyNet
            m = TinyNet(input_scale=float(mc["input_scale"]), dropout=float(mc["dropout"]), l2_hidden=float(mc["l2_hidden"]))
            self.tronco_info = {"tronco": "tiny", "preentrenado": False, "etiqueta_pesos": None}
        elif mc["name"] == "paper1_convnext":
            from src.baselines.paper1_model import Paper1ConvNeXt
            init = mc.get("init") or {"type": "none"}
            m = Paper1ConvNeXt(backbone=mc["backbone"], pretrained=init.get("type") == "timm_imagenet",
                               input_scale=float(mc["input_scale"]), dropout=float(mc["dropout"]),
                               drop_path_rate=float(mc["drop_path_rate"]), l2_hidden=float(mc["l2_hidden"]))
            # tronco y etiqueta efectiva de los pesos de timm, para summary.json (enmienda E8, punto 3; solo lectura)
            self.tronco_info = dict(getattr(m, "pesos_info", None) or {"tronco": mc["backbone"]})
            if mc.get("head_init", "glorot") == "glorot":
                for lin in (m.fc1, m.fc2):
                    torch.nn.init.xavier_uniform_(lin.weight); torch.nn.init.zeros_(lin.bias)
            if init.get("type") in ("keras_h5", "keras_model"):
                from src.convert.keras_convnext_to_timm import convert
                sd, info = convert(resolve_data(init["path"]))
                if init["type"] == "keras_h5":
                    sd = {k: v for k, v in sd.items() if k.startswith("backbone.")}
                missing, unexpected = m.load_state_dict(sd, strict=False)
                bb_missing = [k for k in missing if k.startswith("backbone.")]
                if bb_missing or unexpected:
                    raise RuntimeError("init Keras incompleta: faltan %s, sobran %s" % (bb_missing[:5], unexpected[:5]))
                self._init_info = {"type": init["type"], "path": str(init["path"]), "n_tensors": info["n_tensors"]}
                self.tronco_info["etiqueta_pesos"] = "%s (%s)" % (init["type"], init["path"])
        else:
            raise ValueError("modelo desconocido: %s" % mc["name"])
        self.model = m.to(self.device)

    def _backbone_params(self):
        m = self.model
        return list(m.backbone_parameters()) if hasattr(m, "backbone_parameters") else list(m.backbone.parameters())

    def _configure_trainable(self, phase: dict):
        bb = {id(p) for p in self._backbone_params()}
        frozen = {id(p) for _, p in self.model.backbone_norm_params()} if self.cfg["model"]["freeze_backbone_norms"] else set()
        for p in self.model.parameters():
            if id(p) in bb:
                p.requires_grad_(bool(phase["train_backbone"]) and id(p) not in frozen)
            else:
                p.requires_grad_(True)
        for mod in self.model.modules():     # troncos con BatchNorm: estadísticas congeladas con el tronco (fase A)
            if hasattr(mod, "estadisticas_bn_congeladas"):
                mod.estadisticas_bn_congeladas = not bool(phase["train_backbone"])

    def _param_groups(self):
        oc = self.cfg["train"]["optimizer"]
        bb = {id(p) for p in self._backbone_params()}
        groups = {"backbone": [], "head": [], "backbone_nd": [], "head_nd": []}
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            g = "backbone" if id(p) in bb else "head"
            if oc.get("no_decay_norm_bias") and p.ndim <= 1:
                g += "_nd"
            groups[g].append(p)
        lm = oc.get("lr_mult") or {}
        out = []
        for g, ps in groups.items():
            base = g.replace("_nd", "")
            d = {"name": g, "params": ps, "lr_mult": float(lm.get(base, 1.0))}
            if g.endswith("_nd"):
                d["weight_decay"] = 0.0
            out.append(d)
        return out

    def _setup_phase(self, fresh: bool):
        ph = self.cfg["train"]["phases"][self.state["phase"]]
        self._configure_trainable(ph)
        self.opt = build_optimizer(self._param_groups(), self.cfg["train"]["optimizer"], float(ph["lr"]))
        self.phase_total = self.steps_per_epoch * int(ph["epochs"])
        self.phase_warmup = int(self.phase_total * float(ph.get("warmup_frac", 0.1)))
        em = self.cfg["ema"]["mode"]
        self.ema = None
        if ph.get("ema") and em not in (None, "none", ""):
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            self.ema = WeightEMA(self.model, self.cfg["ema"]["momentum"], em,
                                 params=trainable if em == "tfkeras_exacta" else None)
        if fresh:
            self.log("phase_start", phase=ph["name"], epochs=ph["epochs"], lr=ph["lr"], steps_per_epoch=self.steps_per_epoch,
                     trainable=int(sum(p.numel() for p in self.model.parameters() if p.requires_grad)),
                     ema=self.ema.mode if self.ema else None)
        return ph

    # ================================================================== checkpoint
    def _rng_state(self):
        st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
        if torch.cuda.is_available():
            st["cuda"] = torch.cuda.get_rng_state_all()
        return st

    def _set_rng_state(self, st):
        # Los estados de torch son ByteTensor de CPU. Al cargar el checkpoint con map_location en la GPU llegan en
        # la GPU y torch.set_rng_state los rechaza ("RNG state must be a torch.ByteTensor"): se devuelven a la CPU.
        # Corregido el 17-11-2026, tras fallar la primera reanudación real en GPU.
        torch.set_rng_state(st["torch"].cpu()); np.random.set_state(st["numpy"]); random.setstate(st["python"])
        if "cuda" in st and torch.cuda.is_available():
            estados = [s.cpu() for s in st["cuda"]][:torch.cuda.device_count()]
            for i, s in enumerate(estados):
                torch.cuda.set_rng_state(s, i)

    def save_checkpoint(self, reason="epoch"):
        obj = {"model": self.model.state_dict(), "optimizer": self.opt.state_dict() if self.opt else None,
               "ema": self.ema.state_dict() if self.ema else None, "state": self.state, "rng": self._rng_state(),
               "reason": reason}
        atomic_torch_save(obj, self.ckpt_path)
        self.log("checkpoint", reason=reason, path=str(self.ckpt_path), **{k: self.state[k] for k in
                 ("phase", "epoch", "step_in_epoch", "phase_step", "global_step")})

    def _load_checkpoint(self):
        ck = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        if ck["state"].get("config_hash") != self.state["config_hash"]:
            self.log("aviso", msg="la configuración cambió desde el checkpoint; se reanuda igualmente")
        self.model.load_state_dict(ck["model"])
        self.state = ck["state"]
        if not self.state.get("done") and self.state["phase"] < len(self.cfg["train"]["phases"]):
            self._setup_phase(fresh=False)
            if ck["optimizer"] is not None:
                self.opt.load_state_dict(ck["optimizer"])
            if ck["ema"] is not None and self.ema is not None:
                self.ema.load_state_dict(ck["ema"])
        self._set_rng_state(ck["rng"])
        self.log("resume", **{k: self.state[k] for k in ("phase", "epoch", "step_in_epoch", "phase_step", "global_step")})

    # ================================================================== pasos
    def _autocast(self):
        if self.precision == "bf16" and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        import contextlib
        return contextlib.nullcontext()

    def train_step(self, batch):
        from src.data.datasets import make_generator
        tc = self.cfg["train"]
        self.model.train()
        x = batch["image"].to(self.device, non_blocking=True).float()
        y = batch["label"].to(self.device, non_blocking=True).float()
        if self.augment is not None:
            gen = make_generator(self.seed * 13 + self.state["phase"], self.state["phase_step"], self.device)
            x, y = self.augment(x, y, gen)
        lr = warmup_cosine(self.state["phase_step"], float(self._phase()["lr"]), self.phase_total, self.phase_warmup,
                           float(self._phase().get("min_lr", 1e-7)))
        set_lr(self.opt, lr)
        self.opt.zero_grad(set_to_none=True)
        B = x.shape[0]
        mb = int(tc.get("micro_batch") or B)
        tot = 0.0
        for i in range(0, B, mb):
            with self._autocast():
                z = self.model(x[i:i + mb])
            loss = self.loss_fn(z.float(), y[i:i + mb], reduction="sum") / B
            loss.backward()
            tot += float(loss.detach())
        reg = self.model.l2_penalty() if hasattr(self.model, "l2_penalty") else None
        if reg is not None and reg.requires_grad:
            reg.backward()
            tot += float(reg.detach())
        if tc.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], float(tc["grad_clip"]))
        self.opt.step()
        if self.ema is not None:
            self.ema.update(self.model)
        return tot, lr

    def _phase(self):
        return self.cfg["train"]["phases"][self.state["phase"]]

    @torch.no_grad()
    def predict(self, name: str, tta: bool, max_n: int | None = None):
        from src.eval.predict import predict_loader
        loader = self.eval_loader(name)
        max_batches = None
        if max_n:
            max_batches = max_n // int(self.cfg["data"]["eval_batch_size"])
        return predict_loader(self.model, loader, self.device, tta=tta, precision=self.precision, max_batches=max_batches)

    def validate(self, name=None):
        sc = self.cfg["selection"]
        name = name or sc["subset"]
        n = len(self.frames[name])
        drop = sc.get("drop_remainder_batch")
        pred = self.predict(name, tta=bool(sc.get("tta", False)))
        y, p = pred["label"], pred["prob"]
        if drop:
            keep = (n // int(drop)) * int(drop)
            y, p = y[:keep], p[:keep]
        return selection_metric(sc["metric"], sc.get("impl", "exact"), y, p), {
            "val_auc_roc": exact_metric("auc_roc", y, p), "val_auc_pr": exact_metric("auc_pr", y, p), "val_n": int(len(y))}

    # ================================================================== bucle
    def fit(self) -> int:
        if self.state.get("done"):
            self.log("ya_terminado", out=str(self.out))
            if self.cfg.get("storage"):
                self._apply_storage()
            return 0
        if self._finished_and_cleaned():
            self.log("ya_terminado", out=str(self.out), motivo="summary.json con almacenamiento aplicado y sin checkpoint")
            return 0
        tc = self.cfg["train"]
        stop = GracefulStop().install()
        guard = ThermalSelfCheck(every=int(self.cfg["thermal"]["every"]),
                                 log=lambda m: self.log("thermal", msg=m))
        self.log("start", seed=self.seed, device=str(self.device), precision=self.precision, out=str(self.out),
                 n_train=len(self.train_ds), steps_per_epoch=self.steps_per_epoch, resumed=self.resumed,
                 init=getattr(self, "_init_info", None), tronco=getattr(self, "tronco_info", None))
        if self.opt is None:
            self._setup_phase(fresh=True)
        run_steps = 0
        try:
            while self.state["phase"] < len(tc["phases"]):
                ph = self._phase()
                start = self.state["phase_step"] if tc.get("sampler_restart_each_phase", True) else self.state["global_step"]
                loader_it = iter(self.train_loader(start))
                while self.state["epoch"] < int(ph["epochs"]):
                    t0 = time.time()
                    losses = self.state.setdefault("epoch_losses", [])   # se conserva en el checkpoint
                    while self.state["step_in_epoch"] < self.steps_per_epoch:
                        batch = next(loader_it)
                        loss, lr = self.train_step(batch)
                        losses.append(loss)
                        self.state["step_in_epoch"] += 1; self.state["phase_step"] += 1; self.state["global_step"] += 1
                        run_steps += 1
                        if self.state["global_step"] % int(tc["log_every"]) == 0:
                            extra = {k: v for k, v in (getattr(self, "last_logs", None) or {}).items()
                                     if k not in ("phase", "epoch", "step", "loss", "lr")}   # términos propios del método
                            self.log("step", phase=ph["name"], epoch=self.state["epoch"], step=self.state["phase_step"],
                                     loss=loss, lr=lr, **extra)
                        guard.step()
                        if stop.requested:
                            raise TrainStop("SIGTERM (señal %s)" % stop.signum)
                        if tc.get("stop_after_steps") and run_steps >= int(tc["stop_after_steps"]):
                            raise TrainStop("stop_after_steps=%s" % tc["stop_after_steps"], EXIT_DEBUG_STOP)
                        if tc.get("checkpoint_every_steps") and self.state["global_step"] % int(tc["checkpoint_every_steps"]) == 0:
                            self.save_checkpoint("pasos")
                    if self._end_epoch(ph, losses, time.time() - t0):
                        break
                self._end_phase()
            self._finalize()
            stop.restore()
            if self.cfg.get("storage"):
                self._apply_storage()
            return 0
        except ThermalLimitReached as e:
            self.log("stop", reason="temperatura", detail=str(e), max_temp=guard.max_seen)
            self.save_checkpoint("temperatura")
            stop.restore()
            return EXIT_THERMAL
        except TrainStop as e:
            self.log("stop", reason=e.reason, code=e.code)
            self.save_checkpoint("parada")
            stop.restore()
            return e.code

    # ================================================================== almacenamiento (PLAN.md Section 13.5)
    def _finished_and_cleaned(self) -> bool:
        """Con política de almacenamiento, una corrida terminada borra su checkpoint; si se vuelve a lanzar sin
        --no-resume no debe reentrenar ni sobrescribir sus predicciones."""
        if not self.cfg.get("storage") or not getattr(self, "_resume_flag", True) or self.ckpt_path.exists():
            return False
        p = self.out / "summary.json"
        if not p.exists():
            return False
        try:
            return bool(json.load(open(p, encoding="utf-8")).get("almacenamiento"))
        except (OSError, ValueError):
            return False

    def _apply_storage(self):
        """Se llama solo cuando la corrida termina bien (fit devuelve 0). Conserva los estados de storage.states (y
        sus variantes dfr_), borra los demás y, con keep_checkpoint: false, el checkpoint de reanudación."""
        st = self.cfg.get("storage") or {}
        keep = set(st.get("states") or [])
        keep |= {"dfr_" + k for k in keep}
        sd = self.out / "states"
        removed, kept = [], []
        if sd.is_dir():
            for p in sorted(sd.glob("*.pt")):
                if p.stem in keep:
                    kept.append(p.name)
                else:
                    p.unlink()
                    removed.append(p.name)
        if not st.get("keep_checkpoint", True) and self.ckpt_path.exists():
            self.ckpt_path.unlink()
            removed.append("checkpoint.pt")
        info = {"estados_conservados": kept, "borrados": removed, "politica": st}
        sp = self.out / "summary.json"
        if sp.exists():
            summ = json.load(open(sp, encoding="utf-8"))
            summ["almacenamiento"] = info
            tmp = sp.with_name(".summary.json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(summ, f, indent=2, default=float)
            os.replace(tmp, sp)
        self.log("almacenamiento", **info)

    def _end_epoch(self, ph, losses, secs) -> bool:
        rec = {"phase": ph["name"], "epoch": self.state["epoch"], "loss": float(np.mean(losses)) if losses else None,
               "seconds": round(secs, 1)}
        stop_now = False
        if ph.get("early_stopping"):
            metric, extra = self.validate()
            rec.update(extra); rec["selection"] = metric
            es = self.state["es"]
            es["wait"] += 1
            if metric > es["best"]:
                es["best"], es["best_epoch"], es["wait"] = metric, self.state["epoch"], 0
                atomic_torch_save(self.model.state_dict(), self.out / "states" / "best.pt")
                rec["improved"] = True
            elif es["wait"] >= int(self.cfg["selection"]["patience"]) and self.state["epoch"] > 0:
                es["stopped"] = True
                stop_now = True
            rec["wait"] = es["wait"]
            dcn = self.cfg["selection"].get("subset_dc")
            if dcn:          # selección de sensibilidad (bestdc): no interviene en la parada temprana
                mdc, xdc = self.validate(dcn)
                rec["selection_dc"] = mdc
                rec["dc_auc_roc"], rec["dc_n"] = xdc["val_auc_roc"], xdc["val_n"]
                edc = self.state["es_dc"]
                if mdc > edc["best"]:
                    edc["best"], edc["best_epoch"] = mdc, self.state["epoch"]
                    atomic_torch_save(self.model.state_dict(), self.out / "states" / "bestdc.pt")
                    rec["improved_dc"] = True
        if self.ema is not None:
            rec["ema_t"] = self.ema.t
            rec["ema_init_weight"] = self.ema.momentum ** self.ema.t if self.ema.mode == "tfkeras_exacta" else None
        self.state["history"].append(rec)
        self.log("epoch", **rec)
        self.state["epoch"] += 1
        self.state["step_in_epoch"] = 0
        self.state["epoch_losses"] = []
        if stop_now:
            self.log("early_stopping", epoch=self.state["epoch"], best_epoch=self.state["es"]["best_epoch"] + 1,
                     best=self.state["es"]["best"])
        self.save_checkpoint("epoca")
        return stop_now

    def _end_phase(self):
        ph = self._phase()
        if self.ema is not None:
            atomic_torch_save(self.ema.ema_state_dict(self.model), self.out / "states" / "ema.pt")
            self.ema_info = {"mode": self.ema.mode, "t": self.ema.t, "momentum": self.ema.momentum}
            self.state["ema_info"] = self.ema_info
        self.log("phase_end", phase=ph["name"], epochs_done=self.state["epoch"])
        if self.state["phase"] + 1 < len(self.cfg["train"]["phases"]):
            self.state["phase"] += 1
            self.state["epoch"] = self.state["step_in_epoch"] = self.state["phase_step"] = 0
            self._setup_phase(fresh=True)
        else:
            self.state["phase"] += 1

    def _finalize(self):
        states_dir = self.out / "states"
        atomic_torch_save(self.model.state_dict(), states_dir / "last.pt")
        if not (states_dir / "best.pt").exists():
            atomic_torch_save(self.model.state_dict(), states_dir / "best.pt")
        if self.cfg["selection"].get("subset_dc") and not (states_dir / "bestdc.pt").exists():
            atomic_torch_save(self.model.state_dict(), states_dir / "bestdc.pt")
        last = {k: v.clone() for k, v in self.model.state_dict().items()}
        summary = {"seed": self.seed, "experiment": self.cfg["experiment"], "history": self.state["history"],
                   "early_stopping": self.state["es"], "ema": self.state.get("ema_info"), "results": {},
                   "tronco": getattr(self, "tronco_info", None)}
        if self.cfg.get("bench"):
            summary["bench"] = self.cfg["bench"]
            summary["early_stopping_dc"] = self.state.get("es_dc")
            summary["seleccion_dc"] = self.cfg["selection"].get("subset_dc")
        from src.eval.predict import save_predictions, summarize
        for st in self.cfg["eval"]["states"]:
            p = states_dir / ("%s.pt" % st)
            if not p.exists():
                continue
            self.model.load_state_dict(torch.load(p, map_location=self.device))
            for name in self.cfg["data"]["eval_sets"]:
                pred = self.predict(name, tta=bool(self.cfg["eval"]["tta"]))
                save_predictions(pred, self.out / "predictions" / ("%s_%s.csv" % (st, name)))
                summary["results"]["%s/%s" % (st, name)] = summarize(pred)
                self.log("eval", state=st, set=name, **summary["results"]["%s/%s" % (st, name)])
        self.model.load_state_dict(last)
        self.state["done"] = True
        with open(self.out / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=float)
        self.save_checkpoint("fin")
        self.log("end", out=str(self.out))
