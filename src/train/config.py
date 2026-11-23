"""Configuración del entrenador: valores por defecto, carga de YAML y fusión profunda.

Un YAML puede heredar de otro con la clave `base: <ruta relativa a configs/>` o de varios, fusionados en orden, con
`base: [a.yaml, b.yaml]`.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

DEFAULTS = {
    "experiment": "sin_nombre",
    "output_dir": "results/train/{experiment}/seed{seed}",
    "device": None,                      # None: cuda si hay, si no cpu
    "precision": "auto",                 # auto: bf16 en GPU, fp32 en CPU; o fp32 / bf16
    "deterministic": True,
    "data": {
        "dataset": "covidx_cxr4",
        "manifest": None,                # None: data/manifests/covidx_cxr4_clean.csv
        "mode": "oficial",               # oficial | limpia
        "train_sources": None,           # p. ej. [bimcv] = régimen controlado
        "channel": "p1",                 # p1 | homogeneo
        "res": 224,
        "cache": "auto",                 # auto: data/cache/<dataset>/<canal>_<res> si existe; null: al vuelo
        "require_cache": False,
        "num_workers": 8,
        "eval_batch_size": 256,
        "eval_sets": ["val", "test"],
        "synthetic": None,               # pruebas: {n_train, n_val, n_test, res, n_sources}
        "debug_limit": None,             # pruebas: N primeras filas por clase en cada conjunto
    },
    "model": {
        "name": "paper1_convnext",       # paper1_convnext | tiny
        "backbone": "convnext_tiny",
        "init": {"type": "none", "path": None},   # none | keras_h5 | keras_model | timm_imagenet
        "input_scale": 255.0,
        "dropout": 0.5,
        "drop_path_rate": 0.0,
        "l2_hidden": 1e-2,
        "freeze_backbone_norms": True,
        "head_init": "glorot",
    },
    "train": {
        "batch_size": 768,
        "micro_batch": None,
        "steps_per_epoch": None,         # None: ceil(n_train / batch_size)
        "sampler": "bernoulli",          # bernoulli (sample_from_datasets) | exact
        "sampler_restart_each_phase": True,
        "augment": "p1",                 # p1 | nuevo | none
        "augment_params": {},
        "grad_clip": None,
        "loss": {"name": "focal", "alpha": 0.5, "gamma": 2.0, "apply_class_balancing": True, "label_smoothing": 0.0},
        "optimizer": {"name": "tf_adamw", "weight_decay": 1e-5, "betas": [0.9, 0.999], "eps": 1e-7,
                      "lr_mult": {"backbone": 1.0, "head": 1.0}, "no_decay_norm_bias": False},
        "phases": [
            {"name": "A", "epochs": 5, "lr": 1e-3, "warmup_frac": 0.1, "min_lr": 1e-7, "train_backbone": False,
             "ema": False, "early_stopping": False},
            {"name": "B", "epochs": 100, "lr": 2.5e-5, "warmup_frac": 0.1, "min_lr": 1e-7, "train_backbone": True,
             "ema": True, "early_stopping": True},
        ],
        "log_every": 10,
        "stop_after_steps": None,        # solo pruebas: detiene tras N pasos globales (código 3)
        "checkpoint_every_steps": 0,     # 0: solo al final de cada época y al parar
    },
    "ema": {"mode": "tfkeras_exacta", "momentum": 0.9999},
    "selection": {"metric": "auc_pr", "subset": "val", "impl": "keras200", "patience": 15, "tta": False,
                  "drop_remainder_batch": None},
    "eval": {"tta": True, "states": ["last", "best", "ema"]},
    "thermal": {"every": 50},
}


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path) -> dict:
    import yaml
    path = Path(path)
    cfg = yaml.safe_load(open(path, encoding="utf-8")) or {}
    base = cfg.pop("base", None)
    if isinstance(base, (list, tuple)):        # varias bases, fusionadas en orden (configs/b3a_*.yaml, b2_*.yaml)
        merged = {}
        for b in base:
            merged = deep_merge(merged, load_yaml(path.parent / b))
        cfg = deep_merge(merged, cfg)
    elif base:
        cfg = deep_merge(load_yaml(path.parent / base), cfg)
    return cfg


def resolve(cfg: dict) -> dict:
    return deep_merge(DEFAULTS, cfg)


def config_hash(cfg: dict) -> str:
    c = {k: v for k, v in cfg.items() if k not in ("output_dir",)}
    return hashlib.sha1(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:12]
