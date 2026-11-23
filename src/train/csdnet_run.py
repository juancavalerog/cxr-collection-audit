"""Entrada del entrenamiento de CSD-Net (Algoritmo 1):

    python -m src.train.csdnet_run --config configs/csdnet_piloto.yaml --seed 42 [--out <dir>] [--no-resume]
                                   [--set clave.subclave=valor ...]

--trial T aplica el ensayo T del presupuesto de búsqueda común con las líneas base (docs/07_piloto.md, Section 2;
src/baselines/registry.py): 8 configuraciones (T = 0 son los valores por defecto de la Tabla 6), una semilla en
la búsqueda y las cinco semillas 42, 123, 2024, 7 y 999 para la configuración congelada. La tasa pico de la fase B
se muestrea en 10^U(-5, -3.5) y se aplica al tronco convolucional, con el Transformer a 2 veces y los módulos
nuevos a 20 veces esa tasa (las razones de la Tabla 6); además se muestrean lambda_cons en 10^U(-1, 0.7),
lambda_ind en 10^U(-1, 1) y rho_within en U(0, 0.75) (rangos de la Tabla 6). El muestreo es determinista en
(T, "csdnet"), con la misma convención que sample_hparams de las líneas base.

Códigos de salida: 0 al terminar; 75 por temperatura o SIGTERM (checkpoint guardado, se reanuda al relanzar);
3 en la parada de depuración stop_after_steps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SEARCH_BUDGET = {"configuraciones": 8, "semillas_busqueda": 1, "semillas_finales": [42, 123, 2024, 7, 999]}
CSD_SEARCH = {"lr_B": ("log10", -5.0, -3.5), "lambda_cons": ("log10", -1.0, 0.7), "lambda_ind": ("log10", -1.0, 1.0),
              "rho_within": ("uniform", 0.0, 0.75)}
LR_RATIOS = {"trunk_conv": 1.0, "transformer": 2.0, "new": 20.0}


def sample_csdnet_hparams(trial: int) -> dict:
    import zlib
    import numpy as np
    if int(trial) == 0:
        return {}
    if not 0 <= int(trial) < SEARCH_BUDGET["configuraciones"]:
        raise ValueError("ensayo fuera del presupuesto (0 a %d)" % (SEARCH_BUDGET["configuraciones"] - 1))
    rng = np.random.default_rng([int(trial), zlib.crc32(b"csdnet")])
    out = {}
    for k, (kind, lo, hi) in sorted(CSD_SEARCH.items()):
        v = rng.uniform(lo, hi)
        out[k] = float(10 ** v) if kind == "log10" else float(v)
    return out


def apply_csdnet_hparams(cfg: dict, hp: dict) -> dict:
    import copy
    c = copy.deepcopy(cfg)
    cc = c.setdefault("csdnet", {})
    if "lr_B" in hp:
        cc["lr"] = {k: hp["lr_B"] * r for k, r in LR_RATIOS.items()}
    w = cc.setdefault("weights", {})
    for k in ("lambda_cons", "lambda_ind", "rho_within"):
        if k in hp:
            w[k] = hp[k]
    return c


def main(argv=None):
    import yaml
    from src.train.config import load_yaml
    from src.train.csdnet_loop import CSDNetTrainer
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--trial", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = load_yaml(a.config)
    for kv in a.set:
        k, v = kv.split("=", 1)
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(v)
    if a.trial is not None:
        hp = sample_csdnet_hparams(a.trial)
        cfg = apply_csdnet_hparams(cfg, hp)
        cfg["experiment"] = "%s_t%d" % (cfg.get("experiment", "csdnet"), a.trial)
        cfg.setdefault("csdnet", {})["trial"] = {"t": int(a.trial), "hparams": hp}
    tr = CSDNetTrainer(cfg, a.seed, resume=not a.no_resume, out_dir=a.out)
    return tr.fit()


if __name__ == "__main__":
    sys.exit(main())
