"""Punto de entrada: python -m src.train.run --config configs/<x>.yaml --seed <S> [--out <dir>] [--no-resume]

Código de salida: 0 al terminar; 75 si se detuvo por temperatura o SIGTERM (checkpoint guardado; el ejecutor lo
devuelve a la cola y la siguiente ejecución reanuda); 3 en la parada de depuración stop_after_steps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    from src.train.config import load_yaml
    from src.train.trainer import Trainer
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="sobrescrituras clave.subclave=valor (YAML)")
    a = ap.parse_args(argv)
    cfg = load_yaml(a.config)
    import yaml
    for kv in a.set:
        k, v = kv.split("=", 1)
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(v)
    tr = Trainer(cfg, a.seed, resume=not a.no_resume, out_dir=a.out)
    return tr.fit()


if __name__ == "__main__":
    sys.exit(main())
