"""Punto de entrada de las líneas base:

    python -m src.baselines.run --config configs/baseline_<metodo>.yaml --seed <S> [--trial T] [--out DIR]
                                [--no-resume] [--set clave.subclave=valor ...] [--experimento NOMBRE] [--list]

--trial T aplica los hiperparámetros del ensayo T del protocolo de búsqueda común (src/baselines/registry.py;
T = 0 son los valores por defecto) y añade _t<T> al nombre del experimento. --experimento fija el nombre después de
aplicar el ensayo (corridas finales de B2 y B3a con el nombre que espera src/eval/dosis.py, sin el sufijo _t<T>). Códigos de salida como
src.train.run: 0 al terminar, 75 por temperatura o SIGTERM (checkpoint guardado, se reanuda), 3 en la parada de
depuración y 1 si la pérdida deja de ser finita.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config(path, trial=None, sets=(), experimento=None):
    import yaml
    from src.baselines.registry import apply_hparams, sample_hparams
    from src.train.config import load_yaml, resolve
    cfg = resolve(load_yaml(path))
    for kv in sets:
        k, v = kv.split("=", 1)
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(v)
    if trial is not None:
        hp = sample_hparams((cfg.get("baseline") or {}).get("method", "erm"), int(trial))
        cfg = apply_hparams(cfg, hp)
        cfg["experiment"] = "%s_t%d" % (cfg["experiment"], int(trial))
        cfg.setdefault("baseline", {})["trial"] = int(trial)
        cfg["baseline"]["trial_hparams"] = hp
    if experimento:
        cfg["experiment"] = experimento
    return cfg


def main(argv=None):
    from src.baselines.trainer import BaselineTrainer
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--trial", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--experimento", default=None, help="nombre del experimento tras aplicar --trial (plan B)")
    ap.add_argument("--list", action="store_true", help="lista los métodos registrados y sale")
    a = ap.parse_args(argv)
    if a.list:
        from src.baselines.registry import all_methods
        for n, c in sorted(all_methods().items()):
            print("%-16s %-13s %-34s %s" % (n, c.status, c.ref, c.title))
        return 0
    if not a.config or a.seed is None:
        ap.error("--config y --seed son obligatorios")
    cfg = load_config(a.config, a.trial, a.set, a.experimento)
    tr = BaselineTrainer(cfg, a.seed, resume=not a.no_resume, out_dir=a.out)
    return tr.fit()


if __name__ == "__main__":
    sys.exit(main())
