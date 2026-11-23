#!/usr/bin/env python3
"""Crea un trabajo válido en jobs/pending/ (se usa desde la VM). Solo biblioteca estándar.

Ejemplo:
  python3 runner/submit_job.py --id replica_seed0 --gpu any --prioridad 5 --tiempo-max-min 600 \
      --descripcion "Réplica del artículo 1, semilla 0" -- python src/train/train.py --config configs/replica.yaml

La escritura es atómica: se escribe un temporal oculto en la misma carpeta pending/ y se renombra.
"""
import argparse
import json
import os
import re
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from job_runner import DEFAULT_ALLOW, DEFAULT_ALLOW_BASH_DIR, validate_job  # noqa: E402

DEFAULT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Encola un trabajo en jobs/pending/")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--id", default=None, help="identificador; por defecto se genera con la fecha")
    ap.add_argument("--gpu", default="any", help="0, 1 o any")
    ap.add_argument("--prioridad", type=int, default=0, help="mayor número = antes")
    ap.add_argument("--tiempo-max-min", type=float, default=60.0)
    ap.add_argument("--descripcion", default="")
    ap.add_argument("--no-reencolar", action="store_true",
                    help="si se detiene por temperatura, va a failed/ en lugar de volver a pending/")
    ap.add_argument("--dry-run", action="store_true", help="muestra el JSON sin escribirlo")
    ap.add_argument("comando", nargs=argparse.REMAINDER, help="-- seguido del comando y sus argumentos")
    a = ap.parse_args(argv)
    cmd = a.comando[1:] if a.comando and a.comando[0] == "--" else a.comando
    jid = a.id or ("job_%s_%s" % (time.strftime("%Y%m%d_%H%M%S"), (re.sub(r"[^A-Za-z0-9]+", "-", cmd[0]) if cmd else "x")))[:64]
    gpu = a.gpu if a.gpu == "any" else (int(a.gpu) if a.gpu.isdigit() else a.gpu)
    tmax = int(a.tiempo_max_min) if float(a.tiempo_max_min).is_integer() else a.tiempo_max_min
    job = {"id": jid, "comando": cmd, "gpu": gpu, "prioridad": a.prioridad, "tiempo_max_min": tmax,
           "descripcion": a.descripcion, "creado": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "origen": "submit_job.py"}
    if a.no_reencolar:
        job["reencolar_si_termico"] = False
    errs = validate_job(job, jid, DEFAULT_ALLOW, DEFAULT_ALLOW_BASH_DIR)
    if errs:
        for e in errs:
            print("ERROR: %s" % e, file=sys.stderr)
        return 2
    text = json.dumps(job, ensure_ascii=False, indent=2) + "\n"
    if a.dry_run:
        sys.stdout.write(text)
        return 0
    root = os.path.realpath(a.root)
    pending = os.path.join(root, "jobs", "pending")
    os.makedirs(pending, exist_ok=True)
    for sub in ("pending", "running", "done"):
        if os.path.exists(os.path.join(root, "jobs", sub, jid + ".json")):
            print("ERROR: ya existe un trabajo con id %s en jobs/%s/" % (jid, sub), file=sys.stderr)
            return 3
    final = os.path.join(pending, jid + ".json")
    fd, tmp = tempfile.mkstemp(prefix="." + jid + ".", suffix=".part", dir=pending)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp, 0o644)
    os.replace(tmp, final)
    print(final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
