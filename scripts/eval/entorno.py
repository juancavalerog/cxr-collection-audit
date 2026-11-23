"""Captura del entorno de análisis dentro del contenedor (review/informe_puerta7.md, Section 5, punto 8).

    python scripts/eval/entorno.py --out results/gate7/entorno_contenedor.json

Registra la versión de Python, la lista completa de paquetes instalados (equivalente a pip freeze), las versiones de
torch, CUDA, cuDNN y timm, las GPU visibles, el nombre de la máquina (en Docker, el identificador corto del
contenedor), las variables del trabajo y las diferencias con docker/requirements.lock.txt. El identificador de la
imagen no es visible desde dentro del contenedor: debe anotarse en el anfitrión con
`docker image inspect paper-new:cu126 --format '{{.Id}} {{.RepoDigests}} {{.Created}}' > results/gate7/imagen_docker.txt`.
No lee datos.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def capturar() -> dict:
    sys.path.insert(0, str(ROOT / "scripts" / "eval"))
    from congelar import comparar_lock, entorno
    out = entorno()
    out["fecha_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out["ejecutable"] = sys.executable
    out["en_docker"] = os.path.exists("/.dockerenv")
    out["variables"] = {k: os.environ.get(k) for k in ("PAPER_NEW_JOB_ID", "PAPER_NEW_GPU_INDEX", "TORCH_HOME", "HF_HOME",
                                                       "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")}
    try:
        cg = Path("/proc/self/cgroup").read_text()
        out["cgroup"] = cg.strip().splitlines()[-1][:300]
    except OSError:
        out["cgroup"] = None
    try:
        import torch
        out["torch_cuda"] = torch.version.cuda
        out["cudnn"] = torch.backends.cudnn.version()
        out["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as e:  # noqa: BLE001
        out["torch_error"] = str(e)
    try:
        import timm
        m = timm.create_model("convnext_tiny", pretrained=False, num_classes=0, global_pool="")
        out["timm_convnext_tiny_pesos"] = {k: m.pretrained_cfg.get(k) for k in ("tag", "hf_hub_id", "url")}
    except Exception as e:  # noqa: BLE001
        out["timm_error"] = str(e)
    out["diferencias_con_lock"] = comparar_lock(out["paquetes"], ROOT / "docker" / "requirements.lock.txt")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "results" / "gate7" / "entorno_contenedor.json"))
    a = ap.parse_args(argv)
    e = capturar()
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(e, indent=1, ensure_ascii=False), encoding="utf-8")
    print("python %s; torch %s (CUDA %s); timm %s; %d paquetes; %d diferencias con el lock; en Docker: %s; máquina %s"
          % (platform.python_version(), e.get("version_torch"), e.get("torch_cuda"), e.get("version_timm"),
             len(e["paquetes"]), len(e["diferencias_con_lock"]), e["en_docker"], e["maquina"]))
    for d in e["diferencias_con_lock"]:
        print("  " + d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
