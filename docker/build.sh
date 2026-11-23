#!/usr/bin/env bash
# Construye la imagen paper-new:cu126 y guarda la lista exacta de paquetes instalados.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${PAPER_NEW_IMAGE:-paper-new:cu126}"
docker build -t "$IMAGE" -f "$HERE/Dockerfile" "$HERE"
docker run --rm --entrypoint pip "$IMAGE" freeze > "$HERE/requirements.lock.txt"
echo "Paquetes instalados guardados en $HERE/requirements.lock.txt"
echo "Comprobación de GPU dentro del contenedor:"
docker run --rm --gpus device=0 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility "$IMAGE" \
    python -c "import torch, pynvml; pynvml.nvmlInit(); print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'disponible', torch.cuda.is_available(), 'GPU visibles', torch.cuda.device_count())"
