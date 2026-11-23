#!/usr/bin/env bash
# Muestra el estado: latidos, cola, temperaturas y últimos eventos térmicos.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$ROOT/runner/job_runner.py" --root "$ROOT" --status
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Lectura directa de nvidia-smi:"
    nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv,noheader \
        | sed 's/^/    GPU /' || echo "    nvidia-smi falló"
fi
if command -v docker >/dev/null 2>&1; then
    echo "Contenedores del proyecto (etiqueta paper_new.job=1):"
    docker ps --filter label=paper_new.job=1 --format '    {{.Names}}  {{.Status}}' 2>/dev/null || echo "    docker no responde"
fi
if command -v tmux >/dev/null 2>&1 && tmux has-session -t paper_new 2>/dev/null; then
    echo "Sesión tmux paper_new: $(tmux list-windows -t paper_new -F '#{window_name}' | tr '\n' ' ')"
else
    echo "Sesión tmux paper_new: no existe"
fi
