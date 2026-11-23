#!/usr/bin/env bash
# Arranca el vigilante del anfitrión (CPU, discos, memoria) en una ventana "host" de la sesión tmux "paper_new".
# No toca las ventanas guard ni runner ni los trabajos en marcha.
# Uso, en el servidor:   bash /home/administrador/Proyectos/paper_new/runner/start_host_guard.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="paper_new"
command -v tmux >/dev/null 2>&1 || { echo "ERROR: no se encuentra tmux" >&2; exit 1; }
Q_ROOT="$(printf '%q' "$ROOT")"
HOST_CMD="cd $Q_ROOT && while true; do python3 runner/host_guard.py; rc=\$?; if [ \$rc -eq 0 ]; then echo \"vigilante del anfitrión terminado\"; break; fi; echo \"vigilante del anfitrión caído (código \$rc); se reinicia en 5 s\"; sleep 5; done; exec bash"
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n host "$HOST_CMD"
    echo "[start_host_guard] sesión $SESSION creada solo con la ventana host (el resto se arranca con runner/start_tmux.sh)"
elif tmux list-windows -t "$SESSION" -F '#{window_name}' | grep -qx host; then
    echo "[start_host_guard] la ventana host ya existe: no se hace nada"
else
    tmux new-window -d -t "$SESSION" -n host "$HOST_CMD"
    echo "[start_host_guard] ventana host creada"
fi
sleep 3
echo "Sensores que ve el vigilante del anfitrión:"
python3 "$ROOT/runner/host_guard.py" --print-sensors | sed 's/^/    /'
