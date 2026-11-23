#!/usr/bin/env bash
# Parada ordenada: 1) el ejecutor detiene sus trabajos (SIGTERM, checkpoint, vuelven a pending) y sale;
# 2) se detiene el vigilante; 3) se cierra la sesión tmux. No toca procesos ajenos al proyecto.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/runner/state"
SESSION="paper_new"

hb_pid() {  # PID registrado en un latido
    python3 -c 'import json,sys
try:
    print(int(json.load(open(sys.argv[1])).get("pid", 0)))
except Exception:
    print(0)' "$1"
}

is_ours() {  # el PID existe y su línea de órdenes contiene el script esperado
    local pid="$1" script="$2"
    [[ "$pid" -gt 1 ]] && [[ -r "/proc/$pid/cmdline" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "$script"
}

stop_pid() {  # stop_pid PID SCRIPT ESPERA_MAX
    local pid="$1" script="$2" max="$3" i
    if ! is_ours "$pid" "$script"; then
        echo "[stop_all] $script no está en marcha"
        return 0
    fi
    echo "[stop_all] SIGTERM a $script (PID $pid); esperando hasta $max s"
    kill -TERM "$pid" 2>/dev/null || true
    for i in $(seq 1 "$max"); do
        is_ours "$pid" "$script" || { echo "[stop_all] $script detenido"; return 0; }
        sleep 1
    done
    echo "[stop_all] AVISO: $script sigue vivo tras $max s" >&2
    return 1
}

RUNNER_PID=0; GUARD_PID=0
[[ -f "$STATE/runner_heartbeat.json" ]] && RUNNER_PID="$(hb_pid "$STATE/runner_heartbeat.json")"
[[ -f "$STATE/guard_heartbeat.json" ]] && GUARD_PID="$(hb_pid "$STATE/guard_heartbeat.json")"

# 1) ejecutor primero (el vigilante sigue protegiendo mientras los trabajos guardan checkpoint)
stop_pid "$RUNNER_PID" job_runner.py 120 || true

# Contenedores del proyecto que hubieran quedado vivos
if command -v docker >/dev/null 2>&1; then
    LEFT="$(docker ps -q --filter label=paper_new.job=1 2>/dev/null || true)"
    if [[ -n "$LEFT" ]]; then
        echo "[stop_all] quedan contenedores del proyecto: docker stop -t 30"
        # shellcheck disable=SC2086
        docker stop -t 30 $LEFT >/dev/null || true
    fi
fi

# 2) vigilante
stop_pid "$GUARD_PID" gpu_guard.py 30 || true

# 3) sesión tmux
if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
    echo "[stop_all] sesión tmux $SESSION cerrada"
fi
echo "[stop_all] listo. Los trabajos interrumpidos están en jobs/pending/ con la marca correspondiente."
