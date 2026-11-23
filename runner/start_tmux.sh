#!/usr/bin/env bash
# Arranca (o reutiliza) la sesión tmux "paper_new" con tres ventanas: guard, runner y logs.
# Uso, en el servidor:   bash /home/administrador/Proyectos/paper_new/runner/start_tmux.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="paper_new"
IMAGE="${PAPER_NEW_IMAGE:-paper-new:cu126}"
DATA_DIR="${PAPER_NEW_DATA_DIR:-/home/administrador/notebooks/covid}"
VOLUMES=(paper_new_torch_cache paper_new_hf_cache)

fail() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "[start_tmux] $*"; }

# ---------------------------------------------------------------- comprobaciones previas
for c in tmux docker nvidia-smi python3; do
    command -v "$c" >/dev/null 2>&1 || fail "no se encuentra '$c' en el PATH"
done
nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader >/dev/null 2>&1 \
    || fail "nvidia-smi no responde: no se arranca nada (sin lectura térmica no hay entrenamiento)"
docker info >/dev/null 2>&1 \
    || fail "docker no responde para este usuario sin sudo (¿pertenece al grupo docker?)"
docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || fail "no existe la imagen $IMAGE. Constrúyala con: bash \"$ROOT/docker/build.sh\""
[[ -d "$DATA_DIR" && -r "$DATA_DIR" ]] || fail "no se puede leer la carpeta de datos $DATA_DIR"
for d in jobs jobs/pending jobs/running jobs/done jobs/failed logs logs/jobs runner/state runner/state/running; do
    mkdir -p "$ROOT/$d"
    [[ -w "$ROOT/$d" ]] || fail "sin permiso de escritura en $ROOT/$d"
done
if [[ -n "${PAPER_NEW_STOP_C:-}" || -n "${PAPER_NEW_WARN_C:-}" ]]; then
    info "AVISO: se ignoran PAPER_NEW_STOP_C/PAPER_NEW_WARN_C del entorno; el arranque normal usa 75 °C y 80 °C"
fi

# Volúmenes con nombre para las cachés: deben ser escribibles con el UID del usuario
for v in "${VOLUMES[@]}"; do
    docker volume create "$v" >/dev/null
    if ! docker run --rm --user "$(id -u):$(id -g)" -v "$v:/mnt/v" --entrypoint test "$IMAGE" -w /mnt/v >/dev/null 2>&1; then
        info "ajustando el propietario del volumen $v a $(id -u):$(id -g)"
        docker run --rm -v "$v:/mnt/v" --entrypoint chown "$IMAGE" "$(id -u):$(id -g)" /mnt/v >/dev/null
    fi
done

# ---------------------------------------------------------------- sesión tmux
Q_ROOT="$(printf '%q' "$ROOT")"
GUARD_CMD="cd $Q_ROOT && while true; do env -u PAPER_NEW_STOP_C -u PAPER_NEW_WARN_C python3 runner/gpu_guard.py; rc=\$?; if [ \$rc -eq 0 ] || [ \$rc -eq 3 ]; then echo \"vigilante terminado (código \$rc)\"; break; fi; echo \"vigilante caído (código \$rc); se reinicia en 5 s\"; sleep 5; done; exec bash"
RUNNER_CMD="cd $Q_ROOT && env -u PAPER_NEW_STOP_C -u PAPER_NEW_WARN_C python3 runner/job_runner.py --image $(printf '%q' "$IMAGE") --data-dir $(printf '%q' "$DATA_DIR"); echo \"ejecutor terminado (código \$?)\"; exec bash"
LOGS_CMD="cd $Q_ROOT && touch logs/guard.log logs/runner.log logs/thermal_events.log && tail -n 30 -F logs/thermal_events.log logs/guard.log logs/runner.log"

has_window() { tmux list-windows -t "$SESSION" -F '#{window_name}' 2>/dev/null | grep -qx "$1"; }

wait_guard() {
    local i
    for i in $(seq 1 30); do
        if python3 "$ROOT/runner/job_runner.py" --root "$ROOT" --check-guard >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

if tmux has-session -t "$SESSION" 2>/dev/null; then
    info "la sesión $SESSION ya existe: se reutiliza"
else
    tmux new-session -d -s "$SESSION" -n guard "$GUARD_CMD"
    info "ventana guard creada"
fi
if ! has_window guard; then
    tmux new-window -d -t "$SESSION" -n guard "$GUARD_CMD"
    info "ventana guard creada"
fi
wait_guard || fail "el vigilante no da latido válido tras 30 s; revise: tmux attach -t $SESSION (ventana guard)"
info "vigilante en marcha"
if ! has_window runner; then
    tmux new-window -d -t "$SESSION" -n runner "$RUNNER_CMD"
    info "ventana runner creada"
fi
if ! has_window logs; then
    tmux new-window -d -t "$SESSION" -n logs "$LOGS_CMD"
    info "ventana logs creada"
fi
# Vigilante del anfitrión (CPU, discos, memoria), añadido el 17-11-2026: ventana "host"
if bash "$ROOT/runner/start_host_guard.sh" >/dev/null 2>&1; then
    info "ventana host (vigilante del anfitrión) en marcha"
else
    info "AVISO: no se pudo arrancar el vigilante del anfitrión; pruebe: bash \"$ROOT/runner/start_host_guard.sh\""
fi
sleep 3
bash "$ROOT/runner/status.sh" || true
echo
info "listo. Ver las ventanas: tmux attach -t $SESSION  (Ctrl-b n para cambiar, Ctrl-b d para salir sin detener)"
info "estado: bash \"$ROOT/runner/status.sh\"    parada ordenada: bash \"$ROOT/runner/stop_all.sh\""
