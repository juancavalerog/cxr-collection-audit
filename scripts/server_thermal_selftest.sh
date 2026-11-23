#!/usr/bin/env bash
# Prueba térmica obligatoria en el SERVIDOR antes del primer entrenamiento (PLAN.md, Section 9).
#
# Qué hace:
#  1. Exige que la sesión normal esté detenida (bash runner/stop_all.sh), sin THERMAL_STOP y sin trabajos en curso.
#  2. Lee la temperatura actual y fija un umbral de corte TEMPORAL = máxima actual + 3 °C (siempre < 80).
#     El umbral bajado solo existe en la variable de entorno de los dos procesos que lanza esta prueba.
#  3. Lanza un vigilante y un ejecutor temporales y encola, por la cola normal, un trabajo de carga
#     ficticia en GPU de 2 minutos (tests/dummy_job.py --gpu-load) en la GPU más caliente.
#  4. Verifica: el vigilante crea jobs/THERMAL_STOP, el trabajo escribe "checkpoint guardado",
#     el contenedor desaparece y el evento queda en logs/thermal_events.log.
#  5. Al terminar, aunque falle, detiene los procesos temporales: el umbral vuelve a 80 °C porque el
#     arranque normal (runner/start_tmux.sh) no usa esa variable.
# Resultado en logs/server_thermal_selftest.log.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/logs/server_thermal_selftest.log"
IMAGE="${PAPER_NEW_IMAGE:-paper-new:cu126}"
DATA_DIR="${PAPER_NEW_DATA_DIR:-/home/administrador/notebooks/covid}"
JOB_ID="selftest_$(date +%Y%m%d_%H%M%S)"
CONTAINER="paper_new_job_$JOB_ID"
GUARD_PID=""
RUNNER_PID=""
RESULT="FALLO"

mkdir -p "$ROOT/logs" "$ROOT/logs/jobs"
exec > >(tee -a "$LOG") 2>&1
echo
echo "================ $(date +%Y-%m-%dT%H:%M:%S) prueba térmica del servidor ($JOB_ID) ================"

say() { echo "[selftest $(date +%H:%M:%S)] $*"; }
die() { say "ERROR: $*"; exit 1; }

cleanup() {
    local rc=$?
    say "limpieza (código $rc)"
    if [[ -n "$RUNNER_PID" ]] && kill -0 "$RUNNER_PID" 2>/dev/null; then
        kill -TERM "$RUNNER_PID" 2>/dev/null || true
        for _ in $(seq 1 60); do kill -0 "$RUNNER_PID" 2>/dev/null || break; sleep 1; done
        kill -KILL "$RUNNER_PID" 2>/dev/null || true
    fi
    if docker ps -q --filter "name=^${CONTAINER}\$" | grep -q .; then
        say "el contenedor de la prueba seguía vivo: docker stop"
        docker stop -t 30 "$CONTAINER" >/dev/null 2>&1 || true
    fi
    if [[ -n "$GUARD_PID" ]] && kill -0 "$GUARD_PID" 2>/dev/null; then
        kill -TERM "$GUARD_PID" 2>/dev/null || true
        for _ in $(seq 1 15); do kill -0 "$GUARD_PID" 2>/dev/null || break; sleep 1; done
        kill -KILL "$GUARD_PID" 2>/dev/null || true
    fi
    # El trabajo de prueba no debe quedar en la cola real
    for d in pending running; do
        if [[ -f "$ROOT/jobs/$d/$JOB_ID.json" ]]; then
            mv -- "$ROOT/jobs/$d/$JOB_ID.json" "$ROOT/jobs/failed/$JOB_ID.json"
            say "trabajo de prueba retirado de jobs/$d/ (movido a jobs/failed/)"
        fi
    done
    unset PAPER_NEW_STOP_C
    say "procesos temporales detenidos; el umbral de corte bajado ya no existe en ningún proceso (vuelve a 80 °C)"
    say "RESULTADO: $RESULT"
    if [[ "$RESULT" == "SUPERADA" ]]; then
        say "Siguiente paso: cuando todas las GPU estén < 65 °C, bash runner/clear_thermal_stop.sh y después bash runner/start_tmux.sh"
    fi
}
trap cleanup EXIT

# ---------------------------------------------------------------- comprobaciones
for c in docker nvidia-smi python3; do command -v "$c" >/dev/null 2>&1 || die "falta $c"; done
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "no existe la imagen $IMAGE (bash docker/build.sh)"
if python3 "$ROOT/runner/job_runner.py" --root "$ROOT" --check-guard >/dev/null 2>&1; then
    die "hay un vigilante en marcha. Detenga primero la sesión normal: bash runner/stop_all.sh"
fi
if command -v tmux >/dev/null 2>&1 && tmux has-session -t paper_new 2>/dev/null; then
    die "existe la sesión tmux paper_new. Deténgala primero: bash runner/stop_all.sh"
fi
[[ -f "$ROOT/jobs/THERMAL_STOP" ]] && die "existe jobs/THERMAL_STOP; retírelo antes con bash runner/clear_thermal_stop.sh"
if ls "$ROOT/jobs/running/"*.json >/dev/null 2>&1; then die "hay trabajos en jobs/running/"; fi
if docker ps -q --filter label=paper_new.job=1 | grep -q .; then die "hay contenedores del proyecto en marcha"; fi

# ---------------------------------------------------------------- umbral temporal
READ="$(nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits)"
say "lectura inicial: $(echo "$READ" | tr '\n' ';')"
HOT_GPU="$(echo "$READ" | sort -t, -k2 -n -r | head -1 | cut -d, -f1 | tr -d ' ')"
MAX_T="$(echo "$READ" | sort -t, -k2 -n -r | head -1 | cut -d, -f2 | tr -d ' ')"
[[ "$MAX_T" =~ ^[0-9]+$ ]] || die "temperatura no interpretable: $MAX_T"
TEST_STOP=$((MAX_T + 3))
[[ "$TEST_STOP" -lt 75 ]] || die "las GPU están demasiado calientes para la prueba ($MAX_T °C); espere a que se enfríen"
say "GPU de la prueba: $HOT_GPU; umbral de corte TEMPORAL: $TEST_STOP °C (máxima actual $MAX_T °C + 3)"
OTHER_APPS="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true)"
[[ -z "$OTHER_APPS" ]] || say "AVISO: hay otros procesos en GPU (pueden calentar y adelantar el corte): $OTHER_APPS"

# ---------------------------------------------------------------- procesos temporales
cd "$ROOT"
PAPER_NEW_STOP_C="$TEST_STOP" python3 runner/gpu_guard.py --interval 2 >> "$ROOT/logs/selftest_guard.log" 2>&1 &
GUARD_PID=$!
for _ in $(seq 1 20); do
    python3 runner/job_runner.py --root "$ROOT" --check-guard >/dev/null 2>&1 && break
    sleep 1
done
python3 runner/job_runner.py --root "$ROOT" --check-guard >/dev/null 2>&1 || die "el vigilante temporal no arrancó (ver logs/selftest_guard.log)"
HB_STOP="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["stop_c"])' "$ROOT/runner/state/guard_heartbeat.json")"
say "vigilante temporal PID $GUARD_PID con umbral de corte $HB_STOP °C"
python3 runner/job_runner.py --only-job "$JOB_ID" --exit-after-job --tick 1 \
    --image "$IMAGE" --data-dir "$DATA_DIR" >> "$ROOT/logs/selftest_runner.log" 2>&1 &
RUNNER_PID=$!
say "ejecutor temporal PID $RUNNER_PID (solo procesa $JOB_ID)"

python3 runner/submit_job.py --id "$JOB_ID" --gpu "$HOT_GPU" --prioridad 1000 --tiempo-max-min 5 --no-reencolar \
    --descripcion "Prueba térmica del servidor: carga ficticia de 2 minutos" \
    -- python tests/dummy_job.py --gpu-load --seconds 120 >/dev/null
say "trabajo encolado; esperando el corte (máximo 180 s)"

T0=$(date +%s)
LAUNCHED=0; STOP_AT=""
while (( $(date +%s) - T0 < 180 )); do
    if [[ $LAUNCHED -eq 0 ]] && docker ps -q --filter "name=^${CONTAINER}\$" | grep -q .; then
        LAUNCHED=1; say "contenedor $CONTAINER en marcha"
    fi
    if [[ -f "$ROOT/jobs/THERMAL_STOP" && -z "$STOP_AT" ]]; then
        STOP_AT=$(date +%s); say "jobs/THERMAL_STOP creado a los $((STOP_AT - T0)) s"
    fi
    if [[ -n "$STOP_AT" ]] && ! kill -0 "$RUNNER_PID" 2>/dev/null; then break; fi
    if [[ -f "$ROOT/jobs/failed/$JOB_ID.json" || -f "$ROOT/jobs/done/$JOB_ID.json" ]] && [[ -z "$STOP_AT" ]]; then
        break
    fi
    sleep 2
done
sleep 3

# ---------------------------------------------------------------- verificación
OK=1
check() { if eval "$2"; then say "OK    $1"; else say "FALLO $1"; OK=0; fi; }
check "el contenedor llegó a lanzarse" "[[ $LAUNCHED -eq 1 ]]"
check "se creó jobs/THERMAL_STOP" "[[ -f '$ROOT/jobs/THERMAL_STOP' ]]"
check "el trabajo escribió 'checkpoint guardado'" "grep -q 'checkpoint guardado' '$ROOT/logs/jobs/$JOB_ID.log'"
check "el contenedor ya no existe" "! docker ps -a -q --filter 'name=^${CONTAINER}\$' | grep -q ."
check "el evento quedó en thermal_events.log" "grep -q 'umbral $TEST_STOP' '$ROOT/logs/thermal_events.log'"
check "el trabajo no terminó como done" "[[ ! -f '$ROOT/jobs/done/$JOB_ID.json' ]]"
if [[ -f "$ROOT/logs/jobs/$JOB_ID.json" ]]; then
    say "resumen del trabajo: $(tr -d '\n' < "$ROOT/logs/jobs/$JOB_ID.json" | tr -s ' ')"
fi
[[ -f "$ROOT/jobs/THERMAL_STOP" ]] && say "THERMAL_STOP: $(tr '\n' ' ' < "$ROOT/jobs/THERMAL_STOP")"
if [[ $OK -eq 1 ]]; then RESULT="SUPERADA"; else RESULT="FALLO (no entrenar hasta revisar)"; exit 1; fi
