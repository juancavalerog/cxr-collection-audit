#!/usr/bin/env bash
# Retira jobs/THERMAL_STOP tras una parada térmica. Solo lo ejecuta el usuario.
# Exige que TODAS las GPU estén por debajo de 65 °C y pide confirmación escribiendo SI.
# El fichero no se borra: se archiva en logs/thermal_stop_archive/.
# Pruebas: PAPER_NEW_FAKE_SMI=fichero hace que la lectura salga de ese fichero en vez de nvidia-smi.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STOP_FILE="$ROOT/jobs/THERMAL_STOP"
LIMIT_C=65
ARCHIVE="$ROOT/logs/thermal_stop_archive"
EVENTS="$ROOT/logs/thermal_events.log"

if [[ ! -f "$STOP_FILE" ]]; then
    echo "No existe jobs/THERMAL_STOP: la cola no está bloqueada por temperatura."
    exit 0
fi

echo "Contenido de jobs/THERMAL_STOP:"
sed 's/^/    /' "$STOP_FILE"
echo

SMI_ARGS=()
if [[ -n "${PAPER_NEW_FAKE_SMI:-}" ]]; then
    SMI_ARGS=(--fake-smi "$PAPER_NEW_FAKE_SMI")
fi

set +e
TEMPS="$(python3 "$ROOT/runner/gpu_guard.py" --check-below "$LIMIT_C" ${SMI_ARGS[@]+"${SMI_ARGS[@]}"} 2>&1)"
RC=$?
set -e
echo "Lectura actual:"
echo "$TEMPS" | sed 's/^/    /'
if [[ $RC -eq 2 ]]; then
    echo "NO SE RETIRA: no se pudo leer la temperatura de las GPU." >&2
    exit 2
elif [[ $RC -ne 0 ]]; then
    echo "NO SE RETIRA: alguna GPU está a ${LIMIT_C} °C o más. Espere a que se enfríen y vuelva a intentarlo." >&2
    exit 3
fi

echo "Todas las GPU están por debajo de ${LIMIT_C} °C."

# Vigilante del anfitrión (17-11-2026): la CPU también debe haberse enfriado (por debajo de 70 °C).
CPU_LIMIT_C=70
if [[ -z "${PAPER_NEW_FAKE_SMI:-}" && -f "$ROOT/runner/host_guard.py" ]]; then
    set +e
    CPU_TEMPS="$(python3 "$ROOT/runner/host_guard.py" --check-below "$CPU_LIMIT_C" 2>&1)"
    CPU_RC=$?
    set -e
    echo "$CPU_TEMPS" | sed 's/^/    /'
    if [[ $CPU_RC -eq 3 ]]; then
        echo "NO SE RETIRA: la CPU está a ${CPU_LIMIT_C} °C o más. Espere a que se enfríe y vuelva a intentarlo." >&2
        exit 3
    fi
fi
echo "Al retirar THERMAL_STOP, el ejecutor volverá a lanzar los trabajos de jobs/pending/,"
echo "incluidos los interrumpidos por temperatura (se reanudan desde su checkpoint)."
printf 'Escriba SI para retirar THERMAL_STOP: '
ANSWER=""
if ! read -r ANSWER; then
    ANSWER=""
fi
if [[ "$ANSWER" != "SI" ]]; then
    echo "Cancelado: THERMAL_STOP sigue en su sitio."
    exit 1
fi

mkdir -p "$ARCHIVE"
DEST="$ARCHIVE/THERMAL_STOP_$(date +%Y%m%d_%H%M%S).txt"
mv -- "$STOP_FILE" "$DEST"
printf '%s [LIBERADO] el usuario retiró jobs/THERMAL_STOP (archivado en %s). Lectura: %s\n' \
    "$(date +%Y-%m-%dT%H:%M:%S)" "logs/thermal_stop_archive/$(basename "$DEST")" "$(echo "$TEMPS" | tr '\n' ' ')" >> "$EVENTS"
echo "THERMAL_STOP retirado y archivado en $DEST"
