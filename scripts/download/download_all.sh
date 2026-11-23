#!/usr/bin/env bash
# Orquestador: ejecuta las descargas seleccionadas en orden de menor a mayor tamano. Idempotente.
# Uso: bash download_all.sh            (todo)
#      bash download_all.sh --no-vindr (todo salvo VinDr-CXR; alternativa B)
# ONLY_STEPS="paso1,paso2" ejecuta solo esos pasos (nombres de script sin .sh), util para pruebas.
set -euo pipefail
H="$(cd "$(dirname "$0")" && pwd)"
source "$H/_common.sh"
NO_VINDR=0; [[ "${1:-}" == "--no-vindr" ]] && NO_VINDR=1

want() { [[ -z "${ONLY_STEPS:-}" ]] || [[ ",${ONLY_STEPS}," == *",$1,"* ]]; }
FAILED=()
# Un paso que falla queda registrado y no impide los siguientes (los pasos son independientes entre si).
step() {
  local name="$1" rc=0; shift; want "$name" || return 0; log "== paso $name"
  bash "$H/$name.sh" "$@" || rc=$?
  if (( rc != 0 )); then log "ERROR: el paso $name fallo (codigo $rc); se continua con los siguientes"; FAILED+=("$name"); fi
  return 0
}

# 1. Listado previo: informativo, nunca detiene el orquestador
if want check_kaggle_contents; then bash "$H/check_kaggle_contents.sh" || log "AVISO: el listado previo fallo; se continua"; fi

# 2. Credenciales de Kaggle: se validan antes de cualquier descarga que las necesite
if [[ -z "${ONLY_STEPS:-}" ]]; then require_kaggle; fi

step dl_rsna_mapping
step dl_tb_rahman
step dl_covid19_radiography
step dl_covidgr
step dl_covid_qu_ex
step dl_nlm_tb all
step dl_tbx11k
step dl_chexmask
if want dl_tb_pakistan_mendeley; then
  if [[ -n "${ZIP_URL:-}${ZIP_PATH:-}" ]]; then step dl_tb_pakistan_mendeley
  else log "AVISO: Mendeley (TB Pakistan) requiere ZIP_URL o ZIP_PATH; se omite"; fi
fi
if (( NO_VINDR == 0 )); then step dl_vindr_subset; fi
if (( ${#FAILED[@]} > 0 )); then
  log "Descargas terminadas CON FALLOS en: ${FAILED[*]}. Revise logs/download/ y relance este script: los pasos completados se saltan y los parciales se reanudan."
  exit 1
fi
log "Descargas terminadas. Siguiente paso: python3 scripts/data/hash_images.py sobre cada carpeta de data/raw."
