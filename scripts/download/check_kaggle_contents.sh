#!/usr/bin/env bash
# Lista los ficheros de datasets de Kaggle ANTES de descargarlos y cuenta imagenes por carpeta, sin descargar nada.
# Por defecto solo la base de Rahman (tawsifurrahman/tuberculosis-tb-chest-xray-dataset), la unica cuyo contenido
# decide algo (variante A o B del Benchmark 2). Otros datasets solo si se pasan como argumento.
# Pausa entre paginas (KAGGLE_PAGE_PAUSE, 3 s), reintentos con espera creciente ante 429 y 5xx (ver _common.sh),
# escritura en un temporal que solo sustituye al CSV previo si el listado termina bien.
# NUNCA aborta a quien lo llama: ante cualquier fallo deja un AVISO en el log y sale con codigo 0.
# Salida: logs/kaggle_contents.log y logs/kaggle_files/<owner>_<dataset>.csv
# Uso: bash check_kaggle_contents.sh [slug ...]
DATASET=kaggle_contents; source "$(dirname "$0")/_common.sh"
set +e
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT_LOG="${OUT_LOG:-/home/administrador/Proyectos/paper_new/logs/kaggle_contents.log}"
OUT_DIR="$(dirname "$OUT_LOG")/kaggle_files"
PAGE_PAUSE="${KAGGLE_PAGE_PAUSE:-3}"
mkdir -p "$OUT_DIR"
SLUGS=("$@")
(( ${#SLUGS[@]} == 0 )) && SLUGS=(tawsifurrahman/tuberculosis-tb-chest-xray-dataset)

aviso() { echo "AVISO: $*" >> "$OUT_LOG"; log "AVISO: $*"; }
echo "# Contenido de datasets de Kaggle, $(date '+%F %T')" >> "$OUT_LOG"

if ! check_kaggle_credentials; then
  aviso "no se listan datasets: $KAGGLE_CRED_MSG"
  exit 0
fi

list_one() {
  local slug="$1" base csv tmpraw page token out
  base="$OUT_DIR/$(echo "$slug" | tr '/' '_')"
  csv="$base.csv"
  tmpraw=$(mktemp "$base.raw.XXXXXX") || return 1
  out=$(mktemp "$base.out.XXXXXX") || return 1
  token=""
  for (( page = 1; page <= 2000; page++ )); do
    if [[ -n "$token" ]]; then
      kaggle_retry "$out" datasets files "$slug" -v --page-size 200 --page-token "$token"
    else
      kaggle_retry "$out" datasets files "$slug" -v --page-size 200
    fi
    local rc=$?
    if (( rc != 0 )); then
      aviso "$slug: fallo en la pagina $page (codigo $rc): $(tail -1 "$out" | cut -c1-200). Se conserva el CSV previo si existia."
      rm -f "$tmpraw" "$out"; return 1
    fi
    grep -v -i '^next page token' "$out" >> "$tmpraw"
    token=$(grep -i '^next page token' "$out" | sed -E 's/^[Nn]ext [Pp]age [Tt]oken *= *//' | tr -d '\r' | head -1)
    [[ -z "$token" ]] && break
    sleep "$PAGE_PAUSE"
  done
  rm -f "$out"
  local tmpcsv="$base.csv.tmp"
  if python3 "$HERE/kaggle_list_parse.py" "$slug" "$tmpraw" "$tmpcsv" "$OUT_LOG"; then
    mv -f "$tmpcsv" "$csv"; rm -f "$tmpraw"
    log "$slug: listado completo ($page paginas) en $csv"
    return 0
  fi
  aviso "$slug: el listado no contiene ficheros; se conserva el CSV previo si existia"
  rm -f "$tmpraw" "$tmpcsv"; return 1
}

first=1
for slug in "${SLUGS[@]}"; do
  (( first )) || kaggle_pause
  first=0
  list_one "$slug" || true
done
log "resumen en $OUT_LOG"
exit 0
