#!/usr/bin/env bash
# Funciones comunes de los scripts de descarga de paper_new. Se carga con: source _common.sh
set -euo pipefail

RAW_ROOT="${RAW_ROOT:-/home/administrador/Proyectos/paper_new/data/raw}"
LOG_DIR="${LOG_DIR:-/home/administrador/Proyectos/paper_new/logs/download}"
mkdir -p "$RAW_ROOT" "$LOG_DIR"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/${DATASET:-general}.log" >&2; }
die() { log "ERROR: $*"; exit 1; }
# Cualquier aborto por "set -e" deja constancia en el log (antes el script moria en silencio).
# Solo actua con errexit activo, para no ensuciar los scripts que gestionan sus errores con "set +e".
set -E
trap 'rc=$?; if [[ $- == *e* ]]; then log "ERROR: orden fallida (codigo $rc) en ${BASH_SOURCE[0]##*/}, linea $LINENO: $BASH_COMMAND"; fi' ERR

# check_free_gb <dir> <gb_necesarios>: aborta si no hay espacio (margen extra FREE_MARGIN_GB, 10 GB por defecto).
check_free_gb() {
  local dir="$1" need="$2" avail margin="${FREE_MARGIN_GB:-10}"
  mkdir -p "$dir"
  avail=$(df -B1G --output=avail "$dir" | tail -1 | tr -d ' ')
  log "espacio libre en $(df --output=target "$dir" | tail -1): ${avail} GB; necesarios ${need} GB + ${margin} GB de margen"
  if (( avail < need + margin )); then die "espacio insuficiente"; fi
}

# done_marker: idempotencia. Si existe .done con la misma huella, el script sale sin hacer nada.
is_done() { [[ -f "$1/.done" ]]; }
mark_done() { date '+%F %T' > "$1/.done"; log "completado: $1"; }

# check_kaggle_credentials: valida la CLI y las credenciales sin abortar. Devuelve 0 si son utilizables;
# si no, deja el motivo y el remedio en KAGGLE_CRED_MSG y devuelve 1. Admite:
#   - KAGGLE_USERNAME y KAGGLE_KEY en el entorno;
#   - KAGGLE_API_TOKEN en el entorno o el fichero ~/.kaggle/access_token (tokens de acceso de versiones
#     recientes de la CLI);
#   - kaggle.json en $KAGGLE_CONFIG_DIR o en ~/.kaggle, con los campos "username" y "key".
check_kaggle_credentials() {
  KAGGLE_CRED_MSG=""
  if ! command -v kaggle >/dev/null 2>&1; then
    KAGGLE_CRED_MSG="falta la CLI de Kaggle. Remedio: pip install --user kaggle (y anadir ~/.local/bin al PATH)"
    return 1
  fi
  if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then return 0; fi
  if [[ -n "${KAGGLE_API_TOKEN:-}" ]]; then return 0; fi
  local dir="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}"
  local f="$dir/kaggle.json"
  if [[ -f "$f" ]]; then
    local res
    res=$(python3 - "$f" <<'PY' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("json"); sys.exit(0)
if not isinstance(d, dict):
    print("json"); sys.exit(0)
falta = [k for k in ("username", "key") if not str(d.get(k, "")).strip()]
print(",".join(falta) if falta else "ok")
PY
)
    case "$res" in
      ok)
        local perm; perm=$(stat -c %a "$f" 2>/dev/null || echo "")
        if [[ -n "$perm" && "$perm" != "600" && "$perm" != "400" ]]; then
          log "AVISO: $f tiene permisos $perm; conviene chmod 600 \"$f\""
        fi
        return 0 ;;
      json)
        KAGGLE_CRED_MSG="$f no es un JSON valido. Remedio: descargue de nuevo el token en Kaggle (Settings > API > Create New Token) y guardelo tal cual en $f con chmod 600"
        return 1 ;;
      *)
        KAGGLE_CRED_MSG="a $f le falta el campo o los campos: $res. Remedio: el fichero debe ser {\"username\":\"<su usuario de Kaggle>\",\"key\":\"<clave>\"}; descargue de nuevo el token clasico en Kaggle (Settings > API > Create New Token) o anada el campo que falta a mano y aplique chmod 600. Alternativa: exporte KAGGLE_USERNAME y KAGGLE_KEY en el entorno."
        return 1 ;;
    esac
  fi
  if [[ -s "$dir/access_token" ]]; then return 0; fi
  KAGGLE_CRED_MSG="no hay credenciales de Kaggle: falta $f. Remedio: Kaggle > Settings > API > Create New Token, guarde el fichero en $f y aplique chmod 600; o exporte KAGGLE_USERNAME y KAGGLE_KEY"
  return 1
}

require_kaggle() {
  check_kaggle_credentials || die "credenciales de Kaggle: $KAGGLE_CRED_MSG"
}

# kaggle_run <fichero_salida> <args de kaggle...>: ejecuta la CLI en primer plano con una vigilancia de inactividad
# en segundo plano, porque la CLI no tiene tiempo de espera propio y una llamada sin respuesta quedaria colgada sin
# fin. La senal de avance es la entrada y salida del propio proceso (/proc/<pid>/io, bytes leidos y escritos), ademas
# del fichero de salida y del directorio de destino (-p). El directorio de destino NO basta: la CLI baja el zip a
# una ubicacion temporal y lo mueve al final (visto el 12-11-2026 con usmanshams/tbx-11: 26 minutos sin un solo
# byte en el destino y la descarga iba bien). Si nada cambia durante KAGGLE_STALL_SEC segundos (3600 por defecto;
# KAGGLE_STALL_SEC_NOIO, 14400, si /proc no es legible), se mata la llamada y se devuelve 124. Ctrl+C sigue
# funcionando porque la CLI corre en primer plano.
kaggle_run() {
  local out="$1"; shift
  local stall="${KAGGLE_STALL_SEC:-3600}" stall_noio="${KAGGLE_STALL_SEC_NOIO:-14400}" poll="${KAGGLE_STALL_POLL:-30}" watch="" i rc=0
  local a=("$@")
  for (( i = 0; i < ${#a[@]} - 1; i++ )); do
    if [[ "${a[i]}" == "-p" || "${a[i]}" == "--path" ]]; then watch="${a[i+1]}"; fi
  done
  local pidf="$out.pid" flag="$out.stall"
  rm -f "$pidf" "$flag"
  (
    kp=""; t=0; idle=0; prev=""
    while [[ ! -s "$pidf" ]] && (( t < 40 )); do sleep 0.5; t=$(( t + 1 )); done
    kp=$(cat "$pidf" 2>/dev/null || true)
    [[ -n "$kp" ]] || exit 0
    t=0
    while kill -0 "$kp" 2>/dev/null; do
      sleep 1; t=$(( t + 1 ))
      (( t % poll == 0 )) || continue
      pio=$(LC_ALL=C awk '/^(rchar|wchar):/ {printf "%s:", $2}' "/proc/$kp/io" 2>/dev/null || true)
      cur="$(stat -c %s "$out" 2>/dev/null || echo 0):$pio"
      if [[ -n "$watch" && -d "$watch" ]]; then cur="$cur:$(du -sb "$watch" 2>/dev/null | cut -f1 || echo 0)"; fi
      if [[ "$cur" == "$prev" ]]; then idle=$(( idle + poll )); else idle=0; prev="$cur"; fi
      limit="$stall"; [[ -n "$pio" ]] || limit="$stall_noio"
      if (( idle >= limit )); then
        echo "$idle" > "$flag"
        kill "$kp" 2>/dev/null || true
        sleep 3
        kill -9 "$kp" 2>/dev/null || true
        exit 0
      fi
    done
    exit 0
  ) &
  local wd=$!
  bash -c 'echo $$ > "$0"; exec kaggle "$@"' "$pidf" "$@" > "$out" 2>&1 || rc=$?
  kill "$wd" 2>/dev/null || true
  wait "$wd" 2>/dev/null || true
  if [[ -s "$flag" ]]; then
    echo "timed out: sin avance durante $(cat "$flag") s (vigilancia de inactividad)" >> "$out"
    rc=124
  fi
  rm -f "$pidf" "$flag"
  return "$rc"
}

# kaggle_retry <fichero_salida> <args de kaggle...>: ejecuta la CLI y reintenta con espera creciente ante
# 429 (Too Many Requests), errores 5xx y cortes de red. Los demas errores (401, 403, 404...) no se reintentan.
# Variables: KAGGLE_RETRIES (6), KAGGLE_RETRY_BASE (30 s, se duplica en cada intento), KAGGLE_RETRY_MAX (600 s).
kaggle_retry() {
  local out="$1"; shift
  local n="${KAGGLE_RETRIES:-6}" wait="${KAGGLE_RETRY_BASE:-30}" maxw="${KAGGLE_RETRY_MAX:-600}" i rc stalls=0
  for (( i = 1; i <= n; i++ )); do
    rc=0
    kaggle_run "$out" "$@" || rc=$?
    if (( rc == 124 )); then
      stalls=$(( stalls + 1 ))
      if (( stalls >= ${KAGGLE_STALL_RETRIES:-2} )); then
        log "kaggle $1 $2: Kaggle no responde (llamada colgada $stalls veces); se abandona este paso"
        return 75
      fi
      log "kaggle $1 $2: llamada colgada sin avance; se reintenta (intento $i de $n)"
      sleep "$wait"
      continue
    fi
    if (( rc == 0 )) && ! grep -qiE '429 Client Error|Too Many Requests' "$out"; then return 0; fi
    if grep -qiE '429|Too Many Requests|50[0-9] Server Error|Service Unavailable|Bad Gateway|Gateway Time-?out|Connection (reset|aborted|refused)|timed out|Temporary failure' "$out"; then
      if (( i < n )); then
        log "kaggle $1 $2: error transitorio (intento $i de $n); espera ${wait} s"
        sleep "$wait"
        wait=$(( wait * 2 > maxw ? maxw : wait * 2 ))
        continue
      fi
      log "kaggle $1 $2: error transitorio persistente tras $n intentos"
      return 75
    fi
    return "$rc"
  done
  return 75
}

# pausa breve entre llamadas a Kaggle para no provocar 429 (KAGGLE_PAUSE, 10 s por defecto)
kaggle_pause() { sleep "${KAGGLE_PAUSE:-10}"; }

# bytes_of <dir>: suma de bytes de los ficheros (excluye zips residuales y marcadores)
# Con mawk (awk por defecto en Ubuntu) "print s" pasa a notacion cientifica por encima de 2^31 y, con una
# configuracion regional en espanol, usa coma decimal ("4,25208e+09"): asi fallo la verificacion de TBX11K el
# 12-11-2026 con los datos completos. Se imprime siempre el entero exacto.
bytes_of() { find "$1" -type f ! -name '*.zip' ! -name '.done' -printf '%s\n' | LC_ALL=C awk '{s+=$1} END {printf "%.0f\n", s}'; }

# verify_bytes <dir> <bytes_esperados> <tolerancia_relativa>: compara con el totalBytes publicado.
verify_bytes() {
  local dir="$1" exp="$2" tol="${3:-0.02}" got
  got=$(bytes_of "$dir")
  log "bytes extraidos: $got; esperado (publicado): $exp; tolerancia: $tol"
  if [[ "${SKIP_SIZE_CHECK:-0}" == "1" ]]; then log "AVISO: SKIP_SIZE_CHECK=1, no se compara el tamano"; return 0; fi
  python3 - "$got" "$exp" "$tol" <<'PY' || die "el tamano no coincide con lo publicado (revise; SKIP_SIZE_CHECK=1 para forzar)"
import sys
g, e, t = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
sys.exit(0 if abs(g - e) <= t * e else 1)
PY
}

# count_images <dir>: numero de imagenes (png, jpg, jpeg, dcm, dicom)
count_images() { find "$1" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.dcm' -o -iname '*.dicom' \) | wc -l; }

# kaggle_dataset <slug> <dest> <bytes_publicados> <gb_necesarios>
kaggle_dataset() {
  local slug="$1" dest="$2" exp="$3" need="$4"
  if is_done "$dest"; then log "ya descargado: $dest (borre $dest/.done para repetir)"; return 0; fi
  require_kaggle
  check_free_gb "$dest" "$need"
  log "descargando $slug en $dest"
  local tmp; tmp=$(mktemp "$LOG_DIR/.kaggle_out.XXXXXX")
  if ! kaggle_retry "$tmp" datasets download -d "$slug" -p "$dest"; then
    log "salida de la CLI:"; tail -5 "$tmp" | tee -a "$LOG_DIR/${DATASET:-general}.log" >&2
    rm -f "$tmp"; die "no se pudo descargar $slug"
  fi
  rm -f "$tmp"
  local zip; zip=$(ls -1 "$dest"/*.zip 2>/dev/null | head -1 || true)
  [[ -n "$zip" ]] || die "no se encontro el zip descargado"
  ( cd "$dest" && unzip -q -n "$zip" )
  verify_bytes "$dest" "$exp" 0.02
  sha256sum "$zip" > "$dest/zip.sha256"
  rm -f "$zip"
  log "imagenes: $(count_images "$dest")"
  mark_done "$dest"
  kaggle_pause
}
