#!/usr/bin/env bash
# VinDr-CXR, subconjunto del train de la competicion oficial de Kaggle
#   https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection
# Uso solo para el benchmark controlado: derrame pleural, cardiomegalia y "No finding". La version de Kaggle no
# trae la etiqueta Tuberculosis (es global y solo esta en PhysioNet, con acreditacion).
# Requisitos: ~/.kaggle/kaggle.json y haber aceptado las reglas de la competicion en la web con esa cuenta,
# confirmando que permiten investigacion academica con publicacion (si no, alternativa B: PadChest).
# No se usan copias redimensionadas de terceros: el acuerdo de uso de VinDr (vindr.ai) prohibe redistribuir.
# Las condiciones de uso de los datos de la competicion no se pudieron leer (pagina dinamica): el usuario debe
# revisarlas antes de ejecutar (docs/02_datos.md Section 5).
# El tamano por DICOM no esta publicado: el script mide el acumulado y se detiene al llegar a MAX_GB.
# Uso: MAX_GB=45 N_NOFINDING=3000 JOBS=2 dl_vindr_subset.sh
DATASET=vindr_cxr; source "$(dirname "$0")/_common.sh"
COMP=vinbigdata-chest-xray-abnormalities-detection
DEST="$RAW_ROOT/vindr_cxr"
MAX_GB=${MAX_GB:-45}
N_NOFINDING=${N_NOFINDING:-3000}
JOBS=${JOBS:-2}
HERE="$(cd "$(dirname "$0")" && pwd)"
if is_done "$DEST"; then log "ya descargado: $DEST"; exit 0; fi
require_kaggle
check_free_gb "$DEST" "$MAX_GB"
mkdir -p "$DEST/train"
cd "$DEST"
if [[ ! -s train.csv ]]; then
  t=$(mktemp "$LOG_DIR/.kaggle_out.XXXXXX")
  kaggle_retry "$t" competitions download -c "$COMP" -f train.csv -p "$DEST" \
    || { tail -5 "$t" >&2; rm -f "$t"; die "no se pudo descargar train.csv (acepto las reglas de la competicion?)"; }
  rm -f "$t"
  [[ -f train.csv.zip ]] && unzip -q -o train.csv.zip && rm -f train.csv.zip
fi
[[ -s train.csv ]] || die "no se obtuvo train.csv (acepto las reglas de la competicion?)"
python3 "$HERE/vindr_select.py" --train-csv train.csv --n-nofinding "$N_NOFINDING" --out selected_ids.txt \
  | tee -a "$LOG_DIR/$DATASET.log"

fetch_one() {
  # descarga un DICOM con reintentos y espera creciente ante 429 o 5xx
  local id="$1" f="train/$1.dicom" i wait="${KAGGLE_RETRY_BASE:-30}" out
  [[ -s "$f" ]] && return 0
  for (( i = 1; i <= ${KAGGLE_RETRIES:-6}; i++ )); do
    out=$(kaggle competitions download -c "$COMP" -f "train/$id.dicom" -p train -q 2>&1) && break
    if printf '%s' "$out" | grep -qiE '429|Too Many Requests|50[0-9] Server Error|timed out|Connection'; then
      sleep "$wait"; wait=$(( wait * 2 > ${KAGGLE_RETRY_MAX:-600} ? ${KAGGLE_RETRY_MAX:-600} : wait * 2 ))
    else
      return 1
    fi
  done
  if [[ -f "train/$id.dicom.zip" ]]; then unzip -q -o "train/$id.dicom.zip" -d train && rm -f "train/$id.dicom.zip"; fi
  [[ -s "$f" ]]
}
export -f fetch_one; export COMP KAGGLE_RETRIES KAGGLE_RETRY_BASE KAGGLE_RETRY_MAX

total=$(wc -l < selected_ids.txt); batch=200; i=0
while (( i < total )); do
  used=$(du -s -B1G train | cut -f1)
  if (( used >= MAX_GB )); then log "limite de ${MAX_GB} GB alcanzado tras $i de $total"; break; fi
  sed -n "$((i+1)),$((i+batch))p" selected_ids.txt | xargs -P "$JOBS" -I{} bash -c 'fetch_one "$@" || echo "fallo $1"' _ {} \
    | tee -a "$LOG_DIR/$DATASET.failed"
  i=$((i+batch))
  log "progreso: $(ls train | grep -c '\.dicom$') de $total ficheros; $(du -s -B1M train | cut -f1) MB"
done
n=$(ls train | grep -c '\.dicom$' || true)
log "descargados $n DICOM; media $(du -s -B1M train | LC_ALL=C awk -v n="$n" '{printf "%.1f", $1/(n>0?n:1)}') MB por fichero"
( cd train && ls *.dicom | sort | xargs md5sum ) > MD5SUMS.local
if (( n >= total )); then mark_done "$DEST"; else log "incompleto: relance para continuar (es reanudable)"; fi
