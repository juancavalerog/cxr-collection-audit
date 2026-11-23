#!/usr/bin/env bash
# COVIDGR 1.0 (Tabik et al.), Hospital Universitario Clinico San Cecilio, Granada. 852 PA: 426 positivas, 426 negativas.
# Origen oficial: https://github.com/ari-dasci/OD-covidgr/releases/tag/1.0 (tamano del zip verificado con la API de GitHub).
DATASET=covidgr; source "$(dirname "$0")/_common.sh"
DEST="$RAW_ROOT/covidgr"
BASE=https://github.com/ari-dasci/OD-covidgr/releases/download/1.0
ZIP_BYTES=995465872
if is_done "$DEST"; then log "ya descargado: $DEST"; exit 0; fi
check_free_gb "$DEST" 3
cd "$DEST"
for f in COVIDGR_1.0.zip CrossValidations.zip severity.csv; do
  curl -L --fail --retry 5 -C - -o "$f" "$BASE/$f"
done
got=$(stat -c %s COVIDGR_1.0.zip)
[[ "$got" == "$ZIP_BYTES" ]] || die "COVIDGR_1.0.zip mide $got bytes, se esperaban $ZIP_BYTES"
sha256sum COVIDGR_1.0.zip CrossValidations.zip severity.csv > SHA256SUMS.local
unzip -q -n COVIDGR_1.0.zip && unzip -q -n CrossValidations.zip
n=$(count_images "$DEST"); log "imagenes: $n (esperadas 852)"
[[ "$n" -ge 852 ]] || die "faltan imagenes"
rm -f COVIDGR_1.0.zip
mark_done "$DEST"
