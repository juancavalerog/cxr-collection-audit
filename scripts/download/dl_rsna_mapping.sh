#!/usr/bin/env bash
# Tabla oficial de correspondencia RSNA Pneumonia Challenge 2018 -> NIH ChestX-ray14, publicada por RSNA:
#   https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018
# Sirve para contrastar el enlace por hash data/manifests/covidx_rsna_to_nih_map.csv. Fichero pequeno (tamano no publicado).
DATASET=rsna_mapping; source "$(dirname "$0")/_common.sh"
DEST="$RAW_ROOT/rsna_pneumonia_mapping"
URL=https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-mappings_2018.json
if is_done "$DEST"; then log "ya descargado: $DEST"; exit 0; fi
check_free_gb "$DEST" 1
curl -L --fail --retry 5 -o "$DEST/pneumonia-challenge-dataset-mappings_2018.json" "$URL"
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('entradas:', len(d))" \
  "$DEST/pneumonia-challenge-dataset-mappings_2018.json" | tee -a "$LOG_DIR/$DATASET.log"
sha256sum "$DEST/pneumonia-challenge-dataset-mappings_2018.json" > "$DEST/SHA256SUMS.local"
mark_done "$DEST"
