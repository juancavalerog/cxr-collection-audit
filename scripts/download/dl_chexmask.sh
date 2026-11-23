#!/usr/bin/env bash
# CheXmask 1.0.0 (PhysioNet, acceso abierto, CC BY 4.0; no requiere credenciales):
#   https://physionet.org/content/chexmask-cxr-segmentation-data/1.0.0/
# Se descargan solo las mascaras de las fuentes del proyecto, en resolucion original:
#   ChestX-Ray8.csv 2.0 GB, CheXpert.csv 8.8 GB, VinDr-CXR.csv 0.9 GB (tamanos de la pagina oficial).
# Padchest.csv (4.3 GB) solo si se pasa --padchest. Se verifica con SHA256SUMS.txt oficial.
DATASET=chexmask; source "$(dirname "$0")/_common.sh"
BASE="${CHEXMASK_BASE:-https://physionet.org/files/chexmask-cxr-segmentation-data/1.0.0}"
DEST="$RAW_ROOT/chexmask"
FILES=(OriginalResolution/ChestX-Ray8.csv OriginalResolution/CheXpert.csv OriginalResolution/VinDr-CXR.csv)
NEED=13
if [[ "${1:-}" == "--padchest" ]]; then FILES+=(OriginalResolution/Padchest.csv); NEED=18; fi
if is_done "$DEST" && [[ "${1:-}" != "--padchest" ]]; then log "ya descargado: $DEST"; exit 0; fi
check_free_gb "$DEST" "$NEED"
cd "$DEST"
for f in README.md DATA_DICTIONARY.md LICENSE.txt SHA256SUMS.txt "${FILES[@]}"; do
  mkdir -p "$(dirname "$f")"
  wget -c -nv --tries=8 --timeout=60 --waitretry=15 --retry-connrefused -O "$f" "$BASE/$f"
done
for f in "${FILES[@]}"; do
  exp=$(grep -F " $f" SHA256SUMS.txt | awk '{print $1}' | head -1 || true)
  if [[ -z "$exp" ]]; then exp=$(grep -F "$(basename "$f")" SHA256SUMS.txt | grep -F "$(dirname "$f")" | awk '{print $1}' | head -1 || true); fi
  [[ -n "$exp" ]] || die "no se encontro la suma SHA256 de $f en SHA256SUMS.txt"
  got=$(sha256sum "$f" | awk '{print $1}')
  [[ "$got" == "$exp" ]] || die "SHA256 distinto en $f"
  log "SHA256 correcto: $f"
done
mark_done "$DEST"
