#!/usr/bin/env bash
# "Dataset of Tuberculosis Chest X-rays Images" (Kiran y Jabeen), Mendeley Data, version 2, CC BY 4.0:
#   https://data.mendeley.com/datasets/8j2g3csprk/2   (DOI 10.17632/8j2g3csprk.2)
# 2 494 TB y 514 normales de un hospital de Pakistan. Test externo de TB del Benchmark 2.
# La pagina no publica un enlace directo ni el tamano: copie la URL del boton "Download All" en ZIP_URL,
# o descargue el zip a mano y pase su ruta en ZIP_PATH. Se reservan 3 GB (tamano no verificado).
# Uso: ZIP_URL='<enlace de Download All>' bash dl_tb_pakistan_mendeley.sh
#      ZIP_PATH=/ruta/al/zip bash dl_tb_pakistan_mendeley.sh
DATASET=tb_pakistan_mendeley; source "$(dirname "$0")/_common.sh"
DEST="$RAW_ROOT/tb_pakistan_mendeley"
if is_done "$DEST"; then log "ya descargado: $DEST"; exit 0; fi
check_free_gb "$DEST" 3
cd "$DEST"
if [[ -n "${ZIP_PATH:-}" ]]; then
  cp -n "$ZIP_PATH" mendeley_8j2g3csprk_v2.zip
elif [[ -n "${ZIP_URL:-}" ]]; then
  curl -L --fail --retry 5 -C - -o mendeley_8j2g3csprk_v2.zip "$ZIP_URL"
else
  die "defina ZIP_URL o ZIP_PATH (ver cabecera)"
fi
sha256sum mendeley_8j2g3csprk_v2.zip > zip.sha256
unzip -q -n mendeley_8j2g3csprk_v2.zip
# los zips de Mendeley pueden contener zips internos por carpeta
find . -mindepth 2 -name '*.zip' -print0 | while IFS= read -r -d '' z; do (cd "$(dirname "$z")" && unzip -q -n "$(basename "$z")"); done
n=$(count_images "$DEST"); log "imagenes: $n (publicadas 3 008 = 2 494 TB + 514 normales)"
find "$DEST" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) -printf '%h\n' | sort | uniq -c | tee -a "$LOG_DIR/$DATASET.log"
[[ "$n" -ge 3008 ]] || die "faltan imagenes respecto a lo publicado"
rm -f mendeley_8j2g3csprk_v2.zip
mark_done "$DEST"
