#!/usr/bin/env bash
# Montgomery County y Shenzhen Hospital CXR Sets (NLM), descarga directa del servidor oficial:
#   https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/
# Montgomery: 138 PA (80 normales, 58 TB) con mascaras pulmonares manuales (ManualMask).
# Shenzhen: 662 (326 normales, 336 TB). El servidor no publica tamanos por carpeta; las copias de Kaggle
# miden 0.61 GB (Montgomery) y 3.77 GB (Shenzhen), que se usan como estimacion.
# El servidor publica cada carpeta como <carpeta>/index.html y responde con error a la URL de la carpeta,
# por lo que "wget -r <carpeta>/" falla en la primera peticion (asi aborto la ejecucion del 12-11-2026).
# La descarga la hace nlm_crawl.py, que recorre los index.html, reanuda parciales y reintenta.
# Uso: dl_nlm_tb.sh [montgomery|shenzhen|all]
# Variables para pruebas: NLM_BASE, NLM_EXPECT_MONTGOMERY, NLM_EXPECT_SHENZHEN, NLM_CRAWL_ARGS.
DATASET=nlm_tb; source "$(dirname "$0")/_common.sh"
H="$(cd "$(dirname "$0")" && pwd)"
BASE="${NLM_BASE:-https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets}"
which=${1:-all}

fetch() {  # fetch <nombre> <subruta del index inicial> <carpeta raiz> <gb> <png_esperados_en_CXR_png>
  local name="$1" sub="$2" root="$3" gb="$4" expn="$5" dest="$RAW_ROOT/$1" rc=0
  if is_done "$dest"; then log "ya descargado: $dest"; return 0; fi
  check_free_gb "$dest" "$gb"
  log "$name: descargando desde $BASE/$sub/index.html"
  python3 "$H/nlm_crawl.py" "$BASE/$sub/index.html" "$dest" --root-url "$BASE/$root/" ${NLM_CRAWL_ARGS:-} \
      2> >(tee -a "$LOG_DIR/$DATASET.log" >&2) || rc=$?
  local n; n=$(find "$dest" -path '*CXR_png*' -iname '*.png' | wc -l)
  log "$name: $n PNG en CXR_png (esperados $expn); codigo de nlm_crawl.py: $rc"
  (( rc == 0 )) || die "$name: la descarga no termino bien (codigo $rc); al relanzar se reanuda donde quedo"
  [[ "$n" -eq "$expn" ]] || die "$name: numero de imagenes distinto del publicado"
  find "$dest" -type f ! -name '.done' ! -name 'MD5SUMS.local' -print0 | sort -z | xargs -0 md5sum > "$dest/MD5SUMS.local"
  mark_done "$dest"
}
case "$which" in
  montgomery) fetch montgomery Montgomery-County-CXR-Set/MontgomerySet Montgomery-County-CXR-Set 2 "${NLM_EXPECT_MONTGOMERY:-138}" ;;
  shenzhen)   fetch shenzhen   Shenzhen-Hospital-CXR-Set Shenzhen-Hospital-CXR-Set 6 "${NLM_EXPECT_SHENZHEN:-662}" ;;
  all)        fetch montgomery Montgomery-County-CXR-Set/MontgomerySet Montgomery-County-CXR-Set 2 "${NLM_EXPECT_MONTGOMERY:-138}"
              fetch shenzhen   Shenzhen-Hospital-CXR-Set Shenzhen-Hospital-CXR-Set 6 "${NLM_EXPECT_SHENZHEN:-662}" ;;
  *) die "argumento no valido: $which" ;;
esac
