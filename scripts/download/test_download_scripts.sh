#!/usr/bin/env bash
# Pruebas sin red de los scripts de descarga (kaggle simulado). Se ejecuta en la VM.
S="$HOME/mnt/paper_new/scripts/download"
T=${KTEST_DIR:-/tmp/ktest}  # crear antes $T/bin/kaggle desde test_fixtures/kaggle_simulado
export PATH="$T/bin:$PATH"
export KAGGLE_RETRY_BASE=1 KAGGLE_RETRY_MAX=2 KAGGLE_RETRIES=4 KAGGLE_PAUSE=0 KAGGLE_PAGE_PAUSE=0
pass=0; fail=0
ok()  { echo "  RESULTADO: OK  $*"; pass=$((pass+1)); }
bad() { echo "  RESULTADO: FALLO  $*"; fail=$((fail+1)); }
fresh() {  # HOME, logs y raw temporales; credenciales validas salvo que se indique
  rm -rf "$T/home" "$T/logs" "$T/raw" "$T/state" "$T/calls.log"
  mkdir -p "$T/home/.kaggle" "$T/logs" "$T/raw"
  export HOME="$T/home" LOG_DIR="$T/logs/download" RAW_ROOT="$T/raw" OUT_LOG="$T/logs/kaggle_contents.log"
  echo '{"username":"prueba","key":"0123456789abcdef"}' > "$HOME/.kaggle/kaggle.json"; chmod 600 "$HOME/.kaggle/kaggle.json"
  unset KAGGLE_USERNAME KAGGLE_KEY KAGGLE_API_TOKEN KAGGLE_CONFIG_DIR
}
echo "== Pruebas de scripts/download con kaggle simulado ($(date '+%F %T'))"
echo "which kaggle: $(command -v kaggle)"

echo; echo "-- Prueba 1: 429 dos veces y luego respuesta (listado)"
fresh; export FAKE_MODE=fail429x2
bash "$S/check_kaggle_contents.sh"; rc=$?
echo "  codigo de salida: $rc; llamadas a kaggle: $(wc -l < $T/calls.log)"
grep -h "error transitorio" "$T/logs/download/kaggle_contents.log"
cat "$OUT_LOG"
if [[ $rc == 0 && $(wc -l < $T/calls.log) == 3 ]] && grep -q "imagenes: 4" "$OUT_LOG"; then ok "reintento y exito tras 2 errores 429"; else bad "prueba 1"; fi

echo; echo "-- Prueba 2: fallo persistente; el listado no aborta y conserva el CSV previo"
fresh; mkdir -p "$T/logs/kaggle_files"
echo "name,size,creationDate" > "$T/logs/kaggle_files/tawsifurrahman_tuberculosis-tb-chest-xray-dataset.csv"
echo "previo/valido.png,1,x" >> "$T/logs/kaggle_files/tawsifurrahman_tuberculosis-tb-chest-xray-dataset.csv"
export FAKE_MODE=alwaysfail
bash "$S/check_kaggle_contents.sh"; rc=$?
echo "  codigo de salida de check_kaggle_contents.sh: $rc; llamadas: $(wc -l < $T/calls.log)"
grep -h AVISO "$OUT_LOG"
prev=$(tail -1 "$T/logs/kaggle_files/tawsifurrahman_tuberculosis-tb-chest-xray-dataset.csv")
left=$(ls "$T/logs/kaggle_files" | grep -c -E 'raw|out|tmp')
echo "  CSV previo: $prev; temporales sobrantes: $left"
ONLY_STEPS=check_kaggle_contents bash "$S/download_all.sh" > "$T/da.out" 2>&1; rc2=$?
echo "  codigo de salida de download_all.sh (ONLY_STEPS=check_kaggle_contents): $rc2"
tail -2 "$T/da.out"
if [[ $rc == 0 && $rc2 == 0 && "$prev" == "previo/valido.png,1,x" && $left == 0 ]] && grep -q AVISO "$OUT_LOG"; then ok "fallo persistente sin abortar, CSV previo intacto, sin temporales"; else bad "prueba 2"; fi

echo; echo "-- Prueba 3: paginacion completa (3 paginas con Next Page Token)"
fresh; export FAKE_MODE=paginate
bash "$S/check_kaggle_contents.sh"; rc=$?
csv="$T/logs/kaggle_files/tawsifurrahman_tuberculosis-tb-chest-xray-dataset.csv"
echo "  codigo: $rc; llamadas: $(wc -l < $T/calls.log); filas del CSV: $(( $(wc -l < $csv) - 1 ))"
grep -h "page-token" $T/calls.log | sed 's/^/  /'
cat "$OUT_LOG"
if [[ $rc == 0 && $(wc -l < $T/calls.log) == 3 && $(( $(wc -l < $csv) - 1 )) == 13 ]] && grep -q "Normal: 9 imagenes" "$OUT_LOG" && grep -q "Tuberculosis: 3 imagenes" "$OUT_LOG"; then ok "las 3 paginas se recorren y se cuentan 9 + 3 imagenes"; else bad "prueba 3"; fi

echo; echo "-- Prueba 4: kaggle.json sin username"
fresh; export FAKE_MODE=ok
echo '{"key":"0123456789abcdef"}' > "$HOME/.kaggle/kaggle.json"
bash "$S/check_kaggle_contents.sh"; rc=$?
echo "  check_kaggle_contents.sh codigo: $rc; llamadas a kaggle: $( [[ -f $T/calls.log ]] && wc -l < $T/calls.log || echo 0)"
grep -h AVISO "$OUT_LOG"
bash "$S/dl_tb_rahman.sh" > "$T/dl.out" 2>&1; rc2=$?
echo "  dl_tb_rahman.sh codigo: $rc2; mensaje:"; sed 's/^/    /' "$T/dl.out"
bash "$S/download_all.sh" > "$T/da.out" 2>&1; rc3=$?
echo "  download_all.sh codigo: $rc3 (debe detenerse antes de descargar); ultima linea: $(tail -1 $T/da.out)"
if [[ $rc == 0 && $rc2 != 0 && $rc3 != 0 ]] && grep -q "le falta el campo o los campos: username" "$T/dl.out" && ! grep -q Traceback "$T/dl.out" && [[ ! -f $T/calls.log ]]; then ok "credenciales incompletas detectadas con mensaje claro, sin traza y sin llamar a la CLI"; else bad "prueba 4"; fi

echo; echo "-- Prueba 5: kaggle.json no es JSON; y credenciales por variables de entorno"
fresh; echo 'esto no es json' > "$HOME/.kaggle/kaggle.json"
bash "$S/dl_tb_rahman.sh" > "$T/dl.out" 2>&1; rc=$?
sed 's/^/    /' "$T/dl.out" | tail -1
rm -f "$HOME/.kaggle/kaggle.json"; export KAGGLE_USERNAME=prueba KAGGLE_KEY=abc FAKE_MODE=ok
bash "$S/check_kaggle_contents.sh" >/dev/null 2>&1; rc2=$?
if [[ $rc != 0 ]] && grep -q "no es un JSON valido" "$T/dl.out" && grep -q "imagenes: 4" "$OUT_LOG"; then ok "JSON invalido detectado; variables de entorno aceptadas"; else bad "prueba 5"; fi
unset KAGGLE_USERNAME KAGGLE_KEY

echo; echo "-- Prueba 6: descarga de Kaggle con 429 dos veces y luego exito (dl_tb_rahman.sh)"
fresh; export FAKE_MODE=fail429x2 SKIP_SIZE_CHECK=1 FREE_MARGIN_GB=0
bash "$S/dl_tb_rahman.sh" > "$T/dl.out" 2>&1; rc=$?
grep -E "transitorio|imagenes|completado|AVISO" "$T/dl.out" | sed 's/^/    /'
echo "  codigo: $rc; .done: $( [[ -f $T/raw/tb_rahman/.done ]] && echo si || echo no ); zip borrado: $( ls $T/raw/tb_rahman/*.zip >/dev/null 2>&1 && echo no || echo si )"
bash "$S/dl_tb_rahman.sh" > "$T/dl2.out" 2>&1
if [[ $rc == 0 && -f $T/raw/tb_rahman/.done ]] && grep -q "ya descargado" "$T/dl2.out"; then ok "reintento en la descarga, verificacion e idempotencia"; else bad "prueba 6"; fi
unset SKIP_SIZE_CHECK FREE_MARGIN_GB

echo; echo "-- Prueba 7: reprocesar el CSV real del servidor"
R="$HOME_REAL/mnt/paper_new/logs/kaggle_files/tawsifurrahman_tuberculosis-tb-chest-xray-dataset.csv"
python3 "$S/kaggle_list_parse.py" tawsifurrahman/tuberculosis-tb-chest-xray-dataset "$R" - "$T/real.log"
if grep -q "Normal: 3500 imagenes" "$T/real.log" && grep -q "Tuberculosis: 700 imagenes" "$T/real.log"; then ok "el CSV real da 3 500 normales y 700 de TB (4 200 imagenes)"; else bad "prueba 7"; fi

echo; echo "RESUMEN: $pass pruebas OK, $fail fallidas"
