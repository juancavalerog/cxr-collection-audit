#!/usr/bin/env bash
# Pruebas sin red de la vigilancia de inactividad de las llamadas a Kaggle (kaggle_run en _common.sh).
# Usa el kaggle simulado (modos hang, hangx1 y slow). Se ejecuta en la VM o en el servidor:
#   bash scripts/download/test_kaggle_stall.sh
S="$(cd "$(dirname "$0")" && pwd)"
T=/tmp/ktest   # el kaggle simulado escribe su registro de llamadas en esta ruta fija
rm -rf "$T"; mkdir -p "$T/bin"; cp "$S/test_fixtures/kaggle_simulado" "$T/bin/kaggle"; chmod +x "$T/bin/kaggle"
export PATH="$T/bin:$PATH"
export KAGGLE_RETRY_BASE=1 KAGGLE_RETRY_MAX=2 KAGGLE_RETRIES=4 KAGGLE_PAUSE=0 KAGGLE_STALL_SEC=4 KAGGLE_STALL_POLL=2
export SKIP_SIZE_CHECK=1 FREE_MARGIN_GB=-1000
pass=0; fail=0
ok()  { echo "  RESULTADO: OK  $*"; pass=$((pass+1)); }
bad() { echo "  RESULTADO: FALLO  $*"; fail=$((fail+1)); }
fresh() {
  rm -rf "$T/home" "$T/logs" "$T/raw" "$T/state" "$T/calls.log"; mkdir -p "$T/home/.kaggle" "$T/logs" "$T/raw"
  export HOME="$T/home" LOG_DIR="$T/logs/download" RAW_ROOT="$T/raw"
  echo '{"username":"prueba","key":"0123456789abcdef"}' > "$HOME/.kaggle/kaggle.json"; chmod 600 "$HOME/.kaggle/kaggle.json"
}
colgados() { pgrep -x -a sleep | grep -c " 600$"; }   # solo procesos sleep reales, no ordenes que contengan ese texto
echo "== Pruebas de la vigilancia de inactividad de Kaggle ($(date '+%F %T'))"

echo; echo "-- Prueba 1: la CLI queda colgada siempre; se mata, se reintenta una vez y el paso falla con mensaje"
fresh; export FAKE_MODE=hang; t0=$SECONDS
bash "$S/dl_tb_rahman.sh" > "$T/p1.out" 2>&1; rc=$?; dt=$((SECONDS-t0))
grep -h -E "colgada|no responde|ERROR" "$T/logs/download/tb_rahman.log" | sed 's/^/    /' | cut -c1-200
echo "  codigo: $rc; segundos: $dt; llamadas a kaggle: $(wc -l < "$T/calls.log"); procesos colgados restantes: $(colgados)"
if [[ $rc != 0 && $dt -lt 60 && $(wc -l < "$T/calls.log") == 2 && $(colgados) == 0 && ! -f "$T/raw/tb_rahman/.done" ]] \
   && grep -q "Kaggle no responde" "$T/logs/download/tb_rahman.log" && ! ls "$T"/logs/download/.kaggle_out.* >/dev/null 2>&1; then
  ok "llamada colgada detectada y abandonada sin procesos ni temporales sueltos"; else bad "prueba 1"; tail -5 "$T/p1.out"; ls -la "$T/logs/download"; fi

echo; echo "-- Prueba 2: la primera llamada se cuelga y la segunda responde"
fresh; export FAKE_MODE=hangx1
bash "$S/dl_tb_rahman.sh" > "$T/p2.out" 2>&1; rc=$?
grep -h -E "colgada|completado" "$T/logs/download/tb_rahman.log" | sed 's/^/    /' | cut -c1-200
if [[ $rc == 0 && -f "$T/raw/tb_rahman/.done" && $(colgados) == 0 ]] && grep -q "se reintenta" "$T/logs/download/tb_rahman.log"; then
  ok "reintento tras la llamada colgada"; else bad "prueba 2"; tail -5 "$T/p2.out"; fi

echo; echo "-- Prueba 3: una descarga lenta que avanza no se interrumpe (8 s de escritura con limite de 4 s sin avance)"
fresh; export FAKE_MODE=slow; t0=$SECONDS
bash "$S/dl_tb_rahman.sh" > "$T/p3.out" 2>&1; rc=$?; dt=$((SECONDS-t0))
echo "  codigo: $rc; segundos: $dt; llamadas: $(wc -l < "$T/calls.log")"
if [[ $rc == 0 && -f "$T/raw/tb_rahman/.done" && $(wc -l < "$T/calls.log") == 1 ]] && ! grep -q "colgada" "$T/logs/download/tb_rahman.log"; then
  ok "sin falsos positivos"; else bad "prueba 3"; tail -5 "$T/p3.out"; fi

echo; echo "-- Prueba 3b: descarga a una ubicacion oculta (el destino no cambia durante 8 s); la entrada y salida del proceso cuenta como avance"
fresh; export FAKE_MODE=slowhidden; rm -f "$T/oculto.bin"; t0=$SECONDS
bash "$S/dl_tb_rahman.sh" > "$T/p3b.out" 2>&1; rc=$?; dt=$((SECONDS-t0))
echo "  codigo: $rc; segundos: $dt; llamadas: $(wc -l < "$T/calls.log"); /proc/self/io legible: $( [[ -r /proc/self/io ]] && echo si || echo no )"
if [[ $rc == 0 && -f "$T/raw/tb_rahman/.done" && $(wc -l < "$T/calls.log") == 1 ]] && ! grep -q "colgada" "$T/logs/download/tb_rahman.log"; then
  ok "no se interrumpe una descarga que avanza fuera del destino"; else bad "prueba 3b"; tail -5 "$T/p3b.out"; fi

echo; echo "-- Prueba 3c: suma de bytes por encima de 2^31 (caso de TBX11K) y verificacion de tamano"
fresh; d="$T/grande"; mkdir -p "$d"; truncate -s 3000000000 "$d/a.bin"; truncate -s 1252079748 "$d/b.bin"
got=$(LANG=es_PE.UTF-8 LC_NUMERIC=es_ES.UTF-8 bash -c "source '$S/_common.sh'; bytes_of '$d'" 2>/dev/null)
( unset SKIP_SIZE_CHECK; LANG=es_PE.UTF-8 bash -c "DATASET=prueba; source '$S/_common.sh'; verify_bytes '$d' 4252079748 0.02" > "$T/p3c.out" 2>&1 ); rcv=$?
echo "  bytes_of: $got (esperado 4252079748); verify_bytes: codigo $rcv; awk: $(readlink -f "$(command -v awk)")"
if [[ "$got" == "4252079748" && $rcv == 0 ]]; then ok "entero exacto y verificacion correcta"; else bad "prueba 3c"; cat "$T/p3c.out"; fi

echo; echo "-- Prueba 4: respuesta normal y errores 429 siguen igual que antes"
fresh; export FAKE_MODE=fail429x2
bash "$S/dl_tb_rahman.sh" > "$T/p4.out" 2>&1; rc=$?
echo "  codigo: $rc; llamadas: $(wc -l < "$T/calls.log")"
if [[ $rc == 0 && -f "$T/raw/tb_rahman/.done" && $(wc -l < "$T/calls.log") == 3 ]]; then ok "reintento ante 429 intacto"; else bad "prueba 4"; tail -5 "$T/p4.out"; fi

echo; echo "-- Prueba 5: el orquestador sigue con el paso siguiente cuando Kaggle se cuelga"
fresh; export FAKE_MODE=hang
ONLY_STEPS="dl_tb_rahman,dl_tbx11k" bash "$S/download_all.sh" --no-vindr > "$T/p5.out" 2>&1; rc=$?
grep -h -E "== paso|CON FALLOS" "$T/logs/download/general.log" | sed 's/^/    /' | cut -c1-200
if [[ $rc != 0 && $(colgados) == 0 ]] && grep -q "== paso dl_tbx11k" "$T/logs/download/general.log" \
   && grep -q "CON FALLOS en: dl_tb_rahman dl_tbx11k" "$T/logs/download/general.log"; then ok "continua y resume"; else bad "prueba 5"; tail -6 "$T/p5.out"; fi

echo; echo "RESUMEN: $pass pruebas OK, $fail fallidas"
[[ $fail == 0 ]]
