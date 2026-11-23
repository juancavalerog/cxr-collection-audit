#!/usr/bin/env bash
# Pruebas sin red de dl_nlm_tb.sh, nlm_crawl.py y de la continuidad de download_all.sh ante un paso fallido.
# Usa test_fixtures/nlm_fake_server.py. Se ejecuta en la VM o en el servidor: bash scripts/download/test_nlm_crawl.sh
S="$(cd "$(dirname "$0")" && pwd)"
T=${NTEST_DIR:-/tmp/ntest}; PORT=${NTEST_PORT:-18765}; DEAD=$((PORT+1))
rm -rf "$T"; mkdir -p "$T/www" "$T/logs"
TB="$T/www/public/Tuberculosis-Chest-X-ray-Datasets"
python3 - "$TB" <<'EOF'
import os, sys
tb = sys.argv[1]
def put(rel, n, seed):
    p = os.path.join(tb, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(bytes((i * 31 + seed) % 251 for i in range(n)))
m = "Montgomery-County-CXR-Set/MontgomerySet/"
for i in range(1, 4): put(m + f"CXR_png/MCUCXR_000{i}_0.png", 2000 + i, i)
put(m + "CXR_png/MCUCXR_flaky_1.png", 3000, 7)
put(m + "CXR_png/MCUCXR_grande_1.png", 3_000_000, 9)
for side in ("leftMask", "rightMask"):
    for i in range(1, 4): put(m + f"ManualMask/{side}/MCUCXR_000{i}_0.png", 500 + i, 20 + i)
put(m + "ClinicalReadings/MCUCXR 0001 0.txt", 80, 3)
put(m + "ClinicalReadings/vacio.txt", 0, 0)
put(m + "NLM-MontgomeryCXRSet-ReadMe.pdf", 1049, 4)
put(m + "montgomery_consensus_roi.csv", 406, 5)
s = "Shenzhen-Hospital-CXR-Set/"
for i in range(1, 5): put(s + f"CXR_png/CHNCXR_000{i}_0.png", 1500 + i, 40 + i)
put(s + "ClinicalReadings/CHNCXR_0001_0.txt", 60, 6)
put(s + "Annotations/json/CHNCXR_0001_0.json", 90, 8)
put(s + "NLM-ChinaCXRSet-ReadMe.docx", 529, 2)
EOF
python3 "$S/test_fixtures/nlm_fake_server.py" "$T/www" "$PORT" "$T/requests.log" & SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for _ in $(seq 50); do curl -s -o /dev/null "http://127.0.0.1:$PORT/x" && break; sleep 0.1; done
export NLM_BASE="http://127.0.0.1:$PORT/public/Tuberculosis-Chest-X-ray-Datasets"
export NLM_EXPECT_MONTGOMERY=5 NLM_EXPECT_SHENZHEN=4 NLM_CRAWL_ARGS="--tries 3 --pause 0 --timeout 10" FREE_MARGIN_GB=-1000
pass=0; fail=0
ok()  { echo "  RESULTADO: OK  $*"; pass=$((pass+1)); }
bad() { echo "  RESULTADO: FALLO  $*"; fail=$((fail+1)); }
fresh() { rm -rf "$T/raw" "$T/logs"; mkdir -p "$T/raw" "$T/logs"; : > "$T/requests.log"; export RAW_ROOT="$T/raw" LOG_DIR="$T/logs/download"; }
same() { diff -r -x .done -x MD5SUMS.local "$1" "$2" > "$T/diff.out" 2>&1; }
echo "== Pruebas de la descarga de NLM con servidor simulado ($(date '+%F %T'))"

echo; echo "-- Prueba 1: descarga completa, disposicion de carpetas, reintento ante 503 y ninguna peticion a URL de carpeta"
fresh; bash "$S/dl_nlm_tb.sh" all > "$T/p1.out" 2>&1; rc=$?
nm=$(find "$T/raw/montgomery/MontgomerySet/CXR_png" -name '*.png' 2>/dev/null | wc -l); ns=$(find "$T/raw/shenzhen/CXR_png" -name '*.png' 2>/dev/null | wc -l)
dirreq=$(grep -c -E '^[0-9]+ [^ ]*/ ' "$T/requests.log"); flaky=$(grep -c "flaky" "$T/requests.log")
echo "  codigo: $rc; PNG Montgomery: $nm; PNG Shenzhen: $ns; peticiones a carpetas: $dirreq; peticiones al fichero inestable: $flaky"
grep -h "reintento" "$T/logs/download/nlm_tb.log" | head -2
if [[ $rc == 0 && $nm == 5 && $ns == 4 && $dirreq == 0 && $flaky == 2 && -f "$T/raw/montgomery/.done" && -f "$T/raw/shenzhen/.done" ]] \
   && same "$TB/Montgomery-County-CXR-Set" "$T/raw/montgomery" && same "$TB/Shenzhen-Hospital-CXR-Set" "$T/raw/shenzhen" \
   && [[ -f "$T/raw/montgomery/MontgomerySet/ClinicalReadings/MCUCXR 0001 0.txt" && -f "$T/raw/montgomery/MontgomerySet/ClinicalReadings/vacio.txt" ]]; then
  ok "arbol identico al del servidor"; else bad "prueba 1"; cat "$T/diff.out" | head -5; tail -5 "$T/p1.out"; fi

echo; echo "-- Prueba 2: segunda ejecucion idempotente (sin peticiones)"
: > "$T/requests.log"; bash "$S/dl_nlm_tb.sh" all > "$T/p2.out" 2>&1; rc=$?
echo "  codigo: $rc; peticiones: $(wc -l < "$T/requests.log"); $(grep -c 'ya descargado' "$T/p2.out") avisos de ya descargado"
if [[ $rc == 0 && $(wc -l < "$T/requests.log") == 0 && $(grep -c 'ya descargado' "$T/p2.out") == 2 ]]; then ok "no repite la descarga"; else bad "prueba 2"; fi

echo; echo "-- Prueba 3: reanudacion de un fichero parcial con Range"
fresh; d="$T/raw/montgomery/MontgomerySet/CXR_png"; mkdir -p "$d"
head -c 1000000 "$TB/Montgomery-County-CXR-Set/MontgomerySet/CXR_png/MCUCXR_grande_1.png" > "$d/MCUCXR_grande_1.png.part"
bash "$S/dl_nlm_tb.sh" montgomery > "$T/p3.out" 2>&1; rc=$?
r206=$(grep -c '^206 .*grande.* range=bytes=1000000-' "$T/requests.log")
echo "  codigo: $rc; respuestas 206 desde el byte 1000000: $r206"
if [[ $rc == 0 && $r206 == 1 ]] && cmp -s "$d/MCUCXR_grande_1.png" "$TB/Montgomery-County-CXR-Set/MontgomerySet/CXR_png/MCUCXR_grande_1.png" && [[ ! -e "$d/MCUCXR_grande_1.png.part" ]]; then
  ok "fichero reanudado e identico"; else bad "prueba 3"; tail -5 "$T/p3.out"; fi

echo; echo "-- Prueba 4: recuento distinto del publicado; falla con mensaje y sin marcar .done"
fresh; NLM_EXPECT_MONTGOMERY=138 bash "$S/dl_nlm_tb.sh" montgomery > "$T/p4.out" 2>&1; rc=$?
echo "  codigo: $rc"; grep -h "ERROR" "$T/logs/download/nlm_tb.log" | head -2
if [[ $rc != 0 && ! -f "$T/raw/montgomery/.done" ]] && grep -q "ERROR: montgomery: numero de imagenes distinto" "$T/logs/download/nlm_tb.log"; then ok "fallo declarado"; else bad "prueba 4"; fi

echo; echo "-- Prueba 5: servidor inalcanzable; falla con mensaje en el log (antes moria en silencio)"
fresh; NLM_BASE="http://127.0.0.1:$DEAD/public/Tuberculosis-Chest-X-ray-Datasets" NLM_CRAWL_ARGS="--tries 2 --pause 0 --timeout 5" bash "$S/dl_nlm_tb.sh" montgomery > "$T/p5.out" 2>&1; rc=$?
echo "  codigo: $rc"; grep -h "ERROR" "$T/logs/download/nlm_tb.log" | head -3
if [[ $rc != 0 ]] && grep -q "ERROR: montgomery: la descarga no termino bien" "$T/logs/download/nlm_tb.log"; then ok "fallo declarado"; else bad "prueba 5"; fi

echo; echo "-- Prueba 6: download_all.sh continua tras un paso fallido, resume los fallos y sale con codigo distinto de cero"
fresh
ONLY_STEPS="dl_nlm_tb,dl_chexmask" NLM_BASE="http://127.0.0.1:$DEAD/x" NLM_CRAWL_ARGS="--tries 1 --pause 0 --timeout 5" \
  CHEXMASK_BASE="http://127.0.0.1:$PORT/no-existe" bash "$S/download_all.sh" --no-vindr > "$T/p6.out" 2>&1; rc=$?
echo "  codigo: $rc"; grep -h -E "== paso|CON FALLOS|el paso" "$T/logs/download/general.log"; grep -h "orden fallida" "$T/logs/download/chexmask.log" | head -1
if [[ $rc != 0 ]] && grep -q "== paso dl_chexmask" "$T/logs/download/general.log" \
   && grep -q "CON FALLOS en: dl_nlm_tb dl_chexmask" "$T/logs/download/general.log" \
   && grep -q "orden fallida" "$T/logs/download/chexmask.log"; then ok "continua y resume"; else bad "prueba 6"; tail -8 "$T/p6.out"; fi

echo; echo "-- Prueba 7: download_all.sh sin fallos conserva el mensaje final y el codigo cero"
fresh; ONLY_STEPS="dl_nlm_tb" bash "$S/download_all.sh" --no-vindr > "$T/p7.out" 2>&1; rc=$?
echo "  codigo: $rc; $(tail -1 "$T/logs/download/general.log" | cut -c1-80)"
if [[ $rc == 0 ]] && grep -q "Descargas terminadas. Siguiente paso" "$T/logs/download/general.log" && ! grep -q "CON FALLOS" "$T/logs/download/general.log"; then ok "ejecucion limpia"; else bad "prueba 7"; tail -5 "$T/p7.out"; fi

echo; echo "RESUMEN: $pass pruebas OK, $fail fallidas"
[[ $fail == 0 ]]
