#!/usr/bin/env bash
# Ejecuta las pruebas de src/model dentro del contenedor del servidor (Python con torch, timm y pytest).
# Uso: bash scripts/run_model_tests.sh   (desde cualquier directorio)
# Escribe la salida completa en logs/tests_model.log y devuelve el código de salida de pytest.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/tests_model.log"
cd "$ROOT"
{
  echo "# tests_model $(date -u +%Y-%m-%dT%H:%M:%SZ) en $(hostname)"
  python3 -c "import sys, torch, timm, numpy; print('python', sys.version.split()[0], 'torch', torch.__version__, 'timm', timm.__version__, 'numpy', numpy.__version__, 'cuda', torch.cuda.is_available())"
  python3 -m py_compile src/model/*.py tests/test_model.py && echo "py_compile: ok"
  python3 -m pytest -q -p no:cacheprovider tests/test_model.py
} 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
