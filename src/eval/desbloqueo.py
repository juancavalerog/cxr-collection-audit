"""Desbloqueo único de los conjuntos de test y de COVIDx externo (COVIDGR) tras la congelación del protocolo.

Hallazgo M1 del revisor independiente (review/informe_puerta7.md) y docs/08_evaluacion.md, versión 2, Section V2.7.2.
Todo camino que lea un conjunto test, test_* o COVIDGR (etiquetas, imágenes, características o predicciones) debe
llamar antes a `exigir_congelacion(motivo)`, que solo devuelve el registro de la congelación si:

1. existe `results/gate7/congelacion.json` y es un JSON con la estructura que escribe `scripts/eval/congelar.py`;
2. su SHA256 está anotado en STATUS.md, en una línea que contenga también la palabra `congelacion.json` (registro
   externo al fichero, escrito a mano y enviado fuera del servidor; un fichero de congelación creado
   después, o regenerado, no está anotado y no desbloquea nada);
3. cada fichero de la lista `ficheros` existe y su SHA256 coincide con el registrado.

Si algo falla lanza `TestBloqueadoError` (subclase de la del entrenador de B2 y B3a, de modo que los `except` que ya
la capturan siguen funcionando) y no deja leer nada. Si todo cuadra, añade una línea a
`results/gate7/desbloqueos.jsonl` con la fecha, el motivo, la orden, la máquina y el SHA256 de la congelación, y
devuelve el registro con la clave añadida `sha256_congelacion`.

La firma `exigir_congelacion(motivo) -> dict` es la que usan los informes de B3a y B2; los argumentos con nombre
(`ruta`, `status`, `registro`) solo existen para las pruebas y para el parche del entrenador de B2 y B3a, que
conserva su constante GATE7.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

from src.data.paths import PROJECT_ROOT
from src.train.bench import TestBloqueadoError as _BenchBloqueado

RUTA_CONGELACION = PROJECT_ROOT / "results" / "gate7" / "congelacion.json"
RUTA_STATUS = PROJECT_ROOT / "STATUS.md"
RUTA_REGISTRO = PROJECT_ROOT / "results" / "gate7" / "desbloqueos.jsonl"
VERSION_FORMATO = 1
CLAVES = ("version_formato", "fecha_utc", "orden", "ficheros")


class TestBloqueadoError(_BenchBloqueado):
    """El test (o COVIDGR) sigue bloqueado: no hay congelación válida del protocolo."""
    __test__ = False


def sha256_fichero(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def es_bloqueado(nombre) -> bool:
    """Conjuntos que exigen la congelación: test, test_* y cualquier nombre que contenga covidgr."""
    n = str(nombre).strip().lower()
    return n == "test" or n.startswith("test_") or "covidgr" in n


def _relativa(p: Path, raiz: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(raiz.resolve()))
    except ValueError:
        return str(p)


def sha_registrado_en_status(sha: str, status: Path) -> bool:
    if not Path(status).exists():
        return False
    for linea in Path(status).read_text(encoding="utf-8", errors="replace").splitlines():
        if sha in linea and "congelacion.json" in linea:
            return True
    return False


def verificar(ruta: Optional[Path] = None, status: Optional[Path] = None, raiz: Optional[Path] = None,
              exigir_status: bool = True) -> dict:
    """Comprueba la congelación sin registrar nada. Devuelve {'ok', 'motivos', 'cambiados', 'faltan', 'sha256',
    'registro', 'n_ficheros'}; no lanza excepciones por discrepancias (sí por errores de E/S inesperados)."""
    ruta = Path(ruta or RUTA_CONGELACION)
    status = Path(status or RUTA_STATUS)
    raiz = Path(raiz or PROJECT_ROOT)
    out = {"ok": False, "motivos": [], "cambiados": [], "faltan": [], "sha256": None, "registro": None,
           "n_ficheros": 0, "ruta": str(ruta)}
    if not ruta.exists():
        out["motivos"].append("no existe %s: la Puerta 7 no se ha congelado" % ruta)
        return out
    sha = sha256_fichero(ruta)
    out["sha256"] = sha
    try:
        reg = json.loads(ruta.read_text(encoding="utf-8"))
    except ValueError as e:
        out["motivos"].append("%s no es un JSON válido (%s)" % (ruta, e))
        return out
    falt = [k for k in CLAVES if k not in reg]
    if falt or not isinstance(reg.get("ficheros"), dict) or not reg["ficheros"]:
        out["motivos"].append("%s no tiene la estructura de scripts/eval/congelar.py (faltan %s)" % (ruta, falt or ["ficheros"]))
        return out
    if int(reg.get("version_formato", -1)) != VERSION_FORMATO:
        out["motivos"].append("versión de formato %s distinta de %d" % (reg.get("version_formato"), VERSION_FORMATO))
        return out
    out["registro"] = reg
    out["n_ficheros"] = len(reg["ficheros"])
    if exigir_status and not sha_registrado_en_status(sha, status):
        out["motivos"].append("el SHA256 de congelacion.json (%s) no está anotado en %s; regístrelo antes de abrir el "
                              "test (review/informe_puerta7.md, Section 5, punto 10)" % (sha, status))
    for rel, h in sorted(reg["ficheros"].items()):
        p = raiz / rel
        if not p.exists():
            out["faltan"].append(rel)
        elif sha256_fichero(p) != h:
            out["cambiados"].append(rel)
    if out["faltan"]:
        out["motivos"].append("faltan %d ficheros congelados (p. ej. %s)" % (len(out["faltan"]), out["faltan"][0]))
    if out["cambiados"]:
        out["motivos"].append("%d ficheros congelados han cambiado (p. ej. %s)" % (len(out["cambiados"]), out["cambiados"][0]))
    out["ok"] = not out["motivos"]
    return out


def exigir_congelacion(motivo, *, ruta: Optional[Path] = None, status: Optional[Path] = None,
                       registro: Optional[Path] = None, raiz: Optional[Path] = None) -> dict:
    """Devuelve el registro de la congelación del protocolo si es válida; si no, lanza TestBloqueadoError.

    `motivo` es un texto breve con lo que se va a leer (p. ej. 'evaluar_test_b1: test de COVIDx'). Cada desbloqueo
    concedido queda en results/gate7/desbloqueos.jsonl."""
    if not str(motivo or "").strip():
        raise TestBloqueadoError("exigir_congelacion necesita un motivo")
    v = verificar(ruta, status, raiz)
    if not v["ok"]:
        raise TestBloqueadoError("test bloqueado (Puerta 7) para '%s': %s" % (motivo, "; ".join(v["motivos"])))
    reg = dict(v["registro"])
    reg["sha256_congelacion"] = v["sha256"]
    linea = {"fecha_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "motivo": str(motivo),
             "sha256_congelacion": v["sha256"], "n_ficheros": v["n_ficheros"], "orden": list(sys.argv),
             "pid": os.getpid(), "maquina": socket.gethostname(), "usuario": _usuario(),
             "trabajo": os.environ.get("PAPER_NEW_JOB_ID", "")}
    rp = Path(registro or RUTA_REGISTRO)
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "a", encoding="utf-8") as f:
        f.write(json.dumps(linea, ensure_ascii=False) + "\n")
    return reg


def exigir_si_bloqueado(conjuntos: Iterable[str], motivo: str, **kw) -> Optional[dict]:
    """Llama a exigir_congelacion solo si alguno de los conjuntos pedidos es un test o COVIDGR."""
    pedidos = [c for c in conjuntos if es_bloqueado(c)]
    if not pedidos:
        return None
    return exigir_congelacion("%s (%s)" % (motivo, ", ".join(pedidos)), **kw)


def huella_congelada(registro: dict, path, raiz: Optional[Path] = None) -> str:
    """SHA256 congelado de un fichero; error si el fichero no está en la congelación o si ha cambiado."""
    raiz = Path(raiz or PROJECT_ROOT)
    rel = _relativa(Path(path), raiz)
    h = (registro.get("ficheros") or {}).get(rel)
    if h is None:
        raise TestBloqueadoError("%s no está en la congelación de la Puerta 7: no puede evaluarse en el test" % rel)
    if sha256_fichero(raiz / rel) != h:
        raise TestBloqueadoError("%s ha cambiado después de la congelación" % rel)
    return h


def _usuario() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001  (contenedor sin entrada en /etc/passwd)
        return str(os.getuid()) if hasattr(os, "getuid") else ""
