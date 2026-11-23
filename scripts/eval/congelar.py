"""Congelación del protocolo (review/informe_puerta7.md, Section 5, puntos 8 a 10; docs/08_evaluacion.md, V2.7.2).

Escribe results/gate7/congelacion.json con el SHA256 de cada fichero de la lista declarada en
configs/gate7_congelacion.yaml, el entorno de análisis y la orden usada, y un fichero compatible con `sha256sum -c`.

    python scripts/eval/congelar.py --listar          # qué entraría, por grupo, sin escribir nada
    python scripts/eval/congelar.py                   # crea la congelación (se niega si ya existe o falta algo)
    python scripts/eval/congelar.py --verificar       # 0 si nada cambió; 1 si cambió o falta algún fichero
    python scripts/eval/congelar.py --verificar --exigir-status   # además exige el SHA256 anotado en STATUS.md

Después de crearla: anotar en STATUS.md la línea que imprime (contiene el SHA256 y la palabra congelacion.json),
enviar ese SHA256 fuera del servidor (correo al coautor o registro en OSF) y, antes de cada evaluación del test,
`python scripts/eval/congelar.py --verificar --exigir-status` o `sha256sum -c results/gate7/congelacion.sha256sum`.
Ningún guion lee un test sin `src.eval.desbloqueo.exigir_congelacion`, que hace la misma comprobación.
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import desbloqueo as D  # noqa: E402

CONFIG = ROOT / "configs" / "gate7_congelacion.yaml"
EXCLUIR_SIEMPRE = ("results/gate7/congelacion*", "results/gate7/desbloqueos.jsonl")


def leer_config(path) -> dict:
    import yaml
    c = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(c, dict) or not isinstance(c.get("grupos"), dict) or not c["grupos"]:
        raise SystemExit("%s: falta la clave 'grupos'" % path)
    for g, d in c["grupos"].items():
        if not d.get("patrones"):
            raise SystemExit("%s: el grupo %s no tiene patrones" % (path, g))
    return c


def _excluido(rel: str, excl) -> bool:
    partes = rel.split("/")
    if any(p.startswith(".") for p in partes):
        return True
    for pat in list(excl) + list(EXCLUIR_SIEMPRE):
        if fnmatch.fnmatch(rel, pat) or ("__pycache__" in pat and "__pycache__" in partes):
            return True
    return False


def resolver(cfg: dict, raiz: Path) -> dict:
    """{grupo: [rutas relativas ordenadas]}; un fichero aparece solo en el primer grupo que lo nombra."""
    vistos, out = set(), {}
    excl = cfg.get("excluir") or []
    for g, d in cfg["grupos"].items():
        fs = []
        for pat in d["patrones"]:
            for p in sorted(glob.glob(str(raiz / pat), recursive=True)):
                pp = Path(p)
                if not pp.is_file():
                    continue
                rel = pp.relative_to(raiz).as_posix()
                if rel in vistos or _excluido(rel, excl):
                    continue
                vistos.add(rel)
                fs.append(rel)
        out[g] = sorted(fs)
    return out


def entorno() -> dict:
    """Entorno del proceso que congela (en la VM o en el contenedor) y, si existe, el capturado en el contenedor."""
    try:
        from importlib import metadata
        paquetes = sorted("%s==%s" % (d.metadata["Name"], d.version) for d in metadata.distributions()
                          if d.metadata["Name"])
    except Exception as e:  # noqa: BLE001
        paquetes = ["no disponible: %s" % e]
    out = {"python": sys.version, "plataforma": platform.platform(), "maquina": platform.node(),
           "paquetes": paquetes}
    for k in ("torch", "timm", "numpy", "pandas", "scipy", "sklearn"):
        try:
            out["version_" + k] = __import__(k).__version__
        except Exception:  # noqa: BLE001
            out["version_" + k] = None
    return out


def comparar_lock(paquetes, lock: Path) -> list:
    """Diferencias entre los paquetes de un entorno y docker/requirements.lock.txt (nombre en minúsculas)."""
    if not lock.exists():
        return ["no existe %s" % lock]
    norm = lambda s: s.split("==")[0].strip().lower().replace("_", "-")
    a = {norm(p): p.split("==", 1)[1] if "==" in p else "" for p in paquetes if "==" in p}
    b = {}
    for ln in lock.read_text(encoding="utf-8").splitlines():
        if "==" in ln and not ln.startswith("#"):
            b[norm(ln)] = ln.split("==", 1)[1].strip()
    dif = []
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            dif.append("%s: entorno %s, lock %s" % (k, a.get(k, "ausente"), b.get(k, "ausente")))
    return dif


def listar(cfg, raiz) -> tuple:
    g = resolver(cfg, raiz)
    filas, faltan = [], []
    for k, fs in g.items():
        tam = sum((raiz / f).stat().st_size for f in fs)
        mn = int(cfg["grupos"][k].get("minimo", 0))
        ok = len(fs) >= mn
        if not ok:
            faltan.append(k)
        filas.append((k, len(fs), mn, tam, ok, cfg["grupos"][k].get("nota", "")))
    return g, filas, faltan


def imprimir(filas):
    print("%-26s %7s %7s %12s  %s" % ("grupo", "ficheros", "mínimo", "MB", "estado"))
    for k, n, mn, tam, ok, nota in filas:
        print("%-26s %7d %7d %12.1f  %s  (%s)" % (k, n, mn, tam / 2 ** 20, "ok" if ok else "INCOMPLETO", nota))


def crear(cfg_path, raiz: Path, salida: Path, reemplazar: bool) -> int:
    cfg = leer_config(cfg_path)
    grupos, filas, faltan = listar(cfg, raiz)
    imprimir(filas)
    if faltan:
        print("No se congela: grupos por debajo de su mínimo: %s" % ", ".join(faltan))
        return 2
    if salida.exists():
        if not reemplazar:
            print("Ya existe %s; para sustituirla use --reemplazar (la anterior se conserva con otro nombre y la "
                  "nueva debe registrarse de nuevo en STATUS.md)" % salida)
            return 3
        salida.rename(salida.with_name("congelacion_%s.json.anterior" % time.strftime("%Y%m%dT%H%M%S")))
    t0 = time.time()
    fich = {}
    for k, fs in grupos.items():
        for f in fs:
            fich[f] = D.sha256_fichero(raiz / f)
    env = entorno()
    cont = None
    pc = raiz / "results" / "gate7" / "entorno_contenedor.json"
    if pc.exists():
        cont = json.loads(pc.read_text(encoding="utf-8"))
    lock = raiz / "docker" / "requirements.lock.txt"
    reg = {"version_formato": D.VERSION_FORMATO, "fecha_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "orden": "python " + " ".join([Path(sys.argv[0]).as_posix()] + sys.argv[1:]),
           "config": Path(cfg_path).resolve().relative_to(raiz.resolve()).as_posix()
           if str(Path(cfg_path).resolve()).startswith(str(raiz.resolve())) else str(cfg_path),
           "grupos": grupos, "ficheros": fich,
           "entorno_congelacion": {k: v for k, v in env.items() if k != "paquetes"},
           "paquetes_congelacion": env["paquetes"],
           "diferencias_con_lock": comparar_lock(env["paquetes"], lock),
           "entorno_contenedor": cont,
           "diferencias_contenedor_con_lock": comparar_lock(cont.get("paquetes", []), lock) if cont else None,
           "segundos": None}
    reg["segundos"] = round(time.time() - t0, 1)
    salida.parent.mkdir(parents=True, exist_ok=True)
    tmp = salida.with_name("." + salida.name + ".tmp")
    tmp.write_text(json.dumps(reg, indent=1, ensure_ascii=False, sort_keys=False), encoding="utf-8")
    os.replace(tmp, salida)
    sums = salida.with_suffix(".sha256sum")
    sums.write_text("".join("%s  %s\n" % (h, f) for f, h in sorted(fich.items())), encoding="utf-8")
    sha = D.sha256_fichero(salida)
    print("Congelación escrita: %s (%d ficheros, %.1f s)" % (salida, len(fich), reg["segundos"]))
    print("Línea para STATUS.md:")
    print("Puerta 7: SHA256 de results/gate7/congelacion.json: %s (%s)" % (sha, reg["fecha_utc"]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--raiz", default=str(ROOT), help="raíz del proyecto (solo pruebas)")
    ap.add_argument("--salida", default=None, help="por defecto <raiz>/results/gate7/congelacion.json")
    ap.add_argument("--listar", action="store_true", help="solo lista lo que entraría")
    ap.add_argument("--verificar", action="store_true", help="comprueba las huellas; código 1 si algo cambió")
    ap.add_argument("--exigir-status", action="store_true", help="con --verificar, exige el SHA256 en STATUS.md")
    ap.add_argument("--reemplazar", action="store_true")
    ap.add_argument("--detalle", action="store_true", help="con --listar, imprime cada fichero")
    a = ap.parse_args(argv)
    raiz = Path(a.raiz)
    salida = Path(a.salida) if a.salida else raiz / "results" / "gate7" / "congelacion.json"
    if a.verificar:
        v = D.verificar(salida, raiz / "STATUS.md", raiz, exigir_status=a.exigir_status)
        reg_status = bool(v["sha256"]) and D.sha_registrado_en_status(v["sha256"], raiz / "STATUS.md")
        print("congelación: %s; SHA256 %s; %d ficheros; anotada en STATUS.md: %s"
              % (salida, v["sha256"], v["n_ficheros"], "sí" if reg_status else "no"))
        for f in v["cambiados"]:
            print("CAMBIADO  " + f)
        for f in v["faltan"]:
            print("FALTA     " + f)
        if v["ok"]:
            print("Verificación correcta: ningún fichero congelado ha cambiado.")
            return 0
        print("Verificación FALLIDA: " + "; ".join(v["motivos"]))
        return 1
    if a.listar:
        g, filas, faltan = listar(leer_config(a.config), raiz)
        imprimir(filas)
        if a.detalle:
            for k, fs in g.items():
                for f in fs:
                    print("%s\t%s" % (k, f))
        print("Total: %d ficheros; grupos incompletos: %s" % (sum(len(v) for v in g.values()), ", ".join(faltan) or "ninguno"))
        return 0
    return crear(a.config, raiz, salida, a.reemplazar)


if __name__ == "__main__":
    sys.exit(main())
