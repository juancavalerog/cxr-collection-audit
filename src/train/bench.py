"""Benchmarks del plan B (B2, B3a) en el entrenador común: carga por contrato, bloqueo del test, inicialización,
mitades estratificadas y caché (PLAN.md Section 13.7; docs/13_entrenamiento_plan_b.md).

Se activa con `data.bench: <ruta del manifiesto CSV, relativa a paper_new>`. Sin esa clave nada de este módulo se usa
y el entrenador se comporta como antes con COVIDx.

- Datos: `src.data.bench.load_bench` (fuentes, índices de fuente, fuentes mixtas, subconjuntos y selección salen
  del JSON del manifiesto). Con el test bloqueado, los DataFrames `test` y `test_*` se descartan nada más leer el
  manifiesto (un único CSV contiene todas las particiones): nunca llegan a un Dataset, a la evaluación ni a un
  fichero de predicciones.
- Bloqueo: `protocol.test_bloqueado` (por defecto true). Pedir un conjunto test con el bloqueo activo es un error;
  desbloquearlo exige además `protocol.congelado` y la congelación del protocolo (`results/gate7/congelacion.json`),
  comprobada con sus huellas por `src.eval.desbloqueo.exigir_congelacion` (hallazgo M1).
- Inicialización: en B2 y B3 el tronco parte de ImageNet (`model.init.type: timm_imagenet`); los pesos NIH del
  artículo 1 vieron imágenes de los tests de estos benchmarks (docs/11_benchmarks_plan_b.md, Section 11, punto 4).
  Cualquier otra inicialización es un error de configuración. El modelo diminuto `tiny` solo se admite con
  `protocol.fase: prueba` (pruebas en CPU).
- Mitades: para cada conjunto de selección S (`val` y el subconjunto de sensibilidad, `val_desconf` en B3a o
  `val_robust` en B2) se crean `S_fit` y `S_sel`, estratificadas por celda clase por fuente y fijas por paciente
  (sha1 del identificador, independientes de la semilla). Las usa DFR.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.data.paths import PROJECT_ROOT

N_SOURCES_MAX = 8                    # igual que src/baselines/common.py
# Troncos admitidos en B2 y B3 (siempre desde ImageNet): el principal y el de la sensibilidad S-T (docs/08, E8)
TRONCOS_BENCH = ("convnext_tiny", "resnet50")
GATE7 = PROJECT_ROOT / "results" / "gate7" / "congelacion.json"
HALF_SUFFIXES = ("_fit", "_sel")


class BenchConfigError(ValueError):
    """Configuración no admitida para un benchmark del plan B."""


class TestBloqueadoError(RuntimeError):
    """Se pidió un conjunto de test con el test bloqueado."""
    __test__ = False


def is_test(name) -> bool:
    n = str(name).strip().lower()
    return n == "test" or n.startswith("test_")


def bench_path(value) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else PROJECT_ROOT / p


def rel_project(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(p)


def check_init(cfg: dict):
    mc = cfg["model"]
    fase = (cfg.get("protocol") or {}).get("fase")
    if mc.get("name") == "tiny":
        if fase != "prueba":
            raise BenchConfigError("el modelo 'tiny' solo se admite en pruebas (protocol.fase: prueba)")
        return
    if mc.get("name") != "paper1_convnext":
        raise BenchConfigError("modelo %r no admitido en los benchmarks del plan B" % mc.get("name"))
    if mc.get("backbone", "convnext_tiny") not in TRONCOS_BENCH:
        raise BenchConfigError("tronco %r no admitido en los benchmarks del plan B (admitidos: %s)"
                               % (mc.get("backbone"), ", ".join(TRONCOS_BENCH)))
    t = (mc.get("init") or {}).get("type")
    if t != "timm_imagenet":
        raise BenchConfigError(
            "en B2 y B3 el tronco debe partir de ImageNet (model.init.type: timm_imagenet); se pidió %r. Los pesos "
            "NIH del artículo 1 vieron imágenes de los tests de estos benchmarks (docs/11_benchmarks_plan_b.md, "
            "Section 11, punto 4)." % t)


def requested_sets(cfg: dict) -> list:
    sc = cfg.get("selection") or {}
    out = list((cfg.get("data") or {}).get("eval_sets") or [])
    for k in ("subset", "subset_dc"):
        if sc.get(k):
            out.append(sc[k])
    return out


def check_test(cfg: dict) -> bool:
    """Devuelve True si el test queda bloqueado. Error si se pide test sin un desbloqueo válido."""
    pc = cfg.get("protocol") or {}
    locked = bool(pc.get("test_bloqueado", True))
    sc = cfg.get("selection") or {}
    for k in ("subset", "subset_dc"):
        if sc.get(k) and is_test(sc[k]):
            raise TestBloqueadoError("la selección de modelo no puede hacerse en '%s'" % sc[k])
    wants = [n for n in requested_sets(cfg) if is_test(n)]
    if wants and locked:
        raise TestBloqueadoError("test bloqueado (protocol.test_bloqueado: true) y la configuración pide %s" % wants)
    if wants and not locked:
        if not pc.get("congelado"):
            raise TestBloqueadoError("test desbloqueado sin protocol.congelado")
        from src.eval.desbloqueo import exigir_congelacion      # M1: huellas de la congelación del protocolo, no solo el fichero
        exigir_congelacion("entrenador de B2 y B3a: %s" % wants, ruta=GATE7)
    return locked


def exigir_ficheros_congelados(cfg: dict, reg: dict = None):
    """Con el test desbloqueado, el JSON de hiperparámetros congelados y el manifiesto (CSV y JSON) de la corrida deben
    estar en la congelación del protocolo sin cambios (logs/entrenamiento_r3/parches/, refuerzo del bloqueo). Los
    ficheros fuera de paper_new (pruebas en directorios temporales) no pueden estar congelados y se rechazan igual."""
    from src.data.bench import meta_path
    from src.eval.desbloqueo import TestBloqueadoError as _Bloq, huella_congelada, verificar
    if reg is None:                       # check_test ya concedió el desbloqueo (y lo registró): aquí solo se relee
        v = verificar(ruta=GATE7)
        if not v["ok"]:
            raise _Bloq("test bloqueado (Puerta 7): %s" % "; ".join(v["motivos"]))
        reg = v["registro"]
    pc = cfg.get("protocol") or {}
    rutas = [bench_path(pc["congelado"])]
    if (cfg.get("data") or {}).get("bench"):
        m = bench_path(cfg["data"]["bench"])
        rutas += [m, meta_path(m)]
    for p in rutas:
        huella_congelada(reg, p)


def _phash(pid) -> str:
    return hashlib.sha1(str(pid).encode("utf-8")).hexdigest()


def stratified_halves(frame):
    """Dos mitades por paciente, estratificadas por celda (clase, fuente): dentro de cada celda los pacientes se
    ordenan por sha1 del identificador y se reparten alternando (el primero a `fit`). Cada celda con n pacientes da
    ceil(n / 2) a fit y floor(n / 2) a sel. La celda de un paciente es la de su primera fila."""
    import pandas as pd
    if not len(frame):
        return frame.copy(), frame.copy()
    first = frame.drop_duplicates("patient_id")[["patient_id", "label", "source_idx"]].copy()
    first["h"] = first["patient_id"].astype(str).map(_phash)
    first = first.sort_values(["label", "source_idx", "h"])
    first["rank"] = first.groupby(["label", "source_idx"]).cumcount()
    half = dict(zip(first["patient_id"], (first["rank"] % 2).astype(int)))
    h = frame["patient_id"].map(half)
    return frame[h == 0].reset_index(drop=True), frame[h == 1].reset_index(drop=True)


def resolve_train_sources(value, meta: dict):
    """None (todas), 'mixtas' (las fuentes con las dos clases en train según el JSON) o una lista de fuentes."""
    if value in (None, "", []):
        return None
    if value == "mixtas":
        return list(meta["fuentes_mixtas"])
    if isinstance(value, str):
        value = [value]
    bad = sorted(set(value) - set(meta["fuentes"]))
    if bad:
        raise BenchConfigError("data.train_sources con fuentes que el benchmark no tiene: %s" % bad)
    return list(value)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(cfg: dict, seed: int):
    """(frames, meta, info). `frames` tiene train (filtrado por data.train_sources), val, los subconjuntos que no son
    test y las mitades `<S>_fit` y `<S>_sel` de los conjuntos de selección; info va a cfg['bench']."""
    from src.data.bench import load_bench, meta_path
    dc = cfg["data"]
    check_init(cfg)
    locked = check_test(cfg)
    if not locked and any(is_test(n) for n in requested_sets(cfg)):
        exigir_ficheros_congelados(cfg)
    path = bench_path(dc["bench"])
    if not path.exists():
        raise BenchConfigError("no existe el manifiesto %s" % path)
    frames, meta = load_bench(path)
    if locked:                                   # el test no llega a ningún Dataset
        for k in [k for k in frames if is_test(k)]:
            del frames[k]
    else:
        wanted = set(requested_sets(cfg))
        for k in [k for k in frames if is_test(k) and k not in wanted]:
            del frames[k]
    if len(meta["fuentes"]) > N_SOURCES_MAX:
        raise BenchConfigError("el benchmark tiene %d fuentes; el máximo es %d" % (len(meta["fuentes"]), N_SOURCES_MAX))
    if meta.get("semilla") is not None and int(seed) != int(meta["semilla"]):
        raise BenchConfigError("la semilla de la corrida (%d) no es la del manifiesto (%s): los directorios seed<S> "
                               "deben emparejar métodos por la misma submuestra" % (int(seed), meta["semilla"]))
    fuentes_train = sorted(frames["train"]["source"].unique().tolist())
    ts = resolve_train_sources(dc.get("train_sources"), meta)
    if ts:
        frames["train"] = frames["train"][frames["train"]["source"].isin(ts)].reset_index(drop=True)
    sc = cfg.get("selection") or {}
    dcname = sc.get("subset_dc")
    for base in ["val"] + ([dcname.rsplit("_", 1)[0] if dcname.endswith(HALF_SUFFIXES) else dcname] if dcname else []):
        if base in frames:
            frames[base + "_fit"], frames[base + "_sel"] = stratified_halves(frames[base])
    for k, f in frames.items():
        if k in ("train",):
            continue
        if is_test(k) and locked:
            raise TestBloqueadoError("conjunto %s presente con el test bloqueado" % k)
    info = {"id": meta["benchmark"], "manifiesto": rel_project(path), "sha256_manifiesto": _sha256(path),
            "sha256_json": _sha256(meta_path(path)), "rho": meta.get("rho"), "semilla": meta.get("semilla"),
            "orientacion": meta.get("orientacion"), "fuentes": list(meta["fuentes"]),
            "fuentes_mixtas": list(meta["fuentes_mixtas"]), "fuentes_train": fuentes_train,
            "train_sources": ts, "subconjuntos": list(meta["subconjuntos"]), "seleccion_json": meta.get("seleccion"),
            "test_bloqueado": locked,
            "n": {k: int(len(v)) for k, v in sorted(frames.items())}}
    return frames, meta, info


def cache_auto(meta: dict, channel: str, res: int):
    """data/cache_m/bench_<id>/<canal>_<res> (la construyen los trabajos b_02, b_05 y b_08)."""
    c = PROJECT_ROOT / "data" / "cache_m" / ("bench_%s" % meta["benchmark"]) / ("%s_%d" % (channel, res))
    return c if (c / "index.csv").exists() else None


def check_cache(cache_dir, frames: dict, names):
    """Con require_cache: todas las imágenes de los conjuntos usados deben estar en la caché (falla al arrancar y no
    tras entrenar)."""
    from src.data.cache import read_index
    if cache_dir is None:
        raise FileNotFoundError("require_cache: true y no hay caché del benchmark")
    idx = read_index(cache_dir)
    for n in names:
        miss = [i for i in frames[n]["image_id"] if i not in idx]
        if miss:
            raise FileNotFoundError("faltan %d imágenes de %s en la caché %s (p. ej. %s)" % (len(miss), n, cache_dir, miss[0]))


def write_json(path: Path, obj):
    path = Path(path)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
