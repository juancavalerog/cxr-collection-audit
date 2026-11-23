"""Registro de métodos de línea base (docs/06_lineas_base.md).

Cada método es una subclase de `Method` registrada con `@register`. Declara su referencia (clave de
refs/refs.bib), las etiquetas que usa, su estado, los hiperparámetros por defecto, el espacio de búsqueda y el
coste relativo frente a ERM. `BaselineTrainer` (src/baselines/trainer.py) instancia el método a partir de la
clave `baseline` de la configuración y le delega solo lo propio: pérdida, módulos auxiliares, cambios de datos
y pasos posteriores al entrenamiento. Todo lo demás (datos, optimizador, EMA, parada temprana, checkpoint,
manejo térmico y guardado de probabilidades) es el entrenador común de src/train/trainer.py.

Protocolo de búsqueda común (igual para todos los métodos): N_TRIALS configuraciones por método, la 0 con los
valores por defecto y las demás muestreadas de forma determinista de (método, ensayo), una semilla por
configuración en la búsqueda, selección por AUC-ROC en val_robust y SEEDS_FINAL semillas para la configuración
congelada. Los hiperparámetros comunes (COMMON_SEARCH) se muestrean igual para todos los métodos, ERM incluido.
"""
from __future__ import annotations

import zlib

import numpy as np

METHODS: dict = {}
N_TRIALS = 8
SEEDS_SEARCH = [42]
SEEDS_PILOT = [42, 123]
SEEDS_FINAL = [42, 123, 2024, 7, 999]

# Hiperparámetros comunes: tasa pico de la fase B (rango de DomainBed para troncos preentrenados, 10^U(-5, -3.5)).
COMMON_SEARCH = {"lr_B": ("log10", -5.0, -3.5)}


class Method:
    """Interfaz de un método. Los métodos cambian solo pérdida, muestreo dentro del lote y módulos auxiliares."""
    name = ""
    title = ""
    ref = ""                        # clave(s) de refs/refs.bib
    labels = ("clase",)             # etiquetas que usa en entrenamiento
    status = "implementado"         # implementado | aproximacion | esqueleto | pendiente
    empty_cells = ""                # qué ocurre con las celdas clase por fuente vacías
    defaults: dict = {}
    search: dict = {}               # nombre -> ("log10", lo, hi) | ("uniform", lo, hi) | ("choice", [..]) | ("int", lo, hi)
    hp_source = ""                  # de dónde salen los valores por defecto y el rango
    cost = 1.0                      # coste relativo de GPU frente a ERM (estimación sin medir)
    disk = 1.0                      # disco relativo frente a ERM
    needs_mask = False
    custom_sets = ()                # conjuntos de evaluación propios además de val_robust_fit y val_robust_sel
    noncomparable_sets = ()         # conjuntos cuya métrica no es comparable con otros métodos (DFR: val_robust)

    def __init__(self, params: dict | None = None):
        unknown = set(params or {}) - set(self.defaults)
        if unknown:
            raise ValueError("%s: hiperparámetros desconocidos %s (admitidos: %s)"
                             % (self.name, sorted(unknown), sorted(self.defaults)))
        self.p = dict(self.defaults)
        self.p.update(params or {})

    # ---- ganchos (todos opcionales salvo loss)
    def prepare_data(self, tr):
        """Tras construir los conjuntos del entrenador común."""

    def build(self, tr, net):
        """Crea módulos auxiliares en net.aux (se guardan en el checkpoint y en los estados)."""

    def build_net(self, tr):
        """Si devuelve un módulo, sustituye al modelo del entrenador común (sondas lineales)."""
        return None

    def loss(self, tr, net, x, y, s, gen, B):
        """Pérdida del microlote ya escalada para acumular (suma / B). Devuelve (tensor, dict de registro)."""
        raise NotImplementedError

    def after_step(self, tr):
        pass

    def before_finalize(self, tr):
        """Antes de la evaluación final (DFR añade aquí sus estados)."""

    # ---- benchmarks del plan B (data.bench; docs/13_entrenamiento_plan_b.md, Section 3). Sin data.bench no se usan.
    degenerado_sin_mixtas = ""      # nota si ninguna fuente de entrenamiento tiene las dos clases (B3a, rho = 1.0)

    def aplicabilidad(self, tr):
        """(estado, motivo) con estado en aplicable | degenerado | no_aplicable. Una corrida no aplicable termina sin
        entrenar, con summary.json en estado no_aplicable y sin predicciones."""
        if self.degenerado_sin_mixtas and not fuentes_mixtas_train(tr):
            return "degenerado", self.degenerado_sin_mixtas
        return "aplicable", None

    def noncomparable_bench(self, tr, state, name):
        """Motivo si la predicción (estado, conjunto) no es comparable entre métodos (datos usados para ajustar)."""
        return None


def fuentes_mixtas_train(tr):
    """Índices de las fuentes con las dos clases en el entrenamiento de la corrida (tabla de celdas)."""
    t = tr.cells
    return [s for s in range(t.shape[1]) if t[0, s] > 0 and t[1, s] > 0]


def register(cls):
    if not cls.name:
        raise ValueError("método sin nombre")
    METHODS[cls.name] = cls
    return cls


def _ensure_loaded():
    from src.baselines import methods  # noqa: F401  (registra los métodos al importar)


def get_method(name: str):
    _ensure_loaded()
    if name not in METHODS:
        raise KeyError("método desconocido: %s (registrados: %s)" % (name, ", ".join(sorted(METHODS))))
    return METHODS[name]


def build_method(cfg: dict) -> Method:
    bc = cfg.get("baseline") or {}
    cls = get_method(bc.get("method", "erm"))
    return cls(bc.get("params") or {})


def all_methods():
    _ensure_loaded()
    return dict(METHODS)


def _sample(spec, rng):
    kind = spec[0]
    if kind == "log10":
        return float(10 ** rng.uniform(spec[1], spec[2]))
    if kind == "uniform":
        return float(rng.uniform(spec[1], spec[2]))
    if kind == "choice":
        v = spec[1][int(rng.integers(len(spec[1])))]
        return v.item() if hasattr(v, "item") else v
    if kind == "int":
        return int(rng.integers(spec[1], spec[2] + 1))
    raise ValueError(spec)


def sample_hparams(name: str, trial: int) -> dict:
    """Ensayo 0: valores por defecto. Ensayo t > 0: muestra determinista de (método, t), como DomainBed."""
    cls = get_method(name)
    if int(trial) == 0:
        return {"common": {}, "params": {}}
    rng = np.random.default_rng([int(trial), zlib.crc32(name.encode())])
    common = {k: _sample(v, rng) for k, v in sorted(COMMON_SEARCH.items())}
    params = {k: _sample(v, rng) for k, v in sorted(cls.search.items())}
    return {"common": common, "params": params}


def apply_hparams(cfg: dict, hp: dict) -> dict:
    import copy
    c = copy.deepcopy(cfg)
    if "lr_B" in hp.get("common", {}):
        c["train"]["phases"][-1]["lr"] = hp["common"]["lr_B"]
    if hp.get("params"):
        c.setdefault("baseline", {}).setdefault("params", {}).update(hp["params"])
    return c
