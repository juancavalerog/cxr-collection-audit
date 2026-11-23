"""Árboles de resultados sintéticos con la estructura de los entrenadores, para probar la evaluación.

Reproduce lo que escriben src/train/trainer.py, src/baselines/trainer.py y src/train/csdnet_loop.py:
results/train/<experimento>/seed<S>/ con config_resolved.json, summary.json, train_log.jsonl,
predictions/<estado>_<conjunto>.csv (image_id, label, source, patient, prob) y, en CSD-Net, probes/final_best.json
y probes/V1_fin_A2_ep004.json en el formato de `write_report` de src/eval/probes.py. Las probabilidades son
sintéticas (logística de la etiqueta con efecto aleatorio de paciente y sesgo opcional por fuente); no hay
ningún dato real. Solo para pruebas (tests/test_eval_stats.py) y demostraciones.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

SRC = {"actmed": 0, "bimcv": 1, "cohen": 2, "fig1": 3, "ricord": 4, "rsna": 5, "sirm": 6, "stonybrook": 7}
SEEDS_PILOT = (42, 123)
SEEDS_FINAL = (42, 123, 2024, 7, 999)
CACHE = "data/cache_m/covidx_cxr4/homogeneo_224"


def patient_half(pid: str) -> int:
    """Igual que src/baselines/methods/dfr.py (sha1 del paciente módulo 2); 0 = val_robust_fit, 1 = sel."""
    return int(hashlib.sha1(str(pid).encode("utf-8")).hexdigest(), 16) % 2


def _patients(rng, prefix, n_pat, label_fn, max_img=3):
    rows = []
    for k in range(n_pat):
        pid = "%s_p%04d" % (prefix, k)
        y = int(label_fn(rng))
        for j in range(int(rng.integers(1, max_img + 1))):
            rows.append({"image_id": "%s_%d.png" % (pid, j), "label": y, "patient": pid})
    return rows


def population(kind: str = "val", seed: int = 0, n_bimcv: int = 240, n_rsna: int = 90, n_other: int = 60) -> pd.DataFrame:
    """Imágenes sintéticas. val: bimcv con ambas clases, rsna solo negativos, stonybrook solo positivos.
    test: bimcv, rsna (con y sin fuga) y ricord, con la columna test_subset."""
    rng = np.random.default_rng(seed + (0 if kind == "val" else 1000))
    rows = []
    for r in _patients(rng, kind + "_bimcv", n_bimcv, lambda g: g.random() < 0.6):
        rows.append({**r, "source": SRC["bimcv"], "test_subset": "bimcv"})
    for k, r in enumerate(_patients(rng, kind + "_rsna", n_rsna, lambda g: 0, max_img=1)):
        rows.append({**r, "source": SRC["rsna"], "test_subset": "rsna_con_fuga" if k % 3 else "rsna_sin_fuga"})
    other = "stonybrook" if kind == "val" else "ricord"
    for r in _patients(rng, kind + "_" + other, n_other, lambda g: 1, max_img=2):
        rows.append({**r, "source": SRC[other], "test_subset": other if kind == "test" else ""})
    df = pd.DataFrame(rows)
    eff = {p: float(rng.normal(0, 0.6)) for p in df["patient"].unique()}
    df["patient_effect"] = df["patient"].map(eff)
    return df


def score(pop: pd.DataFrame, strength: float, seed: int, source_bias: Optional[Dict[str, float]] = None,
          noise: float = 1.0) -> np.ndarray:
    """Probabilidad sintética: sigmoide(strength * (2y - 1) + efecto del paciente + sesgo de fuente + ruido)."""
    rng = np.random.default_rng(seed)
    z = strength * (2 * pop["label"].to_numpy() - 1) + pop["patient_effect"].to_numpy()
    z = z + rng.normal(0, noise, len(pop))
    if source_bias:
        for s, b in source_bias.items():
            z = z + b * (pop["source"].to_numpy() == SRC[s])
    return 1.0 / (1.0 + np.exp(-z))


def sets_from(pop: pd.DataFrame, prob: np.ndarray, kind: str, names) -> Dict[str, pd.DataFrame]:
    df = pop[["image_id", "label", "source", "patient"]].copy()
    df["prob"] = prob
    out = {}
    for n in names:
        if n in ("val", "test"):
            out[n] = df
        elif n == "val_robust":
            out[n] = df[df["source"] == SRC["bimcv"]]
        elif n in ("val_robust_sel", "val_robust_fit"):
            rob = df[df["source"] == SRC["bimcv"]]
            h = rob["patient"].map(patient_half)
            out[n] = rob[h == (1 if n.endswith("sel") else 0)]
        elif n.startswith("test_"):
            out[n] = df[pop["test_subset"].to_numpy() == n[5:]]
        else:
            raise ValueError(n)
    return {k: v.reset_index(drop=True) for k, v in out.items()}


def _verif(vid, verdict, **metrics):
    return {"id": vid, "supuesto": "sintético", "metricas": metrics, "umbrales": {}, "veredicto": verdict,
            "detalle": {}}


DEFAULT_VERDICTS = {"V0": "aprueba", "V1": "aprueba", "V1b": "informa", "V2": "aprueba", "V2b": "aprueba",
                    "V3": "aprueba", "V4": "informa", "V5": "aprueba", "V6": "informa", "V7": "informa"}


def write_run(root, experiment: str, seed: int, sets: Dict[str, pd.DataFrame], kind: str = "linea_base",
              states=("last", "best", "ema"), locked: bool = True, congelado: Optional[str] = None,
              verdicts: Optional[Dict[str, str]] = None, v1_a2: str = "aprueba", cache: str = CACHE,
              selection: str = "val_robust_sel", method: Optional[str] = None, state_shift: float = 0.02,
              flags: Optional[dict] = None) -> Path:
    d = Path(root) / experiment / ("seed%d" % seed)
    (d / "predictions").mkdir(parents=True, exist_ok=True)
    cfg = {"experiment": experiment, "seed": seed, "data": {"cache": cache, "eval_sets": list(sets)},
           "selection": {"metric": "auc_roc", "subset": selection},
           "protocol": {"fase": "piloto" if locked else "final", "test_bloqueado": bool(locked), "congelado": congelado},
           "eval": {"tta": True, "states": list(states)}}
    if method:
        cfg["baseline"] = {"method": method, "params": {}}
    (d / "config_resolved.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    summ = {"seed": seed, "experiment": experiment, "history": [], "results": {}}
    for st in states:
        for name, df in sets.items():
            out = df.copy()
            if st != "best" and st != "dfr_best":
                out["prob"] = np.clip(out["prob"] * (1 - state_shift) + state_shift * 0.5, 0, 1)
            out.to_csv(d / "predictions" / ("%s_%s.csv" % (st, name)), index=False, float_format="%.8f")
            summ["results"]["%s/%s" % (st, name)] = {"n": int(len(out))}
    if kind == "linea_base":
        summ["seleccion"] = selection
        summ["comparacion_entre_metodos"] = "val_robust_sel"
        summ["no_comparables"] = {}
    if kind == "csdnet":
        v = dict(DEFAULT_VERDICTS, **(verdicts or {}))
        summ.update({"v1": v1_a2, "flags": dict(flags or {}), "stopped_reason": "parada temprana",
                     "test_bloqueado": bool(locked), "probes": {"best": v}})
        from src.eval.probes import write_report
        res = {k: _verif(k, val, auc=0.55 if k == "V1" else 0.5) for k, val in v.items()}
        write_report(res, d / "probes", "final_best", {"epoca": 20, "estado": "best", "conjunto": "val"})
        write_report({"V1": _verif("V1", v1_a2, auc=0.55), "V1b": _verif("V1b", "informa")}, d / "probes",
                     "V1_fin_A2_ep004", {"epoca": 4, "estado": "actual", "conjunto": "val"})
    (d / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    (d / "train_log.jsonl").write_text(json.dumps({"event": "end", "out": str(d)}) + "\n", encoding="utf-8")
    return d


LEVEL1_NO_CONTROL = ("erm", "groupdro", "dann", "coral", "mixstyle", "fourier", "lisa", "dfa_fuente", "dfr",
                     "erm_mascara", "erm_cuotas")


def make_gate3_tree(root, scenario: str = "aprueba", seed: int = 0, best_strength: float = 0.75,
                    best_method: str = "erm_mascara", test_file_in_csd: bool = False,
                    exp_csd: str = "csdnet_piloto") -> dict:
    """Árbol del piloto (val, val_robust y val_robust_sel; nunca test). Escenarios: 'aprueba',
    'falla_discriminacion' (CSD-Net igual que el control), 'falla_verificaciones' (V2b falla en la semilla 123),
    'falla_v6' (CSD-Net con sesgo en rsna que abre la brecha de especificidad) y 'falla_v1_A2' (V1 falla al final de
    A2 en la semilla 42). `best_method` recibe `best_strength` (la mejor línea base esperada). Con
    `test_file_in_csd`, deja en CSD-Net un fichero best_test.csv ilegible: la Puerta 3 no debe abrirlo nunca."""
    root = Path(root)
    pop = population("val", seed)
    names = ["val", "val_robust", "val_robust_sel"]
    strengths = {"control_fuente": 0.55, "erm": 0.5, "groupdro": 0.6, "dann": 0.45, "coral": 0.5, "mixstyle": 0.55,
                 "fourier": 0.5, "lisa": 0.6, "dfa_fuente": 0.55, "dfr": 0.5, "erm_mascara": 0.55, "erm_cuotas": 0.55}
    strengths[best_method] = best_strength
    bias = {"erm": {"rsna": -1.5}, "groupdro": {"rsna": -1.0}}
    for m, s in strengths.items():
        for sd in SEEDS_PILOT:
            nm = [n for n in names if not (m == "dfr" and n == "val_robust")]
            st = ("last", "best", "ema", "dfr_last", "dfr_best", "dfr_ema") if m == "dfr" else ("last", "best", "ema")
            write_run(root, "baseline_" + m, sd, sets_from(pop, score(pop, s, 100 * sd + sum(map(ord, m)), bias.get(m)),
                                                           "val", nm), "linea_base", states=st, method=m)
    csd = {"falla_discriminacion": 0.55}.get(scenario, 1.6)
    cbias = {"rsna": 3.0} if scenario == "falla_v6" else None
    for sd in SEEDS_PILOT:
        verd = {"V2b": "falla"} if (scenario == "falla_verificaciones" and sd == 123) else {}
        v1 = "falla" if (scenario == "falla_v1_A2" and sd == 42) else "aprueba"
        fl = {"v1_fallo_intercambio_desactivado": True} if v1 == "falla" else None
        if scenario == "v1_informada":       # enmienda 8: V1 falla en las dos semillas y el intercambio sigue activo
            v1, verd = "falla", {"V1": "falla"}
            fl = {"v1_falla_intercambio_activo": True, "v1_enmienda": "docs/07_piloto.md, enmienda 8"}
        d = write_run(root, exp_csd, sd, sets_from(pop, score(pop, csd, 7 * sd, cbias), "val", names), "csdnet",
                      verdicts=verd, v1_a2=v1, flags=fl)
        if test_file_in_csd:
            (d / "predictions" / "best_test.csv").write_text("esto no es un CSV de predicciones\n", encoding="utf-8")
    for ab in ("csdnet_piloto_a14_canonica", "csdnet_piloto_a15_sin_dentro", "csdnet_piloto_a4_sin_compuerta",
               "csdnet_piloto_a2_aleatorio"):
        write_run(root, ab, 42, sets_from(pop, score(pop, 1.0, sum(map(ord, ab))), "val", names), "csdnet")
    return {"poblacion": pop, "raiz": root}


FINAL_SETS = ["val", "val_robust", "val_robust_sel", "test", "test_bimcv", "test_ricord", "test_rsna_sin_fuga",
              "test_rsna_con_fuga"]


def make_final_tree(root, project_root, methods=None, ablations=None, seed: int = 0, locked: bool = False) -> dict:
    """Árbol final (5 semillas, test desbloqueado con protocol.congelado). Escribe los ficheros de congelado en
    project_root/results/baselines/congelados/ y un informe de la Puerta 3 de prueba."""
    root, project_root = Path(root), Path(project_root)
    methods = methods or {"csdnet": 1.4, "control_fuente": 0.9, "erm": 0.8, "dfr": 0.85, "groupdro": 0.95}
    ablations = ablations or {"a2": 1.1, "a4": 1.35, "a14": 1.0}
    vpop, tpop = population("val", seed), population("test", seed)
    cong = project_root / "results" / "baselines" / "congelados"
    cong.mkdir(parents=True, exist_ok=True)
    bias = {"erm": {"rsna": -1.5}}
    exps = {}
    for m, s in list(methods.items()) + [("csdnet_" + a, s) for a, s in ablations.items()]:
        exp = ("csdnet_final" if m == "csdnet" else ("csdnet_final_" + m[7:] if m.startswith("csdnet_")
                                                     else "baseline_%s_final" % m))
        cpath = cong / ("%s.json" % m)
        cpath.write_text(json.dumps({"method": m, "params": {}, "lr_B": 1e-4}), encoding="utf-8")
        kind = "csdnet" if m.startswith("csdnet") else "linea_base"
        for sd in SEEDS_FINAL:
            sets = {}
            vs = sets_from(vpop, score(vpop, s, 11 * sd + len(m), bias.get(m)), "val",
                           [n for n in FINAL_SETS if n.startswith("val") and not (m == "dfr" and n == "val_robust")])
            ts = sets_from(tpop, score(tpop, s, 13 * sd + len(m), bias.get(m)), "test",
                           [n for n in FINAL_SETS if n.startswith("test")])
            sets.update(vs)
            sets.update(ts)
            st = ("last", "best", "ema", "dfr_best") if m == "dfr" else ("last", "best", "ema")
            write_run(root, exp, sd, sets, kind, states=st, locked=locked,
                      congelado=str(cpath.relative_to(project_root)), method=None if kind == "csdnet" else m)
        exps[m] = exp
    g3 = project_root / "results" / "gate3"
    g3.mkdir(parents=True, exist_ok=True)
    (g3 / "informe.json").write_text(json.dumps({"regla": {"aprobado": True}, "rama": "1", "sintetico": True}),
                                     encoding="utf-8")
    return {"experimentos": exps, "poblacion_test": tpop}
