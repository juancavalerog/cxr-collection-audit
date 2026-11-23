"""Elección del ensayo de B1 (COVIDx) tras la búsqueda y ficheros de congelación por método (hallazgos M5 y M6 del
revisor independiente; docs/08_evaluacion.md, enmienda E6). Solo lee summary.json y config_resolved.json de la
búsqueda (semilla 42); no lee ninguna predicción.

    python scripts/eval/elegir_b1.py                        # escribe en results/b1/congelados/ y comprueba
    python scripts/eval/elegir_b1.py --out /tmp/b1c         # en otro directorio

Regla: la de scripts/jobs/bench_jobs.py (elegir): máximo de early_stopping.best (AUC sin promediado de volteos en
val_robust_sel), empate al ensayo menor; el ensayo 0 es la corrida del piloto (`baseline_<m>`), los demás
`baseline_<m>_t<T>`. Escribe `elegidos.json` (mismo formato que el que ya existe: {"baseline_<m>": {"trial",
"metrica"}}, sangría de un espacio) y un JSON por método, `baseline_<m>.json`, con el formato de
results/b3a/congelados/ ({method, params, lr_B, trial, metrica_seleccion, corrida_busqueda}), que es el que lee
`check_test_lock` (docs/06_lineas_base.md, Section 4, paso 1). Comprueba además que params y lr_B coinciden con
`sample_hparams(método, ensayo)` del registro y, si ya existe, con el config_resolved.json de la corrida final
`baseline_<m>_final/seed42`. Si `elegidos.json` ya existe y difiere, no lo sobrescribe y termina con código 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METODOS = ("erm", "erm_cuotas", "erm_mascara", "control_fuente", "groupdro", "dann", "coral", "mixstyle", "fourier",
           "lisa", "dfa_fuente", "dfr")
N_TRIALS = 8
SEMILLA = 42


def corrida(m, t, train):
    return Path(train) / (("baseline_%s" % m) if t == 0 else ("baseline_%s_t%d" % (m, t))) / ("seed%d" % SEMILLA)


def elegir(train):
    eleg, cong, avisos = {}, {}, []
    for m in METODOS:
        best = None
        for t in range(N_TRIALS):
            sp = corrida(m, t, train) / "summary.json"
            if not sp.exists():
                raise SystemExit("falta %s: la búsqueda de B1 no está completa (no se salta ningún ensayo)" % sp)
            v = float(json.load(open(sp, encoding="utf-8"))["early_stopping"]["best"])
            if best is None or v > best[0]:
                best = (v, t)
        v, t = best
        run = corrida(m, t, train)
        cfg = json.load(open(run / "config_resolved.json", encoding="utf-8"))
        params = (cfg.get("baseline") or {}).get("params") or {}
        lr = float(cfg["train"]["phases"][-1]["lr"])
        eleg["baseline_%s" % m] = {"trial": t, "metrica": v}
        rel = str(run.relative_to(ROOT)) if str(run).startswith(str(ROOT)) else str(run)
        cong["baseline_%s" % m] = {"method": m, "params": params, "lr_B": lr, "trial": t, "metrica_seleccion": v,
                                   "corrida_busqueda": rel}
        try:
            from src.baselines.registry import sample_hparams
            hp = sample_hparams(m, t)
            if t > 0 and (hp["params"] != params or abs(hp["common"].get("lr_B", lr) - lr) > 1e-15):
                avisos.append("%s: params o lr_B de la corrida no coinciden con sample_hparams(%s, %d)" % (m, m, t))
        except Exception as e:  # noqa: BLE001
            avisos.append("no se pudo comprobar sample_hparams (%s)" % e)
        fin = Path(train) / ("baseline_%s_final" % m) / ("seed%d" % SEMILLA) / "config_resolved.json"
        if fin.exists():
            cf = json.load(open(fin, encoding="utf-8"))
            if ((cf.get("baseline") or {}).get("params") or {}) != params or abs(float(cf["train"]["phases"][-1]["lr"]) - lr) > 1e-15:
                avisos.append("%s: la corrida final no usa los hiperparámetros congelados" % m)
    return eleg, cong, avisos


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=str(ROOT / "results" / "train"))
    ap.add_argument("--out", default=str(ROOT / "results" / "b1" / "congelados"))
    a = ap.parse_args(argv)
    eleg, cong, avisos = elegir(a.train)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(eleg, indent=1)
    codigo = 0
    pe = out / "elegidos.json"
    if pe.exists():
        actual = pe.read_text(encoding="utf-8")
        if actual != txt:
            print("elegidos.json DIFIERE del existente: no se sobrescribe")
            codigo = 1
        else:
            print("elegidos.json idéntico byte a byte al existente")
    else:
        pe.write_text(txt, encoding="utf-8")
    for k, v in cong.items():
        p = out / (k + ".json")
        nuevo = json.dumps(v, ensure_ascii=False, indent=2) + "\n"
        if p.exists() and p.read_text(encoding="utf-8") != nuevo:
            print("%s existe y difiere: no se sobrescribe" % p)
            codigo = 1
            continue
        tmp = p.with_name("." + p.name + ".tmp")
        tmp.write_text(nuevo, encoding="utf-8")
        os.replace(tmp, p)
        print("escrito %s (ensayo %d, lr_B %.3g, params %s)" % (p.name, v["trial"], v["lr_B"], v["params"]))
    for x in avisos:
        print("AVISO", x)
    return codigo if not any("no coinciden" in x or "no usa" in x for x in avisos) else 1


if __name__ == "__main__":
    sys.exit(main())
