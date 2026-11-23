"""Exploración posterior a la Puerta 3 (no confirmatoria): lectura ajustada con el conjunto reservado.

El tronco (ERM sin el conjunto reservado) no ha visto ninguna imagen reservada. La lectura es una regresión
logística L1 sobre características estandarizadas con el propio conjunto reservado, media de 10 submuestras
equilibradas, como en DFR. Se evalúa en val_robust, que no interviene en el ajuste. Solo CPU; no toca el test.

    python scripts/diag/expl_lectura_reserva.py --dir results/diag/expl_lectura --seeds 42 123
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

CS = [1.0, 0.3, 0.1]
PRIMARIA = ("penult", "antigua", 0.3)


def half(pid):
    return int(hashlib.sha1(str(pid).encode("utf-8")).hexdigest(), 16) % 2


def fit_bal(X, y, C, reps, rng):
    neg, pos = np.nonzero(y == 0)[0], np.nonzero(y == 1)[0]
    n = min(len(neg), len(pos))
    W, B = [], []
    for _ in range(reps):
        idx = np.concatenate([rng.choice(neg, n, replace=False), rng.choice(pos, n, replace=False)])
        lr = LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=2000)
        lr.fit(X[idx], y[idx])
        W.append(lr.coef_[0]); B.append(lr.intercept_[0])
    return np.mean(W, 0), float(np.mean(B))


def cargar(d, tag):
    z = np.load(d / ("%s.npz" % tag))
    m = pd.read_csv(d / ("%s_meta.csv" % tag))
    return z, m


def boot_dif(y, a, b, pid, n=1000, seed=5):
    """IC95 por remuestreo de pacientes de AUC(a) - AUC(b) en las mismas imágenes."""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(pid, return_inverse=True)
    idx_p = [np.nonzero(inv == i)[0] for i in range(len(u))]
    out = []
    for _ in range(n):
        s = np.concatenate([idx_p[i] for i in rng.integers(0, len(u), len(u))])
        if len(set(y[s].tolist())) < 2:
            continue
        out.append(roc_auc_score(y[s], a[s]) - roc_auc_score(y[s], b[s]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def analizar(d: Path, seed: int, exp: str, reps: int):
    z, m = cargar(d, "%s_seed%d_best" % (exp, seed))
    y = m["label"].to_numpy().astype(int)
    cj, src = m["conjunto"].to_numpy(), m["source"].to_numpy()
    pid = m["patient_id"].astype(str).to_numpy()
    vr = (cj == "val") & (src == "bimcv")
    h = m["patient_id"].map(half).to_numpy()
    fit, sel = vr & (h == 0), vr & (h == 1)
    neg, ant, nue = cj == "res_neg", cj == "res_pos_antigua", cj == "res_pos_nueva"
    res = neg | ant | nue
    lg = z["logit"].astype(np.float64)
    auc = lambda mk, s: float(roc_auc_score(y[mk], s[mk]))
    out = {"seed": seed, "n": {"res_neg": int(neg.sum()), "res_pos_antigua": int(ant.sum()), "res_pos_nueva": int(nue.sum()),
                               "val_robust": int(vr.sum())},
           "tronco_cabeza_erm": {"val_robust": auc(vr, lg), "fit": auc(fit, lg), "sel": auc(sel, lg),
                                 "frac_neg_reservados_logit_menor_0": float((lg[neg] < 0).mean()),
                                 "frac_neg_val_logit_menor_0": float((lg[vr & (y == 0)] < 0).mean())},
           "lecturas": {}}
    # referencia: ERM de la línea base (misma semilla), mismas imágenes de validación
    zb, mb = cargar(d, "baseline_erm_seed%d_best" % seed)
    vb = (mb["conjunto"].to_numpy() == "val") & (mb["source"].to_numpy() == "bimcv")
    if mb.loc[vb, "image_id"].tolist() != m.loc[vr, "image_id"].tolist():
        raise SystemExit("las imágenes de validación no coinciden")
    lb = zb["logit"].astype(np.float64)[vb]
    yv, pv = y[vr], pid[vr]
    out["erm_linea_base"] = {"val_robust": float(roc_auc_score(yv, lb)), "sel": float(roc_auc_score(yv[h[vr] == 1], lb[h[vr] == 1])),
                             "fit": float(roc_auc_score(yv[h[vr] == 0], lb[h[vr] == 0]))}
    rng = np.random.default_rng(7)
    prim = None
    for feat in ("penult", "pooled"):
        X = z[feat].astype(np.float64)
        mu, sd = X[res].mean(0), X[res].std(0) + 1e-8          # solo el conjunto reservado
        Z = (X - mu) / sd
        for nombre, mk in (("antigua", neg | ant), ("todas", res), ("nueva", neg | nue)):
            for C in CS:
                w, b = fit_bal(Z[mk], y[mk], C, reps, rng)
                s = Z @ w + b
                r = {"val_robust": auc(vr, s), "fit": auc(fit, s), "sel": auc(sel, s)}
                if (feat, nombre, C) == PRIMARIA:
                    lo, hi = boot_dif(yv, s[vr], lb, pv)
                    r["dif_vs_erm_val_robust"] = r["val_robust"] - out["erm_linea_base"]["val_robust"]
                    r["dif_ic95"] = [lo, hi]
                    prim = r
                out["lecturas"]["%s|%s|C%s" % (feat, nombre, C)] = r
        # referencia retenida sobre este tronco: ajuste en val_robust_fit, evaluación en val_robust_sel
        Xv = z[feat].astype(np.float64)
        mu2, sd2 = Xv[fit].mean(0), Xv[fit].std(0) + 1e-8
        Zv = (Xv - mu2) / sd2
        for C in CS:
            w, b = fit_bal(Zv[fit], y[fit], C, reps, rng)
            out["lecturas"]["%s|retenida_val_fit|C%s" % (feat, C)] = {"sel": auc(sel, Zv @ w + b)}
    out["primaria"] = prim
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--exp", default="expl_erm_reserva")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = [analizar(Path(a.dir), s, a.exp, a.reps) for s in a.seeds]
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    for r in res:
        print("== semilla", r["seed"], r["n"])
        print("  ERM línea base:", {k: round(v, 4) for k, v in r["erm_linea_base"].items()})
        print("  tronco con reserva, cabeza ERM:", {k: round(v, 4) for k, v in r["tronco_cabeza_erm"].items()})
        for k, v in r["lecturas"].items():
            print("   %-34s %s" % (k, {a_: (round(b_, 4) if isinstance(b_, float) else [round(x, 4) for x in b_]) for a_, b_ in v.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
