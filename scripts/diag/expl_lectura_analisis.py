"""Exploración posterior a la Puerta 3 (no confirmatoria): lecturas lineales sobre características ya extraídas
(scripts/diag/expl_lectura_features.py). Solo CPU. No toca el test.

    python scripts/diag/expl_lectura_analisis.py --dir results/diag/expl_lectura --tags baseline_erm_seed42_best ...
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

CS = [1.0, 0.3, 0.1, 0.03, 0.01]


def half(pid):
    return int(hashlib.sha1(str(pid).encode("utf-8")).hexdigest(), 16) % 2


def fit_bal(X, y, n_neg, C, reps, rng, penalty="l1"):
    """Media de `reps` regresiones logísticas con submuestras equilibradas (n_neg por clase)."""
    neg, pos = np.nonzero(y == 0)[0], np.nonzero(y == 1)[0]
    n = min(n_neg, len(neg), len(pos))
    W, B = [], []
    for _ in range(reps):
        idx = np.concatenate([rng.choice(neg, n, replace=False), rng.choice(pos, n, replace=False)])
        lr = LogisticRegression(penalty=penalty, C=C, solver="liblinear", max_iter=2000)
        lr.fit(X[idx], y[idx])
        W.append(lr.coef_[0]); B.append(lr.intercept_[0])
    return np.mean(W, 0), float(np.mean(B))


def analizar(d: Path, tag: str, feat: str, reps: int):
    z = np.load(d / ("%s.npz" % tag))
    m = pd.read_csv(d / ("%s_meta.csv" % tag))
    X, logit = z[feat].astype(np.float64), z["logit"].astype(np.float64)
    y = m["label"].to_numpy().astype(int)
    cj, src = m["conjunto"].to_numpy(), m["source"].to_numpy()
    vr = (cj == "val") & (src == "bimcv")
    h = m["patient_id"].map(half).to_numpy()
    fit, sel = vr & (h == 0), vr & (h == 1)
    trn = (cj == "tr_bimcv_neg") | (cj == "tr_bimcv_pos")
    mu, sd = X[vr | trn].mean(0), X[vr | trn].std(0) + 1e-8
    Z = (X - mu) / sd
    rng = np.random.default_rng(7)
    out = {"tag": tag, "feat": feat, "n": {"val_robust": int(vr.sum()), "fit": int(fit.sum()), "sel": int(sel.sum()),
                                           "fit_neg": int((fit & (y == 0)).sum()), "sel_neg": int((sel & (y == 0)).sum()),
                                           "tr_neg": int((cj == "tr_bimcv_neg").sum())}}
    auc = lambda mask, s: float(roc_auc_score(y[mask], s[mask]))
    out["erm"] = {"val_robust": auc(vr, logit), "sel": auc(sel, logit), "fit": auc(fit, logit)}
    # logit de ERM en negativos de bimcv: vistos en el entrenamiento frente a no vistos
    out["logit_neg_bimcv"] = {"train_mediana": float(np.median(logit[cj == "tr_bimcv_neg"])),
                              "val_mediana": float(np.median(logit[vr & (y == 0)])),
                              "train_frac_menor_0": float((logit[cj == "tr_bimcv_neg"] < 0).mean()),
                              "val_frac_menor_0": float((logit[vr & (y == 0)] < 0).mean()),
                              "pos_train_mediana": float(np.median(logit[cj == "tr_bimcv_pos"])),
                              "pos_val_mediana": float(np.median(logit[vr & (y == 1)]))}
    # A1: lectura retenida (DFR): ajuste en fit, evaluación en sel, por C
    nf = int((fit & (y == 0)).sum())
    out["retenida_fit_a_sel"] = {}
    for C in CS:
        w, b = fit_bal(Z[fit], y[fit], nf, C, reps, rng)
        out["retenida_fit_a_sel"][str(C)] = auc(sel, Z @ w + b)
    # A2/A3: lectura con datos de entrenamiento ya vistos por el tronco; evaluación en val_robust completo y en sel
    out["vista_train"] = {}
    for n in (127, 647):
        for C in CS:
            w, b = fit_bal(Z[trn], y[trn], n, C, reps, rng)
            s = Z @ w + b
            out["vista_train"]["n%d_C%s" % (n, C)] = {"val_robust": auc(vr, s), "sel": auc(sel, s)}
    # A6: curva de aprendizaje de la lectura retenida, validación cruzada por paciente dentro de val_robust
    pats = m.loc[vr, "patient_id"].astype(str).unique()
    r2 = np.random.default_rng(11)
    r2.shuffle(pats)
    folds = np.array_split(pats, 5)
    pid = m["patient_id"].astype(str).to_numpy()
    out["curva_retenida_cv"] = {}
    for n in (30, 60, 127, 200):
        res = {}
        for C in (1.0, 0.1):
            sc = np.full(len(y), np.nan)
            for f in folds:
                te = vr & np.isin(pid, f)
                trm = vr & ~np.isin(pid, f)
                w, b = fit_bal(Z[trm], y[trm], n, C, max(3, reps // 2), r2)
                sc[te] = Z[te] @ w + b
            res[str(C)] = auc(vr, sc)
        out["curva_retenida_cv"]["n%d" % n] = res
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--feats", nargs="+", default=["penult"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    d = Path(a.dir)
    res = [analizar(d, t, f, a.reps) for t in a.tags for f in a.feats]
    txt = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    for r in res:
        print("==", r["tag"], r["feat"], r["n"])
        print("  ERM:", {k: round(v, 4) for k, v in r["erm"].items()})
        print("  logit neg bimcv:", {k: round(v, 3) for k, v in r["logit_neg_bimcv"].items()})
        print("  retenida fit->sel por C:", {k: round(v, 4) for k, v in r["retenida_fit_a_sel"].items()})
        print("  vista (train) :", {k: (round(v["val_robust"], 4), round(v["sel"], 4)) for k, v in r["vista_train"].items()})
        print("  curva retenida CV en val_robust:", {k: {c: round(x, 4) for c, x in v.items()} for k, v in r["curva_retenida_cv"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
