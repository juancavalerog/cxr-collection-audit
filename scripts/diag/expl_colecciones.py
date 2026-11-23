"""Exploración (no confirmatoria): ¿qué separa una lectura dentro de bimcv, la enfermedad o la colección?

Con las características agrupadas de un tronco que no vio el conjunto reservado: (1) AUC en validación cruzada
por paciente de cada contraste dentro del conjunto reservado; (2) transferencia entre colecciones de positivos;
(3) un detector de colección ajustado solo con positivos (antiguos frente a nuevos, sin información de clase)
usado como clasificador en val_robust. Solo CPU; no toca el test.
"""
import argparse, hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


def lr(C=0.3):
    return LogisticRegression(penalty="l2", C=C, max_iter=4000, class_weight="balanced")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/diag/expl_lectura")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--feat", default="pooled")
    ap.add_argument("--C", type=float, default=0.05)
    a = ap.parse_args(argv)
    for tag in a.tags:
        z = np.load(Path(a.dir) / (tag + ".npz")); m = pd.read_csv(Path(a.dir) / (tag + "_meta.csv"))
        X = z[a.feat].astype(np.float64); y = m.label.to_numpy(); cj = m.conjunto.to_numpy(); src = m.source.to_numpy()
        pid = m.patient_id.astype(str).to_numpy()
        neg, ant, nue = cj == "res_neg", cj == "res_pos_antigua", cj == "res_pos_nueva"
        res = neg | ant | nue
        vr = (cj == "val") & (src == "bimcv")
        Z = (X - X[res].mean(0)) / (X[res].std(0) + 1e-8)
        print("==", tag, a.feat, "C =", a.C)

        def cv(mk, t):
            s = np.zeros(mk.sum()); Xm, tm, gm = Z[mk], t[mk], pid[mk]
            for tr, te in GroupKFold(5).split(Xm, tm, gm):
                s[te] = lr(a.C).fit(Xm[tr], tm[tr]).decision_function(Xm[te])
            return roc_auc_score(tm, s)

        print("  (1) validación cruzada en el reservado: neg vs antiguos %.3f | neg vs nuevos %.3f | antiguos vs nuevos %.3f"
              % (cv(neg | ant, y), cv(neg | nue, y), cv(ant | nue, nue.astype(int))))
        # (2) transferencia: mitad de negativos para ajustar, otra mitad para evaluar
        h = np.array([int(hashlib.sha1(p.encode()).hexdigest(), 16) % 2 for p in pid])
        n0, n1 = neg & (h == 0), neg & (h == 1)
        f_ant = lr(a.C).fit(Z[n0 | ant], y[n0 | ant]); f_nue = lr(a.C).fit(Z[n0 | nue], y[n0 | nue])
        ev = lambda f, mk: roc_auc_score(y[mk], f.decision_function(Z[mk]))
        print("  (2) lectura neg vs antiguos -> neg retenidos vs nuevos %.3f | val_robust %.3f" % (ev(f_ant, n1 | nue), ev(f_ant, vr)))
        print("      lectura neg vs nuevos   -> neg retenidos vs antiguos %.3f | val_robust %.3f" % (ev(f_nue, n1 | ant), ev(f_nue, vr)))
        # (3) detector de colección sin información de clase: antiguos (1) frente a nuevos (0)
        d = lr(a.C).fit(Z[ant | nue], ant[ant | nue].astype(int))
        s = d.decision_function(Z)
        print("  (3) detector antiguos-vs-nuevos (sin clase) usado como clasificador: val_robust %.3f | reservado neg vs antiguos %.3f | neg vs nuevos %.3f"
              % (roc_auc_score(y[vr], s[vr]), roc_auc_score(y[neg | ant], s[neg | ant]), roc_auc_score(y[neg | nue], s[neg | nue])))
        # (4) detector de la colección de negativos sin positivos antiguos: neg (0) frente a nuevos (1), ya en (2)
        print("  medianas del detector de colección: neg %.2f | antiguos %.2f | nuevos %.2f | val neg %.2f | val pos %.2f"
              % (np.median(s[neg]), np.median(s[ant]), np.median(s[nue]), np.median(s[vr & (y == 0)]), np.median(s[vr & (y == 1)])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
