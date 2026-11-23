#!/usr/bin/env python3
"""Selecciona el subconjunto de VinDr-CXR (train de la competicion de Kaggle) que se descargara.

La version de Kaggle solo trae 14 hallazgos locales y "No finding" (sin Tuberculosis, que es una etiqueta global
de la version de PhysioNet). Criterio: se descargan todas las imagenes en que al menos un radiologo marco
Pleural effusion o Cardiomegaly (superconjunto; la etiqueta final por voto mayoritario se decide despues), y una
muestra con semilla fija de imagenes en que todos los lectores marcaron "No finding".
Salida: lista ordenada (positivas primero) de image_id, una por linea.
"""
import argparse

import pandas as pd

TARGET = ["Pleural effusion", "Cardiomegaly"]
NOFIND = "no finding"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--n-nofinding", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.train_csv)
    # nombres de clase normalizados (en Kaggle aparecen como "No finding" o "No Finding" segun la fuente)
    canon = {c.lower(): c for c in TARGET}
    df["class_name"] = df.class_name.astype(str).str.strip().map(
        lambda x: canon.get(x.lower(), "No finding" if x.lower() == NOFIND else x))
    if "Tuberculosis" in set(df.class_name):
        print("AVISO: train.csv contiene Tuberculosis; esta version no es la de Kaggle documentada")
    per = df.groupby(["image_id", "class_name"]).rad_id.nunique().unstack(fill_value=0)
    nread = df.groupby("image_id").rad_id.nunique()
    pos_any = per[[c for c in TARGET if c in per.columns]].max(axis=1) >= 1
    nofind = (per.get("No finding", 0) == nread) & ~pos_any
    pos_ids = per.index[pos_any]
    order = []
    for c in ["Pleural effusion", "Cardiomegaly"]:
        if c in per.columns:
            ids = per.index[(per[c] >= 1)]
            seen = set(order)
            order += [i for i in ids if i not in seen]
    nf = pd.Series(per.index[nofind]).sample(n=min(a.n_nofinding, int(nofind.sum())), random_state=a.seed)
    sel = order + nf.tolist()
    with open(a.out, "w") as fh:
        fh.write("\n".join(sel) + "\n")
    for c in TARGET:
        if c in per.columns:
            print(f"{c}: >=1 lector {int((per[c] >= 1).sum())}, >=2 lectores {int((per[c] >= 2).sum())}")
    print(f"positivas (superconjunto): {len(pos_ids)}; sin hallazgos unanimes disponibles: {int(nofind.sum())}; "
          f"seleccionadas sin hallazgos: {len(nf)}; total a descargar: {len(sel)}")


if __name__ == "__main__":
    main()
