#!/usr/bin/env python3
"""Materializa la version limpia de COVIDx CXR-4 (Benchmark 1, docs/02_datos.md Section 3.1).

Entradas (data/manifests/): covidx_cxr4.csv, covidx_rsna_to_nih_map.csv, covidx_cross_split_duplicates.csv,
covidx_label_conflicts.csv.
Salida: covidx_cxr4_clean.csv con las columnas de covidx_cxr4.csv mas
  nih_patient_id  paciente NIH enlazado por hash (solo RSNA con copia confirmada)
  split_clean     train, val, test o excluded
  exclude_reason  vacio o una o varias razones separadas por ';'
  val_robust      1 si la imagen esta en val_clean y es de BIMCV
  test_subset     bimcv, ricord, rsna_sin_fuga, rsna_con_fuga (solo en test)

Reglas, en este orden:
  R1 rsna_nih_patient_in_test: imagen RSNA de train o val cuyo paciente NIH tiene imagenes en test -> excluded.
  R2 nih_patient_regrouped: paciente NIH con imagenes en train y val (tras R1): todas sus imagenes pasan a la
     particion donde tiene mas imagenes (empate: train). No excluye; cambia split_clean y lo anota.
  R3 label_conflict: los dos miembros de cada par casi identico con etiquetas opuestas -> excluded (tambien en test).
  R4 duplicate_cross_split: par casi identico en particiones distintas (tras R2): se excluye el miembro de train
     si el otro esta en val o test, y el de val si el otro esta en test.
  test_subset rsna_con_fuga: imagen RSNA del test cuyo paciente NIH tiene imagenes en train o val oficiales.
"""
import argparse
import os

import pandas as pd


def add_reason(df, mask, reason):
    cur = df.loc[mask, "exclude_reason"]
    df.loc[mask, "exclude_reason"] = cur.where(cur == "", cur + ";") + reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default="/home/administrador/Proyectos/paper_new/data/manifests")
    a = ap.parse_args()
    M = a.manifests
    df = pd.read_csv(os.path.join(M, "covidx_cxr4.csv"), dtype={"patient_id": str})
    mp = pd.read_csv(os.path.join(M, "covidx_rsna_to_nih_map.csv"), dtype=str)
    dup = pd.read_csv(os.path.join(M, "covidx_cross_split_duplicates.csv"))
    df = df.merge(mp[["path_covidx", "nih_patient_id"]].rename(columns={"path_covidx": "image_path"}),
                  on="image_path", how="left")
    df["nih_patient_id"] = df.nih_patient_id.fillna("")
    df["split_clean"] = df.split
    df["exclude_reason"] = ""
    has = df.nih_patient_id != ""

    # R1
    test_p = set(df.loc[has & (df.split == "test"), "nih_patient_id"])
    r1 = has & df.split.isin(["train", "val"]) & df.nih_patient_id.isin(test_p)
    df.loc[r1, "split_clean"] = "excluded"
    add_reason(df, r1, "rsna_nih_patient_in_test")

    # R2
    live = has & df.split_clean.isin(["train", "val"])
    cnt = df[live].groupby(["nih_patient_id", "split_clean"]).size().unstack(fill_value=0)
    for c in ["train", "val"]:
        if c not in cnt.columns:
            cnt[c] = 0
    both = cnt[(cnt.train > 0) & (cnt.val > 0)]
    target = (both.val > both.train).map({True: "val", False: "train"})
    for pid, tgt in target.items():
        m = live & (df.nih_patient_id == pid) & (df.split_clean != tgt)
        df.loc[m, "split_clean"] = tgt
        add_reason(df, m, f"nih_patient_regrouped_to_{tgt}")

    # R3 y R4 sobre pares duplicados entre particiones oficiales
    idx = pd.Series(df.index, index=df.image_path)
    # pares con etiqueta opuesta (entre particiones o dentro de una misma), en covidx_label_conflicts.csv
    conf = pd.read_csv(os.path.join(M, "covidx_label_conflicts.csv"))
    allpairs = pd.concat([dup, conf], ignore_index=True).drop_duplicates(["path_a", "path_b"])
    conflicts = allpairs[allpairs.lab_a != allpairs.lab_b]
    for _, r in conflicts.iterrows():
        for p in (r.path_a, r.path_b):
            i = idx[p]
            df.loc[i, "split_clean"] = "excluded"
            add_reason(df, df.index == i, "label_conflict")
    for _, r in dup[dup.lab_a == dup.lab_b].iterrows():
        ia, ib = idx[r.path_a], idx[r.path_b]
        sa, sb = df.loc[ia, "split_clean"], df.loc[ib, "split_clean"]
        if "excluded" in (sa, sb) or sa == sb:
            continue
        rank = {"train": 0, "val": 1, "test": 2}
        drop = ia if rank[sa] < rank[sb] else ib
        df.loc[drop, "split_clean"] = "excluded"
        add_reason(df, df.index == drop, "duplicate_cross_split")

    # val_robust y test_subset
    df["val_robust"] = ((df.split_clean == "val") & (df.source == "bimcv")).astype(int)
    trval_p = set(df.loc[has & df.split.isin(["train", "val"]), "nih_patient_id"])
    ts = pd.Series("", index=df.index)
    t = df.split == "test"
    ts[t & (df.source == "bimcv")] = "bimcv"
    ts[t & (df.source == "ricord")] = "ricord"
    ts[t & (df.source == "rsna")] = "rsna_sin_fuga"
    ts[t & (df.source == "rsna") & df.nih_patient_id.isin(trval_p)] = "rsna_con_fuga"
    df["test_subset"] = ts
    cols = ["dataset", "source", "image_path", "patient_id", "label", "split",
            "nih_patient_id", "split_clean", "exclude_reason", "val_robust", "test_subset"]
    df[cols].to_csv(os.path.join(M, "covidx_cxr4_clean.csv"), index=False)

    # resumen
    print("filas:", len(df))
    print("split_clean:", df.split_clean.value_counts().to_dict())
    print("razones (imagenes):")
    print(df.exclude_reason.str.split(";").explode().loc[lambda s: s != ""].value_counts().to_string())
    print("tabla clase por fuente en la version limpia (0/1):")
    k = df[df.split_clean != "excluded"]
    print(pd.crosstab([k.split_clean, k.source], k.label).to_string())
    print("excluidas por particion oficial y fuente:")
    e = df[df.split_clean == "excluded"]
    print(pd.crosstab([e.split, e.source], e.label).to_string())
    print("test_subset:", df[t].test_subset.value_counts().to_dict())
    vb = k[(k.split_clean == "val")]
    for s in ["bimcv", "cohen", "actmed"]:
        x = vb[vb.source == s]
        print(f"val_clean {s}: {len(x)} imagenes, {x.patient_id.nunique()} pacientes, "
              f"negativas {int((x.label == 0).sum())}, positivas {int((x.label == 1).sum())}")
    print("pares con etiqueta opuesta:")
    print(conflicts[["path_a", "split_a", "src_a", "lab_a", "path_b", "split_b", "src_b", "lab_b", "corr"]].to_string())


if __name__ == "__main__":
    main()
