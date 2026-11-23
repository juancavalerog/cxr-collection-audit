#!/usr/bin/env python3
"""Construye los manifiestos de los datasets ya presentes en disco.

Raiz de las rutas image_path: /home/administrador/notebooks/covid/ (DATA_ROOT).
Uso:
    python3 build_manifests.py --data-root /home/administrador/notebooks/covid \
        --out /home/administrador/Proyectos/paper_new/data/manifests [--check-exists]
"""
import argparse
import os
import sys

import pandas as pd

NIH_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
              "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
              "Pleural_Thickening", "Pneumonia", "Pneumothorax", "No Finding"]
CHEX_LABELS = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
               "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
               "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]


def col(name):
    return "y_" + name.lower().replace(" ", "_")


def read_covidx_txt(path):
    # Formato real: "<patient_id> <filename> <label> <source>", separado por espacios, sin cabecera.
    rows = []
    with open(path) as fh:
        for ln, line in enumerate(fh, 1):
            parts = line.strip().split(" ")
            if not parts or parts == [""]:
                continue
            if len(parts) != 4:
                raise ValueError(f"{path}:{ln}: se esperaban 4 campos, hay {len(parts)}")
            rows.append(parts)
    return pd.DataFrame(rows, columns=["patient_id", "fname", "label", "source"])


def build_covidx(root, check):
    base = os.path.join(root, "data")
    full = read_covidx_txt(os.path.join(base, "dataset.txt"))
    val = read_covidx_txt(os.path.join(base, "val.txt"))
    test = read_covidx_txt(os.path.join(base, "test.txt"))
    assert full.fname.is_unique, "nombres repetidos en dataset.txt"
    excl = set(val.fname) | set(test.fname)
    missing_in_full = (set(val.fname) | set(test.fname)) - set(full.fname)
    if missing_in_full:
        print(f"AVISO: {len(missing_in_full)} ficheros de val/test no estan en dataset.txt", file=sys.stderr)
    train = full[~full.fname.isin(excl)].copy()
    train["split"], val["split"], test["split"] = "train", "val", "test"
    df = pd.concat([train, val, test], ignore_index=True)
    # Todas las imagenes estan fisicamente en data/dataset/ (las carpetas val/ y test/ estan vacias).
    df["image_path"] = "data/dataset/" + df.fname
    df["dataset"] = "covidx_cxr4"
    df["label"] = df.label.map({"positive": 1, "negative": 0}).astype(int)
    if check:
        present = set(os.listdir(os.path.join(base, "dataset")))
        df["exists"] = df.fname.isin(present)
    return df[["dataset", "source", "image_path", "patient_id", "label", "split"]
              + (["exists"] if check else [])]


def build_nih(root, check):
    src = pd.read_csv(os.path.join(root, "nih", "Data_Entry_2017.csv"))
    df = pd.DataFrame({
        "dataset": "nih_cxr14",
        "source": "nih_cc",
        "image_path": "nih/images/" + src["Image Index"],
        "patient_id": "nih_" + src["Patient ID"].astype(str),
        "view": src["View Position"],
        "labels": src["Finding Labels"],
        "age": src["Patient Age"],
        "sex": src["Patient Gender"],
        "followup": src["Follow-up #"],
    })
    sets = src["Finding Labels"].str.split("|")
    for lab in NIH_LABELS:
        df[col(lab)] = sets.apply(lambda s, l=lab: int(l in s))
    # Sin particion oficial en disco (train_val_list.txt / test_list.txt no estan): split vacio.
    df["split"] = ""
    if check:
        present = set(os.listdir(os.path.join(root, "nih", "images")))
        df["exists"] = src["Image Index"].isin(present).values
    return df


def build_chexpert(root, check):
    base = os.path.join(root, "CheXpert")
    parts = []
    for split, fn in [("train", "train.csv"), ("valid", "valid.csv")]:
        s = pd.read_csv(os.path.join(base, "CheXpert-v1.0-small", fn))
        s["split"] = split
        parts.append(s)
    src = pd.concat(parts, ignore_index=True)
    pid = src.Path.str.extract(r"/(patient\d+)/")[0]
    study = src.Path.str.extract(r"/(study\d+)/")[0]
    df = pd.DataFrame({
        "dataset": "chexpert_small",
        "source": "stanford",
        "image_path": "CheXpert/" + src.Path,
        "patient_id": "chx_" + pid,
        "study_id": study,
        "view": src["Frontal/Lateral"],
        "ap_pa": src["AP/PA"],
        "age": src.Age,
        "sex": src.Sex,
    })
    pos = []
    for lab in CHEX_LABELS:
        v = src[lab]
        # Convencion: 1 positivo, 0 negativo, -1 incierto, vacio = no mencionado (se guarda como NaN).
        df[col(lab)] = v
        pos.append(v.where(v == 1).notna().map({True: lab, False: None}))
    lab_df = pd.concat(pos, axis=1)
    df["labels"] = lab_df.apply(lambda r: "|".join([x for x in r if x]), axis=1)
    df["split"] = src.split
    if check:
        df["exists"] = [os.path.exists(os.path.join(base, p)) for p in src.Path]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/administrador/notebooks/covid")
    ap.add_argument("--out", default="/home/administrador/Proyectos/paper_new/data/manifests")
    ap.add_argument("--check-exists", action="store_true",
                    help="verifica existencia de todos los ficheros (columna temporal, no se guarda)")
    ap.add_argument("--only", nargs="*", default=["covidx", "nih", "chexpert"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    builders = {"covidx": ("covidx_cxr4.csv", build_covidx),
                "nih": ("nih_cxr14.csv", build_nih),
                "chexpert": ("chexpert_small.csv", build_chexpert)}
    for key in a.only:
        name, fn = builders[key]
        df = fn(a.data_root, a.check_exists)
        if a.check_exists:
            miss = int((~df.exists).sum())
            print(f"{name}: {len(df)} filas, ficheros ausentes: {miss}")
            df = df.drop(columns="exists")
        else:
            print(f"{name}: {len(df)} filas")
        df.to_csv(os.path.join(a.out, name), index=False)


if __name__ == "__main__":
    main()
