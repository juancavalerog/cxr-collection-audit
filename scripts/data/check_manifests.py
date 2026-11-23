#!/usr/bin/env python3
"""Comprueba los manifiestos: conteos, existencia de ficheros y pacientes compartidos entre particiones.

Uso:
    python3 check_manifests.py --manifests data/manifests --data-root /home/administrador/notebooks/covid \
        [--sample 2000 | --full] [--seed 0]
Codigo de salida 0 si todas las comprobaciones pasan, 1 si alguna falla.
"""
import argparse
import os
import sys

import pandas as pd

EXPECTED = {
    "covidx_cxr4.csv": {
        "rows": 84818,
        "split": {"train": 67863, "val": 8473, "test": 8482},
        "train_label": {1: 57199, 0: 10664},
        "train_neg_source": {"rsna": 9719, "bimcv": 647},
    },
    "nih_cxr14.csv": {"rows": 112120},
    "chexpert_small.csv": {"rows": 223648, "split": {"train": 223414, "valid": 234}},
}


class Report:
    def __init__(self):
        self.fail = 0

    def check(self, ok, msg):
        print(("  [OK]   " if ok else "  [FALLO] ") + msg)
        if not ok:
            self.fail += 1


def exists_check(df, root, n, seed, rep, name):
    sub = df if n is None or n >= len(df) else df.sample(n=n, random_state=seed)
    miss = [p for p in sub.image_path if not os.path.isfile(os.path.join(root, p))]
    scope = "completo" if len(sub) == len(df) else f"muestra de {len(sub)}"
    rep.check(len(miss) == 0, f"{name}: existencia de ficheros ({scope}): {len(miss)} ausentes")
    for p in miss[:10]:
        print("      ausente:", p)


def patient_overlap(df, rep, name, keys):
    splits = [s for s in df.split.dropna().unique() if s != ""]
    if len(splits) < 2:
        print(f"  [INFO] {name}: sin particiones definidas, no se comprueba solapamiento de pacientes")
        return
    sets = {s: set(map(tuple, df.loc[df.split == s, keys].astype(str).values)) for s in splits}
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            inter = sets[a] & sets[b]
            rep.check(len(inter) == 0,
                      f"{name}: pacientes compartidos {a}-{b} (clave {'+'.join(keys)}): {len(inter)}")
            for x in list(inter)[:10]:
                print("      compartido:", x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default="/home/administrador/Proyectos/paper_new/data/manifests")
    ap.add_argument("--data-root", default="/home/administrador/notebooks/covid")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sample", type=int, default=2000, help="ficheros a verificar por manifiesto")
    g.add_argument("--full", action="store_true", help="verificar todos los ficheros")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    n = None if a.full else a.sample
    rep = Report()
    for name, exp in EXPECTED.items():
        path = os.path.join(a.manifests, name)
        print(f"== {name}")
        if not os.path.exists(path):
            rep.check(False, f"{name}: no existe")
            continue
        df = pd.read_csv(path, dtype={"patient_id": str, "split": str}, keep_default_na=False,
                         low_memory=False)
        rep.check(len(df) == exp["rows"], f"filas {len(df)} (esperado {exp['rows']})")
        rep.check(df.image_path.is_unique, f"image_path unico: {df.image_path.is_unique}")
        rep.check((df.patient_id != "").all(), "patient_id sin vacios")
        if "split" in exp:
            got = df.split.value_counts().to_dict()
            rep.check(got == exp["split"], f"conteo por particion {got}")
        if "train_label" in exp:
            tr = df[df.split == "train"]
            got = {int(k): int(v) for k, v in tr.label.astype(int).value_counts().to_dict().items()}
            rep.check(got == exp["train_label"], f"train por clase (1=positivo) {got}")
            neg = tr[tr.label.astype(int) == 0].source.value_counts().to_dict()
            for s, v in exp["train_neg_source"].items():
                rep.check(neg.get(s, 0) == v, f"negativos de train en {s}: {neg.get(s, 0)} (esperado {v})")
            print("  Tabla clase por fuente y particion:")
            ct = pd.crosstab([df.split, df.source], df.label.astype(int))
            print("    " + ct.to_string().replace("\n", "\n    "))
            patient_overlap(df, rep, name, ["source", "patient_id"])
            patient_overlap(df, rep, name, ["patient_id"])
        else:
            patient_overlap(df, rep, name, ["patient_id"])
        if "view" in df.columns:
            print("  Vistas:", df.view.value_counts().to_dict())
        print(f"  Pacientes unicos: {df.patient_id.nunique()}")
        exists_check(df, a.data_root, n, a.seed, rep, name)
    # Fuga oculta: imagenes RSNA de COVIDx enlazadas por hash a pacientes de NIH ChestX-ray14
    mp = os.path.join(a.manifests, "covidx_rsna_to_nih_map.csv")
    print("== covidx_rsna_to_nih_map.csv (enlace por hash RSNA -> NIH, ver docs/02_datos.md Section 4)")
    if os.path.exists(mp):
        v = pd.read_csv(mp, dtype=str)
        print(f"  imagenes RSNA de COVIDx con copia confirmada en NIH: {len(v)}; pacientes NIH distintos: "
              f"{v.nih_patient_id.nunique()}")
        for x, y in [("train", "val"), ("train", "test"), ("val", "test")]:
            sh = set(v.loc[v.split == x, "nih_patient_id"]) & set(v.loc[v.split == y, "nih_patient_id"])
            n_img = int(v[v.nih_patient_id.isin(sh) & v.split.isin([x, y])].shape[0])
            print(f"  [AVISO] pacientes NIH compartidos {x}-{y} via RSNA: {len(sh)} ({n_img} imagenes)")
        print("  Estos cruces no son visibles con el patient_id oficial de COVIDx (un UUID por imagen RSNA).")
        print("  No se cuentan como fallo de la particion oficial; los benchmarks nuevos reagrupan por paciente NIH.")
    else:
        print("  [INFO] no existe; genere el enlace con hash_images.py y find_overlaps.py")
    # Version limpia de COVIDx (build_covidx_clean.py)
    cp = os.path.join(a.manifests, "covidx_cxr4_clean.csv")
    print("== covidx_cxr4_clean.csv")
    if not os.path.exists(cp):
        rep.check(False, "covidx_cxr4_clean.csv no existe")
    else:
        c = pd.read_csv(cp, dtype={"patient_id": str, "nih_patient_id": str}, keep_default_na=False)
        need = ["dataset", "source", "image_path", "patient_id", "label", "split", "nih_patient_id",
                "split_clean", "exclude_reason", "val_robust", "test_subset"]
        rep.check(all(x in c.columns for x in need), "columnas completas")
        rep.check(len(c) == 84818, f"filas {len(c)} (esperado 84818)")
        sc = c.split_clean.value_counts().to_dict()
        print("  split_clean:", sc)
        r1 = c.exclude_reason.str.contains("rsna_nih_patient_in_test")
        rep.check(int(r1.sum()) == 655, f"excluidas por paciente NIH en test: {int(r1.sum())} (esperado 655)")
        live = c[c.split_clean != "excluded"]
        nih = live[live.nih_patient_id != ""]
        for x, y in [("train", "val"), ("train", "test"), ("val", "test")]:
            sh = set(nih.loc[nih.split_clean == x, "nih_patient_id"]) & set(nih.loc[nih.split_clean == y, "nih_patient_id"])
            rep.check(len(sh) == 0, f"pacientes NIH compartidos {x}-{y} en la version limpia: {len(sh)}")
        patient_overlap(live.assign(split=live.split_clean), rep, "covidx_cxr4_clean", ["source", "patient_id"])
        st = live.set_index("image_path").split_clean
        bad = 0
        for fn in ["covidx_cross_split_duplicates.csv", "covidx_label_conflicts.csv"]:
            fp = os.path.join(a.manifests, fn)
            if os.path.exists(fp):
                d = pd.read_csv(fp)
                for pa, pb in zip(d.path_a, d.path_b):
                    if pa in st.index and pb in st.index and st[pa] != st[pb]:
                        bad += 1
        rep.check(bad == 0, f"pares casi identicos que siguen en particiones distintas: {bad}")
        lc = c.exclude_reason.str.contains("label_conflict")
        rep.check(bool((c.loc[lc, "split_clean"] == "excluded").all()) and int(lc.sum()) == 8,
                  f"imagenes con etiqueta en conflicto excluidas: {int(lc.sum())} (esperado 8)")
        vr = c[c.val_robust.astype(int) == 1]
        rep.check(bool(((vr.split_clean == "val") & (vr.source == "bimcv")).all()),
                  f"val_robust solo BIMCV de val limpia: {len(vr)} imagenes, {vr.patient_id.nunique()} pacientes")
        ts = c[c.split == "test"].test_subset.value_counts().to_dict()
        exp_ts = {"bimcv": 8082, "ricord": 200, "rsna_con_fuga": 129, "rsna_sin_fuga": 71}
        rep.check(ts == exp_ts, f"test_subset {ts}")
        print("  Tabla clase por fuente de la version limpia:")
        ct = pd.crosstab([live.split_clean, live.source], live.label.astype(int))
        print("    " + ct.to_string().replace("\n", "\n    "))
    print(f"\nRESUMEN: {'todas las comprobaciones pasan' if rep.fail == 0 else str(rep.fail) + ' comprobaciones fallan'}")
    sys.exit(1 if rep.fail else 0)


if __name__ == "__main__":
    main()
