#!/usr/bin/env python3
"""Benchmark B2 del plan B: tuberculosis con bases compuestas, variante B (docs/02_datos.md Section 3.2).

Fuentes (índice = posición):
  0 belarus     parte Belarus de la base de Rahman (solo TB)
  1 rsna_nih    parte RSNA de la base de Rahman (solo normales), enlazada con su paciente de NIH por hash
  2 tbx11k      TBX11K train (fuente mixta): 600 TB y 600 no TB submuestreados con la semilla (300 sanas y 300
                enfermas no TB); TBX11K val completo (200 TB y 1 600 no TB) va a validación
  3 shenzhen    test robusto (copia oficial de NLM, ambas clases)
  4 montgomery  test robusto (copia oficial de NLM, ambas clases)

Las imágenes de la base de Rahman que son copias de Shenzhen o Montgomery se identifican por hash contra las copias
oficiales y se excluyen (su metadato URL las atribuye a NLM; se comprueba que ambas fuentes coinciden). Los casi
duplicados (MD5 o pHash <= 4 con correlación de miniaturas >= 0.97) se resuelven dentro de cada fuente y entre
fuentes antes de partir: prioridad test > validación > entrenamiento y, dentro de la misma partición, se conserva
una sola copia. Validación: TBX11K val más el 10 % de los pacientes de cada parte de Rahman (fijo, semilla 20261114).

    python3 scripts/data/build_b2.py --paso dedup
    python3 scripts/data/build_b2.py --paso verificar      # repetir hasta código 0
    python3 scripts/data/build_b2.py --paso build
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_common as bc  # noqa: E402
from src.data.bench import write_bench  # noqa: E402
from src.data.paths import data_root  # noqa: E402

OUT = ROOT / "data" / "manifests" / "bench" / "b2"
H = ROOT / "data" / "manifests" / "bench" / "hashes"
RAW = "data/raw/"
FUENTES = ["belarus", "rsna_nih", "tbx11k", "shenzhen", "montgomery"]
SEEDS = (0, 1, 2, 3, 4)
SUBSETS = ["val_robust", "test_shenzhen", "test_montgomery", "test_robusto"]
N_TBX_NEG = 600
PRIORIDAD = {"test": 3, "val": 2, "train": 1, "": 0}


def abs_proj(p):
    return str(ROOT / p)


def candidatos() -> pd.DataFrame:
    """Todas las imágenes candidatas de B2 con su origen, clase y partición prevista (antes de deduplicar)."""
    rows = []
    rb = RAW + "tb_rahman/TB_Chest_Radiography_Database/"
    for clase, nombre in ((1, "Tuberculosis"), (0, "Normal")):
        md = pd.read_excel(ROOT / rb / ("%s.metadata.xlsx" % nombre))
        for f, url in zip(md["FILE NAME"], md["URL"]):
            origen = "nlm" if "ncbi" in url else ("belarus" if "tuberculosis.by" in url else "rsna")
            rows.append({"dataset": "tb_rahman", "image_path": rb + "%s/%s.png" % (nombre, f), "clase": clase,
                         "origen": origen, "tbx_split": "", "tbx_cat": ""})
    tb = RAW + "tbx11k/TBX11K/"
    for sp, lst in (("train", "TBX11K_train.txt"), ("val", "TBX11K_val.txt")):
        for ln in open(ROOT / tb / "lists" / lst, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                cat = ln.split("/")[0]
                rows.append({"dataset": "tbx11k", "image_path": tb + "imgs/" + ln, "clase": int(cat == "tb"),
                             "origen": "tbx11k", "tbx_split": sp, "tbx_cat": cat})
    for ds, d in (("shenzhen", RAW + "shenzhen/CXR_png"), ("montgomery", RAW + "montgomery/MontgomerySet/CXR_png")):
        for p in sorted((ROOT / d).glob("*.png")):
            rows.append({"dataset": ds, "image_path": "%s/%s" % (d, p.name), "clase": int(p.stem.endswith("_1")),
                         "origen": ds, "tbx_split": "", "tbx_cat": ""})
    return pd.DataFrame(rows)


def hashes() -> pd.DataFrame:
    parts = [bc.cargar_hashes(H / "tb_rahman.csv", RAW + "tb_rahman/"),
             bc.cargar_hashes(H / "tbx11k.csv", RAW + "tbx11k/TBX11K/imgs/"),
             bc.cargar_hashes(H / "shenzhen.csv", RAW + "shenzhen/CXR_png/"),
             bc.cargar_hashes(H / "montgomery.csv", RAW + "montgomery/")]
    return pd.concat(parts, ignore_index=True)


def paso_dedup(log=print):
    cand = candidatos()
    h = hashes().merge(cand[["image_path", "dataset"]], on="image_path", how="inner")
    falta = set(cand.image_path) - set(h.image_path)
    if falta:
        raise SystemExit("faltan hashes de %d imágenes (p. ej. %s)" % (len(falta), sorted(falta)[0]))
    h["grupo"] = h.image_path
    OUT.mkdir(parents=True, exist_ok=True)
    pares = bc.casi_duplicados(h, abs_proj, grupo_col="grupo", verificar=False, log=log)
    pares.drop(columns=[c for c in ("grupo_a", "grupo_b") if c in pares]).to_csv(OUT / "duplicados_hash.csv",
                                                                                 index=False)
    # Parte RSNA de Rahman frente a NIH ChestX-ray14 (para conocer el paciente).
    r = h[h.image_path.isin(cand[cand.origen == "rsna"].image_path)].reset_index(drop=True)
    n = bc.cargar_hashes(ROOT / "logs" / "tmp_datos" / "hash_nih_cxr14.csv")
    to = lambda s: np.array([int(x, 16) for x in s], dtype=np.uint64)
    pa, pb, da, db = to(r.phash), to(n.phash), to(r.dhash), to(n.dhash)
    ia, ib, _ = bc.fo.candidate_pairs(pa, pb, da, db, 5, 20000, 4, 12)
    m = pd.DataFrame({"a": r.image_path.to_numpy()[ia], "b": n.image_path.to_numpy()[ib], "kind": "phash",
                      "phash_dist": bc.fo.popcount64(pa[ia] ^ pb[ib]), "dhash_dist": bc.fo.popcount64(da[ia] ^ db[ib]),
                      "thumb_corr": np.nan, "confirmed": False})
    m.to_csv(OUT / "rahman_rsna_a_nih_candidatos.csv", index=False)
    log("parte RSNA de Rahman: %d imágenes, %d candidatos en NIH" % (len(r), len(m)))


def paso_verificar(max_s=150, log=print):
    t = time.time()
    q1 = bc.verificar_pares(OUT / "duplicados_hash.csv", abs_proj, max_s=max_s, log=log)
    root = data_root()
    q2 = bc.verificar_pares(OUT / "rahman_rsna_a_nih_candidatos.csv",
                            lambda p: str((ROOT / p) if p.startswith("data/raw/") else (root / p)),
                            max_s=max(5, max_s - (time.time() - t)), log=log)
    return 0 if q1 + q2 == 0 else 3


def enlace_nih(log=print) -> pd.DataFrame:
    m = pd.read_csv(OUT / "rahman_rsna_a_nih_candidatos.csv")
    if m.thumb_corr.isna().any():
        raise SystemExit("faltan verificaciones RSNA-NIH: ejecute --paso verificar")
    m = m[m.confirmed].sort_values(["a", "phash_dist", "thumb_corr"], ascending=[True, True, False])
    best = m.drop_duplicates("a")
    nih = pd.read_csv(ROOT / "data" / "manifests" / "nih_cxr14.csv", usecols=["image_path", "patient_id", "labels"])
    best = best.merge(nih.rename(columns={"image_path": "b"}), on="b", how="left")
    best = best.rename(columns={"a": "image_path", "b": "path_nih", "patient_id": "nih_patient_id",
                                "labels": "nih_labels"})
    cols = ["image_path", "path_nih", "nih_patient_id", "nih_labels", "phash_dist", "thumb_corr"]
    best[cols].to_csv(OUT / "rahman_rsna_to_nih_map.csv", index=False)
    log("parte RSNA de Rahman enlazada con NIH: %d imágenes, %d pacientes NIH" % (len(best),
                                                                                 best.nih_patient_id.nunique()))
    return best[cols]


def preparar(log=print):
    cand = candidatos()
    d = pd.read_csv(OUT / "duplicados_hash.csv")
    if d.thumb_corr.isna().any():
        raise SystemExit("faltan verificaciones: ejecute --paso verificar")
    dup = d[d.confirmed]
    # 1) copias de NLM dentro de Rahman (por hash contra las copias oficiales)
    nlm = set(cand[cand.dataset.isin(["shenzhen", "montgomery"])].image_path)
    rah = set(cand[cand.dataset == "tb_rahman"].image_path)
    tbx = set(cand[cand.dataset == "tbx11k"].image_path)
    copia_nlm = set()
    for a, b in zip(dup.a, dup.b):
        if (a in nlm) != (b in nlm):
            copia_nlm.add(b if a in nlm else a)
    meta_nlm = set(cand[cand.origen == "nlm"].image_path)
    info = {"rahman_copias_nlm_por_hash": len(copia_nlm & rah), "rahman_nlm_por_metadato": len(meta_nlm),
            "rahman_nlm_metadato_sin_copia_hash": len(meta_nlm - copia_nlm),
            "rahman_copia_hash_sin_metadato_nlm": len((copia_nlm & rah) - meta_nlm),
            "tbx11k_copias_nlm": len(copia_nlm & tbx)}
    log("NLM: " + json.dumps(info))
    cand["excluir"] = ""
    cand.loc[cand.image_path.isin(copia_nlm | meta_nlm) & cand.dataset.isin(["tb_rahman", "tbx11k"]),
             "excluir"] = "copia_nlm"
    # 2) fuente y paciente
    cand["source"] = cand.dataset.map({"tbx11k": "tbx11k", "shenzhen": "shenzhen", "montgomery": "montgomery"})
    cand.loc[cand.origen == "belarus", "source"] = "belarus"
    cand.loc[cand.origen == "rsna", "source"] = "rsna_nih"
    mapa = enlace_nih(log)
    mp = dict(zip(mapa.image_path, mapa.nih_patient_id))
    stem = cand.image_path.str.split("/").str[-1].str.replace(".png", "", regex=False)
    cand["patient_id"] = cand.dataset + ":" + stem
    cand.loc[cand.source == "tbx11k", "patient_id"] = "tbx11k:" + cand.image_path.str.split("imgs/").str[-1]
    cand["patient_id_origen"] = "imagen"
    sel = cand.source.eq("rsna_nih") & cand.image_path.isin(mp)
    cand.loc[sel, "patient_id"] = cand.loc[sel, "image_path"].map(mp)
    cand.loc[sel, "patient_id_origen"] = "nih_por_hash"
    # Shenzhen y Montgomery: no publican identificador de paciente; una imagen por paciente según NLM.
    # 3) partición prevista por fuente (reserva de pacientes de Rahman: 10 % a validación)
    cand["reserva"] = ""
    cand.loc[cand.source.isin(["shenzhen", "montgomery"]), "reserva"] = "test"
    cand.loc[cand.source == "tbx11k", "reserva"] = cand.tbx_split
    for s in ("belarus", "rsna_nih"):
        m = (cand.source == s) & (cand.excluir == "")
        pac = cand[m].groupby("patient_id").size().rename("n").reset_index()
        pac["clase"] = 0
        r = bc.reservas_pacientes(pac, ["clase"], fracs=(0.9, 0.1), clave="b2_" + s, nombres=("train", "val"))
        cand.loc[m, "reserva"] = cand.loc[m, "patient_id"].map(dict(zip(pac.patient_id, r)))
    # 4) casi duplicados restantes (dentro de cada fuente y entre fuentes)
    vivo = cand[cand.excluir == ""].set_index("image_path")
    rest = dup[dup.a.isin(vivo.index) & dup.b.isin(vivo.index)]
    rep = bc.grupos_union(rest)
    comp = pd.DataFrame({"image_path": list(rep), "grupo": list(rep.values())})
    comp = comp.join(vivo[["reserva", "clase", "source", "patient_id"]], on="image_path")
    dup_rows = []
    for g, gg in comp.groupby("grupo"):
        if gg.clase.nunique() > 1:
            for p in gg.image_path:
                dup_rows.append((p, "duplicado_con_clase_distinta", g))
            continue
        gg = gg.assign(pr=gg.reserva.map(PRIORIDAD)).sort_values(["pr", "image_path"], ascending=[False, True])
        keep = gg.iloc[0]
        for p in gg.image_path.iloc[1:]:
            dup_rows.append((p, "duplicado_de:" + keep.image_path, g))
    dd = pd.DataFrame(dup_rows, columns=["image_path", "motivo", "grupo"])
    cand.loc[cand.image_path.isin(dd.image_path), "excluir"] = \
        cand.image_path.map(dict(zip(dd.image_path, dd.motivo)))
    info["casi_duplicados_resueltos"] = len(dd)
    # paciente común entre reservas (p. ej. dos imágenes RSNA del mismo paciente NIH): por construcción no ocurre
    return cand, info


def paso_build(log=print):
    cand, info = preparar(log)
    OUT.mkdir(parents=True, exist_ok=True)
    cand.to_csv(OUT / "candidatos_y_exclusiones.csv", index=False)
    v = cand[cand.excluir == ""].copy()
    v["image_id"] = v.dataset + "__" + v.image_path.str.split("/").str[-1]
    v.loc[v.dataset == "tbx11k", "image_id"] = "tbx11k__" + v.tbx_cat + "_" + v.image_path.str.split("/").str[-1]
    hechos = []
    for seed in SEEDS:
        tr = v[(v.source == "tbx11k") & (v.reserva == "train")]
        neg = tr[tr.clase == 0]
        keep = []
        for cat in ("health", "sick"):
            c = neg[neg.tbx_cat == cat].sort_values("image_path")
            k = N_TBX_NEG // 2
            if len(c) < k:
                raise SystemExit("TBX11K %s: %d imágenes, se piden %d" % (cat, len(c), k))
            keep += c.image_path.to_numpy()[bc.rng("b2_tbx_neg", seed, cat).permutation(len(c))[:k]].tolist()
        d = v.copy()
        d["split"] = d.reserva
        fuera = (d.source == "tbx11k") & (d.reserva == "train") & (d.clase == 0) & ~d.image_path.isin(keep)
        d = d[~fuera].copy()
        d["label"] = d.clase.astype(int)
        d["raiz"] = "proyecto"
        d["val_robust"] = ((d.source == "tbx11k") & (d.split == "val")).astype(int)
        d["test_shenzhen"] = (d.source == "shenzhen").astype(int)
        d["test_montgomery"] = (d.source == "montgomery").astype(int)
        d["test_robusto"] = d.source.isin(["shenzhen", "montgomery"]).astype(int)
        cols = ["dataset", "source", "image_path", "raiz", "patient_id", "label", "split", "image_id"] + SUBSETS + \
               ["reserva", "patient_id_origen", "tbx_cat"]
        d = d[cols].sort_values(["split", "source", "label", "image_id"]).reset_index(drop=True)
        tr = d[d.split == "train"]
        esperado = {}
        for sp in ("train", "val", "test"):
            g = d[d.split == sp].groupby(["source", "label"]).size().unstack(fill_value=0)
            esperado[sp] = {s: [int(g.loc[s].get(0, 0)), int(g.loc[s].get(1, 0))] for s in g.index}
        meta = {"benchmark": "b2", "variante": "B", "rho": None, "semilla": seed, "fuentes": FUENTES,
                "fuentes_mixtas": [s for s in FUENTES if tr[tr.source == s].label.nunique() == 2],
                "subconjuntos": SUBSETS, "seleccion": "val_robust",
                "recuentos_por_fuente_neg_pos": esperado,
                "notas": ("variante B de docs/02_datos.md Section 3.2; la semilla fija la submuestra de 600 no TB de "
                          "TBX11K train (300 sanas y 300 enfermas no TB); val_robust = TBX11K val (fuente mixta). "
                          "Cohorte externa de Mendeley pendiente de descarga."),
                "deduplicacion": info, "test_bloqueado": True, "semilla_reservas": bc.SEMILLA_RESERVAS,
                "generador": "scripts/data/build_b2.py"}
        p = OUT / ("b2_varB_s%d.csv" % seed)
        write_bench(d, meta, p)
        hechos.append(str(p.relative_to(ROOT)))
    union = pd.concat([pd.read_csv(ROOT / h, usecols=["image_id", "image_path", "raiz", "source"]) for h in hechos])
    union = union.drop_duplicates("image_id")
    union.to_csv(OUT / "imagenes_unicas.csv", index=False)
    with open(OUT / "indice.json", "w", encoding="utf-8") as f:
        json.dump({"manifiestos": hechos, "imagenes_unicas": len(union), "deduplicacion": info,
                   "pendiente": "cohorte externa de Mendeley (scripts/download/dl_tb_pakistan_mendeley.sh)"}, f,
                  indent=2)
    log("B2: %d manifiestos, %d imágenes distintas; %s" % (len(hechos), len(union), json.dumps(info)))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--paso", choices=["dedup", "verificar", "build"], required=True)
    a = ap.parse_args(argv)
    if a.paso == "dedup":
        paso_dedup()
        return 0
    if a.paso == "verificar":
        return paso_verificar()
    return paso_build()


if __name__ == "__main__":
    sys.exit(main())
