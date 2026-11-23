#!/usr/bin/env python3
"""Cifras descriptivas de docs/11_benchmarks_plan_b.md, Section 12 (datos de los experimentos adicionales). Solo lee
manifiestos y etiquetas; escribe logs/datos_extra/resumen_datos_extra.txt.

    python3 scripts/data/resumen_datos_extra.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.bench import list_manifests, read_csv  # noqa: E402

B = ROOT / "data" / "manifests" / "bench"
OUT = ROOT / "logs" / "datos_extra" / "resumen_datos_extra.txt"


def main():
    L = []
    a_all = list_manifests(B / "b3a")
    nuevos = [p for p in a_all if int(p.stem.split("_s")[1]) >= 5]
    orig = [p for p in a_all if p not in nuevos]
    L.append("B3a: %d manifiestos (%d originales, %d nuevos)" % (len(a_all), len(orig), len(nuevos)))
    L.append(json.dumps(json.loads((ROOT / "logs/datos_extra/b3a_s5_9_delta.json").read_text()), ensure_ascii=False))
    # tabla por nivel de los nuevos (semilla 5) y comprobación con la 0
    for rho in ("0.90", "0.95", "0.99"):
        d = read_csv(B / ("b3a/apos/rho%s_s5.csv" % rho))
        t = d[d.split.isin(["train", "val"])].groupby(["split", "source", "label"]).size().unstack("label")
        L.append("B3a rho %s semilla 5 (negativas, positivas): %s" % (rho, {k: tuple(v) for k, v in t.iterrows()}))
    # solape de train entre semillas: fracción media de imágenes de train compartidas con la semilla 0
    for rho in ("0.90", "0.95", "0.99"):
        s0 = set(read_csv(B / ("b3a/apos/rho%s_s0.csv" % rho)).query("split == 'train'").image_id)
        fr = [len(s0 & set(read_csv(B / ("b3a/apos/rho%s_s%d.csv" % (rho, s))).query("split == 'train'").image_id))
              / len(s0) for s in range(1, 10)]
        L.append("B3a rho %s: fracción de train compartida con la semilla 0, semillas 1 a 9: %s"
                 % (rho, " ".join("%.3f" % x for x in fr)))
    c = list_manifests(B / "b3c")
    dc = pd.concat([read_csv(p) for p in c]).drop_duplicates("image_id")
    da = pd.concat([read_csv(p)[["image_id", "patient_id", "source", "label"]] for p in a_all]).drop_duplicates("image_id")
    L.append("B3c: %d manifiestos, %d imágenes distintas (%s)" % (len(c), len(dc), dc.source.value_counts().to_dict()))
    sh = dc.merge(da, on="image_id", suffixes=("", "_a"))
    L.append("Imágenes de B3c que también están en algún manifiesto de B3a: %d (por fuente y clase en B3c: %s; "
             "clase en B3a: %s)" % (len(sh), sh.groupby(["source", "label"]).size().to_dict(),
                                     sh.groupby(["label", "label_a"]).size().to_dict()))
    pa = set(zip(da.source, da.patient_id))
    pc = set(zip(dc.source, dc.patient_id))
    L.append("Pacientes de B3c usados que también se usan en B3a: %d de %d" % (len(pc & pa), len(pc)))
    r = pd.read_csv(B / "b3c/reservas_pacientes.csv", dtype={"patient_id": str})
    L.append("Reservas de B3c: %d pacientes; heredados de B3a %d; nuevos %d (%s)" % (
        len(r), int(r.heredada.sum()), int((~r.heredada).sum()),
        r[~r.heredada].groupby(["source", "clase"]).size().to_dict()))
    L.append("Nuevos por reserva: %s" % r[~r.heredada].groupby(["source", "clase", "reserva"]).size().to_dict())
    L.append("Fracción por reserva (train/val/test) por fuente y clase: %s" % {
        k: tuple(round(x, 3) for x in (g.reserva.value_counts(normalize=True).reindex(["train", "val", "test"])))
        for k, g in r.groupby(["source", "clase"])})
    ex = pd.read_csv(B / "b3c/pacientes_excluidos_hash.csv")
    L.append("Pacientes retirados en B3c: %d (%s)" % (len(ex), ex.motivo.str.replace(r" de .*", "", regex=True)
                                                       .value_counts().to_dict()))
    dup = pd.read_csv(B / "b3c/duplicados_hash.csv")
    L.append("Pares candidatos de B3c: %d; confirmados %d (%s); mediana de correlación de los candidatos por pHash %.2f"
             % (len(dup), int(dup.confirmed.sum()), dup[dup.confirmed].kind.value_counts().to_dict(),
                dup[dup.kind == "phash"].thumb_corr.median()))
    # prevalencias del conjunto candidato (pacientes)
    m = ROOT / "data" / "manifests"
    n = pd.read_csv(m / "nih_cxr14.csv", usecols=["patient_id", "y_cardiomegaly", "y_effusion", "y_no_finding"],
                    dtype={"patient_id": str})
    x = pd.read_csv(m / "chexpert_small.csv", usecols=["patient_id", "view", "ap_pa", "y_cardiomegaly",
                                                       "y_pleural_effusion", "y_no_finding"], dtype={"patient_id": str})
    x = x[(x.view == "Frontal") & x.ap_pa.isin(["AP", "PA"])]
    L.append("NIH: imágenes con Cardiomegaly %d (%.1f %%), pacientes %d; con Effusion %d imágenes"
             % ((n.y_cardiomegaly == 1).sum(), 100 * (n.y_cardiomegaly == 1).mean(),
                n[n.y_cardiomegaly == 1].patient_id.nunique(), (n.y_effusion == 1).sum()))
    L.append("CheXpert frontal AP/PA: imágenes con Cardiomegaly 1 %d, -1 %d, 0 %d, vacío %d; pacientes con algún -1: %d"
             % ((x.y_cardiomegaly == 1).sum(), (x.y_cardiomegaly == -1).sum(), (x.y_cardiomegaly == 0).sum(),
                x.y_cardiomegaly.isna().sum(), x[x.y_cardiomegaly == -1].patient_id.nunique()))
    pos = dc[dc.label == 1]
    L.append("Positivas de B3c con derrame anotado en su fuente: %s" % {
        "nih": float(pos[pos.source == "nih"].merge(pd.read_csv(m / "nih_cxr14.csv", usecols=["image_path", "y_effusion"]),
                                                    on="image_path").y_effusion.eq(1).mean().round(3)),
        "chexpert": float(pos[pos.source == "chexpert"].merge(pd.read_csv(m / "chexpert_small.csv",
                                                                          usecols=["image_path", "y_pleural_effusion"]),
                                                              on="image_path").y_pleural_effusion.eq(1).mean().round(3))})
    t = pd.read_csv(ROOT / "data/masks/teacher_set/selection.csv", usecols=["image_path"])
    tp = set(t.image_path)
    dc["prof"] = dc.image_path.isin(tp)
    L.append("Imágenes de B3c que el profesor de segmentación vio: %s" % {
        k: round(float(v), 3) for k, v in dc.groupby(["source", "label"]).prof.mean().items()})
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
