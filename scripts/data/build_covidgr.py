#!/usr/bin/env python3
"""Cohorte externa COVIDGR 1.0 (un hospital, PA): manifiesto de solo evaluación (split = test).

    python3 scripts/data/build_covidgr.py --paso dedup       # casi duplicados dentro de COVIDGR y frente a COVIDx
    python3 scripts/data/build_covidgr.py --paso verificar   # repetir hasta código 0
    python3 scripts/data/build_covidgr.py --paso build       # manifiesto, JSON y descripción de las imágenes

Clases: carpeta P (positiva, RT-PCR) y N (negativa). Gravedad de las positivas desde severity.csv (NORMAL-PCR+, MILD,
MODERATE, SEVERE). El repositorio no publica identificador de paciente: patient_id = nombre del fichero (md5 del
original), y los pacientes repetidos solo pueden detectarse como imágenes casi idénticas.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_common as bc  # noqa: E402
from src.data.bench import write_bench  # noqa: E402
from src.data.paths import data_root  # noqa: E402

OUT = ROOT / "data" / "manifests" / "bench" / "covidgr"
RAW = "data/raw/covidgr/"


def abs_any(p):
    return str(ROOT / p) if p.startswith("data/raw/") else str(data_root() / p)


def imagenes():
    rows = []
    for cls, lab in (("P", 1), ("N", 0)):
        for p in sorted((ROOT / RAW / cls).glob("*")):
            rows.append({"image_path": RAW + "%s/%s" % (cls, p.name), "label": lab, "stem": p.stem})
    d = pd.DataFrame(rows)
    sev = pd.read_csv(ROOT / RAW / "severity.csv")
    d["severidad"] = d.stem.map(dict(zip(sev.Name, sev.Severity))).fillna("")
    return d


def paso_dedup(log=print):
    g = bc.cargar_hashes(ROOT / "data/manifests/bench/hashes/covidgr.csv", RAW)
    x = bc.cargar_hashes(ROOT / "logs/tmp_datos/hash_covidx_all.csv")
    h = pd.concat([g.assign(grupo=g.image_path, origen="covidgr"), x.assign(grupo=x.image_path, origen="covidx")],
                  ignore_index=True)
    p = bc.casi_duplicados(h, abs_any, grupo_col="grupo", verificar=False, log=log)
    p = p[p.a.str.startswith(RAW) | p.b.str.startswith(RAW)]
    OUT.mkdir(parents=True, exist_ok=True)
    p.to_csv(OUT / "duplicados_hash.csv", index=False)
    log("pares candidatos con alguna imagen de COVIDGR: %d" % len(p))


def describir(path):
    with Image.open(ROOT / path) as im:
        w, h, mode = im.size[0], im.size[1], im.mode
        a = np.asarray(im.convert("L"), dtype=np.uint8)
    neg = a <= 5
    b = max(1, int(0.05 * min(a.shape)))
    borde = np.concatenate([a[:b].ravel(), a[-b:].ravel(), a[:, :b].ravel(), a[:, -b:].ravel()])
    hh = a.shape[0]
    centro = a[hh // 4: 3 * hh // 4, a.shape[1] // 4: 3 * a.shape[1] // 4]
    return {"width": w, "height": h, "mode": mode, "aspecto": round(h / w, 3), "frac_negro": float(neg.mean()),
            "frac_negro_borde": float((borde <= 5).mean()), "frac_negro_centro": float((centro <= 5).mean()),
            "media": float(a.mean())}


def paso_build(log=print):
    d = imagenes()
    p = pd.read_csv(OUT / "duplicados_hash.csv")
    if p.thumb_corr.isna().any():
        raise SystemExit("faltan verificaciones: --paso verificar")
    c = p[p.confirmed]
    dentro = c[c.a.str.startswith(RAW) & c.b.str.startswith(RAW)]
    covidx = c[~(c.a.str.startswith(RAW) & c.b.str.startswith(RAW))]
    log("casi duplicados confirmados: dentro de COVIDGR %d, frente a COVIDx %d" % (len(dentro), len(covidx)))
    desc = pd.DataFrame([describir(x) for x in d.image_path])
    d = pd.concat([d, desc], axis=1)
    d["dataset"], d["source"], d["raiz"], d["split"] = "covidgr", "covidgr", "proyecto", "test"
    d["image_id"] = "covidgr__" + d.label.map({1: "P", 0: "N"}) + "_" + d.image_path.str.split("/").str[-1]
    # pacientes: sin identificador; las imágenes casi idénticas se agrupan como un mismo paciente
    rep = bc.grupos_union(dentro) if len(dentro) else {}
    d["patient_id"] = "covidgr:" + d.image_path.map(lambda x: rep.get(x, x)).str.split("/").str[-1]
    d["dup_covidx"] = d.image_path.isin(set(covidx.a) | set(covidx.b)).astype(int)
    d["test_covidgr"] = 1
    d["test_covidgr_sin_dup"] = (d.dup_covidx == 0).astype(int)
    subs = ["test_covidgr", "test_covidgr_sin_dup"]
    if d.test_covidgr_sin_dup.sum() == len(d):
        subs = ["test_covidgr"]
        d = d.drop(columns="test_covidgr_sin_dup")
    meta = {"benchmark": "covidgr", "variante": "1.0", "rho": None, "semilla": None, "fuentes": ["covidgr"],
            "fuentes_mixtas": ["covidgr"], "subconjuntos": subs, "seleccion": "test_covidgr",
            "notas": ("solo evaluación externa (split = test); seleccion no aplica: ningún modelo se selecciona en "
                      "COVIDGR. Sin identificador de paciente publicado."),
            "recuentos_por_fuente_neg_pos": {"test": {"covidgr": [int((d.label == 0).sum()), int(d.label.sum())]}},
            "duplicados": {"dentro": int(len(dentro)), "frente_a_covidx": int(len(covidx))},
            "test_bloqueado": True, "generador": "scripts/data/build_covidgr.py"}
    cols = ["dataset", "source", "image_path", "raiz", "patient_id", "label", "split", "image_id"] + subs + \
           ["severidad", "dup_covidx", "width", "height", "mode", "aspecto", "frac_negro", "frac_negro_borde",
            "frac_negro_centro", "media"]
    write_bench(d[cols], meta, OUT / "covidgr.csv")
    g = d.groupby("label")[["width", "height", "aspecto", "frac_negro", "frac_negro_borde", "frac_negro_centro",
                            "media"]].describe(percentiles=[0.1, 0.5, 0.9]).T
    g.to_csv(OUT / "descripcion_imagenes.csv")
    log(d.groupby(["label", "severidad"]).size().to_string())
    log(g.to_string())
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--paso", choices=["dedup", "verificar", "build"], required=True)
    a = ap.parse_args(argv)
    if a.paso == "dedup":
        paso_dedup()
        return 0
    if a.paso == "verificar":
        return 0 if bc.verificar_pares(OUT / "duplicados_hash.csv", abs_any) == 0 else 3
    return paso_build()


if __name__ == "__main__":
    sys.exit(main())
