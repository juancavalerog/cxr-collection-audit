#!/usr/bin/env python3
"""Columna de colección de COVIDx CXR-4 (plan B, aporte C1).

Lee data/manifests/covidx_cxr4_clean.csv (sin modificarlo) y escribe data/manifests/covidx_cxr4_colecciones.csv con una
fila por imagen: image_id, source, label, split_clean, coleccion, proyeccion y modalidad.

Reglas de `coleccion`:
- bimcv con label 0: `bimcv_neg`.
- bimcv con label 1 e identificador de sujeto (sub-S<cifras>) de cinco cifras: `bimcv_pos_antigua`.
- bimcv con label 1 e identificador de seis cifras: `bimcv_pos_nueva`.
- resto de fuentes: el nombre de la fuente.
`proyeccion` sale del campo vp-<x> del nombre de fichero de bimcv (ap, pa, ll; vacío si no viene); `modalidad`, del
sufijo _cr- o _dx-. Fuera de bimcv ambas quedan vacías.

    python3 scripts/data/build_colecciones_covidx.py [--tabla logs/datos_plan_b/colecciones_tabla.md]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "manifests" / "covidx_cxr4_clean.csv"
OUT = ROOT / "data" / "manifests" / "covidx_cxr4_colecciones.csv"

# Tabla E1 de docs/07_piloto.md (partición limpia): (coleccion, split) -> imágenes
TABLA_E1 = {("bimcv_neg", "train"): 647, ("bimcv_neg", "val"): 257, ("bimcv_neg", "test"): 4039,
            ("bimcv_pos_antigua", "train"): 5370, ("bimcv_pos_antigua", "val"): 3150,
            ("bimcv_pos_antigua", "test"): 4040,
            ("bimcv_pos_nueva", "train"): 37123, ("bimcv_pos_nueva", "val"): 0, ("bimcv_pos_nueva", "test"): 0}
PACIENTES_E1 = {("bimcv_neg", "train"): 585, ("bimcv_neg", "val"): 207, ("bimcv_pos_antigua", "train"): 2124,
                ("bimcv_pos_antigua", "val"): 792, ("bimcv_pos_nueva", "train"): 20379}


def colecciones(df: pd.DataFrame) -> pd.DataFrame:
    f = df["image_path"].str.split("/").str[-1]
    out = pd.DataFrame({"image_id": f, "source": df["source"], "label": df["label"].astype(int),
                        "split_clean": df["split_clean"], "patient_id": df["patient_id"]})
    sub = f.str.extract(r"^sub-S(\d+)_")[0]
    bim = df["source"].eq("bimcv")
    if bim.any() and sub[bim].isna().any():
        raise ValueError("imágenes bimcv sin identificador sub-S<cifras>")
    col = df["source"].copy()
    col[bim & out["label"].eq(0)] = "bimcv_neg"
    pos = bim & out["label"].eq(1)
    n = sub.str.len()
    if (pos & ~n.isin([5, 6])).any():
        raise ValueError("positivos bimcv con identificador de longitud distinta de 5 o 6")
    col[pos & n.eq(5)] = "bimcv_pos_antigua"
    col[pos & n.eq(6)] = "bimcv_pos_nueva"
    out["coleccion"] = col
    out["sujeto_cifras"] = n.where(bim).astype("Int64")
    out["proyeccion"] = f.str.extract(r"_vp-([a-z]+)_")[0].where(bim).fillna("")
    out["modalidad"] = f.str.extract(r"_(cr|dx)-corrected")[0].where(bim).fillna("")
    return out


def comprobar_e1(c: pd.DataFrame) -> list[str]:
    err = []
    for (col, sp), n in TABLA_E1.items():
        got = int(((c.coleccion == col) & (c.split_clean == sp)).sum())
        if got != n:
            err.append("Tabla E1 %s/%s: %d en lugar de %d" % (col, sp, got, n))
    for (col, sp), n in PACIENTES_E1.items():
        got = c[(c.coleccion == col) & (c.split_clean == sp)].patient_id.nunique()
        if got != n:
            err.append("Tabla E1 pacientes %s/%s: %d en lugar de %d" % (col, sp, got, n))
    return err


def tabla(c: pd.DataFrame) -> str:
    b = c[c.coleccion.str.startswith("bimcv") & (c.split_clean != "excluded")].copy()
    b["proyeccion"] = b["proyeccion"].replace("", "sin dato")
    g = b.groupby(["coleccion", "split_clean", "proyeccion", "modalidad"]).size().unstack(["proyeccion", "modalidad"],
                                                                                         fill_value=0)
    g = g.reindex(sorted(g.columns), axis=1)
    lines = ["| Colección | Partición | " + " | ".join("%s %s" % k for k in g.columns) + " | Total |",
             "|---|---|" + "---|" * (len(g.columns) + 1)]
    for (col, sp), row in g.iterrows():
        lines.append("| %s | %s | " % (col, sp) + " | ".join("%d" % v for v in row.values) + " | %d |" % row.sum())
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabla", default=str(ROOT / "logs" / "datos_plan_b" / "colecciones_tabla.md"))
    a = ap.parse_args(argv)
    df = pd.read_csv(SRC, dtype=str)
    c = colecciones(df)
    err = comprobar_e1(c)
    for e in err:
        print("ERROR", e)
    if err:
        return 1
    c.drop(columns=["patient_id"]).to_csv(OUT, index=False)
    Path(a.tabla).parent.mkdir(parents=True, exist_ok=True)
    Path(a.tabla).write_text(tabla(c) + "\n", encoding="utf-8")
    print("escrito %s (%d filas); Tabla E1 reproducida" % (OUT, len(c)))
    print(tabla(c))
    return 0


if __name__ == "__main__":
    sys.exit(main())
