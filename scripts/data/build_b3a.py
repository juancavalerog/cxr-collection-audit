#!/usr/bin/env python3
"""Benchmark B3a del plan B: derrame pleural con confusión controlada entre clase y fuente.

Diseño: docs/02_datos.md Section 3.3 (variante 3a) con los añadidos de PLAN.md Section 13.3.
Fuente A = NIH ChestX-ray14 (`nih`), fuente B = CheXpert-small frontales (`chexpert`).

Pasos (cada uno cabe en una llamada de 180 s y guarda su resultado en disco):

    python3 scripts/data/build_b3a.py --paso dedup     # casi duplicados entre pacientes del conjunto candidato
    python3 scripts/data/build_b3a.py --paso verificar # correlación de miniaturas; repetir hasta código 0
    python3 scripts/data/build_b3a.py --paso build     # reservas, manifiestos y JSON

Definiciones.
- Positivo: NIH `Effusion` en Finding Labels; CheXpert `Pleural Effusion` = 1 (se excluye -1).
- Negativo: NIH `No Finding`; CheXpert `No Finding` = 1 sin derrame 1 ni -1 en la imagen (vacío o 0).
- Clase por paciente: positivo si tiene alguna imagen frontal positiva; negativo si no la tiene, tiene alguna
  negativa y, en CheXpert, ninguna imagen frontal con derrame incierto (-1). Solo vistas AP y PA.
- Reservas de pacientes fijas (semilla 20261114) por fuente: train 70 %, val 10 %, test 20 %, estratificadas por
  clase y por vistas disponibles (solo AP, solo PA, ambas), para que la cuota AP sea factible en todas las reservas.
- Una imagen por paciente y vista, elegida de forma determinista; cada conjunto tiene una sola imagen por paciente.
- Fracción AP igual entre clases dentro de cada fuente: NIH 0.5, CheXpert 0.7, en todos los conjuntos.
- Prevalencia: orientación A positiva, rho en NIH y 1 - rho en CheXpert; orientación B positiva, al revés.
- train: 2 800 por fuente; val: 400 por fuente con la misma prevalencia (dependen de la semilla).
- val_desconf: 200 por celda fuente x clase de la reserva de validación, fijo; las filas que no están en `val`
  llevan split = fuera.
- test_desconf: 500 positivas y 500 negativas por fuente, fijo. test_invertido: 1 000 por fuente con prevalencia
  invertida (1 - rho en la fuente enriquecida y rho en la otra), prefijo de la misma ordenación que test_desconf.
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

OUT = ROOT / "data" / "manifests" / "bench" / "b3a"
RHOS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
SEEDS = (0, 1, 2, 3, 4)
RHOS_B = (0.5, 0.8, 1.0)
SEEDS_B = (0, 1)
FUENTES = ("nih", "chexpert")
F_AP = {"nih": 0.5, "chexpert": 0.7}
N_TRAIN, N_VAL, N_VALD, N_TESTD, N_TESTI = 2800, 400, 200, 500, 1000
SUBSETS = ["val_desconf", "test_desconf", "test_invertido"]


def rnd(x: float) -> int:
    return int(np.floor(x + 0.5 + 1e-9))


# ------------------------------------------------------------------------------------------ candidatos
def candidatos() -> pd.DataFrame:
    """Una fila por imagen candidata: source, dataset, image_path, patient_id, view, clase (1/0)."""
    m = ROOT / "data" / "manifests"
    n = pd.read_csv(m / "nih_cxr14.csv", usecols=["image_path", "patient_id", "view", "y_effusion", "y_no_finding"],
                    dtype={"patient_id": str})
    pos_p = set(n.loc[n.y_effusion == 1, "patient_id"])
    neg_p = set(n.loc[n.y_no_finding == 1, "patient_id"]) - pos_p
    a = pd.concat([n[(n.y_effusion == 1) & n.patient_id.isin(pos_p)].assign(clase=1),
                   n[(n.y_no_finding == 1) & n.patient_id.isin(neg_p)].assign(clase=0)])
    a = a.assign(source="nih", dataset="nih_cxr14")[["source", "dataset", "image_path", "patient_id", "view", "clase"]]
    c = pd.read_csv(m / "chexpert_small.csv", usecols=["image_path", "patient_id", "view", "ap_pa",
                                                       "y_pleural_effusion", "y_no_finding"], dtype={"patient_id": str})
    c = c[(c.view == "Frontal") & c.ap_pa.isin(["AP", "PA"])]
    pos_p = set(c.loc[c.y_pleural_effusion == 1, "patient_id"])
    unc_p = set(c.loc[c.y_pleural_effusion == -1, "patient_id"])
    isneg = (c.y_no_finding == 1) & ~c.y_pleural_effusion.isin([1, -1])
    neg_p = set(c.loc[isneg, "patient_id"]) - pos_p - unc_p
    b = pd.concat([c[(c.y_pleural_effusion == 1) & c.patient_id.isin(pos_p)].assign(clase=1),
                   c[isneg & c.patient_id.isin(neg_p)].assign(clase=0)])
    b = b.assign(source="chexpert", dataset="chexpert_small", view=b.ap_pa)
    b = b[["source", "dataset", "image_path", "patient_id", "view", "clase"]]
    return pd.concat([a, b], ignore_index=True)


def image_id(source, path):
    if source == "nih":
        return "nih__" + path.split("/")[-1]
    return "chx__" + path.split("CheXpert-v1.0-small/")[-1].replace("/", "_")


# ------------------------------------------------------------------------------------------ paso dedup
def paso_dedup(log=print):
    cand = candidatos()
    hn = bc.cargar_hashes(ROOT / "logs" / "tmp_datos" / "hash_nih_cxr14.csv")
    hc = bc.cargar_hashes(ROOT / "data" / "manifests" / "bench" / "hashes" / "chexpert_pool.csv")
    h = pd.concat([hn, hc], ignore_index=True)
    u = cand.drop_duplicates("image_path")
    h = h.merge(u[["image_path", "source", "patient_id"]], on="image_path", how="inner")
    falta = set(u.image_path) - set(h.image_path)
    log("imágenes candidatas %d, con hash %d, sin hash %d" % (len(u), len(h), len(falta)))
    if falta:
        raise SystemExit("faltan hashes de %d imágenes candidatas (p. ej. %s)" % (len(falta), sorted(falta)[0]))
    h["grupo"] = h.source + ":" + h.patient_id
    root = data_root()
    t = time.time()
    pares = bc.casi_duplicados(h, lambda p: str(root / p), grupo_col="grupo", verificar=False, log=log)
    g = dict(zip(h.image_path, h.grupo))
    pares["grupo_a"], pares["grupo_b"] = pares.a.map(g), pares.b.map(g)
    OUT.mkdir(parents=True, exist_ok=True)
    prev = OUT / "duplicados_hash.csv"
    if prev.exists():       # reutiliza verificaciones ya hechas de los mismos pares
        old = pd.read_csv(prev)[["a", "b", "thumb_corr"]].dropna()
        pares = pares.drop(columns="thumb_corr").merge(old, on=["a", "b"], how="left")
        pares["confirmed"] = (pares.kind == "md5") | (pares.thumb_corr >= 0.97)
    pares.to_csv(prev, index=False)
    log("pares candidatos entre pacientes distintos: %d (%.0f s); falta --paso verificar" % (len(pares), time.time() - t))


def paso_verificar(max_s=150, log=print):
    p = OUT / "duplicados_hash.csv"
    root = data_root()
    n = bc.verificar_pares(p, lambda x: str(root / x), max_s=max_s, log=log)
    return 0 if n == 0 else 3


def pacientes_excluidos(cand: pd.DataFrame) -> pd.DataFrame:
    """Pacientes que se retiran por casi duplicados confirmados con otro paciente: dentro de cada grupo conexo se
    conserva el menor identificador (fuente:paciente) si todos tienen la misma clase; si las clases difieren, se
    retiran todos."""
    p = OUT / "duplicados_hash.csv"
    if not p.exists():
        raise SystemExit("falta %s: ejecute antes --paso dedup" % p)
    d = pd.read_csv(p)
    if d.thumb_corr.isna().any():
        raise SystemExit("hay pares sin verificar en %s: ejecute --paso verificar" % p)
    d = d[d.confirmed]
    if d.empty:
        return pd.DataFrame(columns=["grupo", "motivo"])
    rep = bc.grupos_union(d, "grupo_a", "grupo_b")
    cl = (cand.source + ":" + cand.patient_id).to_frame("grupo").assign(clase=cand.clase.values).drop_duplicates()
    cl = dict(zip(cl.grupo, cl.clase))
    comp = pd.DataFrame({"grupo": list(rep), "rep": list(rep.values())})
    comp["clase"] = comp.grupo.map(cl)
    rows = []
    for r, gg in comp.groupby("rep"):
        if gg.clase.nunique() > 1:
            rows += [(x, "duplicado con clase distinta") for x in gg.grupo]
        else:
            rows += [(x, "duplicado de %s" % r) for x in gg.grupo if x != r]
    return pd.DataFrame(rows, columns=["grupo", "motivo"])


# ------------------------------------------------------------------------------------------ paso build
def preparar(log=print):
    cand = candidatos()
    exc = pacientes_excluidos(cand)
    g = cand.source + ":" + cand.patient_id
    cand = cand[~g.isin(set(exc.grupo))].copy()
    log("pacientes retirados por casi duplicados: %d" % len(exc))
    img = bc.imagen_fija(cand, claves=("source", "patient_id", "clase", "view"), salt="b3a")
    pac = cand.groupby(["source", "patient_id", "clase"]).view.agg(lambda s: set(s)).reset_index()
    pac["has_ap"], pac["has_pa"] = pac.view.map(lambda s: "AP" in s), pac.view.map(lambda s: "PA" in s)
    pac["vistas"] = np.where(pac.has_ap & pac.has_pa, "ambas", np.where(pac.has_ap, "solo_ap", "solo_pa"))
    pac = pac.drop(columns="view")
    pac["reserva"] = ""
    for s in FUENTES:
        m = pac.source == s
        pac.loc[m, "reserva"] = bc.reservas_pacientes(pac[m], ["clase", "vistas"], clave="b3a_" + s).values
    return img, pac, exc


class Ordenes:
    def __init__(self, pac):
        self.pac, self.cache = pac, {}

    def get(self, src, cls, reserva, *tag):
        k = (src, cls, reserva) + tag
        if k not in self.cache:
            p = self.pac[(self.pac.source == src) & (self.pac.clase == cls) & (self.pac.reserva == reserva)]
            self.cache[k] = bc.orden_ap_pa(p, src, cls, reserva, *tag)
        return self.cache[k]


def conjunto(ordenes, img_idx, src, reserva, n_pos, n_neg, tag):
    rows = []
    for cls, n in ((1, n_pos), (0, n_neg)):
        if n == 0:
            continue
        for pid, v in bc.muestra_ap(ordenes.get(src, cls, reserva, *tag), n, F_AP[src]):
            rows.append(img_idx[(src, pid, cls, v)])
    return rows


def prev_fuente(orient, rho):
    """Prevalencia de entrenamiento por fuente."""
    pa = rho if orient == "apos" else 1.0 - rho
    return {"nih": pa, "chexpert": 1.0 - pa}


def tamano_invertido(ordenes, prev):
    """Tamaño por fuente del test invertido: N_TESTI si es factible en las dos fuentes; si no, el mayor tamaño común
    factible (ajuste mínimo, documentado en docs/11_benchmarks_plan_b.md)."""
    for n in range(N_TESTI, 0, -1):
        ok = True
        for s in FUENTES:
            npos = rnd(n * (1.0 - prev[s]))
            for cls, k in ((1, npos), (0, n - npos)):
                o = ordenes.get(s, cls, "test", "fijo")
                kap = rnd(k * F_AP[s])
                ok &= bc.factible(kap, k - kap, o[2], o[3], o[4])
        if ok:
            return n
    raise ValueError("test invertido no factible")


def construir_manifiesto(orient, rho, seed, ordenes, img, img_idx):
    prev = prev_fuente(orient, rho)
    sel = {}   # índice de img -> dict de banderas
    def add(ix, split=None, **flags):
        e = sel.setdefault(ix, {"split": None, "val_desconf": 0, "test_desconf": 0, "test_invertido": 0})
        if split:
            e["split"] = split
        for k, v in flags.items():
            e[k] = v
    for s in FUENTES:
        npos = rnd(N_TRAIN * prev[s])
        for ix in conjunto(ordenes, img_idx, s, "train", npos, N_TRAIN - npos, ("semilla", seed)):
            add(ix, "train")
        npos = rnd(N_VAL * prev[s])
        for ix in conjunto(ordenes, img_idx, s, "val", npos, N_VAL - npos, ("semilla", seed)):
            add(ix, "val")
        for ix in conjunto(ordenes, img_idx, s, "val", N_VALD, N_VALD, ("fijo",)):
            add(ix, None, val_desconf=1)
        for ix in conjunto(ordenes, img_idx, s, "test", N_TESTD, N_TESTD, ("fijo",)):
            add(ix, "test", test_desconf=1)
    n_inv = tamano_invertido(ordenes, prev)
    for s in FUENTES:
        npos = rnd(n_inv * (1.0 - prev[s]))
        for ix in conjunto(ordenes, img_idx, s, "test", npos, n_inv - npos, ("fijo",)):
            add(ix, "test", test_invertido=1)
    d = img.loc[list(sel)].copy()
    for k in ("split",) + tuple(SUBSETS):
        d[k] = [sel[i][k] for i in d.index]
    d["split"] = d["split"].fillna("fuera")
    d["label"] = d.clase.astype(int)
    d["raiz"] = "datos"
    d["image_id"] = [image_id(s, p) for s, p in zip(d.source, d.image_path)]
    d["orientacion"] = orient
    d = d.drop(columns=["clase"]).sort_values(["split", "source", "label", "image_id"]).reset_index(drop=True)
    mixtas = [s for s in FUENTES if d[d.split == "train"].query("source == @s").label.nunique() == 2]
    meta = {
        "benchmark": "b3a", "variante": "3a_derrame_pleural", "orientacion": orient, "rho": rho, "semilla": seed,
        "fuentes": list(FUENTES), "fuentes_mixtas": mixtas, "subconjuntos": SUBSETS, "seleccion": "val",
        "seleccion_alternativa": "val_desconf",
        "prevalencia_train_por_fuente": prev, "fraccion_ap_por_fuente": F_AP,
        "tamanos": {"train_por_fuente": N_TRAIN, "val_por_fuente": N_VAL, "val_desconf_por_celda": N_VALD,
                    "test_desconf_por_celda": N_TESTD, "test_invertido_por_fuente": n_inv,
                    "test_invertido_por_fuente_diseno": N_TESTI},
        "notas": ("val_desconf puede compartir filas con val (las filas solo de val_desconf llevan split=fuera); "
                  "test_desconf y test_invertido salen de la misma ordenación fija de la reserva de test y "
                  "comparten filas; un paciente con AP y PA puede aportar imágenes distintas a dos subconjuntos, "
                  "nunca dos imágenes al mismo conjunto ni pacientes entre reservas."),
        "test_bloqueado": True, "semilla_reservas": bc.SEMILLA_RESERVAS,
        "generador": "scripts/data/build_b3a.py",
    }
    return d, meta


def nombre(orient, rho, seed):
    return OUT / orient / ("rho%.2f_s%d.csv" % (rho, seed))


def paso_build(log=print):
    img, pac, exc = preparar(log)
    img = img.merge(pac[["source", "patient_id", "reserva"]], on=["source", "patient_id"], how="left")
    img = img.reset_index(drop=True)
    img_idx = {(s, p, c, v): i for i, (s, p, c, v) in enumerate(zip(img.source, img.patient_id, img.clase, img.view))}
    OUT.mkdir(parents=True, exist_ok=True)
    exc.to_csv(OUT / "pacientes_excluidos_hash.csv", index=False)
    pac.assign(clase=pac.clase.astype(int)).to_csv(OUT / "reservas_pacientes.csv", index=False)
    resumen = pac.groupby(["source", "reserva", "clase", "vistas"]).size().unstack("vistas", fill_value=0)
    log(resumen.to_string())
    ordenes = Ordenes(pac)
    hechos, fallos = [], []
    combos = [("apos", r, s) for r in RHOS for s in SEEDS] + [("bpos", r, s) for r in RHOS_B for s in SEEDS_B]
    for orient, rho, seed in combos:
        try:
            d, meta = construir_manifiesto(orient, rho, seed, ordenes, img, img_idx)
        except ValueError as e:
            fallos.append((orient, rho, seed, str(e)))
            log("NO FACTIBLE %s rho=%.2f s=%d: %s" % (orient, rho, seed, e))
            continue
        write_bench(d, meta, nombre(orient, rho, seed))
        hechos.append(str(nombre(orient, rho, seed).relative_to(ROOT)))
    union = pd.concat([pd.read_csv(ROOT / h, usecols=["image_id", "image_path", "raiz", "source"]) for h in hechos])
    union = union.drop_duplicates("image_id")
    union.to_csv(OUT / "imagenes_unicas.csv", index=False)
    log("manifiestos escritos: %d; no factibles: %d; imágenes distintas: %d" % (len(hechos), len(fallos), len(union)))
    with open(OUT / "indice.json", "w", encoding="utf-8") as f:
        json.dump({"manifiestos": hechos, "no_factibles": fallos, "imagenes_unicas": len(union)}, f, indent=2)
    return 0 if not fallos else 1


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
