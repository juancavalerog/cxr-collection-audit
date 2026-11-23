#!/usr/bin/env python3
"""Benchmark B3c: cardiomegalia con confusión controlada entre clase y fuente (sensibilidad a la etiqueta, S-E;
docs/08_evaluacion.md, enmienda E8, punto 4; PLAN.md Section 13.12).

Mismas fuentes, vistas y reglas que B3a (scripts/data/build_b3a.py, cuyas funciones se reutilizan sin cambiarlas),
con la etiqueta de cardiomegalia. Fuente A = NIH ChestX-ray14 (`nih`), fuente B = CheXpert-small frontales
(`chexpert`). Solo orientación A positiva, rho 0.50, 0.90, 0.99 y 1.00, semillas 0 a 2.

Pasos (cada uno cabe en una llamada de 180 s y guarda su resultado en disco):

    python3 scripts/data/build_b3c.py --paso hashes        # hashes de las imágenes de CheXpert que faltan; repetir
                                                            # hasta código 0
    python3 scripts/data/build_b3c.py --paso dedup         # casi duplicados entre pacientes del conjunto candidato
    python3 scripts/data/build_b3c.py --paso verificar     # correlación de miniaturas; repetir hasta código 0
    python3 scripts/data/build_b3c.py --paso factibilidad  # reservas, factor de tamaño K y recuentos, sin escribir
    python3 scripts/data/build_b3c.py --paso build         # reservas, manifiestos y JSON
    python3 scripts/data/build_b3c.py --paso tabla         # tabla de recuentos regenerable (logs/datos_extra)

Definiciones (las de B3a con la etiqueta cambiada).
- Positivo: NIH `Cardiomegaly` en Finding Labels (y_cardiomegaly = 1); CheXpert `Cardiomegaly` = 1 (se excluye -1).
- Negativo: NIH `No Finding` (y_no_finding = 1); CheXpert `No Finding` = 1 sin cardiomegalia 1 ni -1 en la imagen.
- Clase por paciente: positivo si tiene alguna imagen frontal positiva; negativo si no la tiene, tiene alguna
  negativa y, en CheXpert, ninguna imagen frontal con cardiomegalia incierta (-1). Solo vistas AP y PA.
- Reservas de pacientes: las de B3a para todo paciente que ya tiene reserva en B3a (data/manifests/bench/b3a/
  reservas_pacientes.csv, por fuente e identificador); los pacientes nuevos se reparten dentro de cada estrato
  fuente x clase x vistas disponibles para que el estrato quede lo más cerca posible del 70/10/20 (función
  `reservas_heredadas`). Se retiran además los 8 pacientes que B3a retiró por casi duplicados.
- Una imagen por paciente y vista, elegida de forma determinista (sal "b3c").
- Fracción AP igual entre clases dentro de cada fuente: NIH 0.5, CheXpert 0.7.
- Tamaños: los de B3a multiplicados por un factor común K (regla de docs/11_benchmarks_plan_b.md, parte
  "Datos de los experimentos adicionales", Section 12.1): K es el mayor múltiplo de 0.05 en (0, 1] con el que todos
  los conjuntos de los cuatro niveles son factibles con las cuotas AP y PA.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_common as bc  # noqa: E402
import build_b3a as B  # noqa: E402
from src.data.bench import write_bench  # noqa: E402
from src.data.paths import data_root  # noqa: E402

OUT = ROOT / "data" / "manifests" / "bench" / "b3c"
B3A = ROOT / "data" / "manifests" / "bench" / "b3a"
HASH_EXTRA = OUT / "hashes_chexpert_b3c.csv"
LOGDIR = ROOT / "logs" / "datos_extra"
RHOS = (0.5, 0.9, 0.99, 1.0)
SEEDS = (0, 1, 2)
ORIENT = "apos"
FUENTES = B.FUENTES
F_AP = B.F_AP
BASE = {"train": 2800, "val": 400, "val_desconf": 200, "test_desconf": 500, "test_invertido": 1000}
SUBSETS = B.SUBSETS
TAG = "b3c"
rnd = B.rnd


# ------------------------------------------------------------------------------------------ candidatos
def candidatos() -> pd.DataFrame:
    """Una fila por imagen candidata: source, dataset, image_path, patient_id, view, clase (1/0)."""
    m = ROOT / "data" / "manifests"
    n = pd.read_csv(m / "nih_cxr14.csv", usecols=["image_path", "patient_id", "view", "y_cardiomegaly",
                                                  "y_no_finding"], dtype={"patient_id": str})
    pos_p = set(n.loc[n.y_cardiomegaly == 1, "patient_id"])
    neg_p = set(n.loc[n.y_no_finding == 1, "patient_id"]) - pos_p
    a = pd.concat([n[(n.y_cardiomegaly == 1) & n.patient_id.isin(pos_p)].assign(clase=1),
                   n[(n.y_no_finding == 1) & n.patient_id.isin(neg_p)].assign(clase=0)])
    a = a.assign(source="nih", dataset="nih_cxr14")[["source", "dataset", "image_path", "patient_id", "view", "clase"]]
    c = pd.read_csv(m / "chexpert_small.csv", usecols=["image_path", "patient_id", "view", "ap_pa",
                                                       "y_cardiomegaly", "y_no_finding"], dtype={"patient_id": str})
    c = c[(c.view == "Frontal") & c.ap_pa.isin(["AP", "PA"])]
    pos_p = set(c.loc[c.y_cardiomegaly == 1, "patient_id"])
    unc_p = set(c.loc[c.y_cardiomegaly == -1, "patient_id"])
    isneg = (c.y_no_finding == 1) & ~c.y_cardiomegaly.isin([1, -1])
    neg_p = set(c.loc[isneg, "patient_id"]) - pos_p - unc_p
    b = pd.concat([c[(c.y_cardiomegaly == 1) & c.patient_id.isin(pos_p)].assign(clase=1),
                   c[isneg & c.patient_id.isin(neg_p)].assign(clase=0)])
    b = b.assign(source="chexpert", dataset="chexpert_small", view=b.ap_pa)
    b = b[["source", "dataset", "image_path", "patient_id", "view", "clase"]]
    return pd.concat([a, b], ignore_index=True)


def excluidos_b3a() -> set:
    p = B3A / "pacientes_excluidos_hash.csv"
    return set(pd.read_csv(p, dtype=str).grupo) if p.exists() else set()


# ------------------------------------------------------------------------------------------ paso hashes
def paso_hashes(max_s=140, log=print) -> int:
    cand = candidatos()
    hc = pd.read_csv(ROOT / "data" / "manifests" / "bench" / "hashes" / "chexpert_pool.csv", usecols=["image_path"])
    falta = sorted(set(cand.loc[cand.source == "chexpert", "image_path"]) - set(hc.image_path))
    OUT.mkdir(parents=True, exist_ok=True)
    lista = LOGDIR / "b3c_chexpert_sin_hash.csv"
    LOGDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_path": falta}).to_csv(lista, index=False)
    log("imágenes de CheXpert del conjunto candidato sin hash en chexpert_pool.csv: %d" % len(falta))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "data" / "hash_images.py"), "--manifest", str(lista),
                        "--root", str(data_root()), "--out", str(HASH_EXTRA), "--workers", "2",
                        "--max-seconds", str(max_s)], capture_output=True, text=True)
    log(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:])
    h = pd.read_csv(HASH_EXTRA, dtype=str, keep_default_na=False)
    hechos = set(h.loc[h.error == "", "image_path"])
    quedan = len(set(falta) - hechos)
    log("hashes nuevos: %d; quedan %d" % (len(hechos), quedan))
    return 0 if quedan == 0 else 3


# ------------------------------------------------------------------------------------------ paso dedup
def hashes_candidatos(cand):
    hn = bc.cargar_hashes(ROOT / "logs" / "tmp_datos" / "hash_nih_cxr14.csv")
    hc = bc.cargar_hashes(ROOT / "data" / "manifests" / "bench" / "hashes" / "chexpert_pool.csv")
    hx = bc.cargar_hashes(HASH_EXTRA)
    h = pd.concat([hn, hc, hx], ignore_index=True).drop_duplicates("image_path")
    u = cand.drop_duplicates("image_path")
    h = h.merge(u[["image_path", "source", "patient_id"]], on="image_path", how="inner")
    falta = set(u.image_path) - set(h.image_path)
    return h, u, falta


def paso_dedup(log=print):
    cand = candidatos()
    h, u, falta = hashes_candidatos(cand)
    log("imágenes candidatas %d, con hash %d, sin hash %d" % (len(u), len(h), len(falta)))
    if falta:
        raise SystemExit("faltan hashes de %d imágenes candidatas: ejecute --paso hashes" % len(falta))
    h["grupo"] = h.source + ":" + h.patient_id
    root = data_root()
    t = time.time()
    pares = bc.casi_duplicados(h, lambda p: str(root / p), grupo_col="grupo", verificar=False, log=log)
    g = dict(zip(h.image_path, h.grupo))
    pares["grupo_a"], pares["grupo_b"] = pares.a.map(g), pares.b.map(g)
    OUT.mkdir(parents=True, exist_ok=True)
    old = []
    for prev in (B3A / "duplicados_hash.csv", OUT / "duplicados_hash.csv"):   # verificaciones ya hechas
        if prev.exists():
            d = pd.read_csv(prev)[["a", "b", "thumb_corr"]].dropna()
            old += [d, d.rename(columns={"a": "b", "b": "a"})]
    if old:
        old = pd.concat(old).drop_duplicates(["a", "b"])
        pares = pares.drop(columns="thumb_corr").merge(old, on=["a", "b"], how="left")
        pares["confirmed"] = (pares.kind == "md5") | (pares.thumb_corr >= 0.97)
    pares.to_csv(OUT / "duplicados_hash.csv", index=False)
    log("pares candidatos entre pacientes distintos: %d, sin verificar %d (%.0f s)"
        % (len(pares), int(pares.thumb_corr.isna().sum()), time.time() - t))


def paso_verificar(max_s=150, log=print):
    root = data_root()
    n = bc.verificar_pares(OUT / "duplicados_hash.csv", lambda x: str(root / x), max_s=max_s, log=log)
    return 0 if n == 0 else 3


def pacientes_excluidos(cand: pd.DataFrame) -> pd.DataFrame:
    """Regla de B3a (build_b3a.pacientes_excluidos) sobre los pares de B3c, más los 8 pacientes retirados en B3a."""
    p = OUT / "duplicados_hash.csv"
    if not p.exists():
        raise SystemExit("falta %s: ejecute antes --paso dedup" % p)
    d = pd.read_csv(p)
    if d.thumb_corr.isna().any():
        raise SystemExit("hay pares sin verificar en %s: ejecute --paso verificar" % p)
    d = d[d.confirmed]
    rows = []
    if not d.empty:
        rep = bc.grupos_union(d, "grupo_a", "grupo_b")
        cl = (cand.source + ":" + cand.patient_id).to_frame("grupo").assign(clase=cand.clase.values).drop_duplicates()
        cl = dict(zip(cl.grupo, cl.clase))
        comp = pd.DataFrame({"grupo": list(rep), "rep": list(rep.values())})
        comp["clase"] = comp.grupo.map(cl)
        for r, gg in comp.groupby("rep"):
            if gg.clase.nunique() > 1:
                rows += [(x, "duplicado con clase distinta") for x in gg.grupo]
            else:
                rows += [(x, "duplicado de %s" % r) for x in gg.grupo if x != r]
    presentes = set(cand.source + ":" + cand.patient_id)
    ya = {x for x, _ in rows}
    rows += [(x, "retirado en b3a") for x in sorted(excluidos_b3a() & presentes) if x not in ya]
    return pd.DataFrame(rows, columns=["grupo", "motivo"])


# ------------------------------------------------------------------------------------------ reservas
def reservas_heredadas(pac: pd.DataFrame, log=print) -> pd.Series:
    """Reserva de B3a para los pacientes que la tienen; los nuevos de cada estrato (fuente, clase, vistas) se
    reparten para acercar el estrato al 70/10/20: objetivo por reserva con los cortes de bc.reservas_pacientes,
    déficit = max(0, objetivo - heredados) y reparto de los nuevos proporcional al déficit (restos mayores; empates
    en el orden test, val, train). Los nuevos se ordenan con una permutación fija (semilla 20261114, clave b3c)."""
    r3a = pd.read_csv(B3A / "reservas_pacientes.csv", dtype={"patient_id": str})
    her = dict(zip(r3a.source + ":" + r3a.patient_id, r3a.reserva))
    out = pd.Series("", index=pac.index, dtype=object)
    k = pac.source + ":" + pac.patient_id
    out[k.isin(her)] = k[k.isin(her)].map(her)
    noms = ("train", "val", "test")
    for key, g in pac.groupby(["source", "clase", "vistas"], sort=True):
        n = len(g)
        cuts = [int(round(x * n)) for x in np.cumsum((0.7, 0.1, 0.2))[:-1]]
        obj = dict(zip(noms, np.diff([0] + cuts + [n])))
        inh = out.loc[g.index].value_counts().to_dict()
        nuevos = g[out.loc[g.index] == ""].sort_values("patient_id").index.to_numpy()
        m = len(nuevos)
        if m == 0:
            continue
        nuevos = nuevos[bc.rng("reservas_b3c", *key).permutation(m)]
        def_ = np.array([max(0, obj[r] - inh.get(r, 0)) for r in noms], dtype=float)
        if def_.sum() == 0:
            def_ = np.array([obj[r] for r in noms], dtype=float)
        q = def_ / def_.sum() * m
        cnt = np.floor(q).astype(int)
        orden = sorted(range(3), key=lambda i: (-(q[i] - cnt[i]), -i))     # restos mayores; empate: test, val, train
        for i in orden[: m - cnt.sum()]:
            cnt[i] += 1
        pos = 0
        for r, c in zip(noms, cnt):
            out.loc[nuevos[pos:pos + c]] = r
            pos += c
    return out


def preparar(log=print):
    cand = candidatos()
    exc = pacientes_excluidos(cand)
    g = cand.source + ":" + cand.patient_id
    cand = cand[~g.isin(set(exc.grupo))].copy()
    log("pacientes retirados: %d (%s)" % (len(exc), exc.motivo.str.split(" de ").str[0].value_counts().to_dict()))
    img = bc.imagen_fija(cand, claves=("source", "patient_id", "clase", "view"), salt=TAG)
    pac = cand.groupby(["source", "patient_id", "clase"]).view.agg(lambda s: set(s)).reset_index()
    pac["has_ap"], pac["has_pa"] = pac.view.map(lambda s: "AP" in s), pac.view.map(lambda s: "PA" in s)
    pac["vistas"] = np.where(pac.has_ap & pac.has_pa, "ambas", np.where(pac.has_ap, "solo_ap", "solo_pa"))
    pac = pac.drop(columns="view")
    r3a = pd.read_csv(B3A / "reservas_pacientes.csv", dtype={"patient_id": str})
    pac["heredada"] = (pac.source + ":" + pac.patient_id).isin(set(r3a.source + ":" + r3a.patient_id))
    pac["reserva"] = reservas_heredadas(pac, log).values
    return img, pac, exc


# ------------------------------------------------------------------------------------------ tamaños
def tamanos(k: float) -> dict:
    return {s: rnd(v * k) for s, v in BASE.items()}


def requisitos(t: dict):
    """(fuente, clase, reserva, n) de todos los conjuntos de los cuatro niveles."""
    req = []
    for rho in RHOS:
        prev = B.prev_fuente(ORIENT, rho)
        for s in FUENTES:
            for name, res, p in (("train", "train", prev[s]), ("val", "val", prev[s]),
                                 ("val_desconf", "val", 0.5), ("test_desconf", "test", 0.5),
                                 ("test_invertido", "test", 1.0 - prev[s])):
                n = t[name]
                npos = rnd(n * p)
                req += [(name, rho, s, 1, res, npos), (name, rho, s, 0, res, n - npos)]
    return req


def factible_k(ordenes, k) -> list:
    """Lista de requisitos no factibles con el factor k (vacía si todo es factible)."""
    malos = []
    for name, rho, s, cls, res, n in requisitos(tamanos(k)):
        if n == 0:
            continue
        o = ordenes.get(s, cls, res, TAG, "fijo")
        nap = int(round(n * F_AP[s]))           # el mismo redondeo que bench_common.muestra_ap
        if not bc.factible(nap, n - nap, o[2], o[3], o[4]):
            malos.append((name, rho, s, cls, res, n))
    return malos


def elegir_k(ordenes, log=print) -> float:
    for i in range(20, 0, -1):
        k = round(i * 0.05, 2)
        malos = factible_k(ordenes, k)
        if not malos:
            return k
        if i in (20, 10) or i <= 8:
            log("K = %.2f no factible: %d requisitos (p. ej. %s)" % (k, len(malos), malos[0]))
    raise ValueError("ningún K factible")


# ------------------------------------------------------------------------------------------ manifiestos
def construir_manifiesto(rho, seed, t, k, ordenes, img, img_idx):
    prev = B.prev_fuente(ORIENT, rho)
    sel = {}

    def add(ix, split=None, **flags):
        e = sel.setdefault(ix, {"split": None, "val_desconf": 0, "test_desconf": 0, "test_invertido": 0})
        if split:
            e["split"] = split
        for kk, v in flags.items():
            e[kk] = v
    semilla, fijo = (TAG, "semilla", seed), (TAG, "fijo")
    for s in FUENTES:
        npos = rnd(t["train"] * prev[s])
        for ix in B.conjunto(ordenes, img_idx, s, "train", npos, t["train"] - npos, semilla):
            add(ix, "train")
        npos = rnd(t["val"] * prev[s])
        for ix in B.conjunto(ordenes, img_idx, s, "val", npos, t["val"] - npos, semilla):
            add(ix, "val")
        for ix in B.conjunto(ordenes, img_idx, s, "val", t["val_desconf"], t["val_desconf"], fijo):
            add(ix, None, val_desconf=1)
        for ix in B.conjunto(ordenes, img_idx, s, "test", t["test_desconf"], t["test_desconf"], fijo):
            add(ix, "test", test_desconf=1)
        npos = rnd(t["test_invertido"] * (1.0 - prev[s]))
        for ix in B.conjunto(ordenes, img_idx, s, "test", npos, t["test_invertido"] - npos, fijo):
            add(ix, "test", test_invertido=1)
    d = img.loc[list(sel)].copy()
    for kk in ("split",) + tuple(SUBSETS):
        d[kk] = [sel[i][kk] for i in d.index]
    d["split"] = d["split"].fillna("fuera")
    d["label"] = d.clase.astype(int)
    d["raiz"] = "datos"
    d["image_id"] = [B.image_id(s, p) for s, p in zip(d.source, d.image_path)]
    d["orientacion"] = ORIENT
    d = d.drop(columns=["clase"]).sort_values(["split", "source", "label", "image_id"]).reset_index(drop=True)
    mixtas = [s for s in FUENTES if d[d.split == "train"].query("source == @s").label.nunique() == 2]
    meta = {
        "benchmark": "b3c", "variante": "3c_cardiomegalia", "orientacion": ORIENT, "rho": rho, "semilla": seed,
        "fuentes": list(FUENTES), "fuentes_mixtas": mixtas, "subconjuntos": SUBSETS, "seleccion": "val",
        "seleccion_alternativa": "val_desconf",
        "prevalencia_train_por_fuente": prev, "fraccion_ap_por_fuente": F_AP,
        "tamanos": {"train_por_fuente": t["train"], "val_por_fuente": t["val"],
                    "val_desconf_por_celda": t["val_desconf"], "test_desconf_por_celda": t["test_desconf"],
                    "test_invertido_por_fuente": t["test_invertido"],
                    "test_invertido_por_fuente_diseno": t["test_invertido"],
                    "factor_k": k, "tamanos_b3a": BASE},
        "etiqueta": {"nih": "positivo y_cardiomegaly = 1 (Finding Labels contiene Cardiomegaly); negativo "
                            "y_no_finding = 1",
                     "chexpert": "positivo Cardiomegaly = 1; negativo No Finding = 1 y Cardiomegaly vacío o 0; "
                                 "se excluye todo paciente con alguna imagen frontal con Cardiomegaly = -1"},
        "reservas": "las de B3a para los pacientes que ya la tienen (data/manifests/bench/b3a/reservas_pacientes.csv); "
                    "los nuevos, repartidos por estrato fuente x clase x vistas hacia el 70/10/20",
        "notas": ("val_desconf puede compartir filas con val (las filas solo de val_desconf llevan split=fuera); "
                  "test_desconf y test_invertido salen de la misma ordenación fija de la reserva de test y "
                  "comparten filas; un paciente con AP y PA puede aportar imágenes distintas a dos subconjuntos, "
                  "nunca dos imágenes al mismo conjunto ni pacientes entre reservas."),
        "test_bloqueado": True, "semilla_reservas": bc.SEMILLA_RESERVAS,
        "generador": "scripts/data/build_b3c.py",
    }
    return d, meta


def nombre(rho, seed):
    return OUT / ORIENT / ("rho%.2f_s%d.csv" % (rho, seed))


def contexto(log=print):
    img, pac, exc = preparar(log)
    img = img.merge(pac[["source", "patient_id", "reserva"]], on=["source", "patient_id"], how="left")
    img = img.reset_index(drop=True)
    img_idx = {(s, p, c, v): i for i, (s, p, c, v) in enumerate(zip(img.source, img.patient_id, img.clase, img.view))}
    return img, pac, exc, img_idx, B.Ordenes(pac)


def resumen_reservas(pac):
    return pac.groupby(["source", "clase", "reserva"]).size().unstack("reserva", fill_value=0)[["train", "val", "test"]]


def paso_factibilidad(log=print) -> int:
    img, pac, exc, img_idx, ordenes = contexto(log)
    log(resumen_reservas(pac).to_string())
    log("con AP / con PA / ambas por fuente, clase y reserva:")
    log(pac.groupby(["source", "clase", "reserva"]).agg(ap=("has_ap", "sum"), pa=("has_pa", "sum"),
                                                        n=("patient_id", "size")).to_string())
    k = elegir_k(ordenes, log)
    log("K = %.2f; tamaños %s" % (k, tamanos(k)))
    return 0


def paso_build(log=print) -> int:
    img, pac, exc, img_idx, ordenes = contexto(log)
    k = elegir_k(ordenes, log)
    t = tamanos(k)
    log("K = %.2f; tamaños %s" % (k, t))
    OUT.mkdir(parents=True, exist_ok=True)
    exc.to_csv(OUT / "pacientes_excluidos_hash.csv", index=False)
    pac.assign(clase=pac.clase.astype(int)).to_csv(OUT / "reservas_pacientes.csv", index=False)
    log(resumen_reservas(pac).to_string())
    hechos = []
    for rho in RHOS:
        for seed in SEEDS:
            d, meta = construir_manifiesto(rho, seed, t, k, ordenes, img, img_idx)
            write_bench(d, meta, nombre(rho, seed))
            hechos.append(str(nombre(rho, seed).relative_to(ROOT)))
    union = pd.concat([pd.read_csv(ROOT / h, usecols=["image_id", "image_path", "raiz", "source"]) for h in hechos])
    union = union.drop_duplicates("image_id")
    union.to_csv(OUT / "imagenes_unicas.csv", index=False)
    log("manifiestos escritos: %d; imágenes distintas: %d" % (len(hechos), len(union)))
    with open(OUT / "indice.json", "w", encoding="utf-8") as f:
        json.dump({"manifiestos": hechos, "imagenes_unicas": len(union), "factor_k": k, "tamanos": t}, f, indent=2)
    return paso_tabla(log)


def paso_tabla(log=print) -> int:
    """Tablas regenerables: reservas y recuentos por conjunto, fuente, clase y vista (semilla 0 de cada nivel y
    comprobación de que las semillas 1 y 2 tienen los mismos recuentos)."""
    from src.data.bench import read_csv
    lines = []
    r = pd.read_csv(OUT / "reservas_pacientes.csv", dtype={"patient_id": str})
    t = r.groupby(["source", "clase", "reserva"]).size().unstack("reserva", fill_value=0)[["train", "val", "test"]]
    h = r.groupby(["source", "clase", "heredada"]).size().unstack("heredada", fill_value=0)
    lines += ["# Reservas de pacientes de B3c", t.to_string(), "", "# Pacientes con reserva heredada de B3a (True)",
              h.to_string(), ""]
    filas = []
    for rho in RHOS:
        base = None
        for seed in SEEDS:
            d = read_csv(nombre(rho, seed))
            sets = {"train": d[d.split == "train"], "val": d[d.split == "val"]}
            for s in SUBSETS:
                sets[s] = d[d[s].astype(int) == 1]
            c = {}
            for name, x in sets.items():
                for (src, lab, v), n in x.groupby(["source", "label", "view"]).size().items():
                    c[(name, src, int(lab), v)] = int(n)
            if base is None:
                base = c
                for (name, src, lab, v), n in sorted(c.items()):
                    filas.append({"rho": rho, "conjunto": name, "fuente": src, "clase": lab, "vista": v, "n": n})
            elif c != base:
                lines.append("AVISO: rho %.2f semilla %d con recuentos distintos de la semilla 0" % (rho, seed))
    f = pd.DataFrame(filas)
    piv = f.pivot_table(index=["rho", "conjunto", "fuente"], columns=["clase", "vista"], values="n", fill_value=0,
                        aggfunc="sum")
    lines += ["# Recuentos por nivel, conjunto, fuente, clase (0/1) y vista; semilla 0 (las semillas 1 y 2 son "
              "iguales salvo aviso)", piv.to_string()]
    LOGDIR.mkdir(parents=True, exist_ok=True)
    (LOGDIR / "b3c_recuentos.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    f.to_csv(LOGDIR / "b3c_recuentos.csv", index=False)
    log("\n".join(lines[:12]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--paso", choices=["hashes", "dedup", "verificar", "factibilidad", "build", "tabla"],
                    required=True)
    a = ap.parse_args(argv)
    if a.paso == "dedup":
        paso_dedup()
        return 0
    return {"hashes": paso_hashes, "verificar": paso_verificar, "factibilidad": paso_factibilidad,
            "build": paso_build, "tabla": paso_tabla}[a.paso]()


if __name__ == "__main__":
    sys.exit(main())
