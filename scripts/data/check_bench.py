#!/usr/bin/env python3
"""Auditoría ejecutable de los manifiestos del plan B (data/manifests/bench/**). Termina con código 1 si falla algo.

Comprobaciones, por manifiesto: contrato (src/data/bench.py); una sola imagen por paciente en cada conjunto; ningún
paciente en dos reservas o particiones; prevalencias por fuente iguales al diseño (tolerancia de una imagen); en B3a,
cuota AP exacta por clase y fuente y diferencia de fracción AP entre clases de 0.02 como máximo en las celdas con al
menos 50 imágenes por clase (en las menores la única diferencia admitida es la del redondeo de la cuota). Por
benchmark: reserva de cada paciente constante entre manifiestos, test fijo entre niveles y semillas (B3a),
casi duplicados por hash entre reservas distintas (MD5 o pHash <= 4 y dHash <= 12, confirmados con correlación de
miniaturas >= 0.97) e imágenes inexistentes.

    python3 scripts/data/check_bench.py [--sin-hash] [--sin-ficheros] [--log logs/datos_plan_b/check_bench.log]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_common as bc  # noqa: E402
from src.data.bench import BENCH_DIR, abs_paths, list_manifests, read_csv, read_meta, validate  # noqa: E402
from src.data.paths import data_root  # noqa: E402

F_AP_B3A = {"nih": 0.5, "chexpert": 0.7}
HB = ROOT / "data" / "manifests" / "bench" / "hashes"
HASHES = [("datos", ROOT / "logs/tmp_datos/hash_nih_cxr14.csv", ""), ("datos", HB / "chexpert_pool.csv", ""),
          ("proyecto", HB / "tb_rahman.csv", "data/raw/tb_rahman/"),
          ("proyecto", HB / "tbx11k.csv", "data/raw/tbx11k/TBX11K/imgs/"),
          ("proyecto", HB / "shenzhen.csv", "data/raw/shenzhen/CXR_png/"),
          ("proyecto", HB / "montgomery.csv", "data/raw/montgomery/"),
          ("proyecto", HB / "covidgr.csv", "data/raw/covidgr/"),
          ("datos", ROOT / "data/manifests/bench/b3c/hashes_chexpert_b3c.csv", "")]   # CheXpert de B3c
BLOQUE3 = ("b3a", "b3c")     # B3c (cardiomegalia) sigue las reglas de B3a con los tamaños de su JSON


def rnd(x):
    return int(np.floor(x + 0.5 + 1e-9))


class Informe:
    def __init__(self):
        self.lines, self.fallos = [], 0

    def ok(self, msg):
        self.lines.append("OK    " + msg)

    def fail(self, msg):
        self.lines.append("FALLO " + msg)
        self.fallos += 1

    def info(self, msg):
        self.lines.append("      " + msg)

    def check(self, cond, msg):
        (self.ok if cond else self.fail)(msg)


def conjuntos(df, meta):
    out = {s: df[df.split == s] for s in ("train", "val", "test")}
    for s in meta["subconjuntos"]:
        out[s] = df[df[s].astype(int) == 1]
    return out


def reserva_col(df):
    return df["reserva"] if "reserva" in df.columns else df["split"]


def check_comunes(rep, name, df, meta):
    err = validate(df, meta)
    rep.check(not err, "%s: contrato%s" % (name, "" if not err else " (" + "; ".join(err) + ")"))
    if err:
        return False
    cj = conjuntos(df, meta)
    rep_dup = [k for k, d in cj.items() if d.patient_id.duplicated().any()]
    if meta["benchmark"] in BLOQUE3:    # requisito del diseño de B3a; en B2 un paciente NIH puede tener varias
        rep.check(not rep_dup, "%s: una imagen por paciente en cada conjunto%s" % (name, "" if not rep_dup else " " + str(rep_dup)))
    elif rep_dup:
        rep.info("%s: pacientes con varias imágenes en %s (admitido fuera de B3a)" % (name, rep_dup))
    r = reserva_col(df)
    n_res = df.assign(_r=r).groupby("patient_id")._r.nunique()
    n_split = df[df.split != "fuera"].groupby("patient_id").split.nunique()
    bad = int((n_res > 1).sum()) + int((n_split > 1).sum())
    rep.check(bad == 0, "%s: pacientes en dos reservas o particiones: %d" % (name, bad))
    if "reserva" in df.columns:
        incons = df[(df.split != "fuera") & (df.split != df.reserva)]
        rep.check(incons.empty, "%s: split coherente con la reserva (%d filas no)" % (name, len(incons)))
    return True


def check_b3a(rep, name, df, meta):
    p_a = meta["rho"] if meta["orientacion"] == "apos" else 1.0 - meta["rho"]
    prev = {"nih": p_a, "chexpert": 1.0 - p_a}
    tm = meta["tamanos"]
    n_inv = int(tm["test_invertido_por_fuente"])
    n_dis = int(tm.get("test_invertido_por_fuente_diseno", 1000))
    if n_inv != n_dis:
        rep.info("%s: test_invertido de %d por fuente (ajuste declarado; diseño %d)" % (name, n_inv, n_dis))
    rep.check(0 < n_inv <= n_dis, "%s: tamaño del test invertido %d en (0, %d]" % (name, n_inv, n_dis))
    if meta["benchmark"] == "b3a":
        tam = (2800, 400, 200, 500)
    else:
        tam = (int(tm["train_por_fuente"]), int(tm["val_por_fuente"]), int(tm["val_desconf_por_celda"]),
               int(tm["test_desconf_por_celda"]))
    esperado = {"train": (tam[0], prev), "val": (tam[1], prev),
                "val_desconf": (2 * tam[2], {"nih": .5, "chexpert": .5}),
                "test_desconf": (2 * tam[3], {"nih": .5, "chexpert": .5}),
                "test_invertido": (n_inv, {s: 1 - v for s, v in prev.items()})}
    cj = conjuntos(df, meta)
    errs, errs_ap, info_ap = [], [], []
    for k, (n, pv) in esperado.items():
        d = cj[k]
        for s in ("nih", "chexpert"):
            ds = d[d.source == s]
            npos_e = rnd(n * pv[s])
            npos, nneg = int(ds.label.sum()), int((ds.label == 0).sum())
            if abs(npos - npos_e) > 1 or abs(nneg - (n - npos_e)) > 1:
                errs.append("%s/%s: %d/%d frente a %d/%d" % (k, s, nneg, npos, n - npos_e, npos_e))
            fr = {}
            for c in (0, 1):
                dc = ds[ds.label == c]
                if len(dc) == 0:
                    continue
                nap = int((dc.view == "AP").sum())
                cuota = int(round(len(dc) * F_AP_B3A[s]))     # el redondeo de bench_common.muestra_ap
                if nap != cuota:
                    errs_ap.append("%s/%s/clase %d: %d AP de %d (cuota %d)" % (k, s, c, nap, len(dc), cuota))
                fr[c] = (nap / len(dc), len(dc))
            if len(fr) == 2:
                diff = abs(fr[0][0] - fr[1][0])
                if min(fr[0][1], fr[1][1]) >= 50 and diff > 0.02:
                    errs_ap.append("%s/%s: diferencia AP entre clases %.3f" % (k, s, diff))
                elif diff > 0.02:
                    info_ap.append("%s/%s %.3f (n mínimo %d)" % (k, s, diff, min(fr[0][1], fr[1][1])))
    rep.check(not errs, "%s: prevalencias por fuente%s" % (name, "" if not errs else " " + "; ".join(errs)))
    rep.check(not errs_ap, "%s: fracción AP por clase y fuente%s" % (name, "" if not errs_ap else " " + "; ".join(errs_ap)))
    if info_ap:
        rep.info("%s: diferencias AP solo por redondeo en celdas pequeñas: %s" % (name, "; ".join(info_ap)))
    mix = [s for s in meta["fuentes"] if cj["train"][cj["train"].source == s].label.nunique() == 2]
    rep.check(sorted(mix) == sorted(meta["fuentes_mixtas"]), "%s: fuentes mixtas declaradas" % name)


def check_b2(rep, name, df, meta):
    cj = conjuntos(df, meta)
    t = lambda d, s: (int((d[d.source == s].label == 0).sum()), int(d[d.source == s].label.sum()))
    tr, va, te = cj["train"], cj["val"], cj["test"]
    e = []
    if t(tr, "belarus")[0] != 0 or t(tr, "belarus")[1] < 250:
        e.append("belarus train %s" % (t(tr, "belarus"),))
    if t(tr, "rsna_nih")[1] != 0 or t(tr, "rsna_nih")[0] < 2700:
        e.append("rsna_nih train %s" % (t(tr, "rsna_nih"),))
    if t(tr, "tbx11k") != (600, 600):
        e.append("tbx11k train %s" % (t(tr, "tbx11k"),))
    if t(va, "tbx11k")[1] != 200 or not (1500 <= t(va, "tbx11k")[0] <= 1600):
        e.append("tbx11k val %s" % (t(va, "tbx11k"),))
    for s in ("belarus", "rsna_nih"):
        ntr, nva = len(tr[tr.source == s]), len(va[va.source == s])
        if abs(nva - 0.1 * (ntr + nva)) > 0.02 * (ntr + nva) + 2:
            e.append("%s: validación %d de %d (diseño 10 %%)" % (s, nva, ntr + nva))
    if t(te, "shenzhen") != (326, 336) or t(te, "montgomery") != (80, 58):
        e.append("test shenzhen %s montgomery %s" % (t(te, "shenzhen"), t(te, "montgomery")))
    if set(te.source) - {"shenzhen", "montgomery"} or set(tr.source) & {"shenzhen", "montgomery"}:
        e.append("fuentes de test en entrenamiento o al revés")
    if not cj["val_robust"].source.eq("tbx11k").all():
        e.append("val_robust con fuentes distintas de tbx11k")
    rep.check(not e, "%s: recuentos por fuente del diseño%s" % (name, "" if not e else " " + "; ".join(e)))
    rep.check(meta["fuentes_mixtas"] == ["tbx11k"], "%s: fuente mixta tbx11k" % name)
    decl = meta.get("recuentos_por_fuente_neg_pos", {})
    for sp, d in (("train", tr), ("val", va), ("test", te)):
        for s, (n0, n1) in decl.get(sp, {}).items():
            if t(d, s) != (n0, n1):
                rep.fail("%s: recuento declarado en el JSON %s/%s" % (name, sp, s))


def check_covidgr(rep, name, df, meta):
    n = (int((df.label == 0).sum()), int(df.label.sum()))
    rep.check(n == (426, 426), "%s: 426 negativas y 426 positivas (%d/%d)" % (name, *n))
    rep.check(df.split.eq("test").all(), "%s: solo evaluación (split = test)" % name)
    pos = df[df.label == 1]
    rep.info("%s: gravedad de las positivas %s" % (name, pos.severidad.replace("", "sin dato").value_counts().to_dict()))
    rep.info("%s: imágenes casi idénticas a COVIDx: %d" % (name, int(df.dup_covidx.sum())))


def cargar_hashes():
    reg = {}
    for raiz, p, pref in HASHES:
        if p.exists():
            h = bc.cargar_hashes(p, pref)
            for r in h[["image_path", "md5", "phash", "dhash"]].itertuples(index=False):
                reg[(raiz, r.image_path)] = r
    return reg


def verificados():
    v = {}
    for p in BENCH_DIR.glob("*/duplicados_hash.csv"):
        d = pd.read_csv(p, usecols=["a", "b", "thumb_corr"])
        for a, b, c in zip(d.a, d.b, d.thumb_corr):
            v[(a, b)] = v[(b, a)] = c
    return v


def check_hash_bench(rep, bench, imgs, reg, ver):
    """imgs: DataFrame con raiz, image_path, reserva (únicas)."""
    keys = list(zip(imgs.raiz, imgs.image_path))
    falta = [k for k in keys if k not in reg]
    rep.check(not falta, "%s: hash disponible para todas las imágenes (%d sin hash)" % (bench, len(falta)))
    sub = imgs[[k in reg for k in keys]].reset_index(drop=True)
    if sub.empty:
        return
    h = pd.DataFrame([reg[k] for k in zip(sub.raiz, sub.image_path)])
    h["image_path"] = sub.image_path.values
    h["grupo"] = sub.reserva.values
    droot = data_root()
    absmap = dict(zip(sub.image_path, abs_paths(sub)))
    pares = bc.casi_duplicados(h, lambda p: absmap[p], grupo_col="grupo", verificar=False, log=lambda *_: None)
    corr = []
    for a, b, k in zip(pares.a, pares.b, pares.kind):
        if k == "md5":
            corr.append(1.0)
        elif (a, b) in ver and not pd.isna(ver[(a, b)]):
            corr.append(ver[(a, b)])
        else:
            corr.append(float(np.mean(bc._thumb(absmap[a]) * bc._thumb(absmap[b]))))
    pares["thumb_corr"] = corr
    conf = pares[(pares.kind == "md5") | (pares.thumb_corr >= 0.97)]
    rep.check(conf.empty, "%s: casi duplicados por hash entre reservas distintas: %d confirmados de %d candidatos"
              % (bench, len(conf), len(pares)))
    for a, b in zip(conf.a.head(5), conf.b.head(5)):
        rep.info("   %s ~ %s" % (a, b))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--sin-hash", action="store_true")
    ap.add_argument("--sin-ficheros", action="store_true")
    ap.add_argument("--log", default=str(ROOT / "logs" / "datos_plan_b" / "check_bench.log"))
    a = ap.parse_args(argv)
    rep = Informe()
    mans = list_manifests(a.root)
    rep.check(len(mans) > 0, "manifiestos encontrados: %d" % len(mans))
    por_bench = defaultdict(list)
    for p in mans:
        df, meta = read_csv(p), read_meta(p)
        name = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
        if not check_comunes(rep, name, df, meta):
            continue
        b = meta["benchmark"]
        fn = {"b3a": check_b3a, "b3c": check_b3a, "b2": check_b2, "covidgr": check_covidgr}.get(b)
        if fn is None:
            rep.fail("%s: benchmark desconocido %s" % (name, b))
            continue
        try:
            fn(rep, name, df, meta)
        except Exception as e:      # columna o clave del diseño ausente: cuenta como fallo, no como caída
            rep.fail("%s: comprobación del diseño no ejecutable (%s: %s)" % (name, type(e).__name__, e))
        por_bench[b].append((name, df, meta))
    reg = None if a.sin_hash else cargar_hashes()
    ver = None if a.sin_hash else verificados()
    for b, items in por_bench.items():
        allr = pd.concat([d.assign(_r=reserva_col(d))[["patient_id", "_r", "raiz", "image_path", "image_id"]]
                          for _, d, _ in items])
        inc = allr.drop_duplicates(["patient_id", "_r"]).patient_id.duplicated().sum()
        rep.check(inc == 0, "%s: reserva de cada paciente constante entre manifiestos (%d incoherentes)" % (b, inc))
        inc = allr.drop_duplicates(["image_id", "image_path"]).image_id.duplicated().sum()
        rep.check(inc == 0, "%s: image_id con la misma ruta en todos los manifiestos" % b)
        if b in BLOQUE3:
            tests = {}
            for name, d, m in items:
                tests.setdefault("todos", set()).add(tuple(sorted(d[d.test_desconf == 1].image_id)))
            rep.check(len(tests["todos"]) == 1, "%s: test_desconf idéntico en los %d manifiestos" % (b, len(items)))
        u = allr.drop_duplicates(["raiz", "image_path"]).rename(columns={"_r": "reserva"})
        rep.info("%s: %d manifiestos, %d imágenes distintas, %d pacientes" % (b, len(items), len(u),
                                                                             u.patient_id.nunique()))
        if not a.sin_ficheros:
            miss = [p for p in abs_paths(u) if not os.path.isfile(p)]
            rep.check(not miss, "%s: imágenes inexistentes: %d%s" % (b, len(miss), "" if not miss else " (" + miss[0] + ")"))
        if not a.sin_hash:
            check_hash_bench(rep, b, u, reg, ver)
    rep.lines.append("RESULTADO: %s (%d fallos)" % ("todas las comprobaciones pasan" if rep.fallos == 0 else "HAY FALLOS",
                                                    rep.fallos))
    txt = "\n".join(rep.lines)
    if a.log:
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        Path(a.log).write_text(txt + "\n", encoding="utf-8")
    print("\n".join(l for l in rep.lines if not l.startswith("OK")))
    print("comprobaciones correctas: %d" % sum(l.startswith("OK") for l in rep.lines))
    return 0 if rep.fallos == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
