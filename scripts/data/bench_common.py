"""Utilidades comunes de los constructores del plan B (B3a, B2, COVIDGR).

- Reservas de pacientes fijas y estratificadas (`reservas_pacientes`).
- Elección determinista de una imagen por paciente, clase y vista (`imagen_fija`).
- Muestreo por prefijos con fracción AP fijada (`muestra_ap`): las listas de pacientes se ordenan una vez por semilla
  y cada conjunto toma un prefijo, de modo que los conjuntos de una misma semilla quedan anidados entre niveles de rho
  y la caché no crece con el número de niveles.
- Casi duplicados por hash entre imágenes de pacientes distintos (`casi_duplicados`), con el mismo criterio que
  scripts/data/find_overlaps.py (pHash <= 4 y dHash <= 12 por bandas, MD5 idéntico, y confirmación con la
  correlación de miniaturas 64 x 64 >= 0.97).
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEMILLA_RESERVAS = 20261114
_spec = importlib.util.spec_from_file_location("find_overlaps", Path(__file__).with_name("find_overlaps.py"))
fo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fo)


def h32(*parts) -> int:
    """Entero estable de 32 bits a partir de cadenas (no depende de PYTHONHASHSEED)."""
    return int(hashlib.md5("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def rng(*parts) -> np.random.Generator:
    return np.random.default_rng([SEMILLA_RESERVAS, h32(*parts)])


def reservas_pacientes(pac: pd.DataFrame, estratos: list[str], fracs=(0.7, 0.1, 0.2), clave="reservas",
                       nombres=("train", "val", "test")) -> pd.Series:
    """Asigna cada paciente (una fila por paciente) a una reserva, estratificando por `estratos`. Devuelve una Serie
    alineada con `pac`. Los cortes se redondean dentro de cada estrato."""
    out = pd.Series("", index=pac.index, dtype=object)
    c = np.cumsum(fracs)
    for key, g in pac.groupby(estratos, sort=True):
        idx = g.sort_values("patient_id").index.to_numpy()
        perm = idx[rng(clave, key).permutation(len(idx))]
        cuts = [int(round(x * len(idx))) for x in c[:-1]]
        for name, part in zip(nombres, np.split(perm, cuts)):
            out.loc[part] = name
    return out


def imagen_fija(imgs: pd.DataFrame, claves=("patient_id", "clase", "view"), salt="imagen") -> pd.DataFrame:
    """Una imagen por (paciente, clase, vista), elegida de forma determinista y fija para todo el benchmark."""
    d = imgs.copy()
    d["_k"] = [h32(salt, p) for p in d["image_path"]]
    d = d.sort_values(list(claves) + ["_k"]).drop_duplicates(list(claves), keep="first")
    return d.drop(columns="_k")


def orden_ap_pa(pac: pd.DataFrame, *seed_parts):
    """Listas ordenadas (AP, PA) de pacientes de un estrato. `pac` necesita patient_id, has_ap y has_pa. Primero van
    los pacientes con una sola vista y después los que tienen ambas; los de ambas se recorren en sentidos opuestos
    en las dos listas, de modo que dos prefijos son disjuntos mientras quepan (ver `muestra_ap`)."""
    p = pac.sort_values("patient_id")
    r = rng("orden", *seed_parts)
    solo_ap = p[p.has_ap & ~p.has_pa].patient_id.to_numpy()
    solo_pa = p[p.has_pa & ~p.has_ap].patient_id.to_numpy()
    ambas = p[p.has_ap & p.has_pa].patient_id.to_numpy()
    solo_ap, solo_pa, ambas = solo_ap[r.permutation(len(solo_ap))], solo_pa[r.permutation(len(solo_pa))], \
        ambas[r.permutation(len(ambas))]
    return (np.concatenate([solo_ap, ambas]), np.concatenate([solo_pa, ambas[::-1]]), len(solo_ap), len(solo_pa),
            len(ambas))


def factible(n_ap, n_pa, n_solo_ap, n_solo_pa, n_ambas) -> bool:
    return max(0, n_ap - n_solo_ap) + max(0, n_pa - n_solo_pa) <= n_ambas


def muestra_ap(orden, n: int, f_ap: float):
    """Prefijo de n pacientes con round(n * f_ap) en AP. Devuelve [(patient_id, vista)] o lanza ValueError."""
    lista_ap, lista_pa, sa, sp, nb = orden
    n_ap = int(round(n * f_ap))
    n_pa = n - n_ap
    if not factible(n_ap, n_pa, sa, sp, nb):
        raise ValueError("no factible: %d AP y %d PA con %d solo AP, %d solo PA y %d con ambas" % (n_ap, n_pa, sa, sp, nb))
    return [(p, "AP") for p in lista_ap[:n_ap]] + [(p, "PA") for p in lista_pa[:n_pa]]


def max_factible(orden, f_ap: float) -> int:
    """Mayor n factible con fracción AP f_ap en un estrato."""
    _, _, sa, sp, nb = orden
    n = 0
    while factible(int(round((n + 1) * f_ap)), (n + 1) - int(round((n + 1) * f_ap)), sa, sp, nb):
        n += 1
    return n


# ---------------------------------------------------------------------------------------------- hashes
def cargar_hashes(csv, prefijo: str = "") -> pd.DataFrame:
    """CSV de hash_images.py con image_path relativo; `prefijo` se antepone a la ruta (p. ej. data/raw/x/)."""
    d = pd.read_csv(csv, dtype=str, keep_default_na=False)
    d = d[(d.error == "") & (d.phash != "")].copy()
    d["image_path"] = prefijo + d["image_path"]
    return d.reset_index(drop=True)


def _thumb(path):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        im.draft("L", (256, 256))
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            a = np.asarray(im, dtype=np.float64)
            lo, hi = np.percentile(a, [0.5, 99.5])
            a = np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
            im2 = Image.fromarray(a, "L")
        else:
            im2 = im.convert("L")
        x = np.asarray(im2.resize((64, 64), Image.BILINEAR), dtype=np.float64).ravel()
    return (x - x.mean()) / (x.std() + 1e-6)


def casi_duplicados(h: pd.DataFrame, abs_of, grupo_col: str | None = "patient_id", max_p=4, max_d=12, min_bits=8,
                    max_bucket=20000, min_corr=0.97, verificar=True, log=print) -> pd.DataFrame:
    """Pares casi idénticos dentro de `h` (una fila por imagen con image_path, md5, phash, dhash y, si se da,
    `grupo_col`). Solo se devuelven pares con distinto `grupo_col` (pacientes distintos); `abs_of(image_path)` da la
    ruta absoluta para la verificación. Columnas: a, b, kind, phash_dist, dhash_dist, thumb_corr, confirmed."""
    h = h.reset_index(drop=True)
    ph = np.array([int(x, 16) for x in h.phash], dtype=np.uint64)
    dh = np.array([int(x, 16) for x in h.dhash], dtype=np.uint64)
    pc = fo.popcount64(ph)
    ok = (pc >= min_bits) & (pc <= 64 - min_bits)
    ia, ib, skipped = fo.candidate_pairs(ph, ph, dh, dh, max_p + 1, max_bucket, max_p, max_d)
    keep = ok[ia] & ok[ib] & (ia < ib)
    ia, ib = ia[keep], ib[keep]
    m = h[["md5"]].reset_index().merge(h[["md5"]].reset_index(), on="md5", suffixes=("_a", "_b"))
    m = m[m.index_a < m.index_b]
    pares = pd.DataFrame({"i": np.r_[m.index_a.to_numpy(), ia], "j": np.r_[m.index_b.to_numpy(), ib],
                          "kind": ["md5"] * len(m) + ["phash"] * len(ia)})
    pares = pares.sort_values("kind").drop_duplicates(["i", "j"]).reset_index(drop=True)
    if grupo_col:
        g = h[grupo_col].to_numpy()
        pares = pares[g[pares.i] != g[pares.j]].reset_index(drop=True)
    log("casi duplicados: %d candidatos entre grupos distintos (cubetas omitidas %d)" % (len(pares), skipped))
    out = pd.DataFrame({"a": h.image_path.to_numpy()[pares.i], "b": h.image_path.to_numpy()[pares.j],
                        "kind": pares.kind.to_numpy(),
                        "phash_dist": fo.popcount64(ph[pares.i] ^ ph[pares.j]) if len(pares) else [],
                        "dhash_dist": fo.popcount64(dh[pares.i] ^ dh[pares.j]) if len(pares) else []})
    corr = []
    if verificar:
        for a, b, k in zip(out.a, out.b, out.kind):
            try:
                corr.append(1.0 if k == "md5" else float(np.mean(_thumb(abs_of(a)) * _thumb(abs_of(b)))))
            except Exception:
                corr.append(float("nan"))
    else:
        corr = [float("nan")] * len(out)
    out["thumb_corr"] = corr
    out["confirmed"] = (out.kind == "md5") | (out.thumb_corr >= min_corr)
    return out


from functools import lru_cache


@lru_cache(maxsize=8000)
def _thumb_c(path):
    return _thumb(path).astype(np.float32)


def verificar_pares(csv, abs_of, max_s=150, min_corr=0.97, log=print) -> int:
    """Completa thumb_corr y confirmed en un CSV de pares (salida de casi_duplicados con verificar=False), por
    tandas y con guardado incremental. Devuelve el número de pares que quedan sin verificar."""
    import time
    d = pd.read_csv(csv)
    t0 = time.time()
    todo = d[d.thumb_corr.isna() & (d.kind != "md5")].sort_values(["a", "b"]).index.tolist()
    d.loc[d.kind == "md5", "thumb_corr"] = 1.0
    hechos = 0
    for i in todo:
        if time.time() - t0 > max_s:
            break
        try:
            x, y = _thumb_c(abs_of(d.at[i, "a"])), _thumb_c(abs_of(d.at[i, "b"]))
            d.at[i, "thumb_corr"] = float(np.mean(x.astype(np.float64) * y))
        except Exception:
            d.at[i, "thumb_corr"] = -9.0     # ilegible: se marca y no se confirma
        hechos += 1
    d["confirmed"] = (d.kind == "md5") | (d.thumb_corr >= min_corr)
    tmp = Path(str(csv) + ".tmp")
    d.to_csv(tmp, index=False)
    os.replace(tmp, csv)
    quedan = int(d.thumb_corr.isna().sum())
    log("verificados %d pares en %.0f s; quedan %d; confirmados %d" % (hechos, time.time() - t0, quedan,
                                                                     int(d.confirmed.sum())))
    return quedan


def grupos_union(pares: pd.DataFrame, a="a", b="b") -> dict:
    """Componentes conexas de los pares (union-find): {elemento: representante}."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for x, y in zip(pares[a], pares[b]):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)
    return {x: find(x) for x in parent}


def tabla_clase_fuente(df: pd.DataFrame, filas="source", cols=("split",)) -> pd.DataFrame:
    return df.groupby([filas] + list(cols) + ["label"]).size().unstack("label", fill_value=0)
