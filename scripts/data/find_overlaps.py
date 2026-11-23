#!/usr/bin/env python3
"""Cruza dos CSV de hashes (salida de hash_images.py) y lista pares candidatos a duplicado.

Criterios:
  md5         fichero identico byte a byte
  phash       distancia de Hamming del pHash <= --max-phash y del dHash <= --max-dhash
Para evitar la comparacion de todos contra todos se indexa por bandas (principio del palomar): con
--max-phash = T se parte el pHash de 64 bits en T+1 bandas; dos hashes a distancia <= T coinciden
al menos en una banda. Todo el cruce esta vectorizado con numpy.
Con --verify cada par se confirma con la correlacion de Pearson de miniaturas 64x64 (>= --min-corr).
Con --self se buscan duplicados dentro de un mismo CSV.

Calibracion medida en la VM (logs/tmp_datos/smoke_hash.log): entre radiografias distintas de COVIDx
y NIH la distancia pHash minima observada fue 6; copias reescaladas y recodificadas en JPEG quedaron
a distancia 0 a 2. Por eso el umbral por defecto es 4 y se recomienda --verify.

Ejemplo:
  python3 find_overlaps.py --a data/hashes/covidx_rsna.csv --b data/hashes/nih_cxr14.csv \
      --name-a covidx --name-b nih --verify --root-a /home/administrador/notebooks/covid \
      --root-b /home/administrador/notebooks/covid --out results/overlap_rsna_nih.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(x):
    x = np.ascontiguousarray(x, dtype=np.uint64)
    return _POP8[x.view(np.uint8)].reshape(-1, 8).sum(axis=1)


def load(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df[(df.error == "") & (df.phash != "")].reset_index(drop=True)
    ph = np.array([int(h, 16) for h in df.phash], dtype=np.uint64)
    dh = np.array([int(h, 16) for h in df.dhash], dtype=np.uint64)
    return df, ph, dh


def band_keys(h, nb):
    width = 64 // nb
    keys = []
    for i in range(nb):
        w = width if i < nb - 1 else 64 - (nb - 1) * width
        shift = 64 - i * width - w
        keys.append((h >> np.uint64(shift)) & np.uint64((1 << w) - 1))
    return keys


def candidate_pairs(phA, phB, dhA, dhB, nb, max_bucket, max_p, max_d, chunk=1000):
    """Devuelve pares (i, j) que comparten alguna banda y pasan los umbrales; filtra por bloques
    para mantener acotada la memoria."""
    keysA, keysB = band_keys(phA, nb), band_keys(phB, nb)
    sorted_b = []
    for kb in keysB:
        order = np.argsort(kb, kind="stable")
        sorted_b.append((order, kb[order]))
    out_i, out_j, skipped = [], [], 0
    for s0 in range(0, len(phA), chunk):
        sl = slice(s0, min(s0 + chunk, len(phA)))
        for ka, (order, kbs) in zip(keysA, sorted_b):
            kac = ka[sl]
            lo = np.searchsorted(kbs, kac, side="left")
            hi = np.searchsorted(kbs, kac, side="right")
            cnt = hi - lo
            big = cnt > max_bucket
            skipped += int(big.sum())
            cnt = np.where(big, 0, cnt)
            tot = int(cnt.sum())
            if tot == 0:
                continue
            ia = np.repeat(np.arange(sl.start, sl.stop), cnt)
            offs = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            ib = order[np.repeat(lo, cnt) + offs]
            ok = (popcount64(phA[ia] ^ phB[ib]) <= max_p) & (popcount64(dhA[ia] ^ dhB[ib]) <= max_d)
            out_i.append(ia[ok])
            out_j.append(ib[ok])
    if not out_i:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), skipped
    pair = np.unique(np.stack([np.concatenate(out_i), np.concatenate(out_j)], axis=1), axis=0)
    return pair[:, 0], pair[:, 1], skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b")
    ap.add_argument("--self", action="store_true", help="duplicados dentro de --a")
    ap.add_argument("--name-a", default="a")
    ap.add_argument("--name-b", default="b")
    ap.add_argument("--max-phash", type=int, default=4)
    ap.add_argument("--max-dhash", type=int, default=12)
    ap.add_argument("--min-bits", type=int, default=8, help="descarta hashes con menos de N bits a 1 o a 0")
    ap.add_argument("--max-bucket", type=int, default=20000, help="ignora cubetas mayores (hashes triviales)")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--root-a", default="")
    ap.add_argument("--root-b", default="")
    ap.add_argument("--min-corr", type=float, default=0.97)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.self and not a.b:
        ap.error("indique --b o --self")

    A, phA, dhA = load(a.a)
    B, phB, dhB = (A, phA, dhA) if a.self else load(a.b)
    okA = (popcount64(phA) >= a.min_bits) & (popcount64(phA) <= 64 - a.min_bits)
    okB = (popcount64(phB) >= a.min_bits) & (popcount64(phB) <= 64 - a.min_bits)
    print(f"A={len(A)} (degenerados {int((~okA).sum())})  B={len(B)} (degenerados {int((~okB).sum())})")

    # md5 exactos
    m = A[["md5"]].reset_index().merge(B[["md5"]].reset_index(), on="md5", suffixes=("_a", "_b"))
    im, jm = m.index_a.to_numpy(), m.index_b.to_numpy()
    # perceptuales
    ia, ib, skipped = candidate_pairs(phA, phB, dhA, dhB, a.max_phash + 1, a.max_bucket,
                                      a.max_phash, a.max_dhash)
    keep = okA[ia] & okB[ib]
    ia, ib = ia[keep], ib[keep]
    allp = pd.DataFrame({"i": np.concatenate([im, ia]), "j": np.concatenate([jm, ib]),
                         "kind": ["md5"] * len(im) + ["phash"] * len(ia)})
    allp = allp.sort_values("kind").drop_duplicates(["i", "j"], keep="first")
    if a.self:
        allp = allp[allp.i < allp.j]
    i, j = allp.i.to_numpy(), allp.j.to_numpy()
    res = pd.DataFrame({
        f"path_{a.name_a}": A.image_path.to_numpy()[i], f"path_{a.name_b}": B.image_path.to_numpy()[j],
        "kind": allp.kind.to_numpy(),
        "phash_dist": popcount64(phA[i] ^ phB[j]) if len(i) else [],
        "dhash_dist": popcount64(dhA[i] ^ dhB[j]) if len(i) else [],
        f"size_{a.name_a}": (A.width + "x" + A.height).to_numpy()[i],
        f"size_{a.name_b}": (B.width + "x" + B.height).to_numpy()[j],
    })
    print(f"candidatos por bandas evaluados, cubetas omitidas={skipped}; pares={len(res)} "
          f"(md5 {int((res.kind == 'md5').sum())})")
    if a.verify and len(res):
        from PIL import Image

        def thumb(root, p):
            with Image.open(os.path.join(root, p)) as im_:
                im_.draft("L", (256, 256))
                x = np.asarray(im_.convert("L").resize((64, 64), Image.BILINEAR), dtype=np.float64).ravel()
            return (x - x.mean()) / (x.std() + 1e-6)
        rb = a.root_a if a.self else a.root_b
        corr = []
        for pa, pb in zip(res[f"path_{a.name_a}"], res[f"path_{a.name_b}"]):
            try:
                corr.append(float(np.mean(thumb(a.root_a, pa) * thumb(rb, pb))))
            except Exception:
                corr.append(float("nan"))
        res["thumb_corr"] = corr
        res["confirmed"] = (res.kind == "md5") | (res.thumb_corr >= a.min_corr)
        print(f"verificacion: {int(res.confirmed.sum())} de {len(res)} pares confirmados (corr >= {a.min_corr})")
    res.to_csv(a.out, index=False)
    if len(res):
        print(f"imagenes de A con pareja={res[f'path_{a.name_a}'].nunique()}, "
              f"de B con pareja={res[f'path_{a.name_b}'].nunique()}")
        print("distribucion de distancia pHash:", res.phash_dist.value_counts().sort_index().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
