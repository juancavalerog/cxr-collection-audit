#!/usr/bin/env python3
"""Semillas 5 a 9 de B3a en rho 0.90, 0.95 y 0.99, orientación A positiva (docs/08_evaluacion.md, enmiendas E1.2 y
E8, punto 1; PLAN.md Section 13.12).

Usa el constructor de B3a sin cambiarlo (scripts/data/build_b3a.py: `preparar`, `Ordenes`, `construir_manifiesto`):
mismas reservas de pacientes, misma imagen fija por paciente y vista, mismas ordenaciones fijas de `val_desconf`,
`test_desconf` y `test_invertido`; solo la semilla cambia las ordenaciones de `train` y `val`. No reescribe ningún
fichero existente: un manifiesto que ya existe se compara byte a byte con el regenerado y, si difiere, el guion
termina con error sin tocarlo. Tampoco toca los auxiliares de data/manifests/bench/b3a (imagenes_unicas.csv,
indice.json, reservas_pacientes.csv), que describen los 46 manifiestos originales; su lista de imágenes propia va a
logs/datos_extra/.

    python3 scripts/data/build_b3a_extra.py --paso build        # escribe los 15 manifiestos nuevos (si faltan)
    python3 scripts/data/build_b3a_extra.py --paso regenerar    # regenera los 46 originales en /tmp y compara
                                                                # con logs/supervisor/sha_manifiestos_bench_20261117.txt
    python3 scripts/data/build_b3a_extra.py --paso delta        # lista de imágenes nuevas para la caché
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_b3a as B  # noqa: E402
from src.data.bench import list_manifests, read_csv, write_bench  # noqa: E402

RHOS_EXTRA = (0.9, 0.95, 0.99)
SEEDS_EXTRA = (5, 6, 7, 8, 9)
ORIENT = "apos"
SHA_REF = ROOT / "logs" / "supervisor" / "sha_manifiestos_bench_20261117.txt"
LOGDIR = ROOT / "logs" / "datos_extra"
DELTA = LOGDIR / "b3a_s5_9_imagenes_nuevas.csv"
LISTA_NUEVAS = LOGDIR / "b3a_s5_9_imagenes.csv"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def originales() -> list[Path]:
    """Los 46 manifiestos originales (los de la lista de huellas de referencia)."""
    out = []
    for ln in SHA_REF.read_text().splitlines():
        h, p = ln.split(None, 1)
        if p.startswith("data/manifests/bench/b3a/") and p.endswith(".csv") and ("/apos/" in p or "/bpos/" in p):
            out.append(ROOT / p)
    return sorted(out)


def nuevos() -> list[tuple[float, int, Path]]:
    return [(r, s, B.nombre(ORIENT, r, s)) for r in RHOS_EXTRA for s in SEEDS_EXTRA]


def contexto(log=print):
    img, pac, exc = B.preparar(log)
    img = img.merge(pac[["source", "patient_id", "reserva"]], on=["source", "patient_id"], how="left")
    img = img.reset_index(drop=True)
    img_idx = {(s, p, c, v): i for i, (s, p, c, v) in enumerate(zip(img.source, img.patient_id, img.clase, img.view))}
    return img, pac, img_idx, B.Ordenes(pac)


def paso_build(log=print) -> int:
    img, pac, img_idx, ordenes = contexto(log)
    ref = {}
    for ln in SHA_REF.read_text().splitlines():
        h, p = ln.split(None, 1)
        ref[p] = h
    escritos, iguales = [], []
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        for rho, seed, dst in nuevos():
            rel = str(dst.relative_to(ROOT))
            if rel in ref:
                raise SystemExit("%s figura entre los manifiestos originales: no se toca" % rel)
            d, meta = B.construir_manifiesto(ORIENT, rho, seed, ordenes, img, img_idx)
            meta = dict(meta)
            meta["generador"] = "scripts/data/build_b3a_extra.py (constructor scripts/data/build_b3a.py)"
            meta["ampliacion"] = "E8 punto 1 (docs/08_evaluacion.md): semillas 5 a 9 en rho 0.90, 0.95 y 0.99"
            tmp = Path(td) / dst.name
            write_bench(d, meta, tmp)
            if dst.exists() or dst.with_suffix(".json").exists():
                same = (dst.exists() and dst.with_suffix(".json").exists() and sha(dst) == sha(tmp)
                        and sha(dst.with_suffix(".json")) == sha(tmp.with_suffix(".json")))
                if not same:
                    raise SystemExit("%s ya existe y difiere del regenerado: no se sobrescribe" % rel)
                iguales.append(rel)
                continue
            dst.write_bytes(tmp.read_bytes())
            dst.with_suffix(".json").write_bytes(tmp.with_suffix(".json").read_bytes())
            escritos.append(rel)
    log("manifiestos nuevos escritos: %d; ya existentes e idénticos: %d" % (len(escritos), len(iguales)))
    LOGDIR.mkdir(parents=True, exist_ok=True)
    with open(LOGDIR / "b3a_s5_9_sha256.txt", "w") as f:
        for rho, seed, dst in nuevos():
            for p in (dst, dst.with_suffix(".json")):
                f.write("%s  %s\n" % (sha(p), p.relative_to(ROOT)))
    return paso_delta(log)


def paso_delta(log=print) -> int:
    """Imágenes de los 15 manifiestos nuevos y, de ellas, las que no aparecen en los 46 originales."""
    cols = ["image_id", "image_path", "raiz", "dataset", "source"]
    orig = pd.concat([read_csv(p)[cols] for p in originales()]).drop_duplicates()
    new = pd.concat([read_csv(p)[cols] for _, _, p in nuevos()]).drop_duplicates()
    if new.image_id.duplicated().any() or orig.image_id.duplicated().any():
        raise SystemExit("image_id con rutas distintas")
    both = pd.concat([orig, new]).drop_duplicates()
    if both.image_id.duplicated().any():
        raise SystemExit("image_id con rutas distintas entre originales y nuevos")
    delta = new[~new.image_id.isin(set(orig.image_id))].sort_values("image_id").reset_index(drop=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)
    new.sort_values("image_id").to_csv(LISTA_NUEVAS, index=False)
    delta.to_csv(DELTA, index=False)
    res = {"imagenes_originales_46": int(len(orig)), "imagenes_15_nuevos": int(len(new)),
           "imagenes_nuevas_para_cache": int(len(delta)),
           "por_fuente": delta.source.value_counts().sort_index().to_dict(),
           "imagenes_union_61": int(len(both))}
    log(json.dumps(res, ensure_ascii=False))
    (LOGDIR / "b3a_s5_9_delta.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    return 0


def paso_regenerar(log=print) -> int:
    """Regenera en /tmp los 46 manifiestos originales con el constructor actual y compara sus huellas con la lista
    de referencia (CSV y JSON)."""
    img, pac, img_idx, ordenes = contexto(log)
    ref = {}
    for ln in SHA_REF.read_text().splitlines():
        h, p = ln.split(None, 1)
        ref[p] = h
    malos, n = [], 0
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        for p in originales():
            orient, stem = p.parent.name, p.stem
            rho, seed = float(stem.split("_")[0][3:]), int(stem.split("_s")[1])
            d, meta = B.construir_manifiesto(orient, rho, seed, ordenes, img, img_idx)
            tmp = Path(td) / orient / p.name
            write_bench(d, meta, tmp)
            for a, b in ((tmp, p), (tmp.with_suffix(".json"), p.with_suffix(".json"))):
                n += 1
                if sha(a) != ref[str(b.relative_to(ROOT))]:
                    malos.append(str(b.relative_to(ROOT)))
    log("regenerados %d ficheros (46 CSV y 46 JSON) en /tmp; distintos de la lista de referencia: %d %s"
        % (n, len(malos), malos[:3]))
    return 0 if not malos and n == 92 else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--paso", choices=["build", "regenerar", "delta"], required=True)
    a = ap.parse_args(argv)
    return {"build": paso_build, "regenerar": paso_regenerar, "delta": paso_delta}[a.paso]()


if __name__ == "__main__":
    sys.exit(main())
