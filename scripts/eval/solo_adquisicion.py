"""Estadísticos de adquisición para el clasificador de solo adquisición (C1-iv; src/eval/adquisicion.py).

    python scripts/eval/solo_adquisicion.py --conjunto ajuste [--segundos 150]    # bimcv de entrenamiento (2 647)
    python scripts/eval/solo_adquisicion.py --conjunto val_robust                # bimcv de validación (3 407)
    python scripts/eval/solo_adquisicion.py --conjunto test_bimcv                # solo tras la congelación del protocolo

Solo CPU, por tandas reanudables (repetir la orden hasta que diga "completo"): las tandas van a
results/c1/adquisicion/_tandas/ y al terminar se escriben `<conjunto>_estadisticos.npz` (X, image_id) y
`<conjunto>_meta.csv`. El test se escribe en results/c1/adquisicion_test/. La evaluación (variantes (a) y (b) y la
diferencia frente a ERM) la hace `scripts/eval/auditoria_covidx.py --c1 val|test`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import adquisicion as AQ  # noqa: E402
from src.eval.desbloqueo import es_bloqueado, exigir_si_bloqueado  # noqa: E402

TANDA = 400


def marco(conjunto: str, registro=None) -> pd.DataFrame:
    from src.data.manifest import load_manifest
    man = load_manifest()
    if conjunto == "ajuste":
        d = AQ.muestra_ajuste(man)
    elif conjunto == "val_robust":
        d = man[(man["split_clean"] == "val") & (man["source"] == "bimcv")]
    elif conjunto == "test_bimcv":
        if not registro:
            raise SystemExit("test_bimcv exige la congelación")
        d = man[(man["split_clean"] == "test") & (man["source"] == "bimcv")]
    else:
        raise SystemExit("conjunto desconocido: %s" % conjunto)
    ci, mi = AQ.indices()
    d = d.merge(ci, on="image_id", how="left").merge(mi, on="image_id", how="left")
    falt = d["rel_path"].isna() | d["path"].isna()
    if falt.any():
        raise SystemExit("%d imágenes de %s sin caché o sin máscara (p. ej. %s)" % (falt.sum(), conjunto, d.loc[falt, "image_id"].iloc[0]))
    return d.reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conjunto", required=True, choices=["ajuste", "val_robust", "test_bimcv"])
    ap.add_argument("--segundos", type=float, default=150)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    reg = exigir_si_bloqueado([a.conjunto], "solo_adquisicion")
    out = Path(a.out or ROOT / "results" / "c1" / ("adquisicion_test" if es_bloqueado(a.conjunto) else "adquisicion"))
    td = out / "_tandas"
    td.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    d = marco(a.conjunto, reg)
    t0 = time.time()
    n = int(np.ceil(len(d) / TANDA))
    for k in range(n):
        f = td / ("%s_%03d.npz" % (a.conjunto, k))
        if f.exists():
            continue
        if time.time() - t0 > a.segundos:
            break
        sub = d.iloc[k * TANDA:(k + 1) * TANDA]
        par = [AQ.cargar_una(r, p, q) for r, p, q in zip(sub["rel_path"], sub["path"], sub["quality"])]
        X = AQ.estadisticos(torch.from_numpy(np.stack([x for x, _ in par])), torch.from_numpy(np.stack([m for _, m in par])))
        tmp = td / (".%s_%03d.tmp.npz" % (a.conjunto, k))
        np.savez(tmp, X=X, image_id=sub["image_id"].to_numpy().astype(str))
        tmp.replace(f)
        print("tanda %d de %d (%.0f s)" % (k + 1, n, time.time() - t0), flush=True)
    hechas = [td / ("%s_%03d.npz" % (a.conjunto, k)) for k in range(n)]
    if not all(f.exists() for f in hechas):
        print("%s: %d de %d tandas; repita la orden" % (a.conjunto, sum(f.exists() for f in hechas), n))
        return 3
    Z = [np.load(f) for f in hechas]
    X = np.concatenate([z["X"] for z in Z])
    ids = np.concatenate([z["image_id"] for z in Z])
    if ids.tolist() != d["image_id"].astype(str).tolist():
        raise SystemExit("orden de imágenes inesperado")
    np.savez_compressed(out / ("%s_estadisticos.npz" % a.conjunto), X=X, image_id=ids, nombres=np.array(AQ.NOMBRES))
    d[["image_id", "patient_id", "label", "source"]].assign(conjunto=a.conjunto).to_csv(out / ("%s_meta.csv" % a.conjunto), index=False)
    print("%s: completo, %d imágenes, %d estadísticos" % (a.conjunto, len(d), X.shape[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
