"""Exploración posterior a la Puerta 3 (no confirmatoria; docs/07_piloto.md, "Exploración de una arquitectura nueva").

Extrae, de corridas YA entrenadas, las características agrupadas, las de la penúltima capa y el logit de:
la validación completa, los negativos de bimcv de entrenamiento, una muestra de positivos de bimcv de
entrenamiento, una muestra de rsna de entrenamiento y una muestra de las fuentes solo positivas.
No entrena nada, no modifica las corridas y no toca el test (exige test bloqueado).

    python scripts/diag/expl_lectura_features.py --runs results/train/baseline_erm/seed42 ... --out results/diag/expl_lectura
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def extraer(run: Path, state: str, out: Path, n_pos: int, n_rsna: int, n_otros: int, max_n: int, workers):
    import pandas as pd
    from src.baselines.common import penultimate
    from src.baselines.trainer import BaselineTrainer
    cfg = json.load(open(run / "config_resolved.json", encoding="utf-8"))
    seed = int(cfg.get("seed", 0))
    cfg["model"]["init"] = {"type": "none", "path": None}      # los pesos vienen del estado guardado
    if cfg["data"].get("manifest") and not Path(cfg["data"]["manifest"]).exists():
        alt = ROOT / "data" / "manifests" / "expl" / Path(cfg["data"]["manifest"]).name      # misma copia, otra máquina
        if not alt.exists():
            raise SystemExit("no se encuentra el manifiesto %s" % cfg["data"]["manifest"])
        cfg["data"]["manifest"] = str(alt)
    if workers is not None:
        cfg["data"]["num_workers"] = int(workers)
    tag = "%s_%s_%s" % (run.parent.name, run.name, state)
    tr = BaselineTrainer(cfg, seed, resume=False, out_dir=str(out / "_trainer" / tag), echo=False)
    if not tr.test_locked:
        raise SystemExit("el test debe estar bloqueado")
    tr.model.load_state_dict(torch.load(run / "states" / ("%s.pt" % state), map_location=tr.device))
    tr.model.eval()
    fr = tr.frames["train"]
    rs = 20261113
    b = fr[fr["source"] == "bimcv"]
    pos = b[b["label"] == 1]
    otros = fr[fr["source"].isin(["stonybrook", "ricord", "sirm"])]
    rsna = fr[fr["source"] == "rsna"]
    sets = {"val": tr.frames["val"],
            "tr_bimcv_neg": b[b["label"] == 0],
            "tr_bimcv_pos": pos.sample(n=min(n_pos, len(pos)), random_state=rs),
            "tr_rsna": rsna.sample(n=min(n_rsna, len(rsna)), random_state=rs),
            "tr_otros_pos": otros.sample(n=min(n_otros, len(otros)), random_state=rs)}
    if cfg["data"].get("manifest"):
        # corrida con reserva: las filas con split_clean == "reserva" no las vio el tronco
        from src.data.manifest import load_manifest
        from src.train.trainer import resolve_data
        man = load_manifest(resolve_data(cfg["data"]["manifest"]))
        r = man[man["split_clean"] == "reserva"]
        if len(r):
            nueva = r["image_id"].str.extract(r"^sub-S(\d+)_")[0].str.len() == 6
            sets["res_neg"] = r[r["label"] == 0]
            sets["res_pos_antigua"] = r[(r["label"] == 1) & ~nueva]
            sets["res_pos_nueva"] = r[(r["label"] == 1) & nueva]
    sets = {k: v for k, v in sets.items() if len(v)}
    meta, P, H, Z = [], [], [], []
    net = tr.model
    with torch.no_grad():
        for name, df in sets.items():
            df = df.reset_index(drop=True)
            if max_n:
                df = df.iloc[:max_n].reset_index(drop=True)
            key = "expl_" + name
            tr.frames[key] = df
            ids = []
            for bt in tr.eval_loader(key):
                x = bt["image"].to(tr.device).float()
                p = net.net.pooled(x)
                h = penultimate(net.net, p)
                z = net.net.fc2(h).reshape(-1)
                P.append(p.float().cpu().numpy()); H.append(h.float().cpu().numpy()); Z.append(z.float().cpu().numpy())
                ids += list(map(str, bt["image_id"]))
            if ids != df["image_id"].astype(str).tolist():
                raise SystemExit("orden de imágenes inesperado en %s" % name)
            m = df[["image_id", "patient_id", "label", "source"]].copy()
            m["conjunto"] = name
            meta.append(m)
            del tr.frames[key]
            print("[%s] %s: %d imágenes" % (tag, name, len(df)), flush=True)
    meta = pd.concat(meta, ignore_index=True)
    meta.to_csv(out / ("%s_meta.csv" % tag), index=False)
    np.savez_compressed(out / ("%s.npz" % tag), pooled=np.concatenate(P).astype(np.float32),
                        penult=np.concatenate(H).astype(np.float32), logit=np.concatenate(Z).astype(np.float32))
    print("[%s] guardado: %d filas, pooled %s, penult %s" % (tag, len(meta), np.concatenate(P).shape, np.concatenate(H).shape))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--state", default="best")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-pos", type=int, default=6000)
    ap.add_argument("--n-rsna", type=int, default=2000)
    ap.add_argument("--n-otros", type=int, default=1500)
    ap.add_argument("--max-n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for r in a.runs:
        extraer(Path(r), a.state, out, a.n_pos, a.n_rsna, a.n_otros, a.max_n, a.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
