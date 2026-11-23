"""Probabilidades por imagen con TTA (imagen y espejo horizontal, como predict_probs del artículo 1).

Uso como biblioteca: `predict_loader(model, loader, device, tta=True, precision="bf16")` devuelve un dict con
image_id, label, source, patient y prob. `save_predictions` escribe CSV (y npz) por estado y conjunto.

Uso como script (modelo ya entrenado por src.train.run):
    python -m src.eval.predict --run-dir results/train/<exp>/seed42 --state ema --sets val val_robust

Los conjuntos test, test_* y COVIDGR solo se predicen tras la congelación del protocolo (src/eval/desbloqueo.py,
hallazgo M1 del revisor); por defecto se predice solo `val`.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _autocast(device, precision):
    if precision == "bf16" and str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def predict_loader(model, loader, device="cpu", tta=True, precision="fp32", max_batches=None):
    model.eval()
    ids, labels, sources, patients, probs = [], [], [], [], []
    for bi, b in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = b["image"].to(device, non_blocking=True).float()
        with _autocast(device, precision):
            p = torch.sigmoid(model(x).float())
            if tta:
                p = 0.5 * (p + torch.sigmoid(model(torch.flip(x, dims=[3])).float()))
        probs.append(p.float().cpu().numpy().reshape(-1))
        ids += list(b["image_id"]); patients += list(b["patient"])
        labels.append(b["label"].numpy().reshape(-1)); sources.append(b["source"].numpy().reshape(-1))
    return {"image_id": np.array(ids), "label": np.concatenate(labels).astype(int) if labels else np.zeros(0, int),
            "source": np.concatenate(sources).astype(int) if sources else np.zeros(0, int),
            "patient": np.array(patients), "prob": np.concatenate(probs) if probs else np.zeros(0)}


def save_predictions(pred: dict, path: Path):
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({k: pred[k] for k in ("image_id", "label", "source", "patient", "prob")})
    tmp = path.with_name("." + path.name + ".tmp")
    df.to_csv(tmp, index=False, float_format="%.8f")
    tmp.replace(path)
    return path


def summarize(pred: dict) -> dict:
    from src.train.metrics import exact_metric
    return {"n": int(len(pred["prob"])), "auc_roc": exact_metric("auc_roc", pred["label"], pred["prob"]),
            "auc_pr": exact_metric("auc_pr", pred["label"], pred["prob"])}


def main(argv=None):
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    from src.train.trainer import Trainer
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--state", default="ema", choices=["last", "best", "ema"])
    ap.add_argument("--sets", nargs="+", default=["val"])
    a = ap.parse_args(argv)
    from src.eval.desbloqueo import exigir_si_bloqueado
    exigir_si_bloqueado(a.sets, "src.eval.predict --run-dir %s --state %s" % (a.run_dir, a.state))
    run = Path(a.run_dir)
    cfg = json.load(open(run / "config_resolved.json", encoding="utf-8"))
    tr = Trainer(cfg, seed=cfg["seed"], resume=False)
    sd = torch.load(run / "states" / ("%s.pt" % a.state), map_location="cpu")
    tr.model.load_state_dict(sd)
    for s in a.sets:
        pred = predict_loader(tr.model, tr.eval_loader(s), tr.device, tta=tr.cfg["eval"]["tta"], precision=tr.precision)
        save_predictions(pred, run / "predictions" / ("%s_%s.csv" % (a.state, s)))
        print(s, summarize(pred))


if __name__ == "__main__":
    main()
