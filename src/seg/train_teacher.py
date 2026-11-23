"""Preparación del conjunto y entrenamiento del profesor de segmentación (pulmones y corazón).

    # 1) conjunto del profesor desde CheXmask (servidor; CheXmask en data/raw/chexmask/OriginalResolution)
    python -m src.seg.train_teacher prepare --chexmask-dir data/raw/chexmask/OriginalResolution \
        --n-nih 12000 --n-chexpert 12000 --out data/masks/teacher_set --workers 16
    # 2) entrenamiento (una GPU; checkpoint y reanudación; SIGTERM y temperatura con salida 75)
    python -m src.seg.train_teacher train --set data/masks/teacher_set --out results/seg_teacher --epochs 30

Partición por paciente (90/10 por hash del paciente). Por defecto se excluyen todos los pacientes NIH con imágenes RSNA
en COVIDx, de cualquier partición (data/manifests/covidx_rsna_to_nih_map.csv), para que el profesor no vea
ninguna imagen de COVIDx y la calidad de sus pseudomáscaras no dependa de la fuente; --exclude-covidx eval o
ninguna lo relajan. Dice de validación por canal en cada época;
se guardan last.pt y best.pt (mejor Dice medio de los dos canales) en <out>/states y el registro en
<out>/train_log.jsonl. El aumento es geométrico suave y fotométrico (sin volteo, para no invertir la anatomía).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runner"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

EXIT_THERMAL = 75


def _log(path, event, **kw):
    rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **kw}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    print(json.dumps(rec, default=float), flush=True)


def prepare(a):
    import pandas as pd
    from src.seg.chexmask import match_manifest, nih_patients_in_covidx
    from src.seg.teacher import build_teacher_set
    d = Path(a.chexmask_dir)
    man_dir = ROOT / "data" / "manifests"
    mode = "ninguna" if a.no_exclude_covidx_eval else a.exclude_covidx
    map_csv = man_dir / "covidx_rsna_to_nih_map.csv"
    excl = (set() if mode == "ninguna" else
            nih_patients_in_covidx(map_csv, None if mode == "todas" else ["val", "test"]))
    print("exclusión de pacientes NIH presentes en COVIDx: %s (%d pacientes)" % (mode, len(excl)))
    parts = []
    for name, csvn, mann, n in (("nih_cxr14", "ChestX-Ray8", "nih_cxr14.csv", a.n_nih),
                                ("chexpert_small", "CheXpert", "chexpert_small.csv", a.n_chexpert)):
        if n <= 0:
            continue
        cands = [p for p in (d / (csvn + ".csv"), d / (csvn + ".csv.gz")) if p.exists()]
        if not cands:
            raise FileNotFoundError("no está %s en %s" % (csvn, d))
        man = pd.read_csv(man_dir / mann, dtype={"patient_id": str})
        sel = match_manifest(cands[0], name, man, a.min_dice, n, seed=a.seed,
                             exclude_patients=excl if name == "nih_cxr14" else ())
        print("%s: %d filas elegidas (Dice RCA medio >= %.2f, una por paciente)" % (name, len(sel), a.min_dice))
        parts.append(sel)
    rows = pd.concat(parts, ignore_index=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows.drop(columns=["Left Lung", "Right Lung", "Heart"]).to_csv(out / "selection.csv", index=False)
    build_teacher_set(rows, out, a.res, a.workers)
    return 0


def train(a) -> int:
    from thermal import GracefulStop, ThermalLimitReached, ThermalSelfCheck
    from src.data.datasets import NewAugment, make_generator
    from src.seg.teacher import SegTeacher, TeacherSet, dice_per_channel, seg_loss
    from src.train.optim import warmup_cosine
    out = Path(a.out)
    (out / "states").mkdir(parents=True, exist_ok=True)
    logp = out / "train_log.jsonl"
    os.environ["PYTHONHASHSEED"] = str(a.seed)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tr_ds, va_ds = TeacherSet(a.set, "train"), TeacherSet(a.set, "val")
    if len(tr_ds) == 0 or len(va_ds) == 0:
        raise SystemExit("conjunto del profesor vacío o sin partición de validación: %s" % a.set)
    model = SegTeacher(a.encoder, pretrained=not a.no_pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    spe = a.steps_per_epoch or math.ceil(len(tr_ds) / a.batch)
    total, warm = spe * a.epochs, int(0.05 * spe * a.epochs)
    aug = NewAugment(rotation=0.03, zoom=0.1, brightness=0.05, contrast=(0.9, 1.1), cutout=0, flip=False)
    st = {"epoch": 0, "step_in_epoch": 0, "global_step": 0, "best": -1.0, "history": [], "done": False}
    ck = out / "checkpoint.pt"
    if ck.exists() and not a.no_resume:
        c = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); st = c["state"]
        torch.set_rng_state(c["rng"])
        _log(logp, "resume", **{k: st[k] for k in ("epoch", "step_in_epoch", "global_step")})
    if st["done"]:
        return 0

    def save(reason):
        tmp = out / (".checkpoint.%d.tmp" % os.getpid())
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": st, "rng": torch.get_rng_state(),
                    "reason": reason, "cfg": {**model.cfg, "res": tr_ds.res}}, tmp)
        os.replace(tmp, ck)

    @torch.no_grad()
    def validate():
        model.eval()
        dl = torch.utils.data.DataLoader(va_ds, batch_size=a.batch, shuffle=False, num_workers=a.workers)
        ds = []
        for b in dl:
            ds.append(dice_per_channel(model(b["image"].to(dev)), b["mask"].to(dev)).cpu())
        d = torch.cat(ds).mean(0)
        return {"dice_pulmones": float(d[0]), "dice_corazon": float(d[1])}

    stop = GracefulStop().install()
    guard = ThermalSelfCheck(every=a.thermal_every, log=lambda m: _log(logp, "thermal", msg=m))
    _log(logp, "start", n_train=len(tr_ds), n_val=len(va_ds), steps_per_epoch=spe, device=str(dev), encoder=a.encoder)
    run = 0
    try:
        while st["epoch"] < a.epochs:
            rng = np.random.default_rng([a.seed, st["epoch"]])
            order = rng.permutation(len(tr_ds))
            bl = [order[(k * a.batch) % len(order):][: a.batch].tolist() for k in range(spe)]
            bl = [b if len(b) == a.batch else (b + order[: a.batch - len(b)].tolist()) for b in bl]
            loader = torch.utils.data.DataLoader(tr_ds, batch_sampler=bl[st["step_in_epoch"]:], num_workers=a.workers,
                                                 generator=torch.Generator().manual_seed(a.seed + st["epoch"]))
            losses = st.setdefault("losses", [])
            model.train()
            for b in loader:
                x, m = b["image"].to(dev), b["mask"].to(dev)
                x, _, m = aug(x, torch.zeros(len(x), device=dev), make_generator(a.seed, st["global_step"], dev), mask=m)
                m = (m > 0.5).float()
                lr = warmup_cosine(st["global_step"], a.lr, total, warm, 1e-6)
                for g in opt.param_groups:
                    g["lr"] = lr
                with torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False):
                    loss = seg_loss(model(x), m)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                losses.append(float(loss.detach())); st["step_in_epoch"] += 1; st["global_step"] += 1; run += 1
                guard.step()
                if stop.requested:
                    _log(logp, "stop", reason="SIGTERM"); save("parada"); stop.restore(); return EXIT_THERMAL
                if a.stop_after_steps and run >= a.stop_after_steps:
                    _log(logp, "stop", reason="stop_after_steps"); save("parada"); stop.restore(); return 3
            v = validate()
            rec = {"epoch": st["epoch"], "loss": float(np.mean(losses)) if losses else None, **v}
            if (v["dice_pulmones"] + v["dice_corazon"]) / 2 > st["best"]:
                st["best"] = (v["dice_pulmones"] + v["dice_corazon"]) / 2
                torch.save({"model": model.state_dict(), "cfg": {**model.cfg, "res": tr_ds.res}, "val": v},
                           out / "states" / "best.pt")
                rec["improved"] = True
            st["history"].append(rec); st["losses"] = []
            st["epoch"] += 1; st["step_in_epoch"] = 0
            _log(logp, "epoch", **rec)
            save("epoca")
        torch.save({"model": model.state_dict(), "cfg": {**model.cfg, "res": tr_ds.res}}, out / "states" / "last.pt")
        st["done"] = True
        save("fin")
        (out / "summary.json").write_text(json.dumps({"history": st["history"], "best_mean_dice": st["best"]}, indent=2))
        _log(logp, "end", best=st["best"])
        stop.restore()
        return 0
    except ThermalLimitReached as e:
        _log(logp, "stop", reason="temperatura", detail=str(e)); save("temperatura"); stop.restore()
        return EXIT_THERMAL


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--chexmask-dir", required=True)
    p.add_argument("--out", default="data/masks/teacher_set")
    p.add_argument("--n-nih", type=int, default=12000)
    p.add_argument("--n-chexpert", type=int, default=12000)
    p.add_argument("--min-dice", type=float, default=0.7)
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude-covidx", choices=["todas", "eval", "ninguna"], default="todas",
                   help="pacientes NIH con imágenes en COVIDx que se excluyen del profesor: todas las particiones "
                        "(por defecto), solo val y test, o ninguna")
    p.add_argument("--no-exclude-covidx-eval", action="store_true", help="equivale a --exclude-covidx ninguna")
    t = sub.add_parser("train")
    t.add_argument("--set", required=True)
    t.add_argument("--out", default="results/seg_teacher")
    t.add_argument("--encoder", default="resnet34")
    t.add_argument("--no-pretrained", action="store_true")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--steps-per-epoch", type=int, default=0)
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default=None)
    t.add_argument("--thermal-every", type=int, default=50)
    t.add_argument("--stop-after-steps", type=int, default=0)
    t.add_argument("--no-resume", action="store_true")
    a = ap.parse_args(argv)
    return prepare(a) if a.cmd == "prepare" else train(a)


if __name__ == "__main__":
    sys.exit(main())
