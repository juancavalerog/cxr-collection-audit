"""Auditoría de las pseudomáscaras del profesor (docs/03_arquitectura.md, Sections 2.5 y 7.1, riesgo 4).

    python -m src.seg.audit_masks --masks data/masks/covidx_cxr4 --out results/mask_audit \
        --qu-ex data/raw/covid_qu_ex --visual-per-group 100

Qué mide:
1. Área pulmonar por clase y por fuente (fracción de la zona de imagen), con imágenes de máscara vacía (< 1 %) o
   diminuta (< 5 %), calidad nula (q = 0), componentes, asimetría y fracción en la mitad inferior.
2. Dentro de cada fuente con ambas clases (bimcv, cohen, actmed), razón entre la mediana del área de las
   positivas y la de las negativas, con IC95 % por bootstrap de pacientes. Una consolidación no cambia el tamaño
   real del pulmón: si las positivas salen con menos área en la misma fuente, el profesor infrasegmenta la
   región enferma y la compuerta la dejaría fuera.
3. Comparación con COVID-QU-Ex donde existan las mismas imágenes (las de BIMCV conservan el nombre original;
   en COVIDx llevan además el sufijo "-corrected"): Dice pulmonar contra su máscara manual o semiautomática, por
   clase y fuente, y, en las imágenes con máscara de infección, la cobertura de la infección por nuestra máscara
   pulmonar (fracción de la infección que queda dentro). Es la prueba directa de la infrasegmentación.
4. Muestras visuales por fuente y clase (contornos sobre la imagen) para la revisión del autor.

Banderas (results/mask_audit/flags.md), fijadas de antemano: razón de áreas < 0.90 con IC superior < 1.0;
cobertura mediana de la infección < 0.90; Dice mediano frente a COVID-QU-Ex < 0.85 en positivas o positivas
0.05 por debajo de negativas; más del 2 % de máscaras vacías o diminutas o más del 10 % con q = 0 en un grupo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

THR = {"area_ratio": 0.90, "infection_coverage": 0.90, "dice_pos": 0.85, "dice_gap": 0.05, "empty_tiny": 0.02,
       "q0": 0.10}


def area_table(idx: pd.DataFrame) -> pd.DataFrame:
    g = idx.groupby(["source", "label"])
    t = g.agg(n=("image_id", "size"), area_mediana=("lung_frac", "median"),
              area_p25=("lung_frac", lambda v: np.percentile(v, 25)), area_p75=("lung_frac", lambda v: np.percentile(v, 75)),
              frac_vacia=("lung_frac", lambda v: float((v < 0.01).mean())),
              frac_diminuta=("lung_frac", lambda v: float((v < 0.05).mean())),
              frac_q0=("q", lambda v: float((v == 0).mean())), asim_mediana=("asym", "median"),
              inferior_mediana=("lower_frac", "median"), corazon_mediana=("heart_frac", "median"))
    return t.reset_index()


def area_ratio(idx: pd.DataFrame, source, n_boot: int = 1000, seed: int = 0) -> dict:
    """Mediana del área de positivas entre la de negativas en una fuente, con IC por bootstrap de pacientes."""
    from src.eval.probes import cluster_bootstrap
    d = idx[idx["source"] == source].reset_index(drop=True)
    if d["label"].nunique() < 2:
        return {}
    a, y = d["lung_frac"].to_numpy(), d["label"].to_numpy()

    def st(i):
        p, n = a[i][y[i] == 1], a[i][y[i] == 0]
        if len(p) == 0 or len(n) == 0:
            return float("nan")
        return float(np.median(p) / max(np.median(n), 1e-9))

    lo, hi = cluster_bootstrap(st, d["patient_id"].astype(str).to_numpy(), n_boot, seed)
    return {"fuente": source, "razon": st(np.arange(len(d))), "ci_lo": lo, "ci_hi": hi,
            "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())}


def scan_reference(root) -> pd.DataFrame:
    """Recorre un árbol con carpetas hermanas images/, lung masks/ (o masks/) e infection masks/ (COVID-QU-Ex y
    COVID-19 Radiography). Devuelve name, stem, image, lung_mask, infection_mask, clase_carpeta."""
    rows = {}
    root = Path(root)
    for d in root.rglob("*"):
        if not d.is_dir() or d.name not in ("lung masks", "masks"):
            continue
        base = d.parent
        inf = base / "infection masks"
        for f in d.iterdir():
            if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            rec = rows.setdefault(f.name + "|" + str(base), {"name": f.name, "stem": f.stem, "clase_carpeta": base.name,
                                                             "image": str(base / "images" / f.name)})
            if "lung_mask" not in rec or "Infection" in str(base):
                rec["lung_mask"] = str(f)
            if (inf / f.name).exists():
                rec["infection_mask"] = str(inf / f.name)
    d = pd.DataFrame(list(rows.values()))
    if len(d) and "infection_mask" not in d:
        d["infection_mask"] = None
    # una fila por nombre: preferir la que tiene máscara de infección
    if len(d):
        d["_inf"] = d["infection_mask"].notna()
        d = d.sort_values("_inf", ascending=False).drop_duplicates("stem").drop(columns="_inf")
    return d.reset_index(drop=True)


def match_reference(ref: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Empareja por nombre: el stem de COVIDx, sin el sufijo '-corrected', contra el stem de la referencia."""
    f = frame.copy()
    f["stem"] = f["image_id"].map(lambda s: os.path.splitext(s)[0])
    f["stem_n"] = f["stem"].str.replace("-corrected", "", regex=False)
    a = ref.merge(f[["image_id", "stem_n", "label", "source"]], left_on="stem", right_on="stem_n", how="inner")
    return a.drop(columns=["stem_n"]).drop_duplicates("image_id").reset_index(drop=True)


def _phash_file(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("hash_images", ROOT / "scripts" / "data" / "hash_images.py")
    hi = importlib.util.module_from_spec(spec); spec.loader.exec_module(hi)
    with Image.open(path) as im:
        return int(hi.phash(im.convert("L")), 16)


def _popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def _thumb(path, n=32):
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((n, n), Image.BILINEAR), dtype=np.float64).ravel()
    return (a - a.mean()) / (a.std() + 1e-9)


def match_by_phash(ref: pd.DataFrame, frame: pd.DataFrame, hash_csv, max_dist: int = 4, min_corr: float = 0.95,
                   resolve=None) -> pd.DataFrame:
    """Empareja imágenes de referencia sin nombre común (p. ej. covid_N.png de COVID-QU-Ex) con COVIDx por pHash
    (distancia de Hamming <= max_dist sobre los hashes de logs/tmp_datos/hash_covidx_all.csv, mismo algoritmo
    que scripts/data/hash_images.py) y verificación por correlación de miniaturas 32 x 32 >= min_corr."""
    from src.data.paths import resolve_data
    resolve = resolve or resolve_data
    h = pd.read_csv(hash_csv, dtype=str)
    h = h[h["phash"].notna() & (h["phash"] != "")]
    h["image_id"] = h["image_path"].str.split("/").str[-1]
    h = h.merge(frame[["image_id", "label", "source", "image_path"]].rename(columns={"image_path": "covidx_path"}),
                on="image_id", how="inner")
    hv = np.array([int(x, 16) for x in h["phash"]], dtype=np.uint64)
    rows = []
    for r in ref.itertuples(index=False):
        try:
            q = np.uint64(_phash_file(r.image))
        except Exception:  # noqa: BLE001
            continue
        d = _popcount64(np.bitwise_xor(hv, q))
        cand = np.nonzero(d <= max_dist)[0]
        best = None
        for c in cand[np.argsort(d[cand])][:5]:
            try:
                corr = float(np.dot(_thumb(r.image), _thumb(resolve(h["covidx_path"].iloc[c]))) / (32 * 32))
            except Exception:  # noqa: BLE001
                continue
            if corr >= min_corr and (best is None or corr > best[1]):
                best = (c, corr)
        if best is not None:
            c = best[0]
            rows.append({**r._asdict(), "image_id": h["image_id"].iloc[c], "label": int(h["label"].iloc[c]),
                         "source": h["source"].iloc[c], "phash_dist": int(d[c]), "thumb_corr": best[1]})
    return pd.DataFrame(rows)


def _read_bin(p, size=None):
    with Image.open(p) as im:
        im = im.convert("L")
        if size is not None and im.size != size:
            im = im.resize(size, Image.NEAREST)
        return np.asarray(im) > 127


def compare_reference(store, matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in matches.itertuples(index=False):
        if r.image_id not in store:
            continue
        ours = store.original(r.image_id)[0]
        ref = _read_bin(r.lung_mask)
        o = np.asarray(Image.fromarray(ours.astype(np.uint8) * 255).resize(ref.shape[::-1], Image.NEAREST)) > 127
        dice = float(2 * (o & ref).sum() / max(o.sum() + ref.sum(), 1))
        cov = np.nan
        if isinstance(r.infection_mask, str) and os.path.exists(r.infection_mask):
            inf = _read_bin(r.infection_mask, ref.shape[::-1])
            if inf.sum() > 0:
                cov = float((inf & o).sum() / inf.sum())
        rows.append({"image_id": r.image_id, "source": r.source, "label": int(r.label), "dice": dice,
                     "cobertura_infeccion": cov, "area_ref": float(ref.mean()), "area_nuestra": float(o.mean())})
    return pd.DataFrame(rows)


def overlay(store, iid, image_path, out_png):
    from src.data.paths import resolve_data
    from src.data.preprocess import load_luminance
    from src.seg.teacher import to_teacher_frame
    lum = load_luminance(resolve_data(image_path))
    m = store.get(iid)
    img, _, _ = to_teacher_frame(lum, int(m.shape[-1]))
    rgb = np.repeat((img * 255).astype(np.uint8)[..., None], 3, -1)
    for c, col in ((0, (255, 0, 0)), (1, (0, 255, 0))):
        mm = m[c].astype(bool)
        edge = mm & ~(np.roll(mm, 1, 0) & np.roll(mm, -1, 0) & np.roll(mm, 1, 1) & np.roll(mm, -1, 1))
        rgb[edge] = col
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_png)


def audit(store, frame: pd.DataFrame, out, qu_ex=None, mixed=("bimcv", "cohen", "actmed"), n_boot=1000,
          visual_per_group=8, seed=0, hash_csv=None, phash_only_infection=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    idx = store.index.reset_index().merge(frame[["image_id", "label", "source", "patient_id", "image_path"]],
                                          on="image_id", how="inner")
    at = area_table(idx)
    at.to_csv(out / "area_por_fuente_clase.csv", index=False)
    flags = []
    for r in at.itertuples(index=False):
        if r.frac_vacia + r.frac_diminuta > THR["empty_tiny"] * 2 or r.frac_diminuta > THR["empty_tiny"]:
            flags.append("máscaras vacías o diminutas en %s/%s: %.1f %%" % (r.source, r.label, 100 * r.frac_diminuta))
        if r.frac_q0 > THR["q0"]:
            flags.append("q = 0 en %s/%s: %.1f %%" % (r.source, r.label, 100 * r.frac_q0))
    ratios = [x for x in (area_ratio(idx, s, n_boot, seed) for s in mixed) if x]
    for x in ratios:
        if x["razon"] < THR["area_ratio"] and x["ci_hi"] < 1.0:
            flags.append("posible infrasegmentación de positivas en %s: razón de áreas %.3f (IC %.3f a %.3f)"
                         % (x["fuente"], x["razon"], x["ci_lo"], x["ci_hi"]))
    ref_summary = {}
    if qu_ex and Path(qu_ex).exists():
        ref = scan_reference(qu_ex)
        mt = match_reference(ref, frame)
        if hash_csv and Path(hash_csv).exists():
            rest = ref[~ref["stem"].isin(mt["stem"])]
            if phash_only_infection:
                rest = rest[rest["infection_mask"].notna()]
            mh = match_by_phash(rest, frame, hash_csv)
            if len(mh):
                mt = pd.concat([mt, mh], ignore_index=True).drop_duplicates("image_id")
        cmp_ = compare_reference(store, mt)
        cmp_.to_csv(out / "comparacion_covid_qu_ex.csv", index=False)
        ref_summary = {"n_referencia": int(len(ref)), "n_emparejadas": int(len(mt)), "n_comparadas": int(len(cmp_))}
        if len(cmp_):
            g = cmp_.groupby(["source", "label"]).agg(n=("dice", "size"), dice_mediana=("dice", "median"),
                                                      cobertura_mediana=("cobertura_infeccion", "median"),
                                                      n_infeccion=("cobertura_infeccion", "count")).reset_index()
            g.to_csv(out / "comparacion_covid_qu_ex_resumen.csv", index=False)
            ref_summary["por_grupo"] = g.to_dict("records")
            pos = cmp_[cmp_["label"] == 1]
            neg = cmp_[cmp_["label"] == 0]
            if len(pos) and pos["dice"].median() < THR["dice_pos"]:
                flags.append("Dice mediano frente a COVID-QU-Ex en positivas %.3f < %.2f" % (pos["dice"].median(), THR["dice_pos"]))
            if len(pos) and len(neg) and pos["dice"].median() < neg["dice"].median() - THR["dice_gap"]:
                flags.append("Dice de positivas %.3f por debajo del de negativas %.3f" % (pos["dice"].median(), neg["dice"].median()))
            cv = cmp_["cobertura_infeccion"].dropna()
            if len(cv) and cv.median() < THR["infection_coverage"]:
                flags.append("cobertura mediana de la infección por la máscara pulmonar %.3f < %.2f: la compuerta "
                             "dejaría fuera parte de la región enferma" % (cv.median(), THR["infection_coverage"]))
    if visual_per_group:
        rng = np.random.default_rng(seed)
        for (s, y), g in idx.groupby(["source", "label"]):
            pick = g.iloc[rng.permutation(len(g))[:visual_per_group]]
            for r in pick.itertuples(index=False):
                try:
                    overlay(store, r.image_id, r.image_path, out / "visual" / ("%s_%d" % (s, y)) / (os.path.splitext(r.image_id)[0] + ".png"))
                except Exception as e:  # noqa: BLE001
                    flags.append("no se pudo dibujar %s: %s" % (r.image_id, str(e)[:80]))
    summary = {"n_imagenes": int(len(idx)), "umbrales": THR, "razon_areas": ratios, "referencia": ref_summary,
               "banderas": flags}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    md = ["# Auditoría de pseudomáscaras", "", "Imágenes: %d" % len(idx), "", "## Banderas", ""]
    md += ["- " + f for f in flags] or ["- ninguna"]
    md += ["", "## Razón de áreas positivas / negativas en fuentes mixtas", ""]
    md += ["- %s: %.3f (IC95 %.3f a %.3f; %d positivas, %d negativas)" % (x["fuente"], x["razon"], x["ci_lo"], x["ci_hi"],
                                                                         x["n_pos"], x["n_neg"]) for x in ratios]
    (out / "flags.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    from src.seg.predict_masks import PseudoMaskStore as MaskStore
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", default="data/masks/covidx_cxr4")
    ap.add_argument("--out", default="results/mask_audit")
    ap.add_argument("--qu-ex", default="data/raw/covid_qu_ex")
    ap.add_argument("--mode", default="limpia")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--visual-per-group", type=int, default=8)
    ap.add_argument("--hash-csv", default="logs/tmp_datos/hash_covidx_all.csv",
                    help="pHash de COVIDx para emparejar las imágenes de referencia sin nombre común")
    ap.add_argument("--phash-all", action="store_true", help="pHash también para las que no tienen máscara de infección")
    a = ap.parse_args(argv)
    from src.data.manifest import load_manifest
    man = load_manifest()
    col = "split_clean" if a.mode == "limpia" else "split"
    man = man[man[col] != "excluded"]
    s = audit(MaskStore(ROOT / a.masks if not os.path.isabs(a.masks) else a.masks), man,
              ROOT / a.out if not os.path.isabs(a.out) else a.out, a.qu_ex if os.path.isabs(a.qu_ex) else ROOT / a.qu_ex,
              n_boot=a.n_boot, visual_per_group=a.visual_per_group,
              hash_csv=ROOT / a.hash_csv if not os.path.isabs(a.hash_csv) else a.hash_csv,
              phash_only_infection=not a.phash_all)
    print(json.dumps({"banderas": s["banderas"], "n": s["n_imagenes"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
