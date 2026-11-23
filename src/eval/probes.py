"""Verificaciones de identificabilidad V0 a V7 de CSD-Net (docs/03_arquitectura.md, Section 3 y Tabla 7).

Cada verificación es una función sobre arrays de numpy (embeddings, etiquetas, fuentes y pacientes) que
devuelve un diccionario con sus métricas, el umbral de la Tabla 7 y el veredicto. Los veredictos posibles son
"aprueba", "falla", "no_concluyente" (V1 sin potencia), "informa" (verificaciones sin umbral: V1b, V6, V7; llevan
además un indicador orientativo) y "no_aplicable" (por ejemplo, V2 a V5 en la ablación canónica, sin vistas
cruzadas). Todas las particiones de entrenamiento y evaluación de los probes son disjuntas por paciente
(validación cruzada agrupada o partición 70/30 agrupada y estratificada) y los IC95 % se calculan por bootstrap
de pacientes (remuestreo de conglomerados) sobre las predicciones fuera de partición.

Además:
- `collect_probe_data`: ejecuta un CSD-Net entrenado sobre un conjunto de validación (con la máscara del
  profesor) y reúne las representaciones reales, dentro de la fuente y re-firmadas hacia cada fuente destino.
- `run_probes`: aplica V1 a V7 a esos datos con la configuración de estratos (nombres de fuente de COVIDx por
  defecto) y `write_report` escribe el informe en JSON y Markdown.

Nunca se pasa el test por aquí durante el desarrollo: la lista de conjuntos la decide el bucle (src/train/
csdnet_loop.py), que bloquea el test en el piloto.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

# ------------------------------------------------------------------------------------- umbrales (Tabla 7)
THRESHOLDS = {
    "V0": {"max_abs_err": 1e-4},
    "V1": {"auc_max": 0.60, "ci_upper_max": 0.65, "power_min": 0.70},
    "V1b": {"r2_orientativo_max": 0.10, "min_n": 120},
    "V2": {"td_min": 0.70, "td_ratio_min": 0.80, "permanence_max": 0.20, "min_n": 120, "cap": 600},
    "V2b": {"ba_max": 0.60, "ci_upper_max": 0.65, "cap": 600},
    "V3": {"dauc_max": 0.02, "mean_abs_dp_max": 0.05, "dpm_abs_max": 0.02, "flip_rate_cordura": 0.05},
    "V4": {"alpha": 0.05, "min_n": 120, "cap": 600},
    "V5": {"auc_mean_max": 0.60, "auc_stratum_max": 0.65, "ci_upper_max": 0.65, "cap": 600},
    "V6": {},
    "V7": {},
}

# Estratos de V2b, V3 y V5 con los nombres de COVIDx (Section 3); el bucle puede sustituirlos (datos sintéticos).
DEFAULT_PROBE_CFG = {
    "robust_source": "bimcv",
    "v1_extra_sources": ["cohen", "actmed"],
    "v2_sources": ["bimcv", "rsna", "stonybrook"],
    # (clase, destino, origen_a, origen_b)
    "v2b_strata": [[1, "rsna", "bimcv", "stonybrook"], [0, "stonybrook", "bimcv", "rsna"]],
    # (clase, fuente efectiva, origen re-firmado)
    "v5_strata": [[0, "bimcv", "rsna"], [0, "rsna", "bimcv"], [1, "stonybrook", "bimcv"], [1, "bimcv", "stonybrook"]],
    "v3_source": "bimcv", "v3_neg_dest": "rsna", "v3_pos_dest": "stonybrook",
    "v6_sources": ["rsna", "bimcv"],
    "cap": 600, "seed": 42, "n_boot": 1000, "folds": 5,
}


def _result(vid, supuesto, metrics, thresholds, verdict, **detail):
    return {"id": vid, "supuesto": supuesto, "metricas": _clean(metrics), "umbrales": thresholds,
            "veredicto": verdict, "detalle": _clean(detail)}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if math.isnan(v) else v
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


# ------------------------------------------------------------------------------------- utilidades
def group_folds(strat: np.ndarray, groups: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Asigna cada muestra a una de k particiones, con todos los registros de un paciente en la misma
    partición y estratificación por la etiqueta mayoritaria del paciente. Devuelve el índice de partición."""
    strat = np.asarray(strat)
    groups = np.asarray(groups).astype(str)
    ug, inv = np.unique(groups, return_inverse=True)
    lab = np.zeros(len(ug), dtype=object)
    for gi in range(len(ug)):
        v, c = np.unique(strat[inv == gi], return_counts=True)
        lab[gi] = v[np.argmax(c)]
    rng = np.random.default_rng(seed)
    gf = np.zeros(len(ug), dtype=int)
    for val in np.unique(lab.astype(str)):
        idx = np.nonzero(lab.astype(str) == val)[0]
        idx = idx[rng.permutation(len(idx))]
        off = int(rng.integers(k))
        gf[idx] = (np.arange(len(idx)) + off) % k
    return gf[inv]


def group_holdout(strat, groups, test_frac: float = 0.3, seed: int = 42) -> np.ndarray:
    """Máscara booleana del conjunto retenido (aprox. test_frac), disjunta por paciente y estratificada."""
    k = max(2, int(round(1.0 / test_frac)))
    f = group_folds(strat, groups, k=k, seed=seed)
    return f == 0


def cap_per_class(labels, cap: int, seed: int = 42) -> np.ndarray:
    """Índices con como mucho `cap` muestras de cada valor de `labels` (semilla fija)."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    out = []
    for v in np.unique(labels):
        idx = np.nonzero(labels == v)[0]
        if len(idx) > cap:
            idx = np.sort(rng.choice(idx, cap, replace=False))
        out.append(idx)
    return np.sort(np.concatenate(out)) if out else np.zeros(0, int)


def cluster_bootstrap(stat: Callable[[np.ndarray], float], groups, n_boot: int = 1000, seed: int = 0,
                      alpha: float = 0.05):
    """IC por bootstrap de pacientes: se remuestrean pacientes con reemplazo y se recalcula `stat(indices)`."""
    groups = np.asarray(groups).astype(str)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.nonzero(inv == g)[0] for g in range(len(ug))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, len(ug), len(ug))
        idx = np.concatenate([members[g] for g in pick])
        try:
            v = stat(idx)
        except ValueError:
            continue
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _ba(y, yhat):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, yhat))


def make_probe(kind: str = "logreg", seed: int = 0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    if kind == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, alpha=1e-3,
                                                             random_state=seed))
    raise ValueError(kind)


def oof_proba(X, y, groups, kind="logreg", k=5, seed=0) -> np.ndarray:
    """Probabilidad de la clase 1 fuera de partición, con validación cruzada agrupada por paciente."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    folds = group_folds(y, groups, k=k, seed=seed)
    p = np.full(len(y), np.nan)
    for f in range(k):
        te = folds == f
        tr = ~te
        if te.sum() == 0:
            continue
        if len(np.unique(y[tr])) < 2:
            p[te] = y[tr].mean() if tr.any() else 0.5
            continue
        m = make_probe(kind, seed).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def _auc_with_ci(y, p, groups, n_boot, seed):
    auc = _auc(y, p)
    lo, hi = cluster_bootstrap(lambda i: _auc(y[i], p[i]), groups, n_boot, seed)
    return auc, lo, hi


# ------------------------------------------------------------------------------------- V0
def v0_reconstruction(real: np.ndarray, recon: np.ndarray, tol: float = THRESHOLDS["V0"]["max_abs_err"]):
    """V0: con donante igual a la propia imagen, la vista re-firmada coincide con la real (|error| <= 1e-4)."""
    real, recon = np.asarray(real, dtype=np.float64), np.asarray(recon, dtype=np.float64)
    err = float(np.max(np.abs(real - recon))) if real.size else 0.0
    return _result("V0", "reconstrucción", {"max_abs_err": err}, {"max_abs_err": tol},
                   "aprueba" if err <= tol else "falla")


def v0_from_model(model, x, lung_mask, tol: float = 1e-4):
    """V0 sobre un modelo: swap_index = within_index = identidad; compara logit y emb_gated."""
    import torch
    model.eval()
    idx = torch.arange(x.shape[0], device=x.device)
    with torch.no_grad():
        out = model(x, swap_index=idx, within_index=idx, lung_mask=lung_mask, gate_mask=lung_mask)
    if "logit_swap" not in out:
        return _result("V0", "reconstrucción", {}, {"max_abs_err": tol}, "no_aplicable",
                       motivo="el modelo no produce vistas re-firmadas (re-firma canónica)")
    real = np.concatenate([out["logit"].float().cpu().numpy().ravel(), out["emb_gated"].float().cpu().numpy().ravel()])
    rec = np.concatenate([out["logit_swap"].float().cpu().numpy().ravel(),
                          out["emb_swap_gated"].float().cpu().numpy().ravel()])
    return v0_reconstruction(real, rec, tol)


# ------------------------------------------------------------------------------------- V1
def v1_signature_class(sig, y, groups, power_emb=None, n_boot: int = 1000, seed: int = 42, folds: int = 5,
                       extra: Optional[Dict[str, tuple]] = None, thr: Optional[dict] = None):
    """V1 (A1): probe de clase sobre la firma dentro de una fuente con ambas clases (bimcv de val), ecuación (35).

    Dos probes (regresión logística y MLP de 64) con validación cruzada de 5 particiones agrupadas por paciente;
    aprueba si AUC <= 0.60 y el extremo superior del IC95 % <= 0.65 en los dos. Control de potencia: el mismo
    probe logístico sobre `power_emb` (emb_content) debe dar AUC >= 0.70; si no, el veredicto es
    "no_concluyente". `extra`: {nombre: (sig, y, groups)} se informa sin umbral (cohen, actmed)."""
    t = dict(THRESHOLDS["V1"], **(thr or {}))
    sig, y, groups = np.asarray(sig), np.asarray(y).astype(int), np.asarray(groups).astype(str)
    if len(np.unique(y)) < 2 or len(y) < 2 * folds:
        return _result("V1", "A1", {"n": len(y)}, t, "no_aplicable", motivo="faltan clases o muestras")
    m: Dict[str, float] = {"n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
                           "n_pacientes": int(len(np.unique(groups)))}
    ok = True
    for kind in ("logreg", "mlp"):
        p = oof_proba(sig, y, groups, kind, folds, seed)
        auc, lo, hi = _auc_with_ci(y, p, groups, n_boot, seed)
        m["auc_" + kind], m["ci_lo_" + kind], m["ci_hi_" + kind] = auc, lo, hi
        ok &= (auc <= t["auc_max"]) and (hi <= t["ci_upper_max"])
    power = None
    if power_emb is not None:
        pp = oof_proba(power_emb, y, groups, "logreg", folds, seed)
        power = _auc(y, pp)
        m["auc_control_potencia"] = power
    detail = {}
    for name, (s2, y2, g2) in (extra or {}).items():
        y2 = np.asarray(y2).astype(int)
        if len(np.unique(y2)) == 2 and len(y2) >= 2 * folds:
            p2 = oof_proba(s2, y2, g2, "logreg", min(folds, max(2, len(np.unique(g2)) // 2)), seed)
            detail["auc_logreg_" + name] = _auc(y2, p2)
            detail["n_" + name] = int(len(y2))
    if power is not None and power < t["power_min"]:
        verdict = "no_concluyente"
    else:
        verdict = "aprueba" if ok else "falla"
    return _result("V1", "A1", m, t, verdict, **detail)


# ------------------------------------------------------------------------------------- V1b
def v1b_signature_content(sig, emb_gated, s, groups, min_n: int = 120, n_pc: int = 5, seed: int = 42,
                          folds: int = 5, source_names: Optional[Dict[int, str]] = None):
    """V1b: R2 de validación cruzada (agrupada por paciente) de las 5 primeras componentes principales de
    emb_gated a partir de la firma, por fuente con >= min_n imágenes. Se informa; umbral orientativo 0.10."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    t = dict(THRESHOLDS["V1b"], min_n=min_n)
    sig, emb, s, groups = np.asarray(sig, float), np.asarray(emb_gated, float), np.asarray(s), np.asarray(groups).astype(str)
    per = {}
    worst = -np.inf
    for src in np.unique(s):
        idx = np.nonzero(s == src)[0]
        if len(idx) < min_n:
            continue
        E = emb[idx]
        k = min(n_pc, E.shape[1], len(idx) - 1)
        Z = PCA(n_components=k, random_state=seed).fit_transform(StandardScaler().fit_transform(E))
        f = group_folds(np.zeros(len(idx)), groups[idx], k=folds, seed=seed)
        pred = np.zeros_like(Z)
        for fi in range(folds):
            te = f == fi
            if te.sum() == 0 or (~te).sum() < 3:
                continue
            reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(sig[idx][~te], Z[~te])
            pred[te] = reg.predict(sig[idx][te])
        r2 = [float(1 - ((Z[:, c] - pred[:, c]) ** 2).sum() / max(((Z[:, c] - Z[:, c].mean()) ** 2).sum(), 1e-12))
              for c in range(k)]
        name = (source_names or {}).get(int(src), str(src))
        per[name] = r2
        worst = max(worst, max(r2))
    if not per:
        return _result("V1b", "A1 (contenido)", {}, t, "no_aplicable", motivo="ninguna fuente con min_n imágenes")
    flag = "por_debajo_del_orientativo" if worst <= t["r2_orientativo_max"] else "supera_el_orientativo"
    return _result("V1b", "A1 (contenido)", {"r2_max": worst, "r2_por_fuente": per}, t, "informa", indicador=flag)


# ------------------------------------------------------------------------------------- V2
def v2_target_source(stats_real, s, groups, stats_swap: Dict[int, np.ndarray], sources: Sequence[int],
                     cap: int = 600, seed: int = 42, test_frac: float = 0.3, n_boot: int = 1000,
                     thr: Optional[dict] = None):
    """V2 (A3): probe multinomial de fuente sobre los estadísticos de la etapa de inserción, entrenado con
    imágenes reales (70 %, por paciente) y aplicado a las del 30 % retenido re-firmadas hacia cada fuente destino.
    `stats_swap[t]` tiene una fila por imagen (NaN si la imagen es de la fuente t). Ecuación (36)."""
    t = dict(THRESHOLDS["V2"], **(thr or {}))
    X, s, groups = np.asarray(stats_real, float), np.asarray(s), np.asarray(groups).astype(str)
    keep = np.isin(s, list(sources))
    base = np.nonzero(keep)[0]
    sel = base[cap_per_class(s[base], cap, seed)]
    hold = group_holdout(s[sel], groups[sel], test_frac, seed)
    tr, te = sel[~hold], sel[hold]
    if len(np.unique(s[tr])) < 2 or len(te) == 0:
        return _result("V2", "A3", {}, t, "no_aplicable", motivo="fuentes insuficientes")
    probe = make_probe("logreg").fit(X[tr], s[tr])
    ba_real = _ba(s[te], probe.predict(X[te]))
    hits, own, tot = {}, 0, 0
    pred_by_t, idx_by_t = {}, {}
    for tgt in sources:
        if tgt not in stats_swap:
            continue
        rows = te[s[te] != tgt]
        if len(rows) == 0:
            continue
        pr = probe.predict(np.asarray(stats_swap[tgt], float)[rows])
        pred_by_t[tgt], idx_by_t[tgt] = pr, rows
        hits[tgt] = float((pr == tgt).mean())
        own += int((pr == s[rows]).sum()); tot += len(rows)
    td = float(np.mean(list(hits.values()))) if hits else float("nan")
    perm = own / max(tot, 1)

    # IC de TD por bootstrap de pacientes del conjunto retenido
    gte = groups[te]
    pos = {r: i for i, r in enumerate(te)}

    def td_boot(ii):
        chosen = set(te[ii].tolist())
        vals = []
        for tgt, rows in idx_by_t.items():
            msk = np.array([r in chosen for r in rows])
            if msk.sum() == 0:
                continue
            # multiplicidad del remuestreo
            cnt = np.bincount(ii, minlength=len(te))
            w = np.array([cnt[pos[r]] for r in rows], float)
            vals.append(float((w * (pred_by_t[tgt] == tgt)).sum() / max(w.sum(), 1e-12)))
        return float(np.mean(vals)) if vals else float("nan")

    lo, hi = cluster_bootstrap(td_boot, gte, n_boot, seed)
    ok = (td >= t["td_min"]) and (td >= t["td_ratio_min"] * ba_real) and (perm <= t["permanence_max"])
    return _result("V2", "A3", {"td": td, "td_ci_lo": lo, "td_ci_hi": hi, "ba_real": ba_real,
                                "permanencia": perm, "acierto_por_destino": {str(k): v for k, v in hits.items()},
                                "n_entrenamiento": int(len(tr)), "n_retenido": int(len(te))}, t,
                   "aprueba" if ok else "falla")


# ------------------------------------------------------------------------------------- V2b
def _binary_strata_metric(strata, metric: str, folds: int, seed: int):
    """Para cada estrato {z, label, groups}: probe logístico con CV agrupada; devuelve (valor por estrato, oof)."""
    vals, oofs = {}, {}
    for st in strata:
        z, lab, g = np.asarray(st["z"], float), np.asarray(st["label"]).astype(int), np.asarray(st["groups"]).astype(str)
        if len(np.unique(lab)) < 2 or len(lab) < 2 * folds:
            continue
        p = oof_proba(z, lab, g, "logreg", folds, seed)
        oofs[st["name"]] = (lab, p, g)
        if metric == "ba":
            vals[st["name"]] = _ba(lab, (p >= 0.5).astype(int))
        else:
            a = _auc(lab, p)
            vals[st["name"]] = max(a, 1 - a)
    return vals, oofs


def _strata_boot(oofs, metric, n_boot, seed):
    """IC del promedio sobre estratos, remuestreando pacientes dentro de cada estrato."""
    rng = np.random.default_rng(seed)
    prep = []
    for name, (lab, p, g) in oofs.items():
        ug, inv = np.unique(g, return_inverse=True)
        prep.append((lab, p, [np.nonzero(inv == k)[0] for k in range(len(ug))]))
    vals = []
    for _ in range(int(n_boot)):
        per = []
        for lab, p, members in prep:
            pick = rng.integers(0, len(members), len(members))
            ii = np.concatenate([members[k] for k in pick])
            if len(np.unique(lab[ii])) < 2:
                continue
            if metric == "ba":
                per.append(_ba(lab[ii], (p[ii] >= 0.5).astype(int)))
            else:
                a = _auc(lab[ii], p[ii]); per.append(max(a, 1 - a))
        if per:
            vals.append(np.mean(per))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def v2b_origin(strata: List[dict], strata_real: Optional[List[dict]] = None, secondary: Optional[List[dict]] = None,
               folds: int = 5, seed: int = 42, n_boot: int = 1000, thr: Optional[dict] = None):
    """V2b: fuente de origen en lo que llega a la decisión (emb_swap_gated), estratificada por clase y fuente
    destino, ecuación (37). Cada estrato es {name, z, label (origen 0/1), groups}. Aprueba si BA_orig <= 0.60 y
    el extremo superior del IC95 % <= 0.65. `strata_real`: mismo probe sobre vistas reales (se informa la razón);
    `secondary`: segunda lectura (emb_swap_decision), se informa."""
    t = dict(THRESHOLDS["V2b"], **(thr or {}))
    vals, oofs = _binary_strata_metric(strata, "ba", folds, seed)
    if not vals:
        return _result("V2b", "atajo de la vista re-firmada", {}, t, "no_aplicable", motivo="sin estratos válidos")
    ba = float(np.mean(list(vals.values())))
    lo, hi = _strata_boot(oofs, "ba", n_boot, seed)
    m = {"ba_orig": ba, "ci_lo": lo, "ci_hi": hi, "por_estrato": vals}
    if strata_real:
        vr, _ = _binary_strata_metric(strata_real, "ba", folds, seed)
        if vr:
            m["ba_orig_vistas_reales"] = float(np.mean(list(vr.values())))
            m["razon_refirmada_real"] = ba / max(m["ba_orig_vistas_reales"], 1e-12)
    if secondary:
        vs, _ = _binary_strata_metric(secondary, "ba", folds, seed)
        if vs:
            m["ba_orig_decision"] = float(np.mean(list(vs.values())))
    ok = ba <= t["ba_max"] and hi <= t["ci_upper_max"]
    return _result("V2b", "atajo de la vista re-firmada", m, t, "aprueba" if ok else "falla")


# ------------------------------------------------------------------------------------- V3
def v3_disease(emb_real, y, groups, emb_swap: Dict[str, np.ndarray], neg_dest: str, pos_dest: str,
               logit_real=None, logit_swap: Optional[Dict[str, np.ndarray]] = None, folds: int = 5,
               seed: int = 42, n_boot: int = 1000, thr: Optional[dict] = None):
    """V3 (A2 y A1): probe logístico de enfermedad sobre emb_content real (bimcv de val, CV agrupada por
    paciente) evaluado en cada partición retenida sobre las mismas imágenes re-firmadas hacia `neg_dest` (rsna)
    y `pos_dest` (stonybrook). Ecuación (38): |ΔAUC| <= 0.02 por destino, media de |Δp| <= 0.05 e IC95 % de
    Δ± = p(pos_dest) - p(neg_dest) con el cero y |Δ±| <= 0.02. Cordura: tasa de cambio de decisión de la cabeza
    al umbral de Youden (se informa)."""
    t = dict(THRESHOLDS["V3"], **(thr or {}))
    X, y, groups = np.asarray(emb_real, float), np.asarray(y).astype(int), np.asarray(groups).astype(str)
    if len(np.unique(y)) < 2:
        return _result("V3", "A2, A1", {}, t, "no_aplicable", motivo="una sola clase")
    f = group_folds(y, groups, k=folds, seed=seed)
    p_real = np.full(len(y), np.nan)
    p_sw = {d: np.full(len(y), np.nan) for d in (neg_dest, pos_dest)}
    for fi in range(folds):
        te = f == fi
        tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        m = make_probe("logreg").fit(X[tr], y[tr])
        p_real[te] = m.predict_proba(X[te])[:, 1]
        for d in p_sw:
            p_sw[d][te] = m.predict_proba(np.asarray(emb_swap[d], float)[te])[:, 1]
    auc0 = _auc(y, p_real)
    dauc = {d: _auc(y, p_sw[d]) - auc0 for d in p_sw}
    mad = float(np.mean([np.mean(np.abs(p_sw[d] - p_real)) for d in p_sw]))
    dpm = float(np.mean(p_sw[pos_dest]) - np.mean(p_sw[neg_dest]))
    lo, hi = cluster_bootstrap(lambda i: float(np.mean(p_sw[pos_dest][i]) - np.mean(p_sw[neg_dest][i])), groups,
                               n_boot, seed)
    m_ = {"auc_real": auc0, "delta_auc": dauc, "media_abs_dp": mad, "delta_pm": dpm, "delta_pm_ci_lo": lo,
          "delta_pm_ci_hi": hi}
    if logit_real is not None and logit_swap:
        lr = np.asarray(logit_real, float)
        pr = 1 / (1 + np.exp(-lr))
        thr_y = youden_threshold(y, pr)
        flips = [float(np.mean((pr >= thr_y) != (1 / (1 + np.exp(-np.asarray(v, float))) >= thr_y)))
                 for v in logit_swap.values()]
        m_["tasa_cambio_decision_cabeza"] = float(np.mean(flips))
    ok = all(abs(v) <= t["dauc_max"] for v in dauc.values()) and mad <= t["mean_abs_dp_max"] \
        and (lo <= 0 <= hi) and abs(dpm) <= t["dpm_abs_max"]
    return _result("V3", "A2, A1", m_, t, "aprueba" if ok else "falla")


# ------------------------------------------------------------------------------------- V4
def v4_levels(reps: Dict[str, np.ndarray], s, groups, sources: Optional[Sequence[int]] = None, cap: int = 600,
              seed: int = 42, test_frac: float = 0.3):
    """V4 por corrida: exactitud del probe multinomial de fuente (protocolo del artículo 1, con partición
    70/30 agrupada por paciente) sobre cada representación, p. ej. {'nivel1_content', 'nivel1_tokens',
    'nivel2_decision'}. Devuelve {nombre: exactitud}. La comparación entre corridas la hace v4_compare."""
    s, groups = np.asarray(s), np.asarray(groups).astype(str)
    if sources is None:
        u, c = np.unique(s, return_counts=True)
        sources = u[c >= THRESHOLDS["V4"]["min_n"]]
    base = np.nonzero(np.isin(s, list(sources)))[0]
    sel = base[cap_per_class(s[base], cap, seed)]
    hold = group_holdout(s[sel], groups[sel], test_frac, seed)
    tr, te = sel[~hold], sel[hold]
    out = {}
    for name, X in reps.items():
        X = np.asarray(X, float)
        if len(np.unique(s[tr])) < 2:
            out[name] = float("nan"); continue
        m = make_probe("logreg").fit(X[tr], s[tr])
        out[name] = float((m.predict(X[te]) == s[te]).mean())
    return out


def v4_compare(csd_level2: Sequence[float], control_level2: Sequence[float], csd_gap: Sequence[float],
               erm_gap: Sequence[float], alpha: float = THRESHOLDS["V4"]["alpha"]):
    """V4 entre corridas: aprueba si el nivel 2 de CSD-Net es menor que el del control de fuente (Welch
    unilateral, p < 0.05, una exactitud por semilla) y si la diferencia nivel 1 menos nivel 2 media de CSD-Net
    es mayor que la de ERM."""
    from scipy import stats
    a, b = np.asarray(csd_level2, float), np.asarray(control_level2, float)
    if len(a) < 2 or len(b) < 2:
        return _result("V4", "dependencia residual", {"n_csd": len(a), "n_control": len(b)}, THRESHOLDS["V4"],
                       "no_aplicable", motivo="hacen falta al menos dos semillas por modelo")
    tt = stats.ttest_ind(a, b, equal_var=False, alternative="less")
    gap_ok = float(np.mean(csd_gap)) > float(np.mean(erm_gap))
    ok = (tt.pvalue < alpha) and gap_ok
    return _result("V4", "dependencia residual",
                   {"nivel2_csd_media": float(a.mean()), "nivel2_control_media": float(b.mean()),
                    "p_welch_unilateral": float(tt.pvalue), "brecha_niveles_csd": float(np.mean(csd_gap)),
                    "brecha_niveles_erm": float(np.mean(erm_gap))}, THRESHOLDS["V4"], "aprueba" if ok else "falla")


# ------------------------------------------------------------------------------------- V5
def v5_view(strata: List[dict], strata_within: Optional[List[dict]] = None, folds: int = 5, seed: int = 42,
            n_boot: int = 1000, thr: Optional[dict] = None):
    """V5: vista propia frente a vista cruzada con la misma clase y la misma fuente efectiva, ecuación (39).
    Cada estrato es {name, z (reales y cruzadas apiladas), label (1 = cruzada), groups}. AUC simetrizada;
    aprueba si la media <= 0.60, cada estrato <= 0.65 y el extremo superior del IC95 % de la media <= 0.65.
    `strata_within`: reales frente a dentro de la fuente (se informa)."""
    t = dict(THRESHOLDS["V5"], **(thr or {}))
    vals, oofs = _binary_strata_metric(strata, "auc", folds, seed)
    if not vals:
        return _result("V5", "atajo de la vista re-firmada", {}, t, "no_aplicable", motivo="sin estratos válidos")
    mean = float(np.mean(list(vals.values())))
    lo, hi = _strata_boot(oofs, "auc", n_boot, seed)
    m = {"auc_vista": mean, "ci_lo": lo, "ci_hi": hi, "por_estrato": vals}
    if strata_within:
        vw, _ = _binary_strata_metric(strata_within, "auc", folds, seed)
        if vw:
            m["auc_real_frente_a_dentro"] = float(np.mean(list(vw.values())))
    ok = mean <= t["auc_mean_max"] and max(vals.values()) <= t["auc_stratum_max"] and hi <= t["ci_upper_max"]
    return _result("V5", "atajo de la vista re-firmada", m, t, "aprueba" if ok else "falla")


# ------------------------------------------------------------------------------------- V6
def youden_threshold(y, p) -> float:
    from sklearn.metrics import roc_curve
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    j = np.argmax(tpr - fpr)
    return float(min(thr[j], 1.0))


def v6_specificity_gap(prob, y, s, groups, threshold: float, src_a: int, src_b: int, n_boot: int = 1000,
                       seed: int = 42):
    """V6: especificidad en src_a (rsna) menos especificidad en src_b (bimcv) al umbral dado (Youden de la
    validación robusta), con IC95 % por bootstrap de pacientes. Sin umbral propio: se informa con el indicador
    'brecha_distinta_de_cero' si el IC excluye el cero."""
    p, y, s, g = np.asarray(prob, float), np.asarray(y).astype(int), np.asarray(s), np.asarray(groups).astype(str)
    neg = y == 0
    ia, ib = np.nonzero(neg & (s == src_a))[0], np.nonzero(neg & (s == src_b))[0]
    if len(ia) == 0 or len(ib) == 0:
        return _result("V6", "dependencia residual", {}, {}, "no_aplicable", motivo="faltan negativos")
    idx = np.concatenate([ia, ib])

    def gap(ii):
        sub = idx[ii]
        a, b = sub[s[sub] == src_a], sub[s[sub] == src_b]
        if len(a) == 0 or len(b) == 0:
            return float("nan")
        return float((p[a] < threshold).mean() - (p[b] < threshold).mean())

    val = gap(np.arange(len(idx)))
    lo, hi = cluster_bootstrap(gap, g[idx], n_boot, seed)
    ind = "brecha_distinta_de_cero" if (lo > 0 or hi < 0) else "brecha_compatible_con_cero"
    return _result("V6", "dependencia residual", {"brecha": val, "ci_lo": lo, "ci_hi": hi, "umbral": threshold,
                                                  "esp_a": float((p[ia] < threshold).mean()),
                                                  "esp_b": float((p[ib] < threshold).mean())}, {}, "informa",
                   indicador=ind)


# ------------------------------------------------------------------------------------- V7
def trivial_descriptors(img: np.ndarray, meta: Optional[dict] = None) -> np.ndarray:
    """Descriptores que no miran la anatomía (Section 2.1, validación i): fracción de borde recortado y relación de
    aspecto original (de los metadatos de la caché), energía relativa de las 6 sub-bandas de detalle de Haar de
    dos niveles, índice de bloques 8 x 8 y percentiles 1, 5, 25, 50, 75, 95 y 99 de intensidad."""
    x = np.asarray(img, dtype=np.float64)
    if x.ndim == 3:
        x = x[0]
    h, w = (x.shape[0] // 4) * 4, (x.shape[1] // 4) * 4
    x = x[:h, :w]
    feats = []
    ll = x
    tot = (x ** 2).sum() + 1e-12
    for _ in range(2):
        a, b, c, d = ll[0::2, 0::2], ll[0::2, 1::2], ll[1::2, 0::2], ll[1::2, 1::2]
        lh, hl, hh = (a + b - c - d) / 2, (a - b + c - d) / 2, (a - b - c + d) / 2
        feats += [(lh ** 2).sum() / tot, (hl ** 2).sum() / tot, (hh ** 2).sum() / tot]
        ll = (a + b + c + d) / 2
    dh = np.abs(np.diff(x, axis=1))
    dv = np.abs(np.diff(x, axis=0))
    cols = np.arange(dh.shape[1])
    rows = np.arange(dv.shape[0])
    bnd = (dh[:, cols % 8 == 7].mean() + dv[rows % 8 == 7].mean()) / 2
    inn = (dh[:, cols % 8 != 7].mean() + dv[rows % 8 != 7].mean()) / 2
    feats.append(bnd / (inn + 1e-12))
    feats += list(np.percentile(x, [1, 5, 25, 50, 75, 95, 99]))
    meta = meta or {}
    feats = [float(meta.get("trim_frac", 0.0) or 0.0), float(meta.get("orig_aspect", 1.0) or 1.0)] + [float(v) for v in feats]
    return np.asarray(feats, dtype=np.float64)


def v7_trivial(desc_sets: Dict[str, tuple], min_n: int = 120, cap: int = 600, seed: int = 42):
    """V7: exactitud balanceada de un probe multinomial de fuente sobre descriptores triviales, por
    preprocesado: {'PRE-0': (X, s, groups), 'P1-P6': (X, s, groups)}. Se informa la reducción de PRE-0 a P1-P6."""
    out = {}
    for name, (X, s, g) in desc_sets.items():
        X, s, g = np.asarray(X, float), np.asarray(s), np.asarray(g).astype(str)
        u, c = np.unique(s, return_counts=True)
        base = np.nonzero(np.isin(s, u[c >= min_n]))[0]
        if len(np.unique(s[base])) < 2:
            out[name] = float("nan"); continue
        sel = base[cap_per_class(s[base], cap, seed)]
        hold = group_holdout(s[sel], g[sel], 0.3, seed)
        m = make_probe("logreg").fit(X[sel[~hold]], s[sel[~hold]])
        out[name] = _ba(s[sel[hold]], m.predict(X[sel[hold]]))
    metr = {"ba_" + k: v for k, v in out.items()}
    ind = None
    if "PRE-0" in out and "P1-P6" in out:
        metr["reduccion"] = out["PRE-0"] - out["P1-P6"]
        ind = "reduce_la_firma_trivial" if metr["reduccion"] > 0 else "no_reduce_la_firma_trivial"
    return _result("V7", "preprocesado", metr, {}, "informa", indicador=ind)


# ------------------------------------------------------------------------------------- recogida desde el modelo
def collect_probe_data(model, dataset, device="cpu", batch_size: int = 64, targets: Sequence[int] = (),
                       rows: Optional[np.ndarray] = None, seed: int = 42, normalize=None, gate: str = "teacher",
                       mask_fn=None):
    """Ejecuta el modelo (eval, sin gradiente) sobre `dataset` (devuelve image en [0, 1], mask, label, source,
    patient) y reúne, por imagen: signature, emb_content, emb_tokens, emb_gated, emb_decision, stats_real,
    logit, la vista dentro de la fuente (emb_within_gated; donante de la misma fuente en el lote, que se arma
    agrupado por fuente) y, para cada fuente destino de `targets`, la vista re-firmada con un donante al azar
    de esa fuente en el mismo conjunto (emb_swap_*, stats_swap, logit_swap). La firma usa siempre la máscara
    del profesor. Devuelve un diccionario de arrays."""
    import torch
    from src.model.swap import sample_within_pairs
    model.eval()
    n = len(dataset)
    rows = np.arange(n) if rows is None else np.asarray(rows)
    meta_y = np.array([int(dataset.labels[i]) for i in rows])
    meta_s = np.array([int(dataset.sources[i]) for i in rows])
    meta_g = np.array([str(dataset.patients[i]) for i in rows])
    order = np.argsort(meta_s, kind="stable")
    rng = np.random.default_rng(seed)
    all_s = np.asarray(dataset.sources)
    donors_of = {t: np.nonzero(all_s == t)[0] for t in targets}
    keys_real = ["signature", "emb_content", "emb_tokens", "emb_gated", "emb_decision", "stats_real", "logit"]
    res: Dict[str, list] = {k: [None] * len(rows) for k in keys_real + ["emb_within_gated"]}
    sw: Dict[int, Dict[str, list]] = {t: {k: [None] * len(rows) for k in
                                          ("emb_swap_gated", "emb_swap_decision", "emb_swap_content", "stats_swap",
                                           "logit_swap")} for t in targets}
    norm = normalize or (lambda x: x)

    def load(ids):
        items = [dataset[int(i)] for i in ids]
        x = torch.stack([it["image"] for it in items]).float().to(device)
        m = torch.stack([it["mask"] for it in items]).float().to(device)
        if mask_fn is not None:
            q = torch.stack([torch.as_tensor(it.get("quality", 1.0)) for it in items]).float().to(device)
            m, gm = mask_fn(m, q)
        else:
            gm = m
        return norm(x), m, gm

    with torch.no_grad():
        for b0 in range(0, len(order), batch_size):
            pos = order[b0:b0 + batch_size]
            ids = rows[pos]
            x, m, gmk = load(ids)
            s_t = torch.as_tensor(meta_s[pos])
            wp = sample_within_pairs(s_t, generator=torch.Generator().manual_seed(seed + b0))
            gm = gmk if gate == "teacher" else None
            canonical = getattr(model.cfg, "refirma_target", "table") == "canonical"
            out = model(x, lung_mask=m, gate_mask=gm, within_index=None if canonical else wp.donor_index.to(device))
            for k in keys_real:
                if k in out:
                    v = out[k].float().cpu().numpy()
                    for j, p in enumerate(pos):
                        res[k][p] = v[j]
            if "emb_within_gated" in out:
                v = out["emb_within_gated"].float().cpu().numpy()
                for j, p in enumerate(pos):
                    res["emb_within_gated"][p] = v[j]
            if canonical:
                continue
            for t in targets:
                need = [j for j in range(len(pos)) if meta_s[pos[j]] != t]
                if not need or len(donors_of[t]) == 0:
                    continue
                dids = rng.choice(donors_of[t], len(need), replace=True)
                xd, md, gd = load(dids)
                xx = torch.cat([x[need], xd]); mm = torch.cat([m[need], md]); gg = torch.cat([gmk[need], gd])
                k = len(need)
                sidx = torch.cat([torch.arange(k) + k, torch.arange(k) + k]).to(device)
                o2 = model(xx, swap_index=sidx, lung_mask=mm, gate_mask=gg if gate == "teacher" else None)
                for key in sw[t]:
                    src_key = key
                    if src_key not in o2:
                        continue
                    v = o2[src_key].float().cpu().numpy()
                    for jj, j in enumerate(need):
                        sw[t][key][pos[j]] = v[jj]
    out_np = {"y": meta_y, "s": meta_s, "groups": meta_g, "rows": rows}
    for k, lst in res.items():
        if all(v is None for v in lst):
            continue
        dim = next(v for v in lst if v is not None).shape
        out_np[k] = np.stack([v if v is not None else np.full(dim, np.nan) for v in lst])
    out_np["swap"] = {}
    for t, d in sw.items():
        out_np["swap"][int(t)] = {}
        for k, lst in d.items():
            if all(v is None for v in lst):
                continue
            dim = next(v for v in lst if v is not None).shape
            out_np["swap"][int(t)][k] = np.stack([v if v is not None else np.full(dim, np.nan) for v in lst])
    return out_np


def _named(cfg, names):
    idx = {n: i for i, n in enumerate(names)}
    return lambda n: idx.get(n, -1)


def run_probes(data: dict, source_names: Sequence[str], cfg: Optional[dict] = None,
               which: Sequence[str] = ("V1", "V1b", "V2", "V2b", "V3", "V5"), prob=None,
               threshold: Optional[float] = None) -> Dict[str, dict]:
    """Aplica las verificaciones pedidas a los datos de `collect_probe_data`. `source_names[i]` es el nombre de
    la fuente de índice i. V6 necesita `prob` (probabilidades por fila) y el umbral de Youden de val_robust."""
    c = dict(DEFAULT_PROBE_CFG, **(cfg or {}))
    ix = _named(c, list(source_names))
    y, s, g = data["y"], data["s"], data["groups"]
    nb, sd, kf = int(c["n_boot"]), int(c["seed"]), int(c["folds"])
    has_swap = bool(data.get("swap")) and any(len(v) for v in data["swap"].values())
    res: Dict[str, dict] = {}
    rob = ix(c["robust_source"])
    if "V1" in which:
        if "signature" not in data:
            res["V1"] = _result("V1", "A1", {}, THRESHOLDS["V1"], "no_aplicable", motivo="sin vía de adquisición")
        else:
            r = s == rob
            extra = {}
            for nm in c.get("v1_extra_sources", []):
                e = s == ix(nm)
                if e.sum() > 0:
                    extra[nm] = (data["signature"][e], y[e], g[e])
            res["V1"] = v1_signature_class(data["signature"][r], y[r], g[r], data["emb_content"][r], nb, sd, kf, extra)
    if "V1b" in which and "signature" in data:
        res["V1b"] = v1b_signature_content(data["signature"], data["emb_gated"], s, g, seed=sd, folds=kf,
                                           source_names=dict(enumerate(source_names)),
                                           min_n=int(c.get("min_n", 120)))
    srcs = [ix(n) for n in c["v2_sources"] if ix(n) >= 0]
    if "V2" in which:
        if not has_swap:
            res["V2"] = _result("V2", "A3", {}, THRESHOLDS["V2"], "no_aplicable", motivo="sin vistas re-firmadas")
        else:
            res["V2"] = v2_target_source(data["stats_real"], s, g,
                                         {t: d["stats_swap"] for t, d in data["swap"].items() if "stats_swap" in d},
                                         srcs, int(c["cap"]), sd, n_boot=nb)
    cap = int(c["cap"])

    def capped(rows_mask, lab):
        idx = np.nonzero(rows_mask)[0]
        keep = idx[cap_per_class(lab[idx], cap, sd)]
        return keep

    if "V2b" in which:
        if not has_swap:
            res["V2b"] = _result("V2b", "atajo", {}, THRESHOLDS["V2b"], "no_aplicable", motivo="sin vistas re-firmadas")
        else:
            st, st_real, st_dec = [], [], []
            for cls, dest, oa, ob in c["v2b_strata"]:
                t, a, b = ix(dest), ix(oa), ix(ob)
                if t not in data["swap"] or "emb_swap_gated" not in data["swap"][t]:
                    continue
                msk = (y == cls) & np.isin(s, [a, b])
                rr = capped(msk, s)
                lab = (s[rr] == b).astype(int)
                name = "%s|%s" % (cls, dest)
                st.append({"name": name, "z": data["swap"][t]["emb_swap_gated"][rr], "label": lab, "groups": g[rr]})
                st_real.append({"name": name, "z": data["emb_gated"][rr], "label": lab, "groups": g[rr]})
                if "emb_swap_decision" in data["swap"][t]:
                    st_dec.append({"name": name, "z": data["swap"][t]["emb_swap_decision"][rr], "label": lab,
                                   "groups": g[rr]})
            res["V2b"] = v2b_origin(st, st_real, st_dec, kf, sd, nb)
    if "V3" in which:
        nd, pd_ = ix(c["v3_neg_dest"]), ix(c["v3_pos_dest"])
        if not has_swap or nd not in data["swap"] or pd_ not in data["swap"]:
            res["V3"] = _result("V3", "A2, A1", {}, THRESHOLDS["V3"], "no_aplicable", motivo="sin vistas re-firmadas")
        else:
            r = s == ix(c["v3_source"])
            res["V3"] = v3_disease(data["emb_content"][r], y[r], g[r],
                                   {c["v3_neg_dest"]: data["swap"][nd]["emb_swap_content"][r],
                                    c["v3_pos_dest"]: data["swap"][pd_]["emb_swap_content"][r]},
                                   c["v3_neg_dest"], c["v3_pos_dest"], data["logit"][r],
                                   {c["v3_neg_dest"]: data["swap"][nd]["logit_swap"][r],
                                    c["v3_pos_dest"]: data["swap"][pd_]["logit_swap"][r]}, kf, sd, nb)
    if "V5" in which:
        if not has_swap:
            res["V5"] = _result("V5", "atajo", {}, THRESHOLDS["V5"], "no_aplicable", motivo="sin vistas re-firmadas")
        else:
            st, stw = [], []
            for cls, eff, orig in c["v5_strata"]:
                e, o = ix(eff), ix(orig)
                if e not in data["swap"] or "emb_swap_gated" not in data["swap"][e]:
                    continue
                rr = capped((y == cls) & (s == e), s)
                rs = capped((y == cls) & (s == o), s)
                if len(rr) == 0 or len(rs) == 0:
                    continue
                z = np.concatenate([data["emb_gated"][rr], data["swap"][e]["emb_swap_gated"][rs]])
                lab = np.r_[np.zeros(len(rr), int), np.ones(len(rs), int)]
                gg = np.r_[g[rr], g[rs]]
                name = "%s|%s" % (cls, eff)
                st.append({"name": name, "z": z, "label": lab, "groups": gg})
                if "emb_within_gated" in data:
                    zw = np.concatenate([data["emb_gated"][rr], data["emb_within_gated"][rr]])
                    stw.append({"name": name, "z": zw, "label": np.r_[np.zeros(len(rr), int), np.ones(len(rr), int)],
                                "groups": np.r_[np.char.add(g[rr], "_r"), np.char.add(g[rr], "_w")]})
            res["V5"] = v5_view(st, stw, kf, sd, nb)
    if "V4" in which:
        reps = {k2: data[k1] for k1, k2 in (("emb_content", "nivel1_content"), ("emb_tokens", "nivel1_tokens"),
                                            ("emb_decision", "nivel2_decision")) if k1 in data}
        acc = v4_levels(reps, s, g, srcs or None, cap, sd)
        res["V4"] = _result("V4", "dependencia residual", acc, THRESHOLDS["V4"], "informa",
                            indicador="pendiente_de_comparacion_entre_corridas (v4_compare)")
    if "V6" in which and prob is not None and threshold is not None:
        a, b = (ix(n) for n in c["v6_sources"])
        res["V6"] = v6_specificity_gap(prob, y, s, g, threshold, a, b, nb, sd)
    return res


def write_report(results: Dict[str, dict], out_dir, name: str, header: Optional[dict] = None):
    """Escribe <out_dir>/<name>.json y <name>.md con una fila por verificación."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"cabecera": _clean(header or {}), "verificaciones": _clean(results)}
    (out / (name + ".json")).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Verificaciones de identificabilidad: %s" % name, ""]
    for k, v in (header or {}).items():
        lines.append("- %s: %s" % (k, v))
    lines += ["", "| Id | Supuesto | Veredicto | Métricas principales | Umbrales |", "|---|---|---|---|---|"]
    for vid, r in results.items():
        mets = ", ".join("%s=%s" % (k, ("%.4f" % val) if isinstance(val, float) else val)
                         for k, val in r["metricas"].items() if not isinstance(val, (dict, list)))
        thr = ", ".join("%s=%s" % (k, val) for k, val in (r.get("umbrales") or {}).items())
        ver = r["veredicto"] + ((" (%s)" % r["detalle"]["indicador"]) if r.get("detalle", {}).get("indicador") else "")
        lines.append("| %s | %s | %s | %s | %s |" % (vid, r["supuesto"], ver, mets, thr))
    (out / (name + ".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out / (name + ".json")
