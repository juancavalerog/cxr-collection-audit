"""Aporte C3: la validación con la distribución de entrenamiento no sirve para elegir modelo cuando la confusión es
fuerte (docs/08_evaluacion.md, "Enmiendas previas a la congelación (17-11-2026)", enmienda E2; hallazgo B3).

Estimandos registrados:

(a) Selección de época, con las corridas finales de B3a: para cada método m, nivel rho y semilla s,
    Δsel = AUC_test_desconf(bestdc) - AUC_test_desconf(best); se informa la media por semilla con IC95 jerárquico y
    la media sobre los métodos, Δsel(rho), que es el estimando de la familia F-C3 (H1: Δsel(rho) > 0 en rho 0.9,
    0.95 y 0.99; Holm sobre tres). En DFR se usan los estados de su etapa ERM (best y bestdc), para aislar la época:
    dfr_bestdc cambia además los datos del reajuste.
(b) Selección de hiperparámetros (solo si el autor aprueba la repetición de la búsqueda con evaluación del test,
    `hiperparametros`): ganancia de la regla registrada sobre la elección al azar, G = AUC_test(regla) - media de
    AUC_test sobre los ensayos, pérdida frente al oráculo de validación desconfundida, L = AUC_test(oráculo) -
    AUC_test(regla), y Spearman entre la métrica de selección y AUC_test entre ensayos.
(c) Descriptivo: Spearman, entre métodos y por nivel, entre la métrica de selección media por semilla y la AUC media
    en test_desconf (`spearman_metodos`).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.eval import c2 as C2
from src.eval import dosis as D
from src.eval import jerarquico as J
from src.eval import stats as S

NIVELES_F_C3 = (0.9, 0.95, 0.99)
N_ENSAYOS = 8


def _estados_epoca(root, bench, orient, conjunto, metodos, verificador):
    """Predicciones de best y bestdc en `conjunto`; en DFR los estados de su etapa ERM (sin prefijo dfr_)."""
    reg = C2.exigir([conjunto], "c3.seleccion_epoca %s" % conjunto, verificador)
    out = {}
    permitir = C2.es_bloqueado(conjunto)
    for estado in ("best", "bestdc"):
        inv = D.descubrir(root, bench, orient, estado, (conjunto,), permitir_test=permitir)
        inv = inv[inv.metodo.isin(list(metodos))].copy()
        es_dfr = inv.metodo == "dfr"
        inv.loc[es_dfr, "estado"] = estado
        inv.loc[es_dfr, "ruta"] = [str(Path(r).parent / ("%s_%s.csv" % (estado, conjunto))) for r in inv.loc[es_dfr, "ruta"]]
        inv["existe"] = [Path(r).exists() for r in inv["ruta"]]
        out[estado] = D.cargar(inv, conjunto, permitir_test=permitir, verificador=lambda motivo: reg)
    return out["best"], out["bestdc"]


def seleccion_epoca(preds_best: Dict[tuple, pd.DataFrame], preds_bestdc: Dict[tuple, pd.DataFrame],
                    metodos: Optional[Sequence[str]] = None, B: int = S.DEFAULT_B, seed: int = S.DEFAULT_BOOT_SEED,
                    seed_semillas: int = J.SEMILLA_SEMILLAS, niveles_familia: Sequence[float] = NIVELES_F_C3,
                    alpha: float = 0.025):
    """Δsel por método y nivel y su media sobre métodos, con IC95 jerárquico. Devuelve (por_metodo, por_nivel)."""
    metodos = list(metodos or sorted({k[0] for k in preds_best}))
    ref = preds_best[sorted(preds_best)[0]]
    sem = sorted({int(k[2]) for k in preds_best})
    niveles = sorted({float(k[1]) for k in preds_best})
    R, K, M = len(niveles), len(sem), len(metodos)
    X = np.full((int(B), R, K, M), np.nan)
    Xp = np.full((R, K, M), np.nan)
    pres = np.zeros((R, K, M), bool)
    for q, m in enumerate(metodos):
        pb, db, rb = C2.cubo(preds_best, m, niveles, sem, ref, B, seed)
        pc, dc, rc = C2.cubo(preds_bestdc, m, niveles, sem, ref, B, seed)
        pr = rb & rc
        X[..., q] = np.where(pr[None], dc - db, np.nan)
        Xp[..., q] = np.where(pr, pc - pb, np.nan)
        pres[..., q] = pr
    pres_sem = pres.any(2)                                  # una semilla existe en un nivel si algún método la tiene
    Cm = J.multiplicidades(pres_sem, int(B), seed_semillas)
    filas, medias_b = [], np.full((int(B), R, M), np.nan)
    for q, m in enumerate(metodos):
        if not pres[..., q].any():
            continue
        hj = J.extraer(X[..., q], Xp[..., q], pres[..., q], Cm)
        medias_b[:, :, q] = hj
        for i, r in enumerate(niveles):
            v = Xp[i, pres[i, :, q], q]
            if not len(v):
                continue
            lo, hi = J.ic(hj[:, i])
            fila = {"metodo": m, "rho": r, "delta_sel": float(v.mean()), "ic95_inf": lo, "ic95_sup": hi,
                    "semillas_a_favor_bestdc": int((v > 0).sum()), "n_semillas": len(v)}
            fila.update(J.t_pareada(v))
            filas.append(fila)
    por_metodo = pd.DataFrame(filas)
    filas = []
    for i, r in enumerate(niveles):
        ok = [q for q in range(M) if pres[i, :, q].any()]
        if not ok:
            continue
        punt = float(np.mean([Xp[i, pres[i, :, q], q].mean() for q in ok]))
        mb = np.nanmean(medias_b[:, i, ok], axis=1)
        lo, hi = J.ic(mb)
        filas.append({"rho": r, "delta_sel_media_metodos": punt, "ic95_inf": lo, "ic95_sup": hi,
                      "p_unilateral": J.p_mayor(mb), "n_metodos": len(ok),
                      "familia": "F-C3" if any(abs(r - x) < 1e-9 for x in niveles_familia) else "",
                      "papel": "confirmatoria" if any(abs(r - x) < 1e-9 for x in niveles_familia) else "descriptiva"})
    por_nivel = pd.DataFrame(filas)
    if len(por_nivel):
        por_nivel["p_holm"] = np.nan
        por_nivel["rechaza_holm"] = False
        fam = por_nivel[por_nivel.familia == "F-C3"]
        if len(fam):
            h = S.holm({i: p for i, p in zip(fam.index, fam.p_unilateral)}, alpha=alpha)
            for i in fam.index:
                por_nivel.loc[i, "p_holm"] = h[i]["p_holm"]
                por_nivel.loc[i, "rechaza_holm"] = bool(h[i]["rechaza"])
    return por_metodo, por_nivel


def metrica_seleccion(root, bench="b3a", orient="A", metodos=D.METODOS_B3) -> Dict[tuple, float]:
    """(método, rho, semilla) -> early_stopping.best del summary.json de la corrida final (la métrica de la regla)."""
    out = {}
    root = Path(root)
    for exp in sorted(p for p in root.iterdir() if p.is_dir()):
        info = D.parse_nombre(exp.name)
        if not info or info["bench"] != bench or info["orient"] != orient or info["metodo"] not in metodos:
            continue
        for s, run in S.find_seed_dirs(root, exp.name).items():
            sp = run / "summary.json"
            if sp.exists():
                sj = json.loads(sp.read_text(encoding="utf-8"))
                if sj.get("estado") != "no_aplicable" and (sj.get("early_stopping") or {}).get("best") is not None:
                    out[(info["metodo"], info["rho"], int(s))] = float(sj["early_stopping"]["best"])
    return out


def _spearman(a, b) -> float:
    """Spearman; NaN si alguna serie es constante (p. ej. la métrica de selección saturada en rho = 1.0)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(a, b).correlation)


def spearman_metodos(preds: Dict[tuple, pd.DataFrame], sel: Dict[tuple, float], B: int = S.DEFAULT_B,
                     seed: int = S.DEFAULT_BOOT_SEED, seed_semillas: int = J.SEMILLA_SEMILLAS) -> pd.DataFrame:
    """Spearman entre métodos, por nivel, entre la métrica de selección media por semilla y la AUC media por semilla
    en el conjunto de `preds`; IC95 con el mismo remuestreo jerárquico (descriptivo, sin valor p)."""
    metodos = sorted({k[0] for k in preds})
    ref = preds[sorted(preds)[0]]
    sem = sorted({int(k[2]) for k in preds})
    filas = []
    for r in sorted({float(k[1]) for k in preds}):
        A = np.full((int(B), len(sem), len(metodos)), np.nan)
        Sm = np.full((len(sem), len(metodos)), np.nan)
        Ap = np.full((len(sem), len(metodos)), np.nan)
        for q, m in enumerate(metodos):
            p, d, pr = C2.cubo(preds, m, [r], sem, ref, B, seed)
            A[:, :, q], Ap[:, q] = d[:, 0], p[0]
            for j, s in enumerate(sem):
                Sm[j, q] = sel.get((m, r, s), np.nan) if pr[0, j] else np.nan
        presq = np.isfinite(Sm) & np.isfinite(Ap)
        ok = presq.any(0)
        if ok.sum() < 3:
            continue
        sel_m = np.array([Sm[presq[:, q], q].mean() if presq[:, q].any() else np.nan for q in range(len(metodos))])
        auc_m = np.array([Ap[presq[:, q], q].mean() if presq[:, q].any() else np.nan for q in range(len(metodos))])
        rho_p = _spearman(sel_m[ok], auc_m[ok])
        C = J.multiplicidades(presq.any(1)[None, :], int(B), seed_semillas)
        rs = []
        for b in range(int(B)):
            w = C[b][:, None] * presq
            den = w.sum(0)
            sb = np.where(den > 0, (np.nan_to_num(Sm) * w).sum(0) / np.maximum(den, 1e-12), np.nan)
            ab = np.where(den > 0, (np.nan_to_num(A[b]) * w).sum(0) / np.maximum(den, 1e-12), np.nan)
            o = np.isfinite(sb) & np.isfinite(ab) & ok
            rs.append(_spearman(sb[o], ab[o]) if o.sum() >= 3 else np.nan)
        lo, hi = J.ic(rs)
        filas.append({"rho": r, "spearman": rho_p, "ic95_inf": lo, "ic95_sup": hi, "n_metodos": int(ok.sum()),
                      "metrica_seleccion_empatada": bool(np.unique(np.round(sel_m[ok], 12)).size < ok.sum())})
    return pd.DataFrame(filas)


# ===================================================================================== (b) hiperparámetros
def elegir_regla(valores: Dict[int, float]) -> int:
    """Regla registrada (scripts/jobs/bench_jobs.py, elegir): máximo de early_stopping.best, empate al ensayo menor."""
    best = None
    for t in sorted(valores):
        v = valores[t]
        if best is None or v > best[0]:
            best = (v, t)
    return best[1]


def hiperparametros(root, metodos=("erm", "lisa", "groupdro", "dfr"), niveles=NIVELES_F_C3, conjunto="test_desconf",
                    B: int = S.DEFAULT_B, seed: int = S.DEFAULT_BOOT_SEED, verificador=None, semilla: int = 0):
    """Análisis registrado (b), condicionado a la aprobación del autor: lee `root/b3a_A_<m>_rho<RRR>_t<T>/seed0` con
    las predicciones del estado best (dfr_best en DFR) en val_desconf y en `conjunto`. El oráculo elige el ensayo con
    la AUC en val_desconf; en DFR esa AUC se calcula sin las imágenes de val (que contienen su reajuste). IC95 por
    pacientes (una semilla: no recoge la variación entre semillas, se declara)."""
    C2.exigir([conjunto], "c3.hiperparametros %s" % conjunto, verificador)
    root = Path(root)
    filas, rhos_s = [], []
    for m in metodos:
        st = "dfr_best" if m == "dfr" else "best"
        for r in niveles:
            sel, vdc, P = {}, {}, {}
            for t in range(N_ENSAYOS):
                run = root / ("b3a_A_%s_rho%03d_t%d" % (m, round(100 * r), t)) / ("seed%d" % semilla)
                sp = run / "summary.json"
                ft = run / "predictions" / ("%s_%s.csv" % (st, conjunto))
                fv = run / "predictions" / ("%s_val_desconf.csv" % st)
                if not (sp.exists() and ft.exists() and fv.exists()):
                    continue
                sj = json.loads(sp.read_text(encoding="utf-8"))
                sel[t] = float(sj["early_stopping"]["best"])
                v = S.read_predictions(fv, forbid_test=True)
                if m == "dfr":
                    fval = run / "predictions" / ("%s_val.csv" % st)
                    if fval.exists():
                        v = v[~v.image_id.isin(set(pd.read_csv(fval, usecols=["image_id"], dtype=str).image_id))]
                vdc[t] = S.auc_roc(v["label"].to_numpy(), v["prob"].to_numpy())
                P[t] = S.read_predictions(ft, forbid_test=False)
            if len(P) < 2:
                continue
            ts = sorted(P)
            ref = P[ts[0]]
            M = np.vstack([S.align(ref, P[t])["prob"].to_numpy() for t in ts])
            y, g = ref["label"].to_numpy(), ref["patient"].to_numpy()
            a = np.array([S.auc_roc(y, M[i]) for i in range(len(ts))])
            Db = D.auc_remuestreo(y, M, g, B, seed)
            tr = elegir_regla({t: sel[t] for t in ts})
            to = max(ts, key=lambda t: (vdc[t], -t))
            ir, io = ts.index(tr), ts.index(to)
            G, L = a[ir] - a.mean(), a[io] - a[ir]
            Gb, Lb = Db[:, ir] - Db.mean(1), Db[:, io] - Db[:, ir]
            sp_ = _spearman([sel[t] for t in ts], a)
            filas.append({"metodo": m, "rho": r, "n_ensayos": len(ts), "ensayo_regla": tr, "ensayo_oraculo": to,
                          "auc_regla": float(a[ir]), "auc_oraculo": float(a[io]), "auc_media_ensayos": float(a.mean()),
                          "ganancia_sobre_azar": float(G), "ic95_inf_G": J.ic(Gb)[0], "ic95_sup_G": J.ic(Gb)[1],
                          "perdida_frente_oraculo": float(L), "ic95_inf_L": J.ic(Lb)[0], "ic95_sup_L": J.ic(Lb)[1],
                          "spearman_sel_test": float(sp_) if np.isfinite(sp_) else np.nan})
    return pd.DataFrame(filas)
