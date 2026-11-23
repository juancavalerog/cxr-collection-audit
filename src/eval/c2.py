"""Análisis de C2 (dosis-respuesta en B3a) y de su réplica en tuberculosis C2-R (B2) con remuestreo jerárquico de
semillas y pacientes (docs/08_evaluacion.md, "Enmiendas previas a la congelación (17-11-2026)", E1 a E3 y E7).

Contenido:
- `cubo`: AUC puntual (R, K) y remuestreada por pacientes (B, R, K) de un método en sus niveles y semillas, con la
  matriz de presencia (cualquier número de semillas por nivel).
- `curva_jerarquica`: curva, medidas de la ecuación (12) y comparación con ERM (Δ por nivel, área, caída, coste en
  rho 0.5), con IC95 jerárquico (principal), IC95 de pacientes con modelos fijos (sensibilidad, el de la versión 2) y
  prueba t pareada entre semillas (sensibilidad).
- `familias_c2`: familias de Holm registradas (F-C2-area confirmatoria; F-C2-099, F-C2-100 y F-C2-coste secundarias;
  el resto descriptivo, sin valor p de Holm).
- `invertido_frente_desconf`, `auc_por_fuente`, `brecha_especificidad`: V2.4.3.
- `orientacion_b`: V2.4.4.
- `cargar_b2`, `informe_b2_tablas`: C2-R (V2.5, familias F-C2R-shenzhen y F-C2R-montgomery).
- `cargar_b3`: lectura del árbol de B3a; si el conjunto es un test exige la congelación del protocolo.

Ninguna función abre un conjunto test sin pasar por `exigir` (src.eval.desbloqueo.exigir_congelacion), salvo que el
llamante pase un verificador explícito (solo las pruebas y las demostraciones sintéticas lo hacen).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from src.eval import dosis as D
from src.eval import jerarquico as J
from src.eval import stats as S

METODOS_B3A = tuple(m for m in D.METODOS_B3 if m != "control_fuente")   # control de fuente no aplicable (enmienda 2)
METODOS_B2 = D.METODOS_B3
CONJUNTOS_B2 = ("test_shenzhen", "test_montgomery")
NOMBRE_B2 = re.compile(r"^b2_(?P<metodo>[a-z0-9_]+?)$")
UMBRAL_ORIENTACION = 0.05


# ===================================================================================== desbloqueo
def es_bloqueado(conjunto) -> bool:
    n = str(conjunto).strip().lower()
    return n == "test" or n.startswith("test_") or "covidgr" in n


def exigir(conjuntos, motivo: str, verificador: Optional[Callable[[str], dict]] = None) -> Optional[dict]:
    """Si algún conjunto es un test, exige la congelación del protocolo antes de leer nada. `verificador` sustituye
    a `exigir_congelacion` solo en pruebas y demostraciones sintéticas."""
    pedidos = [c for c in conjuntos if es_bloqueado(c)]
    if not pedidos:
        return None
    m = "%s (%s)" % (motivo, ", ".join(pedidos))
    if verificador is not None:
        return verificador(m)
    from src.eval.desbloqueo import exigir_congelacion   # importado aquí: solo al leer un test
    return exigir_congelacion(m)


# ===================================================================================== no comparables de DFR
def no_comparable(bench: str, estado: str, conjunto: str) -> Optional[str]:
    """Regla completa de no comparabilidad de DFR (enmienda E5, hallazgo m4): un estado reajustado no es comparable
    en ningún conjunto que contenga imágenes de su conjunto de reajuste. B3a: val contiene val_fit y comparte de 168
    a 283 imágenes con val_desconf (85 a 150 de val_fit); val_desconf contiene val_desconf_fit, que también aparece en
    val. B2: val_robust está dentro de val. COVIDx: val_robust_fit está en val y en val_robust."""
    b, st, c = str(bench).lower(), str(estado), str(conjunto)
    if not st.startswith("dfr_"):
        return None
    if b in ("b3a", "b3b"):
        if c in ("val", "val_desconf"):
            return "%s/%s contiene datos (o imágenes compartidas con) el reajuste de %s" % (st, c, st)
    elif b == "b2":
        if c in ("val", "val_robust"):
            return "%s/%s contiene datos del reajuste de %s (val_robust está dentro de val)" % (st, c, st)
    elif b in ("b1", "covidx"):
        if c in ("val", "val_robust", "val_robust_fit"):
            return "%s/%s contiene val_robust_fit, con el que se reajusta %s" % (st, c, st)
    return None


# ===================================================================================== lectura
def cargar_b3(root, conjunto: str, estado: str = "best", bench: str = "b3a", orient: str = "A",
              metodos: Optional[Sequence[str]] = None, verificador=None, incluir_no_comparables: bool = False):
    """(método, rho, semilla) -> predicciones de `conjunto`, el inventario y las exclusiones por no comparabilidad."""
    reg = exigir([conjunto], "c2.cargar_b3 %s_%s %s %s" % (bench, orient, estado, conjunto), verificador)
    permitir = es_bloqueado(conjunto)
    inv = D.descubrir(root, bench, orient, estado, (conjunto,), permitir_test=permitir)
    if metodos is not None:
        inv = inv[inv.metodo.isin(list(metodos))]
    inv = inv.copy()
    inv["no_comparable"] = [no_comparable(bench, e, c) or "" for e, c in zip(inv.estado, inv.conjunto)]
    excl = inv[(inv.no_comparable != "") & inv.existe]
    if not incluir_no_comparables:
        inv_ok = inv[inv.no_comparable == ""]
    else:
        inv_ok = inv
    preds = D.cargar(inv_ok, conjunto, permitir_test=permitir, verificador=lambda motivo: reg)   # ya verificada arriba
    return preds, inv, excl


def cargar_b2(root, conjunto: str, estado: str = "best", metodos: Optional[Sequence[str]] = None, verificador=None,
              incluir_no_comparables: bool = False):
    """Corridas finales de B2 (`b2_<metodo>/seed<S>`): (método, 0.0, semilla) -> predicciones de `conjunto`."""
    exigir([conjunto], "c2.cargar_b2 %s %s" % (estado, conjunto), verificador)
    permitir = es_bloqueado(conjunto)
    root = Path(root)
    preds, rows = {}, []
    for exp in sorted(p for p in root.iterdir() if p.is_dir()):
        m = NOMBRE_B2.match(exp.name)
        if not m or (metodos is not None and m.group("metodo") not in metodos):
            continue
        met = m.group("metodo")
        if met not in METODOS_B2:
            continue
        st = ("dfr_" + estado) if met == "dfr" else estado
        nc = no_comparable("b2", st, conjunto)
        for seed, run in S.find_seed_dirs(root, exp.name).items():
            f = run / "predictions" / ("%s_%s.csv" % (st, conjunto))
            rows.append({"experimento": exp.name, "metodo": met, "semilla": seed, "estado": st, "conjunto": conjunto,
                         "ruta": str(f), "existe": f.exists(), "no_comparable": nc or ""})
            if f.exists() and (not nc or incluir_no_comparables):
                preds[(met, 0.0, int(seed))] = S.read_predictions(f, forbid_test=not permitir, context="C2-R")
    return preds, pd.DataFrame(rows)


# ===================================================================================== cubo de AUC
def cubo(preds, metodo: str, niveles: Sequence[float], semillas: Sequence[int], ref: pd.DataFrame, B: int,
         seed: int, columna: str = "prob"):
    """AUC puntual (R, K), remuestreada por pacientes (B, R, K) y presencia (R, K) de `metodo` alineado con `ref`.
    Las celdas sin corrida quedan en NaN y con presencia False."""
    y, g = ref["label"].to_numpy(), ref["patient"].to_numpy()
    R, K = len(niveles), len(semillas)
    pres = np.zeros((R, K), bool)
    filas, pos = [], []
    for i, r in enumerate(niveles):
        for j, s in enumerate(semillas):
            k = (metodo, float(r), int(s))
            if k in preds:
                d = S.align(ref, preds[k], "referencia", "%s rho=%s semilla %d" % (metodo, r, s))
                filas.append(d[columna].to_numpy())
                pos.append((i, j))
                pres[i, j] = True
    punt = np.full((R, K), np.nan)
    dr = np.full((int(B), R, K), np.nan)
    if filas:
        M = np.vstack(filas)
        a = np.array([S.auc_roc(y, M[q]) for q in range(len(filas))])
        Db = D.auc_remuestreo(y, M, g, B, seed)
        for q, (i, j) in enumerate(pos):
            punt[i, j] = a[q]
            dr[:, i, j] = Db[:, q]
    return punt, dr, pres


def _media_puntual(punt, pres):
    with np.errstate(invalid="ignore"):
        return np.array([np.nanmean(np.where(pres[i], punt[i], np.nan)) if pres[i].any() else np.nan
                         for i in range(punt.shape[0])])


def _area(niveles, v, diseno):
    r = [round(float(x), 4) for x in niveles]
    if not set(round(float(x), 4) for x in diseno) <= set(r):
        return None
    idx = [r.index(round(float(x), 4)) for x in diseno]
    return D.area_dosis(diseno, np.asarray(v)[..., idx])


def _ref_frame(preds, referencia):
    claves = sorted(k for k in preds if k[0] == referencia)
    if not claves:
        raise ValueError("falta la referencia %s" % referencia)
    return preds[claves[0]]


def curva_jerarquica(preds: Dict[tuple, pd.DataFrame], metodos: Optional[Sequence[str]] = None,
                     B: int = S.DEFAULT_B, seed: int = S.DEFAULT_BOOT_SEED, seed_semillas: int = J.SEMILLA_SEMILLAS,
                     referencia: str = D.REFERENCIA, margen_coste: float = D.MARGEN_COSTE,
                     diseno: Sequence[float] = D.RHOS_AREA):
    """Curva dosis-respuesta, medidas resumen y comparación con ERM con IC95 jerárquico (principal), IC95 de pacientes
    (sensibilidad) y t pareada entre semillas (sensibilidad). Cada nivel usa todas las semillas que tiene.
    Devuelve (curva, resumen, comparacion, info)."""
    metodos = list(metodos or sorted({k[0] for k in preds}))
    ref = _ref_frame(preds, referencia)
    y, g = ref["label"].to_numpy(), ref["patient"].to_numpy()
    semillas = sorted({int(k[2]) for k in preds})
    res = {}
    for m in metodos:
        nv = sorted({float(k[1]) for k in preds if k[0] == m})
        punt, dr, pres = cubo(preds, m, nv, semillas, ref, B, seed)
        res[m] = {"niveles": nv, "punt": punt, "draws": dr, "pres": pres}
    fc, fr, fk = [], [], []
    for m in metodos:
        d = res[m]
        nv, punt, dr, pres = d["niveles"], d["punt"], d["draws"], d["pres"]
        media = _media_puntual(punt, pres)
        hj = J.remuestreo_jerarquico(dr, pres, seed_semillas, punt=punt)
        hp = J.media_semillas(dr, pres)
        for i, r in enumerate(nv):
            ks = [semillas[j] for j in range(len(semillas)) if pres[i, j]]
            vals = punt[i, pres[i]]
            lo, hi = J.ic(np.clip(hj[:, i], 0.0, 1.0))        # el término de semillas es aditivo: se acota a [0, 1]
            lp, hpp = J.ic(hp[:, i])
            fc.append({"metodo": m, "rho": r, "auc_media_semillas": float(media[i]),
                       "dt_semillas": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
                       "ic95_inf": lo, "ic95_sup": hi, "ic95_inf_pacientes": lp, "ic95_sup_pacientes": hpp,
                       "n_semillas": len(ks), "semillas": " ".join(map(str, ks)),
                       "aucs_por_semilla": " ".join("%.4f" % v for v in vals)})
        mp = D.medidas(nv, media, diseno)
        mj = D.medidas(nv, hj, diseno)
        mpp = D.medidas(nv, hp, diseno)
        for k in mp:
            lo, hi = J.ic(mj[k])
            lp, hpp = J.ic(mpp[k])
            fr.append({"metodo": m, "medida": k, "estimacion": float(mp[k]), "ic95_inf": lo, "ic95_sup": hi,
                       "ic95_inf_pacientes": lp, "ic95_sup_pacientes": hpp})
        if "area_dosis" not in mp:
            fr.append({"metodo": m, "medida": "area_dosis", "estimacion": float("nan"), "ic95_inf": float("nan"),
                       "ic95_sup": float("nan"), "nota": "curva incompleta: niveles %s" % nv})
    dref = res[referencia]
    for m in metodos:
        if m == referencia:
            continue
        nv = [r for r in res[m]["niveles"] if r in dref["niveles"]]
        if not nv:
            continue
        im = [res[m]["niveles"].index(r) for r in nv]
        ir = [dref["niveles"].index(r) for r in nv]
        pres = res[m]["pres"][im] & dref["pres"][ir]
        if not pres.any():
            continue
        X = res[m]["draws"][:, im] - dref["draws"][:, ir]
        Xp = res[m]["punt"][im] - dref["punt"][ir]
        hj = J.remuestreo_jerarquico(X, pres, seed_semillas, punt=Xp)
        hd = J.remuestreo_jerarquico(X, pres, seed_semillas, modo="doble")
        hp = J.media_semillas(X, pres)
        comp_v = J.componentes(X, Xp, pres)
        dif = _media_puntual(Xp, pres)
        for i, r in enumerate(nv):
            lo, hi = J.ic(hj[:, i])
            lp, hpp = J.ic(hp[:, i])
            vals = Xp[i, pres[i]]
            t = J.t_pareada(vals)
            fila = {"metodo": m, "medida": "delta_auc_rho%03d" % round(100 * r), "rho": r, "delta": float(dif[i]),
                    "ic95_inf": lo, "ic95_sup": hi, "p_unilateral": J.p_mayor(hj[:, i]),
                    "ic95_inf_pacientes": lp, "ic95_sup_pacientes": hpp, "p_unilateral_pacientes": J.p_mayor(hp[:, i]),
                    "ic95_inf_doble": J.ic(hd[:, i])[0], "ic95_sup_doble": J.ic(hd[:, i])[1],
                    "de_semillas_sigma_S": float(np.sqrt(comp_v["sigma2_S"][i])), "de_pacientes_V_P": float(np.sqrt(comp_v["V_P"][i])),
                    "semillas_a_favor": int((vals > 0).sum()), "n_semillas": int(pres[i].sum())}
            fila.update(t)
            fk.append(fila)
        # medidas que combinan niveles: con los niveles comunes y, para la t, las semillas presentes en todos
        comunes = pres.all(0)
        for k in ("area_dosis", "caida_050_100"):
            mp = D.medidas(nv, dif, diseno)
            if k not in mp:
                continue
            mj, mpp, mdd = D.medidas(nv, hj, diseno)[k], D.medidas(nv, hp, diseno)[k], D.medidas(nv, hd, diseno)[k]
            per_seed = D.medidas(nv, Xp[:, comunes].T, diseno)[k] if comunes.any() else np.array([])
            caida = k.startswith("caida")
            sg = -1.0 if caida else 1.0                      # una caída menor es mejor: H1 es delta de caída < 0
            lo, hi = J.ic(mj)
            lp, hpp = J.ic(mpp)
            t = J.t_pareada(sg * np.asarray(per_seed))
            t["t_media"] = sg * t["t_media"]
            if caida and np.isfinite(t["t_ic95_inf"]):
                t["t_ic95_inf"], t["t_ic95_sup"] = -t["t_ic95_sup"], -t["t_ic95_inf"]
            fila = {"metodo": m, "medida": "delta_" + k, "rho": float("nan"), "delta": float(mp[k]),
                    "ic95_inf": lo, "ic95_sup": hi, "p_unilateral": J.p_mayor(sg * mj),
                    "ic95_inf_pacientes": lp, "ic95_sup_pacientes": hpp, "p_unilateral_pacientes": J.p_mayor(sg * mpp),
                    "ic95_inf_doble": J.ic(mdd)[0], "ic95_sup_doble": J.ic(mdd)[1],
                    "semillas_a_favor": int((sg * np.asarray(per_seed) > 0).sum()), "n_semillas": int(comunes.sum())}
            fila.update(t)
            fk.append(fila)
        r4 = [round(x, 4) for x in nv]
        if 0.5 in r4:
            i = r4.index(0.5)
            lo, hi = J.ic(hj[:, i])
            lp, hpp = J.ic(hp[:, i])
            vals = Xp[i, pres[i]]
            fila = {"metodo": m, "medida": "coste_rho050_no_inferioridad", "rho": 0.5, "delta": float(dif[i]),
                    "ic95_inf": lo, "ic95_sup": hi, "p_unilateral": J.p_mayor(hj[:, i], -float(margen_coste)),
                    "ic95_inf_pacientes": lp, "ic95_sup_pacientes": hpp,
                    "p_unilateral_pacientes": J.p_mayor(hp[:, i], -float(margen_coste)),
                    "ic95_inf_doble": J.ic(hd[:, i])[0], "ic95_sup_doble": J.ic(hd[:, i])[1],
                    "semillas_a_favor": int((vals > 0).sum()), "n_semillas": int(pres[i].sum()),
                    "margen": float(margen_coste), "no_inferior": bool(np.isfinite(lo) and lo > -margen_coste)}
            fila.update(J.t_pareada(vals, -float(margen_coste)))
            fk.append(fila)
    info = {"metodos": metodos, "referencia": referencia, "niveles_por_metodo": {m: res[m]["niveles"] for m in metodos},
            "semillas_por_nivel": {m: {str(r): [semillas[j] for j in range(len(semillas)) if res[m]["pres"][i, j]]
                                       for i, r in enumerate(res[m]["niveles"])} for m in metodos},
            "diseno_rho": list(diseno), "B": int(B), "semilla_bootstrap": int(seed),
            "semilla_remuestreo_semillas": int(seed_semillas), "remuestreo": "jerárquico corregido (ecuación (15)): "
            "pacientes más efectos de semilla reducidos (estratos por patrón de presencia, pareados entre métodos y "
            "niveles); sensibilidades: pacientes con modelos fijos, doble sin corrección y t pareada", "n_imagenes": int(len(y)),
            "n_pacientes": int(len(np.unique(g.astype(str)))), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
    return pd.DataFrame(fc), pd.DataFrame(fr), pd.DataFrame(fk), info


# ===================================================================================== familias
FAMILIAS_C2 = {"delta_area_dosis": ("F-C2-area", "confirmatoria"), "delta_auc_rho099": ("F-C2-099", "secundaria"),
               "delta_auc_rho100": ("F-C2-100", "secundaria"),
               "coste_rho050_no_inferioridad": ("F-C2-coste", "secundaria")}


def familias_c2(comp: pd.DataFrame, alpha: float = 0.025, mapa=FAMILIAS_C2, columna_p: str = "p_unilateral") -> pd.DataFrame:
    """Holm por familia registrada (enmienda E5: solo F-C2-area es confirmatoria). Las filas descriptivas quedan sin
    p de Holm (NaN) y con papel 'descriptiva'."""
    comp = comp.copy()
    comp["familia"] = ""
    comp["papel"] = "descriptiva"
    for medida, (f, papel) in mapa.items():
        sel = comp["medida"] == medida
        comp.loc[sel, "familia"] = f
        comp.loc[sel, "papel"] = papel
    comp["p_holm"] = np.nan
    comp["rechaza_holm"] = False
    comp["n_familia"] = 0
    for f, sub in comp[comp["familia"] != ""].groupby("familia"):
        h = S.holm({i: p for i, p in zip(sub.index, sub[columna_p])}, alpha=alpha)
        for i in sub.index:
            comp.loc[i, "p_holm"] = h[i]["p_holm"]
            comp.loc[i, "rechaza_holm"] = bool(h[i]["rechaza"])
            comp.loc[i, "n_familia"] = int(np.isfinite(sub[columna_p]).sum())
    comp["confirmatoria"] = comp["papel"] == "confirmatoria"
    return comp


# ===================================================================================== V2.4.3
def _pesos_union(frames: Sequence[pd.DataFrame], B: int, seed: int):
    """Pesos por imagen de cada conjunto con los mismos pacientes remuestreados sobre la unión de pacientes."""
    union = np.unique(np.concatenate([f["patient"].astype(str).to_numpy() for f in frames]))
    G = len(union)
    rng = np.random.default_rng(int(seed))
    cnt = np.empty((int(B), G))
    for b in range(int(B)):
        cnt[b] = np.bincount(rng.integers(0, G, G), minlength=G)
    out = []
    for f in frames:
        idx = np.searchsorted(union, f["patient"].astype(str).to_numpy())
        out.append(cnt[:, idx])
    return out


def invertido_frente_desconf(preds_dc, preds_inv, metodos=None, B=S.DEFAULT_B, seed=S.DEFAULT_BOOT_SEED,
                             seed_semillas=J.SEMILLA_SEMILLAS) -> pd.DataFrame:
    """AUC media por semilla en test_invertido y diferencia test_invertido - test_desconf del mismo modelo, por método
    y nivel, con IC95 jerárquico (pacientes remuestreados sobre la unión de los dos conjuntos)."""
    metodos = list(metodos or sorted({k[0] for k in preds_dc}))
    filas = []
    for m in metodos:
        for r in sorted({float(k[1]) for k in preds_dc if k[0] == m}):
            ks = sorted(k[2] for k in preds_dc if k[0] == m and float(k[1]) == r and (m, r, k[2]) in preds_inv)
            if not ks:
                continue
            fd, fi = preds_dc[(m, r, ks[0])], preds_inv[(m, r, ks[0])]
            Wd, Wi = _pesos_union([fd, fi], B, seed)
            X = np.full((int(B), 1, len(ks)), np.nan)
            Xi = np.full((int(B), 1, len(ks)), np.nan)
            pd_, pi_ = [], []
            for j, s in enumerate(ks):
                a = S.align(fd, preds_dc[(m, r, s)])
                b = S.align(fi, preds_inv[(m, r, s)])
                ad = D._auc_pesos(a["label"].to_numpy(), a["prob"].to_numpy(), Wd)
                ai = D._auc_pesos(b["label"].to_numpy(), b["prob"].to_numpy(), Wi)
                X[:, 0, j], Xi[:, 0, j] = ai - ad, ai
                pd_.append(S.auc_roc(a["label"].to_numpy(), a["prob"].to_numpy()))
                pi_.append(S.auc_roc(b["label"].to_numpy(), b["prob"].to_numpy()))
            pres = np.ones((1, len(ks)), bool)
            hj = J.remuestreo_jerarquico(X, pres, seed_semillas, punt=(np.array(pi_) - np.array(pd_))[None])[:, 0]
            hi_ = J.remuestreo_jerarquico(Xi, pres, seed_semillas, punt=np.array(pi_)[None])[:, 0]
            lo, hi = J.ic(hj)
            l2, h2 = J.ic(hi_)
            filas.append({"metodo": m, "rho": r, "auc_invertido": float(np.mean(pi_)), "ic95_inf_invertido": l2,
                          "ic95_sup_invertido": h2, "auc_desconf": float(np.mean(pd_)),
                          "delta_invertido_menos_desconf": float(np.mean(pi_) - np.mean(pd_)), "ic95_inf": lo,
                          "ic95_sup": hi, "n_semillas": len(ks)})
    return pd.DataFrame(filas)


def auc_por_fuente(preds_dc, metodos=None, B=S.DEFAULT_B, seed=S.DEFAULT_BOOT_SEED,
                   seed_semillas=J.SEMILLA_SEMILLAS, nombres=("A (NIH)", "B (CheXpert)")) -> pd.DataFrame:
    """AUC media por semilla dentro de cada fuente de test_desconf (500 positivas y 500 negativas por fuente)."""
    filas = []
    fuentes = sorted({int(s) for f in preds_dc.values() for s in f["source"].unique()})
    for fu in fuentes:
        sub = {k: v[v["source"].astype(int) == fu].reset_index(drop=True) for k, v in preds_dc.items()}
        c, _, _, _ = curva_jerarquica(sub, metodos=metodos, B=B, seed=seed, seed_semillas=seed_semillas,
                                      diseno=D.RHOS_AREA)
        c.insert(1, "fuente", nombres[fu] if fu < len(nombres) else str(fu))
        filas.append(c)
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()


def _umbral_youden(df: pd.DataFrame) -> float:
    return float(S.youden_threshold(df["label"].to_numpy(), df["prob"].to_numpy()))


def brecha_especificidad(preds_dc, preds_umbral, metodos=None, fuente_mas_positivos: int = 0, B=S.DEFAULT_B,
                         seed=S.DEFAULT_BOOT_SEED, seed_semillas=J.SEMILLA_SEMILLAS) -> pd.DataFrame:
    """Brecha de especificidad en test_desconf al umbral de Youden de la validación con la distribución de
    entrenamiento de cada semilla (`preds_umbral`: val, o val_sel en DFR). Brecha = especificidad en la fuente con
    menos positivos en entrenamiento menos especificidad en la fuente con más positivos (orientación A: B - A); un
    valor positivo indica que el modelo penaliza a los negativos de la fuente asociada a la clase positiva."""
    metodos = list(metodos or sorted({k[0] for k in preds_dc}))
    otra = 1 - int(fuente_mas_positivos)
    filas = []
    for m in metodos:
        for r in sorted({float(k[1]) for k in preds_dc if k[0] == m}):
            ks = sorted(k[2] for k in preds_dc if k[0] == m and float(k[1]) == r and (m, r, k[2]) in preds_umbral)
            if not ks:
                continue
            ref = preds_dc[(m, r, ks[0])]
            W = D.pesos_remuestreo(ref["patient"].to_numpy(), B, seed)
            X = np.full((int(B), 1, len(ks)), np.nan)
            pv, thr = [], []
            for j, s in enumerate(ks):
                a = S.align(ref, preds_dc[(m, r, s)])
                t = _umbral_youden(preds_umbral[(m, r, s)])
                neg = a["label"].to_numpy() == 0
                src = a["source"].astype(int).to_numpy()
                ok = (a["prob"].to_numpy() < t).astype(float)
                ea = (W * (neg & (src == fuente_mas_positivos)) * ok).sum(1) / (W * (neg & (src == fuente_mas_positivos))).sum(1)
                eb = (W * (neg & (src == otra)) * ok).sum(1) / (W * (neg & (src == otra))).sum(1)
                X[:, 0, j] = eb - ea
                pa = ok[neg & (src == fuente_mas_positivos)].mean()
                pb = ok[neg & (src == otra)].mean()
                pv.append(pb - pa)
                thr.append(t)
            hj = J.remuestreo_jerarquico(X, np.ones((1, len(ks)), bool), seed_semillas, punt=np.array(pv)[None])[:, 0]
            lo, hi = J.ic(hj)
            filas.append({"metodo": m, "rho": r, "brecha_especificidad": float(np.mean(pv)), "ic95_inf": lo,
                          "ic95_sup": hi, "umbrales": " ".join("%.4f" % x for x in thr), "n_semillas": len(ks)})
    return pd.DataFrame(filas)


# ===================================================================================== V2.4.4
def orientacion_b(preds_a, preds_b, metodo: str = D.REFERENCIA, B=S.DEFAULT_B, seed=S.DEFAULT_BOOT_SEED,
                  seed_semillas=J.SEMILLA_SEMILLAS, umbral: float = UMBRAL_ORIENTACION) -> dict:
    """Diferencia de caídas D(A) - D(B) de `metodo` entre rho 0.5 y 1.0, con los mismos pacientes remuestreados (el
    test desconfundido es el mismo) y las semillas remuestreadas como conglomerados que atraviesan las dos
    orientaciones (la semilla s fija la misma submuestra en rho 0.5 en las dos)."""
    ref = _ref_frame(preds_a, metodo)
    sem = sorted({int(k[2]) for k in list(preds_a) + list(preds_b) if k[0] == metodo})
    filas = [("A", 0.5), ("A", 1.0), ("B", 0.5), ("B", 1.0)]
    R, K = len(filas), len(sem)
    X = np.full((int(B), R, K), np.nan)
    P = np.full((R, K), np.nan)
    pres = np.zeros((R, K), bool)
    for i, (o, r) in enumerate(filas):
        src = preds_a if o == "A" else preds_b
        pu, dr, pr = cubo(src, metodo, [r], sem, ref, B, seed)
        X[:, i], P[i], pres[i] = dr[:, 0], pu[0], pr[0]
    if not (pres[0] & pres[1]).any() or not (pres[2] & pres[3]).any():
        return {"veredicto": "no evaluable", "motivo": "faltan rho 0.5 o 1.0 en alguna orientación"}
    C = J.multiplicidades(pres, int(B), seed_semillas)
    m = J.extraer(X, P, pres, C)
    mp = _media_puntual(P, pres)
    dA, dB = mp[0] - mp[1], mp[2] - mp[3]
    dif = m[:, 0] - m[:, 1] - (m[:, 2] - m[:, 3])
    lo, hi = J.ic(dif)
    return {"metodo": metodo, "caida_A": float(dA), "caida_B": float(dB), "diferencia_A_menos_B": float(dA - dB),
            "ic95": [lo, hi], "umbral": umbral, "limitar_a_orientacion_A": bool(abs(dA - dB) > umbral),
            "semillas_A": [sem[j] for j in range(K) if pres[0, j] and pres[1, j]],
            "semillas_B": [sem[j] for j in range(K) if pres[2, j] and pres[3, j]], "B": int(B),
            "veredicto": "asimetría > %.2f: C2 se limita a la orientación A" % umbral if abs(dA - dB) > umbral
            else "sin asimetría apreciable"}


# ===================================================================================== C2-R
FAMILIAS_C2R = {"delta_auc_rho000": ("F-C2R", "secundaria")}


def informe_b2_tablas(preds_por_conjunto: Dict[str, Dict[tuple, pd.DataFrame]], B=S.DEFAULT_B,
                      seed=S.DEFAULT_BOOT_SEED, seed_semillas=J.SEMILLA_SEMILLAS, delta_c2: Optional[pd.DataFrame] = None):
    """Tablas de C2-R: por conjunto (fuente), AUC media por semilla y Δ frente a ERM con IC95 jerárquico (la unidad es
    la imagen deduplicada, que es lo que trae la columna patient en B2) y Holm por fuente (F-C2R-<fuente>, 11
    comparaciones). Con `delta_c2` (comparación de C2) añade la coherencia de dirección con Δ(0.99) y con el área."""
    curvas, comps = [], []
    for conj, preds in preds_por_conjunto.items():
        if not preds:
            continue
        c, _, k, _ = curva_jerarquica(preds, B=B, seed=seed, seed_semillas=seed_semillas, diseno=(0.0,))
        c.insert(0, "conjunto", conj)
        k = k[k["medida"] == "delta_auc_rho000"].copy()
        fam = {"delta_auc_rho000": ("F-C2R-" + conj.replace("test_", ""), "secundaria")}
        k = familias_c2(k, mapa=fam)
        k.insert(0, "conjunto", conj)
        curvas.append(c.drop(columns=["rho"]))
        comps.append(k.drop(columns=["rho"]))
    curva = pd.concat(curvas, ignore_index=True) if curvas else pd.DataFrame()
    comp = pd.concat(comps, ignore_index=True) if comps else pd.DataFrame()
    if delta_c2 is not None and len(comp):
        ref = {}
        for med in ("delta_auc_rho099", "delta_area_dosis"):
            ref[med] = delta_c2[delta_c2["medida"] == med].set_index("metodo")["delta"]
        for med, col in (("delta_auc_rho099", "coherente_con_c2_rho099"), ("delta_area_dosis", "coherente_con_c2_area")):
            comp[col] = [bool(np.sign(d) == np.sign(ref[med].get(m, np.nan))) if m in ref[med].index else None
                         for m, d in zip(comp["metodo"], comp["delta"])]
    return curva, comp


# ===================================================================================== árbol sintético de B2
def arbol_sintetico_b2(root, metodos=("erm", "groupdro", "dfr", "lisa"), semillas=(0, 1, 2), n=(60, 30), seed=0,
                       mejor: str = "groupdro", estados=("best", "bestdc")) -> Path:
    """Predicciones SINTÉTICAS de las corridas finales de B2 (ningún dato real): val, val_robust, test_shenzhen y
    test_montgomery; `mejor` tiene más señal en los tests."""
    import zlib
    root = Path(root)
    rng0 = np.random.default_rng(seed)
    pops = {}
    for conj, nn in (("val", n[0]), ("val_robust", n[0]), ("test_shenzhen", n[0]), ("test_montgomery", n[1])):
        k = 2 * nn
        ids = ["%s_%04d.png" % (conj, i) for i in range(k)]
        pops[conj] = pd.DataFrame({"image_id": ids, "label": (np.arange(k) < nn).astype(int),
                                   "source": 3 if conj == "test_shenzhen" else (4 if conj == "test_montgomery" else 2),
                                   "patient": ["%s:%s" % (conj, i) for i in ids]})
    for m in metodos:
        for s in semillas:
            rng = np.random.default_rng(zlib.crc32(repr((m, s, seed)).encode()))
            d = root / ("b2_%s" % m) / ("seed%d" % s) / "predictions"
            d.mkdir(parents=True, exist_ok=True)
            st0 = "dfr_" if m == "dfr" else ""
            for st in estados:
                for conj, pop in pops.items():
                    senal = 1.2 + (0.8 if (m == mejor and conj.startswith("test")) else 0.0) + rng.normal(0, 0.1)
                    z = senal * (2 * pop["label"].to_numpy() - 1) + rng.normal(0, 1.0, len(pop))
                    out = pop.copy()
                    out["prob"] = 1 / (1 + np.exp(-z))
                    out.to_csv(d / ("%s%s_%s.csv" % (st0, st, conj)), index=False, float_format="%.8f")
            (d.parent / "config_resolved.json").write_text(json.dumps({"sintetico": True, "seed": s}), encoding="utf-8")
    return root
