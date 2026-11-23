"""Complementos pedidos por la revisión, calculados con las funciones congeladas y el factor corregido de la ecuación
(17) de docs/08 (scripts/correccion/ec6.py). Código posterior a la apertura de los tests; sin entrenar nada y sin
modificar ningún fichero congelado. Todo camino que lee un test pasa por src.eval.desbloqueo.exigir_congelacion y se
pasa a las funciones de lectura el registro real que devuelve.

    python scripts/correccion/complementos.py cubos [--presupuesto-s 120] [--sin-cache]   # cubos de AUC en caché
    python scripts/correccion/complementos.py e1      # F-C1 con D_IN: intervalo de semillas y pacientes
    python scripts/correccion/complementos.py e2e3    # pérdida por selección en escala de área; Holm alternativos

E1. Referencia principal D_IN (fija) frente a cada uno de los doce métodos con sus cinco semillas, parte bimcv del
    test limpio de COVIDx: el procedimiento de `scripts/sensibilidad/intervalos.py c1b-jerarquico` con la referencia
    repetida en cada semilla (ecuación (9) de V2.2.1). Control: delta e IC95 de pacientes frente a
    results/c1/test/f_c1.csv. Se repite D_ERM con el factor corregido para comprobar que coincide con
    results/sensibilidad_intervalos/c1b_jerarquico.csv (un solo estrato). Lectura de sensibilidad: no sustituye a la
    prueba registrada de F-C1.
E2. Pérdida media por selección de época (ecuación (18), media de los once métodos) integrada con los trapecios de la
    ecuación (12) entre rho 0.5 y 0.99, con IC95 del remuestreo corregido; y, como descriptivo, el área restringida a
    rho 0.9 a 0.99 (normalizada por 0.09) de esa pérdida y de las diferencias de cada método frente a ERM.
E3. Valores p de Holm (0.025) con los tres intervalos alternativos de E1 (pacientes con modelos fijos; remuestreo
    doble, cuyo valor p no estaba registrado y se calcula aquí con la ecuación (3) solo como lectura; prueba t entre
    semillas) en F-C2-area y en F-C3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.correccion import ec6  # noqa: E402
from src.eval import c2 as C2  # noqa: E402
from src.eval import c3 as C3  # noqa: E402
from src.eval import dosis as D  # noqa: E402
from src.eval import jerarquico as J  # noqa: E402
from src.eval import stats as S  # noqa: E402

OUT = ROOT / "results" / "correccion_ec6" / "complementos"
CACHE = ROOT / "logs" / "correccion_ec6" / "cache"
TRAIN = ROOT / "results" / "train"
B, SEMILLA = S.DEFAULT_B, S.DEFAULT_BOOT_SEED
METODOS = sorted(m for m in D.METODOS_B3 if m != "control_fuente")     # once métodos de B3a
NIVELES, SEMILLAS = list(D.RHOS), list(range(10))
RHOS_FUERTE = (0.9, 0.95, 0.99)


class Presupuesto(Exception):
    pass


class Ctx:
    def __init__(self, motivo, presupuesto_s=0.0, usar_cache=True):
        from src.eval.desbloqueo import exigir_congelacion
        self.t0 = time.time()
        self.presupuesto = float(presupuesto_s or 0)
        self.usar_cache = usar_cache
        self.registro = exigir_congelacion("complementos de la corrección de la ecuación (17): " + motivo)
        self.sha = self.registro.get("sha256_congelacion")
        CACHE.mkdir(parents=True, exist_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)

    def verificador(self, motivo):
        return self.registro

    def vigilar(self):
        if self.presupuesto and time.time() - self.t0 > self.presupuesto:
            raise Presupuesto("presupuesto de %.0f s agotado" % self.presupuesto)


# ====================================================================================== cubos
def _ok(z, ctx):
    return int(z["B"]) == B and int(z["seed"]) == SEMILLA and str(z["sha"]) == ctx.sha


def cubo_b3(ctx, m):
    """Cubo de `m` en test_desconf, estado best (dfr_best en DFR), alineado a la referencia de
    `c2.curva_jerarquica` (la primera corrida ordenada de ERM), con todas las semillas 0 a 9."""
    f = CACHE / ("b3best_%s.npz" % m)
    if ctx.usar_cache and f.exists():
        z = np.load(f, allow_pickle=False)
        if _ok(z, ctx):
            return [float(x) for x in z["niveles"]], z["punt"], z["draws"], z["pres"]
    ctx.vigilar()
    preds, _, _ = C2.cargar_b3(TRAIN, "test_desconf", "best", metodos=sorted({"erm", m}), verificador=ctx.verificador)
    ref = C2._ref_frame(preds, "erm")
    sem = sorted({int(k[2]) for k in preds})
    assert sem == SEMILLAS, sem
    nv = sorted({float(k[1]) for k in preds if k[0] == m})
    punt, dr, pres = C2.cubo(preds, m, nv, sem, ref, B, SEMILLA)
    np.savez(f, punt=punt, draws=dr, pres=pres, niveles=np.array(nv), B=B, seed=SEMILLA, sha=ctx.sha)
    print("cubo b3 %s (%.0f s)" % (m, time.time() - ctx.t0), flush=True)
    return nv, punt, dr, pres


def cubo_c3(ctx, m):
    """Cubos best y bestdc (en DFR, los de su etapa ERM) alineados como `c3.seleccion_epoca` (primera clave ordenada
    de best de todos los métodos: coral, rho 0.5, semilla 0)."""
    f = CACHE / ("c3_%s.npz" % m)
    if ctx.usar_cache and f.exists():
        z = np.load(f, allow_pickle=False)
        if _ok(z, ctx):
            return tuple(z[k] for k in ("pb", "db", "rb", "pc", "dc", "rc"))
    ctx.vigilar()
    ref_m = METODOS[0]
    pb, pc = C3._estados_epoca(TRAIN, "b3a", "A", "test_desconf", sorted({ref_m, m}), ctx.verificador)
    assert sorted(pb)[0][0] == ref_m
    ref = pb[sorted(pb)[0]]
    cb = C2.cubo(pb, m, NIVELES, SEMILLAS, ref, B, SEMILLA)
    cc = C2.cubo(pc, m, NIVELES, SEMILLAS, ref, B, SEMILLA)
    np.savez(f, pb=cb[0], db=cb[1], rb=cb[2], pc=cc[0], dc=cc[1], rc=cc[2], B=B, seed=SEMILLA, sha=ctx.sha)
    print("cubo c3 %s (%.0f s)" % (m, time.time() - ctx.t0), flush=True)
    return cb + cc


def cubos(ctx):
    for m in METODOS:
        cubo_b3(ctx, m)
    for m in METODOS:
        cubo_c3(ctx, m)
    print("cubos completos (%.0f s)" % (time.time() - ctx.t0))


# ====================================================================================== utilidades
def area(rhos, v, diseno):
    r = [round(float(x), 4) for x in rhos]
    idx = [r.index(round(float(x), 4)) for x in diseno]
    return D.area_dosis(diseno, np.asarray(v)[..., idx])


def holm(ps):
    h = S.holm({i: p for i, p in enumerate(ps)}, alpha=0.025)
    return [h[i]["p_holm"] for i in range(len(ps))]


def _fila_ic(prefijo, v):
    lo, hi = J.ic(v)
    return {prefijo + "ic95_inf": lo, prefijo + "ic95_sup": hi}


# ====================================================================================== E2 y E3: F-C2-area
def area_frente_erm(ctx):
    """Réplica de la parte de comparación de `c2.curva_jerarquica` para el área (ecuación (12)) con los remuestreos
    corregido, original, doble y de pacientes, y el área restringida a rho 0.9 a 0.99."""
    cub = {m: cubo_b3(ctx, m) for m in METODOS}
    nvr, pr_, dr_r, presr = cub["erm"]
    filas = []
    for m in METODOS:
        if m == "erm":
            continue
        nvm, pm, dm, presm = cub[m]
        nv = [r for r in nvm if r in nvr]
        im = [nvm.index(r) for r in nv]
        ir = [nvr.index(r) for r in nv]
        pres = presm[im] & presr[ir]
        X = dm[:, im] - dr_r[:, ir]
        Xp = pm[im] - pr_[ir]
        draws = {}
        draws["original"] = J.remuestreo_jerarquico(X, pres, J.SEMILLA_SEMILLAS, punt=Xp)
        with ec6.parche("corregido"):
            draws["corregido"] = J.remuestreo_jerarquico(X, pres, J.SEMILLA_SEMILLAS, punt=Xp)
        draws["doble"] = J.remuestreo_jerarquico(X, pres, J.SEMILLA_SEMILLAS, modo="doble")
        draws["pacientes"] = J.media_semillas(X, pres)
        dif = C2._media_puntual(Xp, pres)
        comunes = pres.all(0)
        per_seed = D.medidas(nv, Xp[:, comunes].T, D.RHOS_AREA)["area_dosis"]
        t = J.t_pareada(per_seed)
        fila = {"metodo": m, "delta_area": float(D.medidas(nv, dif, D.RHOS_AREA)["area_dosis"]),
                "delta_area_fuerte_090_099": float(area(nv, dif, RHOS_FUERTE)),
                "t_n_semillas": int(comunes.sum()), "t_p_unilateral": t["t_p_unilateral"],
                "t_ic95_inf": t["t_ic95_inf"], "t_ic95_sup": t["t_ic95_sup"]}
        for k, h in draws.items():
            a = D.medidas(nv, h, D.RHOS_AREA)["area_dosis"]
            fila.update(_fila_ic("%s_" % k, a))
            fila["%s_p_unilateral" % k] = J.p_mayor(a)
            if k in ("corregido", "original"):
                fila.update(_fila_ic("%s_fuerte_" % k, area(nv, h, RHOS_FUERTE)))
        filas.append(fila)
    t = pd.DataFrame(filas)
    for k in ("corregido", "original", "doble", "pacientes"):
        t["%s_p_holm" % k] = holm(t["%s_p_unilateral" % k].tolist())
    t["t_p_holm"] = holm(t["t_p_unilateral"].tolist())
    return t


def control_area(t):
    """El área original y la corregida deben coincidir con los CSV de comparación con ERM."""
    filas = []
    for modo, f in (("original", ROOT / "results" / "b3" / "test_A_best" / "comparacion_erm.csv"),
                    ("corregido", ROOT / "results" / "correccion_ec6" / "b3" / "test_A_best" / "comparacion_erm.csv"),
                    ("pacientes", ROOT / "results" / "b3" / "test_A_best" / "comparacion_erm.csv"),
                    ("doble", ROOT / "results" / "b3" / "test_A_best" / "comparacion_erm.csv"),
                    ("t", ROOT / "results" / "b3" / "test_A_best" / "comparacion_erm.csv")):
        if not f.exists():
            continue
        c = pd.read_csv(f)
        c = c[c.medida == "delta_area_dosis"].set_index("metodo").loc[t.metodo]
        par = {"original": [("original_ic95_inf", "ic95_inf"), ("original_ic95_sup", "ic95_sup"),
                            ("original_p_unilateral", "p_unilateral"), ("original_p_holm", "p_holm"),
                            ("delta_area", "delta")],
               "corregido": [("corregido_ic95_inf", "ic95_inf"), ("corregido_ic95_sup", "ic95_sup"),
                             ("corregido_p_unilateral", "p_unilateral"), ("corregido_p_holm", "p_holm")],
               "pacientes": [("pacientes_ic95_inf", "ic95_inf_pacientes"), ("pacientes_ic95_sup", "ic95_sup_pacientes"),
                             ("pacientes_p_unilateral", "p_unilateral_pacientes")],
               "doble": [("doble_ic95_inf", "ic95_inf_doble"), ("doble_ic95_sup", "ic95_sup_doble")],
               "t": [("t_p_unilateral", "t_p_unilateral"), ("t_ic95_inf", "t_ic95_inf"), ("t_ic95_sup", "t_ic95_sup")]}[modo]
        for a, b in par:
            filas.append({"tabla": str(f.relative_to(ROOT)), "columna_mia": a, "columna": b,
                          "max_abs_dif": float(np.max(np.abs(t[a].to_numpy(float) - c[b].to_numpy(float))))})
    return pd.DataFrame(filas)


# ====================================================================================== E2 y E3: F-C3
def seleccion(ctx):
    """Réplica de `c3.seleccion_epoca` (media de los once métodos) con los remuestreos corregido, original, doble y de
    pacientes; área de la pérdida (0.5 a 0.99 y 0.9 a 0.99)."""
    cub = {m: cubo_c3(ctx, m) for m in METODOS}
    R, K, M = len(NIVELES), len(SEMILLAS), len(METODOS)
    X = np.full((B, R, K, M), np.nan)
    Xp = np.full((R, K, M), np.nan)
    pres = np.zeros((R, K, M), bool)
    for q, m in enumerate(METODOS):
        pb, db, rb, pc, dc, rc = cub[m]
        pr = rb & rc
        X[..., q] = np.where(pr[None], dc - db, np.nan)
        Xp[..., q] = np.where(pr, pc - pb, np.nan)
        pres[..., q] = pr
    pres_sem = pres.any(2)
    med = {}
    for modo in ("original", "corregido", "doble", "pacientes"):
        mb = np.full((B, R, M), np.nan)
        with ec6.parche("corregido" if modo == "corregido" else "original"):
            Cm = J.multiplicidades(pres_sem, B, J.SEMILLA_SEMILLAS)
            for q in range(M):
                if modo == "pacientes":
                    mb[:, :, q] = J.media_semillas(X[..., q], pres[..., q])
                else:
                    mb[:, :, q] = J.extraer(X[..., q], Xp[..., q], pres[..., q], Cm,
                                            modo="doble" if modo == "doble" else "corregido")
        med[modo] = np.nanmean(mb, axis=2)                                  # (B, R)
    punt = np.array([np.mean([Xp[i, pres[i, :, q], q].mean() for q in range(M) if pres[i, :, q].any()])
                     for i in range(R)])
    filas = []
    for i, r in enumerate(NIVELES):
        todas = pres[i].all(1)
        v = np.array([Xp[i, j, :].mean() for j in range(K) if todas[j]])
        t = J.t_pareada(v)
        fila = {"rho": r, "familia_f_c3": r in RHOS_FUERTE, "delta_sel_media_metodos": float(punt[i]),
                "t_n_semillas": int(todas.sum()), "t_p_unilateral": t["t_p_unilateral"]}
        for k, h in med.items():
            fila.update(_fila_ic("%s_" % k, h[:, i]))
            fila["%s_p_unilateral" % k] = J.p_mayor(h[:, i])
        filas.append(fila)
    niv = pd.DataFrame(filas)
    fam = niv.familia_f_c3.to_numpy()
    for k in ("original", "corregido", "doble", "pacientes", "t"):
        niv["%s_p_holm" % k] = np.nan
        niv.loc[fam, "%s_p_holm" % k] = holm(niv.loc[fam, "%s_p_unilateral" % k].tolist())
    # pérdida en escala de área
    areas = []
    for nombre, dis in (("area_050_099", D.RHOS_AREA), ("area_fuerte_090_099", RHOS_FUERTE)):
        fila = {"medida": "perdida_seleccion_" + nombre, "estimacion": float(area(NIVELES, punt, dis))}
        for k in ("corregido", "original"):
            fila.update(_fila_ic("%s_" % k, area(NIVELES, med[k], dis)))
        areas.append(fila)
    return niv, pd.DataFrame(areas)


def control_seleccion(niv):
    filas = []
    for modo, f in (("original", ROOT / "results" / "c2c3" / "c3-epoca.csv"),
                    ("corregido", ROOT / "results" / "correccion_ec6" / "c2c3" / "c3-epoca.csv")):
        if not f.exists():
            continue
        c = pd.read_csv(f).set_index("rho").loc[niv.rho]
        for a, b in (("delta_sel_media_metodos", "delta_sel_media_metodos"), ("%s_ic95_inf" % modo, "ic95_inf"),
                     ("%s_ic95_sup" % modo, "ic95_sup"), ("%s_p_unilateral" % modo, "p_unilateral"),
                     ("%s_p_holm" % modo, "p_holm")):
            x, y = niv[a].to_numpy(float), c[b].to_numpy(float)
            ok = np.isfinite(y)
            filas.append({"tabla": str(f.relative_to(ROOT)), "columna_mia": a, "columna": b,
                          "max_abs_dif": float(np.max(np.abs(x[ok] - y[ok]))),
                          "nan_distintos": int((np.isfinite(x) != np.isfinite(y)).sum())})
    alt = ROOT / "results" / "sensibilidad_intervalos" / "c3_epoca_media_alternativos.csv"
    c = pd.read_csv(alt).set_index("rho").loc[niv.rho]
    for a, b in (("pacientes_ic95_inf", "ic95_inf_pacientes"), ("pacientes_p_unilateral", "p_unilateral_pacientes"),
                 ("doble_ic95_inf", "ic95_inf_doble"), ("doble_ic95_sup", "ic95_sup_doble"),
                 ("t_p_unilateral", "t_p_unilateral"), ("pacientes_p_holm", "p_holm_pacientes"), ("t_p_holm", "p_holm_t")):
        x, y = niv[a].to_numpy(float), c[b].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(x)
        filas.append({"tabla": str(alt.relative_to(ROOT)), "columna_mia": a, "columna": b,
                      "max_abs_dif": float(np.max(np.abs(x[ok] - y[ok]))), "nan_distintos": int((np.isfinite(x) != np.isfinite(y)).sum())})
    return pd.DataFrame(filas)


def e2e3(ctx):
    ta = area_frente_erm(ctx)
    ca = control_area(ta)
    niv, areas = seleccion(ctx)
    cs = control_seleccion(niv)
    # E2: comparación en la misma escala
    comp = ta[["metodo", "delta_area", "corregido_ic95_inf", "corregido_ic95_sup", "delta_area_fuerte_090_099",
               "corregido_fuerte_ic95_inf", "corregido_fuerte_ic95_sup"]].copy()
    ta.to_csv(OUT / "e3_area_alternativos.csv", index=False)
    niv.to_csv(OUT / "e3_seleccion_alternativos.csv", index=False)
    areas.to_csv(OUT / "e2_perdida_seleccion_area.csv", index=False)
    comp.to_csv(OUT / "e2_diferencias_area_frente_erm.csv", index=False)
    ctrl = pd.concat([ca, cs], ignore_index=True)
    ctrl.to_csv(OUT / "e2e3_control.csv", index=False)
    md = ["# Complementos E2 y E3", "",
          "B = %d, semilla de pacientes %d, semilla de semillas %d; factor corregido de la ecuación (17) de docs/08. "
          "Congelación: %s." % (B, SEMILLA, J.SEMILLA_SEMILLAS, ctx.sha), "",
          "## E2. Pérdida por selección de época en la escala del área", "", D.a_markdown(areas, 4),
          "## E2. Diferencias de área frente a ERM (0.5 a 0.99 y 0.9 a 0.99), IC95 corregido", "", D.a_markdown(comp, 4),
          "## E3. F-C2-area con los intervalos alternativos (p de Holm al 0.025)", "",
          D.a_markdown(ta[["metodo", "delta_area", "corregido_p_unilateral", "corregido_p_holm", "pacientes_p_unilateral",
                           "pacientes_p_holm", "doble_p_unilateral", "doble_p_holm", "t_p_unilateral", "t_p_holm"]], 4),
          "## E3. F-C3 con los intervalos alternativos", "",
          D.a_markdown(niv[["rho", "familia_f_c3", "delta_sel_media_metodos", "corregido_p_unilateral", "corregido_p_holm",
                            "pacientes_p_unilateral", "pacientes_p_holm", "doble_p_unilateral", "doble_p_holm",
                            "t_n_semillas", "t_p_unilateral", "t_p_holm"]], 4),
          "## Control", "", D.a_markdown(ctrl, 4)]
    (OUT / "e2e3.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(ctrl.to_string())
    return 0 if ctrl["max_abs_dif"].max() <= 1e-9 else 1


# ====================================================================================== E1
CACHE_V28 = ROOT / "logs" / "sensibilidad_v28" / "cache"


def e1(ctx):
    from src.eval import audit as A
    from src.eval.c1_auditoria import Auditoria, Rutas, leer_lista
    L = leer_lista(ROOT / "configs" / "gate7_lista_c1.yaml")
    aud = Auditoria("test", L, Rutas(), registro=ctx.registro)
    semillas = [int(x) for x in L["semillas"]]
    mets, ids0, base = {}, None, None
    for m in L["metodos"]:
        fr = aud.preds(m, "test_bimcv", ("dfr_best" if m == "dfr" else "best"))
        if len(fr) < len(semillas):
            raise SystemExit("%s: %d de %d semillas" % (m, len(fr), len(semillas)))
        df, P, sem = aud.matriz(fr)
        if ids0 is None:
            ids0, base = df["image_id"].tolist(), df
        else:
            P = np.vstack([A.alinear(ids0, df["image_id"], q) for q in P])
        mets[m] = (P, [int(x) for x in sem])
    y = base["label"].to_numpy().astype(int)
    g = base["patient"].to_numpy()
    f = CACHE_V28 / ("punt_%s_C0p05_pooled_espejo.csv" % L["troncos"]["D_IN"])
    pr = pd.read_csv(f, dtype={"image_id": str, "patient_id": str})
    ref = A.alinear(ids0, pr["image_id"], pr["score"].to_numpy())
    Dr = D.auc_remuestreo(y, ref, g, B, SEMILLA)[:, 0]
    a_ref = S.auc_roc(y, ref)
    pres = np.ones((1, len(semillas)), bool)
    pub = pd.read_csv(ROOT / "results" / "c1" / "test" / "f_c1.csv").set_index("metodo")
    filas = []
    for m, (P, sem) in mets.items():
        ctx.vigilar()
        P = P[[sem.index(sd) for sd in semillas]]
        Dm = D.auc_remuestreo(y, P, g, B, SEMILLA)
        a_m = np.array([S.auc_roc(y, P[k]) for k in range(len(semillas))])
        X = (Dr[:, None] - Dm)[:, None, :]
        Xp = (a_ref - a_m)[None, :]
        hj = J.remuestreo_jerarquico(X, pres, J.SEMILLA_SEMILLAS, punt=Xp)[:, 0]
        with ec6.parche("corregido"):
            hjc = J.remuestreo_jerarquico(X, pres, J.SEMILLA_SEMILLAS, punt=Xp)[:, 0]
        hp = np.nanmean(Dr[:, None] - Dm, axis=1)
        comp = J.componentes(X, Xp, pres)
        lo, hi = J.ic(hj)
        lp, hp_ = D._ic(hp)
        fila = {"metodo": m, "n_semillas": len(semillas), "auc_referencia": float(a_ref), "auc_metodo": float(a_m.mean()),
                "delta": float(Xp.mean()), "ic95_inf_jerarquico": lo, "ic95_sup_jerarquico": hi,
                "ic95_inf_pacientes": lp, "ic95_sup_pacientes": hp_,
                "max_abs_dif_corregido_original": float(np.max(np.abs(hjc - hj))),
                "de_semillas_sigma_S": float(np.sqrt(comp["sigma2_S"][0])), "de_pacientes_V_P": float(np.sqrt(comp["V_P"][0])),
                "de_entre_semillas_s": float(np.sqrt(comp["s2"][0])), "lambda": float(comp["lambda"][0]),
                "p_ni_0.02_jerarquico": J.p_mayor(hj, -0.02), "p_ni_0.01_jerarquico": J.p_mayor(hj, -0.01),
                "excluye_cero_jerarquico": bool(lo > 0 or hi < 0),
                "li_jerarquico_sobre_menos_0.02": bool(lo > -0.02), "li_jerarquico_sobre_menos_0.01": bool(lo > -0.01),
                "delta_publicado": float(pub.loc[m, "delta"]), "ic95_inf_publicado": float(pub.loc[m, "ic95_inf"]),
                "ic95_sup_publicado": float(pub.loc[m, "ic95_sup"]),
                "no_inferior_0.02_publicado": bool(pub.loc[m, "no_inferior_0.02"]),
                "no_inferior_0.01_publicado": bool(pub.loc[m, "no_inferior_0.01"])}
        filas.append(fila)
        print("%s: %.4f (%.4f a %.4f)" % (m, fila["delta"], lo, hi), flush=True)
    t = pd.DataFrame(filas)
    for mg in ("0.02", "0.01"):
        t["p_holm_%s_jerarquico" % mg] = holm(t["p_ni_%s_jerarquico" % mg].tolist())
        t["no_inferior_%s_jerarquico_holm" % mg] = t["p_holm_%s_jerarquico" % mg] <= 0.025
    ctrl = pd.DataFrame([{"tabla": "results/c1/test/f_c1.csv", "columna": c,
                          "max_abs_dif": float(np.max(np.abs(t[a] - t[b])))}
                         for c, a, b in (("delta", "delta", "delta_publicado"),
                                         ("ic95_inf", "ic95_inf_pacientes", "ic95_inf_publicado"),
                                         ("ic95_sup", "ic95_sup_pacientes", "ic95_sup_publicado"))])
    ctrl = pd.concat([ctrl, pd.DataFrame([{"tabla": "corregido frente a original (un estrato)", "columna": "remuestreos",
                                           "max_abs_dif": float(t["max_abs_dif_corregido_original"].max())}])])
    res = pd.DataFrame([{"lectura": "IC95 jerárquico con extremo inferior > -0.02", "metodos": int(t["li_jerarquico_sobre_menos_0.02"].sum())},
                        {"lectura": "IC95 jerárquico con extremo inferior > -0.01", "metodos": int(t["li_jerarquico_sobre_menos_0.01"].sum())},
                        {"lectura": "IC95 jerárquico que excluye el cero", "metodos": int(t["excluye_cero_jerarquico"].sum())},
                        {"lectura": "IC95 jerárquico entero por encima de cero (referencia mejor)", "metodos": int((t.ic95_inf_jerarquico > 0).sum())},
                        {"lectura": "IC95 jerárquico entero por debajo de cero (método mejor)", "metodos": int((t.ic95_sup_jerarquico < 0).sum())},
                        {"lectura": "IC95 de pacientes (publicado) que excluye el cero",
                         "metodos": int(((t.ic95_inf_publicado > 0) | (t.ic95_sup_publicado < 0)).sum())},
                        {"lectura": "no inferior 0.02 con Holm, jerárquico (sensibilidad)", "metodos": int(t["no_inferior_0.02_jerarquico_holm"].sum())},
                        {"lectura": "no inferior 0.01 con Holm, jerárquico (sensibilidad)", "metodos": int(t["no_inferior_0.01_jerarquico_holm"].sum())},
                        {"lectura": "no inferior 0.02 con Holm, publicado", "metodos": int(t["no_inferior_0.02_publicado"].sum())},
                        {"lectura": "no inferior 0.01 con Holm, publicado", "metodos": int(t["no_inferior_0.01_publicado"].sum())}])
    # D_ERM (F-C1b, referencia por semilla): con el factor corregido, el IC95 jerárquico coincide con el publicado
    pub_b = pd.read_csv(ROOT / "results" / "sensibilidad_intervalos" / "c1b_jerarquico.csv").set_index("metodo")
    R = []
    for sd in semillas:
        fr = CACHE_V28 / ("punt_%s_C0p05_pooled_espejo.csv" % L["troncos"]["D_ERM"].format(semilla=sd))
        q = pd.read_csv(fr, dtype={"image_id": str, "patient_id": str})
        R.append(A.alinear(ids0, q["image_id"], q["score"].to_numpy()))
    R = np.vstack(R)
    DrK = D.auc_remuestreo(y, R, g, B, SEMILLA)
    a_rK = np.array([S.auc_roc(y, R[k]) for k in range(len(semillas))])
    dmax = 0.0
    for m, (P, sem) in mets.items():
        ctx.vigilar()
        P = P[[sem.index(sd) for sd in semillas]]
        Dm = D.auc_remuestreo(y, P, g, B, SEMILLA)
        a_m = np.array([S.auc_roc(y, P[k]) for k in range(len(semillas))])
        with ec6.parche("corregido"):
            h = J.remuestreo_jerarquico((DrK - Dm)[:, None, :], pres, J.SEMILLA_SEMILLAS, punt=(a_rK - a_m)[None])[:, 0]
        lo, hi = J.ic(h)
        dmax = max(dmax, abs(lo - pub_b.loc[m, "ic95_inf_jerarquico"]), abs(hi - pub_b.loc[m, "ic95_sup_jerarquico"]))
    ctrl = pd.concat([ctrl, pd.DataFrame([{"tabla": "results/sensibilidad_intervalos/c1b_jerarquico.csv (factor corregido)",
                                           "columna": "ic95_jerarquico", "max_abs_dif": float(dmax)}])])
    t.to_csv(OUT / "e1_c1_din_jerarquico.csv", index=False)
    ctrl.to_csv(OUT / "e1_control.csv", index=False)
    res.to_csv(OUT / "e1_resumen.csv", index=False)
    md = ["# E1. F-C1 (D_IN frente a los doce métodos, parte bimcv del test) con IC95 de semillas y pacientes", "",
          "Lectura de sensibilidad; la prueba registrada es la de results/c1/test/f_c1.csv. Congelación: %s." % ctx.sha, "",
          D.a_markdown(t[["metodo", "delta", "ic95_inf_jerarquico", "ic95_sup_jerarquico", "ic95_inf_pacientes",
                          "ic95_sup_pacientes", "de_semillas_sigma_S", "de_pacientes_V_P", "p_holm_0.02_jerarquico",
                          "p_holm_0.01_jerarquico"]], 4), "## Resumen", "", D.a_markdown(res), "## Control", "",
          D.a_markdown(ctrl, 4)]
    (OUT / "e1.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(ctrl.to_string())
    print(res.to_string())
    return 0 if ctrl["max_abs_dif"].max() <= 1e-12 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("orden", choices=["cubos", "e1", "e2e3", "todo"])
    ap.add_argument("--presupuesto-s", type=float, default=0.0)
    ap.add_argument("--sin-cache", action="store_true")
    a = ap.parse_args(argv)
    ctx = Ctx(a.orden, a.presupuesto_s, not a.sin_cache)
    try:
        if a.orden == "cubos":
            cubos(ctx)
            return 0
        if a.orden == "e1":
            return e1(ctx)
        if a.orden == "e2e3":
            return e2e3(ctx)
        r1 = e1(ctx)
        r2 = e2e3(ctx)
        return max(r1, r2)
    except Presupuesto as e:
        print("%s: lo calculado queda en %s; repita la orden" % (e, CACHE))
        return 3


if __name__ == "__main__":
    sys.exit(main())
