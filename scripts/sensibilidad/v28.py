"""Análisis de sensibilidad registrados de C1 (docs/08_evaluacion.md, V2.8, y enmiendas E3, E4 y E5), una orden por
tabla. Código posterior a la congelación del protocolo para análisis registrados antes de ella; reutiliza las
funciones congeladas de `src.eval` (ver scripts/sensibilidad/comun.py). Salidas en results/sensibilidad_v28/.

    python scripts/sensibilidad/v28.py s0    # equivalencia con results/c1/test (f_c1, f_c1b, detectores)
    python scripts/sensibilidad/v28.py s1    # punto 1: detector con C = 0.01 y 0.3 y con la penúltima capa
    python scripts/sensibilidad/v28.py s2    # punto 2: D_ERM y D_RES (ya calculado; extracto de results/c1/test)
    python scripts/sensibilidad/v28.py s3    # punto 3: selección con val_desconf y DFR-dc (extracto de results/b3)
    python scripts/sensibilidad/v28.py s4    # punto 4: estados ema y last en COVIDx (enmienda E4)
    python scripts/sensibilidad/v28.py s5    # punto 5: conjunto de cinco semillas
    python scripts/sensibilidad/v28.py s6    # punto 6: remuestreo por imagen frente a por paciente
    python scripts/sensibilidad/v28.py s7    # punto 7: test limpio completo
    python scripts/sensibilidad/v28.py e3    # enmienda E3: todo sin promediado de espejo
    python scripts/sensibilidad/v28.py todo  # todas, en este orden, y el resumen
    python scripts/sensibilidad/v28.py resumen  # resumen de familias desde las tablas escritas (no lee el test)

Opciones: --presupuesto-s N (detiene el cálculo a los N segundos con código 3; lo hecho queda en
logs/sensibilidad_v28/cache/ y basta con repetir la orden), --sin-cache (recalcula todo).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comun as K  # noqa: E402

ORDEN = "python scripts/sensibilidad/v28.py %s"
VARIANTES = {"base": (0.05, "pooled", "espejo"), "C0.01": (0.01, "pooled", "espejo"), "C0.3": (0.3, "pooled", "espejo"),
             "penult": (0.05, "penult", "espejo"), "sin_espejo": (0.05, "pooled", "no")}


def _detectores(ctx, variantes, conjunto="test_bimcv", unidad="paciente"):
    """AUC con IC95 de cada tronco y variante; también AP y PA en la parte bimcv."""
    filas = []
    for var in variantes:
        C, feat, tta = VARIANTES[var]
        for nombre, tags in ctx.troncos().items():
            for tag in tags:
                p = ctx.puntuaciones(tag, C, feat, tta)
                if p is None:
                    filas.append({"variante": var, "detector": nombre, "tag": tag, "C": C, "caracteristicas": feat,
                                  "tta": tta, "nota": "el tronco no tiene esta capa (sin cabeza entrenada)"})
                    continue
                k = p["test_bimcv"].to_numpy(bool) if conjunto == "test_bimcv" else np.ones(len(p), bool)
                q = p[k]
                g = q["patient_id"].to_numpy() if unidad == "paciente" else q["image_id"].to_numpy()
                r = K.auc_ic(ctx, q["label"], q["score"], g)
                f = {"variante": var, "detector": nombre, "tag": tag, "C": C, "caracteristicas": feat, "tta": tta,
                     "conjunto": conjunto, "unidad": unidad, **r}
                if conjunto == "test_bimcv" and unidad == "paciente":
                    for pr in ("AP", "PA"):
                        mm = (q["proyeccion"] == pr).to_numpy()
                        rr = K.auc_ic(ctx, q.loc[mm, "label"], q.loc[mm, "score"], q.loc[mm, "patient_id"])
                        f.update({"auc_%s" % pr: rr["auc"], "ic95_inf_%s" % pr: rr["ic95_inf"], "ic95_sup_%s" % pr: rr["ic95_sup"]})
                filas.append(f)
    return pd.DataFrame(filas)


def _resumen_det(t):
    """Media y recorrido entre troncos de cada detector y variante."""
    t = t[t["auc"].notna()] if "auc" in t else t
    return (t.groupby(["variante", "detector"], sort=False)
             .agg(n_troncos=("auc", "size"), auc_media=("auc", "mean"), auc_min=("auc", "min"), auc_max=("auc", "max"),
                  ic95_inf_min=("ic95_inf", "min"), ic95_sup_max=("ic95_sup", "max")).reset_index())


def _refs(ctx, var, nombre="D_ERM"):
    C, feat, tta = VARIANTES[var]
    tags = ctx.troncos()[nombre]
    out = {}
    for s, tag in zip(ctx.semillas, tags):
        p = ctx.puntuaciones(tag, C, feat, tta)
        if p is not None:
            out[s] = p
    return out


def _corta(t, ref, variante, extra=None):
    t = t.copy()
    t.insert(0, "variante", variante)
    t.insert(0, "referencia", ref)
    if extra:
        for k, v in extra.items():
            t.insert(2, k, v)
    return t


# ====================================================================================== s0
def s0(ctx):
    mets = ctx.metodos_en("test_bimcv")
    din = ctx.puntuaciones(ctx.L["troncos"]["D_IN"], *VARIANTES["base"])
    a = K.f_c1(ctx, mets, din)
    b = K.f_c1b(ctx, mets, _refs(ctx, "base"))
    det = _detectores(ctx, ["base"])
    fr = pd.read_csv(K.ROOT / "results/c1/test/f_c1.csv")
    frb = pd.read_csv(K.ROOT / "results/c1/test/f_c1b.csv")
    fd = pd.read_csv(K.ROOT / "results/c1/test/detectores.csv")
    filas = []
    cols = ["auc_referencia", "auc_metodo", "delta", "ic95_inf", "ic95_sup", "p_superioridad", "p_ni_0.02", "p_ni_0.01",
            "p_holm_0.02", "p_holm_0.01"]
    for nom, mio, ref in (("f_c1", a, fr), ("f_c1b", b, frb)):
        x = mio.set_index("metodo")
        z = ref.set_index("metodo")
        for c in cols:
            d = np.abs(x.loc[z.index, c].to_numpy(float) - z[c].to_numpy(float))
            filas.append({"tabla": nom, "columna": c, "max_abs_dif": float(d.max())})
        for c in ("no_inferior_0.02", "no_inferior_0.01"):
            filas.append({"tabla": nom, "columna": c, "max_abs_dif": float((x.loc[z.index, c].astype(bool).to_numpy()
                                                                              != z[c].astype(bool).to_numpy()).sum())})
    x = det.set_index("tag")
    z = fd.set_index("tag")
    for c in ("auc", "ic95_inf", "ic95_sup", "auc_AP", "auc_PA", "ic95_inf_AP", "ic95_sup_PA"):
        d = np.abs(x.loc[z.index, c].to_numpy(float) - z[c].to_numpy(float))
        filas.append({"tabla": "detectores", "columna": c, "max_abs_dif": float(d.max())})
    t = pd.DataFrame(filas)
    t["identico"] = t["max_abs_dif"] <= 1e-12
    # Tolerancia de entorno: el ajuste lbfgs del detector depende de la versión de numpy y de BLAS (VM: numpy 2.2.6;
    # contenedor: 1.26.4; docs/14, Section 5). Cambia la AUC de algunos troncos en menos de 1e-4 y, en los valores p,
    # a lo sumo un remuestreo (1/2001); las decisiones deben coincidir siempre.
    dec = t["columna"].str.startswith("no_inferior")
    pv = t["columna"].str.startswith("p_")
    t["dentro_de_tolerancia"] = np.where(dec, t["max_abs_dif"] == 0,
                                         np.where(pv, t["max_abs_dif"] <= 2.0 / 2001, t["max_abs_dif"] <= 1e-4))
    import platform
    import numpy
    K.escribir("s0_equivalencia", {"s0_equivalencia": t}, ctx, ORDEN % "s0",
               ["Compara el cálculo con caché de comun.py con las tablas congeladas de results/c1/test/.",
                "Entorno de esta ejecución: Python %s, numpy %s, máquina %s." % (platform.python_version(),
                                                                               numpy.__version__, platform.node())])
    if not t["dentro_de_tolerancia"].all():
        print("AVISO: hay diferencias con results/c1/test fuera de tolerancia")
        return 1
    print("equivalencia: %d de %d columnas idénticas; el resto dentro de tolerancia" % (t["identico"].sum(), len(t)))
    return 0


# ====================================================================================== s1
def s1(ctx):
    det = _detectores(ctx, ["base", "C0.01", "C0.3", "penult"])
    mets = ctx.metodos_en("test_bimcv")
    fams = []
    din_tag = ctx.L["troncos"]["D_IN"]
    for var in ("C0.01", "C0.3"):
        fams.append(_corta(K.f_c1(ctx, mets, ctx.puntuaciones(din_tag, *VARIANTES[var])), "D_IN", var))
    for var in ("C0.01", "C0.3", "penult"):
        fams.append(_corta(K.f_c1b(ctx, mets, _refs(ctx, var)), "D_ERM", var))
    notas = ["Variantes: C = 0.01 y C = 0.3 sobre las características agrupadas (768), y C = 0.05 sobre la penúltima "
             "capa (128); en todas, el mismo ajuste (6 000 positivos de bimcv de entrenamiento) y promediado de espejo.",
             "D_IN no tiene penúltima capa entrenada (tronco de ImageNet sin cabeza): la variante penúltima se aplica a "
             "D_ERM, D_MASK y D_RES, y la familia de esa variante es F-C1b (D_ERM).",
             "La variante base reproduce results/c1/test/detectores.csv, f_c1.csv y f_c1b.csv (orden s0)."]
    K.escribir("s1_detector", {"s1_detectores": det, "s1_detectores_resumen": _resumen_det(det),
                               "s1_familias": pd.concat(fams, ignore_index=True)}, ctx, ORDEN % "s1", notas)
    return 0


# ====================================================================================== s2
def s2(ctx):
    fd = pd.read_csv(K.ROOT / "results/c1/test/detectores.csv")
    frb = pd.read_csv(K.ROOT / "results/c1/test/f_c1b.csv")
    res = _resumen_det(fd.assign(variante="base"))
    cols = ["metodo", "auc_referencia", "auc_metodo", "delta", "ic95_inf", "ic95_sup", "p_holm_0.02", "no_inferior_0.02",
            "p_holm_0.01", "no_inferior_0.01"]
    K.escribir("s2_referencias", {"s2_detectores_resumen": res, "s2_f_c1b": frb[cols]}, ctx, ORDEN % "s2",
               ["Extracto sin cálculo nuevo de results/c1/test/detectores.csv y f_c1b.csv "
                "(python scripts/eval/auditoria_covidx.py --c1 test)."])
    return 0


# ====================================================================================== s3
def s3(ctx):
    rb = K.ROOT / "results/b3"
    cb, cd = pd.read_csv(rb / "test_A_best/curva.csv"), pd.read_csv(rb / "test_A_bestdc/curva.csv")
    x = cb[["metodo", "rho", "auc_media_semillas"]].merge(cd[["metodo", "rho", "auc_media_semillas", "ic95_inf", "ic95_sup",
                                                              "n_semillas"]], on=["metodo", "rho"], how="outer",
                                                             suffixes=("_best", "_bestdc"))
    x["bestdc_menos_best"] = x["auc_media_semillas_bestdc"] - x["auc_media_semillas_best"]
    comp_b, comp_d = pd.read_csv(rb / "test_A_best/comparacion_erm.csv"), pd.read_csv(rb / "test_A_bestdc/comparacion_erm.csv")
    k = ["metodo", "medida", "delta", "ic95_inf", "ic95_sup", "p_holm", "rechaza_holm", "familia"]
    y = comp_b[k].merge(comp_d[k], on=["metodo", "medida"], how="outer", suffixes=("_best", "_bestdc"))
    y = y[y["medida"].isin(["delta_area_dosis", "delta_auc_rho090", "delta_auc_rho095", "delta_auc_rho099",
                            "delta_auc_rho100", "coste_rho050_no_inferioridad"])]
    K.escribir("s3_bestdc", {"s3_curva_best_bestdc": x, "s3_comparacion_erm_best_bestdc": y}, ctx, ORDEN % "s3",
               ["Extracto sin cálculo nuevo de results/b3/test_A_best y results/b3/test_A_bestdc "
                "(scripts/eval/informe_b3.py con --estado best y bestdc).",
                "En DFR, el estado bestdc es dfr_bestdc: parte de la época elegida en val_desconf y reajusta la última "
                "capa con una mitad de val_desconf (DFR-dc de V2.4.6), definido también en rho = 1.0.",
                "Las familias de la tabla bestdc no son confirmatorias (V2.4.5): la selección con oráculo va aparte."])
    return 0


# ====================================================================================== s4
def s4(ctx):
    din = ctx.puntuaciones(ctx.L["troncos"]["D_IN"], *VARIANTES["base"])
    refs = _refs(ctx, "base")
    aucs, fams = [], []
    for st in ("best", "ema", "last"):
        mets = ctx.metodos_en("test_bimcv", st)
        base = mets["erm"][0]
        for m, (df, P, sem) in mets.items():
            r = K.auc_media(ctx, base["label"], P, base["patient"])
            aucs.append({"estado": st, "metodo": m, "fichero_estado": ctx.estado(m, st), **r})
        if st != "best":
            fams.append(_corta(K.f_c1(ctx, mets, din), "D_IN", st))
            fams.append(_corta(K.f_c1b(ctx, mets, refs), "D_ERM", st))
    t = pd.DataFrame(aucs)
    w = t.pivot(index="metodo", columns="estado", values="auc").reset_index()
    w["ema_menos_best"], w["last_menos_best"] = w["ema"] - w["best"], w["last"] - w["best"]
    K.escribir("s4_estados", {"s4_auc_por_estado": t, "s4_auc_resumen": w, "s4_familias": pd.concat(fams, ignore_index=True)},
               ctx, ORDEN % "s4",
               ["Predicciones del test escritas por evaluar_test_b1.py --modo test (results/b1/test/), estados ema y "
                "last; en DFR, dfr_ema y dfr_last, que las corridas conservan, de modo que DFR entra en esta sensibilidad.",
                "Las referencias no cambian: D_IN no tiene estados y D_ERM usa los troncos best de ERM (las "
                "características solo se extrajeron de ese estado). F-C1 y F-C1b se recalculan como sensibilidad, sin "
                "familia propia (enmienda E4)."])
    return 0


# ====================================================================================== s5
def s5(ctx):
    din = ctx.puntuaciones(ctx.L["troncos"]["D_IN"], *VARIANTES["base"])
    mets = ctx.metodos_en("test_bimcv")
    base = mets["erm"][0]
    ids, y, g = base["image_id"].tolist(), base["label"].to_numpy(), base["patient"].to_numpy()
    filas = []
    ens = {}
    for m, (df, P, sem) in mets.items():
        pe = P.mean(0)
        ens[m] = pe[None]
        ind = K.auc_media(ctx, y, P, g)
        en = K.auc_ic(ctx, y, pe, g)
        filas.append({"metodo": m, "auc_individual_media": ind["auc"], "ic95_inf_individual": ind["ic95_inf"],
                      "ic95_sup_individual": ind["ic95_sup"], "auc_conjunto": en["auc"], "ic95_inf_conjunto": en["ic95_inf"],
                      "ic95_sup_conjunto": en["ic95_sup"], "conjunto_menos_individual": en["auc"] - ind["auc"]})
    ref = K.ref_alineada(din, ids)[None]
    fam = K.familia(ctx, y, g, ref, ens, {m: ["conjunto"] for m in ens})
    K.escribir("s5_conjunto", {"s5_auc_conjunto": pd.DataFrame(filas), "s5_d_in_frente_conjunto": _corta(fam, "D_IN", "conjunto")},
               ctx, ORDEN % "s5",
               ["Conjunto: media de las probabilidades de las cinco semillas (42, 123, 2024, 7, 999) del estado principal.",
                "La comparación con D_IN repite la prueba de F-C1 con el conjunto en lugar de la media por semilla; es "
                "una sensibilidad sin familia propia y no sustituye a la cifra principal (V2.2.1, V2.9 punto 5).",
                "No se forma un conjunto de D_ERM: sus puntuaciones son funciones de decisión de cinco detectores con "
                "escalas distintas y el registro no lo prevé."])
    return 0


# ====================================================================================== s6
def s6(ctx):
    din = ctx.puntuaciones(ctx.L["troncos"]["D_IN"], *VARIANTES["base"])
    refs = _refs(ctx, "base")
    mets = ctx.metodos_en("test_bimcv")
    fams = []
    for u in ("paciente", "imagen"):
        fams.append(_corta(K.f_c1(ctx, mets, din, unidad=u), "D_IN", "base", {"unidad": u}))
        fams.append(_corta(K.f_c1b(ctx, mets, refs, unidad=u), "D_ERM", "base", {"unidad": u}))
    t = pd.concat(fams, ignore_index=True)
    t["semianchura"] = (t["ic95_sup"] - t["ic95_inf"]) / 2
    det = pd.concat([_detectores(ctx, ["base"], unidad="paciente"), _detectores(ctx, ["base"], unidad="imagen")], ignore_index=True)
    det = det[["detector", "tag", "unidad", "auc", "ic95_inf", "ic95_sup", "n", "n_unidades"]]
    a = t[t.unidad == "paciente"].set_index(["referencia", "metodo"])
    b = t[t.unidad == "imagen"].set_index(["referencia", "metodo"])
    cmp_ = pd.DataFrame({"delta": a["delta"], "ic95_inf_paciente": a["ic95_inf"], "ic95_sup_paciente": a["ic95_sup"],
                         "ic95_inf_imagen": b["ic95_inf"], "ic95_sup_imagen": b["ic95_sup"],
                         "cociente_semianchuras_imagen_paciente": b["semianchura"] / a["semianchura"],
                         "no_inferior_0.02_paciente": a["no_inferior_0.02"], "no_inferior_0.02_imagen": b["no_inferior_0.02"],
                         "no_inferior_0.01_paciente": a["no_inferior_0.01"], "no_inferior_0.01_imagen": b["no_inferior_0.01"]}).reset_index()
    K.escribir("s6_unidad", {"s6_familias": t, "s6_comparacion": cmp_, "s6_detectores": det}, ctx, ORDEN % "s6",
               ["Parte bimcv del test: 8 079 imágenes de 4 343 pacientes. La unidad 'imagen' remuestrea cada imagen como "
                "independiente con el mismo generador (pesos_remuestreo con image_id como grupo)."])
    return 0


# ====================================================================================== s7
def s7(ctx):
    din = ctx.puntuaciones(ctx.L["troncos"]["D_IN"], *VARIANTES["base"])
    refs = _refs(ctx, "base")
    mets = ctx.metodos_en("test")
    base = mets["erm"][0]
    y, g = base["label"].to_numpy(), base["patient"].to_numpy()
    src = np.asarray([K.S.COVIDX_SOURCES[int(i)] for i in base["source"]])
    filas = []
    for m, (df, P, sem) in mets.items():
        r = K.auc_media(ctx, y, P, g)
        kb = src == "bimcv"
        rb = K.auc_media(ctx, y[kb], P[:, kb], g[kb])
        filas.append({"metodo": m, "auc_test_completo": r["auc"], "ic95_inf": r["ic95_inf"], "ic95_sup": r["ic95_sup"],
                      "auc_test_bimcv": rb["auc"], "completo_menos_bimcv": r["auc"] - rb["auc"]})
    det_c = _detectores(ctx, ["base"], conjunto="test")
    det_b = _detectores(ctx, ["base"], conjunto="test_bimcv")
    for nombre in ("D_IN", "D_ERM", "D_MASK", "D_RES"):
        c, b = det_c[det_c.detector == nombre], det_b[det_b.detector == nombre]
        filas.append({"metodo": nombre, "auc_test_completo": c["auc"].mean(), "ic95_inf": c["ic95_inf"].min(),
                      "ic95_sup": c["ic95_sup"].max(), "auc_test_bimcv": b["auc"].mean(),
                      "completo_menos_bimcv": c["auc"].mean() - b["auc"].mean(), "n_troncos": len(c)})
    fams = [_corta(K.f_c1(ctx, mets, din), "D_IN", "test_completo"), _corta(K.f_c1b(ctx, mets, refs), "D_ERM", "test_completo")]
    comp = pd.Series(src).value_counts().rename_axis("fuente").reset_index(name="n")
    comp["positivas"] = [int(y[src == f].sum()) for f in comp["fuente"]]
    K.escribir("s7_test_completo", {"s7_auc": pd.DataFrame(filas), "s7_familias": pd.concat(fams, ignore_index=True),
                                    "s7_detectores": det_c, "s7_composicion": comp}, ctx, ORDEN % "s7",
               ["Test limpio completo: 8 479 imágenes, la parte bimcv (8 079) más 200 positivas de ricord y 200 negativas "
                "de rsna. Advertencia registrada (docs/08, versión 1, Section 1): las fuentes de una sola clase permiten "
                "acertar reconociendo la fuente, de modo que estas cifras no miden mejor la enfermedad que las de bimcv.",
                "En las filas de detectores, ic95_inf e ic95_sup son el mínimo y el máximo de los extremos entre troncos "
                "cuando hay varios; el IC de cada tronco está en s7_detectores."])
    return 0


# ====================================================================================== e3
def e3(ctx):
    """Sin promediado de espejo: referencias ahora; métodos cuando exista el trabajo sens_e3_predecir_sin_espejo."""
    det = _detectores(ctx, ["base", "sin_espejo"])
    tablas = {"e3_detectores": det, "e3_detectores_resumen": _resumen_det(det)}
    notas = ["Referencias: mismo ajuste del detector; la puntuación es la función de decisión de la imagen original "
             "(tta 'no'), frente a la media con su espejo de la cifra principal."]
    raiz = K.ROOT / "results" / "sensibilidad_v28" / "sin_espejo_pred"
    try:
        mets = ctx.metodos_en("test_bimcv", "best", test_pred=raiz)
    except FileNotFoundError as e:
        mets = None
        notas.append("PENDIENTE: faltan las predicciones sin espejo de los métodos (%s). Las produce el trabajo "
                     "jobs/staged/sens_e3_predecir_sin_espejo.json (GPU); después, repetir esta orden." % e)
    # comprobación parcial con los logit guardados de los troncos de ERM y de ERM con máscara
    tablas["e3_erm_desde_logit"] = _erm_desde_logit(ctx)
    if mets is not None:
        din = ctx.puntuaciones(ctx.L["troncos"]["D_IN"], *VARIANTES["sin_espejo"])
        refs = _refs(ctx, "sin_espejo")
        base = mets["erm"][0]
        aucs = []
        mets_e = ctx.metodos_en("test_bimcv", "best")
        for m, (df, P, sem) in mets.items():
            r = K.auc_media(ctx, base["label"], P, base["patient"])
            r0 = K.auc_media(ctx, mets_e[m][0]["label"], mets_e[m][1], mets_e[m][0]["patient"])
            aucs.append({"metodo": m, "auc_sin_espejo": r["auc"], "ic95_inf": r["ic95_inf"], "ic95_sup": r["ic95_sup"],
                         "auc_con_espejo": r0["auc"], "sin_menos_con": r["auc"] - r0["auc"]})
        tablas["e3_auc_metodos"] = pd.DataFrame(aucs)
        tablas["e3_familias"] = pd.concat([_corta(K.f_c1(ctx, mets, din), "D_IN", "sin_espejo"),
                                           _corta(K.f_c1b(ctx, mets, refs), "D_ERM", "sin_espejo")], ignore_index=True)
    K.escribir("e3_sin_espejo", tablas, ctx, ORDEN % "e3", notas)
    return 0


def _erm_desde_logit(ctx):
    """ERM y ERM con máscara sin espejo a partir del logit que guardó la extracción de características (fp32), con la
    reconstrucción de la versión con espejo como control frente a las predicciones del test (bf16)."""
    import numpy as np
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
    filas = []
    for met, fmt in (("erm", ctx.L["troncos"]["D_ERM"]), ("erm_mascara", ctx.L["troncos"]["D_MASK"])):
        fr = ctx.aud.preds(met, "test_bimcv", "best")
        for s in ctx.semillas:
            tag = fmt.format(semilla=s)
            z = np.load(K.ROOT / "results/c1/caracteristicas_test" / ("%s__test.npz" % tag))
            m = pd.read_csv(K.ROOT / "results/c1/caracteristicas_test" / ("%s__test_meta.csv" % tag), dtype={"image_id": str, "patient_id": str})
            df = fr[s]
            l0 = K.A.alinear(df["image_id"], m["image_id"], z["logit"])
            l1 = K.A.alinear(df["image_id"], m["image_id"], z["logit_espejo"])
            y, g = df["label"].to_numpy(), df["patient"].to_numpy()
            p_tta, p_no = 0.5 * (sig(l0) + sig(l1)), sig(l0)
            d = np.abs(p_tta - df["prob"].to_numpy())
            filas.append({"metodo": met, "semilla": s, "auc_prediccion_test_con_espejo": K.S.auc_roc(y, df["prob"].to_numpy()),
                          "auc_logit_con_espejo": K.S.auc_roc(y, p_tta), "auc_logit_sin_espejo": K.S.auc_roc(y, p_no),
                          "max_abs_dp_reconstruccion": float(d.max()), "media_abs_dp_reconstruccion": float(d.mean())})
    t = pd.DataFrame(filas)
    t["sin_menos_con_logit"] = t["auc_logit_sin_espejo"] - t["auc_logit_con_espejo"]
    return t


def resumen():
    """Tabla resumen de todas las familias (principal y sensibilidades): referencia, AUC, recuentos de no inferioridad
    y diferencias frente a ERM y DFR. Solo lee tablas ya escritas (results/c1/test y results/sensibilidad_v28)."""
    R = K.OUT
    fuentes = [("principal (results/c1/test/f_c1.csv)", K.ROOT / "results/c1/test/f_c1.csv", None, "base"),
               ("principal (results/c1/test/f_c1b.csv)", K.ROOT / "results/c1/test/f_c1b.csv", None, "base"),
               ("1: detector", R / "s1_familias.csv", "variante", None),
               ("4: estados", R / "s4_familias.csv", "variante", None),
               ("5: conjunto de semillas", R / "s5_d_in_frente_conjunto.csv", "variante", None),
               ("6: unidad de remuestreo", R / "s6_familias.csv", "unidad", None),
               ("7: test completo", R / "s7_familias.csv", "variante", None),
               ("E3: sin espejo", R / "e3_familias.csv", "variante", None)]
    filas = []
    for analisis, f, col, var in fuentes:
        if not Path(f).exists():
            filas.append({"analisis": analisis, "variante": "pendiente", "nota": "falta %s" % Path(f).name})
            continue
        t = pd.read_csv(f)
        claves = ["referencia"] + ([col] if col else [])
        for k, g in t.groupby(claves, sort=False):
            k = k if isinstance(k, tuple) else (k,)
            e, d = g.set_index("metodo").loc["erm"], g.set_index("metodo").loc["dfr"]
            filas.append({"analisis": analisis, "referencia": k[0], "variante": k[1] if col else var,
                          "auc_referencia": g["auc_referencia"].mean(),
                          "no_inferior_0.02": int(g["no_inferior_0.02"].sum()), "no_inferior_0.01": int(g["no_inferior_0.01"].sum()),
                          "n_comparaciones": len(g), "ic_entero_sobre_cero": int((g["ic95_inf"] > 0).sum()),
                          "fallan_0.02": " ".join(g.loc[~g["no_inferior_0.02"].astype(bool), "metodo"]),
                          "fallan_0.01": " ".join(g.loc[~g["no_inferior_0.01"].astype(bool), "metodo"]),
                          "delta_erm": e["delta"], "ic95_inf_erm": e["ic95_inf"], "ic95_sup_erm": e["ic95_sup"],
                          "delta_dfr": d["delta"], "ic95_inf_dfr": d["ic95_inf"], "ic95_sup_dfr": d["ic95_sup"]})
    t = pd.DataFrame(filas)
    t.to_csv(R / "resumen_familias.csv", index=False)
    from src.eval.dosis import a_markdown
    (R / "resumen_familias.md").write_text("# Resumen de las familias F-C1 y F-C1b\n\nGenerado por `python scripts/sensibilidad/v28.py "
                                           "resumen` a partir de las tablas ya escritas.\n\n" + a_markdown(t), encoding="utf-8")
    print("escrito %s (%d filas)" % (R / "resumen_familias.csv", len(t)))
    return 0


ANALISIS = {"s0": s0, "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7, "e3": e3}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("analisis", choices=list(ANALISIS) + ["todo", "resumen"])
    ap.add_argument("--presupuesto-s", type=float, default=0.0)
    ap.add_argument("--sin-cache", action="store_true")
    a = ap.parse_args(argv)
    if a.analisis == "resumen":
        return resumen()
    ctx = K.Ctx(a.analisis, a.presupuesto_s, not a.sin_cache)
    lista = list(ANALISIS) if a.analisis == "todo" else [a.analisis]
    cod = 0
    try:
        for n in lista:
            cod = max(cod, ANALISIS[n](ctx) or 0)
        if a.analisis == "todo":
            resumen()
    except K.Presupuesto as e:
        print("%s; repetir la orden para continuar (lo calculado está en caché)" % e)
        return 3
    return cod


if __name__ == "__main__":
    sys.exit(main())
