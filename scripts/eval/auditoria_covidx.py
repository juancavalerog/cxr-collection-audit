"""Auditoría de COVIDx CXR-4 en validación (aporte C1; docs/08_evaluacion.md, versión 2, Section V2.3).

Regenera con una orden todas las tablas de la auditoría que pueden calcularse sin abrir el test:

    python scripts/eval/auditoria_covidx.py            # B = 2000, semilla 2026, salida en results/audit/

1. Detector de colección sin información de clase en cada fichero de características de results/diag/expl_lectura/,
   con su AUC en val_robust (IC95 por remuestreo de pacientes), los estratos AP y PA, la AUC de la proyección sola y
   la diferencia frente a la cabeza con etiquetas del mismo tronco (y, en los troncos con reserva, frente a la de ERM
   de la misma semilla).
2. Comprobación de que se reproducen las cifras de la Tabla E4 de docs/07_piloto.md con tolerancia de 0.005; si no,
   el programa termina con código 1.
3. Validación cruzada y transferencia entre colecciones de positivos en los troncos con conjunto reservado.
4. Detector (tronco ERM de cada semilla) frente a cada uno de los doce métodos del piloto, emparejado por semilla y
   por imágenes, con la prueba de no inferioridad de margen 0.02 y Holm (ensayo en validación del análisis C1-i).
5. Evaluación por colección (análisis C1-iii) de los doce métodos y del detector en la validación completa.

Solo CPU. No lee ningún conjunto test: los ficheros de características no los contienen y las predicciones se leen
con la guarda de `src.eval.stats.read_predictions`.

Ampliación (17-11-2026; docs/14_auditoria_y_congelacion.md). Sin opciones nuevas, la orden de arriba
hace exactamente lo mismo que antes. Con ellas:

    python scripts/eval/auditoria_covidx.py --din            # AUC de D_IN en val_robust con IC95 (M3)
    python scripts/eval/auditoria_covidx.py --c1 val         # los siete análisis de C1 en validación (ensayo)
    python scripts/eval/auditoria_covidx.py --c1 test        # los mismos en el test; exige la congelación del protocolo

Salida en results/c1/<val|test>/ (o --out-c1): una tabla CSV por cada entrada de configs/gate7_lista_c1.yaml, c1.json
y c1.md. El modo test llama a `src.eval.desbloqueo.exigir_congelacion` antes de abrir nada.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import audit as A  # noqa: E402
from src.eval import stats as S  # noqa: E402
from src.eval.dosis import a_markdown  # noqa: E402

# Tabla E4 de docs/07_piloto.md (detector usado como clasificador en val_robust; AP y PA entre paréntesis)
E4 = {"expl_erm_reserva_seed42_best": (0.907, 0.904, 0.902), "baseline_erm_seed42_best": (0.892, 0.900, 0.886),
      "baseline_erm_seed123_best": (0.877, 0.865, 0.890)}
E4_RESERVADO = {"cv_neg_vs_antiguos": 0.930, "cv_neg_vs_nuevos": 0.736, "cv_antiguos_vs_nuevos": 0.929,
                "transfer_antiguos_a_nuevos": 0.620, "transfer_nuevos_a_antiguos": 0.629,
                "lectura_nuevos_en_val_robust": 0.635}
TOL = 0.005
METODOS = ("erm", "control_fuente", "groupdro", "dann", "coral", "mixstyle", "fourier", "lisa", "dfa_fuente", "dfr",
           "erm_mascara", "erm_cuotas")
SEMILLAS = (42, 123)


def _pred(train_root: Path, metodo: str, seed: int, conjunto: str):
    st = "dfr_best" if metodo == "dfr" else "best"
    f = train_root / ("baseline_" + metodo) / ("seed%d" % seed) / "predictions" / ("%s_%s.csv" % (st, conjunto))
    if not f.exists():
        return None, f
    return S.read_predictions(f, forbid_test=True, context="auditoría"), f


def _alinear(ref_ids, ids, valores):
    s = pd.Series(np.asarray(valores, float), index=pd.Index(ids, dtype=str))
    if not set(ref_ids) <= set(s.index):
        raise S.AlignmentError("faltan imágenes al alinear el detector con las predicciones")
    return s.loc[list(ref_ids)].to_numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(ROOT / "results/diag/expl_lectura"))
    ap.add_argument("--train", default=str(ROOT / "results/train"))
    ap.add_argument("--out", default=str(ROOT / "results/audit"))
    ap.add_argument("--B", type=int, default=S.DEFAULT_B)
    ap.add_argument("--seed", type=int, default=S.DEFAULT_BOOT_SEED)
    ap.add_argument("--tags", nargs="*", default=None, help="por defecto, todos los .npz del directorio")
    ap.add_argument("--sin-metodos", action="store_true", help="omite los apartados 4 y 5")
    g = ap.add_argument_group("análisis C1 completos (docs/14)")
    g.add_argument("--c1", choices=["val", "test"], default=None)
    g.add_argument("--din", action="store_true", help="solo la AUC de D_IN en val_robust con su IC95")
    g.add_argument("--lista", default=str(ROOT / "configs/gate7_lista_c1.yaml"))
    g.add_argument("--out-c1", default=None, help="por defecto results/c1/<val|test>")
    g.add_argument("--caracteristicas", default=None, help="por defecto results/c1/caracteristicas")
    g.add_argument("--caracteristicas-test", default=None)
    g.add_argument("--adquisicion", default=None)
    g.add_argument("--adquisicion-test", default=None)
    g.add_argument("--test-pred", default=None)
    g.add_argument("--covidgr-pred", default=None)
    g.add_argument("--covidgr-csv", default=None, help="manifiesto de COVIDGR (solo modo test)")
    g.add_argument("--experimento", default=None, help="formato del experimento, p. ej. baseline_{metodo} (piloto)")
    g.add_argument("--semillas", nargs="*", type=int, default=None)
    g.add_argument("--tag-derm", default=None, help="formato del tronco D_ERM, p. ej. baseline_erm_seed{semilla}_best")
    g.add_argument("--tag-mask", default=None)
    g.add_argument("--tta-detector", choices=["no", "espejo"], default=None)
    g.add_argument("--permitir-incompleto", action="store_true", help="solo pruebas")
    a = ap.parse_args(argv)
    if a.c1 or a.din:
        return main_c1(a)
    t0 = time.time()
    d, out, tr_root = Path(a.dir), Path(a.out), Path(a.train)
    out.mkdir(parents=True, exist_ok=True)
    tags = a.tags or sorted(p.stem for p in d.glob("*.npz"))
    entradas, avisos = {}, []
    det_rows, res_rows, scores = [], [], {}
    heads = {}
    for tag in tags:
        feats, m = A.cargar(d, tag)
        for ext in (".npz", "_meta.csv"):
            entradas[tag + ext] = S.sha256_file(d / (tag + ext))
        s, info = A.detector(feats, m)
        vr = m["val_robust"].to_numpy()
        y, g = m["label"].to_numpy().astype(int), m["patient_id"].astype(str).to_numpy()
        scores[tag] = (m.loc[vr | (m["conjunto"] == "val"), "image_id"].tolist(), s[(vr | (m["conjunto"] == "val"))], m)
        heads[tag] = (m.loc[vr, "image_id"].tolist(), feats["logit"][vr])
        row = {"tag": tag, **{k: v for k, v in info.items() if k not in ("usa_clase",)}}
        r = A.auc_ic(y[vr], s[vr], g[vr], a.B, a.seed)
        row.update({"auc_val_robust": r["auc"], "ic95_inf": r["ic95_inf"], "ic95_sup": r["ic95_sup"],
                    "n_val_robust": r["n"], "n_neg": r["n_neg"], "n_pacientes": r["n_pacientes"]})
        for pr in ("AP", "PA"):
            mk = vr & (m["proyeccion"].to_numpy() == pr)
            rr = A.auc_ic(y[mk], s[mk], g[mk], a.B, a.seed)
            row.update({"auc_%s" % pr: rr["auc"], "ic95_inf_%s" % pr: rr["ic95_inf"], "ic95_sup_%s" % pr: rr["ic95_sup"],
                        "n_%s" % pr: rr["n"], "n_neg_%s" % pr: rr["n_neg"]})
        row.update(A.referencia_proyeccion(m, vr))
        hd = A.diferencia(y[vr], s[vr], feats["logit"][vr], g[vr], a.B, a.seed, margen=A.MARGEN_NI)
        row.update({"auc_cabeza_mismo_tronco": hd["auc2"], "delta_vs_cabeza": hd["delta"],
                    "delta_ic95_inf": hd["ic95_inf"], "delta_ic95_sup": hd["ic95_sup"],
                    "p_no_inferioridad_0.02": hd["p_no_inferioridad"]})
        det_rows.append(row)
        cr = A.contrastes_reservado(feats, m, B=a.B, seed=a.seed)
        if cr:
            res_rows.append({"tag": tag, **{k: v for k, v in cr.items() if k != "n"}, **{"n_" + k: v for k, v in cr["n"].items()}})
        print("[%s] detector en val_robust %.4f (AP %.4f, PA %.4f); cabeza %.4f" % (tag, row["auc_val_robust"], row["auc_AP"],
                                                                                   row["auc_PA"], row["auc_cabeza_mismo_tronco"]), flush=True)
    det = pd.DataFrame(det_rows)
    # troncos con reserva frente a la cabeza de ERM de la misma semilla
    comp_erm = []
    for tag in tags:
        if not tag.startswith("expl_"):
            continue
        sd = tag.split("_seed")[1].split("_")[0]
        base = "baseline_erm_seed%s_best" % sd
        if base not in heads:
            continue
        ids, sc, m = scores[tag]
        bid, bl = heads[base]
        mm = m[m["val_robust"]]
        sv = _alinear(bid, ids, sc)
        r = A.diferencia(mm.set_index("image_id").loc[bid, "label"].to_numpy(), sv, bl,
                         mm.set_index("image_id").loc[bid, "patient_id"].astype(str).to_numpy(), a.B, a.seed, margen=A.MARGEN_NI)
        comp_erm.append({"tag": tag, "comparador": base + " (cabeza ERM, sin TTA)", **{k: v for k, v in r.items()
                                                                                  if k not in ("delta_por_semilla", "semillas")}})
    # 2. reproducción de la Tabla E4
    rep = []
    for tag, (v, vap, vpa) in E4.items():
        if tag not in set(det["tag"]):
            rep.append({"cifra": tag, "registrada": v, "reproducida": np.nan, "diferencia": np.nan, "cumple": False})
            continue
        r = det.set_index("tag").loc[tag]
        for nombre, reg, val in (("val_robust", v, r["auc_val_robust"]), ("AP", vap, r["auc_AP"]), ("PA", vpa, r["auc_PA"])):
            rep.append({"cifra": "%s, %s" % (tag, nombre), "registrada": reg, "reproducida": float(val),
                        "diferencia": float(val - reg), "cumple": bool(abs(val - reg) <= TOL)})
    resv = pd.DataFrame(res_rows)
    if "expl_erm_reserva_seed42_best" in set(resv.get("tag", [])):
        r = resv.set_index("tag").loc["expl_erm_reserva_seed42_best"]
        for k, reg in E4_RESERVADO.items():
            rep.append({"cifra": "expl_erm_reserva_seed42_best, " + k, "registrada": reg, "reproducida": float(r[k]),
                        "diferencia": float(r[k] - reg), "cumple": bool(abs(r[k] - reg) <= TOL)})
    rep = pd.DataFrame(rep)
    # 4 y 5. métodos del piloto
    vsm, porcol = [], []
    if not a.sin_metodos:
        for conj in ("val_robust", "val_robust_sel"):
            pv = {}
            for meth in METODOS:
                fr = {}
                for sd in SEMILLAS:
                    df, f = _pred(tr_root, meth, sd, conj)
                    if df is not None:
                        fr[sd] = df
                        entradas[str(f.relative_to(ROOT)) if str(f).startswith(str(ROOT)) else str(f)] = S.sha256_file(f)
                if len(fr) == len(SEMILLAS):
                    pv[meth] = fr
                elif fr or (meth != "dfr" or conj == "val_robust_sel"):
                    if not (meth == "dfr" and conj == "val_robust"):
                        avisos.append("%s en %s: faltan semillas (%s)" % (meth, conj, sorted(fr)))
            for meth, fr in pv.items():
                ref = fr[SEMILLAS[0]]
                ids = ref["image_id"].tolist()
                P2 = np.vstack([S.align(ref, fr[sd], "s%d" % SEMILLAS[0], "s%d" % sd)["prob"].to_numpy() for sd in SEMILLAS])
                P1 = np.vstack([_alinear(ids, *scores["baseline_erm_seed%d_best" % sd][:2]) for sd in SEMILLAS])
                r = A.diferencia(ref["label"].to_numpy(), P1, P2, ref["patient"].to_numpy(), a.B, a.seed,
                                 margen=A.MARGEN_NI, semillas=SEMILLAS)
                vsm.append({"conjunto": conj, "metodo": meth, "auc_detector_media": r["auc1"], "auc_metodo_media": r["auc2"],
                            **{"auc_metodo_s%d" % sd: S.auc_roc(ref["label"].to_numpy(), P2[i]) for i, sd in enumerate(SEMILLAS)},
                            "delta": r["delta"], "ic95_inf": r["ic95_inf"], "ic95_sup": r["ic95_sup"],
                            "p_no_inferioridad": r["p_no_inferioridad"], "p_superioridad": r["p_superioridad"],
                            "no_inferior_ic": r["no_inferior"], "n": int(len(ref)), "comparable": not (meth == "dfr" and conj == "val_robust")})
        vsm = pd.DataFrame(vsm)
        if len(vsm):
            vsm["p_holm_no_inferioridad"] = np.nan
            vsm["no_inferior_holm"] = False
            for conj, sub in vsm.groupby("conjunto"):
                h = S.holm({i: p for i, p in zip(sub.index, sub["p_no_inferioridad"])}, alpha=0.025)
                for i in sub.index:
                    vsm.loc[i, "p_holm_no_inferioridad"] = h[i]["p_holm"]
                    vsm.loc[i, "no_inferior_holm"] = h[i]["rechaza"]
        # evaluación por colección en la validación completa
        for meth in METODOS + ("detector",):
            for sd in SEMILLAS:
                if meth == "detector":
                    df, _ = _pred(tr_root, "erm", sd, "val")
                    if df is None:
                        continue
                    sc = _alinear(df["image_id"].tolist(), *scores["baseline_erm_seed%d_best" % sd][:2])
                else:
                    df, f = _pred(tr_root, meth, sd, "val")
                    if df is None:
                        continue
                    entradas[str(f.relative_to(ROOT))] = S.sha256_file(f)
                    sc = df["prob"].to_numpy()
                if meth == "dfr":       # DFR reajustó con val_robust_fit: se excluye esa mitad
                    keep = ~((df["source"].to_numpy() == S.COVIDX_SOURCES.index("bimcv")) &
                             (df["patient"].map(A.mitad).to_numpy() == 0))
                    df, sc = df[keep].reset_index(drop=True), sc[keep]
                t = A.tabla_por_coleccion(df, sc, a.B, a.seed)
                t.insert(0, "semilla", sd)
                t.insert(0, "metodo", meth)
                porcol.append(t)
        porcol = pd.concat(porcol, ignore_index=True) if porcol else pd.DataFrame()
    else:
        vsm, porcol = pd.DataFrame(), pd.DataFrame()
    # salidas
    det.to_csv(out / "detector_coleccion.csv", index=False)
    pd.DataFrame(comp_erm).to_csv(out / "detector_vs_erm_misma_semilla.csv", index=False)
    resv.to_csv(out / "contrastes_reservado.csv", index=False)
    rep.to_csv(out / "reproduccion_tabla_E4.csv", index=False)
    vsm.to_csv(out / "detector_vs_metodos_validacion.csv", index=False)
    porcol.to_csv(out / "por_coleccion_validacion.csv", index=False)
    ok = bool(len(rep) and rep["cumple"].all())
    meta = {"programa": "scripts/eval/auditoria_covidx.py", "fecha_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "B": a.B, "semilla_bootstrap": a.seed, "unidad_remuestreo": "paciente", "margen_no_inferioridad": A.MARGEN_NI,
            "detector": {"C": A.C_DETECTOR, "caracteristicas": A.FEAT_DETECTOR, "penalizacion": "l2",
                         "estandarizacion": "conjunto de ajuste (solo positivos)"},
            "reproduce_tabla_E4": ok, "tolerancia": TOL, "avisos": avisos, "entradas_sha256": entradas,
            "test": "no se lee ningún conjunto test", "segundos": round(time.time() - t0, 1)}
    (out / "auditoria.json").write_text(json.dumps(S.jsonable(meta), indent=1, ensure_ascii=False), encoding="utf-8")
    md = ["# Auditoría de COVIDx CXR-4 en validación", "",
          "Generado por `python scripts/eval/auditoria_covidx.py` (%s UTC). B = %d, semilla %d, remuestreo por paciente. "
          "No se ha leído ningún conjunto test. Plan: docs/08_evaluacion.md, versión 2, Section V2.3. Cifras de validación "
          "con dos semillas: son el ensayo del análisis registrado, no su resultado." % (meta["fecha_utc"], a.B, a.seed), "",
          "## 1. Detector de colección sin información de clase", "",
          a_markdown(det[["tag", "origen_ajuste", "n_ajuste", "n_antigua", "n_nueva", "auc_val_robust", "ic95_inf", "ic95_sup",
                          "auc_AP", "auc_PA", "auc_solo_proyeccion_AP", "auc_cabeza_mismo_tronco", "delta_vs_cabeza",
                          "delta_ic95_inf", "delta_ic95_sup"]]),
          "Troncos con reserva frente a la cabeza de ERM de la misma semilla:", "",
          a_markdown(pd.DataFrame(comp_erm)[["tag", "comparador", "auc1", "auc2", "delta", "ic95_inf", "ic95_sup"]]
                     if comp_erm else pd.DataFrame()),
          "## 2. Reproducción de la Tabla E4 de docs/07_piloto.md (tolerancia %.3f)" % TOL, "", a_markdown(rep),
          "Resultado: %s." % ("se reproducen todas las cifras" if ok else "NO se reproducen todas las cifras"), "",
          "## 3. Contrastes dentro del conjunto reservado y transferencia", "", a_markdown(resv), ""]
    if len(vsm):
        md += ["## 4. Detector (tronco ERM de cada semilla) frente a los métodos con etiquetas", "",
               "Media por semilla de AUC(detector) menos AUC(método), emparejada por semilla y por imágenes. No "
               "inferioridad con margen %.2f: H1 es delta > -%.2f; Holm al 0.025 unilateral dentro de cada conjunto. "
               "DFR solo es comparable en val_robust_sel. El detector no usa TTA; los métodos sí." % (A.MARGEN_NI, A.MARGEN_NI), "",
               a_markdown(vsm.drop(columns=["p_superioridad"])), ""]
    if len(porcol):
        piv = porcol.groupby(["metodo", "negativos", "positivos"], as_index=False)["auc"].mean()
        piv = piv.pivot_table(index="metodo", columns=["negativos", "positivos"], values="auc")
        piv.columns = ["%s / %s" % c for c in piv.columns]
        md += ["## 5. AUC por colección en la validación completa (media de las dos semillas)", "",
               "Cada columna es una pareja negativos / positivos con al menos %d imágenes por lado; los IC95 por semilla "
               "están en `por_coleccion_validacion.csv`. En DFR se excluye la mitad val_robust_fit de bimcv." % A.N_MIN, "",
               a_markdown(piv.reset_index()), ""]
    if avisos:
        md += ["## Avisos", ""] + ["- " + x for x in avisos] + [""]
    (out / "auditoria.md").write_text("\n".join(md), encoding="utf-8")
    print("reproduce la Tabla E4: %s; %.1f s; salidas en %s" % (ok, time.time() - t0, out))
    return 0 if ok else 1


def main_c1(a) -> int:
    from src.eval import c1_auditoria as C
    modo = a.c1 or "val"
    L = C.leer_lista(a.lista)
    R = C.Rutas(ROOT, a.caracteristicas, a.caracteristicas_test, a.dir, a.adquisicion, a.adquisicion_test, a.train,
                a.test_pred, a.covidgr_pred, a.covidgr_csv)
    au = C.Auditoria(modo, L, R, a.B, a.seed, a.experimento, a.semillas, a.tag_derm, a.tag_mask, a.permitir_incompleto,
                     a.tta_detector)
    out = Path(a.out_c1 or ROOT / "results" / "c1" / modo)
    r = au.ejecutar(out, solo_din=bool(a.din and not a.c1))
    det = r["tablas_df"]["detectores"]
    d = det[det["detector"] == "D_IN"] if len(det) else det
    if len(d):
        x = d.iloc[0]
        print("D_IN en %s: AUC %.4f (IC95 %.4f a %.4f), %d imágenes, %d negativas, %d pacientes; TTA del detector: %s"
              % (x["conjunto"], x["auc"], x["ic95_inf"], x["ic95_sup"], x["n"], x["n_neg"], x["n_pacientes"], x["tta"]))
        if a.din:
            (out / ("din_%s.json" % x["conjunto"])).write_text(json.dumps(S.jsonable(x.to_dict()), indent=1, ensure_ascii=False),
                                                           encoding="utf-8")
    else:
        print("sin características de D_IN (%s)" % R.car)
    for k, v in r["tablas"].items():
        print("tabla %s: %d filas" % (k, v))
    for w in r["avisos"]:
        print("aviso: " + w)
    print("salidas en %s" % out)
    return 0 if len(d) else 1


if __name__ == "__main__":
    sys.exit(main())
