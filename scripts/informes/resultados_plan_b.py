#!/usr/bin/env python3
"""Escribe docs/15_resultados_plan_b.md a partir de los CSV de los informes del test (no calcula ninguna cifra nueva).

    python scripts/informes/resultados_plan_b.py

Lee results/c1/test, results/b3/test_A_best, results/c2c3, results/b2/informe_test, results/sensibilidad y
results/b1/finales_validacion.csv. Cada tabla indica el fichero del que sale y la orden que lo regenera.
"""
import os
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NOM = {"erm": "ERM", "erm_cuotas": "ERM con cuotas", "erm_mascara": "ERM con máscara", "control_fuente": "Control de fuente",
       "groupdro": "GroupDRO", "dann": "DANN", "coral": "CORAL", "mixstyle": "MixStyle", "fourier": "Fourier", "lisa": "LISA",
       "dfa_fuente": "DFA con fuente", "dfr": "DFR"}
ORD = list(NOM)


def rd(p):
    return pd.read_csv(os.path.join(R, p))


def f3(x):
    return "no aplica" if pd.isna(x) else ("%.3f" % x)


def ic(a, b):
    return "%.3f a %.3f" % (a, b)


def tabla(cab, filas):
    out = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    out += ["| " + " | ".join(str(c) for c in f) + " |" for f in filas]
    return "\n".join(out)


def main():
    L = []
    A = L.append
    A("# Resultados del plan B en el test")
    A("")
    A("Documento generado por `python scripts/informes/resultados_plan_b.py` a partir de los CSV que escribieron los informes "
      "registrados, todos ejecutados con la congelación de la Puerta 7 válida (SHA256 de `results/gate7/congelacion.json` "
      "anotado en STATUS.md el 17-11-2026; verificación final correcta el 20-11-2026). No se añadió ningún experimento ni se "
      "cambió ningún análisis después de abrir los tests. Las definiciones están en `docs/08_evaluacion.md` (versión 2 y "
      "enmiendas E0 a E8). Los intervalos de C2, C2-R, C3 y de la sensibilidad son los jerárquicos de la enmienda E1.")
    A("")
    # ------------------------------------------------------------------ C1
    det = rd("results/c1/test/detectores.csv"); fc1 = rd("results/c1/test/f_c1.csv"); fb = rd("results/c1/test/f_c1b.csv")
    cg = rd("results/c1/test/c1_vi_covidgr.csv"); adq = rd("results/c1/test/c1_iv_adquisicion.csv"); mk = rd("results/c1/test/c1_v_mascara.csv")
    din = det[det.detector == "D_IN"].iloc[0]
    A("## 1. Hallazgo C1: auditoría de COVIDx CXR-4")
    A("")
    A("Conjunto confirmatorio: parte bimcv del test limpio (%d imágenes, %d negativas). Regenerar con "
      "`python scripts/eval/auditoria_covidx.py --c1 test` (salidas en `results/c1/test/`)." % (din.n, din.n_neg))
    A("")
    A("**Tabla R1.** Detectores de colección sin información de clase en test_bimcv (`detectores.csv`).")
    A("")
    filas = [("D_IN, tronco de ImageNet sin ajuste (principal)", f3(din.auc), ic(din.ic95_inf, din.ic95_sup))]
    for d_, n in (("D_ERM", "D_ERM, troncos finales de ERM (5 semillas)"), ("D_MASK", "D_MASK, troncos de ERM con máscara (5 semillas)"),
                  ("D_RES", "D_RES, troncos con reserva de la exploración (4)")):
        x = det[det.detector == d_]
        filas.append((n, "%.3f (media)" % x.auc.mean(), "%.3f a %.3f (mínimo y máximo entre troncos)" % (x.auc.min(), x.auc.max())))
    A(tabla(["Referencia", "AUC", "IC95 o intervalo"], filas))
    A("")
    A("**Tabla R2.** Familia F-C1: D_IN frente a cada método (media de cinco semillas), no inferioridad con Holm (`f_c1.csv`).")
    A("")
    filas = []
    for m in ORD:
        x = fc1[fc1.metodo == m]
        if x.empty:
            continue
        x = x.iloc[0]
        filas.append((NOM[m], f3(x.auc_metodo), "%.3f" % x.delta, ic(x.ic95_inf, x.ic95_sup),
                      "sí" if x["no_inferior_0.02"] else "no", "sí" if x["no_inferior_0.01"] else "no"))
    A(tabla(["Método", "AUC del método", "D_IN menos método", "IC95", "No inferior (margen 0.02)", "No inferior (margen 0.01)"], filas))
    A("")
    n02 = int(fc1["no_inferior_0.02"].sum()); nsup = int((fc1.ic95_inf > 0).sum())
    dfr = fc1[fc1.metodo == "dfr"].iloc[0]; erm = fc1[fc1.metodo == "erm"].iloc[0]
    A("Lectura. D_IN es no inferior a %d de los %d métodos con el margen registrado y lo es también con el margen de 0.01; "
      "en %d de ellos el intervalo de la diferencia queda entero por encima de cero. Frente a ERM la diferencia es %.3f "
      "(IC95 %s). La excepción es DFR, que supera a D_IN en %.3f (IC95 %s). La referencia secundaria D_ERM es no inferior "
      "a los %d métodos de `f_c1b.csv` (%d de %d comparaciones)."
      % (n02, len(fc1), nsup, erm.delta, ic(erm.ic95_inf, erm.ic95_sup), -dfr.delta, ic(-dfr.ic95_sup, -dfr.ic95_inf),
         len(fb), int(fb["no_inferior_0.02"].sum()), len(fb)))
    A("")
    A("**Tabla R3.** Cohorte externa COVIDGR (`c1_vi_covidgr.csv`): AUC interna en test_bimcv y externa, media de cinco semillas.")
    A("")
    filas = []
    for m in ORD:
        x = cg[cg.metodo == m]
        if x.empty:
            continue
        x = x.iloc[0]
        filas.append((NOM[m], f3(x.auc_interna), f3(x.auc_externa), ic(x.auc_externa_ic95_inf, x.auc_externa_ic95_sup),
                      "%.3f" % x.caida, ic(x.caida_ic95_inf, x.caida_ic95_sup)))
    A(tabla(["Método", "AUC interna", "AUC en COVIDGR", "IC95", "Caída", "IC95 de la caída"], filas))
    A("")
    a_, b_ = adq.iloc[0], adq.iloc[1]; m0 = mk.iloc[0]
    A("Otros análisis registrados. El clasificador de solo adquisición alcanza %.3f (IC95 %s) ajustado en entrenamiento y "
      "%.3f (IC95 %s) con validación cruzada dentro del conjunto (`c1_iv_adquisicion.csv`). ERM con máscara queda %.3f por "
      "debajo de ERM (IC95 %s; `c1_v_mascara.csv`). La evaluación por colección y los estratos AP y PA están en "
      "`c1_iii_resumen.csv` y `c1_vii_estratos.csv`."
      % (a_.auc, ic(a_.ic95_inf, a_.ic95_sup), b_.auc, ic(b_.ic95_inf, b_.ic95_sup), -m0.delta, ic(-m0.ic95_sup, -m0.ic95_inf)))
    A("")
    # ------------------------------------------------------------------ C2
    cur = rd("results/b3/test_A_best/curva.csv"); cmp_ = rd("results/b3/test_A_best/comparacion_erm.csv"); res = rd("results/b3/test_A_best/resumen.csv")
    A("## 2. Hallazgo C2: curva dosis-respuesta bajo confusión controlada (B3a)")
    A("")
    A("Derrame pleural, NIH frente a CheXpert, orientación A, estado elegido con la validación de la misma rho. Cinco semillas "
      "por nivel y diez en rho 0.9, 0.95 y 0.99. Regenerar con "
      "`python scripts/eval/informe_b3.py --conjunto test_desconf --permitir-test --out results/b3/test_A_best`.")
    A("")
    A("**Tabla R4.** AUC en test_desconf, media entre semillas (`curva.csv`).")
    A("")
    niveles = sorted(cur.rho.unique()); pv = cur.pivot(index="metodo", columns="rho", values="auc_media_semillas")
    A(tabla(["Método"] + ["%.2f" % r for r in niveles], [[NOM[m]] + [f3(pv.loc[m, r]) for r in niveles] for m in ORD if m in pv.index]))
    A("")
    ce = res[(res.metodo == "erm") & (res.medida == "caida_050_100")].iloc[0]
    A("ERM cae %.3f entre rho 0.5 y rho 1.0 (IC95 %s)." % (ce.estimacion, ic(ce.ic95_inf, ce.ic95_sup)))
    A("")
    A("**Tabla R5.** Diferencias frente a ERM (`comparacion_erm.csv`): área bajo la curva entre rho 0.5 y 0.99 (familia "
      "confirmatoria F-C2-area, Holm sobre diez comparaciones) y niveles de confusión fuerte (IC95 entre paréntesis).")
    A("")
    filas = []
    for m in ORD:
        x = cmp_[cmp_.metodo == m]
        if x.empty:
            continue
        g = lambda med: x[x.medida == med].iloc[0] if not x[x.medida == med].empty else None
        a = g("delta_area_dosis")
        fila = [NOM[m], "%.4f (%s)" % (a.delta, "%.4f a %.4f" % (a.ic95_inf, a.ic95_sup)), "sí" if a.rechaza_holm else "no"]
        for med in ("delta_auc_rho090", "delta_auc_rho095", "delta_auc_rho099", "delta_auc_rho100"):
            y = g(med)
            fila.append("no aplica" if y is None or pd.isna(y.delta) else "%.3f (%s)" % (y.delta, ic(y.ic95_inf, y.ic95_sup)))
        filas.append(fila)
    A(tabla(["Método", "Área, diferencia", "Supera a ERM (Holm)", "rho 0.90", "rho 0.95", "rho 0.99", "rho 1.00"], filas))
    A("")
    gan = [NOM[m] for m in ORD if not cmp_[(cmp_.metodo == m) & (cmp_.medida == "delta_area_dosis") & (cmp_.rechaza_holm)].empty]
    f100 = cmp_[(cmp_.medida == "delta_auc_rho100") & (cmp_.rechaza_holm)]
    coste = cmp_[(cmp_.medida == "coste_rho050_no_inferioridad")]
    A("Lectura. En la familia confirmatoria superan a ERM %s. Con confusión total (rho 1.0) las AUC van de %.3f a %.3f; "
      "en la familia secundaria de ese nivel superan a ERM %s, con diferencias de %.3f a %.3f. Sin confusión (rho 0.5), todos "
      "los métodos salvo %s son no inferiores a ERM."
      % (", ".join(gan), pv[1.0].min(), pv[1.0].max(), ", ".join(NOM[m] for m in f100.metodo), f100.delta.min(), f100.delta.max(),
         ", ".join(NOM[m] for m in coste[~coste.rechaza_holm].metodo) or "ninguno"))
    A("")
    ori = rd("results/c2c3/orientacion.csv").iloc[0]
    A("Orientación B positiva (`results/c2c3/orientacion.csv`): la caída de ERM es %.3f frente a %.3f en la orientación A; "
      "diferencia %.3f, veredicto registrado: %s. El test invertido, la AUC por fuente y la brecha de especificidad están en "
      "`results/c2c3/invertido.csv`, `fuente.csv` y `brecha.csv` (`python scripts/eval/informe_c2_c3.py todas --root-c3 results/c3_busqueda`)."
      % (ori.caida_B, ori.caida_A, ori.diferencia_A_menos_B, ori.veredicto))
    A("")
    # ------------------------------------------------------------------ C2-R
    b2 = rd("results/b2/informe_test/curva.csv"); b2c = rd("results/b2/informe_test/comparacion_erm.csv")
    A("## 3. Réplica C2-R: tuberculosis (B2)")
    A("")
    A("Regenerar con `python scripts/eval/informe_b2.py --test --delta-c2 results/b3/test_A_best/comparacion_erm.csv`.")
    A("")
    A("**Tabla R6.** AUC por cohorte externa (media de cinco semillas) y diferencia frente a ERM (`curva.csv` y `comparacion_erm.csv`).")
    A("")
    filas = []
    for m in ORD:
        fila = [NOM[m]]
        for c in ("test_shenzhen", "test_montgomery"):
            x = b2[(b2.metodo == m) & (b2.conjunto == c)]
            y = b2c[(b2c.metodo == m) & (b2c.conjunto == c)]
            fila.append(f3(x.auc_media_semillas.iloc[0]) if not x.empty else "no aplica")
            fila.append("referencia" if m == "erm" else ("%.3f (%s)%s" % (y.delta.iloc[0], ic(y.ic95_inf.iloc[0], y.ic95_sup.iloc[0]),
                                                                        ", Holm" if bool(y.rechaza_holm.iloc[0]) else "") if not y.empty else "no aplica"))
        filas.append(fila)
    A(tabla(["Método", "Shenzhen", "Diferencia frente a ERM", "Montgomery", "Diferencia frente a ERM"], filas))
    A("")
    gs = b2c[b2c.rechaza_holm]
    A("Lectura. Familias secundarias. Superan a ERM con Holm: %s. Ningún método gana en las dos cohortes, y los que ganan en "
      "B3a (LISA, GroupDRO, DFR, ERM con cuotas) no muestran ventaja aquí."
      % "; ".join("%s en %s (%.3f)" % (NOM[r.metodo], r.conjunto.replace("test_", "").capitalize(), r.delta) for r in gs.itertuples()))
    A("")
    # ------------------------------------------------------------------ C3
    ep = rd("results/c2c3/c3-epoca.csv"); hp = rd("results/c2c3/c3-hiper.csv"); sp = rd("results/c2c3/c3-spearman.csv")
    A("## 4. Hallazgo C3: selección de modelo")
    A("")
    A("**Tabla R7.** Selección de época (`results/c2c3/c3-epoca.csv`): media sobre once métodos de la AUC en test_desconf del "
      "estado elegido con la validación desconfundida menos la del elegido con la validación de la misma rho.")
    A("")
    A(tabla(["rho", "Pérdida media", "IC95", "Papel", "Holm"],
            [("%.2f" % r.rho, "%.4f" % r.delta_sel_media_metodos, "%.4f a %.4f" % (r.ic95_inf, r.ic95_sup), r.papel,
              ("sí" if r.rechaza_holm else "no") if r.papel == "confirmatoria" else "no aplica") for r in ep.itertuples()]))
    A("")
    c = ep[ep.papel == "confirmatoria"]
    A("Lectura. La familia F-C3 se confirma en sus tres niveles: elegir la época con la validación habitual cuesta entre %.3f y "
      "%.3f de AUC en el test desconfundido, tanto como las mayores ventajas de los métodos sobre ERM en esos niveles."
      % (c.delta_sel_media_metodos.min(), c.delta_sel_media_metodos.max()))
    A("")
    A("**Tabla R8.** Selección de hiperparámetros (`results/c2c3/c3-hiper.csv`; ocho ensayos, semilla 0): AUC en test_desconf "
      "del ensayo elegido con la regla registrada, ganancia sobre la media de los ensayos y pérdida frente al ensayo que\nelegiría la validación desconfundida.")
    A("")
    A(tabla(["Método", "rho", "Regla", "Media de ensayos", "Ensayo de la validación desconfundida", "Ganancia sobre la media (IC95)", "Pérdida frente a ese ensayo (IC95)"],
            [(NOM[r.metodo], "%.2f" % r.rho, f3(r.auc_regla), f3(r.auc_media_ensayos), f3(r.auc_oraculo),
              "%.3f (%s)" % (r.ganancia_sobre_azar, ic(r.ic95_inf_G, r.ic95_sup_G)),
              "%.3f (%s)" % (r.perdida_frente_oraculo, ic(r.ic95_inf_L, r.ic95_sup_L))) for r in hp.itertuples()]))
    A("")
    A("Lectura. La regla coincide con el ensayo de la validación desconfundida en %d de %d casos; en %d casos queda por debajo de la media de los ensayos "
      "(intervalo entero por debajo de cero) y la pérdida frente a ese ensayo llega a %.3f. La correlación de Spearman entre la "
      "métrica de selección y la AUC del test está en `results/c2c3/c3-spearman.csv`."
      % (int((hp.ensayo_regla == hp.ensayo_oraculo).sum()), len(hp), int((hp.ic95_sup_G < 0).sum()), hp.perdida_frente_oraculo.max()))
    A("")
    # ------------------------------------------------------------------ sensibilidad
    A("## 5. Análisis de sensibilidad (enmienda E8)")
    A("")
    for s, tit, reg in (("tronco", "S-T, ResNet-50", "python scripts/eval/informe_sensibilidad.py tronco"),
                        ("etiqueta", "S-E, cardiomegalia (B3c)", "python scripts/eval/informe_sensibilidad.py etiqueta")):
        k = rd("results/sensibilidad/%s/concordancia.csv" % s); c_ = rd("results/sensibilidad/%s/curva.csv" % s)
        e = [x for x in k.columns if x.startswith("estimacion_") and "principal" not in x][0]
        i1 = [x for x in k.columns if x.startswith("ic95_inf_") and "principal" not in x][0]
        i2 = [x for x in k.columns if x.startswith("ic95_sup_") and "principal" not in x][0]
        A("**%s** (tres semillas, valores por defecto; `%s`)." % (tit, reg))
        A("")
        pv2 = c_.pivot(index="metodo", columns="rho", values="auc_media_semillas")
        A(tabla(["Método"] + ["%.2f" % r for r in pv2.columns], [[NOM[m]] + [f3(pv2.loc[m, r]) for r in pv2.columns] for m in ORD if m in pv2.index]))
        A("")
        A(tabla(["Medida", "Método", "rho", "Estimación (IC95)", "Análisis principal (IC95)", "Mismo signo"],
                [("caída de ERM entre 0.5 y 1.0" if r.medida == "caida_050_100" else "diferencia frente a ERM", NOM[r.metodo],
                  "no aplica" if pd.isna(r.rho) else "%.2f" % r.rho, "%.3f (%s)" % (getattr(r, e), ic(getattr(r, i1), getattr(r, i2))),
                  "%.3f (%s)" % (r.estimacion_principal, ic(r.ic95_inf_principal, r.ic95_sup_principal)), "sí" if r.mismo_signo else "no")
                 for r in k.itertuples()]))
        A("")
    A("Lectura. La caída de ERM se reproduce con otro tronco y con otra etiqueta. La ventaja de GroupDRO y de LISA en rho 0.9 se "
      "mantiene en los dos análisis; la de DFR en rho 0.99 no se reproduce (cambia de signo en los dos). La estabilidad de la "
      "búsqueda (S-B) está en docs/13, Tabla S1.")
    A("")
    # ------------------------------------------------------------------ incidencias y límites
    A("## 6. Incidencias de la ejecución")
    A("")
    A("- Las 46 corridas del control de fuente en B3a terminaron con código 1 al no existir su JSON congelado. El método no es "
      "aplicable en B3a (docs/08, enmiendas a la versión 2) y no entra en ningún análisis; las cadenas siguieron con `--continuar`.")
    A("- Corte de luz del 18-11-2026 a las 10:00 (hora de Lima). Dos corridas interrumpidas se relanzaron desde cero con "
      "`--no-resume`; las 275 ya terminadas se comprobaron íntegras; la congelación se verificó de nuevo. El ejecutor quedó "
      "bloqueado unas horas por bytes nulos en el registro de temperaturas, sin efecto en los resultados.")
    A("- Recuento final con código 0: B3a 506, B2 60, diez semillas 165, C3 parte b 96, S-T 45, S-E 45; test de COVIDx y COVIDGR "
      "evaluados desde los 60 estados congelados.")
    A("")
    A("## 7. Límites que el artículo debe declarar")
    A("")
    A("- C2 usa una etiqueta y un par de fuentes como análisis principal; S-E es pequeño (un cuarto del tamaño de B3a).")
    A("- Las configuraciones se eligieron con una sola semilla y la búsqueda es inestable entre semillas (S-B).")
    A("- La ventaja de los métodos que superan a ERM es pequeña y desaparece con confusión total; DFR no es robusto al cambio de "
      "tronco ni de etiqueta en rho 0.99.")
    A("- C2-R no reproduce el patrón de métodos ganadores de B3a.")
    A("- Hubo cuatro usos previos de información del test de COVIDx, declarados en docs/08 (enmienda 6 a la versión 2).")
    A("- La no inferioridad de D_IN no se alcanzaba formalmente en validación (257 negativas); el resultado confirmatorio es el del test.")
    txt = "\n".join(L) + "\n"
    assert "—" not in txt and "§" not in txt
    out = os.path.join(R, "docs", "15_resultados_plan_b.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(txt)
    print("escrito", out, len(txt.splitlines()), "líneas")


if __name__ == "__main__":
    main()
