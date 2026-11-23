"""Intervalos alternativos registrados en la enmienda E1 de docs/08_evaluacion.md que no estaban calculados, una orden
por tabla. Código posterior a la congelación del protocolo para análisis registrados antes de ella (docs/17); no
sustituye a ningún fichero congelado: importa y llama a las funciones de `src.eval` sin cambiarlas.

E1, párrafo "Sensibilidades registradas, que se publican junto al principal y no lo sustituyen": (i) el intervalo de
pacientes con los modelos fijos; (ii) el remuestreo doble sin corrección (`jerarquico.extraer(..., modo="doble")`);
(iii) la prueba t pareada entre semillas con K - 1 grados de libertad (`jerarquico.t_pareada`).

    python scripts/sensibilidad/intervalos.py c3-epoca      # pérdida por selección de época (ecuación (18), F-C3)
    python scripts/sensibilidad/intervalos.py c3-epoca --presupuesto-s 120   # se detiene a los 120 s (código 3);
                                                            # lo hecho queda en caché y basta con repetir la orden
    python scripts/sensibilidad/intervalos.py c3-epoca --sin-cache            # recalcula los cubos
    python scripts/sensibilidad/intervalos.py c1b-jerarquico  # F-C1b (D_ERM por semilla) con IC95 jerárquico (E1)

Método. Las predicciones se leen con `src.eval.c3._estados_epoca` (la misma lectura que `informe_c2_c3.py c3-epoca`,
best y bestdc; en DFR los estados de su etapa ERM), tras `src.eval.desbloqueo.exigir_congelacion`, cuyo registro real
se pasa como verificador. Los cubos de AUC puntual y remuestreada por pacientes son los de `src.eval.c2.cubo` (B =
2 000, semilla 2026, referencia de alineado la primera clave ordenada, como en `c3.seleccion_epoca`); se guardan por
método y estado en logs/sensibilidad_intervalos/cache/. Las multiplicidades de semillas son las de
`jerarquico.multiplicidades` con la presencia por nivel y semilla de todos los métodos y la semilla 20261117, como en
`c3.seleccion_epoca`. Con esos mismos objetos se obtienen:

- principal (control): `jerarquico.extraer(X, Xp, pres, Cm)` (modo corregido), que debe reproducir
  results/c2c3/c3-epoca.csv y c3_epoca_por_metodo.csv;
- (i) pacientes con modelos fijos: `jerarquico.media_semillas(X, pres)` (sin término de semillas);
- (ii) doble sin corrección: `jerarquico.extraer(X, Xp, pres, Cm, modo="doble")`;
- (iii) t pareada: `jerarquico.t_pareada` sobre las diferencias por semilla (por método, como ya hacía
  `seleccion_epoca`) y, para la media de los métodos, sobre la media por semilla de los métodos presentes en esa
  semilla (solo semillas en que están todos los métodos).

La media sobre los métodos de cada remuestreo es `np.nanmean` sobre los métodos presentes, igual que el principal.
Valores p unilaterales: `jerarquico.p_mayor` (ecuación (3)) para (i), como la columna p_unilateral_pacientes de
`c2.curva_jerarquica`; el remuestreo doble no lleva valor p, como en las tablas de C2. Holm sobre los tres niveles de
F-C3 con cada valor p alternativo (`stats.holm`, 0.025), solo como lectura de sensibilidad.
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

from src.eval import c2 as C2  # noqa: E402
from src.eval import c3 as C3  # noqa: E402
from src.eval import dosis as D  # noqa: E402
from src.eval import jerarquico as J  # noqa: E402
from src.eval import stats as S  # noqa: E402

OUT = ROOT / "results" / "sensibilidad_intervalos"
CACHE = ROOT / "logs" / "sensibilidad_intervalos" / "cache"
B, SEMILLA = S.DEFAULT_B, S.DEFAULT_BOOT_SEED
METODOS_C3 = [m for m in D.METODOS_B3 if m != "control_fuente"]      # como informe_c2_c3.py c3-epoca
ORDEN = "python scripts/sensibilidad/intervalos.py %s"
TOL = 1e-12


class Presupuesto(Exception):
    pass


class Ctx:
    def __init__(self, motivo, presupuesto_s=0.0, usar_cache=True):
        from src.eval.desbloqueo import exigir_congelacion
        self.t0 = time.time()
        self.presupuesto = float(presupuesto_s or 0)
        self.usar_cache = usar_cache
        self.registro = exigir_congelacion("intervalos alternativos registrados (E1): " + motivo)
        self.t_exigir = round(time.time() - self.t0, 1)
        CACHE.mkdir(parents=True, exist_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)

    def verificador(self, motivo):
        return self.registro                                   # el registro real que devolvió exigir_congelacion

    def vigilar(self):
        if self.presupuesto and time.time() - self.t0 > self.presupuesto:
            raise Presupuesto("presupuesto de %.0f s agotado" % self.presupuesto)


# ====================================================================================== cubos de c3-epoca
def _cubos_c3(ctx, root, metodos=METODOS_C3, conjunto="test_desconf"):
    """{método: (pb, db, rb, pc, dc, rc)} con `C2.cubo` sobre best y bestdc, alineados a la misma referencia que
    `c3.seleccion_epoca` (la primera clave ordenada de best: ('coral', 0.5, 0))."""
    sha = ctx.registro.get("sha256_congelacion")
    niveles, semillas = list(D.RHOS), list(range(10))
    out, falta = {}, []
    for m in metodos:
        f = CACHE / ("c3_%s_%s.npz" % (conjunto, m))
        if ctx.usar_cache and f.exists():
            z = np.load(f, allow_pickle=False)
            if (int(z["B"]) == B and int(z["seed"]) == SEMILLA and str(z["sha"]) == sha
                    and list(z["niveles"]) == niveles and list(z["semillas"]) == semillas):
                out[m] = tuple(z[k] for k in ("pb", "db", "rb", "pc", "dc", "rc"))
                continue
        falta.append(m)
    if not falta:
        return out, niveles, semillas, None
    ref_m = sorted(metodos)[0]
    claves = {}
    for m in falta:
        ctx.vigilar()
        carga = sorted({ref_m, m})
        pb, pc = C3._estados_epoca(root, "b3a", "A", conjunto, carga, ctx.verificador)
        ref = pb[sorted(pb)[0]]
        assert sorted(pb)[0][0] == ref_m
        nv = sorted({float(k[1]) for k in pb if k[0] == m})
        sm = sorted({int(k[2]) for k in pb if k[0] == m})
        assert set(nv) <= set(niveles) and set(sm) <= set(semillas), (m, nv, sm)
        cb = C2.cubo(pb, m, niveles, semillas, ref, B, SEMILLA)
        cc = C2.cubo(pc, m, niveles, semillas, ref, B, SEMILLA)
        np.savez(CACHE / ("c3_%s_%s.npz" % (conjunto, m)), pb=cb[0], db=cb[1], rb=cb[2], pc=cc[0], dc=cc[1], rc=cc[2],
                 B=B, seed=SEMILLA, sha=sha, niveles=np.array(niveles), semillas=np.array(semillas))
        out[m] = cb + cc
        claves[m] = {"best": len([k for k in pb if k[0] == m]), "bestdc": len([k for k in pc if k[0] == m])}
        print("cubo %s: %d best, %d bestdc (%.0f s)" % (m, claves[m]["best"], claves[m]["bestdc"], time.time() - ctx.t0),
              flush=True)
    return out, niveles, semillas, claves


def _matrices(cubos, metodos, niveles, semillas):
    """Mismas matrices que `c3.seleccion_epoca`: X (B, R, K, M), Xp (R, K, M), pres (R, K, M)."""
    R, K, M = len(niveles), len(semillas), len(metodos)
    X = np.full((int(B), R, K, M), np.nan)
    Xp = np.full((R, K, M), np.nan)
    pres = np.zeros((R, K, M), bool)
    for q, m in enumerate(metodos):
        pb, db, rb, pc, dc, rc = cubos[m]
        pr = rb & rc
        X[..., q] = np.where(pr[None], dc - db, np.nan)
        Xp[..., q] = np.where(pr, pc - pb, np.nan)
        pres[..., q] = pr
    return X, Xp, pres


def _ic(v):
    return J.ic(v)


def c3_epoca(ctx, root):
    metodos = sorted(METODOS_C3)                       # c3.seleccion_epoca ordena los métodos de las claves
    cubos, niveles, semillas, claves = _cubos_c3(ctx, root, metodos)
    X, Xp, pres = _matrices(cubos, metodos, niveles, semillas)
    # niveles y semillas que vería seleccion_epoca: los de las claves de best
    nv_presentes = [r for i, r in enumerate(niveles) if np.any([cubos[m][2][i].any() for m in metodos])]
    sm_presentes = [s for j, s in enumerate(semillas) if np.any([cubos[m][2][:, j].any() for m in metodos])]
    assert nv_presentes == niveles and sm_presentes == semillas, (nv_presentes, sm_presentes)
    pres_sem = pres.any(2)
    Cm = J.multiplicidades(pres_sem, int(B), J.SEMILLA_SEMILLAS)
    R, K, M = Xp.shape
    hj_b = np.full((int(B), R, M), np.nan)
    hp_b = np.full((int(B), R, M), np.nan)
    hd_b = np.full((int(B), R, M), np.nan)
    filas = []
    for q, m in enumerate(metodos):
        if not pres[..., q].any():
            continue
        hj = J.extraer(X[..., q], Xp[..., q], pres[..., q], Cm)
        hp = J.media_semillas(X[..., q], pres[..., q])
        hd = J.extraer(X[..., q], Xp[..., q], pres[..., q], Cm, modo="doble")
        hj_b[:, :, q], hp_b[:, :, q], hd_b[:, :, q] = hj, hp, hd
        for i, r in enumerate(niveles):
            v = Xp[i, pres[i, :, q], q]
            if not len(v):
                continue
            t = J.t_pareada(v)
            fila = {"metodo": m, "rho": r, "familia_f_c3": any(abs(r - x) < 1e-9 for x in C3.NIVELES_F_C3),
                    "delta_sel": float(v.mean()), "n_semillas": len(v),
                    "ic95_inf": _ic(hj[:, i])[0], "ic95_sup": _ic(hj[:, i])[1],
                    "ic95_inf_pacientes": _ic(hp[:, i])[0], "ic95_sup_pacientes": _ic(hp[:, i])[1],
                    "p_unilateral_pacientes": J.p_mayor(hp[:, i]),
                    "ic95_inf_doble": _ic(hd[:, i])[0], "ic95_sup_doble": _ic(hd[:, i])[1]}
            fila.update(t)
            filas.append(fila)
    por_metodo = pd.DataFrame(filas)
    filas = []
    for i, r in enumerate(niveles):
        ok = [q for q in range(M) if pres[i, :, q].any()]
        punt = float(np.mean([Xp[i, pres[i, :, q], q].mean() for q in ok]))
        mj, mp, md = (np.nanmean(a[:, i, ok], axis=1) for a in (hj_b, hp_b, hd_b))
        todas = pres[i][:, ok].all(1)                                # semillas presentes en todos los métodos
        v = np.array([Xp[i, j, ok].mean() for j in range(K) if todas[j]])
        t = J.t_pareada(v)
        fila = {"rho": r, "familia_f_c3": any(abs(r - x) < 1e-9 for x in C3.NIVELES_F_C3), "n_metodos": len(ok),
                "delta_sel_media_metodos": punt, "ic95_inf": _ic(mj)[0], "ic95_sup": _ic(mj)[1],
                "p_unilateral": J.p_mayor(mj), "ic95_inf_pacientes": _ic(mp)[0], "ic95_sup_pacientes": _ic(mp)[1],
                "p_unilateral_pacientes": J.p_mayor(mp), "ic95_inf_doble": _ic(md)[0], "ic95_sup_doble": _ic(md)[1],
                "t_n_semillas": int(todas.sum()), "semillas_con_todos_los_metodos": " ".join(
                    str(semillas[j]) for j in range(K) if todas[j]), "semillas_media_a_favor_bestdc": int((v > 0).sum())}
        fila.update(t)
        filas.append(fila)
    media = pd.DataFrame(filas)
    fam = media[media.familia_f_c3]
    for col, pcol in (("p_holm", "p_unilateral"), ("p_holm_pacientes", "p_unilateral_pacientes"),
                      ("p_holm_t", "t_p_unilateral")):
        media[col] = np.nan
        h = S.holm({i: p for i, p in zip(fam.index, fam[pcol])}, alpha=0.025)
        for i in fam.index:
            media.loc[i, col] = h[i]["p_holm"]
        media[col.replace("p_holm", "rechaza_holm")] = media[col] < 0.025
        media.loc[~media.familia_f_c3, col.replace("p_holm", "rechaza_holm")] = False
    # ------------------------------------------------------------------ control: el principal publicado
    pub_n = pd.read_csv(ROOT / "results" / "c2c3" / "c3-epoca.csv")
    pub_m = pd.read_csv(ROOT / "results" / "c2c3" / "c3_epoca_por_metodo.csv")
    ctrl = []
    a = media.set_index("rho")
    b = pub_n.set_index("rho")
    for c_mio, c_pub in (("delta_sel_media_metodos", "delta_sel_media_metodos"), ("ic95_inf", "ic95_inf"),
                         ("ic95_sup", "ic95_sup"), ("p_unilateral", "p_unilateral"), ("p_holm", "p_holm")):
        x, y = a.loc[b.index, c_mio].to_numpy(float), b[c_pub].to_numpy(float)
        ok = np.isfinite(y)
        ctrl.append({"tabla": "c3-epoca.csv", "columna": c_pub, "filas": int(ok.sum()),
                     "max_abs_dif": float(np.max(np.abs(x[ok] - y[ok]))) if ok.any() else float("nan"),
                     "nan_distintos": int((np.isfinite(x) != np.isfinite(y)).sum())})
    a = por_metodo.set_index(["metodo", "rho"])
    b = pub_m.set_index(["metodo", "rho"])
    if len(a) != len(b) or set(a.index) != set(b.index):
        raise SystemExit("las filas por método no coinciden con c3_epoca_por_metodo.csv: %d frente a %d" % (len(a), len(b)))
    for c in ("delta_sel", "ic95_inf", "ic95_sup", "n_semillas", "t_media", "t_ic95_inf", "t_ic95_sup", "t_p_unilateral",
              "t_gl"):
        x, y = a.loc[b.index, c].to_numpy(float), b[c].to_numpy(float)
        ctrl.append({"tabla": "c3_epoca_por_metodo.csv", "columna": c, "filas": len(y),
                     "max_abs_dif": float(np.nanmax(np.abs(x - y))), "nan_distintos": int((np.isfinite(x) != np.isfinite(y)).sum())})
    ctrl = pd.DataFrame(ctrl)
    ctrl["reproduce"] = (ctrl["max_abs_dif"] <= TOL) & (ctrl["nan_distintos"] == 0)
    escribir("c3_epoca", {"c3_epoca_control": ctrl, "c3_epoca_media_alternativos": media,
                          "c3_epoca_por_metodo_alternativos": por_metodo}, ctx, ORDEN % "c3-epoca",
             ["Control: el principal recalculado con los mismos cubos y multiplicidades frente a results/c2c3/; "
              "diferencia máxima %.3g." % ctrl["max_abs_dif"].max(),
              "Lectura con %s (tiempo de exigir_congelacion: %.1f s)." % (root, ctx.t_exigir)])
    if not ctrl["reproduce"].all():
        print("AVISO: el principal no se reproduce exactamente; ver c3_epoca_control.csv")
        return 1
    print("control: el principal se reproduce (diferencia máxima %.3g)" % ctrl["max_abs_dif"].max())
    return 0


# ====================================================================================== salida
def escribir(nombre, tablas, ctx, orden, notas=()):
    md = ["# %s" % nombre, "",
          "Generado por `%s` (%s UTC). B = %d, semilla de pacientes %d, semilla de semillas %d. Código posterior a la "
          "congelación para análisis registrados antes de ella (docs/17_intervalos_alternativos.md). Congelación: %s." %
          (orden, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), B, SEMILLA, J.SEMILLA_SEMILLAS,
           ctx.registro.get("sha256_congelacion")), ""]
    md += ["- " + n for n in notas] + ([""] if notas else [])
    for k, t in tablas.items():
        t.to_csv(OUT / ("%s.csv" % k), index=False)
        md += ["## %s" % k, "", D.a_markdown(t, 4) if len(t) else "(sin filas)", ""]
    (OUT / ("%s.md" % nombre)).write_text("\n".join(md), encoding="utf-8")
    import platform
    meta = {"orden": orden, "fecha_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "B": B,
            "semilla_bootstrap": SEMILLA, "semilla_semillas": J.SEMILLA_SEMILLAS,
            "congelacion": ctx.registro.get("sha256_congelacion"), "tablas": {k: len(v) for k, v in tablas.items()},
            "notas": list(notas), "segundos": round(time.time() - ctx.t0, 1), "python": platform.python_version(),
            "numpy": np.__version__, "maquina": platform.node()}
    (OUT / ("%s.json" % nombre)).write_text(json.dumps(S.jsonable(meta), indent=1, ensure_ascii=False), encoding="utf-8")
    print("escrito %s (%s)" % (OUT / ("%s.md" % nombre), ", ".join("%s: %d filas" % (k, len(v)) for k, v in tablas.items())))


# ====================================================================================== c1b-jerarquico
CACHE_V28 = ROOT / "logs" / "sensibilidad_v28" / "cache"       # solo lectura: puntuaciones del detector de v28.py


def c1b_jerarquico(ctx, root):
    """F-C1b (D_ERM frente a los doce métodos, emparejado por semilla, parte bimcv del test de COVIDx) con el IC95
    jerárquico corregido de E1, que su alcance pide "también para C1-i con D_ERM, emparejado por semilla". Mismas
    entradas que `results/c1/test/f_c1b.csv`: predicciones best (dfr_best en DFR) con
    `src.eval.c1_auditoria.Auditoria` y puntuaciones del detector de V2.3.1 sobre cada tronco D_ERM (C = 0.05,
    características agrupadas, promediado de espejo) leídas de la caché de v28.py, que la cadena sens_cadena regeneró
    en el contenedor y que reproduce f_c1b.csv exactamente (results/sensibilidad_v28/s0_equivalencia.csv)."""
    from src.eval import audit as A
    from src.eval.c1_auditoria import Auditoria, Rutas, leer_lista
    L = leer_lista(ROOT / "configs" / "gate7_lista_c1.yaml")
    aud = Auditoria("test", L, Rutas(), registro=ctx.registro)            # registro real de exigir_congelacion
    semillas = [int(x) for x in L["semillas"]]                             # orden de la lista cerrada
    conjunto = "test_bimcv"
    # ---- métodos, como comun.Ctx.metodos_en (alineados a las imágenes del primer método)
    mets, ids0, base = {}, None, None
    for m in L["metodos"]:
        fr = aud.preds(m, conjunto, ("dfr_best" if m == "dfr" else "best"))
        if len(fr) < len(semillas):
            raise SystemExit("%s: %d de %d semillas en %s" % (m, len(fr), len(semillas), conjunto))
        df, P, sem = aud.matriz(fr)
        if ids0 is None:
            ids0, base = df["image_id"].tolist(), df
        else:
            P = np.vstack([A.alinear(ids0, df["image_id"], q) for q in P])
        mets[m] = (P, [int(x) for x in sem])
    y = base["label"].to_numpy().astype(int)
    g = base["patient"].to_numpy()
    # ---- referencia por semilla: D_ERM de esa semilla
    tag = L["troncos"]["D_ERM"]
    R = {}
    for sd in semillas:
        f = CACHE_V28 / ("punt_%s_C0p05_pooled_espejo.csv" % tag.format(semilla=sd))
        if not f.exists():
            raise SystemExit("falta %s: ejecute antes python scripts/sensibilidad/v28.py s0" % f)
        pr = pd.read_csv(f, dtype={"image_id": str, "patient_id": str})
        R[sd] = A.alinear(ids0, pr["image_id"], pr["score"].to_numpy())
    ref = np.vstack([R[sd] for sd in semillas])
    ctx.vigilar()
    Dr = D.auc_remuestreo(y, ref, g, B, SEMILLA)                             # (B, K) referencia por semilla
    a_ref = np.array([S.auc_roc(y, ref[k]) for k in range(len(semillas))])
    pres = np.ones((1, len(semillas)), bool)
    pub = pd.read_csv(ROOT / "results" / "c1" / "test" / "f_c1b.csv").set_index("metodo")
    filas = []
    for m, (P, sem) in mets.items():
        ctx.vigilar()
        idx = [sem.index(sd) for sd in semillas]                            # emparejado por semilla
        P = P[idx]
        Dm = D.auc_remuestreo(y, P, g, B, SEMILLA)
        a_m = np.array([S.auc_roc(y, P[k]) for k in range(len(semillas))])
        X = (Dr - Dm)[:, None, :]                                           # (B, 1, K) diferencia por semilla
        Xp = (a_ref - a_m)[None, :]                                         # (1, K)
        hj = J.remuestreo_jerarquico(X, pres, J.SEMILLA_SEMILLAS, punt=Xp)[:, 0]
        hp = np.nanmean(Dr - Dm, axis=1)                                    # pacientes con modelos fijos (f_c1b)
        comp = J.componentes(X, Xp, pres)
        lo, hi = J.ic(hj)
        lp, hp_ = D._ic(hp)
        fila = {"metodo": m, "n_semillas": len(semillas), "semillas": ",".join(map(str, semillas)),
                "auc_referencia": float(a_ref.mean()), "auc_metodo": float(a_m.mean()), "delta": float(Xp.mean()),
                "ic95_inf_jerarquico": lo, "ic95_sup_jerarquico": hi,
                "ic95_inf_pacientes": lp, "ic95_sup_pacientes": hp_,
                "de_semillas_sigma_S": float(np.sqrt(comp["sigma2_S"][0])), "de_pacientes_V_P": float(np.sqrt(comp["V_P"][0])),
                "de_entre_semillas_s": float(np.sqrt(comp["s2"][0])), "lambda": float(comp["lambda"][0])}
        for mg in (0.02, 0.01):
            fila["li_jerarquico_sobre_menos_%.2f" % mg] = bool(np.isfinite(lo) and lo > -mg)
            fila["no_inferior_%.2f_publicado" % mg] = bool(pub.loc[m, "no_inferior_%.2f" % mg])
        fila["delta_publicado"] = float(pub.loc[m, "delta"])
        fila["ic95_inf_publicado"] = float(pub.loc[m, "ic95_inf"])
        fila["ic95_sup_publicado"] = float(pub.loc[m, "ic95_sup"])
        filas.append(fila)
        print("%s: %.4f (%.4f a %.4f) (%.0f s)" % (m, fila["delta"], lo, hi, time.time() - ctx.t0), flush=True)
    t = pd.DataFrame(filas)
    ctrl = []
    for c_mio, c_pub in (("delta", "delta_publicado"), ("ic95_inf_pacientes", "ic95_inf_publicado"),
                         ("ic95_sup_pacientes", "ic95_sup_publicado")):
        ctrl.append({"tabla": "f_c1b.csv", "columna": c_pub.replace("_publicado", ""), "filas": len(t),
                     "max_abs_dif": float(np.max(np.abs(t[c_mio] - t[c_pub])))})
    x = pub.loc[t.metodo, "auc_referencia"].to_numpy(float)
    ctrl.append({"tabla": "f_c1b.csv", "columna": "auc_referencia", "filas": len(t),
                 "max_abs_dif": float(np.max(np.abs(t["auc_referencia"].to_numpy() - x)))})
    x = pub.loc[t.metodo, "auc_metodo"].to_numpy(float)
    ctrl.append({"tabla": "f_c1b.csv", "columna": "auc_metodo", "filas": len(t),
                 "max_abs_dif": float(np.max(np.abs(t["auc_metodo"].to_numpy() - x)))})
    ctrl = pd.DataFrame(ctrl)
    ctrl["reproduce"] = ctrl["max_abs_dif"] <= 1e-12
    resumen = pd.DataFrame([{"margen": mg, "metodos": len(t),
                             "li_jerarquico_sobre_menos_margen": int(t["li_jerarquico_sobre_menos_%.2f" % mg].sum()),
                             "no_inferior_publicado_holm": int(t["no_inferior_%.2f_publicado" % mg].sum())}
                            for mg in (0.02, 0.01)])
    escribir("c1b_jerarquico", {"c1b_jerarquico": t, "c1b_jerarquico_control": ctrl, "c1b_jerarquico_resumen": resumen},
             ctx, ORDEN % "c1b-jerarquico",
             ["Control: delta e IC95 de pacientes frente a results/c1/test/f_c1b.csv; diferencia máxima %.3g." %
              ctrl["max_abs_dif"].max(),
              "Las columnas li_jerarquico_sobre_menos_* son solo una lectura de sensibilidad: la prueba registrada es "
              "la de f_c1b.csv (no inferioridad con Holm sobre doce comparaciones)."])
    if not ctrl["reproduce"].all():
        print("AVISO: no se reproduce f_c1b.csv exactamente; ver c1b_jerarquico_control.csv")
        return 1
    print("control: f_c1b.csv se reproduce (diferencia máxima %.3g)" % ctrl["max_abs_dif"].max())
    return 0


ORDENES = {"c3-epoca": c3_epoca, "c1b-jerarquico": c1b_jerarquico}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tabla", choices=sorted(ORDENES))
    ap.add_argument("--root", default=str(ROOT / "results" / "train"))
    ap.add_argument("--presupuesto-s", type=float, default=0.0)
    ap.add_argument("--sin-cache", action="store_true")
    a = ap.parse_args(argv)
    ctx = Ctx(a.tabla, a.presupuesto_s, not a.sin_cache)
    try:
        return ORDENES[a.tabla](ctx, Path(a.root))
    except Presupuesto as e:
        print("%s: lo calculado queda en %s; repita la orden" % (e, CACHE))
        return 3


if __name__ == "__main__":
    sys.exit(main())
