#!/usr/bin/env python3
"""Informes de los análisis de sensibilidad registrados en la enmienda E8 de docs/08_evaluacion.md (17-11-2026).

Tres apartados, cada uno con su orden:

- estabilidad (S-B, punto 5; solo validación, ejecutable en cuanto terminen las 32 corridas de
  `make_jobs_extra.py --fase estabilidad`): para ERM y LISA en rho 0.9 y 0.99, con la regla registrada (máxima
  early_stopping.best, empate al ensayo menor), si el ensayo elegido con la semilla 1 coincide con el de la semilla 0,
  la correlación de rangos de Spearman entre las métricas de selección de los ocho ensayos en las dos semillas y la
  diferencia de AUC (con promediado de espejo, estado best) en val_desconf entre el ensayo elegido con la semilla 1 y
  el elegido con la semilla 0, medida en las corridas de cada semilla, con IC95 por pacientes. Descriptivo; no cambia
  ninguna configuración congelada.
- tronco (S-T, punto 3) y etiqueta (S-E, punto 4): leen test_desconf y llaman antes a
  `src.eval.desbloqueo.exigir_congelacion`. Con el intervalo jerárquico de la enmienda E1 (src/eval/jerarquico.py, a
  través de src.eval.c2.curva_jerarquica), sin prueba de hipótesis: la caída de ERM entre rho 0.5 y 1.0, la diferencia
  de cada método frente a ERM en rho 0.9 y 0.99 y si el signo de cada diferencia (y el de la caída) coincide con el del
  análisis principal (ConvNeXt-Tiny, derrame pleural, results/train/b3a_A_<m>_rho<RRR>, todas sus semillas). En DFR
  se usa dfr_best y no existe en rho 1.0.

Salida en results/sensibilidad/<apartado>/: CSV por tabla, informe.md e informe.json (entradas con su número, semillas
por nivel, B y, en los apartados de test, el SHA256 de la congelación). `--demo` ejecuta los tres apartados sobre
árboles sintéticos (ningún dato real) y es la prueba de extremo a extremo.

Uso: python scripts/eval/informe_sensibilidad.py estabilidad|tronco|etiqueta|todos [--out DIR] [--B 2000]
     [--root ...] [--root-principal ...] [--root-busqueda ...] [--permitir-incompleto] [--demo]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
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

METODOS = ("erm", "groupdro", "lisa", "dfr")
NIVELES = (0.5, 0.9, 0.99, 1.0)
NIVELES_DIF = (0.9, 0.99)
ESTAB_METODOS = ("erm", "lisa")
ESTAB_NIVELES = (0.9, 0.99)
N_ENSAYOS = 8
CONJUNTO = "test_desconf"
APARTADOS = {
    "tronco": {"prefijo": "st_r50_b3a", "root": "results/sens_tronco", "tronco": "resnet50",
               "titulo": "S-T, segundo tronco (ResNet-50 de ImageNet), B3a"},
    "etiqueta": {"prefijo": "se_card_b3c", "root": "results/sens_etiqueta", "tronco": "convnext_tiny",
                 "titulo": "S-E, segunda etiqueta (cardiomegalia, B3c), ConvNeXt-Tiny"},
}


class Incompleto(SystemExit):
    pass


# ===================================================================================== S-B
def _summary(run):
    p = Path(run) / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def estabilidad(root_busqueda, root_estab, metodos=ESTAB_METODOS, niveles=ESTAB_NIVELES, B=S.DEFAULT_B,
                seed=S.DEFAULT_BOOT_SEED, permitir_incompleto=False, semillas=(0, 1)):
    """Tabla de S-B: una fila por método y nivel. Solo lee validación (best_val_desconf.csv con forbid_test)."""
    roots = {semillas[0]: Path(root_busqueda), semillas[1]: Path(root_estab)}
    filas, faltan = [], []
    for m in metodos:
        for r in niveles:
            sel, auc, preds = {s: {} for s in semillas}, {s: {} for s in semillas}, {s: {} for s in semillas}
            for s in semillas:
                for t in range(N_ENSAYOS):
                    run = roots[s] / ("b3a_A_%s_rho%03d_t%d" % (m, round(100 * r), t)) / ("seed%d" % s)
                    sj = _summary(run)
                    f = run / "predictions" / "best_val_desconf.csv"
                    if not sj or (sj.get("early_stopping") or {}).get("best") is None or not f.exists():
                        faltan.append(str(run))
                        continue
                    sel[s][t] = float(sj["early_stopping"]["best"])
                    preds[s][t] = S.read_predictions(f, forbid_test=True, context="S-B")
            if any(len(sel[s]) < N_ENSAYOS for s in semillas) and not permitir_incompleto:
                continue
            if any(len(sel[s]) < 2 for s in semillas):
                continue
            t_sel = {s: C3.elegir_regla(sel[s]) for s in semillas}
            comunes = sorted(set(sel[semillas[0]]) & set(sel[semillas[1]]))
            sp = C3._spearman([sel[semillas[0]][t] for t in comunes], [sel[semillas[1]][t] for t in comunes])
            fila = {"metodo": m, "rho": r, "ensayo_semilla%d" % semillas[0]: t_sel[semillas[0]],
                    "ensayo_semilla%d" % semillas[1]: t_sel[semillas[1]],
                    "coincide": bool(t_sel[semillas[0]] == t_sel[semillas[1]]),
                    "spearman_metricas_seleccion": float(sp) if np.isfinite(sp) else np.nan,
                    "n_ensayos_comunes": len(comunes),
                    "metrica_constante": bool(any(np.unique(np.round(list(sel[s].values()), 12)).size < 2 for s in semillas))}
            for s in semillas:
                a, b = t_sel[semillas[0]], t_sel[semillas[1]]
                ref = preds[s][a]
                pb = S.align(ref, preds[s][b], "ensayo %d" % a, "ensayo %d" % b)
                y, g = ref["label"].to_numpy(), ref["patient"].to_numpy()
                M = np.vstack([ref["prob"].to_numpy(), pb["prob"].to_numpy()])
                auc_a, auc_b = S.auc_roc(y, M[0]), S.auc_roc(y, M[1])
                Db = D.auc_remuestreo(y, M, g, B, seed)
                lo, hi = J.ic(Db[:, 1] - Db[:, 0])
                fila.update({"auc_val_desconf_s%d_ensayo_s%d" % (s, semillas[0]): float(auc_a),
                             "auc_val_desconf_s%d_ensayo_s%d" % (s, semillas[1]): float(auc_b),
                             "dif_auc_s%d" % s: float(auc_b - auc_a), "ic95_inf_s%d" % s: lo, "ic95_sup_s%d" % s: hi})
            filas.append(fila)
    if faltan and not permitir_incompleto:
        raise Incompleto("S-B incompleto: faltan %d corridas o sus ficheros (p. ej. %s); con --permitir-incompleto se "
                         "informa con los ensayos presentes y se rotula" % (len(faltan), faltan[0]))
    return pd.DataFrame(filas), faltan


# ===================================================================================== S-T y S-E
def _inventario(root, prefijo, conjunto, metodos, niveles):
    rx = re.compile(r"^%s_A_(?P<metodo>[a-z0-9_]+?)_rho(?P<rho>\d{3})$" % re.escape(prefijo))
    rows, troncos = [], {}
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        mm = rx.match(d.name)
        if not mm or mm.group("metodo") not in metodos:
            continue
        rho = int(mm.group("rho")) / 100.0
        if not any(abs(rho - x) < 1e-9 for x in niveles):
            continue
        for s, run in S.find_seed_dirs(root, d.name).items():
            sj = _summary(run) or {}
            troncos[(mm.group("metodo"), rho, int(s))] = (sj.get("tronco") or {})
            rc = D._rho_config(run)
            if rc is not None and abs(rc - rho) > 1e-9:
                raise ValueError("%s: rho del nombre (%.2f) distinto del de la configuración (%.2f)" % (run, rho, rc))
            st = "dfr_best" if mm.group("metodo") == "dfr" else "best"
            f = run / "predictions" / ("%s_%s.csv" % (st, conjunto))
            rows.append({"experimento": d.name, "metodo": mm.group("metodo"), "rho": rho, "semilla": int(s),
                         "conjunto": conjunto, "estado": st, "ruta": str(f), "existe": f.exists()})
    return pd.DataFrame(rows, columns=["experimento", "metodo", "rho", "semilla", "conjunto", "estado", "ruta",
                                       "existe"]), troncos


def _extraer(resumen, comp, metodos):
    """{('caida', 'erm'): (delta, lo, hi), ('dif', m, rho): (delta, lo, hi)}."""
    out = {}
    r = resumen[(resumen.metodo == "erm") & (resumen.medida == "caida_050_100")]
    if len(r):
        out[("caida", "erm")] = (float(r.estimacion.iloc[0]), float(r.ic95_inf.iloc[0]), float(r.ic95_sup.iloc[0]))
    for m in metodos:
        if m == "erm":
            continue
        for x in NIVELES_DIF:
            c = comp[(comp.metodo == m) & (comp.medida == "delta_auc_rho%03d" % round(100 * x))]
            if len(c):
                out[("dif", m, x)] = (float(c.delta.iloc[0]), float(c.ic95_inf.iloc[0]), float(c.ic95_sup.iloc[0]))
    return out


def _signo(v):
    return 0 if not np.isfinite(v) or v == 0 else int(np.sign(v))


def sensibilidad(apartado, root, root_principal, B=S.DEFAULT_B, seed=S.DEFAULT_BOOT_SEED, verificador=None,
                 metodos=METODOS, niveles=NIVELES, conjunto=CONJUNTO, permitir_incompleto=False,
                 comprobar_tronco=True):
    """Tablas de S-T o S-E. Devuelve (concordancia, curva, comparacion, resumen, info)."""
    ap = APARTADOS[apartado]
    reg = C2.exigir([conjunto], "informe_sensibilidad %s: %s" % (apartado, conjunto), verificador)
    inv, troncos = _inventario(root, ap["prefijo"], conjunto, metodos, niveles)
    esperadas = {(m, r, s) for m in metodos for r in niveles for s in (0, 1, 2) if not (m == "dfr" and r >= 1.0)}
    presentes = {(f.metodo, float(f.rho), int(f.semilla)) for f in inv[inv.existe].itertuples()}
    falt = sorted(esperadas - presentes)
    if falt and not permitir_incompleto:
        raise Incompleto("%s incompleto: faltan %d de %d corridas (p. ej. %s); --permitir-incompleto para informar "
                         "con las presentes" % (apartado, len(falt), len(esperadas), falt[0]))
    if comprobar_tronco:
        malos = sorted(k for k, t in troncos.items() if k in presentes and t.get("tronco") != ap["tronco"])
        if malos:
            raise SystemExit("%s: summary.json sin el tronco %s en %d corridas (p. ej. %s): ¿se lanzaron sin los parches "
                             "de logs/entrenamiento_r3/?" % (apartado, ap["tronco"], len(malos), malos[0]))
    ver = (lambda motivo: reg)
    preds = D.cargar(inv, conjunto, permitir_test=True, verificador=ver)
    curva, resumen, comp, info = C2.curva_jerarquica(preds, [m for m in metodos if any(k[0] == m for k in preds)], B,
                                                     seed)
    sens = _extraer(resumen, comp, metodos)
    inv_p = D.descubrir(root_principal, "b3a", "A", "best", (conjunto,), permitir_test=True)
    inv_p = inv_p[inv_p.metodo.isin(list(metodos)) & inv_p.rho.isin(list(niveles))]
    preds_p = D.cargar(inv_p, conjunto, permitir_test=True, verificador=ver)
    prin = {}
    if preds_p and any(k[0] == "erm" for k in preds_p):
        _, res_p, comp_p, info_p = C2.curva_jerarquica(preds_p, [m for m in metodos if any(k[0] == m for k in preds_p)],
                                                       B, seed)
        prin = _extraer(res_p, comp_p, metodos)
        info["semillas_principal"] = info_p["semillas_por_nivel"]
    filas = []
    for k in [("caida", "erm")] + [("dif", m, x) for m in metodos if m != "erm" for x in NIVELES_DIF]:
        a = sens.get(k, (np.nan,) * 3)
        b = prin.get(k, (np.nan,) * 3)
        sa, sb = _signo(a[0]), _signo(b[0])
        filas.append({"medida": "caida_050_100" if k[0] == "caida" else "delta_auc_frente_a_erm", "metodo": k[1],
                      "rho": np.nan if k[0] == "caida" else k[2],
                      "estimacion_%s" % apartado: a[0], "ic95_inf_%s" % apartado: a[1], "ic95_sup_%s" % apartado: a[2],
                      "estimacion_principal": b[0], "ic95_inf_principal": b[1], "ic95_sup_principal": b[2],
                      "mismo_signo": (bool(sa == sb) if (sa and sb) else None),
                      "ic95_%s_excluye_cero" % apartado: bool(np.isfinite(a[1]) and (a[1] > 0 or a[2] < 0))})
    info.update({"apartado": apartado, "titulo": ap["titulo"], "conjunto": conjunto, "corridas": int(inv.existe.sum()),
                 "faltan": [list(f) for f in falt], "sha256_congelacion": (reg or {}).get("sha256_congelacion"),
                 "troncos": sorted({json.dumps(t, sort_keys=True) for t in troncos.values() if t}),
                 "corridas_principal": int(len(preds_p)), "prueba_de_hipotesis": False})
    return pd.DataFrame(filas), curva, comp, resumen, info


# ===================================================================================== salida
def _md(df, cifras=3):
    return D.a_markdown(df, cifras) if len(df) else "(sin filas)\n"


def escribir(out, nombre, tablas: dict, info: dict, texto: str):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    partes = ["# %s\n" % nombre, texto, ""]
    for k, df in tablas.items():
        df.to_csv(out / ("%s.csv" % k), index=False)
        partes += ["## %s\n" % k, _md(df, 4), ""]
    (out / "informe.md").write_text("\n".join(partes), encoding="utf-8")
    (out / "informe.json").write_text(json.dumps(info, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


TEXTO_SB = ("S-B (docs/08, enmienda E8, punto 5). Validación solamente; descriptivo. La regla registrada elige el ensayo "
            "con la máxima early_stopping.best (empate al menor). dif_auc_s<S> es la AUC en val_desconf (estado best, "
            "promediado de espejo) del ensayo elegido con la semilla 1 menos la del elegido con la semilla 0, en las "
            "corridas de la semilla S; IC95 por pacientes. Las configuraciones congeladas siguen siendo las de la "
            "semilla 0.")
TEXTO_ST = ("%s (docs/08, enmienda E8). test_desconf, estado best (dfr_best en DFR), IC95 jerárquico de la enmienda E1, "
            "sin prueba de hipótesis. mismo_signo compara con el análisis principal (ConvNeXt-Tiny, derrame pleural) "
            "con todas sus semillas; vacío si alguna de las dos cifras es cero o falta.")


def correr(apartado, a, verificador=None):
    t0 = time.time()
    if apartado == "estabilidad":
        df, faltan = estabilidad(a.root_busqueda, a.root_estab, B=a.B, seed=a.seed, permitir_incompleto=a.permitir_incompleto)
        info = {"apartado": "estabilidad", "B": a.B, "semilla_bootstrap": a.seed, "faltan": faltan,
                "incompleto": bool(faltan), "raices": [str(a.root_busqueda), str(a.root_estab)]}
        escribir(Path(a.out) / "estabilidad", "S-B, estabilidad de la búsqueda", {"estabilidad": df}, info, TEXTO_SB)
        print("estabilidad: %d filas en %s" % (len(df), Path(a.out) / "estabilidad"))
        return df
    root = getattr(a, "root_" + apartado) or str(ROOT / APARTADOS[apartado]["root"])
    conc, curva, comp, res, info = sensibilidad(apartado, root, a.root_principal, a.B, a.seed, verificador,
                                                permitir_incompleto=a.permitir_incompleto,
                                                comprobar_tronco=not getattr(a, "sin_comprobar_tronco", False))
    info["segundos"] = round(time.time() - t0, 1)
    escribir(Path(a.out) / apartado, APARTADOS[apartado]["titulo"],
             {"concordancia": conc, "curva": curva, "comparacion_erm": comp, "resumen": res}, info,
             TEXTO_ST % APARTADOS[apartado]["titulo"])
    print("%s: %d filas de concordancia en %s" % (apartado, len(conc), Path(a.out) / apartado))
    return conc


# ===================================================================================== demostración sintética
def _demo_sb(raiz: Path, n=200):
    """Árboles sintéticos de S-B: dos semillas, ocho ensayos, métrica de selección y predicciones en val_desconf."""
    rng0 = np.random.default_rng(11)
    y = np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]
    base = pd.DataFrame({"image_id": ["v%04d" % i for i in range(n)], "label": y,
                         "source": np.where(np.arange(n) % 2 == 0, "nih", "chexpert"),
                         "patient": ["p%04d" % i for i in range(n)]})
    for s, sub in ((0, "busqueda"), (1, "estab")):
        for m in ESTAB_METODOS:
            for r in ESTAB_NIVELES:
                for t in range(N_ENSAYOS):
                    calidad = 0.6 + 0.08 * t + 0.03 * (m == "lisa") - 0.5 * (r > 0.95)
                    rng = np.random.default_rng([s, t, int(100 * r), len(m)])
                    d = raiz / sub / ("b3a_A_%s_rho%03d_t%d" % (m, round(100 * r), t)) / ("seed%d" % s)
                    (d / "predictions").mkdir(parents=True, exist_ok=True)
                    sel = 0.9 + 0.01 * ((t * (3 if s else 5)) % 8) + rng.normal(0, 0.001)
                    (d / "summary.json").write_text(json.dumps({"early_stopping": {"best": sel}, "sintetico": True}))
                    p = base.copy()
                    z = calidad * (2 * y - 1) + rng0.normal(0, 1, n)
                    p["prob"] = 1 / (1 + np.exp(-z))
                    p.to_csv(d / "predictions" / "best_val_desconf.csv", index=False, float_format="%.8f")
    return raiz / "busqueda", raiz / "estab"


def _demo_sens(raiz: Path, apartado):
    prin = D.arbol_sintetico_b3(raiz / "principal", "informativo", metodos=METODOS, semillas=(0, 1, 2, 3, 4))
    tmp = D.arbol_sintetico_b3(raiz / ("_%s" % apartado), "informativo", metodos=METODOS, rhos=NIVELES,
                               semillas=(0, 1, 2), seed=5)
    dest = raiz / apartado
    dest.mkdir(parents=True, exist_ok=True)
    for d in sorted(p for p in tmp.iterdir() if p.is_dir()):
        nuevo = dest / ("%s_%s" % (APARTADOS[apartado]["prefijo"], d.name[len("b3a_"):]))
        d.rename(nuevo)
        for run in nuevo.iterdir():
            (run / "summary.json").write_text(json.dumps({"tronco": {"tronco": APARTADOS[apartado]["tronco"],
                                                                     "etiqueta_pesos": "sintetico"}, "sintetico": True}))
    return dest, prin


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("apartado", choices=["estabilidad", "tronco", "etiqueta", "todos"])
    ap.add_argument("--out", default=str(ROOT / "results" / "sensibilidad"))
    ap.add_argument("--B", type=int, default=S.DEFAULT_B)
    ap.add_argument("--seed", type=int, default=S.DEFAULT_BOOT_SEED)
    ap.add_argument("--root-busqueda", default=str(ROOT / "results" / "train"), help="S-B: búsqueda original, semilla 0")
    ap.add_argument("--root-estab", default=str(ROOT / "results" / "estabilidad_b3a"), help="S-B: semilla 1")
    ap.add_argument("--root-tronco", default=None)
    ap.add_argument("--root-etiqueta", default=None)
    ap.add_argument("--root-principal", default=str(ROOT / "results" / "train"), help="finales de B3a (ConvNeXt, derrame)")
    ap.add_argument("--permitir-incompleto", action="store_true")
    ap.add_argument("--demo", action="store_true", help="árboles sintéticos; ningún dato real")
    a = ap.parse_args(argv)
    verificador = None
    if a.demo:
        raiz = Path(tempfile.mkdtemp(prefix="sens_demo_"))
        a.root_busqueda, a.root_estab = _demo_sb(raiz)
        for x in ("tronco", "etiqueta"):
            r, p = _demo_sens(raiz / x, x)
            setattr(a, "root_" + x, str(r))
            a.root_principal = str(p)
        verificador = lambda motivo: {"demostracion_sintetica": motivo, "sha256_congelacion": None}
    apartados = ("estabilidad", "tronco", "etiqueta") if a.apartado == "todos" else (a.apartado,)
    for x in apartados:
        if a.demo and x != "estabilidad":
            a.root_principal = str(Path(getattr(a, "root_" + x)).parent / "principal")
        correr(x, a, verificador)
    return 0


if __name__ == "__main__":
    sys.exit(main())
