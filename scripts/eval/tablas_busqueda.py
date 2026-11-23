"""Regenera, desde los summary.json y las predicciones de VALIDACIÓN de las búsquedas de hiperparámetros, las tablas
de resultados/b3a, results/b2 y results/b1 y las Tablas H1 a H4 de docs/13_entrenamiento_plan_b.md (hallazgo M5 del
revisor independiente; docs/08_evaluacion.md, enmienda E6). No lee ningún test.

    python scripts/eval/tablas_busqueda.py todo --comparar            # regenera en logs/eval_r2/regenerado y compara
    python scripts/eval/tablas_busqueda.py b3a --out DIR               # solo B3a (los ocho CSV)
    python scripts/eval/tablas_busqueda.py b2 | b1 | tablas            # B2, B1 o las Tablas H1 a H4 en Markdown
    python scripts/eval/tablas_busqueda.py b3a --regla-antigua         # reproduce también busqueda_ensayo_elegido.csv
                                                                       # tal como se escribió (ver enmienda E6)

Definiciones (las mismas que fija la enmienda E6 y el texto propuesto para docs/13):
- sel: summary.json, early_stopping.best (AUC sin promediado de volteos en el conjunto de selección de la corrida:
  val, o val_sel en DFR); sel_dc: early_stopping_dc.best (en val_desconf, o val_desconf_sel en DFR).
- val, vdc: AUC con promediado de volteos (summary.json, results) del estado principal (best; dfr_best en DFR) en
  val y val_desconf; si DFR no tiene dfr_best (rho = 1.0), val es la de best y vdc queda vacía.
- vdc_etapa_erm: best/val_desconf; vdc_oraculo: bestdc/val_desconf (en DFR, el bestdc de su etapa ERM).
- Regla registrada (scripts/jobs/bench_jobs.py, elegir): máximo de sel, empate al ensayo menor (comparación estricta
  de coma flotante, como en elegir).
Las cifras de DFR en val y val_desconf contienen sus datos de reajuste (enmienda E5): se reproducen, pero no son
comparables con los demás métodos.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import stats as S  # noqa: E402

RHOS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
ORDEN_REGLA = ("erm", "groupdro", "dann", "coral", "mixstyle", "fourier", "lisa", "dfa_fuente", "dfr", "erm_mascara",
               "erm_cuotas")                                            # orden de bench_jobs (sin control de fuente)
ORDEN_DOCS = ("erm", "erm_cuotas", "erm_mascara", "groupdro", "dann", "coral", "mixstyle", "fourier", "lisa",
              "dfa_fuente", "dfr")                                      # orden de las Tablas H1 y H2
ORDEN_B2 = ("erm", "control_fuente", "groupdro", "dann", "coral", "mixstyle", "fourier", "lisa", "dfa_fuente", "dfr",
            "erm_mascara", "erm_cuotas")
ORDEN_B1 = ("erm", "erm_cuotas", "erm_mascara", "control_fuente", "groupdro", "dann", "coral", "mixstyle", "fourier",
            "lisa", "dfa_fuente", "dfr")
NOMBRES = {"erm": "ERM", "erm_cuotas": "ERM con cuotas", "erm_mascara": "ERM con máscara", "groupdro": "GroupDRO",
           "dann": "DANN", "coral": "CORAL", "mixstyle": "MixStyle", "fourier": "Fourier", "lisa": "LISA",
           "dfa_fuente": "DFA con fuente", "dfr": "DFR", "control_fuente": "Control de fuente"}
TOL_EMPATE = 1e-9
TOL_COMPARACION = 1e-12
TRAIN = ROOT / "results" / "train"
DIFERENCIAS_CONOCIDAS = {"busqueda_ensayo_elegido.csv": "columna 1.0 escrita con idxmax en el orden del disco; la regla "
                         "registrada y results/b3a/congelados dan el ensayo 0 en los empates (enmienda E6); "
                         "--regla-antigua reproduce el fichero"}


def _auc_res(res, clave):
    v = (res.get(clave) or {}).get("auc_roc")
    return float(v) if v is not None else np.nan


def _elegir(g, col="sel"):
    """Regla registrada sobre un grupo de ensayos ordenados por ensayo: máximo estricto, empate al menor."""
    best = None
    for r in g.sort_values("trial").itertuples():
        v = getattr(r, col)
        if not np.isfinite(v):
            continue
        if best is None or v > best[0]:
            best = (v, r.Index)
    return None if best is None else best[1]


# ===================================================================================== B3a
def b3a_todas(train=TRAIN):
    rows = []
    for sp in sorted(glob.glob(str(Path(train) / "b3a_A_*_rho*_t*" / "seed0" / "summary.json"))):
        m = re.search(r"b3a_A_(.+)_rho(\d{3})_t(\d)/", sp)
        met, rho, t = m.group(1), int(m.group(2)) / 100, int(m.group(3))
        s = json.load(open(sp, encoding="utf-8"))
        R = s.get("results") or {}
        st = "dfr_best" if met == "dfr" else "best"
        val = _auc_res(R, st + "/val")
        if not np.isfinite(val):
            val = _auc_res(R, "best/val")
        rows.append({"metodo": met, "rho": rho, "trial": t, "estado": s.get("estado"),
                     "sel": float((s.get("early_stopping") or {}).get("best", np.nan)),
                     "sel_dc": float((s.get("early_stopping_dc") or {}).get("best", np.nan)), "val": val,
                     "vdc": _auc_res(R, st + "/val_desconf"), "vdc_etapa_erm": _auc_res(R, "best/val_desconf"),
                     "vdc_oraculo": _auc_res(R, "bestdc/val_desconf"),
                     "mejor_epoca": int((s.get("early_stopping") or {}).get("best_epoch", -1))})
    return pd.DataFrame(rows).sort_values(["metodo", "rho", "trial"]).reset_index(drop=True)


def _auc_y_dentro(path):
    from sklearn.metrics import roc_auc_score     # la implementación con la que se escribió el fichero original
    d = pd.read_csv(path)
    a = float(roc_auc_score(d["label"], d["prob"]))
    w = [float(roc_auc_score(g["label"], g["prob"])) for _, g in d.groupby("source") if g["label"].nunique() == 2]
    return a, (float(np.mean(w)) if w else np.nan), len(w)


def b3a_criterios(todas, train=TRAIN):
    """Criterios de selección alternativos (exploratorio, Tabla H3): AUC en val del estado principal (dfr_best en DFR,
    que contiene val_fit) y AUC media dentro de cada fuente con las dos clases. Excluye DFR en rho = 1.0."""
    rows = []
    for r in todas.itertuples():
        st = "dfr_best" if r.metodo == "dfr" else "best"
        f = Path(train) / ("b3a_A_%s_rho%03d_t%d" % (r.metodo, round(100 * r.rho), r.trial)) / "seed0" / "predictions" / ("%s_val.csv" % st)
        if not f.exists():
            continue
        a, w, n = _auc_y_dentro(f)
        rows.append({"metodo": r.metodo, "rho": r.rho, "trial": r.trial, "sel": r.sel, "auc_val": a, "auc_dentro": w,
                     "n_fuentes_2clases": n, "vdc": r.vdc})
    sel = pd.DataFrame(rows)
    res = []
    for (m, rho), g in sel.groupby(["metodo", "rho"]):
        g = g.sort_values("trial")
        i = _elegir(g, "sel")
        j = _elegir(g, "auc_dentro")
        conj = g.loc[i, "vdc"]
        dent = g.loc[j, "vdc"] if j is not None else np.nan
        res.append({"metodo": m, "rho": rho, "conjunta": conj, "dentro": dent, "oraculo_ensayo": g["vdc"].max(),
                    "peor_ensayo": g["vdc"].min(), "gan": dent - conj})
    return sel, pd.DataFrame(res)


def b3a_elegidos(todas, orden=ORDEN_REGLA):
    rows = []
    for m in orden:
        for rho in RHOS:
            g = todas[(todas.metodo == m) & (todas.rho == rho)]
            if not len(g):
                continue
            i = _elegir(g, "sel")
            mx = g["sel"].max()
            rows.append({"metodo": m, "rho": rho, "trial": int(g.loc[i, "trial"]), "sel": g.loc[i, "sel"],
                         "val": g.loc[i, "val"], "vdc": g.loc[i, "vdc"],
                         "empates": int((g["sel"] >= mx - TOL_EMPATE).sum())})
    return pd.DataFrame(rows)


def _pivote(df, col, orden=ORDEN_DOCS, redondeo=3):
    p = df.pivot(index="metodo", columns="rho", values=col).reindex(list(orden))
    p.columns = [str(c) for c in p.columns]
    p.index.name = "metodo"
    return p.round(redondeo) if redondeo is not None else p


def b3a_tablas(todas, regla_antigua=False):
    det = b3a_elegidos(todas)
    out = {"busqueda_elegidos_detalle.csv": det}
    ens = _pivote(det, "trial", redondeo=None).astype("Int64")
    if regla_antigua:
        # reproducción del fichero anterior: idxmax de sel en el orden en que el sistema de ficheros
        # devuelve los directorios (sin ordenar por ensayo); difiere de la regla registrada solo en los empates de rho 1.0
        filas = []
        for sp in glob.glob(str(TRAIN / "b3a_A_*_rho*_t*" / "seed0" / "summary.json")):
            m = re.search(r"b3a_A_(.+)_rho(\d{3})_t(\d)/", sp)
            filas.append((m.group(1), int(m.group(2)) / 100, int(m.group(3))))
        orden_disco = pd.DataFrame(filas, columns=["metodo", "rho", "trial"]).merge(todas, on=["metodo", "rho", "trial"])
        idx = orden_disco.groupby(["metodo", "rho"])["sel"].idxmax()
        ant = orden_disco.loc[idx, ["metodo", "rho", "trial"]]
        out["busqueda_ensayo_elegido_regla_antigua.csv"] = _pivote(ant, "trial", redondeo=None).astype("Int64")
    out["busqueda_ensayo_elegido.csv"] = ens
    out["busqueda_val_desconf_elegido.csv"] = _pivote(det, "vdc")
    out["busqueda_val_elegido.csv"] = _pivote(det, "val")
    orac = []
    for (m, rho), g in todas.groupby(["metodo", "rho"]):
        i = _elegir(g, "sel_dc")
        orac.append({"metodo": m, "rho": rho, "vdc_oraculo": g.loc[i, "vdc_oraculo"]})
    out["busqueda_val_desconf_oraculo.csv"] = _pivote(pd.DataFrame(orac), "vdc_oraculo")
    return out


def b3a(out_dir, regla_antigua=False):
    todas = b3a_todas()
    sel, res = b3a_criterios(todas)
    t = {"busqueda_todas.csv": todas, "busqueda_criterios_seleccion.csv": sel, "busqueda_criterios_resumen.csv": res}
    t.update(b3a_tablas(todas, regla_antigua))
    return _escribir(out_dir / "b3a", t)


# ===================================================================================== B2
def b2(out_dir, train=TRAIN):
    rows = []
    for m in ORDEN_B2:
        for t in range(8):
            sp = Path(train) / ("b2_%s_t%d" % (m, t)) / "seed0" / "summary.json"
            if not sp.exists():
                rows.append({"metodo": m, "trial": t, "estado": "falta"})
                continue
            s = json.load(open(sp, encoding="utf-8"))
            R = s.get("results") or {}
            st = "dfr_best" if m == "dfr" else "best"
            rows.append({"metodo": m, "trial": t, "estado": s.get("estado"), "elegido": False,
                         "sel": float((s.get("early_stopping") or {}).get("best", np.nan)), "val": _auc_res(R, st + "/val"),
                         "val_robust": _auc_res(R, st + "/val_robust"),
                         "val_robust_oraculo": _auc_res(R, "bestdc/val_robust"), "claves": np.nan})
    d = pd.DataFrame(rows)
    for m, g in d.groupby("metodo"):
        i = _elegir(g, "sel")
        if i is not None:
            d.loc[i, "elegido"] = True
    return _escribir(out_dir / "b2", {"busqueda_todas.csv": d})


# ===================================================================================== B1
def _b1_run(m, t):
    return "baseline_%s" % m if t == 0 else "baseline_%s_t%d" % (m, t)


def b1_todas(train=TRAIN):
    rows = []
    for m in ORDEN_B1:
        for t in range(8):
            run = Path(train) / _b1_run(m, t) / "seed42"
            sp = run / "summary.json"
            if not sp.exists():
                rows.append({"metodo": m, "trial": t, "falta": True})
                continue
            s = json.load(open(sp, encoding="utf-8"))
            R = s.get("results") or {}
            cfg = json.load(open(run / "config_resolved.json", encoding="utf-8"))
            st = "dfr_best" if m == "dfr" else "best"
            rows.append({"metodo": m, "trial": t, "falta": False, "sel": float(s["early_stopping"]["best"]),
                         "mejor_epoca": int(s["early_stopping"]["best_epoch"]), "vr_sel": _auc_res(R, st + "/val_robust_sel"),
                         "vr": _auc_res(R, st + "/val_robust"), "val": _auc_res(R, st + "/val"),
                         "lr_B": float(cfg["train"]["phases"][-1]["lr"])})
    return pd.DataFrame(rows)


def b1_elegidos(todas):
    rows = []
    for m in ORDEN_B1:
        g = todas[(todas.metodo == m) & (~todas.falta)]
        i = _elegir(g, "sel")
        d0 = g[g.trial == 0].iloc[0]
        e = g.loc[i]
        rows.append({"metodo": m, "ensayo": int(e.trial), "lr_B": "%.1e" % e.lr_B, "sel_sin_TTA": round(e.sel, 3),
                     "val_robust_sel": round(e.vr_sel, 3), "val_robust": round(e.vr, 3) if np.isfinite(e.vr) else np.nan,
                     "val": round(e.val, 3), "defecto_sel": round(d0.sel, 3), "defecto_val_robust_sel": round(d0.vr_sel, 3),
                     "rango_sel": "%.3f a %.3f" % (g.sel.min(), g.sel.max())})
    return pd.DataFrame(rows)


def b1(out_dir):
    todas = b1_todas()
    return _escribir(out_dir / "b1", {"busqueda_todas.csv": todas, "busqueda_elegidos.csv": b1_elegidos(todas)})


# ===================================================================================== Tablas H1 a H4
def _md(df, cols, fmt="%.3f"):
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines) + "\n"


def _f(v, fmt="%.3f", vacio="no aplicable"):
    return vacio if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt % v


def tablas_md(out_dir):
    todas = b3a_todas()
    t = b3a_tablas(todas)
    sel, res = b3a_criterios(todas)
    cols = ["Método"] + ["%.2f" % r for r in RHOS]
    h1 = [[NOMBRES[m]] + [_f(t["busqueda_val_desconf_elegido.csv"].loc[m, str(r)]) for r in RHOS] for m in ORDEN_DOCS]
    h2 = [[NOMBRES[m]] + [_f(t["busqueda_val_desconf_oraculo.csv"].loc[m, str(r)]) for r in RHOS] for m in ORDEN_DOCS]
    m3 = res.groupby("rho")[["peor_ensayo", "conjunta", "dentro", "oraculo_ensayo"]].mean()
    h3 = [["%.2f" % r] + [_f(m3.loc[r, c], vacio="no definida") for c in ("peor_ensayo", "conjunta", "dentro", "oraculo_ensayo")]
          for r in RHOS]
    from scipy.stats import spearmanr
    sp = []
    for r in RHOS:
        v, w = [], []
        for m, g in sel[sel.rho == r].groupby("metodo"):
            if g.sel.nunique() > 1:
                v.append(spearmanr(g.sel, g.vdc).correlation)
            if g.auc_dentro.notna().sum() > 1:
                w.append(spearmanr(g.auc_dentro, g.vdc).correlation)
        sp.append(["%.2f" % r, _f(np.mean(v) if v else np.nan, "%.2f", "no definida"),
                   _f(np.min(v) if v else np.nan, "%.2f", "no definida"), _f(np.max(v) if v else np.nan, "%.2f", "no definida"),
                   _f(np.mean(w) if w else np.nan, "%.2f", "no definida"), str(len(v))])
    b1t = b1_todas()
    e = b1_elegidos(b1t)
    h4 = [[NOMBRES[r.metodo], str(r.ensayo), "%.2g" % float(b1t[(b1t.metodo == r.metodo) & (b1t.trial == r.ensayo)].lr_B.iloc[0]),
           "%.3f" % r.val_robust_sel, _f(r.val_robust, vacio="no comparable"), "%.3f" % r.defecto_val_robust_sel]
          for r in e.itertuples()]
    md = ["# Tablas H1 a H4 regeneradas (`python scripts/eval/tablas_busqueda.py tablas`)", "",
          "Cifras de validación, una semilla, sin intervalos. Definiciones en la enmienda E6 de docs/08.", "",
          "**Tabla H1.** AUC en `val_desconf` del ensayo elegido con la regla registrada.", "", _md(h1, cols),
          "**Tabla H2.** Oráculo: ensayo elegido con `early_stopping_dc.best` y AUC del estado `bestdc` en `val_desconf` "
          "(en DFR, el `bestdc` de su etapa ERM).", "", _md(h2, cols),
          "**Tabla H3.** Media sobre los métodos de la AUC en `val_desconf` (estado `best`; `dfr_best` en DFR; en rho "
          "1.0, diez métodos).", "",
          _md(h3, ["rho", "Peor ensayo", "Regla registrada", "AUC media dentro de cada fuente", "Mejor ensayo (oráculo)"]),
          "**Spearman entre ensayos** (8 por método) de la métrica de selección con la AUC en `val_desconf`: media, mínimo "
          "y máximo sobre los métodos con métrica no constante, y media del criterio dentro de cada fuente.", "",
          _md(sp, ["rho", "media", "mín", "máx", "media, dentro de fuente", "métodos"]),
          "**Tabla H4.** B1: ensayo elegido y AUC del estado de mejor época.", "",
          _md(h4, ["Método", "Ensayo elegido", "Tasa de la fase B", "val_robust_sel", "val_robust",
                   "val_robust_sel con los valores por defecto"])]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tablas_H1_H4.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"tablas_H1_H4.md": out / "tablas_H1_H4.md"}, {"H1": h1, "H2": h2, "H3": h3, "H4": h4}


def _celdas_docs13(nombre):
    """Celdas numéricas de una tabla de docs/13 (solo lectura), por fila."""
    txt = (ROOT / "docs" / "13_entrenamiento_plan_b.md").read_text(encoding="utf-8").splitlines()
    i = next(k for k, l in enumerate(txt) if l.startswith("**Tabla %s.**" % nombre))
    filas, en = [], False
    for l in txt[i + 1:]:
        if l.startswith("|"):
            en = True
            if set(l.replace("|", "").strip()) <= set("-"):
                continue
            filas.append([c.strip() for c in l.strip("|").split("|")])
        elif en:
            break
    return filas[1:]


def comparar_docs13(celdas):
    difs = []
    for nombre in ("H1", "H2", "H3", "H4"):
        doc = _celdas_docs13(nombre)
        mia = celdas[nombre]
        for a, b in zip(doc, mia):
            for x, y in zip(a, b):
                if x != y:
                    difs.append((nombre, a[0], x, y))
    return difs


# ===================================================================================== escritura y comparación
def _escribir(d, tablas):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for n, df in tablas.items():
        p = d / n
        df.to_csv(p, index=not isinstance(df.index, pd.RangeIndex))
        out[n] = p
    return out


def _clave(df):
    for k in (["metodo", "rho", "trial"], ["metodo", "trial"], ["metodo", "rho"], ["metodo"]):
        if all(c in df.columns for c in k):
            return k
    return list(df.columns[:1])


def comparar(nuevo: Path, actual: Path):
    """Compara valor a valor (los ficheros nuevos se escriben ordenados; el orden de filas no cuenta)."""
    a = pd.read_csv(actual, dtype=str, keep_default_na=False)
    b = pd.read_csv(nuevo, dtype=str, keep_default_na=False)
    if list(a.columns) != list(b.columns):
        return "columnas distintas: %s frente a %s" % (list(a.columns), list(b.columns))
    k = _clave(a)
    a = a.sort_values(k).reset_index(drop=True)
    b = b.sort_values(k).reset_index(drop=True)
    if len(a) != len(b):
        return "filas: %d frente a %d" % (len(a), len(b))
    dif = (a != b)
    if not dif.values.any():
        return None
    r, c = np.nonzero(dif.values)
    peor = 0.0
    for i, j in zip(r, c):
        try:
            x, y = float(a.iat[i, j]), float(b.iat[i, j])
        except ValueError:
            x = y = None
        if x is None or not (abs(x - y) <= TOL_COMPARACION or (np.isnan(x) and np.isnan(y))):
            ej = "%s=%s: %s=%r frente a %r" % (k, a.loc[i, k].tolist(), a.columns[j], a.iat[i, j], b.iat[i, j])
            return "%d celdas distintas; p. ej. %s" % (len(r), ej)
        peor = max(peor, abs(x - y))
    return "~%d celdas difieren solo en el último dígito de coma flotante (máximo %.1e)" % (len(r), peor)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("que", choices=["b3a", "b2", "b1", "tablas", "todo"])
    ap.add_argument("--out", default=str(ROOT / "logs" / "eval_r2" / "regenerado"))
    ap.add_argument("--comparar", action="store_true", help="compara con results/<bench>/ y con docs/13")
    ap.add_argument("--regla-antigua", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    hechos = {}
    if a.que in ("b3a", "todo"):
        hechos.update({("b3a", k): v for k, v in b3a(out, a.regla_antigua or a.comparar).items()})
    if a.que in ("b2", "todo"):
        hechos.update({("b2", k): v for k, v in b2(out).items()})
    if a.que in ("b1", "todo"):
        hechos.update({("b1", k): v for k, v in b1(out).items()})
    celdas = None
    if a.que in ("tablas", "todo"):
        f, celdas = tablas_md(out)
        hechos.update({("docs13", k): v for k, v in f.items()})
    codigo = 0
    for (bench, n), p in sorted(hechos.items()):
        print("escrito %s" % p)
    if a.comparar:
        for (bench, n), p in sorted(hechos.items()):
            if bench == "docs13":
                continue
            nombre = n.replace("_regla_antigua", "")
            act = ROOT / "results" / bench / nombre
            if not act.exists():
                print("SIN REFERENCIA %s/%s" % (bench, n))
                continue
            r = comparar(p, act)
            ok = r is None or r.startswith("~")
            print("%s %s/%s%s" % ("IGUAL " if ok else "DIFIERE", bench, n, "" if r is None else ": " + r))
            if not ok and n == nombre and nombre not in DIFERENCIAS_CONOCIDAS:
                codigo = 1
            elif not ok and nombre in DIFERENCIAS_CONOCIDAS and n == nombre:
                print("        (conocida: %s)" % DIFERENCIAS_CONOCIDAS[nombre])
        if celdas is not None:
            d = comparar_docs13(celdas)
            print("docs/13 Tablas H1 a H4: %s" % ("IGUALES celda a celda" if not d else "%d celdas distintas: %s" % (len(d), d[:8])))
            codigo = codigo or (1 if d else 0)
    return codigo


if __name__ == "__main__":
    sys.exit(main())
