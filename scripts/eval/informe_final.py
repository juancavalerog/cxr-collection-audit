#!/usr/bin/env python3
"""Evaluación final sobre el test (docs/08_evaluacion.md): tablas en CSV, Markdown y LaTeX, y figuras en PDF y PNG.

Solo se ejecuta cuando el test está desbloqueado: exige el informe de la Puerta 3 (results/gate3/informe.json) y,
en cada corrida, protocol.test_bloqueado = false con un fichero protocol.congelado existente (docs/06_lineas_base.md,
Section 4). Si una corrida sigue bloqueada, termina con error antes de abrir ningún fichero de test.

Uso:
  python scripts/eval/informe_final.py [--resultados results/train] [--gate3 results/gate3/informe.json]
         [--salida results/final] [--spec spec.json] [--B 2000] [--semilla-bootstrap 2026]

La especificación (por defecto DEFAULT_SPEC; con --spec, un JSON con las mismas claves) da los métodos con su
experimento, las ablaciones A1 a A15, las semillas (42, 123, 2024, 7, 999) y el estado evaluado (best; dfr_best en
DFR). Para cada método: conjunto medio de las cinco semillas, umbral de Youden del conjunto medio en la validación
robusta (val_robust_sel en DFR), métricas con IC95 % por bootstrap de pacientes, media y desviación entre semillas y
diferencia frente a CSD-Net con bootstrap pareado y DeLong, con Holm por conjunto. Las cifras del artículo 1 solo
aparecen en tabla_otro_protocolo_articulo1, rotulada como de otro protocolo.

Salidas en <salida>/: tablas/*.csv|md|tex, figuras/*.pdf|png e informe_final.json (todas las cifras, B, semillas y
huellas SHA256 de las entradas).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval import stats as S  # noqa: E402

SEMILLAS_FINALES = [42, 123, 2024, 7, 999]
B_DEFECTO = 2000
SEMILLA_DEFECTO = 2026
CONJUNTO_PRIMARIO = "test_bimcv_limpio"
ALFA = 0.025          # nivel unilateral de las pruebas confirmatorias (docs/08_evaluacion.md, Section 1)
CONJUNTOS = ["test", "test_bimcv_limpio", "test_bimcv", "test_ricord", "test_rsna_sin_fuga", "test_rsna_con_fuga"]
NOMBRE_CONJUNTO = {"test": "Test limpio", "test_bimcv_limpio": "BIMCV del test limpio",
                   "test_bimcv": "Subconjunto bimcv (test oficial)", "test_ricord": "RICORD",
                   "test_rsna_sin_fuga": "RSNA sin fuga", "test_rsna_con_fuga": "RSNA con fuga"}

METODOS = [("csdnet", "CSD-Net"), ("control_fuente", "Control de fuente"), ("erm", "ERM"), ("groupdro", "GroupDRO"),
           ("dann", "DANN"), ("coral", "CORAL"), ("mixstyle", "MixStyle"), ("fourier", "Fourier (FACT)"),
           ("lisa", "LISA"), ("dfa_fuente", "DFA con fuente"), ("dfr", "DFR"), ("erm_mascara", "ERM con máscara"),
           ("erm_cuotas", "ERM con cuotas"), ("cdann", "CDANN"), ("irm", "IRM"), ("lai2026", "Lai et al."), ("rad_dino_probe", "Sonda RAD-DINO")]
# Paleta apta para daltonismo (Okabe e Ito, completada con la de Paul Tol), fija por método en todas las figuras.
COLOR = {"csdnet": "#000000", "control_fuente": "#0072B2", "erm": "#D55E00", "groupdro": "#E69F00",
         "dann": "#56B4E9", "coral": "#009E73", "mixstyle": "#CC79A7", "fourier": "#882255", "lisa": "#44AA99",
         "dfa_fuente": "#999933", "dfr": "#AA4499", "erm_mascara": "#117733", "cdann": "#332288", "irm": "#88CCEE",
         "lai2026": "#777777", "rad_dino_probe": "#CC6677", "erm_cuotas": "#DDCC77"}
MARCA = {m: mk for (m, _), mk in zip(METODOS, ["o", "s", "^", "v", "D", "P", "X", "<", ">", "h", "p", "*", "d", "H",
                                                "8", "1", "2"])}
ABLACIONES = ["A%d" % k for k in range(1, 16)]

DEFAULT_SPEC = {
    "metodos": [{"id": m, "nombre": n, "experimento": "csdnet_final" if m == "csdnet" else "baseline_%s_final" % m}
                for m, n in METODOS],
    "ablaciones": [{"id": a, "experimento": "csdnet_final_%s" % a.lower()} for a in ABLACIONES],
    "semillas": SEMILLAS_FINALES, "estado": "best", "estado_dfr": "dfr_best",
    "obligatorios": ["csdnet", "control_fuente"], "conjuntos_externos": [],
}

# docs/00_inventario.md, Sections 5.1, 5.2, 5.3 y 5.8: test oficial, protocolo del artículo 1 (ConvNeXt NIH).
ARTICULO1 = [
    {"configuracion": "Ingenuo, ConvNeXt-Tiny NIH (artículo 1)", "auc_conjunto": 0.725, "auc_semillas": "0.722 ± 0.009",
     "esp_bimcv": 0.216, "sens_bimcv": 0.921, "esp_rsna": 0.955, "brecha_rsna_menos_bimcv": 0.739, "ece": 0.190},
    {"configuracion": "Control de fuente, ConvNeXt-Tiny NIH (artículo 1)", "auc_conjunto": 0.723,
     "auc_semillas": "0.717 ± 0.006", "esp_bimcv": 0.731, "sens_bimcv": 0.602, "esp_rsna": 0.600,
     "brecha_rsna_menos_bimcv": -0.131, "ece": 0.077},
]

PRINCIPAL_V = {"V0": ["max_abs_err"], "V1": ["auc_logreg", "auc_mlp"], "V1b": ["r2_max"], "V2": ["td"],
               "V2b": ["ba_orig"], "V3": ["media_abs_dp", "delta_pm"], "V4": ["nivel2_decision"],
               "V5": ["auc_vista"], "V6": ["brecha"], "V7": []}


class FinalError(RuntimeError):
    pass


def _abs(p, base=ROOT) -> Path:
    p = Path(p)
    return p if p.is_absolute() else Path(base) / p


def _rel(p) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


# ===================================================================================== carga y desbloqueo
class Carga:
    def __init__(self, a, spec):
        self.a, self.spec = a, spec
        self.res = _abs(a.resultados)
        self.raiz = _abs(a.raiz_proyecto) if a.raiz_proyecto else ROOT
        self.huellas = {}
        self.pmap = None
        man = None if a.sin_mapa_pacientes else _abs(a.manifiesto)
        if man is not None and man.exists():
            self.pmap = S.patient_map_from_manifest(man)
            self.huellas[_rel(man)] = S.sha256_file(man)
        self.avisos = []

    def corridas(self, exp, obligatorio):
        dirs = S.find_seed_dirs(self.res, exp)
        sem = list(self.spec["semillas"])
        falta = [s for s in sem if s not in dirs]
        if len(falta) == len(sem):
            if obligatorio:
                raise FinalError("no hay corridas de %s en %s" % (exp, self.res))
            return None
        if falta:
            msg = "%s: faltan las semillas %s" % (exp, falta)
            if not self.a.permitir_semillas_incompletas:
                raise FinalError(msg + " (no se quitan ni se sustituyen semillas; docs/08_evaluacion.md)")
            self.avisos.append(msg)
        runs = {s: S.read_run(dirs[s]) for s in sem if s in dirs}
        for s, r in runs.items():
            self._desbloqueo(r)
            for f in ("config_resolved.json", "summary.json"):
                p = Path(r["ruta"]) / f
                if p.exists():
                    self.huellas[_rel(p)] = S.sha256_file(p)
        return runs

    def _desbloqueo(self, r):
        p = r["protocolo"]
        if p.get("test_bloqueado") is not False:
            raise FinalError("el test sigue bloqueado en %s (protocol.test_bloqueado = %s); la evaluación final exige "
                             "el desbloqueo con hiperparámetros congelados" % (r["ruta"], p.get("test_bloqueado")))
        c = p.get("congelado")
        if not c or not _abs(c, self.raiz).exists():
            raise FinalError("%s no declara un fichero protocol.congelado existente (%s)" % (r["ruta"], c))
        self.huellas[_rel(_abs(c, self.raiz))] = S.sha256_file(_abs(c, self.raiz))

    def pred(self, run, estado, conjunto):
        path = Path(run["ruta"]) / "predictions" / ("%s_%s.csv" % (estado, conjunto))
        if not path.exists():
            return None
        df = S.read_predictions(path)
        self.huellas[_rel(path)] = S.sha256_file(path)
        return S.apply_patient_map(df, self.pmap)


def _bimcv_limpio(df):
    if df is None:
        return None
    out = df[df["source"].astype(int) == S.COVIDX_SOURCES.index("bimcv")].reset_index(drop=True)
    out.attrs.update(df.attrs)
    return out


def carga_metodo(C: Carga, mid, exp, obligatorio):
    runs = C.corridas(exp, obligatorio)
    if runs is None:
        return None
    est = C.spec["estado_dfr"] if mid == "dfr" else C.spec["estado"]
    cu = "val_robust_sel" if mid == "dfr" else "val_robust"
    sem = sorted(runs, key=lambda s: C.spec["semillas"].index(s))
    out = {"id": mid, "experimento": exp, "estado": est, "conjunto_umbral": cu, "semillas": sem, "por_semilla": {},
           "conjuntos": {}, "corridas": runs}
    for s in sem:
        d = {"umbral": C.pred(runs[s], est, cu)}
        if d["umbral"] is None:
            raise FinalError("faltan las predicciones %s_%s de %s/seed%d" % (est, cu, exp, s))
        for c in CONJUNTOS + list(C.spec.get("conjuntos_externos") or []):
            if c == "test_bimcv_limpio":
                continue
            d[c] = C.pred(runs[s], est, c)
        if d.get("test") is None:
            raise FinalError("faltan las predicciones %s_test de %s/seed%d" % (est, exp, s))
        d["test_bimcv_limpio"] = _bimcv_limpio(d["test"])
        out["por_semilla"][s] = d
    names = [str(s) for s in sem]
    out["ens_umbral"] = S.ensemble([out["por_semilla"][s]["umbral"] for s in sem], names)
    out["umbral"] = S.youden_threshold(out["ens_umbral"]["label"], out["ens_umbral"]["prob"])
    for c in CONJUNTOS + list(C.spec.get("conjuntos_externos") or []):
        dfs = [out["por_semilla"][s].get(c) for s in sem]
        if any(d is None for d in dfs):
            if any(d is not None for d in dfs):
                raise FinalError("%s: el conjunto %s falta en algunas semillas" % (exp, c))
            continue
        out["conjuntos"][c] = S.ensemble(dfs, names)
    return out


# ===================================================================================== cálculos
def _per_seed(M, c, fn):
    vals = {}
    for s in M["semillas"]:
        d = M["por_semilla"][s].get(c)
        if d is None:
            continue
        thr_s = S.youden_threshold(M["por_semilla"][s]["umbral"]["label"], M["por_semilla"][s]["umbral"]["prob"])
        vals[str(s)] = fn(d, thr_s)
    return S.seed_summary(vals) if vals else None


def metricas_conjunto(M, c, B, seed):
    e = M["conjuntos"][c]
    bm = S.bootstrap_metrics(e["label"].to_numpy(), e["prob"].to_numpy(), e["patient"].to_numpy(), M["umbral"], B, seed)
    sem = {k: _per_seed(M, c, (lambda d, t, k=k: S.point_metrics(d["label"], d["prob"], t)[k]))
           for k in ("auc_roc", "auc_pr", "sensibilidad", "especificidad", "exactitud_balanceada", "ece", "brier")}
    return {"conjunto_medio": bm, "semillas": sem, "umbral": M["umbral"], "conjunto_umbral": M["conjunto_umbral"]}


def _dos_clases(df):
    return df is not None and df["label"].nunique() == 2


def comparaciones(Ms, ref, conjunto, B, seed):
    """CSD-Net frente a cada método en un conjunto. Prueba primaria (docs/08_evaluacion.md, Section 1 y enmienda
    2): media por semilla de la diferencia de AUC, con bootstrap de pacientes. Secundarias: la misma diferencia en
    el conjunto medio de las semillas (bootstrap y DeLong) y las semillas como réplicas (Welch y t pareada). Holm
    sobre la familia, por separado para cada prueba."""
    R = Ms[ref]
    if conjunto not in R["conjuntos"] or not _dos_clases(R["conjuntos"][conjunto]):
        return {}
    a = R["conjuntos"][conjunto]
    y, g = a["label"].to_numpy(), a["patient"].to_numpy()
    out = {}
    for mid, M in Ms.items():
        if mid == ref or conjunto not in M["conjuntos"]:
            continue
        b = S.align(a, M["conjuntos"][conjunto], ref, mid)
        tc = S.paired_auc_test(y, a["prob"].to_numpy(), b["prob"].to_numpy(), g, B, seed)
        comunes = [s_ for s_ in R["semillas"] if s_ in M["semillas"] and conjunto in R["por_semilla"][s_]
                   and conjunto in M["por_semilla"][s_]]
        if not comunes:
            raise FinalError("%s y %s no comparten ninguna semilla en %s" % (ref, mid, conjunto))
        P1 = np.stack([S.align(a, R["por_semilla"][s_][conjunto], ref, "%s semilla %s" % (ref, s_))["prob"].to_numpy()
                       for s_ in comunes])
        P2 = np.stack([S.align(a, M["por_semilla"][s_][conjunto], ref, "%s semilla %s" % (mid, s_))["prob"].to_numpy()
                       for s_ in comunes])
        t = S.paired_auc_test_seedmean(y, P1, P2, g, B, seed, seeds=comunes)
        t["semillas_omitidas"] = [s_ for s_ in R["semillas"] if s_ not in comunes]
        t["conjunto_medio"] = tc
        t["delong"] = tc.get("delong")                       # DeLong solo existe para el conjunto medio
        va = {str(s_): S.auc_roc(R["por_semilla"][s_][conjunto]["label"], R["por_semilla"][s_][conjunto]["prob"])
              for s_ in R["semillas"]}
        vb = {str(s_): S.auc_roc(M["por_semilla"][s_][conjunto]["label"], M["por_semilla"][s_][conjunto]["prob"])
              for s_ in M["semillas"]}
        t["semillas_replicas"] = S.compare_seed_values(va, vb)
        out[mid] = t
    if out:
        hb = S.holm({k: v["p_unilateral"] for k, v in out.items()}, ALFA)
        hc = S.holm({k: v["conjunto_medio"]["p_unilateral"] for k, v in out.items()}, ALFA)
        hd = S.holm({k: (v.get("delong") or {}).get("p_unilateral", float("nan")) for k, v in out.items()}, ALFA)
        hw = S.holm({k: v["semillas_replicas"]["welch"]["p"] for k, v in out.items()}, ALFA)
        for k in out:
            out[k]["p_holm"] = hb[k]["p_holm"]
            out[k]["p_holm_conjunto"] = hc[k]["p_holm"]
            out[k]["p_holm_delong"] = hd[k]["p_holm"]
            out[k]["p_holm_welch"] = hw[k]["p_holm"]
            out[k]["rechaza_holm"] = hb[k]["rechaza"]
            out[k]["alfa_unilateral"] = ALFA
    return out


def brecha(M, B, seed):
    e = M["conjuntos"].get("test")
    g = S.specificity_gap(e, M["umbral"], "rsna", "bimcv", B, seed)
    g["semillas"] = _per_seed(M, "test", lambda d, t: S.specificity_gap(d, t, "rsna", "bimcv", 1, seed)["brecha"])
    return g


def verificaciones(M):
    """Veredictos por semilla de los informes probes/final_*.json: recuento y media ± dt de las métricas."""
    if M is None:
        return None
    rows = {}
    for s in M["semillas"]:
        run = M["corridas"][s]
        for name, path in run["informes_verificaciones"].items():
            if not name.startswith("final_"):
                continue
            r = S.read_probe_report(path)
            for vid, ver in r["veredictos"].items():
                key = (name, vid)
                d = rows.setdefault(key, {"veredictos": {}, "metricas": {}})
                d["veredictos"][str(s)] = ver
                for mk, mv in (r["metricas"].get(vid) or {}).items():
                    if isinstance(mv, (int, float)) and not isinstance(mv, bool):
                        d["metricas"].setdefault(mk, {})[str(s)] = float(mv)
    return rows


# ===================================================================================== escritura de tablas
def _fmt(x, nd=3):
    if x is None:
        return "n.d."
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return "n.d."
    return ("%." + str(nd) + "f") % xf


def _fp(p):
    if p is None or not np.isfinite(p):
        return "n.d."
    return "< 0.001" if p < 0.001 else "%.3f" % p


def _ci(t, nd=3):
    if not t or t[0] is None or not np.isfinite(t[0]):
        return "n.d."
    return "[%s, %s]" % (_fmt(t[0], nd), _fmt(t[1], nd))


def _eic(x, t, nd=3):
    """Estimación con su IC, o n.d. si la métrica no está definida en el conjunto."""
    if x is None or not np.isfinite(float(x)):
        return "n.d."
    return "%s %s" % (_fmt(x, nd), _ci(t, nd))


def _ms(s, nd=3):
    if not s or not np.isfinite(s.get("media", float("nan"))):
        return "n.d."
    return "%s ± %s" % (_fmt(s["media"], nd), _fmt(s["dt"], nd))


def _tex_escape(s):
    s = str(s)
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_"),
                 ("±", "$\\pm$"), ("Δ", "$\\Delta$"), ("<", "$<$"), (">", "$>$"), ("≤", "$\\leq$")):
        s = s.replace(a, b)
    return s


def escribe_tabla(df: pd.DataFrame, base: Path, titulo: str, nota: str = ""):
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(base.with_suffix(".csv"), index=False)
    cols = list(df.columns)
    md = ["<!-- %s -->" % titulo, "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        md.append("| " + " | ".join(str(v) for v in r.values) + " |")
    if nota:
        md += ["", nota]
    base.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    tex = ["% " + titulo, "\\begin{tabular}{" + "l" * min(2, len(cols)) + "r" * max(0, len(cols) - 2) + "}",
           "\\toprule", " & ".join(_tex_escape(c) for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        tex.append(" & ".join(_tex_escape(v) for v in r.values) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    if nota:
        tex.append("% " + nota)
    base.with_suffix(".tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return [base.with_suffix(x) for x in (".csv", ".md", ".tex")]


# ===================================================================================== figuras
def _estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                         "axes.linewidth": 0.6, "lines.linewidth": 1.0})
    return plt


def _guarda(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    return [base.with_suffix(".pdf"), base.with_suffix(".png")]


def figuras(Ms, nombres, comp, gaps, out: Path, B, seed):
    plt = _estilo()
    files = []
    orden = [m for m, _ in METODOS if m in Ms]
    c = CONJUNTO_PRIMARIO
    # ROC con banda
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    for m in orden:
        e = Ms[m]["conjuntos"].get(c)
        if not _dos_clases(e):
            continue
        rb = S.roc_band(e["label"], e["prob"], e["patient"], B if m in ("csdnet", "control_fuente") else 1, seed)
        ax.plot(rb["fpr"], rb["tpr"], color=COLOR.get(m, "#777777"), lw=1.4 if m == "csdnet" else 0.9,
                label=nombres[m], zorder=3 if m == "csdnet" else 2)
        if m in ("csdnet", "control_fuente"):
            ax.fill_between(rb["fpr_rejilla"], rb["tpr_lo"], rb["tpr_hi"], color=COLOR.get(m), alpha=0.18, lw=0)
    ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.6, ls=":")
    ax.set_xlabel("Tasa de falsos positivos, 1 - especificidad (proporción)")
    ax.set_ylabel("Sensibilidad (proporción)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(loc="lower right", frameon=False, fontsize=8, handlelength=1.2, borderaxespad=0.2, ncol=1 if len(orden) <= 6 else 2,
              columnspacing=0.6)
    files += _guarda(fig, out / "fig_roc_test_bimcv_limpio"); plt.close(fig)
    # Fiabilidad
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.6, ls=":")
    for m in orden:
        e = Ms[m]["conjuntos"].get("test")
        rb = S.reliability_bins(e["label"], e["prob"])
        rb = rb[rb["n"] > 0]
        ax.plot(rb["p_media"], rb["frecuencia"], color=COLOR.get(m), marker=MARCA.get(m, "o"), ms=3, lw=0.8,
                label=nombres[m])
    ax.set_xlabel("Probabilidad predicha media del bin (proporción)")
    ax.set_ylabel("Frecuencia observada de positivos (proporción)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(loc="lower right", frameon=False, fontsize=8, handlelength=1.2, ncol=1 if len(orden) <= 6 else 2,
              columnspacing=0.6)
    files += _guarda(fig, out / "fig_fiabilidad_test"); plt.close(fig)
    # Bosque
    cc = comp.get(c, {})
    ms_ = [m for m in orden if m in cc]
    fig, ax = plt.subplots(figsize=(3.4, 0.3 * len(ms_) + 0.8))
    for i, m in enumerate(ms_):
        t = cc[m]
        lo, hi = t["ic95"]
        ax.errorbar(t["delta"], i, xerr=[[t["delta"] - lo], [hi - t["delta"]]], fmt=MARCA.get(m, "o"),
                    color=COLOR.get(m), ms=4, capsize=2, lw=0.9)
    ax.axvline(0, color="#777777", lw=0.6, ls="--")
    ax.set_yticks(range(len(ms_))); ax.set_yticklabels([nombres[m] for m in ms_]); ax.invert_yaxis()
    ax.set_xlabel("Δ AUC-ROC, CSD-Net menos método (unidades de AUC)")
    files += _guarda(fig, out / "fig_bosque_delta_auc"); plt.close(fig)
    # Brecha de especificidad
    fig, ax = plt.subplots(figsize=(3.4, 0.3 * len(orden) + 0.8))
    for i, m in enumerate(orden):
        g = gaps[m]
        lo, hi = g["ic95"]
        if g["brecha"] is None:
            continue
        ax.errorbar(g["brecha"], i, xerr=[[g["brecha"] - lo], [hi - g["brecha"]]], fmt=MARCA.get(m, "o"),
                    color=COLOR.get(m), ms=4, capsize=2, lw=0.9)
    ax.axvline(0, color="#777777", lw=0.6, ls="--")
    ax.set_yticks(range(len(orden))); ax.set_yticklabels([nombres[m] for m in orden]); ax.invert_yaxis()
    ax.set_xlabel("Brecha de especificidad, RSNA menos BIMCV (proporción)")
    files += _guarda(fig, out / "fig_brecha_especificidad"); plt.close(fig)
    return files


# ===================================================================================== principal
def ejecutar(a) -> dict:
    spec = dict(DEFAULT_SPEC)
    if a.spec:
        spec.update(json.loads(_abs(a.spec).read_text(encoding="utf-8")))
    g3 = _abs(a.gate3)
    if not g3.exists():
        raise FinalError("falta el informe de la Puerta 3 (%s): el test solo se desbloquea después de la Puerta 3"
                         % _rel(g3))
    B, seed = int(a.B), int(a.semilla_bootstrap)
    C = Carga(a, spec)
    C.huellas[_rel(g3)] = S.sha256_file(g3)
    nombres = {m["id"]: m.get("nombre", m["id"]) for m in spec["metodos"]}
    # 1) desbloqueo de todas las corridas antes de abrir ningún test
    runs_all = {}
    for m in spec["metodos"]:
        runs_all[m["id"]] = C.corridas(m["experimento"], m["id"] in spec["obligatorios"])
    for ab in spec["ablaciones"]:
        runs_all["abl_" + ab["id"]] = C.corridas(ab["experimento"], False)
    # 2) carga
    Ms, no_disp = {}, []
    for m in spec["metodos"]:
        if runs_all[m["id"]] is None:
            no_disp.append(m["id"])
            continue
        Ms[m["id"]] = carga_metodo(C, m["id"], m["experimento"], m["id"] in spec["obligatorios"])
    ref = "csdnet"
    out = _abs(a.salida)
    T, F = out / "tablas", out / "figuras"
    res = {"fecha": _dt.datetime.now().astimezone().isoformat(timespec="seconds"), "B": B, "semilla_bootstrap": seed,
           "semillas": spec["semillas"], "unidad_remuestreo": "paciente", "conjunto_primario": CONJUNTO_PRIMARIO,
           "comparacion_primaria": "CSD-Net frente a control de fuente, media por semilla de la diferencia de AUC-ROC "
                                   "(modelo individual) en %s, bootstrap pareado de pacientes, unilateral; se declara si "
                                   "el extremo inferior del IC95 %% > 0 (alfa unilateral %.3f). El conjunto medio de las "
                                   "semillas es secundario" % (CONJUNTO_PRIMARIO, ALFA),
           "no_disponibles": no_disp, "avisos": C.avisos, "metodos": {}, "comparaciones": {}, "ablaciones": {},
           "ficheros": []}
    # 3) métricas por método y conjunto
    filas, filas_c, met = [], [], {}
    sets_all = CONJUNTOS + list(spec.get("conjuntos_externos") or [])
    comp = {c: comparaciones(Ms, ref, c, B, seed) for c in sets_all}
    COLS_P = ["Método", "Conjunto", "n", "AUC-ROC, media ± dt por semilla", "AUC-PR, media ± dt",
              "Sensibilidad, media ± dt", "Especificidad, media ± dt", "Exactitud balanceada, media ± dt",
              "Δ frente a CSD-Net, media por semilla (IC95 %)", "Semillas en que gana CSD-Net", "p Holm (bootstrap)"]
    COLS_C = ["Método", "Conjunto", "n", "AUC-ROC del conjunto (IC95 %)", "AUC-PR", "Sensibilidad (IC95 %)",
              "Especificidad (IC95 %)", "Exactitud balanceada", "Δ del conjunto frente a CSD-Net (IC95 %)",
              "p Holm (bootstrap)", "p Holm (DeLong)"]
    for mid in [m for m, _ in METODOS if m in Ms] + [m for m in Ms if m not in dict(METODOS)]:
        M = Ms[mid]
        met[mid] = {}
        for c in sets_all:
            if c not in M["conjuntos"]:
                continue
            mc = metricas_conjunto(M, c, B, seed)
            met[mid][c] = mc
            e, ic = mc["conjunto_medio"]["estimacion"], mc["conjunto_medio"]["ic95"]
            sm = mc["semillas"]
            t = comp.get(c, {}).get(mid)
            es_ref = "referencia" if mid == ref and _dos_clases(M["conjuntos"][c]) else ""
            filas.append({"Método": nombres.get(mid, mid), "Conjunto": NOMBRE_CONJUNTO.get(c, c), "n": e["n"],
                          "AUC-ROC, media ± dt por semilla": _ms(sm["auc_roc"]),
                          "AUC-PR, media ± dt": _ms(sm["auc_pr"]),
                          "Sensibilidad, media ± dt": _ms(sm["sensibilidad"]),
                          "Especificidad, media ± dt": _ms(sm["especificidad"]),
                          "Exactitud balanceada, media ± dt": _ms(sm["exactitud_balanceada"]),
                          "Δ frente a CSD-Net, media por semilla (IC95 %)":
                              ("%s %s" % (_fmt(-t["delta"]), _ci((-t["ic95"][1], -t["ic95"][0])))) if t else es_ref,
                          "Semillas en que gana CSD-Net": ("%d de %d" % (t["semillas_a_favor"], t["n_semillas"])) if t else "",
                          "p Holm (bootstrap)": _fp(t["p_holm"]) if t else ""})
            tc = t["conjunto_medio"] if t else None
            filas_c.append({"Método": nombres.get(mid, mid), "Conjunto": NOMBRE_CONJUNTO.get(c, c), "n": e["n"],
                            "AUC-ROC del conjunto (IC95 %)": _eic(e["auc_roc"], ic.get("auc_roc")),
                            "AUC-PR": _fmt(e["auc_pr"]),
                            "Sensibilidad (IC95 %)": _eic(e.get("sensibilidad"), ic.get("sensibilidad")),
                            "Especificidad (IC95 %)": _eic(e.get("especificidad"), ic.get("especificidad")),
                            "Exactitud balanceada": _fmt(e.get("exactitud_balanceada")),
                            "Δ del conjunto frente a CSD-Net (IC95 %)":
                                ("%s %s" % (_fmt(-tc["delta"]), _ci((-tc["ic95"][1], -tc["ic95"][0])))) if tc else es_ref,
                            "p Holm (bootstrap)": _fp(t["p_holm_conjunto"]) if t else "",
                            "p Holm (DeLong)": _fp(t["p_holm_delong"]) if t else ""})
        res["metodos"][mid] = {"experimento": M["experimento"], "estado": M["estado"], "semillas": M["semillas"],
                               "umbral_youden": M["umbral"], "conjunto_umbral": M["conjunto_umbral"],
                               "metricas": met[mid]}
    for mid in no_disp:
        filas.append(dict({k: "" for k in COLS_P}, **{"Método": nombres.get(mid, mid), "Conjunto": "no disponible"}))
        filas_c.append(dict({k: "" for k in COLS_C}, **{"Método": nombres.get(mid, mid), "Conjunto": "no disponible"}))
    res["comparaciones"] = comp
    tp = comp.get(CONJUNTO_PRIMARIO, {}).get("control_fuente")
    res["resultado_primario"] = None if tp is None else {
        "estadistico": tp["estadistico"],
        "delta_csdnet_menos_control": tp["delta"], "ic95": tp["ic95"], "p_unilateral": tp["p_unilateral"],
        "delta_por_semilla": tp["delta_por_semilla"], "dt_delta_entre_semillas": tp["dt_delta_entre_semillas"],
        "semillas_a_favor": tp["semillas_a_favor"], "n_semillas": tp["n_semillas"],
        "B": tp["B"], "B_validos": tp["B_validos"], "semilla_bootstrap": tp["semilla_bootstrap"],
        "se_declara_mejora": bool(tp["ic95"][0] is not None and tp["ic95"][0] > 0),
        "secundario_conjunto_medio": {"delta": tp["conjunto_medio"]["delta"], "ic95": tp["conjunto_medio"]["ic95"],
                                      "p_unilateral": tp["conjunto_medio"]["p_unilateral"],
                                      "delong_p_unilateral": (tp.get("delong") or {}).get("p_unilateral")}}
    nota = ("Cifra principal: modelo individual, media y desviación típica entre %d semillas. Umbral de Youden de cada "
            "semilla en su validación robusta (val_robust_sel en DFR). Δ = método menos CSD-Net, media por semilla, con "
            "IC95 %% por bootstrap de pacientes (B = %d, semilla %d); p unilateral de que CSD-Net supere al método, "
            "corregido por Holm dentro de cada conjunto." % (len(spec["semillas"]), B, seed))
    res["ficheros"] += escribe_tabla(pd.DataFrame(filas, columns=COLS_P), T / "tabla_principal", "Tabla principal", nota)
    nota_c = ("Dato secundario: conjunto medio de %d semillas (probabilidades promediadas); IC95 %% por bootstrap de "
              "pacientes (B = %d, semilla %d). Umbral de Youden del conjunto medio en la validación robusta. Δ = método "
              "menos CSD-Net; p unilateral corregido por Holm dentro de cada conjunto." % (len(spec["semillas"]), B, seed))
    res["ficheros"] += escribe_tabla(pd.DataFrame(filas_c, columns=COLS_C), T / "tabla_conjunto_medio",
                                     "Conjunto medio de las semillas (secundario)", nota_c)
    # semillas como réplicas
    fs = []
    for c, cc in comp.items():
        for mid, t in cc.items():
            w, pr = t["semillas_replicas"]["welch"], t["semillas_replicas"]["pareada"]
            fs.append({"Conjunto": NOMBRE_CONJUNTO.get(c, c), "Método": nombres.get(mid, mid),
                       "Diferencia de medias (CSD-Net menos método)": _fmt(t["semillas_replicas"]["diferencia"]),
                       "Welch t (gl)": "%s (%s)" % (_fmt(w["t"], 2), _fmt(w["gl"], 1)), "p Welch": _fp(w["p"]),
                       "p Holm (Welch)": _fp(t["p_holm_welch"]),
                       "t pareada": _fmt(pr["t"], 2) if pr else "n.d.", "p pareada": _fp(pr["p"]) if pr else "n.d."})
    res["ficheros"] += escribe_tabla(pd.DataFrame(fs), T / "tabla_semillas", "Comparación con las semillas como réplicas",
                                     "AUC-ROC por semilla; unilateral (CSD-Net mayor). Holm dentro de cada conjunto.")
    # calibración
    fc = []
    for mid in met:
        for c in ("test", "test_bimcv_limpio"):
            if c not in met[mid]:
                continue
            mc = met[mid][c]
            e, ic = mc["conjunto_medio"]["estimacion"], mc["conjunto_medio"]["ic95"]
            fc.append({"Método": nombres.get(mid, mid), "Conjunto": NOMBRE_CONJUNTO[c],
                       "ECE (IC95 %)": "%s %s" % (_fmt(e["ece"]), _ci(ic["ece"])),
                       "ECE por semilla": _ms(mc["semillas"]["ece"]),
                       "Brier (IC95 %)": "%s %s" % (_fmt(e["brier"]), _ci(ic["brier"])),
                       "Brier por semilla": _ms(mc["semillas"]["brier"])})
        rb = S.reliability_bins(Ms[mid]["conjuntos"]["test"]["label"], Ms[mid]["conjuntos"]["test"]["prob"])
        rb.insert(0, "metodo", mid)
        T.mkdir(parents=True, exist_ok=True)
        rb.to_csv(T / ("fiabilidad_%s.csv" % mid), index=False)
    res["ficheros"] += escribe_tabla(pd.DataFrame(fc), T / "tabla_calibracion", "Calibración",
                                     "ECE de 15 bins de igual anchura y puntuación de Brier.")
    # brecha
    gaps = {mid: brecha(Ms[mid], B, seed) for mid in met}
    fb = [{"Método": nombres.get(m, m), "Especificidad RSNA": _fmt(g["esp_a"]), "Especificidad BIMCV": _fmt(g["esp_b"]),
           "Brecha RSNA menos BIMCV (IC95 %)": "%s %s" % (_fmt(g["brecha"]), _ci(g["ic95"])),
           "Brecha por semilla": _ms(g["semillas"]), "Negativos RSNA / BIMCV": "%d / %d" % (g["n_neg_a"], g["n_neg_b"])}
          for m, g in gaps.items()]
    res["brecha_especificidad"] = gaps
    res["ficheros"] += escribe_tabla(pd.DataFrame(fb), T / "tabla_brecha_especificidad", "Brecha de especificidad",
                                     "Negativos del test limpio al umbral de Youden de la validación robusta.")
    # ablaciones
    fa, pa = [], {}
    A0 = Ms.get(ref)
    for ab in spec["ablaciones"]:
        if runs_all["abl_" + ab["id"]] is None:
            fa.append({"Id": ab["id"], "Experimento": ab["experimento"], "AUC-ROC, media ± dt por semilla": "no disponible",
                       "Δ frente a A0, media por semilla (IC95 %)": "", "Semillas en que gana A0": "", "p Holm": "",
                       "AUC-ROC del conjunto (IC95 %)": "", "Brecha (IC95 %)": ""})
            continue
        Ma = carga_metodo(C, "abl_" + ab["id"], ab["experimento"], False)
        e = Ma["conjuntos"][CONJUNTO_PRIMARIO]
        a0 = A0["conjuntos"][CONJUNTO_PRIMARIO]
        b = S.align(a0, e, "A0", ab["id"])
        tc = S.paired_auc_test(a0["label"].to_numpy(), a0["prob"].to_numpy(), b["prob"].to_numpy(),
                               a0["patient"].to_numpy(), B, seed)
        comunes = [s_ for s_ in A0["semillas"] if s_ in Ma["semillas"] and CONJUNTO_PRIMARIO in A0["por_semilla"][s_]
                   and CONJUNTO_PRIMARIO in Ma["por_semilla"][s_]]
        if not comunes:
            raise FinalError("A0 y %s no comparten ninguna semilla" % ab["id"])
        P1 = np.stack([S.align(a0, A0["por_semilla"][s_][CONJUNTO_PRIMARIO], "A0", "A0 semilla %s" % s_)["prob"].to_numpy()
                       for s_ in comunes])
        P2 = np.stack([S.align(a0, Ma["por_semilla"][s_][CONJUNTO_PRIMARIO], "A0", "%s semilla %s" % (ab["id"], s_))["prob"].to_numpy()
                       for s_ in comunes])
        t = S.paired_auc_test_seedmean(a0["label"].to_numpy(), P1, P2, a0["patient"].to_numpy(), B, seed, seeds=comunes)
        t["conjunto_medio"] = tc
        g = brecha(Ma, B, seed)
        pa[ab["id"]] = t["p_unilateral"]
        res["ablaciones"][ab["id"]] = {"experimento": ab["experimento"], "comparacion_A0": t, "brecha": g,
                                       "auc_semillas": _per_seed(Ma, CONJUNTO_PRIMARIO,
                                                                 lambda d, th: S.auc_roc(d["label"], d["prob"])),
                                       "verificaciones": _verif_json(verificaciones(Ma))}
        fa.append({"Id": ab["id"], "Experimento": ab["experimento"],
                   "AUC-ROC, media ± dt por semilla": _ms(res["ablaciones"][ab["id"]]["auc_semillas"]),
                   "Δ frente a A0, media por semilla (IC95 %)": "%s %s" % (_fmt(-t["delta"]), _ci((-t["ic95"][1], -t["ic95"][0]))),
                   "Semillas en que gana A0": "%d de %d" % (t["semillas_a_favor"], t["n_semillas"]),
                   "p Holm": None,
                   "AUC-ROC del conjunto (IC95 %)": "%s %s" % (_fmt(tc["auc2"]), _ci(S.bootstrap_auc(e["label"], e["prob"], e["patient"], B, seed)["ic95"])),
                   "Brecha (IC95 %)": "%s %s" % (_fmt(g["brecha"]), _ci(g["ic95"]))})
    hol = S.holm(pa, ALFA) if pa else {}
    for r in fa:
        if r["p Holm"] is None:
            r["p Holm"] = _fp(hol[r["Id"]]["p_holm"])
            res["ablaciones"][r["Id"]]["p_holm"] = hol[r["Id"]]["p_holm"]
    res["ficheros"] += escribe_tabla(pd.DataFrame(fa), T / "tabla_ablaciones", "Ablaciones A1 a A15",
                                     "BIMCV del test limpio; Δ = ablación menos A0 (CSD-Net completa), media por semilla "
                                     "sobre las semillas comunes, con IC95 % por bootstrap de pacientes; p unilateral de "
                                     "que A0 supere a la ablación, Holm sobre las ablaciones disponibles. El conjunto "
                                     "medio es secundario.")
    # verificaciones
    fv = []
    exps = [("CSD-Net (A0)", A0)] + [(ab["id"], None) for ab in spec["ablaciones"]]
    vtabs = {"A0": verificaciones(A0)}
    for ab in spec["ablaciones"]:
        if ab["id"] in res["ablaciones"]:
            vtabs[ab["id"]] = res["ablaciones"][ab["id"]]["verificaciones"]
    for nm, V in vtabs.items():
        if V is None:
            continue
        items = V.items() if not isinstance(V, list) else ((tuple(x["clave"]), x) for x in V)
        for (rep, vid), d in items:
            cnt = pd.Series(list(d["veredictos"].values())).value_counts().to_dict()
            keys = [k for k in PRINCIPAL_V.get(vid, []) if k in d["metricas"]] or list(d["metricas"])[:1]
            fv.append({"Modelo": nm, "Informe": rep, "Verificación": vid,
                       "Veredictos": ", ".join("%s %d" % kv for kv in sorted(cnt.items())),
                       "Métrica principal (media ± dt)": "; ".join(
                           "%s %s" % (k, _ms(S.seed_summary(d["metricas"][k]))) for k in keys)})
    res["verificaciones"] = {k: _verif_json(v) if not isinstance(v, list) else v for k, v in vtabs.items()}
    res["V4_entre_corridas"] = _v4(Ms)
    nota_v = ("Recuento de veredictos sobre las semillas de cada modelo (informes probes/final_*.json). V4 entre "
              "corridas: %s." % res["V4_entre_corridas"].get("veredicto", res["V4_entre_corridas"].get("motivo")))
    res["ficheros"] += escribe_tabla(pd.DataFrame(fv) if fv else pd.DataFrame(columns=["Modelo", "Informe", "Verificación",
                                                                                        "Veredictos", "Métrica principal (media ± dt)"]),
                                     T / "tabla_verificaciones", "Verificaciones V1 a V7", nota_v)
    # artículo 1, otro protocolo
    fo = pd.DataFrame([{"Configuración": r["configuracion"], "AUC del conjunto": _fmt(r["auc_conjunto"]),
                        "AUC por semilla": r["auc_semillas"], "Especificidad BIMCV": _fmt(r["esp_bimcv"]),
                        "Sensibilidad BIMCV": _fmt(r["sens_bimcv"]), "Especificidad RSNA": _fmt(r["esp_rsna"]),
                        "Brecha RSNA menos BIMCV": _fmt(r["brecha_rsna_menos_bimcv"]), "ECE": _fmt(r["ece"])}
                       for r in ARTICULO1])
    res["ficheros"] += escribe_tabla(fo, T / "tabla_otro_protocolo_articulo1",
                                     "Otro protocolo: cifras del artículo 1 (no comparables con la tabla principal)",
                                     "Otro protocolo: test oficial con fuga, partición oficial, preentreno NIH, umbral de "
                                     "Youden de la validación oficial, IC de 1 000 remuestreos por imagen. Fuente: "
                                     "docs/00_inventario.md, Sections 5.1, 5.2, 5.3 y 5.8. No comparable con la tabla principal.")
    res["articulo1_otro_protocolo"] = ARTICULO1
    # figuras
    res["ficheros"] += figuras(Ms, nombres, comp, gaps, F, B, seed)
    res["entradas"] = C.huellas
    res["ficheros"] = [_rel(p) for p in res["ficheros"]]
    out.mkdir(parents=True, exist_ok=True)
    (out / "informe_final.json").write_text(json.dumps(S.jsonable(res), indent=1, ensure_ascii=False), encoding="utf-8")
    return res


def _verif_json(V):
    if V is None:
        return None
    return [{"clave": list(k), **v} for k, v in V.items()]


def _v4(Ms):
    from src.eval.probes import v4_compare
    def lv(M, key):
        if M is None:
            return []
        V = verificaciones(M) or {}
        d = V.get(("final_best", "V4"))
        return list((d or {}).get("metricas", {}).get(key, {}).values())
    csd2 = lv(Ms.get("csdnet"), "nivel2_decision")
    ctl2 = lv(Ms.get("control_fuente"), "nivel2_decision")
    if len(csd2) < 2 or len(ctl2) < 2:
        return {"motivo": "no disponible: hacen falta informes V4 por semilla de CSD-Net y del control de fuente "
                          "(las líneas base no generan probes)"}
    gap = lambda M: [a - b for a, b in zip(lv(M, "nivel1_content"), lv(M, "nivel2_decision"))]
    r = v4_compare(csd2, ctl2, gap(Ms.get("csdnet")), gap(Ms.get("erm")) or [float("nan")])
    return {"veredicto": r["veredicto"], "metricas": r["metricas"]}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--resultados", default="results/train")
    ap.add_argument("--gate3", default="results/gate3/informe.json")
    ap.add_argument("--salida", default="results/final")
    ap.add_argument("--spec", default=None)
    ap.add_argument("--B", type=int, default=B_DEFECTO)
    ap.add_argument("--semilla-bootstrap", type=int, default=SEMILLA_DEFECTO)
    ap.add_argument("--manifiesto", default="data/manifests/covidx_cxr4_clean.csv")
    ap.add_argument("--sin-mapa-pacientes", action="store_true")
    ap.add_argument("--raiz-proyecto", default=None, help="raíz para resolver protocol.congelado (por defecto, el repo)")
    ap.add_argument("--permitir-semillas-incompletas", action="store_true",
                    help="no quita semillas: solo permite continuar marcando en avisos las que faltan")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        r = ejecutar(a)
    except (FinalError, S.AlignmentError, S.PredictionFileError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    print("informe final: %d ficheros en %s" % (len(r["ficheros"]), a.salida))
    return 0


if __name__ == "__main__":
    sys.exit(main())
