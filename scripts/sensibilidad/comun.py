"""Utilidades comunes de los análisis de sensibilidad registrados de C1 (docs/08_evaluacion.md, V2.8, puntos 1 a 7,
y enmiendas E3, E4 y E5), calculados después de abrir el test con el protocolo congelado.

Código posterior a la congelación del protocolo para análisis registrados antes de ella (docs/16). No sustituye a
ningún fichero congelado: importa y llama a las funciones de `src.eval` sin cambiarlas.

- Lectura de predicciones y características: `src.eval.c1_auditoria.Auditoria.preds` y `.matriz`,
  `src.eval.audit.cargar_c1` y `src.eval.audit.detector_c1` (mismo detector de V2.3.1, con C, capa y TTA como
  parámetros; el ajuste es el registrado: solo positivos de bimcv de entrenamiento).
- Remuestreo: `src.eval.dosis.auc_remuestreo` (pesos de `pesos_remuestreo`, B = 2 000, semilla 2026). Cada columna
  de su salida depende solo de las etiquetas, de las unidades de remuestreo y de la serie, de modo que las AUC
  remuestreadas de cada serie se guardan en caché (logs/sensibilidad_v28/cache/) y se combinan después con la misma
  aritmética que `src.eval.audit.diferencia` y `familia_no_inferioridad`: media por semilla de las diferencias
  (nanmean), percentiles con `dosis._ic`, valores p con `dosis._p_mayor` y Holm con `stats.holm` al 0.025. La
  equivalencia exacta con `results/c1/test/f_c1.csv`, `f_c1b.csv` y `detectores.csv` se comprueba con
  `python scripts/sensibilidad/v28.py s0`.
- Todo camino que lee el test pasa por `src.eval.desbloqueo.exigir_congelacion` (una vez por proceso).
"""
from __future__ import annotations

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

from src.eval import audit as A  # noqa: E402
from src.eval import stats as S  # noqa: E402
from src.eval import dosis as DO  # noqa: E402
from src.eval.c1_auditoria import Auditoria, Rutas, leer_lista  # noqa: E402

LISTA = ROOT / "configs" / "gate7_lista_c1.yaml"
OUT = ROOT / "results" / "sensibilidad_v28"
CACHE = ROOT / "logs" / "sensibilidad_v28" / "cache"
B, SEMILLA = S.DEFAULT_B, S.DEFAULT_BOOT_SEED
MARGENES = (A.MARGEN_NI, A.MARGEN_NI_SENS)
NOMBRES = {"erm": "ERM", "control_fuente": "Control de fuente", "groupdro": "GroupDRO", "dann": "DANN",
           "coral": "CORAL", "mixstyle": "MixStyle", "fourier": "Fourier", "lisa": "LISA",
           "dfa_fuente": "DFA con fuente", "dfr": "DFR", "erm_mascara": "ERM con máscara", "erm_cuotas": "ERM con cuotas"}


class Presupuesto(Exception):
    """Se agotó el tiempo pedido con --presupuesto-s; lo calculado queda en caché y basta con repetir la orden."""


class Ctx:
    """Estado de un proceso: lista cerrada, registro de la congelación, auditoría para leer predicciones y reloj."""

    def __init__(self, motivo: str, presupuesto_s: float = 0.0, usar_cache: bool = True, test_pred=None):
        from src.eval.desbloqueo import exigir_congelacion
        self.t0 = time.time()
        self.presupuesto = float(presupuesto_s or 0)
        self.usar_cache = usar_cache
        self.L = leer_lista(LISTA)
        self.registro = exigir_congelacion("sensibilidad registrada V2.8: " + motivo)
        self.aud = Auditoria("test", self.L, Rutas(test_pred=test_pred), registro=self.registro)
        self.metodos = list(self.L["metodos"])
        self.semillas = [int(s) for s in self.L["semillas"]]
        CACHE.mkdir(parents=True, exist_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)

    def vigilar(self):
        if self.presupuesto and time.time() - self.t0 > self.presupuesto:
            raise Presupuesto("presupuesto de %.0f s agotado" % self.presupuesto)

    # ------------------------------------------------------------------ predicciones de los métodos
    def estado(self, metodo: str, estado: str) -> str:
        """Nombre del fichero de estado: best, ema o last; en DFR, dfr_best, dfr_ema o dfr_last (enmienda E4)."""
        return ("dfr_" + estado) if metodo == "dfr" else estado

    def metodos_en(self, conjunto: str, estado: str = "best", test_pred=None):
        """{método: (df de referencia, P (S, n), semillas)} alineados a las imágenes del primer método (ERM)."""
        aud = self.aud if test_pred is None else Auditoria("test", self.L, Rutas(test_pred=test_pred), registro=self.registro)
        out, ids0 = {}, None
        for m in self.metodos:
            fr = aud.preds(m, conjunto, self.estado(m, estado))
            if len(fr) < len(self.semillas):
                raise FileNotFoundError("%s, %s, %s: %d de %d semillas" % (m, conjunto, estado, len(fr), len(self.semillas)))
            df, P, sem = aud.matriz(fr)
            if ids0 is None:
                ids0, base = df["image_id"].tolist(), df
            else:
                P = np.vstack([A.alinear(ids0, df["image_id"], p) for p in P])
                df = S.align(base, df, "erm", m)
            out[m] = (base, P, sem)
        return out

    # ------------------------------------------------------------------ detectores
    def puntuaciones(self, tag: str, C: float, feat: str, tta: str) -> pd.DataFrame | None:
        """Puntuación del detector de V2.3.1 (src.eval.audit.detector_c1) en el test limpio completo, con caché.
        Devuelve None si el tronco no tiene la capa pedida (D_IN no tiene penúltima capa entrenada)."""
        f = CACHE / ("punt_%s_C%s_%s_%s.csv" % (tag, ("%g" % C).replace(".", "p"), feat, tta))
        if self.usar_cache and f.exists():
            return pd.read_csv(f, dtype={"image_id": str, "patient_id": str})
        self.vigilar()
        d = ROOT / "results" / "c1" / "caracteristicas"
        dt = ROOT / "results" / "c1" / "caracteristicas_test"
        z = np.load(d / ("%s.npz" % tag))
        if feat not in z.files:
            return None
        feats, m = A.cargar_c1(d, tag, dt, ["test"], self.registro)
        s, info = A.detector_c1(feats, m, C, feat, tta)
        k = (m["conjunto"] == "test").to_numpy()
        r = m.loc[k, ["image_id", "patient_id", "label", "source", "conjunto"]].copy()
        r["score"] = s[k]
        r["test_bimcv"] = m.loc[k, "test_bimcv"].to_numpy()
        r["proyeccion"] = m.loc[k, "proyeccion"].to_numpy()
        r.to_csv(f, index=False, float_format="%.10g")
        (CACHE / (f.stem + "_info.json")).write_text(json.dumps(S.jsonable(info), ensure_ascii=False), encoding="utf-8")
        return pd.read_csv(f, dtype={"image_id": str, "patient_id": str})

    def troncos(self) -> dict:
        L, sem = self.L, self.semillas
        return {"D_IN": [L["troncos"]["D_IN"]],
                "D_ERM": [L["troncos"]["D_ERM"].format(semilla=s) for s in sem],
                "D_MASK": [L["troncos"]["D_MASK"].format(semilla=s) for s in sem],
                "D_RES": list(L["troncos"]["D_RES"])}

    # ------------------------------------------------------------------ remuestreo con caché
    def remuestreo(self, y, g, series: dict) -> dict:
        """{nombre: vector (B,) de AUC remuestreadas}, idéntico columna a columna a `dosis.auc_remuestreo` con todas
        las series juntas. Se calcula por tandas de 12 series y se guarda cada una."""
        y = np.asarray(y).astype(int)
        g = np.asarray(g).astype(str)
        hb = hashlib.sha1(y.tobytes() + "\x1f".join(g.tolist()).encode("utf-8") + ("%d_%d" % (B, SEMILLA)).encode()).hexdigest()[:16]
        out, falta = {}, []
        for k, v in series.items():
            v = np.asarray(v, float)
            h = hashlib.sha1(v.tobytes()).hexdigest()[:20]
            f = CACHE / "rem" / hb / ("%s.npy" % h)
            if self.usar_cache and f.exists():
                out[k] = np.load(f)
            else:
                falta.append((k, v, f))
        for i in range(0, len(falta), 12):
            self.vigilar()
            lote = falta[i:i + 12]
            D = DO.auc_remuestreo(y, np.vstack([v for _, v, _ in lote]), g, B, SEMILLA)
            for j, (k, _, f) in enumerate(lote):
                f.parent.mkdir(parents=True, exist_ok=True)
                np.save(f, D[:, j])
                out[k] = D[:, j]
        return out


# ====================================================================================== estadísticos
def auc_ic(ctx: Ctx, y, s, g) -> dict:
    """Como `src.eval.audit.auc_ic`."""
    D = ctx.remuestreo(y, g, {"s": s})["s"]
    lo, hi = DO._ic(D)
    y = np.asarray(y).astype(int)
    return {"auc": S.auc_roc(y, np.asarray(s, float)), "ic95_inf": lo, "ic95_sup": hi, "n": int(len(y)),
            "n_neg": int(len(y) - y.sum()), "n_unidades": int(len(np.unique(np.asarray(g).astype(str)))),
            "B_validos": int(np.isfinite(D).sum())}


def auc_media(ctx: Ctx, y, P, g) -> dict:
    """Como `src.eval.audit.auc_media`: media por semilla y percentiles de la media remuestreada."""
    P = np.atleast_2d(np.asarray(P, float))
    R = ctx.remuestreo(y, g, {i: P[i] for i in range(P.shape[0])})
    D = np.column_stack([R[i] for i in range(P.shape[0])])
    a = np.array([S.auc_roc(np.asarray(y).astype(int), p) for p in P])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = DO._ic(np.nanmean(D, axis=1))
    return {"auc": float(a.mean()), "ic95_inf": lo, "ic95_sup": hi,
            "desv_semillas": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "n_semillas": int(len(a))}


def familia(ctx: Ctx, y, g, ref, metodos: dict, semillas: dict, comparables: dict | None = None,
            emparejar: bool = False, semillas_ref=None, margenes=MARGENES) -> pd.DataFrame:
    """Réplica de `src.eval.audit.familia_no_inferioridad` sobre el remuestreo en caché (misma aritmética)."""
    y = np.asarray(y).astype(int)
    ref = np.atleast_2d(np.asarray(ref, float))
    Rr = ctx.remuestreo(y, g, {("ref", i): ref[i] for i in range(ref.shape[0])})
    a_ref = [S.auc_roc(y, ref[i]) for i in range(ref.shape[0])]
    filas = []
    for met, P in metodos.items():
        P = np.atleast_2d(np.asarray(P, float))
        sem = list(semillas.get(met, range(P.shape[0])))
        if emparejar:
            idx = {str(s): i for i, s in enumerate(semillas_ref)}
            comunes = [i for i, s in enumerate(sem) if str(s) in idx]
            ir = [idx[str(sem[i])] for i in comunes]
            P, sem = P[comunes], [sem[i] for i in comunes]
        else:
            ir = [0] * P.shape[0]
        Rm = ctx.remuestreo(y, g, {(met, i): P[i] for i in range(P.shape[0])})
        D1 = np.column_stack([Rr[("ref", j)] for j in ir])
        D2 = np.column_stack([Rm[(met, i)] for i in range(P.shape[0])])
        dd = np.nanmean(D1 - D2, axis=1)
        a1 = np.array([a_ref[j] for j in ir])
        a2 = np.array([S.auc_roc(y, P[i]) for i in range(P.shape[0])])
        lo, hi = DO._ic(dd)
        fila = {"metodo": met, "n_semillas": int(P.shape[0]), "semillas": ",".join(map(str, sem)),
                "auc_referencia": float(a1.mean()), "auc_metodo": float(a2.mean()), "delta": float((a1 - a2).mean()),
                "ic95_inf": lo, "ic95_sup": hi, "p_superioridad": DO._p_mayor(dd),
                "comparable": True if comparables is None else bool(comparables.get(met, True)), "n": int(len(y))}
        for mg in margenes:
            fila["p_ni_%.2f" % mg] = DO._p_mayor(dd, -float(mg))
        filas.append(fila)
    t = pd.DataFrame(filas)
    for mg in margenes:
        c = "p_ni_%.2f" % mg
        t["p_holm_%.2f" % mg] = np.nan
        t["no_inferior_%.2f" % mg] = False
        sub = t[t["comparable"]]
        if len(sub):
            h = S.holm({i: p for i, p in zip(sub.index, sub[c])}, alpha=A.ALPHA_HOLM)
            for i in sub.index:
                t.loc[i, "p_holm_%.2f" % mg] = h[i]["p_holm"]
                t.loc[i, "no_inferior_%.2f" % mg] = bool(h[i]["rechaza"])
    return t


def ref_alineada(p: pd.DataFrame, ids, columna="score") -> np.ndarray:
    return A.alinear(ids, p["image_id"], p[columna].to_numpy())


def f_c1(ctx, mets, ref_scores, unidad="paciente"):
    """F-C1 con una referencia única (D_IN o una variante) frente a los doce métodos, en las imágenes de `mets`."""
    base = next(iter(mets.values()))[0]
    ids, y = base["image_id"].tolist(), base["label"].to_numpy()
    g = base["patient"].to_numpy() if unidad == "paciente" else base["image_id"].to_numpy()
    ref = ref_alineada(ref_scores, ids)[None]
    return familia(ctx, y, g, ref, {m: v[1] for m, v in mets.items()}, {m: v[2] for m, v in mets.items()})


def f_c1b(ctx, mets, refs_por_semilla: dict, unidad="paciente"):
    """F-C1b: referencia por semilla (D_ERM o una variante), emparejada con las semillas de cada método."""
    base = next(iter(mets.values()))[0]
    ids, y = base["image_id"].tolist(), base["label"].to_numpy()
    g = base["patient"].to_numpy() if unidad == "paciente" else base["image_id"].to_numpy()
    rs = list(refs_por_semilla)
    R = np.vstack([ref_alineada(refs_por_semilla[s], ids) for s in rs])
    return familia(ctx, y, g, R, {m: v[1] for m, v in mets.items()}, {m: v[2] for m, v in mets.items()},
                   emparejar=True, semillas_ref=rs)


# ====================================================================================== salida
def escribir(nombre: str, tablas: dict, ctx: Ctx, orden: str, notas=()):
    """CSV por tabla, un Markdown con todas y un JSON con la procedencia."""
    from src.eval.dosis import a_markdown
    md = ["# %s" % nombre, "",
          "Generado por `%s` (%s UTC). B = %d, semilla %d. Código posterior a la congelación para un análisis "
          "registrado antes de ella (docs/16_sensibilidad_registrada.md). Congelación: %s." %
          (orden, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), B, SEMILLA, ctx.registro.get("sha256_congelacion")), ""]
    for n in notas:
        md += ["- " + n]
    if notas:
        md.append("")
    for k, t in tablas.items():
        t.to_csv(OUT / ("%s.csv" % k), index=False)
        md += ["## %s" % k, "", a_markdown(t) if len(t) else "(sin filas)", ""]
    (OUT / ("%s.md" % nombre)).write_text("\n".join(md), encoding="utf-8")
    meta = {"orden": orden, "fecha_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "B": B,
            "semilla_bootstrap": SEMILLA, "congelacion": ctx.registro.get("sha256_congelacion"),
            "tablas": {k: len(v) for k, v in tablas.items()}, "notas": list(notas),
            "segundos": round(time.time() - ctx.t0, 1)}
    (OUT / ("%s.json" % nombre)).write_text(json.dumps(S.jsonable(meta), indent=1, ensure_ascii=False), encoding="utf-8")
    print("escrito %s (%s)" % (OUT / ("%s.md" % nombre), ", ".join("%s: %d filas" % (k, len(v)) for k, v in tablas.items())))
