"""Los siete análisis de C1 (docs/08_evaluacion.md, V2.3.1 a V2.3.7) en validación y, tras la congelación del protocolo, en el test.

Orquestador de `scripts/eval/auditoria_covidx.py --c1 val|test` y `--din`. Las funciones estadísticas están en
`src/eval/audit.py`; la lista cerrada de métodos, semillas, troncos y tablas, en configs/gate7_lista_c1.yaml. Cada tabla
de la lista (T-C1-1 a T-C1-9) se escribe como CSV en la carpeta de salida, con un JSON de entradas y huellas y un
resumen en Markdown. En modo test, `exigir_congelacion` se llama antes de abrir ningún fichero de test o de COVIDGR.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.data.paths import PROJECT_ROOT
from src.eval import audit as A
from src.eval import stats as S
from src.eval.dosis import a_markdown

ROOT = PROJECT_ROOT


def leer_lista(path) -> dict:
    import yaml
    L = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    tta = (L.get("detector") or {}).get("tta", "no")
    L.setdefault("detector", {})["tta"] = "no" if tta in (False, None, "no") else str(tta)
    return L


class Rutas:
    def __init__(self, raiz=ROOT, caracteristicas=None, caracteristicas_test=None, exploracion=None, adquisicion=None,
                 adquisicion_test=None, train=None, test_pred=None, covidgr_pred=None, covidgr_csv=None):
        r = Path(raiz)
        self.raiz = r
        self.car = Path(caracteristicas or r / "results/c1/caracteristicas")
        self.car_test = Path(caracteristicas_test or r / "results/c1/caracteristicas_test")
        self.expl = Path(exploracion or r / "results/diag/expl_lectura")
        self.aq = Path(adquisicion or r / "results/c1/adquisicion")
        self.aq_test = Path(adquisicion_test or r / "results/c1/adquisicion_test")
        self.train = Path(train or r / "results/train")
        self.test_pred = Path(test_pred or r / "results/b1/test")
        self.covidgr_pred = Path(covidgr_pred or r / "results/b1/covidgr")
        self.covidgr_csv = Path(covidgr_csv or r / "data/manifests/bench/covidgr/covidgr.csv")


def _estado(L, metodo):
    ep = L.get("estado_principal") or {}
    return ep.get(metodo, ep.get("defecto", "best"))


class Auditoria:
    def __init__(self, modo: str, L: dict, rutas: Rutas, B=S.DEFAULT_B, seed=S.DEFAULT_BOOT_SEED, experimento=None,
                 semillas=None, tag_derm=None, tag_mask=None, permitir_incompleto=False, tta=None, registro=None):
        if modo not in ("val", "test"):
            raise ValueError(modo)
        self.modo, self.L, self.R, self.B, self.seed = modo, L, rutas, int(B), int(seed)
        self.exp = experimento or L["experimento"]
        self.semillas = [int(s) for s in (semillas or L["semillas"])]
        self.metodos = list(L["metodos"])
        self.tag_derm = tag_derm or L["troncos"]["D_ERM"]
        self.tag_mask = tag_mask or L["troncos"]["D_MASK"]
        self.tta = tta or L["detector"]["tta"]
        self.permitir_incompleto = permitir_incompleto
        self.entradas, self.avisos = {}, []
        self.registro = registro
        if modo == "test" and not registro:
            from src.eval.desbloqueo import exigir_congelacion
            self.registro = exigir_congelacion("auditoría C1 en el test: F-C1, F-C1b, C1-ii a C1-vii y COVIDGR")
        self.ev = "val_robust" if modo == "val" else "test_bimcv"
        self.full = "val" if modo == "val" else "test"
        self.conj_test = ["test", "test_covidgr"] if modo == "test" else []

    # ------------------------------------------------------------------ entradas
    def _sha(self, p: Path):
        try:
            k = str(Path(p).resolve().relative_to(self.R.raiz.resolve()))
        except ValueError:
            k = str(p)
        self.entradas[k] = S.sha256_file(p)

    def tronco(self, tag: str):
        """(feats, meta, puntuación del detector) o None si no hay fichero."""
        for d in (self.R.car, self.R.expl):
            if (d / ("%s.npz" % tag)).exists():
                conj = [c for c in self.conj_test if (self.R.car_test / ("%s__%s.npz" % (tag, c))).exists()]
                if self.modo == "test" and "test" not in conj:
                    raise FileNotFoundError("faltan las características del test de %s en %s" % (tag, self.R.car_test))
                feats, m = A.cargar_c1(d, tag, self.R.car_test, conj, self.registro)
                for f in [d / ("%s.npz" % tag), d / ("%s_meta.csv" % tag)] + \
                         [self.R.car_test / ("%s__%s%s" % (tag, c, e)) for c in conj for e in (".npz", "_meta.csv")]:
                    self._sha(f)
                s, info = A.detector_c1(feats, m, self.L["detector"]["C"], self.L["detector"]["caracteristicas"], self.tta)
                return feats, m, s, info
        return None

    def preds(self, metodo: str, conjunto: str, estado: Optional[str] = None) -> Dict[int, pd.DataFrame]:
        st = estado or _estado(self.L, metodo)
        out = {}
        for sd in self.semillas:
            exp = self.exp.format(metodo=metodo)
            if conjunto in ("test", "test_bimcv"):
                f = self.R.test_pred / exp / ("seed%d" % sd) / "predictions" / ("%s_%s.csv" % (st, conjunto))
            elif conjunto == "test_covidgr":
                f = self.R.covidgr_pred / exp / ("seed%d" % sd) / "predictions" / ("%s_%s.csv" % (st, conjunto))
            else:
                f = self.R.train / exp / ("seed%d" % sd) / "predictions" / ("%s_%s.csv" % (st, conjunto))
            if not f.exists():
                continue
            prohibido = not S.is_test_set(conjunto) and "covidgr" not in conjunto
            if not prohibido and not self.registro:
                from src.eval.desbloqueo import TestBloqueadoError
                raise TestBloqueadoError("predicciones de %s sin congelación" % conjunto)
            out[sd] = S.read_predictions(f, forbid_test=prohibido, context="auditoría C1")
            self._sha(f)
        return out

    def matriz(self, fr: Dict[int, pd.DataFrame]):
        sem = sorted(fr, key=self.semillas.index)
        ref = fr[sem[0]]
        P = np.vstack([S.align(ref, fr[s], "s%d" % sem[0], "s%d" % s)["prob"].to_numpy() for s in sem])
        return ref, P, sem

    def todos(self, conjunto, minimo_completo=True):
        """{método: (df, P, semillas)} con las predicciones disponibles; en el test exige las cinco semillas."""
        out = {}
        for m in self.metodos:
            fr = self.preds(m, conjunto)
            if len(fr) < len(self.semillas):
                msg = "%s en %s: %d de %d semillas" % (m, conjunto, len(fr), len(self.semillas))
                if self.modo == "test" and minimo_completo and not self.permitir_incompleto:
                    raise FileNotFoundError(msg + " (el test exige todas; --permitir-incompleto solo para pruebas)")
                self.avisos.append(msg)
            if fr:
                out[m] = self.matriz(fr)
        return out

    # ------------------------------------------------------------------ análisis
    def ejecutar(self, out: Path, solo_din: bool = False) -> dict:
        t0 = time.time()
        out.mkdir(parents=True, exist_ok=True)
        L, B, sd0 = self.L, self.B, self.seed
        tablas = {}
        # T-C1-1: detectores
        troncos = {"D_IN": [L["troncos"]["D_IN"]]}
        if not solo_din:
            troncos.update({"D_ERM": [self.tag_derm.format(semilla=s) for s in self.semillas],
                            "D_MASK": [self.tag_mask.format(semilla=s) for s in self.semillas],
                            "D_RES": list(L["troncos"]["D_RES"])})
        det, puntuaciones = [], {}
        for nombre, tags in troncos.items():
            for tag in tags:
                r = self.tronco(tag)
                if r is None:
                    self.avisos.append("sin características del tronco %s (%s)" % (tag, nombre))
                    continue
                feats, m, s, info = r
                puntuaciones[tag] = (m, s, feats)
                mk = m[self.ev].to_numpy()
                y, g = m["label"].to_numpy().astype(int), m["patient_id"].astype(str).to_numpy()
                fila = {"detector": nombre, "tag": tag, "conjunto": self.ev, **info}
                rr = A.auc_ic(y[mk], s[mk], g[mk], B, sd0)
                fila.update({"auc": rr["auc"], "ic95_inf": rr["ic95_inf"], "ic95_sup": rr["ic95_sup"], "n": rr["n"],
                             "n_neg": rr["n_neg"], "n_pacientes": rr["n_pacientes"]})
                for pr in ("AP", "PA"):
                    mm = mk & (m["proyeccion"].to_numpy() == pr)
                    if mm.sum() and len(set(y[mm].tolist())) == 2:
                        q = A.auc_ic(y[mm], s[mm], g[mm], B, sd0)
                        fila.update({"auc_%s" % pr: q["auc"], "ic95_inf_%s" % pr: q["ic95_inf"],
                                     "ic95_sup_%s" % pr: q["ic95_sup"], "n_%s" % pr: q["n"]})
                det.append(fila)
        tablas["detectores"] = pd.DataFrame(det)
        din_tag = L["troncos"]["D_IN"]
        if solo_din:
            return self._escribir(out, tablas, t0)
        metodos_ev = self.todos(self.ev)
        conj_comp = [self.ev] + (["val_robust_sel"] if self.modo == "val" else [])
        # T-C1-2 y T-C1-3: F-C1 y F-C1b
        fc1, fc1b, est = [], [], []
        for conj in conj_comp:
            mets = metodos_ev if conj == self.ev else self.todos(conj)
            if not mets:
                continue
            ids0 = None
            Ps, sems, comp = {}, {}, {}
            for met, (df, P, sem) in mets.items():
                if ids0 is None:
                    ids0, y0, g0, proy = df["image_id"].tolist(), df["label"].to_numpy(), df["patient"].to_numpy(), A.proyeccion(df["image_id"])
                    base = df
                else:
                    P = np.vstack([A.alinear(ids0, df["image_id"], p) for p in P])
                Ps[met], sems[met] = P, sem
                comp[met] = not (met == "dfr" and conj == "val_robust")
            if din_tag in puntuaciones:
                m_, s_, _ = puntuaciones[din_tag]
                ref = A.alinear(ids0, m_["image_id"], s_)[None]
                t = A.familia_no_inferioridad(y0, g0, ref, Ps, sems, comparables=comp, B=B, seed=sd0)
                t.insert(0, "conjunto", conj)
                t.insert(0, "referencia", "D_IN")
                fc1.append(t)
                if conj == self.ev:
                    e = A.estratos_proyeccion(y0, g0, proy, ref, {k: v for k, v in Ps.items() if comp[k]}, B, sd0)
                    e.insert(0, "referencia", "D_IN")
                    est.append(e)
            rs = [s for s in self.semillas if self.tag_derm.format(semilla=s) in puntuaciones]
            if rs:
                R = np.vstack([A.alinear(ids0, puntuaciones[self.tag_derm.format(semilla=s)][0]["image_id"],
                                         puntuaciones[self.tag_derm.format(semilla=s)][1]) for s in rs])
                t = A.familia_no_inferioridad(y0, g0, R, Ps, sems, comparables=comp, B=B, seed=sd0, emparejar=True,
                                              semillas_ref=rs)
                t.insert(0, "conjunto", conj)
                t.insert(0, "referencia", "D_ERM")
                fc1b.append(t)
        tablas["f_c1"] = pd.concat(fc1, ignore_index=True) if fc1 else pd.DataFrame()
        tablas["f_c1b"] = pd.concat(fc1b, ignore_index=True) if fc1b else pd.DataFrame()
        # T-C1-4: C1-ii en los troncos con reserva
        res = []
        for tag in L["troncos"]["D_RES"]:
            if tag not in puntuaciones:
                continue
            m_, s_, feats = puntuaciones[tag]
            mm = m_.copy()
            mm["val_robust"] = mm[self.ev]
            cr = A.contrastes_reservado(feats, mm, B=B, seed=sd0)
            if cr:
                ren = (lambda k: k.replace("val_robust", self.ev)) if self.ev != "val_robust" else (lambda k: k)
                res.append({"tag": tag, "conjunto_lectura_c": self.ev,
                            **{ren(k): v for k, v in cr.items() if k != "n"},
                            **{"n_" + k: v for k, v in cr["n"].items()}})
        tablas["c1_ii_reservado"] = pd.DataFrame(res)
        # T-C1-5: C1-iii por colección
        pc, resumen = [], []
        mets_full = self.todos(self.full)
        refs_full = {}
        if din_tag in puntuaciones:
            refs_full["D_IN"] = [din_tag]
        rs_full = [self.tag_derm.format(semilla=s) for s in self.semillas if self.tag_derm.format(semilla=s) in puntuaciones]
        if rs_full:
            refs_full["D_ERM"] = rs_full
        series = {}
        for met, (df, P, sem) in mets_full.items():
            if met == "dfr" and self.modo == "val":
                keep = ~((df["source"].to_numpy() == S.COVIDX_SOURCES.index("bimcv")) & (df["patient"].map(A.mitad).to_numpy() == 0))
                df, P = df[keep].reset_index(drop=True), P[:, keep]
            series[met] = (df, P)
        if series:
            base_df = next(iter(series.values()))[0]
            for nombre, tags in refs_full.items():
                Pm = np.vstack([A.alinear(base_df["image_id"], puntuaciones[t][0]["image_id"], puntuaciones[t][1]) for t in tags])
                series[nombre] = (base_df, Pm)
        for met, (df, P) in series.items():
            dfx = df.copy()
            if np.issubdtype(np.asarray(dfx["source"]).dtype, np.number):
                dfx["source"] = [S.COVIDX_SOURCES[int(i)] for i in dfx["source"]]
            t = A.tabla_por_coleccion_media(dfx, P, B, sd0)
            t.insert(0, "metodo", met)
            pc.append(t)
            for pr in ("todos", "AP", "PA"):
                mk = np.ones(len(dfx), bool) if pr == "todos" else (A.proyeccion(dfx["image_id"]) == pr) | (dfx["source"].to_numpy() != "bimcv")
                for r in A.contrastes_resumen(dfx[mk].reset_index(drop=True), P[:, mk], B, sd0):
                    resumen.append({"metodo": met, "estrato_bimcv": pr, **r})
        tablas["c1_iii_por_coleccion"] = pd.concat(pc, ignore_index=True) if pc else pd.DataFrame()
        tablas["c1_iii_resumen"] = pd.DataFrame(resumen)
        # T-C1-6: C1-iv solo adquisición
        tablas["c1_iv_adquisicion"] = self._adquisicion(metodos_ev.get("erm"))
        # T-C1-7: C1-v pulmones enmascarados
        cv = []
        if "erm" in metodos_ev and "erm_mascara" in metodos_ev:
            dfe, Pe, se = metodos_ev["erm"]
            dfm, Pm, sm = metodos_ev["erm_mascara"]
            comunes = [s for s in se if s in sm]
            Pm2 = np.vstack([A.alinear(dfe["image_id"], dfm["image_id"], Pm[sm.index(s)]) for s in comunes])
            Pe2 = Pe[[se.index(s) for s in comunes]]
            r = A.diferencia(dfe["label"].to_numpy(), Pm2, Pe2, dfe["patient"].to_numpy(), B, sd0, semillas=comunes)
            cv.append({"estimando": "a_erm_mascara_menos_erm", "auc_1": r["auc1"], "auc_2": r["auc2"], "delta": r["delta"],
                       "ic95_inf": r["ic95_inf"], "ic95_sup": r["ic95_sup"], "n_semillas": r["n_semillas"],
                       "semillas": ",".join(map(str, comunes))})
        mt = [self.tag_mask.format(semilla=s) for s in self.semillas if self.tag_mask.format(semilla=s) in puntuaciones]
        if mt:
            m0 = puntuaciones[mt[0]][0]
            mk = m0[self.ev].to_numpy()
            ids = m0.loc[mk, "image_id"].tolist()
            P = np.vstack([A.alinear(ids, puntuaciones[t][0]["image_id"], puntuaciones[t][1]) for t in mt])
            r = A.auc_media(m0.loc[mk, "label"].to_numpy(), P, m0.loc[mk, "patient_id"].astype(str).to_numpy(), B, sd0)
            cv.append({"estimando": "b_detector_tronco_enmascarado", "auc_1": r["auc"], "ic95_inf": r["ic95_inf"],
                       "ic95_sup": r["ic95_sup"], "n_semillas": len(mt), "semillas": ",".join(t.split("_seed")[1].split("_")[0] for t in mt)})
        tablas["c1_v_mascara"] = pd.DataFrame(cv)
        # T-C1-8: COVIDGR
        if self.modo == "test":
            tablas["c1_vi_covidgr"], tablas["c1_vi_gravedad"] = self._covidgr(metodos_ev, puntuaciones)
        else:
            self.avisos.append("C1-vi (COVIDGR) solo existe en el modo test")
        # T-C1-9: C1-vii
        tablas["c1_vii_estratos"] = pd.concat(est, ignore_index=True) if est else pd.DataFrame()
        return self._escribir(out, tablas, t0)

    def _adquisicion(self, erm):
        from src.eval import adquisicion as AQ
        fa = self.R.aq / "ajuste_estadisticos.npz"
        d_ev = self.R.aq if self.modo == "val" else self.R.aq_test
        fe = d_ev / ("%s_estadisticos.npz" % self.ev)
        if not (fa.exists() and fe.exists()):
            self.avisos.append("C1-iv: faltan los estadísticos de adquisición (%s, %s)" % (fa, fe))
            return pd.DataFrame()
        za, ze = np.load(fa), np.load(fe)
        ma = pd.read_csv(self.R.aq / "ajuste_meta.csv", dtype={"patient_id": str})
        me = pd.read_csv(d_ev / ("%s_meta.csv" % self.ev), dtype={"patient_id": str})
        for f in (fa, fe, self.R.aq / "ajuste_meta.csv", d_ev / ("%s_meta.csv" % self.ev)):
            self._sha(f)
        y, g = me["label"].to_numpy().astype(int), me["patient_id"].astype(str).to_numpy()
        p_a = AQ.variante_a(za["X"], ma["label"], ze["X"])
        p_b = AQ.variante_b(ze["X"], y, g)
        P_erm = None
        if erm is not None:
            df, P, sem = erm
            P_erm = np.vstack([A.alinear(me["image_id"], df["image_id"], p) for p in P])
            g = A.alinear(me["image_id"], df["image_id"], np.arange(len(df)))
            g = df["patient"].astype(str).to_numpy()[g.astype(int)]     # mismos pacientes que el resto de C1
        t = pd.DataFrame(A.solo_adquisicion(y, g, p_a, p_b, P_erm, self.B, self.seed))
        t.insert(0, "conjunto", self.ev)
        t["n_ajuste"] = len(ma)
        return t

    def _covidgr(self, metodos_ev, puntuaciones):
        filas, grav = [], []
        if not (self.registro or {}).get("sha256_congelacion"):
            from src.eval.desbloqueo import TestBloqueadoError
            raise TestBloqueadoError("COVIDGR exige la congelación")
        man = pd.read_csv(self.R.covidgr_csv, dtype=str)
        self._sha(self.R.covidgr_csv)
        sev = dict(zip(man["image_id"], man["severidad"].fillna("")))
        for m in self.metodos:
            ext = self.preds(m, "test_covidgr")
            vr = self.preds(m, "val_robust_sel" if m == "dfr" else "val_robust")
            if m not in metodos_ev or not ext or not vr:
                self.avisos.append("C1-vi: faltan predicciones de %s" % m)
                continue
            dfi, Pi, si = metodos_ev[m]
            dfe, Pe, se = self.matriz(ext)
            dfv, Pv, sv = self.matriz(vr)
            com = [s for s in si if s in se and s in sv]
            u = A.youden_por_semilla(dfv["label"].to_numpy(), Pv[[sv.index(s) for s in com]])
            r = A.externa(dfi["label"], Pi[[si.index(s) for s in com]], dfi["patient"], dfe["label"],
                          Pe[[se.index(s) for s in com]], dfe["patient"], u, dfe["image_id"].map(sev).to_numpy(), self.B, self.seed)
            filas.append({"metodo": m, "semillas": ",".join(map(str, com)), **{k: v for k, v in r.items() if k != "sensibilidad_por_gravedad"}})
            for k, v in (r.get("sensibilidad_por_gravedad") or {}).items():
                grav.append({"metodo": m, "gravedad": k, **v})
        refs = {"D_IN": [self.L["troncos"]["D_IN"]],
                "D_ERM": [self.tag_derm.format(semilla=s) for s in self.semillas if self.tag_derm.format(semilla=s) in puntuaciones]}
        for nombre, tags in refs.items():
            tags = [t for t in tags if t in puntuaciones]
            if not tags:
                continue
            Pi, Pe, Pv, u = [], [], [], []
            for t in tags:
                m_, s_, _ = puntuaciones[t]
                ki, ke, kv = m_["test_bimcv"].to_numpy(), (m_["conjunto"] == "test_covidgr").to_numpy(), m_["val_robust"].to_numpy()
                if not ke.any():
                    break
                Pi.append(s_[ki]); Pe.append(s_[ke]); u.append(A.youden_por_semilla(m_.loc[kv, "label"].to_numpy(), s_[kv][None])[0])
                mi, me = m_[ki], m_[ke]
            if len(Pe) != len(tags):
                self.avisos.append("C1-vi: faltan características de COVIDGR de %s" % nombre)
                continue
            r = A.externa(mi["label"], np.vstack(Pi), mi["patient_id"], me["label"], np.vstack(Pe), me["patient_id"], u,
                          me["image_id"].map(sev).to_numpy(), self.B, self.seed)
            filas.append({"metodo": nombre, "semillas": len(tags), **{k: v for k, v in r.items() if k != "sensibilidad_por_gravedad"}})
            for k, v in (r.get("sensibilidad_por_gravedad") or {}).items():
                grav.append({"metodo": nombre, "gravedad": k, **v})
        return pd.DataFrame(filas), pd.DataFrame(grav)

    def _escribir(self, out: Path, tablas: dict, t0) -> dict:
        for k, t in tablas.items():
            t.to_csv(out / ("%s.csv" % k), index=False)
        meta = {"programa": "scripts/eval/auditoria_covidx.py --c1 %s" % self.modo, "modo": self.modo,
                "fecha_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "B": self.B, "semilla_bootstrap": self.seed,
                "unidad_remuestreo": "paciente", "conjunto_confirmatorio": self.ev, "experimento": self.exp,
                "semillas": self.semillas, "detector": dict(self.L["detector"], tta=self.tta),
                "margenes": [A.MARGEN_NI, A.MARGEN_NI_SENS], "congelacion": (self.registro or {}).get("sha256_congelacion"),
                "avisos": self.avisos, "entradas_sha256": self.entradas, "segundos": round(time.time() - t0, 1),
                "papel": "ensayo en validación (no confirmatorio)" if self.modo == "val" else "test tras la Puerta 7"}
        (out / "c1.json").write_text(json.dumps(S.jsonable(meta), indent=1, ensure_ascii=False), encoding="utf-8")
        md = ["# Auditoría C1 (%s)" % ("validación, ensayo" if self.modo == "val" else "test, Puerta 7"), "",
              "Generado por `%s` (%s UTC). B = %d, semilla %d, remuestreo por paciente; detector con TTA '%s'."
              % (meta["programa"], meta["fecha_utc"], self.B, self.seed, self.tta), ""]
        for k, t in tablas.items():
            md += ["## %s" % k, "", a_markdown(t) if len(t) else "(sin datos)", ""]
        if self.avisos:
            md += ["## Avisos", ""] + ["- " + a for a in self.avisos] + [""]
        (out / "c1.md").write_text("\n".join(md), encoding="utf-8")
        return {"tablas": {k: len(v) for k, v in tablas.items()}, "avisos": self.avisos, "tablas_df": tablas}
