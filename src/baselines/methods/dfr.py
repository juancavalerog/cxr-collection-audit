"""DFR: reajuste de la última capa sobre un conjunto retenido equilibrado [kirichenko2023last].

Conjunto retenido: val_robust partida por paciente en dos mitades fijas (sha1 del identificador de paciente,
independiente de la semilla): `val_robust_fit` para el reajuste y `val_robust_sel` para elegir la regularización y
para la parada temprana de la etapa ERM. BaselineTrainer crea las dos mitades para todos los métodos, y todos
seleccionan y se comparan en val_robust_sel; en DFR, val_robust completa y val_robust_fit no son comparables
(contienen los datos del reajuste) y se marcan así en el registro y en summary.json. CSD-Net importa
`patient_half` y `split_val_robust` de este módulo: no cambiar sus nombres ni su firma. El test no interviene (y
está bloqueado en la fase de piloto y búsqueda).

Etapa 1: ERM con el entrenador común, seleccionando en val_robust_sel. Etapa 2 (antes de la evaluación final), para
cada estado guardado (last, best, ema): rasgos de la penúltima capa (entrada de fc2, modo evaluación, sin aumento
ni TTA) en las dos mitades; estandarización con la media y desviación de val_robust_fit; n_reps submuestras
equilibradas sobre las celdas (y, s) presentes en val_robust_fit, cada una del tamaño de la celda menor;
regresión logística L1 (liblinear) para cada C de la rejilla, con los coeficientes promediados entre submuestras;
se elige C por AUC-ROC en val_robust_sel y se escribe en fc2 (estandarización plegada en los pesos). Solo cambia
fc2: el resto del estado es idéntico. Se guardan los estados dfr_<estado> y se evalúan como los demás.

Benchmarks del plan B (data.bench; docs/13_entrenamiento_plan_b.md, Section 3; docs/08_evaluacion.md, versión 2,
Section V2.4.6). El conjunto retenido es la validación con la distribución de entrenamiento, `val`, partida en
`val_fit` y `val_sel` por paciente y estratificada por celda (src/train/bench.py). La etapa 1 selecciona en `val_sel`
y la variante principal `dfr_best` parte del estado `best`, ajusta en `val_fit` y elige C en `val_sel`. La variante
de sensibilidad `dfr_bestdc` parte de `bestdc` (elegido en `<dc>_sel`) y ajusta en `<dc>_fit`, con `<dc>` el
subconjunto de selección de sensibilidad (`val_desconf` en B3a, `val_robust` en B2). Una variante es no aplicable si
en su mitad de ajuste ninguna fuente tiene las dos clases (B3a con rho = 1.0: sin celdas minoritarias) o si su mitad
de selección no tiene las dos clases: no se escribe su estado ni sus predicciones y summary.json lo registra en
`no_aplicable`. Con menos de 10 imágenes en la celda menor la cifra se marca como inestable (B3a, rho = 0.99).

Limitación: DFR solo puede equilibrar sobre las celdas presentes en el conjunto retenido. val_robust contiene solo
bimcv, así que el equilibrado es por clase dentro de una única fuente; las celdas vacías de la tabla siguen sin
representarse y el método no puede corregirlas.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from src.baselines.common import penultimate
from src.baselines.methods.erm import ERM
from src.baselines.registry import register
from src.train.trainer import atomic_torch_save


def patient_half(pid: str) -> int:
    return int(hashlib.sha1(str(pid).encode("utf-8")).hexdigest(), 16) % 2


def split_val_robust(frame):
    h = frame["patient_id"].astype(str).map(patient_half)
    return frame[h == 0].reset_index(drop=True), frame[h == 1].reset_index(drop=True)


@register
class DFR(ERM):
    name = "dfr"
    title = "DFR (reajuste de la última capa en val_robust equilibrada)"
    ref = "kirichenko2023last"
    labels = ("clase", "fuente (para equilibrar el conjunto retenido)")
    empty_cells = "indefinido en las celdas vacías: solo equilibra las celdas presentes en el retenido (en val_robust, solo bimcv)"
    defaults = {"C_grid": [1.0, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01], "n_reps": 10, "states": ["last", "best", "ema"]}
    search = {}
    hp_source = "artículo original (rejilla de C y 10 submuestras); C se elige en val_robust_sel, sin búsqueda adicional"
    noncomparable_sets = ("val_robust", "val_robust_fit")
    cost = 1.05

    def prepare_data(self, tr):
        if tr.cfg["data"].get("bench"):
            sc = tr.cfg["selection"]
            if sc["subset"] != "val_sel":
                raise ValueError("DFR en un benchmark del plan B exige selection.subset: val_sel (no puede seleccionar "
                                 "con los datos de ajuste val_fit)")
            if sc.get("subset_dc") and not str(sc["subset_dc"]).endswith("_sel"):
                raise ValueError("DFR exige selection.subset_dc: <dc>_sel (p. ej. val_desconf_sel)")
            return
        fit, sel = tr.frames["val_robust_fit"], tr.frames["val_robust_sel"]     # creadas por BaselineTrainer
        if not len(fit) or not len(sel):
            raise ValueError("DFR: una de las mitades de val_robust está vacía")
        if tr.cfg["selection"]["subset"] != "val_robust_sel":
            raise ValueError("DFR exige selection.subset: val_robust_sel (no puede seleccionar con datos de ajuste)")

    @torch.no_grad()
    def features(self, tr, name):
        net = tr.model
        net.eval()
        F_, Y, S, ids = [], [], [], []
        for b in tr.eval_loader(name):
            x = b["image"].to(tr.device).float()
            F_.append(penultimate(net.net, net.net.pooled(x)).float().cpu().numpy())
            Y.append(b["label"].numpy()); S.append(b["source"].numpy()); ids += list(b["image_id"])
        return np.concatenate(F_), np.concatenate(Y).astype(int), np.concatenate(S).astype(int), ids

    def fit_last_layer(self, tr, fit_name="val_robust_fit", sel_name="val_robust_sel"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        Xf, yf, sf, idf = self.features(tr, fit_name)
        Xs, ys, ss, ids_ = self.features(tr, sel_name)
        mu, sd = Xf.mean(0), Xf.std(0) + 1e-8
        Zf, Zs = (Xf - mu) / sd, (Xs - mu) / sd
        groups = sorted(set(zip(yf.tolist(), sf.tolist())))
        nmin = min(int(((yf == g[0]) & (sf == g[1])).sum()) for g in groups)
        rng = np.random.default_rng(tr.seed)
        best = None
        for C in self.p["C_grid"]:
            coefs, ints = [], []
            for _ in range(int(self.p["n_reps"])):
                idx = np.concatenate([rng.choice(np.nonzero((yf == g[0]) & (sf == g[1]))[0], nmin, replace=False)
                                      for g in groups])
                if len(set(yf[idx].tolist())) < 2:
                    raise ValueError("DFR: el conjunto equilibrado no tiene las dos clases")
                lr = LogisticRegression(penalty="l1", C=float(C), solver="liblinear", max_iter=1000)
                lr.fit(Zf[idx], yf[idx])
                coefs.append(lr.coef_[0]); ints.append(lr.intercept_[0])
            w, b = np.mean(coefs, 0), float(np.mean(ints))
            auc = float(roc_auc_score(ys, Zs @ w + b)) if len(set(ys.tolist())) > 1 else float("nan")
            if best is None or auc > best[0]:
                best = (auc, float(C), w, b)
        auc, C, w, b = best
        info = {"C": C, "auc_val_robust_sel": auc, "fit": fit_name, "sel": sel_name, "groups": [list(g) for g in groups], "n_per_group": nmin,
                "n_fit": int(len(yf)), "n_sel": int(len(ys)), "fit_ids": list(map(str, idf)), "sel_ids": list(map(str, ids_))}
        return w / sd, b - float((w * mu / sd).sum()), info

    # ------------------------------------------------------------------ plan B
    @staticmethod
    def _dc_base(tr):
        dc = tr.cfg["selection"].get("subset_dc")
        return dc.rsplit("_", 1)[0] if dc else None

    def bench_sets(self, tr, st):
        """(conjunto de ajuste, conjunto de selección) de la variante dfr_<st> en un benchmark del plan B."""
        if st == "bestdc":
            b = self._dc_base(tr)
            return b + "_fit", b + "_sel"
        return "val_fit", "val_sel"

    @staticmethod
    def bench_variant_reason(tr, fit_name, sel_name):
        fit, sel = tr.frames[fit_name], tr.frames[sel_name]
        mixed = [s for s, g in fit.groupby("source_idx") if g["label"].nunique() == 2]
        if not mixed:
            return ("%s no tiene ninguna fuente con las dos clases (sin celdas minoritarias): el reajuste equilibrado "
                    "por celdas no está definido" % fit_name)
        if sel["label"].nunique() < 2:
            return "%s no tiene las dos clases: no se puede elegir C" % sel_name
        return None

    def noncomparable_bench(self, tr, state, name):
        """Regla completa de la enmienda E5 de docs/08 (hallazgo m4): un estado reajustado no es comparable en ningún
        conjunto que contenga imágenes de su conjunto de reajuste o las comparta con él. Siempre: `val` y el
        subconjunto de sensibilidad completo (`val_desconf` en B3a, `val_robust` en B2), como `src.eval.c2.
        no_comparable`, y la propia mitad de ajuste; además, cualquier otro conjunto evaluado que comparta imágenes
        con la mitad de ajuste (p. ej. `val_sel` en dfr_bestdc). Solo añade marcas: ninguna cifra cambia."""
        if not state.startswith("dfr_"):
            return None
        fit_name, _ = self.bench_sets(tr, state[4:])
        base = fit_name[:-4]
        if name in (base, fit_name):
            return "contiene los datos del reajuste de %s (%s)" % (state, fit_name)
        dc = self._dc_base(tr)
        if name in ("val", dc) or (name.endswith("_fit") and name[:-4] in ("val", dc)):
            return "comparte imágenes con %s, el reajuste de %s (docs/08, enmienda E5)" % (fit_name, state)
        frames = getattr(tr, "frames", {}) or {}
        if name in frames and fit_name in frames:
            com = set(frames[name]["image_id"].astype(str)) & set(frames[fit_name]["image_id"].astype(str))
            if com:
                return "comparte %d imágenes con %s, el reajuste de %s (docs/08, enmienda E5)" % (len(com), fit_name, state)
        return None

    def before_finalize(self, tr):
        bench = bool(tr.cfg["data"].get("bench"))
        sd_dir = tr.out / "states"
        cur = {k: v.clone() for k, v in tr.model.state_dict().items()}
        all_info = {}
        added = []
        for st in self.p["states"]:
            p = sd_dir / ("%s.pt" % st)
            if not p.exists():
                continue
            if bench:
                fit_name, sel_name = self.bench_sets(tr, st)
                why = self.bench_variant_reason(tr, fit_name, sel_name)
                if why:
                    tr.no_aplicable_estados["dfr_" + st] = why
                    all_info[st] = {"no_aplicable": why}
                    tr.log("dfr", state=st, no_aplicable=why)
                    continue
            tr.model.load_state_dict(torch.load(p, map_location=tr.device))
            if bench:
                w, b, info = self.fit_last_layer(tr, fit_name, sel_name)
                info["inestable"] = bool(info["n_per_group"] < 10)
            else:
                w, b, info = self.fit_last_layer(tr)
            fc2 = tr.model.net.fc2
            with torch.no_grad():
                fc2.weight.copy_(torch.tensor(w, dtype=fc2.weight.dtype).view_as(fc2.weight))
                fc2.bias.fill_(b)
            atomic_torch_save(tr.model.state_dict(), sd_dir / ("dfr_%s.pt" % st))
            all_info[st] = info
            added.append("dfr_" + st)
            tr.log("dfr", state=st, C=info["C"], auc_val_robust_sel=info["auc_val_robust_sel"], groups=info["groups"],
                   n_per_group=info["n_per_group"])
        tr.model.load_state_dict(cur)
        with open(sd_dir / "dfr_info.json", "w", encoding="utf-8") as f:
            json.dump(all_info, f, indent=1)
        ev = tr.cfg["eval"]["states"]
        for a in added:
            if a not in ev:
                ev.append(a)
        self.info = all_info
