"""Evaluación del test de COVIDx desde los estados guardados de las corridas finales de B1 (hallazgos B1 y M6).

Tres modos, en este orden:

    python scripts/eval/evaluar_test_b1.py --modo huellas     # SHA256 de cada estado y configuración (sin test)
    python scripts/eval/evaluar_test_b1.py --modo comprobar   # recalcula la validación desde los estados (sin test)
    python scripts/eval/evaluar_test_b1.py --modo test        # solo tras la congelación del protocolo (exigir_congelacion)

- huellas: results/gate7/estados_b1.json con el SHA256 y el tamaño de cada `states/*.pt`, de `config_resolved.json` y
  de `summary.json` de `results/train/baseline_<metodo>_final/seed<S>` (lista cerrada de configs/gate7_lista_c1.yaml).
  Código 0 si están las 60 corridas terminadas; 2 si falta alguna (el fichero se escribe igual, con "completo": false).
- comprobar: reconstruye el entrenador de cada corrida, carga el estado y predice los conjuntos de validación que el
  entrenador escribió (`predictions/<estado>_<conjunto>.csv`) con el mismo preprocesado, el mismo promediado de
  volteos (eval.tta) y la misma precisión; compara imagen a imagen. Tolerancia: |Δp| máximo de 2e-3 y |ΔAUC| de 2e-4
  (docs/14, Section 4: el entrenador evalúa en bf16 con algoritmos deterministas en modo de aviso; las únicas
  diferencias admisibles son de redondeo de bf16 en el logit, del orden de 1e-3 en la probabilidad, mientras que un
  estado o un preprocesado equivocados dan diferencias de 1e-2 o más). Escribe results/gate7/comprobacion_estados_b1.json;
  código 0 si todo cuadra, 1 si algo no, 2 si faltan corridas.
- test: exige la congelación; cada estado y su configuración deben estar en ella con la misma huella, y la
  comprobación debe haber terminado sin discrepancias. Predice el test limpio completo y escribe
  `results/b1/test/<experimento>/seed<S>/predictions/<estado>_test.csv` y `<estado>_test_bimcv.csv` (la parte bimcv,
  conjunto confirmatorio de F-C1, extraída de las mismas predicciones).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import c1_datos as CD  # noqa: E402
from src.eval import desbloqueo as D  # noqa: E402

LISTA = ROOT / "configs" / "gate7_lista_c1.yaml"
TOL_P = 2e-3
TOL_AUC = 2e-4
CONJ_VAL = ("val_robust_sel", "val_robust", "val")


def lista(path=LISTA) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def corridas(L: dict, raiz: Path, metodos=None, semillas=None):
    for m in metodos or L["metodos"]:
        for s in semillas or L["semillas"]:
            yield m, int(s), raiz / "results" / "train" / L["experimento"].format(metodo=m) / ("seed%d" % int(s))


def estados_de(run: Path):
    d = run / "states"
    return sorted(p.stem for p in d.glob("*.pt")) if d.is_dir() else []


def estado_principal(L, metodo):
    return (L.get("estado_principal") or {}).get(metodo, (L.get("estado_principal") or {}).get("defecto", "best"))


def modo_huellas(L, raiz: Path, out: Path, metodos=None, semillas=None) -> int:
    filas, faltan = [], []
    for m, s, run in corridas(L, raiz, metodos, semillas):
        term = (run / "summary.json").exists()
        f = {"metodo": m, "semilla": s, "corrida": CD.rel(run), "terminada": term, "estados": {}}
        for k in ("config_resolved.json", "summary.json"):
            if (run / k).exists():
                f["sha256_" + k.split(".")[0]] = D.sha256_fichero(run / k)
        for st in estados_de(run):
            p = run / "states" / ("%s.pt" % st)
            f["estados"][st] = {"sha256": D.sha256_fichero(p), "bytes": p.stat().st_size}
        if not term or estado_principal(L, m) not in f["estados"]:
            faltan.append("%s s%d" % (m, s))
        filas.append(f)
    doc = {"fecha_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "completo": not faltan, "faltan": faltan,
           "n_corridas": len(filas), "corridas": filas, "orden": "python scripts/eval/evaluar_test_b1.py --modo huellas"}
    CD.escribir_atomico(out, json.dumps(doc, indent=1, ensure_ascii=False))
    print("huellas de %d corridas en %s; faltan %d%s" % (len(filas), out, len(faltan), (": " + ", ".join(faltan)) if faltan else ""))
    return 0 if not faltan else 2


def comparar(pred: dict, ref) -> dict:
    """Compara unas predicciones recalculadas con el CSV del entrenador, imagen a imagen."""
    import pandas as pd
    from src.eval import stats as S
    a = pd.DataFrame({k: pred[k] for k in ("image_id", "label", "prob")})
    a["image_id"] = a["image_id"].astype(str)
    b = ref.set_index("image_id").loc[a["image_id"]]
    if (b["label"].to_numpy() != a["label"].to_numpy()).any():
        raise S.AlignmentError("etiquetas distintas al recalcular")
    d = np.abs(a["prob"].to_numpy() - b["prob"].to_numpy())
    auc_a, auc_b = S.auc_roc(a["label"].to_numpy(), a["prob"].to_numpy()), S.auc_roc(b["label"].to_numpy(), b["prob"].to_numpy())
    return {"n": int(len(a)), "max_abs_dp": float(d.max()), "media_abs_dp": float(d.mean()),
            "frac_iguales_1e-6": float((d <= 1e-6).mean()), "auc_recalculada": auc_a, "auc_entrenador": auc_b,
            "abs_dauc": float(abs(auc_a - auc_b)), "cumple": bool(d.max() <= TOL_P and abs(auc_a - auc_b) <= TOL_AUC)}


def modo_comprobar(L, raiz: Path, out: Path, estados=None, metodos=None, semillas=None, max_n=0, workers=None,
                   dispositivo=None) -> int:
    from src.eval import stats as S
    vig = CD.Vigilante()
    filas, faltan = [], []
    try:
        for m, s, run in corridas(L, raiz, metodos, semillas):
            if not (run / "summary.json").exists():
                faltan.append("%s s%d" % (m, s))
                continue
            sts = [x for x in (estados or [estado_principal(L, m)]) if (run / "states" / ("%s.pt" % x)).exists()]
            if not sts:
                faltan.append("%s s%d (sin estados %s)" % (m, s, estados or [estado_principal(L, m)]))
                continue
            tr = CD.trainer_de_corrida(run, workers, dispositivo)
            for st in sts:
                p = run / "states" / ("%s.pt" % st)
                CD.cargar_estado(tr, p)
                for conj in CONJ_VAL:
                    f = run / "predictions" / ("%s_%s.csv" % (st, conj))
                    if not f.exists() or conj not in tr.frames:
                        continue
                    ref = S.read_predictions(f, forbid_test=True, context="comprobación de estados")
                    pred = CD.predecir_como_entrenador(tr, conj, max_n)
                    r = comparar(pred, ref)
                    r.update({"metodo": m, "semilla": s, "estado": st, "conjunto": conj, "sha256_estado": D.sha256_fichero(p),
                              "predicciones_entrenador": CD.rel(f), "sha256_predicciones": D.sha256_fichero(f)})
                    filas.append(r)
                    print("%s s%d %s %s: n %d, max|dp| %.2e, |dAUC| %.2e, %s" % (m, s, st, conj, r["n"], r["max_abs_dp"],
                          r["abs_dauc"], "ok" if r["cumple"] else "NO CUMPLE"), flush=True)
                    vig.step()
    except CD.Parada as e:
        print("parada: %s; no se escribe la comprobación" % e)
        return CD.EXIT_PARADA
    finally:
        vig.cerrar()
    malos = [f for f in filas if not f["cumple"]]
    doc = {"fecha_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tolerancia": {"max_abs_dp": TOL_P, "abs_dauc": TOL_AUC},
           "max_n": int(max_n), "completo": not faltan and not max_n, "faltan": faltan, "n_comparaciones": len(filas),
           "n_fuera_de_tolerancia": len(malos), "todo_cumple": bool(filas) and not malos, "comparaciones": filas,
           "orden": "python " + " ".join([Path(sys.argv[0]).as_posix()] + sys.argv[1:])}
    CD.escribir_atomico(out, json.dumps(doc, indent=1, ensure_ascii=False))
    print("%d comparaciones, %d fuera de tolerancia; faltan %d corridas; salida %s" % (len(filas), len(malos), len(faltan), out))
    if malos:
        return 1
    return 0 if not faltan else 2


def modo_test(L, raiz: Path, out_root: Path, comprobacion: Path, estados=None, metodos=None, semillas=None,
              workers=None, dispositivo=None) -> int:
    reg = D.exigir_congelacion("evaluar_test_b1: test limpio de COVIDx (B1) desde los estados finales")
    if not comprobacion.exists():
        raise D.TestBloqueadoError("falta %s (modo comprobar)" % comprobacion)
    D.huella_congelada(reg, comprobacion, raiz)
    cp = json.loads(comprobacion.read_text(encoding="utf-8"))
    if not cp.get("todo_cumple") or not cp.get("completo"):
        raise D.TestBloqueadoError("la comprobación de los estados no está completa o tiene discrepancias")
    vig = CD.Vigilante()
    escritos = []
    try:
        for m, s, run in corridas(L, raiz, metodos, semillas):
            D.huella_congelada(reg, run / "config_resolved.json", raiz)
            principal = estado_principal(L, m)
            sts = estados or ([principal] + [("dfr_" + x if m == "dfr" else x) for x in L.get("estados_sensibilidad", [])])
            sts = [x for x in dict.fromkeys(sts) if (run / "states" / ("%s.pt" % x)).exists()]
            if principal not in sts:
                raise D.TestBloqueadoError("falta el estado principal %s de %s s%d" % (principal, m, s))
            tr = CD.trainer_de_corrida(run, workers, dispositivo)
            df = CD.marco(tr, "test", reg)
            od = out_root / run.parent.name / run.name / "predictions"
            for st in sts:
                p = run / "states" / ("%s.pt" % st)
                D.huella_congelada(reg, p, raiz)
                CD.cargar_estado(tr, p)
                pred = CD.predecir(tr, df, "test")
                from src.eval.predict import save_predictions
                save_predictions(pred, od / ("%s_test.csv" % st))
                bim = np.asarray(df["source"]) == "bimcv"
                if not tr.cfg["data"].get("synthetic"):
                    sub = {k: np.asarray(v)[bim] for k, v in pred.items()}
                else:
                    sub = {k: np.asarray(v)[np.asarray(df["source_idx"]) == 0] for k, v in pred.items()}
                save_predictions(sub, od / ("%s_test_bimcv.csv" % st))
                escritos += [CD.rel(od / ("%s_test.csv" % st)), CD.rel(od / ("%s_test_bimcv.csv" % st))]
                print("%s s%d %s: test %d imágenes, parte bimcv %d" % (m, s, st, len(pred["prob"]), len(sub["prob"])), flush=True)
                vig.step()
    except CD.Parada as e:
        print("parada: %s" % e)
        return CD.EXIT_PARADA
    finally:
        vig.cerrar()
    CD.escribir_atomico(out_root / "evaluacion_test_b1.json", json.dumps(
        {"fecha_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "congelacion": reg["sha256_congelacion"],
         "ficheros": escritos}, indent=1, ensure_ascii=False))
    print("%d ficheros de predicciones en %s" % (len(escritos), out_root))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modo", choices=["huellas", "comprobar", "test"], required=True)
    ap.add_argument("--lista", default=str(LISTA))
    ap.add_argument("--raiz", default=str(ROOT), help="solo pruebas")
    ap.add_argument("--metodos", nargs="*", default=None)
    ap.add_argument("--semillas", nargs="*", type=int, default=None)
    ap.add_argument("--estados", nargs="*", default=None)
    ap.add_argument("--max-n", type=int, default=0, help="solo pruebas (modo comprobar)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--dispositivo", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    L, raiz = lista(a.lista), Path(a.raiz)
    g7 = raiz / "results" / "gate7"
    if a.modo == "huellas":
        return modo_huellas(L, raiz, Path(a.out or g7 / "estados_b1.json"), a.metodos, a.semillas)
    if a.modo == "comprobar":
        if not a.max_n:            # las huellas de la congelación solo con la comprobación completa
            modo_huellas(L, raiz, g7 / "estados_b1.json", a.metodos, a.semillas)
        return modo_comprobar(L, raiz, Path(a.out or g7 / "comprobacion_estados_b1.json"), a.estados, a.metodos,
                              a.semillas, a.max_n, a.workers, a.dispositivo)
    return modo_test(L, raiz, Path(a.out or raiz / "results" / "b1" / "test"), g7 / "comprobacion_estados_b1.json",
                     a.estados, a.metodos, a.semillas, a.workers, a.dispositivo)


if __name__ == "__main__":
    sys.exit(main())
