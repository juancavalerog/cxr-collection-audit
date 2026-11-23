"""Predicciones del test de COVIDx sin promediado de espejo, desde los estados congelados de las 60 finales de B1
(sensibilidad de la enmienda E3 de docs/08: "todo sin volteo, referencias y métodos").

Código posterior a la congelación para un análisis registrado antes de ella (docs/16). Sigue el camino de
`scripts/eval/evaluar_test_b1.py --modo test` (mismas comprobaciones: congelación, comprobación de estados completa y
sin discrepancias, huella congelada de cada estado y configuración; mismo entrenador reconstruido, mismo conjunto y
mismo preprocesado de `src.eval.c1_datos`), con una diferencia: en cada lote se calculan por separado la probabilidad
de la imagen original, p0 = sigmoid(f(x)), y la de su espejo, p1 = sigmoid(f(flip(x))), con la misma precisión y el
mismo autocast que `src.eval.predict.predict_loader`. Se guarda p0 (sin espejo) y se comprueba que 0.5 (p0 + p1)
reproduce, imagen a imagen, las predicciones con espejo ya publicadas en results/b1/test/ (tolerancia de
evaluar_test_b1.py: |dp| <= 2e-3 y |dAUC| <= 2e-4). Si alguna corrida no cumple, el programa termina con código 1.

    python scripts/sensibilidad/predecir_sin_espejo.py                  # 60 corridas, estado principal (GPU)
    python scripts/sensibilidad/predecir_sin_espejo.py --solo-listar    # comprobaciones sin cargar ningún estado

Salida: results/sensibilidad_v28/sin_espejo_pred/<experimento>/seed<S>/predictions/<estado>_test.csv y
<estado>_test_bimcv.csv, y results/sensibilidad_v28/sin_espejo_pred/control_espejo.json.
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

sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.dont_write_bytecode = True              # no escribir nada en scripts/eval/ (congelado) al importar
import evaluar_test_b1 as EV  # noqa: E402
sys.dont_write_bytecode = False

OUT = ROOT / "results" / "sensibilidad_v28" / "sin_espejo_pred"


def predecir_ambas(tr, df, nombre):
    """Como src.eval.predict.predict_loader con tta=True, pero devuelve p0 y p1 por separado."""
    import torch
    from src.eval.predict import _autocast
    model, device = tr.model, tr.device
    model.eval()
    ids, labels, sources, patients, p0s, p1s = [], [], [], [], [], []
    with torch.no_grad():
        for b in CD.cargador(tr, df, nombre):
            x = b["image"].to(device, non_blocking=True).float()
            with _autocast(device, tr.precision):
                p0 = torch.sigmoid(model(x).float())
                p1 = torch.sigmoid(model(torch.flip(x, dims=[3])).float())
            p0s.append(p0.float().cpu().numpy().reshape(-1))
            p1s.append(p1.float().cpu().numpy().reshape(-1))
            ids += list(b["image_id"]); patients += list(b["patient"])
            labels.append(b["label"].numpy().reshape(-1)); sources.append(b["source"].numpy().reshape(-1))
    base = {"image_id": np.array(ids), "label": np.concatenate(labels).astype(int),
            "source": np.concatenate(sources).astype(int), "patient": np.array(patients)}
    return base, np.concatenate(p0s), np.concatenate(p1s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metodos", nargs="*", default=None)
    ap.add_argument("--semillas", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--dispositivo", default=None)
    ap.add_argument("--solo-listar", action="store_true")
    a = ap.parse_args(argv)
    L = EV.lista()
    reg = D.exigir_congelacion("sensibilidad E3: test de COVIDx sin promediado de espejo desde los estados")
    comp = ROOT / "results" / "gate7" / "comprobacion_estados_b1.json"
    D.huella_congelada(reg, comp, ROOT)
    cp = json.loads(comp.read_text(encoding="utf-8"))
    if not cp.get("todo_cumple") or not cp.get("completo"):
        raise D.TestBloqueadoError("la comprobación de los estados no está completa o tiene discrepancias")
    import pandas as pd
    from src.eval import stats as S
    from src.eval.predict import save_predictions
    filas, escritos = [], []
    vig = None if a.solo_listar else CD.Vigilante()
    try:
        for m, s, run in EV.corridas(L, ROOT, a.metodos, a.semillas):
            st = EV.estado_principal(L, m)
            D.huella_congelada(reg, run / "config_resolved.json", ROOT)
            D.huella_congelada(reg, run / "states" / ("%s.pt" % st), ROOT)
            pub = ROOT / "results" / "b1" / "test" / run.parent.name / run.name / "predictions" / ("%s_test.csv" % st)
            if not pub.exists():
                raise FileNotFoundError("faltan las predicciones con espejo publicadas: %s" % pub)
            if a.solo_listar:
                print("%s s%d %s: estado y configuración congelados; control %s" % (m, s, st, CD.rel(pub)))
                continue
            tr = CD.trainer_de_corrida(run, a.workers, a.dispositivo)
            df = CD.marco(tr, "test", reg)
            CD.cargar_estado(tr, run / "states" / ("%s.pt" % st))
            base, p0, p1 = predecir_ambas(tr, df, "test")
            ref = S.read_predictions(pub, context="control del espejo")
            r = EV.comparar({**base, "prob": 0.5 * (p0 + p1)}, ref)
            r.update({"metodo": m, "semilla": s, "estado": st, "publicadas": CD.rel(pub),
                      "auc_sin_espejo": S.auc_roc(base["label"], p0)})
            filas.append(r)
            od = OUT / run.parent.name / run.name / "predictions"
            save_predictions({**base, "prob": p0}, od / ("%s_test.csv" % st))
            bim = np.asarray(df["source"]) == "bimcv"
            save_predictions({k: np.asarray(v)[bim] for k, v in {**base, "prob": p0}.items()}, od / ("%s_test_bimcv.csv" % st))
            escritos += [CD.rel(od / ("%s_test.csv" % st)), CD.rel(od / ("%s_test_bimcv.csv" % st))]
            print("%s s%d %s: %d imágenes; control con espejo max|dp| %.2e, |dAUC| %.2e, %s" % (
                m, s, st, len(p0), r["max_abs_dp"], r["abs_dauc"], "ok" if r["cumple"] else "NO CUMPLE"), flush=True)
            vig.step()
    except CD.Parada as e:
        print("parada: %s" % e)
        return CD.EXIT_PARADA
    finally:
        if vig is not None:
            vig.cerrar()
    if a.solo_listar:
        return 0
    malos = [f for f in filas if not f["cumple"]]
    CD.escribir_atomico(OUT / "control_espejo.json", json.dumps(
        {"fecha_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "congelacion": reg["sha256_congelacion"],
         "tolerancia": {"max_abs_dp": EV.TOL_P, "abs_dauc": EV.TOL_AUC}, "n_corridas": len(filas),
         "n_fuera_de_tolerancia": len(malos), "todo_cumple": bool(filas) and not malos, "corridas": filas,
         "ficheros": escritos, "orden": "python " + " ".join([Path(sys.argv[0]).as_posix()] + sys.argv[1:])},
        indent=1, ensure_ascii=False))
    print("%d corridas, %d fuera de tolerancia en el control del espejo" % (len(filas), len(malos)))
    return 1 if malos else 0


if __name__ == "__main__":
    sys.exit(main())
