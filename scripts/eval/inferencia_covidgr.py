"""Inferencia de los doce métodos de B1 en la cohorte externa COVIDGR (C1-vi; docs/08_evaluacion.md, V2.3.6).

Bloqueado: exige la congelación del protocolo (`exigir_congelacion`) y que cada estado y su configuración estén en
ella con la misma huella. Ningún modelo se ajusta, se selecciona ni se recalibra con COVIDGR: se cargan los estados
principales de las corridas finales (best; dfr_best en DFR) y se predice con el mismo preprocesado, el mismo
promediado de volteos y la misma precisión que el entrenador, sobre la caché homogénea de COVIDGR
(data/cache_m/bench_covidgr/homogeneo_224; en ERM con máscara, con las máscaras de data/masks/bench_covidgr).

    python scripts/eval/inferencia_covidgr.py        # salida: results/b1/covidgr/<experimento>/seed<S>/predictions/

Las características de COVIDGR de los detectores (D_IN, D_ERM) se extraen con
`scripts/eval/extraer_caracteristicas.py ... --conjuntos test_covidgr`. La evaluación la hace
`scripts/eval/auditoria_covidx.py --c1 test` (tabla T-C1-8).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import c1_datos as CD  # noqa: E402
from src.eval import desbloqueo as D  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from evaluar_test_b1 import LISTA, corridas, estado_principal, lista  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lista", default=str(LISTA))
    ap.add_argument("--raiz", default=str(ROOT), help="solo pruebas")
    ap.add_argument("--metodos", nargs="*", default=None)
    ap.add_argument("--semillas", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--dispositivo", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    L, raiz = lista(a.lista), Path(a.raiz)
    reg = D.exigir_congelacion("inferencia_covidgr: cohorte externa COVIDGR (C1-vi)")
    out_root = Path(a.out or raiz / "results" / "b1" / "covidgr")
    from src.eval.predict import save_predictions
    vig = CD.Vigilante()
    escritos = []
    try:
        for m, s, run in corridas(L, raiz, a.metodos, a.semillas):
            D.huella_congelada(reg, run / "config_resolved.json", raiz)
            st = estado_principal(L, m)
            p = run / "states" / ("%s.pt" % st)
            D.huella_congelada(reg, p, raiz)
            tr = CD.trainer_de_corrida(run, a.workers, a.dispositivo)
            CD.cargar_estado(tr, p)
            df = CD.marco(tr, "test_covidgr", reg)
            pred = CD.predecir(tr, df, "test_covidgr")
            f = out_root / run.parent.name / run.name / "predictions" / ("%s_test_covidgr.csv" % st)
            save_predictions(pred, f)
            escritos.append(CD.rel(f))
            print("%s s%d %s: COVIDGR %d imágenes" % (m, s, st, len(pred["prob"])), flush=True)
            vig.step()
    except CD.Parada as e:
        print("parada: %s" % e)
        return CD.EXIT_PARADA
    finally:
        vig.cerrar()
    CD.escribir_atomico(out_root / "inferencia_covidgr.json", json.dumps(
        {"fecha_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "congelacion": reg["sha256_congelacion"],
         "ficheros": escritos}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
