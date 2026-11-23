"""Informe de la réplica en tuberculosis, C2-R (benchmark B2; docs/08_evaluacion.md, V2.5 y enmiendas E1, E2 y E7).

    python scripts/eval/informe_b2.py                          # ensayo en val_robust (no abre ningún test)
    python scripts/eval/informe_b2.py --test                   # Shenzhen y Montgomery; exige la congelación del protocolo
    python scripts/eval/informe_b2.py --test --delta-c2 results/b3/b3a_A_test_desconf_best/comparacion_erm.csv
    python scripts/eval/informe_b2.py --demo --out /tmp/b2_demo

Lee `results/train/b2_<metodo>/seed<S>/predictions/<estado>_<conjunto>.csv` de las corridas finales. Por conjunto
(fuente): AUC media por semilla, Δ frente a ERM con IC95 jerárquico (semillas e imágenes; Shenzhen y Montgomery no
publican paciente y la unidad es la imagen deduplicada) y Holm en F-C2R-shenzhen y F-C2R-montgomery (once
comparaciones cada una, secundarias). Con --delta-c2, coherencia de dirección con Δ(0.99) y con el área de C2. Añade
la selección de época de C3 en B2 (bestdc, elegido en val_robust, frente a best), descriptiva. En validación, DFR no es
comparable en val_robust (enmienda E5) y queda fuera.
"""
from __future__ import annotations

import argparse
import json
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
from src.eval import stats as S  # noqa: E402


def informe(root, out, conjuntos, B, seed, verificador=None, delta_c2=None, nota=""):
    pc, invs = {}, []
    for c in conjuntos:
        pc[c], inv = C2.cargar_b2(root, c, "best", verificador=verificador)
        invs.append(inv)
    inv = pd.concat(invs, ignore_index=True) if invs else pd.DataFrame()
    if inv.empty:
        raise SystemExit("no hay corridas b2_<metodo> en %s" % root)
    dc2 = pd.read_csv(delta_c2) if delta_c2 else None
    curva, comp = C2.informe_b2_tablas(pc, B=B, seed=seed, delta_c2=dc2)
    sel = []
    for c in conjuntos:
        pb, _ = C2.cargar_b2(root, c, "best", verificador=verificador)
        pdc, _ = C2.cargar_b2(root, c, "bestdc", verificador=verificador)
        pb = {(k[0], k[1], k[2]): v for k, v in pb.items() if k[0] != "dfr"}
        pdc = {(k[0], k[1], k[2]): v for k, v in pdc.items() if k[0] != "dfr"}
        if pb and pdc:
            pm, _ = C3.seleccion_epoca(pb, pdc, B=B, seed=seed, niveles_familia=())
            pm.insert(0, "conjunto", c)
            sel.append(pm.drop(columns=["rho"]))
    sel = pd.concat(sel, ignore_index=True) if sel else pd.DataFrame()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    curva.to_csv(out / "curva.csv", index=False)
    comp.to_csv(out / "comparacion_erm.csv", index=False)
    sel.to_csv(out / "seleccion_epoca.csv", index=False)
    info = {"programa": "scripts/eval/informe_b2.py", "fecha_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "conjuntos": list(conjuntos), "B": B, "semilla_bootstrap": seed, "nota": nota,
            "excluidos_no_comparables": inv[(inv.no_comparable != "") & inv.existe]["ruta"].tolist(),
            "ausentes": inv[~inv.existe]["ruta"].tolist(),
            "entradas_sha256": {r.ruta: S.sha256_file(r.ruta) for r in inv[inv.existe].itertuples()}}
    (out / "informe.json").write_text(json.dumps(S.jsonable(info), indent=1, ensure_ascii=False), encoding="utf-8")
    md = ["# C2-R, réplica en tuberculosis (B2)", "", "Generado por `python scripts/eval/informe_b2.py` (%s UTC). B = %d, "
          "semilla %d; remuestreo jerárquico de semillas e imágenes. Advertencia de potencia de V2.5: Montgomery tiene "
          "58 positivas y 80 negativas. %s" % (info["fecha_utc"], B, seed, nota), "",
          "## AUC media por semilla", "", D.a_markdown(curva.drop(columns=["aucs_por_semilla"]) if len(curva) else curva),
          "## Diferencia frente a ERM (F-C2R por fuente, Holm al 0.025 unilateral)", "", D.a_markdown(comp),
          "## Selección de época (C3 en B2, descriptivo)", "", D.a_markdown(sel)]
    (out / "informe.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return curva, comp, sel


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "results" / "train"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--test", action="store_true", help="Shenzhen y Montgomery; solo tras la Puerta 7")
    ap.add_argument("--delta-c2", default=None)
    ap.add_argument("--B", type=int, default=S.DEFAULT_B)
    ap.add_argument("--seed", type=int, default=S.DEFAULT_BOOT_SEED)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args(argv)
    root, nota, verificador = Path(a.root), "", None
    conjuntos = C2.CONJUNTOS_B2 if (a.test or a.demo) else ("val_robust",)
    if a.demo:
        root = C2.arbol_sintetico_b2(Path(tempfile.mkdtemp(prefix="b2_demo_")))
        nota = "DATOS SINTÉTICOS: ninguna cifra es un resultado."
        verificador = lambda motivo: {"demostracion_sintetica": motivo}
    elif a.test:
        try:
            C2.exigir(conjuntos, "informe_b2")
        except Exception as e:  # noqa: BLE001
            raise SystemExit("--test exige la congelación válida de la Puerta 7: %s" % e)
        verificador = lambda motivo: {"verificado_en_main": motivo}
    out = a.out or str(ROOT / "results" / "b2" / ("informe_%s" % ("test" if a.test else "val_robust")))
    curva, comp, _ = informe(root, out, conjuntos, a.B, a.seed, verificador, a.delta_c2, nota)
    print("C2-R: %d métodos, conjuntos %s; salida en %s" % (curva["metodo"].nunique() if len(curva) else 0, list(conjuntos), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
