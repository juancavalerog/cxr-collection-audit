"""Informe de la curva dosis-respuesta de B3a (aporte C2; docs/08_evaluacion.md, versión 2, Section V2.4).

    python scripts/eval/informe_b3.py                                   # ensayo en val_desconf (no abre el test)
    python scripts/eval/informe_b3.py --conjunto test_desconf --permitir-test   # solo tras la congelación del protocolo
    python scripts/eval/informe_b3.py --demo informativo --out /tmp/b3_demo     # árbol sintético, sin datos reales

Lee `results/train/<bench>_<orient>_<metodo>_rho<RRR>/seed<S>/predictions/<estado>_<conjunto>.csv` y escribe en
`--out` (por defecto results/b3/<bench>_<orient>_<conjunto>_<estado>/): `curva.csv`, `resumen.csv`,
`comparacion_erm.csv` (con las familias de Holm registradas) y `informe.md` e `informe.json` con B, semillas, niveles
y la huella SHA256 de cada fichero leído. El estado `best` es el elegido con la validación con la distribución de
entrenamiento (regla principal) y `bestdc` el elegido con val_desconf (sensibilidad, Section V2.4.5).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import c2 as C2  # noqa: E402
from src.eval import dosis as D  # noqa: E402
from src.eval import stats as S  # noqa: E402


def informe(root, out, bench="b3a", orient="A", conjunto=D.CONJ_VAL_DESCONF, estado="best", B=S.DEFAULT_B,
            seed=S.DEFAULT_BOOT_SEED, permitir_test=False, metodos=None, nota="", verificador=None,
            incluir_no_comparables=False):
    """Curva, medidas y comparación con ERM. Desde el 17-11-2026 (enmienda E1 de docs/08) el intervalo y el p
    principales son los del remuestreo jerárquico de semillas y pacientes (src/eval/c2.py); las columnas
    *_pacientes son el remuestreo de pacientes con modelos fijos de la versión 2 (sensibilidad) y t_* la prueba t
    pareada entre semillas (sensibilidad). Las familias son las de la enmienda E5 (solo F-C2-area confirmatoria).
    Un test solo se abre con la congelación válida del protocolo (src.eval.desbloqueo.exigir_congelacion)."""
    if C2.es_bloqueado(conjunto) and not permitir_test:
        S.assert_not_test(conjunto, "informe de B3a antes de la Puerta 7")
    preds, inv, excl = C2.cargar_b3(root, conjunto, estado, bench, orient, metodos, verificador, incluir_no_comparables)
    if inv.empty:
        raise SystemExit("no hay corridas %s_%s_* en %s" % (bench, orient, root))
    faltan = inv[~inv["existe"]]
    curva, resumen, comp, info = C2.curva_jerarquica(preds, B=B, seed=seed)
    comp = C2.familias_c2(comp) if len(comp) else comp
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    curva.to_csv(out / "curva.csv", index=False)
    resumen.to_csv(out / "resumen.csv", index=False)
    comp.to_csv(out / "comparacion_erm.csv", index=False)
    info.update({"programa": "scripts/eval/informe_b3.py", "fecha_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                 "bench": bench, "orientacion": orient, "conjunto": conjunto, "estado": estado, "nota": nota,
                 "ficheros_ausentes": faltan["ruta"].tolist(),
                 "excluidos_no_comparables": dict(zip(excl["ruta"], excl["no_comparable"])),
                 "entradas_sha256": {r.ruta: S.sha256_file(r.ruta) for r in inv[inv["existe"]].itertuples()
                                     if (r.no_comparable == "" or incluir_no_comparables)}})
    (out / "informe.json").write_text(json.dumps(S.jsonable(info), indent=1, ensure_ascii=False), encoding="utf-8")
    conf = comp[comp["papel"] != "descriptiva"] if len(comp) else comp
    cols = [c for c in ("metodo", "medida", "familia", "papel", "delta", "ic95_inf", "ic95_sup", "p_unilateral", "p_holm",
                        "rechaza_holm", "n_familia", "n_semillas", "semillas_a_favor", "ic95_inf_pacientes",
                        "ic95_sup_pacientes", "t_p_unilateral") if c in comp.columns]
    md = ["# Curva dosis-respuesta de %s, orientación %s, conjunto %s, estado %s" % (bench, orient, conjunto, estado), "",
          "Generado por `python scripts/eval/informe_b3.py` (%s UTC). B = %d, semilla %d; remuestreo jerárquico de "
          "semillas y pacientes (enmienda E1 de docs/08); cifra principal: media por semilla. %s" % (
              info["fecha_utc"], B, seed, nota), "",
          "## Curva (AUC media por semilla, IC95 jerárquico y de pacientes)", "",
          D.a_markdown(curva.drop(columns=["aucs_por_semilla"])),
          "## Medidas resumen", "", D.a_markdown(resumen),
          "## Familias registradas frente a ERM (Holm al 0.025 unilateral; solo F-C2-area es confirmatoria)", "",
          D.a_markdown(conf[cols] if len(conf) else conf), "## Todas las comparaciones con ERM", "",
          D.a_markdown(comp[cols] if len(comp) else comp)]
    if len(excl):
        md += ["## Excluidos por no comparables (enmienda E5)", ""] + ["- %s: %s" % kv for kv in zip(excl["ruta"], excl["no_comparable"])] + [""]
    if len(faltan):
        md += ["## Ficheros ausentes", ""] + ["- " + x for x in faltan["ruta"]]
    (out / "informe.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return curva, resumen, comp, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "results/train"))
    ap.add_argument("--bench", default="b3a")
    ap.add_argument("--orient", default="A", choices=["A", "B"])
    ap.add_argument("--conjunto", default=D.CONJ_VAL_DESCONF)
    ap.add_argument("--estado", default="best", choices=["best", "bestdc"],
                    help="en B3a solo existen best y bestdc (enmienda E4)")
    ap.add_argument("--B", type=int, default=S.DEFAULT_B)
    ap.add_argument("--seed", type=int, default=S.DEFAULT_BOOT_SEED)
    ap.add_argument("--permitir-test", action="store_true", help="solo tras la Puerta 7 (protocolo congelado)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--demo", choices=D.ESCENARIOS, default=None, help="usa un árbol sintético (sin datos reales)")
    ap.add_argument("--incluir-no-comparables", action="store_true",
                    help="incluye estados de DFR evaluados en conjuntos que contienen su reajuste (enmienda E5)")
    a = ap.parse_args(argv)
    root, nota = Path(a.root), ""
    if a.demo:
        root = Path(tempfile.mkdtemp(prefix="b3_demo_"))
        D.arbol_sintetico_b3(root, a.demo, metodos=("erm", "groupdro", "dfr", "lisa"))
        nota = "DATOS SINTÉTICOS (escenario %s): ninguna cifra es un resultado." % a.demo
        if a.conjunto.startswith("test"):
            a.permitir_test = True
    verificador = None
    if a.demo:
        verificador = lambda motivo: {"demostracion_sintetica": motivo}       # árbol sintético: no hay test real
    elif a.permitir_test:
        try:
            C2.exigir([a.conjunto], "informe_b3 %s_%s %s %s" % (a.bench, a.orient, a.estado, a.conjunto))
        except Exception as e:  # noqa: BLE001  (TestBloqueadoError u otro fallo de la verificación)
            raise SystemExit("--permitir-test exige la congelación válida de la Puerta 7: %s" % e)
        verificador = lambda motivo: {"verificado_en_main": motivo}
    out = a.out or str(ROOT / "results" / "b3" / ("%s_%s_%s_%s" % (a.bench, a.orient, a.conjunto, a.estado)))
    curva, resumen, comp, info = informe(root, out, a.bench, a.orient, a.conjunto, a.estado, a.B, a.seed,
                                         a.permitir_test, nota=nota, verificador=verificador,
                                         incluir_no_comparables=a.incluir_no_comparables)
    print("métodos %s; niveles %s; salida en %s" % (info["metodos"], info["niveles_por_metodo"].get("erm"), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
