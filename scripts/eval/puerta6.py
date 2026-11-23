"""Piloto de B3a del plan B: ¿es informativo el benchmark B3a? (PLAN.md Section 13.9; docs/08_evaluacion.md, versión 2,
Section V2.6.1).

    python scripts/eval/puerta6.py                 # lee results/train/b3a_A_<metodo>_rho<RRR>/seed<S>/
    python scripts/eval/puerta6.py --demo no_informativo --out /tmp/p6_demo

Criterio registrado: con ERM, al menos dos semillas comunes y rho en {0.5, 1.0}, la caída de la AUC media por semilla
en val_desconf (validación desconfundida, nunca el test) entre rho = 0.5 y rho = 1.0 es de al menos 0.05, su IC95 por
remuestreo de pacientes excluye el cero y la caída es positiva en cada semilla: "informativo". Si el extremo superior
del IC95 es menor que 0.05: "no informativo" (se rediseña antes de gastar el presupuesto). En otro caso,
"indeterminado" y decide el autor. DFR y GroupDRO, y el nivel 0.9, se informan como descripción sin intervenir.
Códigos de salida: 0 informativo, 2 no informativo, 3 indeterminado, 4 no evaluable.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import dosis as D  # noqa: E402
from src.eval import stats as S  # noqa: E402

CODIGO = {"informativo": 0, "no informativo": 2, "indeterminado": 3, "no evaluable": 4}
METODOS_P6 = ("erm", "dfr", "groupdro")
RHOS_P6 = (0.5, 0.9, 1.0)


def evaluar(root, out, B=S.DEFAULT_B, seed=S.DEFAULT_BOOT_SEED, nota=""):
    inv = D.descubrir(root, "b3a", "A", "best", (D.CONJ_VAL_DESCONF,))
    preds = D.cargar(inv, D.CONJ_VAL_DESCONF) if len(inv) else {}
    preds = {k: v for k, v in preds.items() if k[0] in METODOS_P6 and round(k[1], 4) in RHOS_P6}
    res = D.puerta6(preds, B=B, seed=seed)
    desc = None
    if res["veredicto"] != "no evaluable" and any(k[0] != "erm" for k in preds):
        curva, _, comp, _ = D.curva(preds, B=B, seed=seed, diseno=RHOS_P6)
        desc = {"curva": curva.drop(columns=["aucs_por_semilla"]).to_dict("records"),
                "comparacion_erm": comp.to_dict("records")}
    res.update({"programa": "scripts/eval/puerta6.py", "fecha_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                "criterio": "caída de ERM >= %.2f entre rho 0.5 y 1.0 en val_desconf, IC95 > 0 y positiva en cada semilla"
                            % D.UMBRAL_P6, "descriptivo": desc, "nota": nota,
                "entradas_sha256": {r.ruta: S.sha256_file(r.ruta) for r in inv[inv["existe"]].itertuples()} if len(inv) else {}})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "puerta6.json").write_text(json.dumps(S.jsonable(res), indent=1, ensure_ascii=False), encoding="utf-8")
    md = ["# Puerta 6: piloto de B3a", "", "Veredicto: **%s**. %s" % (res["veredicto"], nota), "",
          "Criterio: " + res["criterio"] + ".", ""]
    if "caida" in res:
        md += ["Caída media por semilla: %.3f (IC95 de %.3f a %.3f); por semilla: %s." % (
            res["caida"], res["ic95"][0], res["ic95"][1],
            ", ".join("%s: %.3f" % kv for kv in res["caida_por_semilla"].items())), ""]
    else:
        md += ["Motivo: %s." % res.get("motivo", ""), ""]
    if desc:
        import pandas as pd
        md += ["## Descriptivo (no interviene en el veredicto)", "", D.a_markdown(pd.DataFrame(desc["curva"])),
               D.a_markdown(pd.DataFrame(desc["comparacion_erm"]))]
    (out / "puerta6.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "results/train"))
    ap.add_argument("--out", default=str(ROOT / "results/gate6"))
    ap.add_argument("--B", type=int, default=S.DEFAULT_B)
    ap.add_argument("--seed", type=int, default=S.DEFAULT_BOOT_SEED)
    ap.add_argument("--demo", choices=D.ESCENARIOS, default=None)
    a = ap.parse_args(argv)
    root, nota = Path(a.root), ""
    if a.demo:
        root = Path(tempfile.mkdtemp(prefix="p6_demo_"))
        D.arbol_sintetico_b3(root, a.demo, metodos=METODOS_P6, rhos=RHOS_P6, semillas=(0, 1))
        nota = "DATOS SINTÉTICOS (escenario %s): no es la Puerta 6." % a.demo
    res = evaluar(root, a.out, a.B, a.seed, nota)
    print("Puerta 6: %s%s" % (res["veredicto"], (" (caída %.3f, IC95 %.3f a %.3f)" % (res["caida"], *res["ic95"]))
                              if "caida" in res else ""))
    return CODIGO[res["veredicto"]]


if __name__ == "__main__":
    sys.exit(main())
