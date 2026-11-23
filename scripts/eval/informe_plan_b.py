"""Informe final del plan B para C2, C2-R y C3: reúne en un Markdown las tablas del artículo a partir de las salidas
de los guiones de una tabla (docs/08_evaluacion.md, enmienda E7). No calcula ninguna cifra nueva ni abre ningún
conjunto: solo lee los CSV que escriben

    python scripts/eval/informe_b3.py --conjunto test_desconf --permitir-test --out results/b3/test_A_best
    python scripts/eval/informe_b3.py --conjunto test_desconf --permitir-test --estado bestdc --out results/b3/test_A_bestdc
    python scripts/eval/informe_c2_c3.py todas
    python scripts/eval/informe_b2.py --test --delta-c2 results/b3/test_A_best/comparacion_erm.csv

y, si existen, las tablas de C1 del otro guion de auditoría (directorio --c1, cualquier CSV que contenga).

    python scripts/eval/informe_plan_b.py                     # results/informe_plan_b/informe.md
    python scripts/eval/informe_plan_b.py --demo --out /tmp/pb_demo   # toda la cadena con árboles sintéticos

Lo que falte se lista como pendiente; ninguna tabla se omite en silencio (V2.2.5).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import dosis as D  # noqa: E402

NOMBRES = {"erm": "ERM", "erm_cuotas": "ERM con cuotas", "erm_mascara": "ERM con máscara", "groupdro": "GroupDRO",
           "dann": "DANN", "coral": "CORAL", "mixstyle": "MixStyle", "fourier": "Fourier", "lisa": "LISA",
           "dfa_fuente": "DFA con fuente", "dfr": "DFR", "control_fuente": "Control de fuente"}


def _fmt(v, d=3):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else "%.*f" % (d, v)


def _ic(r, a="ic95_inf", b="ic95_sup", d=3):
    return "%s (%s a %s)" % (_fmt(r["delta"] if "delta" in r else r["estimacion"], d), _fmt(r[a], d), _fmt(r[b], d))


def _leer(p):
    p = Path(p)
    return pd.read_csv(p) if p.exists() else None


def tabla_curva(curva):
    piv = {}
    for r in curva.itertuples():
        piv.setdefault(r.metodo, {})[r.rho] = "%s (%s a %s)" % (_fmt(r.auc_media_semillas), _fmt(r.ic95_inf), _fmt(r.ic95_sup))
    rhos = sorted(curva.rho.unique())
    filas = [[NOMBRES.get(m, m)] + [piv[m].get(x, "no aplicable") for x in rhos] for m in piv]
    return pd.DataFrame(filas, columns=["Método"] + ["%.2f" % x for x in rhos])


def tabla_familias(comp):
    cols = {"F-C2-area": "Área 0.5 a 0.99 (confirmatoria)", "F-C2-099": "rho 0.99", "F-C2-100": "rho 1.0",
            "F-C2-coste": "Coste en rho 0.5 (no inferioridad, 0.01)"}
    filas = {}
    for _, r in comp[comp.familia.isin(list(cols))].iterrows():
        filas.setdefault(r.metodo, {})[r.familia] = "%s; p Holm %s%s" % (_ic(r), _fmt(r.p_holm), " *" if r.rechaza_holm else "")
    return pd.DataFrame([[NOMBRES.get(m, m)] + [v.get(f, "no aplicable") for f in cols] for m, v in filas.items()],
                        columns=["Método"] + list(cols.values()))


def construir(b3, b3dc, c2c3, b2, c1, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Informe del plan B: C2, C2-R y C3", "", "Generado por `python scripts/eval/informe_plan_b.py` (%s UTC). "
          "Todas las cifras proceden de los CSV citados; ninguna se recalcula aquí." % time.strftime("%Y-%m-%d %H:%M", time.gmtime()), ""]
    pend, leidos = [], {}

    def usar(nombre, df):
        if df is None:
            pend.append(nombre)
        return df

    curva = usar("curva de B3a (informe_b3.py, estado best)", _leer(Path(b3) / "curva.csv"))
    comp = usar("comparación con ERM de B3a (informe_b3.py)", _leer(Path(b3) / "comparacion_erm.csv"))
    if curva is not None:
        md += ["## Tabla 2. Curva dosis-respuesta (AUC media por semilla en test_desconf, IC95 jerárquico)", "",
               D.a_markdown(tabla_curva(curva))]
    if comp is not None:
        md += ["## Tabla 3. Comparación con ERM (Δ, IC95 jerárquico y p de Holm; * rechaza al 0.025)", "",
               D.a_markdown(tabla_familias(comp))]
    b2c = usar("comparación de C2-R (informe_b2.py --test)", _leer(Path(b2) / "comparacion_erm.csv"))
    if b2c is not None:
        b2c = b2c.copy()
        b2c["Δ (IC95)"] = [_ic(r) for _, r in b2c.iterrows()]
        cols = ["conjunto", "metodo", "Δ (IC95)", "p_holm", "rechaza_holm"] + [c for c in b2c.columns if c.startswith("coherente")]
        md += ["## Tabla 4. C2-R, tuberculosis: Δ frente a ERM por fuente (secundaria, con la advertencia de potencia)", "",
               D.a_markdown(b2c[cols])]
    c3n = usar("C3 selección de época (informe_c2_c3.py c3-epoca)", _leer(Path(c2c3) / "c3-epoca.csv"))
    c3m = _leer(Path(c2c3) / "c3_epoca_por_metodo.csv")
    if c3n is not None:
        md += ["## Tabla 5. C3: Δsel = AUC(bestdc) - AUC(best) en test_desconf, media sobre métodos (F-C3 en 0.9, 0.95 y 0.99)", "",
               D.a_markdown(c3n)]
    if c3m is not None:
        md += ["### Tabla 5b. Δsel por método y nivel", "", D.a_markdown(c3m[["metodo", "rho", "delta_sel", "ic95_inf", "ic95_sup", "n_semillas"]])]
    for n, f, t in (("S1", "invertido.csv", "Test invertido frente a test desconfundido (V2.4.3)"),
                    ("S2", "fuente.csv", "AUC por fuente en test_desconf (V2.4.3)"),
                    ("S3", "brecha.csv", "Brecha de especificidad al umbral de Youden de val (V2.4.3)"),
                    ("S4", "orientacion.csv", "Orientación B positiva: diferencia de caídas de ERM (V2.4.4)"),
                    ("S5", "c3-spearman.csv", "C3 (c): Spearman entre métodos, métrica de selección frente a AUC en test_desconf"),
                    ("S6", "c3-hiper.csv", "C3 (b): búsqueda repetida con test (solo si el autor la aprobó)")):
        df = _leer(Path(c2c3) / f)
        if df is None:
            pend.append("Tabla %s (%s)" % (n, f))
            continue
        if n == "S2":
            df = df[["metodo", "fuente", "rho", "auc_media_semillas", "ic95_inf", "ic95_sup", "n_semillas"]]
        md += ["## Tabla %s. %s" % (n, t), "", D.a_markdown(df)]
    if b3dc is not None:
        cdc = _leer(Path(b3dc) / "comparacion_erm.csv")
        if cdc is None:
            pend.append("selección con oráculo (informe_b3.py --estado bestdc)")
        else:
            md += ["## Tabla S7. Selección con oráculo (estado bestdc; no entra en ninguna familia, V2.4.5)", "",
                   D.a_markdown(tabla_familias(cdc))]
    if comp is not None:
        sens = comp[comp.papel != "descriptiva"][["metodo", "medida", "delta", "ic95_inf", "ic95_sup", "ic95_inf_pacientes",
                                                  "ic95_sup_pacientes", "ic95_inf_doble", "ic95_sup_doble", "t_p_unilateral"]]
        md += ["## Tabla S8. Sensibilidad del intervalo: jerárquico corregido, pacientes con modelos fijos, doble y t pareada", "",
               D.a_markdown(sens)]
    if c1 and Path(c1).exists():
        for f in sorted(Path(c1).glob("*.csv")):
            md += ["## C1: %s" % f.name, "", D.a_markdown(pd.read_csv(f))]
    else:
        pend.append("tablas de C1 (directorio %s)" % c1)
    for d in (b3, b3dc, c2c3, b2):
        if d and Path(d).exists():
            for f in sorted(Path(d).glob("*.csv")):
                leidos[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    md += ["## Pendientes", ""] + (["- " + p for p in pend] if pend else ["- ninguno"]) + ["", "## Ficheros leídos (SHA256)", ""]
    md += ["- `%s` %s" % kv for kv in sorted(leidos.items())]
    (out / "informe.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return out / "informe.md", pend


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def demo(out, B=150):
    """Cadena completa con árboles sintéticos: informe_b3 (best y bestdc), informe_c2_c3 y informe_b2."""
    t = Path(tempfile.mkdtemp(prefix="pb_demo_"))
    cc = _load("scripts/eval/informe_c2_c3.py", "informe_c2_c3")
    train, c3 = cc._demo(t)
    ver = lambda motivo: {"demostracion_sintetica": motivo}
    b3 = _load("scripts/eval/informe_b3.py", "informe_b3")
    for st in ("best", "bestdc"):
        b3.informe(train, t / ("b3_" + st), conjunto="test_desconf", estado=st, B=B, permitir_test=True,
                   nota="DATOS SINTÉTICOS", verificador=ver)
    for n in cc.TABLAS:
        cc.tabla(n, train, t / "c2c3", B, 2026, ver, c3)
    b2 = _load("scripts/eval/informe_b2.py", "informe_b2")
    from src.eval import c2 as C2
    rb2 = C2.arbol_sintetico_b2(t / "b2train")
    b2.informe(rb2, t / "b2", C2.CONJUNTOS_B2, B, 2026, ver, t / "b3_best" / "comparacion_erm.csv", "DATOS SINTÉTICOS")
    return construir(t / "b3_best", t / "b3_bestdc", t / "c2c3", t / "b2", None, out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--b3", default=str(ROOT / "results" / "b3" / "test_A_best"))
    ap.add_argument("--b3dc", default=str(ROOT / "results" / "b3" / "test_A_bestdc"))
    ap.add_argument("--c2c3", default=str(ROOT / "results" / "c2c3"))
    ap.add_argument("--b2", default=str(ROOT / "results" / "b2" / "informe_test"))
    ap.add_argument("--c1", default=str(ROOT / "results" / "c1_test"))
    ap.add_argument("--out", default=str(ROOT / "results" / "informe_plan_b"))
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args(argv)
    if a.demo:
        p, pend = demo(a.out)
    else:
        p, pend = construir(a.b3, a.b3dc, a.c2c3, a.b2, a.c1, a.out)
    print("informe en %s; pendientes: %d" % (p, len(pend)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
