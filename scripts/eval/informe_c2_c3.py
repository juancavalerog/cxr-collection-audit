"""Tablas de C2 que no da informe_b3.py (V2.4.3 y V2.4.4) y tablas de C3 (enmienda E2 de docs/08_evaluacion.md).
Una orden por tabla; todas leen test_desconf o test_invertido y exigen antes la congelación válida del protocolo
(src.eval.desbloqueo.exigir_congelacion). Con --demo usan un árbol sintético (ninguna cifra es un resultado).

    python scripts/eval/informe_c2_c3.py invertido       # V2.4.3: test invertido frente a test desconfundido
    python scripts/eval/informe_c2_c3.py fuente          # V2.4.3: AUC por fuente en test_desconf
    python scripts/eval/informe_c2_c3.py brecha          # V2.4.3: brecha de especificidad al umbral de Youden de val
    python scripts/eval/informe_c2_c3.py orientacion     # V2.4.4: orientación B positiva, diferencia de caídas
    python scripts/eval/informe_c2_c3.py c3-epoca        # C3 (a): bestdc frente a best, familia F-C3
    python scripts/eval/informe_c2_c3.py c3-spearman     # C3 (c): Spearman entre métodos (descriptivo)
    python scripts/eval/informe_c2_c3.py c3-hiper --root-c3 results/c3_busqueda   # C3 (b), solo si el autor lo aprueba
    python scripts/eval/informe_c2_c3.py todas --demo --out /tmp/c2c3_demo

Salida por defecto: results/c2c3/<tabla>.csv y .md. B = 2 000, semilla 2026, semillas remuestreadas con
SEMILLA_SEMILLAS (src/eval/jerarquico.py).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import c2 as C2  # noqa: E402
from src.eval import c3 as C3  # noqa: E402
from src.eval import dosis as D  # noqa: E402
from src.eval import stats as S  # noqa: E402

TABLAS = ("invertido", "fuente", "brecha", "orientacion", "c3-epoca", "c3-spearman", "c3-hiper")


def _demo(root: Path):
    """Árboles sintéticos de B3a (A y B) y de la búsqueda repetida de C3."""
    D.arbol_sintetico_b3(root / "train", "uno_mejor", metodos=("erm", "groupdro", "dfr", "lisa"), semillas=(0, 1, 2),
                         n_test=120, n_val=50, estados=("best", "bestdc"))
    D.arbol_sintetico_b3(root / "train", "uno_mejor", metodos=("erm",), rhos=(0.5, 0.8, 1.0), semillas=(0, 1),
                         n_test=120, n_val=50, orient="B", estados=("best", "bestdc"))
    for f in (root / "train").glob("b3a_A_dfr_*/seed*/predictions/dfr_*.csv"):     # estados de la etapa ERM de DFR
        shutil.copy(f, f.with_name(f.name[4:]))
    for f in (root / "train").glob("b3a_*/seed*"):
        (f / "summary.json").write_text(json.dumps({"early_stopping": {"best": 0.99}, "estado": "terminado"}))
    import numpy as np
    rng = np.random.default_rng(0)
    for m in ("erm", "lisa"):
        for t in range(8):
            d = root / "c3" / ("b3a_A_%s_rho090_t%d" % (m, t)) / "seed0"
            (d / "predictions").mkdir(parents=True, exist_ok=True)
            (d / "summary.json").write_text(json.dumps({"early_stopping": {"best": 0.98 + 0.001 * rng.integers(0, 5)}}))
            y = (np.arange(200) < 100).astype(int)
            for conj in ("val_desconf", "test_desconf"):
                p = (0.6 + 0.1 * t) * y + rng.normal(0, 1, 200)
                pd.DataFrame({"image_id": ["%s%d" % (conj, i) for i in range(200)], "label": y, "source": 0,
                              "patient": ["p%s%d" % (conj, i) for i in range(200)],
                              "prob": 1 / (1 + np.exp(-p))}).to_csv(d / "predictions" / ("best_%s.csv" % conj), index=False)
    return root / "train", root / "c3"


def tabla(nombre, root, out, B, seed, verificador=None, root_c3=None):
    t0 = time.time()
    meta = {"tabla": nombre, "B": B, "semilla_bootstrap": seed, "root": str(root)}
    if nombre == "invertido":
        dc, _, _ = C2.cargar_b3(root, "test_desconf", verificador=verificador)
        iv, _, _ = C2.cargar_b3(root, "test_invertido", verificador=verificador)
        df = C2.invertido_frente_desconf(dc, iv, B=B, seed=seed)
    elif nombre == "fuente":
        dc, _, _ = C2.cargar_b3(root, "test_desconf", verificador=verificador)
        df = C2.auc_por_fuente(dc, B=B, seed=seed)
    elif nombre == "brecha":
        dc, _, _ = C2.cargar_b3(root, "test_desconf", verificador=verificador)
        um, _, _ = C2.cargar_b3(root, "val", metodos=[m for m in D.METODOS_B3 if m != "dfr"])
        um_dfr, _, _ = C2.cargar_b3(root, "val_sel", metodos=["dfr"])      # DFR: val contiene su reajuste
        um.update(um_dfr)
        df = C2.brecha_especificidad(dc, um, B=B, seed=seed)
    elif nombre == "orientacion":
        da, _, _ = C2.cargar_b3(root, "test_desconf", verificador=verificador)
        db, _, _ = C2.cargar_b3(root, "test_desconf", orient="B", verificador=verificador)
        res = C2.orientacion_b(da, db, B=B, seed=seed)
        df = pd.DataFrame([{k: (v if not isinstance(v, list) else " ".join(map(str, v))) for k, v in res.items()}])
        if db:
            cb, _, _, _ = C2.curva_jerarquica(db, B=B, seed=seed, diseno=(0.5, 0.8, 1.0))
            cb.to_csv(Path(out) / "orientacion_b_curva.csv", index=False)
    elif nombre == "c3-epoca":
        pb, pc = C3._estados_epoca(root, "b3a", "A", "test_desconf", [m for m in D.METODOS_B3 if m != "control_fuente"],
                                   verificador)
        pm, pn = C3.seleccion_epoca(pb, pc, B=B, seed=seed)
        pm.to_csv(Path(out) / "c3_epoca_por_metodo.csv", index=False)
        df = pn
    elif nombre == "c3-spearman":
        dc, _, _ = C2.cargar_b3(root, "test_desconf", verificador=verificador)
        df = C3.spearman_metodos(dc, C3.metrica_seleccion(root), B=B, seed=seed)
    elif nombre == "c3-hiper":
        if root_c3 is None or not Path(root_c3).exists():
            raise SystemExit("c3-hiper necesita --root-c3 con la búsqueda repetida (análisis condicionado, enmienda E2)")
        df = C3.hiperparametros(root_c3, B=B, seed=seed, verificador=verificador)
    else:
        raise ValueError(nombre)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / ("%s.csv" % nombre), index=False)
    meta["segundos"] = round(time.time() - t0, 1)
    (out / ("%s.md" % nombre)).write_text("# %s\n\n%s\n%s\n" % (nombre, json.dumps(meta, ensure_ascii=False),
                                                                D.a_markdown(df, 4)), encoding="utf-8")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tabla", choices=TABLAS + ("todas",))
    ap.add_argument("--root", default=str(ROOT / "results" / "train"))
    ap.add_argument("--root-c3", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "c2c3"))
    ap.add_argument("--B", type=int, default=S.DEFAULT_B)
    ap.add_argument("--seed", type=int, default=S.DEFAULT_BOOT_SEED)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args(argv)
    root, rc3, verificador = Path(a.root), a.root_c3, None
    if a.demo:
        root, rc3 = _demo(Path(tempfile.mkdtemp(prefix="c2c3_demo_")))
        verificador = lambda motivo: {"demostracion_sintetica": motivo}
    else:
        try:
            C2.exigir(["test_desconf"], "informe_c2_c3 %s" % a.tabla)
        except Exception as e:  # noqa: BLE001
            raise SystemExit("estas tablas leen el test: exigen la congelación válida de la Puerta 7: %s" % e)
        verificador = lambda motivo: {"verificado_en_main": motivo}
    nombres = TABLAS if a.tabla == "todas" else (a.tabla,)
    for n in nombres:
        if n == "c3-hiper" and rc3 is None:
            print("c3-hiper: omitida (sin --root-c3; análisis condicionado a la aprobación del autor)")
            continue
        df = tabla(n, root, a.out, a.B, a.seed, verificador, rc3)
        print("%s: %d filas en %s" % (n, len(df), a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
