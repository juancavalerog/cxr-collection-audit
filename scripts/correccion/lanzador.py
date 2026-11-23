"""Recalcula con el factor corregido de la ecuación (17) de docs/08 (ecuación (6) del manuscrito) las tablas cuyo
intervalo usa el término de semillas con más de un estrato, llamando a las funciones congeladas de los informes con
su directorio de salida apuntando a results/correccion_ec6/ (o, para el control con el factor original, a
logs/correccion_ec6/control/). Código posterior a la apertura de los tests; no modifica ningún fichero congelado.

    python scripts/correccion/lanzador.py control b3-best          # factor original: debe reproducir lo publicado
    python scripts/correccion/lanzador.py corregido b3-best        # factor corregido
    python scripts/correccion/lanzador.py ambos todas              # control y corregido de todas las tablas

Tablas: b3-best, b3-bestdc, fuente, orientacion, c3-epoca, invertido, brecha, b2, tronco, etiqueta.
Cada proceso llama una vez a src.eval.desbloqueo.exigir_congelacion y pasa a los informes el registro real que
devuelve. El control compara cada CSV con el publicado (diferencia máxima absoluta por columna numérica y
coincidencia exacta de las no numéricas) y lo escribe en logs/correccion_ec6/control/control_<tabla>.csv; el modo
'ambos' solo escribe la versión corregida si el control reproduce lo publicado.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.correccion import ec6  # noqa: E402
from src.eval import jerarquico as J  # noqa: E402
from src.eval import stats as S  # noqa: E402

OUT = ROOT / "results" / "correccion_ec6"
CTRL = ROOT / "logs" / "correccion_ec6" / "control"
TRAIN = ROOT / "results" / "train"
B, SEMILLA = S.DEFAULT_B, S.DEFAULT_BOOT_SEED
TOL = 1e-9

# tabla -> (subdirectorio relativo a results/ del original, ficheros CSV que se comparan)
TABLAS = {
    "b3-best": ("b3/test_A_best", ["curva.csv", "resumen.csv", "comparacion_erm.csv"]),
    "b3-bestdc": ("b3/test_A_bestdc", ["curva.csv", "resumen.csv", "comparacion_erm.csv"]),
    "fuente": ("c2c3", ["fuente.csv"]),
    "orientacion": ("c2c3", ["orientacion.csv", "orientacion_b_curva.csv"]),
    "c3-epoca": ("c2c3", ["c3-epoca.csv", "c3_epoca_por_metodo.csv"]),
    "invertido": ("c2c3", ["invertido.csv"]),
    "brecha": ("c2c3", ["brecha.csv"]),
    "b2": ("b2/informe_test", ["curva.csv", "comparacion_erm.csv", "seleccion_epoca.csv"]),
    "tronco": ("sensibilidad/tronco", ["concordancia.csv", "curva.csv", "comparacion_erm.csv", "resumen.csv"]),
    "etiqueta": ("sensibilidad/etiqueta", ["concordancia.csv", "curva.csv", "comparacion_erm.csv", "resumen.csv"]),
}


class Ctx:
    def __init__(self, motivo):
        from src.eval.desbloqueo import exigir_congelacion
        self.t0 = time.time()
        self.registro = exigir_congelacion("corrección del factor de la ecuación (17) de docs/08: " + motivo)
        self.t_exigir = round(time.time() - self.t0, 1)

    def verificador(self, motivo):
        return self.registro


def ejecutar(tabla, out, ctx):
    """Llama a la función congelada que produce `tabla` con salida en `out`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if tabla in ("b3-best", "b3-bestdc"):
        from scripts.eval import informe_b3 as IB3
        IB3.informe(TRAIN, out, "b3a", "A", "test_desconf", "best" if tabla == "b3-best" else "bestdc", B, SEMILLA,
                    permitir_test=True, verificador=ctx.verificador)
    elif tabla in ("fuente", "orientacion", "c3-epoca", "invertido", "brecha"):
        from scripts.eval import informe_c2_c3 as IC
        IC.tabla(tabla, TRAIN, out, B, SEMILLA, ctx.verificador)
    elif tabla == "b2":
        from scripts.eval import informe_b2 as IB2
        from src.eval import c2 as C2
        IB2.informe(TRAIN, out, C2.CONJUNTOS_B2, B, SEMILLA, ctx.verificador,
                    str(ROOT / "results" / "b3" / "test_A_best" / "comparacion_erm.csv"))
    elif tabla in ("tronco", "etiqueta"):
        from scripts.eval import informe_sensibilidad as IS
        a = argparse.Namespace(out=str(out.parent), B=B, seed=SEMILLA, root_tronco=None, root_etiqueta=None,
                               root_principal=str(TRAIN), permitir_incompleto=False)
        IS.correr(tabla, a, ctx.verificador)
    else:
        raise ValueError(tabla)


def comparar(dir_a, dir_b, ficheros):
    """Diferencias por fichero y columna entre dos directorios de CSV."""
    filas = []
    for f in ficheros:
        a, b = pd.read_csv(Path(dir_a) / f), pd.read_csv(Path(dir_b) / f)
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            filas.append({"fichero": f, "columna": "(forma)", "max_abs_dif": np.inf, "nan_distintos": -1,
                          "texto_distinto": -1, "filas": len(a)})
            continue
        for c in a.columns:
            x, y = a[c], b[c]
            if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
                x, y = x.to_numpy(float), y.to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                d = float(np.max(np.abs(x[ok] - y[ok]))) if ok.any() else 0.0
                filas.append({"fichero": f, "columna": c, "max_abs_dif": d,
                              "nan_distintos": int((np.isfinite(x) != np.isfinite(y)).sum()), "texto_distinto": 0,
                              "filas": len(x)})
            else:
                filas.append({"fichero": f, "columna": c, "max_abs_dif": 0.0, "nan_distintos": 0,
                              "texto_distinto": int((x.astype(str) != y.astype(str)).sum()), "filas": len(x)})
    return pd.DataFrame(filas)


LOGS = ROOT / "logs" / "correccion_ec6"


def directorio(tabla, modo):
    """Salida provisional de cada ejecución: control (factor original) o corregido, una carpeta por tabla."""
    return (CTRL if modo == "original" else LOGS / "corregido") / TABLAS[tabla][0] / tabla


def publicar(tabla, origen, multiestrato):
    """La versión corregida se copia a results/correccion_ec6/<subdirectorio original>/ solo si el término de
    semillas tuvo más de un estrato en algún nivel; si no, el resultado es idéntico al publicado (se comprueba) y se
    deja en logs/correccion_ec6/identidad/<tabla>/ como demostración, sin publicarse."""
    import shutil
    dest = (OUT / TABLAS[tabla][0]) if multiestrato else (LOGS / "identidad" / tabla)
    dest.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(origen).iterdir()):
        if f.is_file():
            shutil.copy2(f, dest / f.name)
    return dest


def correr(tabla, modo, ctx):
    t0 = time.time()
    out = directorio(tabla, modo)
    ec6.REGISTRO.clear()
    ec6.AVISOS.clear()
    with ec6.parche(modo):
        ejecutar(tabla, out, ctx)
    assert ec6.intacto()
    pub = ROOT / "results" / TABLAS[tabla][0]
    dif = comparar(out, pub, TABLAS[tabla][1])
    dif.insert(0, "tabla", tabla)
    dif.insert(1, "modo", modo)
    CTRL.mkdir(parents=True, exist_ok=True)
    nombre = ("control_%s.csv" if modo == "original" else "cambios_%s.csv") % tabla
    dif.to_csv(CTRL / nombre, index=False)
    meta = {"tabla": tabla, "modo": modo, "segundos": round(time.time() - t0, 1), "t_exigir": ctx.t_exigir,
            "congelacion": ctx.registro.get("sha256_congelacion"), "numpy": np.__version__,
            "python": platform.python_version(), "maquina": platform.node(),
            "max_abs_dif_con_publicado": float(dif["max_abs_dif"].max()),
            "texto_distinto": int(dif["texto_distinto"].sum()), "nan_distintos": int(dif["nan_distintos"].sum()),
            "avisos_V0": list(ec6.AVISOS), "llamadas_extraer_corregido": len(ec6.REGISTRO),
            "niveles_con_mas_de_un_estrato": int(sum(sum(1 for h in r["H"] if h > 1) for r in ec6.REGISTRO)),
            "registro_lambdas": ec6.REGISTRO[:400]}
    (CTRL / (nombre.replace(".csv", ".json"))).write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=float),
                                                         encoding="utf-8")
    reproduce = bool(dif["max_abs_dif"].max() == 0.0 and dif["texto_distinto"].sum() == 0
                     and dif["nan_distintos"].sum() == 0) if modo == "corregido" else bool(
        dif["max_abs_dif"].max() <= TOL and dif["texto_distinto"].sum() == 0 and dif["nan_distintos"].sum() == 0)
    if modo == "corregido":
        multi = meta["niveles_con_mas_de_un_estrato"] > 0
        if not multi and not reproduce:
            raise SystemExit("%s: sin niveles con más de un estrato y, sin embargo, distinto de lo publicado" % tabla)
        dest = publicar(tabla, out, multi)
        meta["publicado_en"] = str(dest.relative_to(ROOT))
        proc = {k: v for k, v in meta.items() if k != "registro_lambdas"}
        proc["nota"] = ("Tabla generada con la función congelada del informe y el factor corregido del término de "
                        "semillas (scripts/correccion/ec6.py; docs/18_correccion_ecuacion6.md). Los textos de "
                        "informe.md/.json que describen el remuestreo son los del programa congelado.")
        (dest / ("procedencia_%s.json" % tabla)).write_text(json.dumps(proc, indent=1, ensure_ascii=False, default=float),
                                                          encoding="utf-8")
        meta["identico_a_lo_publicado"] = reproduce
        (CTRL / (nombre.replace(".csv", ".json"))).write_text(
            json.dumps(meta, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("%s %s: %.1f s; diferencia máxima con lo publicado %.3g; %s" % (
        tabla, modo, meta["segundos"], meta["max_abs_dif_con_publicado"],
        ("reproduce" if reproduce else "difiere") if modo == "original" else
        "%d niveles con más de un estrato; texto distinto en %d celdas; copiado a %s" % (
            meta["niveles_con_mas_de_un_estrato"], meta["texto_distinto"], meta["publicado_en"])), flush=True)
    return reproduce


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modo", choices=["control", "corregido", "ambos"])
    ap.add_argument("tablas", nargs="+", help="nombres de TABLAS o 'todas'")
    a = ap.parse_args(argv)
    tablas = list(TABLAS) if a.tablas == ["todas"] else a.tablas
    for t in tablas:
        if t not in TABLAS:
            raise SystemExit("tabla desconocida: %s" % t)
    ctx = Ctx("%s %s" % (a.modo, " ".join(tablas)))
    fallos = []
    for t in tablas:
        if a.modo in ("control", "ambos"):
            ok = correr(t, "original", ctx)
            if not ok:
                fallos.append(t)
                print("AVISO: %s no reproduce lo publicado con el factor original; no se escribe la versión corregida" % t)
                continue
        if a.modo in ("corregido", "ambos"):
            correr(t, "corregido", ctx)
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
