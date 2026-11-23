#!/usr/bin/env python3
"""Solapamiento entre el conjunto de entrenamiento del segmentador y los benchmarks de derrame y cardiomegalia.

    python3 scripts/suplemento/solape_segmentador.py              # escribe los dos CSV
    python3 scripts/suplemento/solape_segmentador.py --verificar  # reconstruye y compara con lo escrito

Escribe en results/suplemento_fuentes/:

  solape_segmentador.csv            una fila por benchmark, conjunto, fuente y clase: imágenes, imágenes vistas por el
                                    segmentador, fracción de imágenes vistas y fracción de imágenes cuyo paciente tiene
                                    alguna imagen en el conjunto del segmentador.
  solape_segmentador_calidad.csv    calidad media de las pseudomáscaras del benchmark de derrame según la imagen fuera
                                    vista o no por el segmentador, por fuente, papel y clase.

Fuentes (solo texto, manifiestos y la lista de imágenes del segmentador; no se lee ninguna predicción):

  - Reservas de pacientes del benchmark de derrame (entrenamiento, validación y test): fracciones de
    logs/datos_plan_b/b3a_solape_profesor.txt (columnas en_prof y pac_prof) y, para la reserva de test, recuentos de
    imágenes y de imágenes vistas de review/comprobaciones/a4_profesor.out (columna n, filas visto False y True).
    Para las reservas de entrenamiento y de validación ese fichero solo da el recuento conjunto, así que esas filas
    no llevan recuentos.
  - Test desconfundido (el mismo en todos los niveles y semillas): recuento sobre
    data/manifests/bench/b3a/apos/rho0.50_s0.csv (derrame) y data/manifests/bench/b3c/apos/rho0.50_s0.csv
    (cardiomegalia), filas con test_desconf = 1, con la clave de imagen de review/comprobaciones/a4_profesor.py
    (tkey) sobre data/masks/teacher_set/selection.csv. Un paciente cuenta como visto si su identificador figura en esa
    lista, que es la definición con la que se calcularon las fracciones de las reservas.
  - Calidad de las máscaras: columna q_media de a4_profesor.out.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
SALIDA = RAIZ / "results" / "suplemento_fuentes"
F_RESERVAS = "logs/datos_plan_b/b3a_solape_profesor.txt"
F_A4 = "review/comprobaciones/a4_profesor.out"
F_SEL = "data/masks/teacher_set/selection.csv"
MANIFIESTOS = {"derrame": "data/manifests/bench/b3a/apos/rho0.50_s0.csv",
               "cardiomegalia": "data/manifests/bench/b3c/apos/rho0.50_s0.csv"}
FUENTE = {"nih": "NIH", "chexpert": "CheXpert"}
CLASE = {"0": "negativa", "1": "positiva"}
RESERVA = {"train": "reserva de entrenamiento", "val": "reserva de validación", "test": "reserva de test"}
ROL = {"test": "reserva de test", "trval": "reservas de entrenamiento y validación"}


def tkey(p):
    """Clave de imagen de review/comprobaciones/a4_profesor.py (misma regla, copiada sin cambios)."""
    p = p.replace('\\', '/')
    if 'nih/images/' in p:
        return 'nih__' + p.split('/')[-1]
    q = p.split('CheXpert-v1.0-small/')[-1] if 'CheXpert-v1.0-small/' in p else p
    return 'chx__' + q.replace('/', '_', 1).replace('/', '_')


def tabla_jerarquica(texto, n_indices, n_valores):
    """Lee una tabla impresa por pandas con índice jerárquico (los niveles vacíos repiten el anterior)."""
    filas, prev = [], [None] * n_indices
    for lin in texto.splitlines():
        tok = lin.split()
        if len(tok) < n_valores + 1:
            continue
        vals, idx = tok[-n_valores:], tok[:-n_valores]
        try:
            vals = [float(v) for v in vals]
        except ValueError:
            continue
        if len(idx) > n_indices:
            continue
        idx = prev[:n_indices - len(idx)] + idx
        if any(x is None for x in idx):
            continue
        prev = idx
        filas.append(idx + vals)
    return filas


def leer_reservas():
    filas = tabla_jerarquica((RAIZ / F_RESERVAS).read_text(encoding="utf-8"), 3, 2)
    d = pd.DataFrame(filas, columns=["source", "reserva", "label", "en_prof", "pac_prof"])
    if len(d) != 12:
        raise SystemExit("%s: se esperaban 12 filas y hay %d" % (F_RESERVAS, len(d)))
    return d


def leer_a4():
    texto = (RAIZ / F_A4).read_text(encoding="utf-8")
    filas = tabla_jerarquica(texto, 4, 4)
    d = pd.DataFrame(filas, columns=["source", "rol", "label", "visto", "n", "q_media", "q_p05", "pulmon"])
    if len(d) != 16:
        raise SystemExit("%s: se esperaban 16 filas y hay %d" % (F_A4, len(d)))
    m = re.search(r"vistas por el profesor:\s*(\d+)\s+de\s+(\d+)", texto)
    tot = (int(m.group(1)), int(m.group(2))) if m else None
    return d, tot


def construir():
    res = leer_reservas()
    a4, _ = leer_a4()
    filas = []
    # reservas del benchmark de derrame
    for reserva in ("test", "train", "val"):
        for src in ("nih", "chexpert"):
            for lab in ("0", "1"):
                r = res[(res.source == src) & (res.reserva == reserva) & (res.label == lab)].iloc[0]
                n = v = None
                if reserva == "test":
                    g = a4[(a4.source == src) & (a4.rol == "test") & (a4.label == lab)]
                    n = int(g.n.sum())
                    v = int(g[g.visto == "True"].n.sum())
                    if abs(v / n - r.en_prof) > 0.0005 + 1e-12:
                        raise SystemExit("reserva de test %s %s: %d/%d no reproduce en_prof %.3f" % (src, lab, v, n, r.en_prof))
                filas.append({"benchmark": "derrame", "conjunto": RESERVA[reserva], "fuente": FUENTE[src],
                              "clase": CLASE[lab], "n_imagenes": n, "n_vistas": v,
                              "frac_imagenes": (v / n) if n else r.en_prof, "frac_pacientes": r.pac_prof,
                              "origen": "reservas" if n is None else "reservas y recuento de la reserva de test"})
    # test desconfundido
    sel = pd.read_csv(RAIZ / F_SEL, dtype=str)
    vistas = set(sel.image_path.map(tkey))
    pacientes = set(sel.patient_id)
    for bench, man in MANIFIESTOS.items():
        m = pd.read_csv(RAIZ / man, dtype=str)
        t = m[m.test_desconf == "1"].copy()
        t["visto"] = t.image_id.isin(vistas)
        t["pac_visto"] = t.patient_id.isin(pacientes)
        for src in ("nih", "chexpert"):
            for lab in ("0", "1"):
                g = t[(t.source == src) & (t.label == lab)]
                filas.append({"benchmark": bench, "conjunto": "test desconfundido", "fuente": FUENTE[src],
                              "clase": CLASE[lab], "n_imagenes": len(g), "n_vistas": int(g.visto.sum()),
                              "frac_imagenes": float(g.visto.mean()), "frac_pacientes": float(g.pac_visto.mean()),
                              "origen": "manifiesto rho0.50_s0"})
    solape = pd.DataFrame(filas)
    for c in ("n_imagenes", "n_vistas"):
        solape[c] = solape[c].astype("Int64")
    # calidad de las máscaras
    cal = []
    for src in ("nih", "chexpert"):
        for rol in ("test", "trval"):
            for lab in ("0", "1"):
                g = a4[(a4.source == src) & (a4.rol == rol) & (a4.label == lab)].set_index("visto")
                cal.append({"fuente": FUENTE[src], "conjunto": ROL[rol], "clase": CLASE[lab],
                            "n_no_vistas": int(g.loc["False", "n"]), "n_vistas": int(g.loc["True", "n"]),
                            "q_media_no_vistas": g.loc["False", "q_media"], "q_media_vistas": g.loc["True", "q_media"],
                            "diferencia_vistas_menos_no_vistas": round(g.loc["True", "q_media"] - g.loc["False", "q_media"], 4)})
    return solape, pd.DataFrame(cal)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verificar", action="store_true")
    a = ap.parse_args()
    solape, cal = construir()
    rutas = {SALIDA / "solape_segmentador.csv": solape, SALIDA / "solape_segmentador_calidad.csv": cal}
    if a.verificar:
        fallos = 0
        for p, df in rutas.items():
            nuevo = df.to_csv(index=False)
            if not p.exists() or p.read_text(encoding="utf-8") != nuevo:
                print("distinto de lo escrito: %s" % p.relative_to(RAIZ))
                fallos += 1
        print("%d fallos" % fallos)
        return 1 if fallos else 0
    SALIDA.mkdir(parents=True, exist_ok=True)
    for p, df in rutas.items():
        p.write_text(df.to_csv(index=False), encoding="utf-8")
        print("escrito %s (%d filas)" % (p.relative_to(RAIZ), len(df)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
