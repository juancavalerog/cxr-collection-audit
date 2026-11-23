#!/usr/bin/env python3
"""Añade a la caché y a las máscaras alineadas de un benchmark (por ejemplo, bench_b3a) las entradas de imágenes
nuevas construidas aparte, sin reescribir ni cambiar ningún fichero de imagen o de máscara existente.

Por qué existe. Las herramientas de caché y máscaras (scripts/build_cache.py, src.data.homogeneo_masks) son
reanudables por fichero, pero reescriben su index.csv con las filas que reciben: lanzadas sobre el directorio de
bench_b3a con solo las imágenes nuevas, dejarían un índice con esas imágenes y nada más; lanzadas con todas,
reescribirían el índice de las máscaras alineadas sin las cajas torácicas de las entradas ya hechas. Por eso las
imágenes nuevas se procesan en un directorio propio (trabajos e_01 a e_03) y este guion las incorpora.

Garantías.
1. Antes de escribir nada comprueba: todas las imágenes pedidas están en los índices de origen con estado ok y su
   fichero existe; la cabecera de cada índice de origen es idéntica, byte a byte, a la del destino; ninguna imagen
   pedida está ya en el destino con contenido distinto (si está con contenido idéntico, se omite).
2. Copia cada fichero nuevo con un enlace atómico que falla si el destino existe (os.link sobre un temporal): nunca
   sobrescribe.
3. Cada index.csv del destino se sustituye de forma atómica (os.replace) por sus bytes anteriores, intactos, seguidos
   de las líneas de las imágenes nuevas tomadas tal cual del índice de origen (mismo escritor, mismo formato).
4. Registra antes y después el tamaño y la fecha de modificación (ns) de todos los ficheros del destino que figuran en
   su índice, la huella SHA256 de una muestra fija de ellos y la de cada index.csv, y comprueba que nada de lo
   anterior cambió y que cada índice nuevo empieza por los bytes del anterior. Termina con código 1 si algo no cuadra.

    python3 scripts/data/fusionar_cache_bench.py --ids logs/datos_extra/b3a_s5_9_imagenes_nuevas.csv \
        --cache-origen data/cache_m/bench_b3a_s5_9/homogeneo_224 --cache-destino data/cache_m/bench_b3a/homogeneo_224 \
        --mascaras-origen data/masks/bench_b3a_s5_9/homogeneo_224 \
        --mascaras-destino data/masks/bench_b3a/homogeneo_224 --registro logs/datos_extra/fusion_b3a_s5_9 [--comprobar]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


class Indice:
    """index.csv leído en bytes: cabecera, líneas por image_id (bytes exactos) y ruta relativa de cada entrada."""

    def __init__(self, d: Path, col_ruta: str):
        self.dir, self.path = d, d / "index.csv"
        self.raw = self.path.read_bytes()
        lines = self.raw.splitlines(keepends=True)
        self.header = lines[0]
        cols = next(csv.reader([lines[0].decode("utf-8")]))
        self.i_ruta, self.i_estado = cols.index(col_ruta), cols.index("status")
        self.lineas, self.ruta, self.estado = {}, {}, {}
        for ln in lines[1:]:
            row = next(csv.reader([ln.decode("utf-8")]))
            iid = row[0]
            if iid in self.lineas:
                raise SystemExit("image_id repetido en %s: %s" % (self.path, iid))
            self.lineas[iid], self.ruta[iid], self.estado[iid] = ln, row[self.i_ruta], row[self.i_estado]

    def ok(self):
        return [i for i, s in self.estado.items() if s == "ok" and self.ruta[i]]


def inventario(idx: Indice) -> dict:
    out = {}
    for iid in idx.ok():
        p = idx.dir / idx.ruta[iid]
        st = p.stat()
        out[iid] = (idx.ruta[iid], st.st_size, st.st_mtime_ns)
    return out


def muestra(ids, n):
    ids = sorted(ids)
    if len(ids) <= n:
        return ids
    paso = len(ids) / n
    return [ids[int(i * paso)] for i in range(n)]


def enlazar_sin_sobrescribir(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(".%s.fusion.%d.tmp" % (dst.name, os.getpid()))
    tmp.write_bytes(src.read_bytes())
    try:
        os.link(tmp, dst)                  # falla con FileExistsError si dst existe: nunca sobrescribe
        return "nuevo"
    except FileExistsError:
        return "existente_igual" if dst.read_bytes() == src.read_bytes() else "existente_distinto"
    finally:
        tmp.unlink()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="CSV con image_id de las imágenes a incorporar")
    ap.add_argument("--cache-origen", required=True)
    ap.add_argument("--cache-destino", required=True)
    ap.add_argument("--mascaras-origen", required=True)
    ap.add_argument("--mascaras-destino", required=True)
    ap.add_argument("--registro", required=True)
    ap.add_argument("--muestra", type=int, default=1000)
    ap.add_argument("--comprobar", action="store_true", help="solo las comprobaciones previas; no escribe")
    a = ap.parse_args(argv)
    t0 = time.time()
    reg = _abs(a.registro)
    reg.mkdir(parents=True, exist_ok=True)
    with open(_abs(a.ids), newline="", encoding="utf-8") as f:
        ids = sorted({r["image_id"] for r in csv.DictReader(f)})
    pares = [("cache", Indice(_abs(a.cache_origen), "rel_path"), Indice(_abs(a.cache_destino), "rel_path")),
             ("mascaras", Indice(_abs(a.mascaras_origen), "path"), Indice(_abs(a.mascaras_destino), "path"))]
    informe = {"imagenes_pedidas": len(ids), "inicio": time.strftime("%Y-%m-%dT%H:%M:%S")}
    errores, plan = [], {}
    for nom, o, d in pares:
        if o.header != d.header:
            errores.append("%s: cabeceras distintas (%r frente a %r)" % (nom, o.header, d.header))
        falta = [i for i in ids if o.estado.get(i) != "ok" or not o.ruta.get(i) or not (o.dir / o.ruta[i]).is_file()]
        if falta:
            errores.append("%s: %d imágenes sin entrada ok en el origen (p. ej. %s)" % (nom, len(falta), falta[0]))
        nuevas, iguales = [], []
        for i in ids:
            if i in d.lineas:
                same = (d.estado[i] == "ok" and d.ruta.get(i) == o.ruta.get(i) and (d.dir / d.ruta[i]).is_file()
                        and (d.dir / d.ruta[i]).read_bytes() == (o.dir / o.ruta[i]).read_bytes())
                (iguales if same else errores).append(i if same else "%s: %s ya está en el destino y difiere" % (nom, i))
            else:
                if (d.dir / o.ruta.get(i, "")).exists() and o.ruta.get(i):
                    if (d.dir / o.ruta[i]).read_bytes() != (o.dir / o.ruta[i]).read_bytes():
                        errores.append("%s: %s tiene fichero en el destino, fuera del índice, y difiere" % (nom, i))
                nuevas.append(i)
        plan[nom] = (o, d, nuevas)
        informe[nom] = {"destino": str(d.dir), "entradas_destino_antes": len(d.lineas), "nuevas": len(nuevas),
                        "ya_presentes_identicas": len(iguales), "sha256_index_antes": sha_bytes(d.raw)}
    if errores:
        informe["errores"] = errores
        (reg / "informe.json").write_text(json.dumps(informe, indent=2, ensure_ascii=False) + "\n")
        print("\n".join(errores[:20]))
        return 2
    antes = {}
    for nom, (o, d, nuevas) in plan.items():
        inv = inventario(d)
        sm = muestra(inv, a.muestra)
        antes[nom] = (inv, {i: sha_file(d.dir / inv[i][0]) for i in sm})
        with open(reg / ("huellas_antes_%s.csv" % nom), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "ruta", "bytes", "mtime_ns", "sha256_muestra"])
            for i in sorted(inv):
                w.writerow([i, inv[i][0], inv[i][1], inv[i][2], antes[nom][1].get(i, "")])
    if a.comprobar:
        informe["modo"] = "comprobar (sin escritura)"
        (reg / "informe_comprobar.json").write_text(json.dumps(informe, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(informe, ensure_ascii=False))
        return 0
    # copia de ficheros (todos los de los dos destinos) y después los índices
    for nom, (o, d, nuevas) in plan.items():
        res = {}
        for i in nuevas:
            r = enlazar_sin_sobrescribir(o.dir / o.ruta[i], d.dir / o.ruta[i])
            res[r] = res.get(r, 0) + 1
            if r == "existente_distinto":
                errores.append("%s: %s apareció en el destino con contenido distinto durante la copia" % (nom, i))
        informe[nom]["copias"] = res
    if errores:
        informe["errores"] = errores
        (reg / "informe.json").write_text(json.dumps(informe, indent=2, ensure_ascii=False) + "\n")
        print("\n".join(errores[:20]))
        return 1
    for nom in ("mascaras", "cache"):
        o, d, nuevas = plan[nom]
        old = d.path.read_bytes()
        if old != d.raw:
            errores.append("%s: el índice del destino cambió durante la fusión" % nom)
            continue
        if not old.endswith(b"\n"):
            errores.append("%s: el índice del destino no termina en salto de línea" % nom)
            continue
        new = old + b"".join(o.lineas[i] for i in nuevas)
        tmp = d.dir / ".index.csv.fusion.tmp"
        tmp.write_bytes(new)
        os.replace(tmp, d.path)
    # comprobación posterior
    for nom, (o, d, nuevas) in plan.items():
        inv0, sm0 = antes[nom]
        d2 = Indice(d.dir, "rel_path" if nom == "cache" else "path")
        cambios = []
        for i, (rel, size, mt) in inv0.items():
            st = (d.dir / rel).stat()
            if st.st_size != size or st.st_mtime_ns != mt:
                cambios.append(i)
        sha_mal = [i for i, h in sm0.items() if sha_file(d.dir / inv0[i][0]) != h]
        pref_ok = d2.raw[:len(d.raw)] == d.raw
        lineas_ok = all(d2.lineas.get(i) == d.lineas[i] for i in d.lineas)
        nuevas_ok = all(d2.estado.get(i) == "ok" and (d.dir / d2.ruta[i]).is_file() for i in nuevas)
        informe[nom].update({"entradas_destino_despues": len(d2.lineas), "sha256_index_despues": sha_bytes(d2.raw),
                             "ficheros_existentes_comprobados": len(inv0), "ficheros_existentes_cambiados": len(cambios),
                             "muestra_sha256": len(sm0), "muestra_sha256_distinta": len(sha_mal),
                             "indice_empieza_por_el_anterior": pref_ok, "lineas_anteriores_intactas": lineas_ok,
                             "nuevas_presentes": nuevas_ok})
        if cambios or sha_mal or not pref_ok or not lineas_ok or not nuevas_ok:
            errores.append("%s: comprobación posterior fallida" % nom)
        with open(reg / ("huellas_despues_%s.csv" % nom), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "ruta", "bytes", "mtime_ns", "sha256_muestra"])
            for i in sorted(inv0):
                st = (d.dir / inv0[i][0]).stat()
                w.writerow([i, inv0[i][0], st.st_size, st.st_mtime_ns,
                            sha_file(d.dir / inv0[i][0]) if i in sm0 else ""])
    informe["errores"] = errores
    informe["segundos"] = round(time.time() - t0, 1)
    (reg / "informe.json").write_text(json.dumps(informe, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(informe, ensure_ascii=False))
    return 0 if not errores else 1


if __name__ == "__main__":
    sys.exit(main())
