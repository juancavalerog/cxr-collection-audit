#!/usr/bin/env python3
"""Analiza el listado de ficheros de un dataset de Kaggle y cuenta imagenes por carpeta.

Acepta la salida cruda de `kaggle datasets files -v` (varias paginas concatenadas, con o sin cabecera y
lineas "Next Page Token") o un CSV ya limpio con columnas name,size,creationDate.
Uso: kaggle_list_parse.py <slug> <entrada> <csv_salida> <log>   (csv_salida puede ser '-' para no escribirlo)
"""
import collections
import csv
import io
import sys

EXT = (".png", ".jpg", ".jpeg", ".dcm", ".dicom", ".bmp", ".tif", ".tiff")


def parse(path):
    rows, seen = [], set()
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        low = line.lower()
        if not line or low.startswith(("name,", "warning", "dataset url", "next page token")):
            continue
        rec = next(csv.reader(io.StringIO(line)))
        if len(rec) >= 2 and rec[0] not in seen:
            seen.add(rec[0])
            rows.append((rec + ["", ""])[:3])
    return rows


def main():
    slug, src, out, log = sys.argv[1:5]
    rows = parse(src)
    if out != "-":
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "size", "creationDate"])
            w.writerows(rows)
    cnt, tot = collections.Counter(), collections.Counter()
    for name, _size, _date in rows:
        folder = "/".join(name.split("/")[:-1]) or "."
        tot[folder] += 1
        if name.lower().endswith(EXT):
            cnt[folder] += 1
    lines = [f"\n## {slug}", f"ficheros listados: {len(rows)}; imagenes: {sum(cnt.values())}"]
    lines += [f"  {f}: {cnt[f]} imagenes de {tot[f]} ficheros" for f in sorted(tot)]
    with open(log, "a") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines).strip())
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
