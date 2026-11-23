#!/usr/bin/env python3
"""Descarga recursiva de un arbol publicado como paginas index.html estaticas.

El servidor de datos de NLM (data.lhncbc.nlm.nih.gov) publica cada carpeta como <carpeta>/index.html y
responde con error a la URL de la carpeta sin index.html, de modo que "wget -r <carpeta>/" falla en la
primera peticion. Este script recorre los index.html, descarga los ficheros con reanudacion y reintentos,
y no depende del marcado exacto de la pagina: solo de los atributos href.

Uso: nlm_crawl.py <url_index.html> <destino> [--root-url URL] [--timeout 60] [--tries 5] [--pause 0.1]
  --root-url  prefijo que se quita de cada URL para formar la ruta local (por defecto, la carpeta del
              index inicial).
Codigo de salida: 0 si todo se descargo; 2 si fallo alguna pagina o fichero; 3 si no se encontro ninguno.
"""
import argparse
import http.client
import os
import re
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import unquote, urldefrag, urljoin

HREF = re.compile(r"""href\s*=\s*(?:"([^"]+)"|'([^']+)')""", re.I)
UA = "Wget/1.21.2 (paper_new; descarga academica)"
SKIP_NAMES = {"index.html", "index.json"}
FATAL_HTTP = {400, 401, 403, 404, 410}


def log(msg):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg, file=sys.stderr, flush=True)


def open_url(url, timeout, headers=None):
    h = {"User-Agent": UA}
    h.update(headers or {})
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout)


def retry(fn, tries, what):
    delay = 2.0
    for k in range(1, tries + 1):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if e.code in FATAL_HTTP or k == tries:
                raise
            err = e
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            if k == tries:
                raise
            err = e
        log(f"aviso: {what}: {err}; reintento {k} de {tries - 1} en {delay:.0f} s")
        time.sleep(delay)
        delay = min(delay * 2, 60.0)


def links(page_url, html):
    out = []
    for m in HREF.finditer(html):
        href = (m.group(1) or m.group(2)).strip()
        if not href or href.startswith(("#", "mailto:", "javascript:")):
            continue
        out.append(urldefrag(urljoin(page_url, href))[0].split("?", 1)[0])
    return out


def crawl(start, timeout, tries):
    """Devuelve (ficheros, paginas_fallidas). Solo desciende: nunca sale de la carpeta de cada pagina."""
    pages, files, seen, failed = [start], {}, set(), []
    while pages:
        page = pages.pop(0)
        if page in seen:
            continue
        seen.add(page)
        base_dir = page.rsplit("/", 1)[0] + "/"
        try:
            html = retry(lambda: open_url(page, timeout).read().decode("utf-8", "replace"), tries, page)
        except Exception as e:  # noqa: BLE001
            log(f"ERROR: no se pudo leer {page}: {e}")
            failed.append(page)
            continue
        for u in links(page, html):
            if not u.startswith(base_dir) or u in (base_dir, page):
                continue
            name = u.rsplit("/", 1)[1]
            if u.endswith("/"):
                pages.append(u + "index.html")
            elif name == "index.html":
                pages.append(u)
            elif name not in SKIP_NAMES:
                files.setdefault(u, None)
    return list(files), failed


def download(url, path, timeout, tries):
    """Descarga a <path>.part y renombra al terminar. Devuelve (bytes, nuevo)."""
    if os.path.exists(path):
        return os.path.getsize(path), False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"

    def attempt():
        have = os.path.getsize(part) if os.path.exists(part) else 0
        try:
            r = open_url(url, timeout, {"Range": f"bytes={have}-"} if have else None)
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:  # parcial incoherente con el servidor: se empieza de cero
                open(part, "wb").close()
                raise OSError("rango no satisfacible; se reinicia el fichero") from e
            raise
        with r:
            resumed = bool(have) and getattr(r, "status", 200) == 206
            if not resumed:
                have = 0
            total = r.headers.get("Content-Length")
            expected = have + int(total) if total and total.isdigit() else None
            with open(part, "ab" if resumed else "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
        size = os.path.getsize(part)
        if expected is not None and size != expected:
            raise OSError(f"tamano {size} distinto del anunciado {expected}")
        os.replace(part, path)
        return size

    return retry(attempt, tries, url), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start")
    ap.add_argument("dest")
    ap.add_argument("--root-url", default=None)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--tries", type=int, default=5)
    ap.add_argument("--pause", type=float, default=0.1)
    a = ap.parse_args()
    if not a.start.endswith("index.html"):
        sys.exit("la URL inicial debe terminar en index.html")
    root = a.root_url or (a.start.rsplit("/", 1)[0] + "/")
    if not root.endswith("/"):
        root += "/"
    if not a.start.startswith(root):
        sys.exit("--root-url debe ser un prefijo de la URL inicial")

    files, failed_pages = crawl(a.start, a.timeout, a.tries)
    log(f"listado: {len(files)} ficheros; paginas ilegibles: {len(failed_pages)}")
    if not files:
        log("ERROR: el listado no contiene ningun fichero")
        return 3
    new = nbytes = 0
    failed = []
    for i, u in enumerate(sorted(files), 1):
        rel = os.path.normpath(unquote(u[len(root):]))
        if rel.startswith("..") or os.path.isabs(rel):
            log(f"ERROR: ruta fuera del destino, se omite: {u}")
            failed.append(u)
            continue
        try:
            size, was_new = download(u, os.path.join(a.dest, rel), a.timeout, a.tries)
        except Exception as e:  # noqa: BLE001
            log(f"ERROR: {u}: {e}")
            failed.append(u)
            continue
        nbytes += size
        if was_new:
            new += 1
            time.sleep(a.pause)
        if i % 100 == 0 or i == len(files):
            log(f"progreso: {i} de {len(files)} ficheros, {nbytes / 1e6:.1f} MB")
    log(f"resumen: {len(files)} ficheros, {new} nuevos, {nbytes} bytes, {len(failed)} fallidos, "
        f"{len(failed_pages)} paginas ilegibles")
    return 2 if (failed or failed_pages) else 0


if __name__ == "__main__":
    sys.exit(main())
