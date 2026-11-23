#!/usr/bin/env python3
"""Servidor HTTP simulado del arbol de datos de NLM, para pruebas sin red.

Imita lo observado en el servidor real el 12-11-2026:
  - cada carpeta se publica como <carpeta>/index.html, con enlaces a <subcarpeta>/index.html;
  - la URL de la carpeta (terminada en /) responde con error (aqui, 403).
Ademas: admite Range; el primer GET de un fichero cuyo nombre contiene "flaky" devuelve 503; el arbol de
Montgomery usa enlaces absolutos y el de Shenzhen relativos; cada pagina incluye enlaces que el descargador
debe ignorar (carpeta superior, hoja de estilo, sitio externo, index.json). Registra las peticiones en
<log>. Uso: nlm_fake_server.py <dir_raiz> <puerto> <log>
"""
import html
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import quote, unquote

ROOT, PORT, LOG = os.path.abspath(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
SEEN_FLAKY = set()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _log(self, code):
        with open(LOG, "a") as f:
            f.write(f"{code} {self.path} range={self.headers.get('Range', '-')}\n")

    def _send(self, code, body=b"", ctype="text/plain", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)
        self._log(code)

    def do_GET(self):
        url_path = self.path.split("?", 1)[0]
        rel = unquote(url_path).lstrip("/")
        fs = os.path.normpath(os.path.join(ROOT, rel))
        if not fs.startswith(ROOT):
            return self._send(403, b"fuera de la raiz")
        if url_path.endswith("/"):
            return self._send(403, b"AccessDenied")
        if os.path.basename(fs) == "index.html" and os.path.isdir(os.path.dirname(fs)):
            return self._send(200, self._listing(os.path.dirname(fs), url_path).encode(), "text/html; charset=utf-8")
        if not os.path.isfile(fs):
            return self._send(404, b"NoSuchKey")
        if "flaky" in os.path.basename(fs) and fs not in SEEN_FLAKY:
            SEEN_FLAKY.add(fs)
            return self._send(503, b"SlowDown")
        data = open(fs, "rb").read()
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            start = int(rng[6:].split("-")[0])
            if start >= len(data):
                return self._send(416, b"", extra={"Content-Range": f"bytes */{len(data)}"})
            return self._send(206, data[start:], "application/octet-stream",
                              {"Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}"})
        return self._send(200, data, "application/octet-stream")

    def _listing(self, d, url_path):
        absolute = "Montgomery" in url_path
        base_url = url_path.rsplit("/", 1)[0] + "/"
        prefix = f"http://127.0.0.1:{PORT}{base_url}" if absolute else ""
        rows = []
        for name in sorted(os.listdir(d)):
            full = os.path.join(d, name)
            if os.path.isdir(full):
                rows.append(f'<tr><td><a href="{prefix}{quote(name)}/index.html">{html.escape(name)}/</a></td><td>-</td><td>-</td></tr>')
            else:
                rows.append(f'<tr><td><a href="{prefix}{quote(name)}">{html.escape(name)}</a></td>'
                            f'<td>{os.path.getsize(full)}</td><td>2024-08-28 21:54:07 +0000 UTC</td></tr>')
        return ("<html><head><title>Index of " + html.escape(base_url) + "</title>"
                '<link rel="stylesheet" href="/static/style.css"></head><body>'
                '<a href="../index.html">..</a> <a href="https://www.nlm.nih.gov/">NLM</a> '
                '<a href="index.json">json</a> <a href="#top">arriba</a>'
                "<table><tr><th>File Name</th><th>File Size</th><th>Date</th></tr>" + "".join(rows) + "</table></body></html>")


HTTPServer(("127.0.0.1", PORT), H).serve_forever()
