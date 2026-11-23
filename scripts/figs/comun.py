"""Utilidades comunes de las figuras: lectura de tablas, formato de cifras y guardado.

leer(rel): lee results/<rel>, o results/correccion_ec6/<rel> si existe (intervalos con el factor de calibración
    corregido del término de semillas; las estimaciones puntuales no cambian). Registra qué versión se leyó.
num(v, nd, signo): cifra con signo menos tipográfico y sin cero negativo ("0.000", nunca con signo).
guardar(fig, base, datos): normaliza tipografía y signos, comprueba el estilo, la ausencia de ceros negativos y de
    solapes de texto, recorta los márgenes (bbox ajustado con 0.5 mm, que incluye todos los textos) y escribe PDF vectorial, PNG y TIFF a 600 ppp
    (TIFF en RGB sin canal alfa, LZW) con metadatos neutros, más el CSV de datos.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.viz import estilo as E  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text  # noqa: E402

RES = ROOT / "results"
CORR = RES / "correccion_ec6"
MENOS = "−"
PPP = 600
PAD_MM = 0.5
ANCHO_MAX_MM = 190.0
TAMANOS = (6.0, 6.5, 7.0, 7.5, 9.0)          # tamaños de letra admitidos (pt)
EJE_RHO_ES = "Nivel de confusión ρ (niveles equiespaciados)"
EJE_RHO_EN = "Confounding level ρ (levels evenly spaced)"
LEIDOS: list = []                             # (tabla pedida, versión leída)
INFORME: list = []                            # una entrada por figura guardada


def ruta(rel) -> Path:
    """Ruta de la tabla `rel` (relativa a results/), con preferencia por la versión corregida."""
    p = Path(rel)
    if p.is_absolute():
        p = p.relative_to(RES)
    q = CORR / p
    return q if q.exists() else RES / p


def leer(rel, **kw) -> pd.DataFrame:
    p = ruta(rel)
    LEIDOS.append((str(Path(rel)), "corregida" if CORR in p.parents else "original"))
    return pd.read_csv(p, **kw)


_CERO_NEG = re.compile(r"[-−+](0(?:\.0+)?)(?![\d.])")
_GUION_NUM = re.compile(r"(?<![\w−])-(?=[\d.])")


def menos(s: str) -> str:
    """Signo menos tipográfico en las cifras negativas y ceros sin signo."""
    return _CERO_NEG.sub(r"\1", _GUION_NUM.sub(MENOS, s))


def num(v, nd: int = 3, signo: bool = False) -> str:
    s = ("%+.*f" if signo else "%.*f") % (nd, v)
    return menos(s)


def p_txt(p: float) -> str:
    """Valor p de Holm: cuatro decimales por debajo de 0.01 (como la Tabla S17), dos en el resto."""
    return "%.4f" % p if p < 0.01 else "%.2f" % p


def hueco_ms(marca: str, ms: float = 5.0) -> float:
    """Tamaño algo mayor para los marcadores huecos de trazo fino (P, X, *), que si no parecen rellenos."""
    return ms + 1.0 if marca in ("P", "X", "*") else ms


def ejes_mm(fig, izq, abajo, ancho, alto):
    """Ejes con posición en mm desde la esquina inferior izquierda de la figura."""
    W, H = fig.get_size_inches() / E.MM
    return fig.add_axes([izq / W, abajo / H, ancho / W, alto / H])


def nota_bajo(fig, ejes, texto, sep_mm=2.0, x=0.04, **kw):
    """Nota de la figura alineada arriba, sep_mm por debajo de lo más bajo de `ejes` (rótulos incluidos)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    y0 = min(ax.get_tightbbox(r).y0 for ax in ejes) / fig.bbox.height
    y = y0 - sep_mm * E.MM * fig.dpi / fig.bbox.height
    kw = dict(dict(fontsize=6, color=E.GRIS["texto"], va="top", ha="left", linespacing=1.35), **kw)
    return fig.text(x, y, texto, **kw)


def mapa(ax, M, **kw):
    """Mapa de calor vectorial (pcolormesh) con la misma geometría que imshow: celdas centradas en enteros."""
    import numpy as np
    M = np.ma.masked_invalid(np.asarray(M, dtype=float))
    f, c = M.shape
    q = ax.pcolormesh(np.arange(c + 1) - 0.5, np.arange(f + 1) - 0.5, M, shading="flat", edgecolors="none",
                      linewidth=0, antialiased=False, **kw)
    ax.set_xlim(-0.5, c - 0.5); ax.set_ylim(f - 0.5, -0.5)
    return q


def _textos(fig):
    return [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]


def _normalizar(fig):
    for t in _textos(fig):
        s = t.get_text()
        n = menos(s)
        if n != s:
            t.set_text(n)
        fs = t.get_fontsize()
        k = min(TAMANOS, key=lambda a: (abs(a - fs), a))
        if abs(k - fs) > 1e-6:
            t.set_fontsize(k)


def _cajas(fig):
    r = fig.canvas.get_renderer()
    ocultos = set()                                    # rótulos de marcas fuera de los límites: no se dibujan
    for ax in fig.axes:
        for eje in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(eje.get_view_interval())
            for tk in eje.get_major_ticks() + eje.get_minor_ticks():
                if not (lo - 1e-9 * max(1.0, abs(hi)) <= tk.get_loc() <= hi + 1e-9 * max(1.0, abs(hi))):
                    ocultos.update((id(tk.label1), id(tk.label2)))
    out = []
    for t in _textos(fig):
        if id(t) in ocultos:
            continue
        bb = t.get_window_extent(r)
        if bb.width > 0 and bb.height > 0:
            out.append((t, bb))
    return out


def solapes(fig, tol_px: float = 1.0):
    """Pares de textos cuyas cajas se cortan en más de tol_px en los dos ejes."""
    c = _cajas(fig)
    out = []
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            (a, A), (b, B) = c[i], c[j]
            if min(A.x1, B.x1) - max(A.x0, B.x0) > tol_px and min(A.y1, B.y1) - max(A.y0, B.y0) > tol_px:
                out.append((a.get_text().replace("\n", " ")[:40], b.get_text().replace("\n", " ")[:40]))
    return out


def fuera_del_lienzo(fig):
    """Textos que el borde de la figura recortaría."""
    W, H = fig.bbox.width, fig.bbox.height
    return [t.get_text().replace("\n", " ")[:40] for t, bb in _cajas(fig)
            if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5]


def caja_total(fig, r):
    """Caja ajustada (en pulgadas) que incluye todos los textos visibles. Figure.get_tightbbox reduce al ancho del eje
    los rótulos de eje más anchos que él, de modo que el recorte los cortaría; aquí se une la caja de cada texto."""
    from matplotlib.transforms import Bbox
    inv = fig.dpi_scale_trans.inverted()
    cajas = [fig.get_tightbbox(r)] + [bb.transformed(inv) for _, bb in _cajas(fig)]
    return Bbox.union(cajas)


def guardar(fig, base, datos=None):
    """Guarda PDF, PNG y TIFF (RGB, LZW, 600 ppp) con bbox ajustado y metadatos neutros, y el CSV de datos."""
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):                                 # los rótulos de marcas se crean al dibujar
        fig.canvas.draw()
        _normalizar(fig)
    fallos = E.comprobar(fig)
    if fallos:
        raise ValueError("%s incumple el estilo: %s" % (base.name, "; ".join(fallos[:8])))
    neg = [s for s in (t.get_text() for t in _textos(fig)) if re.search(r"[-−]0\.0+(?!\d)", s)]
    if neg:
        raise ValueError("%s: cero negativo en %r" % (base.name, neg[:3]))
    sol, fuera = solapes(fig), fuera_del_lienzo(fig)
    r = fig.canvas.get_renderer()
    bb = caja_total(fig, r).padded(PAD_MM * E.MM)
    ancho_mm, alto_mm = bb.width / E.MM, bb.height / E.MM
    if ancho_mm > ANCHO_MAX_MM + 0.05:
        raise ValueError("%s: ancho %.1f mm mayor que %.0f mm" % (base.name, ancho_mm, ANCHO_MAX_MM))
    out = {}
    p = base.with_suffix(".pdf")
    fig.savefig(p, bbox_inches=bb, metadata={"Creator": None, "Producer": None, "Author": None, "Title": None,
                                             "CreationDate": None})
    out["pdf"] = str(p)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=PPP, bbox_inches=bb, metadata={"Software": None})
    p = base.with_suffix(".png")
    p.write_bytes(buf.getvalue())
    out["png"] = str(p)
    from PIL import Image
    im = Image.open(io.BytesIO(buf.getvalue())).convert("RGBA")
    fondo = Image.new("RGB", im.size, (255, 255, 255))
    fondo.paste(im, mask=im.split()[3])
    p = base.with_suffix(".tiff")
    fondo.save(p, format="TIFF", compression="tiff_lzw", dpi=(PPP, PPP))
    out["tiff"] = str(p)
    if datos is not None:
        p = base.with_suffix(".csv")
        datos.to_csv(p, index=False)
        out["csv"] = str(p)
    INFORME.append({"figura": base.name, "ancho_mm": round(ancho_mm, 1), "alto_mm": round(alto_mm, 1),
                    "solapes": sol, "fuera": fuera})
    if sol:
        print("AVISO solapes en %s: %s" % (base.name, sol))
    if fuera:
        print("AVISO texto fuera del lienzo en %s: %s" % (base.name, fuera))
    plt.close(fig)
    return out
