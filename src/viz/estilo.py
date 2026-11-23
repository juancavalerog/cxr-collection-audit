"""Estilo común de las figuras del artículo (docs/09_figuras.md, Section 1).

Reglas que aplica: anchuras de 90, 140 o 190 mm, formato vertical con altura máxima de 240 mm, tipografía sin
remates de 7 a 8 pt, líneas de 0.5 pt como mínimo, paleta apta para daltonismo con color y marcador fijos por
método, y entrega en PDF vectorial más PNG y TIFF a 600 ppp (el mínimo que pide el autor es 300). Cada figura
guarda al lado un CSV con los datos dibujados.

Uso:
    from src.viz import estilo as E
    E.aplicar()
    fig = E.figura(90, 150)
    ...
    E.guardar(fig, "results/figs/fig01_problema", datos=df)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MM = 1.0 / 25.4
ANCHOS_MM = (90.0, 140.0, 190.0)          # una columna, columna y media, página completa (Elsevier)
ALTO_MAX_MM = 240.0
PPP = 600                                  # mapas de bits; el mínimo pedido es 300
PPP_MIN = 300
FUENTES_TIPO = ["Arial", "Helvetica", "Liberation Sans", "Arimo", "TeX Gyre Heros", "Nimbus Sans", "DejaVu Sans"]
PT_BASE, PT_MIN, PT_PANEL = 7.5, 6.0, 9.0
GROSOR_MIN = 0.5

# Paleta apta para daltonismo (Okabe e Ito, completada con la de Paul Tol). CSD-Net siempre en negro.
OKABE_ITO = {"negro": "#000000", "naranja": "#E69F00", "celeste": "#56B4E9", "verde": "#009E73",
             "amarillo": "#F0E442", "azul": "#0072B2", "bermellon": "#D55E00", "purpura": "#CC79A7"}
COLOR_METODO = {"csdnet": "#000000", "control_fuente": "#0072B2", "erm": "#D55E00", "groupdro": "#E69F00",
                "dann": "#56B4E9", "coral": "#009E73", "mixstyle": "#CC79A7", "fourier": "#882255",
                "lisa": "#44AA99", "dfa_fuente": "#999933", "dfr": "#AA4499", "erm_mascara": "#117733",
                "erm_cuotas": "#DDCC77", "cdann": "#332288", "irm": "#88CCEE", "lai2026": "#777777",
                "rad_dino_probe": "#CC6677"}
MARCA_METODO = {"csdnet": "o", "control_fuente": "s", "erm": "^", "groupdro": "v", "dann": "D", "coral": "P",
                "mixstyle": "X", "fourier": "<", "lisa": ">", "dfa_fuente": "h", "dfr": "p", "erm_mascara": "*",
                "erm_cuotas": "d", "cdann": "H", "irm": "8", "lai2026": "1", "rad_dino_probe": "2"}
COLOR_CLASE = {0: "#0072B2", 1: "#D55E00"}            # negativa, positiva
GRIS = {"texto": "#222222", "eje": "#444444", "rejilla": "#D9D9D9", "vacio": "#F2F2F2", "tenue": "#8C8C8C"}


def aplicar():
    """Fija los parámetros de matplotlib del proyecto. Llamar una vez antes de crear figuras."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FUENTES_TIPO, "font.size": PT_BASE,
        "axes.labelsize": PT_BASE, "axes.titlesize": PT_BASE, "xtick.labelsize": PT_BASE - 0.5,
        "ytick.labelsize": PT_BASE - 0.5, "legend.fontsize": PT_BASE - 0.5, "legend.frameon": False,
        "axes.linewidth": 0.6, "axes.edgecolor": GRIS["eje"], "axes.labelcolor": GRIS["texto"],
        "text.color": GRIS["texto"], "xtick.color": GRIS["eje"], "ytick.color": GRIS["eje"],
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.direction": "out", "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.0, "lines.markersize": 4.0, "patch.linewidth": 0.6, "grid.linewidth": GROSOR_MIN,
        "grid.color": GRIS["rejilla"], "axes.axisbelow": True, "figure.dpi": 150, "savefig.dpi": PPP,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "mathtext.fontset": "dejavusans",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def figura(ancho_mm: float, alto_mm: float, **kw):
    """Figura al tamaño final de impresión. Avisa con error si no cumple las reglas de tamaño o no es vertical."""
    if not any(abs(ancho_mm - a) < 0.5 for a in ANCHOS_MM):
        raise ValueError("anchura %.1f mm no admitida (admitidas: %s)" % (ancho_mm, ANCHOS_MM))
    if alto_mm > ALTO_MAX_MM + 1e-6:
        raise ValueError("altura %.1f mm mayor que el máximo de %.0f mm" % (alto_mm, ALTO_MAX_MM))
    if alto_mm < ancho_mm:
        raise ValueError("la figura debe ser vertical (alto %.1f < ancho %.1f mm)" % (alto_mm, ancho_mm))
    return plt.figure(figsize=(ancho_mm * MM, alto_mm * MM), **kw)


def rotulo(ax, letra: str, dx_mm: float = -9.0, dy_mm: float = 2.0):
    """Rótulo de panel '(a)' en negrita de 9 pt, fuera de la esquina superior izquierda del eje."""
    from matplotlib.transforms import ScaledTranslation
    tr = ax.transAxes + ScaledTranslation(dx_mm * MM, dy_mm * MM, ax.figure.dpi_scale_trans)
    return ax.text(0.0, 1.0, "(%s)" % letra, transform=tr, fontsize=PT_PANEL, fontweight="bold", va="bottom",
                   ha="left")


def comprobar(fig):
    """Devuelve la lista de textos por debajo del tamaño mínimo y de líneas por debajo del grosor mínimo."""
    fallos = []
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text().strip() and t.get_visible() and t.get_fontsize() < PT_MIN - 1e-6:
            fallos.append("texto de %.1f pt: %r" % (t.get_fontsize(), t.get_text()[:30]))
    for ln in fig.findobj(matplotlib.lines.Line2D):
        if ln.get_visible() and ln.get_linestyle() not in ("None", "", " ") and 0 < ln.get_linewidth() < GROSOR_MIN - 1e-6:
            fallos.append("línea de %.2f pt" % ln.get_linewidth())
    return fallos


def guardar(fig, base, datos=None, formatos=("pdf", "png", "tiff"), ppp: int = PPP, estricto: bool = True):
    """Guarda la figura en PDF vectorial, PNG y TIFF (LZW) a `ppp`, y los datos dibujados en CSV.

    `base` es la ruta sin extensión. Devuelve el diccionario formato -> ruta. Con `estricto`, falla si algún texto
    o línea incumple los mínimos o si `ppp` es menor que 300."""
    if ppp < PPP_MIN:
        raise ValueError("resolución %d ppp menor que el mínimo de %d" % (ppp, PPP_MIN))
    fallos = comprobar(fig)
    if fallos and estricto:
        raise ValueError("la figura incumple el estilo: " + "; ".join(fallos[:8]))
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    out = {}
    for f in formatos:
        p = base.with_suffix("." + f)
        kw = {"dpi": ppp}
        if f == "tiff":
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(p, **kw)
        out[f] = str(p)
    if datos is not None:
        p = base.with_suffix(".csv")
        datos.to_csv(p, index=False)
        out["csv"] = str(p)
    return out
