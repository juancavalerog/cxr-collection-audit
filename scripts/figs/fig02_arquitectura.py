"""Figura 2: arquitectura de CSD-Net (docs/09_figuras.md; docs/03_arquitectura.md, Section 2). Borrador en español.

(a) Flujo de una imagen: vía de decisión (tronco, re-firma, Transformer, compuerta y cabeza), vía de adquisición
    (entrada sin tórax, codificador, firma, modulador y cabeza de fuente) y cabeza de segmentación.
(b) Las tres vistas de entrenamiento de cada imagen.

Dibujo vectorial sin datos. Uso: python scripts/figs/fig02_arquitectura.py [--out results/figs/fig02_arquitectura]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.viz import estilo as E  # noqa: E402

W, H = 190.0, 208.0
AZUL, AZUL_F = "#0072B2", "#DCEAF5"          # se usa en entrenamiento y en inferencia
NAR, NAR_F = "#D55E00", "#FBE7D6"            # solo en entrenamiento
GRIS_B, GRIS_F = "#666666", "#FFFFFF"        # datos
PERD_F = "#EFEFEF"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "figs" / "fig02_arquitectura"))
    a = ap.parse_args(argv)
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    E.aplicar()
    fig = E.figura(W, H)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    PT = E.PT_BASE

    def caja(x, y, w, h, titulo, lineas=(), borde=AZUL, fondo=AZUL_F, grosor=0.8, negrita=True, pt=PT):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", linewidth=grosor,
                                    edgecolor=borde, facecolor=fondo, zorder=2))
        n = 1 + len(lineas)
        paso = 3.3
        y0 = y + h / 2 - (n - 1) * paso / 2
        ax.text(x + w / 2, y0, titulo, ha="center", va="center", fontsize=pt, fontweight="bold" if negrita else "normal",
                zorder=3)
        for k, t in enumerate(lineas):
            ax.text(x + w / 2, y0 + (k + 1) * paso, t, ha="center", va="center", fontsize=pt - 0.5, zorder=3)
        return (x, y, w, h)

    def perdida(x, y, texto, ancho=None):
        w = ancho or (2.2 + 1.25 * len(texto.replace("$", "").replace("\\mathrm", "").replace("{", "").replace("}", "")))
        ax.add_patch(FancyBboxPatch((x, y - 2.3), w, 4.6, boxstyle="round,pad=0,rounding_size=2.2", linewidth=0.5,
                                    edgecolor=E.GRIS["tenue"], facecolor=PERD_F, zorder=2))
        ax.text(x + w / 2, y, texto, ha="center", va="center", fontsize=PT - 0.5, zorder=3)
        return w

    def flecha(p, q, color="#333333", discontinua=False, grosor=0.8):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, linewidth=grosor, color=color,
                                     linestyle=(0, (3, 2)) if discontinua else "solid", shrinkA=0, shrinkB=0, zorder=4))

    def linea(pts, color="#333333", discontinua=False, grosor=0.8):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=grosor, linestyle=(0, (3, 2)) if discontinua else "solid", zorder=4,
                solid_capstyle="round")

    def nota(x, y, texto, ha="left", pt=PT - 0.5, color=None, **kw):
        ax.text(x, y, texto, ha=ha, va="center", fontsize=pt, color=color or E.GRIS["texto"], zorder=5, **kw)

    # ============================================================================ (a) arquitectura
    ax.text(3.0, 5.0, "(a)", fontsize=E.PT_PANEL, fontweight="bold", va="center")
    # columnas: izquierda (segmentación), centro (decisión), derecha (adquisición)
    xi, wi = 6.0, 46.0
    xc, wc = 60.0, 64.0
    xd, wd = 136.0, 48.0
    cc, ci, cd = xc + wc / 2, xi + wi / 2, xd + wd / 2

    # ---- fila de datos
    caja(xi, 9, wi, 11, "Máscara pulmonar $m_i$", ("del profesor de segmentación",), GRIS_B, GRIS_F, 0.6, False)
    caja(xc, 9, wc, 11, "Radiografía preprocesada $x_i$", ("224 × 224, preprocesado común",), GRIS_B, GRIS_F, 0.6, False)
    caja(xd, 9, wd, 11, "Donante $x_j$", ("otra imagen del lote",), GRIS_B, GRIS_F, 0.6, False)

    # ---- vía de decisión (centro)
    caja(xc, 28, wc, 14, "Tronco convolucional", ("ConvNeXt-Tiny, etapas 1 a 3", "96, 192 y 384 canales"))
    caja(xc, 56, wc, 17, "Capa de re-firma", ("lleva la media y la escala de cada canal", "de la firma propia a la del donante",
                                              "sin parámetros propios"), grosor=1.4)
    caja(xc, 83, wc, 11, "Transformer", ("4 bloques, 384 dimensiones, 196 tokens",))
    caja(xc, 104, wc, 11, "Compuerta anatómica", ("promedio de tokens con peso pulmonar",))
    caja(xc, 125, wc, 11, "Cabeza de decisión", ("128 unidades",))
    flecha((cc, 20), (cc, 28))
    flecha((cc, 42), (cc, 56))
    nota(cc + 2, 49, "$F_i$  (384 × 14 × 14)")
    flecha((cc, 73), (cc, 83))
    nota(cc + 2, 78, r"$\tilde{F}_{i \leftarrow j}$")
    flecha((cc, 94), (cc, 104))
    flecha((cc, 115), (cc, 125))
    nota(cc + 2, 120, "$z$  (384)")
    flecha((cc, 136), (cc, 143))
    ax.text(cc, 146, "Probabilidad de COVID-19", ha="center", va="center", fontsize=PT, fontweight="bold")
    perdida(xc - 15.5, 127.8, r"$L_{cls}$", 12.5)
    perdida(xc - 15.5, 133.2, r"$L_{cons}$", 12.5)
    linea([(xc - 3, 127.8), (xc - 1.5, 127.8), (xc - 1.5, 133.2), (xc - 3, 133.2)], color=E.GRIS["tenue"], grosor=0.5)
    linea([(xc - 1.5, 130.5), (xc, 130.5)], color=E.GRIS["tenue"], grosor=0.5)

    # ---- cabeza de segmentación (izquierda)
    caja(xi, 56, wi, 17, "Cabeza de segmentación", ("pirámide ligera sobre", "las etapas 1 a 3"))
    linea([(xc, 35), (ci + 8, 35), (ci + 8, 50)])
    flecha((ci + 8, 50), (ci + 8, 56))
    linea([(ci - 8, 20), (ci - 8, 50)], discontinua=True, color=NAR)
    flecha((ci - 8, 50), (ci - 8, 56), color=NAR, discontinua=True)
    perdida(ci - 8 - 14.5, 38, r"$L_{seg}$", 11.5)
    linea([(ci - 8 - 3, 38), (ci - 8, 38)], color=E.GRIS["tenue"], grosor=0.5)
    linea([(ci, 73), (ci, 109.5), (xc - 6, 109.5)])
    flecha((xc - 6, 109.5), (xc, 109.5))
    nota(ci + 2, 91, "peso de cada token")
    nota(ci + 2, 94.6, "(cero fuera del pulmón)")

    # ---- vía de adquisición (derecha)
    caja(xd, 28, wd, 14, "Entrada de adquisición", ("detalle de Haar y periferia,", "sin la caja torácica"), NAR, NAR_F)
    caja(xd, 50, wd, 11, "Codificador de adquisición", ("red convolucional de 4 etapas",), NAR, NAR_F)
    caja(xd, 73, wd, 14, "Modulador", ("predice la media y la escala", "de canal desde la firma"), NAR, NAR_F)
    caja(xd, 104, wd, 11, "Cabeza de fuente", ("lineal, 8 fuentes",), NAR, NAR_F)
    flecha((cd, 20), (cd, 28), color=NAR)
    linea([(xc + wc, 14.5), (xc + wc + 6, 14.5), (xc + wc + 6, 24), (cd - 12, 24)], color=NAR)
    flecha((cd - 12, 24), (cd - 12, 28), color=NAR)
    flecha((cd, 42), (cd, 50), color=NAR)
    flecha((cd, 61), (cd, 73), color=NAR)
    nota(cd + 2, 67, "firmas $a_i$ y $a_j$  (64)")
    linea([(xd, 80), (xc + wc + 6, 80), (xc + wc + 6, 64.5)], color=NAR)
    flecha((xc + wc + 6, 64.5), (xc + wc, 64.5), color=NAR)
    linea([(xd + wd, 55.5), (xd + wd + 3, 55.5), (xd + wd + 3, 109.5)], color=NAR)
    flecha((xd + wd + 3, 109.5), (xd + wd, 109.5), color=NAR)
    perdida(xd + 2, 92, r"$L_{sig}$", 11.5)
    linea([(xd + 7.75, 87), (xd + 7.75, 89.7)], color=E.GRIS["tenue"], grosor=0.5)
    perdida(xd + 2, 120, r"$L_{src}$", 11.5)
    linea([(xd + 7.75, 115), (xd + 7.75, 117.7)], color=E.GRIS["tenue"], grosor=0.5)
    perdida(cd - 3 - 11.5, 67, r"$L_{ind}$", 11.5)          # actúa sobre la firma, no sobre la cabeza de fuente
    linea([(cd - 3, 67), (cd, 67)], color=E.GRIS["tenue"], grosor=0.5)
    nota(xd, 128.0, "$L_{src}$: la firma reconoce la fuente")
    nota(xd, 131.8, "$L_{ind}$: la firma no distingue la clase")
    nota(xd, 135.6, "$L_{sig}$: el modulador explica la firma")

    # ---- leyenda de (a)
    yl = 153.0
    for k, (bo, fo, t) in enumerate(((AZUL, AZUL_F, "Entrenamiento e inferencia"), (NAR, NAR_F, "Solo en entrenamiento"),
                                     (GRIS_B, GRIS_F, "Datos"), (E.GRIS["tenue"], PERD_F, "Pérdida"))):
        x0 = 20.0 + k * 42.0
        ax.add_patch(FancyBboxPatch((x0, yl - 1.8), 6, 3.6, boxstyle="round,pad=0,rounding_size=0.9", linewidth=0.7,
                                    edgecolor=bo, facecolor=fo, zorder=2))
        nota(x0 + 7.5, yl, t)

    # ============================================================================ (b) tres vistas
    yb = 161.0
    ax.text(3.0, yb, "(b)", fontsize=E.PT_PANEL, fontweight="bold", va="center")
    x0, wl, wv = 6.0, 34.0, 48.0
    cols = ["Vista real", "Vista dentro de la fuente", "Vista cruzada"]
    filas = [("Contenido", ["imagen $i$", "imagen $i$", "imagen $i$"]),
             ("Firma que se aplica", ["la propia, $a_i$", "la de otra imagen\nde la misma fuente", "la de una imagen de otra\nfuente, elegida por el plan"]),
             ("Fuente efectiva", ["la de la imagen $i$", "la de la imagen $i$", "la del donante"]),
             ("Etiqueta", ["$y_i$", "$y_i$", "$y_i$"]),
             ("Pérdidas", ["$L_{cls}$, peso $1 - \\rho$", "$L_{cls}$, peso $\\rho$, y $L_{cons}$", "$L_{cls}$, peso $\\lambda$, y $L_{cons}$"])]
    alt_cab, alt = 6.5, [5.5, 8.5, 5.5, 5.5, 5.5]
    y = yb + 4.0
    for k, c in enumerate(cols):
        xk = x0 + wl + k * wv
        fo, bo = (AZUL_F, AZUL) if k == 0 else (NAR_F, NAR)
        ax.add_patch(plt.Rectangle((xk + 0.6, y), wv - 1.2, alt_cab, facecolor=fo, edgecolor=bo, linewidth=0.7, zorder=2))
        ax.text(xk + wv / 2, y + alt_cab / 2, c, ha="center", va="center", fontsize=PT, fontweight="bold", zorder=3)
    y += alt_cab
    for (nombre, celdas), h in zip(filas, alt):
        ax.plot([x0, x0 + wl + 3 * wv], [y, y], color=E.GRIS["rejilla"], linewidth=0.5, zorder=1)
        ax.text(x0 + 1.0, y + h / 2, nombre, ha="left", va="center", fontsize=PT, fontweight="bold")
        for k, t in enumerate(celdas):
            ax.text(x0 + wl + k * wv + wv / 2, y + h / 2, t, ha="center", va="center", fontsize=PT - 0.5, linespacing=1.15)
        y += h
    ax.plot([x0, x0 + wl + 3 * wv], [y, y], color=E.GRIS["rejilla"], linewidth=0.5, zorder=1)

    out = E.guardar(fig, a.out)
    for k, v in out.items():
        print(k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
