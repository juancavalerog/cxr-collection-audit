"""Texto en inglés común a las figuras de la versión inglesa (manuscript/en/GLOSARIO.md).

Solo texto: los datos, tamaños, colores, marcas y posiciones son los de los guiones españoles de scripts/figs/.
"""
from __future__ import annotations

NOMBRE = {"erm": "ERM", "erm_cuotas": "Quota ERM", "erm_mascara": "Masked ERM", "control_fuente": "Source control",
          "groupdro": "GroupDRO", "dann": "DANN", "coral": "CORAL", "mixstyle": "MixStyle",
          "fourier": "Fourier amplitude mixing", "lisa": "LISA", "dfa_fuente": "Source-aware DFA", "dfr": "DFR",
          "D_IN": "Primary reference", "D_ERM": "Secondary reference"}
RHO_ET = {0.5: "0.5", 0.6: "0.6", 0.7: "0.7", 0.8: "0.8", 0.9: "0.9", 0.95: "0.95", 0.99: "0.99", 1.0: "1.0"}
EJE_RHO = "Confounding level ρ (levels evenly spaced)"   # niveles de ρ equiespaciados en el eje
MENOS = "−"


def menos(s: str) -> str:
    """Signo menos tipográfico (U+2212) en las cifras negativas de un texto."""
    return s.replace("-", MENOS)


def miles(v: int) -> str:
    """Miles según el glosario: sin separador hasta cuatro cifras y con coma desde cinco (8079, 37,123)."""
    v = int(v)
    return str(v) if abs(v) < 10000 else "{:,}".format(v)

# Variante en dos líneas para los ejes estrechos (margen izquierdo de 0.30 a 0.31 de 90 mm), donde
# "Fourier amplitude mixing" en una línea queda cortado por el borde de la figura.
NOMBRE2 = dict(NOMBRE, fourier="Fourier amplitude\nmixing")
