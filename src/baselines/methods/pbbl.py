"""PBBL [luo2022pseudo]: esqueleto registrado, pendiente.

Plan de la reimplementación (no hecha): etapa 1, modelo sesgado entrenado con entropía cruzada generalizada (GCE,
q = 0.7) sobre la clase para que capture el atajo; sus predicciones dan una pseudoetiqueta de sesgo por imagen
(alineado o en conflicto); etapa 2, entrenamiento equilibrado sobre los grupos (clase, pseudo-sesgo). Con la
fuente conocida, la variante natural sustituye la pseudoetiqueta por la fuente, lo que la reduce a un
reequilibrado por celdas y queda indefinida en las celdas vacías como GroupDRO. La función de pérdida GCE ya está
en src/baselines/common.py (gce_binary). Al ejecutarlo, el entrenador falla con NotImplementedError.
"""
from __future__ import annotations

from src.baselines.registry import Method, register


@register
class PBBL(Method):
    name = "pbbl"
    title = "PBBL (sesgo pseudoetiquetado con GCE)"
    ref = "luo2022pseudo"
    labels = ("clase",)
    status = "pendiente"
    empty_cells = "con la fuente como pseudo-sesgo, indefinido en las celdas vacías (como GroupDRO)"
    defaults = {"gce_q": 0.7}
    search = {"gce_q": ("choice", [0.5, 0.7, 0.9])}
    hp_source = "artículo original (GCE q 0.7, por verificar)"
    cost = 2.0

    def build(self, tr, net):
        raise NotImplementedError("PBBL está registrado como esqueleto y pendiente de implementar "
                                  "(docs/06_lineas_base.md, Section 5)")
