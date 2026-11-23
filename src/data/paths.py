"""Resolución de rutas del proyecto y de los datos (VM de trabajo, contenedor del servidor y anfitrión).

- Raíz del proyecto: dos niveles por encima de este fichero (paper_new/).
- Raíz de datos (carpeta notebooks/covid): variable PAPER_NEW_DATA_ROOT o, en este orden, /data/covid
  (contenedor), $HOME/mnt/notebooks/covid (VM de trabajo) y /home/administrador/notebooks/covid (anfitrión).
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CANDIDATES = ("/data/covid", os.path.expanduser("~/mnt/notebooks/covid"), "/home/administrador/notebooks/covid")


def data_root() -> Path:
    env = os.environ.get("PAPER_NEW_DATA_ROOT", "").strip()
    if env:
        return Path(env)
    for c in _CANDIDATES:
        if os.path.isdir(os.path.join(c, "data", "dataset")):
            return Path(c)
    return Path(_CANDIDATES[0])


def project_path(*parts) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def resolve_data(rel: str | os.PathLike) -> Path:
    """Ruta absoluta de un fichero de datos dado relativo a la raíz de datos (p. ej. data/dataset/x.png)."""
    p = Path(rel)
    return p if p.is_absolute() else data_root() / p


def saved_root() -> Path:
    """Carpeta SAVED del artículo 1 (modelos .keras y probs.npz)."""
    return data_root() / "article1" / "saved"
