"""Clasificador de solo adquisición (C1-iv; docs/08_evaluacion.md, V2.3.4).

Versión de producción del prototipo del diagnóstico de la firma del 13-11-2026 (scripts/diag/r2_cache.py, r2_lib.py, r2_origen_clase.py y
r2_asociacion_particiones.py; docs/10_correccion_independencia.md, Section 4, Tabla 1). Rasgos: los 14 estadísticos
de la entrada de adquisición de la ecuación (16) de CSD-Net (`src.model.layers.AcquisitionInput`, valores por
defecto): media del valor absoluto y desviación típica, dentro de la zona visible, de las seis sub-bandas de detalle
de Haar y de LL1 en la banda periférica. La zona visible excluye la región torácica derivada de la unión de las
máscaras del profesor (pulmones o corazón; vacía si la calidad es 0); no se usa ninguna otra información anatómica.
Sonda: la de V1 (`src.eval.probes.make_probe('logreg')`: estandarización y regresión logística con C = 1).

- Variante (a): sonda ajustada con bimcv de entrenamiento (todas las negativas y la muestra de 2 000 positivas del
  prototipo, semilla 0) y aplicada al conjunto evaluado.
- Variante (b): validación cruzada de cinco pliegues agrupados por paciente dentro del conjunto evaluado (semilla 42),
  como el prototipo; mide cuánta clase hay en la adquisición de ese conjunto y no es un clasificador desplegable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.paths import PROJECT_ROOT

CACHE = PROJECT_ROOT / "data" / "cache_m" / "covidx_cxr4" / "homogeneo_224"
MASCARAS = PROJECT_ROOT / "data" / "masks" / "covidx_cxr4" / "homogeneo_224"
N_POS_AJUSTE = 2000            # CAP_TRAIN[('bimcv', 1)] de scripts/diag/r2_cache.py
SEMILLA_AJUSTE = 0
NOMBRES = ["media_abs_%s" % c for c in ("LH1", "HL1", "HH1", "LH2", "HL2", "HH2", "LL1_periferia")] + \
          ["desv_%s" % c for c in ("LH1", "HL1", "HH1", "LH2", "HL2", "HH2", "LL1_periferia")]
_ACQ = None


def entrada(img_u8: torch.Tensor, mk_u8: torch.Tensor):
    """(x_acq, keep) desde imágenes uint8 (n, 224, 224) y la unión de máscaras uint8 (n, 224, 224)."""
    global _ACQ
    if _ACQ is None:
        from src.model.layers import AcquisitionInput
        _ACQ = AcquisitionInput()
    g = img_u8.float().div(255.0)[:, None]
    m = mk_u8.float()[:, None]
    with torch.no_grad():
        return _ACQ(g, m)


def estadisticos(img_u8: torch.Tensor, mk_u8: torch.Tensor, lote: int = 256) -> np.ndarray:
    """Los 14 estadísticos (ver NOMBRES; el orden de los canales es el de AcquisitionInput)."""
    st = []
    for i in range(0, len(img_u8), lote):
        xa, k = entrada(img_u8[i:i + lote], mk_u8[i:i + lote])
        mass = k.sum((2, 3)).clamp_min(1)
        mu = (xa.abs() * k).sum((2, 3)) / mass
        sd = (((xa - (xa * k).sum((2, 3), keepdim=True) / mass[..., None, None]) ** 2) * k).sum((2, 3)) / mass
        st.append(torch.cat([mu, sd.sqrt()], 1))
    return torch.cat(st).numpy().astype(np.float64) if st else np.zeros((0, 14))


def indices():
    ci = pd.read_csv(CACHE / "index.csv")
    mi = pd.read_csv(MASCARAS / "index.csv")
    return ci[ci.status == "ok"][["image_id", "rel_path"]], mi[mi.status == "ok"][["image_id", "path", "quality"]]


def cargar_una(rel_path: str, mask_path: str, q: float, cache: Path = CACHE, mascaras: Path = MASCARAS):
    from PIL import Image
    from src.data.homogeneo_masks import decode_code
    with Image.open(Path(cache) / rel_path) as im:
        x = np.asarray(im, dtype=np.uint8)
    if x.ndim == 3:
        x = x[..., 0]
    with Image.open(Path(mascaras) / mask_path) as im:
        m = decode_code(np.asarray(im)).max(0).astype(np.uint8)
    if q <= 0:
        m[:] = 0
    return x, m


def muestra_ajuste(man: pd.DataFrame) -> pd.DataFrame:
    """bimcv de entrenamiento de la caché del prototipo: todas las negativas y 2 000 positivas (semilla 0)."""
    t = man[(man["split_clean"] == "train") & (man["source"] == "bimcv")]
    neg = t[t["label"] == 0]
    pos = t[t["label"] == 1]
    pos = pos.sample(min(N_POS_AJUSTE, len(pos)), random_state=SEMILLA_AJUSTE)
    return pd.concat([neg, pos]).reset_index(drop=True)


def variante_a(X_aj, y_aj, X_ev):
    from src.eval.probes import make_probe
    return make_probe("logreg", 0).fit(np.asarray(X_aj, float), np.asarray(y_aj).astype(int)).predict_proba(np.asarray(X_ev, float))[:, 1]


def variante_b(X_ev, y_ev, g_ev, k: int = 5, semilla: int = 42):
    from src.eval.probes import oof_proba
    return oof_proba(np.asarray(X_ev, float), np.asarray(y_ev).astype(int), np.asarray(g_ev).astype(str), "logreg", k, semilla)
