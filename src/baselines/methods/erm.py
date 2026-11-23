"""ERM, control de fuente y ERM con enmascarado pulmonar."""
from __future__ import annotations

from src.baselines.registry import Method, register


def erm_loss(tr, net, x, y, B):
    z = net(x)
    return tr.loss_fn(z.float(), y, reduction="sum") / B


@register
class ERM(Method):
    name = "erm"
    title = "ERM"
    ref = "gulrajani2021search"
    labels = ("clase",)
    empty_cells = "no usa la fuente; las celdas vacías no le afectan de forma explícita (aprende el atajo)"
    defaults = {}
    search = {}
    hp_source = "DomainBed (hparams_registry.py): solo los comunes (tasa)"
    cost = 1.0

    def loss(self, tr, net, x, y, s, gen, B):
        return erm_loss(tr, net, x, y, B), {}


@register
class ControlFuente(ERM):
    """ERM con el entrenamiento restringido a bimcv (data.train_sources: [bimcv]) [valero2026learning]."""
    name = "control_fuente"
    title = "Control de fuente (solo bimcv)"
    ref = "valero2026learning"
    labels = ("clase", "fuente (para filtrar)")
    empty_cells = "elimina el problema descartando todas las fuentes salvo bimcv (94 % de los negativos fuera)"
    hp_source = "artículo 1; solo los comunes (tasa)"
    cost = 0.65

    def aplicabilidad(self, tr):
        """En los benchmarks del plan B el control entrena solo con las fuentes mixtas del manifiesto
        (data.train_sources: mixtas). Sin fuentes mixtas (B3a, rho = 1.0) no tiene datos; si todas las fuentes son
        mixtas (B3a, rho < 1) el filtro no quita nada y la corrida sería idéntica a ERM."""
        if not tr.cfg["data"].get("bench"):
            return "aplicable", None
        b = tr.cfg["bench"]
        mix, ftr = set(b["fuentes_mixtas"]), set(b["fuentes_train"])
        if not mix:
            return "no_aplicable", "ninguna fuente de entrenamiento tiene las dos clases: el control de fuente no tiene datos"
        if ftr <= mix:
            return "no_aplicable", ("todas las fuentes de entrenamiento tienen las dos clases: el filtro no quita ninguna "
                                    "imagen y la corrida sería idéntica a ERM")
        return "aplicable", None

    def prepare_data(self, tr):
        srcs = set(tr.frames["train"]["source"].unique())
        want = set(tr.cfg["data"].get("train_sources") or [])
        if tr.cfg["data"].get("bench"):
            want = set(tr.cfg["bench"].get("train_sources") or [])
        if not want or not srcs <= want:
            raise ValueError("control_fuente exige data.train_sources (p. ej. [bimcv]); fuentes en train: %s" % sorted(srcs))


@register
class ERMCuotas(ERM):
    """ERM con el muestreador de cuotas de CSD-Net (src/model/swap.py, `TableGuidedBatchSampler`): en cada lote,
    `min_per_source` imágenes de cada fuente, `min_per_class_mixed` de cada clase en cada fuente con las dos clases
    y el resto equilibrado por clase. Es el control que separa el efecto del muestreador del efecto del método:
    CSD-Net entrena con este muestreador y las demás líneas base con el equilibrado por clase. Equilibrar grupos
    por remuestreo es por sí solo una línea base fuerte [idrissi2022simple]."""
    name = "erm_cuotas"
    degenerado_sin_mixtas = ("sin fuentes mixtas la cuota por clase dentro de fuente no actúa: solo quedan "
                             "min_per_source imágenes por fuente y el resto equilibrado por clase")
    title = "ERM con el muestreador de cuotas de CSD-Net"
    ref = "idrissi2022simple"
    labels = ("clase", "fuente (para muestrear)")
    empty_cells = ("sobremuestrea las celdas escasas de las fuentes con las dos clases; no puede poblar las celdas "
                   "vacías")
    defaults = {"min_per_source": 2, "min_per_class_mixed": 4}
    search = {}
    hp_source = ("cuotas por defecto de CSD-Net (docs/03_arquitectura.md, Tabla 6), fijas; lote de 64 como CSD-Net; "
                 "solo los comunes (tasa)")
    cost = 1.3

    def prepare_data(self, tr):
        tr.quota_sampler = {"min_per_source": int(self.p["min_per_source"]),
                            "min_per_class_mixed": int(self.p["min_per_class_mixed"])}


@register
class ERMMascara(ERM):
    """ERM sobre la imagen multiplicada por la máscara pulmonar [teixeira2021impact]; máscaras del profesor de
    segmentación (data/masks/, CheXmask [gaggion2024chexmask]). Falla con un mensaje claro si falta alguna."""
    name = "erm_mascara"
    title = "ERM con enmascarado pulmonar"
    ref = "teixeira2021impact; gaggion2024chexmask"
    labels = ("clase", "máscara pulmonar")
    empty_cells = "no usa la fuente; reduce la firma fuera del tórax pero no completa celdas"
    defaults = {"mask_dir": "data/masks/covidx_cxr4/homogeneo_224", "mask_resize": False}
    hp_source = "sin hiperparámetros propios (preprocesado); solo los comunes"
    needs_mask = True
    cost = 1.0

    def prepare_data(self, tr):
        if tr.cfg["data"].get("bench") and "covidx" in str(self.p["mask_dir"]):
            raise ValueError("erm_mascara: en un benchmark del plan B mask_dir debe ser el del benchmark "
                             "(data/masks/bench_<id>/homogeneo_224), no el de COVIDx (%s)" % self.p["mask_dir"])
