"""Paquete del modelo CSD-Net (docs/03_arquitectura.md)."""
from .csdnet import CSDNet, CSDNetConfig, build_csdnet, config_to_dict
from .layers import (
    AcquisitionEncoder,
    AcquisitionInput,
    AnatomicalGate,
    HaarHighPass,
    ReSignature,
    SignatureModulator,
    TransformerStage,
    instance_stats,
    thoracic_exclusion,
)
from .losses import LossWeights, L_cls_real, L_cls_swap, L_cons, L_ind, L_seg, L_sig, L_src, L_total, focal_bce
from .swap import (ClassSourceTable, RoutingPlan, SwapPairs, TableGuidedBatchSampler, WithinPairs,
                   sample_swap_pairs, sample_within_pairs)

__all__ = [
    "CSDNet", "CSDNetConfig", "build_csdnet", "config_to_dict",
    "AcquisitionEncoder", "AcquisitionInput", "AnatomicalGate", "HaarHighPass", "ReSignature",
    "SignatureModulator", "TransformerStage", "instance_stats", "thoracic_exclusion",
    "LossWeights", "L_cls_real", "L_cls_swap", "L_cons", "L_ind", "L_seg", "L_sig", "L_src", "L_total",
    "focal_bce", "ClassSourceTable", "RoutingPlan", "SwapPairs", "TableGuidedBatchSampler", "sample_swap_pairs",
    "WithinPairs", "sample_within_pairs",
]
