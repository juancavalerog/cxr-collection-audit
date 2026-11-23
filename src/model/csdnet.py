"""CSD-Net: tronco híbrido ConvNeXt + Transformer con vía de adquisición, re-firma y compuerta anatómica.

El forward devuelve un diccionario con los logits de enfermedad (reales y re-firmados), los logits de
fuente, los logits de segmentación y los embeddings que usan los probes de identificabilidad
(docs/03_arquitectura.md, Sections 2 y 3). Dependencias: torch, timm.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .layers import (
    AcquisitionEncoder,
    AcquisitionInput,
    AnatomicalGate,
    ReSignature,
    SegHead,
    SignatureModulator,
    TransformerStage,
    instance_stats,
    thoracic_exclusion,
)


@dataclass
class CSDNetConfig:
    """Configuración de CSD-Net. Los valores por defecto son los de la Section 2 del documento."""

    img_size: int = 224
    num_sources: int = 8
    backbone: str = "convnext_tiny.in12k_ft_in1k"
    pretrained: bool = True
    backbone_ckpt: Optional[str] = None
    model_ckpt: Optional[str] = None
    num_conv_stages: int = 3
    transformer_depth: int = 4
    transformer_heads: int = 6
    transformer_init: str = "vit_small"  # "vit_small" o "random"
    vit_name: str = "vit_small_patch16_224.augreg_in21k_ft_in1k"
    drop_path: float = 0.1
    # vía de adquisición
    use_acquisition: bool = True
    use_highpass: bool = True
    use_lung_exclusion: bool = True
    exclusion_mode: str = "box"
    box_margin: float = 0.05
    dilate_frac: float = 0.08
    periphery_frac: float = 0.125
    acq_widths: Tuple[int, ...] = (32, 64, 128, 128)
    sig_dim: int = 64
    modulator_hidden: int = 256
    # re-firma
    refirma_stage: int = 2
    refirma_mode: str = "relative"  # "relative", "absolute" o "mixstyle"
    log_scale_clamp: float = 2.0
    mix_alpha: float = 0.1
    refirma_target: str = "table"  # "table" (completación dirigida) o "canonical" (ablación A14)
    canonical_momentum: float = 0.99
    # compuerta anatómica
    use_gate: bool = True
    gate_region: str = "lungs"  # "lungs" (COVID, TB), "lungs_heart" (cardiomegalia, derrame) o "thorax"
    seg_channels: int = 2  # canal 0 pulmones, canal 1 corazón (CheXmask)
    gate_threshold: float = 0.1
    gate_detach: bool = True
    seg_width: int = 64
    # cabeza de decisión
    head_hidden: int = 128
    head_dropout: float = 0.5
    # normalización de entrada (la del tronco preentrenado)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        if not self.use_acquisition and self.refirma_mode != "mixstyle":
            self.refirma_mode = "mixstyle"
        if self.img_size % 32 != 0:
            raise ValueError("img_size debe ser múltiplo de 32")
        if self.refirma_stage not in range(self.num_conv_stages):
            raise ValueError("refirma_stage fuera de rango")
        if self.gate_region not in ("lungs", "lungs_heart", "thorax"):
            raise ValueError(f"gate_region desconocida: {self.gate_region}")
        if self.gate_region == "lungs_heart" and self.seg_channels < 2:
            raise ValueError("gate_region='lungs_heart' requiere seg_channels >= 2")
        if self.refirma_target not in ("table", "canonical"):
            raise ValueError(f"refirma_target desconocido: {self.refirma_target}")
        if self.refirma_target == "canonical" and not self.use_acquisition:
            raise ValueError("la re-firma canónica requiere la vía de adquisición")


class CSDNet(nn.Module):
    """Modelo CSD-Net. Ver docs/03_arquitectura.md, Section 2."""

    def __init__(self, cfg: CSDNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        use_timm_weights = cfg.pretrained and cfg.backbone_ckpt is None and cfg.model_ckpt is None
        cnx = timm.create_model(cfg.backbone, pretrained=use_timm_weights, drop_path_rate=cfg.drop_path)
        if cfg.backbone_ckpt is not None:
            sd = torch.load(cfg.backbone_ckpt, map_location="cpu")
            sd = sd.get("state_dict", sd)
            cnx.load_state_dict(sd, strict=False)
        chs = [cnx.feature_info[i]["num_chs"] for i in range(cfg.num_conv_stages)]
        self.channels: List[int] = chs
        self.stem = cnx.stem
        self.stages = nn.ModuleList(list(cnx.stages)[: cfg.num_conv_stages])
        del cnx
        dim = chs[-1]
        vit_init = cfg.transformer_init == "vit_small"
        self.transformer = TransformerStage(dim, cfg.transformer_depth, cfg.transformer_heads, 4.0, cfg.drop_path,
                                            ls_init=1.0 if vit_init else 1e-5)
        if vit_init and use_timm_weights:
            self._init_transformer_from_vit()
        self.seg_head = SegHead(tuple(chs), cfg.seg_width, cfg.seg_channels)
        self.gate = AnatomicalGate(cfg.gate_threshold)
        c_ins = chs[cfg.refirma_stage]
        if cfg.use_acquisition:
            self.acq_input = AcquisitionInput(cfg.periphery_frac, cfg.exclusion_mode, cfg.box_margin, cfg.dilate_frac,
                                              exclude=cfg.use_lung_exclusion, highpass=cfg.use_highpass)
            self.acq_encoder = AcquisitionEncoder(self.acq_input.out_channels, tuple(cfg.acq_widths), cfg.sig_dim)
            self.src_head = nn.Linear(cfg.sig_dim, cfg.num_sources)
            self.modulator = SignatureModulator(cfg.sig_dim, c_ins, cfg.modulator_hidden)
        self.resig = ReSignature(cfg.refirma_mode, cfg.log_scale_clamp, mix_alpha=cfg.mix_alpha)
        # firma de referencia para la ablación canónica: media móvil de (g, b) en entrenamiento
        self.register_buffer("ref_g", torch.zeros(c_ins))
        self.register_buffer("ref_b", torch.zeros(c_ins))
        self.register_buffer("ref_init", torch.zeros((), dtype=torch.bool))
        self.head_norm = nn.LayerNorm(dim)
        self.head_fc = nn.Linear(dim, cfg.head_hidden)
        self.head_act = nn.GELU()
        self.head_drop = nn.Dropout(cfg.head_dropout)
        self.head_out = nn.Linear(cfg.head_hidden, 1)
        self.register_buffer("in_mean", torch.tensor(cfg.mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("in_std", torch.tensor(cfg.std).view(1, 3, 1, 1), persistent=False)
        if cfg.model_ckpt is not None:
            self.load_checkpoint(cfg.model_ckpt)

    # ------------------------------------------------------------------ inicialización y carga
    def _init_transformer_from_vit(self) -> None:
        """Copia los últimos `depth` bloques de un ViT-S de timm (dim 384, 6 cabezas)."""
        vit = timm.create_model(self.cfg.vit_name, pretrained=True)
        src_blocks = list(vit.blocks)[-self.cfg.transformer_depth :]
        for dst, src in zip(self.transformer.blocks, src_blocks):
            dst.load_state_dict(src.state_dict(), strict=False)
        del vit

    def load_checkpoint(self, path: str, strict: bool = True) -> Tuple[List[str], List[str]]:
        """Carga un state_dict completo de CSD-Net (acepta envoltorio {'state_dict': ...} y prefijo 'module.')."""
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("state_dict", sd)
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        res = self.load_state_dict(sd, strict=strict)
        return list(res.missing_keys), list(res.unexpected_keys)

    def param_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Grupos de parámetros (para el optimizador y para las pruebas de flujo de gradiente)."""
        groups: Dict[str, List[nn.Parameter]] = {
            "trunk_conv": list(self.stem.parameters()) + list(self.stages.parameters()),
            "transformer": list(self.transformer.parameters()),
            "seg_head": list(self.seg_head.parameters()),
            "decision_head": [p for m in (self.head_norm, self.head_fc, self.head_out) for p in m.parameters()],
        }
        if self.cfg.use_acquisition:
            groups["acq_encoder"] = list(self.acq_encoder.parameters())
            groups["src_head"] = list(self.src_head.parameters())
            groups["modulator"] = list(self.modulator.parameters())
        return groups

    # ------------------------------------------------------------------ piezas del forward
    def _gray(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1] * self.in_std[:, :1] + self.in_mean[:, :1]

    def _head(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.head_act(self.head_fc(self.head_norm(z)))
        return self.head_out(self.head_drop(h)).squeeze(-1), h

    def _run_stages(self, f: torch.Tensor, start: int) -> torch.Tensor:
        for st in list(self.stages)[start:]:
            f = st(f)
        return f

    def gate_region_mask(self, prob: torch.Tensor) -> torch.Tensor:
        """Máscara de la compuerta según el benchmark (Section 2.5): pulmones, pulmones y corazón o región
        torácica (caja envolvente de los pulmones). `prob` es (B, C, H, W) con C = 1 o 2."""
        region = self.cfg.gate_region
        if prob.shape[1] == 1 or region == "lungs":
            return prob[:, :1]
        if region == "lungs_heart":
            return torch.maximum(prob[:, :1], prob[:, 1:2])
        return thoracic_exclusion(prob[:, :1], "box", margin=0.02)

    def signature(self, x: torch.Tensor, lung_mask: torch.Tensor) -> torch.Tensor:
        """Firma de adquisición a partir de la imagen normalizada y de una máscara anatómica (B, C, H, W);
        con varios canales se excluye la unión (Section 2.3)."""
        m = lung_mask.float().amax(dim=1, keepdim=True)
        x_acq, keep = self.acq_input(self._gray(x).float(), m)
        return self.acq_encoder(x_acq, keep)

    def _view(self, fk: torch.Tensor, g: torch.Tensor, b: torch.Tensor, g_dst: torch.Tensor,
              b_dst: torch.Tensor) -> torch.Tensor:
        """Re-firma con modulaciones separadas del grafo y etapas conv restantes."""
        k = self.cfg.refirma_stage
        f = self.resig(fk, g.detach(), b.detach(), g_dst.detach(), b_dst.detach())
        return self._run_stages(f, k + 1) if k + 1 < len(self.stages) else f

    def forward(
        self,
        x: torch.Tensor,
        swap_index: Optional[torch.Tensor] = None,
        lung_mask: Optional[torch.Tensor] = None,
        gate_mask: Optional[torch.Tensor] = None,
        compute_signature: bool = True,
        within_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward.

        x: (B, 3, H, W) normalizada. swap_index: (B,) donante de otra fuente (vista cruzada) o None.
        within_index: (B,) donante de la misma fuente (vista dentro de la fuente, Section 2.4) o None.
        lung_mask: máscara del profesor (B, C, H, W) para la entrada de adquisición; si es None se usa la
        predicha, separada del grafo. gate_mask: máscara para la compuerta (B, C, H, W); si es None, la
        predicha. compute_signature=False omite la vía de adquisición (inferencia de la decisión).
        Con refirma_target="canonical" todas las vistas, también en inferencia, se llevan a la firma de
        referencia y se ignoran swap_index y within_index.
        """
        cfg = self.cfg
        out: Dict[str, torch.Tensor] = {}
        f = self.stem(x)
        feats: List[torch.Tensor] = []
        for st in self.stages:
            f = st(f)
            feats.append(f)
        f_last = feats[-1]
        seg_logits = self.seg_head(*feats[-3:])
        out["seg_logits"] = seg_logits
        seg_prob = torch.sigmoid(seg_logits.float())
        gate_src = gate_mask.float() if gate_mask is not None else (seg_prob.detach() if cfg.gate_detach else seg_prob)
        gate_src = self.gate_region_mask(gate_src)
        grid = (f_last.shape[-2], f_last.shape[-1])
        if cfg.use_gate:
            w = self.gate.token_weights(gate_src, grid)
        else:
            w = torch.ones(x.shape[0], grid[0] * grid[1], device=x.device)
        out["token_weights"] = w

        k = cfg.refirma_stage
        fk = feats[k]
        mu, sd = instance_stats(fk)
        log_sd, mu_f = torch.log(sd).flatten(1), mu.flatten(1)
        out["stats_real"] = torch.cat([log_sd, mu_f], dim=1).detach()
        anchor = seg_logits.float().sum() * 0.0

        g: Optional[torch.Tensor] = None
        b: Optional[torch.Tensor] = None
        canonical = cfg.refirma_target == "canonical"
        if canonical and not compute_signature:
            raise ValueError("la re-firma canónica necesita la firma también en inferencia")
        if cfg.use_acquisition and compute_signature:
            acq_mask = lung_mask if lung_mask is not None else (
                F.interpolate(seg_prob.detach(), size=x.shape[-2:], mode="bilinear", align_corners=False))
            a = self.signature(x, acq_mask)
            out["signature"] = a
            out["src_logits"] = self.src_head(a)
            g, b = self.modulator(a.detach())
            out["sig_pred"] = torch.cat([g, b], dim=1)
            anchor = anchor + 0.0 * (out["src_logits"].float().sum() + out["sig_pred"].sum())
            if self.training:
                with torch.no_grad():
                    gm, bm = g.detach().mean(0), b.detach().mean(0)
                    if not bool(self.ref_init):
                        self.ref_g.copy_(gm)
                        self.ref_b.copy_(bm)
                        self.ref_init.fill_(True)
                    else:
                        m = cfg.canonical_momentum
                        self.ref_g.mul_(m).add_((1 - m) * gm)
                        self.ref_b.mul_(m).add_((1 - m) * bm)
        elif not cfg.use_acquisition:
            g, b = log_sd.detach(), mu_f.detach()

        bsz = x.shape[0]
        views: List[Tuple[str, torch.Tensor]] = []
        if canonical:
            rg, rb = self.ref_g.expand_as(g), self.ref_b.expand_as(b)
            views.append(("real", self._view(fk, g, b, rg, rb)))
        else:
            views.append(("real", f_last))
            if within_index is not None and g is not None:
                jw = within_index.long()
                views.append(("within", self._view(fk, g, b, g.detach()[jw], b.detach()[jw])))
            if swap_index is not None and g is not None:
                j = swap_index.long()
                fk_sw = self.resig(fk, g.detach(), b.detach(), g.detach()[j], b.detach()[j])
                mu_s, sd_s = instance_stats(fk_sw)
                out["stats_swap"] = torch.cat([torch.log(sd_s).flatten(1), mu_s.flatten(1)], dim=1).detach()
                views.append(("swap", self._view(fk, g, b, g.detach()[j], b.detach()[j])))
        nv = len(views)
        f_in = torch.cat([v for _, v in views], dim=0) if nv > 1 else views[0][1]
        w_in = w.repeat(nv, 1) if nv > 1 else w
        bias = self.gate.attn_bias(w_in, f_in.dtype) if cfg.use_gate else None
        tokens = self.transformer(f_in, bias)
        z = self.gate.pool(tokens, w_in) if cfg.use_gate else tokens.mean(dim=1)
        logit, h = self._head(z)
        out["logit"] = logit[:bsz]
        out["emb_content"] = views[0][1].mean(dim=(2, 3)) if canonical else f_last.mean(dim=(2, 3))
        out["emb_tokens"] = tokens[:bsz].mean(dim=1)
        out["emb_gated"] = z[:bsz]
        out["emb_decision"] = h[:bsz]
        for v_i, (name, fv) in enumerate(views[1:], start=1):
            sl = slice(v_i * bsz, (v_i + 1) * bsz)
            out[f"logit_{name}"] = logit[sl]
            out[f"emb_{name}_content"] = fv.mean(dim=(2, 3))
            out[f"emb_{name}_gated"] = z[sl]
            out[f"emb_{name}_decision"] = h[sl]
        out["ddp_anchor"] = anchor
        return out


def build_csdnet(**kwargs) -> CSDNet:
    """Atajo: construye CSD-Net a partir de argumentos de CSDNetConfig."""
    return CSDNet(CSDNetConfig(**kwargs))


def config_to_dict(cfg: CSDNetConfig) -> Dict[str, object]:
    """Serializa la configuración (para guardar junto al checkpoint)."""
    return asdict(cfg)
