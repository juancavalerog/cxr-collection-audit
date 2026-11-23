"""Sonda lineal sobre un codificador congelado: RAD-DINO [perezgarcia2025exploring] (microsoft/rad-dino).

El codificador se congela (no está entre los parámetros del tronco entrenables: la fase única de la configuración
tiene train_backbone: false y además se fija requires_grad = False) y se entrena solo una capa lineal sobre el
token CLS. Los rasgos de entrenamiento se guardan en memoria por image_id cuando el aumento está desactivado
(cache_train: true), de modo que solo la primera época pasa el codificador. Codificadores:
- "rad_dino": transformers.AutoModel.from_pretrained(hf_id); entrada redimensionada a input_res (518 en la ficha)
  y normalizada con la media y desviación de su AutoImageProcessor. Requiere `transformers` (transformers==5.20.2
  en docker/requirements.txt) y la descarga de pesos en el servidor. Sin probar en la VM.
- "timm": cualquier modelo de timm sin preentreno, como sustituto diminuto en las pruebas de CPU.
Riesgo declarado (docs/01_novedad.md, Section 5): RAD-DINO se preentrenó con NIH-CXR completo y probablemente vio
las imágenes RSNA de COVIDx.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.baselines.registry import Method, register


class FrozenProbe(nn.Module):
    def __init__(self, encoder, dim, input_res, mean, std, input_scale=1.0, kind="timm"):
        super().__init__()
        self.backbone = encoder
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.kind, self.input_res, self.input_scale = kind, int(input_res), float(input_scale)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.linear = nn.Linear(dim, 1)
        nn.init.zeros_(self.linear.weight); nn.init.zeros_(self.linear.bias)
        self.cache = {}
        self.fc1 = nn.Linear(dim, 1)     # solo para la interfaz común (n_features); no se usa
        self.fc1.requires_grad_(False)

    def backbone_parameters(self):
        return list(self.backbone.parameters())

    def backbone_norm_params(self):
        return []

    def l2_penalty(self):
        return None

    @torch.no_grad()
    def embed(self, x):
        self.backbone.eval()
        x = x / self.input_scale
        if x.shape[-1] != self.input_res:
            x = F.interpolate(x, size=(self.input_res, self.input_res), mode="bicubic", align_corners=False)
        x = (x - self.mean) / self.std
        if self.kind == "rad_dino":
            return self.backbone(pixel_values=x).pooler_output.float()
        return self.backbone(x).float()

    def pooled(self, x):
        return self.embed(x)

    def forward(self, x):
        return self.linear(self.embed(x)).squeeze(-1)


@register
class RadDinoProbe(Method):
    name = "rad_dino_probe"
    title = "Sonda lineal sobre RAD-DINO congelado"
    ref = "perezgarcia2025exploring"
    labels = ("clase",)
    status = "pendiente_servidor"   # implementado; probado solo con un sustituto diminuto
    empty_cells = "no usa la fuente; la representación preentrenada puede codificar la firma de fuente"
    defaults = {"encoder": "rad_dino", "hf_id": "microsoft/rad-dino", "timm_name": "vit_tiny_patch16_224",
                "input_res": 518, "cache_train": True, "timm_kwargs": {}}
    search = {}
    hp_source = "ficha del modelo (sonda lineal sobre CLS); solo los comunes (tasa)"
    cost = 0.6

    def build_net(self, tr):
        p = self.p
        scale = float(tr.cfg["model"].get("input_scale", 1.0))
        if p["encoder"] == "rad_dino":
            from transformers import AutoImageProcessor, AutoModel    # solo en el servidor
            enc = AutoModel.from_pretrained(p["hf_id"])
            proc = AutoImageProcessor.from_pretrained(p["hf_id"])
            return FrozenProbe(enc, enc.config.hidden_size, p["input_res"], proc.image_mean, proc.image_std, scale,
                               "rad_dino")
        import timm
        enc = timm.create_model(p["timm_name"], pretrained=False, num_classes=0, img_size=int(p["input_res"]),
                                **dict(p.get("timm_kwargs") or {}))
        return FrozenProbe(enc, enc.num_features, p["input_res"], (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), scale,
                           "timm")

    def loss(self, tr, net, x, y, s, gen, B):
        probe = net.net
        ids = getattr(tr, "_batch_ids", None)
        use_cache = bool(self.p["cache_train"]) and tr.augment is None and ids is not None and x.shape[0] == len(ids)
        if use_cache and all(i in probe.cache for i in ids):
            f = torch.stack([probe.cache[i] for i in ids]).to(x.device)
        else:
            f = probe.embed(x)
            if use_cache:
                for i, v in zip(ids, f.detach().cpu()):
                    probe.cache[i] = v
        z = probe.linear(f).squeeze(-1)
        return tr.loss_fn(z.float(), y, reduction="sum") / B, {}
