"""Modelo del artículo 1 en PyTorch: ConvNeXt-Tiny de timm más la cabeza de Keras.

Estructura de Keras (covid_official.ipynb, celda 1, build_backbone y build_model; tf_keras 2.16,
applications/convnext.py con include_top=False y pooling=None):

    entrada 0..255 -> PreStem Normalization((x - 255*mean) / sqrt(var)) -> ConvNeXt-Tiny (stem, 4 etapas)
    -> LayerNormalization(eps=1e-6) por píxel (sobre el mapa 7 x 7 x 768) -> GlobalAveragePooling2D
    -> Dense(128, relu, L2 1e-2) -> LayerNormalization(eps=1e-3) -> Dropout(0.5) -> Dense(1, sigmoide)

En timm, convnext_tiny con num_classes=0 y global_pool='' aplica head.norm (LayerNorm2d, eps=1e-6) sobre el
mapa espacial y no agrega: equivale exactamente a la LayerNorm final de Keras, así que el agregado medio se
hace aquí, después. El modelo devuelve el logit; la probabilidad es sigmoid(logit).

input_scale fija el rango de entrada: 255 para el canal "p1" (como Keras) y 1 para el canal "homogeneo"
(imágenes en [0, 1]); en ambos casos la normalización interna es la de ImageNet.

Otros troncos de timm (sensibilidad S-T, docs/08_evaluacion.md, enmienda E8, punto 3: `backbone: resnet50`). La
cabeza es la misma; `feature_map` es el mapa del tronco tras su `forward_head(pre_logits=True)` con `global_pool=''`
(en ConvNeXt, tras la LayerNorm final; en ResNet-50, la salida de la última etapa, 2048 canales) y `pooled` su media
espacial, de modo que la dimensión de fc1 sale de `num_features` del tronco. `pesos_info` registra el tronco y la
etiqueta efectiva de los pesos de timm (p. ej. convnext_tiny.in12k_ft_in1k o resnet50.a1_in1k), que el entrenador
copia en summary.json. Con un tronco con BatchNorm (ResNet-50), `estadisticas_bn_congeladas` deja las BatchNorm del
tronco en modo de evaluación mientras el tronco está congelado (fase A): congelar el tronco congela también sus
estadísticas, como `trainable = False` en Keras. ConvNeXt no tiene BatchNorm y nada de esto cambia su cálculo.

emulate_bf16=True reproduce en float32 los redondeos de la política mixed_bfloat16 de tf_keras (entradas y
salidas de cada capa en bfloat16, LayerNorm con eps < 1.001e-5 calculada en float32, última Dense en float32).
Solo sirve para comparar con las probabilidades guardadas por el artículo 1 (Puerta 2a); no se entrena con él.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

KERAS_MEAN_255 = (123.675, 116.28, 103.53)                       # 255 x (0.485, 0.456, 0.406)
KERAS_VAR_255 = (3409.9760250000004, 3262.6944000000003, 3291.890625)   # (255 x (0.229, 0.224, 0.225))^2


def _r(t: torch.Tensor) -> torch.Tensor:
    """Redondeo a bfloat16 (round-to-nearest-even) y vuelta a float32."""
    return t.to(torch.bfloat16).to(torch.float32)


class Paper1ConvNeXt(nn.Module):
    def __init__(self, backbone: str = "convnext_tiny", pretrained: bool = False, input_scale: float = 255.0,
                 hidden: int = 128, dropout: float = 0.5, head_ln_eps: float = 1e-3, drop_path_rate: float = 0.0,
                 l2_hidden: float = 1e-2):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="",
                                          drop_path_rate=drop_path_rate)
        self.pesos_info = pesos_info(self.backbone, backbone, pretrained)
        self.estadisticas_bn_congeladas = False
        self.input_scale = float(input_scale)
        k = self.input_scale / 255.0
        self.register_buffer("in_mean", torch.tensor(KERAS_MEAN_255).view(1, 3, 1, 1) * k, persistent=False)
        self.register_buffer("in_std", torch.tensor(KERAS_VAR_255).sqrt().view(1, 3, 1, 1) * k, persistent=False)
        nf = self.backbone.num_features
        self.fc1 = nn.Linear(nf, hidden)
        self.ln = nn.LayerNorm(hidden, eps=head_ln_eps)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)
        self.l2_hidden = float(l2_hidden)
        self.emulate_bf16 = False
        # Variantes de la emulación (ver docstring de _forward_bf16): fusión del sesgo con conv/matmul (un solo
        # redondeo, como _FusedConv2D y _FusedMatMul de grappler en GPU) y GELU calculada en float32.
        self.bf16_fuse_bias = False
        self.bf16_gelu_fp32 = False

    # ------------------------------------------------------------------ partes
    def normalize(self, x):
        return (x - self.in_mean) / self.in_std.clamp_min(1e-7)

    def feature_map(self, x):
        """Mapa del tronco antes del agregado: en ConvNeXt-Tiny, 768 x 7 x 7 tras la LayerNorm final; en ResNet-50,
        2048 x 7 x 7."""
        b = self.backbone
        x = b.forward_features(self.normalize(x))
        return b.forward_head(x, pre_logits=True)

    def pooled(self, x):
        return self.feature_map(x).mean(dim=(2, 3))

    def head(self, f):
        h = F.relu(self.fc1(f))
        h = self.drop(self.ln(h))
        return self.fc2(h).squeeze(-1)

    def forward(self, x):
        if self.emulate_bf16 and not self.training:
            return self._forward_bf16(x)
        return self.head(self.pooled(x))

    def train(self, mode: bool = True):
        """Como nn.Module.train; con `estadisticas_bn_congeladas`, las BatchNorm del tronco quedan en evaluación (no
        actualizan sus medias). Sin BatchNorm (ConvNeXt) no cambia nada."""
        super().train(mode)
        if mode and self.estadisticas_bn_congeladas:
            for m in self.backbone.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    # ------------------------------------------------------------------ utilidades de entrenamiento
    def backbone_norm_params(self):
        """Parámetros de las LayerNorm que Keras dejaba congeladas: las 18 de los bloques y la final
        (las de stem y downsampling estaban dentro de Sequential y quedaban entrenables)."""
        out = []
        for name, p in self.backbone.named_parameters():
            if (".blocks." in name and ".norm." in name) or name.startswith("head.norm."):
                out.append((name, p))
        return out

    def l2_penalty(self):
        """Regularizador L2 de Keras sobre el núcleo de la Dense 128: l2 * sum(W^2)."""
        return self.l2_hidden * self.fc1.weight.pow(2).sum()

    def param_groups(self):
        bb = [p for p in self.backbone.parameters()]
        hd = [p for n, p in self.named_parameters() if not n.startswith("backbone.")]
        return {"backbone": bb, "head": hd}

    # ------------------------------------------------------------------ emulación de mixed_bfloat16
    @torch.no_grad()
    def _forward_bf16(self, x):
        """Emulación de mixed_bfloat16: cada capa recibe y devuelve bf16 (redondeo r), pesos leídos en bf16,
        acumulación en float32. Las LayerNorm con eps 1e-6 (ruta no fusionada) se calculan en float32; la de la
        cabeza (eps 1e-3, ruta fusionada) normaliza en float32 y aplica escala y sesgo en bf16. La Dense final es
        float32. Variantes: bf16_fuse_bias (sesgo dentro de la conv/matmul, un redondeo) y bf16_gelu_fp32."""
        b = self.backbone
        r = _r
        # PreStem Normalization con política mixed_bfloat16: todo en bf16
        mean = r(torch.tensor(KERAS_MEAN_255, device=x.device).view(1, 3, 1, 1) * (self.input_scale / 255.0))
        std = r(torch.sqrt(r(torch.tensor(KERAS_VAR_255, device=x.device).view(1, 3, 1, 1)
                              * (self.input_scale / 255.0) ** 2)))
        x = r(r(r(x) - mean) / std)

        fuse, gelu32 = self.bf16_fuse_bias, self.bf16_gelu_fp32

        def conv(m, t):
            if fuse:
                return r(F.conv2d(t, r(m.weight), r(m.bias), m.stride, m.padding, m.dilation, m.groups))
            y = r(F.conv2d(t, r(m.weight), None, m.stride, m.padding, m.dilation, m.groups))
            return r(y + r(m.bias).view(1, -1, 1, 1)) if m.bias is not None else y

        def ln2d(m, t):      # LayerNorm eps 1e-6: Keras la calcula en float32 y devuelve bf16
            t = t.permute(0, 2, 3, 1)
            t = F.layer_norm(t, m.normalized_shape, m.weight, m.bias, m.eps)
            return r(t.permute(0, 3, 1, 2))

        def dense(m, t):
            if fuse:
                return r(F.linear(t, r(m.weight), r(m.bias)))
            y = r(F.linear(t, r(m.weight)))
            return r(y + r(m.bias))

        def gelu(t):
            if gelu32:
                return r(F.gelu(t))
            c = r(0.5 * t)
            e = r(torch.erf(r(t / r(torch.tensor(math.sqrt(2.0))))))
            return r(c * r(1.0 + e))

        x = conv(b.stem[0], x)
        x = ln2d(b.stem[1], x)
        for st in b.stages:
            if not isinstance(st.downsample, nn.Identity):
                x = ln2d(st.downsample[0], x)
                x = conv(st.downsample[1], x)
            for blk in st.blocks:
                sc = x
                y = conv(blk.conv_dw, x).permute(0, 2, 3, 1)
                y = r(F.layer_norm(y, blk.norm.normalized_shape, blk.norm.weight, blk.norm.bias, blk.norm.eps))
                y = dense(blk.mlp.fc1, y)
                y = gelu(y)
                y = dense(blk.mlp.fc2, y)
                if blk.gamma is not None:
                    y = r(y * r(blk.gamma))
                x = r(sc + y.permute(0, 3, 1, 2))
        x = ln2d(b.head.norm, x)
        f = r(x.mean(dim=(2, 3)))
        h = r(F.relu(dense(self.fc1, f)))
        # LayerNorm eps 1e-3: ruta "fused" de Keras; normaliza en float32, salida bf16, escala y sesgo en bf16
        n = r(F.layer_norm(h, self.ln.normalized_shape, None, None, self.ln.eps))
        h = r(r(n * r(self.ln.weight)) + r(self.ln.bias))
        return F.linear(h, self.fc2.weight, self.fc2.bias).squeeze(-1)     # Dense final en float32


def pesos_info(model, backbone: str, pretrained: bool) -> dict:
    """Tronco y etiqueta efectiva de los pesos de timm (solo lectura de pretrained_cfg; no altera el modelo)."""
    pc = dict(getattr(model, "pretrained_cfg", None) or {})
    arq, tag = pc.get("architecture") or backbone, pc.get("tag")
    etiqueta = "%s.%s" % (arq, tag) if tag else arq
    n_bn = sum(1 for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))
    return {"tronco": backbone, "preentrenado": bool(pretrained),
            "etiqueta_pesos": etiqueta if pretrained else None, "etiqueta_por_defecto_timm": etiqueta,
            "hf_hub_id": pc.get("hf_hub_id") if pretrained else None, "num_features": int(model.num_features),
            "n_batchnorm": int(n_bn), "timm": _version_timm()}


def _version_timm():
    try:
        import timm
        return timm.__version__
    except Exception:  # noqa: BLE001
        return None


def build_paper1_convnext(state_dict=None, **kw) -> Paper1ConvNeXt:
    m = Paper1ConvNeXt(**kw)
    if state_dict is not None:
        missing, unexpected = m.load_state_dict(state_dict, strict=False)
        missing = [k for k in missing]
        if missing or unexpected:
            raise RuntimeError("state_dict incompatible: faltan %s; sobran %s" % (missing[:5], unexpected[:5]))
    return m
