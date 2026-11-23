"""Mide parámetros y FLOPs (torch.utils.flop_counter) de CSD-Net, ConvNeXt-Tiny y Swin-Tiny a 224 y 384.

Uso: python3 scripts/model/complexity.py > logs/complexity_model.txt
FlopCounterMode cuenta 2 FLOPs por multiplicación-suma en convoluciones, matmul y atención; no cuenta
normalizaciones ni activaciones. Se informa en GMAC = FLOPs / 2 / 1e9.
"""
import sys
from pathlib import Path

import torch
import timm
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model import CSDNet, CSDNetConfig  # noqa: E402


def gmac(model, *args, **kw):
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(*args, **kw)
    return fc.get_total_flops() / 2 / 1e9


def nparams(model, exclude=()):
    return sum(p.numel() for n, p in model.named_parameters() if not any(n.startswith(e) for e in exclude))


torch.manual_seed(0)
print("modelo,resolucion,parametros,GMAC")
for res in (224, 384):
    m = CSDNet(CSDNetConfig(pretrained=False, img_size=res)).eval()
    x = torch.randn(1, 3, res, res)
    mask = torch.zeros(1, 1, res, res)
    mask[..., res // 4 : 3 * res // 4, res // 6 : 5 * res // 6] = 1
    print(f"CSD-Net inferencia (sin via de adquisicion),{res},{nparams(m, ('acq_', 'src_head', 'modulator'))},{gmac(m, x, compute_signature=False):.3f}")
    x2, m2 = x.repeat(2, 1, 1, 1), mask.repeat(2, 1, 1, 1)
    print(f"CSD-Net entrenamiento (real + cruzada + firma),{res},{nparams(m)},{gmac(m, x2, swap_index=torch.tensor([1,0]), lung_mask=m2) / 2:.3f}")
    print(f"CSD-Net entrenamiento (real + dentro de la fuente + cruzada + firma),{res},{nparams(m)},{gmac(m, x2, swap_index=torch.tensor([1,0]), within_index=torch.tensor([0,1]), lung_mask=m2) / 2:.3f}")
    sig = gmac(m.acq_encoder, *m.acq_input(m._gray(x), mask))
    print(f"  de ello codificador de adquisicion,{res},{nparams(m.acq_encoder)},{sig:.3f}")
    c = timm.create_model("convnext_tiny", pretrained=False, num_classes=1).eval()
    print(f"ConvNeXt-Tiny,{res},{nparams(c)},{gmac(c, x):.3f}")
    if res == 224:
        s = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=1).eval()
        name = "Swin-Tiny (ventana 7)"
    else:
        s = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=1, img_size=384, window_size=12).eval()
        name = "Swin-Tiny (ventana 12)"
    print(f"{name},{res},{nparams(s)},{gmac(s, x):.3f}")
