"""Conversión de pesos ConvNeXt-Tiny de Keras (tf_keras 2.16) a timm `convnext_tiny` más la cabeza del artículo 1.

Lee los ficheros solo con h5py (sin TensorFlow):

1. `.h5` del preentreno de dominio (`base.save_weights`, formato h5 heredado): grupos con el nombre de cada
   capa (`convnext_tiny_stage_0_block_0_depthwise_conv/.../kernel:0`). Solo tronco (180 tensores).
2. `.keras` de los modelos COVID (zip con `config.json` y `model.weights.h5`, formato de saving_lib): los grupos
   usan nombres genéricos por clase y orden (`layers/functional/layers/conv2d_7/vars/0`). El orden se reconstruye
   con `config.json`: dentro de cada Functional o Sequential se cuenta cada clase en el orden de la lista de
   capas (en Functional cuenta también la InputLayer; en Sequential no) y el contador da el sufijo `_N`. Así se
   recupera el nombre semántico de Keras de cada grupo y la correspondencia es la misma que en el caso 1.

Correspondencia (Keras -> timm, con k = convnext_tiny):

| Keras | timm | Transformación |
|---|---|---|
| k_prestem_normalization (sin pesos; media 255 x ImageNet, varianza (255 x std)^2 en config) | buffers in_mean / in_std de Paper1ConvNeXt | constantes |
| k_stem_conv kernel (4, 4, 3, 96) y bias | backbone.stem.0.weight (96, 3, 4, 4) y bias | HWIO -> OIHW (permute 3, 2, 0, 1) |
| k_stem_layernorm gamma, beta | backbone.stem.1.weight, bias | ninguna |
| k_downsampling_layernorm_i | backbone.stages.{i+1}.downsample.0 | ninguna |
| k_downsampling_conv_i kernel (2, 2, Cin, Cout) | backbone.stages.{i+1}.downsample.1.weight | HWIO -> OIHW |
| k_stage_i_block_j_depthwise_conv kernel (7, 7, 1, C) | backbone.stages.i.blocks.j.conv_dw.weight (C, 1, 7, 7) | permute 3, 2, 0, 1 |
| k_stage_i_block_j_layernorm | ...blocks.j.norm | ninguna |
| k_stage_i_block_j_pointwise_conv_1 kernel (C, 4C) | ...blocks.j.mlp.fc1.weight (4C, C) | transponer |
| k_stage_i_block_j_pointwise_conv_2 kernel (4C, C) | ...blocks.j.mlp.fc2.weight (C, 4C) | transponer |
| k_stage_i_block_j_layer_scale gamma (C,) | ...blocks.j.gamma | ninguna |
| LayerNormalization final del tronco (eps 1e-6, sobre el mapa) | backbone.head.norm (LayerNorm2d) | ninguna |
| Dense 128 (cabeza, solo .keras) kernel (768, 128) | fc1.weight (128, 768) | transponer |
| LayerNormalization de la cabeza (eps 1e-3) | ln.weight, ln.bias | ninguna |
| Dense 1 kernel (128, 1) | fc2.weight (1, 128) | transponer |

Los relleno "same" del depthwise 7 x 7 (3 por lado) y "valid" de stem y downsampling coinciden con timm.
GELU exacta (erf) en ambos. Los `.keras` contienen además el estado del optimizador (`optimizer/vars`), que se
ignora; tras `finalize_variable_values` los pesos guardados ya son la EMA.

Uso:
    python -m src.convert.keras_convnext_to_timm --src <fichero.h5|.keras> --out <salida.pt>
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

PREFIX = "convnext_tiny"


def snake(name: str) -> str:
    """Igual que tf_keras generic_utils.to_snake_case."""
    intermediate = re.sub("(.)([A-Z][a-z0-9]+)", r"\1_\2", name)
    insecure = re.sub("([a-z])([A-Z])", r"\1_\2", intermediate).lower()
    if insecure[0] != "_":
        return insecure
    return "private" + insecure


def _generic_names(layer_cfgs, is_sequential):
    cnt, out = Counter(), []
    for lc in layer_cfgs:
        cls = lc["class_name"]
        if is_sequential and cls == "InputLayer":
            continue
        s = snake(cls)
        gen = s if cnt[s] == 0 else "%s_%d" % (s, cnt[s])
        cnt[s] += 1
        out.append((gen, lc))
    return out


def _semantic_target(name: str, inner: str | None, var_index: int):
    """Clave timm (sin prefijo 'backbone.') y transformación para un tensor del tronco."""
    p = PREFIX
    if name == p + "_stem":
        if inner == "conv2d":
            return ("stem.0.weight", "conv") if var_index == 0 else ("stem.0.bias", None)
        if inner == "layer_normalization":
            return ("stem.1.weight", None) if var_index == 0 else ("stem.1.bias", None)
    m = re.fullmatch(p + r"_downsampling_block_(\d)", name)
    if m:
        s = int(m.group(1)) + 1
        if inner == "layer_normalization":
            return ("stages.%d.downsample.0.%s" % (s, "weight" if var_index == 0 else "bias"), None)
        if inner == "conv2d":
            return ("stages.%d.downsample.1.%s" % (s, "weight" if var_index == 0 else "bias"),
                    "conv" if var_index == 0 else None)
    m = re.fullmatch(p + r"_stage_(\d)_block_(\d+)_(depthwise_conv|layernorm|pointwise_conv_1|pointwise_conv_2|layer_scale)",
                     name)
    if m:
        i, j, kind = int(m.group(1)), int(m.group(2)), m.group(3)
        base = "stages.%d.blocks.%d." % (i, j)
        wb = "weight" if var_index == 0 else "bias"
        if kind == "depthwise_conv":
            return base + "conv_dw." + wb, ("conv" if var_index == 0 else None)
        if kind == "layernorm":
            return base + "norm." + wb, None
        if kind == "pointwise_conv_1":
            return base + "mlp.fc1." + wb, ("dense" if var_index == 0 else None)
        if kind == "pointwise_conv_2":
            return base + "mlp.fc2." + wb, ("dense" if var_index == 0 else None)
        if kind == "layer_scale":
            return base + "gamma", None
    return None


def _transform(arr: np.ndarray, how):
    if how == "conv":       # HWIO -> OIHW (depthwise: H W 1 C -> C 1 H W)
        return np.ascontiguousarray(arr.transpose(3, 2, 0, 1))
    if how == "dense":
        return np.ascontiguousarray(arr.T)
    return np.ascontiguousarray(arr)


# ------------------------------------------------------------------ .keras (saving_lib)
def read_keras_zip(path):
    import h5py
    z = zipfile.ZipFile(path)
    cfg = json.loads(z.read("config.json"))
    meta = json.loads(z.read("metadata.json")) if "metadata.json" in z.namelist() else {}
    f = h5py.File(z.open("model.weights.h5"), "r")
    top = cfg["config"]["layers"]
    sd, mapping = {}, []
    top_named = _generic_names(top, is_sequential=(cfg["class_name"] == "Sequential"))
    dense_seen = 0
    for gen, lc in top_named:
        cls = lc["class_name"]
        if cls == "Functional":
            fl = lc["config"]["layers"]
            final_ln = [l["config"]["name"] for l in fl if l["class_name"] == "LayerNormalization"][-1]
            for g2, l2 in _generic_names(fl, is_sequential=False):
                grp = "layers/%s/layers/%s" % (gen, g2)
                nm = l2["config"]["name"]
                if l2["class_name"] == "Sequential":
                    for g3, l3 in _generic_names(l2["config"]["layers"], is_sequential=True):
                        key3 = grp + "/layers/%s/vars" % g3
                        if key3 not in f:
                            continue
                        inner = "conv2d" if l3["class_name"] == "Conv2D" else "layer_normalization"
                        for vi in range(len(f[key3])):
                            tgt = _semantic_target(nm, inner, vi)
                            if tgt is None:
                                raise KeyError("sin correspondencia: %s/%s" % (nm, l3["config"]["name"]))
                            sd["backbone." + tgt[0]] = _transform(f[key3][str(vi)][()], tgt[1])
                            mapping.append((key3 + "/" + str(vi), nm + "/" + l3["config"]["name"], "backbone." + tgt[0]))
                    continue
                key = grp + "/vars"
                if key not in f or len(f[key]) == 0:
                    continue
                for vi in range(len(f[key])):
                    if l2["class_name"] == "LayerNormalization" and nm == final_ln:
                        tgt = ("head.norm.weight" if vi == 0 else "head.norm.bias", None)
                    else:
                        tgt = _semantic_target(nm, None, vi)
                    if tgt is None:
                        raise KeyError("sin correspondencia: %s (%s)" % (nm, grp))
                    sd["backbone." + tgt[0]] = _transform(f[key][str(vi)][()], tgt[1])
                    mapping.append((key + "/" + str(vi), nm, "backbone." + tgt[0]))
        elif cls in ("Dense", "LayerNormalization"):
            key = "layers/%s/vars" % gen
            if cls == "Dense":
                dst = "fc1" if dense_seen == 0 else "fc2"
                dense_seen += 1
                sd[dst + ".weight"] = _transform(f[key]["0"][()], "dense")
                sd[dst + ".bias"] = _transform(f[key]["1"][()], None)
            else:
                dst = "ln"
                sd["ln.weight"] = _transform(f[key]["0"][()], None)
                sd["ln.bias"] = _transform(f[key]["1"][()], None)
            mapping.append((key, lc["config"]["name"], dst))
    info = {"format": "keras_v3_zip", "keras_version": meta.get("keras_version"), "n_tensors": len(sd),
            "policy": _policy_of(cfg), "mapping": mapping}
    return sd, info


def _policy_of(cfg):
    s = json.dumps(cfg)
    return "mixed_bfloat16" if "mixed_bfloat16" in s else ("float32" if '"float32"' in s else "desconocida")


# ------------------------------------------------------------------ .h5 heredado (save_weights)
def read_legacy_h5(path):
    import h5py
    f = h5py.File(path, "r")
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in f.attrs["layer_names"]]
    ln_names = [n for n in names if re.fullmatch(r"layer_normalization(_\d+)?", n)]
    final_ln = ln_names[-1] if ln_names else None
    sd, mapping = {}, []
    for n in names:
        g = f[n]
        wn = [w.decode() if isinstance(w, bytes) else str(w) for w in g.attrs.get("weight_names", [])]
        for w in wn:
            arr = g[w][()]
            inner_name, var = w.split("/")[-2], w.split("/")[-1].split(":")[0]
            vi = 0 if var in ("kernel", "gamma") else 1
            if n == final_ln:
                tgt = ("head.norm.weight" if vi == 0 else "head.norm.bias", None)
            elif n == PREFIX + "_stem" or n.startswith(PREFIX + "_downsampling_block_"):
                inner = "layer_normalization" if "layernorm" in inner_name else "conv2d"
                tgt = _semantic_target(n, inner, vi)
            else:
                tgt = _semantic_target(n, None, vi)
            if tgt is None:
                raise KeyError("sin correspondencia: %s/%s" % (n, w))
            sd["backbone." + tgt[0]] = _transform(arr, tgt[1])
            mapping.append((n + "/" + w, n, "backbone." + tgt[0]))
    info = {"format": "legacy_h5", "keras_version": str(f.attrs.get("keras_version", b"")),
            "n_tensors": len(sd), "mapping": mapping}
    return sd, info


def convert(path, as_torch=True):
    path = str(path)
    if zipfile.is_zipfile(path):
        sd, info = read_keras_zip(path)
    else:
        sd, info = read_legacy_h5(path)
    if as_torch:
        import torch
        sd = {k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in sd.items()}
    return sd, info


def check_against_timm(sd, require_head=True):
    """Compara claves y formas con Paper1ConvNeXt; devuelve (faltan, sobran, formas_distintas)."""
    from src.baselines.paper1_model import Paper1ConvNeXt
    ref = Paper1ConvNeXt(pretrained=False).state_dict()
    keys = set(ref) if require_head else {k for k in ref if k.startswith("backbone.")}
    missing = sorted(keys - set(sd))
    extra = sorted(set(sd) - set(ref))
    bad = sorted(k for k in set(sd) & set(ref) if tuple(ref[k].shape) != tuple(sd[k].shape))
    return missing, extra, bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch
    sd, info = convert(a.src)
    missing, extra, bad = check_against_timm(sd, require_head=info["format"] != "legacy_h5")
    if missing or extra or bad:
        raise SystemExit("conversión incompleta: faltan %s, sobran %s, formas %s" % (missing[:5], extra[:5], bad[:5]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sd, "info": {k: v for k, v in info.items() if k != "mapping"}, "src": a.src}, a.out)
    print("ok: %d tensores (%s) -> %s" % (info["n_tensors"], info["format"], a.out))


if __name__ == "__main__":
    main()
