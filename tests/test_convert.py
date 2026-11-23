"""Pruebas del conversor Keras -> timm (src/convert/keras_convnext_to_timm.py).

1. Estructura sintética: a partir de un state_dict aleatorio de Paper1ConvNeXt se escriben un `.keras` (zip con
   config.json y model.weights.h5 con nombres genéricos de saving_lib) y un `.h5` heredado con la misma
   disposición que los ficheros reales de tf_keras 2.16 (inspeccionados en la VM); la conversión debe devolver
   exactamente los tensores de partida.
2. Ficheros reales (se saltan si no están): el .h5 del preentreno NIH.
3. La pasada manual de la emulación bf16 sin redondeo coincide con la pasada normal del modelo.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import paper1_model as pm  # noqa: E402
from src.convert.keras_convnext_to_timm import check_against_timm, convert, snake  # noqa: E402
from src.data.paths import data_root  # noqa: E402

K = "convnext_tiny"
DEPTHS = (3, 3, 9, 3)


def _inv(key, t):
    a = t.numpy()
    if a.ndim == 4:
        return a.transpose(2, 3, 1, 0)          # OIHW -> HWIO
    if a.ndim == 2:
        return a.T
    return a


def _random_sd(seed=0):
    torch.manual_seed(seed)
    m = pm.Paper1ConvNeXt(pretrained=False)
    return {k: torch.randn_like(v) for k, v in m.state_dict().items()}


def _layer(cls, name, **cfg):
    c = {"name": name}
    c.update(cfg)
    return {"class_name": cls, "config": c}


def _functional_layers():
    """Lista de capas del Functional de ConvNeXtTiny en el orden de tf_keras, con (clase, nombre semántico)."""
    L = [_layer("InputLayer", "input_1"), _layer("Normalization", K + "_prestem_normalization"),
         _layer("Sequential", K + "_stem", layers=[_layer("InputLayer", "x"), _layer("Conv2D", K + "_stem_conv"),
                                                    _layer("LayerNormalization", K + "_stem_layernorm")])]
    for i, d in enumerate(DEPTHS):
        if i > 0:
            L.append(_layer("Sequential", K + "_downsampling_block_%d" % (i - 1),
                            layers=[_layer("InputLayer", "y"),
                                    _layer("LayerNormalization", K + "_downsampling_layernorm_%d" % (i - 1)),
                                    _layer("Conv2D", K + "_downsampling_conv_%d" % (i - 1))]))
        for j in range(d):
            b = K + "_stage_%d_block_%d" % (i, j)
            L += [_layer("Conv2D", b + "_depthwise_conv"), _layer("LayerNormalization", b + "_layernorm", epsilon=1e-6),
                  _layer("Dense", b + "_pointwise_conv_1"), _layer("Activation", b + "_gelu"),
                  _layer("Dense", b + "_pointwise_conv_2"), _layer("LayerScale", b + "_layer_scale"),
                  _layer("Activation", b + "_identity"), _layer("TFOpLambda", "tf.__operators__.add_%d" % (i * 10 + j))]
    L.append(_layer("LayerNormalization", "layer_normalization_2", epsilon=1e-6))
    return L


def _target_of(name, inner=None):
    """Clave timm (sin 'backbone.') de cada capa semántica."""
    if name == K + "_stem":
        return {"Conv2D": "stem.0", "LayerNormalization": "stem.1"}[inner]
    if name.startswith(K + "_downsampling_block_"):
        s = int(name[-1]) + 1
        return "stages.%d.downsample.%d" % (s, 0 if inner == "LayerNormalization" else 1)
    if name == "layer_normalization_2" or name == "layer_normalization":
        return "head.norm"
    i, j = int(name.split("_stage_")[1][0]), int(name.split("_block_")[1].split("_")[0])
    base = "stages.%d.blocks.%d." % (i, j)
    for suf, t in (("_depthwise_conv", "conv_dw"), ("_layernorm", "norm"), ("_pointwise_conv_1", "mlp.fc1"),
                   ("_pointwise_conv_2", "mlp.fc2"), ("_layer_scale", "gamma")):
        if name.endswith(suf):
            return base + t
    raise KeyError(name)


def _write_keras(path, sd):
    fl = _functional_layers()
    top = [_layer("InputLayer", "in"), {"class_name": "Functional", "config": {"name": K, "layers": fl}},
           _layer("GlobalAveragePooling2D", "gap"), _layer("Dense", "dense_2"),
           _layer("LayerNormalization", "layer_normalization_3", epsilon=1e-3), _layer("Dropout", "dropout_1"),
           _layer("Dense", "dense_3")]
    cfg = {"module": "keras", "class_name": "Sequential", "config": {"name": "sequential_1", "layers": top}}
    h5p = path.with_suffix(".weights.h5")
    with h5py.File(h5p, "w") as f:
        def put(grp, arrays):
            g = f.require_group(grp + "/vars")
            for i, a in enumerate(arrays):
                g.create_dataset(str(i), data=a)
        cnt = {}
        for lc in fl:
            s = snake(lc["class_name"]); n = cnt.get(s, 0); cnt[s] = n + 1
            gen = s if n == 0 else "%s_%d" % (s, n)
            name = lc["config"]["name"]
            if lc["class_name"] == "Sequential":
                c2 = {}
                for l2 in lc["config"]["layers"]:
                    if l2["class_name"] == "InputLayer":
                        continue
                    s2 = snake(l2["class_name"]); n2 = c2.get(s2, 0); c2[s2] = n2 + 1
                    g2 = s2 if n2 == 0 else "%s_%d" % (s2, n2)
                    t = "backbone." + _target_of(name, l2["class_name"])
                    put("layers/functional/layers/%s/layers/%s" % (gen, g2),
                        [_inv(t, sd[t + ".weight"]), _inv(t, sd[t + ".bias"])])
                continue
            if lc["class_name"] in ("InputLayer", "Normalization", "Activation", "TFOpLambda"):
                f.require_group("layers/functional/layers/%s/vars" % gen)
                continue
            t = "backbone." + _target_of(name)
            if lc["class_name"] == "LayerScale":
                put("layers/functional/layers/%s" % gen, [sd[t].numpy()])
            else:
                put("layers/functional/layers/%s" % gen, [_inv(t, sd[t + ".weight"]), _inv(t, sd[t + ".bias"])])
        put("layers/dense", [sd["fc1.weight"].numpy().T, sd["fc1.bias"].numpy()])
        put("layers/layer_normalization", [sd["ln.weight"].numpy(), sd["ln.bias"].numpy()])
        put("layers/dense_1", [sd["fc2.weight"].numpy().T, sd["fc2.bias"].numpy()])
        f.require_group("optimizer/vars").create_dataset("0", data=np.zeros(3))
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("metadata.json", json.dumps({"keras_version": "2.16.0"}))
        z.writestr("config.json", json.dumps(cfg))
        z.write(h5p, "model.weights.h5")
    return path


def _write_legacy_h5(path, sd):
    fl = _functional_layers()
    fl[-1]["config"]["name"] = "layer_normalization"
    with h5py.File(path, "w") as f:
        names = [l["config"]["name"] for l in fl]
        f.attrs["layer_names"] = np.array([n.encode() for n in names])
        f.attrs["keras_version"] = b"2.16.0"
        f.attrs["backend"] = b"tensorflow"
        for lc in fl:
            n = lc["config"]["name"]
            g = f.create_group(n)
            wn = []
            if lc["class_name"] == "Sequential":
                for l2 in lc["config"]["layers"]:
                    if l2["class_name"] == "InputLayer":
                        continue
                    t = "backbone." + _target_of(n, l2["class_name"])
                    a, b = ("kernel", "bias") if l2["class_name"] == "Conv2D" else ("gamma", "beta")
                    for v, arr in ((a, _inv(t, sd[t + ".weight"])), (b, sd[t + ".bias"].numpy())):
                        key = "%s/%s:0" % (l2["config"]["name"], v)
                        g.create_dataset(key, data=arr); wn.append(key)
            elif lc["class_name"] in ("Conv2D", "Dense", "LayerNormalization", "LayerScale"):
                t = "backbone." + _target_of(n)
                if lc["class_name"] == "LayerScale":
                    items = [("gamma", sd[t].numpy())]
                elif lc["class_name"] == "LayerNormalization":
                    items = [("gamma", sd[t + ".weight"].numpy()), ("beta", sd[t + ".bias"].numpy())]
                else:
                    items = [("kernel", _inv(t, sd[t + ".weight"])), ("bias", sd[t + ".bias"].numpy())]
                for v, arr in items:
                    key = "%s/%s:0" % (n, v)
                    g.create_dataset(key, data=arr); wn.append(key)
            g.attrs["weight_names"] = np.array([w.encode() for w in wn])
    return path


def test_snake_case_like_keras():
    assert snake("LayerNormalization") == "layer_normalization"
    assert snake("TFOpLambda") == "tf_op_lambda" and snake("Conv2D") == "conv2d"
    assert snake("GlobalAveragePooling2D") == "global_average_pooling2d"


def test_convert_synthetic_keras_zip(tmp_path):
    sd = _random_sd(0)
    p = _write_keras(tmp_path / "m.keras", sd)
    out, info = convert(p)
    assert info["format"] == "keras_v3_zip" and info["n_tensors"] == 186
    assert check_against_timm(out) == ([], [], [])
    for k, v in sd.items():
        assert torch.equal(out[k], v), k
    m = pm.Paper1ConvNeXt(pretrained=False)
    m.load_state_dict(out, strict=True)


def test_convert_synthetic_legacy_h5(tmp_path):
    sd = _random_sd(1)
    p = _write_legacy_h5(tmp_path / "bb.h5", sd)
    out, info = convert(p)
    assert info["format"] == "legacy_h5" and info["n_tensors"] == 180
    assert check_against_timm(out, require_head=False) == ([], [], [])
    for k, v in out.items():
        assert torch.equal(v, sd[k]), k


H5 = data_root() / "nih" / "saved" / "nih_chest_convnext_tiny_weights.h5"


@pytest.mark.skipif(not H5.exists(), reason="sin el .h5 del preentreno NIH")
def test_convert_real_nih_h5():
    out, info = convert(H5)
    assert info["n_tensors"] == 180
    assert check_against_timm(out, require_head=False) == ([], [], [])
    assert sum(v.numel() for v in out.values()) == 27820128
    g = out["backbone.stages.0.blocks.0.gamma"]
    assert g.abs().max() < 10 and g.abs().max() > 0                     # LayerScale entrenada, no la inicial 1e-6


def test_bf16_emulation_path_equals_normal_forward_without_rounding(monkeypatch):
    m = pm.Paper1ConvNeXt(pretrained=False)
    torch.manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.02 * torch.randn_like(p))
    m.eval()
    x = torch.rand(2, 3, 64, 64) * 255
    ref = m(x)
    monkeypatch.setattr(pm, "_r", lambda t: t)
    m.emulate_bf16 = True
    m.bf16_gelu_fp32 = True        # sin redondeo, la GELU por partes y la exacta coinciden salvo coma flotante
    out = m(x)
    assert torch.allclose(out, ref, atol=1e-4), (out, ref)
