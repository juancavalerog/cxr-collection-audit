"""Diagnóstico de la firma (13-11-2026). Pregunta previa: ¿de dónde sale la clase que se lee en la firma
dentro de bimcv de val? Separa la geometría de la zona visible (que depende de la máscara y por tanto del tamaño
pulmonar) y de la imagen original, del contenido de la entrada de adquisición.

Uso: python3 scripts/diag/r2_origen_clase.py CHECKPOINT [n_boot]   (CHECKPOINT en /tmp/r2/runs, sin .pt)
Todas las AUC: sonda de V1 (StandardScaler + regresión logística, 5 particiones agrupadas por paciente, semilla 42)
sobre las 3 407 imágenes bimcv de val, con IC95 por remuestreo de pacientes. Salida: logs/diag_r2/origen_<ck>.json.
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/diag')
import r2_lib as L
from src.eval.probes import oof_proba, _auc_with_ci

torch.set_num_threads(2)
name = sys.argv[1]
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
vm, VI, VM = L.load_split('val')
r = vm.source_idx.values == L.BIMCV
y, g = vm.label.values[r].astype(int), vm.patient_id.values[r].astype(str)
VIb, VMb, mb = VI[torch.from_numpy(r)], VM[torch.from_numpy(r)], vm[r].reset_index(drop=True)
out = {'n': int(r.sum()), 'n_neg': int((y == 0).sum()), 'n_pos': int(y.sum()), 'n_pacientes': int(len(set(g)))}


def auc(X, tag):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    p = oof_proba(X, y, g, 'logreg', 5, 42)
    a, lo, hi = _auc_with_ci(y, p, g, nb, 42)
    out[tag] = [round(a, 4), round(lo, 4), round(hi, 4)]
    print(tag, out[tag], flush=True)
    return p


# 1. geometría de la zona visible: fracción visible y caja excluida (de keep), sin mirar píxeles de la imagen
_, keep = L.acq_input(VIb[:1], VMb[:1])
geo = []
for i in range(0, len(VIb), 256):
    _, k = L.acq_input(VIb[i:i + 256], VMb[i:i + 256])
    k = k[:, 0]
    ex = 1 - k
    rows, cols = ex.amax(2) > 0, ex.amax(1) > 0
    H = k.shape[-1]
    ar = torch.arange(H).float()
    y0 = torch.where(rows, ar, torch.full_like(ar, H)).amin(1); y1 = torch.where(rows, ar, torch.full_like(ar, -1)).amax(1)
    x0 = torch.where(cols, ar, torch.full_like(ar, H)).amin(1); x1 = torch.where(cols, ar, torch.full_like(ar, -1)).amax(1)
    geo.append(torch.stack([k.mean((1, 2)), y0, y1, x0, x1, y1 - y0, x1 - x0], 1))
geo = torch.cat(geo).numpy()
out['fraccion_visible_media'] = float(geo[:, 0].mean())
auc(geo[:, 0], 'auc_fraccion_visible')
auc(geo, 'auc_geometria_keep')  # fracción visible y caja (7 rasgos)
mask_meta = mb[['lung_frac', 'heart_frac', 'box_y0', 'box_y1', 'box_x0', 'box_x1']].values
auc(mask_meta, 'auc_mascara_profesor')  # área pulmonar y cardiaca y caja de la máscara del profesor
img_meta = np.c_[np.log(mb.orig_h), np.log(mb.orig_w), mb.orig_aspect, mb.trim_frac, np.log(mb.crop_side)]
auc(img_meta, 'auc_imagen_original')  # tamaño, aspecto, recorte de la imagen original
auc(np.c_[geo, mask_meta, img_meta], 'auc_geometria_toda')
auc(mb.quality.values, 'auc_calidad_mascara')

# 2. contenido: estadísticos simples por canal de la entrada de adquisición en la zona visible
st = []
for i in range(0, len(VIb), 256):
    xa, k = L.acq_input(VIb[i:i + 256], VMb[i:i + 256])
    mass = k.sum((2, 3)).clamp_min(1)
    mu = (xa.abs() * k).sum((2, 3)) / mass
    sd = (((xa - (xa * k).sum((2, 3), keepdim=True) / mass[..., None, None]) ** 2) * k).sum((2, 3)) / mass
    st.append(torch.cat([mu, sd.sqrt()], 1))
st = torch.cat(st).numpy()
auc(st[:, list(range(0, 6)) + list(range(7, 13))], 'auc_estadisticos_subbandas')
auc(st[:, [6, 13]], 'auc_estadisticos_periferia_LL1')
auc(st, 'auc_estadisticos_todos')
auc(np.c_[st, geo], 'auc_estadisticos_mas_geometria')

# 3. firma del codificador entrenado solo con L_src
ck = torch.load(os.path.join(L.R2, 'runs', name + '.pt'), weights_only=False)
enc = L.AcquisitionEncoder(); enc.load_state_dict(ck['enc'])
A = L.encode(enc, VIb, VMb)
auc(A, 'auc_firma')
auc(np.c_[A, geo, mask_meta, img_meta], 'auc_firma_mas_geometria')
# residuo de la firma tras quitar la parte lineal explicada por la geometría (ajuste por mínimos cuadrados)
G = np.c_[np.ones(len(A)), geo, mask_meta, img_meta]
res = A - G @ np.linalg.lstsq(G, A, rcond=None)[0]
auc(res, 'auc_firma_residuo_sin_geometria')
# intervención: la misma firma con una caja torácica común para todas las imágenes (la caja media de bimcv val)
bx = tuple(int(v) for v in np.round(np.median(mask_meta[:, 2:6], 0)))
fb = (max(0, bx[0] - 11), min(224, bx[1] + 12), max(0, bx[2] - 11), min(224, bx[3] + 12))
out['caja_comun'] = fb
A2 = L.encode(enc, VIb, VMb, fixed_box=fb)
auc(A2, 'auc_firma_caja_comun')
# referencia: codificador con los pesos iniciales (semilla 1), sin entrenar
torch.manual_seed(1); enc0 = L.AcquisitionEncoder()
auc(L.encode(enc0, VIb, VMb), 'auc_firma_sin_entrenar')
json.dump(out, open('logs/diag_r2/origen_%s.json' % name, 'w'), indent=1)
