"""Diagnóstico de la firma (13-11-2026). ¿Es igual de fuerte la asociación clase-adquisición dentro de
bimcv en entrenamiento y en validación? Sonda de V1 (sin remuestreo) sobre (1) los 14 estadísticos simples de la
entrada de adquisición en la zona visible (caché de r2_cache.py) y (2) los metadatos de tamaño, aspecto y recorte de
la imagen original y de la máscara del profesor (todas las imágenes bimcv de cada partición, desde los índices).
Uso: python3 scripts/diag/r2_asociacion_particiones.py. Salida: logs/diag_r2/asociacion_particiones.json"""
import sys, json
import numpy as np, pandas as pd, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/diag')
import r2_lib as L
from src.eval.probes import oof_proba, _auc
torch.set_num_threads(2)
out = {}


def stats(I, M):
    st = []
    for i in range(0, len(I), 256):
        xa, k = L.acq_input(I[i:i + 256], M[i:i + 256])
        mass = k.sum((2, 3)).clamp_min(1)
        mu = (xa.abs() * k).sum((2, 3)) / mass
        sd = (((xa - (xa * k).sum((2, 3), keepdim=True) / mass[..., None, None]) ** 2) * k).sum((2, 3)) / mass
        st.append(torch.cat([mu, sd.sqrt()], 1))
    return torch.cat(st).numpy()


for split in ('train', 'val'):
    m, I, M = L.load_split(split)
    r = m.source_idx.values == L.BIMCV
    mm = m[r]
    y, g = mm.label.values.astype(int), mm.patient_id.values.astype(str)
    X = stats(I[torch.from_numpy(r)], M[torch.from_numpy(r)])
    out['estadisticos_%s' % split] = round(_auc(y, oof_proba(X, y, g, 'logreg', 5, 42)), 4)
    out['n_%s_cache' % split] = [int((y == 0).sum()), int(y.sum())]
man = pd.read_csv('data/manifests/covidx_cxr4_clean.csv', low_memory=False, dtype={'patient_id': str})
man['image_id'] = man.image_path.map(lambda s: s.split('/')[-1])
ci = pd.read_csv('data/cache_m/covidx_cxr4/homogeneo_224/index.csv')
mi = pd.read_csv('data/masks/covidx_cxr4/homogeneo_224/index.csv')
d = man[man.source == 'bimcv'].merge(ci[ci.status == 'ok'], on='image_id').merge(mi[mi.status == 'ok'], on='image_id')
for split in ('train', 'val'):
    s = d[d.split_clean == split]
    if split == 'train':  # todas las negativas y 5 000 positivas al azar, para que la sonda quepa en el tiempo
        s = pd.concat([s[s.label == 0], s[s.label == 1].sample(5000, random_state=0)])
    y, g = s.label.values.astype(int), s.patient_id.values.astype(str)
    Xi = np.c_[np.log(s.orig_h), np.log(s.orig_w), s.orig_aspect, s.trim_frac, np.log(s.crop_side)]
    Xm = np.nan_to_num(s[['lung_frac', 'heart_frac', 'box_y0', 'box_y1', 'box_x0', 'box_x1']].values.astype(float), nan=-1.0)
    out['imagen_original_%s' % split] = round(_auc(y, oof_proba(Xi, y, g, 'logreg', 5, 42)), 4)
    out['mascara_profesor_%s' % split] = round(_auc(y, oof_proba(Xm, y, g, 'logreg', 5, 42)), 4)
    out['n_%s_meta' % split] = [int((y == 0).sum()), int(y.sum())]
    out['aspecto_mediano_%s' % split] = [float(s[s.label == c].orig_aspect.median()) for c in (0, 1)]
print(json.dumps(out, indent=1))
json.dump(out, open('logs/diag_r2/asociacion_particiones.json', 'w'), indent=1)
