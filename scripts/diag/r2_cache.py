"""Diagnóstico de la firma (13-11-2026). Construye, por tandas reanudables, una caché compacta de las
imágenes (uint8, 224 x 224) y de la unión de las máscaras del profesor (pulmones o corazón; vacía si la calidad es
0, como hace `CSDNetTrainer.masks_for_quality`) para los diagnósticos de la firma en CPU.

Uso: python3 scripts/diag/r2_cache.py {train|val} [segundos]
Muestra fija (semilla 0): entrenamiento, por celda fuente x clase, como mucho CAP_TRAIN imágenes; validación,
todo bimcv de val (257 negativas y 3 150 positivas) y como mucho CAP_VAL por celda en el resto de fuentes.
Salida en /tmp/r2: <split>_meta.csv y <split>_NNN.npz (img, mk, q). Nunca toca el test.
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from multiprocessing import Pool
OUT = os.environ.get('R2_DIR', '/tmp/r2')
CAP_TRAIN = {('bimcv', 0): 647, ('bimcv', 1): 2000, ('cohen', 0): 219, ('cohen', 1): 246, ('actmed', 0): 78,
             ('actmed', 1): 23, ('fig1', 1): 21, ('rsna', 0): 1500, ('stonybrook', 1): 1500, ('sirm', 1): 804,
             ('ricord', 1): 834}
CAP_VAL = 300
CHUNK = 400


def meta(split):
    from src.data.manifest import COVIDX_SOURCES
    p = os.path.join(OUT, '%s_meta.csv' % split)
    if os.path.exists(p):
        return pd.read_csv(p, dtype={'patient_id': str})
    m = pd.read_csv('data/manifests/covidx_cxr4_clean.csv', low_memory=False, dtype={'patient_id': str})
    assert split in ('train', 'val')
    m = m[m.split_clean == split].copy()
    m['image_id'] = m.image_path.map(lambda s: s.split('/')[-1])
    m['source_idx'] = m.source.map(COVIDX_SOURCES.index)
    parts = []
    for (s, c), g in m.groupby(['source', 'label']):
        cap = CAP_TRAIN.get((s, int(c)), 0) if split == 'train' else (len(g) if s == 'bimcv' else CAP_VAL)
        if cap > 0:
            parts.append(g.sample(min(len(g), cap), random_state=0))
    df = pd.concat(parts).reset_index(drop=True)
    ci = pd.read_csv('data/cache_m/covidx_cxr4/homogeneo_224/index.csv')
    mi = pd.read_csv('data/masks/covidx_cxr4/homogeneo_224/index.csv')
    ci = ci[ci.status == 'ok'][['image_id', 'rel_path', 'orig_h', 'orig_w', 'trim_frac', 'crop_side', 'orig_aspect']]
    mi = mi[mi.status == 'ok'][['image_id', 'path', 'quality', 'lung_frac', 'heart_frac', 'box_y0', 'box_y1', 'box_x0', 'box_x1']]
    df = df.merge(ci, on='image_id', how='inner').merge(mi, on='image_id', how='inner')
    df = df.sample(frac=1.0, random_state=1).reset_index(drop=True)  # orden aleatorio: cada tanda mezcla celdas
    df[['image_id', 'source', 'source_idx', 'label', 'patient_id', 'rel_path', 'path', 'quality', 'lung_frac',
        'heart_frac', 'box_y0', 'box_y1', 'box_x0', 'box_x1', 'orig_h', 'orig_w', 'trim_frac', 'crop_side',
        'orig_aspect']].to_csv(p, index=False)
    return pd.read_csv(p, dtype={'patient_id': str})


def load_one(a):
    from PIL import Image
    from src.data.homogeneo_masks import decode_code
    rel, mp, q = a
    with Image.open('data/cache_m/covidx_cxr4/homogeneo_224/' + rel) as im:
        x = np.asarray(im, dtype=np.uint8)
    if x.ndim == 3:
        x = x[..., 0]
    with Image.open('data/masks/covidx_cxr4/homogeneo_224/' + mp) as im:
        m = decode_code(np.asarray(im)).max(0).astype(np.uint8)
    if q <= 0:
        m[:] = 0
    return x, m


if __name__ == '__main__':
    split = sys.argv[1]
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 150
    df = meta(split)
    t0 = time.time()
    n_chunks = int(np.ceil(len(df) / CHUNK))
    with Pool(2) as pool:
        for k in range(n_chunks):
            f = os.path.join(OUT, '%s_%03d.npz' % (split, k))
            if os.path.exists(f):
                continue
            if time.time() - t0 > budget:
                break
            sub = df.iloc[k * CHUNK:(k + 1) * CHUNK]
            res = pool.map(load_one, list(zip(sub.rel_path, sub.path, sub.quality)), chunksize=20)
            np.savez(f + '.tmp.npz', img=np.stack([r[0] for r in res]), mk=np.stack([r[1] for r in res]),
                     q=sub.quality.values.astype(np.float32))
            os.replace(f + '.tmp.npz', f)
            print('tanda', k, 'de', n_chunks, '%.0f s' % (time.time() - t0), flush=True)
    done = sum(os.path.exists(os.path.join(OUT, '%s_%03d.npz' % (split, k))) for k in range(n_chunks))
    print(split, 'n', len(df), 'tandas hechas', done, 'de', n_chunks)
    print(df.groupby(['source', 'label']).size().to_dict())
