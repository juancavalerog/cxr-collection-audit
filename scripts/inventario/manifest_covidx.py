"""Tabla clase por fuente por particion de COVIDx CXR-4.
Formato real de los .txt (4 columnas separadas por espacio): patient_id filename label source.
train = dataset.txt menos (val.txt U test.txt), emparejando por nombre de fichero."""
import os, sys, pandas as pd
D = os.path.expanduser('~/mnt/notebooks/covid/data')
OUT = os.path.expanduser('~/mnt/paper_new/data/manifests')
cols = ['patient', 'filename', 'label', 'source']
rd = lambda f: pd.read_csv(os.path.join(D, f), sep=' ', header=None, names=cols, dtype=str)
full, val, test = rd('dataset.txt'), rd('val.txt'), rd('test.txt')
log = []
for n, d in [('dataset', full), ('val', val), ('test', test)]:
    log.append(f'{n}: filas={len(d)} ficheros_unicos={d.filename.nunique()} pacientes_unicos={d.patient.nunique()}')
vt = set(val.filename) | set(test.filename)
log.append(f'val&test interseccion ficheros={len(set(val.filename)&set(test.filename))}')
log.append(f'val no en dataset={len(set(val.filename)-set(full.filename))} test no en dataset={len(set(test.filename)-set(full.filename))}')
train = full[~full.filename.isin(vt)].copy()
log.append(f'train: filas={len(train)}')
# coherencia de etiqueta/fuente entre dataset.txt y val/test
m = pd.concat([val, test]).merge(full, on='filename', suffixes=('', '_full'))
for c in ['patient', 'label', 'source']:
    log.append(f'discrepancias {c} val/test vs dataset.txt: {(m[c]!=m[c+"_full"]).sum()}')
# solapamiento de pacientes entre particiones
P = {k: set(v.patient) for k, v in [('train', train), ('val', val), ('test', test)]}
for a, b in [('train','val'),('train','test'),('val','test')]:
    log.append(f'pacientes compartidos {a}-{b}: {len(P[a]&P[b])}')
rows = []
for s, d in [('train', train), ('val', val), ('test', test)]:
    g = d.groupby(['source', 'label']).agg(n_images=('filename', 'size'), n_patients=('patient', 'nunique')).reset_index()
    g.insert(0, 'split', s); rows.append(g)
    for lab in ['positive', 'negative']:
        log.append(f'{s} {lab}: {(d.label==lab).sum()}')
tab = pd.concat(rows)
tab.to_csv(os.path.join(OUT, 'covidx_cxr4_clase_fuente.csv'), index=False)
# imagenes en disco
files = set(os.listdir(os.path.join(D, 'dataset')))
log.append(f'ficheros en data/dataset/: {len(files)}; de dataset.txt ausentes en disco: {len(set(full.filename)-files)}; en disco sin fila: {len(files-set(full.filename))}')
for sub in ['val', 'test']:
    log.append(f'ficheros en data/{sub}/: {len(os.listdir(os.path.join(D, sub)))}')
print('\n'.join(log)); print(tab.to_string(index=False))
