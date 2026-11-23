"""Diagnóstico de la firma (13-11-2026). Compara la AUC de clase desde la firma en bimcv de ENTRENAMIENTO
(las imágenes que ve la penalización; 647 negativas y 2 000 positivas de la caché) con la de bimcv de VALIDACIÓN
(lo que mide V1). Misma sonda que V1, sin remuestreo. Uso: python3 scripts/diag/r2_dentro_fuera.py CK1 [CK2 ...]
Salida: logs/diag_r2/dentro_fuera.json (se añade cada punto de control)."""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/diag')
import r2_lib as L
from src.eval.probes import oof_proba, _auc
torch.set_num_threads(2)
tm, TI, TM = L.load_split('train')
r = tm.source_idx.values == L.BIMCV
TI, TM, tm = TI[torch.from_numpy(r)], TM[torch.from_numpy(r)], tm[r].reset_index(drop=True)
y, g = tm.label.values.astype(int), tm.patient_id.values.astype(str)
path = 'logs/diag_r2/dentro_fuera.json'
res = json.load(open(path)) if os.path.exists(path) else {}
for name in sys.argv[1:]:
    ck = torch.load(os.path.join(L.R2, 'runs', name + '.pt'), weights_only=False)
    enc = L.AcquisitionEncoder(); enc.load_state_dict(ck['enc'])
    A = L.encode(enc, TI, TM)
    a_tr = _auc(y, oof_proba(A, y, g, 'logreg', 5, 42))
    ev = ck['st']['evals']
    last = ev[max(ev, key=int)] if ev else {}
    res[name] = {'auc_bimcv_train': round(a_tr, 4), 'auc_bimcv_val': last.get('v1_logreg', [None])[0],
                 'paso': ck['st']['step']}
    print(name, res[name], flush=True)
json.dump(res, open(path, 'w'), indent=1)
