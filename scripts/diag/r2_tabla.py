"""Diagnóstico de la firma (13-11-2026). Tabla comparativa de las variantes de r2_train.py a partir de
logs/diag_r2/<variante>.json. L_src y H: media de los tramos de 50 pasos entre los pasos 751 y 900; V1: AUC de clase
desde la firma en bimcv de val (regresión logística y MLP, IC95 por remuestreo de pacientes) en el paso final; trayectoria
de V1 (logística) en los pasos 300, 600 y final. Uso: python3 scripts/diag/r2_tabla.py > logs/diag_r2/tabla.txt"""
import json, glob, os
import numpy as np
ORDER = ['sin_penalizacion', 'sin_penalizacion_s2', 'sin_penalizacion_s3', 'sin_penalizacion_largo', 'mmd_sesgada_w1',
         'mmd_insesgada_w1', 'mmd_insesgada_norm_b0.5', 'adversario_eta1', 'banco256_insesgada_w1',
         'banco256_w10_tope0.5', 'banco256_w10_tope0.5_s2', 'calendario_banco256_w10_tope0.5']
print('| variante | pasos | L_src final | H composición | L_src - H | exact. bal. fuente val | desv. firma val | V1 log. (IC95) | V1 MLP (IC95) | V1 log. 300/600/final | razón grad. media final |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for n in ORDER:
    p = 'logs/diag_r2/%s.json' % n
    if not os.path.exists(p):
        continue
    st = json.load(open(p))
    N = st['step']
    lg = [r for r in st['log'] if N - 150 < r['paso'] <= N]
    ls, h = np.mean([r['L_src'] for r in lg]), np.mean([r['H_composicion'] for r in lg])
    rat = [r['razon_grad'] for r in lg if r.get('razon_grad') is not None]
    ev = st['evals']; fin = ev[str(N)]
    f = lambda t: '%.3f (%.3f a %.3f)' % tuple(t)
    traj = '/'.join('%.3f' % ev[k]['v1_logreg'][0] for k in sorted(ev, key=int) if k in ('300', '600', str(N)))
    print('| %s | %d | %.3f | %.3f | %+.3f | %.3f | %.3f | %s | %s | %s | %s |' % (
        n, N, ls, h, ls - h, fin['acc_fuente_bal'], fin['desv_firma'], f(fin['v1_logreg']), f(fin['v1_mlp']), traj,
        '%.2f' % np.mean(rat) if rat else 'sin dato'))
