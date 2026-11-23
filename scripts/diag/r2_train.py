"""Diagnóstico de la firma (13-11-2026). Entrena la vía de adquisición en CPU con una penalización de
independencia candidata y mide en validación (nunca en test) lo que miden L_src y V1.

Uso: python3 scripts/diag/r2_train.py VARIANTE [segundos]   (reanudable; repetir hasta que diga "terminado")
Variantes en VARIANTS. Salida: logs/diag_r2/<variante>.json (registro por tramos de 50 pasos y evaluaciones).
Punto de control en /tmp/r2/runs/<variante>.pt.
"""
import os, sys, json, time
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/diag')
import r2_lib as L

N_STEPS = int(os.environ.get('R2_STEPS', 900))
EVAL_AT = (300, 600)
BASE = dict(mode='none', w=0.0, beta=0.0, est='insesgada', eta=1.0, adv_lr_mult=5.0, bank=0, start=0, ramp=0,
            lam_src=0.5, seed=1)
VARIANTS = {
    'sin_penalizacion': dict(),
    'sin_penalizacion_largo': dict(),  # continúa el punto de control de sin_penalizacion con R2_STEPS=2100
    'sin_penalizacion_s2': dict(seed=2),
    'sin_penalizacion_s3': dict(seed=3),
    'mmd_sesgada_w1': dict(mode='mmd', est='sesgada', w=1.0),
    'mmd_sesgada_w0.1': dict(mode='mmd', est='sesgada', w=0.1),
    'mmd_insesgada_w1': dict(mode='mmd', est='insesgada', w=1.0),
    'mmd_insesgada_norm_b0.5': dict(mode='norm', est='insesgada', beta=0.5),
    'mmd_insesgada_norm_b1': dict(mode='norm', est='insesgada', beta=1.0),
    'mmd_sesgada_norm_b0.5': dict(mode='norm', est='sesgada', beta=0.5),
    'adversario_eta1': dict(mode='adv', eta=1.0),
    'banco256_insesgada_w1': dict(mode='mmd', est='insesgada', w=1.0, bank=256),
    'banco256_norm_b0.5': dict(mode='norm', est='insesgada', beta=0.5, bank=256),
    'banco256_w10_tope0.5': dict(mode='tope', est='insesgada', w=10.0, beta=0.5, bank=256),
    'banco256_w10_tope0.5_s2': dict(mode='tope', est='insesgada', w=10.0, beta=0.5, bank=256, seed=2),
    'calendario_banco256_w10_tope0.5': dict(mode='tope', est='insesgada', w=10.0, beta=0.5, bank=256, start=450, ramp=150),
    'calendario_norm_b0.5': dict(mode='norm', est='insesgada', beta=0.5, start=450, ramp=150),
    'calendario_banco_norm_b0.5': dict(mode='norm', est='insesgada', beta=0.5, bank=256, start=450, ramp=150),
}


def ramp_factor(cfg, step):
    if step < cfg['start']:
        return 0.0
    if cfg['ramp'] <= 0:
        return 1.0
    return min(1.0, (step - cfg['start'] + 1) / cfg['ramp'])


def main():
    name = sys.argv[1]
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 150
    cfg = dict(BASE, **VARIANTS[name])
    torch.set_num_threads(2)
    os.makedirs(os.path.join(L.R2, 'runs'), exist_ok=True)
    os.makedirs('logs/diag_r2', exist_ok=True)
    ck = os.path.join(L.R2, 'runs', name + '.pt')
    jpath = 'logs/diag_r2/%s.json' % name
    t0 = time.time()
    meta, IMG, MK = L.load_split('train')
    stream = L.BatchStream(meta)
    S = torch.tensor(meta.source_idx.values).long()
    Y = torch.tensor(meta.label.values).long()
    torch.manual_seed(cfg['seed'])
    enc = L.AcquisitionEncoder()
    head = torch.nn.Linear(64, L.K)
    adv = L.ClassAdversary() if cfg['mode'] == 'adv' else None
    mods = [enc, head]
    dec = [p for m in mods for p in m.parameters() if p.ndim > 1]
    nod = [p for m in mods for p in m.parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{'params': dec, 'weight_decay': 0.05}, {'params': nod, 'weight_decay': 0.0}], lr=1e-3)
    opt_d = torch.optim.AdamW(adv.parameters(), lr=1e-3 * cfg['adv_lr_mult'], weight_decay=0.0) if adv else None
    bank = {}
    st = {'step': 0, 'log': [], 'evals': {}, 'cfg': cfg}
    if os.path.exists(ck):
        c = torch.load(ck, weights_only=False)
        enc.load_state_dict(c['enc']); head.load_state_dict(c['head']); opt.load_state_dict(c['opt'])
        if adv:
            adv.load_state_dict(c['adv']); opt_d.load_state_dict(c['opt_d'])
        bank, st = c['bank'], c['st']
        torch.set_rng_state(c['rng'])
    win = {'ls': [], 'pen': [], 'sd': [], 'cnt': np.zeros(L.K), 'H': [], 'ratio': []}

    def save():
        c = {'enc': enc.state_dict(), 'head': head.state_dict(), 'opt': opt.state_dict(), 'bank': bank, 'st': st,
             'rng': torch.get_rng_state()}
        if adv:
            c['adv'], c['opt_d'] = adv.state_dict(), opt_d.state_dict()
        torch.save(c, ck + '.tmp'); os.replace(ck + '.tmp', ck)
        json.dump(st, open(jpath + '.tmp', 'w'), indent=1, default=float); os.replace(jpath + '.tmp', jpath)

    def evaluate(full):
        vm, VI, VM = L.load_split('val')
        A, m = L.source_metrics(enc, head, VI, VM, vm)
        v1 = L.v1_like(A, vm, n_boot=1000 if full else 0, kinds=('logreg', 'mlp') if full else ('logreg',))
        m.update({'v1_' + k: v for k, v in v1.items()})
        if full:
            np.save(os.path.join(L.R2, 'runs', name + '_val_sig.npy'), A)
        return m

    while st['step'] < N_STEPS and time.time() - t0 < budget:
        step = st['step']
        b = stream.batch(step)
        xa, k = L.acq_input(IMG[b], MK[b])
        s, y = S[b], Y[b]
        a = enc(xa, k)
        ls = F.cross_entropy(head(a), s)
        r = ramp_factor(cfg, step)
        loss = cfg['lam_src'] * ls
        pen = torch.zeros(())
        ratio = None
        est = L.mmd2_unbiased if cfg['est'] == 'insesgada' else L.mmd2_biased
        if cfg['mode'] in ('mmd', 'norm', 'tope') and r > 0:
            pen = L.per_source_penalty(a, s, y, est, bank=bank if cfg['bank'] else None)
            if cfg['mode'] == 'mmd':
                loss = loss + r * cfg['w'] * pen
                if step % 25 == 0 and pen.requires_grad:
                    gs = torch.autograd.grad(cfg['lam_src'] * ls, a, retain_graph=True)[0]
                    gi = torch.autograd.grad(cfg['w'] * pen, a, retain_graph=True)[0]
                    ratio = float(gi.norm() / gs.norm().clamp_min(1e-12))
            elif cfg['mode'] == 'tope' and pen.requires_grad:
                # recorte relativo: el gradiente de w*L_ind sobre la firma no supera beta veces el de lam_src*L_src
                gs = torch.autograd.grad(cfg['lam_src'] * ls, a, retain_graph=True)[0]
                gi = torch.autograd.grad(cfg['w'] * pen.clamp_min(0.0), a, retain_graph=True)[0]
                c = torch.clamp(cfg['beta'] * gs.norm() / gi.norm().clamp_min(1e-12), max=1.0)
                loss = loss + r * (a * (c * gi).detach()).sum()
                ratio = float(r * c * gi.norm() / gs.norm().clamp_min(1e-12))
            elif pen.requires_grad:
                gs = torch.autograd.grad(cfg['lam_src'] * ls, a, retain_graph=True)[0]
                gi = torch.autograd.grad(pen, a, retain_graph=True)[0]
                c = cfg['beta'] * gs.norm() / gi.norm().clamp_min(1e-12)
                loss = loss + r * (a * (c * gi).detach()).sum()
                ratio = float(r * cfg['beta'])
        if cfg['mode'] == 'adv' and r > 0:
            mm = torch.zeros(len(s), dtype=torch.bool)
            for src in L.MIXED:
                mm |= (s == src)
            oh = torch.stack([(s[mm] == src).float() for src in L.MIXED], 1)
            ys, ss = y[mm].float(), s[mm]
            wts = torch.zeros(len(ys))
            for src in L.MIXED:
                for c_ in (0, 1):
                    q = (ss == src) & (ys == c_)
                    if q.any():
                        wts[q] = 1.0 / float(q.sum())
            wts = wts / wts.sum()
            dl = adv(L._GRL.apply(a[mm], cfg['eta'] * r), oh)
            pen = (F.binary_cross_entropy_with_logits(dl, ys, reduction='none') * wts).sum()
            loss = loss + pen
            opt_d.zero_grad()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters() and [p for m in (enc, head) for p in m.parameters()], 1.0)
        opt.step()
        if opt_d is not None:
            opt_d.step()
        if cfg['bank']:
            with torch.no_grad():
                ad = a.detach()
                for src in L.MIXED:
                    for c_ in (0, 1):
                        q = (s == src) & (y == c_)
                        if q.any():
                            old = bank.get((src, c_), ad[:0])
                            bank[(src, c_)] = torch.cat([ad[q], old], 0)[:cfg['bank']]
        cnt = np.bincount(s.numpy(), minlength=L.K)
        win['ls'].append(float(ls.detach())); win['pen'].append(float(pen.detach())); win['sd'].append(float(a.detach().std(0).mean()))
        win['cnt'] += cnt; win['H'].append(L.entropy_counts(cnt))
        if ratio is not None:
            win['ratio'].append(ratio)
        st['step'] += 1
        if st['step'] % 50 == 0:
            rec = {'paso': st['step'], 'L_src': float(np.mean(win['ls'])), 'H_lote_medio': float(np.mean(win['H'])),
                   'H_composicion': L.entropy_counts(win['cnt']), 'pen': float(np.mean(win['pen'])),
                   'desv_firma_lote': float(np.mean(win['sd'])),
                   'razon_grad': float(np.mean(win['ratio'])) if win['ratio'] else None}
            st['log'].append(rec)
            print(json.dumps(rec), flush=True)
            win = {'ls': [], 'pen': [], 'sd': [], 'cnt': np.zeros(L.K), 'H': [], 'ratio': []}
        if st['step'] in EVAL_AT and str(st['step']) not in st['evals']:
            st['evals'][str(st['step'])] = evaluate(False)
            print('eval', st['step'], json.dumps(st['evals'][str(st['step'])], default=float), flush=True)
            save()
    if st['step'] >= N_STEPS and str(N_STEPS) not in st['evals']:
        save()
        st['evals'][str(N_STEPS)] = evaluate(True)
        print('eval', N_STEPS, json.dumps(st['evals'][str(N_STEPS)], default=float), flush=True)
    save()
    print(name, 'paso', st['step'], 'terminado' if str(N_STEPS) in st['evals'] else 'sigue', '%.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
