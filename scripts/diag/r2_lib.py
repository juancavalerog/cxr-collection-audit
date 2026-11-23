"""Diagnóstico de la firma (13-11-2026). Biblioteca común de los diagnósticos de la firma en CPU.

Entrena solo la vía de adquisición (AcquisitionInput + AcquisitionEncoder + cabeza de fuente lineal) sobre la caché
de `r2_cache.py`, con lotes de 64 que tienen exactamente la composición por celda fuente x clase del muestreador real
(`src.model.swap.TableGuidedBatchSampler` sobre las 67 932 imágenes de entrenamiento, semilla 42, cuotas 2 por fuente
y 4 por clase en cada fuente mixta): cada índice del lote real se sustituye por una imagen al azar de la misma celda
en la caché. AdamW (lr 1e-3, wd 0.05 sin decaimiento en normas y sesgos), recorte de norma 1.0, sin aumento de datos
y con tasa constante (el bucle real usa calentamiento del 5 % y coseno; ver docs/10_correccion_independencia.md).
"""
import os, sys, glob, json, math, time
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, '.')
from src.model.layers import AcquisitionInput, AcquisitionEncoder
from src.data.manifest import COVIDX_SOURCES
R2 = os.environ.get('R2_DIR', '/tmp/r2')
K = len(COVIDX_SOURCES)
MIXED = [COVIDX_SOURCES.index(s) for s in ('actmed', 'bimcv', 'cohen')]
BIMCV = COVIDX_SOURCES.index('bimcv')
STEPS_PER_EPOCH = 1062


def load_split(split):
    meta = pd.read_csv(os.path.join(R2, '%s_meta.csv' % split), dtype={'patient_id': str})
    fs = sorted(glob.glob(os.path.join(R2, '%s_[0-9][0-9][0-9].npz' % split)))
    img, mk = [], []
    for f in fs:
        z = np.load(f)
        img.append(z['img']); mk.append(z['mk'])
    img, mk = np.concatenate(img), np.concatenate(mk)
    assert len(img) == len(meta), 'caché incompleta: ejecute scripts/diag/r2_cache.py %s' % split
    return meta, torch.from_numpy(img), torch.from_numpy(mk)


_ACQ = AcquisitionInput()


def acq_input(img_u8, mk_u8, fixed_box=None):
    """Entrada de adquisición, ecuación (16), desde la imagen uint8 y la unión de máscaras.
    fixed_box=(y0, y1, x0, x1) en píxeles de 224 sustituye la máscara por una caja común (intervención)."""
    g = img_u8.float().div(255.0)[:, None]
    m = mk_u8.float()[:, None]
    if fixed_box is not None:
        m = torch.zeros_like(m)
        y0, y1, x0, x1 = fixed_box
        m[..., y0:y1, x0:x1] = 1.0
    with torch.no_grad():
        return _ACQ(g, m)


def full_train_cells():
    m = pd.read_csv('data/manifests/covidx_cxr4_clean.csv', low_memory=False, usecols=['source', 'label', 'split_clean'])
    m = m[m.split_clean == 'train']
    return m.label.astype(int).to_numpy(), m.source.map(COVIDX_SOURCES.index).to_numpy()


class BatchStream:
    """Lotes con la composición del muestreador real, trasladados a la caché por celda."""

    def __init__(self, meta, seed=42, batch_size=64):
        from src.model.swap import TableGuidedBatchSampler
        self.y_full, self.s_full = full_train_cells()
        self.smp = TableGuidedBatchSampler(self.y_full, self.s_full, batch_size, STEPS_PER_EPOCH, 2, 4, seed=seed)
        self.cell = {}
        for i, (s, y) in enumerate(zip(meta.source_idx.values, meta.label.values)):
            self.cell.setdefault((int(s), int(y)), []).append(i)
        self.cell = {k: np.asarray(v) for k, v in self.cell.items()}
        self.seed = seed
        self._epoch, self._batches = None, None

    def batch(self, step):
        e, j = divmod(step, STEPS_PER_EPOCH)
        if e != self._epoch:
            self.smp.set_epoch(e)
            self._batches, self._epoch = list(self.smp), e
        rng = np.random.default_rng([self.seed, 7, step])
        idx = self._batches[j]
        out = [int(rng.choice(self.cell[(int(self.s_full[i]), int(self.y_full[i]))])) for i in idx]
        return torch.tensor(out)


def entropy_counts(c):
    p = np.asarray(c, float); p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum())


# ------------------------------------------------------------------------- penalizaciones candidatas
def _kernel(z, scales=(0.5, 1.0, 2.0)):
    d2 = torch.cdist(z, z).pow(2)
    dd = d2.detach()
    off = dd[~torch.eye(len(z), dtype=torch.bool)]
    med = off.median() if bool((off > 0).any()) else d2.new_tensor(1.0)
    return sum(torch.exp(-d2 / (s * med + 1e-8)) for s in scales)


def mmd2_biased(x, y):
    from src.model.losses import _rbf_mmd2
    return _rbf_mmd2(x, y)


def mmd2_unbiased(x, y):
    n, m = len(x), len(y)
    k = _kernel(torch.cat([x, y], 0))
    kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    return (kxx.sum() - kxx.diag().sum()) / (n * (n - 1)) + (kyy.sum() - kyy.diag().sum()) / (m * (m - 1)) - 2 * kxy.mean()


def per_source_penalty(a, s, y, f, mixed=MIXED, bank=None):
    """Suma ponderada por n_s de f(positivas, negativas) en cada fuente mixta del lote (ecuación 33).
    bank: {(s, y): tensor de firmas pasadas, sin gradiente} que se añaden a cada conjunto."""
    tot, mass = a.new_zeros(()), 0.0
    for src in mixed:
        p, q = (s == src) & (y == 1), (s == src) & (y == 0)
        xp, xq = a[p], a[q]
        if bank is not None:
            bp, bq = bank.get((src, 1)), bank.get((src, 0))
            if bp is not None and len(bp):
                xp = torch.cat([xp, bp], 0)
            if bq is not None and len(bq):
                xq = torch.cat([xq, bq], 0)
        if len(xp) >= 2 and len(xq) >= 2 and int(p.sum()) + int(q.sum()) > 0:
            w = float(p.sum() + q.sum())
            tot = tot + w * f(xp, xq)
            mass += w
    return tot / mass if mass > 0 else tot


class ClassAdversary(nn.Module):
    """Discriminador de clase condicionado a la fuente mixta: D([a, onehot(s)]) -> logit de clase."""

    def __init__(self, sig_dim=64, n_mixed=len(MIXED), hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sig_dim + n_mixed, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, a, s_onehot):
        return self.net(torch.cat([a, s_onehot], 1)).squeeze(1)


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eta):
        ctx.eta = eta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.eta * g, None


# ------------------------------------------------------------------------- evaluación
@torch.no_grad()
def encode(enc, img, mk, bs=256, fixed_box=None):
    enc.eval()
    out = []
    for i in range(0, len(img), bs):
        xa, k = acq_input(img[i:i + bs], mk[i:i + bs], fixed_box)
        out.append(enc(xa, k))
    enc.train()
    return torch.cat(out).numpy()


def v1_like(sig, meta, n_boot=1000, kinds=('logreg',)):
    """AUC de clase desde la firma en bimcv de val, con la sonda de V1 (src/eval/probes.py: StandardScaler +
    regresión logística C = 1, 5 particiones agrupadas por paciente, semilla 42) e IC95 por remuestreo de pacientes."""
    from src.eval.probes import oof_proba, _auc_with_ci
    r = (meta.source_idx.values == BIMCV)
    X, y, g = sig[r], meta.label.values[r].astype(int), meta.patient_id.values[r].astype(str)
    res = {}
    for kind in kinds:
        p = oof_proba(X, y, g, kind, 5, 42)
        auc, lo, hi = _auc_with_ci(y, p, g, n_boot, 42) if n_boot else (None, None, None)
        if not n_boot:
            from src.eval.probes import _auc
            auc = _auc(y, p)
        res[kind] = (auc, lo, hi)
    return res


def source_metrics(enc, head, img, mk, meta):
    """Exactitud de fuente de la cabeza en val (exactitud global y balanceada entre las fuentes con >= 30 imágenes),
    pérdida de la cabeza en val y dispersión de la firma (media de la desviación típica por dimensión)."""
    from sklearn.metrics import balanced_accuracy_score
    A = encode(enc, img, mk)
    with torch.no_grad():
        lo = head(torch.from_numpy(A))
    s = meta.source_idx.values
    keep = np.isin(s, [k for k in np.unique(s) if (s == k).sum() >= 30])
    pred = lo.argmax(1).numpy()
    ce = float(F.cross_entropy(lo[torch.from_numpy(keep)], torch.from_numpy(s[keep]).long()))
    return A, {'acc_fuente_bal': float(balanced_accuracy_score(s[keep], pred[keep])),
               'acc_fuente': float((pred[keep] == s[keep]).mean()), 'ce_fuente_val': ce,
               'desv_firma': float(A.std(0).mean())}
