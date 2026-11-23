import sys, numpy as np, pandas as pd, torch, time, os
sys.path.insert(0,'.')
torch.set_num_threads(8)
from src.model.layers import AcquisitionInput, AcquisitionEncoder
from src.model.losses import L_ind
F=["actmed","bimcv","cohen","fig1","ricord","rsna","sirm","stonybrook"]
C='/tmp/diag_cache2.pt'
if os.path.exists(C):
    XA, KP, S, Yc = torch.load(C)
else:
    from src.data.homogeneo_masks import CXRMaskDataset
    m=pd.read_csv('data/manifests/covidx_cxr4_clean.csv', low_memory=False, dtype={'patient_id':str})
    m=m[m.split_clean=='val'].copy()
    m['image_id']=m.image_path.map(lambda p: p.split('/')[-1]); m['source_idx']=m.source.map(F.index)
    parts=[]
    for s,c,n in (('bimcv',0,110),('bimcv',1,110),('cohen',0,76),('cohen',1,19),('rsna',0,110),('stonybrook',1,110),('sirm',1,65),('ricord',1,62)):
        g=m[(m.source==s)&(m.label==c)]; parts.append(g.sample(min(len(g),n), random_state=0))
    df=pd.concat(parts).reset_index(drop=True)
    ds=CXRMaskDataset(df,'data/cache_m/covidx_cxr4/homogeneo_224','data/masks/covidx_cxr4/homogeneo_224', require=True)
    acq=AcquisitionInput(); xs=[]; ks=[]
    for i in range(len(ds)):
        it=ds[i]; xa,k=acq(it['image'][None][:, :1].float(), it['mask'][None].amax(1,keepdim=True).float()); xs.append(xa[0]); ks.append(k[0])
    XA=torch.stack(xs); KP=torch.stack(ks); S=torch.tensor(df.source_idx.values); Yc=torch.tensor(df.label.values)
    torch.save((XA,KP,S,Yc), C)
print('n', len(S), 'celdas', {(F[int(s)],int(c)): int(((S==s)&(Yc==c)).sum()) for s in S.unique() for c in (0,1) if ((S==s)&(Yc==c)).any()})
rng=np.random.default_rng(0)
idx_of=lambda s,c: torch.nonzero((S==F.index(s))&(Yc==c)).flatten().numpy()
def lote():
    sel=[]
    for s,c,n in (('bimcv',0,4),('bimcv',1,10),('cohen',0,4),('cohen',1,4),('rsna',0,6),('stonybrook',1,4),('sirm',1,2),('ricord',1,2)):
        sel += list(rng.choice(idx_of(s,c), n, replace=False))
    return torch.tensor(sel)
mixtas=[F.index('bimcv'), F.index('cohen')]
w_ind=float(sys.argv[1]); N=int(sys.argv[2])
torch.manual_seed(1)
enc=AcquisitionEncoder(); head=torch.nn.Linear(64, 8)
opt=torch.optim.AdamW(list(enc.parameters())+list(head.parameters()), lr=1e-3, weight_decay=0.05)
hs=[]; hi=[]; t0=time.time()
for step in range(N+1):
    b=lote(); a=enc(XA[b],KP[b])
    ls=torch.nn.functional.cross_entropy(head(a), S[b]); li=L_ind(a, S[b], Yc[b], mixtas)
    if step in (0, N//2, N):
        gs=torch.autograd.grad(0.5*ls, list(enc.parameters()), retain_graph=True); gi=torch.autograd.grad(li, list(enc.parameters()), retain_graph=True)
        ns=float(torch.sqrt(sum((g**2).sum() for g in gs))); ni=float(torch.sqrt(sum((g**2).sum() for g in gi)))
        print('   paso %d: norma del gradiente en el codificador: 0.5 L_src %.4f | 1.0 L_ind %.4f | razón %.1f' % (step, ns, ni, ni/max(ns,1e-12)))
    loss=0.5*ls + w_ind*li
    opt.zero_grad(); loss.backward(); opt.step(); hs.append(float(ls.detach())); hi.append(float(li.detach()))
    if step%(N//3)==0:
        print('peso L_ind=%.1f paso %3d  L_src (media 15) %.3f  L_ind (media 15) %.3f  [%.0f s]' % (w_ind, step, np.mean(hs[-15:]), np.mean(hi[-15:]), time.time()-t0), flush=True)
