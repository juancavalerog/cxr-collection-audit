import sys, numpy as np, pandas as pd, torch, time, pickle, os
sys.path.insert(0,'.')
torch.set_num_threads(8)
from src.data.datasets import build_augment, make_generator
from src.model.layers import AcquisitionInput, AcquisitionEncoder
F=["actmed","bimcv","cohen","fig1","ricord","rsna","sirm","stonybrook"]
C='/tmp/diag_cache.pt'
if os.path.exists(C):
    IM, MK, Y = torch.load(C)
else:
    from src.data.homogeneo_masks import CXRMaskDataset
    m=pd.read_csv('data/manifests/covidx_cxr4_clean.csv', low_memory=False, dtype={'patient_id':str})
    m=m[m.split_clean=='val'].copy()
    m['image_id']=m.image_path.map(lambda p: p.split('/')[-1]); m['source_idx']=m.source.map(F.index)
    parts=[m[m.source==s].sample(min((m.source==s).sum(),100), random_state=0) for s in ('bimcv','rsna','stonybrook','cohen','sirm','ricord')]
    df=pd.concat(parts).reset_index(drop=True)
    ds=CXRMaskDataset(df,'data/cache_m/covidx_cxr4/homogeneo_224','data/masks/covidx_cxr4/homogeneo_224', require=True)
    IM=torch.stack([ds[i]['image'] for i in range(len(ds))]); MK=torch.stack([ds[i]['mask'] for i in range(len(ds))]); ys=[int(ds[i]['source']) for i in range(len(ds))]
    cls=sorted(set(ys)); Y=torch.tensor([cls.index(v) for v in ys]); torch.save((IM,MK,Y), C)
acq=AcquisitionInput()
aug=build_augment('nuevo', rotation=0.03, zoom=0.1, brightness=0.05, contrast=[0.9,1.1], cutout=24)
torch.manual_seed(0); perm=torch.randperm(len(Y)); tr, te = perm[:480], perm[480:]
def entrada(idx, paso, aumentar):
    x, mk = IM[idx], MK[idx]
    if aumentar:
        x, _, mk = aug(x, torch.zeros(len(idx)), make_generator(7, paso, 'cpu'), mask=mk); mk=(mk>0.5).float()
    return acq(x[:, :1].float(), mk.amax(1, keepdim=True))
modo = sys.argv[1]=='1'; N=int(sys.argv[2])
torch.manual_seed(1)
enc=AcquisitionEncoder(); head=torch.nn.Linear(64, int(Y.max())+1)
opt=torch.optim.AdamW(list(enc.parameters())+list(head.parameters()), lr=1e-3, weight_decay=0.05)
hist=[]; t0=time.time()
for step in range(N+1):
    idx=tr[torch.randint(0,len(tr),(32,))]
    xa,k=entrada(idx, step, modo)
    loss=torch.nn.functional.cross_entropy(head(enc(xa,k)), Y[idx]); opt.zero_grad(); loss.backward(); opt.step(); hist.append(float(loss.detach()))
    if step%(N//3)==0:
        with torch.no_grad():
            enc.eval(); xa,k=entrada(te, 10**6, False); lo=head(enc(xa,k)); enc.train()
        print('aumento=%s paso %3d  CE entrenamiento (media 20) %.3f  CE retenido sin aumento %.3f  exactitud %.3f  [%.0f s]' % (modo, step, np.mean(hist[-20:]), float(torch.nn.functional.cross_entropy(lo, Y[te])), float((lo.argmax(1)==Y[te]).float().mean()), time.time()-t0), flush=True)
xa,k=entrada(tr[:8], 3, modo); print('fracción visible', [round(float(v),3) for v in k.mean(dim=(1,2,3))])
