import os, sys, numpy as np, torch, time
sys.path.insert(0,'.')
torch.set_num_threads(8)
from src.model.layers import AcquisitionEncoder
from src.model.losses import _rbf_mmd2
F=["actmed","bimcv","cohen","fig1","ricord","rsna","sirm","stonybrook"]
XA, KP, S, Yc = torch.load(os.environ.get('DIAG_CACHE', '/tmp/diag_cache2.pt'))
def mmd_u(x, y, scales=(0.5,1.0,2.0)):
    z=torch.cat([x,y],0); d2=torch.cdist(z,z).pow(2); dd=d2.detach()
    med=dd[dd>0].median() if bool((dd>0).any()) else d2.new_tensor(1.0)
    k=sum(torch.exp(-d2/(s*med+1e-8)) for s in scales); n=x.shape[0]; m=y.shape[0]
    kxx,kyy,kxy=k[:n,:n],k[n:,n:],k[:n,n:]
    return (kxx.sum()-kxx.diag().sum())/(n*(n-1)) + (kyy.sum()-kyy.diag().sum())/(m*(m-1)) - 2*kxy.mean()
def ind(a, s, y, mixtas, f):
    tot=a.new_zeros(()); mass=0.0
    for src in mixtas:
        p=(s==src)&(y==1); q=(s==src)&(y==0)
        if int(p.sum())>=2 and int(q.sum())>=2:
            w=float(p.sum()+q.sum()); tot=tot+w*f(a[p],a[q]); mass+=w
    return tot/mass if mass>0 else tot
rng=np.random.default_rng(0)
# retenido: 40 por clase de bimcv para medir cuánta clase lleva la firma
bim0=torch.nonzero((S==1)&(Yc==0)).flatten().numpy(); bim1=torch.nonzero((S==1)&(Yc==1)).flatten().numpy()
ret=np.r_[bim0[:40], bim1[:40]]; excl=set(ret.tolist())
idx_of=lambda s,c: np.array([i for i in torch.nonzero((S==F.index(s))&(Yc==c)).flatten().numpy() if i not in excl])
def lote():
    sel=[]
    for s,c,n in (('bimcv',0,4),('bimcv',1,10),('cohen',0,4),('cohen',1,4),('rsna',0,6),('stonybrook',1,4),('sirm',1,2),('ricord',1,2)):
        sel += list(rng.choice(idx_of(s,c), n, replace=False))
    return torch.tensor(sel)
mixtas=[1,2]; modo=sys.argv[1]; w=float(sys.argv[2]); N=int(sys.argv[3])
f={'sesgado':_rbf_mmd2,'insesgado':mmd_u}[modo]
torch.manual_seed(1); enc=AcquisitionEncoder(); head=torch.nn.Linear(64,8)
opt=torch.optim.AdamW(list(enc.parameters())+list(head.parameters()), lr=1e-3, weight_decay=0.05)
hs=[]; hi=[]; t0=time.time()
for step in range(N+1):
    b=lote(); a=enc(XA[b],KP[b]); ls=torch.nn.functional.cross_entropy(head(a), S[b]); li=ind(a,S[b],Yc[b],mixtas,f)
    loss=0.5*ls+w*li; opt.zero_grad(); loss.backward(); opt.step(); hs.append(float(ls.detach())); hi.append(float(li.detach()))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
with torch.no_grad():
    enc.eval(); A=enc(XA[ret],KP[ret]).numpy()
yy=np.r_[np.zeros(40),np.ones(40)]
p=cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=1.0,max_iter=3000)), A, yy, cv=StratifiedKFold(5,shuffle=True,random_state=0), method='predict_proba')[:,1]
print('%-9s peso %.1f | L_src inicial %.3f -> final %.3f (prior 1.565) | L_ind medio final %.3f | AUC de clase en bimcv retenido (80 imágenes) %.3f | desviación media de la firma %.3f | %.0f s' % (modo, w, np.mean(hs[:5]), np.mean(hs[-15:]), np.mean(hi[-15:]), roc_auc_score(yy,p), float(A.std(0).mean()), time.time()-t0))
