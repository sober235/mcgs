import numpy as np, os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
OUT=os.path.expanduser("~/figs/ce_retro_af16_audit")
ce=np.load("/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16/sim/mask/mask_af16_pe.npy").astype(bool)
J="/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/data/"
m=loadmat(J+"mask_vdpois_256x176_af16.mat"); print({k:(v.shape if hasattr(v,'shape') else v) for k,v in m.items() if not k.startswith("__")})
key=[k for k in m if not k.startswith("__")][0]; jz=np.abs(m[key])>0.5
if jz.ndim==3: jz=jz[:,:,0]
print("jyz",jz.shape,jz.sum(),jz.size/jz.sum(),"ce",ce.shape,ce.sum(),ce.size/ce.sum())
def prof(mk,nb=20):
    ny,nz=mk.shape; u=(np.arange(ny)-ny//2)/(ny/2); v=(np.arange(nz)-nz//2)/(nz/2)
    r=np.sqrt(u[:,None]**2+v[None,:]**2); edges=np.linspace(0,1.0,nb+1); out=[]
    for a,b in zip(edges[:-1],edges[1:]):
        s=(r>=a)&(r<b); out.append(mk[s].mean() if s.any() else np.nan)
    return (edges[:-1]+edges[1:])/2, np.array(out), r
def ifft2c(x): return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x),norm="ortho"))
fig,ax=plt.subplots(2,3,figsize=(14,8))
res={}
for i,(name,mk) in enumerate([("CE AF16 (BART poisson -v, 232x176)",ce),("jyz AF16 (vdPoisMex, 256x176)",jz)]):
    c,p,r=prof(mk); res[name]={"af":float(mk.size/mk.sum()),"profile_centres":c.tolist(),"profile":p.tolist()}
    ax[i,0].imshow(mk.T,cmap="gray",origin="lower"); ax[i,0].set_title(name,fontsize=9)
    ax[i,1].plot(c,p,"o-"); ax[i,1].set_ylim(0,1.02); ax[i,1].set_xlabel("normalised radius"); ax[i,1].set_ylabel("sampled fraction"); ax[i,1].grid(alpha=.3); ax[i,1].set_title(f"radial density, AF {mk.size/mk.sum():.2f}",fontsize=9)
    P=np.abs(ifft2c(mk.astype(np.complex64))); P/=P.max()
    ax[i,2].imshow(np.log10(P.T+1e-5),cmap="magma",origin="lower",vmin=-4,vmax=0); ax[i,2].set_title("log10|PSF|",fontsize=9)
    # cumulative: fraction of samples inside radius
    for rr in (0.1,0.15,0.2,0.3,0.5):
        res[name][f"frac_samples_within_r{rr}"]=float(mk[r<rr].sum()/mk.sum())
        res[name][f"density_within_r{rr}"]=float(mk[r<rr].mean())
plt.tight_layout(); plt.savefig(f"{OUT}/G_mask_compare_ce_vs_jyz.png",dpi=110)
print(json.dumps(res,indent=1))
json.dump(res,open(f"{OUT}/audit_mask_compare.json","w"),indent=1)
