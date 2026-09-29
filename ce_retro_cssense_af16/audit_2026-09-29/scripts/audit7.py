# mask-only test on the CE truth: how much of the truth's k-space energy does each sampling design leave unmeasured, per band?
import numpy as np, json, os
from scipy.io import loadmat
T="/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"
ce=np.load(f"{T}/sim/mask/mask_af16_pe.npy").astype(bool)
J="/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/data/mask_vdpois_256x176_af16.mat"
m=loadmat(J); key=[k for k in m if not k.startswith("__")][0]; jz=np.abs(m[key])>0.5
if jz.ndim==3: jz=jz[:,:,0]
def radial(shape):
    ny,nz=shape; u=(np.arange(ny)-ny//2)/(ny/2); v=(np.arange(nz)-nz//2)/(nz/2); return np.sqrt(u[:,None]**2+v[None,:]**2)
def density_fn(mk,nb=40):
    r=radial(mk.shape); edges=np.linspace(0,1.45,nb+1); p=[]
    for a,b in zip(edges[:-1],edges[1:]):
        s=(r>=a)&(r<b); p.append(mk[s].mean() if s.any() else np.nan)
    p=np.array(p); c=(edges[:-1]+edges[1:])/2; ok=~np.isnan(p); return lambda rr: np.interp(rr,c[ok],p[ok])
def fft2c(x): return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x,axes=(-2,-1)),axes=(-2,-1),norm="ortho"),axes=(-2,-1))
out={}
for scan in ["pre","post"]:
    gt=np.load(f"{T}/recon/gt_{scan}_complex.npy"); G=np.mean(np.abs(fft2c(gt))**2,axis=0)   # (232,176) energy over readout
    r=radial(G.shape); tot=G.sum()
    bands=[(0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.0),(1.0,1.5)]
    res={}
    for name,mk in [("CE_bart_poisson_actual",ce),("jyz_vdpoismex_density_applied",None),("CE_density_applied",None)]:
        if mk is not None: samp=mk.astype(float)
        else:
            f=density_fn(jz if name.startswith("jyz") else ce); samp=f(r)
        res[name]={"unsampled_energy_frac_total":float(np.sum(G*(1-samp))/tot),
                   "af_equiv":float(samp.size/samp.sum()),
                   "per_band_unsampled_frac":{f"{a}-{b}":float(np.sum((G*(1-samp))[(r>=a)&(r<b)])/max(np.sum(G[(r>=a)&(r<b)]),1e-30)) for a,b in bands},
                   "per_band_energy_share":{f"{a}-{b}":float(np.sum(G[(r>=a)&(r<b)])/tot) for a,b in bands}}
    out[scan]=res
    print(scan); 
    for k,v in res.items(): print(" ",k,"AF_equiv %.2f  unsampled energy %.4f"%(v["af_equiv"],v["unsampled_energy_frac_total"]), {kk:round(vv,3) for kk,vv in v["per_band_unsampled_frac"].items()})
    print("  energy share per band:",{k:round(v,4) for k,v in res["CE_bart_poisson_actual"]["per_band_energy_share"].items()})
json.dump(out,open(os.path.expanduser("~/figs/ce_retro_af16_audit/audit_mask_energy.json"),"w"),indent=1)
