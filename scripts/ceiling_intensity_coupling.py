"""Ceiling probe section 1 (record docs/superpowers/records/2026-09-26-widen-ceiling-probes.md): can a hand-crafted
intensity coupling predict one contrast's unsampled k-space from the other contrast's PERFECT image? jiangyizhou, AF16 topic
mask and ground truth, numpy only. Every predictor gets the target's sampled k-space back (hard DC) and the phase of the
target's GS-single reconstruction; scored as head PSNR of |x| vs |GT| (peak 1 at the p99.9 normalisation, ~7 dB below the
topic scorer -- read differences only). Anchors: the registered gs_single_wd / gs_joint_wd volumes. The last line leaks
the true residual through a 5-voxel local-linear model and is an upper bound, not a method.
(Was scratchpad predictability4.py in the 2026-09-26 session; the earlier energy-fraction variants were not kept.)
"""
import numpy as np, h5py
from scipy.ndimage import uniform_filter
from scipy.cluster.vq import kmeans2
T="/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/"
gt={c:np.load(T+f"recon/gt_{c}_complex.npy") for c in ("T1","FLAIR")}
m=np.asarray(h5py.File(T+"data/mask_af16_3d.h5")["mask"]).astype(bool)
mag={c:np.abs(gt[c]).astype(np.float32) for c in gt}
def fft3(x): return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x)))
def ifft3(k): return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(k)))
PH={}
K_full={c:fft3(gt[c]) for c in gt}
zfc={c:ifft3(K_full[c]*m) for c in gt}
def dc(pred_c,c): return ifft3(fft3(pred_c)*(~m)+K_full[c]*m)
def head_psnr(x,c):
    g=mag[c]; h=g>0.05; e=np.mean((np.abs(x)[h]-g[h])**2); return 10*np.log10(1.0/e)   # peak = 1 (p99.9 normalisation)
def with_phase(pred_mag,c): return pred_mag*np.exp(1j*np.angle(PH[c]))
def kclass(src,tgt_obs,head,K,seed=0):
    X=np.stack([src[head],tgt_obs[head]],1).astype(np.float64)
    idx=np.random.RandomState(seed).choice(len(X),200000,replace=False)
    cent,_=kmeans2(X[idx],K,minit='++',seed=0)
    lab=np.argmin(((X[:,None,:]-cent[None,:,:])**2).sum(2),1); ph=np.zeros(len(X),np.float32)
    for k in range(K):
        s=lab==k
        if s.sum()<10: continue
        A=np.stack([X[s,0],np.ones(s.sum())],1); coef=np.linalg.lstsq(A,X[s,1],rcond=None)[0]; ph[s]=A@coef
    pred=np.zeros_like(src); pred[head]=ph; return pred
def local_linear(src,tgt_obs,head,w):
    s=src*head; t=tgt_obs*head; n=uniform_filter(head.astype(np.float32),w)+1e-6
    ms=uniform_filter(s,w)/n; mt=uniform_filter(t,w)/n
    cov=uniform_filter(s*t,w)/n-ms*mt; var=uniform_filter(s*s,w)/n-ms*ms
    a=cov/(var+1e-4); b=mt-a*ms; return (a*src+b)*head
for s,t in (("T1","FLAIR"),("FLAIR","T1")):
    head=(mag[s]>0.05)|(mag[t]>0.05)
    print(f"== target {t}, source {s} GT (head PSNR after DC, dB)")
    print(f"  ZF                      {head_psnr(zfc[t],t):.3f}")
    recs={}; PH[t]=np.load(T+f"recon/gs_single_wd_{t}_complex.npy")
    for key in ("gs_single_wd","gs_joint_wd"):
        r=np.load(T+f"recon/{key}_{t}_complex.npy"); recs[key]=np.abs(r).astype(np.float32)
        print(f"  {key:22s}  {head_psnr(dc(r,t),t):.3f}   (as saved: {head_psnr(r,t):.3f})")
    for obs_name,obs in (("ZF",np.abs(zfc[t]).astype(np.float32)),("GS-single",recs["gs_single_wd"])):
        for K in (8,32): print(f"  K-class K={K:2d} fit on {obs_name:9s} -> DC  {head_psnr(dc(with_phase(kclass(mag[s],obs,head,K),t),t),t):.3f}")
        for w in (3,5): print(f"  local-lin w={w} fit on {obs_name:9s} -> DC  {head_psnr(dc(with_phase(local_linear(mag[s],obs,head,w),t),t),t):.3f}")
    # residual test: does the source GT explain the GS-single residual? (local-linear of the residual on the source)
    res=mag[t]-recs["gs_single_wd"]; pr=local_linear(mag[s],res,head,5)
    print(f"  GS-single + local-lin(w=5) residual model (LEAKS via residual, upper bound only): {head_psnr(dc(PH[t]+pr*np.exp(1j*np.angle(PH[t])),t),t):.3f}")
