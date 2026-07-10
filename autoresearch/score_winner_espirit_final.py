#!/usr/bin/env python3
"""Score the exported winner reconstruction with the EXTERNAL ESPIRiT CSM,
apples-to-apples with the corrected-baseline (T1 37.44 / FLAIR 37.82) numbers.

GT = |sum_c conj(CSM_espirit_c) * ifft3c(ksp_fullysampled_c)|, alpha-fit, PSNR/SSIM.
Also dumps recon + GT volumes to .npy for figure generation.
"""
import sys, json
from pathlib import Path
import numpy as np, h5py
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim

ROOT = Path("/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC")
EXP = ROOT / "results_espirit_winner_hi20/mcgs_espirit_winner_hi20/test_results_best"
KSP = {
 "T1":    "/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_T1_trainer.mat",
 "FLAIR": "/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/Train_3dBrain_huangyuwei_20230705-160927-1451_161015_FLAIR_trainer.mat",
}
CSM = {"T1": ROOT/"csm_espirit/csm_espirit_T1.mat", "FLAIR": ROOT/"csm_espirit/csm_espirit_FLAIR.mat"}
RECON = {"T1": EXP/"reconstruction_plain.mat", "FLAIR": EXP/"reconstruction_contrast.mat"}
OUT = ROOT / "results_espirit_winner_hi20"

def ifft3c(x):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x,axes=(0,1,2)),axes=(0,1,2),norm="ortho"),axes=(0,1,2))

def load_ksp(p):
    with h5py.File(p,"r") as f:
        a=np.asarray(f["ksp"])
    if a.dtype.names and {"real","imag"}<=set(a.dtype.names):
        a=a["real"]+1j*a["imag"]
    a=np.asarray(a)
    # want (X,Y,Z,coils)=(256,220,256,12)
    if a.shape[0]==12: a=np.moveaxis(a,0,-1)
    return a

def load_csm(p):
    try: m=loadmat(p)["map"]
    except Exception:
        with h5py.File(p,"r") as f: m=np.asarray(f["map"])
    m=np.asarray(m)
    if m.shape[0]==12: m=np.moveaxis(m,0,-1)
    return m

def load_recon(p):
    m=loadmat(p); 
    for k in m:
        if not k.startswith("__"): return np.asarray(m[k])

res={}
for con in ["T1","FLAIR"]:
    ksp=load_ksp(KSP[con]); csm=load_csm(CSM[con])
    coil=np.stack([ifft3c(ksp[...,cc]) for cc in range(ksp.shape[-1])],axis=-1)
    gt=np.abs(np.sum(np.conj(csm)*coil,axis=-1)).astype(np.float32)
    rec=np.abs(load_recon(RECON[con])).astype(np.float32)
    alpha=float(np.sum(rec*gt)/np.sum(rec*rec)); r=rec*alpha
    mx=float(gt.max())
    psnr=10*np.log10(mx**2/np.mean((r-gt)**2))
    ss=ssim(gt,r,data_range=mx)
    res[con]=dict(psnr=round(float(psnr),3),ssim=round(float(ss),4),alpha=alpha,gt_max=mx)
    np.save(OUT/f"gt_{con}.npy", gt)
    np.save(OUT/f"recon_{con}.npy", r)
    print(f"{con}: PSNR={psnr:.2f} SSIM={ss:.4f}")

json.dump(res, open(OUT/"espirit_scores.json","w"), indent=2)
print("SCORE_DONE", json.dumps(res))
