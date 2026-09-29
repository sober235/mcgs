import numpy as np, json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,"/data0/congcong/code/GS/mcgs_ce_retro_wt/ce_retro_cssense_af16/scripts")
import arms
T=arms.R; OUT=os.path.expanduser("~/figs/ce_retro_af16_audit"); os.makedirs(OUT,exist_ok=True)
mask=np.load(f"{T}/sim/mask/mask_af16_pe.npy").astype(bool)   # (232,176)
def fft2c(x): return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x,axes=(-2,-1)),axes=(-2,-1),norm="ortho"),axes=(-2,-1))
def ifft2c(x): return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x,axes=(-2,-1)),axes=(-2,-1),norm="ortho"),axes=(-2,-1))
KEEP=["cssense_o200","spirit","zsssdu","gs_single","gs_joint"]
report={}
for scan in ["pre","post"]:
    gt=np.load(f"{T}/recon/gt_{scan}_complex.npy")
    gmag=np.abs(gt); p999=float(np.quantile(gmag,0.999)); head=gmag>0.05*p999
    vols={}
    for key,label,p in arms.arm_files(scan):
        if key in KEEP: vols[label]=np.load(p)
    labels=["GT"]+list(vols)
    # ---------- A. whole-FOV axial slices, bright window ----------
    zs=[120,160,201,230]
    fig,ax=plt.subplots(2*len(zs),len(labels),figsize=(2.2*len(labels),2.2*2*len(zs)))
    for i,z in enumerate(zs):
        for j,l in enumerate(labels):
            v=gmag if l=="GT" else np.abs(vols[l])
            ax[2*i,j].imshow(v[z].T,cmap="gray",vmin=0,vmax=0.4*p999,origin="lower"); ax[2*i,j].set_title(f"{l} z{z}" if i==0 else f"z{z}",fontsize=8)
            if l!="GT":
                e=np.abs(v[z]-gmag[z])/p999
                ax[2*i+1,j].imshow(e.T,cmap="jet",vmin=0,vmax=0.1,origin="lower")
            else:
                ax[2*i+1,j].imshow(head[z].T,cmap="gray",origin="lower"); ax[2*i+1,j].set_title("head mask",fontsize=8)
    for a in ax.ravel(): a.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/A_axial_wholefov_{scan}.png",dpi=110); plt.close()
    # ---------- B. error spectrum over PE plane (avg over readout) ----------
    fig,ax=plt.subplots(2,len(labels),figsize=(2.6*len(labels),5.2))
    G=fft2c(gt); Gpe=np.mean(np.abs(G)**2,axis=0)
    ax[0,0].imshow(mask.T,cmap="gray",origin="lower"); ax[0,0].set_title("mask (ky,kz)",fontsize=8)
    ax[1,0].imshow(np.log10(Gpe.T+1e-12),cmap="viridis",origin="lower"); ax[1,0].set_title("log|GT|^2",fontsize=8)
    rep={}
    yy,zz=np.meshgrid(np.arange(232)-116,np.arange(176)-88,indexing="ij"); rad=np.sqrt((yy/116)**2+(zz/88)**2)
    bands=[(0,0.1),(0.1,0.2),(0.2,0.4),(0.4,0.7),(0.7,1.0),(1.0,1.5)]
    for j,l in enumerate(labels[1:],start=1):
        E=fft2c(vols[l]-gt); Epe=np.mean(np.abs(E)**2,axis=0)
        ax[0,j].imshow(np.log10(Epe.T+1e-12),cmap="viridis",origin="lower",vmin=np.log10(Gpe.max())-7,vmax=np.log10(Gpe.max())-1); ax[0,j].set_title(f"{l} log|err|^2",fontsize=8)
        # autocorrelation of error in PE plane (coherence test)
        R=np.abs(ifft2c(Epe)); R/=R.max()
        ax[1,j].imshow(np.log10(R.T+1e-6),cmap="magma",origin="lower",vmin=-4,vmax=0); ax[1,j].set_title("err autocorr (log)",fontsize=8)
        # numbers
        es=float(Epe[mask].sum()); eu=float(Epe[~mask].sum())
        gs_=float(Gpe[mask].sum()); gu=float(Gpe[~mask].sum())
        bandnmse={f"{a}-{b}":float(Epe[(rad>=a)&(rad<b)].sum()/max(Gpe[(rad>=a)&(rad<b)].sum(),1e-30)) for a,b in bands}
        # off-centre autocorr peak
        Rm=R.copy(); Rm[(np.abs(yy)<=3)&(np.abs(zz)<=3)]=0
        k=np.unravel_index(np.argmax(Rm),Rm.shape)
        rep[l]={"err_energy_sampled_frac":es/(es+eu),"nmse_sampled_band":es/gs_,"nmse_unsampled_band":eu/gu,"band_nmse":bandnmse,
                "autocorr_offpeak":float(Rm.max()),"autocorr_offpeak_lag":(int(k[0]-116),int(k[1]-88))}
    for a in ax.ravel(): a.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/B_error_spectrum_{scan}.png",dpi=110); plt.close()
    # mask PSF and its autocorr
    P=ifft2c(mask.astype(np.complex64)); Pa=np.abs(P)/np.abs(P).max()
    Rm_=np.abs(ifft2c(mask.astype(np.float64))); 
    rep["mask_psf_max_sidelobe"]=float(np.sort(Pa.ravel())[-2]); 
    k=np.unravel_index(np.argsort(Pa.ravel())[-2],Pa.shape); rep["mask_psf_sidelobe_lag"]=(int(k[0]-116),int(k[1]-88))
    report[scan]=rep
    print(scan, json.dumps(rep,indent=1))
json.dump(report,open(f"{OUT}/audit_spectrum.json","w"),indent=1)
# mask PSF figure
P=ifft2c(mask.astype(np.complex64)); Pa=np.abs(P)/np.abs(P).max()
fig,ax=plt.subplots(1,2,figsize=(8,4)); ax[0].imshow(mask.T,cmap="gray",origin="lower"); ax[0].set_title("AF16 PE mask")
im=ax[1].imshow(np.log10(Pa.T+1e-5),cmap="magma",origin="lower",vmin=-4,vmax=0); ax[1].set_title("log10 |PSF| (image domain)"); plt.colorbar(im,ax=ax[1])
plt.tight_layout(); plt.savefig(f"{OUT}/C_mask_psf.png",dpi=120)
