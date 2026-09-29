import numpy as np, json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,"/data0/congcong/code/GS/mcgs_ce_retro_wt/ce_retro_cssense_af16/scripts")
import arms
T=arms.R; OUT=os.path.expanduser("~/figs/ce_retro_af16_audit")
KEEP=["cssense_o200","spirit","zsssdu","gs_single","gs_joint"]
rep={}
for scan in ["pre","post"]:
    gt=np.load(f"{T}/recon/gt_{scan}_complex.npy"); gmag=np.abs(gt); p999=float(np.quantile(gmag,0.999)); head=gmag>0.05*p999
    vols={l:np.load(p) for k,l,p in arms.arm_files(scan) if k in KEEP}
    labels=["GT"]+list(vols)
    # sagittal (fix axis2=48) and coronal (fix axis1=131), gamma display to expose low-level artefacts, whole FOV
    fig,ax=plt.subplots(4,len(labels),figsize=(2.6*len(labels),12))
    for j,l in enumerate(labels):
        v=gmag if l=="GT" else np.abs(vols[l])
        sag=v[:,:,48]; cor=v[:,131,:]
        ax[0,j].imshow((sag/p999).clip(0,1)**0.4,cmap="gray",vmin=0,vmax=1,origin="lower"); ax[0,j].set_title(f"{l} sag48 gamma0.4",fontsize=8)
        ax[2,j].imshow((cor/p999).clip(0,1)**0.4,cmap="gray",vmin=0,vmax=1,origin="lower"); ax[2,j].set_title(f"{l} cor131 gamma0.4",fontsize=8)
        if l!="GT":
            ax[1,j].imshow(np.abs(sag-gmag[:,:,48])/p999,cmap="jet",vmin=0,vmax=0.1,origin="lower")
            ax[3,j].imshow(np.abs(cor-gmag[:,131,:])/p999,cmap="jet",vmin=0,vmax=0.1,origin="lower")
        else:
            ax[1,j].imshow(head[:,:,48],cmap="gray",origin="lower"); ax[3,j].imshow(head[:,131,:],cmap="gray",origin="lower")
    for a in ax.ravel(): a.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/D_sag_cor_gamma_{scan}.png",dpi=110); plt.close()
    # background (outside head) statistics + in-head error stats
    r={}
    for l,v in vols.items():
        m=np.abs(v); e=m-gmag
        r[l]={"bg_rms_rel":float(np.sqrt(np.mean(m[~head]**2))/p999),"gt_bg_rms_rel":float(np.sqrt(np.mean(gmag[~head]**2))/p999),
              "head_err_rms_rel":float(np.sqrt(np.mean(e[head]**2))/p999),"head_err_p99_rel":float(np.quantile(np.abs(e[head]),0.99)/p999),
              "head_err_max_rel":float(np.abs(e[head]).max()/p999),"mean_bias_head":float(e[head].mean()/p999)}
    rep[scan]=r; print(scan,json.dumps(r,indent=1))
json.dump(rep,open(f"{OUT}/audit_bg.json","w"),indent=1)
