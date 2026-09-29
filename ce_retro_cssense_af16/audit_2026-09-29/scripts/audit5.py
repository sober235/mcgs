import numpy as np, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
sys.path.insert(0,"/data0/congcong/code/GS/mcgs_ce_retro_wt/ce_retro_cssense_af16/scripts")
import arms
T=arms.R; OUT=os.path.expanduser("~/figs/ce_retro_af16_audit")
for scan in ["pre","post"]:
    gt=np.abs(np.load(f"{T}/recon/gt_{scan}_complex.npy")); p999=float(np.quantile(gt,0.999))
    files=dict((k,p) for k,l,p in arms.arm_files(scan))
    zf=np.abs(np.load(files["zf"])); gj=np.abs(np.load(files["gs_joint"])); gs=np.abs(np.load(files["gs_single"])); sp=np.abs(np.load(files["spirit"]))
    z=201; y0,y1,x0,x1=60,180,40,140   # zoom box in the PE plane
    fig,ax=plt.subplots(3,5,figsize=(17,10.5))
    for j,(l,v) in enumerate([("GT",gt),("ZF",zf),("L1-SPIRiT",sp),("GS-single",gs),("MC-GS",gj)]):
        im=v[z,y0:y1,x0:x1].T/p999
        ax[0,j].imshow(im,cmap="gray",vmin=0,vmax=1,origin="lower"); ax[0,j].set_title(f"{l}  z{z} zoom, window P99.9",fontsize=9)
        hp=(v[z]-ndimage.gaussian_filter(v[z],2))[y0:y1,x0:x1].T/p999
        ax[1,j].imshow(hp,cmap="gray",vmin=-0.08,vmax=0.08,origin="lower"); ax[1,j].set_title("high-pass (sigma2) +-0.08",fontsize=9)
        sd=(v[z]-gt[z])[y0:y1,x0:x1].T/p999
        ax[2,j].imshow(sd,cmap="RdBu_r",vmin=-0.1,vmax=0.1,origin="lower"); ax[2,j].set_title("signed error +-0.1",fontsize=9)
    for a in ax.ravel(): a.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/F_zoom_texture_{scan}_z{z}.png",dpi=110); plt.close()
print("ok")
