import numpy as np, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
N="/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois"; CM=N+"_cm"
masks={"pre":np.load(N+"/sim/mask/mask_af16_pe.npy")>0,"post":np.load(CM+"/sim/mask/mask_af16_pe_post.npy")>0}
cols={"same mask: GS-single":{"pre":N+"/recon/ar_s_pre_wf00v_pre_complex.npy","post":N+"/recon/ar_s_post_wf00bv_post_complex.npy"},
      "same mask: MC-GS":{"pre":N+"/recon/ar_e1b_wf00bv_pre_complex.npy","post":N+"/recon/ar_e1b_wf00bv_post_complex.npy"},
      "different masks: GS-single":{"pre":CM+"/recon/ar_cm_s_pre_pre_complex.npy","post":CM+"/recon/ar_cm_s_post_post_complex.npy"},
      "different masks: MC-GS":{"pre":CM+"/recon/ar_cm_mcgs_pre_complex.npy","post":CM+"/recon/ar_cm_mcgs_post_complex.npy"}}
score={"same mask: GS-single":(34.583,36.809),"same mask: MC-GS":(34.646,36.897),"different masks: GS-single":(34.583,36.863),"different masks: MC-GS":(34.684,36.977)}
z=201; y0,y1,x0,x1=30,200,15,165
fig,ax=plt.subplots(4,6,figsize=(19,13.2))
for i,sc in enumerate(["pre","post"]):
    gt=np.abs(np.load(f"{N}/recon/gt_{sc}_complex.npy")); p=float(np.quantile(gt,0.999)); head=gt[z]>0.05*p
    r=2*i
    ax[r,0].imshow(gt[z,y0:y1,x0:x1].T/p,cmap="gray",vmin=0,vmax=0.67,origin="lower"); ax[r,0].set_title(f"GT {sc} z{z}",fontsize=10)
    ax[r+1,0].imshow(masks[sc].T,cmap="gray",origin="lower",interpolation="nearest"); ax[r+1,0].set_title(f"{sc} mask (different-mask run)",fontsize=9)
    for j,(name,fs) in enumerate(cols.items(),start=1):
        v=np.abs(np.load(fs[sc])[z])
        ax[r,j].imshow(v[y0:y1,x0:x1].T/p,cmap="gray",vmin=0,vmax=0.67,origin="lower"); ax[r,j].set_title(f"{name}\nhead PSNR {score[name][i]:.3f}",fontsize=9)
        e=np.where(head,np.abs(v-gt[z])/p,0)
        ax[r+1,j].imshow(e[y0:y1,x0:x1].T,cmap="jet",vmin=0,vmax=0.1,origin="lower")
    m=masks["pre"]; o=masks["pre"]&masks["post"]
    ov=np.zeros(m.shape+(3,)); ov[...,0]=masks["pre"]; ov[...,1]=masks["post"]; ov[...,2]=o
    ax[r,5].imshow(ov.transpose(1,0,2),origin="lower",interpolation="nearest"); ax[r,5].set_title("red=pre only, green=post only,\nwhite=both",fontsize=9)
    ax[r+1,5].axis("off")
    ax[r+1,5].text(0.02,0.5,f"MC-GS - GS-single ({sc})\nsame mask: {score['same mask: MC-GS'][i]-score['same mask: GS-single'][i]:+.3f} dB\ndifferent masks: {score['different masks: MC-GS'][i]-score['different masks: GS-single'][i]:+.3f} dB",fontsize=12,va="center",transform=ax[r+1,5].transAxes)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
fig.suptitle("Different-mask probe, AF 16.00 per scan: pre keeps its mask, post uses a second vdPoisMex draw. Error |rec-GT|/P99.9, jet [0, 0.1], head only",fontsize=12)
plt.tight_layout(); out="/home/congcongliu/figs/ce_retro_af16_audit/different_mask_probe_z201.png"; plt.savefig(out,dpi=110); print(out)
