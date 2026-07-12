
import numpy as np, json, matplotlib.pyplot as plt, matplotlib as mpl

FW="/data0/congcong/code/GS/Paired_ARC_GS_general_framework"
d = np.load(f"{FW}/results_arc_final_tf/final_volumes.npz")
sc = json.load(open(f"{FW}/results_arc_final_tf/final_scores.json"))

def three_views(vol):
    D,H,W = vol.shape
    return [vol[D//2,:,:], vol[:,H//2,:], vol[:,:,W//2]]   # transverse, sagittal, coronal-ish

def make_fig(rec, gt, name, psnr, ssim, ho, path):
    apply_figure_style(sizes=(8,7,6))
    gtv, rcv = three_views(gt), three_views(rec)
    errv = [np.abs(r-g) for r,g in zip(rcv, gtv)]
    vmax = max(g.max() for g in gtv)
    emax = 0.10*vmax
    titles = ["TRANSVERSE","SAGITTAL","CORONAL"]
    fig, axes = plt.subplots(3,3, figsize=(8.4,8.6))
    for j,(g,r,e,tt) in enumerate(zip(gtv,rcv,errv,titles)):
        axes[0,j].imshow(np.rot90(g), cmap="gray", vmin=0, vmax=vmax); axes[0,j].set_title(tt, fontsize=8)
        axes[1,j].imshow(np.rot90(r), cmap="gray", vmin=0, vmax=vmax)
        im=axes[2,j].imshow(np.rot90(e), cmap="jet", vmin=0, vmax=emax)
        for i in range(3): axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
    axes[0,0].set_ylabel("GT (ESPIRiT CSM)", fontsize=7)
    axes[1,0].set_ylabel("Paired-ARC-GS", fontsize=7)
    axes[2,0].set_ylabel("|error| (jet 0..0.1·max)", fontsize=7)
    fig.colorbar(im, ax=axes[2,:].tolist(), fraction=0.02, pad=0.01)
    fig.suptitle(f"Paired-ARC-GS {name}  —  PSNR {psnr:.2f} dB / SSIM {ssim:.3f}  (held-out k-NRMSE {ho:.3f})\n"
                 f"full model: shared anatomy + response decoder + change(audit) + deform + phase + Gabor + CG (§4-13)",
                 fontsize=8.2)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    r_=fig.canvas.get_renderer()
    tx=[(t,t.get_window_extent(r_)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
    ov=[(a.get_text(),b.get_text()) for i,(a,ba) in enumerate(tx) for b,bb in tx[i+1:] if ba.overlaps(bb)]
    return len(ov)

o1=make_fig(d["recT1"], d["gtT1"], sc["T1"]["name"] if "name" in sc["T1"] else "T1",
            sc["T1"]["psnr"], sc["T1"]["ssim"], sc["T1"]["holdout_nrmse"], f"{FW}/results_arc_final_tf/recon_T1.png")
o2=make_fig(d["recFL"], d["gtFL"], "FLAIR",
            sc["FLAIR"]["psnr"], sc["FLAIR"]["ssim"], sc["FLAIR"]["holdout_nrmse"], f"{FW}/results_arc_final_tf/recon_FLAIR.png")
print("overlaps T1", o1, "FLAIR", o2)
print("T1  psnr", round(sc["T1"]["psnr"],3), "ssim", round(sc["T1"]["ssim"],4))
print("FLAIR psnr", round(sc["FLAIR"]["psnr"],3), "ssim", round(sc["FLAIR"]["ssim"],4))
