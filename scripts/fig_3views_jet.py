"""GT | ZF | recon | error(jet,[0,0.1]) in three orthogonal views for T1 and FLAIR.
usage: fig_3views_jet.py <volumes.npz> <score.json> <out_prefix> "<title>"
views (house convention): transverse v[:,:,128], sagittal v[:,110,:], coronal v[128,:,:]
"""
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
npz, score, out_prefix, title = sys.argv[1:5]
z = np.load(npz); sc = json.load(open(score))
def psnr(a, g, msk, gmax): return 20*np.log10(gmax/np.sqrt(np.mean((a[msk]-g[msk])**2)))
def views(v):
    return {"transverse (z=128)": np.rot90(v[:, :, 128]),
            "sagittal (y=110)": np.flipud(v[:, 110, :].T),
            "coronal (x=128)": np.flipud(v[128, :, :].T)}
for con, key, rk, gk, zk in [("T1", "plain", "recT1", "gtT1", "zfT1"), ("FLAIR", "contrast", "recFL", "gtFL", "zfFL")]:
    rec = z[rk].astype(np.float32); gt = z[gk].astype(np.float32); zf = z[zk].astype(np.float32)
    gmax = float(gt.max()); head = gt > 0.05*np.percentile(gt, 99.9); allm = np.ones_like(head)
    vmax = float(np.percentile(gt[head], 99.5))
    err = np.abs(rec-gt)/gmax; errz = np.abs(zf-gt)/gmax
    s = sc[key]
    hdr = (f"{title}  |  {con}: recon PSNR whole@max {s['psnr']:.2f} dB / head@max {psnr(rec,gt,head,gmax):.2f} dB, SSIM {s['ssim']:.3f}, "
           f"NMSE {s['nmse']:.4f}, HFEN {s['hfen']:.3f}   |   ZF whole {psnr(zf,gt,allm,gmax):.2f} / head {psnr(zf,gt,head,gmax):.2f} dB")
    V = {"GT": views(gt), "Zero-filled": views(zf), "Recon": views(rec), "|Recon-GT|/max(GT)": views(err)}
    names = list(V["GT"].keys())
    fig, axes = plt.subplots(3, 4, figsize=(17, 12.5))
    for r, vn in enumerate(names):
        for c, (cn, vv) in enumerate(V.items()):
            ax = axes[r, c]; img = vv[vn]
            if c < 3:
                ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
            else:
                im = ax.imshow(img, cmap="jet", vmin=0, vmax=0.1)
                cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.04)
                fig.colorbar(im, cax=cax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(cn, fontsize=13)
            if c == 0: ax.set_ylabel(vn, fontsize=12)
    fig.suptitle(hdr, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = f"{out_prefix}_{con}_3views.png"; fig.savefig(out, dpi=110); plt.close(fig); print("saved", out)
