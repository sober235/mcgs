#!/usr/bin/env python3
"""Strict-bar verdict + figure for edge-TV vs baseline. Uses the P0 re-run as the noise
floor. Reads results_arc/edge_scores.tsv (merge per-GPU TSVs into it first).
Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/fig_edge_tv_verdict.py"""
import csv, collections, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import sobel

R = "results_arc"
rows = list(csv.DictReader(open(f"{R}/edge_scores.tsv"), delimiter="\t"))
by = collections.defaultdict(dict)
for r in rows: by[r["tag"]][r["contrast"]] = r

def mean(tag, key): return (float(by[tag]["T1"][key]) + float(by[tag]["FLAIR"][key])) / 2
# noise floor = |baseline_4k - p0_noise_4k| on each metric: TWO isotropic-TV runs at the SAME 4000-iter
# budget (only seed-nondeterminism differs). Must match edgeTV_best's budget — a 1500-iter p0 would fold the
# 1500->4000 convergence gap into the "floor" and inflate the PASS bar. (p0_noise@1500 is selection-only.)
floor = {k: abs(mean("baseline_4k", k) - mean("p0_noise_4k", k)) for k in ["nmse_edge", "ssim", "nmse_whole"]}
d_edge = mean("baseline_4k", "nmse_edge") - mean("edgeTV_best", "nmse_edge")   # + = improvement
d_ssim = mean("edgeTV_best", "ssim") - mean("baseline_4k", "ssim")            # + = improvement
d_nmse = mean("edgeTV_best", "nmse_whole") - mean("baseline_4k", "nmse_whole")  # + = worse
PASS = (d_edge > floor["nmse_edge"]) and (d_ssim >= -floor["ssim"]) and (d_nmse <= floor["nmse_whole"])
verdict = "PASS" if PASS else ("RESHAPE/FAIL" if d_edge > floor["nmse_edge"] else "NO-GAIN/FAIL")
print(f"noise floor: {floor}")
print(f"edge_gain={d_edge:+.4f} (floor {floor['nmse_edge']:.4f}) | dSSIM={d_ssim:+.4f} | dNMSE_whole={d_nmse:+.4f}")
print(f"VERDICT: {verdict}")

B = np.load(f"{R}/reg_aligned_4k_volumes.npz"); E = np.load(f"{R}/edgeTV_best_volumes.npz")
def gmag(v): return np.sqrt(sobel(v,0)**2+sobel(v,1)**2+sobel(v,2)**2)
AX = lambda v: v[:, :, 160]
fig, ax = plt.subplots(2, 4, figsize=(15, 7.4))
for r, c in enumerate(["T1", "FL"]):
    gt, rb, re = np.abs(B[f"gt{c}"]), np.abs(B[f"rec{c}"]), np.abs(E[f"rec{c}"]); peak = gt.max()
    eb, ee = np.abs(AX(rb)-AX(gt))/peak, np.abs(AX(re)-AX(gt))/peak
    for j,(im,ttl,cm,vm) in enumerate([(AX(gt),"GT","gray",peak),(eb,"baseline |err|","jet",0.1),
                                       (ee,"edge-TV |err|","jet",0.1),(AX(gmag(gt)),"|grad(GT)|","inferno",np.percentile(gmag(gt),99))]):
        h=ax[r,j].imshow(im.T,cmap=cm,vmin=0,vmax=vm,origin="lower"); ax[r,j].set_xticks([]); ax[r,j].set_yticks([])
        if r==0: ax[r,j].set_title(ttl,fontsize=11)
        if cm=="jet": plt.colorbar(h,ax=ax[r,j],fraction=0.046,pad=0.03,ticks=[0,.02,.04,.06,.08,.1])
    ax[r,0].set_ylabel(c,fontsize=11)
fig.suptitle(f"edge-TV vs baseline (butterfly z=160)  VERDICT={verdict}\n"
             f"mean edge-NMSE gain {d_edge:+.4f} (noise {floor['nmse_edge']:.4f}), dSSIM {d_ssim:+.4f}, dNMSE {d_nmse:+.4f}",
             fontsize=12)
plt.tight_layout(rect=[0,0,1,0.92]); plt.savefig(f"{R}/edgeTV_verdict.png", dpi=125, bbox_inches="tight")
print(f"saved {R}/edgeTV_verdict.png")
