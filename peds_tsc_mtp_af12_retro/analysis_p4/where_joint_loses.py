"""Where does the joint arm lose T1w to the single arm? Primary head region, magnitude, as the table.
Voxels are binned by the ground-truth edge strength of T1w and of T2s (Gaussian-derivative gradient
magnitude, each normalised by its 99th percentile in the head). For each bin: the share of the
joint-minus-single squared-error excess. If the excess sits at edges that only T2s has, the shared
cloud is spending T1w's capacity on T2s-only structure (a case for private capacity per contrast);
if it sits at T1w-only edges, T2s is smoothing T1w's own detail away."""
import sys, json
import numpy as np
from scipy import ndimage
sys.path.insert(0, "/data0/congcong/code/GS/mcgs_peds_wt")
from diag_artefact.headonly import head_mask
from uih_mtp.rescore import partition_region
T = "/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro/"
ga = np.abs(np.load(T + "recon/gt_T1w_complex.npy")).astype(np.float32)
gb = np.abs(np.load(T + "recon/gt_T2s_complex.npy")).astype(np.float32)
j = np.abs(np.load(T + "recon/gs_joint_p3_T1w_complex.npy")).astype(np.float32)
s = np.abs(np.load(T + "recon/gs_single_p3_T1w_complex.npy")).astype(np.float32)
head = head_mask(ga) & partition_region(ga.shape, 2, 12, 92)
def edge(v):
    g = ndimage.gaussian_gradient_magnitude(v, sigma=(1.0, 1.0, 0.52))   # voxel 0.7x0.7x1.35 mm
    return g / np.percentile(g[head], 99)
ea, eb = edge(ga), edge(gb)
dj, ds = (j - ga) ** 2, (s - ga) ** 2
excess = (dj - ds)[head].astype(np.float64)
tot_excess = excess.sum()
print(f"head voxels {head.sum()}, joint MSE {dj[head].mean():.3e}, single MSE {ds[head].mean():.3e}, excess {tot_excess / head.sum():+.3e} per voxel")
bins = [0, 0.1, 0.3, 1e9]
labels = ["flat", "mid", "edge"]
A = np.digitize(ea[head], bins) - 1; B = np.digitize(eb[head], bins) - 1
out = {}
print(f"{'T1w edge':>9s} {'T2s edge':>9s} {'voxels%':>8s} {'excess%':>8s} {'joint-single dB-equiv in bin':>28s}")
for ia, la in enumerate(labels):
    for ib, lb in enumerate(labels):
        m = (A == ia) & (B == ib)
        if m.sum() == 0: continue
        share = excess[m].sum() / tot_excess
        dbin = 10 * np.log10(dj[head][m].mean() / ds[head][m].mean())
        out[f"{la}|{lb}"] = dict(vox=float(m.mean()), excess_share=float(share), mse_ratio_db=float(dbin))
        print(f"{la:>9s} {lb:>9s} {100 * m.mean():8.2f} {100 * share:8.1f} {dbin:+28.3f}")
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
