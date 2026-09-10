"""SSIM-map comparison of the per-contrast run (T1 wavelet 0.06, T2 0.03) against v3 (shared 0.05), same bins as ssim_breakdown.py."""
import numpy as np
from pathlib import Path
from scipy import ndimage
R = Path("/data0/congcong/code/GS/Results_20260908/T1T2_AF12_retro/recon")
NPZ = Path("/data0/congcong/code/GS/Paired_ARC_GS_general_framework/results_arc/ar14_n24_wavP006_C003_full4k_volumes.npz")
W, K1, K2 = 7, 0.01, 0.03
def parts(x, y, L):
    f = lambda a: ndimage.uniform_filter(a, size=W)
    cn = W ** 3 / (W ** 3 - 1)
    ux, uy = f(x), f(y)
    vx, vy, vxy = cn * (f(x * x) - ux * ux), cn * (f(y * y) - uy * uy), cn * (f(x * y) - ux * uy)
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    l = (2 * ux * uy + C1) / (ux * ux + uy * uy + C1); cs = (2 * vxy + C2) / (vx + vy + C2)
    sx, sy = np.sqrt(np.maximum(vx, 0)), np.sqrt(np.maximum(vy, 0))
    return dict(S=l * cs, l=l, cs=cs, rho=vxy / (sx * sy + 1e-12), sr=sx / (sy + 1e-12))
with np.load(NPZ) as d:
    P = {"T1": np.abs(np.asarray(d["recT1"], dtype=np.float64)), "T2": np.abs(np.asarray(d["recFL"], dtype=np.float64))}
for c, lam in (("T1", "0.06 vs 0.05"), ("T2", "0.03 vs 0.05")):
    gt = np.abs(np.load(R / f"gt_{c}_complex.npy")).astype(np.float64); L = float(gt.max())
    head = gt > 0.05 * float(np.quantile(gt, 0.999))
    gm = ndimage.gaussian_gradient_magnitude(gt, sigma=1.0)[head]; gbin = np.digitize(gm, np.quantile(gm, [0.2, 0.4, 0.6, 0.8]))
    a = {k: v[head] for k, v in parts(P[c], gt, L).items()}
    b = {k: v[head] for k, v in parts(np.abs(np.load(R / f"gs_joint_v3_{c}_mag.npy")).astype(np.float64), gt, L).items()}
    d = a["S"] - b["S"]
    print(f"\n=== {c}: wavelet {lam}; head SSIM per-contrast {a['S'].mean():.4f} vs v3 {b['S'].mean():.4f} (diff {d.mean():+.5f})")
    print("   bin   contrib   dS_mean   dl        dcs       rho pc/v3        sigma ratio pc/v3")
    for q in range(5):
        k = gbin == q
        print(f"   Q{q + 1}   {d[k].sum() / d.size:+.5f}  {d[k].mean():+.5f}  {(a['l'][k] - b['l'][k]).mean():+.5f}  {(a['cs'][k] - b['cs'][k]).mean():+.5f}  "
              f"{np.median(a['rho'][k]):.4f}/{np.median(b['rho'][k]):.4f}  {np.median(a['sr'][k]):.3f}/{np.median(b['sr'][k]):.3f}")
