"""Where does MC-GS joint v3 lose head SSIM to L1-ESPIRiT? Scorer-identical 3-D SSIM map (uniform 7^3 window,
K1 .01, K2 .03, data_range max|GT|, sample covariance), split into luminance l and contrast-structure cs, and cs
further into correlation rho and variance match c = 2 sx sy / (sx^2 + sy^2). Binned by quintiles of the GT's
gradient magnitude (Q1 flattest ... Q5 edges) inside the head."""
import sys, json
import numpy as np
from pathlib import Path
from scipy import ndimage
R = Path("/data0/congcong/code/GS/Results_20260908/T1T2_AF12_retro/recon")
W, K1, K2 = 7, 0.01, 0.03
def load(m, c):
    for suf in ("_complex.npy", "_mag.npy"):
        p = R / f"{m}_{c}{suf}"
        if p.exists():
            return np.abs(np.load(p)).astype(np.float64)
def parts(x, y, L):
    f = lambda a: ndimage.uniform_filter(a, size=W)
    cn = W ** 3 / (W ** 3 - 1)
    ux, uy = f(x), f(y)
    vx, vy, vxy = cn * (f(x * x) - ux * ux), cn * (f(y * y) - uy * uy), cn * (f(x * y) - ux * uy)
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    l = (2 * ux * uy + C1) / (ux * ux + uy * uy + C1)
    cs = (2 * vxy + C2) / (vx + vy + C2)
    sx, sy = np.sqrt(np.maximum(vx, 0)), np.sqrt(np.maximum(vy, 0))
    return dict(S=l * cs, l=l, cs=cs, rho=vxy / (sx * sy + 1e-12), c=2 * sx * sy / (vx + vy + 1e-12), sr=sx / (sy + 1e-12))
for c, arms in (("T1", ["gs_joint_v3", "l1espirit"]), ("T2", ["gs_joint_v3", "gs_single_v3", "l1espirit"])):
    gt = load("gt", c); L = float(gt.max()); head = gt > 0.05 * float(np.quantile(gt, 0.999))
    gm = ndimage.gaussian_gradient_magnitude(gt, sigma=1.0)[head]
    qb = np.quantile(gm, [0.2, 0.4, 0.6, 0.8]); gbin = np.digitize(gm, qb)
    P = {m: {k: v[head] for k, v in parts(load(m, c), gt, L).items()} for m in arms}
    print(f"\n=== {c}: head SSIM (scorer-identical) " + ", ".join(f"{m} {P[m]['S'].mean():.4f}" for m in arms))
    ref = "l1espirit"
    for m in [a for a in arms if a != ref]:
        d = P[m]["S"] - P[ref]["S"]
        print(f"  {m} - {ref}: total {d.mean():+.5f} = sum of bin contributions; mean dl {(P[m]['l'] - P[ref]['l']).mean():+.5f}, mean dcs {(P[m]['cs'] - P[ref]['cs']).mean():+.5f}")
        print("   bin (|grad GT|)   frac  contrib   dS_mean   dl       dcs      rho m/ref       c m/ref        sigma ratio m/ref")
        for b in range(5):
            k = gbin == b
            print(f"   Q{b + 1}            {k.mean():.2f}  {d[k].sum() / d.size:+.5f}  {d[k].mean():+.5f}  "
                  f"{(P[m]['l'][k] - P[ref]['l'][k]).mean():+.5f}  {(P[m]['cs'][k] - P[ref]['cs'][k]).mean():+.5f}  "
                  f"{np.median(P[m]['rho'][k]):.4f}/{np.median(P[ref]['rho'][k]):.4f}  {np.median(P[m]['c'][k]):.4f}/{np.median(P[ref]['c'][k]):.4f}  "
                  f"{np.median(P[m]['sr'][k]):.3f}/{np.median(P[ref]['sr'][k]):.3f}")
