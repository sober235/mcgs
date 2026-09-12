"""Scorer-identical head-SSIM breakdown for any topic/contrast: ours minus each reference, by GT-gradient quintile.
usage: ssim_topic.py <topic dir> <contrast> <ours> <ref1,ref2,...>"""
import sys
import numpy as np
from pathlib import Path
from scipy import ndimage
T, c, ours, refs = Path(sys.argv[1]) / "recon", sys.argv[2], sys.argv[3], sys.argv[4].split(",")
W, K1, K2 = 7, 0.01, 0.03
def load(m):
    if m.startswith("npz:"):
        _, p, k = m.split(":"); return np.abs(np.asarray(np.load(p)[k], dtype=np.float64))
    for suf in ("_complex.npy", "_mag.npy"):
        p = T / f"{m}_{c}{suf}"
        if p.exists():
            return np.abs(np.load(p)).astype(np.float64)
    raise FileNotFoundError(m)
def parts(x, y, L):
    f = lambda a: ndimage.uniform_filter(a, size=W)
    cn = W ** 3 / (W ** 3 - 1)
    ux, uy = f(x), f(y)
    vx, vy, vxy = cn * (f(x * x) - ux * ux), cn * (f(y * y) - uy * uy), cn * (f(x * y) - ux * uy)
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    l = (2 * ux * uy + C1) / (ux * ux + uy * uy + C1); cs = (2 * vxy + C2) / (vx + vy + C2)
    sx, sy = np.sqrt(np.maximum(vx, 0)), np.sqrt(np.maximum(vy, 0))
    return dict(S=l * cs, l=l, cs=cs, rho=vxy / (sx * sy + 1e-12), sr=sx / (sy + 1e-12))
gt = load("gt"); L = float(gt.max()); head = gt > 0.05 * float(np.quantile(gt, 0.999))
gm = ndimage.gaussian_gradient_magnitude(gt, sigma=1.0)[head]; gbin = np.digitize(gm, np.quantile(gm, [0.2, 0.4, 0.6, 0.8]))
P = {m: {k: v[head] for k, v in parts(load(m), gt, L).items()} for m in [ours] + refs}
print(f"=== {T.parent.name} {c}: head SSIM " + ", ".join(f"{m} {P[m]['S'].mean():.4f}" for m in P))
for r in refs:
    d = P[ours]["S"] - P[r]["S"]
    print(f"  {ours} - {r}: {d.mean():+.5f}; mean dl {(P[ours]['l'] - P[r]['l']).mean():+.5f}, mean dcs {(P[ours]['cs'] - P[r]['cs']).mean():+.5f}")
    print("   bin   contrib   dS_mean   dl        dcs       rho ours/ref     sigma ratio ours/ref")
    for q in range(5):
        k = gbin == q
        print(f"   Q{q + 1}   {d[k].sum() / d.size:+.5f}  {d[k].mean():+.5f}  {(P[ours]['l'][k] - P[r]['l'][k]).mean():+.5f}  {(P[ours]['cs'][k] - P[r]['cs'][k]).mean():+.5f}  "
              f"{np.median(P[ours]['rho'][k]):.4f}/{np.median(P[r]['rho'][k]):.4f}  {np.median(P[ours]['sr'][k]):.3f}/{np.median(P[r]['sr'][k]):.3f}")
