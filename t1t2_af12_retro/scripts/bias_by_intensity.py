"""Signed error and local-mean error of each arm by GT-intensity decile inside the head; share of the flattest
fifth (GT gradient quintile 1) in each decile; mean SSIM luminance term. usage: bias_by_intensity.py <topic dir> <contrast> <arm1,arm2,...>"""
import sys
import numpy as np
from pathlib import Path
from scipy import ndimage
T, c, arms = Path(sys.argv[1]) / "recon", sys.argv[2], sys.argv[3].split(",")
def load(m):
    if m.startswith("npz:"):
        _, p, k = m.split(":"); return np.abs(np.asarray(np.load(p)[k], dtype=np.float64))
    for suf in ("_complex.npy", "_mag.npy"):
        p = T / f"{m}_{c}{suf}"
        if p.exists():
            return np.abs(np.load(p)).astype(np.float64)
    raise FileNotFoundError(m)
gt = load("gt"); L = float(gt.max()); head = gt > 0.05 * float(np.quantile(gt, 0.999)); C1 = (0.01 * L) ** 2
f = lambda a: ndimage.uniform_filter(a, size=7)
uy = f(gt)[head]; g = gt[head]
gm = ndimage.gaussian_gradient_magnitude(gt, sigma=1.0)[head]; flat = gm <= np.quantile(gm, 0.2)
ib = np.digitize(g, np.quantile(g, np.linspace(0.1, 0.9, 9)))
A = {}
for m in arms:
    x = load(m); ux = f(x)[head]
    A[m] = dict(e=(x[head] - g) / L, de=(ux - uy) / L, l=(2 * ux * uy + C1) / (ux * ux + uy * uy + C1))
print(f"=== {T.parent.name} {c}: signed error and local-mean error in % of max|GT|, SSIM luminance term; flat = GT-gradient quintile 1")
print("  decile  GT/L   flat share | " + " | ".join(f"{m[:10]:>10s}: err  lmean   l" for m in arms))
for q in range(10):
    k = ib == q
    row = f"  D{q + 1:<2d}    {g[k].mean() / L:.3f}  {flat[k].mean():.2f}      | "
    row += " | ".join(f"{100 * A[m]['e'][k].mean():+6.2f} {100 * A[m]['de'][k].mean():+6.2f} {A[m]['l'][k].mean():.4f}" for m in arms)
    print(row)
kk = flat
print("  flat Q1 all deciles        | " + " | ".join(f"{100 * A[m]['e'][kk].mean():+6.2f} {100 * A[m]['de'][kk].mean():+6.2f} {A[m]['l'][kk].mean():.4f}" for m in arms))
