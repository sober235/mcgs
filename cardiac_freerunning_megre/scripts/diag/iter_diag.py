import sys, numpy as np
from scipy import ndimage
sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from diag_artefact.headonly import head_mask
R = "/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/recon/V07"
x100 = np.load(f"{R}/ref_e0.npy"); x60 = np.load(f"{R}/ref_it60_e0.npy")
mag = np.abs(x100); body = head_mask(mag / np.quantile(mag, 0.999))
d = x100 - x60
def smooth(v): return ndimage.gaussian_filter(v.real, 2) + 1j * ndimage.gaussian_filter(v.imag, 2)
lo_d, hi_d = smooth(d), d - smooth(d)
n = np.linalg.norm
print(f"body fraction {body.mean():.3f}")
print(f"||x100-x60||/||x100|| (body)         = {n(d[body]) / n(x100[body]):.4f}")
print(f"  low-pass part (gauss sigma 2)       = {n(lo_d[body]) / n(x100[body]):.4f}")
print(f"  high-pass remainder                 = {n(hi_d[body]) / n(x100[body]):.4f}")
print(f"||x100|| / ||x60|| (body)             = {n(x100[body]) / n(x60[body]):.4f}")
hp = lambda v: v - smooth(v)
print(f"high-pass energy x60 {n(hp(x60)[body]):.4g} -> x100 {n(hp(x100)[body]):.4g} (growth = noise amplification if large)")
print(f"low-pass relative change (smooth x100 vs smooth x60) = {n((smooth(x100) - smooth(x60))[body]) / n(smooth(x100)[body]):.4f}")
