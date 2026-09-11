# Diagnostic, 2026-09-11: split the error of every self-check reconstruction against REF into a low-pass part
# (Gaussian sigma 2 voxels) and the rest, inside the body mask, and compare each image's high-pass content and
# mean intensity with REF's. Reads recon/V07 only.
import sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr.gauge import norm_scale
from diag_artefact.headonly import head_mask
R = "/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/recon/V07"
for e in (0, 4):
    ref = np.abs(np.load(f"{R}/ref_e{e}.npy")); s = norm_scale(ref); ref = ref / s
    body = head_mask(ref)
    lp = lambda v: ndimage.gaussian_filter(v, 2)
    ref_hp = np.linalg.norm((ref - lp(ref))[body]); nb = np.linalg.norm(ref[body])
    print(f"echo {e}: body {body.mean():.3f}, |hp(ref)|/|ref| {ref_hp / nb:.4f}")
    for m in ("cg", "l1"):
        for a in ("af8", "af12", "af16"):
            x = np.abs(np.load(f"{R}/{m}_{a}_e{e}.npy")) / s
            err = x - ref
            print(f"  {m} {a:5s} err lp {np.linalg.norm(lp(err)[body]) / nb:.4f}  hp {np.linalg.norm((err - lp(err))[body]) / nb:.4f}"
                  f"  |hp(x)|/|hp(ref)| {np.linalg.norm((x - lp(x))[body]) / ref_hp:.3f}  mean(x)/mean(ref) {x[body].mean() / ref[body].mean():.4f}")
