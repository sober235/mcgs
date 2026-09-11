import json, os, sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr import bart
from diag_artefact.headonly import head_mask
D = "/data2/congcong/data/cardiac_3d_cine_public/derived/megre/V07"
W = D + "/bart"
S = "/tmp/claude-1002/-home-congcongliu--claude/4016652d-b661-45ca-ba05-927ae4a7ee12/scratchpad/lamscan"
w = json.load(open("/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/recon/V07/ref_scaling.json"))["0"]
def smooth(v):
    return ndimage.gaussian_filter(v.real, 2) + 1j * ndimage.gaussian_filter(v.imag, 2)
n = np.linalg.norm
for lam in (1e-2, 3e-2):
    xs, secs = {}, 0.0
    for it in (60, 100):
        out = f"{S}/rec_lam{lam:g}_it{it}"
        secs += bart.pics(f"{W}/ksp_ref_e0", f"{D}/sens", out, f"{W}/traj_ref", reg="l2", lam=lam, iters=it, w=w)["seconds"]
        xs[it] = bart.read_cfl(out)
    mag = np.abs(xs[100]); body = head_mask(mag / np.quantile(mag, 0.999))
    d = xs[100] - xs[60]
    print(f"lambda {lam:g}: 60 vs 100 body rel diff {n(d[body]) / n(xs[100][body]):.4f} "
          f"(low-pass {n(smooth(d)[body]) / n(xs[100][body]):.4f}, high-pass {n((d - smooth(d))[body]) / n(xs[100][body]):.4f}), "
          f"{secs:.0f} s", flush=True)
    np.save(f"{S}/x100_lam{lam:g}.npy", xs[100].astype(np.complex64))
