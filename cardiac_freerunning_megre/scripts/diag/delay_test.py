# Diagnostic (scratchpad), 2026-09-11: how much of the in-sample held-out residual floor of REF (echo 0, 0.4516)
# comes from a gradient-delay-like trajectory offset? Fit the echo-0 peak shift of the REF spokes with an even
# (delay-tensor) model in the spoke direction, shift REF and held-out trajectories by it, rerun REF echo 0.
import json, sys
import numpy as np
sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr import bart
from cardiac_fr.gauge import image_metrics, kspace_nrmse, norm_scale
D = "/data2/congcong/data/cardiac_3d_cine_public/derived/megre/V07"; W = D + "/bart"
S = "/tmp/claude-1002/-home-congcongliu--claude/4016652d-b661-45ca-ba05-927ae4a7ee12/scratchpad/trajdiag"
T = "/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE"; R = T + "/recon/V07"
scal = json.load(open(R + "/ref_scaling.json"))["0"]
lam = json.load(open(T + "/logs/checks_V07.json"))["lam"]
C = 109                                                     # k-space centre in the kept samples 3..220


def geometry(name):
    t = bart.read_cfl(f"{W}/traj_{name}").real              # (3, 218, n)
    u = t[:, -1, :] - t[:, 0, :]
    u /= np.linalg.norm(u, axis=0, keepdims=True)
    X = np.stack([u[0]**2, u[1]**2, u[2]**2, u[0]*u[1], u[0]*u[2], u[1]*u[2]], 1)
    return t, u, X


def bands(y, p, label):
    r = p - y; ey = (np.abs(y) ** 2).sum()
    ey_s = (np.abs(y) ** 2).sum(axis=(0, 2, 3)); er_s = (np.abs(r) ** 2).sum(axis=(0, 2, 3))
    dist = np.abs(np.arange(218) - C)
    out = [f"{label}: nrmse {np.sqrt(er_s.sum() / ey):.5f}"]
    for lo, hi in ((0, 4), (4, 12), (12, 28), (28, 113)):
        sel = (dist >= lo) & (dist < hi)
        out.append(f"|k| {lo}-{hi - 1}: res/data {np.sqrt(er_s[sel].sum() / ey_s[sel].sum()):.3f}")
    print("  " + ";  ".join(out), flush=True)


y_ho = bart.read_cfl(f"{W}/ksp_heldout_e0")
bands(y_ho, bart.read_cfl(f"{W}/pred_ref_e0"), "nominal REF, in-sample held-out")
p = bart.read_cfl(f"{W}/pred_ref_e0")
num = (np.abs(p - y_ho) ** 2).sum(axis=(0, 1, 2)); den = (np.abs(y_ho) ** 2).sum(axis=(0, 1, 2))
print("  per-coil residual/data:", " ".join(f"{v:.2f}" for v in np.sqrt(num / den)), flush=True)

t, u, X = geometry("ref")
y = bart.read_cfl(f"{W}/ksp_ref_e0")
m = (np.abs(y) ** 2).sum(axis=3)[0]                         # (218, n)
k = np.argmax(m[C - 12:C + 13], axis=0) + C - 12
cols = np.arange(m.shape[1])
a, b, c = (np.log(m[k + d, cols]) for d in (-1, 0, 1))
s = k + 0.5 * (a - c) / (a - 2 * b + c) - C
coef, *_ = np.linalg.lstsq(X, s, rcond=None)
print(f"REF spokes echo 0: peak shift median {np.median(s):+.3f} sd {s.std():.3f}; delay model [xx yy zz xy xz yz] "
      f"{np.round(coef, 3)}, residual sd {np.std(s - X @ coef):.3f}", flush=True)

for name in ("ref", "heldout"):
    t, u, X = geometry(name)
    tc = t - 0.5 * (X @ coef)[None, None, :] * u[:, None, :]      # one sample = 0.5 BART units along the spoke
    bart.write_cfl(f"{S}/traj_{name}_corr", tc.astype(np.complex64))
info = bart.pics(f"{W}/ksp_ref_e0", f"{D}/sens", f"{S}/ref_corr_e0", f"{S}/traj_ref_corr", reg="l2", lam=lam, iters=100, w=scal)
bart.forward(f"{S}/traj_heldout_corr", f"{S}/ref_corr_e0", f"{D}/sens", f"{S}/pred_corr", S)
bands(y_ho, bart.read_cfl(f"{S}/pred_corr"), "delay-corrected REF, in-sample held-out")
ref = np.abs(np.load(f"{R}/ref_e0.npy")); sc = norm_scale(ref)
x = np.abs(bart.read_cfl(f"{S}/ref_corr_e0"))
mm = image_metrics(x / sc, ref / sc)
print(f"  corrected vs nominal REF image: body_psnr {mm['body_psnr']:.2f} ssim {mm['body_ssim']:.4f} ({info['seconds']:.0f} s)", flush=True)
np.save(f"{S}/ref_corr_e0.npy", bart.read_cfl(f"{S}/ref_corr_e0").astype(np.complex64))
print("DELAY_TEST_DONE", flush=True)
