# Diagnostic (scratchpad), 2026-09-11: how much of each coil's signal lies outside the 220 mm reconstruction FOV?
# Density-compensated, low-pass (|k| < half of kmax) adjoint NUFFT of the REF spokes (echo 0) onto a 2x FOV grid
# (224^3, same 2 mm voxels, which the 2x oversampled readout supports); energy outside the central 112^3 cube per
# coil, and its rank correlation with the per-coil in-sample held-out residual of REF.
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr
sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr import bart
D = "/data2/congcong/data/cardiac_3d_cine_public/derived/megre/V07"; W = D + "/bart"
S = "/tmp/claude-1002/-home-congcongliu--claude/4016652d-b661-45ca-ba05-927ae4a7ee12/scratchpad/trajdiag"
t = bart.read_cfl(f"{W}/traj_ref")                               # (3, 218, n), BART units of the 112 grid
y = bart.read_cfl(f"{W}/ksp_ref_e0")                             # (1, 218, n, 24)
kr = np.sqrt((t.real ** 2).sum(axis=0))
w = (kr ** 2 + 0.25) * np.where(kr < 28, np.cos(np.pi / 2 * kr / 28) ** 2, 0.0)
bart.write_cfl(f"{S}/traj2x", 2 * t)                             # FOV doubled -> coordinates doubled
bart.write_cfl(f"{S}/ksp_w", (y * w[None, :, :, None]).astype(np.complex64))
bart.run(["nufft", "-a", "-x", "224:224:224", f"{S}/traj2x", f"{S}/ksp_w", f"{S}/img2x"])
x = bart.read_cfl(f"{S}/img2x")                                  # (224, 224, 224, 24)
e = np.abs(x) ** 2
inner = e[56:168, 56:168, 56:168].sum(axis=(0, 1, 2))
total = e.sum(axis=(0, 1, 2))
frac_out = 1 - inner / total
rss = np.sqrt(e.sum(axis=-1))
print("RSS energy outside the 112^3 FOV: %.3f" % (1 - (rss[56:168, 56:168, 56:168] ** 2).sum() / (rss ** 2).sum()))
for ax, name in ((0, "axis0 (LR)"), (1, "axis1 (AP)"), (2, "axis2 (SI)")):
    prof = (rss ** 2).sum(axis=tuple(i for i in range(3) if i != ax))
    print(f"  along {name}: {(prof[:56].sum() + prof[168:].sum()) / prof.sum():.3f} of the energy beyond the FOV on this axis")
p = bart.read_cfl(f"{W}/pred_ref_e0"); yh = bart.read_cfl(f"{W}/ksp_heldout_e0")
res = np.sqrt((np.abs(p - yh) ** 2).sum(axis=(0, 1, 2)) / (np.abs(yh) ** 2).sum(axis=(0, 1, 2)))
print("per coil  outside-FOV energy fraction:", " ".join(f"{v:.2f}" for v in frac_out))
print("per coil  in-sample residual/data     :", " ".join(f"{v:.2f}" for v in res))
rho, pv = spearmanr(frac_out, res)
print(f"Spearman rho {rho:.2f} (p = {pv:.2g})")
vmax = np.quantile(rss, 0.995)
fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
for a_, (img, lab) in zip(axs, ((rss[:, :, 112].T, "axis2 = 112 (x axis0, y axis1)"), (rss[:, 112, :].T, "axis1 = 112 (x axis0, y axis2)"),
                                 (rss[112, :, :].T, "axis0 = 112 (x axis1, y axis2)"))):
    a_.imshow(img, cmap="gray", vmin=0, vmax=vmax, origin="lower")
    a_.add_patch(Rectangle((56, 56), 112, 112, fill=False, ec="y", lw=1.2))
    a_.set_title(lab, fontsize=9)
fig.suptitle("V07 REF echo 0, low-pass gridding on a 2x FOV grid (448 mm); yellow = the 220 mm reconstruction FOV")
fig.tight_layout(); fig.savefig("/home/congcongliu/figs/cardiac_fr/V07_fov2x_gridding.png", dpi=90)
print("http://localhost:8765/cardiac_fr/V07_fov2x_gridding.png\n/home/congcongliu/figs/cardiac_fr/V07_fov2x_gridding.png")
print("OUTFOV_DONE", flush=True)
