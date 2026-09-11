"""Feasibility test for option 1 of the 2026-09-11 report (user: "continue"): V07 echo 0 on a 2x FOV grid,
224^3 with the same 2 mm voxels (448 mm, which the 2x oversampled readout supports).
Coil maps: inverse NUFFT of the R0 spokes' central 63 samples on a 64^3 grid over the same 448 mm (7 mm), unitary
FFT, ESPIRiT with the pipeline's flags and the same calibration k-extent as on the 112^3 grid (-r 48 at the halved
k spacing = -r 24 on 112^3), maps Fourier-interpolated to 224^3 (ESPIRiT on 224^3 would take hours: ecalib alone
took 1,487 s on 112^3). REF: its 6,813 spokes, lambda = 1e-3 x lambda_max of the 2x
operator, 100 CG steps. Then REF's in-sample held-out residual (same k bands and per-coil split as for the 112^3
REF), the FOV-edge band of the central 112^3 against the 112^3 REF, and time and peak memory of every BART step."""
import json
import resource
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

T = "/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE"
sys.path.insert(0, T + "/scripts")
from common import LOGS, bart, recon_dir, save_fig  # noqa: E402
from cardiac_fr.gauge import norm_scale  # noqa: E402

D = Path("/data2/congcong/data/cardiac_3d_cine_public/derived/megre/V07")
W, F = D / "bart", D / "fov2x"
F.mkdir(exist_ok=True)
BLOG = LOGS / "bart_V07_fov2x_test.log"
C = 109                                                    # k-space centre among the kept samples 3..220
steps = {}


def step(name, fn):
    t0 = time.time()
    out = fn()
    steps[name] = {"seconds": round(time.time() - t0, 1),
                   "peak_rss_gb_any_bart_so_far": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6, 2)}
    print(f"{name}: {steps[name]}", flush=True)
    return out


t_r0, y_r0 = bart.read_cfl(W / "traj_r0"), bart.read_cfl(W / "ksp_r0_e0")
cal = slice(C - 31, C + 32)                                # |k| < 32 points on the 64^3 grid
bart.write_cfl(F / "traj_cal", 2 * t_r0[:, cal, :])        # same k positions in units of the doubled FOV
bart.write_cfl(F / "ksp_cal", y_r0[:, cal])
del t_r0, y_r0
for name in ("ref", "heldout"):
    bart.write_cfl(F / f"traj2x_{name}", 2 * bart.read_cfl(W / f"traj_{name}"))

step("coil images 64^3 over 448 mm (nufft -i)",
     lambda: bart.run(["nufft", "-i", "-x", "64:64:64", F / "traj_cal", F / "ksp_cal", F / "coilimg64"], log=BLOG))
step("fft", lambda: bart.run(["fft", "-u", "7", F / "coilimg64", F / "kgrid64"], log=BLOG))
step("ecalib on 64^3 (-r 48)",
     lambda: bart.run(["ecalib", "-t", "0.001", "-c", "0.0", "-r", "48", "-m", "1", F / "kgrid64", F / "sens64"], log=BLOG))
step("maps Fourier-interpolated to 224^3",
     lambda: (bart.run(["fft", "-u", "7", F / "sens64", F / "ksens64"], log=BLOG),
              bart.run(["resize", "-c", "0", "224", "1", "224", "2", "224", F / "ksens64", F / "ksens224"], log=BLOG),
              bart.run(["fft", "-u", "-i", "7", F / "ksens224", F / "sens224u"], log=BLOG)))
bart.write_cfl(F / "sens2x", (bart.read_cfl(F / "sens224u") * (224 / 64) ** 1.5).astype(np.complex64))   # unitary FFTs
lmax = step("lambda_max of the 2x REF operator",
            lambda: bart.max_eigenvalue(W / "ksp_ref_e0", F / "sens2x", F / "traj2x_ref", F, log=BLOG))
info = step("REF echo 0 on 224^3, 100 CG steps",
            lambda: bart.pics(W / "ksp_ref_e0", F / "sens2x", F / "rec2x_ref_e0", F / "traj2x_ref", reg="l2",
                              lam=1e-3 * lmax, iters=100, debug=True, log=BLOG))
step("forward onto the held-out spokes",
     lambda: bart.forward(F / "traj2x_heldout", F / "rec2x_ref_e0", F / "sens2x", F / "pred2x_ref_e0", F, log=BLOG))

y = bart.read_cfl(W / "ksp_heldout_e0")


def bands(p):
    r = p - y
    ey_s, er_s = (np.abs(y) ** 2).sum(axis=(0, 2, 3)), (np.abs(r) ** 2).sum(axis=(0, 2, 3))
    dist = np.abs(np.arange(y.shape[1]) - C)
    out = {"nrmse": float(np.sqrt(er_s.sum() / ey_s.sum()))}
    for lo, hi in ((0, 4), (4, 12), (12, 28), (28, 113)):
        sel = (dist >= lo) & (dist < hi)
        out[f"res_over_data_k{lo}-{hi - 1}"] = float(np.sqrt(er_s[sel].sum() / ey_s[sel].sum()))
    out["per_coil"] = [round(float(v), 3) for v in
                       np.sqrt((np.abs(r) ** 2).sum(axis=(0, 1, 2)) / (np.abs(y) ** 2).sum(axis=(0, 1, 2)))]
    return out


b112, b224 = bands(bart.read_cfl(W / "pred_ref_e0")), bands(bart.read_cfl(F / "pred2x_ref_e0"))
x224 = np.abs(bart.read_cfl(F / "rec2x_ref_e0"))
x112 = np.abs(np.load(recon_dir("V07") / "ref_e0.npy"))
crop = x224[56:168, 56:168, 56:168]                       # the 112^3 grid's FOV (both grids centred at N/2)


def edge_ratio(v):
    """mean |x| in the 4-voxel band along the faces of the 112^3 cube over the body mean inside it"""
    v = v / norm_scale(v)
    band = np.ones(v.shape, bool)
    band[4:-4, 4:-4, 4:-4] = False
    body = (v > 0.05) & ~band
    return float(v[band].mean() / v[body].mean())


sens = bart.read_cfl(F / "sens2x")
sup = x224 > 0.05 * np.quantile(x224, 0.999)
sos = np.sqrt((np.abs(sens) ** 2).sum(-1))
rep = {"subject": "V07", "echo": 0, "grid": 224, "fov_mm": 448, "lambda_max_2x": lmax, "lambda": 1e-3 * lmax,
       "bart_scaling": info["scaling"], "steps": steps,
       "sens_sos_within_5pct_in_support": float(np.mean(np.abs(sos[sup] - 1) < 0.05)),
       "in_sample_heldout_112": b112, "in_sample_heldout_224": b224,
       "fov_edge_band_over_body_112": edge_ratio(x112), "fov_edge_band_over_body_224_central": edge_ratio(crop)}
(LOGS / "diag" / "fov2x_test_V07.json").write_text(json.dumps(rep, indent=1))
print(json.dumps({k: v for k, v in rep.items() if k != "steps"}, indent=1), flush=True)

c224, c112 = (117, 120, 113), (61, 64, 57)                # through the heart-box centre
fig, ax = plt.subplots(3, 3, figsize=(14, 14))
rows = ((x224, c224, "REF 224^3 (448 mm), yellow = 112^3 FOV"), (crop, c112, "central 112^3 of the 224^3 REF"),
        (x112, c112, "REF 112^3 (current pipeline)"))
for r, (v, c, lab) in enumerate(rows):
    v = v / norm_scale(v if v.shape[0] == 112 else crop)
    for k, (img, xl, yl, up) in enumerate(((v[:, :, c[2]].T, "axis0", "axis1", False), (v[:, c[1], :].T, "axis0", "axis2", True),
                                          (v[c[0], :, :].T, "axis1", "axis2", True))):
        a = ax[r, k]
        a.imshow(img, cmap="gray", vmin=0, vmax=1, origin="upper" if up else "lower")
        if v.shape[0] == 224:
            a.add_patch(plt.Rectangle((56, 56), 112, 112, fill=False, ec="y", lw=1))
        a.set_title(f"{lab}\n{('axis2', 'axis1', 'axis0')[k]} = {c[(2, 1, 0)[k]]}", fontsize=8)
        a.set_xlabel(xl, fontsize=7)
        a.set_ylabel(yl, fontsize=7)
        a.tick_params(labelsize=6)
fig.suptitle("V07 REF echo 0: 2x FOV grid against the current 112^3 grid (each row scaled to its P99.9 in the 112^3 FOV)")
fig.tight_layout()
save_fig(fig, "V07_fov2x_ref_e0.png")
print("FOV2X_TEST_DONE", flush=True)
