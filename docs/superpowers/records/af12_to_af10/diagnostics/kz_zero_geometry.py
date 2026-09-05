"""Read-only diagnostic v3 (fast): numerically-zero region (|y| < 1e-10*max) in 3-D k-space, HYW T1 / FLAIR.
Per PE point: readout extent of the nonzero support; regress extent^2 on (di^2, dj^2) -> ellipsoid semi-axes."""
import sys, os, numpy as np, yaml, torch
ROOT = "/data0/congcong/code/GS/Paired_ARC_GS_general_framework"
sys.path.insert(0, ROOT + "/paired_gs")
from trainers.joint_trainer import _load_dataset
from data.private_paths import resolve_subject_paths
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
cfg = yaml.safe_load(open(ROOT + "/configs/winner_v2_edgeTV.yaml"))
outdir = "/home/congcongliu/figs/af12_to_af10"; os.makedirs(outdir, exist_ok=True)
REL = 1e-10
for sub, con in (("plain", "T1"), ("contrast", "FLAIR")):
    dc_ = dict(cfg["data"][sub])
    for k in ("data_path", "csm_path", "mask_path"): dc_[k] = resolve_subject_paths(dc_[k])
    ds = _load_dataset(dc_)
    yf = ds.kspace_multicoil; m = ds.mask; ro = int(ds.readout_axis)
    a3 = torch.movedim(yf.abs().amax(0), ro, -1)            # (P0,P1,R) max over coils
    gmax = float(a3.max()); tiny = (a3 < REL * gmax).numpy(); P0, P1, R = tiny.shape
    nz = ~tiny
    print(f"\n===== {con}: max|y| {gmax:.3e}, min nonzero |y| {float(a3[a3>0].min()):.3e}, tiny voxel fraction {tiny.mean():.4f} =====", flush=True)
    e_pe = (torch.movedim((yf.abs() ** 2).sum(0), ro, -1).sum(-1)).numpy(); dc = np.unravel_index(int(np.argmax(e_pe)), e_pe.shape)
    e_ro = (torch.movedim((yf.abs() ** 2).sum(0), ro, -1).sum((0, 1))).numpy(); dcr = int(np.argmax(e_ro))
    frac_line = tiny.mean(-1); full_line = frac_line >= 0.999
    print(f"  DC={dc} readout DC={dcr} array centre={(P0//2, P1//2, R//2)}", flush=True)
    print(f"  PE points: whole line tiny {full_line.mean():.4f} | any tiny {(frac_line>0).mean():.4f} | >=50% tiny {(frac_line>=0.5).mean():.4f}", flush=True)
    # readout support per PE line: indices of nonzero samples -> [lo, hi]; extent = min(dcr-lo, hi-dcr)
    idx = np.arange(R)
    lo = np.where(nz.any(-1), np.argmax(nz, axis=-1), -1)                       # first nonzero
    hi = np.where(nz.any(-1), R - 1 - np.argmax(nz[:, :, ::-1], axis=-1), -1)  # last nonzero
    print(f"  readout support of nonzero samples at DC line: [{lo[dc]}..{hi[dc]}] of 0..{R-1}; at PE corner (0,0): [{lo[0,0]}..{hi[0,0]}]", flush=True)
    # interior tiny samples (holes inside the support)?
    inside = np.zeros_like(tiny)
    for i in range(P0):
        for j in range(P1):
            if lo[i, j] >= 0: inside[i, j, lo[i, j]:hi[i, j] + 1] = True
    holes = tiny & inside
    print(f"  tiny samples INSIDE the per-line support (holes): {holes.sum()} of {inside.sum()} ({holes.sum()/max(inside.sum(),1):.5f})", flush=True)
    # regression: ext^2 = c^2 * (1 - (di/a)^2 - (dj/b)^2)  ->  ext^2 = A + B*di^2 + C*dj^2  with A=c^2, B=-c^2/a^2, C=-c^2/b^2
    ok = lo >= 0
    ii, jj = np.meshgrid(np.arange(P0) - dc[0], np.arange(P1) - dc[1], indexing="ij")
    ext = np.minimum(dcr - lo, hi - dcr).astype(float)          # symmetric half-extent about the readout DC
    partial = ok & ((lo > 0) | (hi < R - 1))                     # lines whose support does NOT reach both array ends
    print(f"  lines with nonzero support reaching both readout ends: {(ok & ~partial).mean():.4f}; partially zeroed lines: {partial.mean():.4f}; fully tiny lines: {(~ok).mean():.4f}", flush=True)
    if partial.sum() > 50:
        X = np.stack([np.ones(partial.sum()), ii[partial] ** 2, jj[partial] ** 2], 1); yv = ext[partial] ** 2
        coef, *_ = np.linalg.lstsq(X, yv, rcond=None); A, B, C = coef
        c = np.sqrt(max(A, 1e-9)); a = np.sqrt(max(-A / B, 1e-9)) if B < 0 else float("inf"); b = np.sqrt(max(-A / C, 1e-9)) if C < 0 else float("inf")
        resid = yv - X @ coef
        print(f"  ellipsoid fit on partially-zeroed lines: a(axis0)={a:.1f} px (rho {a/(P0/2):.3f}), b(axis1)={b:.1f} px (rho {b/(P1/2):.3f}), c(readout)={c:.1f} px (rho {c/(R/2):.3f}); rms resid of ext^2 {np.sqrt(np.mean(resid**2)):.1f} vs ext^2 scale {np.mean(yv):.1f}", flush=True)
    # sampled view: among acquired PE points, how many are whole-line tiny / partial
    m2 = torch.movedim(m > 0.5, ro, -1).any(-1).numpy()
    print(f"  acquired PE points {m2.sum()}: whole-line tiny {(full_line & m2).sum()}, partially zeroed {(partial & m2).sum()}, clean {(ok & ~partial & m2).sum()}", flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    im = axes[0].imshow(frac_line.T, cmap="viridis", origin="lower", vmin=0, vmax=1); axes[0].set_title(f"{con}: fraction of readout samples numerically zero, per PE point"); plt.colorbar(im, ax=axes[0]); axes[0].plot(dc[0], dc[1], "r+", ms=12)
    axes[1].imshow(tiny[:, :, dcr].T, cmap="gray_r", origin="lower"); axes[1].set_title(f"zero map at readout DC (index {dcr}); black = zero")
    axes[2].imshow(tiny[dc[0], :, :].T, cmap="gray_r", origin="lower", aspect="auto"); axes[2].set_title("zero map: PE axis1 (x) vs readout (y) at PE axis0 = DC")
    out = f"{outdir}/kz_tiny_geometry_{con}.png"; fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig); print("  saved", out, flush=True)
