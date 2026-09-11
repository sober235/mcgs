"""Spec 5.5 on the 2x FOV grid (correction 5, 2026-09-11): ESPIRiT maps for REF, the held-out spokes and the gauge
subsets on RECON_GRID (224^3, 448 mm, 2 mm voxels). ecalib's per-voxel step costs about 1 ms per voxel (hours on
224^3), so the maps are computed on 112^3 over the same 448 mm (4 mm voxels) from the central |k| < 56 samples of the
R0 spokes (the same k-extent as the 112^3 maps: -r 48 at the halved k spacing = -r 24 there), Fourier-interpolated to
224^3 and renormalised to unit sum of squares (the ESPIRiT convention with -c 0.0). Check statistic: the fraction of
the support where the interpolated maps' sum of squares was within 5 % of 1 before renormalisation (gate 0.90; it
measures the interpolation, which the renormalisation then removes; the 112^3 maps have 1.0 by construction)."""
import json
import sys

import h5py
import numpy as np

from common import DERIVED_ROOT, LOGS, TRAJ_R, bart, set_ids, three_plane, work_dir
from cardiac_fr import GRID, KEEP, RECON_GRID

s = sys.argv[1]
ids = set_ids(s, "r0")
wd = work_dir(s)
LOG = LOGS / f"bart_{s}_coilmaps2x.log"
C = 112 - KEEP.start                                     # k-space centre among the kept samples
cal = slice(C - GRID // 2 + 1, C + GRID // 2)            # 111 samples: |k| < 56 in 224-grid units
k = np.load(TRAJ_R, mmap_mode="r")[:, KEEP][:, cal][:, :, ids] * RECON_GRID
bart.write_cfl(wd / "traj_cal2x", bart.bart_traj(np.asarray(k)))
with h5py.File(DERIVED_ROOT / s / "raw.h5", "r") as h:
    y = h["ksp"][0][ids][:, :, KEEP][:, :, cal]
bart.write_cfl(wd / "ksp_cal2x_e0", bart.bart_ksp(y))
del k, y
bart.run(["nufft", "-i", "-x", f"{GRID}:{GRID}:{GRID}", wd / "traj_cal2x", wd / "ksp_cal2x_e0", wd / "coilimg_fov2x"],
         log=LOG)
bart.run(["fft", "-u", "7", wd / "coilimg_fov2x", wd / "kgrid_fov2x"], log=LOG)
bart.run(["ecalib", "-t", "0.001", "-c", "0.0", "-r", "48", "-m", "1", wd / "kgrid_fov2x", wd / "sens_fov2x_112"],
         log=LOG)
bart.interp_maps(wd / "sens_fov2x_112", wd / "sens2x_raw", RECON_GRID, wd, log=LOG)
sens = bart.read_cfl(wd / "sens2x_raw")
sos = np.sqrt((np.abs(sens) ** 2).sum(-1))
rss = np.sqrt((np.abs(bart.read_cfl(wd / "coilimg_fov2x")) ** 2).sum(-1))       # 4 mm voxels, 448 mm
support = (rss > 0.05 * np.quantile(rss, 0.999)).repeat(2, 0).repeat(2, 1).repeat(2, 2)
frac = float(np.mean(np.abs(sos[support] - 1) < 0.05))
sens /= np.maximum(sos, 1e-6)[..., None]
bart.write_cfl(DERIVED_ROOT / s / "sens2x", sens)
rep = {"subject": s, "n_spokes": int(ids.size), "calib_samples": [cal.start, cal.stop], "espirit_grid": GRID,
       "sens_shape": list(sens.shape), "support_frac": float(support.mean()),
       "sos_within_5pct_in_support_before_renorm": frac, "renormalised": True}
(LOGS / f"coilmaps2x_{s}.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1), flush=True)
three_plane({"RSS coil images (4 mm, 448 mm)": rss.repeat(2, 0).repeat(2, 1).repeat(2, 2), "|map 0|": sens[..., 0],
             "|map 8|": sens[..., 8], "|map 16|": sens[..., 16]}, f"{s}_coilmaps2x.png",
            title=f"{s} ESPIRiT maps on the 2x FOV grid (echo 0, {ids.size} spokes)")
sys.exit(0 if frac >= 0.90 else 1)
