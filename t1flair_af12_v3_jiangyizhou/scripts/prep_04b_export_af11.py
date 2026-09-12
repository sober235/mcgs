#!/usr/bin/env python3
"""prep_04b_export_af11.py -- the AF11 vd-Poisson mask (FLAIR channel only, user request 2026-09-12) broadcast along the
readout to the 3-D mask h5 the trainer reads, exactly as prep_04_export.py did for AF12; plus the effective FLAIR
acceleration against its own acquired support (elliptical shutter), read from one PE plane of the FLAIR k-space."""
import json
import h5py, numpy as np, scipy.io
R = "/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou"
m2 = scipy.io.loadmat(f"{R}/data/mask_vdpois_256x176_af11.mat")["mask"].astype(np.float32)      # (256,176) = (ky,kz)
m12 = scipy.io.loadmat(f"{R}/data/mask_vdpois_256x176_af12.mat")["mask"].astype(np.float32)
mask3d = np.broadcast_to(m2[None, :, :], (256, 256, 176)).astype(np.uint8)
with h5py.File(f"{R}/data/mask_af11_3d.h5", "w") as h:
    h.create_dataset("mask", data=mask3d, compression="gzip", compression_opts=1)
with h5py.File(f"{R}/data/ksp_FLAIR.h5", "r") as h:
    key = [k for k in h.keys()][0]; ds = h[key]; print("ksp key", key, ds.shape)
    plane = ds[0, 128] if ds.shape[0] < ds.shape[1] else ds[128, 0]     # one coil, the central readout position -> (ky,kz)
support = np.abs(plane) > 0
rep = dict(mask2d_samples=int(m2.sum()), mask2d_frac=float(m2.mean()), mask2d_AF_rect=float(1 / m2.mean()),
           overlap_with_af12_mask=int((m2 * m12).sum()), af12_mask_samples=int(m12.sum()),
           FLAIR=dict(support_frac=float(support.mean()), sampled_frac_of_plane=float((m2 * support).mean()),
                      AF_vs_full_rect=float(1 / (m2 * support).mean()), AF_vs_own_support=float(support.mean() / (m2 * support).mean()),
                      af12_AF_vs_own_support=float(support.mean() / (m12 * support).mean())),
           mask3d_path=f"{R}/data/mask_af11_3d.h5")
json.dump(rep, open(f"{R}/logs/prep_04b_export_af11.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
