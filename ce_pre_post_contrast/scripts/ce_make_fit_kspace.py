#!/usr/bin/env python3
"""ce_make_fit_kspace.py -- write the 95%-line k-space for the GS held-out runs (2026-09-15).

ce_gs.py reads ksp.mat / ksp2.mat, layout (coil, 256, 176, 232), and infers its sampling mask from the non-zero
samples. The fit-only input is therefore the same array with the 5% held-out lines of
heldout_split/split_95_5_seed0.npz set to zero. It is built from ksp_full.npy (coil, 256, 232, 176), which equals
ksp*.mat after swapping the last two axes (checked 2026-09-15 on readout index 130: relative difference 0.0 for both
scans), and it is written as heldout_split/ksp_fit_{pre,post}.h5, dataset "ksp". The check reads back one readout
position and compares it with ksp*.mat restricted to the fit lines.
"""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ce_common as C  # noqa: E402

MAT = {"pre": "/data1/data1_congcong/data/3D/CE_Data/t1_3D_gre_fsp_sag_iso_ACS_16887049370010/ksp.mat",
       "post": "/data1/data1_congcong/data/3D/CE_Data/t1_3D_gre_fsp_sag_iso_ACS_16887049370060/ksp2.mat"}
fit = np.load(C.SPLIT)["fit"]                                   # (232,176) bool
report = {}
for scan in C.SCANS:
    t0 = time.time()
    out = C.SPLIT.parent / f"ksp_fit_{scan}.h5"
    y = np.load(C.WORK[scan] / "ksp_full.npy", mmap_mode="r")    # (42,256,232,176)
    arr = np.ascontiguousarray(np.transpose(np.asarray(y) * fit, (0, 1, 3, 2)))   # (42,256,176,232)
    tmp = out.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("ksp", data=arr)
    tmp.replace(out)
    del arr
    with h5py.File(out, "r") as f, h5py.File(MAT[scan], "r") as g:
        a = f["ksp"][:, 130]                                     # (42,176,232) complex
        b = g["ksp"][:, 130]
        b = b["real"].astype(np.complex64) + 1j * b["imag"].astype(np.complex64)
        b = b * fit.T                                            # fit plane in the (176,232) layout
        rel = float(np.linalg.norm(a - b) / np.linalg.norm(b))
        nz = int((np.abs(f["ksp"][:, 7]).sum(0) > 0).sum())     # sampled positions on one readout plane
    report[scan] = dict(path=str(out), shape=list(h5py.File(out, "r")["ksp"].shape), readback_rel_diff=rel,
                        nonzero_lines_one_readout=nz, fit_lines=int(fit.sum()), seconds=time.time() - t0)
    print(scan, report[scan], flush=True)
    assert rel == 0.0 and nz == int(fit.sum()), "fit k-space check failed"
C.dump_json(report, C.SPLIT.parent / "ksp_fit_check.json")
print("DONE", flush=True)
