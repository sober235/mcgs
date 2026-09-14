#!/usr/bin/env python3
"""prep_04d_flair_support.py -- write FLAIR's acquired support on the (ky,kz) PE plane as data/flair_support_2d.h5
(dataset 'support', uint8 (256,176)). Rule: a PE line is acquired if any coil has nonzero energy anywhere along the
readout (the same rule as prep_04_export.py / prep_04c_export_af24c.py, which report 34139 of 45056 lines). Used by the
trainer's `data.contrast.known_zero_support` (phase 3, 2026-09-12): with a sampling pattern drawn inside the shutter
the known-zero estimator has no zero sampled line to learn the shutter from, so the shutter is given explicitly.
"""
import json, os, sys
import h5py, numpy as np
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sup = None
with h5py.File(f"{R}/data/ksp_FLAIR.h5", "r") as f:          # symlink into the AF12 topic: READ ONLY
    C = f["kspace"].shape[0]
    for c in range(C):
        k = np.abs(np.asarray(f["kspace"][c]))                 # (256 ro, 256, 176)
        s = k.max(axis=0) > 0
        sup = s if sup is None else (sup | s)
n = int(sup.sum()); print(f"[support] {n} of {sup.size} = {sup.mean():.4f} (coils {C})")
assert n == 34139, n
with h5py.File(f"{R}/data/flair_support_2d.h5", "w") as h:
    h.create_dataset("support", data=sup.astype(np.uint8))
json.dump({"support_points": n, "plane_points": int(sup.size), "support_frac": float(sup.mean()),
           "path": f"{R}/data/flair_support_2d.h5", "rule": "any coil nonzero along the readout"},
          open(f"{R}/logs/prep_04d_flair_support.json", "w"), indent=1)
print("wrote", f"{R}/data/flair_support_2d.h5")
