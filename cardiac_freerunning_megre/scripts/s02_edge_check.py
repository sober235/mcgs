"""Spec 5.2: mean |s| per readout sample over 250 mid-scan spokes x 8 echoes x 24 coils. Samples above
1.2x the kept-edge level must all lie in the dropped set {0, 1, 2, 221, 222, 223}."""
import sys

import h5py
import numpy as np

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr import DERIVED_ROOT, SUBJECTS  # noqa: E402

DROPPED = {0, 1, 2, 221, 222, 223}
ok_all = True
for s in sys.argv[1:] or SUBJECTS:
    with h5py.File(DERIVED_ROOT / s / "raw.h5", "r") as h:
        m = np.abs(h["ksp"][:, 10000:10250]).mean(axis=(0, 1, 2))
    level = float(np.median(np.r_[m[3:9], m[215:221]]))
    flagged = [j for j in [*range(0, 8), *range(216, 224)] if m[j] > 1.2 * level]
    ok = set(flagged) <= DROPPED
    ok_all &= ok
    print(f"{s}: head/level {np.round(m[:5] / level, 2).tolist()}  tail/level "
          f"{np.round(m[219:] / level, 2).tolist()}  flagged {flagged}  {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok_all else 1)
