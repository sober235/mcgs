"""Spec 5.1: 1,000 imaging lines (seed 0) read by the authors' MATLAB reader must equal ours bitwise."""
import subprocess
import sys

import h5py
import numpy as np
import scipy.io as sio

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr import DERIVED_ROOT, N_ECHO, RAW_ROOT, TOPIC  # noqa: E402

S = sys.argv[1] if len(sys.argv) > 1 else "V07"
d = DERIVED_ROOT / S
rows = np.sort(np.random.default_rng(0).choice(176000, size=1000, replace=False))
idx_file, out = d / "xcheck_idx.txt", d / "xcheck_matlab_lines.mat"
np.savetxt(idx_file, rows + 1, fmt="%d")                       # MATLAB is 1-based
dat = (RAW_ROOT / S / "freerunning_8echo.dat").resolve()
subprocess.run(["matlab", "-batch", f"addpath('{TOPIC / 'scripts'}'); xcheck_reader('{dat}', '{idx_file}', '{out}')"],
               check=True)
theirs = sio.loadmat(out)["lines"].transpose(2, 1, 0)         # (1000, coil, sample)
with h5py.File(d / "raw.h5", "r") as h:
    ours = np.empty((rows.size, *h["ksp"].shape[2:]), np.complex64)
    for e in range(N_ECHO):
        sel = np.flatnonzero(rows % N_ECHO == e)
        ours[sel] = h["ksp"][e][rows[sel] // N_ECHO]
equal = bool(np.array_equal(ours, theirs))
print(f"{S}: {rows.size} lines, ours {ours.shape} vs theirs {theirs.shape}, bitwise equal = {equal}, "
      f"max |diff| = {float(np.abs(ours.astype(np.complex128) - theirs).max()):.3g}")
sys.exit(0 if equal else 1)
