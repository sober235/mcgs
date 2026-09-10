"""Spec 5.3: our phyllotaxis port vs the authors' computePhyllotaxis (r units); max |diff| < 1e-9."""
import subprocess
import sys

import numpy as np
import scipy.io as sio

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr import DERIVED_ROOT, N_SAMP, N_SEG, N_SHOT, TOPIC  # noqa: E402

out = DERIVED_ROOT / "common" / "traj_matlab.mat"
subprocess.run(["matlab", "-batch", f"addpath('{TOPIC / 'scripts'}'); xcheck_traj('{out}')"], check=True)
m = sio.loadmat(out)
k = np.load(DERIVED_ROOT / "common" / "traj_r.npy")
worst = 0.0
for i, name in enumerate(("kx", "ky", "kz")):
    ref = m[name].reshape(N_SAMP, N_SEG * N_SHOT, order="F")      # MATLAB (N, nseg, nshot) -> column = Lin
    diff = float(np.abs(ref - k[i]).max())
    worst = max(worst, diff)
    print(f"{name}: max |diff| = {diff:.3g}")
print("PASS" if worst < 1e-9 else "FAIL")
sys.exit(0 if worst < 1e-9 else 1)
