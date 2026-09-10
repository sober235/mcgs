"""Spec 5.3: the phyllotaxis trajectory, shared by all four subjects (same 22 x 1000 protocol)."""
import sys

import numpy as np

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr import DERIVED_ROOT, N_SAMP, N_SEG, N_SHOT  # noqa: E402
from cardiac_fr.trajectory import spoke_coordinates  # noqa: E402

out = DERIVED_ROOT / "common" / "traj_r.npy"
k = spoke_coordinates(N_SHOT, N_SEG, N_SAMP, self_nav=True)
np.save(out, k)
print(f"saved {out} {k.shape}, max |k| = {float(np.abs(k).max()):.4f}")
