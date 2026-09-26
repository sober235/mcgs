"""Copy the widen-loop winners' volumes into the AF topics under new keys, for the figure scripts.

usage: wd_register_vols.py AF   (16 | 20 | 24)
Writes <topic>/recon/gs_joint_wd_<c>_{mag,complex}.npy from results_arc/wd_v_af<AF>_j_volumes.npz and
gs_single_wd_<c>_{mag,complex}.npy from the single runs' plain channel (both channels carry that contrast).
Refuses to overwrite: every file is new, nothing registered before is touched.
"""
import json
import sys
from pathlib import Path

import numpy as np

ARC = Path(__file__).resolve().parents[1] / "results_arc"
AF = int(sys.argv[1])
TOPIC = Path(f"/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF{AF}_v3_jiangyizhou/recon")
NPZ = {"plain": "recT1", "contrast": "recFL"}          # cycle.py's fixed npz keys, by channel
SRC = [("gs_joint_wd", "T1", f"wd_v_af{AF}_j", "plain"), ("gs_joint_wd", "FLAIR", f"wd_v_af{AF}_j", "contrast"),
       ("gs_single_wd", "T1", f"wd_v_af{AF}_s_t1", "plain"), ("gs_single_wd", "FLAIR", f"wd_v_af{AF}_s_fl", "plain")]

for key, c, exp, ch in SRC:
    d = json.load(open(ARC / f"{exp}_score.json"))
    print(f"{key} {c} <- {exp}:{ch}  head PSNR {d[ch]['head_psnr']:.3f}")
    with np.load(ARC / f"{exp}_volumes.npz") as z:
        for suffix, npz_key in (("mag", NPZ[ch]), ("complex", NPZ[ch] + "_complex")):
            out = TOPIC / f"{key}_{c}_{suffix}.npy"
            if out.exists():
                raise SystemExit(f"refusing to overwrite {out}")
            arr = np.asarray(z[npz_key])
            np.save(out, arr.astype(np.complex64 if suffix == "complex" else np.float32))
            print("  wrote", out)
