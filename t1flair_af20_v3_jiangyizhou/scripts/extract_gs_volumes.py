#!/usr/bin/env python3
"""extract_gs_volumes.py -- pull the MC-GS magnitude volumes out of arc/cycle.py's npz archives.

cycle.py hardcodes the npz keys as recT1/gtT1/zfT1 (the PLAIN channel) and recFL/gtFL/zfFL (the
CONTRAST channel) whatever the contrasts are actually called, so the mapping here is by channel,
not by name:
    t1flair_v3_af20_joint      plain = T1, contrast = FLAIR
    t1flair_v3_af20_single_T1  both channels are T1  -> take the plain one
    t1flair_v3_af20_single_FLAIR  both channels are FLAIR -> take the plain one
Writes recon/<method>_<contrast>_mag.npy (float32 magnitude) plus the GT and ZF references.
"""
import json
from pathlib import Path
import numpy as np

R = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF20_v3_jiangyizhou")
ARC = Path("/data0/congcong/code/GS/mcgs_af16_wt/results_arc")

JOBS = [("t1flair_v3_af20_joint", "gs_joint", [("recT1", "T1"), ("recFL", "FLAIR")],
         [("gtT1", "T1"), ("gtFL", "FLAIR")], [("zfT1", "T1"), ("zfFL", "FLAIR")]),
        ("t1flair_v3_af20_single_T1", "gs_single", [("recT1", "T1")], [], []),
        ("t1flair_v3_af20_single_FLAIR", "gs_single", [("recT1", "FLAIR")], [], [])]


def main():
    wrote = []
    for exp, method, recs, gts, zfs in JOBS:
        p = ARC / f"{exp}_volumes.npz"
        if not p.exists():
            print(f"MISSING {p}"); continue
        with np.load(p) as d:
            for key, c in recs:
                out = R / f"recon/{method}_{c}_mag.npy"
                np.save(out, np.asarray(d[key], dtype=np.float32)); wrote.append(str(out))
            for key, c in gts:
                np.save(R / f"recon/gtGS_{c}_mag.npy", np.asarray(d[key], dtype=np.float32))
            for key, c in zfs:
                np.save(R / f"recon/zfGS_{c}_mag.npy", np.asarray(d[key], dtype=np.float32))
        print(f"{exp}: extracted {[c for _, c in recs]}")
    print("\n".join(wrote))


if __name__ == "__main__":
    main()
