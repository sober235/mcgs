#!/usr/bin/env python3
"""add_gs_arm.py -- register an autoresearch MC-GS run (worktree mcgs_af12_wt) as an extra arm here.

usage: add_gs_arm.py --exp EXP --key KEY --label LABEL [--map T1:plain,FLAIR:contrast]
  --map: which channel of the run is which contrast (joint run: T1:plain,FLAIR:contrast;
         a single-T1 run: T1:plain; a single-FLAIR run: FLAIR:plain).
Writes recon/<KEY>_<c>_mag.npy (and recon/<KEY>_<c>_complex.npy when the npz carries complex volumes, runs
saved from 2026-09-11 on) from results_arc/<EXP>_volumes.npz when the run saved volumes, and
records the arm in logs/extra_gs_arms.json; summarize.py then reads the arm's metrics straight from
results_arc/<EXP>_score.json, so the table always shows the run's own scorer output.
"""
import argparse, json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1]
ARC = Path("/data0/congcong/code/GS/mcgs_af12_wt/results_arc")
NPZ = {"plain": "recT1", "contrast": "recFL"}   # cycle.py's fixed npz keys, by channel
NPZ_C = {"plain": "recT1_complex", "contrast": "recFL_complex"}   # complex volumes, when saved

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--key", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--map", default="T1:plain,FLAIR:contrast")
a = ap.parse_args()
m = dict(x.split(":") for x in a.map.split(","))
sj = ARC / f"{a.exp}_score.json"
assert sj.exists(), f"no score json for {a.exp}"
d = json.load(open(sj))
for c, ch in m.items():
    name = d.get(f"{ch}_name", c)
    assert name == c, f"channel {ch} of {a.exp} is {name}, not {c}"
vol = ARC / f"{a.exp}_volumes.npz"
if vol.exists():
    with np.load(vol) as z:
        for c, ch in m.items():
            np.save(R / f"recon/{a.key}_{c}_mag.npy", np.asarray(z[NPZ[ch]], dtype=np.float32))
            print(f"wrote recon/{a.key}_{c}_mag.npy")
            if NPZ_C[ch] in z.files:
                np.save(R / f"recon/{a.key}_{c}_complex.npy", np.asarray(z[NPZ_C[ch]], dtype=np.complex64))
                print(f"wrote recon/{a.key}_{c}_complex.npy")
else:
    print(f"(no volumes for {a.exp}: table row only)")
p = R / "logs/extra_gs_arms.json"
arms = json.load(open(p)) if p.exists() else []
arms = [x for x in arms if not (x["key"] == a.key and set(x["map"]) & set(m))] + [dict(key=a.key, label=a.label, exp=a.exp, map=m)]   # one entry per (key, contrast): the two single-contrast runs share a key
json.dump(arms, open(p, "w"), indent=1)
print(f"registered {a.key} ({a.label}) <- {a.exp} {m}")
