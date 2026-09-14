#!/usr/bin/env python3
"""score_candidate.py -- one autoresearch run, scored where the table is judged.

The primary table averages the frozen gauge over the head inside the vendor's central 80 partitions.
A candidate is scored there, with the same function (uih_mtp.rescore.score_region) and the same
ground truth the table uses, and compared against the bars of logs/rescore.json: the best value any
other arm reaches, GS-single counted as a competitor, every joint GS arm excluded (spec
2026-09-14-peds-af12-mcgs-autoresearch-design.md, section 2).

  ./scripts/py scripts/score_candidate.py --exp peds_af12_joint
  ./scripts/py scripts/score_candidate.py --exp peds_af12_single_T1w --map T1w:plain

Writes <worktree>/results_arc/<exp>_primary.json.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_peds_wt")
from uih_mtp.bars import competitor_bars, margins  # noqa: E402
from uih_mtp.rescore import partition_region, score_region  # noqa: E402

R = Path(__file__).resolve().parents[1]
ARC = Path("/data0/congcong/code/GS/mcgs_peds_wt/results_arc")
NPZ = {"plain": ("recT1_complex", "recT1"), "contrast": ("recFL_complex", "recFL")}

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--map", default="T1w:plain,T2s:contrast", help="contrast:channel pairs of this run")
ap.add_argument("--lo", type=int, default=12)
ap.add_argument("--hi", type=int, default=92)
ap.add_argument("--jobs", type=int, default=2)
args = ap.parse_args()
CHANNEL = dict(x.split(":") for x in args.map.split(","))


def volume(z, channel):
    """The run's reconstruction for one channel: the complex volume when it was saved, else magnitude."""
    for key in NPZ[channel]:
        if key in z.files:
            return np.asarray(z[key])
    raise KeyError(f"{args.exp}: npz has none of {NPZ[channel]}, only {z.files}")


def score_one(job):
    c, rec = job
    gt = np.load(R / f"recon/gt_{c}_complex.npy")
    assert rec.shape == gt.shape, (c, rec.shape, gt.shape)
    row = score_region(rec, gt, partition_region(gt.shape, 2, args.lo, args.hi))
    return c, row


with np.load(ARC / f"{args.exp}_volumes.npz") as z:
    jobs = [(c, volume(z, ch)) for c, ch in CHANNEL.items()]
with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as ex:
    rows = dict(ex.map(score_one, jobs))

bars = competitor_bars(json.load(open(R / "logs/rescore.json"))["rows"], tuple(rows))
out = {"exp": args.exp, "region": f"head inside partitions {args.lo}..{args.hi}", "rows": rows,
       "bars": {c: {m: {"bar": b, "arm": who} for m, (b, who) in bars[c].items()} for c in rows},
       "margins": {c: margins(rows[c], bars[c]) for c in rows}}
out["min_margin"] = min(v for c in out["margins"] for v in out["margins"][c].values())
out["leads_every_cell"] = out["min_margin"] > 0
json.dump(out, open(ARC / f"{args.exp}_primary.json", "w"), indent=1)

for c in rows:
    print(f"\n{c}  (central {args.lo}..{args.hi})")
    for m in ("head_psnr", "head_ssim", "head_hfen"):
        bar, who = bars[c][m]
        d = out["margins"][c][m]
        print(f"  {m:10s} {rows[c][m]:9.4f}   bar {bar:9.4f} ({who:14s})   margin {d:+.4f}"
              f"   {'lead' if d > 0 else 'BEHIND'}")
print(f"\nmin margin over the six cells: {out['min_margin']:+.4f}  "
      f"({'leads every cell' if out['leads_every_cell'] else 'does not lead yet'})")
print(f"wrote {ARC / f'{args.exp}_primary.json'}")
