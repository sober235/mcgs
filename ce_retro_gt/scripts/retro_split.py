#!/usr/bin/env python3
"""retro_split.py -- freeze a 95/5 held-out split of the sampled phase-encode lines of a simulated case (2026-09-17).

The prospective topic selects lambda on held-out lines, never on truth. The simulation has truth, but using it to
tune one arm and not the others would tilt the comparison, so CS-SENSE gets its lambda the same GT-free way here.

  python scripts/retro_split.py --case UCSF-PDGM-0004 [--holdout 0.05] [--seed 0]
"""
import argparse
from pathlib import Path

import numpy as np

T = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True)
ap.add_argument("--holdout", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
m2 = np.load(T / "sim" / a.case / "gt.npz")["mask2d"]
idx = np.argwhere(m2)
rng = np.random.default_rng(a.seed)
perm = rng.permutation(len(idx))
nho = int(round(a.holdout * len(idx)))
fit = np.zeros_like(m2); ho = np.zeros_like(m2)
fit[tuple(idx[perm[nho:]].T)] = True
ho[tuple(idx[perm[:nho]].T)] = True
out = T / "sim" / a.case / "split_95_5_seed0.npz"
np.savez(out, fit=fit, ho=ho)
print(f"{a.case}: {len(idx)} lines -> fit {int(fit.sum())}, held out {int(ho.sum())} -> {out}")
