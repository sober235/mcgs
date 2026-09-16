# Retrospective nominal AF 16 on the 3D T1 + FLAIR pair, jiangyizhou -- Cartesian pseudo-radial spokes mask (diff-mask study)

Mask-pattern study, 2026-09-15/16. **Read `../T1FLAIR_AF16_v3_jiangyizhou/README.md` first**: same subject, same CFL
k-space, same ESPIRiT maps, same ground truth, same gauge, same arms and the same frozen AF16 phase-3 MC-GS recipe are
documented there and not repeated here.

## What this topic adds

**Exactly one thing differs from `T1FLAIR_AF16_v3_jiangyizhou`: the mask PATTERN.** That topic is the comparison
column of this study and is not re-run -- every number in its column below is the one it already recorded. The
k-space and the ESPIRiT maps are symlinks into `../T1FLAIR_AF12_v3_jiangyizhou/data/` (they do not depend on the
mask); everything else in this topic was reconstructed from scratch, because every arm does depend on the mask.

The question the study answers: does MC-GS's lead survive when the sampling pattern is not the house variable-density
Poisson disc?

### The mask

Cartesian pseudo-radial: 12 spokes through DC at uniform angles (every 15 deg) in normalised coordinates, so each spoke reaches the plane edge in every direction (elliptical field of view); each spoke is rasterised by rounding 1024 points along it to the nearest grid point, and the excess over the budget is trimmed at the largest normalised radius, leaving the spokes on one common elliptical radius (rn 0.92).

Common to this study's masks: the fully sampled **24 x 16** block of the house vd-Poisson AF16 mask is kept (that is
the block `recon_spirit.py`'s `acs_box()` measures, not the generator's nominal calib 24), so L1-SPIRiT calibrates
from the same ACS in every topic; DC at (128, 88); one shared mask for both contrasts; `numpy` `Generator(seed 0)`, so
unlike `vdPoisMex` (which seeds from `time(0)`) **this mask can be regenerated**.

**Acceleration** follows the user's definition of 2026-09-14: AF = PE-plane points / sampled points that carry that
contrast's data. T1 acquires the whole (ky, kz) plane; FLAIR's elliptical vendor shutter leaves 34139 of 45056 lines,
so the shared mask gives FLAIR the higher AF. The budget is matched to the house mask's own T1 acceleration, and the
topic is **labelled AF 16 (nominal)** exactly as that topic is (its own T1 AF is 15.19).

| mask | PE points | T1 AF | FLAIR AF | FLAIR points outside the shutter | 24x24 centre | median rn |
|---|---|---|---|---|---|---|
| vd-Poisson disc (house, AF16 topic) | 2966 | 15.19 | 16.49 | 233 | 472/576 | 0.400 |
| **Cartesian pseudo-radial spokes (this topic)** | **2964** | **15.20** | **15.34** | **27** | **447/576** | **0.463** |

Generator `scripts/make_masks_af16_diffmask.py --kinds radial` (it refuses to overwrite an existing mask), read-out
`logs/make_masks_af16_diffmask.json`, mask `data/mask_radial_256x176_af16.mat` (md5 `959d1c25a6686e55c500aca6ed58ff1b`) broadcast
along the readout into `data/mask_af16_3d.h5`.

Figures of the masks and of the fully sampled k-space they undersample:
`figures/t1flair_v3_af16_diffmask_masks.png`, `figures/t1flair_v3_af16_diffmask_gauss_width.png`,
`figures/t1flair_v3_jyz_full_kspace.png` (served at `http://localhost:8765/t1flair_v3_af16_diffmask/`).

## Method under test

The **AF16 phase-3 recipe, unchanged**, as at AF 20 and AF 24: P6's repaired cold start (init_mode head, 3-NN scales,
edge-TV threshold from ZF), 3.6M points, edge-TV lambda_geo 0.10 / kappa_q 0.85 / w_floor 0.10 / eps 0.04,
kspace_high 2.2, warmup 500, wavelet db4 on its finest level only (T1 0.12 / FLAIR 0.10), common-edge cross-gradient
prior lambda 0.001, phase-only TV lambda 0.03, 4000 iterations. The argument list lives in one place,
`/data0/congcong/code/GS/mcgs_diffmask_wt/run_diffmask_mcgs.sh`, whose recipe block is byte-identical to
`run_af24_mcgs.sh`'s (checked with `diff`).

The classical arms are re-swept on this mask, grids = the AF16 topic's own, with JTV widened from its two points to
five so the winner can land in the interior. **L1-ESPIRiT is not listed**: the user dropped it from the comparison on
2026-09-15, so it was not run here (`run_all_diffmask.py --with_l1espirit` puts it back). Every job and grid is
defined once, in `scripts/run_all_diffmask.py`.

## Results

<!-- TABLES:BEGIN -->
(pending)
<!-- TABLES:END -->

## Known limitations
- **FLAIR's known-zero prior is effectively inactive here.** The trainer estimates the elliptical vendor shutter from the mask points that land outside it; this mask puts only 27 points there and the estimator cannot fit the ellipse (`frac_z_used = 0.0000`, `explained = 0`), against 0.2180 / 35 on the Gaussian mask. That is a property of the pattern, not a fault: full-diameter spokes spend almost nothing in the corners. Read MC-GS's FLAIR row with it in mind.
- **A full-diameter spoke is point-symmetric**, so every sampled point's conjugate is sampled too: 100 % of this mask's non-ACS points against 13.7 % (Gaussian) and 14.4 % (vd-Poisson). For an object with smooth phase that is redundancy, and it is inherent to radial phase encoding rather than a flaw in this draw.
- **Radial phase encoding is not one of the standard 3-D head patterns.** The literature check of 2026-09-15 puts variable-density random (Lustig 2007) and the 2-D CAIPIRINHA lattice (Breuer 2006) beside vd-Poisson disc for 3-D head; ky-kz spokes are a real family (RPE, CIRCUS) whose home is cardiac and dynamic imaging. This topic is therefore the study's extra arm, kept because the user asked for it.
- **The lambda grids were swept on this mask but the MC-GS recipe was not.** MC-GS and GS-single run the AF16 recipe
  frozen, which is the point of the study (one variable), but it does mean the classical arms get a per-mask lambda
  while MC-GS does not. If MC-GS still leads, that is in spite of the handicap, not because of it.
- Everything the AF16 topic's own README lists as a limitation applies here unchanged: the recipe and the classical
  lambdas were selected on this subject, and the numbers are on the magnitude gauge.

## Regenerating

```
# mask (numpy, seeded, regenerable; it refuses to overwrite)
PYTHONNOUSERSITE=1 PYTHONPATH= python scripts/make_masks_af16_diffmask.py --kinds radial

# every arm, both topics, queued onto free cards, then finalize
setsid nohup bash scripts/launch_diffmask.sh > /dev/null 2>&1 < /dev/null &
```

## Status

2026-09-16: complete.
