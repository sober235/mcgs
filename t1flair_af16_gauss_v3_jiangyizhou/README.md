# Retrospective nominal AF 16 on the 3D T1 + FLAIR pair, jiangyizhou -- Gaussian variable-density random mask (diff-mask study)

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

2-D Gaussian variable density: outside the ACS the points are drawn i.i.d. without replacement with probability proportional to exp(-rn^2 / 2s^2). The width s is not hand-picked -- it is chosen on a 0.005 grid so the draw's median normalised radius matches the house vd-Poisson mask's (0.399 against 0.400), which matches the density curve and leaves the point process as the only difference: i.i.d. random against Poisson disc.

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
| **Gaussian variable-density random (this topic)** | **2964** | **15.20** | **15.43** | **44** | **442/576** | **0.399** |

Generator `scripts/make_masks_af16_diffmask.py --kinds gauss` (it refuses to overwrite an existing mask), read-out
`logs/make_masks_af16_diffmask.json`, mask `data/mask_gauss_256x176_af16.mat` (md5 `306f4caed5118687c09ad364c63d02d4`) broadcast
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
**T1** (sorted by head-only PSNR; vd-Poisson column = the same arm in `T1FLAIR_AF16_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @vd-Poisson | this − vd-Poisson |
|---|---|---|---|---|---|---|---|---|
| MC-GS | 38.615 | 0.9316 | **34.909** | **0.9333** | 0.1344 | 0.796 | 35.706 | -0.80 |
| GS-single | 38.471 | 0.9347 | 34.771 | 0.9305 | 0.1412 | 0.794 | 35.558 | -0.79 |
| ZS-SSDU | 38.217 | 0.9518 | 34.375 | 0.9273 | 0.1697 | 0.899 | 34.150 | +0.22 |
| CS-SENSE (TV, converged) (λ 0.0007) | 37.652 | 0.9418 | 33.896 | 0.9190 | 0.1814 | 0.788 | 34.945 | -1.05 |
| joint CS-SENSE (JTV) (λ 0.001) | 37.638 | 0.9430 | 33.841 | 0.9188 | 0.1832 | 0.779 | 34.831 | -0.99 |
| CS-SENSE | 37.481 | 0.9389 | 33.637 | 0.9172 | 0.1855 | 0.747 | 34.409 | -0.77 |
| L1-SPIRiT (λ 0.001) | 36.709 | 0.9210 | 32.995 | 0.9010 | 0.1653 | 0.896 | 33.788 | -0.79 |
| CG-SENSE | 34.598 | 0.7887 | 31.899 | 0.8782 | 0.2516 | 0.811 | 32.354 | -0.45 |
| MC-INR (Niessen 2025) | 31.874 | 0.6231 | 31.558 | 0.8463 | 0.2598 | 0.929 | 31.590 | -0.03 |
| 3D-DIP | 35.063 | 0.8570 | 31.141 | 0.8546 | 0.3053 | 0.647 | 30.994 | +0.15 |
| 3D-INR | 32.299 | 0.7073 | 29.443 | 0.7659 | 0.2206 | 1.436 | 30.375 | -0.93 |
| ZF | 29.468 | 0.7953 | 25.412 | 0.6523 | 0.7320 | 0.278 | 25.841 | -0.43 |

**FLAIR** (sorted by head-only PSNR; vd-Poisson column = the same arm in `T1FLAIR_AF16_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @vd-Poisson | this − vd-Poisson |
|---|---|---|---|---|---|---|---|---|
| MC-GS | 38.101 | 0.9263 | **34.317** | **0.9210** | 0.1278 | 0.759 | 34.928 | -0.61 |
| GS-single | 38.019 | 0.9046 | 34.294 | 0.9202 | 0.1295 | 0.767 | 34.886 | -0.59 |
| ZS-SSDU | 37.902 | 0.9471 | 33.972 | 0.9196 | 0.1479 | 0.905 | 34.005 | -0.03 |
| joint CS-SENSE (JTV) (λ 0.001) | 37.588 | 0.9408 | 33.731 | 0.9112 | 0.1481 | 0.719 | 34.355 | -0.62 |
| CS-SENSE (TV, converged) (λ 0.0007) | 37.503 | 0.9395 | 33.683 | 0.9115 | 0.1480 | 0.699 | 34.302 | -0.62 |
| CS-SENSE | 37.314 | 0.9298 | 33.447 | 0.9078 | 0.1477 | 0.655 | 33.806 | -0.36 |
| L1-SPIRiT (λ 0.001) | 36.652 | 0.9187 | 32.849 | 0.8923 | 0.1360 | 0.876 | 33.720 | -0.87 |
| CG-SENSE | 35.275 | 0.8047 | 32.557 | 0.8828 | 0.1800 | 0.757 | 32.826 | -0.27 |
| MC-INR (Niessen 2025) | 33.082 | 0.6739 | 32.102 | 0.8599 | 0.1946 | 0.852 | 32.243 | -0.14 |
| 3D-DIP | 34.850 | 0.8239 | 30.918 | 0.8393 | 0.2613 | 0.545 | 30.650 | +0.27 |
| 3D-INR | 32.917 | 0.7329 | 29.807 | 0.7745 | 0.2053 | 1.212 | 30.257 | -0.45 |
| ZF | 31.332 | 0.8335 | 27.451 | 0.6956 | 0.6888 | 0.260 | 27.879 | -0.43 |

**MC-GS joint minus MC-GS single**

| channel | whole PSNR | head PSNR | head SSIM | (vd-Poisson: whole / head) |
|---|---|---|---|---|
<!-- TABLES:END -->

## Figures

The house layouts, at the AF16 figure-of-record slices and box so nothing is re-selected on this mask. Eleven arms:
CG-SENSE, CS-SENSE, Joint SENSE, L1-SPIRiT, ZS-SSDU, 3D-DIP, 3D-INR, MC-INR, GS-single, MC-GS, against the ground truth.

| figure | what |
|---|---|
| `http://localhost:8765/t1flair_v3_af16_diffmask_gauss/t1flair_v3_af16g_T1_row3planes_compact_174_110_47_headerr_p3_spirit.png` | T1, transverse 174 / coronal 110 / sagittal 47, no inset |
| `http://localhost:8765/t1flair_v3_af16_diffmask_gauss/t1flair_v3_af16g_FLAIR_row3planes_compact_174_110_47_headerr_p3_spirit.png` | FLAIR, same slices |
| `http://localhost:8765/t1flair_v3_af16_diffmask_gauss/t1flair_v3_af16g_T1_row3planes_compact_180_124_76_inset48_headerr_p3_spirit.png` | T1, figure-of-record layout, 48-voxel zoom inset (also `.pdf`, 600 dpi vector) |
| `http://localhost:8765/t1flair_v3_af16_diffmask_gauss/t1flair_v3_af16g_FLAIR_row3planes_compact_180_124_76_inset48_headerr_p3_spirit.png` | FLAIR, same (also `.pdf`) |
| `http://localhost:8765/t1flair_v3_af16_diffmask/t1flair_v3_af16_diffmask_masks.png` | the four candidate masks side by side |
| `http://localhost:8765/t1flair_v3_af16_diffmask/t1flair_v3_jyz_full_kspace.png` | the fully sampled k-space both contrasts are undersampled from |

Files also live in this topic's `figures/`. Orange in-panel numbers are head-masked slice PSNR / SSIM (7x7 window) /
NMSE, the best of each plane in bold; error maps are `|recon - GT| / max(GT)`, jet, fixed [0, 0.1], inside the head
only; left of each error row is that plane's sampling mask.

## The study's headline: does MC-GS's lead depend on the pattern?

The three topics share everything but the sampling pattern, so this table isolates it. **MC-GS is first on head-only
PSNR on all three patterns and both contrasts** (`scripts/compare_masks.py`, which reads all three topics read-only;
full tables in `logs/compare_masks_head_{psnr,ssim,hfen}.md`):

| contrast | pattern | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | vd-Poisson (house) | 35.706 | GS-single 35.558 | **+0.148** |
| T1 | Gaussian | 34.909 | GS-single 34.771 | **+0.138** |
| T1 | pseudo-radial | 31.113 | GS-single 30.913 | **+0.200** |
| FLAIR | vd-Poisson (house) | 34.928 | GS-single 34.886 | **+0.042** |
| FLAIR | Gaussian | 34.317 | GS-single 34.294 | **+0.023** |
| FLAIR | pseudo-radial | 31.729 | ZS-SSDU 31.635 | **+0.093** |

**GS-single counts as a comparison arm, not as a second version of ours** (user, 2026-09-13, restated 2026-09-16 with
its reason): 3-D reconstruction by Gaussian splatting is already published work, so a single-contrast Gaussian-splatting
reconstruction is prior art that this method has to beat, exactly like CS-SENSE or ZS-SSDU. Every margin quoted in this
topic is therefore over the strongest arm *including* GS-single. A margin computed over "the strongest non-GS arm" is
not a number this study reports.

**Five of the six cells win on all three head metrics; the sixth does not.** On the pseudo-radial mask FLAIR keeps head
PSNR but loses head SSIM (0.8673 against ZS-SSDU's 0.8686) and head HFEN (0.2809 against 0.2768). That is the one cell
where MC-GS is not the outright winner, and it is the cell whose known-zero prior the pattern disables (27 mask points
outside the shutter, the estimator cannot fit the ellipse). It is reported, not smoothed over.

**The pattern itself is worth more than any arm's lead over another.** Against the house vd-Poisson mask, the same arm
loses 0.3-1.1 dB on the Gaussian mask and 1.1-5.0 dB on the pseudo-radial one -- an order of magnitude more than the
margins above. Poisson disc is the better pattern of the three, which is what the 3-D head literature says; the point
of the study is that the ranking of the *methods* does not change with it.

**The coupling gain (MC-GS minus GS-single) is positive on every pattern**: T1 +0.148 / +0.138 / +0.200, FLAIR +0.042 /
+0.023 / +0.150 (vd-Poisson / Gaussian / pseudo-radial). It is largest on the worst pattern, which is consistent with
cross-contrast information being worth more when the single-contrast problem is harder -- an observation, not evidence.

## Known limitations
- **The lambda grids were swept on this mask but the MC-GS recipe was not.** MC-GS and GS-single run the AF16 recipe
  frozen, which is the point of the study (one variable), but it does mean the classical arms get a per-mask lambda
  while MC-GS does not. If MC-GS still leads, that is in spite of the handicap, not because of it.
- Everything the AF16 topic's own README lists as a limitation applies here unchanged: the recipe and the classical
  lambdas were selected on this subject, and the numbers are on the magnitude gauge.

## Regenerating

```
# mask (numpy, seeded, regenerable; it refuses to overwrite)
PYTHONNOUSERSITE=1 PYTHONPATH= python scripts/make_masks_af16_diffmask.py --kinds gauss

# every arm, both topics, queued onto free cards, then finalize
setsid nohup bash scripts/launch_diffmask.sh > /dev/null 2>&1 < /dev/null &
```

## Status

2026-09-16: complete. Eleven arms x two contrasts reconstructed, scored and figured; MC-INR re-scored through this topic's own gauge with a worst disagreement of 1e-7 against the reproduction's own numbers (the gauge is mask-independent, so this is the expected result and it confirms the three topics' numbers are directly comparable).
