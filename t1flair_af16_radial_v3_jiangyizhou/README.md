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
**T1** (sorted by head-only PSNR; vd-Poisson column = the same arm in `T1FLAIR_AF16_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @vd-Poisson | this − vd-Poisson |
|---|---|---|---|---|---|---|---|---|
| MC-GS | 35.047 | 0.9011 | **31.113** | **0.8702** | 0.3417 | 0.635 | 35.706 | -4.59 |
| GS-single | 34.840 | 0.9005 | 30.913 | 0.8629 | 0.3587 | 0.631 | 35.558 | -4.65 |
| ZS-SSDU | 34.856 | 0.9174 | 30.854 | 0.8619 | 0.3516 | 0.704 | 34.150 | -3.30 |
| L1-SPIRiT (λ 0.0005) | 33.781 | 0.8799 | 30.004 | 0.8240 | 0.3682 | 0.866 | 33.788 | -3.78 |
| CS-SENSE | 33.835 | 0.8926 | 29.961 | 0.8439 | 0.4045 | 0.606 | 34.409 | -4.45 |
| CS-SENSE (TV, converged) (λ 0.0014) | 33.902 | 0.8874 | 29.955 | 0.8452 | 0.4084 | 0.613 | 34.945 | -4.99 |
| joint CS-SENSE (JTV) (λ 0.0014) | 33.785 | 0.8958 | 29.862 | 0.8456 | 0.4192 | 0.631 | 34.831 | -4.97 |
| 3D-DIP | 33.830 | 0.8412 | 29.855 | 0.8134 | 0.4301 | 0.650 | 30.994 | -1.14 |
| CG-SENSE | 32.449 | 0.7849 | 29.109 | 0.8136 | 0.4576 | 0.652 | 32.354 | -3.24 |
| MC-INR (Niessen 2025) | 30.760 | 0.6210 | 28.952 | 0.7845 | 0.4857 | 0.706 | 31.590 | -2.64 |
| 3D-INR | 30.776 | 0.6412 | 27.868 | 0.6997 | 0.4575 | 1.248 | 30.375 | -2.51 |
| ZF | 29.405 | 0.7790 | 25.406 | 0.6455 | 0.7545 | 0.288 | 25.841 | -0.43 |

**FLAIR** (sorted by head-only PSNR; vd-Poisson column = the same arm in `T1FLAIR_AF16_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @vd-Poisson | this − vd-Poisson |
|---|---|---|---|---|---|---|---|---|
| MC-GS | 35.625 | 0.9012 | **31.729** | 0.8673 | 0.2809 | 0.615 | 34.928 | -3.20 |
| ZS-SSDU | 35.646 | 0.9218 | 31.635 | **0.8686** | 0.2768 | 0.701 | 34.005 | -2.37 |
| GS-single | 35.459 | 0.8833 | 31.579 | 0.8624 | 0.2991 | 0.617 | 34.886 | -3.31 |
| CS-SENSE | 34.829 | 0.8955 | 30.932 | 0.8455 | 0.3545 | 0.547 | 33.806 | -2.87 |
| CS-SENSE (TV, converged) (λ 0.001) | 34.692 | 0.8953 | 30.777 | 0.8460 | 0.3735 | 0.564 | 34.302 | -3.52 |
| joint CS-SENSE (JTV) (λ 0.0014) | 34.510 | 0.8950 | 30.609 | 0.8421 | 0.3899 | 0.584 | 34.355 | -3.75 |
| L1-SPIRiT (λ 0.0005) | 34.286 | 0.8829 | 30.432 | 0.8192 | 0.3136 | 0.831 | 33.720 | -3.29 |
| CG-SENSE | 33.642 | 0.8065 | 30.301 | 0.8200 | 0.4060 | 0.642 | 32.826 | -2.53 |
| 3D-DIP | 34.256 | 0.8179 | 30.290 | 0.8141 | 0.3416 | 0.539 | 30.650 | -0.36 |
| MC-INR (Niessen 2025) | 32.161 | 0.6879 | 29.966 | 0.7969 | 0.4282 | 0.693 | 32.243 | -2.28 |
| 3D-INR | 31.158 | 0.6395 | 28.247 | 0.6946 | 0.4302 | 1.158 | 30.257 | -2.01 |
| ZF | 31.230 | 0.8248 | 27.377 | 0.6851 | 0.7194 | 0.286 | 27.879 | -0.50 |

**MC-GS joint minus MC-GS single**

| channel | whole PSNR | head PSNR | head SSIM | (vd-Poisson: whole / head) |
|---|---|---|---|---|
<!-- TABLES:END -->

## Figures

House layouts, at the AF16 figure-of-record slices and box so nothing is re-selected on this mask. Twelve columns:
ground truth, zero-filling, CG-SENSE, CS-SENSE, Joint SENSE, L1-SPIRiT, ZS-SSDU, 3D-DIP, 3D-INR, MC-INR, GS-single,
MC-GS.

| figure | what |
|---|---|
| `http://localhost:8765/t1flair_v3_af16_diffmask_radial/t1flair_v3_af16r_T1_row3planes_compact_174_110_47_headerr_p3_spirit.png` | T1, transverse 174 / coronal 110 / sagittal 47 |
| `http://localhost:8765/t1flair_v3_af16_diffmask_radial/t1flair_v3_af16r_FLAIR_row3planes_compact_174_110_47_headerr_p3_spirit.png` | FLAIR, same slices |
| `http://localhost:8765/t1flair_v3_af16_diffmask_radial/t1flair_v3_af16r_T1_row3planes_compact_180_124_76_inset48_headerr_p3_spirit.png` | T1, figure of record, 48-voxel zoom inset (also `.pdf`, 600 dpi vector) |
| `http://localhost:8765/t1flair_v3_af16_diffmask_radial/t1flair_v3_af16r_FLAIR_row3planes_compact_180_124_76_inset48_headerr_p3_spirit.png` | FLAIR, same (also `.pdf`) |
| `http://localhost:8765/t1flair_v3_af16_diffmask/t1flair_v3_af16_diffmask_masks.png` | the four candidate masks side by side |
| `http://localhost:8765/t1flair_v3_af16_diffmask/t1flair_v3_jyz_full_kspace.png` | the fully sampled k-space both contrasts are undersampled from |

Files also live in this topic's `figures/`. Orange in-panel numbers are head-masked slice PSNR / SSIM (7x7 window) /
NMSE, the best of each plane in bold; error maps are `|recon - GT| / max(GT)`, jet, fixed [0, 0.1], inside the head
only; left of each error row is that plane's sampling mask.

Zero-filling is the adjoint, so it barely uses the pattern and lands within 0.4 dB of itself on all three masks (head
PSNR 25.4-25.8 on T1, 27.4-27.9 on FLAIR). It is the anchor showing the three studies start from the same place; what
differs in its panel is the *shape* of the aliasing, noise-like on the random patterns and streaky on the radial one.

The served folder and the file-name prefix are derived from the topic path inside `fig_three_planes_compact.py`, not
hardcoded: a hardcoded copy of that script wrote this study's radial renders into the Gaussian folder under Gaussian
names on 2026-09-16 and overwrote them. Both sets were re-rendered after the fix.

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
- **FLAIR's known-zero prior is effectively inactive here.** The trainer estimates the elliptical vendor shutter from the mask points that land outside it; this mask puts only 27 points there and the estimator cannot fit the ellipse (`frac_z_used = 0.0000`, `explained = 0`), against 0.2180 / 35 on the Gaussian mask. That is a property of the pattern, not a fault: full-diameter spokes spend almost nothing in the corners. Read MC-GS's FLAIR row with it in mind.
- **A full-diameter spoke is point-symmetric**, so every sampled point's conjugate is sampled too: 100 % of this mask's non-ACS points against 13.7 % (Gaussian) and 14.4 % (vd-Poisson). For an object with smooth phase that is redundancy, and it is inherent to radial phase encoding rather than a flaw in this draw.
- **Radial phase encoding is not one of the standard 3-D head patterns.** The literature check of 2026-09-15 puts variable-density random (Lustig 2007) and the 2-D CAIPIRINHA lattice (Breuer 2006) beside vd-Poisson disc for 3-D head; ky-kz spokes are a real family (RPE, CIRCUS) whose home is cardiac and dynamic imaging. This topic is therefore the study's extra arm, kept because the user asked for it.
- **Parameter selection, for the record.** MC-GS and GS-single run one frozen recipe (the AF16 phase-3 one) on all
  three masks -- that is what makes the study single-variable. The classical arms' lambdas were re-selected on this
  mask from the AF16 grids.
- **On this mask the TV arms' lambdas are edge winners and the grid was not extended.** Converged CS-SENSE picks the
  grid's top point on both contrasts (T1 0.0014, FLAIR 0.001) and joint TV picks 0.0014 on both: a pseudo-radial mask
  needs stronger regularisation than the AF16 grids were drawn for, so those two arms sit at the edge of their sweep
  rather than at an interior optimum, and their true optimum is somewhat above what this table shows. L1-SPIRiT's
  winner is interior on both contrasts, so it is unaffected. The study's own headline margin, MC-GS over GS-single,
  is unaffected too -- both run the same frozen recipe -- but a margin quoted over converged TV or joint TV on this
  mask would be optimistic. Recording an edge winner rather than silently extending the grid is this project's
  convention (the AF24 and cardiac topics do the same).
- **L1-ESPIRiT is retired from this series, not merely unlisted** (user, 2026-09-16): it is replaced everywhere by
  L1-SPIRiT as the k-space comparison arm. It was not run on either diff-mask topic and will not be; the older topics
  keep their recorded L1-ESPIRiT volumes and summary rows as history. `run_all_diffmask.py --with_l1espirit` still
  exists but is not something this series uses.
- **The 2-D CAIPIRINHA mask was built but not reconstructed** (user, 2026-09-16). Its topic
  `../T1FLAIR_AF16_caipi_v3_jiangyizhou` holds the mask, its read-out and the inspection figures only; the study
  compares vd-Poisson, Gaussian and pseudo-radial.
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

2026-09-16: complete. Eleven arms x two contrasts reconstructed, scored and figured; MC-INR re-scored through this topic's own gauge with a worst disagreement of 1e-7 against the reproduction's own numbers (the gauge is mask-independent, so this is the expected result and it confirms the three topics' numbers are directly comparable).

## Seven-method cut of the figure of record (user, 2026-09-18)

The user asked for the same figure with only **CS-SENSE, Joint SENSE, ZS-SSDU, 3D-INR, MC-INR,
GS-single and MC-GS**. ZF, CG-SENSE, the L1 arm and 3D-DIP are dropped; **GT keeps its column** --
it is what the error maps are taken against, and the sampling mask of each plane sits in its cell of
the error rows. Slices, zoom box, grey window, error colour scale, orientation, typography, metrics
and resolution are the figure of record's, unchanged.

```
figures/t1flair_v3_af16r_<c>_row3planes_compact_180_124_76_inset48_headerr_p3_sel7.{png,pdf}
bash scripts/render_sel7.sh
```

served from `http://localhost:8765/t1flair_v3_af16_diffmask_radial/`, mirrored in `/home/congcongliu/figs/t1flair_v3_af16_diffmask_radial/`.
**The figure of record is untouched** -- the new `--keep` flag appends `_sel7` to the file name.

**The MC-INR column is new in this topic**, exactly as in the Gaussian topic: the record figure predates the column, the volumes and the summary rows were already there, so the cut adds `--with_mcinr`.

**The numbers did not move.** Each panel's PSNR / SSIM / NMSE is that arm against the ground truth
and does not depend on which other arms are on the figure; dropping columns only changes which panel
the bold marks, since bold is the best of what is shown. Within these seven arms:

- **FLAIR** -- MC-GS ranks (transverse 180: PSNR #2, SSIM #1, NMSE #2; coronal 124: PSNR #1, SSIM #1, NMSE #1; sagittal 76: PSNR #1, SSIM #1, NMSE #1)
- **T1** -- MC-GS ranks (transverse 180: PSNR #2, SSIM #2, NMSE #2; coronal 124: PSNR #1, SSIM #1, NMSE #1; sagittal 76: PSNR #1, SSIM #1, NMSE #1)

`--keep` and the caption fix that comes with it are described in
`../T1FLAIR_AF20_v3_jiangyizhou/README.md` (same patch, applied by one generator to all ten scripts).
Without `--keep` the script is byte-for-byte the old one: re-rendering this topic's figure of record
reproduces it exactly.
