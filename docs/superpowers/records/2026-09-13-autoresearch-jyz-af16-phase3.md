# jiangyizhou AF16, phase 3 — widening the MC-GS lead (session record, 2026-09-12 / 13)

Branch `autoresearch-jyz-af16-phase3` (worktree `/data0/congcong/code/GS/mcgs_af16_wt`), opened from the phase-2 tip
5348815 on the user's instruction to reopen the frozen AF16 line and "拉大" the gap to the comparison methods.
Design `docs/superpowers/specs/2026-09-12-af16-phase3-nonlocal-design.md`; programme, rules and full round log
`autoresearch_af16_p3.md`; ledger `autoresearch_af16_p3.tsv`; probe read-out
`docs/superpowers/records/2026-09-12-af16-probe-c-readout.md`.

## 1. What the user set

Arena jiangyizhou AF16 starting from P6; levers = model-structure changes, the AF12 line's findings (as parameter
values only), acquisition-side design; learned priors excluded; success = head PSNR at least **1.0 dB** above the
strongest comparison arm on both contrasts, all six head cells first, sharpness not below L1-ESPIRiT's. Cards: empty
cards only (a split with the AF12 phase-2 session gave this line cards 4 and 5).

Bar, L1-ESPIRiT at its best λ on this subject: head PSNR 35.209 / 34.503, SSIM 0.9334 / 0.9201, HFEN 0.1286 / 0.1210,
sharpness 0.849 / 0.789.

## 2. Result — the phase-3 recipe

`p3_r3_e0001_ptv003_full4k`, registered in the topic as **`gs_joint_p3`** ("MC-GS (phase 3)"); P6 keeps `gs_joint_ar`.

**Recipe = P6 + three changes**: the wavelet prior becomes **db4 on its finest level only, T1 0.12 / FLAIR 0.10**
(the AF12 line's shape, as a parameter value); **common-edge cross-gradient prior λ 0.001** (`--lambda_edge`);
**phase-only TV λ 0.03** (`--lambda_ptv`). Everything else is P6: repaired cold start (head init, 3-NN scales, ZF-sourced
edge-TV threshold), 3.6M points, edge-TV λ_geo 0.10 / κ_q 0.85 / ε 0.04, kspace_high 2.2, 4000 iterations.

| head-only, 4000 it | T1 | FLAIR |
|---|---|---|
| **MC-GS phase 3** | **35.706** | **34.928** |
| MC-GS P6 (phase 1) | 35.552 | 34.807 |
| L1-ESPIRiT (best other arm) | 35.209 | 34.503 |
| **margin, phase 3** | **+0.497** | **+0.425** |
| margin, P6 | +0.343 | +0.304 |
| head SSIM (phase 3 / P6 / L1) | 0.9396 / 0.9351 / 0.9334 | 0.9261 / 0.9203 / 0.9201 |
| head HFEN (phase 3 / P6 / L1) | 0.1139 / 0.1218 / 0.1286 | 0.1121 / 0.1188 / 0.1210 |
| sharpness (phase 3 / P6 / L1) | **0.855** / 0.830 / 0.849 | **0.804** / 0.788 / 0.789 |

- **Every gate of the pre-registered criterion is met except the size of the margin**: six head cells first, and — for
  the first time in the three phases — **sharpness at or above L1-ESPIRiT's on both contrasts** (P6 was below on both).
- Against P6: +0.154 / +0.121 dB with every structural cell better.
- Seed floor at this recipe, 4000 it: **0.001 / 0.000 dB** (`p3_r3_ptv003_s43_full4k`).
- GS-single at the same recipe (`gs_single_p3`): T1 35.558, FLAIR 34.886 → **MC-GS − GS-single = +0.148 / +0.042 dB**.
  In phase 1 that difference was +0.037 / −0.003: the joint arm now wins on both contrasts, because the common-edge
  term is itself a coupling term.
- Compact three-plane figures 174 / 110 / 47, error inside the head only, orange in-panel metrics:
  `figures/t1flair_v3_af16_{T1,FLAIR}_row3planes_compact_174_110_47_headerr_p3.png` — MC-GS is first on PSNR, SSIM and
  NMSE in **every plane of both contrasts** (at 174/110/47 P6 lost FLAIR coronal SSIM to CS-SENSE).

**Capacity is a documented trade, not a null** (this contradicts the AF12 line, where the wavelet was db2 / 3 levels):
under the new wavelet shape the full recipe at 4000 iterations reads

| points | T1 | FLAIR | margins | min margin |
|---|---|---|---|---|
| **3.6M (the recipe)** | **35.706** | 34.928 | +0.497 / +0.425 | 0.425 |
| 4.8M (`p3_r6_n48_full4k`) | 35.697 | 34.962 | +0.488 / +0.459 | 0.459 |
| 6.0M (`p3_r6_n60_full4k`) | 35.675 | **34.977** | +0.465 / +0.474 | **0.465** |

All three are first on every head cell and clear both sharpness gates; the two margins cross at 6.0M, which is where
`min_margin_mag` peaks. **By the pre-registered promotion rule the recipe stays at 3.6M** (6.0M costs 0.031 dB of T1,
outside the 0.001 dB seed floor, and the full budget does not repair it). 3.6M maximises T1; 6.0M maximises the
worst-case margin and costs ~50 % more VRAM and wall time. The choice is the user's.

### Which single slice shows the phase-3 arm best (user request, 2026-09-13)

`scripts/best_slice_mcgs.py --mcgs_key gs_joint_p3 --single_key gs_single_p3` (now takes those two keys and a `--tag`)
was re-run for all three axes on the phase-3 arm. Gauge: head-masked slice PSNR, volume peak; a slice counts only if
its head area is at least half the largest slice's in both contrasts; the score is
**min over the two contrasts of [PSNR(MC-GS) − best of the seven competing arms]** (CG-SENSE, CS-SENSE, Joint SENSE,
L1-ESPIRiT, ZS-SSDU, 3D-DIP, 3D-INR — GS-single is ours, so it is excluded from "competing" but MC-GS is still required
to beat it). The runner-up is L1-ESPIRiT on essentially every slice.

Restricted to comfortably interior slices (the 80 %-head-area range shrunk by 5 at each end, so the pick cannot be a
head-edge artefact) the optimum is the point **transverse 135 / coronal 173 / sagittal 73**:

| plane, slice | lead over the best competing arm, T1 | FLAIR | current figure's slice | its lead, T1 / FLAIR |
|---|---|---|---|---|
| transverse 135 | +0.537 | +0.561 | 174 | +0.504 / +0.403 |
| coronal 173 | +0.541 | +0.577 | 110 | +0.442 / +0.444 |
| sagittal 73 | +0.529 | +0.508 | 47 | +0.705 / +0.464 |

The three planes meet at one voxel and that voxel is inside the head in both contrasts. At 135 / 173 / 73 MC-GS is
first on PSNR, SSIM **and** NMSE in every plane of both contrasts, as it is at 174 / 110 / 47. Figures
`figures/t1flair_v3_af16_{T1,FLAIR}_row3planes_compact_135_173_73_headerr_p3.png`.

**Caveat the user should weigh:** the optimum sits low and posterior — transverse 135 is at the level of the posterior
fossa, coronal 173 is behind the brainstem and carries a lot of neck — which is where the competing arms struggle
most. The gain over the existing 174 / 110 / 47 view is only about 0.1 dB on the weaker contrast, so if the figure
should show canonical supratentorial anatomy, 174 / 110 / 47 remains the better picture and is already first on every
metric. The full per-slice curves are in `logs/best_slice_mcgs{,_axis1,_axis2}_p3.json`.

### How much bigger can the gap be made? (follow-up, 2026-09-13)

**Whole slices are capped at about +0.55 dB.** Dropping the head-area gate entirely (`--min_area 0`) does not find a
better slice on any axis: the maxima stay at transverse 135 (+0.537), coronal 69 (+0.548) and sagittal 151 (+0.560),
and the slices below the gate only produce NaNs where the head is a few voxels. So no whole-slice view of this data
shows more than roughly half a dB.

Zooming does more, because whole-slice PSNR is diluted by the large smooth regions where every method agrees.
`scripts/best_zoom_mcgs.py` (sliding square window, same gauge, window at least 85 % head, MC-GS still required to beat
GS-single) and `scripts/best_cube_mcgs.py` (**one** 3-D box, its three orthogonal centre planes, six panels scored
together) give:

| what is shown | smallest lead over the best competing arm across all six panels |
|---|---|
| whole slices, best triple 135 / 173 / 73 | +0.508 dB |
| one 96³ box, centre (152, 104, 72) | +0.521 dB |
| one 64³ box, centre (180, 124, 72) | **+0.697 dB** |
| one 48³ box, centre (180, 124, 76) | **+0.824 dB** (largest panel +1.10, FLAIR coronal) |
| three separate 48² windows, one per plane | +1.09 / +1.04 / +0.99 dB |

The 64³ and 48³ optima land on the same place, (180, 124, 72–76) — the periventricular white matter around the lateral
ventricles, which the coronal face shows as the classic butterfly. That the two window sizes agree says this is a real
region, not a noise spike. Figures: `figures/t1flair_v3_af16_{T1,FLAIR}_cube64_180_124_72_headerr_p3.png` and
`..._cube48_180_124_76_headerr_p3.png`; `fig_three_planes_compact.py` grew a `--cube I J K W` option that sets the three
centre planes and crops every panel and every metric to the box (the sampling-mask thumbnail stays whole, it is k-space).

**Disclosure that belongs with any of these numbers:** the box is chosen by maximising the gap against the ground
truth, so the smaller the box the larger the quotable number, and the choice is a selection effect. One 3-D box scored
on all six panels at once is the honest version of a zoom — three independently placed 2-D windows are three separate
cherry-picks. The defensible headline stays the whole-volume margin (+0.497 / +0.425 dB).

**The figure the user asked for** (whole slices with the zoom pasted in): `fig_three_planes_compact.py --slices I J K
--inset W` draws the three full slices through the box centre, outlines the W x W face in green on every panel, and
pastes the magnified box in the bottom-left corner of every panel — image rows and error rows alike. **User decision
2026-09-13: the zoom carries no numbers of its own and its outline and border are the same orange-yellow as the
in-panel metrics, drawn at line width 2.2 (inset border 2.8)**; the box's PSNR is printed to stdout by the script instead. Head-masked PSNR inside the 48 box at
(180, 124, 76), MC-GS against the best competing arm of that panel:

| panel | MC-GS | best competing arm | lead |
|---|---|---|---|
| T1 transverse 180 | 42.30 | 41.41 (Joint SENSE) | +0.892 |
| T1 coronal 124 | 41.94 | 41.11 (CS-SENSE) | +0.826 |
| T1 sagittal 76 | 42.37 | 41.53 (Joint SENSE) | +0.836 |
| FLAIR transverse 180 | 40.26 | 39.43 (L1-ESPIRiT) | +0.824 |
| FLAIR coronal 124 | 39.47 | 38.37 (L1-ESPIRiT) | **+1.102** |
| FLAIR sagittal 76 | 39.57 | 38.51 (Joint SENSE) | **+1.060** |

Files: `figures/t1flair_v3_af16_{T1,FLAIR}_row3planes_compact_180_124_76_inset48_headerr_p3.png` and the 64-box
version `..._180_124_72_inset64_...` (64 box: T1 leads +0.72 / +0.75 / +0.72, FLAIR +0.74 / +0.72 / +0.69). The cropped-only versions (`..._cube48_...`, `..._cube64_...`) stay as well.

**Print output and the data bundle (2026-09-13).** `--pdf` writes a vector PDF beside each PNG: text, boxes, colorbar
frames and tick labels stay vector (fonts embedded) and only the image panels are rasterised, at `--pdf_dpi` 600
(`pdfimages -list` reports 600 ppi; about 5.6 MB per figure). All four figures exist as PDF and PNG.
`scripts/export_phase3_bundle.py` collects everything behind the figure of record into
`Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/export/af16_phase3/` — 43 files, 1.95 GiB, real copies so the folder
travels on its own: every arm's volume for both contrasts plus the ground truth (`volumes/`), the 3-D sampling mask and
the 2-D pattern it came from (`mask/`), the topic summary, the competitor bar, the preparation report, the extra-arm
registry and the four phase-3 scorer outputs (`logs/`), the four figures as PDF and PNG, a `MANIFEST.tsv` giving each
file's shape, dtype, size and md5, and `checksums.md5` (43/43 verify). Its `README.md` states the acquisition, the
recipe, the head-only table, what each figure shows and what is deliberately left out (the coil k-space and ESPIRiT
maps, shared with the AF 12 topic, and the Gaussian checkpoints, which live in the scratch `results_arc/`). The bundle
carries the subject's real name in its paths and stays on the server.

Figure conventions the user fixed on 2026-09-13, all as switches on `fig_three_planes_compact.py`: the zoom box and its
inset border are the in-panel orange-yellow at line width 2.2 / 2.8; `--no_metrics` takes every number off the plane
panels (the panel label keeps only the plane name, the slice indices are in the title) and the ranking still goes to
stdout; `--gray_cbar` puts a grey bar with that plane's window right of every image row and the jet error bar right of
every error row; no title in the header row is bold. `--inset_corner TRA COR SAG` places each plane's inset; the transverse one goes **top-right** because bottom-left
leaves only 3 voxels between the inset border and the outlined box (top-right leaves 11), while the coronal and
sagittal insets stay bottom-left and are 89 and 59 voxels clear. `--cbar_planes` picks which rows carry the two bars.

**User decision 2026-09-13: the AF16 figure of record is the 48 version**, made with

```
scripts/fig_three_planes_compact.py --contrast T1|FLAIR --slices 180 124 76 --inset 48 --inset_corner tr bl bl \
    --mask_error --gray_cbar --cbar_planes 0 --tra_rot 1 --err_vmax 0.1 --mcgs_key gs_joint_p3 --single_key gs_single_p3 --tag _p3
```

i.e. the in-panel PSNR / SSIM / NMSE stay (best of each plane in bold), the zoom carries no numbers of its own, and the
grey and error bars appear on the transverse row only. The 64 version (`--slices 180 124 72 --inset 64`) is kept in the
same style as the alternative with the wider zoom field: it trades 0.13 dB of the smallest lead (+0.697 against +0.824)
for a larger field of view, and its sagittal slice is 72 rather than 76 because each box size was optimised separately.
Note the side effect of `--cbar_planes 0`: the coronal and sagittal rows have their own grey windows and now show no
bar for them.

**Transverse orientation (user 2026-09-13):** the transverse panel is turned a quarter turn so the eyes point up
(`--tra_rot`, default 1; `np.rot90(x.T, 1)`, verified on the orbit slice, transverse 120, where the two globes move
from the left edge to the top). The anterior-posterior axis is then vertical, so the transverse panel is taller than
wide as the anatomy is (about 180 mm front to back against 150 mm across) and that row grows accordingly. `--tra_rot 0`
restores the orientation the topic's older figures use; `fig_row_final.py` and `fig_three_planes.py` are unchanged and
still draw the old one. The zoom box is now derived by pushing a 3-D box mask through the same display transform, so it
lands correctly in any orientation, and the script asserts that the transformed box is still the requested square.
None of the numbers change. The label on the leftmost panel of each row is the plane name alone — `transverse`,
`coronal`, `sagittal` — with no slice index (user 2026-09-13); the indices stay in the title.

**Error-map scale (user 2026-09-13):** the error maps are back on the house-standard fixed [0, 0.1]. Reason, measured on this data: inside the head the error / max|GT| has P90 0.0266 and P99 0.0506 for
MC-GS, 0.0280 and 0.0542 for L1-ESPIRiT, 0.0370 and 0.0807 for CG-SENSE — at a ceiling of 0.1 every arm but CG-SENSE
sits in the bottom half of the colormap and all the panels read as uniform dark blue, which is what made the methods
look indistinguishable. 0.05, 0.08 and 0.09 were tried on the way (0.05 saturates about 1 % of head voxels, the others almost none, and all
three separate the arms far better than 0.1); the user came back to **0.1**, the house convention, so the figures of
record use it and `--err_vmax` is the switch for the tighter scales. The outer end of each bar is flush with its row and the two
ends that face each other are pulled in by 0.10 in (`CBTRIM`), so the grey bar's 0 and the error bar's ceiling are both
labelled and clearly apart; the error bar's ticks are round steps that always include the ceiling.
The scale is
**fixed and identical for every panel**, so panels remain directly comparable; only the ceiling changed. `--err_vmax 0.1`
restores the house convention.

**Not done, on request 2026-09-13:** de-tuning GS-single to land near 36 dB so the gap would look larger. GS-single is
the single-contrast ablation of the same recipe; weakening it on purpose would misstate what the single-contrast
variant achieves and the comparison would no longer be like-for-like. The honest routes to a larger visible gap were
offered instead: this error scale (taken), dropping the GS-single column (it is our arm, not a competitor), the
existing un-tuned cold-start single arm (head 32.331 / 32.137, a 3.2 dB gap, honestly labelled), and running a higher
acceleration where the coupling is worth more (joint − single went from +0.36 / −0.02 at AF12 to +0.56 / +0.27 at AF16
for the cold-start pair).

**The 1.0 dB target was not reached.** The margin roughly doubled (T1 +45 %, FLAIR +40 %) and every quality gate is now
met, but +1.0 dB over a comparison arm whose λ is itself tuned on this subject's ground truth is out of reach of the
lever classes this phase was allowed to use — see §4 and §6.

## 3. What moved it, in order of size

1. **The AF12 wavelet shape on P6** (db4, finest level only, T1 0.12 / FLAIR 0.10): +0.177 / +0.081 dB at 4000 it.
   Phase 2 had tested the same shape stacked on `p6ptv` and found no magnitude gain; on P6 it works — the phase-TV
   term at 0.05 was in the way.
2. **Common-edge cross-gradient prior, λ 0.001**: +0.056 / +0.051 dB at 1500 it, SSIM and HFEN better on both.
   Bracketed: 0.0005 lower, 0.002 level, 0.005 collapses (−0.13 dB).
3. **Phase-only TV, λ 0.03**: the sharpness lever. On the db4 base the trade curve is
   0.01 → sharp 0.839/0.787 at −0.005/+0.019 dB; 0.02 → 0.846/0.792 at −0.033/+0.013; 0.03 → 0.850/0.799;
   0.05 → 0.865/0.815 at −0.182/−0.069. 0.03 is the cheapest point clearing both sharpness gates.

## 4. Lever classes closed in this phase (do not retry)

- **Non-local self-similarity (RED with a lagged BM4D target)** — the phase's main bet, implemented as
  `paired_gs/losses/nonlocal_red.py` (config `loss.lambda_nl`, `nl_refresh`, `nl_start`, `nl_sigma_rel`, `nl_profile`,
  `nl_denoiser`, `nl_threads`, `nl_sharpen`; 14 tests; default 0 leaves the trainer bit-identical). Three settings —
  σ_rel 0.04 at λ 4, σ_rel 0.01 at λ 1, and σ_rel 0.02 at λ 1 with BM4D's transform-domain sharpening α 1.3 — all lose
  sharpness (up to −0.16 / −0.20) and never gain FLAIR PSNR. Mechanism, from the logs: at the refresh the weighted term
  matches the two data terms, within 50 iterations the fit snaps onto the denoised target (R_nl → 1e-3) and the
  k-space residual rises 15 %. The residual the denoiser removes is real high-frequency signal, which is what the
  2026-09-04 data audit said of this data. Cost: BM4D runs on the whole volume (the aliased ZF support fills both PE
  axes, so the head bounding box is the full grid), 156 s per contrast per refresh.
- **Adaptive density control** (never enabled on this line before): gradient score and evidence-energy score, split
  quantile 0.90, cap 5.0M, until iteration 1000 — both null (+0.003 / +0.003 and +0.002 / +0.013 with slightly worse
  HFEN). The alternating split/prune ended at +0.23 % points: at 3.6M the representation is not capacity-limited.
- **Vectorial TV** (λ 0.005 and 0.02): smooths — +0.02 dB of T1 at best, HFEN and sharpness worse. Same signature on
  the probe mask.
- **Contrast-private residual Gaussians** (0.3M): null (+0.005 / +0.015, nothing else moves).

## 5. The acquisition probe (full read-out in its own record)

A complementary FLAIR mask drawn **inside the vendor shutter** (2966 samples, all in-support, AF 11.51 against FLAIR's
own support against 12.49 today, 733 samples in common with the current mask) lifts **every** arm by about the same
amount: L1-ESPIRiT +0.368 dB, MC-GS +0.347. The FLAIR margin goes from +0.222 to +0.201, T1 does not move, and adding
vectorial TV to carry the new samples across contrasts moves T1 by +0.03. Both pre-registered thresholds (+0.3 dB of
margin, +0.15 dB of T1) are missed, so the full re-run of every arm on a two-mask design was not done. The in-shutter
mask is nevertheless a **better FLAIR acquisition for everyone** — 233 of the current mask's 2966 samples fall outside
the shutter and are wasted.

**By-product, never measured before:** the known-zero region (the shutter, used by the GS arms only) is worth
**−0.013 dB** of FLAIR head PSNR on the current mask — i.e. nothing — but **+0.225 dB** on the in-shutter mask, where
no sampled line tells the model the corners are empty. The "GS arms use the shutter, the others do not" asymmetry
carried in the phase-1 and phase-2 records is therefore not a material advantage on the mask the results use.
Enabling it needs `data.<x>.known_zero_support` (new in this phase, `known_zero_from_support` + 5 tests) when the
sampling lies inside the shutter.

## 6. Caveats

- One subject; the MC-GS knobs and the classical λ are both selected on its ground truth — fair, not held out.
- Magnitude gauge. The complex-gauge ranking was not re-measured for the phase-3 recipe; phase 2 showed phase TV lifts
  it, and this recipe carries phase TV at 0.03 (phase 2 used 0.05), so it is likely better than P6 there, but that is
  an expectation, not a measurement.
- Nominal AF 16 (T1 15.19x, FLAIR 12.49x of its acquired k-space).
- Whole-volume SSIM stays sixth (the air-noise gauge), as in every phase.
- The `p3_r0_control` row (the reference for the "db4 base minus control" delta) ran before the non-local module was
  merged; `p3_r5_control_postmerge` re-runs plain P6 after the merge to close that gap. The module is guarded by
  `lambda_nl > 0` and an import-isolation test.

## 7. Open decisions (for the user)

1. **Which recipe is quoted externally.** P6 is the current answer (CLAUDE.md, user decision 2026-09-12). The phase-3
   recipe is better on every head cell and on sharpness, and its figures exist with the `_p3` suffix.
2. **The 1.0 dB target.** Not reached and, under "no learned priors, same data, same mask", not reachable by anything
   this phase found. The options are: accept the doubled margin; re-state the target as a margin ratio; or allow a
   learned prior (and then give the comparison arms one too, which changes what the comparison means).
3. **Capacity**: 3.6M (as promoted, maximum T1) or 6.0M (maximum worst-case margin, +0.04 dB on the weaker side, ~50 % more VRAM and time). Nothing else separates them.
4. A held-out subject at AF16 with the recipe and λ frozen — still never run, from phase 1 onward.
5. Whether to adopt the in-shutter FLAIR mask for future acquisitions (it helps every method, not the margin).
