# Retrospective nominal AF 20 on the 3D T1 + FLAIR pair — jiangyizhou

Sixth topic in the series. **Read `../T1FLAIR_AF16_v3_jiangyizhou/README.md` first** (the same
subject at nominal AF 16) and, through it, `../T1FLAIR_AF16_retro/README.md` and
`../T1T2_AF12_retro/README.md`: the data provenance, the geometry adjudication, the ESPIRiT recipe,
the known-zero region, the arms and the scoring gauge are documented there and not repeated.

## What this topic adds

**Exactly one thing differs from `T1FLAIR_AF16_v3_jiangyizhou`: the mask.** Same subject, same CFL
k-space, same two ESPIRiT maps, same ground truth, same gauge, same arms, same MC-GS recipe. The
k-space and the maps are not copied — `data/ksp_*.h5` and `data/csm_espirit_*.h5` are symlinks into
`../T1FLAIR_AF12_v3_jiangyizhou/data/`, and `scripts/prep_04_export.py` asserts the symlink so it can
never write through them. Nothing is reused from the AF 16 results: every arm depends on the mask.

Created with `scripts/setup_af20_jyz.sh`; configs `configs/t1flair_v3_af20_jyz{,_singleT1,_singleFLAIR}.yaml`
in the worktree `/data0/congcong/code/GS/mcgs_af16_wt` on branch `autoresearch-jyz-af20`.

| mask | generator call | PE points | of the plane | AF, full plane | T1 AF vs own support | FLAIR AF vs own support |
|---|---|---|---|---|---|---|
| AF 12 (`../T1FLAIR_AF12_v3_jiangyizhou`) | `vdPoisMex(256,176,256,176, 3, 4, 24, 0, 1)` | 3847 | 8.54 % | 11.71 | 11.71 | 9.74 |
| AF 16 (`../T1FLAIR_AF16_v3_jiangyizhou`) | `vdPoisMex(256,176,256,176, 4, 4, 24, 0, 1)` | 2966 | 6.58 % | 15.19 | 15.19 | 12.49 |
| **AF 20 (this topic)** | `vdPoisMex(256,176,256,176, 5, 4, 24, 0, 1)` | **2427** | **5.39 %** | **18.56** | **18.56** | **15.13** |

Same house generator and convention (`accely = 4`, `accelx = AF/4`, `calib = 24`, `ellipse = 0`,
`pp = 1`): a nominal AF 20 lands at 18.56 on the full plane the way the nominal AF 16 landed at
15.19 and the nominal AF 12 at 11.71. FLAIR keeps its elliptical vendor shutter (support 0.7577), so
against the data it actually acquires its acceleration is 15.13 — which is, to within 0.4 %, the
acceleration T1 saw at AF 16. The 24x24 calibration block keeps 82.3 % of its points (81.9 % at
AF 16). One mask for both contrasts, `data/mask_vdpois_256x176_af20.mat`, broadcast along the
readout into `data/mask_af20_3d.h5`; all of it measured in `logs/prep_04_export.json`.

## Method under test

The **AF 16 phase-3 recipe, unchanged** (user decision 2026-09-13): P6's repaired cold start
(init_mode head, 3-NN scales, edge-TV threshold from ZF), 3.6M points, edge-TV λ_geo 0.10 /
κ_q 0.85 / w_floor 0.10 / ε 0.04, kspace_high 2.2, warmup 500, **wavelet db4 on its finest level
only (T1 0.12 / FLAIR 0.10), common-edge cross-gradient prior λ 0.001, phase-only TV λ 0.03**,
4000 iterations. The argument list lives in one place, `mcgs_af16_wt/run_af20_mcgs.sh`. Nothing was
re-tuned for AF 20 — that is the point of the study: the same parameters, a harder mask.

The classical arms are re-swept: their λ is chosen on this subject against the ground truth, the
same privilege the MC-GS recipe had when it was designed, with the grids centred on the AF 16
optima (`scripts/run_classical_af20.sh`).

## Results

<!-- TABLES:BEGIN -->
**T1** (sorted by head-only PSNR; AF16 column = the same arm in `T1FLAIR_AF16_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @AF16 | AF20 − AF16 |
|---|---|---|---|---|---|---|---|---|
| MC-GS | 38.482 | 0.9144 | **34.832** | **0.9289** | 0.1355 | 0.833 | 35.706 | -0.87 |
| GS-single | 38.335 | 0.9205 | 34.667 | 0.9257 | 0.1414 | 0.837 | 35.558 | -0.89 |
| CS-SENSE (TV, converged) (λ 0.0007) | 37.808 | 0.9418 | 34.003 | 0.9191 | 0.1683 | 0.781 | 34.945 | -0.94 |
| joint CS-SENSE (JTV) (λ 0.001) | 37.736 | 0.9421 | 33.885 | 0.9183 | 0.1733 | 0.768 | 34.831 | -0.95 |
| CS-SENSE | 37.269 | 0.9276 | 33.418 | 0.9131 | 0.1864 | 0.725 | 34.409 | -0.99 |
| ZS-SSDU | 36.795 | 0.9440 | 32.892 | 0.9108 | 0.2241 | 1.014 | 34.150 | -1.26 |
| L1-SPIRiT (λ 0.001) | 36.572 | 0.9149 | 32.882 | 0.8936 | 0.1759 | 0.903 | 33.788 | -0.91 |
| CG-SENSE | 34.030 | 0.7646 | 31.425 | 0.8665 | 0.2741 | 0.791 | 32.354 | -0.93 |
| 3D-DIP | 34.926 | 0.8541 | 30.996 | 0.8537 | 0.3034 | 0.640 | 30.994 | +0.00 |
| MC-INR (Niessen 2025) | 30.432 | 0.5642 | 30.969 | 0.8272 | 0.2893 | 0.913 | 31.590 | -0.62 |
| 3D-INR | 31.826 | 0.6520 | 29.456 | 0.7596 | 0.2723 | 1.308 | 30.375 | -0.92 |
| ZF | 29.744 | 0.8165 | 25.639 | 0.6636 | 0.7206 | 0.263 | 25.841 | -0.20 |

**FLAIR** (sorted by head-only PSNR; AF16 column = the same arm in `T1FLAIR_AF16_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @AF16 | AF20 − AF16 |
|---|---|---|---|---|---|---|---|---|
| MC-GS | 37.892 | 0.9028 | **34.179** | **0.9142** | 0.1290 | 0.776 | 34.928 | -0.75 |
| GS-single | 37.801 | 0.8875 | 34.127 | 0.9127 | 0.1313 | 0.778 | 34.886 | -0.76 |
| joint CS-SENSE (JTV) (λ 0.0007) | 37.358 | 0.9364 | 33.544 | 0.9044 | 0.1522 | 0.726 | 34.355 | -0.81 |
| CS-SENSE (TV, converged) (λ 0.0005) | 37.241 | 0.9328 | 33.494 | 0.9049 | 0.1523 | 0.698 | 34.302 | -0.81 |
| CS-SENSE | 36.883 | 0.9107 | 33.027 | 0.8968 | 0.1571 | 0.616 | 33.806 | -0.78 |
| L1-SPIRiT (λ 0.001) | 36.604 | 0.9117 | 32.824 | 0.8863 | 0.1430 | 0.849 | 33.720 | -0.90 |
| ZS-SSDU | 36.642 | 0.9351 | 32.660 | 0.8958 | 0.1962 | 1.058 | 34.005 | -1.34 |
| CG-SENSE | 34.868 | 0.7963 | 32.089 | 0.8717 | 0.1972 | 0.718 | 32.826 | -0.74 |
| MC-INR (Niessen 2025) | 32.459 | 0.6481 | 31.666 | 0.8486 | 0.2079 | 0.808 | 32.243 | -0.58 |
| 3D-DIP | 34.572 | 0.8167 | 30.637 | 0.8269 | 0.2768 | 0.515 | 30.650 | -0.01 |
| 3D-INR | 32.851 | 0.7233 | 29.809 | 0.7716 | 0.2448 | 1.040 | 30.257 | -0.45 |
| ZF | 31.548 | 0.8329 | 27.636 | 0.6970 | 0.6874 | 0.251 | 27.879 | -0.24 |

**MC-GS joint minus MC-GS single**

| channel | whole PSNR | head PSNR | head SSIM | (AF16: whole / head) |
|---|---|---|---|---|
<!-- TABLES:END -->

### L1-ESPIRiT dropped from the comparison (user, 2026-09-15)

L1-ESPIRiT is no longer carried as a comparison arm: it is gone from the tables above and from the
figures. **Nothing was deleted** — its reconstructions, its lambda sweep logs and its row in
`logs/summary.json` are untouched, and `DROP` in the table generator is the single line that decides
what is listed. Sections of this file written before 2026-09-15 still discuss it; they are the record
of what was measured at the time, and are left as they are.

**It changes no headline number.** GS-single counts as a comparison arm (user, 2026-09-13), and
GS-single — not L1-ESPIRiT — is the strongest comparison arm on both contrasts here, as in every
topic of this series. The margin that MC-GS is quoted on is therefore MC-GS − GS-single, **+0.165 dB
on T1 and +0.052 dB on FLAIR**, and dropping L1-ESPIRiT leaves it untouched.

What does change is the strongest arm *below* the GS pair:

| head-only | T1 | FLAIR |
|---|---|---|
| strongest comparison arm (unchanged) | GS-single 34.667 | GS-single 34.127 |
| strongest non-GS arm, was | L1-ESPIRiT 34.371 | L1-ESPIRiT 33.688 |
| strongest non-GS arm, now | CS-SENSE (TV, converged) 34.003 | joint CS-SENSE (JTV) 33.544 |

So a sentence of the form "MC-GS leads the best classical arm by X" gets 0.368 dB bigger on T1 and
0.144 dB bigger on FLAIR with L1-ESPIRiT gone, purely from the arm no longer being shown. The
headline margin over GS-single does not move.

## Regenerating

```
# mask (MATLAB R2021a + the house vdPoisMex)
matlab -batch "run('scripts/prep_02_mask.m')"
PYTHONNOUSERSITE=1 PYTHONPATH= python scripts/prep_04_export.py      # 3-D mask + the AF read-out

# classical arms, every lambda (GPU for TV / joint TV, CPU BART for L1-ESPIRiT)
GPU_BASE=7 GPU_TV=7 GPU_L1=7 GPU_JTV=5 scripts/run_classical_af20.sh

# MC-GS, the AF16 phase-3 recipe (worktree mcgs_af16_wt, branch autoresearch-jyz-af20)
SAVE_VOLUMES=1 ./run_af20_mcgs.sh af20_p3_joint_full4k     6 4000
SINGLE=T1    SAVE_VOLUMES=1 ./run_af20_mcgs.sh af20_p3_singleT1_full4k    7 4000
SINGLE=FLAIR SAVE_VOLUMES=1 ./run_af20_mcgs.sh af20_p3_singleFLAIR_full4k 7 4000

# table and figures
scripts/finalize_af20_jyz.sh af20_p3_joint_full4k af20_p3_singleT1_full4k af20_p3_singleFLAIR_full4k
```

## Figures

- `figures/t1flair_v3_af20_{T1,FLAIR}_row3planes_compact_180_124_76_inset48_headerr_p3.{png,pdf}` — the AF 16 figure of
  record's layout, slices and 48-voxel box (chosen on the AF 16 phase-3 arm, not re-optimised here).
- `figures/t1flair_v3_af20_{T1,FLAIR}_row3planes_compact_174_110_47_headerr_p3.png` — the plain compact view.
- `figures/t1flair_v3_af20_mask_T1_vs_FLAIR.png` — the mask against each contrast's acquired region.
- `figures/t1flair_v3_af20_mask_T1_FLAIR_reinspect.png` (2026-09-14, `scripts/fig_mask_reinspect.py`) — the mask as each channel
  loads it: T1 and FLAIR read the same file (0 lines differ); T1 has data on all 2427 lines (AF 18.56), FLAIR on 2257, with 170
  lines in its unacquired corners (AF 19.96). AF here = PE-plane lines / sampled lines carrying that contrast's data (user
  decision 2026-09-14); the "nominal AF 20, actual T1 18.56x / FLAIR 15.13x" labels elsewhere in this topic are the older
  convention and have not been relabelled. The cross-AF review of this subject (AF 12 / 16 / 20, every arm re-scored with one
  scorer, same-slice trends) is the topic `T1FLAIR_AF12_16_20_trend_v3_jiangyizhou/`.
- `figures/t1flair_v3_af20_{T1,FLAIR}_row3planes_compact_180_124_76_inset48_headerr_p3_mcinr.{png,pdf}` — the same layout
  with the MC-INR (Niessen 2025) column added (`fig_three_planes_compact.py --with_mcinr`), as in the AF 16 topic.

MC-GS is first on PSNR, SSIM and NMSE in every plane of both contrasts in both layouts. The 48-box leads are smaller
than at AF 16 (T1 +0.571 / +0.487 / +0.653, FLAIR +0.448 / +0.900 / +0.703): the box was picked for the largest gap on
the AF 16 mask and regresses toward the whole-volume margin on this one. Quote the whole-volume margin.

## Findings

- Margin over L1-ESPIRiT (the strongest competing arm) **+0.461 / +0.491** against +0.497 / +0.425 at AF 16: it holds,
  it does not grow. The SSIM and HFEN leads widen on both contrasts.
- T1 sharpness 0.833 is below L1-ESPIRiT's 0.845 (it was above at AF 16); FLAIR stays above (0.776 against 0.732).
- MC-GS − GS-single: +0.165 / +0.052 dB (AF 16 +0.148 / +0.042).
- ZS-SSDU pays the most for the extra acceleration (−1.26 / −1.34 dB); 3D-DIP does not move because it is budget-limited
  (best held-out residual at the last iterations at both accelerations).
- The comparison is conservative for MC-GS (classical λ re-swept at AF 20, MC-GS parameters frozen at AF 16).

## Status and caveats

Complete (2026-09-14 00:42). One subject. The house masks are independent draws (698 of the 2427 AF 20 points are in
the AF 16 mask).

**Mask-draw variance (2026-09-14, three more AF 20 draws b / c / d, MC-GS joint and L1-ESPIRiT only;
`scripts/score_mask_draws.py`, `logs/mask_draws_summary.json`):** the margin over L1-ESPIRiT is **+0.459 ± 0.002** (T1)
and **+0.488 ± 0.004** (FLAIR) over the four draws. Each arm's absolute head PSNR moves far more with the draw
(SD 0.009 / 0.007 on T1, 0.040 / 0.040 on FLAIR) but the two move together (correlation +0.97 / +0.99), so the paired
margin is 4–10x steadier. The AF 16 → AF 20 margin shifts (−0.038 / +0.063) are 16 / 15 draw SDs. T1 sharpness is below
L1-ESPIRiT's on all four draws; FLAIR's is above on all four. Draw masks: `data/mask_vdpois_256x176_af20_d{b,c,d}.mat`
(+ `mask_af20_d{b,c,d}_3d.h5`), configs `configs/t1flair_v3_af20_jyz_d{b,c,d}.yaml` in the worktree.
`vdPoisMex` seeds from `time(0)`: **do not re-run `prep_02_mask.m` into this `data/`** — the mask (md5
1c90baf850a8c1e64b43f6c64146e2c9) cannot be regenerated. MC-INR (Niessen 2025) was run by a parallel session on this
topic's own AF 20 mask (`../T1FLAIR_MCINR_Niessen2025_v3_jiangyizhou/`) and is registered here through
`scripts/score_mcinr_arm.py` (re-scored on this topic's gauge, worst relative disagreement 2.2e-7): head PSNR 30.969 /
31.666, eighth / seventh of the ten comparison arms. User decisions 2026-09-14: no held-out subject (huangyuwei not
considered); the masks stay out of git for now. Details: the record `docs/superpowers/records/2026-09-14-af20-jyz.md` in the worktree.

## L1-SPIRiT registered as a comparison arm (2026-09-15)

Method, GPU port, fidelity against Lustig's MATLAB toolbox and the solver change (`pocsSPIRiT`
diverges on this data and the original diverges with it; the arm runs a damped proximal-gradient
step with FISTA momentum on the same objective) are documented once in
`../T1FLAIR_AF16_v3_jiangyizhou/README.md`. Same scripts and same 150 iterations, on this topic's
mask. **L1-ESPIRiT stays in the table; nothing was replaced.**

Same grid as the other two topics (kernel size 3 and 5 x lambda in {0, 5e-4, 1e-3, 2e-3, 4e-3,
8e-3}); both contrasts peak at ksize 5 / lambda 1e-3, interior to it.

| head-only, AF 20 | T1 | FLAIR |
|---|---|---|
| **L1-SPIRiT** | **32.882** / 0.8936 / 0.1759 | **32.824** / 0.8863 / 0.1430 |
| plain SPIRiT (lambda 0) | 29.413 | 29.299 |
| ZS-SSDU | 32.892 | 32.660 |
| GS-single, the strongest comparison arm | 34.667 | 34.127 |
| CS-SENSE (TV, converged) / joint CS-SENSE, the strongest non-GS arm | 34.003 | 33.544 |

- **Sixth of eleven comparison arms on T1 — 0.010 dB behind ZS-SSDU, a tie — and fifth on FLAIR,
  0.164 dB ahead of it.** Eleven counts GS-single, a comparison arm here (user, 2026-09-13), and
  excludes L1-ESPIRiT, dropped 2026-09-15. It is 1.785 / 1.303 dB behind GS-single, the strongest
  comparison arm, and 1.121 / 0.720 dB behind the strongest non-GS arm.
- **Its fall-off matches the field.** AF16 -> AF20 costs it 0.906 dB on T1 and 0.896 on FLAIR,
  against 0.84 to 0.99 for every classical arm; ZS-SSDU alone falls faster (1.26 / 1.34). Across
  the series the arm is monotone, as the cross-AF review found every arm except 3D-DIP to be:
  T1 35.017 -> 33.788 -> 32.882, FLAIR 35.103 -> 33.720 -> 32.824.
- **It changes no margin.** MC-GS and GS-single keep the top two rows on both contrasts.

**Figure set of record since 2026-09-16: the `_zf_caps` variant at publication resolution.**
`figures/t1flair_v3_af20_{T1,FLAIR}_row3planes_compact_180_124_76_inset48_headerr_p3_mcinr_spirit_zf_caps.{png,pdf}`
— the same slices, 48-voxel box, grey windows, error scale, arms and gauge as the `_mcinr_spirit` set described
below, which it supersedes without overwriting: every plane's PSNR / SSIM / NMSE line is identical. What changed
is a zero-filling column after GT and the caps typography (user 2026-09-16, `--with_zf --caps`, with labels
containing `SPIRiT` exempt from upper-casing), the PNG at `--png_dpi 200` (6453 x 4352 px, which is 911 dpi at a
180 mm page width) and TrueType fonts in the PDF (`pdf.fonttype = 42`; matplotlib's Type 3 default is rejected by
IEEE and Elsevier submission systems). All four topics' copies are gathered in
`Results_20260908/T1FLAIR_AF12_16_20_24_figures_hires/`; verification and the render command are in
`mcgs_af16_wt/docs/handoff/2026-09-16-figures-hires.md`.

**Figures with the k-space arm in the L1 column** (user request 2026-09-15). The figure of record
re-rendered with `--l1_arm spirit` (L1-SPIRiT replaces L1-ESPIRiT in that column), same slices,
style, gauge and recipe keys. An eleven-column `--l1_arm both` render was made first and then
deleted: the user dropped L1-ESPIRiT from the comparison on 2026-09-15. The flag still regenerates it. **The existing `_p3_mcinr`
pair is untouched**; the flag appends its own suffix:

```
figures/t1flair_v3_af20_{T1,FLAIR}_row3planes_compact_180_124_76_inset48_headerr_p3_mcinr_spirit.{png,pdf}
```

served from `http://localhost:8765/t1flair_v3_af20/`, mirrored in `/home/congcongliu/figs/t1flair_v3_af20/`.
Rendered by `scripts/fig_three_planes_compact.py --with_mcinr`.

Re-running the same command with `--l1_arm espirit` reproduces the existing figure **pixel for
pixel** (0 pixels differing), which is what establishes that this is the command that made it, and
the `_spirit` render differs from it only inside one column slot (a 337-pixel span, 0.96 of the
1/11 width). The swap makes MC-GS's lead look larger, because the column it removes was the
strongest comparison arm; how much is quantified in "L1-ESPIRiT dropped from the comparison" above.
The same note in full is in `../T1FLAIR_AF16_v3_jiangyizhou/README.md`.

**AF label corrected, 2026-09-15.** The title's acceleration now reads `AF_vs_full_rect` — whole
phase-encode plane over the points that contrast actually has data on, the user's 2026-09-14
definition, the same number BART's `pics` prints as `Acc`. It used to read `AF_vs_own_support`,
against the contrast's own vendor shutter, which is the superseded gauge; only FLAIR moves
(15.13x -> 19.96x), T1 was already right at 18.56x. Both numbers were always in
`logs/prep_04_export.json`; the figure took the wrong field. Consequence for the check above: the
pixel-identity re-render was done BEFORE this fix, so repeating it now reproduces the old figure
exactly on T1 and everywhere but the title on FLAIR.

**The figures of record themselves, re-archived with the corrected label (user, 2026-09-15: keep the
old, re-archive the new).** The `_p3` and `_p3_mcinr` pairs at 180/124/76 inset48 -- the historical figures, L1-ESPIRiT
column and all -- were re-rendered under their canonical names by `scripts/archive_and_relabel_figs.sh`
(in the AF16 topic) with the corrected FLAIR title; the originals are in `figures/archive_aflabel_pre20260915/` and in the
served mirror `/home/congcongliu/figs/t1flair_v3_af20/archive_aflabel_pre20260915/`, PNG and PDF, 8 files. Verified
against those originals: every T1 re-render is **pixel-identical**, every FLAIR re-render differs
**only in the title band** (image rows 13-35 of 2829). So the canonical names now carry the right
label and nothing else changed.

One thing went wrong on the way and is recorded here rather than hidden: the first pass of that
script built the archive file name with a doubled underscore, matched nothing, and the re-render
overwrote the originals before they had been moved. They were recovered from the committed
snapshot of this topic in the framework worktree (`mcgs_af16_wt/t1flair_af20_v3_jiangyizhou/figures`, PNGs, 2026-09-13/14) and from this topic's
`export/` bundle (the PDFs the worktree does not carry), then the verification above was run against
the recovered files. The script is fixed. Older renders of this topic (the 174/110/47 pairs)
still carry the old FLAIR label and were not touched.

## Seven-method cut of the figure of record (user, 2026-09-18)

The user asked for the same figure with only **CS-SENSE, Joint SENSE, ZS-SSDU, 3D-INR, MC-INR,
GS-single and MC-GS**. ZF, CG-SENSE, L1-SPIRiT and 3D-DIP are dropped; **GT keeps its column** --
it is what the error maps are taken against, and the sampling mask of each plane sits in its cell
of the error rows. Eight columns instead of twelve; everything else is the figure of record
unchanged: slices 180 / 124 / 76, 48x48 inset (top-right / bottom-left / bottom-left), head-only
error maps in jet on the fixed [0, 0.1] scale, per-plane grey window, transverse eyes-up, caps
typography, `--title_fs 22`, the same MC-GS and GS-single keys (`gs_joint_p3` / `gs_single_p3`).

```
figures/t1flair_v3_af20_{T1,FLAIR}_row3planes_compact_180_124_76_inset48_headerr_p3_mcinr_caps_sel7.{png,pdf}
bash scripts/render_sel7.sh          # both contrasts, PNG 200 dpi + vector PDF, panels rasterised at 600 dpi
```

served from `http://localhost:8765/t1flair_v3_af20/`, mirrored in `/home/congcongliu/figs/t1flair_v3_af20/`,
short names `af20_{T1,FLAIR}_sel7.png` in `/home/congcongliu/figs/t1flair_v3_af_hires/`.
**The twelve-column `_spirit_zf_caps` pair is untouched** -- the new `--keep` flag appends `_sel7`.

**The numbers did not move.** Each panel's PSNR / SSIM / NMSE is that arm against the ground truth
and does not depend on which other arms are on the figure; dropping four columns only changes which
panel the bold marks, since bold is the best of what is shown. Verified independently of the figure
code: MC-GS at transverse 180 is 36.25 dB on T1 (GS-single 36.03), the number the user quoted.
Within these seven arms MC-GS is first on all three metrics in all three planes on **both**
contrasts (`logs/sel7/fig_{T1,FLAIR}.log`) -- which it also is in the twelve-column figure on T1;
what the cut removes on FLAIR is no arm that was ahead of it in these planes.

**Resolution.** 4353 x 4416 px at 200 dpi = 614 dpi at a 180 mm page width (the twelve-column set
reaches 911 dpi only because that figure is 32.3 in wide). PDF text is CID TrueType, not Type 3.

**One layout fix came with it.** The caption under the title is one long line sized for nothing in
particular. At twelve columns it is already ~65 px wider than the figure and is clipped at *both*
ends (`head-masked` prints as `d-masked`, `has its own` as `has its o`) -- a real defect of the wide
figures, visible in `figures/..._p3_mcinr_spirit_zf_caps.png`. At eight columns it loses a third of
itself. With `--keep`, `scripts/fig_three_planes_compact.py` now measures the caption, wraps each of
its lines to the figure width, and sizes the header from the resulting line count and `--title_fs`
instead of the hard-coded 1.15 in (1.15 stays the floor, and the formula reproduces it for a
two-line caption at `--title_fs 14`, the original design point).

**Without `--keep` not one line of that runs**, so every existing figure of this topic and of the
eight other topics that got the same patch re-renders byte for byte -- checked, not assumed:
re-rendering `T1FLAIR_AF16_noise1x_v3_jiangyizhou`'s figure of record with its patched script
returns md5 `c0ec35a2f5a6b6734e301d6e77fb0703`, the md5 it had before. The wide figures therefore
keep their clipped caption until someone decides to fix it there too. The patch was applied by one
generator to all ten scripts (`--keep`, the ARMS filter, the caption block moved above the layout);
that generator reproduces this topic's hand-written patch exactly, and the figure it renders is md5
`4c8f5bb3b258a5c16e784002c3304a6f`, the T1 PNG delivered above.
