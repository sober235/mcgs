# Retrospective nominal AF 16 on the 3D T1 + FLAIR pair, jiangyizhou -- uniform-density mask (diff-mask study, row 2 redone 2026-11-11 / 12)

**Read `../T1FLAIR_AF16_v3_jiangyizhou/README.md` first**: same subject, same CFL k-space, same ESPIRiT maps, same
ground truth, same gauge, same arms and the same frozen AF16 phase-3 MC-GS recipe are documented there. The September
diff-mask study is documented in `../T1FLAIR_AF16_gauss_v3_jiangyizhou/README.md`; this topic repeats its procedure.

## What this topic is

**Exactly one thing differs from `T1FLAIR_AF16_v3_jiangyizhou`: the mask PATTERN.** On 2026-11-11 the user said the
three masks of the September study were not the intended ones and approved three new ones (GitHub branch
`diffmask-af16-masks-20261111`, folder `autoresearch/diffmask_af16_masks_20261111`). This topic is the new **row 2**:
a mask whose centre block is fully sampled and whose other points have a uniform density. User request of 2026-11-11
22:05 PST: reconstruct on it, "the core objective is for MCGS to outperform all the comparison methods", and afterwards
"the old uniform density experimental results should be replaced". The old result is the September row 2
(`../T1FLAIR_AF16_gauss_v3_jiangyizhou`, Gaussian variable density; confirmed by the user on 2026-11-12 20:48 PST):
this topic replaces it in the study's figures and tables.
**The September topics are not changed and not deleted**; every study-level file made here lives in this topic and in
`~/figs/t1flair_v3_af16_diffmask_redo/`.

The k-space and the ESPIRiT maps are symlinks into `../T1FLAIR_AF12_v3_jiangyizhou/data/` (they do not depend on the
mask); every arm was reconstructed from scratch.

### The mask

Michael Lustig's ESPIRiT-toolbox generator `vdPoisMex` with the polynomial order of the variable density set to zero
("0 = uniform" in the toolbox's help): `vdPoisMex(256, 176, 256, 176, 4.1979, 4, 24, 0, 0)`. Everything else is the
house call. The points keep the generator's minimum spacing: a uniform-density Poisson-disc pattern, not an i.i.d.
random one. `vdPoisMex` seeds from `time(0)`, so **this mask cannot be regenerated**; the file is the record:
`data/mask_uniform_256x176_af16.mat`, md5 `d0796ac1a198ae2aba8983146b3830fc`, a byte copy of the approved
`~/codex_artifacts/mcgs_diffmask_af16_redo_20261111/mask_vdpoismex_pp0_256x176_af16_preview.mat`, broadcast along the
readout into `data/mask_af16_3d.h5`. Read-out: `logs/register_mask.json`.

| mask | PE points | T1 AF | FLAIR AF | FLAIR points outside the shutter | 24x24 centre | fully sampled block | median rn |
|---|---|---|---|---|---|---|---|
| vd-Poisson disc (house, AF16 topic) | 2966 | 15.19 | 16.49 | 233 | 472/576 | 24 x 16 | 0.400 |
| **uniform density (this topic)** | **2964** | **15.20** | **19.48** | **651** | **396/576** | **24 x 16** | **0.742** |

AF = PE-plane points / sampled points that carry that contrast's data (user definition, 2026-09-14). **FLAIR runs at
19.48x here**: a uniform density spends 651 points in the corners FLAIR's elliptical shutter never acquired. Reported,
not compensated, as in September.

## Method under test

The **AF16 phase-3 recipe, unchanged**: `/data0/congcong/code/GS/mcgs_diffmask_wt/run_diffmask_mcgs.sh`'s argument
list (init_mode head, 3-NN scales, edge-TV threshold from ZF, 3.6M points, edge-TV lambda_geo 0.10 / kappa_q 0.85 /
w_floor 0.10 / eps 0.04, kspace_high 2.2, warmup 500, wavelet db4 finest level T1 0.12 / FLAIR 0.10, common-edge
prior 0.001, phase-only TV 0.03, 4000 iterations). The driver asserts that its argument list equals the argv recorded
by the September Gaussian-mask runs, argument for argument, and that the trainer worktree (HEAD 246e833) has no
uncommitted change. **Nothing was tuned for MC-GS or GS-single on this mask.** Runs:
`mcgs_diffmask_wt/results_arc/af16u_p3_{joint,singleT1,singleFLAIR}_full4k`, registered here as `gs_joint_p3` /
`gs_single_p3`.

The comparison arms are the September ones with the September arguments (ZF, CG-SENSE, CS-SENSE, converged TV sweep,
joint-TV sweep, L1-SPIRiT sweep, ZS-SSDU, 3D-DIP, 3D-INR, MC-INR). L1-ESPIRiT stays retired.

**The lambda grids had to be extended upward.** On the September grids the converged-TV and joint-TV sweeps were still
rising at the top on this mask (T1 TV at 0.0014, FLAIR TV at 0.001, JTV at 0.0014), so those arms would have been
scored below their optimum. Points were added until every winner sat inside its grid:

| arm | grid on this mask | winner (head PSNR) |
|---|---|---|
| CS-SENSE, TV converged, T1 | 0.00035 ... 0.0014 (September) + 0.002, 0.0028, 0.004, 0.0056 | 0.0014 |
| CS-SENSE, TV converged, FLAIR | 0.00025 ... 0.001 (September) + 0.0014, 0.002, 0.0028, 0.004, 0.0056, 0.008 | 0.0014 |
| joint CS-SENSE, JTV | 0.00035 ... 0.0014 (September) + 0.002, 0.0028, 0.004, 0.0056 | 0.002 |
| L1-SPIRiT | kernels 3, 5 x lambda 0, 0.0005, 0.001, 0.002, 0.004, 0.008 (September) | kernel 5, 0.001 |

## Results

<!-- TABLES:BEGIN -->
<!-- TABLES:END -->

**MC-GS is first on head PSNR, head SSIM and head HFEN on both contrasts.** Margins over the strongest other arm
(GS-single counts as a comparison arm, user 2026-09-13 / 16):

| contrast | head PSNR | runner-up | head SSIM | runner-up | head HFEN (lower is better) | runner-up |
|---|---|---|---|---|---|---|
| T1 | 34.903, **+0.507 dB** | GS-single 34.396 | 0.9151, +0.0109 | GS-single 0.9042 | 0.2178, -0.0264 | GS-single 0.2443 |
| FLAIR | 34.984, **+0.142 dB** | GS-single 34.842 | 0.9166, +0.0005 | ZS-SSDU 0.9161 | 0.2079, -0.0117 | GS-single 0.2196 |

- **The thinnest lead is FLAIR head SSIM over ZS-SSDU, +0.0005.** That is a tie in practice.
- **Whole-volume SSIM (background included) is higher for ZS-SSDU**: T1 0.9278 against 0.9034, FLAIR 0.9477 against
  0.9077. The study's gauge is the head-masked one, as in September; the whole-volume columns are in the tables above.
- Over the best arm that is not a Gaussian model: +2.17 dB on T1 (L1-SPIRiT), +0.76 dB on FLAIR (ZS-SSDU).

## The study with the new row 2

`scripts/compare_masks.py` (ported: row 2 = this topic), full tables in `logs/compare_masks_head_{psnr,ssim,hfen}.md`.
**Row 3 is the September pseudo-radial topic (with its ACS block) and stays so** (user, 2026-11-12 20:48 PST: "the
radial direction still uses the old result"); the conventional-radial masks drawn on 2026-11-11 are not used.

| contrast | pattern | MC-GS head PSNR | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | vd-Poisson (house) | 35.706 | GS-single 35.558 | +0.148 |
| T1 | **uniform density** | **34.903** | GS-single 34.396 | **+0.507** |
| T1 | pseudo-radial (September) | 31.113 | GS-single 30.913 | +0.200 |
| FLAIR | vd-Poisson (house) | 34.928 | GS-single 34.886 | +0.042 |
| FLAIR | **uniform density** | **34.984** | GS-single 34.842 | **+0.142** |
| FLAIR | pseudo-radial (September) | 31.729 | ZS-SSDU 31.635 | +0.093 |

For reference, the row this one replaces (September, Gaussian variable density): T1 34.909, +0.138 over GS-single;
FLAIR 34.317, +0.023.

- **The uniform density is much harder for the regularised SENSE arms than for the Gaussian models.** Against the
  house mask, converged TV loses 4.02 dB on T1 and 2.23 dB on FLAIR, joint TV 4.08 / 2.28, CG-SENSE 3.79 / 2.04;
  MC-GS loses 0.80 dB on T1 and gains 0.06 dB on FLAIR (at 19.48x against 16.49x), GS-single loses 1.16 / 0.04.
  L1-SPIRiT (-1.06 / +0.10) and ZS-SSDU (-1.88 / +0.22) sit between.
- **The coupling gain (MC-GS minus GS-single) is the largest of the study on this mask**: +0.507 dB on T1 against
  +0.148 on the house mask and +0.200 on the pseudo-radial one. One subject, one draw of the mask: an observation.
- MC-GS's head SSIM and head HFEN on T1 are worse than on the September Gaussian mask (0.9151 / 0.2178 against
  0.9333 / 0.1344) at the same head PSNR.

## Figures

Twelve columns: ground truth, zero-filling, CG-SENSE, CS-SENSE, Joint SENSE, L1-SPIRiT, ZS-SSDU, 3D-DIP, 3D-INR,
MC-INR, GS-single, MC-GS. Orange in-panel numbers are head-masked slice PSNR / SSIM (7x7 window) / NMSE, the best of
each plane in bold; error maps are `|recon - GT| / max(GT)`, jet, fixed [0, 0.1], inside the head only.

| file (in `figures/`) | what |
|---|---|
| `t1flair_v3_af16u_{T1,FLAIR}_row3planes_compact_180_124_76_inset48_headerr_p3_spirit.png` (+ `.pdf`) | this mask alone, the AF16 figure-of-record slices (transverse 180 / coronal 124 / sagittal 76) with the 48-voxel zoom; served in `~/figs/t1flair_v3_af16_diffmask_uniform/` |
| `t1flair_v3_af16u_{T1,FLAIR}_row3planes_compact_174_110_47_headerr_p3_spirit.png` | this mask alone, transverse 174 / coronal 110 / sagittal 47 |
| `t1flair_v3_af16_masks_stack_T1_tra126_inset32_headerr_p3_caps_nongsmax_sel7.png` (+ `.pdf`) | **the three-pattern figure with the new row 2**, seven-method cut of 2026-09-18 (CS-SENSE, Joint SENSE, ZS-SSDU, 3D-INR, MC-INR, GS-single, MC-GS), T1, transverse 126, 32x32 box at (146, 29); served in `~/figs/t1flair_v3_af16_diffmask_redo/` |
| `t1flair_v3_af16_masks_stack_FLAIR_tra205_inset32_headerr_p3_caps_nongsmax_sel7.png` (+ `.pdf`) | the same for FLAIR, transverse 205, 32x32 box at (148, 146) |
| `t1flair_v3_af16_masks_stack_{T1_tra126,FLAIR_tra205}_inset32_headerr_p3_zf_caps_nongsmax.png` (+ `.pdf`) | the same two figures with all twelve columns |
| `t1flair_v3_af16_masks_stack_{T1,FLAIR}_tra168_zoomrows64_headerr_p3_zf_caps_butterfly.png` | **the butterfly slice** (user request, 2026-11-12): transverse 168, the whole slice, all twelve columns (GT, ZF, the nine comparison methods, MC-GS), three patterns, with the 64 x 64 box on the lateral ventricles at (128, 88) magnified under every image row and error row |

`finalize_diffmask.sh`'s own figure step stops because it asks for the retired L1-ESPIRiT column (it did in September
too); the three-plane figures were drawn with `--l1_arm spirit --with_mcinr --with_zf`, the September layout.

**The slices and boxes of the three-pattern figures were NOT re-selected.** They are the September picks (chosen then
with the Gaussian row). With the new row 2 they were only checked (`scripts/eval_zoom_box.py`,
`scripts/find_slice_nongs_gap.py`, logs in `logs/study_*.log`):

- T1, transverse 126, box (146, 29): MC-GS first on box PSNR and box SSIM in all three rows; smallest box lead
  +0.225 dB (vd-Poisson, over GS-single); on the uniform row +0.550 dB over GS-single and +1.512 dB over the best
  arm that is not a Gaussian model.
- FLAIR, transverse 205, box (148, 146): first in all three rows; smallest box lead +0.142 dB (vd-Poisson); on the
  uniform row +0.173 dB over GS-single and +1.411 dB over the best non-Gaussian arm.
- Each figure's box is chosen for its own contrast, as in September. In the other contrast MC-GS is not first
  everywhere: FLAIR in the T1 figure's box on the uniform row is 0.42 dB behind L1-SPIRiT, T1 in the FLAIR figure's
  box on the pseudo-radial row is 0.11 dB behind ZS-SSDU.
- Whole-slice gauge: MC-GS is first over every column in all six rows (three patterns x two contrasts) on 44 of the
  178 eligible transverse slices; slices 126 and 205 are among them. **Quote the whole-volume numbers, not the
  slice or box numbers**: slices and boxes were selected on these data.
- **Butterfly slice (transverse 168, box 64 at (128, 88)).** The slice is the house butterfly slice of this subject
  (`fig_butterfly_final.py`, `fig_row_final.py`); the box is the bounding box of the lateral ventricles read from the
  ground truth (`butterfly_probe.py` in the GitHub folder), not chosen by any score. Inside the box MC-GS is first on
  PSNR and SSIM in all six rows (leads T1 +0.306 / +0.829 / +0.403 dB, FLAIR +0.079 / +0.237 / +0.298 dB for
  vd-Poisson / uniform density / pseudo-radial). On the whole slice it is first on PSNR, SSIM and NMSE on the
  vd-Poisson and uniform-density rows of both contrasts; **on the pseudo-radial row ZS-SSDU has the better slice PSNR
  and NMSE** (T1 32.17 against 31.96 dB, FLAIR 32.12 against 32.07 dB) and MC-GS the better SSIM. The figure shows
  that as it is (bold = best of the row).
- The seven-method cut drops L1-SPIRiT, which is the strongest non-Gaussian arm on T1 on this mask (32.73 dB); it
  is in the twelve-column figures and in every table.

## Known limitations

- One subject, one draw of the mask, one seed per run.
- MC-GS and GS-single run one frozen recipe on every mask; the classical arms' lambdas are re-selected per mask by
  head PSNR against the ground truth (the September rule), which favours the classical arms.
- FLAIR's acceleration is 19.48x here against 16.49x on the house mask, so the FLAIR column mixes pattern and
  acceleration.
- Row 3 is the September pseudo-radial result by the user's decision, so the study mixes one row reconstructed in
  November with two reconstructed in September (same data, gauge, scorer and frozen recipe; the classical lambda
  grids of the uniform row reach higher than the September ones because its optimum lies higher).
- Everything the AF16 topic's README lists as a limitation applies unchanged: the recipe was selected on this
  subject, and the numbers are on the magnitude gauge.

## Regenerating

Everything here was produced by `autoresearch/diffmask_af16_masks_20261111/run_uniform.py` (GitHub `sober235/mcgs`,
branch `diffmask-af16-masks-20261111`) through the workspace's `gpu.py`:

```
gpu.py run --cpu   -- @python autoresearch/diffmask_af16_masks_20261111/run_uniform.py --setup
gpu.py run --gpu N -- @python autoresearch/diffmask_af16_masks_20261111/run_uniform.py --worker \
    --add tv:T1:0.002 --add tv:T1:0.0028 --add tv:T1:0.004 --add tv:T1:0.0056 \
    --add tv:FLAIR:0.0014 --add tv:FLAIR:0.002 --add tv:FLAIR:0.0028 --add tv:FLAIR:0.004 --add tv:FLAIR:0.0056 --add tv:FLAIR:0.008 \
    --add jtv:0.002 --add jtv:0.0028 --add jtv:0.004 --add jtv:0.0056
gpu.py run --cpu   -- @python autoresearch/diffmask_af16_masks_20261111/run_uniform.py --finalize
gpu.py run --cpu   -- @python autoresearch/diffmask_af16_masks_20261111/run_uniform.py --study "<script.py args>" ...
```

`--setup` refuses to touch an existing topic (it only verifies it), and the mask itself cannot be redrawn. The trainer
start runs ninja on the CUDA voxelizer of the September worktree; the driver puts `/usr/local/cuda-12.6/bin` first on
`PATH` and proves on a private copy of the build directory that nothing would be compiled before it starts a trainer.

## Status

2026-11-12: complete. Twelve arms x two contrasts reconstructed and scored (65 jobs on one GPU, 2026-11-11 22:49 to
2026-11-12 11:58 PST, no failure; queue log `logs/run_uniform.log`), registered, summarised and figured; the
study-level table and three-pattern figures are redrawn with this topic as row 2. 2026-11-12 20:48 PST, user: the
replaced row is the September Gaussian one, and row 3 keeps the September pseudo-radial result. Nothing is open.
