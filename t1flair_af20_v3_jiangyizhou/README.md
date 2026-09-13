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
| L1-ESPIRiT (λ 0.001) | 37.814 | 0.9279 | 34.371 | 0.9206 | 0.1518 | 0.845 | 35.209 | -0.84 |
| CS-SENSE (TV, converged) (λ 0.0007) | 37.808 | 0.9418 | 34.003 | 0.9191 | 0.1683 | 0.781 | 34.945 | -0.94 |
| joint CS-SENSE (JTV) (λ 0.001) | 37.736 | 0.9421 | 33.885 | 0.9183 | 0.1733 | 0.768 | 34.831 | -0.95 |
| CS-SENSE | 37.269 | 0.9276 | 33.418 | 0.9131 | 0.1864 | 0.725 | 34.409 | -0.99 |
| CG-SENSE | 34.030 | 0.7646 | 31.425 | 0.8665 | 0.2741 | 0.791 | 32.354 | -0.93 |
| ZF | 29.744 | 0.8165 | 25.639 | 0.6636 | 0.7206 | 0.263 | 25.841 | -0.20 |

**FLAIR** (sorted by head-only PSNR; AF16 column = the same arm in `T1FLAIR_AF16_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @AF16 | AF20 − AF16 |
|---|---|---|---|---|---|---|---|---|
| MC-GS | 37.892 | 0.9028 | **34.179** | **0.9142** | 0.1290 | 0.776 | 34.928 | -0.75 |
| GS-single | 37.801 | 0.8875 | 34.127 | 0.9127 | 0.1313 | 0.778 | 34.886 | -0.76 |
| L1-ESPIRiT (λ 0.003) | 37.402 | 0.9322 | 33.688 | 0.9076 | 0.1406 | 0.732 | 34.503 | -0.81 |
| joint CS-SENSE (JTV) (λ 0.0007) | 37.358 | 0.9364 | 33.544 | 0.9044 | 0.1522 | 0.726 | 34.355 | -0.81 |
| CS-SENSE (TV, converged) (λ 0.0005) | 37.241 | 0.9328 | 33.494 | 0.9049 | 0.1523 | 0.698 | 34.302 | -0.81 |
| CS-SENSE | 36.883 | 0.9107 | 33.027 | 0.8968 | 0.1571 | 0.616 | 33.806 | -0.78 |
| CG-SENSE | 34.868 | 0.7963 | 32.089 | 0.8717 | 0.1972 | 0.718 | 32.826 | -0.74 |
| ZF | 31.548 | 0.8329 | 27.636 | 0.6970 | 0.6874 | 0.251 | 27.879 | -0.24 |

**MC-GS joint minus MC-GS single**

| channel | whole PSNR | head PSNR | head SSIM | (AF16: whole / head) |
|---|---|---|---|---|
<!-- TABLES:END -->

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

## Status

Stage 1 (this session, 2026-09-13): the mask, the classical arms with their λ sweeps, and MC-GS
joint. Stage 2, if the user asks for it: ZS-SSDU, 3D-DIP and 3D-INR, which complete the nine-arm
table; they cost about eight more GPU-hours and change nothing about the MC-GS margin over the
strongest classical arm.
