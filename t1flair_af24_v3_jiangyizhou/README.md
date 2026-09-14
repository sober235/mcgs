# Retrospective AF 24 on the 3D T1 + FLAIR pair — jiangyizhou

Seventh topic in the series. **Read `../T1FLAIR_AF20_v3_jiangyizhou/README.md` first** (the same subject one
acceleration down) and, through it, the AF 16 / AF 12 topics: the data provenance, the geometry adjudication, the
ESPIRiT recipe, the known-zero region, the arms and the scoring gauge are documented there and not repeated.

## What this topic adds

**Exactly one thing differs from `T1FLAIR_AF20_v3_jiangyizhou`: the mask.** Same subject, same CFL k-space, same two
ESPIRiT maps, same ground truth, same gauge, same arms, same MC-GS recipe. The k-space and the maps are symlinks into
`../T1FLAIR_AF12_v3_jiangyizhou/data/`; `scripts/prep_04_export.py` asserts the symlink so it can never write through
them. Nothing is reused from the AF 20 results: every arm depends on the mask.

Created with `scripts/setup_af24_jyz.sh`; configs `configs/t1flair_v3_af24_jyz{,_singleT1,_singleFLAIR}.yaml` in the
worktree `/data0/congcong/code/GS/mcgs_af16_wt` on branch `autoresearch-jyz-af24`.

**Acceleration** follows the user's definition of 2026-09-14: AF = PE-plane points / sampled points that carry that
contrast's data. T1 acquires the whole (ky, kz) plane; FLAIR's elliptical vendor shutter leaves the corners empty
(34139 of 45056 lines), so one shared mask gives FLAIR the higher AF. **This topic puts T1 at exactly 24x** (1877
points; 45056 / 1877 = 24.004), so neither contrast is below 24x.

| mask | generator call | PE points | T1 AF | FLAIR AF | FLAIR points outside the shutter | 24x24 centre | (older convention) FLAIR vs its shutter |
|---|---|---|---|---|---|---|---|
| AF 12 topic | `vdPoisMex(256,176,256,176, 3, 4, 24, 0, 1)` | 3847 | 11.71 | 12.86 | 343 | 482 | 9.74 |
| AF 16 topic | `vdPoisMex(256,176,256,176, 4, 4, 24, 0, 1)` | 2966 | 15.19 | 16.49 | 233 | 472 | 12.49 |
| AF 20 topic | `vdPoisMex(256,176,256,176, 5, 4, 24, 0, 1)` | 2427 | 18.56 | 19.96 | 170 | 474 | 15.13 |
| **AF 24 (this topic)** | `vdPoisMex(256,176,256,176, 6.8656, 4, 24, 0, 1)` | **1877** | **24.00** | **25.51** | **111** | **463** | **19.33** |

Same house generator and settings; only `accelx` differs. A draw's point count is random (the four AF 20 draws at
accelx 5 have 2400–2427 points), so `scripts/prep_02_mask_pool_af24.m` drew a pool of 134 masks around the accelx
that gives 1877 points on average (stage A: accelx 6.4–7.0, fit n = c + a/accelx; stage B: 120 draws around
accelx* = 6.8456), and `scripts/prep_03_pick_mask_af24.py` took, by a rule fixed before the pool was drawn, the
lowest-index draw with exactly 1877 points: **draw_025 (accelx 6.8656), untrimmed**. Two draws had exactly 1877
points; the other one gives FLAIR 25.37x. The mask is `data/mask_vdpois_256x176_af24.mat` (md5
c3688d6182ba20df7404519e560dd9b4; `vdPoisMex` seeds from `time(0)`, it cannot be regenerated, masks are not in git),
broadcast along the readout into `data/mask_af24_3d.h5`. Measured in `logs/prep_03_pick_mask_af24.json` and
`logs/prep_04_export.json`, and re-read through both configs' `mask_path` by `scripts/fig_mask_reinspect.py`.

Figures: `figures/t1flair_v3_af24_mask_T1_vs_FLAIR.png` (the mask on each contrast's acquired region),
`figures/t1flair_v3_af24_mask_series_af12_16_20_24.png` (the four topics' masks side by side),
`figures/t1flair_v3_af24_mask_T1_FLAIR_reinspect.png` (the mask as each channel loads it).

## Method under test

The **AF 16 phase-3 recipe, unchanged**, as at AF 20: P6's repaired cold start (init_mode head, 3-NN scales,
edge-TV threshold from ZF), 3.6M points, edge-TV λ_geo 0.10 / κ_q 0.85 / w_floor 0.10 / ε 0.04, kspace_high 2.2,
warmup 500, wavelet db4 on its finest level only (T1 0.12 / FLAIR 0.10), common-edge cross-gradient prior λ 0.001,
phase-only TV λ 0.03, 4000 iterations. The argument list lives in one place, `mcgs_af16_wt/run_af24_mcgs.sh`
(recipe block byte-identical to `run_af20_mcgs.sh`).

The classical arms are re-swept on this subject against the ground truth, grids = the AF 20 topic's final grids plus
one point below each. ZS-SSDU / 3D-DIP / 3D-INR use the AF 16 topic's arguments. Every job and grid is defined once, in
`scripts/run_all_af24.py`.

## Results

<!-- TABLES:BEGIN -->
**T1** (sorted by head-only PSNR; AF20 column = the same arm in `T1FLAIR_AF20_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @AF20 | AF24 − AF20 |
|---|---|---|---|---|---|---|---|---|
| MC-GS | 37.499 | 0.9046 | **33.767** | **0.9131** | 0.1692 | 0.803 | 34.832 | -1.06 |
| GS-single | 37.331 | 0.9100 | 33.588 | 0.9090 | 0.1772 | 0.806 | 34.667 | -1.08 |
| L1-ESPIRiT (λ 0.001) | 37.118 | 0.9331 | 33.401 | 0.9069 | 0.1869 | 0.787 | 34.371 | -0.97 |
| CS-SENSE (TV, converged) (λ 0.0007) | 36.711 | 0.9291 | 32.820 | 0.8991 | 0.2158 | 0.730 | 34.003 | -1.18 |
| joint CS-SENSE (JTV) (λ 0.001) | 36.656 | 0.9291 | 32.730 | 0.8986 | 0.2186 | 0.715 | 33.885 | -1.15 |
| ZS-SSDU | 36.519 | 0.9386 | 32.578 | 0.8989 | 0.2398 | 0.948 | 32.892 | -0.31 |
| CS-SENSE | 36.154 | 0.9140 | 32.231 | 0.8913 | 0.2349 | 0.671 | 33.418 | -1.19 |
| 3D-DIP | 34.442 | 0.8476 | 30.500 | 0.8378 | 0.3450 | 0.612 | 30.996 | -0.50 |
| CG-SENSE | 33.329 | 0.7669 | 30.340 | 0.8434 | 0.3361 | 0.707 | 31.425 | -1.08 |
| 3D-INR | 30.364 | 0.5637 | 28.444 | 0.7108 | 0.3473 | 1.339 | 29.456 | -1.01 |
| ZF | 29.436 | 0.8049 | 25.333 | 0.6390 | 0.7578 | 0.246 | 25.639 | -0.31 |

**FLAIR** (sorted by head-only PSNR; AF20 column = the same arm in `T1FLAIR_AF20_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @AF20 | AF24 − AF20 |
|---|---|---|---|---|---|---|---|---|
| MC-GS | 36.962 | 0.8925 | **33.182** | **0.8966** | 0.1579 | 0.738 | 34.179 | -1.00 |
| GS-single | 36.861 | 0.8762 | 33.110 | 0.8943 | 0.1620 | 0.741 | 34.127 | -1.02 |
| L1-ESPIRiT (λ 0.002) | 36.434 | 0.9121 | 32.782 | 0.8893 | 0.1724 | 0.704 | 33.688 | -0.91 |
| joint CS-SENSE (JTV) (λ 0.0007) | 36.496 | 0.9262 | 32.605 | 0.8849 | 0.1903 | 0.668 | 33.544 | -0.94 |
| CS-SENSE (TV, converged) (λ 0.0005) | 36.421 | 0.9237 | 32.573 | 0.8858 | 0.1886 | 0.639 | 33.494 | -0.92 |
| CS-SENSE | 36.061 | 0.8941 | 32.165 | 0.8762 | 0.1937 | 0.561 | 33.027 | -0.86 |
| ZS-SSDU | 36.145 | 0.9315 | 32.107 | 0.8849 | 0.2256 | 0.989 | 32.660 | -0.55 |
| CG-SENSE | 34.405 | 0.8071 | 31.259 | 0.8510 | 0.2551 | 0.644 | 32.089 | -0.83 |
| 3D-DIP | 34.298 | 0.8094 | 30.360 | 0.8144 | 0.3023 | 0.485 | 30.637 | -0.28 |
| 3D-INR | 31.767 | 0.6444 | 29.146 | 0.7352 | 0.3034 | 1.071 | 29.809 | -0.66 |
| ZF | 31.208 | 0.8212 | 27.301 | 0.6748 | 0.7271 | 0.237 | 27.636 | -0.34 |

**MC-GS joint minus MC-GS single**

| channel | whole PSNR | head PSNR | head SSIM | (AF20: whole / head) |
|---|---|---|---|---|
<!-- TABLES:END -->

## Regenerating

```
# mask (MATLAB R2021a + the house vdPoisMex); the pool and the pick, never re-run into this data/
matlab -batch "run('scripts/prep_02_mask_pool_af24.m')"
PYTHONNOUSERSITE=1 PYTHONPATH= python scripts/prep_03_pick_mask_af24.py
PYTHONNOUSERSITE=1 PYTHONPATH= python scripts/prep_04_export.py      # 3-D mask + the AF read-out

# everything else: 38 jobs queued by priority onto free cards (one GPU job of this user per card, at most 4 at once;
# 12 L1-ESPIRiT runs in 4 CPU slots), then finalize_af24_jyz.sh; --dry_run shows the queue and the cards
setsid nohup python -u scripts/run_all_af24.py > /dev/null 2>&1 < /dev/null &
```

## Status

Complete. 2026-09-14 21:00–21:40 the mask, topic, configs and scripts were prepared; the user looked at the mask figures
and approved them before any GPU job; a 60-iteration smoke run passed; `scripts/run_all_af24.py` started 21:45:49 and
the queue (38 jobs, no failure) and `scripts/finalize_af24_jyz.sh` were done at 05:30 on 2026-09-15.

Headline (head-only PSNR, T1 / FLAIR): MC-GS **33.767 / 33.182**, margin over L1-ESPIRiT **+0.366 / +0.400** (AF 20:
+0.461 / +0.491), first on head PSNR, SSIM and HFEN on both contrasts; MC-GS − GS-single +0.179 / +0.071; T1 sharpness
0.803 against L1-ESPIRiT's 0.787. Every classical λ optimum is interior. Figures in the AF 16 figure-of-record layout:
`figures/t1flair_v3_af24_{T1,FLAIR}_row3planes_compact_180_124_76_inset48_headerr_p3.{png,pdf}` and
`..._174_110_47_headerr_p3.png`. Session record: `mcgs_af16_wt/docs/superpowers/records/2026-09-15-af24-jyz.md`.
