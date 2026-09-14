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
(not run yet: the study waits for the user's look at the mask. The table appears when `scripts/summarize.py` has the arms.)
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

2026-09-14 21:00–21:40: the mask, topic, configs and scripts are prepared; nothing has run on a GPU. The user asked to
see the mask before the full experiment starts.
