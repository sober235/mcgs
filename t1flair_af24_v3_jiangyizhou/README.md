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

**Acceleration label (user, 2026-09-16): AF = PE-plane points / sampled points = 45056 / 1877 = 24.00x, the same number on
both contrasts**, because both read the same mask. The mask was drawn to put that number at exactly 24x. What differs
between the contrasts is how much of it carries data: T1 acquired the whole (ky, kz) plane, FLAIR only 34139 of the 45056
lines (elliptical scanning plus ky rows 0-19, verified on the raw CFL — `figures/t1flair_v3_af24_raw_kspace_support_T1_vs_FLAIR.png`),
so 1766 of the 1877 sampled points carry FLAIR data. Two other gauges are reported where they matter and are NOT the label:
45056 / points-with-data (the 2026-09-14 per-contrast gauge: T1 24.00x, FLAIR 25.51x) and 34139 / 1766 = 19.33x, FLAIR against
its own acquisition. Sections written before 2026-09-16 quote the per-contrast gauge.

| mask | generator call | PE points | AF label (both contrasts) | FLAIR AF, 09-14 gauge | FLAIR points outside its scan | 24x24 centre | FLAIR vs its own acquisition |
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
| CS-SENSE (TV, converged) (λ 0.0007) | 36.711 | 0.9291 | 32.820 | 0.8991 | 0.2158 | 0.730 | 34.003 | -1.18 |
| joint CS-SENSE (JTV) (λ 0.001) | 36.656 | 0.9291 | 32.730 | 0.8986 | 0.2186 | 0.715 | 33.885 | -1.15 |
| ZS-SSDU | 36.519 | 0.9386 | 32.578 | 0.8989 | 0.2398 | 0.948 | 32.892 | -0.31 |
| CS-SENSE | 36.154 | 0.9140 | 32.231 | 0.8913 | 0.2349 | 0.671 | 33.418 | -1.19 |
| L1-SPIRiT (λ 0.001) | 35.575 | 0.9026 | 31.783 | 0.8717 | 0.2246 | 0.864 | 32.882 | -1.10 |
| 3D-DIP | 34.442 | 0.8476 | 30.500 | 0.8378 | 0.3450 | 0.612 | 30.996 | -0.50 |
| CG-SENSE | 33.329 | 0.7669 | 30.340 | 0.8434 | 0.3361 | 0.707 | 31.425 | -1.08 |
| MC-INR (Niessen 2025) | 30.448 | 0.5712 | 30.076 | 0.8103 | 0.3593 | 0.803 | 30.969 | -0.89 |
| 3D-INR | 30.364 | 0.5637 | 28.444 | 0.7108 | 0.3473 | 1.339 | 29.456 | -1.01 |
| ZF | 29.436 | 0.8049 | 25.333 | 0.6390 | 0.7578 | 0.246 | 25.639 | -0.31 |

**FLAIR** (sorted by head-only PSNR; AF20 column = the same arm in `T1FLAIR_AF20_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @AF20 | AF24 − AF20 |
|---|---|---|---|---|---|---|---|---|
| MC-GS | 36.962 | 0.8925 | **33.182** | **0.8966** | 0.1579 | 0.738 | 34.179 | -1.00 |
| GS-single | 36.861 | 0.8762 | 33.110 | 0.8943 | 0.1620 | 0.741 | 34.127 | -1.02 |
| joint CS-SENSE (JTV) (λ 0.0007) | 36.496 | 0.9262 | 32.605 | 0.8849 | 0.1903 | 0.668 | 33.544 | -0.94 |
| CS-SENSE (TV, converged) (λ 0.0005) | 36.421 | 0.9237 | 32.573 | 0.8858 | 0.1886 | 0.639 | 33.494 | -0.92 |
| CS-SENSE | 36.061 | 0.8941 | 32.165 | 0.8762 | 0.1937 | 0.561 | 33.027 | -0.86 |
| ZS-SSDU | 36.145 | 0.9315 | 32.107 | 0.8849 | 0.2256 | 0.989 | 32.660 | -0.55 |
| L1-SPIRiT (λ 0.001) | 35.653 | 0.8975 | 31.803 | 0.8635 | 0.1804 | 0.802 | 32.824 | -1.02 |
| CG-SENSE | 34.405 | 0.8071 | 31.259 | 0.8510 | 0.2551 | 0.644 | 32.089 | -0.83 |
| MC-INR (Niessen 2025) | 32.442 | 0.6644 | 30.947 | 0.8303 | 0.2688 | 0.721 | 31.666 | -0.72 |
| 3D-DIP | 34.298 | 0.8094 | 30.360 | 0.8144 | 0.3023 | 0.485 | 30.637 | -0.28 |
| 3D-INR | 31.767 | 0.6444 | 29.146 | 0.7352 | 0.3034 | 1.071 | 29.809 | -0.66 |
| ZF | 31.208 | 0.8212 | 27.301 | 0.6748 | 0.7271 | 0.237 | 27.636 | -0.34 |

**MC-GS joint minus MC-GS single**

| channel | whole PSNR | head PSNR | head SSIM | (AF20: whole / head) |
|---|---|---|---|---|
<!-- TABLES:END -->

### L1-ESPIRiT dropped from the comparison (user, 2026-09-15)

As in the AF 12 / 16 / 20 topics: L1-ESPIRiT is no longer carried as a comparison arm and is gone from the tables above
(`DROP = {"l1espirit"}` in `scripts/readme_tables.py`) and from the figure set of record. Nothing was deleted: its sweep
logs, volumes and rows in `logs/summary.json` stay. GS-single counts as a comparison arm (user, 2026-09-13), so the
margin to quote is over GS-single, and dropping L1-ESPIRiT does not change it:

| head PSNR, AF 24 | T1 | FLAIR |
|---|---|---|
| strongest comparison arm (GS-single counts) | GS-single 33.588 | GS-single 33.110 |
| **MC-GS margin over it** (unchanged by the drop) | **+0.179** | **+0.071** |
| strongest non-GS arm, was | L1-ESPIRiT 33.401 | L1-ESPIRiT 32.782 |
| strongest non-GS arm, now | CS-SENSE (TV, converged) 32.820 | joint CS-SENSE (JTV) 32.605 |
| MC-GS over the strongest non-GS arm, was / now | +0.366 / +0.947 | +0.400 / +0.577 |

The "was" column is only the arm that is no longer listed; sections of this file and of the session record written
before the drop quote margins against L1-ESPIRiT and are history (`scripts/compare_topics.py` prints both framings).

## L1-SPIRiT and MC-INR registered (2026-09-15)

Both arms were added to the AF 12 / 16 / 20 topics on 2026-09-14/15; they are run here the same way so that AF 24 has the
same comparison set (user request 2026-09-15). The method, the GPU port's fidelity against Lustig's MATLAB toolbox and
the solver change (FISTA, because `pocsSPIRiT` diverges on this data) are documented once in
`../T1FLAIR_AF16_v3_jiangyizhou/README.md`; `scripts/recon_spirit.py`, `run_spirit_sweep.sh` and `spirit_align*` are the AF 20
topic's files with the topic path changed. The ACS block of this mask is 24 x 16 at (116, 80), as at AF 20.

**L1-SPIRiT sweep** (`logs/spirit_<c>_ks<K>_lam<L>.json`, `logs/spirit/`; the AF 20 grid, 150 iterations; head PSNR):

| | lam 0 | 5e-4 | 1e-3 | 2e-3 | 4e-3 | 8e-3 |
|---|---|---|---|---|---|---|
| T1 ksize 3 | 29.514 | **31.779** | 31.724 | 31.352 | 30.824 | 30.251 |
| T1 ksize 5 | 28.852 | 31.738 | **31.783** | 31.478 | 30.981 | 30.412 |
| FLAIR ksize 3 | 29.345 | 31.687 | **31.708** | 31.498 | 31.153 | 30.746 |
| FLAIR ksize 5 | 28.790 | 31.687 | **31.803** | 31.643 | 31.299 | 30.868 |

Kept row, both contrasts: ksize 5 / lambda 1e-3, the optimum of the AF 16 and AF 20 topics too; every optimum is interior.

| head-only, AF 24 | T1 PSNR / SSIM / HFEN | FLAIR PSNR / SSIM / HFEN | AF 20 → AF 24 (head PSNR) |
|---|---|---|---|
| L1-SPIRiT (ksize 5, lambda 1e-3) | 31.783 / 0.8717 / 0.2246 | 31.803 / 0.8635 / 0.1804 | −1.099 / −1.021 |
| MC-INR (Niessen 2025) | 30.076 / 0.8103 / 0.3593 | 30.947 / 0.8303 / 0.2688 | −0.893 / −0.719 |

- L1-SPIRiT is sixth of the eleven comparison rows (ZF included) on both contrasts, below every CS-SENSE variant and
  ZS-SSDU, above CG-SENSE, MC-INR, 3D-DIP, 3D-INR and ZF. On FLAIR its head HFEN 0.1804 is the lowest
  of the non-GS comparison arms. Its fall-off from AF 20 (1.10 / 1.02 dB) is in the range of the classical arms.
- MC-INR ran in `mcinr_niessen2025` (commits 2f0b07f + 5ff090a: AF 24 added as a source topic; `run_volume.py --af 24`,
  256 slices, loop 707 s) on this topic's own mask, k-space and ESPIRiT maps; `score.py --af 24` re-checked the gauge on
  CG-SENSE first. `scripts/score_mcinr_arm.py` re-scores its volumes through this topic's `Pair.score`: worst relative
  disagreement with the reproduction's own scores 1.2e-7, so the row is comparable; `recon/mcinr_<c>_complex.npy` link to
  its `recon/af24/`. Ninth (T1) and eighth (FLAIR) of the eleven comparison rows.
- Neither arm changes the ranking at the top: MC-GS first on all six head cells, GS-single second.

## Mask-draw variance (2026-09-15)

Three more draws b / c / d with the registered mask's rule (a second pool, `scripts/prep_02b_mask_pool_draws_af24.m`, 67
draws; the three lowest-index draws with exactly 1877 points, untrimmed; `scripts/prep_03b_pick_export_draws_af24.py`):
FLAIR 25.43 / 25.41 / 25.50x, 540–572 points shared with draw a. As at AF 20, per draw only MC-GS joint (frozen recipe,
`MASK_DRAW=<d>`, no volumes) and L1-ESPIRiT at the optimum and both neighbours (`scripts/run_all_af24.py --study draws`,
`scripts/score_mask_draws_af24.py`, `logs/mask_draws_af24_summary.{txt,json}`).

| head PSNR over 4 draws | T1 | FLAIR |
|---|---|---|
| MC-GS − L1-ESPIRiT, mean ± SD (range) | +0.359 ± 0.006 (+0.354 … +0.366) | +0.397 ± 0.006 (+0.389 … +0.403) |
| same at AF 20 (4 draws) | +0.459 ± 0.002 | +0.488 ± 0.004 |
| shift AF 20 → AF 24 | −0.099 (17 x the larger SD) | −0.092 (15 x) |
| SD of MC-GS / of L1-ESPIRiT alone (correlation) | 0.010 / 0.013 (+0.91) | 0.024 / 0.022 (+0.97) |

L1-ESPIRiT's best lambda is interior on every draw (T1 1e-3, FLAIR 2e-3); MC-GS's sharpness is above L1-ESPIRiT's on 4 of 4
draws on both contrasts (at AF 20 the T1 gate failed on 4 of 4). **This study was designed before the 2026-09-15 conventions:
it measures the spread of the margin over L1-ESPIRiT, which is no longer the quoted margin. The draw spread of the quoted
margin, MC-GS − GS-single, was not measured (GS-single was not run on draws b–d).**

## Figures

- **Figure set of record since 2026-09-16: the `_zf_caps` variant at publication resolution.**
  `figures/t1flair_v3_af24_{T1,FLAIR}_row3planes_compact_180_124_76_inset48_headerr_p3_mcinr_spirit_zf_caps.{png,pdf}`
  — the same slices, 48-voxel box, grey windows, error scale, arms and gauge as the `_mcinr_spirit` set described
  below, which it supersedes without overwriting: every plane's PSNR / SSIM / NMSE line is identical. What changed
  is a zero-filling column after GT and the caps typography (user 2026-09-16, `--with_zf --caps`, with labels
  containing `SPIRiT` exempt from upper-casing), the PNG at `--png_dpi 200` (6453 x 4352 px, which is 911 dpi at a
  180 mm page width) and TrueType fonts in the PDF (`pdf.fonttype = 42`; matplotlib's Type 3 default is rejected by
  IEEE and Elsevier submission systems). All four topics' copies are gathered in
  `Results_20260908/T1FLAIR_AF12_16_20_24_figures_hires/`; verification and the render command are in
  `mcgs_af16_wt/docs/handoff/2026-09-16-figures-hires.md`.

- **Previous figure set of record**, kept and not overwritten (as in the AF 12 / 16 / 20 topics since 2026-09-15):
  `figures/t1flair_v3_af24_{T1,FLAIR}_row3planes_compact_180_124_76_inset48_headerr_p3_mcinr_spirit.{png,pdf}` — the AF 16 layout,
  slices and 48-voxel box; ten method columns with L1-SPIRiT in the L1 column and MC-INR added; title
  `nominal AF 24, actual 24.00x` on both contrasts (user 2026-09-16), with the FLAIR panel note "1766 sampled points carry FLAIR
  data" under the panels. MC-GS is first on PSNR,
  SSIM and NMSE in every plane of both contrasts. Command:
  `fig_three_planes_compact.py --contrast <c> --slices 180 124 76 --inset 48 --inset_corner tr bl bl --mask_error --gray_cbar
  --cbar_planes 0 --tra_rot 1 --err_vmax 0.1 --pdf --mcgs_key gs_joint_p3 --single_key gs_single_p3 --tag _p3_mcinr --with_mcinr --l1_arm spirit`.
- Archive, not re-rendered: the nine-column `..._180_124_76_inset48_headerr_p3.{png,pdf}` and `..._174_110_47_headerr_p3.png`
  of 2026-09-15 05:30 (L1-ESPIRiT column, title `AF 24.00` / `AF 25.51`).
- Mask figures: `t1flair_v3_af24_mask_T1_vs_FLAIR.png`, `..._mask_series_af12_16_20_24.png`, `..._mask_T1_FLAIR_reinspect.png`.
- 48-voxel box leads are printed by the figure script only (not on the figure) and do not transfer between masks.

## Raw data on this server (2026-09-16)

Everything behind the figures of record is archived in one self-contained folder, symlinks dereferenced, so it can be
moved or tarred on its own:

```
export/af24_p3_mcinr_spirit/          72 files, 12 GiB    md5sum -c checksums.md5  ->  72/72 OK (re-run 2026-09-16)
  raw/      ksp_{T1,FLAIR}.h5 (26-coil FULL k-space), csm_espirit_{T1,FLAIR}.h5
  mask/     mask_af24_3d.h5 + the .mat of record, and draws b/c/d
  volumes/  ground truth, ZF and every arm at its kept parameters, complex64 256x256x176
  mcinr/    MC-INR outputs and score logs;  logs/  summary.json/tsv and every prep/score json
  figures/  the _p3_mcinr_spirit set of record (PNG + PDF) and the archived renders
```

Provenance, checked element-wise on 2026-09-16: `raw/ksp_{T1,FLAIR}.h5` are **bit for bit** the vendor CFL in
`/data1/data1_congcong/data/MC_Jia/Train_SpiritDiffusion_CFL/vol3_jiangyizhou/{t1_fsp,flair}_train_jiangyizhou_cc.cfl`
(md5 c449fdbd... / ce3627e9...) reshaped `(X,Y,Z,C)` -> `(C,X,Y,Z)`; 0 differing elements out of 299,892,736 per
contrast, all 256 readout planes compared, nothing sampled. The ESPIRiT maps are derived (BART `ecalib -t 0.001 -c 0.0
-r 24 -m 1`) and reproducible; the masks are not (clock-seeded generator), so the bundle and this topic's `data/` are
their only copies. Details in `export/af24_p3_mcinr_spirit/README.md`.

## Regenerating

```
# mask (MATLAB R2021a + the house vdPoisMex); the pool and the pick, never re-run into this data/
matlab -batch "run('scripts/prep_02_mask_pool_af24.m')"
PYTHONNOUSERSITE=1 PYTHONPATH= python scripts/prep_03_pick_mask_af24.py
PYTHONNOUSERSITE=1 PYTHONPATH= python scripts/prep_04_export.py      # 3-D mask + the AF read-out

# everything else: 38 jobs queued by priority onto free cards (one GPU job of this user per card, at most 4 at once;
# 12 L1-ESPIRiT runs in 4 CPU slots), then finalize_af24_jyz.sh; --dry_run shows the queue and the cards
setsid nohup python -u scripts/run_all_af24.py > /dev/null 2>&1 < /dev/null &
# mask-draw variance (draws b, c, d) and the two added arms, same launcher
matlab -batch "run('scripts/prep_02b_mask_pool_draws_af24.m')"; python scripts/prep_03b_pick_export_draws_af24.py
setsid nohup python -u scripts/run_all_af24.py --study draws --max_gpu_jobs 3 --max_cpu_jobs 3 > /dev/null 2>&1 < /dev/null &
setsid nohup python -u scripts/run_all_af24.py --study extra --max_gpu_jobs 3 --max_cpu_jobs 1 > /dev/null 2>&1 < /dev/null &
# then, in mcinr_niessen2025: score.py --af 24; here: score_mcinr_arm.py --gpu <g>, summarize.py, readme_tables.py, the figure command above
```

## Status

Complete. 2026-09-14 21:00–21:40 the mask, topic, configs and scripts were prepared; the user looked at the mask figures
and approved them before any GPU job; a 60-iteration smoke run passed; `scripts/run_all_af24.py` started 21:45:49 and
the queue (38 jobs, no failure) and `scripts/finalize_af24_jyz.sh` were done at 05:30 on 2026-09-15.

Headline (head-only PSNR, T1 / FLAIR): MC-GS **33.767 / 33.182**, first on head PSNR, SSIM and HFEN on both contrasts over
every comparison arm including GS-single. The quoted margin is over the strongest comparison arm, GS-single (user
2026-09-13): **+0.179 / +0.071** (AF 20: +0.165 / +0.052). L1-ESPIRiT was dropped from the comparison on 2026-09-15 (user;
its volumes, logs and summary rows are kept, only no longer listed): the strongest non-GS arm is CS-SENSE (TV, converged)
32.820 on T1 and joint CS-SENSE (JTV) 32.605 on FLAIR. Numbers written here earlier against L1-ESPIRiT (+0.366 / +0.400,
T1 sharpness 0.803 against its 0.787) are history. L1-SPIRiT and MC-INR, added to the AF 12 / 16 / 20 topics on
2026-09-14/15, were run here on 2026-09-15 afternoon and are registered (section "L1-SPIRiT and MC-INR registered"); the
figure set of record is now `..._p3_mcinr_spirit` (section "Figures"). The mask-draw variance of the margin over
L1-ESPIRiT was measured on three more draws (section "Mask-draw variance"). Every classical λ optimum is interior. Figures in the AF 16 figure-of-record layout:
`figures/t1flair_v3_af24_{T1,FLAIR}_row3planes_compact_180_124_76_inset48_headerr_p3.{png,pdf}` and
`..._174_110_47_headerr_p3.png`. Session record: `mcgs_af16_wt/docs/superpowers/records/2026-09-15-af24-jyz.md`.
