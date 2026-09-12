# Cardiac_FreeRunning_MEGRE — free-running multi-echo 3D radial whole heart, phase 1

Reading, BART reference reconstructions and the evaluation gauge for the free-running 8-echo GRE data of
Mackowiak et al. (MRM 2023; Zenodo 7621356 / 7615780, CC-BY-4.0), subjects V07, V03, V04, V08.
No MC-GS yet (phase 2). No fully sampled ground truth exists for this data.
Spec: /data0/congcong/code/GS/mcgs_cardiac_wt/docs/superpowers/specs/2026-09-10-cardiac-freerunning-phase1-design.md
Plan: /data0/congcong/code/GS/mcgs_cardiac_wt/docs/superpowers/plans/2026-09-10-cardiac-freerunning-phase1.md
Code: worktree /data0/congcong/code/GS/mcgs_cardiac_wt, branch cardiac-freerunning, package cardiac_fr/;
drivers in scripts/, always run through scripts/py (GSTMI python, CPU only, nice 10, 32 BART threads).
Status 2026-09-12: phase 1 complete — four subjects with R0 (112^3) and REF (224^3, 8 echoes), every check passed,
gauge self-check on V07 passed under the amended gate (correction 6). Handoff tag handoff/2026-09-12-cardiac.

Isolation: this topic wrote only to this folder, the worktree, `/data2/congcong/data/cardiac_3d_cine_public/derived/megre/`
and `/home/congcongliu/figs/cardiac_fr/`. The main checkout (`Paired_ARC_GS_general_framework`) and `mcgs_af16_wt`
were not touched, the raw data was only read, no GPU was used, and BART ran one job at a time until 2026-09-12 12:40,
two jobs (32 threads each, nice 10) afterwards with the user's approval.

## Seven corrections made during the run

Each came from a failing check, was approved by the user, and is recorded with its date in the spec and the plan.

1. **Polarity criterion (b), spec 5.4 (09-11).** The first version compared SI profiles along the whole FOV at echo 0 and
   was dominated by the bright band at the FOV edge (best lag 0). The second used a central block (16–95 on every axis)
   and was dominated by the spine, costal cartilage and chest wall, which do not move with breathing (+0.77 voxel, wrong
   sign). The criterion measures the SI shift between the two respiratory thirds (echo 3) by normalised cross-correlation
   inside a liver-dome box and a heart box placed on each subject's own image (`polarity_boxes.json`); both boxes must
   agree in sign; the third lying further towards the head is end-expiration. The boxes need the subject's R0, so
   `run_subject.sh` runs in phases.
2. **REF regularisation, spec 5.6 (09-11).** For the radial operator the largest eigenvalue of AᴴA is thousands (5,616.5 on
   the 112^3 V07 REF; about 1 for the Cartesian brain data), so the planned absolute λ = 1e-3 regularised nothing: 60
   against 100 CG steps differed by 31.2 % inside the body (`logs/run_v07_ref_cont2.log`), 120 against 200 by 31.9 %
   (`logs/checks_V07_abs_lambda_1e-3.json`), and absolute λ 0.01 and 0.03 still gave 30 % between 60 and 100 steps,
   almost all of it high-pass (`logs/diag/ref_abs_lambda_scan_V07.log`). REF and the CG subsets use λ = λ_rel × λ_max
   with λ_max measured per subject on its REF spokes (`recon/<Vxx>/ref_lmax.json`); λ_rel = 1e-3 is the smallest
   candidate that passes the iteration check (`logs/lambda_select_V07_grid112.json`, re-confirmed on 224^3) and is frozen.
3. **The L1 λ stays absolute, spec 5.7 (09-11).** BART applies the L1 threshold in its own normalised data scale. Scaling
   the L1 candidates by λ_max as well pushed the images towards zero and failed the first self-check
   (`logs/selfcheck_V07_l1_times_lmax.*`, `figures/V07_selfcheck_e0_l1_times_lmax.png`).
4. **The L1 arm is early-stopped (09-11).** FISTA has not converged after 100 steps: at λ = 1e-3, AF 12, echo 0 on 112^3,
   100 and 200 steps differed by 27 % and 200 steps scored lower (`logs/diag/l1_fista_steps_V07.log`). The arm is
   reported at 100 steps, as in the brain topics' L1-ESPIRiT arm, and its numbers depend on that choice.
5. **2x FOV reconstruction grid, spec 5.5/5.6/7 (09-11 evening).** The 220 mm FOV holds about 40 % of the signal: gridding
   the REF data (low-pass) on a 448 mm grid showed 62 % of the coil-combined energy outside 220 mm (arms, lateral chest
   wall, back, upper abdomen), and the per-coil out-of-FOV fraction (0.19–0.87) had rank correlation 0.86 with the
   per-coil in-sample residual (`logs/diag/out_of_fov_V07.log`, `figures/V07_fov2x_gridding.png`). Consequences on
   112^3: REF's own held-out prediction had NRMSE 0.452 (echo 0), 38 % residual at the k-space centre, all arms within
   0.1–1 % of each other, and the second self-check failed on the L1 arm. A gradient-delay model was excluded (shifting
   the trajectory by the fitted delay raised the residual to 0.534, `logs/diag/delay_model_V07.log`). REF, the held-out
   spokes and every gauge subset are therefore reconstructed on 224^3 (448 mm, the same 2 mm voxels, which the 2x
   oversampled readout supports) and stored, scored and shown as the central 112^3 (both grids centre on N/2,
   `cardiac_fr.bart.crop_center`, pinned by `tests/test_cardiac_fr_bart.py`). V07 echo 0: in-sample held-out NRMSE
   0.452 → 0.130, k-centre residual 38 % → 11 %, per-coil 0.17–0.81 → 0.08–0.26, FOV-edge band / body 1.84 → 0.93
   (`logs/diag/fov2x_test_V07.json`, `figures/V07_fov2x_ref_e0.png`). R0 and the polarity thirds stay on 112^3 (they
   only serve orientation and the shift measurement; their images are folded). Cost: 7.9x per reconstruction.
   The 224^3 coil maps come from a 112^3 ESPIRiT over 448 mm (4 mm voxels, the same k-extent as the 112^3 maps: -r 48
   at the halved k spacing) Fourier-interpolated to 224^3 and renormalised to unit sum of squares (ecalib's per-voxel
   step costs about 1 ms per voxel, hours on 224^3); the check statistic is the sum of squares before renormalisation.
   BART's NUFFT scales like a unitary FFT, so -S images on the 224^3 grid are 2.83x brighter in data units than on
   112^3; every metric is scale-free and held-out predictions use the same grid's operator.
6. **Self-check gate, spec 5.7 (09-12).** On 224^3 the CG arm was monotone in all 8 verdicts and the L1 arm in held-out
   NRMSE and SSIM, but not in body PSNR/HFEN: the three L1 images differ by < 0.8 dB because their error is dominated
   by REF's own noise (`logs/diag/selfcheck_error_split_V07.log`; the CG arm's AF dependence is mostly Tikhonov
   shrinkage, body mean 0.96/0.94/0.92 of REF at echo 0). The gate is now: held-out NRMSE (the primary metric) monotone
   for every arm, the image metrics monotone for the CG arm (a check of the metric implementation); the L1 image
   metrics are reported with the note that they saturate against a noisy reference. The self-check passes 12/12
   required verdicts (`logs/selfcheck_V07.json`; the all-metrics run is kept as `logs/selfcheck_V07_grid224_allmetrics.*`).
7. **Polarity criterion (b) resolution and per-subject orientation (09-12).** The image criterion is valid only at
   |shift| ≥ 1 voxel (2 mm) in both boxes; below that criterion (a) is adopted alone if its dwell asymmetry
   (`physio.dwell_asymmetry`: time in the top fifth of the respiratory range over time in the bottom fifth) is ≥ 2
   towards the high end or ≤ 0.5 towards the low end. V03 needed this: thirds +0.67/+0.62 voxel, dome edge −0.42, extreme
   fifths +0.82/+0.89 and −0.65 — all sub-voxel with contradictory signs, while V07's fifths move 5.5/3.2 voxels (NCC)
   and 8.3 voxels (dome edge) (`logs/diag/fifths_polarity.log`); V03's dwell asymmetry is 2.54 and the SI navigator
   points the same way (`logs/diag/si_nav_polarity.log`, weak evidence). V03 is flagged "weak gating": its respiratory
   extremes are < 2 mm apart, so its end-expiration third is nearly a random third. V04 first came out "undetermined"
   for a different reason: **V04 was acquired with a coronal slab** (twix `sNormal.dCor = 1`; the other three
   `dTra = 1`), so its image axes are axis0 = inferior→superior, axis1 = right→left, axis2 = posterior→anterior, and the
   09-11 boxes (placed assuming V07's axes on a heavily folded R0) measured the shift along AP. `orientation.json` is
   per subject since 09-12; with boxes placed in V04's own frame the criterion gives +2.17/+1.97 voxels, agreeing with
   the histogram (dwell asymmetry 6.65). Note that the phyllotaxis "SI" self-navigation spoke runs along the slab normal,
   i.e. along AP for V04.

## Data

Raw Siemens twix (MAGNETOM Sola, 1.494 T, HFS), read-only, under
`/data2/congcong/data/cardiac_3d_cine_public/zenodo_7621356_7615780_freerunning_multiecho_GRE_5D/V<n>/V<n>/`; the
project archive `/data1/data1_congcong/data/3D/cardiac_freerunning_multiecho_5D/<Vxx>/freerunning_8echo.dat` holds
symlinks to them.

| item | value | source |
|---|---|---|
| measurements per .dat | 2: `AdjCoilSens` prescan + imaging; the last one is read | multi-raid header, all four |
| imaging ADCs | 176,001 = ADC 0 is a noise prescan (evalInfoMask bit 25, all loop counters 0) + 22,000 spokes × 8 echoes | MDH, all four |
| order | the 8 echoes of a spoke are consecutive; `Lin = shot × 22 + seg`, the `Seg` counter is the shot (0–999) | MDH |
| readout | 224 samples (2× oversampled), dwell 5 µs, k-space centre at sample 112 | MDH |
| echoes | TE = 1.25 + 2.05·n ms (n = 0…7), monopolar; fat–water period 4.62 ms, echo 4 (9.45 ms) is the closest to in-phase (16°), echo 0 is quadrature (97°) | ASCCONV |
| trajectory | phyllotaxis, 22 segments × 1,000 interleaves; seg 0 of every interleave is the self-navigation spoke along the slab normal | authors' `phyllotaxis3D.m`; MDH bit 34 exactly at `Lin % 22 == 0` |
| slab | transversal for V07, V03, V08; **coronal for V04** | twix `sSliceArray.asSlice[0].sNormal` |
| physiology | the 176,000 rows of `PhysioInfo.mat` are the imaging ADCs without the noise ADC; timestamps equal | PhysioInfo + MDH |
| readout edges | samples 0, 1, 2 are 18×, 4.4×, 1.5× the normal level at the same \|k\| | `logs/s02_edge_check.txt` |
| Nyquist reference | π/2 × 112² ≈ 19,704 spokes for the 112 acquisition matrix; AF = 19,704 / spokes | computed |

| subject | file | twix read time (s) | median RR (ms) | slab |
|---|---|---|---|---|
| V07 | `meas_MID00613_FID145567_FRFF_MEGRE_2P0MM_8XC_BW893_FA12_DTE2P05_MONO_22x1000.dat` | 473.5 | 748 | transversal |
| V03 | `meas_MID00063_FID24429_FRFF_MEGRE_2P0MM_8XC_BW893_FA12_DTE2P05_MONO_22x1000.dat` | 900.9 | 847 | transversal |
| V04 | `meas_MID00049_FID29919_FRFF_MEGRE_2P0MM_8XC_BW893_FA12_DTE2P05_MONO_22x1000.dat` | 566.6 | 962 | coronal |
| V08 | `meas_MID00496_FID147175_FRFF_MEGRE_2P0MM_8XC_BW893_FA12_DTE2P05_MONO_22x1000.dat` | 159.9 | 900.5 | transversal |

All four share the protocol `FRFF_MEGRE_2P0MM_8XC_BW893_FA12_DTE2P05_MONO_22x1000` (112 base resolution and partitions,
2 mm isotropic, FA 12°, one 22-spoke interleave every 374.44 ms), `ksp` (8, 22000, 24, 224), and the same spoke
counts (`logs/physio_<Vxx>.json`): 1,000 self-navigation spokes, 588 spokes inside the first 10 s, 20,439 steady non-SI
spokes (R0) and 6,813 in each respiratory third (REF is the end-expiration third). The read time follows the page
cache (`prefetch_raw.sh`), not the subject.

Derived data (not in git), `/data2/congcong/data/cardiac_3d_cine_public/derived/megre/`, linked as `data/<Vxx>`:

| file | content |
|---|---|
| `<Vxx>/raw.h5` (7.6 GB) | `ksp` (8, 22000, 24, 224) complex64 (echo, spoke, coil, sample), `lin`, `seg`, `eco`, `ts_ticks` |
| `<Vxx>/noise.npy` | the noise ADC, (24, 224); stored, not used |
| `<Vxx>/meta.json`, `physio.json` | copies of `logs/meta_<Vxx>.json`, `logs/physio_<Vxx>.json` |
| `<Vxx>/spokes.npz` | per-spoke table (respiration, time, cardiac phase) and `r0_ids`, `low_ids`, `high_ids` |
| `<Vxx>/sens.cfl` | ESPIRiT maps 112^3 (R0, polarity thirds) |
| `<Vxx>/sens2x.cfl` | ESPIRiT maps 224^3 (REF, held-out, subsets), 2.2 GB |
| `<Vxx>/ref_ids.npy` | REF spokes; written by `s07_polarity.py` only when the polarity is decided |
| `V07/gauge_split.npz` | `heldout`, `pool`, `af8`, `af12`, `af16` spoke ids |
| `<Vxx>/bart/` | BART inputs per spoke set and echo (trajectories in the set's grid units), full 224^3 reconstructions `rec_<tag>_e<k>` |
| `common/traj_r.npy` | the trajectory in r units, shared by the four subjects |

Reconstructions in this topic: `recon/<Vxx>/<tag>_e<echo>.npy`, complex64, 112^3 in data units — the full image for the
112^3 sets (r0, low, high) and the central 112^3 of the 224^3 image for ref and the subsets. Superseded 112^3 V07
results are in `recon/V07/grid112_fov220/` and `recon/V07/ref_abs_lambda_1e-3/` with README files.

## Pipeline and how to rerun

Every command runs from `scripts/`; `./py` is the GSTMI python with `PYTHONNOUSERSITE=1`, no CUDA devices,
`CARDIAC_FR_THREADS=32` and nice 10. `logs/recon_<Vxx>.jsonl` has one line per BART run (with `grid`),
`logs/bart_<Vxx>_<tag>_e<echo>.log` BART's output. Figures go to `/home/congcongliu/figs/cardiac_fr/` and `figures/`.

| spec | command | writes |
|---|---|---|
| 5.1 | `bash prefetch_raw.sh` | nothing (a read-only `cat` of the four files into the page cache); `logs/prefetch_raw.log` |
| 5.1 | `bash run_read_all.sh` (runs `./py s01_read.py <Vxx>` for the four) | `raw.h5`, `noise.npy`, `meta.json`; `logs/meta_<Vxx>.json`, `logs/s01_read_all.log` |
| 5.1 | `./py xcheck_reader_compare.py V07` (calls `xcheck_reader.m` through `matlab -batch`) | `logs/xcheck_reader_V07.txt` |
| 5.2 | `./py s02_edge_check.py` | `logs/s02_edge_check.txt` |
| 5.3 | `./py s03_traj.py`; `./py xcheck_traj_compare.py` (calls `xcheck_traj.m`) | `common/traj_r.npy`; `logs/xcheck_traj.txt` |
| 5.4 | `./py s04_physio.py` | `spokes.npz`, `physio.json`; `logs/physio_<Vxx>.json` |
| 5.5 | `./py s05_coilmaps.py <Vxx>` | `sens.cfl` (112^3); `logs/coilmaps_<Vxx>.json`; `<Vxx>_coilmaps.png` |
| 5.6 | `./py s06_recon.py <Vxx> --set r0 --iters 100`; `./py s11_figures.py <Vxx> --tag r0`; `./py orient_views.py <Vxx>` | `recon/<Vxx>/r0_e0..7.npy`; `<Vxx>_r0.png`, `<Vxx>_r0_orientation.png` |
| 5.4 (b) | `./py s06_recon.py <Vxx> --set low --echoes 3 --iters 100`, then `--set high` | `recon/<Vxx>/{low,high}_e3.npy` |
| 5.4 (b) | decide the axes (`orientation.json[<Vxx>]`, twix slab normal + images), place the dome and heart boxes in the subject's own frame (`./py box_views.py <Vxx>` draws them), `./py s07_polarity.py <Vxx>` | `logs/polarity_<Vxx>.json`, `ref_ids.npy`, `<Vxx>_polarity.png` |
| 5.5 | `./py s05b_coilmaps_2x.py <Vxx>` | `sens2x.cfl` (224^3); `logs/coilmaps2x_<Vxx>.json`; `<Vxx>_coilmaps2x.png` |
| 5.6 | `./py s08b_lambda_select.py V07` (V07 only) | `recon/V07/{ref_lmax,ref_lambda}.json`, `logs/lambda_select_V07.json` |
| 5.6 | `./py s06_recon.py <Vxx> --set ref --debug --iters 100 --lam 0.001 --lam-rel` | `recon/<Vxx>/ref_e0..7.npy`, `ref_scaling.json`, `ref_lmax.json` |
| 5.6 | `./py s08_checks.py <Vxx> --ref-iters 100 --lam-rel 0.001`; `./py s11_figures.py <Vxx> --tag ref` | `logs/checks_<Vxx>.json`; `<Vxx>_ref.png` |
| 5.7 | `./py s09_gauge_split.py V07`; heart box by hand in `gauge.json`; `./py s11_figures.py V07 --tag ref --echoes 4 --center 61,64,57 --box --name V07_ref_heart.png` | `gauge_split.npz`, `gauge.json`; `V07_ref_heart.png` |
| 5.7 | `./py s10_selfcheck.py` (`--rescore` re-scores existing reconstructions) | `gauge.json` (`l1_lambda`, sweep), `logs/selfcheck_V07.{tsv,json}`, `V07_selfcheck_e0.png` |
| 5.7 | `./py s12_compare_figure.py V07 --echo 0` (and 4) | `V07_compare_e<k>.png` |
| 5.8 | `./py freeze_params.py` | `params_frozen.json` |
| 5.8 | `bash run_rest.sh 1` (phase 1 for the other subjects), boxes, then `bash run_lanes.sh` (phase 2a/2b in two lanes) | `logs/pipeline_<Vxx>_phase<1\|2a\|2b>.log`, `logs/run_rest_phase1.log`, `logs/run_lanes.log` |

The V07 chain as it was actually run: `run_v07_r0.sh`, `run_v07_ref.sh`, `run_v07_ref_cont.sh`, `run_v07_iter200_check.sh`,
`run_v07_ref_lam.sh`, `run_v07_selfcheck.sh` (112^3, superseded), then `run_v07_fov2x.sh` (224^3: maps, λ selection,
REF, checks, figures, self-check). Long runs were started detached: `setsid nohup bash <script> > logs/<name>.log 2>&1 < /dev/null &`,
PID in `logs/<name>.pid`. Diagnostics behind the corrections are in `scripts/diag/` (README.txt there) with their
outputs in `logs/diag/`.

## Verification

- **5.1 reader.** `logs/xcheck_reader_V07.txt`: `V07: 1000 lines, ours (1000, 24, 224) vs theirs (1000, 24, 224), bitwise equal = True, max |diff| = 0`
  (the authors' `mapVBVD_JH`, unsorted, `flagIgnoreSeg`). All four files pass the reader's assertions
  (176,000 imaging lines + 1 noise line, `logs/meta_<Vxx>.json`).
- **5.2 readout edges.** `logs/s02_edge_check.txt`, level of samples 0, 1, 2 relative to the same |k|:
  V07 18.28, 4.39, 1.52; V03 20.25, 4.77, 1.59; V04 18.44, 4.42, 1.51; V08 18.74, 4.52, 1.56; flagged `[0, 1, 2]`
  and PASS in all four; the last five samples are at 0.87–1.0. Samples 3–220 are kept (dropped, not zeroed).
- **5.3 trajectory.** `logs/xcheck_traj.txt`: max |diff| kx 1.67e-16, ky 2.22e-16, kz 1.67e-16 against the authors'
  `computePhyllotaxis(224, 22, 1000, false)`; PASS (< 1e-9).
- **5.4 physiology and polarity.** `logs/physio_<Vxx>.json`: `timestamp_max_abs_diff_ms` 0.0 in all four; the histogram
  mode puts end-expiration at the high end of the respiratory signal in all four (dwell asymmetry V07 4.92, V03 2.54,
  V04 6.65, V08 2.10). Criterion (b), `logs/polarity_<Vxx>.json` (shift of the high third, voxels, + = feet):
  V07 dome −3.80 / heart −2.55 (NCC 0.73/0.77) → high, basis both; V03 +0.67 / +0.62 → undetermined, basis histogram
  (weak gating, see correction 7); V04 +2.17 / +1.97 (0.80/0.83) in its own coronal-slab frame → high, basis both;
  V08 −2.67 / −1.34 (0.83/0.83) → high, basis both. REF = the high third (6,813 spokes) in all four.
- **5.5 coil maps.** 112^3 (`logs/coilmaps_<Vxx>.json`, sum of squares within 5 % of 1 inside the support): 1.000 in all
  four (support 0.254 / 0.175 / 0.294 / 0.222 for V07 / V03 / V04 / V08). 224^3 (`logs/coilmaps2x_<Vxx>.json`, before
  renormalisation, gate 0.90): V07 0.983, V03 0.982, V04 0.982, V08 0.978.
- **5.6 iterations and scaling** (`logs/checks_<Vxx>.json`, 224^3, λ_rel 1e-3): λ_max V07 2,185.1 / V03 2,829.0 /
  V04 2,488.2 / V08 2,265.6; 60 vs 100 CG steps inside the body 0.25 % / 0.10 % / 0.14 % / 0.21 % (gate 1 %); `-w 2X`
  over `-w X` norm ratio 1.0000 in all four. BART's scaling (`recon/<Vxx>/ref_scaling.json`, echo 0): 0.1147 / 0.1079 /
  0.1021 / 0.0979.
- **5.7 gauge** (V07): held-out 341 of 6,813 REF spokes (seed 0), pool 6,472; whole-interleave subsets AF 8 (2,464 spokes,
  7.997), AF 12 (1,642, 12.000), AF 16 (1,231, 16.006); heart box axis0 27–96, axis1 33–96, axis2 30–84 (REF echo 4,
  all four chambers + 3 voxels). L1 λ sweep at AF 12, echo 0 (body PSNR vs REF, dB): 1e-4 39.92, 3e-4 40.42, 1e-3 41.55,
  **3e-3 42.72**, 1e-2 40.22 → λ = 3e-3 frozen. Self-check (`logs/selfcheck_V07.tsv`; body PSNR / body SSIM /
  held-out NRMSE at AF 8 | 12 | 16):

  | echo | arm | AF 8 | AF 12 | AF 16 |
  |---|---|---|---|---|
  | 0 | REF in-sample | held-out NRMSE 0.1319 | | |
  | 0 | CG-SENSE | 40.73 / 0.982 / 0.1408 | 37.52 / 0.966 / 0.1467 | 36.02 / 0.954 / 0.1526 |
  | 0 | L1-wavelet | 41.95 / 0.983 / 0.1325 | 42.72 / 0.982 / 0.1332 | 42.39 / 0.979 / 0.1344 |
  | 4 | REF in-sample | held-out NRMSE 0.3406 | | |
  | 4 | CG-SENSE | 32.28 / 0.945 / 0.3669 | 29.13 / 0.899 / 0.3817 | 27.26 / 0.853 / 0.3990 |
  | 4 | L1-wavelet | 31.86 / 0.945 / 0.3387 | 32.76 / 0.944 / 0.3389 | 33.35 / 0.942 / 0.3408 |

  Verdicts: 12/12 required monotone (held-out NRMSE for both arms, all image metrics for CG); the L1 body PSNR and HFEN
  are not monotone at either echo (correction 6). PASS.

## Orientation

`orientation.json` is per subject (decided from the images, `figures/orient_check_4subjects.png`, and the twix slab
normal). V07, V03, V08 (transversal slab): axis 0 patient right → left (liver at low axis 0 in coronal slices); axis 1
posterior → anterior (spine low, sternum high); axis 2 superior → inferior, the head at the LOW index (the tissue
fraction of transverse slices rises from 0.38 at axis2 = 16 to 0.84 at axis2 = 88 for V07, `logs/orient_views_V07.txt`).
V04 (coronal slab): axis 0 inferior → superior (head at the HIGH index), axis 1 right → left, axis 2 posterior → anterior.
Relative to LPS the transversal layout is a rotation, not a mirror. The self-navigation spoke runs along image axis 2 =
the slab normal, i.e. SI for the transversal subjects and AP for V04.

## Timings

Median seconds per BART run from `logs/recon_<Vxx>.jsonl` (32 threads, nice 10; the machine's load varied from 30 to
200 of 112 cores, which explains the spread):

| run | 112^3 | 224^3 |
|---|---|---|
| twix read (`s01_read.py`) | 160–900 s per subject (page cache) | |
| coil maps (`nufft -i` + `ecalib`) | 105–1,104 s + 344–1,487 s | 92–107 s + 398–448 s on the 112^3/448 mm grid, + 8 s interpolation |
| R0, one echo, 20,439 spokes, 100 CG | 164–461 s | |
| respiratory third, echo 3, 6,813 spokes, 100 CG | 130–333 s | |
| REF, one echo, 6,813 spokes, 100 CG | 140 s (V07) | 1,199 s alone (V07), 1,274–1,378 s with two lanes (V03/V04/V08) |
| REF, 60 CG (iteration check) | 87 s | 695 s alone, 797–994 s with two lanes |
| λ_max probe (`pics -e -i 1 -d5`) | 59 s | 364 s (V07), 418–539 s (others, two lanes) |
| gauge subset, CG-SENSE 100 steps | 124 s | 1,144 s |
| gauge subset, L1-wavelet FISTA 100 steps | 160 s | 1,471 s |
| held-out forward (`fmac` + `nufft`) | 4 s | 8 s |

Wall clock of the whole line: V07 112^3 chain 09-10 21:00 → 09-11 14:44 (with the failed self-checks); 2x FOV test
09-11 18:16–18:51; V07 224^3 chain 09-11 20:46 → 09-12 07:19; V03/V04/V08 phase 1 (112^3) 18:55–20:46 on 09-11; their
224^3 maps, polarity and REF 09-12 12:40–20:16 in two lanes.

## Figures

- `V03_box_views.png`
  http://localhost:8765/cardiac_fr/V03_box_views.png
  /home/congcongliu/figs/cardiac_fr/V03_box_views.png
- `V03_coilmaps.png`
  http://localhost:8765/cardiac_fr/V03_coilmaps.png
  /home/congcongliu/figs/cardiac_fr/V03_coilmaps.png
- `V03_coilmaps2x.png`
  http://localhost:8765/cardiac_fr/V03_coilmaps2x.png
  /home/congcongliu/figs/cardiac_fr/V03_coilmaps2x.png
- `V03_polarity.png`
  http://localhost:8765/cardiac_fr/V03_polarity.png
  /home/congcongliu/figs/cardiac_fr/V03_polarity.png
- `V03_r0.png`
  http://localhost:8765/cardiac_fr/V03_r0.png
  /home/congcongliu/figs/cardiac_fr/V03_r0.png
- `V03_r0_orientation.png`
  http://localhost:8765/cardiac_fr/V03_r0_orientation.png
  /home/congcongliu/figs/cardiac_fr/V03_r0_orientation.png
- `V03_ref.png`
  http://localhost:8765/cardiac_fr/V03_ref.png
  /home/congcongliu/figs/cardiac_fr/V03_ref.png
- `V04_box_views.png`
  http://localhost:8765/cardiac_fr/V04_box_views.png
  /home/congcongliu/figs/cardiac_fr/V04_box_views.png
- `V04_coilmaps.png`
  http://localhost:8765/cardiac_fr/V04_coilmaps.png
  /home/congcongliu/figs/cardiac_fr/V04_coilmaps.png
- `V04_coilmaps2x.png`
  http://localhost:8765/cardiac_fr/V04_coilmaps2x.png
  /home/congcongliu/figs/cardiac_fr/V04_coilmaps2x.png
- `V04_polarity.png`
  http://localhost:8765/cardiac_fr/V04_polarity.png
  /home/congcongliu/figs/cardiac_fr/V04_polarity.png
- `V04_r0.png`
  http://localhost:8765/cardiac_fr/V04_r0.png
  /home/congcongliu/figs/cardiac_fr/V04_r0.png
- `V04_r0_orientation.png`
  http://localhost:8765/cardiac_fr/V04_r0_orientation.png
  /home/congcongliu/figs/cardiac_fr/V04_r0_orientation.png
- `V04_ref.png`
  http://localhost:8765/cardiac_fr/V04_ref.png
  /home/congcongliu/figs/cardiac_fr/V04_ref.png
- `V07_box_views.png`
  http://localhost:8765/cardiac_fr/V07_box_views.png
  /home/congcongliu/figs/cardiac_fr/V07_box_views.png
- `V07_coilmaps.png`
  http://localhost:8765/cardiac_fr/V07_coilmaps.png
  /home/congcongliu/figs/cardiac_fr/V07_coilmaps.png
- `V07_coilmaps2x.png`
  http://localhost:8765/cardiac_fr/V07_coilmaps2x.png
  /home/congcongliu/figs/cardiac_fr/V07_coilmaps2x.png
- `V07_compare_e0.png`
  http://localhost:8765/cardiac_fr/V07_compare_e0.png
  /home/congcongliu/figs/cardiac_fr/V07_compare_e0.png
- `V07_compare_e4.png`
  http://localhost:8765/cardiac_fr/V07_compare_e4.png
  /home/congcongliu/figs/cardiac_fr/V07_compare_e4.png
- `V07_fov2x_gridding.png`
  http://localhost:8765/cardiac_fr/V07_fov2x_gridding.png
  /home/congcongliu/figs/cardiac_fr/V07_fov2x_gridding.png
- `V07_fov2x_ref_e0.png`
  http://localhost:8765/cardiac_fr/V07_fov2x_ref_e0.png
  /home/congcongliu/figs/cardiac_fr/V07_fov2x_ref_e0.png
- `V07_polarity.png`
  http://localhost:8765/cardiac_fr/V07_polarity.png
  /home/congcongliu/figs/cardiac_fr/V07_polarity.png
- `V07_r0.png`
  http://localhost:8765/cardiac_fr/V07_r0.png
  /home/congcongliu/figs/cardiac_fr/V07_r0.png
- `V07_r0_orientation.png`
  http://localhost:8765/cardiac_fr/V07_r0_orientation.png
  /home/congcongliu/figs/cardiac_fr/V07_r0_orientation.png
- `V07_ref.png`
  http://localhost:8765/cardiac_fr/V07_ref.png
  /home/congcongliu/figs/cardiac_fr/V07_ref.png
- `V07_ref_grid112.png`
  http://localhost:8765/cardiac_fr/V07_ref_grid112.png
  /home/congcongliu/figs/cardiac_fr/V07_ref_grid112.png
- `V07_ref_heart.png`
  http://localhost:8765/cardiac_fr/V07_ref_heart.png
  /home/congcongliu/figs/cardiac_fr/V07_ref_heart.png
- `V07_ref_heart_grid112.png`
  http://localhost:8765/cardiac_fr/V07_ref_heart_grid112.png
  /home/congcongliu/figs/cardiac_fr/V07_ref_heart_grid112.png
- `V07_resp_trace.png`
  http://localhost:8765/cardiac_fr/V07_resp_trace.png
  /home/congcongliu/figs/cardiac_fr/V07_resp_trace.png
- `V07_selfcheck_e0.png`
  http://localhost:8765/cardiac_fr/V07_selfcheck_e0.png
  /home/congcongliu/figs/cardiac_fr/V07_selfcheck_e0.png
- `V07_selfcheck_e0_grid112.png`
  http://localhost:8765/cardiac_fr/V07_selfcheck_e0_grid112.png
  /home/congcongliu/figs/cardiac_fr/V07_selfcheck_e0_grid112.png
- `V08_box_views.png`
  http://localhost:8765/cardiac_fr/V08_box_views.png
  /home/congcongliu/figs/cardiac_fr/V08_box_views.png
- `V08_coilmaps.png`
  http://localhost:8765/cardiac_fr/V08_coilmaps.png
  /home/congcongliu/figs/cardiac_fr/V08_coilmaps.png
- `V08_coilmaps2x.png`
  http://localhost:8765/cardiac_fr/V08_coilmaps2x.png
  /home/congcongliu/figs/cardiac_fr/V08_coilmaps2x.png
- `V08_polarity.png`
  http://localhost:8765/cardiac_fr/V08_polarity.png
  /home/congcongliu/figs/cardiac_fr/V08_polarity.png
- `V08_r0.png`
  http://localhost:8765/cardiac_fr/V08_r0.png
  /home/congcongliu/figs/cardiac_fr/V08_r0.png
- `V08_r0_orientation.png`
  http://localhost:8765/cardiac_fr/V08_r0_orientation.png
  /home/congcongliu/figs/cardiac_fr/V08_r0_orientation.png
- `V08_ref.png`
  http://localhost:8765/cardiac_fr/V08_ref.png
  /home/congcongliu/figs/cardiac_fr/V08_ref.png
- `orient_check_4subjects.png`
  http://localhost:8765/cardiac_fr/orient_check_4subjects.png
  /home/congcongliu/figs/cardiac_fr/orient_check_4subjects.png
- `resp_traces_4subjects.png`
  http://localhost:8765/cardiac_fr/resp_traces_4subjects.png
  /home/congcongliu/figs/cardiac_fr/resp_traces_4subjects.png

Files with `_grid112` / `_l1_times_lmax` / `_grid224_allmetrics` in the name are the superseded runs kept for the record.

## Known limitations

- **REF is not ground truth.** It is an end-expiration, cardiac-averaged CG-SENSE at AF ≈ 2.9 with noise and residual
  streaks (its in-sample held-out NRMSE floor is 0.132 at echo 0 and 0.341 at echo 4). Image metrics against it favour
  reconstructions of the same kind (CG-SENSE) and saturate for denoising arms (correction 6): rank methods by held-out
  k-space NRMSE and read image metrics together with their error maps.
- **Cardiac averaging.** REF is not a cardiac phase; the heart is blurred by its motion.
- **Respiratory gating quality differs between subjects.** V03's extreme fifths move < 2 mm (weak gating); V04's image
  criterion needed its own coronal-slab frame; V07 and V08 move 2–8 voxels. As a held-out subject in phase 2, V03 needs
  a footnote.
- **L1 arm.** Its λ was chosen on the same subject, AF 12 and echo 0 it is scored on (optimistic for L1, as in the brain
  topics), and it is early-stopped FISTA at 100 steps (correction 4).
- **CG subsets share REF's λ_max** (spec 5.6): with fewer spokes the Tikhonov term is relatively stronger, so part of the
  CG arm's AF dependence is intensity shrinkage (body mean 0.96/0.94/0.92 of REF at AF 8/12/16, echo 0). Kept as the
  spec's convention; a per-subset λ_max would change the arm.
- **Noise prescan.** 224 samples per coil give a coarse covariance; stored, no pre-whitening (as in the brain topics).
- **FOV.** Signal up to 448 mm is now modelled; the arms sit inside that grid for these subjects (`V07_fov2x_gridding.png`).
- **What the images show** (`V*_ref.png`): heart, great vessels, lungs, spine, descending aorta and liver are clear in all
  eight echoes and all four subjects; the even echoes are near fat–water in-phase and the odd ones opposed (dark lines at
  fat–water boundaries), on top of T2* decay (V07 R0 central mean per echo relative to echo 0: 1.0, 0.778, 0.822, 0.561,
  0.61, 0.489, 0.488, 0.455).

## Gotchas that cost time

- **`bart pics -l1` on radial data needs `-e`** (maximum eigenvalue for the FISTA step). Without it FISTA diverged
  (residual 6.5e10 against 0.043 on a 16³ phantom).
- **Always `-S`.** Without it the image stays in BART's internal scale (a factor 91,008 on V07). BART prints its
  scaling ("Inverse scaling of the data") only with `-d5`; REF runs with `--debug` to record it for `-w`, and
  `common.ref_scaling()` measures it with a one-step probe when the file is missing.
- **λ scale on radial data.** AᴴA has λ_max in the thousands here, so an `-l2` λ copied from Cartesian work is no
  regularisation; use λ_rel × λ_max (`cardiac_fr.bart.max_eigenvalue`). The `-l1` λ is absolute (corrections 2, 3).
- **The 220 mm FOV folds 62 % of the signal** (correction 5); reconstruct on 224^3 and crop. `bart traj -x 32` reaches
  |k| 15.5, twice a 16-grid's Nyquist — a matched phantom trajectory is `traj -x 16 -o 2`.
- **ecalib on 224^3 takes hours** (about 1 ms per voxel in its per-voxel step): compute the maps on 112^3 over the full
  FOV and Fourier-interpolate (`cardiac_fr.bart.interp_maps`, same FOV, half the voxel).
- **Orientation differs between subjects**: read the twix slab normal (`sSliceArray.asSlice[0].sNormal`) before placing
  anything; V04 is coronal. A folded 112^3 R0 is easy to misread.
- **Polarity by image needs local boxes** (correction 1) and a resolution floor (correction 7): whole-FOV profiles see
  the FOV edge, central blocks see the static chest wall, sub-voxel shifts have random sign.
- **ADC 0 of the imaging measurement is a noise scan**, and each .dat holds two measurements (take the last).
- **Readout samples 0–2 are contaminated**; dropped, not zeroed.
- **Cold disk.** The twix reader seeks per ADC; on a cold page cache the disk delivered 35–55 MB/s. Run
  `prefetch_raw.sh` first.
- **Shared machine.** nice 10, 32 threads, one BART job at a time (two with approval). Start long jobs with
  `setsid nohup ... &` and a PID file, wait with `kill -0 <pid>`. Never `pgrep -f`/`pkill -f` a pattern that also appears
  in the waiting shell's own command line. The session's memory watchdog kills background waiters when MemFree is low
  (page cache): detached jobs survive, waiters do not — poll the PID file.
- **zsh** does not word-split unquoted `$VAR`, and a line starting with `=` (e.g. `echo =====`) fails; quote it.
