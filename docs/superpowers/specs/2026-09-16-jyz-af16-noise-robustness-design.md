# jiangyizhou AF16 — measurement-noise robustness, every arm on noisy data (design, 2026-09-16)

Branch `jyz-af16-noise` (worktree `/data0/congcong/code/GS/mcgs_noise_wt`), cut from the diff-mask tip bbba53b.
Topic `/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou/`.

User decisions taken in chat on 2026-09-16 before this document:
- data = the jiangyizhou 3D T1 + FLAIR pair; AF 16; the house Poisson-disc 3-D mask; every arm of the series;
  Gaussian noise added to the measurements to test robustness;
- one noise level: added per-sample standard deviation = 2 x the data's own noise (2 sigma0);
- the design below approved as presented, including the upward extension of the TV / JTV lambda grids and the
  GS render-only diagnostic (cherry-pick 02e53dd);
- GPUs: the Gaussian-mask autoresearch loop (tmux `gauss_sched`) gives up two cards for this study.

## 1. Question and pre-registered reading

With the measurements carrying extra complex Gaussian noise at 2 sigma0 (total noise about 2.24 sigma0, SNR down about
7 dB), does MC-GS with the frozen AF16 phase-3 recipe stay first on the six head cells (head PSNR / SSIM / HFEN x T1 /
FLAIR) against every comparison arm, GS-single included, and how much does each arm lose against its own zero-noise
result?

- Primary: per contrast, the rank of MC-GS on each head metric, and its head-PSNR margin over the strongest comparison
  arm (GS-single counts, L1-ESPIRiT is retired), set against the zero-noise margin (+0.148 T1 / +0.042 FLAIR).
- Secondary: the loss of every arm, noise minus zero noise, on each head metric.
- Diagnostic, not an arm: the GS render without hard data consistency (section 6).
- The outcome is reported as it comes. MC-GS is not retuned for noise.

## 2. What is held fixed

| item | value |
|---|---|
| subject, k-space | vol3_jiangyizhou CFL, 26 coils, (256, 256, 176); h5 files of `T1FLAIR_AF12_v3_jiangyizhou/data/` |
| coil maps | the same ESPIRiT maps (`csm_espirit_{T1,FLAIR}.h5`), for every arm |
| mask | the house vd-Poisson disc mask of `T1FLAIR_AF16_v3_jiangyizhou/data/mask_af16_3d.h5` (md5 `d08179b8e74f35846455983dbe299294`), 2966 PE points broadcast along the readout; T1 15.19x, 2733 of the points carry FLAIR data |
| ground truth, scale | from the CLEAN full k-space, exactly as in the AF16 topic (`GT = sum_c conj(S_c) ifft3c(y_c)`, scale = P99.9 of abs(GT)) |
| scorer | `t1t2_common.Pair.score` (head-only PSNR / SSIM / HFEN, whole-volume metrics, sharpness) |
| arms and settings | the AF16 topic's, see section 5 |
| figure slices | 180 / 124 / 76 with the 48-voxel box (figure of record), and 174 / 110 / 47 |

The zero-noise column is the AF16 topic as recorded (`logs/summary.json`): zf, cgsense, cssense, cssense_o200,
jcssense, spirit, zsssdu, dip, inr, mcinr, gs_single_p3, gs_joint_p3. Nothing is re-run there.

## 3. The one variable: measurement noise

- **Model.** n ~ CN(0, sigma^2), independent over coils and samples, added only where the scanner acquired data:
  the support is the set of k-space samples that are non-zero in the clean file (any coil). T1 acquires the whole PE
  plane; FLAIR acquires the 34139 of 45056 lines inside the vendor's elliptical shutter, and everything outside stays
  exactly zero. The generator records the support and checks it is constant along the readout.
- **Level.** sigma = 2 sigma0_k(c). sigma0_k is the data's own per-sample noise, measured once per contrast from the
  clean ground truth: sigma0_img = sqrt(var Re + var Im) over air voxels (abs(GT) above 0.05 P99.9, closed 3x,
  dilated 12 voxels, complement, 8-voxel border excluded), then sigma0_k = sigma0_img / sqrt(f) with f the acquired
  fraction of the PE plane (T1 1, FLAIR 0.7577). Measured on 2026-09-16 in gauge units (P99.9 = 1): sigma0_img 0.01408
  (T1) and 0.01282 (FLAIR), so sigma0_k is about 0.0141 and 0.0147. The generator recomputes and records them; the raw
  standard deviation written to the file is sigma x scale(c).
- **Why this definition.** The maps satisfy sum_c abs(S_c)^2 = 1 and the FFT is unitary, so per-sample noise sigma
  appears in the SENSE-combined full image with standard deviation sigma sqrt(f). The noisy full image should
  therefore show air noise sqrt(sigma0_img^2 + sigma^2 f) = sqrt(5) sigma0_img (0.0315 T1, 0.0287 FLAIR).
- **Realization.** numpy PCG64 from `SeedSequence(20260916)`, one spawned child per contrast (T1 first), generated
  coil by coil, stored as complex64 with the source file's layout and chunking. One realization, shared by every arm.
- **Files.** `data/meas_{T1,FLAIR}.h5` (key `kspace`, about 2.4 GB each) and `logs/make_noisy_kspace.json`
  (sigma0, sigma, f, scale, seed, support counts, md5 of inputs and outputs, check results). The generator
  `scripts/make_noisy_kspace.py` refuses to overwrite.
- **Generator checks, all must pass.**
  1. meas minus clean is exactly zero outside the support.
  2. On the support, each coil's mean is within 3 standard errors of 0, and the standard deviations of the real and
     imaginary parts are sigma / sqrt(2) within 0.5 %.
  3. The air noise of the SENSE-combined noisy full image matches sqrt(sigma0_img^2 + sigma^2 f) within 2 %.
  4. The ground truth computed from the clean file equals the AF16 topic's `recon/gt_{c}_complex.npy` (single
     precision) within 1e-5 in gauge units (max absolute difference), and the scale equals the AF16 topic's
     `scale_p999` to 1e-6 relative.

## 4. Data flow and code changes (defaults unchanged everywhere)

Topic `data/`: `ksp_{T1,FLAIR}.h5` and `csm_espirit_{T1,FLAIR}.h5` are symlinks into the AF12 topic (clean; used for
the ground truth only); `mask_af16_3d.h5` and `flair_support_2d.h5` are symlinks into the AF16 topic (md5 checked);
`meas_{T1,FLAIR}.h5` are the new noisy measurements. The repo mirrors the topic without volumes as
`t1flair_af16_noise2x_v3_jiangyizhou/`.

1. **Topic scripts**, cloned from the Gaussian-mask topic (it carries the newest scheduler and figure-script fixes),
   with the topic path set to the noise topic:
   - `t1t2_common.Pair`: the ground truth and scale come from `ksp_{c}.h5` as now; the measurements come from
     `meas_{c}.h5`: `y_full = meas / scale` and `y = y_full * mask`. The environment variable `T1T2_MEAS_<C>` overrides
     the measurement path (used by the gauge check). Every arm that uses `Pair` inherits this: JTV, ZS-SSDU, 3D-DIP,
     3D-INR and L1-SPIRiT, whose kernel calibration reads the ACS block from `y_full`, i.e. from noisy data, as on a
     scanner.
   - `recon_classical.py` (ZF, CG-SENSE, CS-SENSE) has its own loader; it gets the same split.
2. **Trainer** (`paired_gs/data/dataset.py`, `paired_gs/trainers/joint_trainer.py::_load_dataset`,
   `arc/cycle.py::build_cfg` path resolution): optional config key `data.<channel>.measurement_path`. When set,
   `_apply_undersampling` takes the k-space from that file (same layout normalisation, same DC roll, divided by the
   clean `normalization_scale`) instead of `kspace_multicoil`. The ground truth, the scale, `kspace_full` and
   `kspace_multicoil` stay clean (`kspace_multicoil` is read afterwards only by the e_unobs diagnostic). The fit-side
   physics (known-zero estimate, prewhitening) sees the noisy measurements, as it would on a scanner. Unset, the code
   path is unchanged. Unit tests:
   - key unset vs key set to `data_path`: identical tensors;
   - key set to a perturbed copy: ground truth, scale and `kspace_full` identical to the clean run,
     `kspace_undersampled_complex == perturbed / scale * mask`;
   - a measurement file with a different shape raises.
3. **cycle.py**: cherry-pick 02e53dd (`--save_render`) from `16yo-mtp` (same lineage as this branch).
4. **MC-INR** (`/data0/congcong/code/GS/mcinr_niessen2025`, local repository, never pushed): `run_volume.py` gets
   `--ksp_T1 / --ksp_FLAIR` (default: its own topic files), and the run is tagged `_noise2x`. Its `score.py` needs no
   change: the source topic for AF 16 is the house AF16 topic, i.e. clean ground truth and clean scale, and M comes
   from the run's own meta. The noise topic's `score_mcinr_arm.py` then re-scores the volume through its own `Pair`.
5. **GS configs and runner**: `configs/t1flair_v3_af16n_jyz{,_singleT1,_singleFLAIR}.yaml` = the Gaussian-mask configs
   with the noise topic's paths plus `measurement_path`; FLAIR keeps `known_zero: true`. `run_noise_mcgs.sh` =
   `run_diffmask_mcgs.sh` with the experiment prefix `af16n_`; its recipe block stays byte-identical (checked with
   `diff`).

## 5. Arms and parameters (57 jobs)

| arm | jobs | settings |
|---|---|---|
| MC-GS | 1 | AF16 phase-3 recipe, frozen, 4000 iterations, seed 42, `SAVE_VOLUMES=1`, `--save_render` |
| GS-single | 2 | the same recipe, single-contrast configs |
| ZF, CG-SENSE, CS-SENSE (35 outer) | 2 | `recon_classical.py` base run per contrast |
| CS-SENSE, converged TV (200 outer) | 14 | T1 lambda 0.00035 0.0005 0.0007 0.001 0.0014 **0.002 0.0028**; FLAIR 0.00025 0.00035 0.0005 0.0007 0.001 **0.0014 0.002** |
| Joint SENSE (JTV, 200 outer) | 7 | lambda 0.00035 0.0005 0.0007 0.001 0.0014 **0.002 0.0028** |
| L1-SPIRiT | 24 | kernel 3 / 5 x lambda 0 5e-4 1e-3 2e-3 4e-3 8e-3, 150 iterations |
| ZS-SSDU | 2 | 100 epochs, held-out early stopping |
| 3D-INR | 2 | 1500 iterations, held-out checkpoint |
| 3D-DIP | 2 | 5000 iterations, lr 3e-3, held-out checkpoint |
| MC-INR | 1 | paper configuration, both contrasts in one run |

Bold = the two upward steps added because noise moves the regularisation optimum up. If a winner still sits on the
upper edge, the curve is extended by at most two more steps of x sqrt(2), and the extension is recorded. L1-ESPIRiT is
not run (retired 2026-09-16). The held-out data of ZS-SSDU, 3D-DIP and 3D-INR are noisy, as they would be on a scanner.
Lambda selection follows the AF16 topic's `summarize.py` unchanged (each family at its best lambda against the
ground truth, the oracle privilege every table of the series gives the classical arms). The row at the zero-noise
optimum lambda is reported as well, as the frozen-lambda view.

## 6. The hard-data-consistency asymmetry and the render diagnostic

The GS arms are scored, as in every topic of the series, on `x_dc`: the model's k-space with the measured samples put
back on the acquired points. With noisy measurements this returns the noise on the acquired points (6.6 % of the PE
plane, including the whole calibration block) to the image unfiltered; CS-SENSE, JTV and the other regularised arms
have no such step. The main table keeps the GS arms exactly as defined. With `--save_render` the same runs also save the
render before data consistency; it is scored through the topic's `Pair` under the keys `gs_joint_p3_render` and
`gs_single_p3_render` and shown in a separate diagnostic table, never in the main ranking.

## 7. Verification, in order

1. Baseline test suite in the new worktree before any change; the count is recorded.
2. The new trainer unit tests pass; the full suite passes again.
3. The generator checks of section 3.
4. Gauge check: with `T1T2_MEAS_<C>` pointing at the clean file, the base `recon_classical.py` run reproduces the AF16
   topic's recorded cgsense row (relative difference at most 2e-4 per metric, the `check_gauge.py` tolerance).
5. GS smoke, 60 iterations on the noisy data: the log shows the measurement path, the FLAIR known-zero estimate finds
   the shutter as in the clean runs (zero lines 233 of 2966), the npz carries the render keys. Numbers are NOT_EVIDENCE.
6. MC-INR smoke on one slice with the new flags.
7. After the queue: 57 results present, every arm registered, 11 arms x 2 contrasts in `summary.json`, and the
   `scale_p999` recorded by every job equal to the AF16 topic's (same clean ground truth for every arm).

## 8. Scheduling

`scripts/run_all_noise.py`, cloned from `run_all_diffmask.py` for one topic, keeps its rules: one job of this user per
card, the shared `.gpulock.<card>` lock, the working directory in the job fingerprint, one retry, MC-GS never retried
blindly. `--cards 1,2,5`: the Gaussian-mask loop's session (claude-64) released cards 1, 2 and 5 on 2026-09-16 and
restarted `gauss_sched` with `ALLOWED='3 4 7'` at 16:41; card 5 also carries another user's job. At most 3 GPU jobs. Launched detached from a bash file; `finalize_noise.sh` runs at the end: register the GS arms and the
render diagnostic, `summarize.py`, README tables with the zero-noise column, `compare_noise.py`, figures. Expected wall
time with 2 to 3 cards: 12 to 16 hours.

## 9. Outputs

- Topic README: what differs, the noise model and its checks, main table with the zero-noise column and the
  difference, the diagnostic table, figures, limitations; `logs/compare_noise_head_{psnr,ssim,hfen}.md`.
- Figures (served at `http://localhost:8765/t1flair_v3_af16_noise2x/`): the noisy-data figure (clean full image,
  noisy full image, added noise; three planes, both contrasts); the 12-column figure of record (GT, ZF, CG-SENSE,
  CS-SENSE, Joint SENSE, L1-SPIRiT, ZS-SSDU, 3D-DIP, 3D-INR, MC-INR, GS-single, MC-GS) at 180 / 124 / 76 with the
  48 box, `--with_zf --caps`, title under the AF label gauge B (AF 15.19) plus the noise level; the 174 / 110 / 47
  version; PNG and 600-dpi PDF, at most two PDF renders at a time (about 100 GB of RAM each).
- English record `docs/superpowers/records/YYYY-MM-DD-jyz-af16-noise.md`, Chinese summary
  `docs/YYYY-MM-DD-af16-noise-jyz-summary.md`, STATUS.md, a CLAUDE.md section, a handoff document (YYYY-MM-DD = the
  day the study completes).
- Branch `jyz-af16-noise` pushed to origin with a tag `handoff/YYYY-MM-DD-af16-noise`; not merged into main (the two
  lineages are never merged).

## 10. Limitations to state

- One subject, one noise level, one realization; the spread over realizations is not measured.
- The added noise is white and independent across coils; the scanner's own noise is correlated across coils. The level
  is defined on the combined image, so the ratio to sigma0 holds on average, not voxel by voxel.
- Maps and ground truth are clean. The maps were estimated from the fully sampled data, not re-estimated from noisy
  calibration data; only L1-SPIRiT's kernel sees the noisy calibration block.
- The ground truth carries the scanner's own noise sigma0, which bounds every arm's achievable score equally.
- The classical lambdas are chosen against the ground truth while MC-GS runs a frozen recipe; this favours the
  classical arms.
- The hard-data-consistency asymmetry of section 6.

## 11. Risks

- GPU time depends on the Gaussian-mask loop releasing two cards.
- Disk: about 4.8 GB of measurements plus about 8 GB of volumes and npz archives; /data0 has 5.0 T free.
- A 600-dpi PDF render holds about 100 GB of RAM; at most two at a time.
