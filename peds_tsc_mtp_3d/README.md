# Pediatric TSC exam, 3D MTP Cartesian — stage 1: classical parallel-imaging reference (GRAPPA)

Opened 2026-09-12. Code: worktree `/data0/congcong/code/GS/mcgs_peds_wt`, branch `peds-tsc-mtp`
(package `uih_mtp/`, tests `tests/test_uih_mtp_*.py`, design
`docs/superpowers/specs/2026-09-12-peds-tsc-mtp-grappa-design.md`). Figures served at
`http://localhost:8765/peds_tsc_age16/`. **The DICOM and raw headers carry the patient's name: never copy them
into file names, figures or documents.**

## Data

One exam (UIH uMR 790, 3 T, 2022-01-24, age 16, F, head). Raw file
`/data0/congcong/data/3D_data/Children_data/TSC_20220124_age16/rawdata/UID_126819142341469_mtp_cartesian_.7x.7x1.raw`
(22 GB): 3D multi-echo spoiled GRE "MTP", 5 echoes (TE 3.25 / 9.29 / 15.33 / 21.37 / 27.41 ms) x 2 flip
angles (4°, 16°), TR 40.9 ms, 32 channels, 0.7 x 0.7 x 1.35 mm, FOV 224 x 210 mm.

| item | value |
|---|---|
| k-space grid (RO x PE x SPE) | 320 x 304 x 104, DC at (160, 152, 52); readout acquired with 2x oversampling (640 samples) |
| acquired per contrast | 11257 (PE, SPE) positions: PE 0..226 only (0.75 partial Fourier, lines 227..303 never acquired), even lines outside the ACS block, ACS = PE 140..163 fully sampled, elliptical corner cut |
| acceleration | 2.10x against the 227 x 104 band, 2.81x against the full 304 x 104 grid; vendor label "PPA 2, CombinedAccel 1.8" |
| loop counters | `con` = echo; user index `u1` = flip angle (0 -> 4°, 1 -> 16°), whole 4° pass first then the 16° pass; `set` 0 = a repeat readout of echo 0 (unused here, used for the noise floor) |
| vendor reference | `dicom/mtp_cartesian_.7x.7x1_M_802`, magnitude, 10 contrasts x 80 slices, exported to `data/vendor/M_FA*_TE*.npy`; raw image axes -> vendor (Z, rows, cols) = (SPE reversed, central 80 of 104; RO; PE cropped by 2 on each side) |

No fully sampled acquisition exists. The two contrasts processed here are the two that differ most in
tissue contrast (brain-mask Pearson r = 0.57 between them): **FA16 / TE 3.25 ms** (T1-weighted) and
**FA4 / TE 27.41 ms** (PD/T2*-weighted).

## Method (stage 1)

1. `uih_mtp.uih_raw`: header-only index of the raw file (135084 imaging lines, 32 noise lines, 24 PPA
   reference lines per contrast), selective reading of one (set, echo, u1), receive-gain scaling as in the
   vendor reader, readout oversampling removed line by line (IFFT, central 320 samples, FFT).
2. `uih_mtp.grappa`: 3D GRAPPA, acceleration 2 along PE. Sources = PE offsets {-3, -1, +1, +3} x RO
   offsets {-2..2} x SPE offsets {-2..2} x 32 coils (3200 sources per target coil); weights from the ACS
   block (9 target lines 143..159, 284400 equations), Tikhonov 1e-3 relative to the mean Gram diagonal.
   Filled: the 8657 missing odd-line positions inside the elliptical envelope. Acquired lines are
   bit-identical to the measurement; the partial-Fourier region and the corners (11702 positions) stay
   zero and are saved as `m_zero`.
3. Root-sum-of-squares images for the stage-1 figures.
4. Stage 1b: ESPIRiT maps with the frozen house recipe `bart ecalib -t 0.001 -c 0.0 -r 24 -m 1` at full
   resolution on the completed k-space (the 24-cube calibration region is the ACS block itself; unit sum
   of squares over the whole FOV), the SENSE combination x = sum conj(S) . ifft(k) of the GRAPPA k-space, and
   an unregularised CG-SENSE (40 iterations, torch on the GPU) on the acquired lines only as the independent
   cross-check.

## Results

Both contrasts on GPU 2 (nothing else on the card): read 9 s, GRAPPA 8 s, save 4 s per contrast.

| contrast | acquired | GRAPPA-filled | empty by design | interpolated energy | ACS hold-out error | noise floor of that check |
|---|---|---|---|---|---|---|
| FA16 / TE 3.25 | 11257 | 8657 | 11702 | 10.4 % of total | 0.096 | 0.067 |
| FA4 / TE 27.41 | 11257 | 8657 | 11702 | 9.2 % of total | 0.188 | 0.144 |

*ACS hold-out error*: the odd lines 153..163 of the ACS block are hidden, the kernel is calibrated on the
lower half (140..151) and the hidden lines are predicted; relative k-space error against the measurement.
*Noise floor*: the same quantity for a perfect predictor, from the repeated echo 0 (`set` 0 vs `set` 1,
`logs/noise_floor.json`): ||k1 - k0|| / (sqrt(2) ||k1||) on the hidden lines, scaled to echo 4 by the
signal ratio. GRAPPA sits 1.3-1.4x above the noise floor on both contrasts.

Vendor check (`logs/vendor_check_*.json`, GRAPPA RSS vs vendor magnitude after the axis mapping, PE
offset 2; the vendor image is coil-normalised and partial-Fourier reconstructed, the RSS is not):

| contrast | Pearson r, raw intensities (zero-filled) | per-slice median / min | r of gradient magnitudes after shading removal (zero-filled) |
|---|---|---|---|
| FA16 / TE 3.25 | 0.910 (0.906) | 0.914 / 0.872 | 0.759 (0.543) |
| FA4 / TE 27.41 | 0.964 (0.941) | 0.965 / 0.937 | 0.780 (0.488) |

The raw-intensity correlation is dominated by the coil-shading difference (RSS vs the vendor's normalised
image), which is why it barely moves between zero-filled and GRAPPA on FA16; the gradient correlation and
the figures show the 2x ghost removed. The spec's a-priori 0.95 bar is met by FA4 only on raw intensities;
a fair comparison needs the SENSE-combined image (stage 1b).

Kernel choice (`logs/grappa_kernel_loo_FA16_TE3.25.json`, leave-one-line-out over the 9 calibration lines,
relative k-space error): 4 PE sources / dx 2 / dz 2 gives 0.0970 at lambda 1e-3 (0.0969 at 1e-4, 0.0999 at
1e-2); 2 PE sources 0.1021; dx 1 / dz 1 0.0970; dx 3 0.0977; dz 3 0.0975; 6 PE sources 0.0944. The
kernel is not the limiting factor (the floor is ~0.067), so the default stays.

## Stage 1b: ESPIRiT maps, SENSE combination, CG-SENSE cross-check

`ecalib` took 23 / 27 min per contrast (CPU, 8 threads, host load ~80). Maps: `data/<contrast>/csm_espirit.h5`,
dataset `map` (coil, RO, PE, SPE), SoS = 1.0000 everywhere. Images: `sense_grappa.npy` (SENSE combination of
the GRAPPA k-space) and `cgsense.npy` (CG-SENSE on the acquired lines), both complex (RO, PE, SPE).

| contrast | CG-SENSE vs SENSE(GRAPPA), head, complex NRMSE | magnitude NRMSE, raw / after 1-voxel Gaussian | high-pass noise level SENSE(GRAPPA) / CG-SENSE | vendor r, SENSE(GRAPPA) / CG-SENSE |
|---|---|---|---|---|
| FA16 / TE 3.25 | 0.078 | 0.056 / 0.010 | 0.234 / 0.241 | 0.912 / 0.903 |
| FA4 / TE 27.41 | 0.196 | 0.138 / 0.046 | 0.204 / 0.244 | 0.962 / 0.930 |

The two classical reconstructions agree up to noise: smoothing both magnitudes with a 1-voxel Gaussian
removes 80 % (FA16) and 67 % (FA4) of their difference, and the difference maps (`figures/*_sense.png`, last
column, jet [0, 0.1] of P99.9) show no anatomy, only a noise texture that is stronger in the central
overlap band where the 2x g-factor is highest. Unregularised CG-SENSE is the noisier of the two on the
low-SNR FA4 contrast (high-pass noise 0.244 vs 0.204), which is why its vendor correlation is lower.

Correction to the stage-1 note: with unit-SoS single-map ESPIRiT the SENSE combination has the same
coil-sensitivity shading as the RSS image, so the vendor correlation does not move (0.912 vs 0.910 on FA16).
The remaining gap to the vendor image is its intensity normalisation (PALAdvNorm prescans), partial-Fourier
reconstruction and filtering, not the parallel-imaging step. The chosen reference for stage 2 is the GRAPPA
k-space with the SENSE combination as its image; CG-SENSE stays a comparison arm.

Figures (`figures/`, mirrored to `~/figs/peds_tsc_age16/`):
- `FA16_TE3.25_sense.png`, `FA4_TE27.41_sense.png`: SENSE(GRAPPA) / CG-SENSE / vendor, three planes, plus the
  normalised difference map; `*_espirit_maps.png`: eight of the 32 maps (magnitude and phase).
- `FA16_TE3.25_kspace.png`, `FA4_TE27.41_kspace.png`: log-magnitude coil-RSS k-space before and after GRAPPA
  (PE x SPE plane at the RO centre, RO x PE plane at the SPE centre), the three masks, the PE profile.
- `FA16_TE3.25_image.png`, `FA4_TE27.41_image.png`: zero-filled RSS / GRAPPA RSS / vendor, three planes.

## Files

```
data/line_index.npz                     header index of the raw file (offsets, counters, flags)
data/vendor/M_FA{4,16}_TE*.npy          vendor magnitude volumes (80, 320, 300) + meta.json
data/<contrast>/kspace_grappa.h5        dataset "kspace": (coil, RO, PE, SPE) complex64, DC-centred, 2.6 GB
data/<contrast>/masks.npz               m_acq, m_grappa, m_zero, envelope (PE x SPE), acs
data/<contrast>/rss_grappa.npy, rss_zf.npy   (RO, PE, SPE) float32
data/<contrast>/meta.json               counts, kernel, timings, self-check numbers
logs/run_stage1.log, vendor_check_*.json, noise_floor.json, grappa_kernel_loo_*.json
scripts/py                              GSTMI python wrapper (PYTHONPATH -> the worktree)
scripts/run_stage1.sh                   both contrasts + figures; picks GPU 2 only if it is free, else CPU
scripts/prep_mtp_grappa.py --fa 16 --echo 0 --device cuda:0
scripts/fig_mtp_grappa.py --contrast FA16_TE3.25
scripts/noise_floor.py, grappa_kernel_loo.py, vendor_dicom_to_npy.py (nvgen env, pydicom)
scripts/run_stage1b.sh, stage1b_sense.py --contrast FA16_TE3.25 --device cuda:0   (stage 1b)
data/<contrast>/csm_espirit.h5 (dataset "map"), sense_grappa.npy, cgsense.npy; logs/stage1b_*.json, stage1b_agreement.json
```

Rerun everything: `bash scripts/run_stage1.sh --overwrite` (about 2 minutes on a free GPU).

## Caveats

- The reference k-space carries 2x parallel-imaging noise amplification in the interpolated lines and
  no data in the partial-Fourier region (effective PE resolution 227/304 of nominal, no homodyne/POCS).
  Stage-2 numbers will be relative to this reference, not to a fully sampled truth.
- Only the GS arms of the earlier topics use known-zero regions; the corners and the partial-Fourier
  block here are such a region and the asymmetry must be decided before stage 2.
- Noise prewhitening is not applied (the 32 noise lines are indexed and available).

## Isolation

GPU 2 only when no other process is on it (GPU 1 carries the user's own training, 0 and 3-7 the other
reconstruction lines); no BART on the CPU in this stage; nothing written outside this folder, the worktree,
`~/figs/peds_tsc_age16/` and the session scratchpad.
