# Pediatric TSC exam, 3D MTP Cartesian: classical parallel-imaging reference (stage 1) — design

Date 2026-09-12. Branch `16yo-mtp`, worktree `/data0/congcong/code/GS/mcgs_peds_wt` (cut from `3a44aba`,
the common base of the two running autoresearch loops). Topic folder
`/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_3D/`. Approved in chat by the user on 2026-09-12
(GRAPPA as the primary reconstruction, ESPIRiT-SENSE as the cross-check, partial-Fourier region left empty,
two contrasts first).

## Goal

Reconstruct the prospectively undersampled 3D MTP Cartesian raw data of the age-16 TSC exam with a standard
parallel-imaging method, and show (a) the completed multi-coil k-space and (b) the 3D image, for the two
acquired contrasts that differ most (FA 16° / TE 3.25 ms, T1-weighted; FA 4° / TE 27.41 ms, PD/T2*-weighted).
The completed k-space is the reference for stage 2 (retrospective undersampling, not designed here).

## Data facts (verified 2026-09-12; see the memory note `peds-tsc-age16-uih-raw`)

- Raw: `/data0/congcong/data/3D_data/Children_data/TSC_20220124_age16/rawdata/UID_126819142341469_mtp_cartesian_.7x.7x1.raw`
  (22 GB, `RAWDATA-V004`, DHL version 4, 32 channels, 640 samples per readout = 2x oversampling of 320).
- Grid: RO 320 (centre 160) x PE 304 (`FTLengthPE`, centre 152) x SPE 104 (centre 52, includes 30.75 %
  slice oversampling; the vendor exports the central 80). Voxel 0.7 x 0.7 x 1.35 mm, FOV 224 x 210 mm.
- Loop counters: `con` = echo 0..4 (TE 3.25 / 9.29 / 15.33 / 21.37 / 27.41 ms); user index `u1` = flip
  angle (0 -> 4°, 1 -> 16°), the file holds the whole 4° pass then the whole 16° pass; `set` 0 holds only
  echo 0 and is a repeat readout of echo 0 (correlation 0.9997 with `set` 1 echo 0) — ignored here.
- Sampling, identical for all 12 (set, echo, u1) combinations: PE lines 0..226 acquired (partial Fourier,
  lines 227..303 never acquired), even lines only outside the ACS block, the ACS block = PE 140..163 fully
  sampled, elliptical corner cut in the PE x SPE plane. 11257 (PE, SPE) positions per contrast:
  2.10x against the 227 x 104 band, 2.81x against the full 304 x 104 rectangle.
- Vendor reference: `dicom/mtp_cartesian_.7x.7x1_M_802` (magnitude, 10 contrasts x 80 slices, 320 x 300).
  Raw image axes map to DICOM (Z, rows, cols) = (SPE flipped, RO, PE); established with a low-resolution
  reconstruction of the ACS block (correlation 0.92–0.97 per contrast).
- No fully sampled acquisition exists. The DICOM headers carry the patient's name: never copy them into
  figures, file names or documents.

## Method

1. **Reader** (`uih_mtp/uih_raw.py`): parse the embedded protocol XML and walk the 192-byte DHL headers;
   read only the lines of the requested (set=1, echo, u1); per channel skip the 2 complex header words
   (`ReadLineDataV2`); scale by the receive-gain factor as the vendor reader does. Output: complex64
   `k_os[coil, 640, PE 304, SPE 104]` plus the acquisition mask `m_acq[PE, SPE]` and the ref-line flags.
2. **Readout oversampling removal**: inverse FFT along RO, keep the central 320 samples, FFT back
   (vendor recipe; the echo is symmetric, ROCenter = 160).
3. **3D GRAPPA** (`uih_mtp/grappa.py`, torch on one GPU): acceleration 2 along PE only. Kernel sources at
   PE offsets {-3, -1, +1, +3} x RO offsets {-2..2} x SPE offsets {-2..2} x 32 coils (3200 sources) for every
   target coil; weights calibrated on the ACS block (targets = odd PE lines 143..159 with all sources inside
   the block), Tikhonov-regularised least squares (lambda relative to the Gram trace). Applied to every
   missing odd PE line inside the acquired band whose (PE, SPE) lies inside the elliptical envelope of the
   acquired pattern (envelope = per-line SPE span of the neighbouring acquired lines); missing sources at
   the envelope or partial-Fourier edge enter as zeros. Acquired lines are never modified.
4. **What stays zero by design**: PE 227..303 (partial Fourier) and the elliptical corners. They are saved
   as a separate mask so stage 2 can treat them as known-unacquired, as the AF12 line treats the FLAIR
   shutter. No POCS / homodyne synthesis.
5. **Coil combination**: root-sum-of-squares for the stage-1 figures. ESPIRiT maps (BART `ecalib` on a
   half-resolution grid then Fourier-interpolated, or sigpy on the GPU) and the SENSE combination
   sum(conj(S) * ifft(k)) follow as stage 1b, together with a CG-SENSE reconstruction of the same raw data
   as the independent cross-check.

## Outputs (topic folder, one sub-directory per contrast `FA16_TE3.25`, `FA4_TE27.41`)

- `data/<contrast>/kspace_grappa.h5` dataset `kspace` (coil, RO, PE, SPE) complex64 — the CE-topic layout;
  `masks.npz` with `m_acq`, `m_grappa`, `m_zero`, `envelope`; `rss.npy` (RO, PE, SPE) float32; `meta.json`
  (raw UID, counters, kernel, lambda, calibration size, timings, self-check numbers).
- `figures/`: per contrast one k-space figure (log magnitude before / after GRAPPA in the PE x SPE plane at
  the RO centre and in the RO x PE plane at the SPE centre, plus the three masks) and one image figure
  (zero-filled RSS, GRAPPA RSS, vendor DICOM; transverse, coronal, sagittal). Mirrored to
  `/home/congcongliu/figs/peds_tsc_age16/` (served at `http://localhost:8765/peds_tsc_age16/`).
- `logs/`: run logs and the self-check JSON.

## Self-checks and acceptance

- Unit tests (bit-level): reader header counts equal the probe (135084 imaging lines, 11257 positions,
  identical masks); oversampling removal of a synthetic line is exact; GRAPPA on a synthetic 8-coil
  phantom (full k-space -> 2x + 24 ACS -> GRAPPA) reaches image NRMSE < 5 % and leaves acquired lines
  bit-identical.
- On the real data: hide the odd ACS lines 152..163, calibrate on 140..151 + the rest, predict them, report
  the relative error (expected a few per cent); correlation between the GRAPPA RSS volume and the vendor
  magnitude after the known axis mapping > 0.95; no visible 2x ghost in the GRAPPA image where the zero-filled
  image shows one.

## Isolation (user requirement: no interference with the other reconstruction lines)

- Only GPU 1 (free at launch, re-checked with `nvidia-smi --query-compute-apps` before every launch); one
  process at a time; nothing on GPUs 0, 3–7. CPU work is limited to figure generation and, in stage 1b,
  BART `ecalib` with `nice 10` and at most 8 threads because the cardiac line runs BART on the CPU.
- Nothing is written outside this worktree, the topic folder, `~/figs/peds_tsc_age16/` and the scratchpad.
  The other worktrees, their branches, ledgers and topic folders are not touched; the archive root README gets
  one added row.
- Long steps run detached (`setsid nohup … &`) with a log and PID file; waiting uses `kill -0`.

## Out of scope for stage 1

Retrospective undersampling and the method comparison (stage 2), MC-GS on this data, noise prewhitening
(the 32 noise lines are available and can be added later), the wave-CAIPI MTP scan (non-Cartesian PSF), the
2D FSE series.

## Tasks

1. Reader + tests → verify: pytest passes, counts equal the probe.
2. GRAPPA + phantom test → verify: NRMSE < 5 %, acquired lines untouched.
3. Pipeline script and figure script in the topic folder → verify: dry run on the FA16/TE3.25 contrast
   produces the files listed above.
4. Run both contrasts on GPU 1, self-checks, README, commit, report figures with URL and path.
5. Stage 1b: ESPIRiT maps, SENSE combination, CG-SENSE cross-check.
