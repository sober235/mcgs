# Record 2026-09-12 — the 16-year-old TSC exam: from the raw-data survey to the two-contrast 3D reference dataset

Branch `16yo-mtp` (worktree `/data0/congcong/code/GS/mcgs_peds_wt`, cut from `3a44aba`, the common base of the two
autoresearch loops). Topic folder `/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_3D/` (README = the
authoritative numbers; mirrored here as `peds_tsc_mtp_3d/`). Dataset
`/data1/data1_congcong/data/3D/Peds_TSC_age16_MTP_GT/`. Design spec
`docs/superpowers/specs/2026-09-12-peds-tsc-mtp-grappa-design.md`. Figures served at
`http://localhost:8765/peds_tsc_age16/`. **DICOM and raw headers carry the patient's name; nothing in this branch
or the dataset carries it.**

## 1. What was asked and what exists now

The user asked for a reconstruction line on a newly transferred pediatric exam, isolated from every other running
reconstruction. Over the session the request settled to: (a) classical parallel-imaging reconstruction of the
prospectively undersampled 3D MTP scan, (b) the ESPIRiT-SENSE combination of that reconstruction as the ground truth,
(c) a saved two-contrast reference dataset (T1-weighted and T2*-weighted) for later retrospective undersampling.
All three exist and are verified; retrospective undersampling (stage 2) is designed but not started.

## 2. Data survey (what the exam contains)

Exam root `/data0/congcong/data/3D_data/Children_data/TSC_20220124_age16/` (UIH uMR 790, 3 T, 2022-01-24, age 16, F;
the folder label says TSC; no diagnosis field exists in the headers). Raw files are UIH `RAWDATA-V004`; the vendor
MATLAB reader in `readers/` documents the layout, re-implemented in Python in `uih_mtp/uih_raw.py`.

| scan | what | usable for a 3D multi-contrast line |
|---|---|---|
| t1_fse_flair_sag, t1_fse_flair_tra (x2), t2_fse_tra, t2_fse_flair_tra_fs | 2D multi-slice FSE, 19 slices x 5 mm, ACS ~2.25x prospective | no: 2D, the requested sag FLAIR + tra T2 pair is orthogonal; a co-planar tra FLAIR + tra T2 pair would need per-contrast grids |
| **mtp_cartesian_.7x.7x1** | **3D multi-echo spoiled GRE, 5 TE (3.25…27.41 ms) x 2 flip angles (4°, 16°), 0.7 x 0.7 x 1.35 mm, 32 ch, prospective 2x PPA + 24-line ACS + elliptical cut + 0.75 partial Fourier along PE** | **yes: chosen** |
| mtp_wave_iso1 | same contrasts at 1 mm iso, wave-CAIPI 5.1x | no: non-Cartesian PSF |
| tof3d_tra_uCS_I | 3D TOF MRA, 0.78 x 0.63 x 1.1 mm, 75 mm slab, prospective CS 3.5x, 28 ch | a third 3D contrast but a different grid |
| gre_scout, PAL prescans | localizer, noise, coil-sensitivity prescans | no |

MTP raw facts (all verified by header walks and a low-resolution reconstruction matched to the vendor DICOM):
grid RO 320 x PE 304 x SPE 104 (DC 160 / 152 / 52), readout 2x oversampled; `con` = echo; **user index `u1` =
flip angle (0 -> 4°, 1 -> 16°)**, the file holds the whole 4° pass then the 16° pass; `set` 0 is a repeat readout
of echo 0 (correlation 0.9997 with `set` 1); 11257 (PE, SPE) positions per contrast, identical for all twelve
(set, echo, u1) combinations: 2.10x against the 227 x 104 band, 2.81x against the full grid; PE 227..303 never
acquired. Image axes -> vendor DICOM (Z, rows, cols) = (SPE reversed, central 80 of 104; RO; PE cropped by 2 each side).

## 3. Which two contrasts

Brain-mask Pearson correlation between the vendor magnitude images of the ten acquired contrasts
(`logs/mtp_contrast_metrics_brain.json`): the least similar pair is **FA4 / TE 27.41 (PD-T2*-weighted) vs
FA16 / TE 3.25 (T1-weighted), r = 0.57**; next 0.62 and 0.65; neighbouring echoes at one flip angle r >= 0.94.
The vendor's T1W / PDW / SWI / QSM / T1Map / T2* / R2* / PDMap series are derived from these ten images, not acquired.

Label verification from physics (`scripts/survey/contrast_physics_check.py`): the 16°/4° signal ratio at TE 3.25 is
1.51 in white matter and 0.69 in CSF (ratio of ratios 2.19; spoiled-GRE theory 2.3); the 16° echo train gives
T2* = 53 ms (WM) and 213 ms (CSF); the 4° pass gives WM T2* = 54 ms from TE 27.41 / TE 3.25; CSF/WM contrast 0.45
(T1w) vs 1.39 (T2*w). The second contrast is the plain long-echo magnitude — the input of SWI, not SWI itself.

## 4. Stage 1 — 3D GRAPPA (commits 0cbd25a, e51d921)

`uih_mtp/grappa.py`: PE acceleration 2, sources {-3,-1,+1,+3} PE x 5 RO x 5 SPE x 32 coils, Tikhonov 1e-3,
calibrated on ACS lines 143..159 (284400 equations / 3200 unknowns), applied to the 8657 missing odd-line positions
inside the elliptical envelope; acquired lines untouched; 11702 positions (partial Fourier, corners) left zero and
saved as `m_zero`. GPU 2, 8 s per contrast.

| contrast | ACS hold-out error | noise floor (repeat echo 0) | vendor r (zero-filled) | gradient r after shading removal (zero-filled) |
|---|---|---|---|---|
| FA16 / TE 3.25 | 0.096 | 0.067 | 0.910 (0.906) | 0.759 (0.543) |
| FA4 / TE 27.41 | 0.188 | 0.144 | 0.964 (0.941) | 0.780 (0.488) |

Leave-one-line-out over eight kernel settings: 0.094–0.102, the default 0.097 — the kernel is not the limit.

## 5. Stage 1b — ESPIRiT, SENSE, CG-SENSE (commits d308413, 5f73be1)

House recipe `bart ecalib -t 0.001 -c 0.0 -r 24 -m 1` at full resolution (23 / 27 min, CPU 8 threads), unit SoS.
SENSE combination of the GRAPPA k-space vs unregularised CG-SENSE (40 iterations) on the acquired lines: complex
NRMSE in the head 0.078 / 0.196; the magnitude difference falls from 0.056 / 0.138 to 0.010 / 0.046 after a
1-voxel Gaussian, and the difference maps show no anatomy — the two classical reconstructions agree up to noise.
Correction recorded: with unit-SoS single-map ESPIRiT the SENSE image keeps the RSS coil shading, so the vendor
correlation does not move (0.912 vs 0.910); the remaining gap is the vendor's intensity normalisation and
partial-Fourier reconstruction, not the parallel-imaging step.

## 6. Ground truth and the dataset (user decisions, commits bba60b3, 9d59296, 9843ef9)

GT per contrast = `sum_c conj(S_c) ifft3c(ksp_c)` with the GRAPPA k-space and the ESPIRiT maps (the first column of
`figures/*_sense.png`). Dataset `/data1/data1_congcong/data/3D/Peds_TSC_age16_MTP_GT/`, 9.8 GB, **only
FA16_TE3.25 (T1w) and FA4_TE27.41 (T2*w)** (the user first asked for the multi-contrast set, then restricted it to
these two; the batch for the other eight echoes was stopped at the sixth contrast). Per contrast: `ksp_<c>.h5`
(dataset `kspace`), `csm_espirit_<c>.h5` (dataset `map`), `gt_sense_<c>.npy`, `masks_<c>.npz`, `meta_<c>.json`;
`index.json` with md5 sums; `README.md`. The finished-but-unwanted outputs of FA16 TE 9.29 / 15.33 / 21.37 / 27.41
and the GRAPPA half of FA4 TE 3.25 remain in the topic `data/` (23 GB, not exported, deletable).

## 7. Lesion survey (visual only, not a radiological reading)

`figures/lesion_survey_*.png`: dark blooming foci at the lateral-ventricle walls on the vendor SWI (partitions 38
and 46) that the T2*w GT reproduces at the same positions — the appearance of calcified subependymal nodules; multiple
bright cortical / subcortical patches on the vendor 2D T2-FLAIR (slices 10, 13, 14) — the appearance of cortical
tubers, not conspicuous on the MTP contrasts. The nodules are a natural lesion-fidelity target for stage 2; tubers
need FLAIR, which only exists in 2D.

## 8. Isolation and compute

Only GPU 2, and only when `nvidia-smi --query-compute-apps` showed it empty (GPU 1 carries the user's own nnUNet
training; 0 and 3–7 belong to the other reconstruction lines); BART on the CPU with 8 threads and nice 10; long
steps detached with `setsid nohup`. Tests: 194 pass (183 baseline + 11 new). Nothing outside the worktree, the
topic folder, `~/figs/peds_tsc_age16/`, the dataset folder and the session scratchpad was written; one row was
added to the archive root README.

## 9. Open decisions for stage 2 (retrospective undersampling)

1. Masks relative to the acquired support (the elliptical 227 x 104 band; recommended) or the full 304 x 104 grid.
2. Acceleration levels and mask type (recommended: 4 / 8 / 12 against the support, vd-Poisson with a 24-line
   calibration region as in the AF12 / AF16 topics, one mask for both contrasts).
3. Whether the partial-Fourier block and the corners count as known zeros for every method (recommended) or only
   for the GS arms as in the earlier topics.
4. Method table: classical + GS first (CG-SENSE, L1-ESPIRiT, GS-single, MC-GS; recommended) or the full set.
Also pending: whether to delete the 23 GB of unwanted contrast outputs.
