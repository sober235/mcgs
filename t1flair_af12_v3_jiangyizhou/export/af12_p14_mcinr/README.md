# AF 12 result of record (MC-GS P14, with MC-INR registered) — data bundle

Everything behind the jiangyizhou AF 12 comparison figure, in one self-contained folder. Real copies, no symlinks, so
this directory can be moved or tarred on its own. Written by
`Results_20260908/T1FLAIR_AF12_v3_jiangyizhou/scripts/export_af12_bundle.py --with_raw`; `MANIFEST.tsv` lists every
file with its shape, dtype, size and md5, and `checksums.md5` verifies the copy
(`md5sum -c checksums.md5`; **66/66 OK on 2026-09-13**, log kept as `md5check.out`).

Same layout and manifest format as `../../T1FLAIR_AF16_v3_jiangyizhou/export/af16_phase3/`, the AF 16 bundle of the
same subject, so the two can be read side by side.

## What the reconstruction is

Subject vol3_jiangyizhou, 3D T1 + FLAIR, retrospective undersampling with one vd-Poisson mask for both contrasts:
3847 of 45056 phase-encode points (8.54 % of the plane, calib 24), nominal AF 12 — **11.71x** against the full grid on
T1 and **9.74x** of FLAIR's own acquired k-space (its vendor elliptical shutter keeps 75.77 % of the plane). Volumes
are 256 x 256 x 176, the readout (axis 0) fully sampled. Ground truth is the conjugate-SENSE image of the FULL
k-space with the same ESPIRiT maps, normalised by P99.9 of its magnitude, so every arm is in the same gauge.

**MC-GS, P14** (`gs_joint_p14`, run `p3_e0001_ptv001_full4k`): repaired cold start (head-support initialisation, 3-NN
scales, ZF-sourced edge-TV threshold), 3.6M Gaussians, edge-TV lambda_geo 0.10 / kappa_q 0.85, k-space high-band
weight 2.2, wavelet sparsity db4 on its finest level only at 0.12 (T1) / 0.10 (FLAIR), rotation learning rate 0.008,
symmetric common-edge prior 0.001, phase-only TV 0.01, 4000 iterations. `gs_single_p14` is the same recipe with one
contrast, i.e. the single-contrast ablation — which under the user's 2026-09-13 gauge counts as a comparison arm.

**MC-INR** (`mcinr`) is the multi-contrast INR of Niessen et al. 2025 (MICCAI RIME, arXiv 2509.04888), reproduced
per-slice in 2-D from the authors' code at commit 08d37c5, run on this same mask and these same ESPIRiT maps and
re-scored through this topic's own scorer. It is a comparison arm, not ours.

## Head-only numbers (the gauge of the tables; head = |GT| > 0.05 x P99.9(|GT|), PSNR peak = max|GT|)

| arm | T1 PSNR / SSIM / HFEN | FLAIR PSNR / SSIM / HFEN |
|---|---|---|
| **MC-GS (P14)** | **37.015 / 0.9516 / 0.0928** | **36.360 / 0.9415 / 0.0935** |
| GS-single (P14) | 36.865 / 0.9496 / 0.0969 | 36.324 / 0.9409 / 0.0943 |
| L1-ESPIRiT | 36.426 / 0.9469 / 0.1036 | 35.757 / 0.9368 / 0.1000 |
| CS-SENSE (TV, converged) | 36.162 / 0.9446 / 0.1102 | 35.581 / 0.9353 / 0.1073 |
| Joint SENSE (JTV) | 36.070 / 0.9442 / 0.1143 | 35.624 / 0.9352 / 0.1073 |
| ZS-SSDU | 35.724 / 0.9422 / 0.1330 | 35.034 / 0.9353 / 0.1297 |
| CG-SENSE | 33.514 / 0.8981 / 0.1741 | 33.957 / 0.9032 / 0.1285 |
| MC-INR (Niessen 2025) | 32.192 / 0.8407 / 0.1778 | 33.042 / 0.8696 / 0.1391 |
| 3D-INR | 31.638 / 0.8373 / 0.1577 | 31.190 / 0.8277 / 0.1690 |
| 3D-DIP | 31.363 / 0.8634 / 0.2750 | 30.810 / 0.8333 / 0.2656 |

MC-GS is first in all six head cells over every other arm, GS-single included: margins **+0.150 / +0.036 dB** PSNR,
+0.0020 / +0.0005 SSIM, +0.0041 / +0.0008 HFEN, against a seed-to-seed spread of 0.0037 / 0.0026 dB. Over L1-ESPIRiT,
the strongest arm that is not ours, +0.589 / +0.603 dB.

**Two asymmetries, stated rather than hidden.** Every classical lambda was swept on this subject's ground truth and
every MC-GS knob was tuned on it, the same privilege; ZS-SSDU / 3D-DIP / 3D-INR early-stop on held-out k-space;
MC-INR ran at the paper's config values with nothing selected here. That last one does not explain where MC-INR
lands — the whole lambda re-sweep was worth at most +0.092 dB to any classical arm, and CG-SENSE has no lambda to
tune and is still 1.32 / 0.92 dB above it. Separately, the GS arms fit FLAIR's measured shutter as known zeros while
the comparison arms use the mask only. Nothing here is held out; the held-out check of P14 is on huangyuwei, and its
scorer output is in `logs/p3_hyw_P14_*_score.json`.

## Figures

`figures/` holds both versions of the figure of record, each as 600-dpi PDF (vector text and lines, image panels
rasterised at 600 ppi) and as PNG — whole transverse 180 / coronal 124 / sagittal 76, each panel carrying head-masked
slice PSNR / SSIM / NMSE, with a 48-voxel box outlined and magnified in a corner:

- `..._inset48_headerr_p14_mcinr.*` — **ten methods, MC-INR included**. This is the current figure.
- `..._inset48_headerr_p14.*` — the nine-method version delivered before MC-INR was registered. Kept because the two
  render logs are byte-identical apart from the file name: adding the column changed nothing else on the figure.

Inside the 48-box MC-GS leads the best competing arm by +1.114 / +0.963 / +1.010 dB (T1, the three planes) and
+0.635 / +1.074 / +0.685 dB (FLAIR).

Slices 180 / 124 / 76 are the AF 16 bundle's slices, deliberately: the two topics share k-space and ESPIRiT maps by
symlink and their ground-truth volumes are identical to the last bit, so the same slice is the same anatomy in both
and the panels compare one to one.

Conventions: transverse turned so the eyes point up; error maps |recon - GT| / max|GT| on a fixed jet [0, 0.1] inside
the head only; grey and error colorbars on the transverse row; the left cell of each error row is that plane's cut of
the 3-D sampling mask (k-space, never cropped).

## Contents

- `volumes/T1/`, `volumes/FLAIR/` — one array per arm, plus the ground truth. Complex arrays are the reconstruction
  itself; `_mag` files are its magnitude (what the tables and the error maps use). numpy `.npy`, 256 x 256 x 176.
  The classical file names carry the lambda that was selected on this subject.
- `mask/` — the 3-D sampling mask actually applied and the 2-D vd-Poisson pattern it came from.
- `logs/` — the topic's `summary.{json,tsv}` (every arm, every metric), the complex-gauge table, the export report of
  the data preparation, the registry of the extra GS arms, the two MC-INR rows as this topic scored them, and the
  scorer output of the six P14 runs (tuning subject, seed repeat, both single arms, and the huangyuwei held-out pair).
- `mcinr/` — the MC-INR arm's own provenance: its network output in the paper's normalised units before the linear
  rescale to topic units, the per-slice loss curves, the run metadata (mask, hyperparameters, per-slice timings), its
  self-scoring and gauge re-check, the real-data synthesis check, the slice-jitter measurement, and three documents —
  the reproduction's README and Chinese summary, and its fidelity review against the paper.
- `raw/` — **the acquisition itself**: `ksp_{T1,FLAIR}.h5`, the 26-coil fully sampled k-space each arm is
  undersampled from, and `csm_espirit_{T1,FLAIR}.h5`, the ESPIRiT maps every arm uses, both complex64
  (26, 256, 256, 176). Dereferenced real copies, so the bundle stands alone.
- `README_topic.md`, `RESULTS_AF12_T1W_FLAIR.md` — the topic's full README at the time of export and the generated
  consolidated results sheet. (That sheet is still written around the P10 recipe; the P14 promotion and the MC-INR
  registration are in `README_topic.md`.)

## Not in here

The trained Gaussian checkpoints, which live with their runs in the framework worktree
(`/data0/congcong/code/GS/mcgs_af12_wt/results_arc/<run>/`) — that directory is scratch, not archive. A reconstruction
can be reproduced from `raw/` with the runner named in the phase-3 handoff. The design-B study (FLAIR on the AF 11
mask) is not here either: P14 is an all-AF-12 result and design B is a side study of P13, with its own mask and
figures in the topic.

## Provenance

Framework branch `autoresearch-jyz-af12-phase3` (github.com/sober235/mcgs), phase-3 record
`docs/superpowers/records/2026-09-13-autoresearch-jyz-af12-phase3.md`, programme and round log
`autoresearch_jyz_af12_p3.md`, ledger `autoresearch_jyz_af12.tsv`. The MC-INR arm comes from
`/data0/congcong/code/GS/mcinr_niessen2025` (local git, tag `handoff/2026-09-13`) and its results topic
`../../T1FLAIR_MCINR_Niessen2025_v3_jiangyizhou`. Subject data is not de-identified — the directory names carry the
subject's real name, so this bundle stays on the server.
