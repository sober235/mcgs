# AF16 phase-3 figure of record — data bundle

Everything behind the jiangyizhou AF 16 phase-3 comparison figure, in one self-contained folder. Real copies, so this
directory can be moved or tarred on its own. Written by
`Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/scripts/export_phase3_bundle.py`; `MANIFEST.tsv` lists every file with
its shape, dtype, size and md5, and `checksums.md5` verifies the copy (`md5sum -c checksums.md5`; 48/48 OK on 2026-09-13).

## What the reconstruction is

Subject vol3_jiangyizhou, 3D T1 + FLAIR, retrospective undersampling with one vd-Poisson mask for both contrasts:
2966 of 45056 phase-encode points, nominal AF 16 — **15.19x** against the full grid on T1 and **12.49x** of FLAIR's
own acquired k-space (its vendor elliptical shutter keeps 75.8 % of the plane). Volumes are 256 x 256 x 176, the
readout (axis 0) fully sampled. Ground truth is the conjugate-SENSE image of the FULL k-space with the same ESPIRiT
maps, normalised by P99.9 of its magnitude, so every arm is in the same gauge.

**MC-GS, phase-3 recipe** (`gs_joint_p3`, run `p3_r3_e0001_ptv003_full4k`): repaired cold start (head-support
initialisation, 3-NN scales, ZF-sourced edge-TV threshold), 3.6M Gaussians, edge-TV lambda_geo 0.10 / kappa_q 0.85 /
eps 0.04, k-space high-band weight 2.2, wavelet sparsity db4 on its finest level only at 0.12 (T1) / 0.10 (FLAIR),
symmetric common-edge prior 0.001, phase-only TV 0.03, 4000 iterations. `gs_single_p3` is the same recipe with one
contrast, i.e. the single-contrast ablation.

## Head-only numbers (the gauge of the tables; head = |GT| > 0.05 x P99.9(|GT|), PSNR peak = max|GT|)

| arm | T1 PSNR / SSIM / HFEN | FLAIR PSNR / SSIM / HFEN |
|---|---|---|
| **MC-GS (phase 3)** | **35.706 / 0.9396 / 0.1139** | **34.928 / 0.9261 / 0.1121** |
| GS-single (phase 3) | 35.558 / 0.9371 / 0.1186 | 34.886 / 0.9249 / 0.1137 |
| L1-ESPIRiT (best other arm) | 35.209 / 0.9334 / 0.1286 | 34.503 / 0.9201 / 0.1210 |
| CS-SENSE (TV, converged) | 34.945 / 0.9317 / 0.1380 | 34.302 / 0.9180 / 0.1309 |
| Joint SENSE (JTV) | 34.831 / 0.9311 / 0.1429 | 34.355 / 0.9177 / 0.1302 |
| ZS-SSDU | 34.150 / 0.9216 / 0.1646 | 34.005 / 0.9185 / 0.1529 |
| 3D-DIP | 30.994 / 0.8549 / 0.3024 | 30.650 / 0.8290 / 0.2769 |
| 3D-INR | 30.375 / 0.7951 / 0.2125 | 30.257 / 0.7889 / 0.1998 |
| CG-SENSE | 32.354 / 0.8825 / 0.2239 | 32.826 / 0.8860 / 0.1608 |

Margin of MC-GS over the strongest comparison arm: **+0.497 dB (T1) and +0.425 dB (FLAIR)**; seed-to-seed spread of
the recipe is 0.001 / 0.000 dB. Every classical lambda was swept on this subject's ground truth, the same privilege
the MC-GS knobs had — a fair comparison, not a held-out one.

## Figures

`figures/` holds the figure of record and its alternative, each as 600-dpi PDF (vector text and lines, image panels
rasterised at 600 ppi) and as PNG:

- `..._180_124_76_inset48_headerr_p3.*` — **the figure of record**: whole transverse 180 / coronal 124 / sagittal 76,
  each panel carrying head-masked slice PSNR / SSIM / NMSE, with a 48-voxel box outlined and magnified in a corner.
  Inside that box MC-GS leads the best competing arm by +0.892 / +0.826 / +0.836 dB (T1, the three planes) and
  +0.824 / +1.102 / +1.060 dB (FLAIR).
- `..._180_124_72_inset64_headerr_p3.*` — the same with a 64-voxel box (wider field, smallest lead +0.697 dB).

Conventions: transverse turned so the eyes point up; error maps |recon - GT| / max|GT| on a fixed jet [0, 0.1] inside
the head only; grey and error colorbars on the transverse row; the left cell of each error row is that plane's cut of
the 3-D sampling mask (k-space, never cropped).

## Contents

- `volumes/T1/`, `volumes/FLAIR/` — one array per arm, plus the ground truth. Complex arrays are the reconstruction
  itself; `_mag` files are its magnitude (what the tables and the error maps use). numpy `.npy`, 256 x 256 x 176.
- `mask/` — the 3-D sampling mask actually applied and the 2-D vd-Poisson pattern it came from.
- `logs/` — the topic's `summary.{json,tsv}` (every arm, every metric), the competitor bar, the export report of the
  data preparation, the registry of the extra GS arms, and the scorer output of the four phase-3 runs.

- `raw/` — **the acquisition itself** (user request 2026-09-13): `ksp_{T1,FLAIR}.h5`, the 26-coil fully sampled
  k-space each arm is undersampled from, and `csm_espirit_{T1,FLAIR}.h5`, the ESPIRiT maps every arm uses, both
  complex64 (26, 256, 256, 176). In the topic these two pairs are symlinks into the AF 12 topic of the same subject;
  here they are dereferenced real copies, so the bundle stands alone. Plus `flair_support_2d.h5`, FLAIR's acquired
  support on the phase-encode plane (the vendor elliptical shutter, 34139 of 45056 lines).

## Not in here

The trained Gaussian checkpoints, which live with their runs in the framework worktree
(`/data0/congcong/code/GS/mcgs_af16_wt/results_arc/<run>/checkpoints/`) — that directory is scratch, not archive. A
reconstruction can be reproduced from `raw/` with the command in the phase-3 STATUS.

## Provenance

Framework branch `autoresearch-jyz-af16-phase3` (github.com/sober235/mcgs), phase-3 record
`docs/superpowers/records/2026-09-13-autoresearch-jyz-af16-phase3.md`, programme and round log
`autoresearch_af16_p3.md`, ledger `autoresearch_af16_p3.tsv`. Subject data is not de-identified — the directory names
carry the subject's real name, so this bundle stays on the server.
