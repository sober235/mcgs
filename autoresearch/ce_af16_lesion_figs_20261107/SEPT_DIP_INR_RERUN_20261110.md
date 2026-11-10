# September CE retro AF16: 3D-DIP and 3D-INR rerun with the October audit settings (2026-11-09 / 11-10)

## Why

The September topic `Results_20260908/CE_retro_cssense_af16` (2026-09-18) is the only run in which MC-GS beats every
arm on a lesion plane of all three axes (see `README.md`, "September 2026 data"). Its two untrained network arms were
weak, though: they used the registered September settings, with no output support and no TV term. That left 97 % of 3D-DIP's
error energy outside the head, and 3D-INR was still at 28 dB head PSNR on pre.

The October comparison-arm audit (branch `ce-af16-mask-audit`, `ce_retro_cssense_af16/CMP_ARMS_AUDIT.md`) fixed both
arms on the vdPois data. This round reruns those audited settings, unchanged, on the September data and redraws the
three-plane figure with them. No other arm changes.

## Setup

- **Data:** September CE retro topic, BART Poisson-disc AF16 (2552 of 40832 phase-encode points), CS-SENSE truth,
  42 coils, 256 × 232 × 176. The September `sim/` is used through a symlink and is never written.
- **Recon script:** `mcgs_ce_mask_wt/ce_retro_cssense_af16/scripts/recon_dip_inr.py` at `b01fadf`. The audited
  script is run unchanged, and the wrapper asserts that the scripts directory is clean.
- **Settings:** the October audit winners, each picked by its own pre-registered rule (held-out k-space residual on
  5 % of the phase-encode lines, no GT):

  | arm | settings |
  |---|---|
  | 3D-DIP | `--support 6 --dip_base 96 --tv 0.03 --iters 12000 --lr 3e-3 --eval_every 200` (8.55 M weights, 0.41 × outputs) |
  | 3D-INR | `--support 6 --inr_nmax 512 --tv 0.03 --iters 1500 --eval_every 50` |

  Support means the zero-filled head support (|ZF| > 0.05 P99.9), dilated by 6 voxels. It uses acquired data only, no GT
  or ROI.
- **Wrapper:** `sept_cmp_support.py --scan {pre,post}` builds a new topic directory and refuses to overwrite.
  The wrapper is not in this repository; its SHA-256 is `3985f421…cbb4c4`.
- **Runs:** each on one A800 (physical GPU 2).
  - pre: job `j-20261109T100902-c9266335`, DIP 13433 s, then INR 2276 s.
  - post: DIP 13574 s. It was launched without `MCGS_ARTIFACT_DIR`, so the wrapper's final self-copy raised
    `SameFileError` after DIP had finished and written its volume and score. INR post was rerun alone with
    `--methods inr` (2311 s).
  - No `record_post.json` exists for DIP post. Its volume hash is in the figure record below.

## Volume-level results (truth-scored, head mask |GT| > 0.05 P99.9)

| arm | scan | head PSNR | head SSIM | head NMSE | whole PSNR | HFEN | selected iter | held-out kres |
|---|---|---|---|---|---|---|---|---|
| 3D-DIP Sept | pre | 29.324 | 0.8093 | 0.0351 | 18.444 | 0.938 | 4800 | 0.11134 |
| **3D-DIP new** | pre | **30.758** | 0.8610 | 0.0252 | 34.155 | 0.805 | 11600 | 0.10572 |
| 3D-DIP Sept | post | 30.235 | 0.7949 | 0.0473 | 21.062 | 0.990 | 4700 | 0.10719 |
| **3D-DIP new** | post | **32.067** | 0.8565 | 0.0310 | 35.468 | 0.809 | 12000 | 0.09791 |
| 3D-INR Sept | pre | 28.084 | 0.6937 | 0.0467 | 26.287 | 1.686 | 750 | 0.13453 |
| **3D-INR new** | pre | **32.393** | 0.8819 | 0.0173 | 35.663 | 0.694 | 1500 | 0.10562 |
| 3D-INR Sept | post | 30.609 | 0.7677 | 0.0434 | 27.968 | 1.675 | 1250 | 0.12940 |
| **3D-INR new** | post | **34.262** | 0.9032 | 0.0187 | 37.513 | 0.650 | 1500 | 0.09608 |

- **3D-DIP:** head PSNR rises by 1.43 dB (pre) and 1.83 dB (post). Whole-image PSNR rises from 18–21 dB to 34–35 dB,
  because the support removes the signal outside the head. The held-out residual still falls at the last
  evaluation on post (selected iteration = 12000), as it did on vdPois.
- **3D-INR:** head PSNR rises by 4.31 dB (pre) and 3.65 dB (post). Both scans select the last iteration.
- In both arms, a lower held-out residual goes with a better truth score, so the GT-free selection rule
  and the truth agree.

## Three-plane figure: transverse 201, coronal 131, sagittal 48, pre and post

Same planes, house style and options as the September figure (`--three 201 131 48 --err_layout zoom --cbar single`).
Only the 3D-DIP and 3D-INR columns change. Lesion-plane PSNR / SSIM / NMSE are computed on the head mask with
`plane_metrics` arithmetic (PSNR at max |GT|):

| plane | scan | 3D-DIP Sept → new | 3D-INR Sept → new | ZS-SSDU | GS-single | MC-GS |
|---|---|---|---|---|---|---|
| transverse 201 | pre | 31.46 → 32.62 | 28.62 → 33.40 | 32.91 | 34.00 | **34.25** |
| transverse 201 | post | 31.53 → 33.28 | 31.59 → 36.12 | 34.97 | 37.10 | **37.53** |
| coronal 131 | pre | 31.47 → 33.12 | 28.86 → 33.62 | 33.20 | 34.62 | **34.81** |
| coronal 131 | post | 31.68 → 33.41 | 31.01 → 34.84 | 34.41 | 35.77 | **36.07** |
| sagittal 48 | pre | 30.28 → 31.61 | 28.62 → 32.95 | 33.01 | 34.02 | **34.23** |
| sagittal 48 | post | 31.44 → 33.10 | 31.43 → 34.96 | 34.61 | 35.95 | **36.32** |

SSIM and NMSE move the same way. For example, transverse 201 post: 3D-INR SSIM 0.7659 → 0.9235, NMSE 0.0413 → 0.0146.

- **MC-GS still wins all 18 conditions** (3 planes × pre / post × PSNR / SSIM / NMSE). The runner-up is GS-single in
  every row, as before. MC-GS margin over the best other arm, PSNR, pre / post:
  - transverse: +0.25 / +0.43 dB
  - coronal: +0.19 / +0.30 dB
  - sagittal: +0.21 / +0.38 dB
- **3D-INR is now third** in five of six rows, ahead of ZS-SSDU, the classical arms and MC-INR. On sagittal pre it is just behind ZS-SSDU
  (32.95 vs 33.01). 3D-DIP moves from the bottom group to the middle.
- The figure script recomputed all 66 panel numbers per plane and reported `strict True` on each plane.

## Caveats

- **ZS-SSDU is still the September 5-block, ≤ 100-epoch arm.** On vdPois the October audit's published-size network
  (15 blocks, 400 epochs) is 1.5–2.2 dB stronger, and it is the arm MC-GS does not beat on the transverse axis.
  This figure is therefore fair for 3D-DIP and 3D-INR, but not yet for ZS-SSDU.
- Single case, truth-scored, and the truth is a CS-SENSE reconstruction of the real AF 5.33 data, resimulated with
  the same coil maps.
- The figure numbers come from the figure script (topic mode), not from a frozen evaluator.

## Files (server, not in git)

- New topic: `/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_sept_cmp_20261109/`, containing:
  - `recon/{dip_sup_b96_tv0.03,inr_i10_n512_sup6_tv0.03}_{pre,post}_complex.npy`
  - `logs/*.json` (scores)
  - `run_post.log`, `run_post_inr.log`
  - `artifacts_post/record_post.json` (INR post)
  - `code/` (wrapper and figure-script copies)
  - The pre record is `~/.cache/mcgs-vscode/jobs/j-20261109T100902-c9266335/artifacts/record_pre.json`.
- Figure topic: `.../mcgs_ce_sept_cmp_20261109/fig_topic/`, symlinks only. Every September volume is linked unchanged
  except `dip_{pre,post}` and `inr_{pre,post}`, which point at the new volumes.
- Figure: `.../mcgs_ce_sept_cmp_20261109/figs/sept_bart_three_cmpfix/`, containing
  `ce_af16_lesion_three_planes_201_131_48_prepost_Sept_2026_09_18_recipe_BART_Poisson_disc_AF16_3D_DIP_and_3D_INR_rerun_with_the_Oct_audit_settings_zoomerr{.png,_preview.jpg}`
  (PNG 16 MB) and the record JSON with every drawn volume's SHA-256.
- New volume SHA-256:

  | volume | SHA-256 |
  |---|---|
  | DIP pre | `43e4426e182c079a05fb17d4d1983e1f2a51dc92bea678383f309e24724cbce9` |
  | DIP post | `d09c4ec3ff93453ea25f102c7482ea34268f0fefc16b1b0a23d68bc61f054651` |
  | INR pre | `ff28a2b835c5b9410795f550e69a88fd87901383a6832c032d5f06eda3d1fef1` |
  | INR post | `13beec6d7ab4f6f1ab16ae51676d3a7e2202259e9ad7138c7cbadf3d5087517d` |

## Reproduce

```sh
O=/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_sept_cmp_20261109
# reruns (one GPU each; MCGS_ARTIFACT_DIR must differ from $O/logs)
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 PYTHONPATH= MCGS_ARTIFACT_DIR=$O/artifacts_post \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -u $O/code/sept_cmp_support.py --scan post
# figure (CPU)
cd $O/code && env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python \
  fig_lesion_plane_prepost.py --topic $O/fig_topic \
  --label "Sept 2026-09-18 recipe (BART Poisson-disc AF16); 3D-DIP and 3D-INR rerun with the Oct audit settings" \
  --out $O/figs/sept_bart_three_cmpfix --three 201 131 48 --err_layout zoom --cbar single
```

## Next

To make the September comparison fair for every arm, rerun ZS-SSDU at the published network size on the September data.
A cleaner alternative is the open item from `README.md`: rerun the current TS01 pipeline on the September BART mask.
