# MC-GS — vdPois joint reconstruction (T1 + FLAIR)

Multi-contrast 3D Gaussian-Splatting (MC-GS) joint reconstruction of multi-coil
k-space: a single model with a shared Gaussian representation reconstructs both
the **plain (T1)** and **contrast (FLAIR)** volumes in one training run.

This repository captures the **corrected ESPIRiT vdPois** reconstruction runs
and the scripts needed to reproduce the undersampling mask and the scoring.

## Axis / sampling convention (house convention)

Volume shape `(256, 220, 256) = (axis0, axis1, axis2)`:

| axis  | size | anatomical plane it indexes |
|-------|------|-----------------------------|
| axis0 | 256  | coronal                     |
| axis1 | 220  | sagittal                    |
| axis2 | 256  | transverse (**readout**, fully sampled) |

The variable-density Poisson-disc (**vdPois**) undersampling acts on the
**transverse plane** (phase-encoding plane = `axis0 x axis1`); the readout axis
`axis2` (S-I) is fully sampled. The Poisson pattern therefore only appears when
you slice the transverse plane (`fix axis2`); sagittal / coronal slices show
fully-sampled readout stripes.

## Reconstruction scores (ESPIRiT-CSM self-consistent metric, AF ~11.83)

| contrast          | corrected (Poisson on transverse) | earlier (Poisson on coronal, wrong readout axis) | incoh baseline (AF 11.33) |
|-------------------|-----------------------------------|--------------------------------------------------|---------------------------|
| T1 (plain)        | **37.44 dB / SSIM 0.881**         | 37.63 / 0.870                                    | 38.51 / 0.910             |
| FLAIR (contrast)  | **37.82 dB / SSIM 0.904**         | 36.84 / 0.880                                    | 37.76 / 0.901             |

After fixing the readout axis, FLAIR improved by ~1.0 dB (SSIM 0.880 -> 0.904),
reaching/slightly exceeding the incoherent-mask baseline; T1 is essentially on
par (SSIM up). vdPois and incoherent masks should be close at matched AF and
geometry — the earlier FLAIR drop was an artifact of placing the readout axis
on the wrong dimension, now removed.

## Contents

```
configs/
  ce_joint_espirit_vdpois_transverse.yaml   # CORRECT: readout_axis=2, phase_axes=[0,1]
  ce_joint_espirit_vdpois.yaml              # earlier variant (readout_axis=0) — kept for reference
scripts/
  gen_vdpois_transverse_mask.py             # build the canonical 3D mask from a 2D ESPIRiT-utils pattern
  score_espirit_csm.py                      # PSNR/SSIM against an ESPIRiT-CSM self-consistent GT
figures/                                    # three-plane recon vs GT vs error; mask per-plane; CSM alignment
NOTES.md                                    # full working notes (conventions, diagnostics, decisions)
```

## Reproduce the mask

```bash
python scripts/gen_vdpois_transverse_mask.py \
    --src /path/to/ESPIRiT/utils/mask_vdpois_220x256_af12.mat \
    --out mask3d_vdpois_af12_transverse.mat
```

## Reproduce the score

```bash
python scripts/score_espirit_csm.py \
    --recon results_.../test_results_best/reconstruction_plain.mat \
    --ksp   /path/to/T1_trainer.mat \
    --csm   /path/to/csm_espirit_T1.mat
```

## Notes on excluded data

Multi-coil k-space, ESPIRiT CSM maps, reconstruction volumes (`.mat`),
checkpoints (`.pth`), and numpy dumps (`.npy`) are **not** tracked here — they
are large binary data, not code. The scripts above regenerate the mask and the
scores from those inputs. Training itself uses the full 3dgsVC engine
(`train_joint.py`), which lives in the parent project.


## 平扫／增强 AF16 研究归档

[2026-10-26 MC-GS autoresearch 进展、代码与验证记录](research/ce_af16_autoresearch/README.md)。当前全面领先目标尚未实现，最新状态和实验边界见归档。
