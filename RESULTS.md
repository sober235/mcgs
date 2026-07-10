# MC-GS vdPois joint reconstruction — latest results

Corrected run: variable-density Poisson-disc undersampling on the **transverse
plane** (`readout_axis = 2`, phase-encoding plane = `axis0 x axis1`), ESPIRiT
CSM, AF ~= 11.83, 4000 iterations, warm-started. A single MC-GS model jointly
reconstructs both contrasts (plain = T1, contrast = FLAIR) in one training run.

## Scores (ESPIRiT-CSM self-consistent metric — same recipe across all runs)

| contrast          | corrected (Poisson on transverse) | earlier (Poisson on coronal, wrong readout axis) | incoh baseline (AF 11.33) |
|-------------------|-----------------------------------|--------------------------------------------------|---------------------------|
| T1 (plain)        | **37.44 dB / SSIM 0.881**         | 37.63 / 0.870                                    | 38.51 / 0.910             |
| FLAIR (contrast)  | **37.82 dB / SSIM 0.904**         | 36.84 / 0.880                                    | 37.76 / 0.901             |

**Takeaway.** After fixing the readout axis, FLAIR improved by ~1.0 dB
(SSIM 0.880 -> 0.904), reaching/slightly exceeding the incoherent-mask baseline;
T1 is essentially on par (SSIM up). vdPois and incoherent masks should be close
at matched AF and geometry — the earlier FLAIR drop was an artifact of placing
the readout axis on the wrong dimension, now removed.

## Axis / sampling convention

Volume `(256, 220, 256) = (axis0, axis1, axis2)`:

| axis  | size | plane                                    |
|-------|------|------------------------------------------|
| axis0 | 256  | coronal                                  |
| axis1 | 220  | sagittal                                 |
| axis2 | 256  | transverse — **readout, fully sampled**  |

The Poisson pattern appears only when slicing the transverse plane
(`fix axis2`); sagittal / coronal slices show fully-sampled readout stripes.
The ESPIRiT CSM was verified to be aligned with the reconstruction in all three
planes (coil sensitivities hug the head outline; no transpose/flip mismatch).

## Figures

| file | content |
|------|---------|
| `figures/recon_transverse_T1.png`     | T1 (plain): three-plane GT / recon / error |
| `figures/recon_transverse_FLAIR.png`  | FLAIR (contrast): three-plane GT / recon / error |
| `figures/mask_3planes_corrected.png`  | corrected mask per plane (Poisson on transverse) |
| `figures/csm_alignment_3planes.png`   | recon vs ESPIRiT CSM alignment, three planes |
| `vdpois_scores.csv`                   | full three-run score table |

## Reproduce

```bash
# 1. build the canonical 3D mask from a 2D ESPIRiT-utils vdPois pattern
python scripts/gen_vdpois_transverse_mask.py \
    --src /path/to/mask_vdpois_220x256_af12.mat \
    --out mask3d_vdpois_af12_transverse.mat

# 2. train with the corrected config (full 3dgsVC engine)
python train_joint.py --config configs/ce_joint_espirit_vdpois_transverse.yaml --gpu 0

# 3. score against an ESPIRiT-CSM self-consistent ground truth
python scripts/score_espirit_csm.py \
    --recon results_.../test_results_best/reconstruction_plain.mat \
    --ksp   /path/to/T1_trainer.mat \
    --csm   /path/to/csm_espirit_T1.mat
```
