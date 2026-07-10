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

---

# Autoresearch tuning (ar_mcgs) — winner

A Karpathy-style autoresearch sweep (`autoresearch/cycle.py`, `autoresearch/program.md`)
was run on the corrected transverse mask. The scorer is artifact-centric:
`comp = psnr + 25*ssim - 60*nmse - 25*hfen - 40*max(0,1-sharp) - 200*grain`,
where `hfen` (3D-Laplacian high-freq error) measures aliasing/speckle residual
and `sharp` (gradient-energy ratio vs GT) measures detail fidelity.

**Lever findings (batch 1+2, 1500-iter cycles, harness self-CSM caliber):**

| lever delta | combined | note |
|-------------|----------|------|
| baseline (kspace_high=1.0) | 29.08 | sharp 0.73/0.78 (too blurry) |
| shared_warmup 500→700 | 29.08 | no effect (saturated) |
| lambda_dtv 0→0.02 | 28.46 | slightly negative |
| lambda_geo 0.03→0.3 | 7.84 | over-smooths, sharp collapses |
| **kspace_high 1.0→1.5** | 32.36 | sharp 0.82/0.87 |
| **kspace_high 1.0→2.0** | **33.56** | sharp 0.88/0.92 — WINNER lever |
| kspace_high 2.0 + max_radius 12 | 33.53 | no gain (kept mr20, simpler) |
| kspace_high 1.5 + curriculum | 25.52 | negative |

**Winner = `kspace_high_weight = 2.0`** (everything else at the corrected
baseline: lambda_geo 0.03, warmup 500, lambda_dtv 0, max_radius 20). The primary
lever is the high-frequency k-space weight — it directly recovers detail
(sharp 0.73→0.88) without inflating aliasing (hfen essentially flat).

**Winner re-run at 4000 iters on the ESPIRiT CSM (apples-to-apples with the baselines):**

| contrast | winner (hi=2.0) | corrected baseline (hi=1.0) | incoh baseline (AF 11.33) |
|----------|-----------------|-----------------------------|---------------------------|
| T1 (plain)       | **37.51 dB / SSIM 0.878** | 37.44 / 0.881 | 38.51 / 0.910 |
| FLAIR (contrast) | **38.22 dB / SSIM 0.903** | 37.82 / 0.904 | 37.76 / 0.901 |

Winner improves PSNR on both contrasts (T1 +0.07 dB, FLAIR +0.40 dB) with SSIM
held; FLAIR now clears the incoherent-mask baseline by ~0.5 dB. Detail fidelity
(sharpness) is visibly higher; see `figures/MC_GS_winner_{T1,FLAIR}.png`.

Winner config: `configs/ce_joint_espirit_winner_hi20.yaml`.
Full sweep table: `results/MC_GS_sweep_summary.csv`.
