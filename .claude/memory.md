# Project Memory — MC-GS winner-v2 (AF=12 dual-contrast T1+FLAIR 3D recon)

> Source of truth: `configs/winner_v2.yaml`. This file freezes the WINNER method +
> the 28-experiment autoresearch conclusion so it is not re-derived or re-tuned.
> Last verified: 2026-07-15 (all training idle; winner reproduced).

## 1. Winner method = MC-GS "winner-v2"

**Algorithm.** Multi-Contrast Gaussian Splatting joint reconstruction of a dual-contrast
3D MRI pair (`JointTrainer` + `JointGaussianModel3D`, `paired_gs/trainers/joint_trainer.py`).
- ONE shared 3D-Gaussian geometry cloud: per primitive μ (position), Σ = R(q)·diag(sx²,sy²,sz²)·Rᵀ
  (scale s + rotation quaternion q). This shared geometry IS the "基础结构" (shared anatomy).
- Per-primitive **dual complex density**: α_plain (→T1) and α_contrast (→FLAIR). Geometry shared,
  amplitudes independent. (This is "shared dictionary + independent per-contrast coefficients",
  NOT a response-decoder model — that is the separate Paired-ARC-GS frontier.)
- Warm-started from a dual-warm checkpoint of **576 902 Gaussians**; densification OFF.
- Forward operator = SENSE  A_m = M_m · F_3D · S_m  (per contrast m).
- Scoring (no alpha-fit on scored path): hard-DC synthesis  k_dc = m·y + (1−m)·k_model,
  then conjugate-SENSE combine  x = Σ_coils ifft3c(k_dc)·conj(csm).

## 2. Data / acquisition (both contrasts)
- Pair: **T1 = plain** (high-SNR guide channel) + **FLAIR = contrast** (guided channel).
  `data.*.name` is the ONLY thing making a pair "T1/FLAIR" — code is contrast-agnostic.
- **AF = 12**, mask `mask3d_vdpois_af12_transverse.mat` (separable vd-Poisson, **net AF 11.83,
  8.45 % sampled**, prospectively-Cartesian-realizable). SAME mask for both contrasts.
- Volume (256,220,256) = (coronal,sagittal,transverse); 12 coils. readout_axis=2 (S-I full),
  phase_axes=[0,1] (transverse PE plane).
- CSM: **ESPIRiT** `ecalib -t 0.001 -c 0.0 -r 24 -m 1`, key `map`, (256,220,256,12), SoS≡1 full-FOV.
  Estimated from full ACS only, independent of undersampling mask → reusable.
- normalization_percentile 99.9, normalize_kspace true, mask_type external.
- Data paths (amax): T1/FLAIR `.mat` under `/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/
  Train_3dBrain_huangyuwei_20230705-160927-1451_161015_{T1,FLAIR}_trainer.mat`; CSM under
  `/data0/.../3dgsVC/csm_espirit/csm_espirit_{T1,FLAIR}.mat`; mask under `/data0/.../3dgsVC/mask/`.

## 3. Gaussian hyperparameters (`gaussian:`)
| key | value | note |
|---|---|---|
| initial_num_points | 200000 | overridden by warm-start (576902) |
| init_scale | 0.008 | |
| position_lr | 0.0003 | |
| scale_lr | 0.002 | |
| rotation_lr | 0.0003 | |
| density_lr | 0.015 | |
| warm_start_checkpoint | `/data0/.../results_dualwarm_v2_t1_flairTV5/checkpoints/latest.pth` | 576902 Gaussians |

## 4. Loss hyperparameters (`loss:`)
| key | value | role |
|---|---|---|
| loss_type | l2 | |
| **tv_weight** (=lambda_geo) | **0.06** | PRIMARY anti-grain; sweet spot 0.05–0.06 |
| kspace_freq_adaptive | true | |
| kspace_center_fraction | 0.05 | |
| kspace_center_weight | 0.3 | low-freq band weight (de-emphasized) |
| kspace_mid_weight | 1.0 | |
| **kspace_high_weight** | **2.2** | sharpness lever (opposes TV; 2.0–2.2) |
| lambda_diff / lambda_enh | 0.01 / 0 | **IGNORED by v2 trainer** (q(a) posterior replaces) — do not tune |

## 5. Training hyperparameters (`training:` / `voxelizer:`)
| key | value |
|---|---|
| max_iterations | 4000 (winner re-run; sweep cells 1500) |
| shared_warmup_iters | 500 |
| seed | 42 |
| max_grad_norm | 1.0 |
| lr_scheduler | exponential, gamma 0.9995 |
| eval/log/save_every | 1000 / 50 / 1000 |
| voxelizer.max_radius | 20 |
| voxelizer.tile_size | 8 |
| adaptive_control | **absent → densification DISABLED** (`enable` defaults False, joint_trainer.py:556) |

## 6. Canonical results
| contrast | PSNR | SSIM | source |
|---|---|---|---|
| T1 (plain) | **37.74 dB** | 0.876 | yaml header @4000 iters |
| FLAIR (contrast) | **38.14 dB** | 0.894 | yaml header @4000 iters |

Reproduction range (my runs, 1500–3000 iters): T1 37.63–37.75, FLAIR 38.06–38.13. Both beat ZF on ssim∧nmse∧hfen.

## 7. Reproduce
```
# base config = configs/winner_v2.yaml. Sweep harness = arc/cycle.py.
# WARNING: cycle.py CLI defaults CLOBBER yaml (lambda_geo default 0.03, kspace_high default 1.0),
# so you MUST pass --lambda_geo 0.06 --kspace_high 2.2 --warmup 500 to reproduce the winner.
CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1 PYTHONPATH= \
  python arc/cycle.py --exp winner --gpu N --iters 4000 \
  --lambda_geo 0.06 --kspace_high 2.2 --warmup 500 --max_radius 20 --save_volumes
```

## 8. KEY FINDING — the "AF=12 ceiling" was a MASK-REGISTRATION ARTIFACT (CONFIRMED 2026-07-15)

> ✅ **CONFIRMED — this OVERTURNS the old "saturated ceiling / data-limited" conclusion (§8b).**
> The undersampling **mask's ACS was misregistered from the data's k-space DC by ~22 px on the PE axis**
> (data DC at array index axis1=132, matrix center=110; `M@DC=0`). The pipeline (`_apply_undersampling`
> dataset.py:728, `kspace_multicoil * mask`, no fftshift) captured only **27% (T1) / 31% (FLAIR)** of
> k-space energy vs **74% / 89%** when the mask is rolled onto the DC.
> **Fix = roll the shared mask by (2,24,-6) so its ACS lands on the k-space DC** (`mask3d_vdpois_af12_transverse_ALIGNED.mat`,
> config `configs/winner_v2_aligned.yaml`). Retrained winner-v2 (same warm-start / 3000 iters / hyperparams,
> only mask differs):
>
> | | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM |
> |---|---|---|---|---|---|
> | misaligned (aa_base3k) | 37.629 | 0.8773 | 0.0247 | 38.131 | 0.8930 |
> | **aligned (reg_aligned)** | **38.505** | **0.8921** | **0.0202** | **38.258** | **0.9018** |
> | Δ | **+0.88 dB** | +0.015 | −0.0045 | +0.13 dB | +0.009 |
>
> **T1 crosses 38 (38.505); both contrasts now >38 = the user's original PSNR>38 target, MET.** This is a
> CONSERVATIVE result (the shared warm-start was fit to the misaligned mask, biasing the A/B toward the
> baseline). Likely larger with a re-derived warm-start / 4000 iters. So T1≈37.75 was NOT an AF=12
> information limit — it was throwing away most measurable central energy. **Real lever = fix mask/k-space
> registration (likely affects ALL training data — a systematic fftshift/centering mismatch), not the AF or
> more hyperparameter tuning.** Credit: user hypothesized the DC-vs-ACS offset. See [[mcgs-winner-v2-af12]].
>
> ⚠️ **Gauge correction (2026-07-16):** the 38.505 is PSNR@`data_range=max(gt)`. T1's peak (1.887) exceeds
> FLAIR's (1.568), inflating T1 by ~1.6 dB — so the "T1 crosses 38 / surpasses FLAIR" reading is a gauge
> artifact. Under a unified data_range **FLAIR leads T1 by ~1.4 dB** (NMSE 0.0202 vs 0.0085, 2.4×). The
> per-contrast >38 milestone and "mask-fix was the real win" still stand; only the cross-contrast comparison
> was wrong. Root cause T1 is harder = spectrum–sampling mismatch (T1 energy is higher-freq, vd-Poisson
> drops it); contrast-adaptive sampling ruled out. Full analysis → project `memory.md` §3.

## 8b. (SUPERSEDED by §8 — kept for context) old "saturated ceiling" claim
28-experiment single-knob autoresearch (2026-07-13). All levers inert-or-worse:
- **TV**: peak 0.06–0.08 (T1 max 37.79 @ TV0.08), falls off both sides.
- **kspace_high_weight**: peak 2.2; worse at 1.0 / 1.5 / 3.5.
- **kspace_center_weight**: inert 0.3→2.0 (center band already densely sampled → no gain).
- **capacity** (forced densify to 900k / 1.1M Gaussians): **MONOTONICALLY WORSE**
  (T1 37.75→37.50→37.39) ⇒ **data-limited, not capacity-limited** (over-fits undersampling).
- shared_warmup 500→700 inert (bit-identical to baseline); lambda_dtv / curriculum / max_radius inert-to-neutral.

**Conclusion:** T1 stuck ~0.2 dB below the PSNR-38 target = **AF=12 information limit** (92 % of
k-space unmeasured). Residual on both contrasts is **incoherent grain / detail-loss, NOT coherent
wrap-around ghosts** → classical aliasing is essentially eliminated (background clean); the 38 dB
line simply sits ~0.2 dB above the T1 data ceiling. Breaking it needs changing a PILLAR, not tuning:
lower AF (project note: AF12→AF6 ≈ MC-GS +1.3/+1.5 dB) or a learned/generative prior.

## 9. Harness notes
- `arc/cycle.py` (this repo, uncommitted edits): added CLI `--kspace_center/--kspace_mid/
  --split_quantile/--scale_threshold/--use_cloning/--save_volumes`; densify MASTER switch
  `adaptive_control.enable=True` auto-set when any capacity knob is passed (else densify is a no-op).
- Figures (jet error maps, range [0,0.1] per global convention): `scripts/fig_recon_view.py`,
  `fig_dealias.py`, `fig_alias_zoom.py`, `fig_compare_zoom.py`. Volume npz saved by `--save_volumes`
  (recT1/gtT1/zfT1/recFL/gtFL/zfFL) under `results_arc/<exp>_volumes.npz`.
- Metrics TSV: `results_arc/results_grain.tsv` (col `plain_*`=T1, `contrast_*`=FLAIR).

## 10. Undersampling mask — how it is built (verified 2026-07-15)
- **Type:** 2D **variable-density Poisson-disc** on the PE plane, **broadcast along the fully-sampled
  readout axis** (separable → prospectively-Cartesian-realizable). Same mask for T1 and FLAIR.
- **Generator:** MATLAB **`vdPoisMex`** (Lustig ESPIRiT toolbox), `/data0/congcong/code/ESPIRiT/utils/`.
- **Command (AF=12)** from `gen_vdpois_masks.m`:
  `mask = vdPoisMex(220,256,220,256, 3,4, 24, 0, 1)` → AF = accelx·accely = 3·4 = 12.
  Signature `(sx,sy,fovx,fovy,accelx,accely,calib,ellipse,pp)`: PE 220×256, **accelx=3 accely=4**,
  **calib=24** (ACS), **ellipse=0**, **pp=1** (linear variable density). Change AF: `accelx=AF/4`.
- **Algorithm:** VD radius `R=(y²+z²)^(pp/2)` → sparser at edges; central calib forced sampled;
  bisection to target accel; Bridson Poisson-disc (blue-noise) rejection sampling.
- **Verified on file** `mask3d_vdpois_af12_transverse.mat`: (256,220,256), 8.45% samp, **net AF 11.83**
  (<12 because calib=24 ACS added back), axis2=100% identical slabs (readout broadcast), radial density
  0.98→0.04, central 16×16=100% ACS, blue-noise NN 2.54 vs random 1.85.
- Helper: `scripts/make_mask.py --af 12` (prints MATLAB cmd) / `--verify`. Full detail → project `memory.md`.

## 11. CSM — how it is estimated (verified 2026-07-15)
- **Tool:** BART ESPIRiT **`ecalib`** (amax `TOOLBOX_PATH=/home/congcongliu/src/bart`). **Per contrast**
  (T1 CSM ≠ FLAIR CSM, max|diff|=1.59). Same CSM defines BOTH forward operator AND GT.
- **Command:** `bart ecalib -t 0.001 -c 0.0 -r 24 -m 1 <acs_kspace> <csm_out>`
  → `-t 0.001` smoothest eigen-threshold; `-c 0.0` no crop → **SoS≡1 whole FOV incl. air**;
  `-r 24` calib region; `-m 1` one map. Output `.mat` key `map`, complex (256,220,256,12).
- **Input ACS** = readout axis full + central **32×32** in transverse PE plane = the fully-sampled center
  of the full `ksp` (data is `presampled:false`, nonzero_frac=1.0 → retrospective undersampling). ACS is
  the fully-sampled low-freq center; CSM smooth→low-freq→small center suffices; **independent of mask →
  reusable across AF**.
- **Verified on files** `csm_espirit_{T1,FLAIR}.mat`: complex64, 12 coils, SoS=1.000 everywhere
  (std 0.0000), per-coil |S|∈[0,0.99], smooth. Data `.mat` also bundles a `map` CSM but winner uses these
  **external ESPIRiT** ones.
- **Usage** (`dataset.py`): forward `M·F3c(S·x)`; GT `Σ_c conj(S_c)·ifft3c(y_full_c)` (:313–315).
  Helper: `scripts/estimate_csm.py --recipe`/`--verify`. NOTE: ecalib `-r 24` ≠ mask `calib 24` (different roles).
  Full detail → project `memory.md`.
