# Project Memory — MC-GS winner-v2 (AF=12 dual-contrast T1+FLAIR 3D recon)

> Source of truth: `configs/winner_v2.yaml` + the `--tv_eps` flag recorded in §0. This file freezes
> the WINNER method and the autoresearch conclusions so they are not re-derived or re-tuned.
> Last verified: 2026-08-01 (working point moved to edge-TV + Charbonnier ε=0.04; §14, §15).

## 0. OUR METHOD (user decision 2026-07-31) = **MC-GS edge-TV + Charbonnier ε=0.04**

> Supersedes the 2026-07-24 pick (same method, ε=0). Everything below about edge-TV still holds;
> the only change is the SHAPE of the TV penalty. See §14 for why, and for the metrics that decided it.

**Frozen config:** edge-TV `κ_q 0.85 / λ_geo 0.10 / w_floor 0.1` **+ `--tv_eps 0.04`**, 4000 iters,
exact-bbox voxelizer (§15). Run `tvs4k_e040` → `results_arc/tvs4k_e040_volumes.npz`.
Reproduce: `... arc/cycle.py --exp tvs4k_e040 --gpu N --iters 4000 --kspace_high 2.2 --warmup 500
--max_radius 20 --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.1 --lambda_geo 0.10 --tv_eps 0.04
--base_cfg configs/winner_v2_edgeTV.yaml --save_volumes`
**Result (whole@max):** T1 PSNR **38.379** / SSIM **0.9103** / NMSE 0.02075; FLAIR PSNR **38.221** /
SSIM **0.9137** / NMSE 0.00856. Against the ε=0 pick: staircase index 8.58 → **2.04** (GT = 1.00),
T1 −0.092 dB, **FLAIR +0.035 dB**, **SSIM +0.011 on both contrasts**, and the reconstruction stays
*cleaner than the ground truth* (grain 0.944 of GT) — i.e. the sheet artefact is removed without
handing the noise back. Final figures `results_arc/{FINAL_recon_edgeTV_eps040,
FINAL_staircase_before_after}.png`.

**Full eps series at 4000 iters** (staircase S / axis-bias A / grain N, all relative to GT = 1.00;
figure `results_arc/FINAL_eps_series_4k.png`, raw numbers in the session scratchpad's
`eps_series_4k.json`). eps is a MONOTONE knob on S — it does not saturate — but three things turn
over past 0.04, which is what fixes the working point:

| eps | S | A | N | T1 PSNR | FLAIR PSNR | T1 SSIM | FLAIR SSIM | sharp |
|---|---|---|---|---|---|---|---|---|
| 0 | 8.580 | 1.241 | 0.799 | 38.471 | 38.186 | 0.8994 | 0.9102 | 0.768 |
| 0.015 | 4.126 | 1.198 | 0.851 | 38.441 | 38.195 | 0.9104 | 0.9137 | 0.781 |
| 0.025 | 2.902 | 1.181 | 0.892 | 38.419 | 38.211 | 0.9104 | **0.9138** | 0.796 |
| **0.04** | 2.037 | 1.166 | 0.944 | 38.379 | **38.221** | 0.9103 | 0.9137 | 0.819 |
| 0.06 | **1.516** | **1.153** | **1.000** | 38.323 | 38.209 | 0.9099 | 0.9126 | 0.846 |

**eps=0.06 reaches N = 1.000 exactly** — grain equal to the ground truth, i.e. the edge of the
territory the user rejected in the low-lambda arm — and FLAIR PSNR (peaks at 0.04), T1 SSIM and
FLAIR SSIM all turn down there. 0.04 keeps a 5.6% cleanliness margin under GT while still cutting
the staircase 4.2x, so it is the balanced point; **0.06 is a legitimate alternative if a lower
staircase is wanted and grain at exactly GT level is acceptable — it is further along the same
trade, not worse.** Do not push eps past 0.06 (at 1500 iters eps=0.08 already gives N 1.060).

**Why edge-TV at all (user decision 2026-07-24, still the governing criterion):** chosen on the
SSIM / whole-slice perceptual criterion, not on PSNR. Confirmed again on 2026-07-31: shown four
working points spanning the texture range (iso-TV λ0.03/0.04/0.06 vs edge-TV λ0.10), the user
preferred edge-TV's look and rejected the low-λ arm as "worse", even though low λ matches the
ground-truth texture statistics. **Judge by the whole slice, not by texture-matching metrics, and
do not re-propose lowering λ on texture-matching grounds.** "Do not consider the iso-TV winner for now."
- **edge-TV method** = MC-GS + **edge-weighted TV** (`w=clamp(1/(1+(|∇u|_sg/κ)²),w_floor,1)`, detached
  Perona-Malik, per-axis, within-contrast) instead of isotropic TV. Best cell **κ_q0.85 / λ_geo0.10 /
  w_floor0.1**, 4000 iters. `edge_tv` config-gated (`configs/winner_v2_edgeTV.yaml`, branch
  `edge-preserving-tv`, PR #1). Run `edgeTV_best` → `results_arc/edgeTV_best_volumes.npz`.
  Reproduce: `... arc/cycle.py --exp edgeTV_best --iters 4000 --kspace_high 2.2 --warmup 500 --max_radius 20
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.1 --lambda_geo 0.10 --base_cfg configs/winner_v2_edgeTV.yaml --save_volumes`
- **Metrics (same-gauge, brain, @max):** T1 PSNR 33.90 / **SSIM 0.898** / NMSE 0.0137 / edge-NMSE 0.0150;
  FLAIR 33.31 / **SSIM 0.909** / NMSE 0.0059 / edge-NMSE 0.0063. (⚠️ §7 called edge-TV a "FAIL" only against
  the strict *edge-NMSE-down* bar; on the SSIM axis it is the preferred variant — the user's chosen criterion.)
- **Baselines it beats (same AF=12 data/mask/CSM, same gauge, robust to N4-deshading):** conventional
  **CS-SENSE (TV)** — T1 32.80/0.871/0.0176, FLAIR 33.11/0.886/0.0062 (grainier, all metrics worse); every
  MC-GS variant ≫ CS-SENSE. Figs `results_arc/three_way_cssense_mcgs_edgeTV.png` (+ `_N4` version).
- **PSNR/NMSE-optimal ALTERNATIVE (set aside per user) = iso-TV winner `reg_aligned_4k`** — details below.

**iso-TV winner `reg_aligned_4k`** (PSNR/NMSE-optimal; ALIGNED AF=12 mask, 4000 iters):
- **Algorithm:** MC-GS (`JointTrainer`+`JointGaussianModel3D`): ONE shared 3D-Gaussian geometry (μ,Σ=R(q)diag(s²)Rᵀ)
  + per-primitive **dual complex density** α_plain→T1 / α_contrast→FLAIR. Forward `A_m=M_m·F3c·S_m`; scoring =
  hard-DC `k_dc=m·y+(1−m)·F3c(S·x)` then conj-SENSE combine `x=Σ_c ifft3c(k_dc)·conj(S_c)`. Warm-start 576902
  Gaussians, **densification OFF**.
- **Key params:** `lambda_geo(TV)=0.06`, `kspace_freq_adaptive` (center_frac 0.05 / center_w 0.3 / mid 1.0 /
  **high 2.2**), loss l2, **4000 iters**, warmup 500, seed 42, max_radius 20, init_scale 0.008,
  lr {pos 3e-4, scale 2e-3, rot 3e-4, density 1.5e-2}, lr_sched exp γ0.9995.
- **Data:** AF=12 **ALIGNED** vd-Poisson mask (net 11.83, 8.45% PE, readout fully sampled), ESPIRiT **1-map** CSM
  (`ecalib -t0.001 -c0.0 -r24 -m1`, SoS≡1), norm P99.9. Config `configs/winner_v2_aligned.yaml`.
- **Result @4000 (@max gauge):** T1 PSNR **38.512** / SSIM **0.8936** / NMSE **0.0201**; FLAIR PSNR **38.291** /
  SSIM **0.9017** / NMSE **0.00842**. Volumes `results_arc/reg_aligned_4k_volumes.npz`; fig
  `results_arc/FINAL_recon_winner_reg_aligned_4k.png`.
- **Reproduce:** `CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python arc/cycle.py --exp reg_aligned_4k --gpu N --iters 4000 --lambda_geo 0.06 --kspace_high 2.2 --warmup 500 --max_radius 20 --base_cfg configs/winner_v2_aligned.yaml --save_volumes`
- **Everything tested this session that does NOT beat it (do not re-try):** DC-centering (project memory §6),
  N4 error-map de-shading (§6), **edge-preserving TV / edge_tv (§7 — reshape, WORSE on edge-NMSE)**, 2-map CSM
  (§4/§12), **frozen MAISI VAE as a prior (§8 — INADMISSIBLE: the ORACLE VAE(GT) is 49%/89% worse than MC-GS
  in the null space, the only region hard-DC lets a prior touch; best oracle blend ≤0.38 dB, achievable
  ≤0.19 dB)**. Remaining residual = AF=12 unsampled high-freq (physics). Levers left: a **measurement-supervised
  learned** recon (MAISI only as data-engine, never at inference), or lower AF.

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

## 12. Is 2-map CSM / recalibration a lever? — NO (verified 2026-07-17/21)
- **2-map soft-SENSE ≈ 0 gain.** `ecalib -m2` on DC-centered ksp; coil-energy captured map0 vs span(map0,map1),
  brain: default `-c0.8` T1/FLAIR **+0.00 pt** (2nd map cropped); uncropped `-c0` **T1 +0.51 / FLAIR +0.18 pt**.
  2nd map is noise + faint edge ring (λ2≈0.24, 0.1–0.6% voxels λ2>0.5), not wrapped anatomy. **Opposite of
  cheng_data** (limited FOV → `-m2` recovers ~11%); huangyuwei full-FOV → 1-map suffices.
- **Shipped 1-map CSM already well-calibrated:** captures T1 **94.8%** / FLAIR **98.4%** ≈ fresh (96.1/98.6).
  Raw ksp DC offset **+22 px (axis1)** but `ecalib` auto-detects the calib region → no harm. Recalibration
  ≤1.3 pt (T1) / ~0 (FLAIR) → not worth a rerun.
- **Actual 2-map recon confirms (decisive):** AF≈12 CG-SENSE (readout-decoupled), 1-map vs 2-map primary,
  Tikhonov-λ sweep. 2-map only "wins" at under-reg λ=0.001 (noise-regime **implicit-smoothing artifact**,
  1-map SSIM there 0.44); at best λ=0.01 **1-map 26.82 dB > 2-map 24.91 dB**, heavier reg 1-map ≥ 2-map,
  overall-best recon = **1-map**. ⚠️ single-λ 2-map compare misleads → **sweep reg**. GS = strong prior
  (≈ good reg) ⇒ 2-map won't help GS.
- **Conclusion: CSM side (2-map + recalibration) is NOT a lever** (consistent with §8 + project `memory.md`
  §3: bottleneck = prior-limited unsampled high-freq → learned prior). Figs `results_arc/{csm_1map_vs_2map_T1_FLAIR,
  recon_1map_vs_2map_T1_frontier,recon_1map_vs_2map_T1_lam0.01}.png`. Do NOT re-try 2-map/recalibration.

## 13. Spatial operations / registration — OUT OF SCOPE (user decision, 2026-07-21)
- **User: do NOT bring T1↔FLAIR spatial registration into the project** (stated twice). Recorded so it isn't
  silently re-enabled.
- **ARC frontier HAS an inter-contrast registration module; MC-GS winner does NOT.** ARC §8 `deformation`
  (`paired_arc_gs/modules/deformation.py`): `T_m(r)=R_m·r+t_m+u_m(r)` (rigid+nonrigid) on the shared Gaussian
  centers, `reference_index=0` pins T1, FLAIR deforms relative = registration. `arc/cycle_arc.py` had it
  `enable:True` + ran `s3_deformation` 200 iters → the **P1/P3 experiments ran with it ON**.
- **But estimated deform ≈ identity** (log `def` reg 1e-7–1e-8 → nonrigid ∇u≈0; T1/FLAIR co-acquired, already
  aligned; `u_max=0.05` caps it) → **P1/P3 negative results NOT confounded**. Caveat: `def` reg can't see a
  rigid translation `t_m`, but co-aligned data → `t_m≈0` too.
- **Rule:** MC-GS winner (`JointTrainer`) has NO deformation → main line = clean zero-registration. **Future
  ARC runs MUST set `model.deformation.enable=False`**; do not pursue T1-FLAIR registration / spatial alignment.

## 14. TV penalty SHAPE — the sheet/staircase artefact and the `--tv_eps` fix (2026-07-31)

**Symptom (user, on the final figure):** "sheet-like distortions and pixel discontinuities".
**Verified in the volume, not in the figure:** exact 6x nearest upscale with no resampling anywhere,
`results_arc/zoom6x_GT_vs_edgeTV.png` — flat plateaus separated by abrupt steps, jagged staircase edges
along the ventricle boundary, against uniform fine grain in the GT.

**Root cause = the shape of the penalty, not its strength.** `JointTrainer._tv_loss` and
`losses/edge_tv.py` were both **anisotropic L1** (`|gx|+|gy|+|gz|`). L1's sub-gradient has magnitude 1
however small the gradient is, so flattening a whole region to exactly zero gradient always pays →
piecewise-constant plateaus (the "sheets") with steps between them (the "pixel discontinuities"). The
per-axis form additionally charges a diagonal gradient **sqrt(3)x** more than an axis-aligned one of
equal magnitude (measured 1.732 in `tests/test_tv_charbonnier.py`), which is why the sheets come out
axis-aligned. This is why tuning lambda could never fix it: lambda only slides along one trade-off
(staircase <-> grain), and both ends were rejected.

**Metrics that decided it** (white matter only, each normalised so GT = 1.00; the eroded WM mask is
`gt in [P55,P90] of brain` then a sigma=2 opening):
- `S` **staircase / sheet index** = P(|grad| < 0.25 * median|grad_GT|) — the plateau mass
- `A` **axis bias** = mean|grad_diagonal| / mean|grad_axis|
- `N` **grain** = rms of the (x - gaussian_filter(x, 4)) high-pass — this is the user's "clean" axis
Validated before use: they rank the existing runs exactly as the eye does (GT 1.00 / iso-TV lambda0.03
1.87 / 0.04 2.74 / 0.06 4.64 / edge-TV lambda0.10 **8.58**). ⚠️ Two other candidates were tried and
**dropped as degenerate/redundant** — an "isolated jump" fraction (0.000 for every volume including GT)
and a mid-range gradient deficit (merely tracks the distribution shift). Do not re-invent them.

**Fix (opt-in, defaults bit-identical):** `losses/edge_tv.py` gained `charbonnier(g, eps) =
sqrt(g^2+eps^2) - eps` and `tv_magnitude(vol, eps, vector)`; `edge_preserving_tv` takes the same two
arguments; CLI `--tv_eps` / `--tv_vector` in `arc/cycle.py` (config keys `loss.tv_eps`,
`loss.tv_vector`). With eps > 0 the slope near zero is g/sqrt(g^2+eps^2) instead of sign(g), so
sub-eps gradients stop being worth removing while |g| >> eps still costs |g| and edges survive.
The vector form is normalised by **3/2** so lambda transfers between the two forms (sum_a E|g_a| =
3*sigma*sqrt(2/pi) vs E||g||_2 = 2*sigma*sqrt(2/pi)) — without it the arms would be compared at
different effective strengths, the same trap that once manufactured a fake multi-contrast gain (§ the
CRISP vectorial-TV note). Tests: `tests/test_tv_charbonnier.py`, 6 cases incl. bit-identical defaults.

**eps is a genuinely NEW lever, not another way to slide along the old curve.** In a 1500-iter sweep the
eps arm lands **below the frontier reachable by lambda and kappa_q together** (their full 3x3 grid at the
same 1500 iters is in `results_arc/etv_kq{0.60,0.75,0.85}_l{0.06,0.10,0.15}`): at matched grain N≈0.87
lambda can only reach S≈6.1 while eps=0.015 reaches 3.33; at N≈0.96, S 3.4 vs 1.85. Figure
`results_arc/tv_shape_sweep.png`. The 1500-iter ranking held at 4000. **The vector norm is NOT the
lever** — it raised S to 6.635 (||g||_2 is equally non-differentiable at 0 and flattens the whole
gradient vector at once) and only moved A from 1.211 to 1.192. Chosen working point and the full eps
series are in §0.

## 15. Voxelizer — the tiled render must not depend on `tile_size` (fixed 2026-07-30)

`TileVoxelizer`'s bounding box used an isotropic `3*max(scales*half_shape)` **clamped to `max_radius`**,
while the CUDA kernel applies only `mahal <= 9` and no radius test of its own. Any gaussian whose 3-sigma
support exceeded its bbox was therefore truncated at the edge of whichever **tiles** the bbox happened to
touch — so the rendered volume depended on `tile_size`, which is a pure compute-partitioning parameter.
Two independent ways it fired, on the shipped 576 902-gaussian checkpoint:
- **max_radius clamp**: 26 772 gaussians (4.64%) clamped, 3-sigma reaching a median 25.2 and up to 206 voxels
- **anisotropic under-coverage**: the isotropic bound falls short on the long axes when the largest scale
  sits on a short one; `half_shape = (128,110,128)` → shortfall up to 128/110 = 1.1636
Together **59 458 gaussians (10.31%)** had support outside their bbox; rel-L2 0.33% in brain, up to 7.3%
of the GT peak locally.

**Fix:** `_bbox_radius_vox()` = the exact per-axis half-extent `3*sqrt(diag(R diag(s^2) R^T))*half_shape`,
never clamped; `build_tile_pairs` accepts a per-axis `(N,3)` radius. `max_radius` is still accepted for
config compatibility but no longer truncates. **Cost is negative**: tile-gaussian pairs 18 836 184 →
9 673 491 (51%), training 1.21 → 1.39 it/s. Verified: `tests/test_tile_voxelizer_invariance.py` (fails
before the fix by 6.65% of peak, passes after; also checks against an untiled brute-force reference), and
at full scale the render is now **bit-identical** across tile_size 4/8/16 on both contrasts.
**PSNR-neutral** (|delta| <= 0.007 dB on both edge-TV and iso-TV), so all earlier numbers stand.
Not touched: `paired_gs/gaussian/voxelizer.py`, the non-tiled Voxelizer, has its own radius logic and was
never audited — the main line does not use it.
