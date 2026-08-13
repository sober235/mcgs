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

> ✅ **§8b's "not coherent ghosts" claim was RE-TESTED and NOT OVERTURNED (2026-08-12, §16).** A user
> reported "a lot of aliasing artefacts" on `results_arc/LOOK_T1_axial.png`, which would have overturned
> this. Four pre-registered propositions were adjudicated with calibrated metrics: the outer-rim band,
> the ventricle streaks and background ghosts are all **false**; the white-matter speckle is **undecidable**
> and its available evidence leans toward "it IS residual incoherent aliasing" — which is what §8b says.
> The claim stands. Details, verdicts and their calibration stakes → §16.

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

> ⚠️ **CORRECTION (2026-08-01, root-cause follow-up).** The `S` index below is **mostly measuring
> global smoothing, not staircasing** — it uses an absolute threshold, so an image whose whole
> gradient distribution is shifted down scores high automatically. Re-measured with a self-normalised
> threshold, the shape-only excess is **+14%**, not the +758% that "S = 8.58" suggests, and there is
> **no atom of exactly-zero gradients** (P(|g|<t)/t rises only 1.9x over 4.5 decades; a true atom
> would rise 10x per decade). A sum of smooth Gaussians cannot realise exactly-constant plateaus, so
> "L1 pins gradients to zero" is right in direction and wrong in mechanism. The S numbers below are
> still valid as a *ranking* — they order the runs the way the eye does — but must not be read as
> "N times more plateau voxels". **The better index is `selectivity` (see the end of this section).**

**Metrics that decided it** (white matter only, each normalised so GT = 1.00; the eroded WM mask is
`gt in [P55,P90] of brain` then a sigma=2 opening — ⚠️ that prose does NOT determine the code, see the
exact definition in §14c, and do not re-derive it from this sentence):
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

### 14b. Root cause, verified rather than argued (2026-08-01)

The account in §14 was a mechanism *story* fitted to the eps sweep. This closes it with controls.

**The measurement that separates the candidates.** For each reconstruction take the q-quantile of
the in-white-matter per-axis gradient magnitude and divide by the ground truth's q-quantile at the
same q. The SHAPE of that curve identifies the mechanism with no further modelling:
flat and below 1 = every gradient attenuated alike = a low-pass (blur); rising and below 1 = small
gradients cut harder than large ones = selective suppression, i.e. flat tissue interiors with the
boundaries kept, which is exactly the sheet look; above 1 = under-regularised roughness.
Define **`selectivity` = ratio at q=0.999 divided by ratio at q=0.10**.

| run | q0.10 | q0.50 | q0.90 | q0.999 | selectivity | T1 PSNR |
|---|---|---|---|---|---|---|
| ground truth | 1.000 | 1.000 | 1.000 | 1.000 | 1.00 | — |
| **zero-fill** (pure low-pass control) | 0.503 | 0.502 | 0.478 | 0.490 | **0.97** | — |
| **lambda=0** (no TV at all) | 1.666 | 1.736 | 1.916 | 1.997 | 1.20 | 36.315 |
| lambda0.10 eps=0 | 0.394 | 0.455 | 0.565 | 0.825 | **2.09** | 38.471 |
| lambda0.10 eps=0.015 | 0.576 | 0.599 | 0.653 | 0.838 | 1.46 | 38.441 |
| lambda0.10 eps=0.04 | 0.759 | 0.770 | 0.803 | 0.898 | 1.18 | 38.379 |
| lambda0.10 eps=0.06 | 0.842 | 0.862 | 0.894 | 0.953 | 1.13 | 38.323 |
| lambda0.10 eps=0.15 | 1.086 | 1.112 | 1.161 | 1.153 | 1.06 | 37.978 |

**Both readings are anchored by a control, not asserted.** Zero-fill is literally a k-space
truncation and comes out flat at 0.97 — so a rising curve cannot be blur. **lambda=0 flips the whole
curve above 1** (0.39-0.83 becomes 1.67-2.00): with no TV there is no selective suppression at all,
the image is simply rougher than the GT everywhere. **That refutes the Gaussian representation and
the sampling mask as sources of the artefact** — it is the regulariser.

**The mechanism is two layers, and only the second is a defect.** (1) By design the edge weight
`w = clamp(1/(1+(|grad|/kappa)^2), w_floor, 1)` penalises flat regions up to `1/w_floor = 10x` more
than edges. (2) L1's marginal cost stays at 1 however small the gradient is, so layer (1) presses
all the way down and never eases off. Charbonnier removes only (2) — the marginal cost becomes
`g/sqrt(g^2+eps^2)`, which fades as g shrinks — and the selectivity falls monotonically
2.09 -> 1.46 -> 1.18 -> 1.13 -> 1.06 across the eps series, with the whole curve lifting toward 1.

**eps=0.15 is past the useful window** (selectivity 1.06 but T1 PSNR 37.978, under 38, and the
ratios cross above 1). Confirms 0.04-0.06 as the operating range independently of the earlier
S/N argument. Figure `results_arc/RC_selective_suppression.png`; controls are runs `rc_lam000`
and `rc_eps015x` (1500 iters); scripts in session c6621154's scratchpad
(`rc_zero_mass.py`, `rc_figure.py`).

### 14c. The EXACT selectivity definition — the prose above is not reproducible (recorded 2026-08-12)

⚠️ **§14/§14b describe this measurement in prose that does NOT determine the code.** A later re-implementation
that followed the prose reproduced the OPPOSITE of the published result — it read `rc_lam000` at **0.726 with
a FALLING curve** where the published control is **1.20 with a RISING one**, i.e. it would have overturned
§14b's attribution of the artefact to the regulariser on the strength of an implementation difference and no
new evidence. The producing script was recovered from session c6621154's scratchpad (`rc_figure.py` — it is
the one that prints `selectivity {t:.3f}`) and the definitions transcribed verbatim. **Use these, not the prose:**

```python
gt    = np.abs(gtT1).astype(np.float64)
brain = gt > 0.05 * gt.max()                       # NOT 0.15*P99.9, no fill-holes, no largest-CC
lo, hi = np.quantile(gt[brain], [0.55, 0.90])
m     = brain & (gt > lo) & (gt < hi)              # STRICT inequalities
wm    = m & (gaussian_filter(m.astype(np.float32), 2.0) > 0.95)   # the "sigma=2 opening": a STRONG
                                                   # erosion keeping only deep interiors. The outer
                                                   # `m &` is NOT redundant — sigma=2's central weight
                                                   # is ~0.008, so a voxel outside the band can clear
                                                   # 0.95 on its neighbours alone (measured: 439 of
                                                   # 151512 WM voxels leak in without it).
def per_axis(v):
    v = v * ((v[brain] * gt[brain]).sum() / (v[brain] ** 2).sum())   # <-- LEAST-SQUARES AMPLITUDE
    out = []                                                         #     PROJECTION onto GT. Absent
    for ax in range(3):                                              #     from §14b's prose entirely,
        s0, s1 = [slice(None)]*3, [slice(None)]*3                    #     and it is the DOMINANT term
        s0[ax], s1[ax] = slice(None,-1), slice(1,None)               #     for rc_lam000, whose global
        both = wm[tuple(s0)] & wm[tuple(s1)]                         #     scale differs. No-op on GT.
        out.append(np.abs(v[tuple(s1)] - v[tuple(s0)])[both])        # BOTH endpoints must be in WM
    return np.concatenate(out)
Q   = [0.05,0.10,0.20,0.30,0.50,0.70,0.80,0.90,0.95,0.99,0.999]
r   = np.quantile(per_axis(v), Q) / np.quantile(per_axis(gt), Q)
selectivity = r[-1] / r[1]                          # ratio@q0.999 / ratio@q0.10
```

**Now reproduced in committed code** — `diag_artefact/volumes.py` (`brain_mask`/`wm_mask`) and
`diag_artefact/metrics.py` (`gradient_ratio_curve`/`selectivity`) on branch `diag-t1-artefact`, verified to
match the published table at EVERY quantile to three decimals:

| run | q0.10 | q0.50 | q0.90 | q0.999 | selectivity |
|---|---|---|---|---|---|
| zero-fill | 0.503 | 0.502 | 0.478 | 0.490 | 0.973 (published 0.97, flat) |
| rc_lam000 | 1.666 | 1.736 | 1.916 | 1.997 | 1.199 (published 1.20, above 1) |
| tvs4k_e040 | 0.759 | 0.770 | 0.803 | 0.898 | 1.183 (published 1.18) |

⚠️ **GAUGE WARNING, new and not in §14b.** `selectivity`'s zero-point MOVES between acquisitions: the
zero-fill anchor reads 0.9728 on the AF=12 shipped volume but **0.8182** on AF=2 volumes (a second AF=12
draw reads 0.9945, so it is acquisition-specific, not noise). **Never compare raw selectivity across
different sampling rates** — normalise each arm by its own zero-fill anchor first. This is memory.md §3's
metric-gauge trap in a new place, and it flipped a control's conclusion when missed.

---

## 16. The T1 artefact report, adjudicated — §8b stands (2026-08-12)

> User, on `results_arc/LOOK_T1_axial.png`: "a lot of aliasing artefacts, plus the kind of patchy artefact
> TV produces". Rather than answer from memory, four **pre-registered** propositions were adjudicated with
> calibrated metrics. Spec `docs/superpowers/specs/2026-08-07-t1-artefact-diagnosis-design.md`; plan
> `docs/superpowers/plans/2026-08-07-t1-artefact-diagnosis.md`; code `diag_artefact/` + `scripts/`
> (branch `diag-t1-artefact`, 25 commits, 77 tests). Records: `results_arc/diag_artefact/verdicts.json`
> (+ `_FLAIR`), `adjudication.png`, `run_manifest.json`, `nonvacuity_report.json`, `m0_provenance.json`.

**Verdicts.** Each row carries the implemented verdict, the spec's literal reading, and the number that
decided it. FLAIR replicates all four independently.

| # | proposition | implemented | spec-literal | decided by |
|---|---|---|---|---|
| P1 | WM speckle = residual incoherent aliasing | insufficient | false (**NOT supported**) | its own stake fails: zf's pure aliasing ALSO escapes the band |
| P2 | outer-rim / skull band = aliasing | **false** | false (supported) | ncc(err_ctrlB, err_shipped) = 0.398 vs valid null 0.094, p=0.000 |
| P3 | ventricle streaks = displaced structure | insufficient | false (supported) | coherent displacement ≤ **1.6%** of a detectable half-voxel shift |
| P4 | background contains recon-introduced ghosts | insufficient | false (supported) | air energy is **below** GT's at every threshold (0.888→0.483) |

- **P2 is the one firm positive finding**, and its content is mechanistic: change the mask INSTANCE and the
  error pattern barely moves (shipped 38.379 dB / ctrlB 38.362 dB), so the rim band does not follow the
  sampling pattern and is not aliasing.
- **P1's spec-literal "false" is marked `supported: false` in the JSON and must not be cited.** It derives
  from the very clause its stake invalidated, and the other conjunct points the other way
  (`ncc_rec_zf` 0.452 vs null 0.005 = the residual IS same-family with zf's aliasing).
- **§8b is NOT overturned.** Nothing found coherent wrap-around ghosts; the evidence on the one open
  proposition leans toward §8b's own reading.

### 16a. TRAP — a fully-sampled control is VACUOUS under hard-DC. Never run one.

Scoring is `k_dc = m*y + (1-m)*k_model`. With an **all-ones mask `(1-m) = 0`**, so the model's output is
multiplied by zero and **the scored reconstruction IS the ground truth** — measured `max|rec-gt| = 8.3e-07`,
with `zf` identical to `rec`. It scored T1 psnr **156.4**, ssim **1.0000**. The run trains normally, reports
a beautiful number, and contains no information whatsoever.

Left unnoticed, the artefact metric would have read "identical to GT" and that would have been misread as
"the patchiness vanishes at full sampling ⇒ the mask caused it" — a fabricated causal conclusion, and the
opposite of §14b's controlled attribution to the regulariser. The plan had guarded against control A
self-confirming via TV *dilution* (a second lambda arm); that guard is useless here because **both arms are
equally vacuous**.

- **Fix if you need a data-rich control: use a MILD ACCELERATION (AF=2), not full sampling**, so hard-DC
  leaves a real null space. Configs `configs/ctrlA_af2_lam{010,060}.yaml`; mask built by
  `scripts/build_ctrl_mask.py`. λ must be dilution-scaled (AF 12→2 multiplies measurements ~6x, so
  λ 0.10 → 0.60) and **both arms must agree in direction or the control is undecided**.
- **Gate every control with `scripts/check_nonvacuity.py`** — it asserts the scored recon differs materially
  from GT and **exits non-zero** otherwise. The retired all-ones run is recorded as a permanent FAIL in
  `results_arc/diag_artefact/nonvacuity_report.json`. The two retired configs
  `configs/ctrlA_fullsample_edgetv_lam{010,120}.yaml` carry a RETIRED/VACUOUS banner — do not run them.
- Note the same algebra kills a naive self-supervised split on a static model: partitioning the loss into
  Θ/Λ gives `L_Θ + L_Λ = L_Ω`, which only re-blocks DC and produces no extrapolating prior.

### 16b. The AF=2 control did NOT establish that the patchiness is data-limited

T1's attribution came out **"control A undecided"**, and this is the honest result, not a shortfall:
- the raw scalar comparison is invalid across acquisitions (the gauge warning in §14c);
- gauge-normalised, the two λ arms **disagree in direction** (+0.025 vs −0.115);
- and the gauge-normalised spread between two nominally IDENTICAL AF=12 runs is **0.031** — larger than the
  +0.025 that drove the "agreement", so the difference sits inside run-to-run variation.

What *is* readable is the curve SHAPE (§14b's own taxonomy): shipped **rising** (selective suppression),
AF=2 λ0.10 **flat** (low-pass), AF=2 λ0.60 **falling** (edge blur) — the artefact changes KIND with more
data, it does not simply shrink. FLAIR's arms do yield an attribution ("persists or worsens at AF=2") but it
is flagged `attribution_not_for_citation` because the post-hoc gauge normalisation created it and its M5
stake is only a direction proxy. **Do not cite either as evidence that more encoding fixes the patchiness.**

### 16c. Method note — twelve pre-registered bounds failed on contact with reality

Every one was handled by DOCUMENTING rather than retuning, on a standing user ruling that pre-registered
rules stay as written and their limitations get recorded. Recurring shapes, worth checking for by default:
a bound never computed against the object it constrains (a volume-fraction assertion demanding <0.25 for a
phantom that is 0.2663 by construction); a threshold above its own pipeline's measured ceiling (M0's
`top>0.99` vs 0.9893); an assertion that cannot distinguish the two readings it exists to separate (an
energy ratio from an amplitude ratio); a "curve" whose points all select the same voxels; a test flaky by
construction 45% of the time; and **a null model whose validity assumption the same report elsewhere
refutes** (a permutation null on a residual with lag-1 autocorrelation 0.19-0.23). The last one gated the
branch's only definite verdict until the final review caught it — the conclusion survived a valid null, the
gate did not.

---

## 17. Is the shared *dictionary* enough? — the response-parameterisation ladder (2026-08-13)

> Scope decision (user): restricted to the **Gaussian representation itself**. CAIPI (ky-kz lattice,
> g-factor) is acquisition-side and J-LORAKS / SPIRiT (Hankel low-rank, coil self-consistency) is a
> different reconstruction mechanism; neither touches the Gaussian parameterisation, so both were cut.
> Spec `docs/superpowers/specs/2026-08-12-shared-structure-response-design.md`; driver
> `scripts/run_fourarm_response.py`; records `results_arc/fourarm/*.json`; harness validation
> `scripts/smoke_fourarm_harness.py`.

**Why this and not capacity.** G0's representation-ceiling gate passes the 2x line on both contrasts;
forced densification to 900k/1.1M is monotonically worse (§8b); §14b's `lambda=0` control removes the
artefact from both the Gaussian representation and the sampling mask. Capacity is closed. What had never
been tested is the STRUCTURE of the sharing: `JointGaussianModel3D` shares `mu`/`Sigma` while the two
complex densities are fully independent tables — a shared dictionary with independent coefficients.

### 17a. Vocabulary, fixed against the code (the prose was ambiguous)

From `paired_arc_gs/modules/response_decoder.py`: `z_A (N, latent_dim)` and a per-contrast condition code
`e_m` are concatenated, passed through `Linear(latent+cond -> hidden) -> Tanh -> ...`, and each contrast
gets a `Linear(hidden -> 2)` head whose 2 outputs are **(real, imag) of one complex density**.
- **`decoder-24/2` = hidden_dim 24 / n_layers 2.** NOT "outputs 2 contrasts".
- **`K` = `model.anatomy.latent_dim`.**
- Each Gaussian carries `K` free reals and the two contrasts together emit **4 reals**, so the shipped
  `latent_dim = 12` is an EXPANSION, not a bottleneck. A genuine bottleneck needs **K < 4**.
- ⚠️ `K` is NOT a rank on the contrast axis. Had it been, `K=2` would be full rank for two contrasts and
  therefore identical to `independent` — two of four arms would have been the same thing and the
  resulting "no difference" would have read as "the constraint does nothing".

### 17b. Setup

576902 Gaussians warm-started from `results_arc/tvs4k_e040/tvs4k_e040/checkpoints/best.pth`, **geometry
frozen** (`training.lr.geometry = 0.0`), the **ALIGNED** mask, 2000 iters, 4 arms x 3 seeds.
Primary metric = `PairedARCTrainer.holdout_kspace_nrmse`, per contrast, **never averaged across
contrasts**. That metric reserves a sampling set that is never fit and never DC-projected, so unlike an
in-sample score it **cannot be gamed by hard data consistency** — the property that makes §16a's
vacuity trap impossible here.

**A shipped guardrail was deliberately relaxed:** `config_schema.py:187` rejects `latent_dim` outside
"recommended 8-16 (hard bounds 4-32, §6.2)", which forbids every genuine bottleneck. Relaxed to `K >= 2`
for this experiment's configs only, on three grounds: the lower bound arrived with the scaffold commit
`a01ecf1` and no ablation on `latent_dim` exists anywhere in the repo; where `docs/validation_layer1.md`
invokes §6.2 the argument is an UPPER bound ("far too small to memorize either image"); and a lower bound
of 4 removes precisely the only new information. The exemption is recorded per-run in
`validate_config_exempted`.

### 17c. Results — and the sharp line between what is established and what is not

| arm | free reals/Gaussian | T1 held-out | seed spread | FLAIR held-out | seed spread |
|---|---|---|---|---|---|
| `independent` | 4 | **0.1618** | 0.0003 | **0.1499** | 0.0000 |
| `K12` | 12 -> decoder | 0.1627 | 0.0001 | 0.1505 | 0.0005 |
| `K3` | 3 -> decoder | 0.1741 | 0.0040 | 0.1606 | 0.0031 |
| `K2` | 2 -> decoder | 0.1799 | 0.0037 | 0.1873 | 0.0085 |

Geometry freeze read **exactly 0.0** on positions, scales and rotations in **all 12 runs**, so the arms
differ only in the response parameterisation.

**ESTABLISHED — a shared response decoder neither helps nor hurts.** `K12` vs `independent` is
+0.0008 (T1) / +0.0006 (FLAIR): reproducible across seeds but of the same order as the within-arm
wobble, and matching in absolute size `docs/validation_layer1.md`'s independently obtained
0.0002-0.0032. Two independent measurements now agree. Do not re-run this comparison expecting a
different answer, and do not report it as a win for either side.

**NOT ESTABLISHED — whether a genuine bottleneck (K<4) hurts.** `K3` and `K2` read 0.011-0.037 worse,
comfortably outside the seed noise, and the tempting conclusion is "compressing the two contrasts
through a shared bottleneck loses information". **That reading does not stand yet.** At 2000 iters the
three constrained arms were still improving by 0.002-0.008 per the convergence gate — the SAME order as
their deficit — while `independent` had converged (0.0004-0.0007). The ranking is therefore
systematically biased toward the one arm that can no longer improve, and "the constraint is harmful" is
not separable from "the constraint converges more slowly". Per the spec, the remedy is more iterations
for EVERY arm; the user's decision was to report only the established result and stop, so **the
bottleneck question is left OPEN, not answered.**

### 17d. Method — the convergence gate was defective twice, in two different ways

Recorded because both defects produced confident, wrong readings before being caught, and neither was
caught by a reviewer — both were caught by looking at the raw trajectory.

1. **Endpoint differencing is not robust.** Defined as "improvement over the final 20% of iterations"
   and computed as a two-point difference, a single excursion reads as improvement: `K12` spikes to
   0.2008 at iter 1600 and returns to 0.1627 by 2000, giving 0.038 and the appearance of an arm 50x
   from convergence, while its trajectory is flat from ~iter 1000. Replaced with a median-of-halves.
2. **Anchoring the gate to the seed spread inverts it.** "Improvement < that arm's own between-seed
   spread" degenerates when the spread approaches zero: `independent`'s FLAIR spread is **0.0000**
   (three seeds identical), so the gate demanded improvement below zero and failed the most stable arm,
   while passing the noisiest one because its loose spread made the bar generous. **A stability-based
   denominator punishes stability.** Replaced, on the user's decision, with an absolute threshold of
   0.001 — chosen an order of magnitude below the smallest gap that carries a conclusion (K3's FLAIR
   +0.0106) so that residual improvement cannot flip any comparison. This is the 13th invented bound in
   this project; the preceding 12 all failed on contact with reality, so treat it as provisional.
3. A related sizing point: with `--chunk 100` over 2000 iters the trajectory has 21 points, so a
   "final 20%" window is 4 points and a median of 2 is not a median. The window was widened to 40%.

### 17e. What this closes and what it leaves

The shared-decoder rung is closed by two independent measurements. The bottleneck rung is **open** and
would need all four arms re-run at a budget where the constrained arms converge (~6000 iters, about 3
hours on 4 GPUs). Until then, no claim should be made in either direction about whether shared structure
beyond a shared dictionary carries information. The remaining model-vNext items from the prompting audit
(bounded per-scan rigid + low-frequency phase in the forward operator, local cross-fitted
shared/private/absent routing, wiring `confidence/pi` and `lambda_route` into render and loss) were never
started and are unaffected by this result.
