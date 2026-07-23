# Project memory — house conventions (mask + CSM 口径)

> The reproducible recipes for the two acquisition-side artifacts of the AF=12 T1+FLAIR
> dual-contrast recon: the undersampling **mask** and the **CSM** (coil sensitivity maps).
> Every fact below was verified against the real files on 2026-07-15 (not recited).
> Winner method / hyperparameters live in `.claude/memory.md`. Source configs: `configs/winner_v2.yaml`.

---

## 1. Undersampling mask — vd-Poisson (separable, prospectively-Cartesian-realizable)

**What it is.** A 2D **variable-density Poisson-disc** pattern on the two phase-encode (PE) axes,
**broadcast identically along the fully-sampled readout axis** (S-I) → separable → acquirable on a
real Cartesian scanner. Same mask for BOTH contrasts.

**Generator.** MATLAB **`vdPoisMex`** (Lustig ESPIRiT toolbox), on amax at
`/data0/congcong/code/ESPIRiT/utils/` (`vdPoisMex.mexa64` + `.c` source + `.m` help + `gen_vdpois_masks.m`).

**Exact generation command** (from `gen_vdpois_masks.m`, AF=12):
```matlab
mask = vdPoisMex(220, 256, 220, 256, 3, 4, 24, 0, 1);  % AF = accelx*accely = 3*4 = 12
mask = single(mask ~= 0);  save('mask_vdpois_220x256_af12.mat','mask','-v7');
```
Signature `vdPoisMex(sx, sy, fovx, fovy, accelx, accely, calib, ellipse, pp)`:
| arg | value | meaning |
|---|---|---|
| sx,sy / fovx,fovy | 220,256 | PE-plane size |
| accelx, accely | **3, 4** | per-axis accel → nominal AF 3×4 = **12** |
| calib | **24** | central fully-sampled ACS (calibration) lines |
| ellipse | 0 | no corner cut (rectangular FOV) |
| pp | **1** | polynomial order of variable density (0=uniform, 1=linear) |

**Algorithm (vdPoisMex.c).** (1) variable-density radius field `R=(y²+z²)^(pp/2)` on normalized
y,z∈[−1,1] → larger min-distance at the edges (sparser); (2) central calib region forced sampled;
(3) numerical bisection to hit target accel; (4) Bridson Poisson-disc rejection sampling
(`est_accl = 1.24²·masksum/dsum`, 1.24 = blue-noise packing constant) → blue-noise pattern.

**To change AF:** set `accelx = AF/4` (AF12→3, AF10→2.5, AF6→1.5); accely=4, calib=24, ellipse=0, pp=1 fixed.
Prebuilt masks exist: `/data0/congcong/code/ESPIRiT/utils/mask_vdpois_220x256_af{08,10,12,14,16}.mat`.
Helper: `python scripts/make_mask.py --af 12` prints the MATLAB command; `--verify <mat>` checks samp/AF/ACS.

**Shipped file** (winner): `/data0/.../3dgsVC/mask/mask3d_vdpois_af12_transverse.mat` (keys `mask`,`net_af`).
Config uses `mask_type: external`; `paired_gs/data/dataset.py::_conform_mask_shape` stacks the 2D pattern
along the readout axis and transposes to the (256,220,256) volume layout.

**Verified on the real file:**
| property | measured | meaning |
|---|---|---|
| shape / samp / net AF | (256,220,256) / **8.45%** / **11.83** | matches `net_af` key |
| separable | axis2 = **100% identical slabs** | 2D PE pattern (256×220) broadcast along readout |
| ACS | central 16×16 = **1.000**, 24×24 = 0.918 | calib=24 fully-sampled center (disc) |
| variable density | radial samp 0.98→0.44→0.20→0.10→0.06→**0.04** | dense center, sparse edge |
| blue-noise | NN dist mean **2.54** vs white-random 1.85 | Poisson-disc min-distance property |

**Why net AF 11.83 (not 12):** the calib=24 ACS is added back fully sampled, raising total samples above the
nominal 1/12. **Why reusable/realizable:** undersampling is entirely in the 2D PE plane; readout is free.

---

## 2. CSM — ESPIRiT (BART `ecalib`), per contrast, SoS≡1 full-FOV

**What it is.** Coil sensitivity maps estimated by **BART ESPIRiT `ecalib`** from a **fully-sampled ACS**
(central low-freq k-space). Estimated **separately per contrast** (T1 CSM ≠ FLAIR CSM). The SAME CSM
defines BOTH the forward operator and the ground truth, so a clean smooth CSM is essential.

**Exact recipe** (from `scripts/estimate_csm.py`; BART on amax `TOOLBOX_PATH=/home/congcongliu/src/bart`):
```bash
# ACS input = readout axis fully sampled + central 32×32 in the transverse PE plane
bart ecalib -t 0.001 -c 0.0 -r 24 -m 1  <acs_kspace>  <csm_out>
# save .mat key 'map', complex (256,220,256,12) = (axis0,axis1,axis2,coil)
```
| flag | value | meaning / why |
|---|---|---|
| -t | **0.001** | eigenvalue threshold — empirically smoothest (U-shaped vs -t) |
| -c | **0.0** | no soft crop → unit-modulus maps span whole FOV → **SoS ≡ 1 everywhere (incl. air)** |
| -r | **24** | 24×24 calibration region |
| -m | **1** | one set of ESPIRiT maps (single map, not multi-map soft-SENSE) |

**"Fully-sampled ACS" = the central, no-undersampling low-freq block of k-space.** CSM are smooth →
low-spatial-frequency → a small fully-sampled center (32×32) suffices for full-resolution maps. Because the
ACS is the fully-sampled center, it is **independent of the peripheral undersampling mask → CSM reusable
across AF** (mask changes, CSM does not). In this project the raw `ksp` is fully sampled (nonzero_frac=1.0,
`presampled:false`) so undersampling is retrospective; the ACS is just the central crop of that full data.

**Usage in pipeline** (`paired_gs/data/dataset.py`): forward `A·x = M·F3c(S·x)`; GT
`x_gt = Σ_c conj(S_c)·ifft3c(y_full_c)` (:313–315, conjugate-SENSE / matched-filter combine of fully-sampled
multicoil). SoS≡1 makes this the minimum-variance estimate. Coil axis auto-inferred (`_infer_coil_axis`)→dim0.

**Shipped files** (winner): `/data0/.../3dgsVC/csm_espirit/csm_espirit_{T1,FLAIR}.mat`, key `map`.
NOTE: the raw data `.mat` ALSO bundles a `map` CSM, but the winner points `csm_path` at these **external
ESPIRiT** files, not the bundled one.

**Verified on the real files:**
| property | T1 | FLAIR |
|---|---|---|
| shape / dtype / coils | (256,220,256,12) complex64 / 12 | same |
| SoS (∑\|S_c\|²)^½ | **1.000** everywhere, in-FOV std **0.0000** | same (the `-c 0.0` signature) |
| per-coil \|S\| range | [0, 0.991], complex (phase used) | [0, 0.989] |
| smoothness mean\|∇\| | 0.0030 | 0.0048 |
| per-contrast | **T1 ≠ FLAIR**, max\|diff\|=1.59 (separately estimated) | — |

**Two "24"s that are unrelated:** ecalib `-r 24` (CSM calibration kernel) vs vdPoisMex `calib=24`
(undersampling ACS). Same number, different roles. Reproduce/verify: `scripts/estimate_csm.py --recipe` / `--verify <csm.mat>`.

---

## 3. Why T1(plain) is harder than FLAIR(contrast) — spectrum–sampling mismatch (2026-07-16)

> In the winner recon T1's relative error is ~2.4× FLAIR's (NMSE). Root cause is a **spectrum–sampling
> mismatch**, NOT a bug / CSM / geometry issue — and it CANNOT be recovered by reallocating samples; the
> only lever is a learned prior for unsampled high-freq. Every number verified against real files
> (`reg_aligned_volumes.npz` + raw multicoil `ksp` + pre-gen masks); multicoil-gauge self-check reproduces
> §8's af12 73.8% / 89.8% at rel-err 0.00%.

**H2 (main cause) — k-space energy distribution.** T1 energy sits at higher freq. PE radius to reach:
| energy | T1 | FLAIR |
|---|---|---|
| 90% | 18.2 px | 11.4 px |
| 95% | 34.2 px | 23.1 px |

Real multicoil retained energy (‖zf‖²/‖gt‖², brain): **T1 73.8% / FLAIR 89.8%** (loses 26% vs 10%, 2.5×).
vd-Poisson is sparse at high freq ⇒ T1 loses more. FLAIR is water-suppressed ⇒ energy concentrated low-freq.

**H1 (image-domain root, mild).** FLAIR near-zero brain voxels 16.3% vs T1 10.7%, sparser (L1/L2 0.746 vs
0.805); but TV nearly equal and entropy higher, so the image-domain gap is moderate — H2 (frequency) is the
strong evidence.

**H3 (metric-gauge artifact, important).** Pipeline/TSV PSNR uses `data_range = max(gt)`. T1's peak is
higher (1.887 vs FLAIR 1.568) ⇒ inflates T1 PSNR by ~1.6 dB ⇒ makes T1 (38.51) *appear* to beat FLAIR
(38.26). Under a unified data_range **FLAIR 34.35 > T1 32.99 (leads ~1.36 dB)**, consistent with NMSE
(2.4×), SSIM, HFEN. ⚠️ Never compare contrasts with @max-PSNR; use NMSE or a fixed data_range. This
**corrects `.claude/memory.md` §8**'s "T1 crosses 38 / surpasses FLAIR" reading — T1's 38.51 value is fine,
but it does NOT actually beat FLAIR.

**Contrast-adaptive sampling = dead end (ruled out).** At fixed total budget (2×8.45%), reallocating
samples between T1/FLAIR barely moves retained energy — doubling sampling lifts T1 only +1.4 pts;
budget-fixed reallocation min 73.8→74.3 (+0.5). T1's missing energy is a far high-freq tail that even
af08 (12.65%) can't reach ⇒ no free lunch on the acquisition side ⇒ locks in the learned-prior direction.

**Figures** (`results_arc/`, jet error maps [0,0.1]): `tf_diff_H123.png` (H1/H2/H3 diagnosis),
`tf_adaptive_mask_multicoil.png` (adaptive-sampling null result, faithful multicoil gauge),
`final_recon_reg_aligned_T1_FLAIR.png` (winner, 3 views × [ZF|GT|Recon|err]). Analysis scripts live in the
session scratchpad (not committed).

---

## 4. Is 2-map (soft-SENSE) CSM or recalibration a lever? — NO (verified 2026-07-17/21)

> After the self-contained-prior dead-ends, checked whether the **CSM itself** is the bottleneck:
> (a) 1-map→2-map ESPIRiT, (b) whether the shipped 1-map CSM is well-calibrated, (c) an actual 2-map
> reconstruction. All ruled out. Measured on real files (raw multicoil `ksp` + shipped `csm_espirit_*.mat`,
> DC-centered before `ecalib`). Scripts: `scratchpad/csm2map_{prep,diag}.py`, `csm_existing_check.py`,
> `csm2map_reconCG.py`.

**4a. 2-map soft-SENSE recovers essentially nothing here.** BART `ecalib -m2` on the DC-centered raw ksp;
coil-image energy captured by map0 alone vs span(map0,map1), brain:
| | default `-c0.8` (cropped) | `-c0` (uncropped, true 2nd subspace) |
|---|---|---|
| T1 | 96.12% → 96.12% (**+0.00 pt**) | 96.55% → 97.06% (**+0.51 pt**) |
| FLAIR | 98.64% → 98.64% (**+0.00 pt**) | 98.72% → 98.91% (**+0.18 pt**) |

The 2nd ESPIRiT map is **noise-dominated** (λ2≈0.24, only 0.1–0.6% of voxels λ2>0.5; visually speckle + faint
edge ring, NOT wrapped anatomy) → the default crop correctly zeros it. **Opposite of cheng_data** (limited
FOV/wrap → `-m1` underfits ~11%, `-m2` recovers it, see auto-memory [[cheng-data-csm-bart-espirit]]).
huangyuwei is **full-FOV, no wrap → 1-map suffices**.

**4b. Shipped 1-map CSM is already well-calibrated.** Existing `csm_espirit_*.mat` captures **T1 94.8% /
FLAIR 98.4%** of brain coil energy ≈ a fresh centered calibration (96.1% / 98.6%). The raw ksp DC is offset
**+22 px on axis1** (the same offset the mask-registration fix addressed), but BART `ecalib` auto-detects the
calibration region by energy → the offset **did not hurt the CSM**. Recalibration gain ≤1.3 pt (T1) / ~0
(FLAIR) → **not worth a full rerun**. (Aside: raw-vs-centered frame gives identical capture — a k-space roll
is an image-domain phase ramp the projection magnitude cancels.)

**4c. Actually running a 2-map reconstruction confirms it (the decisive test).** AF≈12 CG-SENSE
(readout-decoupled, GPU-batched), 1-map vs 2-map soft-SENSE (primary image x1), Tikhonov-λ sweep:
| λ | 1-map PSNR / SSIM | 2-map PSNR / SSIM |
|---|---|---|
| 0.001 (under-reg, noisy) | 24.26 / 0.44 | 25.02 / 0.60 |
| **0.01 (best op. pt)** | **26.82 / 0.76** | 24.91 / 0.83 |
| 0.05 | 24.34 / 0.87 | 22.83 / 0.87 |
| 0.2 | 20.40 / 0.87 | 19.82 / 0.86 |

2-map only "wins" at λ=0.001 — a **noise-regime artifact** (its 2× unknowns act as implicit smoothing over
the under-regularized 1-map noise; note 1-map SSIM there is only 0.44). At the best operating point (λ=0.01,
overall-best recon) and every heavier setting, **1-map ≥ 2-map**; 2-map only trades PSNR for a little SSIM.
**⚠️ Lesson: a single-λ 2-map recon comparison MISLEADS — you must sweep regularization.** Since GS is a
**strong-prior (≈ well-regularized) recon**, this is exactly the regime where 2-map ≤ 1-map ⇒ **2-map would
not help GS either.**

**Conclusion:** the **CSM side (2-map + recalibration) is NOT a lever** — consistent with §3: the AF≈12
bottleneck is prior-limited *unsampled high-freq*, whose only lever is a learned/generative prior. **Do not
re-try 2-map or CSM recalibration.** Figures (`results_arc/`): `csm_1map_vs_2map_T1_FLAIR.png`
(maps / 2nd-map / λ2), `recon_1map_vs_2map_T1_frontier.png` (λ-sweep), `recon_1map_vs_2map_T1_lam0.01.png`
(best-op-pt recon).

---

## 5. Spatial operations / inter-contrast registration — OUT OF SCOPE (user decision, 2026-07-21)

> **User decision (stated twice): do NOT bring T1↔FLAIR spatial registration / any spatial-alignment
> operation into the project.** Recorded so it is not silently re-enabled.

**The ARC frontier HAS a registration module; the MC-GS winner does NOT.** Paired-ARC-GS ships the §8
`deformation` module (`paired_arc_gs/modules/deformation.py`): `T_m(r) = R_m·r + t_m + u_m(r)` (rigid +
low-freq nonrigid) applied to the shared-anatomy Gaussian centers, with `reference_index=0` pinning
T1(plain) to identity and FLAIR deforming relative to it = **inter-contrast registration**. The
`arc/cycle_arc.py` harness sets `deformation:{enable:True, rigid:True, nonrigid:True, u_max:0.05}` and runs
`s3_deformation` for 200 iters — so the **P1/P3 prior experiments ran WITH the registration machinery on**.

**But the estimated deformation was ≈ identity** → the results are not confounded. The deformation
regularizer `R_def = ‖∇u‖² + λ_J‖det J − 1‖²` stayed at **1e-7–1e-8** throughout every run → nonrigid
`∇u ≈ 0`. Reason: T1/FLAIR here are **co-acquired (same subject/session/geometry) → already voxel-aligned**,
so the optimizer correctly finds ~nothing to warp (deforming FLAIR only raises the DC loss; `u_max=0.05`
also caps it). Caveat: `R_def` does not penalize a pure rigid translation `t_m` (∇u=0), so the logs alone
can't rule out a tiny rigid shift — but on co-aligned data `t_m` should also be ≈0.

**Standing rule going forward:** the **MC-GS winner (`JointTrainer`) has NO deformation module** — the
shipped main line is clean zero-registration. Any future **ARC** run must set
**`model.deformation.enable=False`**; do NOT pursue T1↔FLAIR registration / spatial alignment.

---

## 6. Algorithm–code alignment audit of the 1-map winner — code is FAITHFUL, T1 artifact = physics (2026-07-21)

> User re-opened T1's severe artifacts as a suspected **code-vs-algorithm mismatch**. Audited the winner
> code (`paired_gs/trainers/joint_trainer.py`, `data/dataset.py`, `data/transforms.py`) line-by-line
> against the spec AND independently re-measured on real files. **Verdict: no T1-specific bug; the code
> implements the algorithm faithfully and is contrast-symmetric. T1's artifact is the §3 spectrum–sampling
> mismatch, now independently reconfirmed.** Fig `results_arc/diag_T1_artifact_rootcause.png` (3-panel).

**What was verified correct & contrast-symmetric:**
- Forward = true multicoil SENSE `pred_ksp = fft3c(vol·csm)` (`joint_trainer.py:428`), DC loss vs measured
  multicoil `kspace_undersampled_complex`, freq-weighted, masked (`_kspace_loss`). Not a combined-channel
  shortcut.
- hard-DC export (`:927-931`): `ksp_dc = pred·(1-m) + y_meas` → sampled points EXACT → 100% of residual is
  unsampled k. GT = `Σ conj(csm)·ifft3c(ksp)` (conj-SENSE). Per-contrast P99.9 normalization, self-consistent.
- `fft3c/ifft3c = fftshift∘fftn∘ifftshift` → DC at array geometric center.

**Independent re-measurement (raw multicoil ksp + ALIGNED mask):**
- **Mask is OPTIMALLY aligned to T1's DC**: T1 3D DC = `(128,132,126)`, offset `(0,+22,-2)` from center;
  per-contrast roll sweep → T1 peak retained-energy is **exactly at shipped roll 0** (re-align gain
  **+0.00 pt**). Kills the "shared mask only aligned FLAIR, not T1" hypothesis. (FLAIR best roll (-4,-1) = +0.21pt, negligible.)
- **Retained energy** (aligned mask): **T1 73.79% / FLAIR 89.28%** (reconfirms §3's 73.8/89.8).
- **Radial spectrum**: 90% energy T1@59px vs FLAIR@21px (95%: 87 vs 45) → T1 intrinsically higher-freq.
- **Recon works, residual all high-freq**: error-energy ↓**82%** vs zero-filled for BOTH (T1 NMSE
  0.1106→0.0202, FLAIR 0.0484→0.0085); recon-error radial profile shifts to high-k.

**Only code-algorithm inconsistency found — quantified NEGLIGIBLE:** the +22px(axis1) DC offset is
compensated by rolling the *mask* but NOT the frequency-adaptive loss weight — `_build_freq_weight`
(`joint_trainer.py:526-537`) centers its center/mid/high bands on the array center, not the true DC. But
only **7.3% of sampled pts** are mis-weighted and **97.1% of high-band pts still get the 2.2 boost** (a
0.2-normalized-radius shift only moves the r>0.5 shell). Same offset also forces the model to represent an
image-domain phase ramp (magnitude/TV unaffected). Both effects are **contrast-symmetric → do NOT explain
T1>FLAIR**. The clean fix (center data k-space DC once, up front) is principled but expected small &
symmetric — it does NOT touch the 73.79% physics limit (retained energy is invariant to a global k roll).
**Real lever for T1 stays: learned/generative high-freq prior, or lower AF (§3, §8b).**

**RETRAIN VERDICT (2026-07-23) — the DC-centering cleanup was IMPLEMENTED + tested; it does NOT help.**
Config-gated `center_kspace_dc` added (default-off, reversible): `dataset.py` rolls ksp so DC→array
center + rolls the external mask by the same shift; `_load_dataset` plumbs the flag;
`configs/winner_v2_centered.yaml`. Dry-run verified (DC→center, |GT| invariant 3e-7, retained energy
73.79% unchanged, phase ramp 0.71→0.20 rad/vox). Retrained `centered_4k` (GPU5, 4000 iters, same
warm-start/seed/hyperparams as `reg_aligned_4k` — ONLY diff is centering):
| | T1 comp | T1 SSIM | T1 PSNR@max | T1 NMSE | FLAIR comp | FLAIR SSIM |
|---|---|---|---|---|---|---|
| reg_aligned_4k (base) | 30.85 | 0.8936 | 38.51 | 0.0201 | 35.10 | 0.9017 |
| centered_4k | **29.77** | **0.8632** | 38.59 | 0.0198 | **34.36** | **0.8809** |
| Δ | **−1.08** | **−0.030** | +0.08 | −0.0004 | **−0.74** | **−0.021** |
Centering makes the recon **smoother**: pixel-error metrics (PSNR@max/NMSE/HFEN/mean|err| −4%) marginally
better, but **structural/composite (comp −1.08/−0.74, SSIM −0.03, sharp −0.036) REGRESS**. Error maps
visually unchanged (`results_arc/cmp_centered_vs_baseline_T1_FLAIR.png`). Two confounds, same conclusion:
(a) `kspace_center_weight=0.3` was tuned UNDER the miscentered freq-weight → centering makes 0.3 de-emphasize
the TRUE low-freq structure → SSIM drop; (b) warm-start fit to ramped data biases vs centered. But retained
energy is roll-invariant (73.79% both) so the physics ceiling is untouched → no re-tune/re-warm can win big.
**Conclusion: algorithm-code alignment audit CLOSED — code is faithful, the one cleanup (DC-centering)
empirically does not reduce T1 artifacts (slightly regresses). T1 artifact = AF=12 spectrum–sampling physics.
Do NOT re-pursue DC-centering; keep `center_kspace_dc` default-off. Real lever = learned prior / lower AF.**

**N4 bias-field "normalization" of error maps is a METHODOLOGICAL ARTIFACT, not a new recon bug (2026-07-23).**
> User asked to N4-correct recon+GT then recompute |err|; the result showed error DOUBLING with a bright
> boundary ring. Debugged with systematic-debugging: first hypothesis (independently-thresholded N4 masks
> disagree at the blurred-vs-sharp boundary → ring) was REFUTED by a minimal test (forcing recon's N4 fit to
> use GT's mask did not shrink the ring; T1 ring NMSE 0.0353→0.0375, slightly worse). Real mechanism, verified
> with matched 4-voxel ring/interior masks on both contrasts:
> 1. **Pre-existing** (= the already-established AF=12 physics artifact): boundary ring already has
>    3.2–3.9× higher NMSE than interior in the ORIGINAL (non-N4) error map — undersampling blur/Gibbs
>    concentrates at the sharpest edge in the volume (skull/scalp-air).
> 2. **N4-specific**: N4 fits recon and GT INDEPENDENTLY (two separate smooth B-spline bias fields). The
>    B-spline fit is least-constrained at the edge of its domain, so it responds differently to recon's
>    blurred boundary vs GT's sharp boundary → the two fitted fields diverge 2.7–4.0× more at the ring than
>    in the interior (mean|bias_recon/bias_gt − 1|: ring≈0.11–0.12, interior≈0.03–0.05). Dividing by two
>    DIFFERENT fields injects a genuinely new error term absent from the original comparison.
> Both effects compound multiplicatively on the same boundary ring → after N4 the ring dominates a [0,0.1]
> error-map colorbar even though nothing about the underlying reconstruction changed.
> **Takeaway: do NOT independently N4-correct recon and GT for error-map comparison — it is not a fair
> normalization for this undersampled-recon use case.** If bias-field-invariant comparison is ever needed,
> either (a) skip N4 entirely (as in all other figures this session), or (b) fit N4 ONCE (e.g. on GT) and
> apply the SAME field to both, or (c) restrict any N4-based metric to a deeply-eroded interior mask
> (≥10–15 voxels from the boundary) where independent fits actually agree (~3% mismatch, tolerable).
> Figures: `results_arc/n4_error_centered_butterfly.png` (the inflated-ring effect),
> `results_arc/n4_mask_mismatch_diag.png` (mask XOR + bias-ratio map — mask-mismatch panel is a red herring,
> bias-ratio-divergence panel is the real signal).
>
> **Fair version confirmed (option b executed):** fit N4 ONCE on GT only, apply the SAME field to recon (no
> independent recon fit) → ring NMSE inflation drops from ×1.84–1.87 (independent-fit) to **×1.15–1.20**
> (T1 0.0192→0.0231, FLAIR 0.0104→0.0120), interior barely moves (×1.07–1.08, the expected residual from
> dividing by any non-constant field). Visually, shared-field `|err|` ≈ orig `|err|` (no bright ring) —
> `results_arc/n4_shared_field_fair_deshade.png`. **Confirms: no hidden shading defect: after fair de-shading
> the error is still diffuse high-freq cortical/ventricular-boundary grain, unchanged from the non-N4 story.**

**Why the ventricle ("butterfly" CSF) region shows high error (2026-07-23) — same edge/Gibbs mechanism, not new.**
> Brain-wide (gradient-independent of any mask bug): binning all brain voxels by `|grad(GT)|` gives a clean
> MONOTONIC error increase, T1 quintile-1→5 mean|err|/peak 0.0117→0.0189, FLAIR 0.0107→0.0201 — confirms
> error scales with local edge sharpness everywhere, not just at the skull.
> Ventricle-specific: the lateral ventricles are a THIN structure (~2–4 voxels wide in cross-section at the
> butterfly slice) → there is no voxel far from the CSF-tissue wall, so "core" and "wall" are nearly the same
> population: core≈wall error (T1 ratio 0.99×, FLAIR 1.16×), both ~12–41% above generic (thick) WM/GM
> interior. `|grad(GT)|` edge map shows the ventricle wall as bright as the skull rim — same mechanism,
> different location. Fig: `results_arc/ventricle_error_diag.png`.
> ⚠️ **Methodology gotcha hit during this analysis (fixed): an initial 3D connected-component ventricle
> segmentation (fill-holes + label) merged the ventricles with peripheral sulcal CSF into one blob spanning
> nearly the WHOLE brain (z 0–197) — this gave a WRONG "core low-error, wall high-error" result (NMSE-based,
> also confounded by near-zero-GT denominator blowup). Fixed by per-slice local-percentile segmentation,
> visually verified against the raw image (contour must visibly trace the clover/butterfly shape) before
> trusting any statistic — do not trust an anatomical mask's stats without plotting its contour on the image
> first, especially for thin/small structures where 3D connectivity can silently merge unrelated regions.
