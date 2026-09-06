# Design — Self-contained multi-contrast priors for T1/FLAIR artifact removal

- **Date:** 2026-07-16
- **Status:** Draft (awaiting user review)
- **Base commit:** `d0a47c0` (branch `paired-arc-gs`)
- **Scope:** One implementation plan. Stage 2 (P2) is optional and gated on Stage 1 results.

## 1. Motivation

At AF≈12 the winner `reg_aligned` (MC-GS, aligned vd-Poisson) has **eliminated coherent aliasing** —
the residual error is diffuse, edge/texture-concentrated, no wrap/ghost. The remaining gap is:

- **Unsampled high-frequency loss** (T1 worse than FLAIR): T1's image energy sits higher in frequency;
  real multicoil retained energy is **T1 73.8% / FLAIR 89.8%** → T1's relative error is ~2.4× FLAIR's
  (NMSE). This is prior-limited: hard-DC makes the 8.5% sampled k perfect, so all residual is the 91.5%
  unsampled k.

**Disproven paths (do NOT re-tread):** hand-crafted gradient couplings (dtv/vtv/edge — only reshape
PSNR↔SSIM), contrast-adaptive sampling *reallocation* at fixed budget (energy retention insensitive,
min 73.8→74.3), densification, warm-start re-derivation. GT also carries a shared bias field (matched-
filter combine's SoS weighting); de-shading is deferred (needs a combine-gauge redefinition).

**Gauge lesson:** PSNR with `data_range=max(gt)` inflates T1 (its single-voxel peak 1.887 > FLAIR 1.568)
→ false "T1 beats FLAIR". **Unified gauge (used throughout this work):** PSNR with
`data_range = P99.9(gt_contrast)` — a fixed per-contrast value robust to the outlier peak, held constant
across all experiments so within-contrast deltas are honest; **NMSE (gauge-free) is always co-reported**
and is the arbiter for any cross-contrast statement. **Never @max-PSNR.**

## 2. Goals & Success Criteria

Improve artifact removal on **both** T1 and FLAIR, targeting two axes simultaneously:

- **A (quantitative fidelity):** recover unsampled high-frequency → higher PSNR (unified gauge) / lower
  NMSE & HFEN.
- **B (structure/perception):** higher SSIM / edge sharpness.

**The improvement must be self-contained** — priors constructed from the multi-contrast measurement +
the shared-structure model ONLY. No external data, no pretrained/generative model (MAISI etc. explicitly
out of scope for this design).

**Success:**
- A: `ARC + P1 + P3` unified-gauge PSNR ≥ MC-GS winner `reg_aligned`, OR HFEN over the genuinely-unsampled
  k-band drops significantly vs the ARC baseline.
- B: SSIM / sharpness ≥ ARC baseline AND ≥ MC-GS winner.
- Both contrasts beat ZF on ssim ∧ nmse ∧ hfen.

## 3. Architecture

**Base:** Paired-ARC-GS frontier (`paired_arc_gs/`, `PairedARCTrainer`) — the model that *is* "shared
implicit structure + specificity": shared anatomy `G_A(μ,Σ,z_A)` + response decoder `D_ψ(z_A,e_m)` +
measurement-audited change branch. (MC-GS winner is chosen as the reference target, not the base — it
lacks the shared latent `z_A`, so it doesn't match the prior philosophy.)

**Core prior = the model itself:** shared structure compresses each contrast's null space; specificity is
only expressible as a constrained function of the shared structure.

**Training objective** (added on top of the base loss):
```
L = L_DC(Θ)                    # existing hard-DC data consistency on sampled points Θ
  + λ₁ · L_freqExtrap(Λ)       # P1: self-supervised extrapolation to held-out high-freq k (Λ)
  + λ₃ · L_LLR                 # P3: cross-contrast local low-rank
  + [λ_tv · TV_iso]            # optional isotropic TV (image-quality dependent)
  (+ P2: complementary mask at dataset level, Stage 2, does not change the loss)
```
(Subscripts track the prior number: P1→λ₁, P3→λ₃. There is no λ₂ — P2 is a sampling-mask
change, not a loss term.)

At training time P1 splits each contrast's sampled k into Θ (fit) / Λ (held-out); at inference/eval the
full sampled set Ω is used for hard-DC (Λ is a training-only holdout).

## 4. P1 — Self-supervised frequency extrapolation

**Idea:** hold out a subset of the *acquired* mid/high-frequency k and force the model to predict it from
the rest — using the real acquired high-freq as a self-supervised target. This is the no-external-data
analog of a learned prior: it trains the extrapolation ability that then generalizes to the truly-
unsampled k.

**k-split per contrast m (each iteration or every K iters):**
- ACS center (low-freq) is **always kept in Θ** (guarantees DC / basic recon does not collapse).
- `Λ_m`: random `ρ ≈ 0.3` fraction sampled **only from the mid/high-frequency** part of `Ω_m`.
- `Θ_m = Ω_m \ Λ_m`.
- Re-sample `Λ` every `K ≈ 50` iters (bootstrap; prevents overfitting one split).

**Loss:**
```
L_freqExtrap = Σ_{m∈{T1,FLAIR}} ‖ M_Λm ⊙ (F₃c(csm_m · x_m) − y_m) ‖² / |Λ_m|
```
where `x_m` is the ARC-rendered contrast-m image from the shared `G_A`. **Cross-contrast boost is
automatic:** both contrasts share `G_A`, so T1's held-out extrapolation implicitly borrows FLAIR's
sampled structure.

**Schedule:** warm up `λ₁` — let Θ-DC converge (~500 iters) then ramp `λ₁` linearly, so early extrapolation
pressure doesn't disturb geometry convergence.

**Failure modes / mitigations:**
- Λ takes too much high-freq → Θ starved of high-freq, DC collapses → lock ACS in Θ, `ρ≤0.3`, draw Λ only
  from mid/high band.
- `λ₁` too large → overfits Λ, artifacts → warmup + sweep `λ₁∈[0.1,1]`.

**Direct validation signal:** held-out `Λ` NRMSE should fall over training (extrapolation ability growing).

## 5. P3 — Cross-contrast local low-rank (LLR)

**Idea:** within a small block the two contrasts share anatomy → strongly correlated (locally near-linear
intensity relation). A nuclear-norm penalty on the block Casorati matrix softly extracts this redundancy —
more complete than dtv/vtv (which only couple gradients; this constrains the full intensity relation).

**Construction:**
- Sliding blocks (`8×8×8`, stride 4, overlapping for smoothness).
- Per block `b`: flatten T1 and FLAIR block voxels into two columns → Casorati `M_b ∈ R^{512×2}`
  (**magnitude**, to avoid phase coupling).
- `L_LLR = Σ_b ‖M_b‖_*` (nuclear norm). For an N×2 matrix the nuclear norm has a **closed form** via the
  2×2 Gram matrix `M_bᵀM_b` eigenvalues (σ₁+σ₂) — fully differentiable, cheap.

**Why principled:** the nuclear norm pushes `M_b` low-rank = the two contrasts locally lie in a shared
low-dimensional subspace = "FLAIR ≈ local linear map of T1 within the block" = shared structure with
response-based specificity — exactly the model's philosophy, made explicit as a patch-level prior.

**Complementarity with the change branch (key):** LLR penalizes regions where the contrasts are
uncorrelated, but genuine differences (lesion, enhancement) must be preserved. The change branch (§7)
models exactly those measurement-audited differences → LLR tightens shared structure while the change
branch admits real differences. They do not conflict.

**Failure modes / mitigations:**
- `λ₃` too large → forces T1∝FLAIR, washes out real contrast differences → moderate `λ₃` (nuclear norm is
  soft) + change branch protects difference regions.
- Complex phase coupling → build the Casorati from magnitude.

**Validation:** SSIM/HFEN improvement; verify contrast-specific structures (features visible in only one
contrast) are not washed out.

## 6. P2 — Complementary sampling (Stage 2, optional)

**Idea:** give T1/FLAIR complementary masks (disjoint mid/high-freq samples) at the same total budget, so
jointly they cover ~2× unique k; the shared `G_A` lets contrast-m's sampled k constrain contrast-m''s
unsampled k at the same location (structure shared, intensity via response). Distinct from the disproven
budget *reallocation* — this is joint coverage + structural cross-fill.

**Gated on Stage 1.** It changes the dataset mask generation and re-touches DC/alignment (the §8 pitfall),
and its payoff is uncertain (the adaptive-mask probe showed energy retention is sampling-insensitive,
though P2's mechanism differs). Do P1+P3 first (loss-only, low risk); evaluate P2 only if warranted.

## 7. Validation plan

**Baseline note:** the ARC frontier was previously run on the *pre-alignment* mask; first rebuild an ARC
baseline on the aligned mask.

**Ablation:**
| Experiment | Purpose |
|---|---|
| ARC baseline (aligned mask, no P1/P3) | aligned ARC reference |
| ARC + P1 | frequency-extrapolation gain alone |
| ARC + P3 | LLR gain alone |
| **ARC + P1 + P3** | full method |
| (reference) MC-GS winner `reg_aligned` | target to match/beat |

**Metrics:** unified-`data_range` PSNR + NMSE + SSIM + HFEN (**never @max-PSNR**); held-out `Λ` NRMSE
(P1's direct signal); contrast-specific-structure preservation check; both contrasts beat ZF on
ssim∧nmse∧hfen.

## 8. Hyperparameters & experiment phases

Reuse the `arc/cycle_arc.py` single-knob sweep harness (reduced budget ~1500 iters; winner re-run at full
budget).

- **P1:** `ρ≈0.3`, re-sample `K≈50`, `λ₁` warmup@500 then sweep `[0.1, 1]`.
- **P3:** block `8³` / stride 4, `λ₃` sweep `[1e-3, 1e-1]`.

**Phases:** Phase0 ARC aligned baseline → Phase1 P1 sweep → Phase2 P3 sweep → Phase3 P1+P3 joint tune →
(optional) Phase4 P2.

## 9. Module landing points

- **P1:** new loss in `paired_arc_gs/losses/arc_losses.py` + Θ/Λ split logic in `PairedARCTrainer`
  (`paired_arc_gs/train/trainer.py`); CLI knobs `--p1_rho --p1_K --lambda_freqextrap` in `arc/cycle_arc.py`.
- **P3:** new LLR term in `arc_losses.py` (block Casorati + 2×2 closed-form nuclear norm); CLI knobs
  `--llr_block --llr_stride --lambda_llr`.
- **P2 (Stage 2):** complementary mask generation in the dataset / `paired_arc_gs/train/` mask path.
- **Metrics:** ensure the scorer reports unified-gauge PSNR + NMSE (extend the existing scoring path;
  do NOT rely on @max-PSNR).

## 10. Risks

- ARC frontier currently trails MC-GS by ~1 dB PSNR (pre-alignment numbers); P1 is expected to close this,
  but if it does not, fall back to adding P1+P3 onto MC-GS as a lower-risk variant.
- P1 and P3 both pull toward "shared/smooth"; over-weighting could blur genuine high-freq / contrast
  difference. The change branch + moderate weights + the ablation guard against this.
- Compute: each sweep cell ~1500 iters on one idle GPU (5/7); full pipeline is several GPU-hours.

## 11. Non-goals

- No external pretrained/generative prior (MAISI, diffusion) in this design.
- No de-shading / combine-gauge redefinition (deferred).
- No budget *reallocation* sampling (disproven).
