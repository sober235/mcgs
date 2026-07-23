# Design — Edge-preserving TV for sharp-boundary (CSF / ventricle) error at AF≈12

- **Date:** 2026-07-23
- **Status:** Draft (awaiting user review)
- **Base:** MC-GS winner-v2 `reg_aligned` (`JointTrainer`, isotropic magnitude-TV `lambda_geo=0.06`, 4000 iters,
  `configs/winner_v2_aligned.yaml`). Baseline run = `results_arc/reg_aligned_4k`.
- **Scope:** ONE self-contained loss-side change + its sweep & validation. Single implementation plan.
- **Approach chosen:** A (edge-preserving within-contrast regularizer). Approaches B (shared-geometry edge
  anisotropy) and C (Mumford-Shah / piecewise-smooth) are explicitly OUT of scope, gated on A's result.

## 1. Motivation

Established this session (project `memory.md` §6, `results_arc/ventricle_error_diag.png`): the "butterfly" /
ventricle-region error the user flagged is **not a distinct defect** — it is the general AF=12
spectrum–sampling artifact **localized at sharp boundaries**. Brain-wide, `mean|err|/peak` rises
**monotonically** with local edge sharpness `|∇GT|` (T1 gradient-quintile 1→5: 0.0117→0.0189; FLAIR
0.0107→0.0201), and the ventricle wall is one of the sharpest CSF–tissue boundaries in the brain, so error
concentrates there. Hard-DC makes the 8.5 % measured k exact ⇒ **all residual is UNMEASURED high-freq**
(Gibbs ringing + blur) at edges.

**Constraint (user decision):** SELF-CONTAINED, edge-directed — no external/generative model; work only
inside the MC-GS framework.

**Physics ceiling (bounds this whole design):** no self-contained prior has new high-freq information, so
none can make the edge "correct." The one lever available: the current within-contrast prior is
**isotropic** magnitude-TV (`_tv_loss`, `lambda_geo=0.06`), which conflates "smooth interior grain" and
"don't blur the edge" into ONE knob (turn up → blur edges; turn down → grain remains) — a known-suboptimal
edge-vs-grain tradeoff. An edge-preserving prior can reach a **Pareto-better** tradeoff that isotropic TV
structurally cannot. The expected gain is modest and the strict bar (§2/§4) guards against calling a null a
win.

**Disproven, do NOT re-tread:** cross-contrast gradient couplings (dtv/vtv/edge) "only reshape PSNR↔SSIM";
P1 (self-supervised freq extrapolation) and P3 (cross-contrast LLR) negative; DC-centering no gain (§6);
N4 de-shading is a methodological artifact, not a lever (§6). **This design is a WITHIN-contrast edge-aware
smoother — a different lever from all of the above.**

## 2. Goals & success criteria

Reduce edge-region error WITHOUT the classic "reshape." **Strict, pre-registered quantitative bar** (details §4):

- edge-region `NMSE` (or `HFEN`) **down by > run-to-run noise floor**; AND
- whole-brain `SSIM` **not down** (within the noise band counts as not-down); AND
- whole-brain `NMSE` **not up**.

"edge NMSE down but SSIM down" = **RESHAPE = FAIL** (the exact trap that killed dtv/vtv/edge/P1/P3).

## 3. The prior

Replace isotropic magnitude-TV with **edge-weighted TV (detached Perona-Malik weight)**:

```
L_ewtv = mean( w(x) · |∇u(x)| ),   w(x) = clamp( 1 / (1 + (|∇u|_sg / κ)²),  w_floor, 1 )
```

- `u` = rendered per-contrast **magnitude** volume (as `_tv_loss` already uses `vol.abs()`); `|∇u|` = the
  same finite-difference gradients (∂x,∂y,∂z) `_tv_loss` already computes.
- `|∇u|_sg` = **stop-gradient (detach)** of `|∇u|` ⇒ `w` is a per-voxel spatial WEIGHT, not full anisotropic
  diffusion ⇒ gradient wrt `u` stays ~ `w·sign(∇u)` ⇒ **numerically stable** (matches the codebase's
  "magnitude-based, bounded-gradient, STABLE" preference; the normalized `_edge_align_loss` is the unstable
  counter-example).
- Flat region `|∇u|≈0 → w≈1 → full TV` (smooth grain). Edge `|∇u|≫κ → w→w_floor` (edge nearly free).
- `w_floor` keeps a residual smoothing at edges to **damp Gibbs ringing** (ringing is high-gradient and would
  otherwise be entirely un-penalized).
- **Per-contrast, within-contrast** (T1 and FLAIR each use their own `|∇u|`). No cross-contrast coupling in v1.
- **New hyperparameters:** `κ` (edge threshold), `w_floor`. Overall weight reuses/retunes `lambda_geo`.
- **`κ` is data-adaptive:** set once at init from a quantile of the initial render's in-brain `|∇u|` (so it
  does not drift as training changes the gradient scale). Store the resolved absolute `κ` in the run log.

Why not Huber / TGV: Huber is quadratic for small gradients (soft on exactly the grain we want to kill) and
linear on edges (still penalizes them) — wrong shape. TGV (second-order, anti-staircase) needs an auxiliary
variable and is higher-complexity — deferred.

## 4. Validation

**Edge set (recon-independent).** On GT, `edge = dilate( |∇GT| ≥ Q80(in-brain), 2–3 vox )` — top-quintile
gradient, dilated to include the ~3–4 vox Gibbs ring measured this session. Computed per contrast. Plus the
**verified ventricle mask** (per-slice local-percentile segmentation, contour visually validated — see §6
methodology note in `memory.md`) as the flagship region.

**Metrics (unified gauge, NEVER @max-PSNR):**
- edge-region: `NMSE_edge`, `HFEN`
- whole-brain: `SSIM`, `NMSE`, `PSNR@P99.9`
- ventricle: `NMSE_vent`
- gradient-quintile `mean|err|` curve (the tool built this session)

**Noise floor.** Re-run the BASELINE once (same warm-start / seed42 / 4000 iters) to get the run-to-run band
on `NMSE_edge`/`SSIM`/`NMSE` (training is `deterministic=False`: cuDNN-fast + TF32). Any "win" must exceed
this band.

**Strict acceptance:** the §2 bar, all three conditions, pre-registered.

**Confounds controlled:** the ONLY difference vs baseline `reg_aligned_4k` is the edge-weighted TV;
warm-start / seed 42 / 4000 iters / all other hyperparameters identical.

## 5. Experiment plan

**Code landing (config-gated, default-off, reversible — same pattern as `center_kspace_dc`):**
- `paired_gs/trainers/joint_trainer.py`: add `_edge_preserving_tv_loss(vol, kappa, w_floor)` beside
  `_tv_loss`; call it in the training loop when `loss.edge_tv == true` (else legacy `_tv_loss`).
- `arc/cycle.py`: CLI `--edge_tv --edge_tv_kappa_q --edge_tv_wfloor` (following existing override pattern;
  `--edge_tv_kappa_q` is the in-brain gradient quantile that resolves to absolute κ at init).
- `configs/winner_v2_edgeTV.yaml`: copy of `winner_v2_aligned.yaml` + `loss.edge_tv: true` + κ-quantile +
  `w_floor`.
- `scripts/score_edge.py`: compute the §4 edge-set / ventricle / quintile metrics from a run's
  `*_volumes.npz` → append a TSV row. Reproducible, committed.

**Sweep:** `κ_q ∈ {0.60, 0.75, 0.85}` × `λ_edgeTV ∈ {0.06, 0.10, 0.15}`, `w_floor = 0.1` fixed for v1;
refine `w_floor ∈ {0, 0.1}` only at the best cell. (`λ_edgeTV` = the edge-weighted TV term's weight; when
`edge_tv` is on it simply REPLACES the isotropic-TV `lambda_geo` — there is one TV term, not two. Note `κ_q`
is a quantile of the RENDER gradient `|∇u|` used to build the prior's weight `w`; the separate `Q80` in §4 is
a quantile of the GT gradient `|∇GT|` used only to define the evaluation edge-set — different quantiles, do
not conflate.)

**Phases:**
- **P0** — noise floor: baseline re-run (1 cell).
- **P1** — coarse `κ_q × λ` 9-cell sweep @ **1500 iters** (~17 min/cell); pick best by `NMSE_edge` down with
  `SSIM` guard.
- **P2** — best cell @ **4000 iters**; full strict-bar evaluation vs baseline `reg_aligned_4k`.
- **Decision gate:** P2 clears the §2 bar → success; reshape-only → **documented negative result**, then
  consider Approach B or stop. Record the outcome in `memory.md` either way.

**Compute:** P1 ≈ 2.5 GPU-hours; P2 ≈ 44 min/cell. Runs on one idle GPU (GPU 5 was free this session).

**Deliverables:** gated edge-TV code, sweep TSV, one strict-bar figure (baseline vs best edge-TV: edge-region
error maps + gradient-quintile curve + metric table), `memory.md` update.

## 6. Risks

- **Ringing amplification:** edge-weighted TV un-penalizes edge-adjacent Gibbs ringing (high-gradient) →
  `NMSE_edge` could *rise*. Mitigations: `w_floor` (residual edge smoothing) + `κ` set so mid-gradient ringing
  sidelobes still get partial smoothing. This risk is exactly why `NMSE_edge` is a PASS condition, not an
  assumption.
- **Small headroom:** `lambda_geo=0.06` is already a swept sweet spot ⇒ the available Pareto gain may be tiny.
  The strict bar + noise floor keep us honest.
- **Weight stability:** `w` depends on the evolving `|∇u|`; keep it detached and cheap per-iter; fix `κ` at
  init so the bands don't drift.

## 7. Non-goals

- No cross-contrast coupling in v1 (disproven family; a T1-guided edge-weight variant is a possible *later*
  extension, not this design).
- No external / pretrained / generative prior.
- No geometry / Gaussian-shape change (that is Approach B, gated on A).
- No new acquisition / AF change.
- No change to the shipped winner default (`edge_tv` defaults off).
