# arc — paired dual-contrast MC-GS autoresearch: kill aliasing/grain artifacts on the configured mask

> **TEMPLATE** — the shipped instance targets the T1(plain)+FLAIR(contrast) pair; grep keys in
> cycle.py logs are the configured contrast names (default 'plain'/'contrast', shipped 'T1'/'FLAIR').

Placeholders `{plain}` / `{contrast}` denote the two channels, named by `cfg["data"]["plain"]["name"]`
/ `cfg["data"]["contrast"]["name"]` (defaults `"plain"` / `"contrast"`). Retarget the harness to a new
contrast pair by editing the config's `data.plain` / `data.contrast` subdicts (paths + optional `name`);
no code change is needed — the scorer and logs are channel-neutral.

## Objective (PRIMARY = eliminate undersampling artifacts in ALL 3 views)
Tune the **paired MC-GS** (`JointTrainer`, shared geometry + per-primitive dual density) on the
**{plain}+{contrast}** pair jointly with the **configured undersampling mask**. Drive
axial/coronal/sagittal error maps to clean/blue: remove coherent ghosts + speckle while holding fidelity.
- SHIPPED INSTANCE: separable vd-Poisson mask `mask3d_vdpois_af12_transverse.mat` (net-AF 11.79,
  prospectively-Cartesian-realizable — regular phase-encode undersampling → coherent aliasing is a REAL
  artifact here, unlike a 3D-incoherent r8 mask).

ARCHITECTURE NOTE: this loop tunes the GENUINE joint model (shared geometry + per-primitive dual
density), NOT a single-contrast frozen-scaffold ablation. Keep {plain} and {contrast} coupled through
shared geometry; do not regress to independent per-contrast reconstruction.

## Metric (owned by cycle.py scorer — DO NOT weaken)
- `hfen`: 3D-Laplacian high-freq error norm vs GT — the artifact metric (lower = fewer aliasing/speckle
  residuals). 3D operator → covers axial+coronal+sagittal simultaneously.
- `psnr`/`ssim`/`nmse`: hard-DC synthesis + SENSE-combine vs GT, on raw model output (no alpha-fit).
- `sharp`: edge-sharpness ratio (mean gradient rec/GT over the brain mask); one-sided anti-blur penalty at <1.
- `grain`: std of the high-pass residual over GT-flat brain voxels; anti-speckle in homogeneous tissue.
- Per-contrast composite `c = psnr + 25*ssim - 60*nmse - 25*hfen - 40*max(0,1-sharp) - 200*grain`
  (artifact-centric; rises as artifacts drop). `grain` and `sharp` pull opposite ways → the optimum is
  an edge-preserving denoise balance.
- Combined score = mean(c_plain, c_contrast). Both contrasts must beat their ZF anchor on ssim AND nmse AND hfen.
- Grep keys per contrast in cycle log: each score line starts with the contrast name padded to 5 chars —
  `^{plain}` / `^{contrast}` (shipped: `^T1` / `^FLAIR`), plus `^SCORE combined=`.

## v2 trainer reality (confirmed from joint_trainer warnings)
- `loss.lambda_diff` / `loss.lambda_enh` are **silently ignored** (v2 q(a_{n,c}) posterior replaces them) — do NOT tune.
- `loss.tv_weight` is read as `lambda_geo` (geometric/TV smoothness).
- Tunable v2 knobs: `loss.tv_weight`(=lambda_geo), `loss.kspace_{center,mid,high}_weight`,
  `loss.kspace_center_fraction`, `voxelizer.max_radius`, `gaussian.{position,scale,rotation,density}_lr`,
  `training.shared_warmup_iters`, `gaussian.initial_num_points`.

## Anti-aliasing mechanism order (DO NOT lead with TV)
At high AF (shipped ≈11.8, ~92% of k unsampled) the dominant aliasing source is MISSING k-space.
Single-contrast TV is a WEAK secondary lever for ALIASING (suppresses speckle, cannot recover unsampled
energy). The PRIMARY anti-aliasing mechanism is **shared geometry / cross-contrast coupling** (compresses
the inverse-problem null-space). MC-GS already shares positions+scales; the knobs that ADJUST coupling
strength are primary:
1. `training.shared_warmup_iters` — longer shared-geometry tying → stronger null-space compression (hard guard: < 0.5·max_iter).
2. `loss.lambda_dtv` (+dtv_eta_scale/dtv_w_floor) — edge-aware cross-contrast difference-TV coupling (baseline leaves it 0).
3. geometry expressivity: `adaptive_control.densify_until_iter`, `gaussian.max_num_points`, `voxelizer.max_radius`.
4. `loss.kspace_freq_curriculum` — progressive high-k fitting.
SECONDARY (cleanup only): `loss.tv_weight`(=lambda_geo), `loss.kspace_high_weight`, private lrs.

## Shipped priors (from the winner-v2 sweep — starting point; re-verify per contrast pair)
- **TV** (`loss.tv_weight` → `lambda_geo`) is the PRIMARY anti-grain lever: grain decreases monotonically
  with TV, but sharpness also drops → sweet spot **TV = 0.05–0.06**.
- **`kspace_high_weight`** (~2.0–2.2) recovers sharpness but UNIFORMLY amplifies flat-region grain; the
  two levers oppose each other, so the optimum is an edge-preserving denoise balance.
- **`shared_warmup_iters` = 500** baseline; warmup 500→700 saturated (no gain).
- **Winner-v2 = TV 0.06, kspace_high 2.2** (v1 was kspace_high 2.0).

## Search plan (propose → run → score → keep/revert; 1 cycle = 1 git commit; single-knob delta)
- Batch 1 (4-wide parallel, iters 1500): A=baseline · B=shared_warmup 500→700 (joint tying↑) ·
  C=lambda_dtv 0→0.02 (cross-contrast coupling on) · D=lambda_geo sweep around the TV sweet spot (0.05–0.06).
- Batch 2: around the winning primary lever, combine with geometry expressivity (max_radius 20→12, densify_until↑).
- Winner re-run at full `iters=4000`; compare vs the {plain}/{contrast} baseline + prior deliverable
  (shipped instance: r8-incoherent T1 38.19 / FLAIR 36.21).

## Keep / reset
- keep only if combined score ↑ AND no contrast's psnr drops while its ssim<prev−0.005 or nmse>prev+0.003 (alignment gate).
- both contrasts beat ZF on ssim/nmse/hfen.
- ties → simpler (fewer prims / lower VRAM).

## Budget
- Sweep cycles: `iters≈1500` (~25-35 min/cycle on 80GB). Winner re-run: 4000.
- GPU via `--gpu`. Warm-start from the dualwarm checkpoint named in the config (kept; part of MC-GS).

## Forbidden
- Editing the scorer/metric/GT/mask, alpha-fit or display-norm on scored path, changing mask/AF to ease the problem, new packages.

## Run command (one cycle)
`PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID python arc/cycle.py --exp <name> --gpu N --lambda_geo X --iters 1500`
