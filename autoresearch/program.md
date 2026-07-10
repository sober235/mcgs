# ar_mcgs — MC-GS (joint) autoresearch: kill aliasing artifacts on vd-Poisson AF≈12

## Objective (PRIMARY = eliminate undersampling aliasing in ALL 3 views)
Tune the **real MC-GS** (`JointTrainer`, shared geometry + per-primitive dual density, `train_joint.py`)
on HYW **T1+FLAIR jointly** with the **separable vd-Poisson mask** `mask/mask3d_vdpois_af12.mat`
(net-AF 11.79, prospectively-Cartesian-realizable — regular phase-encode undersampling → coherent
aliasing is a REAL artifact here, unlike the 3D-incoherent r8 mask). Drive axial/coronal/sagittal
error maps to clean/blue: remove coherent ghosts + speckle while holding fidelity.

NOTE the architecture lesson (cs:s1-448): `scheme_v1/ar` is the SINGLE-CONTRAST FROZEN-SCAFFOLD
ABLATION, NOT MC-GS. This loop (`ar_mcgs`) tunes the genuine joint model. Do not regress to ar.

## Metric (owned by cycle.py scorer — DO NOT weaken)
- `hfen`: 3D-Laplacian high-freq error norm vs GT — the artifact metric (lower = fewer aliasing/speckle
  residuals). 3D operator → covers axial+coronal+sagittal simultaneously.
- `psnr`/`ssim`/`nmse`: hard-DC synthesis + SENSE-combine vs GT, on raw model output (no alpha-fit).
- Per-contrast composite `c = psnr + 25*ssim - 60*nmse - 25*hfen` (artifact-centric; rises as artifacts drop).
- Combined score = mean(c_T1, c_FLAIR). Both contrasts must beat their ZF anchor on ssim AND nmse AND hfen.
- Grep keys per contrast in cycle log: `^T1  psnr=` `^FLAIR psnr=` and `^SCORE combined=`.

## v2 trainer reality (confirmed from train_joint.py warnings)
- `loss.lambda_diff` / `loss.lambda_enh` are **silently ignored** (v2 q(a_{n,c}) posterior replaces them) — do NOT tune.
- `loss.tv_weight` is read as `lambda_geo` (geometric/TV smoothness) — PRIMARY anti-aliasing knob.
- Tunable v2 knobs: `loss.tv_weight`(=lambda_geo), `loss.kspace_{center,mid,high}_weight`,
  `loss.kspace_center_fraction`, `voxelizer.max_radius`, `gaussian.{position,scale,rotation,density}_lr`,
  `training.shared_warmup_iters`, `gaussian.initial_num_points`.

## Anti-aliasing mechanism order (cs:s1-449 — DO NOT lead with TV)
At AF≈11.8, ~92% of k is unsampled → the dominant aliasing source is MISSING k-space. Single-contrast
TV is a WEAK secondary lever (suppresses speckle, cannot recover unsampled energy). The PRIMARY
anti-aliasing mechanism is **shared geometry / cross-contrast coupling** (compresses the inverse-problem
null-space). MC-GS already shares positions+scales; the knobs that ADJUST coupling strength are primary:
1. `training.shared_warmup_iters` — longer shared-geometry tying → stronger null-space compression (hard guard: < 0.5·max_iter).
2. `loss.lambda_dtv` (+dtv_eta_scale/dtv_w_floor) — edge-aware cross-contrast difference-TV coupling (r8 leaves it 0).
3. geometry expressivity: `adaptive_control.densify_until_iter`, `gaussian.max_num_points`, `voxelizer.max_radius`.
4. `loss.kspace_freq_curriculum` — progressive high-k fitting.
SECONDARY (cleanup only): `loss.tv_weight`(=lambda_geo), `loss.kspace_high_weight`, private lrs.

## Search plan (Karpathy: propose → run → score → keep/revert; 1 cycle = 1 git commit; single-knob delta)
- Batch 1 (4-wide parallel, iters 1500): A=baseline(vdpois r8) · B=shared_warmup 500→700 (joint tying↑) ·
  C=lambda_dtv 0→0.02 (cross-contrast coupling on) · D=lambda_geo 0.03→0.3 (secondary TV control).
- Batch 2: around the winning primary lever, combine with geometry expressivity (max_radius 20→12, densify_until↑).
- Winner re-run at full `iters=4000`; compare vs vdpois baseline (bm8udag2b) + r8 incoh deliverable (T1 38.19/FLAIR 36.21).

## Keep / reset
- keep only if combined score ↑ AND no contrast's psnr drops while its ssim<prev−0.005 or nmse>prev+0.003 (alignment gate).
- both contrasts beat ZF on ssim/nmse/hfen.
- ties → simpler (fewer prims / lower VRAM).

## Budget
- Sweep cycles: `iters≈1500` (~25-35 min/cycle on 80GB). Winner re-run: 4000.
- GPU via `--gpu` (idle 1,3,7,8). Warm-start from dualwarm ckpt (kept; part of MC-GS).

## Forbidden
- Editing the scorer/metric/GT/mask, alpha-fit or display-norm on scored path, changing mask/AF to ease the problem, new packages.

## Run command (one cycle)
`PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID python scheme_v1/ar_mcgs/cycle.py --exp <name> --gpu N --lambda_geo X --iters 1500`
