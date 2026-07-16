# Paired-ARC-GS — project instructions (auto-loaded)

MC-GS joint 3D reconstruction of an AF=12 dual-contrast **T1 (plain) + FLAIR (contrast)** brain pair:
one shared 3D-Gaussian geometry + per-primitive dual complex density (α_plain/α_contrast), warm-started.
The frozen method, every hyperparameter, canonical results, and the verified mask/CSM recipes are in the
imported memory at the bottom of this file — read them before proposing changes.

## Environment (run with `PYTHONNOUSERSITE=1` — user-site torch has a broken CUDA lib)
- Winner / joint trainer (`arc/cycle.py` → `JointTrainer`): conda env **GSTMI**
  (`/home/congcongliu/anaconda3/envs/GSTMI/bin/python`).
- Legacy single-contrast proxy trainer (warm-start derivation, `…/3dgsVC/train.py`): env **vediffusion0328**.
- GPUs are shared; pick idle ones via `nvidia-smi`. Long trainings: launch in background + poll (don't foreground-block).

## Harness gotcha (critical)
`arc/cycle.py` CLI defaults **CLOBBER** the yaml. To reproduce the winner you MUST pass
`--lambda_geo 0.06 --kspace_high 2.2 --warmup 500 --max_radius 20`. Base = `configs/winner_v2.yaml`;
aligned-mask variant = `configs/winner_v2_aligned.yaml`. Final metrics print as `T1 psnr=… ` / `FLAIR psnr=…`.

## Settled findings — don't re-run these dead ends
- The "AF=12 ceiling" was a **mask-registration bug** (mask ACS ~22 px off the raw k-space DC at PE index
  132 vs matrix center 110). The lever was aligning the mask → `mask3d_vdpois_af12_transverse_ALIGNED.mat`.
  After alignment both contrasts clear the **>38 dB target** (T1 38.5 / FLAIR 38.26). This was the real win.
- Above ~38.5 the residual is **unsampled-k extrapolation** — the scorer applies hard-DC
  (`k_dc = m·y + (1−m)·k_model`, `arc/cycle.py`), so the 8.5% sampled k is already perfect; only the 91.5%
  unsampled k remains, and it is **prior-limited**.
- Tested null (do not repeat expecting a PSNR gain): warm-start re-derivation on aligned data (joint
  optimization washes it out, −0.08 dB), capacity/densification, hyperparameter sweeps, and 3 inter-contrast
  structural priors — dtv (`--lambda_dtv`, small +), VTV (`--lambda_vtv`, PSNR↔SSIM trade), edge-align
  (`--lambda_edge`, catastrophic −10 dB). They only reshape PSNR↔SSIM. The real PSNR lever = a **learned
  prior** for unsampled-k (aligns with the MAISI/generative direction), not hand-crafted regularization.
- **Why T1 < FLAIR = spectrum–sampling mismatch** (2026-07-16 verified): T1's image energy sits higher in
  freq (95%-energy PE radius 34 vs FLAIR 23); real multicoil retained energy T1 73.8% / FLAIR 89.8% (NMSE
  2.4× worse). Not a bug/CSM/geometry — a data×mask physical inevitability; fix needs a prior, not tuning.
- **Contrast-adaptive sampling is a dead end**: at fixed total budget, reallocating samples between the two
  contrasts barely changes retained energy (min 73.8→74.3, +0.5 pt); T1's missing far-high-freq isn't
  reachable even at af08. Don't touch the acquisition protocol to chase this.
- **Metric-gauge trap**: PSNR with `data_range=max(gt)` inflates T1 by ~1.6 dB (higher peak) → false "T1
  beats FLAIR". Compare contrasts with NMSE or a unified data_range; unified, FLAIR leads T1 by ~1.4 dB.
  See memory.md §3.

## Conventions
- Error / difference maps: `jet` colormap, fixed range `[0, 0.1]` (normalized to the GT peak). Never autoscale.
- Result figures: copy to an accessible path and always give the **full absolute path** — the user works over
  VS Code Remote-SSH and cannot see images the agent opens with Read.
- Do **not** `git commit`/`push` unless explicitly asked.

# Project memory (imported — auto-loaded every session)
@memory.md
@../memory.md
