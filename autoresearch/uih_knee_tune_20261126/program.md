# Knee MC-GS tuning loop (autoresearch), started 2026-11-26 20:36 PST

- **Question:** which trainer settings raise the MC-GS (joint T1 + T2) quality on the UIH knee, S06, AF12?
- **Metric (one number):** mean of the T1 and T2 head PSNR of the joint model (trainer gauge, seed 42).
  Baseline 32.158 (33.213 / 31.102). Watched beside it: SSIM, and the margin over GS-single (33.217 / 31.214).
- **Goal:** metric up, and the joint model ahead of GS-single on both contrasts. GS-single is re-run with every
  setting that is kept and is not specific to the joint model, so the comparison stays like for like.
- **Rules:** one change per candidate against the current keeper; a change is kept if the metric rises by more
  than 0.02 dB and neither contrast drops by more than 0.02 dB; every run logged in experiments.jsonl; nothing
  overwritten; one card, one job (project cap 3 cards); stop and ask after three rounds without a kept change.
- **Caveat:** S06 becomes the tuning subject; a kept recipe must be checked on other subjects and a second seed
  before it counts as a result. The reference (never a training input) is used only for scoring.
- **Files:** tune_knee.py (round driver), rounds/<round>.json (keeper + candidates), experiments.jsonl.

## Round 1 (reconstructions are smoother than the reference -> less regularisation / more capacity)
wavelet weight halved; lambda_geo halved; 3.6 M Gaussians; 6000 iterations; coupling off (lambda_edge 0).
