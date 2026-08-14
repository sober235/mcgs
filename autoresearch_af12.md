# AF12 T1W + FLAIR artifact autoresearch

Target acquisition: nominal AF=12, measured net AF=11.83193, aligned vd-Poisson mask.

Baseline: `tvs_e040` at 1500 iterations for screening and `tvs4k_e040` at 4000
iterations for full-budget promotion. Effective parameters: `lambda_geo=0.10`,
`tv_eps=0.04`, `kspace_high=2.2`, `edge_tv_kappa_q=0.85`,
`edge_tv_wfloor=0.10`.

Promotion is modality-local. T1W cannot compensate for FLAIR. A candidate must:

- beat zero fill in both modalities;
- remain within same-budget repeat variation for HFEN, grain, gradient-log RMSE,
  flat-region SEAC RMS/score, and unobserved complex multicoil k-space error;
- improve strong-edge SEAC at q90 and at least one neighboring q85/q95 beyond
  repeat variation in both modalities;
- pass a fixed-window transverse/coronal/sagittal visual inspection with no
  coherent alias fringe, sheet, or fragment regression.

The legacy combined score is ranking-only; it cannot override any hard gate.
Runner: `autoresearch_af12.sh`. Decisions: `autoresearch_af12.jsonl`.

## Round 2 outcome (2026-08-13): all three candidates rejected

Cells at 1500 iterations, `kspace_high=2.2`, `edge_tv_kappa_q=0.85`,
`edge_tv_wfloor=0.10` fixed: baseline repeat `af12_ar_b1_repeat_e040`
(0.10 / 0.04), and candidates `af12_ar_r2_l0105_e050` (0.105 / 0.05),
`af12_ar_r2_l011_e050` (0.11 / 0.05), `af12_ar_r2_l011_e060` (0.11 / 0.06).
Gate records: `results_arc/autoresearch_af12/*_repeat_gate.json`.

Baseline A is `tvs_e040`, which predates the in-run `e_unobs` scorer; its
unobserved-k-space value came from a checkpoint replay
(`scripts/autoresearch_kspace_score.py`, iteration 1500), not from retraining.

The round-2 hypothesis was that a larger `lambda_geo` would pay for the HFEN/grain
cost that round 1 charged to a larger `tv_eps`. It does, partially — but two gates
fail in **both** modalities for **every** candidate:

- `e_unobs` (oracle complex multicoil unobserved-k-space error) regresses by
  **47-156x the two-baseline repeat delta**. Raising lambda does not offset it, and
  the regression grows with eps (0.05 -> +0.0005..0.0008, 0.06 -> +0.0010..0.0014).
  Larger eps restores small-gradient content in image space and buys it with a worse
  extrapolation into unmeasured k.
- flat-region false-edge RMS regresses (+1.3% to +5.8%) — the same gate that rejected
  the AF10 full-budget candidate.

Strong-edge SEAC, sharpness and the gradient-ratio curve do improve for T1W, so this
is another reshape, not a gain.

**Two limitations of the gate, recorded rather than retuned** (standing rule: a
pre-registered rule stays as written):

1. The two 1500-iteration baselines are near-deterministic (T1W `e_unobs` delta
   1.4e-05, `seac_q90` delta 5e-05). Every repeat-anchored tolerance is therefore
   near-zero by construction, so those gates are effectively "no regression at all"
   rather than "within run-to-run noise". This is the AF10 §17d failure mode
   (a stability-based denominator punishing stability) in a milder form; it did not
   decide round 2, because the `e_unobs` margins are 47-156x rather than marginal.
2. `flat_region_rms` and `grain` rank the three candidates **identically** in both
   modalities (the only ordering disagreement is between the two baselines, which
   differ at repeat-noise level). `flat_region_rms` carries zero tolerance while
   `grain` carries 3%, so on this parameter family the flat-region gate acts as a
   zero-tolerance restatement of the noise gate. It is not independent evidence, and
   a future round should not count it as a second failing axis.
