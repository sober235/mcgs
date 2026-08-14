# AF10 joint reconstruction autoresearch

## Scope and fixed baseline

- Target: nominal-AF=10 T1W + FLAIR joint reconstruction.
- Starting point: the effective AF=12 edge-TV working configuration, transferred without retuning:
  `lambda_geo=0.10`, `tv_eps=0.04`, `edge_tv_kappa_q=0.85`,
  `edge_tv_wfloor=0.1`, `kspace_high=2.2`, 4000 iterations.
- Fixed across candidates: AF10 aligned mask (actual net AF 9.8738), data, CSM, warm-start checkpoint,
  seed 42, shared geometry, optimizer/LR schedule, hard-DC scorer, warmup 500, max radius 20.
- Screening budget: 1500 iterations. Final promotion: 4000 iterations.

## User goal

Suppress both residual undersampling aliasing and visible patch/fragment/staircase artifacts in both
contrasts. Absolute artifact-free output cannot be certified by a scalar; the final decision requires
fixed-window axial/coronal/sagittal inspection in addition to quantitative gates.

## Final outcome

No tuned candidate was promoted. The retained configuration is the transferred AF12 parameter
baseline (`lambda_geo=0.10`, `tv_eps=0.04`, `edge_tv_kappa_q=0.85`,
`edge_tv_wfloor=0.10`, `kspace_high=2.2`) at 4000 iterations.

The closest full-budget candidate (`lambda_geo=0.095`, `tv_eps=0.03725`) improved combined score,
complex unobserved-k-space error, gradient error, strong-edge SEAC, and flat-region RMS. It was still
rejected because flat-region spatial autocorrelation regressed in both contrasts relative to two
full-budget baseline runs: T1W `0.150855 > 0.149478`, FLAIR `0.113009 > 0.112720`.
Fixed-window views showed no obvious new artifact, but visual similarity was not allowed to override
the predeclared no-regression gate. Later floor and high-band micro-sweeps produced no repeat-stable,
dual-modality feasible cell. The result therefore supports baseline retention, not a claim of absolute
artifact absence or out-of-subject generalization.

## Non-gameable gates

Every kept candidate must satisfy:

1. both contrasts beat their own zero-filled anchor on SSIM, NMSE, and HFEN;
2. all arrays and metrics are finite and both scored reconstructions are non-vacuous;
3. no blur trade: per-contrast sharpness must be at least `min(baseline A, baseline B) - 0.01`;
4. no noise trade: per-contrast grain must be at most `max(baseline A, baseline B) * 1.03`;
5. aliasing direction, separately for each contrast: direct complex multicoil unobserved-k-space
   error (`e_unobs`) must not regress beyond the same-budget repeat floor; HFEN must be at most
   `max(baseline A, baseline B) * 1.01`; magnitude-NPZ re-encoding is forbidden because candidate
   phase has been discarded;
6. fragmentation direction, separately for each contrast: the gradient-ratio curve's log-RMSE to 1
   and spatial edge-error autocorrelation (`SEAC`, GT-fixed q=0.85/0.90/0.95) are retained. The SEAC
   flat-region RMS and score must each be no worse than the worse baseline repeat. SEAC q=0.90 plus at
   least one adjacent threshold must improve beyond the baseline-repeat floor; every check is
   modality-local, so no cross-contrast averaging can rescue a failure;
7. a second 1500-iteration baseline run estimates nondeterministic repeat floors before promotion;
8. final 4000-iteration promotion must pass fixed-window three-view visual inspection for T1W and FLAIR.

`scripts/autoresearch_repeat_gate.py` is the sole automated promotion gate. It requires two distinct
same-budget baseline runs plus a distinct candidate, canonical and distinct checkpoint paths, rehashed
score/volume/config/checkpoint sources, and identical per-modality GT/ZF array hashes across all three
runs. `scripts/autoresearch_compare.py` is explicitly `SCREEN_ONLY`; its output is never promotion
eligible even when the pairwise screen passes. The artifact and complex checkpoint-replay scorers only
produce evidence for those commands.

Gradient and SEAC metrics remain proxies: SEAC is spatially sensitive but also rises under blur, so
sharpness remains a hard companion gate. The unobserved-k-space `e_unobs` value is an oracle only for
the fixed acquisition instance used by these runs. It can detect residual aliasing on that instance,
but repeated tuning against it can overfit the instance and cannot establish subject, scanner, mask,
or distribution-level generalization.

## Search discipline

- Karpathy-style propose -> run -> score -> keep/discard.
- One changed knob per screening experiment.
- Do not edit the trainer during hyperparameter search.
- Do not repeat known inert/negative AF12 levers: warmup, DTV coupling, max-radius reduction,
  densification, or vector-TV, unless new evidence overturns the historical result.
- Combine knobs only after each direction independently improves a required artifact axis.
- Never overwrite an output; every experiment gets a unique tag.

## Initial coordinate probes

| experiment | changed knob | hypothesis |
|---|---|---|
| `ar10_b0_e040_l010_h22` | none | same-budget reference |
| `ar10_p1_l008` | lambda 0.10 -> 0.08 | extra AF10 measurements permit less TV, reducing segmentation |
| `ar10_p2_e060` | eps 0.04 -> 0.06 | smoother Charbonnier TV reduces staircase transitions |
| `ar10_p3_w020` | edge floor 0.10 -> 0.20 | reduce the flat/edge regularization ratio that can segment tissue |

## Session artifacts

- runner: `autoresearch.sh`
- append-only decisions: `autoresearch.jsonl`
- per-run artifact reports: `results_arc/autoresearch_af10/*_artifact.json`
- cycle logs/scores/volumes: `results_arc/<experiment>*`
