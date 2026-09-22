# Coupling attribution — Stage 3, the placebo second contrast (2026-09-22)

Spec: `docs/superpowers/specs/2026-09-20-coupling-attribution-design.md` §3 Stage 3 (axis
justification corrected in `dcb229e`), decision rule §4. Branch `coupling-attribution`, worktree
`mcgs_hparam_wt`. Code: `scripts/ca_flip_flair.py`, checks `scripts/ca_verify_placebo.py`, test
`tests/test_ca_flip_flair.py`.

## The question

Stages 1 and 2 showed the common-edge prior is a cross-contrast mechanism, but both stages are still
joint modelling. Neither tests whether the gain needs **shared anatomy** or merely a **second
well-posed channel**. Stage 3 keeps jiangyizhou's own FLAIR k-space, coils, SNR and vendor shutter
and flips the FLAIR volume so it is a real brain that does not register to T1.

## What was run

Two placebo arms, both joint, λ_edge 0.001, 4000 iterations, seed 42, 3.6 M points, house AF16 mask.

| arm | flip axis | k-space support vs original | role |
|---|---|---|---|
| `ca_j_placebo` | readout x (head–foot) | **identical, 0 differing samples** | **primary** |
| `ca_j_placebo_ky` | ky (anterior–posterior) | mirrored about DC, 2454 PE lines move | secondary, context only |

The readout axis is the primary because FLAIR's shutter is a DC-centred ellipse **with a one-sided
partial-Fourier cut at ky ≥ 20** (support 34139/45056 lines, modelled to 1 line in 45056 by
`ellipse(128,88; 128,88) ∧ ky ≥ 20`), so it is not DC-symmetric in ky and a ky flip must mirror it.
FLAIR's support is complete along the readout (34139 PE points at every one of the 256 x indices), so
that flip returns it untouched and the placebo's `[physics:contrast]` line is **identical to the
real-FLAIR anchor's in every known-zero field** — `cuts` `axis0_lo=20`, ellipse `(128.0, 88.0, 1.0)`,
`frac_z_used=0.2423`, `zero_lines=233/2966`, `explained=233`, `nonzero_in_z=0`, `outliers=0/0`.
Field-by-field, exactly one of the fifteen differs: `noise_level_used`, 0.0003062865580432117 for
`hp_n36` against 0.00030644499929621816 for `ca_j_placebo`, **5.1730e-04 relative**, which is the
float32 FFT round trip and carries no known-zero content. The ky arm's line differs where it matters
(`cuts` all `None`, `frac_z_used=0.2151`, 199/229 explained).
The readout flip is also the more adversarial: head-masked NCC to T1 **−0.0957** vs **+0.0581** for ky
(whole-volume 0.349 vs 0.427; unflipped 0.801 / head-masked +0.5628).

Primary/secondary was a controller ruling recorded in `dcb229e` **before either run scored** —
verified, not asserted: `results_arc/ca_j_placebo*_score.json` was checked for non-existence in the
same shell command as that commit.

## Numbers

`plain` channel only (`plain_name = T1` on all six); the contrast channel of a placebo run is
reconstructing a flipped brain and its score is meaningless. Read from `results_arc/<exp>_score.json`.

| run | partner | λ_edge | T1 head PSNR | margin | init `support_frac` |
|---|---|---|---|---|---|
| `hp_n36` | real FLAIR, aligned | 0.001 | 35.706153 | **M1 = +0.148154** | 0.4264 |
| `ca_s_t1_p3` | T1 in both slots | 0.001 | 35.557999 | — | 0.4219 |
| `ca_j_edge0` | real FLAIR, aligned | 0 | 35.644656 | **M0 = +0.045714** | 0.4264 |
| `ca_s_t1_edge0` | T1 in both slots | 0 | 35.598942 | — | 0.4219 |
| **`ca_j_placebo`** | FLAIR, head–foot flip | 0.001 | **35.617065** | **+0.059066** | 0.4801 |
| `ca_j_placebo_ky` | FLAIR, ky flip | 0.001 | 35.628940 | +0.070941 | 0.4802 |

Both placebo runs `rc=0`, scored 16:58–16:59 (`results_arc/ca_runs.log`). Seed floor on this line
is 0.001 dB.

## Verdict against the pre-registered rule

Applied to the **primary arm only**, per `dcb229e`:

```
|placebo_primary − M1| = |+0.059066 − (+0.148154)| = 0.089088 dB
band 0.02 dB                                        = 4.45× the band
```

**The §4 Stage-3 row does not fire.** The reading "the second channel's anatomy is irrelevant; the
joint-modelling claim fails; the effect is regularisation / conditioning" is **not** triggered. The
secondary ky arm sits 0.077213 dB from M1 and would not fire either.

## What this means, both halves

- **60.1 % of the published margin (0.089088 dB) is lost** when the partner stops registering.
- **39.9 % (+0.059066 dB, 59× the seed floor) survives** a partner that is a *flipped brain*. This is
  real and must be reported: two fifths of the +0.148 does not require the partner's anatomy at all.
  It is what a second well-posed channel with brain-like statistics is worth to the shared cloud,
  whatever it depicts.

**Placebo vs M0.** M0 = +0.045714 is a registered partner with the prior off; the placebo is a
misregistered partner with the prior on; they differ by +0.013351 dB. Both mechanisms are
individually small. This is **not** an additive split of M1 into geometry + prior and must not be
written as one: the two are measured at different λ_edge, their single-arm baselines differ
(35.598942 vs 35.557999), and M0's arms differ in cloud support by 0.0045 against the placebo's
0.0582 — a 13× larger mismatch.

**Ordering (free consistency check, two points only).** 35.617065 (NCC −0.0957) < 35.628940 (+0.0581)
< 35.706153 (+0.5628): monotone in measured registration quality, and the ky arm achieves its higher
score *despite* a weaker `known_zero`. But the two steps are not of the same standing, and they must
be read against **this study's own 0.02 dB indistinguishability band**, not against the seed floor:

| step | Δ | vs seed floor 0.001 | vs the 0.02 dB band | standing |
|---|---|---|---|---|
| readout → ky | +0.011875 | 11.9× | **0.59×** | **below the band — the two placebo arms are indistinguishable from each other by the standard this record applies to every other comparison** |
| ky → real FLAIR | +0.077213 | 77.2× | 3.86× | above the band, solid |

So the honest reading is: the *placebo-to-real* step is established; the *placebo-to-placebo* step is
**suggestive only** and cannot be claimed, because it falls below the tolerance this study
pre-registered. The observation is kept because it points the right way and costs nothing, but quoting
it against the seed floor while quoting M0, M1 and the Stage-3 verdict against the band would be two
yardsticks in one document with the more favourable one chosen where the evidence is thinnest. It is
not a dose-response curve and it rests on two points.

## The limit of the claim

`init_mode=head` draws the shared cloud from the union of the two channels' ZF head supports
(`joint_trainer.py:251-272`), so **both** placebo arms grow it from 0.4264 to 0.4801/0.4802 — the same
3.6 M points cover ~12.6 % more volume. The two arms agree to one part in 4801, so the effect is
axis-independent and intrinsic to any misregistered partner; it is not an artefact of how the flip
was built (confirmed independently in review).

**Therefore the claim this stage supports is: 60 % of the published margin requires the two volumes to
register.** Whether "register" means the partner's *anatomy* specifically, or merely that the partner
occupies the same support so the shared cloud is not spread thinner, **Stage 3 does not separate**.
The paper's mechanism sentence must use the broader wording unless a later experiment separates them.

## Reproducing

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH= python scripts/ca_flip_flair.py --axis 0      # or --axis 1
env PYTHONNOUSERSITE=1 PYTHONPATH= python scripts/ca_verify_placebo.py --axis 0  # or --axis 1
env PYTHONNOUSERSITE=1 PYTHONPATH= python -m pytest tests/test_ca_flip_flair.py -v
bash scripts/ca_launch_one.sh ca_j_placebo <card> 4000 configs/t1flair_v3_af16_jyz_placebo.yaml 0.001 SAVE_VOLUMES=1
```

`ca_verify_placebo.py` checks support, complex SENSE self-consistency (primary: max 2.324430e-09,
rel L2 5.484372e-09 in float64 — a magnitude-only check would miss a phase error in the CSM flip),
the round-off the zero re-imposition removes, NCC whole and head-masked, and runs the same
`estimate_known_zero` the trainer runs, comparing Z.

**The two arms are a positive control on the zero-set logic.** The one way this task could have failed
invisibly is re-imposing the *original* support instead of the reflected one. The ky file equals
`rev0(orig)` exactly and differs from `orig` in 628 224 samples; the readout file equals **both**,
because for a support complete along that axis the two sets coincide — not because the code took a
shortcut. Same code path, opposite answers, both correct.

## Known gaps and open questions

- One subject, one mask, one acceleration, one seed per arm; the seed floor (0.001 dB) is carried over
  from the AF16 line, not re-measured for the placebo.
- Two placebo points, and the step between them (0.011875 dB) is below the 0.02 dB band, so the
  registration ordering is suggestive, not a curve.
- **Open, not analysed here: λ_edge costs the single arm.** `ca_s_t1_edge0` (λ_edge 0) is
  35.598942 against `ca_s_t1_p3` (λ_edge 0.001) 35.557999 — the prior makes the single arm
  **0.040943 dB worse**, which is 27.6 % of M1 and 2.05× the band. This record cites both baselines as
  reason (b) for refusing an additive split but does not comment on the direction or the size. It
  bears on how M1 itself should be read, and it is already carried in the Stage-1/2 ledger as the
  equal-budget number; no new runs are needed, but someone should write down what it means.
- **Open: "39.9 % survives" is arguably a lower bound.** The placebo pays a cloud-dilution handicap
  its comparator does not — `support_frac` 0.4801 against the single arm's 0.4219 and `hp_n36`'s
  0.4264 — so the surviving fraction measured here may understate what a misregistered partner is
  worth at matched cloud density. This record already treats the mirror-image fact as making the
  60 % figure compound; the same caveat applies to the 40 %.
- This dataset's readout k-space DC sits at x = 127, one sample off the array centre (masked readout
  energy 1.000 at 127 vs 0.701 at 126 and 0.674 at 128). Pre-existing, affects every run in this line;
  the flip moves it to 129, equal and opposite. It does not enter `Z` (verified bit-identical),
  `prewhiten` is off, and the frequency-adaptive k-space weighting centres on `size // 2`
  (`kspace_loss.py:40-57`), so anchor and placebo are 1 sample off in opposite directions — matched.
