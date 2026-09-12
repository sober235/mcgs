# jiangyizhou AF 12, phase 2 — the optimiser, not the prior (2026-09-12 16:00 → 2026-09-13 04:16)

Branch `autoresearch-jyz-af12-phase2` (from `autoresearch-jyz-af12` at 2887187), worktree
`/data0/congcong/code/GS/mcgs_af12_wt`. Program and round log: `autoresearch_jyz_af12_p2.md`. Ledger:
`autoresearch_jyz_af12.tsv` (phase-2 rows carry the `p2_` prefix). Final runner: `autoresearch_jyz_af12_P13.sh`.
Topic archive: `Results_20260908/T1FLAIR_AF12_v3_jiangyizhou`.

## What the user asked for

Two decisions opened this phase. The current AF 12 result was fixed as two figures of the P10 recipe — T1 with every
arm at AF 12, FLAIR in **design B** (MC-GS on an AF 11 mask, every comparison arm at AF 12) — and **design B became
the setting to tune in**. The goal, in the user's words: raise MC-GS further above the comparison methods, or widen
the gap. Lever scope: configuration knobs plus small code levers. The two marginal items the P10 loop had left
undecided (scale lr 0.005, 6000 iterations) were to be judged as one candidate under the loop's own rules.

## Result

**P13 = P10 + `gaussian.rotation_lr = 0.008`** (the shipped default is 0.0003; P10 = P6 + wavelet db4 on one level,
T1 0.12 / FLAIR 0.10). Nothing else changed. At 4000 iterations, head-only metrics against the bar — the best value
per cell over every non-MC-GS arm at every swept λ on this subject:

| head cell | T1 (all AF 12) | bar | margin | FLAIR (design B) | bar | margin |
|---|---|---|---|---|---|---|
| PSNR | **36.969** | 36.426 | **+0.543** | **36.683** | 35.757 | **+0.926** |
| SSIM | **0.9504** | 0.9473 | +0.0031 | **0.9444** | 0.9368 | +0.0076 |
| HFEN | **0.0948** | 0.1034 | +0.0086 | **0.0904** | 0.1000 | +0.0096 |

Against the phase's starting point (P10): **T1 +0.025 dB, FLAIR +0.046 dB**, every other head cell equal or better,
sharpness unchanged (0.858 / 0.831). Two seeds averaged (2x compute, its own row): **37.005 / 36.703** — the first
time T1 crosses 37 dB on this subject. Joint − single: T1 **+0.065**, FLAIR **−0.030** (P10: +0.047 / −0.055).
Seed floor of the final recipe at 4000 in design B: 0.002 / 0.005 dB.

## How it was found, and what it cost

27 experiments, 13 of them at the full budget, on two cards. The loop's pre-registered six-cell rule was kept
verbatim from the P10 phase: a keep must raise `min_margin` (the smaller of the two contrasts' head-PSNR margins)
by more than the seed floor without any of the other five cells falling below the parent's by more than its own
floor, and without losing 0.01 of sharpness.

**Round 1 (12 single knobs) produced exactly one keep.** Everything structural in the working point turned out to
sit at a local optimum on this subject:

| knob | result |
|---|---|
| per-contrast edge-TV strength, both directions (new code lever) | loses both ways — 0.10 is each channel's own optimum, and moving one channel moves the other through the shared geometry |
| isotropic (vector) TV | loses (sharper, less accurate — the same shape as weaker TV) |
| uniform k-space band weights | loses by 1.65 / 2.10 dB — the high-band emphasis is load-bearing, not fine tuning |
| high band 2.0 instead of 2.2 | reshape (T1 bought with FLAIR and sharpness) |
| vectorial joint TV, common-edge cross-gradient prior | lose by 0.09 and 1.33 dB — explicit cross-contrast coupling still hurts under the wavelet prior |
| gradient clipping 10 instead of 1.0 | null — the clip does not bind |
| **rotation lr 0.001 (3.3x the default)** | **keep, +0.023 / +0.025** |

**Round 2 opened the optimiser family and closed the prior's shape.** A second code lever gave the wavelet prior
per-level and per-sub-band weights; all four screens (half-weight second level, diagonal band ×0.5 and ×2.0,
single-axis bands ×0.5) lose or reshape, so **uniform weighting of the seven detail sub-bands on the single finest
level — the shape the P10 loop arrived at — is a local optimum in both directions of both knobs.**

The rotation-learning-rate bracket rose monotonically at the screening budget (0.0005 +0.008, 0.001 +0.023,
0.002 +0.038, 0.004 +0.053, 0.008 +0.063, 0.016 +0.072 dB on T1) and **flattened at the full budget** (0.001 +0.011,
0.002 +0.017, 0.004 +0.022, 0.008 +0.023, 0.016 +0.024): 0.008 is the smallest value on the plateau and became P13.

**Every stacked row that led the screening budget failed at the full one.** Scale lr 0.005 on top of a faster
rotation was the best 1500-iteration row of the whole phase (+0.087 / +0.160) and at 4000 iterations it is a trade:

| scale lr | rotation lr | T1 Δ vs P12 | FLAIR Δ | T1 SSIM Δ | T1 HFEN Δ | verdict |
|---|---|---|---|---|---|---|
| 0.003 | 0.002 | +0.003 | +0.023 | −0.0001 | −0.0001 | null |
| 0.005 | 0.004 | −0.002 | +0.047 | −0.0002 | −0.0004 | null |
| 0.005 | 0.008 | −0.005 | +0.052 | −0.0003 | −0.0005 | loses |

Recorded as a calibrated **FLAIR dial**: if the FLAIR panel is ever worth T1's fourth decimal, `gaussian.scale_lr`
is the lever and this is its dose-response. The bundle the user asked to have judged (scale lr 0.005 at 6000
iterations) reads the same way: +0.013 / +0.062 dB for a 0.0001 SSIM and 0.0002 HFEN cost on T1, a `reshape` by the
letter of the rule at a seed floor that is zero to four decimals. Not folded in; the user can overrule it by hand.

## Method notes worth keeping

- **A screening keep is not a keep.** Four times in two phases, the row with the largest 1500-iteration margin became
  a trade at 4000. The rule that catches it (promote before folding in) earned its keep again; the corollary is that
  screening is a filter, never a ranking.
- **Never edit a launched script in place.** The any-trainer guard was written into the runner while two bash
  instances were executing it; bash reads by byte offset, so one of them resumed inside the new text, ran a garbage
  line, truncated its run's log and reported `crash` for a run whose results were fine. New behaviour goes in a new
  file (`…_p2g.sh`, `…_P13.sh`).
- **`kill -0` waiters are not safe on this host.** PIDs wrapped past 4194304 during the night, and a wrapper was
  still inside a 60-second poll when its runner exited, so a hand-launched duplicate chain nearly put two trainers on
  one card. File-triggered waits (`until [ -e …_score.json ]`) and the runner's own any-trainer guard are the
  reliable pair.
- **A selector that reads hand-written ledger status can fire early.** `p2_promote_best_grid.sh` filtered on status
  `keep`, which is written after a human read, so seconds after a better row landed as `pending` it promoted the
  second-best. Caught and corrected the same minute; a correct version would apply the rule to the score jsons.
- **`pgrep -f` matches the watcher itself.** A monitor waiting for `summarize.py` to disappear waited 40 minutes for
  its own command line. The project memory already carries this trap; it applies to Monitor commands too.
- **GPU etiquette with a parallel loop.** The AF16 phase-3 session launched into the gaps between this loop's runs
  twice. Resolved by message: cards 2 and 3 here, 4 and 5 there, and a runner guard that waits while any
  `arc/cycle.py` trainer sits on the target card, whichever session owns it.

## What is closed after two phases

On this subject, at this working point: the wavelet prior's strength, family, depth, level weights and sub-band
weights; edge-TV strength shared and per contrast; TV shape; the k-space band weights; capacity; warm-up; the
iteration budget; density lr; gradient clipping; every explicit cross-contrast coupling term; and now the rotation
and scale learning rates. The remaining levers are not knobs: a learned prior, a different representation, or
acquisition-side changes.
