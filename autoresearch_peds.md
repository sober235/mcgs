# Autoresearch — pediatric MTP AF12, stage 3: make MC-GS lead the table

Program: `docs/superpowers/specs/2026-09-14-peds-af12-mcgs-autoresearch-design.md`.
Ledger: `autoresearch_peds.tsv` (one row per run, written by the runner).
Topic: `/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro/` (README = the table).

## The bar

MC-GS first in all six cells of the **primary** table — head PSNR / SSIM / HFEN × T1w / T2s —
against every other arm, GS-single included (user, 2026-09-14). `scripts/score_candidate.py` of the
topic prints the six margins of a run; a positive `min_margin` in the ledger is the target.

Bars at the start (the arm that sets each in brackets):

| | T1w | T2s |
|---|---|---|
| head PSNR | 38.830 [L1-ESPIRiT] | 34.175 [GS-single] |
| head SSIM | 0.9674 [L1-ESPIRiT] | 0.9052 [GS-single] |
| head HFEN | 0.0756 [L1-ESPIRiT] | 0.1855 [GS-single] |

Baseline (the stage-2 joint run `peds_af12_joint`, P10 transferred unchanged): 37.855 / 0.9618 /
0.0922 and 34.160 / 0.9047 / 0.1858, so `min_margin = −0.9744`, set by T1w head PSNR.

Bars move as the rounds go on: whatever recipe MC-GS gets, GS-single gets, and the tuned single is
then the competitor. Adding MC-INR to the table on 2026-09-14 moved nothing — it is eighth of ten.

## Why T1w is the gap

MC-GS trails L1-ESPIRiT on T1w by 0.975 dB, on HFEN by 18 % and on sharpness by 0.813 vs 0.924. That
is an over-smoothed reconstruction. The GS arms ran at a working point (P10) chosen on a 1 mm adult
MPRAGE + FLAIR pair, while L1-ESPIRiT's λ was swept on this subject over 14 values and CS-SENSE's
over 11. Round 1 removes that asymmetry before anything more interesting is tried.

## Round log

### Round 1 — λ recalibration on this subject (2026-09-14)

Four runs, one knob each, everything else at P10; joint arm only (the singles are re-run for the
finalists). Queues `scripts/p3_round1_{a,b}.txt` of the topic, two scheduler instances with disjoint
card lists (`ALLOWED="1 0 2" TAG=r1a`, `ALLOWED="3 4 5 7" TAG=r1b`), one training per card.

| run | change | verdict |
|---|---|---|
| `p3_wavT1_006` | T1w wavelet 0.12 → 0.06 (T2s pinned at 0.10) | pending |
| `p3_wavT1_003` | T1w wavelet 0.12 → 0.03 | pending |
| `p3_geo_006` | edge-TV λ_geo 0.10 → 0.06 (shared by both channels) | pending |
| `p3_geo_003` | edge-TV λ_geo 0.10 → 0.03 | pending |

The run log's `[wavelet] lambda=... (plain ..., contrast ...)` line is the proof the per-contrast
override reached the trainer: `p3_wavT1_006` printed `lambda=0.1 (plain 0.06, contrast 0.1)`.
