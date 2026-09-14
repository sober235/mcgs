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

## Round 0 — where the T1w gap actually is (2026-09-14, diagnostic)

`scripts/diag_kspace_error.py` of the topic splits each arm's error energy over the phase-encode
plane (`logs/diag_kspace_error.log`). Energies are relative to the ground truth's, x1e3:

| T1w | total | on the sampled 2635 | unsampled, in support | on the measured zeros | radius 0.25–0.50 | radius 0.50–1.00 |
|---|---|---|---|---|---|---|
| MC-GS | 8.74 | 0.161 | 8.542 | 0.039 | 1.513 | 6.877 |
| GS-single | 8.34 | 0.155 | 8.144 | 0.039 | 1.474 | 6.510 |
| L1-ESPIRiT | 7.93 | 0.166 | 7.724 | 0.041 | 1.381 | 6.169 |

Three things follow, and they set the order of the levers.

1. **MC-GS is not underfitting the data.** On the sampled positions its error is 0.161 against
   L1-ESPIRiT's 0.166 — it fits what it was given slightly *better*. Nothing is to be won by pushing
   the data term.
2. **The whole gap is unsampled high spatial frequency.** Of the 0.81 total gap to L1-ESPIRiT, 0.71
   sits in the 0.50–1.00 radius band and 0.13 in 0.25–0.50; below 0.25 MC-GS is the better arm
   (0.013 + 0.247 against 0.015 + 0.268). This is the signature of a prior that is too smooth, or of
   a representation that cannot carry that band — which is what rounds 1 and 2 test, in that order.
3. **A bound on the frequency-shaping lever.** The sampled positions carry 1.8 % of the error and the
   measured zeros 0.45 %, so `kspace_high_weight`, `kspace_center_weight` and anything else that only
   reweights the fit to sampled data cannot buy more than about 0.08 dB. That lever moves from
   "round 5" to "not worth a run on its own".

T2s shows the same structure (98 % of the error unsampled); there the three arms are within 1.5 % of
each other in every band, which is why those three cells are near-ties.

## Round log

### Round 1 — λ recalibration on this subject (2026-09-14)

Four runs, one knob each, everything else at P10; joint arm only (the singles are re-run for the
finalists). Queues `scripts/p3_round1_{a,b}.txt` of the topic, two scheduler instances with disjoint
card lists (`ALLOWED="1 0 2" TAG=r1a`, `ALLOWED="3 4 5 7" TAG=r1b`), one training per card.

| run | change | verdict |
|---|---|---|
| `p3_wavT1_006` | T1w wavelet 0.12 → 0.06 (T2s pinned at 0.10) | **T1w +0.136 dB** (37.855 → 37.991), HFEN 0.0922 → 0.0903, sharp 0.813 → 0.819, SSIM flat; T2s unmoved as designed. min_margin −0.9744 → −0.8389 |
| `p3_wavT1_003` | T1w wavelet 0.12 → 0.03 | back to baseline: T1w 37.855 (HFEN 0.0912, sharp 0.821). min_margin −0.9751 — the wavelet optimum is **interior, near 0.06** |
| `p3_geo_006` | edge-TV λ_geo 0.10 → 0.06 (shared by both channels) | **T1w +0.260 dB** (→ 38.116), SSIM +0.0027, HFEN 0.0922 → 0.0852, sharp 0.813 → 0.836; T2s PSNR −0.047 but SSIM +0.0010, HFEN −0.0011, sharp 0.759 → 0.809. min_margin −0.7139 |
| `p3_geo_003` | edge-TV λ_geo 0.10 → 0.03 | pending |

λ_geo is the stronger of the two knobs — 0.260 dB against the wavelet's 0.136 — and it is the one
that moves HFEN (0.0922 → 0.0852 against a 0.0756 bar). On T2s it trades 0.047 dB of PSNR for better
SSIM, HFEN and sharpness, which is the shared-knob cost the queue comment predicted; whether that
trade is acceptable depends on where the re-tuned GS-single lands, since it sets all three T2s bars.

Round 0's reading holds: the prior *is* too heavy for this exam, and unloading it moves T1w in the
right direction on PSNR, HFEN and sharpness at once. But halving the weight bought 0.136 dB of the
0.975 needed, so λ alone is not going to close the gap — which is why the ceiling run is queued.

The run log's `[wavelet] lambda=... (plain ..., contrast ...)` line is the proof the per-contrast
override reached the trainer: `p3_wavT1_006` printed `lambda=0.1 (plain 0.06, contrast 0.1)`.

### Planned for round 2 — the representation ceiling

Round 0 says the contest is whether the cloud can carry the 0.50–1.00 band at all. Before sweeping
capacity blindly, fit the **fully sampled** ground truth with the same 3.6M-point cloud (no
undersampling, same iterations): its head PSNR is the ceiling that no amount of λ tuning can pass at
that capacity. If the ceiling is near 38.8 dB, capacity is the binding constraint and round 2 is the
whole campaign; if it is far above, the prior weights are, and round 1's direction is the right one.
One GPU run, and it settles which of the two round-0 explanations is the operative one.
