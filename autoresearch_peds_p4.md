# Autoresearch — pediatric MTP AF12, stage 4: MC-GS at least 1 dB above every other arm

Asked 2026-09-14 (evening): start a Karpathy-style autoresearch loop so that MC-GS beats every
comparison method by 1 dB. This continues stage 3 (`autoresearch_peds.md`, record
`docs/superpowers/records/2026-09-14-16yo-mtp-stage3.md`, tag `handoff/2026-09-14-16yo-mtp-stage3`),
whose bar was "first in all six cells"; the bar here is stricter.

Ledger: the same `autoresearch_peds.tsv` (the runner writes it; stage-4 runs are named `p4_*`).
Queue: `Results_20260908/Peds_TSC_MTP_AF12_retro/scripts/p4_queue.txt`, scheduler `scripts/p4_sched.sh`
of the topic (one detached instance; log `logs/p4_sched.log`).

## The bar

- **Head PSNR in the primary region** (head inside partitions 12..92, `scripts/score_candidate.py`)
  **at least 1.00 dB above the best other arm, on T1w and on T2s.**
- "Other arm" = all nine other arms of the table, **GS-single included** (user, 2026-09-14), at its best
  available setting. Whatever generic lever MC-GS gets (a weight on one contrast, iterations,
  capacity, optimiser, phase TV), GS-single gets too, and the bar moves with it (stage-3 rule).
- Head SSIM and HFEN are reported beside it; a PSNR margin bought by losing either is said so.

Where it starts (primary region, from `logs/rescore.json` and the ledger):

| | best other arm | target | MC-GS now |
|---|---|---|---|
| T1w head PSNR | 38.848 (GS-single tuned; L1-ESPIRiT 38.830) | **39.848** | 38.516 at 4000 it, 38.588 at 6000 |
| T2s head PSNR | 34.175 (GS-single; L1-ESPIRiT 33.860) | **35.175** | 34.163 at 4000 it, 34.328 at 6000 |

## Why only the joint part can meet it

The margin over a GS-single that receives every generic lever is the joint-minus-single gap, which is
−0.33 dB on T1w and −0.01 dB on T2s at the stage-3 working point. Levers that act on one contrast by
itself move both arms and leave that gap where it is; they are only worth running to stay ahead of
L1-ESPIRiT. The rounds therefore go first to mechanisms that pass information between the contrasts.

## Round 0 — is there a dB left in this data? (CPU, 2026-09-14)

The reference is GRAPPA-completed, SENSE-combined data, so it carries noise, and the noise at the
unsampled positions is unpredictable by any method. Measured on the complex background far from the
head (MAD, per component): σ = 0.0044 (T1w), 0.0057 (T2s), gmax 2.20. The noise at the 17279 unsampled
in-support positions caps head PSNR near **54.6 dB (T1w) and 52.4 dB (T2s)**; the current error is
16–18 dB above that. What is missing is unrecovered signal, so noise does not rule the target out.
(Two other estimators were discarded: the k-space tail is still signal-dominated at the support edge,
and the finest Haar band sits on the measured-zero corners.)
Scripts and numbers: `peds_tsc_mtp_af12_retro/analysis_p4/ceiling_bg.{py,json}`, `ceiling_noise.{py,json}`.

## Round 0b — can T1w carry T2s's detail through a smooth ratio? No (CPU, 2026-09-14)

The one cross-contrast mechanism that could give T2s's unsampled high frequencies real information
from T1w is a signal model: both come from one MTP exam, so T2s = T1w · ρ with ρ a smooth map of the
T1 and T2* weighting. Tested on the ground truth before writing any code: ρ restricted to a low-pass
band on every axis, fitted by CG in the head, and T1w·ρ scored against T2s in the primary region.

| model | band per axis | head PSNR of the approximation |
|---|---|---|
| complex ρ | 1/16 | 20.29 |
| complex ρ | 1/8 | 21.89 |
| complex ρ | 1/4 | 24.52 (24.60 with 120 CG iterations: converged) |
| magnitude ratio, real r | 1/8 | 25.08 |
| magnitude ratio, real r | 1/4 | 26.96 |
| one global scalar | — | 19.09 |
| (current T2s reconstruction from its own data) | | 34.16 |

Even with the ground-truth T1w and a ρ that keeps a quarter of the band, the product explains T2s
10 dB worse than T2s's own undersampled data already do; the reverse direction is worse still
(T1w from T2s: 18.8 / 20.2 / 22.0). The two contrasts differ in fine detail far beyond any smooth
modulation, so the ratio coupling is dropped without a GPU run. What a joint prior can still use
here is where edges sit, not what the intensities are. Script and numbers:
`peds_tsc_mtp_af12_retro/analysis_p4/ratio_fast.{py,json}`, `ratio_check.{py,json}`.

## Levers queued (priority order, `scripts/p4_queue.txt`)

1. `p4_dcw02` — **who drives the shared geometry.** At the end of a c1 run the T2s data term is ~5x
   the T1w one, so the cloud is pulled mostly by the noisier contrast. `loss.dc_weight_contrast` (new,
   `losses/balance.py`, 2 tests, default bit-identical) scales T2s's data term; T2s's own regularisers
   get the same factor, so under Adam T2s's densities see no change and only the geometry split moves.
2. `p4_gwav` — joint (group) wavelet sparsity across the contrasts (new `loss.wav_group`, 6 tests).
3. `p4_dtvswap` — half of T2s's own edge-TV replaced by T1w-guided directional TV.
4. `p4_dcw05`, `p4_dtv005`, `p4_gwav_x14` — the second points of those three scans.

## Round log
