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

## Round log
