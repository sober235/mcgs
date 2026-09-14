# Design — pediatric MTP AF12 stage 3: tune MC-GS until it leads the table

Opened 2026-09-14. Branch `16yo-mtp`, worktree `/data0/congcong/code/GS/mcgs_peds_wt`.
Topic `/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro/` (its README is the
authoritative table). Stage 1 = the reference dataset, stage 2 = the nine-arm comparison with the
recipe transferred unchanged, which MC-GS did not lead. This stage tunes on this subject.

## 1. The question this stage answers

Stage 2 asked whether the transferred recipe leads here, and the answer was no. It could not
distinguish two causes: a joint prior that is wrong for this contrast pair (r = 0.57), or a working
point that was selected on a different subject, a different pair and a different resolution. This
stage removes the second cause by tuning on this subject, and reports whichever answer comes out.

The asymmetry it repairs is real and recorded in stage 2: L1-ESPIRiT's λ was swept on this subject
over 14 values and CS-SENSE's over 11, while the GS arms ran at the jiangyizhou AF12 P10 point with
nothing swept. The tuned numbers are therefore comparable with the classical arms, and the record
must say so; they are **not** comparable with ZS-SSDU, 3D-DIP, 3D-INR or MC-INR, which were not tuned.

## 2. Success criterion (user decision, 2026-09-14)

MC-GS first in all **six** cells of the primary table — head PSNR, head SSIM, head HFEN × T1w, T2s —
against **every** other arm, GS-single included.

Bars to clear, from `logs/rescore.json` (head inside partitions 12..92), with the arm that sets each:

| cell | bar | set by | MC-GS now | gap |
|---|---|---|---|---|
| T1w head PSNR | 38.830 | L1-ESPIRiT | 37.855 | **+0.975 dB** |
| T1w head SSIM | 0.9674 | L1-ESPIRiT | 0.9618 | +0.0056 |
| T1w head HFEN | 0.0756 | L1-ESPIRiT | 0.0922 | −0.0166 (18 % lower) |
| T2s head PSNR | 34.175 | GS-single | 34.160 | +0.015 |
| T2s head SSIM | 0.9052 | GS-single | 0.9047 | +0.0005 |
| T2s head HFEN | 0.1855 | GS-single | 0.1858 | −0.0003 |

The T1w column is the whole problem; T2s is three near-ties with its own single arm. Note what the
T1w numbers say about the mechanism: MC-GS trails L1-ESPIRiT on HFEN by 18 % and on sharpness by
0.813 vs 0.924. That is an over-smoothed reconstruction, which is what a wavelet and edge-TV weight
chosen for a 1 mm adult MPRAGE would do to a 0.7 mm pediatric GRE.

**GS-single is tuned alongside.** Every recipe MC-GS gets, GS-single gets, because they are the same
trainer with one contrast; only then does "joint beats single" mean anything. A recipe that wins the
six cells only because the single arm was left at the old working point does not count. The bar in a
cell is therefore the best of every other arm **at its own best available setting**, the tuned single
included, and it moves upward as the rounds go on — on T2s, where the single arm sets all three bars,
that is the whole contest.

## 3. Gauge and bookkeeping

- Judged on the **primary** table: head metrics inside the vendor's central 80 partitions (SPE 12..92),
  `scripts/rescore.py`'s region. Candidates are scored there by `scripts/score_candidate.py` (new,
  §6), so a run is compared on the same footing as the table it wants to enter.
- The full-volume numbers stay in the ledger as the cross-check, exactly as stage 2 kept both.
- Ledger `autoresearch_peds.tsv`, one row per run, written by `scripts/autoresearch_peds_row.py`;
  the six margins go in the row. Round log in `autoresearch_peds.md` (new, this stage's program).
- Nothing in `Results_20260908/Peds_TSC_MTP_AF12_retro/` is overwritten until a recipe wins. The
  stage-2 rows keep their keys (`gs_joint_ar`, `gs_single_ar`); a winner enters as a **new** arm
  (`gs_joint_p3` / `gs_single_p3`, labels "MC-GS (tuned)" / "GS-single (tuned)") through
  `scripts/add_gs_arm.py`, so the stage-2 result stays visible beside it.

## 4. Levers, in order

Each round changes one thing, at the P10 point otherwise. A lever is kept only if it loses none of
the six cells relative to the round's incumbent; ties are resolved on T1w head PSNR, the binding gap.

1. **λ recalibration on this subject** (expected to matter most). Wavelet weight per contrast
   (now T1w 0.12 / T2s 0.10) and the edge-TV geometry weight `lambda_geo` (now 0.10), swept down:
   λ_wav ∈ {0.03, 0.06, 0.09}, λ_geo ∈ {0.03, 0.06}. Scanned on the joint arm only; the singles are
   re-run for the finalists.
2. **Capacity and iterations.** 3.6M points at 10.1M voxels here against 11.5M voxels on jiangyizhou:
   try 6.0M points, and 6000 iterations against 4000. The AF16 phase-3 record found capacity a
   trade-off rather than a free gain, so this is a check, not an expectation.
3. **Joint coupling.** The jiangyizhou AF12 phase-3 pair that turned joint−single from −0.038 to
   +0.036: common-edge weight (scan 0.0005 .. 0.004 — the phase-1 lesson there was that the λ which
   made it look dead was 20× the useful one) and phase TV (0.01 .. 0.05). This is the lever aimed
   squarely at the T2s near-ties and at the T1w −0.231 against the single arm.
4. **Optimiser.** `rotation_lr` 0.008 (the phase-2 winner there), `scale_lr`, `warmup`.
5. **Frequency shaping.** `kspace_high` ∈ {1.6, 2.2, 3.0}, `edge_tv_kappa_q`, `edge_tv_wfloor`.

Rounds 1–2 are about the T1w gap to L1-ESPIRiT; rounds 3–5 about the margin over GS-single.

## 5. Stopping rule (user decision: run until the target is met or the levers are exhausted)

- **Met:** all six cells first. Then: re-run the singles at the winning point, register the arms,
  regenerate both README tables and the figures, write the record.
- **Exhausted:** rounds 1–5 done and the last two rounds each gained < 0.05 dB on T1w head PSNR.
  Then the stage-2 conclusion is upgraded from "the transferred recipe does not lead" to "tuned on
  this subject it still does not lead, and the gap is X dB", which is a stronger negative result and
  is worth the same write-up.
- Either way the ledger and the round log are committed, so a later session sees what was tried.

## 6. New code

- `scripts/score_candidate.py` (topic, new): score one run's saved volumes in the primary region and
  print the six margins. It reads `results_arc/<exp>_volumes.npz` (the complex volumes when the run
  saved them, `recT1*` = plain channel, `recFL*` = contrast channel), scores against
  `recon/gt_<c>_complex.npy` with `uih_mtp.rescore.score_region` — the same function and region the
  primary table uses — and takes the bars from `logs/rescore.json`, so a candidate cannot be judged
  against a bar that differs from the published table. Writes `results_arc/<exp>_primary.json`.
- `scripts/autoresearch_peds_row.py` (repo, exists): **its competitor set is wrong for this stage.**
  It currently excludes every GS arm from the bar ("any MC-GS variant never counts"), which was right
  when the question was "does MC-GS beat the classical and learned arms" and is wrong now that
  GS-single counts. Change: a competitor is any arm that is not a *joint* GS arm; `gs_single*` counts,
  `gs_joint*` does not. The row also gains the six primary-region margins, read from the candidate's
  `_primary.json` when it exists, with the full-volume numbers kept beside them as the cross-check.
- No change to the trainer. Every knob in §4 already exists as a flag or a config key.

## 7. Isolation and cost

- GPUs: any card with no compute process and utilisation ≤ 10 % for two consecutive polls, at most
  **two** of this line's trainings at once (user decision 2026-09-14, replacing the stage-2 rule of
  card 2 only). One training per card. The scheduler runs detached from the session.
- One 4000-iteration run is 65 minutes on a free card; a λ scan of six candidates is ~3.5 hours of
  wall clock on two cards. Rounds 1–5 with finalist singles are on the order of 20 hours.
- Writes go to `results_arc/` in the worktree and to the topic folder only when an arm is registered.

## 8. Caveats that must reach the record

- One subject, tuned on itself, no held-out subject exists on this line. The tuned row is honest
  only if labelled as tuned on this subject.
- The classical arms were λ-swept on the same subject, so MC-GS tuned here is comparable with them;
  ZS-SSDU, 3D-DIP, 3D-INR and MC-INR were not tuned and the tuned row is not a fair comparison
  against those four.
- The reference is GRAPPA-completed 2× data, not a fully sampled acquisition.
- `grain_ratio` on T2s rests on 664 white-matter voxels and is not quoted.
