# CE AF16 three-axis autoresearch (program, 2026-11-07)

Human layer of a Karpathy-style loop. The user sets the question, the metric and the scope. The agent
implements, runs, logs and proposes the next hypotheses. `threeaxis.py` is the driver. `batchN.json` files
are the hypothesis queues.

## Research question

Can the proposed MC-GS strictly beat all ten opponents on all three anatomical axes, on both the pre-contrast
(平扫) and post-contrast (增强) scans?

The opponents are ZF, CG-SENSE, CS-SENSE, Joint SENSE, L1-SPIRiT, ZS-SSDU, 3D-DIP, 3D-INR, MC-INR and GS-single.

## Success criterion (user, 2026-11-07)

On each axis — transverse (a0), coronal (a1) and sagittal (a2) — at least one frozen lesion-core plane must
meet two conditions:

- MC-GS strictly wins all six of pre/post × PSNR/SSIM/NMSE against every opponent.
- This holds on both development seeds, 42 and 7.

The planes come from the evaluator's `lesion_planes` block: 59 planes, 17/24/18 per axis. Each must contain at
least 150 px of lesion core. Metrics are computed on the head mask of the whole plane, not on the lesion ROI.
A tie is not a win.

## Metric (single number, higher is better)

```
M3(seed) = min over axes of ( max over that axis's lesion planes of score_six )
M3       = min(M3(42), M3(7))
```

`score_six` is the evaluator's own number: the smallest relative margin over the six conditions against the
best opponent. Success means M3 > 0. It must then be confirmed on seeds 201 and 207 and pass the visual check
(the three-plane pre/post figure).

## Starting point (mined from 875 scored outputs, 2026-11-07)

- Keeper 0: `FV06_lambda_nl_plain_0.5`, M3 = −0.12704.

  | seed | transverse (a0) | coronal (a1) | sagittal (a2) |
  |---|---|---|---|
  | 42 | −0.1270, plane 196, 0/6 | +0.0192, plane 130, 6/6 | −0.0195, plane 43, 0/6 |
  | 7 | −0.1270 | +0.0233 | −0.0175 |

- **Transverse is the bottleneck.** No scored output ever exceeded −0.109 on it. That value came from the
  19-member ensemble E4; the best single run reached −0.1195.
- **The limiter is always ZS-SSDU**, by about 0.55 dB PSNR (pre 36.8 vs 37.3, post 38.7 vs 39.3).
- **Why transverse:** readout runs along array axis 0, so a transverse plane is the phase-encode plane and
  carries the full 2-D AF16 aliasing.
- **Knob tuning has saturated on transverse.** About 90 one-knob runs since the BM4D prior was added stayed
  within −0.127 to −0.137. Adding the non-local BM4D prior itself (C47 → C54) was the only large step
  (−0.20 → −0.137). Densification, longer training, DC weights, TV/wavelet weights and LR were flat.
- **Diagnosis (2026-11-06): MC-GS over-smooths.** It keeps 59–66 % of GT high-frequency energy, against
  91–97 % for ZS-SSDU.

## Scope (user, 2026-11-07): method-level changes + knobs

**Allowed:**

- Recipe knobs.
- Trainer keys the frozen trainer already reads, such as `nl_sharpen`.
- Non-local target variants registered at runtime, in the same process. This follows the precedent of
  `bm4d_pos`.

**Unchanged:**

- Frozen trainer files, the evaluator and its hashes.
- Case, masks, CSMs, references and opponents.
- Normalisation, the 625-step stage-2 budget from the iteration-4500 bases, and relax α.

**Never:**

- GT or ROI as a training input.
- Post-hoc denoising of the output.
- Ensembles or opponent outputs.
- Plane cherry-picking beyond the frozen plane set.
- Changing the metric to manufacture a success.

## Loop rules

1. **Build:** each queue item is the current keeper plus one change (`loss_edits`, `edits` or `variant`).
   Every item has a written hypothesis.
2. **Screen:** seed 42 runs first, about 15 min of training plus a few minutes of scoring.
3. **Confirm:** seed 7 runs only if M3(42) beats the keeper's M3(42) by more than 1e-4.
4. **Keep rule:** keep the item if M3 > keeper M3 + 1e-4. The keeper then moves greedily.
5. **Success path:** a keep with M3 > 0 runs confirmation seeds 201 and 207. If both are > 0, the loop writes
   `STOP` and waits for the user's visual check.
6. **Ledger:** `ROOT/experiments.jsonl` and `results.tsv`. Attempts are never resumed; a retry gets a new id.
7. **Stop:** a worker stops after three consecutive infrastructure failures, when its queue is exhausted, or
   when `ROOT/STOP` exists. After each batch the agent analyses the results and proposes the next batch.
   Changing direction outside the approved scope needs the user's approval.
8. **Compute:** GPUs 2 and 7 only, one worker each, each launched with explicit `--gpu`. GPU 4 stays empty;
   GPUs 0 and 1 belong to the user's other project.

`ROOT = /data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_threeaxis_20261107`

Git: experiments are queue entries (data), so they are not code branches. Code and queues are committed on
the local dev branch before each launch. The source digest of every run is in its `spec.json`.

## Hypothesis families (priority order)

| family | rationale | batch |
|---|---|---|
| A. Sharper NL target (`nl_sharpen` α) | directly attacks the high-frequency deficit; BM4D alpha-rooting boosts detail coefficients | B1 |
| B. Joint pre/post block matching | uses the shared anatomy (MC-GS's thesis) for grouping; same mask, so the gain is in grouping, not in data | B1 |
| C. NL weight × sharpening | if A helps, raise λ_nl to pull harder toward the sharper target | B2 if A works |
| D. Transverse-aware BM4D blocks | aliasing is incoherent in the phase-encode plane but coherent along readout; anisotropic blocks or search | B2/B3 |
| E. Sigma per contrast | pre and post have different SNR and limiters | later |
| F. Base retraining with the NL prior from stage 1 | the 4500-step base was trained without it; stage 2 may be too short to undo the smoothing | expensive, ask first |

## Findings so far (2026-11-08 03:30 CST)

Keeper chain (M3, min over seeds 42/7 of the worst axis's best-plane S6):

| keeper | change | M3 |
|---|---|---|
| FV06 (keeper0) | — | −0.12704 |
| TA02 | joint pre/post block matching | −0.12620 |
| TB06 | BM4D step 3 → 2 | −0.12467 |
| TD01 | dc_target: BM4D of the data-consistent image (frozen `hard_dc_combine`), on TA02 | −0.12293 |
| TC02 | joint guide σ 1.5 | −0.12236 |
| TE01 | + BM4D step 2 | −0.12137 |

Coronal passes 6/6 on both seeds; sagittal plane 43 reaches 1/6. Transverse is the limiter. On plane 196 the keeper
is 0.52 dB short of ZS-SSDU on PSNR, pre and post alike. The limiter metric is post NMSE.

Negative results:
- Sharpening the target hurts monotonically (TA01, TA03).
- An abrupt Gaussian shrink breaks the fit (TB01).
- Repeated NL refreshes over-smooth, both annealed (TB05, −0.1448) and with the DC target (TD03, −0.2765).
- A higher NL weight changes nothing (TC03, TC04).
- A higher σ_rel hurts (TD02).
- The export relaxation alpha must stay 1 (`probe-relax`: alpha 0.95/0.85/0.7 all lower).
- BM4D refilter (`denoise_residual`) is broken in bm4d 4.2.5.

Two diagnostics settle where the gap is:
- **k-space error (`diag_kspace.py`):** FV06 → TE01 reduces the transverse magnitude-error energy by only 0.4%. The
  gap to ZS-SSDU is 13%, almost all at unsampled (ky, kz) in |f| 0.2–0.5. Pre and post share one mask (sim/gt.npz
  mask2d), so neither contrast holds the other's missing samples.
- **BM4D ceiling (`diag_target_ceiling.py`):** this probe is post-hoc and not admissible. BM4D of TE01's final
  export, blended at w 0.5/1 with σ_rel 0.01/0.02, never improves the transverse axis: the best a0 is −0.1213 vs
  −0.1204 as exported. The missing transverse 0.5 dB is wrong or absent unsampled high frequency, not noise, so
  denoising the result cannot supply it.
- **Correction (03:50):** the dose of the data-consistency gain does move the transverse axis. TE03 (gain 2, on
  TE01) is kept at M3 −0.11809 (+3.3e-3); s42 a0 −0.1180, s7 sagittal −0.0090. Over-relaxing the measured residual
  into the BM4D input acts before the fit, unlike a post-hoc denoise. Batch 7 continues the dose (gain 3, gain 4).
  The earlier reading, that the BM4D family sits near its transverse ceiling, was premature.
- **Gain ladder (06:45):** all runs at σ_rel 0.0225, which TC06 kept at gain 2 with M3 −0.11623.

  | gain | M3 | notes |
  |---|---|---|
  | 4 (TF02) | −0.10968 | |
  | 6 (TH01) | −0.10394 | |
  | 8 (TH02) | **−0.10260** | coronal and sagittal pass 6/6 on both dev seeds |
  | 12 (TI01) | s42 −0.1106 | screen discard |

  On transverse plane 196, the PSNR gap to ZS-SSDU (pre/post, dB):

  | gain | pre | post |
  |---|---|---|
  | 4 | −0.460 | −0.474 |
  | 6 | −0.443 | −0.451 |
  | 8 | −0.446 | −0.432 |
  | 12 | −0.480 | −0.421 |

  The gain is a gradient step x + g·Aᴴ(y − Ax) on the DC loss before BM4D. It compensates BM4D's shrinkage of the
  missed high frequency. Transverse plateaus about 0.44 dB behind ZS-SSDU. σ_rel 0.025 at gain 2 (TG01) and the wide
  in-plane search (TC07) hurt.
- **High-gain follow-ups (08:15):**
  - **TJ02 kept.** σ_rel 0.025 at gain 8 gives M3 −0.09651; transverse is −0.0965 with 1/6 wins. Sagittal falls back
    to −0.003.
  - **TK01 discarded.** A second gain-8 refresh diverges: pre |x_t − x|/|x| goes 0.135 → 0.82 → 3.8, because
    over-relaxation above 2 is unstable when iterated. Transverse still improved (−0.0994 vs −0.1026), while coronal
    and sagittal collapsed. `later_gain` (TM01/TM02) makes later refreshes a stable PnP step.
  - **TK02, re-test when transverse nears 0.** `lambda_nl_plain` 1 on TJ02 is neutral on transverse (s42 −0.0968 vs
    −0.0965, discarded by the M3 rule). It lifts sagittal to +0.0019 and coronal to +0.040.

## Final state (2026-11-08 11:35 CST, loop stopped at the user's request after the queue)

The goal is **not met**. M3 went from −0.12704 to **−0.08173**. Coronal and sagittal pass 6/6 on both dev seeds;
transverse does not. No confirmation run on seeds 201/207 was triggered, because M3 never reached > 0.

**Final keeper TN15:**
- **BM4D:** step 2, joint pre/post block matching with guide σ 1.5.
- **NL target:** dc_target with gain 14, i.e. BM4D(|x + 14·Aᴴ(y − Ax)|), formed once at 4501.
- **NL settings:** nl_sigma_rel 0.0275, nl_refresh 1000; λ_nl_plain 0.5, λ_nl_contrast 4.
- **Unchanged:** every other recipe value (FV06).

| seed | transverse (plane 196) | coronal | sagittal | whole head |
|---|---|---|---|---|
| 42 | −0.0817 (1/6) | +0.0152 (plane 112) | +0.0005 (plane 44) | −0.052 |
| 7 | −0.0807 (1/6) | +0.0134 (plane 112) | +0.0048 (plane 43) | −0.050 |

On transverse plane 196 the PSNR gap to ZS-SSDU narrowed from −0.52 to −0.33 dB (pre) and −0.35 dB (post); the
limiter is post NMSE/PSNR. Coronal margins shrink as gain rises (+0.03 → +0.015), and sagittal passes only narrowly.

Rest of the keeper chain after TE01:

| keeper | M3 |
|---|---|
| TE03 gain 2 | −0.11809 |
| TC06 σ 0.0225 | −0.11623 |
| TF02 gain 4 | −0.10968 |
| TH01 gain 6 | −0.10394 |
| TH02 gain 8 | −0.10260 |
| TJ02 σ 0.025 | −0.09651 |
| TL02 gain 10 | −0.09176 |
| TN11 gain 12 | −0.09068 |
| TN13 σ 0.0275 | −0.08657 |
| **TN15 gain 14** | **−0.08173** |

Ridge discards:

| item | gain | σ_rel | M3 |
|---|---|---|---|
| TI01 | 12 | 0.0225 | −0.1106 |
| TI02 | 16 | 0.0225 | −0.1357 |
| TL01 | 8 | 0.0275 | −0.1042 |
| TN12 | 10 | 0.0275 | −0.0943 |
| TN14 | 14 | 0.025 | −0.0958 |
| TN16 | 14 | 0.03 | −0.0908 |

Any second target hurts: TK01 −0.0994 at gain 8 (transverse only; the other axes collapse), TM01 −0.148 at
later_gain 1, TM02 −0.139 at later_gain 2. The guide σ of 2.5 (TJ01) and λ_nl_plain 1 (TK02) are neutral on transverse.

The ledger holds 52 rows: 15 keep, 26 discard, 10 skipped, 1 control.

**Reading:**
- The single over-relaxed data-consistency step before BM4D is the only lever that moved transverse materially.
- It climbs a gain × σ ridge whose steps have shrunk to about 4e-3 per point, while coronal margins erode.
- The remaining ~0.33 dB transverse deficit is unsampled phase-encode high frequency. Closing it within this case
  would need either many more ridge points (gain 16+/σ 0.03, untested at the joint optimum) or a prior that can
  inpaint unsampled (ky, kz), such as a learned one. That second option needs the user's decision.

## Robustness follow-up (batch 19, 2026-11-08 14:25–15:21 CST)

After the branch review the user asked to run the two suggested checks. Driver change (commit f8569d0): an item with
`"screen": false` runs both dev seeds before the unchanged keep rule, and a control item may give `reference_scores`
to record per-seed and per-axis deviations. ROOT/STOP was set aside for the batch and rewritten afterwards.

**TO01_repeat_tn15** (control, GPU 7 = TN15's GPU). Not bit-identical, but every axis reproduces within 2.5e-4:

| seed | M3 | Δ vs TN15 | transverse | coronal | sagittal |
|---|---|---|---|---|---|
| 42 | −0.08148 | +2.5e-4 | −0.0815 @196 | +0.0151 @112 | +0.0005 @42 |
| 7 | −0.08063 | +4.9e-5 | −0.0806 @196 | +0.0135 @112 | +0.0046 @43 |

The s42 sagittal pass survives one rerun, but its margin is still only about twice the run-to-run noise.

**TO02_lambda_nl_plain_1_on_tn15** (TK02 on TN15, unscreened). Status `discard`: M3 −0.08174 (Δ −8e-6, a tie within
noise; transverse limits M3). Every other axis improves on both seeds:

| seed | transverse | coronal (passing planes) | sagittal (passing planes) | whole head |
|---|---|---|---|---|
| 42 | −0.0817 @196 | +0.0162 @121 (13, was 8) | +0.0098 @45 (7, was 2) | −0.0409 (was −0.052) |
| 7 | −0.0805 @196 | +0.0158 @112 (12, was 5) | +0.0098 @45 (7, was 4) | −0.0395 (was −0.050) |

The sagittal margin rises from +0.0005 to +0.0098, about 40× the noise. On transverse plane 196 the pre PSNR gap to
ZS-SSDU narrows to about −0.27 dB, but the limiter (post NMSE, ZS-SSDU, about −0.35 dB) does not move.

**Reading:** λ_nl_plain 1 gives the same M3 as TN15 with robust coronal and sagittal passes and a better whole head. Under
the user's keep rule (M3 > keeper + 5e-4) it is not kept. Making TO02 the keeper on a tie needs the user's decision.
The keeper stays TN15. The ledger holds 54 rows: 15 keep, 27 discard, 10 skipped, 2 control.

## Commands

```sh
# from mcgs/
/usr/bin/python3 ../gpu-tools/gpu.py run --cpu -- @python autoresearch/ce_af16_threeaxis_20261107/threeaxis.py selftest --queue autoresearch/ce_af16_threeaxis_20261107/batch1.json
/usr/bin/python3 ../gpu-tools/gpu.py run --gpu 2 -- @python autoresearch/ce_af16_threeaxis_20261107/threeaxis.py worker --gpu 2 --queue autoresearch/ce_af16_threeaxis_20261107/batch1.json
/usr/bin/python3 ../gpu-tools/gpu.py run --gpu 7 -- @python autoresearch/ce_af16_threeaxis_20261107/threeaxis.py worker --gpu 7 --queue autoresearch/ce_af16_threeaxis_20261107/batch1.json
/usr/bin/python3 ../gpu-tools/gpu.py run --cpu -- @python autoresearch/ce_af16_threeaxis_20261107/threeaxis.py summary
```

## Disclosure

This is single-case, GT-scored adaptive development. The AF16 GT is a CS-SENSE (TV) reconstruction of real
AF5.33 data, re-simulated with the same CSM, which is an inverse crime. The recipe is tuned against the GT
evaluator, while ZS-SSDU used no GT. A pass is evidence on this case only.
