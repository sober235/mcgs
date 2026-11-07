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
