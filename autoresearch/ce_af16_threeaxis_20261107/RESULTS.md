# CE AF16 pre (平扫) + post (增强): three-axis autoresearch results (2026-11-07 – 11-08)

**Status: goal not met. The loop is stopped.**

The final keeper is `TN15_gain_14_sigma_0.0275`, with **M3 = −0.08173**. The run started from FV06 at −0.12704, a gain of
+0.0453.

- **Pass:** coronal and sagittal win 6/6 on both development seeds.
- **Fail:** transverse does not; it wins 1/6 at best.
- **Not run:** the confirmation on seeds 201/207 and the visual check, because M3 never became positive.

`program.md` in this folder has the loop design, and `threeaxis.py` is the driver. The ledger and run directories are on
the server under `ROOT = /data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_threeaxis_20261107`.

## Goal and metric

**Question:** does MC-GS strictly beat all ten opponents on all three anatomical axes, for both the pre-contrast and the
post-contrast scan?

The ten opponents are ZF, CG-SENSE, CS-SENSE, Joint SENSE, L1-SPIRiT, ZS-SSDU, 3D-DIP, 3D-INR, MC-INR and GS-single.

**Success criterion (set by the user):**

- Each axis must have at least one frozen lesion-core plane where MC-GS strictly wins all six conditions: pre/post ×
  PSNR/SSIM/NMSE. A tie is not a win.
- This must hold on both dev seeds, 42 and 7.
- The planes come from the evaluator's `lesion_planes` block: 59 planes, 17/24/18 per axis, each with ≥ 150 px of lesion
  core. Metrics use the head mask of the whole plane.

**Metric (one number, higher is better):**

```
score_six(plane) = smallest relative margin over the six conditions against the best opponent (evaluator's number)
M3(seed)         = min over axes of ( max over that axis's lesion planes of score_six )
M3               = min(M3(42), M3(7))        success: M3 > 0, then confirm on seeds 201/207 + visual check
```

**Scope (set by the user):**

- **Allowed:** method changes and knobs. Non-local (NL) target variants are registered at runtime, in the same process.
- **Unchanged:** the frozen trainer, evaluator, masks, CSMs, references and opponents; normalisation; the 625-step stage-2
  budget from the iteration-4500 bases; relax α = 1.
- **Not allowed:** GT or ROI as training input, post-hoc denoising, ensembles, and plane cherry-picking.

## Answer: final keeper TN15

| axis | seed 42 | seed 7 | criterion |
|---|---|---|---|
| transverse (a0) | −0.0817, plane 196, 1/6 | −0.0807, plane 196, 1/6 | ✗ |
| coronal (a1) | +0.0152, plane 112, 6/6 | +0.0134, plane 112, 6/6 | ✓ |
| sagittal (a2) | +0.0005, plane 44, 6/6 | +0.0048, plane 43, 6/6 | ✓ (thin margin) |
| whole head S6 (not part of the criterion) | −0.052 | −0.050 | — |

**What blocks transverse:** on transverse plane 196, MC-GS still trails ZS-SSDU in PSNR, by about 0.33 dB (pre) and 0.35 dB
(post). At the start of the loop the gap was 0.52 dB. The binding condition is post NMSE/PSNR.

**TN15 recipe:** FV06 plus the changes below; every other value is unchanged from FV06.

| component | setting |
|---|---|
| NL denoiser | `bm4d_pos` with BM4D profile step 2 (`step_ht`, `step_wiener` = 2,2,2) |
| block matching | joint pre/post block matching, guide σ 1.5 |
| NL target | `dc_target`, gain 14: target = BM4D(\|x + 14·Aᴴ(y − Ax)\|), formed once at iteration 4501 |
| NL knobs | `nl_sigma_rel` 0.0275, `nl_refresh` 1000 (a single target), `lambda_nl_plain` 0.5, `lambda_nl_contrast` 4 |

In the `dc_target` formula:

- `x` is the current render.
- `x + Aᴴ(y − Ax)` is the frozen `data.known_zero.hard_dc_combine`: measured k-space is put back into the sampled
  locations, and the image is coil-combined with the frozen CSMs.
- The gain over-relaxes this data-consistency step before BM4D.

## Keeper chain

| # | keeper | change (one per step) | M3 | Δ vs previous keeper |
|---|---|---|---|---|
| 0 | `FV06_lambda_nl_plain_0.5` | starting point, mined from 875 scored outputs | −0.12704 | — |
| 1 | `TA02_joint_bm` | joint pre/post block matching | −0.12620 | +0.00084 |
| 2 | `TB06_bm4d_step2` | BM4D step 3 → 2 | −0.12467 | +0.00153 |
| 3 | `TD01_dc_target` | NL target = BM4D of the data-consistent image (built on TA02) | −0.12293 | +0.00174 |
| 4 | `TC02_joint_guide_1.5` | joint guide σ 1.5 | −0.12236 | +0.00057 |
| 5 | `TE01_dc_bm4d_step2` | + BM4D step 2 | −0.12137 | +0.00099 |
| 6 | `TE03_dc_gain_2` | DC gain 2 | −0.11809 | +0.00328 |
| 7 | `TC06_nl_sigma_0.0225` | σ_rel 0.02 → 0.0225 | −0.11623 | +0.00186 |
| 8 | `TF02_dc_gain_4` | gain 4 | −0.10968 | +0.00655 |
| 9 | `TH01_dc_gain_6` | gain 6 | −0.10394 | +0.00574 |
| 10 | `TH02_dc_gain_8` | gain 8 | −0.10260 | +0.00134 |
| 11 | `TJ02_nl_sigma_0.025_high_gain` | σ_rel 0.025 at gain 8 | −0.09651 | +0.00609 |
| 12 | `TL02_dc_gain_10` | gain 10 | −0.09176 | +0.00475 |
| 13 | `TN11_gain_12_sigma_0.025` | gain 12 | −0.09068 | +0.00108 |
| 14 | `TN13_gain_12_sigma_0.0275` | σ_rel 0.0275 at gain 12 | −0.08657 | +0.00411 |
| 15 | **`TN15_gain_14_sigma_0.0275`** | gain 14 at σ_rel 0.0275 | **−0.08173** | +0.00484 |

**Transverse plane 196, PSNR gap to ZS-SSDU (dB):**

| keeper | pre | post |
|---|---|---|
| FV06 (start) | ≈ −0.55 | ≈ −0.55 |
| TE01 | −0.52 | −0.52 |
| TF02 (gain 4) | −0.460 | −0.474 |
| TH01 (gain 6) | −0.443 | −0.451 |
| TH02 (gain 8) | −0.446 | −0.432 |
| TN15 (gain 14, σ 0.0275) | ≈ −0.33 | ≈ −0.35 |

## What worked

- **The gain on the data-consistency step, together with σ_rel.**
  - This was the only lever that moved transverse materially. The NL target is BM4D of a single over-relaxed
    data-consistency step, x + g·Aᴴ(y − Ax). It is formed once, before the fit, and compensates BM4D's shrinkage of
    the missed high frequency.
  - The best σ_rel rises with the gain, so the gain and σ_rel trace a ridge:

    | gain | best σ_rel | M3 |
    |---|---|---|
    | 8 | 0.025 | −0.0965 |
    | 12 | 0.0275 | −0.0866 |
    | 14 | 0.0275 | −0.0817 |

- **Smaller structural gains:**
  - joint pre/post block matching, best at guide σ 1.5 (2.5 is neutral);
  - BM4D step 2.

**Gain × σ_rel ridge.** Each cell gives the M3, or the seed-42 screen value marked "(s42)". A blank cell was not run.

| gain \ σ_rel | 0.0225 | 0.025 | 0.0275 | 0.03 |
|---|---|---|---|---|
| 8 | −0.1026 | **−0.0965** | −0.1042 (s42) | |
| 10 | | **−0.0918** | −0.0943 (s42) | |
| 12 | −0.1106 (s42) | **−0.0907** | **−0.0866** | |
| 14 | | −0.0958 (s42) | **−0.0817** | −0.0908 (s42) |
| 16 | −0.1357 (s42) | | | |

## What did not work

| idea | runs | result |
|---|---|---|
| sharpen the NL target (`nl_sharpen`) | TA01, TA03 | hurts monotonically (−0.145, −0.162) |
| abrupt Gaussian shrink (`init_scale` 0.85) | TB01 | breaks the fit (−1.31) |
| repeated NL refreshes, annealed or with the DC target | TB05, TD03 | over-smooth (−0.145, −0.277) |
| a second DC target at gain 8 | TK01 | diverges, \|x_t − x\|/\|x\| 0.135 → 0.82 → 3.8; transverse improves, but coronal and sagittal collapse |
| a second DC target at a stable `later_gain` of 1 or 2 | TM01, TM02 | over-smooths (−0.148, −0.139) |
| higher NL weight (`lambda_nl_contrast` 8, `lambda_nl_plain` 1) | TC03, TC04, TK02 | neutral on transverse |
| σ_rel away from the ridge | TC05, TD02, TG01, TL01 | lower |
| wide in-plane BM4D search | TC07 | lower |
| export relax α < 1 (`probe-relax`: 0.95 / 0.85 / 0.7) | probe | all lower; α = 1 (hard DC) is best |
| BM4D refilter (`denoise_residual`) | — | broken in bm4d 4.2.5 (NaN at block size 2) |

TK02 (`lambda_nl_plain` 1) is neutral on transverse but lifts sagittal to +0.0019 and coronal to +0.040. It is worth
re-testing once transverse is no longer the limiter.

## Diagnostics: where the transverse gap is

- **k-space error (`diag_kspace.py`):**
  - From FV06 to TE01, the transverse magnitude-error energy falls by only 0.4 %.
  - The gap to ZS-SSDU is about 13 %, almost all at unsampled (ky, kz) with |f| 0.2–0.5.
  - Readout is array axis 0, so a transverse plane is the phase-encode plane and carries the full 2-D AF16 aliasing.
  - Pre and post share one mask (`sim/gt.npz` `mask2d`), so neither contrast holds the samples the other is missing.
- **BM4D ceiling (`diag_target_ceiling.py`):**
  - This probe is post-hoc and inadmissible as a method output.
  - BM4D of TE01's final export (σ_rel 0.01 or 0.02, blend weight 0.5 or 1) never improves transverse. The best was
    −0.1213, against −0.1204 for the export itself.
  - So denoising the result cannot supply the missing detail.
  - The gain result above shows that over-relaxing the measured residual before the fit does help. The earlier reading
    that the BM4D family was at its ceiling was premature.

## Reading and next options

- The DC gain × σ_rel ridge is still rising, but each step now adds only about 4e-3, while M3 needs about +0.08.
- The coronal margin erodes as the gain rises, from about +0.03 to +0.015, and sagittal passes only narrowly.
- The remaining ~0.33 dB transverse deficit is unsampled phase-encode high frequency.

There are two ways forward:

1. **Continue the ridge**, for example gain 16 at σ_rel 0.03. This is cheap, but unlikely to reach M3 > 0 on its own.
2. **Add a prior that can inpaint unsampled (ky, kz)**, for example a learned one. This is outside the agreed scope and
   needs the user's decision. The same applies to family F: retraining the base with the NL prior from stage 1.

## Ledger (all 52 rows)

Status counts: 15 keep, 2 discard (seed 7 ran, but the M3 margin was not met), 24 screen_discard (seed 42 only),
10 skipped, 1 control.

How to read the table:

- Each cell shows the best-plane S6, the plane index and the number of the six conditions won.
- For a screen discard, the M3 column shows the seed-42 minimum, marked "(s42)".
- A skipped item was withdrawn before running. Some of these were superseded. Others were unsafe compositions: for
  example, TN01/TN02 would have inherited an iterated high gain, and were re-issued as TN11/TN12.
- The control TA00 reruns FV06 under the new driver. It differs by about 2e-4, within the nondeterminism.
- Keep margin: 1e-4 for batch 1 and 5e-4 from batch 2 on.

| # | id | status | M3 | s42 transverse | s42 coronal | s42 sagittal | s7 transverse | s7 coronal | s7 sagittal |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `TA00_ctrl_fv06` | control_differs | -0.12724 | -0.1272 @196 (0/6) | +0.0192 @130 (6/6) | -0.0196 @43 (0/6) | — | — | — |
| 2 | `TA01_nl_sharpen_1.25` | screen_discard | -0.1447 (s42) | -0.1447 @196 (0/6) | +0.0165 @130 (6/6) | -0.0285 @43 (0/6) | — | — | — |
| 3 | `TA04_nl_sharpen_1.1` | skipped | — | — | — | — | — | — | — |
| 4 | `TA05_nl_sharpen_2` | skipped | — | — | — | — | — | — | — |
| 5 | `TA03_nl_sharpen_1.5` | screen_discard | -0.1622 (s42) | -0.1622 @196 (0/6) | +0.0057 @130 (6/6) | -0.0388 @43 (0/6) | — | — | — |
| 6 | `TB02_sigma_anneal_r125_d0.6` | skipped | — | — | — | — | — | — | — |
| 7 | `TB03_bm4d_step2` | skipped | — | — | — | — | — | — | — |
| 8 | `TB04_init_scale_0.7` | skipped | — | — | — | — | — | — | — |
| 9 | `TB01_init_scale_0.85` | screen_discard | -1.3110 (s42) | -1.3110 @194 (0/6) | -0.7432 @115 (0/6) | -0.9683 @46 (0/6) | — | — | — |
| 10 | `TB07_init_scale_0.7` | skipped | — | — | — | — | — | — | — |
| 11 | `TA02_joint_bm` | keep | **-0.12620** | -0.1262 @196 (0/6) | +0.0226 @130 (6/6) | -0.0194 @43 (0/6) | -0.1259 @196 (0/6) | +0.0254 @130 (6/6) | -0.0170 @43 (0/6) |
| 12 | `TB05_sigma_anneal_r125_d0.6` | screen_discard | -0.1448 (s42) | -0.1448 @196 (0/6) | +0.0145 @131 (6/6) | -0.0310 @50 (0/6) | — | — | — |
| 13 | `TB06_bm4d_step2` | keep | **-0.12467** | -0.1243 @196 (0/6) | +0.0231 @130 (6/6) | -0.0185 @43 (0/6) | -0.1247 @196 (0/6) | +0.0270 @130 (6/6) | -0.0152 @43 (0/6) |
| 14 | `TD01_dc_target` | keep | **-0.12293** | -0.1227 @196 (0/6) | +0.0263 @130 (6/6) | -0.0161 @43 (1/6) | -0.1229 @196 (0/6) | +0.0298 @130 (6/6) | -0.0138 @43 (1/6) |
| 15 | `TD02_dc_target_sigma_0.025` | screen_discard | -0.1300 (s42) | -0.1300 @202 (0/6) | +0.0154 @130 (6/6) | -0.0304 @45 (0/6) | — | — | — |
| 16 | `TC01_joint_guide_0.7` | screen_discard | -0.1226 (s42) | -0.1226 @196 (0/6) | +0.0260 @130 (6/6) | -0.0166 @43 (1/6) | — | — | — |
| 17 | `TD03_dc_target_refresh_125` | screen_discard | -0.2765 (s42) | -0.2765 @204 (0/6) | -0.1149 @129 (2/6) | -0.1586 @53 (0/6) | — | — | — |
| 18 | `TC03_lambda_nl_contrast_8` | screen_discard | -0.1228 (s42) | -0.1228 @196 (0/6) | +0.0255 @130 (6/6) | -0.0166 @43 (0/6) | — | — | — |
| 19 | `TC02_joint_guide_1.5` | keep | **-0.12236** | -0.1221 @196 (0/6) | +0.0266 @130 (6/6) | -0.0160 @43 (1/6) | -0.1224 @196 (0/6) | +0.0295 @130 (6/6) | -0.0136 @43 (1/6) |
| 20 | `TC04_lambda_nl_plain_1` | screen_discard | -0.1224 (s42) | -0.1224 @196 (0/6) | +0.0274 @130 (6/6) | -0.0161 @43 (1/6) | — | — | — |
| 21 | `TE01_dc_bm4d_step2` | keep | **-0.12137** | -0.1204 @196 (0/6) | +0.0271 @130 (6/6) | -0.0152 @43 (1/6) | -0.1214 @196 (0/6) | +0.0306 @130 (6/6) | -0.0122 @43 (1/6) |
| 22 | `TE02_dc_gain_1.5` | discard | -0.12130 | -0.1208 @196 (0/6) | +0.0289 @130 (6/6) | -0.0148 @43 (1/6) | -0.1213 @196 (0/6) | +0.0319 @130 (6/6) | -0.0124 @43 (1/6) |
| 23 | `TC05_nl_sigma_0.0175` | screen_discard | -0.1303 (s42) | -0.1303 @196 (0/6) | +0.0239 @130 (6/6) | -0.0178 @43 (1/6) | — | — | — |
| 24 | `TE03_dc_gain_2` | keep | **-0.11809** | -0.1180 @196 (0/6) | +0.0315 @130 (6/6) | -0.0121 @43 (1/6) | -0.1181 @196 (0/6) | +0.0348 @130 (6/6) | -0.0090 @43 (1/6) |
| 25 | `TE04_dc_gain_1.5_step2` | discard | -0.11964 | -0.1191 @196 (0/6) | +0.0296 @130 (6/6) | -0.0136 @43 (1/6) | -0.1196 @196 (0/6) | +0.0329 @130 (6/6) | -0.0103 @43 (1/6) |
| 26 | `TC06_nl_sigma_0.0225` | keep | **-0.11623** | -0.1161 @196 (0/6) | +0.0269 @130 (6/6) | -0.0176 @45 (1/6) | -0.1162 @196 (0/6) | +0.0306 @130 (6/6) | -0.0168 @43 (1/6) |
| 27 | `TF01_dc_gain_3` | screen_discard | -0.1161 (s42) | -0.1161 @196 (0/6) | +0.0341 @130 (6/6) | -0.0099 @43 (1/6) | — | — | — |
| 28 | `TG02_dc_gain_3_sigma_0.0225` | skipped | — | — | — | — | — | — | — |
| 29 | `TG01_nl_sigma_0.025` | screen_discard | -0.1229 (s42) | -0.1229 @202 (0/6) | +0.0206 @130 (6/6) | -0.0243 @45 (0/6) | — | — | — |
| 30 | `TF02_dc_gain_4` | keep | **-0.10968** | -0.1092 @196 (0/6) | +0.0338 @130 (6/6) | -0.0096 @43 (1/6) | -0.1097 @196 (0/6) | +0.0375 @130 (6/6) | -0.0076 @43 (1/6) |
| 31 | `TH01_dc_gain_6` | keep | **-0.10394** | -0.1038 @196 (0/6) | +0.0385 @130 (6/6) | -0.0024 @43 (3/6) | -0.1039 @196 (1/6) | +0.0414 @130 (6/6) | +0.0009 @43 (6/6) |
| 32 | `TH02_dc_gain_8` | keep | **-0.10260** | -0.1026 @196 (0/6) | +0.0288 @127 (6/6) | +0.0011 @43 (6/6) | -0.1000 @196 (1/6) | +0.0272 @132 (6/6) | +0.0014 @43 (6/6) |
| 33 | `TC07_wide_inplane_search` | screen_discard | -0.1080 (s42) | -0.1080 @196 (0/6) | +0.0354 @130 (6/6) | -0.0053 @43 (3/6) | — | — | — |
| 34 | `TI01_dc_gain_12` | screen_discard | -0.1106 (s42) | -0.1106 @196 (0/6) | +0.0116 @112 (6/6) | -0.0108 @42 (2/6) | — | — | — |
| 35 | `TI02_dc_gain_16` | screen_discard | -0.1357 (s42) | -0.1357 @196 (0/6) | -0.0145 @112 (3/6) | -0.0340 @56 (0/6) | — | — | — |
| 36 | `TJ01_joint_guide_2.5` | screen_discard | -0.1032 (s42) | -0.1032 @196 (0/6) | +0.0287 @127 (6/6) | +0.0007 @43 (6/6) | — | — | — |
| 37 | `TJ02_nl_sigma_0.025_high_gain` | keep | **-0.09651** | -0.0965 @196 (1/6) | +0.0328 @130 (6/6) | -0.0028 @45 (4/6) | -0.0960 @196 (1/6) | +0.0332 @130 (6/6) | -0.0026 @50 (4/6) |
| 38 | `TK01_dc_two_refreshes` | screen_discard | -0.1154 (s42) | -0.0994 @202 (0/6) | -0.1154 @113 (3/6) | -0.0396 @56 (0/6) | — | — | — |
| 39 | `TK02_lambda_nl_plain_1_high_gain` | screen_discard | -0.0968 (s42) | -0.0968 @196 (1/6) | +0.0401 @130 (6/6) | +0.0019 @45 (6/6) | — | — | — |
| 40 | `TL01_nl_sigma_0.0275` | screen_discard | -0.1042 (s42) | -0.1042 @202 (1/6) | +0.0315 @130 (6/6) | -0.0067 @41 (3/6) | — | — | — |
| 41 | `TL03_nl_sigma_0.03` | skipped | — | — | — | — | — | — | — |
| 42 | `TN01_gain_12_sigma_0.025` | skipped | — | — | — | — | — | — | — |
| 43 | `TN02_gain_10_sigma_0.0275` | skipped | — | — | — | — | — | — | — |
| 44 | `TL02_dc_gain_10` | keep | **-0.09176** | -0.0918 @196 (1/6) | +0.0242 @127 (6/6) | +0.0011 @43 (6/6) | -0.0910 @196 (1/6) | +0.0222 @127 (6/6) | +0.0039 @43 (6/6) |
| 45 | `TM01_second_refresh_gain_1` | screen_discard | -0.1484 (s42) | -0.1484 @204 (0/6) | -0.0081 @129 (5/6) | -0.0667 @53 (0/6) | — | — | — |
| 46 | `TM02_second_refresh_gain_2` | screen_discard | -0.1389 (s42) | -0.1389 @204 (0/6) | -0.0014 @129 (5/6) | -0.0586 @53 (0/6) | — | — | — |
| 47 | `TN11_gain_12_sigma_0.025` | keep | **-0.09068** | -0.0907 @196 (0/6) | +0.0183 @112 (6/6) | +0.0022 @42 (6/6) | -0.0890 @196 (1/6) | +0.0168 @112 (6/6) | +0.0028 @42 (6/6) |
| 48 | `TN12_gain_10_sigma_0.0275` | screen_discard | -0.0943 (s42) | -0.0943 @202 (1/6) | +0.0245 @127 (6/6) | -0.0021 @41 (4/6) | — | — | — |
| 49 | `TN14_gain_14_sigma_0.025` | screen_discard | -0.0958 (s42) | -0.0958 @196 (0/6) | +0.0134 @112 (6/6) | -0.0039 @42 (2/6) | — | — | — |
| 50 | `TN13_gain_12_sigma_0.0275` | keep | **-0.08657** | -0.0866 @196 (1/6) | +0.0181 @127 (6/6) | +0.0015 @45 (6/6) | -0.0863 @196 (1/6) | +0.0111 @121 (6/6) | +0.0011 @45 (6/6) |
| 51 | `TN16_gain_14_sigma_0.03` | screen_discard | -0.0908 (s42) | -0.0908 @202 (1/6) | +0.0052 @121 (6/6) | -0.0001 @41 (5/6) | — | — | — |
| 52 | `TN15_gain_14_sigma_0.0275` | keep | **-0.08173** | -0.0817 @196 (1/6) | +0.0152 @112 (6/6) | +0.0005 @44 (6/6) | -0.0807 @196 (1/6) | +0.0134 @112 (6/6) | +0.0048 @43 (6/6) |

## Files

| file | role |
|---|---|
| `program.md` | loop design: question, metric, scope, rules, hypothesis families, dated findings |
| `threeaxis.py` | driver: `selftest`, `worker`, `enqueue [--front]`, `summary` and `probe-relax`, plus the runtime NL variants `profile`, `joint_bm`, `dc_target`, `sigma`, `init_scale` and `stack` |
| `batch1.json` … `batch18.json`, `batch2b.json`, `batch16b.json` | hypothesis queues; each item is the current keeper plus one change |
| `diag_kspace.py` | per-band k-space error energy (CPU) |
| `diag_target_ceiling.py` | post-hoc BM4D ceiling probe (CPU, diagnostic only) |

**Files on the server (ROOT):**

- the ledger: `experiments.jsonl`, `results.tsv`;
- the keeper state: `keeper.json`, `keeper_history.json`;
- `runs/<id>/` with `spec.json` (which holds the source digest) and the scores;
- `probes/`;
- `STOP` (the loop finished at the user's request).

Commands, run from `mcgs/`:

```sh
/usr/bin/python3 ../gpu-tools/gpu.py run --cpu -- @python autoresearch/ce_af16_threeaxis_20261107/threeaxis.py selftest --queue autoresearch/ce_af16_threeaxis_20261107/batch1.json
/usr/bin/python3 ../gpu-tools/gpu.py run --gpu 2 -- @python autoresearch/ce_af16_threeaxis_20261107/threeaxis.py worker --gpu 2
/usr/bin/python3 ../gpu-tools/gpu.py run --cpu -- @python autoresearch/ce_af16_threeaxis_20261107/threeaxis.py summary
```

## Disclosure

- **Single case, scored against GT.** This is adaptive development on one case, tuned against the GT evaluator.
  ZS-SSDU used no GT.
- **Inverse crime.** The AF16 GT is a CS-SENSE (TV) reconstruction of real AF5.33 data, re-simulated with the same CSM.
- **Limited claim.** A pass, had it occurred, would be evidence on this case only.
