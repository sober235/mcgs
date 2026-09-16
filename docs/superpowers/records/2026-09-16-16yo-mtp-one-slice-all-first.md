# Record 2026-09-16: one slice on which MC-GS leads every comparison arm (16yo MTP, AF 12)

Branch `16yo-mtp`, worktree `/data0/congcong/code/GS/mcgs_peds_wt`, topic `Results_20260908/Peds_TSC_MTP_AF12_retro/`.
Session claude-e7. User, 2026-09-16: "you need to find one slice, mc-gs全部超过对比方法".

## 1. Search

`scripts/find_slice_all_first.py` (topic; mirrored here). Every slice of the three planes with at least 15 % head
pixels; per slice the figures' own panel gauge — head = |GT| > 0.05 · P99.9 of the volume, PSNR at the volume's max |GT|
over the head pixels, SSIM = 7x7 uniform-window map averaged over the head pixels, NMSE over the head pixels (NMSE
ranks as PSNR within one slice, so "all three" = PSNR and SSIM). MC-GS = `gs_joint_p3f`, the stage-3 final
configuration. Comparison arms = every counted table row at its table point (L1 = L1-SPIRiT), every registered
GS-single volume of the contrast and every results_arc single run of the contrast (the stage-4 rule: the bar on a slice
is the best any single reaches there): 16 arms on T1w (incl. `gs_single_ar`, `_p3`, `_p3f` and the five
`p3_single_T1w_*` / `peds_af12_single_T1w` runs), 10 on T2s. Earlier MC-GS versions are the same method, not
competitors. Output `logs/find_slice_all_first_{T1w,T2s}.json` (every slice, PSNR of every arm; SSIM where MC-GS
leads PSNR).

## 2. Counts (selection on the reported metric is illustration, not evidence — these are the evidence)

| contrast | slices | MC-GS first on PSNR / NMSE | first on PSNR, SSIM and NMSE | by plane (large sections, area ≥ 90 % of the plane's largest) |
|---|---|---|---|---|
| T1w | 578 | 40 | **29** | transverse 1 of 96 (102, a skull-base sliver, 0 large); coronal 9 of 256 (RO 30–40, the anterior/posterior caps, area ≤ 0.41, 0 large); **sagittal 19 of 226, all large** |
| T2s | 561 | 489 | **331** | transverse 69 of 90 (28 large); coronal 126 of 249 (45); sagittal 136 of 222 (73) |

Same slice winning on both contrasts: **10, all sagittal** — 137, 156, 157, 160, 161, 162, 163, 166, 167, 173
(PE index; the head-area centroid is 152, so these are 2–15 mm off the midline).

## 3. The slice: sagittal 161

Largest T1w margin among the ten. Head-masked slice metrics, MC-GS against the best other arm:

| contrast | MC-GS | best other arm | margin PSNR / SSIM | strongest non-GS arm |
|---|---|---|---|---|
| T1w | 39.344 dB / 0.9587 | GS-single (tuned, 9.6M) 39.326 / 0.9586 | **+0.018 / +0.0001** | L1-SPIRiT 38.53 |
| T2s | 33.449 dB / 0.8858 | GS-single 33.388 / 0.8850 | **+0.061 / +0.0008** | ZS-SSDU 32.43 |

Runner-up choice: sagittal 160 has the largest T2s margin of the ten (+0.095 / +0.0017) but its T1w SSIM lead rounds to
+0.0000 and the runner-up there is the unregistered `p3_single_T1w_n60` run (+0.016 dB); drawn as well.

Figures (`fig_row_final.py --axis 1 --mcgs_key gs_joint_p3f --single_key gs_single_p3f` on T1w / `gs_single_ar` on T2s,
L1 column = L1-SPIRiT; `--single_key` added today):
`figures/peds_af12_{T1w,T2s}_allfirst_sag161_axis1_161.png`, `..._allfirst_sag160_axis1_160.png`, served at
`http://localhost:8765/peds_tsc_af12/`.

## 4. What travels with it

- On T1w the lead exists only on sagittal sections near the midline and is hundredths of a dB (no eye can see it);
  there is no full transverse or coronal section of T1w on which MC-GS beats every arm, and the volume-level T1w
  cells are still GS-single's (PSNR, SSIM) and L1-SPIRiT's (HFEN). On T2s the lead is broad (331 of 561 slices).
- The panel metrics are PSNR / SSIM / NMSE; HFEN is not a per-panel metric of the figures.
- One subject, arms tuned on it (the GS arms and the classical λ alike); the reference is GRAPPA-completed 2x data.
