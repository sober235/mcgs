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

## 5. Follow-up (user, 2026-09-16 morning): the three planes on which MC-GS leads, zoomed on the lesion

User: one slice that beats every comparison method is enough; then show the three planes on which it leads, and use
the zoom-in view on the lesion region to show the finer reconstruction of the lesion against the comparison methods.

Lesions are visible on T2*w only, so this figure is T2s. From `find_slice_all_first_T2s.json`, the slices through each
lesion centre (stage-4 coordinates, RO / PE / SPE) and whether MC-GS leads every arm on PSNR, SSIM and NMSE there:

- right subependymal nodules (148 / 166, 172, 45): transverse 43–47 all lead, sagittal 170–173 all lead, coronal 146,
  149 and 150 lead (147 and 148 lose SSIM by ≤ 0.0003). **Chosen triple: transverse 46 / coronal 146 / sagittal 172**,
  margins over the best other arm +0.081 dB / +0.0010, +0.041 / +0.0006, +0.095 / +0.0011; the runner-up is GS-single
  in every cell.
- left nodules (138–147, 139–140, 53): transverse 51–55 lead; a full triple exists at 53 / 146 / 140
  (+0.048 / +0.0008, +0.041 / +0.0006, +0.068 / +0.0004).
- calcified focus (166, 196, 69): transverse 67–71 and sagittal 194–198 lead (transverse 69: +0.095 / +0.0031, the
  largest SSIM lead of any lesion slice), but every coronal slice through it (164–168) loses SSIM to GS-single by
  0.0004–0.0013, so no full triple.
- T1w at the chosen triple: MC-GS second to GS-single on all three planes (−0.059 / −0.041 / −0.001 dB); the lesions
  are not visible on T1w in any case.

Figure: `fig_three_planes_inset.py --contrast T2s --slices 46 146 172 --inset 48 --inset_corner tr bl bl --mask_error
--gray_cbar --cbar_planes 0 --pdf --mcgs_key gs_joint_p3f --single_key gs_single_ar --tag _lesionB_allfirst` →
`figures/peds_af12_T2s_row3planes_compact_46_146_172_inset48_headerr_lesionB_allfirst.{png,pdf}`; an `inset32` variant
(tighter box) beside it. The zoom cube sits on the intersection of the three slices, 1.4 mm from the nodule centre
(148, 172, 45), so every plane's inset holds the lesion. MC-GS is first on all three metrics in every plane (bold).
Looked at: the two nodules are distinct dark foci on the ventricle wall in GT, MC-GS and GS-single; 3D-DIP blurs them
into blobs, CG-SENSE, L1-SPIRiT, ZS-SSDU and MC-INR are grainy around them. MC-GS and GS-single are not separable by
eye — the lead over GS-single is the numbers (+0.04 to +0.10 dB per plane); the visible gain is over the other eight
arms. Selected on the reported metric, as everything in this record.

Revision (user, 2026-09-16 10:xx): the sagittal zoom box must be a landscape rectangle covering the lesion region, and
no inset may overlap the metrics in the panel's top-left corner. `fig_three_planes_inset.py` now takes `--inset RO PE
SPE` (three extents; one value still means a cube), `--inset_center RO PE SPE` (-1 = the slice index), and keeps a
left-side inset below the three metric lines (`TEXT_KEEPOUT_IN`). **Figure of record:**
`figures/peds_af12_T2s_row3planes_compact_46_146_172_inset48x48x16_box157_172_46_headerr_lesionB_allfirst.{png,pdf}`
— box 48 x 48 x 16 voxels (33.6 x 33.6 x 21.6 mm) centred at (RO 157, PE 172, SPE 46), midway between the two
nodules (RO 148 and 166), so the transverse and sagittal insets hold both and the coronal inset (RO 146) holds the
medial one; landscape on the coronal and sagittal panels, square on the transverse; every inset below the metrics.
Looked at. The cube-box versions (`_inset48_`, `_inset32_`) are superseded.
