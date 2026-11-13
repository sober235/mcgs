# Record result of the different-mask experiment (jiangyizhou T1 + FLAIR, nominal AF 16)

**Designated by the user on 2026-11-13 (about 00:14 PST, local clock):** this is the record result of the
different-mask experiments, the one that shows the best generalisation of MC-GS across sampling masks.

## What it is

Transverse slice 205 of the diff-mask study, drawn in the plain-scan / enhanced figure-of-record format: six rows
(vd-Poisson T1, vd-Poisson FLAIR, uniform density T1, uniform density FLAIR, pseudo-radial T1, pseudo-radial FLAIR),
twelve columns (GT, ZF, CG-SENSE, CS-SENSE, Joint SENSE, L1-SPIRiT, ZS-SSDU, 3D-DIP, 3D-INR, MC-INR, GS-single,
MC-GS), a zoom row under every image row (64 x 64 box at (J, K) = (128, 88): image | error; GT column: GT | the
row's sampling mask), head-masked slice PSNR / SSIM (7x7) / NMSE in every panel, one grey and one error bar.

**MC-GS is first on PSNR, SSIM and NMSE in all six rows, against all ten other columns: 18 of 18 conditions.**

| row | AF | MC-GS PSNR (dB) | PSNR lead | SSIM lead | NMSE lead | box PSNR lead |
|---|---|---|---|---|---|---|
| vd-Poisson T1 | 15.19x | 35.737 | +0.131 (GS-single) | +0.0022 | +0.00026 | +0.274 (Joint SENSE) |
| vd-Poisson FLAIR | 16.49x | 33.355 | +0.050 (GS-single) | +0.0019 | +0.00013 | +0.186 (GS-single) |
| uniform density T1 | 15.20x | 35.326 | +0.730 (GS-single) | +0.0143 | +0.00171 | +1.282 (GS-single) |
| uniform density FLAIR | 19.48x | 34.742 | +0.163 (GS-single) | +0.0039 | +0.00031 | +0.513 (GS-single) |
| pseudo-radial T1 | 15.20x | 30.528 | +0.140 (ZS-SSDU) | +0.0030 | +0.00093 | +0.729 (GS-single) |
| pseudo-radial FLAIR | 15.34x | 29.553 | +0.197 (GS-single) | +0.0124 | +0.00125 | +0.650 (GS-single) |

AF = phase-encode plane points / sampled points that carry that contrast's data (2966 / 2964 / 2964 points of 45056;
FLAIR's k-space covers 75.8 % of the plane).

## How the slice was found

`fig_masks_record_format.py --scan 0 255 --no_fig` scored every transverse slice with more than 2000 head pixels (243
slices). MC-GS is first on all 18 conditions on five of them: **203, 204, 205, 206, 213** (all above the lateral
ventricles). On PSNR alone it is first in all six rows on 60 of the 243 slices; SSIM is the condition that removes the
rest. 205 was taken because it is the house FLAIR slice of the September three-pattern figures; its smallest leads
are PSNR +0.050 dB and SSIM +0.0019 (both vd-Poisson FLAIR over GS-single). 203 has the largest smallest PSNR lead
(+0.059 dB). No slice of the butterfly range (156-180) passes: there ZS-SSDU has the better slice PSNR on the
pseudo-radial T1 row.

## Read it with

- The slice was selected on these data (5 of 243 slices pass). The study's volume-level numbers are the head PSNR
  margins of MC-GS over the best other arm: T1 +0.148 / +0.507 / +0.200 dB, FLAIR +0.042 / +0.142 / +0.093 dB
  (vd-Poisson / uniform density / pseudo-radial), first on all three patterns and both contrasts.
- One subject, one draw of each mask, one seed. MC-GS and GS-single run one frozen recipe on every mask; the classical
  arms' lambdas are re-selected per mask by head PSNR against the ground truth.
- Row 3 is the September pseudo-radial result (user decision, 2026-11-12); rows 1 and 3 were reconstructed in
  September, row 2 in November, with the same data, gauge, scorer and recipe.

## Where

| what | where |
|---|---|
| 600-dpi PNG (19569 x 19317 px), 600-dpi PDF, preview, record JSON, MANIFEST | GPU server `/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_diffmask_masks_record_tra205_20261112/` |
| 200-dpi PNG | topic `Results_20260908/T1FLAIR_AF16_uniform_v3_jiangyizhou/figures/t1flair_v3_af16_masks_record_format_tra205_box64.png` |
| script, preview, per-panel record, all-slice scan | GitHub `sober235/mcgs`, branch `diffmask-af16-masks-20261111`, `autoresearch/diffmask_af16_masks_20261111/` |
| the same figure at the butterfly slice (transverse 168, not 18 of 18) | GPU server `.../codex_artifacts/mcgs_diffmask_masks_record_20261112/` |

Regenerate (CPU, a few minutes; refuses a non-empty output folder):

```sh
gpu.py run --cpu -- @python autoresearch/diffmask_af16_masks_20261111/fig_masks_record_format.py --tra 205 --box 64 \
    --center 128 88 --what "MC-GS first on PSNR, SSIM and NMSE in all six rows" --png_dpi 600 --pdf --pdf_dpi 600 \
    --out_dir <new folder>
```
