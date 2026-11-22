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

## Article edition drawn the way of the AF 20 figure (user, 2026-11-22)

User: "You need to first understand how the figure with af = 20 was plotted. Then redraw this graph again."
The AF 20 article figure (`t1flair_af_record_fig_20261120`, `article_af20_acceleration`) was drawn in two steps:
`find_zoom_box.py` places one 48-voxel box per plane (the same for T1 and FLAIR) where MC-GS's box-mean error is
furthest below the best other method's; `fig_t1flair_af_record_format.py --box_centres ... --err_cmap inferno
--err_vmax 0.1 --arrow --arrow_mode edge --arrow_on main --role ...` then draws it, the window at the slice P99.9 and
the red arrow on each slice. The same here, with the three sampling masks in place of the three planes:

    fig_masks_record_format.py --tra 205 --box 48 --box_select --what "" --err_cmap inferno --arrow --arrow_mode edge
        --arrow_on main --brightness 1 --role "Sampling-mask generalization" --save_data --pdf --tag _article_boxsel

- Figure: `uniform_recon/figures/t1flair_v3_af16_masks_record_format_tra205_box48_article_boxsel.png` (PNG sha256
  `b63f3c20fdff…`), 600-dpi PDF and `_data.npz` beside it, record JSON in `uniform_recon/`.
- Boxes (J, K): vd-Poisson (147, 69), uniform density (167, 75), pseudo-radial (163, 58). In the box MC-GS's mean
  error is 4.7 / 2.2 % (vd-Poisson T1 / FLAIR), 14.1 / 5.0 % (uniform density) and 6.6 / 4.6 % (pseudo-radial) below
  the best other column. Boxes and arrows are chosen against GT, for display only.
- Slice 205 and every panel number are those of the record figure; MC-GS is first on all 18 conditions.
- The script's defaults are unchanged: the earlier article edition (64 x 64 box at (128, 88), arrows in the magnified
  boxes) was redrawn with its old options after the change and is byte-identical (sha256 `4fab5779…`).

**Arrows corrected (user, 2026-11-22):** "The arrows are only marked on the zoom-in weighted graph in the area with
the largest error map." The figure above (arrows on the slices) is superseded by

    fig_masks_record_format.py --tra 205 --box 48 --box_select --what "" --err_cmap inferno --arrow --arrow_mode gap
        --arrow_on zoom --brightness 1 --role "Sampling-mask generalization" --save_data --pdf --tag _article_boxsel_zoomarrow

- Figure: `uniform_recon/figures/t1flair_v3_af16_masks_record_format_tra205_box48_article_boxsel_zoomarrow.png` (PNG
  sha256 `2122a1c2f90f…`), 600-dpi PDF and `_data.npz` beside it. Boxes and panel numbers as above.
- The arrow sits in the magnified weighted-image box only, at the point of the box where the error of the other
  methods (median of the nine reconstructions other than ZF and MC-GS, 7x7 local mean) exceeds MC-GS's most.
- `..._zoomarrow_edge.png` is the same figure with that point restricted to tissue boundaries (`--arrow_mode edge`).

## Optimal result of the different-mask stage (user, 2026-11-22, local clock PST)

On 2026-11-22 the user asked for the result figure of the different masks as a 600-dpi PDF with vector graphics, for
the reconstructed data to be saved on the GPU server, and for the current result to be recorded as the optimal result
of the different-mask stage. The current figure is the last one of the section above:
`t1flair_v3_af16_masks_record_format_tra205_box48_article_boxsel_zoomarrow` (transverse 205; one 48-voxel box per
mask, placed where MC-GS's error is furthest below the best other method; inferno error [0, 0.1]; red arrow only in
the magnified weighted-image box, at the largest error gap; title "Sampling-mask generalization: T1 + FLAIR,
acceleration factor 16"). The panel numbers are those of the record of 2026-11-13: MC-GS first on 18 of 18 conditions.
The user's message did not name the arrow variant; the largest-error-gap one (`gap`) is taken, the `_edge` spare is not.

| mask | T1 slice PSNR (lead) | FLAIR slice PSNR (lead) | box PSNR lead T1 / FLAIR | volume head PSNR lead T1 / FLAIR |
|---|---|---|---|---|
| vd-Poisson | 35.737 dB (+0.131, GS-single) | 33.355 dB (+0.050, GS-single) | +0.363 / +0.212 dB | +0.148 / +0.042 dB |
| uniform density | 35.326 dB (+0.730, GS-single) | 34.742 dB (+0.163, GS-single) | +1.379 / +0.406 dB | +0.507 / +0.142 dB |
| pseudo-radial | 30.528 dB (+0.140, ZS-SSDU) | 29.553 dB (+0.197, GS-single) | +0.480 / +0.413 dB | +0.200 / +0.093 dB |

GPU server folder `/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_diffmask_masks_article_tra205_20261122/`:

| file | content | sha256 (first 12) |
|---|---|---|
| `optimal_boxsel_zoomarrow/..._article_boxsel_zoomarrow_600dpi.pdf` | THE FIGURE: text, frames and bars vector (embedded TrueType), all 218 image panels embedded at 600 dpi | 7a4b44e49edc |
| `optimal_boxsel_zoomarrow/..._600dpi.png` | 600-dpi raster | f0389beb1ed8 |
| `optimal_boxsel_zoomarrow/..._600dpi_data.npz` | every slice the figure draws (each method, GT, head mask, error region, sampling masks) | dbc3a2cbb065 |
| `optimal_boxsel_zoomarrow/` record JSON, `MANIFEST.json`, preview, script | numbers of every panel, source files and SHA-256 | |
| `recon_data/{vd-Poisson,Uniform_density,Pseudo-radial}/` | the reconstructed volumes: 11 columns x T1 / FLAIR per mask, the GT volumes and the 3-D sampling mask (75 files, 5.67 GiB), each copy re-hashed against the source and the figure record | list in `DATA_MANIFEST.json` (8260ccdd19cb) |
| top level `..._box64_article_600dpi.*` | superseded edition (one 64 x 64 box at (128, 88), arrows at tissue boundaries), drawn earlier the same day | |

The reconstructed images are raster data, so "vector" applies to the text and line work of the PDF; the image panels
are embedded at 600 dpi.

Regenerate (CPU; refuses a non-empty output folder), then save the volumes:

```sh
gpu.py run --cpu -- @python autoresearch/diffmask_af16_masks_20261111/fig_masks_record_format.py --tra 205 --box 48 \
    --box_select --what "" --err_cmap inferno --arrow --arrow_mode gap --arrow_on zoom --brightness 1 \
    --role "Sampling-mask generalization" --save_data --tag _article_boxsel_zoomarrow --png_dpi 600 --pdf --pdf_dpi 600 \
    --out_dir <new folder>
gpu.py run --cpu -- @python autoresearch/diffmask_af16_masks_20261111/save_recon_data.py --folder <folder with the record JSON>
```

The cautions of "Read it with" above apply unchanged (slice selected on these data, one subject, one draw per mask,
one seed); boxes and arrows are chosen against GT, for display only.
