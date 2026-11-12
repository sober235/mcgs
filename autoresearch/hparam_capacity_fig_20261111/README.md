# Gaussian-capacity ablation: figure of record

**Record (user decision, 2026-11-11 22:49 PST).** The current best result for the capacity ablation
(0.9M / 1.8M / 3.6M / 7.2M shared Gaussians) is the single-slice figure

- `hp_capacity_slice181.png` (600 dpi, 4320 x 3379 px = 183 x 143 mm) and `hp_capacity_slice181.pdf`
- PNG sha256 `7acfa24fafed15c9ebc7fa59045699e53ebe4cd49791bbba1e921415c625afd3`
- server copy: `/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_hparam_capacity_fig_20261111/`

Use this figure and the numbers below when the capacity ablation is shown or quoted, until the user
names a newer one.

## What it shows

Statement supported: too little capacity lowers both contrasts; doubling beyond 3.6M trades T1 for FLAIR
without a consistent gain, at extra time and memory.

Data: jiangyizhou T1 + FLAIR, AF16 (house vd-Poisson mask), frozen phase-3 recipe, 4000 iterations,
seed 42, one run per setting; runs `hp_n09 / hp_n18 / hp_n36 / hp_n72` of
`Results_20260908/T1FLAIR_AF16_hparam_v3_jiangyizhou` (2026-09-16). Only the number of shared Gaussians
changes. All image metrics are computed on transverse slice 181, head pixels only.

| Transverse slice 181 | 0.9M | 1.8M | 3.6M | 7.2M |
|---|---|---|---|---|
| T1 PSNR (dB) | 36.851 | 37.130 | 37.210 | 37.151 |
| FLAIR PSNR (dB) | 34.407 | 34.841 | 35.066 | 35.100 |
| T1 SSIM | 0.95548 | 0.95617 | 0.95635 | 0.95585 |
| FLAIR SSIM | 0.92610 | 0.92969 | 0.93320 | 0.93421 |
| T1 NMSE | 0.00759 | 0.00711 | 0.00698 | 0.00708 |
| FLAIR NMSE | 0.01023 | 0.00926 | 0.00879 | 0.00872 |
| Training time (min) | 47 | 53 | 65 | 81 |
| Peak GPU memory (GB) | 32.3 | 32.5 | 32.9 | 33.8 |

Against 3.6M: 0.9M is -0.36 dB (T1) and -0.66 dB (FLAIR); 7.2M is -0.06 dB (T1) and +0.03 dB (FLAIR),
with SSIM moving the same way. Time and memory are per run on one A100 80 GB.

Panels: a, slice 181 for GT and the four capacities, a 48 x 48 zoom and its error map
(|recon - GT| / max GT, jet [0, 0.1]); b, slice PSNR; c, slice SSIM; d, slice NMSE; e, compute cost.

## What must be said with it

- **One slice, chosen by rule.** Slice 181 is the transverse slice nearest to the figure-of-record
  slice (180) on which the whole statement holds on PSNR and SSIM for both contrasts. Slice 180 fails
  one item (T1 SSIM is not lower at 0.9M).
- **How typical it is** (`slice_metrics.csv`, 175 transverse slices with enough head area): the full
  pattern holds on 78; "0.9M worse on both contrasts" on 142; "7.2M: T1 down, FLAIR up" on 123 and its
  reverse on 0 (coronal 101 / 135, sagittal 121 / 128, reverse 0 on both).
- **NMSE is not independent evidence**: on the same pixels it is a monotone transform of PSNR.
- **Zoom box**: the 48 x 48 window of this slice where 0.9M loses most against 3.6M (both contrasts
  summed). In the box of the figure of record (deep white matter) the trend reverses.
- **One seed.** Seed-to-seed variation was not measured; the 7.2M differences are 0.03 to 0.06 dB.
- HFEN, whole-volume metrics and the private-Gaussian ablation are not part of this record
  (user 2026-11-11).

## Files

| File | Content |
|---|---|
| `hp_capacity_slice181.{png,pdf}`, `_source_data.csv` | the record figure and its plotted values |
| `fig_capacity_slice.py` | draws it (`gpu.py run --cpu`; `--slice N` for another slice; stops if the pattern fails) |
| `slice_scan.py`, `slice_metrics.csv` | per-slice PSNR / SSIM / NMSE on all three axes |
| `hp_capacity_full_nmse.png`, `hp_capacity_full.png`, `hp_capacity_full_housebox.png` | earlier whole-head versions (slice 180, error scale [0, 0.05]) |
| `hp_capacity_main.png` | first two-panel version |
| `fig_capacity_full.py`, `fig_capacity_main.py` | scripts of the earlier versions |
