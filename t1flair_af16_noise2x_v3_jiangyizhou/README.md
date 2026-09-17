# Retrospective nominal AF 16 on the 3D T1 + FLAIR pair, jiangyizhou -- measurements with 2 sigma0 of added noise

Noise-robustness study, 2026-09-16. **Read `../T1FLAIR_AF16_v3_jiangyizhou/README.md` first**: same subject, same CFL
k-space, same ESPIRiT maps, same house vd-Poisson disc mask, same ground truth, same gauge, same arms and the same frozen
AF16 phase-3 MC-GS recipe are documented there and not repeated here. Spec:
`/data0/congcong/code/GS/mcgs_noise_wt/docs/superpowers/specs/2026-09-16-jyz-af16-noise-robustness-design.md`.

## What this topic adds

**Exactly one thing differs from `T1FLAIR_AF16_v3_jiangyizhou`: every arm fits noisier measurements.** That topic is
the zero-noise column of this study and is not re-run. `data/ksp_<c>.h5` (symlinks into the AF12 topic) stay the
clean reference and define only the ground truth and the scale; `data/meas_<c>.h5` are what every arm fits.

A second, milder level (1 sigma0, the same draws halved) is `../T1FLAIR_AF16_noise1x_v3_jiangyizhou/`; the
three-level tables (no noise, 1 sigma0, 2 sigma0) are in its `logs/compare_noise_levels_<metric>.md`.

### The noise

Complex Gaussian noise, independent over coils and samples, added only on the acquired samples (T1 the whole PE plane,
FLAIR the lines inside the vendor shutter; outside stays exactly zero). Per-sample standard deviation
sigma = 2 sigma0_k, where sigma0_k is the data's own noise measured on the clean ground truth's air
(sigma0_img, gauge units, P99.9 = 1) and divided by sqrt(f_acq). `numpy` PCG64 from `SeedSequence(20260916)`, one
realization shared by every arm. Generator `scripts/make_noisy_kspace.py`, report `logs/make_noisy_kspace.json`.

| contrast | acquired lines (fraction) | sigma0_img | sigma0_k | added sigma | expected air noise | measured air noise | checks |
|---|---|---|---|---|---|---|---|
| T1 | 45056 (1.0000) | 0.01408 | 0.01408 | 0.02815 | 0.03148 | 0.03147 | all pass |
| FLAIR | 34139 (0.7577) | 0.01282 | 0.01472 | 0.02944 | 0.02866 | 0.02865 | all pass |

Checks: meas minus clean exactly zero outside the support; per-coil mean within 4 SE of 0 (52 tests); per-coil std
within 0.5 %; air noise of the noisy full image within 2 % of sqrt(sigma0_img^2 + sigma^2 f_acq); ground truth within
1e-4 of the AF16 topic's single-precision volume and scale within 1e-5 relative. Gauge check (`logs/gauge_check_clean.log`):
with the measurements pointed at the clean file this topic's code reproduces the AF16 topic's CG-SENSE row on every metric.

## Results

<!-- TABLES:BEGIN -->
**T1** (sorted by head-only PSNR; no noise column = the same arm in `T1FLAIR_AF16_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @no noise | this − no noise |
|---|---|---|---|---|---|---|---|---|
| ZS-SSDU | 36.432 | 0.8784 | **33.131** | 0.8945 | 0.1870 | 0.952 | 34.150 | -1.02 |
| CS-SENSE (TV, converged) (λ 0.002) | 36.388 | 0.9153 | 32.766 | **0.8951** | 0.1967 | 0.810 | 34.945 | -2.18 |
| joint CS-SENSE (JTV) (λ 0.0028) | 36.419 | 0.9194 | 32.697 | 0.8949 | 0.2044 | 0.790 | 34.831 | -2.13 |
| CS-SENSE | 34.423 | 0.7633 | 32.227 | 0.8711 | 0.2221 | 0.908 | 34.409 | -2.18 |
| L1-SPIRiT (λ 0.004) | 34.910 | 0.8720 | 31.453 | 0.8601 | 0.2427 | 0.883 | 33.788 | -2.33 |
| 3D-DIP | 34.903 | 0.8612 | 30.951 | 0.8525 | 0.3062 | 0.637 | 30.994 | -0.04 |
| CG-SENSE | 29.564 | 0.5041 | 29.924 | 0.7646 | 0.2801 | 1.208 | 32.354 | -2.43 |
| MC-GS | 33.084 | 0.8301 | 29.802 | 0.7771 | 0.2252 | 1.331 | 35.706 | -5.90 |
| GS-single | 32.571 | 0.7931 | 29.571 | 0.7671 | 0.2326 | 1.350 | 35.558 | -5.99 |
| 3D-INR | 30.229 | 0.5442 | 28.698 | 0.7224 | 0.3397 | 1.265 | 30.375 | -1.68 |
| ZF | 29.938 | 0.8260 | 25.835 | 0.6782 | 0.6955 | 0.289 | 25.841 | -0.01 |
| MC-INR (Niessen 2025) | 20.334 | 0.2694 | 25.034 | 0.5522 | 0.3746 | 1.800 | 31.590 | -6.56 |

**FLAIR** (sorted by head-only PSNR; no noise column = the same arm in `T1FLAIR_AF16_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @no noise | this − no noise |
|---|---|---|---|---|---|---|---|---|
| joint CS-SENSE (JTV) (λ 0.0028) | 36.262 | 0.9102 | **32.553** | 0.8801 | 0.1847 | 0.708 | 34.355 | -1.80 |
| CS-SENSE (TV, converged) (λ 0.002) | 36.116 | 0.9014 | 32.526 | **0.8809** | 0.1847 | 0.674 | 34.302 | -1.78 |
| ZS-SSDU | 35.854 | 0.8682 | 32.348 | 0.8731 | 0.1867 | 0.936 | 34.005 | -1.66 |
| CS-SENSE | 34.395 | 0.7582 | 32.172 | 0.8635 | 0.2004 | 0.789 | 33.806 | -1.63 |
| L1-SPIRiT (λ 0.004) | 35.210 | 0.8865 | 31.636 | 0.8493 | 0.2030 | 0.827 | 33.720 | -2.08 |
| 3D-DIP | 34.582 | 0.8399 | 30.611 | 0.8269 | 0.2816 | 0.509 | 30.650 | -0.04 |
| CG-SENSE | 29.989 | 0.5127 | 30.297 | 0.7656 | 0.2319 | 1.083 | 32.826 | -2.53 |
| GS-single | 33.493 | 0.8404 | 30.062 | 0.7699 | 0.2161 | 1.219 | 34.886 | -4.82 |
| MC-GS | 33.222 | 0.8233 | 29.946 | 0.7644 | 0.2205 | 1.227 | 34.928 | -4.98 |
| 3D-INR | 31.669 | 0.6245 | 29.389 | 0.7483 | 0.2965 | 0.950 | 30.257 | -0.87 |
| ZF | 31.783 | 0.8524 | 27.868 | 0.7111 | 0.6600 | 0.275 | 27.879 | -0.01 |
| MC-INR (Niessen 2025) | 22.443 | 0.3025 | 26.507 | 0.5950 | 0.3081 | 1.479 | 32.243 | -5.74 |

**MC-GS joint minus MC-GS single**

| channel | whole PSNR | head PSNR | head SSIM | (no noise: whole / head) |
|---|---|---|---|---|
| T1 | +0.512 | +0.232 | +0.0100 | +0.124 / +0.148 |
| FLAIR | -0.270 | -0.116 | -0.0056 | +0.086 / +0.042 |
<!-- TABLES:END -->

### The study's headline (2 sigma0)

**The frozen MC-GS recipe is not robust to 2 sigma0 of added measurement noise.** Its head PSNR falls 5.9 dB (T1) and
5.0 dB (FLAIR), its head SSIM by 0.16, and from first on all six head cells it drops to 8th / 7th / 5th (T1: PSNR /
SSIM / HFEN) and 9th / 9th / 7th (FLAIR) of the twelve listed arms (`logs/compare_noise_head_{psnr,ssim,hfen}.md`).

| contrast | metric | MC-GS, no noise -> 2 sigma0 | strongest comparison arm at 2 sigma0 | margin, no noise -> 2 sigma0 |
|---|---|---|---|---|
| T1 | head PSNR | 35.706 -> 29.802 | ZS-SSDU 33.131 | +0.148 -> -3.328 |
| T1 | head SSIM | 0.9396 -> 0.7771 | CS-SENSE (TV, converged) 0.8951 | +0.0025 -> -0.1180 |
| T1 | head HFEN | 0.1139 -> 0.2252 | ZS-SSDU 0.1870 | +0.0047 -> -0.0382 |
| FLAIR | head PSNR | 34.928 -> 29.946 | Joint SENSE 32.553 | +0.042 -> -2.606 |
| FLAIR | head SSIM | 0.9261 -> 0.7644 | CS-SENSE (TV, converged) 0.8809 | +0.0013 -> -0.1165 |
| FLAIR | head HFEN | 0.1121 -> 0.2205 | Joint SENSE 0.1847 | +0.0016 -> -0.0359 |

(no noise = `T1FLAIR_AF16_v3_jiangyizhou`, where GS-single is the strongest comparison arm on every cell.)

**Why.** The GS render fits the noise: the sharpness ratio of MC-GS rises from 0.855 / 0.804 to 1.331 / 1.227 (T1 /
FLAIR), GS-single's to 1.35 / 1.22. Hard data consistency is not the cause: scored without it, the same volumes are
+0.06 dB head PSNR and +0.002 head SSIM better, nothing more (diagnostic table in `logs/compare_noise_head_psnr.md`).
The recipe's regularisation was set on data carrying only the scanner's own noise and is frozen here, while every
classical family re-selects its lambda on the noisy data: TV 0.0007 -> 0.002 (T1) and 0.0005 -> 0.002 (FLAIR), JTV
0.001 -> 0.0028 and 0.0007 -> 0.0028, L1-SPIRiT 0.001 -> 0.004. At their zero-noise lambdas those families lose another
1.1-1.9 dB (frozen-lambda view in the comparison tables), which is the size of the gap a frozen recipe pays.
MC-INR, which has no regulariser, loses the most (-6.6 / -5.7 dB); 3D-DIP, early-stopped on held-out k-space, loses
almost nothing (-0.04 dB) but starts low; ZS-SSDU loses 1.0 / 1.7 dB and becomes the best T1 arm on head PSNR.

**Grids.** Three winners first sat on the upper edge (T1 JTV 0.0028, FLAIR TV 0.002, FLAIR JTV 0.0028). The grids were
extended by two steps each (JTV 0.004, 0.0056; FLAIR TV 0.0028, 0.004; `run_all_noise.py --extend`); the winners did not
move, and every selected lambda is now interior.

### Figures

| figure | what |
|---|---|
| `http://localhost:8765/t1flair_v3_af16_noise2x/t1flair_v3_af16n_noise_data.png` | the data: clean and noisy fully sampled images and their difference, both contrasts |
| `http://localhost:8765/t1flair_v3_af16_noise2x/t1flair_v3_af16n_T1_row3planes_compact_180_124_76_inset48_headerr_p3_mcinr_spirit_zf_caps.png` | T1, figure of record (also `.pdf`, 600 dpi) |
| `http://localhost:8765/t1flair_v3_af16_noise2x/t1flair_v3_af16n_FLAIR_row3planes_compact_180_124_76_inset48_headerr_p3_mcinr_spirit_zf_caps.png` | FLAIR, figure of record (also `.pdf`) |
| `http://localhost:8765/t1flair_v3_af16_noise2x/t1flair_v3_af16n_T1_row3planes_compact_174_110_47_headerr_p3_mcinr_spirit_zf_caps.png` | T1, 174 / 110 / 47 |
| `http://localhost:8765/t1flair_v3_af16_noise2x/t1flair_v3_af16n_FLAIR_row3planes_compact_174_110_47_headerr_p3_mcinr_spirit_zf_caps.png` | FLAIR, 174 / 110 / 47 |

The same files are in `figures/` (served from `/home/congcongliu/figs/t1flair_v3_af16_noise2x/`). Columns: GT, ZF,
CG-SENSE, CS-SENSE (converged TV at its selected lambda), Joint SENSE, L1-SPIRiT, ZS-SSDU, 3D-DIP, 3D-INR, MC-INR,
GS-single, MC-GS; slices and box as in the AF16 topic; error maps `|recon - GT| / max(GT)`, jet [0, 0.1], inside the
head. The caption is set on three lines here: the one-line caption of the AF16 topic's `_zf_caps` figures runs past
both edges of the image.

## Known limitations

- One subject and one realization (the 1 sigma0 topic uses the same draws, halved); the spread over realizations is
  not measured.
- The zero-noise JTV column comes from a two-point sweep in `T1FLAIR_AF16_v3_jiangyizhou` (lambda 0.0007 and 0.001),
  so its zero-noise value is a lower bound; it is not the strongest comparison arm at zero noise on any cell.
- The added noise is white and independent across coils; the scanner's own noise is correlated across coils.
- Maps and ground truth are clean; only L1-SPIRiT's kernel sees the noisy calibration block.
- The ground truth carries the scanner's own noise sigma0.
- The classical lambdas are chosen against the ground truth while MC-GS runs a frozen recipe.
- The GS arms are scored with hard data consistency, which puts the noise on the acquired points back into the image;
  the render-only diagnostic (`logs/gs_render_<c>.json`) shows how much that costs.

## Regenerating

```
python scripts/make_noisy_kspace.py                       # refuses to overwrite
bash scripts/launch_noise.sh --dry_run                    # the 57 jobs
setsid nohup bash scripts/launch_noise.sh > /dev/null 2>&1 < /dev/null &
```

## Status

2026-09-17 08:05: complete. 61 jobs (57 + 4 grid extensions), no failures; finalized twice (after the extension).
2026-09-17 09:21: the noise figure re-laid out (axes placed in inches, no column slack). 09:30: the 1 sigma0 level is complete
(`../T1FLAIR_AF16_noise1x_v3_jiangyizhou/`).
