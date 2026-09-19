# Retrospective nominal AF 16 on the 3D T1 + FLAIR pair, jiangyizhou -- measurements with 0.75 sigma0 of added noise

Noise-robustness study, fourth level, 2026-09-17. **Read `../T1FLAIR_AF16_noise2x_v3_jiangyizhou/README.md` first** (the
2 sigma0 level; everything there applies here), `../T1FLAIR_AF16_noise1x_v3_jiangyizhou/README.md` (1 sigma0) and
`../T1FLAIR_AF16_v3_jiangyizhou/README.md` (the zero-noise column). Spec:
`/data0/congcong/code/GS/mcgs_noise_wt/docs/superpowers/specs/2026-09-16-jyz-af16-noise-robustness-design.md`,
section "Addendum (2026-09-17 13:40)".

## What this topic adds

The second point of the search for the noise range in which MC-GS leads (user, 2026-09-17): "favourable" means first on
all six head cells (head PSNR / SSIM / HFEN x T1 / FLAIR) against every comparison arm, GS-single included, with the
classical lambdas re-selected on the noisy data. The rule: 0.75 sigma0 follows if 0.5 sigma0
(`../T1FLAIR_AF16_noise0p5x_v3_jiangyizhou/`) is favourable, otherwise 0.25 sigma0. Every arm is run again with the frozen recipe and the same grids; the GS experiments carry the prefix
`af16n075_`, the MC-INR run the tag `_noise0p75x`. The other levels stay in the record.

### The noise

Per-sample standard deviation sigma = 0.75 sigma0_k, **the same seed and the same draws as the other levels**, so this
topic's noise is 3/4 of the 1 sigma0 topic's (checked sample by sample as `nested_vs_1x`, the largest ratio of the
difference to a single-precision rounding bound, limit 1). Added noise power x1.5625, SNR down about 1.9 dB. Generator
`scripts/make_noisy_kspace.py`, report `logs/make_noisy_kspace.json`.

| contrast | acquired lines (fraction) | sigma0_img | sigma0_k | added sigma | expected air noise | measured air noise | nested_vs_1x | checks |
|---|---|---|---|---|---|---|---|---|
| T1 | 45056 (1.0000) | 0.01408 | 0.01408 | 0.01056 | 0.01760 | 0.01759 | 0.29 | all pass |
| FLAIR | 34139 (0.7577) | 0.01282 | 0.01472 | 0.01104 | 0.01602 | 0.01602 | 0.28 | all pass |

## Results

<!-- TABLES:BEGIN -->
**T1** (sorted by head-only PSNR; no noise column = the same arm in `T1FLAIR_AF16_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @no noise | this − no noise |
|---|---|---|---|---|---|---|---|---|
| MC-GS | 38.375 | 0.9221 | **34.749** | **0.9250** | 0.1272 | 0.911 | 35.706 | -0.96 |
| GS-single | 38.168 | 0.9253 | 34.555 | 0.9213 | 0.1323 | 0.918 | 35.558 | -1.00 |
| CS-SENSE (TV, converged) (λ 0.001) | 38.108 | 0.9473 | 34.367 | 0.9240 | 0.1499 | 0.811 | 34.945 | -0.58 |
| joint CS-SENSE (JTV) (λ 0.0014) | 38.051 | 0.9488 | 34.248 | 0.9231 | 0.1561 | 0.798 | 34.831 | -0.58 |
| CS-SENSE | 37.741 | 0.9441 | 34.007 | 0.9199 | 0.1646 | 0.785 | 34.409 | -0.40 |
| ZS-SSDU | 37.267 | 0.9206 | 33.710 | 0.9125 | 0.1717 | 1.018 | 34.150 | -0.44 |
| L1-SPIRiT (λ 0.002) | 36.797 | 0.9241 | 33.113 | 0.8998 | 0.1740 | 0.875 | 33.788 | -0.68 |
| CG-SENSE | 33.480 | 0.6951 | 31.922 | 0.8627 | 0.2330 | 0.922 | 32.354 | -0.43 |
| 3D-DIP | 35.001 | 0.8513 | 31.071 | 0.8556 | 0.3004 | 0.639 | 30.994 | +0.08 |
| MC-INR (Niessen 2025) | 27.636 | 0.4718 | 30.545 | 0.7865 | 0.2508 | 1.127 | 31.590 | -1.04 |
| 3D-INR | 32.052 | 0.6573 | 29.806 | 0.7716 | 0.2413 | 1.334 | 30.375 | -0.57 |
| ZF | 29.947 | 0.8252 | 25.840 | 0.6793 | 0.6951 | 0.278 | 25.841 | -0.00 |

**FLAIR** (sorted by head-only PSNR; no noise column = the same arm in `T1FLAIR_AF16_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @no noise | this − no noise |
|---|---|---|---|---|---|---|---|---|
| MC-GS | 37.906 | 0.9149 | **34.213** | 0.9121 | 0.1250 | 0.855 | 34.928 | -0.71 |
| GS-single | 37.854 | 0.9001 | 34.188 | 0.9111 | 0.1260 | 0.856 | 34.886 | -0.70 |
| joint CS-SENSE (JTV) (λ 0.001) | 37.538 | 0.9310 | 33.872 | 0.9088 | 0.1417 | 0.764 | 34.355 | -0.48 |
| ZS-SSDU | 37.663 | 0.9389 | 33.845 | **0.9126** | 0.1489 | 0.998 | 34.005 | -0.16 |
| CS-SENSE (TV, converged) (λ 0.0007) | 37.369 | 0.9240 | 33.826 | 0.9091 | 0.1424 | 0.739 | 34.302 | -0.48 |
| CS-SENSE | 37.307 | 0.9343 | 33.538 | 0.9063 | 0.1460 | 0.678 | 33.806 | -0.27 |
| L1-SPIRiT (λ 0.002) | 36.916 | 0.9233 | 33.157 | 0.8928 | 0.1417 | 0.826 | 33.720 | -0.56 |
| CG-SENSE | 34.075 | 0.7151 | 32.377 | 0.8659 | 0.1726 | 0.837 | 32.826 | -0.45 |
| MC-INR (Niessen 2025) | 29.891 | 0.5365 | 31.414 | 0.8176 | 0.1890 | 0.974 | 32.243 | -0.83 |
| 3D-DIP | 34.619 | 0.8213 | 30.681 | 0.8298 | 0.2734 | 0.511 | 30.650 | +0.03 |
| 3D-INR | 32.893 | 0.7083 | 30.083 | 0.7810 | 0.2241 | 1.065 | 30.257 | -0.17 |
| ZF | 31.792 | 0.8442 | 27.877 | 0.7124 | 0.6596 | 0.264 | 27.879 | -0.00 |

**MC-GS joint minus MC-GS single**

| channel | whole PSNR | head PSNR | head SSIM | (no noise: whole / head) |
|---|---|---|---|---|
| T1 | +0.207 | +0.194 | +0.0037 | +0.124 / +0.148 |
| FLAIR | +0.052 | +0.025 | +0.0010 | +0.086 / +0.042 |
<!-- TABLES:END -->

### Reading at 0.75 sigma0

**At 0.75 sigma0 MC-GS is first on five of the six head cells; FLAIR head SSIM is second, 0.0005 behind ZS-SSDU.** So the
range in which MC-GS is first on all six cells ends between 0.5 and 0.75 sigma0 (`logs/compare_noise_levels_{head_psnr,
head_ssim,head_hfen}.md`, which list every level run).

| contrast | metric | MC-GS, no noise -> 0.75 sigma0 | strongest comparison arm at 0.75 sigma0 | margin | rank |
|---|---|---|---|---|---|
| T1 | head PSNR | 35.706 -> 34.749 | GS-single 34.555 | +0.194 | 1/12 |
| T1 | head SSIM | 0.9396 -> 0.9250 | CS-SENSE (TV, converged) 0.9240 | +0.0011 | 1/12 |
| T1 | head HFEN | 0.1139 -> 0.1272 | GS-single 0.1323 | +0.0051 | 1/12 |
| FLAIR | head PSNR | 34.928 -> 34.213 | GS-single 34.188 | +0.025 | 1/12 |
| FLAIR | head SSIM | 0.9261 -> 0.9121 | ZS-SSDU 0.9126 | -0.0005 | 2/12 |
| FLAIR | head HFEN | 0.1121 -> 0.1250 | GS-single 0.1260 | +0.0010 | 1/12 |

Against the best non-GS arm MC-GS leads head PSNR by +0.382 dB (T1, converged TV 34.367) and +0.341 dB (FLAIR, JTV
33.872). It loses 0.96 / 0.71 dB against zero noise (GS-single 1.00 / 0.70, TV 0.58 / 0.48, JTV 0.58 / 0.48); its
sharpness ratio is 0.911 / 0.855. The lambdas are interior (TV 0.001 / 0.0007, JTV 0.0014 / 0.001, L1-SPIRiT 0.002 with
kernel 5); against their zero-noise lambdas they gain 0.02 to 0.21 dB. Hard data consistency changes the GS arms by
+0.025 (T1) and 0.000 dB (FLAIR).

The FLAIR SSIM cell is lost to a single ZS-SSDU run: ZS-SSDU is not monotonic in the noise level (T1 head PSNR 33.836 /
33.710 / 33.950 at 0.5 / 0.75 / 1 sigma0; FLAIR 33.836 / 33.845 / 33.289; it stops early on held-out data at a different
epoch each run), so a 0.0005 difference is inside its run-to-run spread; 3D-DIP likewise reads higher at 0.75 sigma0 than at
zero noise (+0.08 / +0.03 dB). One run per arm per level.

The tables across all levels are `logs/compare_noise_levels_<metric>.md` (`scripts/compare_noise_levels.py`, written by
the finalize step). Figures (same layout as the other levels): `http://localhost:8765/t1flair_v3_af16_noise0p75x/t1flair_v3_af16n075_{T1,FLAIR}_row3planes_compact_180_124_76_inset48_headerr_p3_mcinr_spirit_zf_caps.png`,
`..._174_110_47_headerr_p3_mcinr_spirit_zf_caps.png`, `t1flair_v3_af16n075_noise_data.png`.

## Known limitations

As in the 2 sigma0 and 1 sigma0 topics; all noisy levels share one realization.

## Regenerating

```
python scripts/make_noisy_kspace.py                       # needs the 1 sigma0 topic's meas files for the nested check
bash scripts/launch_noise.sh --dry_run                    # the 57 jobs
setsid nohup bash scripts/launch_noise.sh --cards 1,2,5,6 --max_gpu_jobs 4 > /dev/null 2>&1 < /dev/null &
```

## Status

2026-09-17 14:58: measurements made ahead of the decision, all checks pass.
2026-09-17 15:21: queue started on cards 1, 2, 5, 6 next to the 0.5 sigma0 queue (same locks, one job per card). Basis:
at 15:20 every strong arm of 0.5 sigma0 but ZS-SSDU had finished (`scripts/six_cells_now.py`) and MC-GS was first on
all six head cells, GS-single second on each; ZS-SSDU cannot change that, since even its zero-noise values are below
MC-GS's 0.5 sigma0 values on all six cells (closest: FLAIR head SSIM 0.9185 against 0.9204). To be confirmed on the
finalized 0.5 sigma0 tables.
2026-09-17 20:21: complete. 57 jobs, no failures, every selected lambda interior, finalize rc 0; five of six head cells
first (FLAIR head SSIM second by 0.0005). The 0.5 sigma0 decision was confirmed on its finalized tables at 17:22.

## Seven-method cut of the figure of record (user, 2026-09-18)

The user asked for the same figure with only **CS-SENSE, Joint SENSE, ZS-SSDU, 3D-INR, MC-INR,
GS-single and MC-GS**. ZF, CG-SENSE, the L1 arm and 3D-DIP are dropped; **GT keeps its column** --
it is what the error maps are taken against, and the sampling mask of each plane sits in its cell of
the error rows. Slices, zoom box, grey window, error colour scale, orientation, typography, metrics
and resolution are the figure of record's, unchanged.

```
figures/t1flair_v3_af16n075_<c>_row3planes_compact_180_124_76_inset48_headerr_p3_mcinr_caps_sel7.{png,pdf}
bash scripts/render_sel7.sh
```

served from `http://localhost:8765/t1flair_v3_af16_noise0p75x/`, mirrored in `/home/congcongliu/figs/t1flair_v3_af16_noise0p75x/`.
**The figure of record is untouched** -- the new `--keep` flag appends `_sel7` to the file name.

**The numbers did not move.** Each panel's PSNR / SSIM / NMSE is that arm against the ground truth
and does not depend on which other arms are on the figure; dropping columns only changes which panel
the bold marks, since bold is the best of what is shown. Within these seven arms:

- **FLAIR** -- MC-GS is first on all three metrics in all three planes (transverse 180: PSNR #1, SSIM #1, NMSE #1; coronal 124: PSNR #1, SSIM #1, NMSE #1; sagittal 76: PSNR #1, SSIM #1, NMSE #1)
- **T1** -- MC-GS ranks (transverse 180: PSNR #1, SSIM #3, NMSE #1; coronal 124: PSNR #1, SSIM #1, NMSE #1; sagittal 76: PSNR #1, SSIM #1, NMSE #1)

`--keep` and the caption fix that comes with it are described in
`../T1FLAIR_AF20_v3_jiangyizhou/README.md` (same patch, applied by one generator to all ten scripts).
Without `--keep` the script is byte-for-byte the old one: re-rendering this topic's figure of record
reproduces it exactly.
