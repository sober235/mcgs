# Retrospective nominal AF 16 on the 3D T1 + FLAIR pair, jiangyizhou -- measurements with 0.5 sigma0 of added noise

Noise-robustness study, third level, 2026-09-17. **Read `../T1FLAIR_AF16_noise2x_v3_jiangyizhou/README.md` first** (the
2 sigma0 level; everything there applies here), `../T1FLAIR_AF16_noise1x_v3_jiangyizhou/README.md` (1 sigma0) and
`../T1FLAIR_AF16_v3_jiangyizhou/README.md` (the zero-noise column). Spec:
`/data0/congcong/code/GS/mcgs_noise_wt/docs/superpowers/specs/2026-09-16-jyz-af16-noise-robustness-design.md`,
section "Addendum (2026-09-17 13:40)".

## What this topic adds

The first point of the search for the noise range in which MC-GS leads (user, 2026-09-17): "favourable" means first on
all six head cells (head PSNR / SSIM / HFEN x T1 / FLAIR) against every comparison arm, GS-single included, with the
classical lambdas re-selected on the noisy data. If this level is favourable, 0.75 sigma0 follows, otherwise
0.25 sigma0. Every arm is run again with the frozen recipe and the same grids; the GS experiments carry the prefix
`af16n05_`, the MC-INR run the tag `_noise0p5x`. The 1 sigma0 and 2 sigma0 levels stay in the record.

### The noise

Per-sample standard deviation sigma = 0.5 sigma0_k, **the same seed and the same draws as the other levels**, so this
topic's noise is exactly half of the 1 sigma0 topic's (checked sample by sample as `nested_vs_1x`, the largest ratio of
the difference to a single-precision rounding bound, limit 1). Added noise power x1.25, SNR down about 1 dB. Generator
`scripts/make_noisy_kspace.py`, report `logs/make_noisy_kspace.json`.

| contrast | acquired lines (fraction) | sigma0_img | sigma0_k | added sigma | expected air noise | measured air noise | nested_vs_1x | checks |
|---|---|---|---|---|---|---|---|---|
| T1 | 45056 (1.0000) | 0.01408 | 0.01408 | 0.00704 | 0.01574 | 0.01574 | 0.29 | all pass |
| FLAIR | 34139 (0.7577) | 0.01282 | 0.01472 | 0.00736 | 0.01433 | 0.01433 | 0.29 | all pass |

## Results

<!-- TABLES:BEGIN -->
**T1** (sorted by head-only PSNR; no noise column = the same arm in `T1FLAIR_AF16_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @no noise | this − no noise |
|---|---|---|---|---|---|---|---|---|
| MC-GS | 38.880 | 0.9222 | **35.280** | **0.9336** | 0.1197 | 0.879 | 35.706 | -0.43 |
| GS-single | 38.718 | 0.9275 | 35.109 | 0.9306 | 0.1245 | 0.884 | 35.558 | -0.45 |
| CS-SENSE (TV, converged) (λ 0.001) | 38.401 | 0.9454 | 34.636 | 0.9284 | 0.1436 | 0.799 | 34.945 | -0.31 |
| joint CS-SENSE (JTV) (λ 0.001) | 38.270 | 0.9492 | 34.551 | 0.9264 | 0.1496 | 0.823 | 34.831 | -0.28 |
| CS-SENSE | 37.999 | 0.9447 | 34.223 | 0.9242 | 0.1583 | 0.775 | 34.409 | -0.19 |
| ZS-SSDU | 37.479 | 0.9319 | 33.836 | 0.9166 | 0.1710 | 1.024 | 34.150 | -0.31 |
| L1-SPIRiT (λ 0.001) | 36.849 | 0.9088 | 33.404 | 0.9001 | 0.1527 | 0.948 | 33.788 | -0.38 |
| CG-SENSE | 34.053 | 0.7294 | 32.156 | 0.8735 | 0.2280 | 0.888 | 32.354 | -0.20 |
| MC-INR (Niessen 2025) | 29.016 | 0.5143 | 31.129 | 0.8135 | 0.2406 | 1.062 | 31.590 | -0.46 |
| 3D-DIP | 34.841 | 0.8462 | 30.898 | 0.8512 | 0.3083 | 0.627 | 30.994 | -0.10 |
| 3D-INR | 32.641 | 0.6951 | 30.198 | 0.7889 | 0.2316 | 1.282 | 30.375 | -0.18 |
| ZF | 29.948 | 0.8247 | 25.840 | 0.6794 | 0.6950 | 0.277 | 25.841 | -0.00 |

**FLAIR** (sorted by head-only PSNR; no noise column = the same arm in `T1FLAIR_AF16_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @no noise | this − no noise |
|---|---|---|---|---|---|---|---|---|
| MC-GS | 38.298 | 0.9135 | **34.617** | **0.9204** | 0.1177 | 0.826 | 34.928 | -0.31 |
| GS-single | 38.224 | 0.8985 | 34.581 | 0.9193 | 0.1190 | 0.828 | 34.886 | -0.30 |
| joint CS-SENSE (JTV) (λ 0.001) | 37.901 | 0.9415 | 34.126 | 0.9140 | 0.1344 | 0.751 | 34.355 | -0.23 |
| CS-SENSE (TV, converged) (λ 0.0007) | 37.760 | 0.9377 | 34.071 | 0.9142 | 0.1351 | 0.726 | 34.302 | -0.23 |
| ZS-SSDU | 37.671 | 0.9397 | 33.836 | 0.9135 | 0.1520 | 1.000 | 34.005 | -0.17 |
| CS-SENSE | 37.492 | 0.9329 | 33.683 | 0.9093 | 0.1405 | 0.670 | 33.806 | -0.12 |
| L1-SPIRiT (λ 0.002) | 37.177 | 0.9243 | 33.394 | 0.8989 | 0.1351 | 0.811 | 33.720 | -0.33 |
| CG-SENSE | 34.683 | 0.7520 | 32.622 | 0.8769 | 0.1662 | 0.808 | 32.826 | -0.20 |
| MC-INR (Niessen 2025) | 31.252 | 0.5863 | 31.881 | 0.8410 | 0.1788 | 0.926 | 32.243 | -0.36 |
| 3D-DIP | 34.536 | 0.8183 | 30.591 | 0.8265 | 0.2803 | 0.509 | 30.650 | -0.06 |
| 3D-INR | 33.089 | 0.7270 | 30.168 | 0.7841 | 0.2075 | 1.127 | 30.257 | -0.09 |
| ZF | 31.792 | 0.8425 | 27.878 | 0.7125 | 0.6596 | 0.263 | 27.879 | -0.00 |

**MC-GS joint minus MC-GS single**

| channel | whole PSNR | head PSNR | head SSIM | (no noise: whole / head) |
|---|---|---|---|---|
| T1 | +0.162 | +0.171 | +0.0030 | +0.124 / +0.148 |
| FLAIR | +0.074 | +0.035 | +0.0012 | +0.086 / +0.042 |
<!-- TABLES:END -->

### Reading at 0.5 sigma0

**At 0.5 sigma0 the frozen MC-GS recipe is first on all six head cells.** The strongest comparison arm is GS-single on
every cell (`logs/compare_noise_head_{psnr,ssim,hfen}.md`, `logs/compare_noise_levels_{head_psnr,head_ssim,head_hfen}.md`).

| contrast | metric | MC-GS, no noise -> 0.5 sigma0 | strongest comparison arm at 0.5 sigma0 | margin, no noise -> 0.5 sigma0 | best non-GS arm, margin |
|---|---|---|---|---|---|
| T1 | head PSNR | 35.706 -> 35.280 | GS-single 35.109 | +0.148 -> +0.171 | CS-SENSE (TV, converged) 34.636, +0.644 |
| T1 | head SSIM | 0.9396 -> 0.9336 | GS-single 0.9306 | +0.0025 -> +0.0030 | CS-SENSE (TV, converged) 0.9284, +0.0052 |
| T1 | head HFEN | 0.1139 -> 0.1197 | GS-single 0.1245 | +0.0047 -> +0.0048 | CS-SENSE (TV, converged) 0.1436, +0.0239 |
| FLAIR | head PSNR | 34.928 -> 34.617 | GS-single 34.581 | +0.042 -> +0.036 | Joint SENSE 34.126, +0.491 |
| FLAIR | head SSIM | 0.9261 -> 0.9204 | GS-single 0.9193 | +0.0013 -> +0.0012 | CS-SENSE (TV, converged) 0.9142, +0.0062 |
| FLAIR | head HFEN | 0.1121 -> 0.1177 | GS-single 0.1190 | +0.0016 -> +0.0013 | Joint SENSE 0.1344, +0.0167 |

(HFEN is lower-is-better; its margins are the other arm minus MC-GS.)

MC-GS loses 0.43 / 0.31 dB head PSNR against zero noise (GS-single 0.45 / 0.30, converged TV 0.31 / 0.23, JTV 0.28 /
0.23). The classical lambdas move little (TV 0.0007 -> 0.001 on T1, 0.0005 -> 0.0007 on FLAIR; JTV 0.001 -> 0.001 and
0.0007 -> 0.001; L1-SPIRiT T1 kernel 5 -> 3 at 0.001, FLAIR 0.001 -> 0.002), worth at most 0.06 dB (frozen-lambda view);
every selected lambda is interior. Hard data consistency changes the GS arms by +0.010 (T1) and -0.016 dB (FLAIR). The
sharpness ratio of MC-GS is 0.879 / 0.826 (zero noise 0.855 / 0.804). ZS-SSDU stopped early at epoch 28 (T1) and reads
33.836 dB, below its 1 sigma0 run (33.950): single runs of ZS-SSDU vary by that much.

In all four compact figures (180 / 124 / 76 with the box, 174 / 110 / 47; both contrasts) MC-GS is first on PSNR, SSIM
and NMSE in every plane.

**Slices with the largest lead** (`scripts/best_planes_noise.py`, `logs/best_planes_noise.{json,log}`; the panel gauge,
every slice with at least half the largest head area, ten comparison columns including GS-single). MC-GS is first on
slice PSNR and SSIM on every eligible T1 slice of all three planes (180 / 135 / 128) and on 136 / 129 / 127 FLAIR
slices (of 178 / 140 / 128); both contrasts at once on 136 transverse, 127 coronal and 127 sagittal slices. The paired
pick (largest smaller-of-the-two PSNR lead, GS-single second everywhere): transverse 170 (T1 +0.240 dB, FLAIR +0.100),
coronal 166 (+0.150, +0.062), sagittal 147 (+0.207, +0.083; a lateral slice). Figures:

| figure | what |
|---|---|
| `http://localhost:8765/t1flair_v3_af16_noise0p5x/t1flair_v3_af16n05_T1_row3planes_compact_170_166_147_headerr_p3_mcinr_spirit_zf_caps.png` | T1, largest-lead slices (also `.pdf`) |
| `http://localhost:8765/t1flair_v3_af16_noise0p5x/t1flair_v3_af16n05_FLAIR_row3planes_compact_170_166_147_headerr_p3_mcinr_spirit_zf_caps.png` | FLAIR, largest-lead slices (also `.pdf`) |
| `http://localhost:8765/t1flair_v3_af16_noise0p5x/t1flair_v3_af16n05_T1_row3planes_compact_180_124_76_inset48_headerr_p3_mcinr_spirit_zf_caps.png` | T1, figure-of-record slices (also `.pdf`) |
| `http://localhost:8765/t1flair_v3_af16_noise0p5x/t1flair_v3_af16n05_FLAIR_row3planes_compact_180_124_76_inset48_headerr_p3_mcinr_spirit_zf_caps.png` | FLAIR, figure-of-record slices (also `.pdf`) |
| `http://localhost:8765/t1flair_v3_af16_noise0p5x/t1flair_v3_af16n05_T1_row3planes_compact_174_110_47_headerr_p3_mcinr_spirit_zf_caps.png` | T1, 174 / 110 / 47 |
| `http://localhost:8765/t1flair_v3_af16_noise0p5x/t1flair_v3_af16n05_FLAIR_row3planes_compact_174_110_47_headerr_p3_mcinr_spirit_zf_caps.png` | FLAIR, 174 / 110 / 47 |
| `http://localhost:8765/t1flair_v3_af16_noise0p5x/t1flair_v3_af16n05_noise_data.png` | the data at this level |

The same PNGs are in `figures/` (served from `/home/congcongliu/figs/t1flair_v3_af16_noise0p5x/`).

## Known limitations

As in the 2 sigma0 and 1 sigma0 topics; all noisy levels share one realization. One run per arm per level; ZS-SSDU's
run-to-run spread is visible above.

## Regenerating

```
python scripts/make_noisy_kspace.py                       # needs the 1 sigma0 topic's meas files for the nested check
bash scripts/launch_noise.sh --dry_run                    # the 57 jobs
setsid nohup bash scripts/launch_noise.sh --cards 1,2,5,6 --max_gpu_jobs 4 > /dev/null 2>&1 < /dev/null &
```

## Status

2026-09-17 13:50: measurements made, all checks pass; queue about to start on cards 1, 2, 5, 6.
2026-09-17 17:22: complete. 57 jobs, no failures, every selected lambda interior, finalize rc 0; MC-GS first on all six
head cells. 18:00: slice search and the largest-lead figures (170 / 166 / 147). The next level, 0.75 sigma0, started
15:21 on the decision rule (`../T1FLAIR_AF16_noise0p75x_v3_jiangyizhou/`).
