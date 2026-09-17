# Retrospective nominal AF 16 on the 3D T1 + FLAIR pair, jiangyizhou -- measurements with 1 sigma0 of added noise

Noise-robustness study, second level, 2026-09-16. **Read `../T1FLAIR_AF16_noise2x_v3_jiangyizhou/README.md` first** (the
2 sigma0 level; everything there applies here) and `../T1FLAIR_AF16_v3_jiangyizhou/README.md` (the zero-noise column).
Spec: `/data0/congcong/code/GS/mcgs_noise_wt/docs/superpowers/specs/2026-09-16-jyz-af16-noise-robustness-design.md`,
section "Addendum (2026-09-16 23:20)".

## What this topic adds

The same study at half the added noise. The user asked for this level on 2026-09-16 after the interim 2 sigma0 numbers
showed the frozen GS recipe losing 5-6 dB. Every arm is run again with the frozen recipe and the same grids; the GS
experiments carry the prefix `af16n1_`, the MC-INR run the tag `_noise1x`.

### The noise

Per-sample standard deviation sigma = 1 sigma0_k, **the same seed and the same draws as the 2 sigma0 level**, so this
topic's noise is exactly half of that topic's (common random numbers; checked sample by sample as `nested_vs_2x`, the
largest ratio of the difference to a single-precision rounding bound, limit 1). Generator `scripts/make_noisy_kspace.py`,
report `logs/make_noisy_kspace.json`.

| contrast | acquired lines (fraction) | sigma0_img | sigma0_k | added sigma | expected air noise | measured air noise | nested_vs_2x | checks |
|---|---|---|---|---|---|---|---|---|
| T1 | 45056 (1.0000) | 0.01408 | 0.01408 | 0.01408 | 0.01991 | 0.01990 | 0.29 | all pass |
| FLAIR | 34139 (0.7577) | 0.01282 | 0.01472 | 0.01472 | 0.01812 | 0.01812 | 0.29 | all pass |

## Results

<!-- TABLES:BEGIN -->
**T1** (sorted by head-only PSNR; no noise column = the same arm in `T1FLAIR_AF16_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @no noise | this − no noise |
|---|---|---|---|---|---|---|---|---|
| CS-SENSE (TV, converged) (λ 0.001) | 37.653 | 0.9383 | **34.018** | **0.9175** | 0.1586 | 0.828 | 34.945 | -0.93 |
| MC-GS | 37.645 | 0.9185 | 34.016 | 0.9109 | 0.1385 | 0.958 | 35.706 | -1.69 |
| ZS-SSDU | 37.589 | 0.9330 | 33.950 | 0.9167 | 0.1701 | 0.991 | 34.150 | -0.20 |
| joint CS-SENSE (JTV) (λ 0.0014) | 37.652 | 0.9408 | 33.935 | 0.9171 | 0.1645 | 0.814 | 34.831 | -0.90 |
| GS-single | 37.382 | 0.9176 | 33.799 | 0.9062 | 0.1438 | 0.968 | 35.558 | -1.76 |
| CS-SENSE | 37.343 | 0.9324 | 33.730 | 0.9140 | 0.1729 | 0.800 | 34.409 | -0.68 |
| L1-SPIRiT (λ 0.002) | 36.391 | 0.9163 | 32.772 | 0.8913 | 0.1836 | 0.896 | 33.788 | -1.02 |
| CG-SENSE | 32.770 | 0.6558 | 31.613 | 0.8481 | 0.2397 | 0.966 | 32.354 | -0.74 |
| 3D-DIP | 34.888 | 0.8525 | 30.944 | 0.8518 | 0.3086 | 0.632 | 30.994 | -0.05 |
| MC-INR (Niessen 2025) | 26.091 | 0.4259 | 29.713 | 0.7492 | 0.2663 | 1.219 | 31.590 | -1.88 |
| 3D-INR | 31.527 | 0.6216 | 29.538 | 0.7601 | 0.2519 | 1.358 | 30.375 | -0.84 |
| ZF | 29.946 | 0.8257 | 25.839 | 0.6792 | 0.6951 | 0.279 | 25.841 | -0.00 |

**FLAIR** (sorted by head-only PSNR; no noise column = the same arm in `T1FLAIR_AF16_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @no noise | this − no noise |
|---|---|---|---|---|---|---|---|---|
| MC-GS | 37.317 | 0.9128 | **33.635** | 0.8982 | 0.1360 | 0.899 | 34.928 | -1.29 |
| GS-single | 37.302 | 0.8992 | 33.628 | 0.8977 | 0.1364 | 0.899 | 34.886 | -1.26 |
| joint CS-SENSE (JTV) (λ 0.0014) | 37.354 | 0.9331 | 33.602 | 0.9039 | 0.1483 | 0.739 | 34.355 | -0.75 |
| CS-SENSE (TV, converged) (λ 0.001) | 37.218 | 0.9281 | 33.556 | **0.9043** | 0.1488 | 0.710 | 34.302 | -0.75 |
| CS-SENSE | 36.999 | 0.9242 | 33.345 | 0.9018 | 0.1535 | 0.691 | 33.806 | -0.46 |
| ZS-SSDU | 37.082 | 0.9244 | 33.289 | 0.9019 | 0.1644 | 0.984 | 34.005 | -0.72 |
| L1-SPIRiT (λ 0.002) | 36.559 | 0.9187 | 32.848 | 0.8842 | 0.1506 | 0.847 | 33.720 | -0.87 |
| CG-SENSE | 33.326 | 0.6730 | 32.055 | 0.8511 | 0.1813 | 0.875 | 32.826 | -0.77 |
| MC-INR (Niessen 2025) | 28.332 | 0.4826 | 30.734 | 0.7839 | 0.2038 | 1.042 | 32.243 | -1.51 |
| 3D-DIP | 34.591 | 0.8265 | 30.637 | 0.8296 | 0.2740 | 0.513 | 30.650 | -0.01 |
| 3D-INR | 32.580 | 0.6876 | 29.900 | 0.7712 | 0.2294 | 1.099 | 30.257 | -0.36 |
| ZF | 31.791 | 0.8460 | 27.876 | 0.7122 | 0.6596 | 0.266 | 27.879 | -0.00 |

**MC-GS joint minus MC-GS single**

| channel | whole PSNR | head PSNR | head SSIM | (no noise: whole / head) |
|---|---|---|---|---|
| T1 | +0.263 | +0.217 | +0.0047 | +0.124 / +0.148 |
| FLAIR | +0.015 | +0.007 | +0.0005 | +0.086 / +0.042 |
<!-- TABLES:END -->

### Reading at 1 sigma0

**At 1 sigma0 the frozen MC-GS recipe keeps head PSNR and head HFEN, and loses head SSIM.** It is first on FLAIR head
PSNR and on head HFEN in both contrasts, second on T1 head PSNR (0.003 dB behind the converged TV; the recipe's
seed-to-seed spread on clean data is 0.001 dB, on noisy data it was not measured), and fifth on head SSIM in both
contrasts (`logs/compare_noise_head_{psnr,ssim,hfen}.md`).

| contrast | metric | MC-GS, no noise -> 1 sigma0 | strongest comparison arm at 1 sigma0 | margin, no noise -> 1 sigma0 | rank at 1 sigma0 |
|---|---|---|---|---|---|
| T1 | head PSNR | 35.706 -> 34.016 | CS-SENSE (TV, converged) 34.018 | +0.148 -> -0.003 | 2/12 |
| T1 | head SSIM | 0.9396 -> 0.9109 | CS-SENSE (TV, converged) 0.9175 | +0.0025 -> -0.0066 | 5/12 |
| T1 | head HFEN | 0.1139 -> 0.1385 | GS-single 0.1438 | +0.0047 -> +0.0053 | 1/12 |
| FLAIR | head PSNR | 34.928 -> 33.635 | GS-single 33.628 | +0.042 -> +0.007 | 1/12 |
| FLAIR | head SSIM | 0.9261 -> 0.8982 | CS-SENSE (TV, converged) 0.9043 | +0.0013 -> -0.0061 | 5/12 |
| FLAIR | head HFEN | 0.1121 -> 0.1360 | GS-single 0.1364 | +0.0016 -> +0.0004 | 1/12 |

(no noise = `T1FLAIR_AF16_v3_jiangyizhou`, where GS-single is the strongest comparison arm on every cell; HFEN is
lower-is-better, its margin is the other arm minus MC-GS.)

**The loss is about twice the re-tuned classical arms' loss.** MC-GS loses 1.69 / 1.29 dB head PSNR (T1 / FLAIR),
GS-single 1.76 / 1.26; the converged TV loses 0.93 / 0.75, JTV 0.90 / 0.75, ZS-SSDU 0.20 / 0.72. The classical
families again move their lambda up (TV 0.0007 -> 0.001 and 0.0005 -> 0.001, JTV 0.001 -> 0.0014 and 0.0007 -> 0.0014,
L1-SPIRiT 0.001 -> 0.002), worth 0.20 / 0.30 dB (TV), 0.12 / 0.34 dB (JTV) and 0.40 / 0.42 dB (L1-SPIRiT) over their
zero-noise lambda (frozen-lambda view in `logs/compare_noise_head_psnr.md`). With those families also frozen at their
zero-noise lambda, as MC-GS is, MC-GS would lead T1 head PSNR by +0.066 dB (over ZS-SSDU, which has no lambda) and
FLAIR by +0.007 (GS-single), and would still trail ZS-SSDU on head SSIM (-0.0058 / -0.0037); this is read off the two
tables, no extra run. Hard data consistency costs the GS arms 0.04 dB (T1) and 0.02 dB (FLAIR) head PSNR here
(render-only diagnostic, same file). Every selected lambda is interior; no grid was extended at this level.

### The two noise levels side by side

`logs/compare_noise_levels_{head_psnr,head_ssim,head_hfen}.md` (`scripts/compare_noise_levels.py`): no noise, 1 sigma0,
2 sigma0, one gauge.

| contrast | | no noise | 1 sigma0 | 2 sigma0 |
|---|---|---|---|---|
| T1 | MC-GS head PSNR | 35.706 | 34.016 | 29.802 |
| T1 | MC-GS rank, head PSNR / SSIM / HFEN | 1 / 1 / 1 | 2 / 5 / 1 | 8 / 7 / 5 |
| T1 | head-PSNR margin over the strongest comparison arm | +0.148 (GS-single) | -0.003 (TV) | -3.328 (ZS-SSDU) |
| T1 | MC-GS sharpness ratio | 0.855 | 0.958 | 1.331 |
| T1 | MC-GS minus GS-single, head PSNR | +0.148 | +0.217 | +0.232 |
| FLAIR | MC-GS head PSNR | 34.928 | 33.635 | 29.946 |
| FLAIR | MC-GS rank, head PSNR / SSIM / HFEN | 1 / 1 / 1 | 1 / 5 / 1 | 9 / 9 / 7 |
| FLAIR | head-PSNR margin over the strongest comparison arm | +0.042 (GS-single) | +0.007 (GS-single) | -2.606 (JTV) |
| FLAIR | MC-GS sharpness ratio | 0.804 | 0.899 | 1.227 |
| FLAIR | MC-GS minus GS-single, head PSNR | +0.042 | +0.007 | -0.116 |

Doubling the noise (the same draws, scaled) multiplies the MC-GS head-PSNR loss by 3.5 (T1) and 3.9 (FLAIR), the
converged TV's by 2.4 and 2.4. The GS render takes in more noise as the level rises (sharpness ratio above 1 at
2 sigma0). The coupling gain of MC-GS over GS-single grows on T1 and shrinks to a loss on FLAIR.

### Figures

| figure | what |
|---|---|
| `http://localhost:8765/t1flair_v3_af16_noise1x/t1flair_v3_af16n1_noise_data.png` | the data: clean and noisy fully sampled images and their difference, both contrasts |
| `http://localhost:8765/t1flair_v3_af16_noise1x/t1flair_v3_af16n1_T1_row3planes_compact_180_124_76_inset48_headerr_p3_mcinr_spirit_zf_caps.png` | T1, figure of record (also `.pdf`, 600 dpi) |
| `http://localhost:8765/t1flair_v3_af16_noise1x/t1flair_v3_af16n1_FLAIR_row3planes_compact_180_124_76_inset48_headerr_p3_mcinr_spirit_zf_caps.png` | FLAIR, figure of record (also `.pdf`) |
| `http://localhost:8765/t1flair_v3_af16_noise1x/t1flair_v3_af16n1_T1_row3planes_compact_174_110_47_headerr_p3_mcinr_spirit_zf_caps.png` | T1, 174 / 110 / 47 |
| `http://localhost:8765/t1flair_v3_af16_noise1x/t1flair_v3_af16n1_FLAIR_row3planes_compact_174_110_47_headerr_p3_mcinr_spirit_zf_caps.png` | FLAIR, 174 / 110 / 47 |

The same files are in `figures/` (served from `/home/congcongliu/figs/t1flair_v3_af16_noise1x/`); columns, slices,
box, error maps and the three-line caption as in the 2 sigma0 topic.

**The slice with the largest MC-GS lead** (`scripts/best_planes_noise.py`, `logs/best_planes_noise.{json,log}`; the panel
gauge of the compact figure, every slice of every plane whose head area is at least half the largest, all ten comparison
columns including GS-single). MC-GS is first on slice PSNR and SSIM (hence NMSE) on 3 T1 and 12 FLAIR transverse slices,
1 FLAIR sagittal slice and no coronal slice; no slice is first in both contrasts, so no three-plane figure shows MC-GS
ahead everywhere. The largest lead: **T1 transverse 107** (skull base), PSNR 32.99 dB, +0.246 dB over GS-single, SSIM
0.8797, +0.0013 over CS-SENSE; FLAIR's best is transverse 124, +0.062 dB. At 2 sigma0 MC-GS is first on PSNR on no slice
(`../T1FLAIR_AF16_noise2x_v3_jiangyizhou/logs/best_planes_noise.log`). Figure (single plane, `--planes 0`):
`http://localhost:8765/t1flair_v3_af16_noise1x/t1flair_v3_af16n1_T1_compact_tra107_headerr_p3_mcinr_spirit_zf_caps.png`
(also `.pdf`, 600 dpi).

## Known limitations

As in the 2 sigma0 topic, and:
- The two noisy levels share one realization, so their difference is the effect of the noise scale alone, not of a new
  draw.
- The zero-noise JTV column comes from a two-point sweep in `T1FLAIR_AF16_v3_jiangyizhou` (lambda 0.0007 and 0.001;
  T1 chose the upper point, FLAIR the lower one), so its zero-noise value is a lower bound and its loss here may be
  understated by a little. It is not the strongest comparison arm at zero noise on any cell.
- The T1 head-PSNR ranking at this level rests on a 0.003 dB difference.

## Regenerating

```
python scripts/make_noisy_kspace.py                       # needs the 2 sigma0 topic's meas files for the nested check
bash scripts/launch_noise.sh --dry_run                    # the 57 jobs
setsid nohup bash scripts/launch_noise.sh --cards 2,1,5 > /dev/null 2>&1 < /dev/null &
```

The queue actually ran as `launch_noise.sh --cards 2,1,5 --light_only_cards 5` (card 5 carries another user's job and
takes light jobs only), alongside the 2 sigma0 queue on the same three cards.

## Status

2026-09-17 09:30: complete. 57 jobs, no failures, every selected lambda interior; finalize rc 0 at 09:29. The noise
figure was re-laid out at 09:21 (axes placed in inches, no column slack) in both topics.
