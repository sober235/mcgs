# Retrospective AF 16 on the 3D T1 + FLAIR pair — second subject (vol3_jiangyizhou)

Fifth topic in the series. **Read `../T1FLAIR_AF16_retro/README.md` first**: the AF 16 mask, the
worktree, the scheduler, the gauge proof and the gotchas are documented there and not repeated.
The data, geometry, maps, gauge and eight arms are those of `../T1FLAIR_AF12_v3_jiangyizhou/`
(this subject at AF 12), which in turn defers to `../T1T2_AF12_retro/README.md`.

## What this topic adds

The **same AF 16 study on a different person**. Its own T1 and FLAIR k-space and its own two
ESPIRiT maps (symlinks into `../T1FLAIR_AF12_v3_jiangyizhou/data/`, never written), the **same
mask** as the first subject (`data/mask_vdpois_256x176_af16.mat`, byte-identical; the AF 12 topics
shared their mask the same way), sixteen new reconstructions. Created from the first-subject topic
with `../T1FLAIR_AF16_retro/scripts/setup_af16_jyz.sh`; configs
`configs/t1flair_v3_af16_jyz{,_singleT1,_singleFLAIR}.yaml` in the worktree; run by the same
scheduler queue, after the first subject's core arms.

Mask against this subject's data: T1 support 1.0000 → AF 15.19; FLAIR support 0.7577 → AF 16.49
on the full grid, 12.49 against its own shutter (`logs/prep_04_export.json`). Identical to the
first subject to four digits, as at AF 12.

## Results

`logs/summary.tsv`; gauge proof `logs/post_gauge.log` (`GAUGE OK`). Head-only PSNR / SSIM are the
numbers to quote. AF 12 values per arm from `../T1FLAIR_AF12_v3_jiangyizhou/logs/summary.json`.

<!-- TABLES:BEGIN -->
**T1** (sorted by head-only PSNR; AF12 column = the same arm in `T1FLAIR_AF12_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @AF12 | AF16 − AF12 |
|---|---|---|---|---|---|---|---|---|
| MC-GS joint (autoresearch) | 39.016 | 0.9124 | **35.552** | **0.9351** | 0.1218 | 0.830 | -- | -- |
| L1-ESPIRiT (λ 0.0015) | 38.758 | 0.9479 | 35.209 | 0.9334 | 0.1286 | 0.849 | 36.334 | -1.12 |
| CS-SENSE (TV, converged) (λ 0.0007) | 38.646 | 0.9506 | 34.945 | 0.9317 | 0.1380 | 0.823 | 36.144 | -1.20 |
| joint CS-SENSE (JTV) (λ 0.001) | 38.595 | 0.9514 | 34.831 | 0.9311 | 0.1429 | 0.811 | 36.070 | -1.24 |
| CS-SENSE | 38.187 | 0.9393 | 34.409 | 0.9276 | 0.1531 | 0.768 | 35.762 | -1.35 |
| ZS-SSDU | 37.823 | 0.9399 | 34.150 | 0.9216 | 0.1646 | 1.016 | 35.724 | -1.57 |
| MC-GS joint | 34.628 | 0.8093 | 32.890 | 0.8893 | 0.1882 | 0.921 | 34.160 | -1.27 |
| CG-SENSE | 34.559 | 0.7621 | 32.354 | 0.8825 | 0.2239 | 0.859 | 33.514 | -1.16 |
| MC-GS single | 33.750 | 0.7608 | 32.331 | 0.8679 | 0.1875 | 1.018 | 33.799 | -1.47 |
| 3D-DIP | 34.938 | 0.8507 | 30.994 | 0.8549 | 0.3024 | 0.634 | 31.363 | -0.37 |
| 3D-INR | 32.903 | 0.7131 | 30.375 | 0.7951 | 0.2125 | 1.310 | 31.638 | -1.26 |
| ZF | 29.949 | 0.8242 | 25.841 | 0.6795 | 0.6950 | 0.276 | 26.130 | -0.29 |

**FLAIR** (sorted by head-only PSNR; AF12 column = the same arm in `T1FLAIR_AF12_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @AF12 | AF16 − AF12 |
|---|---|---|---|---|---|---|---|---|
| MC-GS joint (autoresearch) | 38.342 | 0.9045 | **34.807** | **0.9203** | 0.1188 | 0.788 | -- | -- |
| L1-ESPIRiT (λ 0.003) | 38.027 | 0.9310 | 34.503 | 0.9201 | 0.1210 | 0.789 | 35.720 | -1.22 |
| joint CS-SENSE (JTV) (λ 0.0007) | 38.048 | 0.9408 | 34.355 | 0.9177 | 0.1302 | 0.775 | 35.624 | -1.27 |
| CS-SENSE (TV, converged) (λ 0.0005) | 37.904 | 0.9365 | 34.302 | 0.9180 | 0.1309 | 0.751 | 35.581 | -1.28 |
| ZS-SSDU | 37.882 | 0.9461 | 34.005 | 0.9185 | 0.1529 | 1.036 | 35.034 | -1.03 |
| CS-SENSE | 37.613 | 0.9248 | 33.806 | 0.9116 | 0.1360 | 0.663 | 35.084 | -1.28 |
| CG-SENSE | 35.220 | 0.7873 | 32.826 | 0.8860 | 0.1608 | 0.783 | 33.957 | -1.13 |
| MC-GS joint | 35.743 | 0.8923 | 32.402 | 0.8834 | 0.1724 | 0.807 | 33.391 | -0.99 |
| MC-GS single | 34.989 | 0.8420 | 32.137 | 0.8701 | 0.1719 | 0.922 | 33.409 | -1.27 |
| 3D-DIP | 34.581 | 0.8188 | 30.650 | 0.8290 | 0.2769 | 0.510 | 30.810 | -0.16 |
| 3D-INR | 33.316 | 0.7454 | 30.257 | 0.7889 | 0.1998 | 1.139 | 31.190 | -0.93 |
| ZF | 31.792 | 0.8410 | 27.879 | 0.7126 | 0.6595 | 0.262 | 28.193 | -0.31 |

**MC-GS joint minus MC-GS single**

| channel | whole PSNR | head PSNR | head SSIM | (AF12: whole / head) |
|---|---|---|---|---|
| T1 | +0.878 | +0.559 | +0.0214 | +0.820 / +0.361 |
| FLAIR | +0.754 | +0.266 | +0.0133 | +0.442 / -0.018 |
<!-- TABLES:END -->

### Reading

1. **The classical arms were re-run converged and λ-swept on this subject (2026-09-10)**, after the
   AF12 topics' audit found the 09-09 CS-SENSE under-converged (35 ADMM iterations inherited from the CE
   study). **L1-wavelet ESPIRiT leads both contrasts (35.209 / 34.503)**, then converged TV
   (34.945 / 34.302) and joint vectorial TV (34.831 / 34.355); every λ curve is bracketed
   (`logs/classical_*_o200_lam*.json`, `logs/l1espirit_*_lam*.json`, `logs/jcssense_o200_lam*.json`,
   `scripts/run_classical_revisited*.sh`). λ was chosen on this subject against the ground truth — the
   same privilege the MC-GS autoresearch loop has. The 09-09 CS-SENSE row is kept for continuity; it is
   0.8 / 0.7 dB below L1-ESPIRiT.
2. **The cold-start MC-GS joint is sixth on T1 and seventh on FLAIR of eleven arms**, 2.3 / 2.1 dB behind
   L1-ESPIRiT. The shared-geometry effect still holds between the two cold-start MC-GS arms: joint − single
   = +0.56 dB (T1) and +0.27 dB (FLAIR) head-only, +0.021 / +0.013 head SSIM.
3. **The autoresearch loop on this subject** (worktree `mcgs_af16_wt`, branch `autoresearch-jyz-af16`,
   program `autoresearch_jyz_af16.md`, ledger `autoresearch_jyz_af16.tsv`) repairs the cold start
   (head-support initialisation, 3-NN scales, ZF-sourced edge-TV threshold), adds the wavelet-sparsity
   prior and doubles the capacity. Status 2026-09-10 22:20: best screened configuration 35.169 / 34.426
   at 1500 iterations (0.04 / 0.08 dB behind L1-ESPIRiT), full-budget promotion running. When it lands,
   the arm is registered with `scripts/add_gs_arm.py` and appears in the tables above and in the
   visual-gate figure.
4. Per-slice sweep (`logs/per_slice.json`) covers the original eight arms only: there, MC-GS joint ranked
   first on 5 of 600 T1 slices and 0 of 606 FLAIR slices.
5. 3D-INR (30.4 / 30.3) and 3D-DIP (31.0 / 30.7) stay 3 to 4 dB behind the leaders.

## Figures

`figures/`, mirrored to `/home/congcongliu/figs/t1flair_v3_af16/`, served at
`http://localhost:8765/t1flair_v3_af16/<file>`: `t1flair_v3_af16_{T1,FLAIR}_plane{0,1,2}.png`
(three mid-planes, all arms, jet error maps fixed [0, 0.1]) and
`t1flair_v3_af16_{T1,FLAIR}_butterfly_axis0_168.png` (the transverse slice the AF 12 topic of this
subject uses).

## Regenerating

See `../T1FLAIR_AF16_retro/README.md`; the scheduler runs both subjects, and
`scripts/postprocess.sh jyz <gpu>` does everything after the arms.
