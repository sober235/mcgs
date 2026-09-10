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
| CS-SENSE | 38.187 | 0.9393 | **34.409** | **0.9276** | 0.1531 | 0.768 | 35.762 | -1.35 |
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
| ZS-SSDU | 37.882 | 0.9461 | **34.005** | **0.9185** | 0.1529 | 1.036 | 35.034 | -1.03 |
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

1. **The leader is split on this subject**: CS-SENSE keeps T1 (34.41 vs ZS-SSDU 34.15) and ZS-SSDU
   takes FLAIR (34.01 vs 33.81). Both lose 1.0 to 1.6 dB from AF 12; here ZS-SSDU is not the
   robust outlier it is on the first subject (−1.57 / −1.03 dB against −0.47 / −0.58 there).
2. **MC-GS joint is third on T1 and fourth on FLAIR**, 1.52 dB behind CS-SENSE on T1 and 1.60 dB
   behind ZS-SSDU on FLAIR (AF 12: 1.60 / 1.69). As on the first subject it degrades less than its
   single arm and than CS-SENSE on both contrasts (−1.27 / −0.99 dB).
3. **Shared geometry gains inside the head in both channels again**: joint − single = +0.56 dB (T1)
   and +0.27 dB (FLAIR) head-only, +0.021 / +0.013 head SSIM. Over the two AF 16 subjects that is
   four positive channel-pairings of four (+0.05, +0.27, +0.56, +0.27), against two of four at AF 12
   (−0.11, +0.05, +0.36, −0.02). The whole-volume gain is unchanged in kind (+0.9 / +0.8 dB).
4. Per-slice sweep (`logs/per_slice.json`, all eight arms): MC-GS joint ranks first on 5 of
   600 scored T1 slices and 0 of 606 FLAIR slices; winners are CS-SENSE / ZS-SSDU
   almost everywhere (T1 {'cssense': 503, 'zsssdu': 91, 'gs_joint': 5, 'gs_single': 1}, FLAIR {'zsssdu': 374, 'cssense': 231, 'cgsense': 1}).
5. 3D-INR (30.4 / 30.3 head-only) and 3D-DIP (31.0 / 30.7) stay 3 to 4 dB behind the leaders.

## Figures

`figures/`, mirrored to `/home/congcongliu/figs/t1flair_v3_af16/`, served at
`http://localhost:8765/t1flair_v3_af16/<file>`: `t1flair_v3_af16_{T1,FLAIR}_plane{0,1,2}.png`
(three mid-planes, all arms, jet error maps fixed [0, 0.1]) and
`t1flair_v3_af16_{T1,FLAIR}_butterfly_axis0_168.png` (the transverse slice the AF 12 topic of this
subject uses).

## Regenerating

See `../T1FLAIR_AF16_retro/README.md`; the scheduler runs both subjects, and
`scripts/postprocess.sh jyz <gpu>` does everything after the arms.
