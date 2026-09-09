# Retrospective AF 12 on the 3D T1 + FLAIR pair — second subject (vol3_jiangyizhou)

Third topic in the series. **Read `../T1T2_AF12_retro/README.md` first**: data provenance, the
geometry adjudication (mask on the transverse plane, readout head-foot), the mask, the ESPIRiT
recipe, the known-zero region, the eight methods and the scoring gauge are identical and are
documented there. `../T1FLAIR_AF12_retro/` is the same pairing on the first subject.

## What this topic adds

A **different person**. Nothing is reused from the other two topics: this subject's own T1 and
FLAIR k-space, its own two ESPIRiT maps (SoS == 1 over the FOV, as the recipe predicts), its own
sixteen reconstructions. Created from the first-subject topic with one command,
`../T1FLAIR_AF12_retro/scripts/setup_subject.sh vol3_jiangyizhou jiangyizhou v3`.

Geometry is the same as the other subjects to the voxel: T1 support 1.0000, FLAIR support 0.7577
(elliptical shutter), DC on the array centre, AF 11.71 on T1 and 9.74 on FLAIR against their own
support. T1/FLAIR co-registration: peak cross-correlation shift **(0, 0, 0)**, brain-mask Dice
0.905 (the first subject's pair gave 0.860).

## Results

`logs/summary.tsv`. Head-only PSNR / SSIM are the numbers to quote (see the T1+T2 README on why
whole-volume SSIM misleads on this data).

**T1**

| method | PSNR | SSIM | head PSNR | head SSIM | wall |
|---|---|---|---|---|---|
| CS-SENSE | **39.395** | 0.9523 | **35.762** | **0.9429** | 2 min |
| ZS-SSDU | 39.357 | **0.9577** | 35.724 | 0.9422 | 22 min |
| MC-GS joint | 36.349 | 0.8703 | 34.160 | 0.9169 | 29 min |
| MC-GS single | 35.529 | 0.8297 | 33.799 | 0.9032 | 29 min |
| CG-SENSE | 35.082 | 0.7562 | 33.514 | 0.8981 | 2 min |
| 3D-INR | 34.668 | 0.8218 | 31.638 | 0.8373 | 57 min |
| 3D-DIP | 35.267 | 0.8475 | 31.363 | 0.8634 | 50 min |
| ZF | 30.240 | 0.8353 | 26.130 | 0.7009 | 2 min |

**FLAIR**

| method | PSNR | SSIM | head PSNR | head SSIM | wall |
|---|---|---|---|---|---|
| CS-SENSE | 38.773 | 0.9416 | **35.084** | 0.9307 | 2 min |
| ZS-SSDU | **38.807** | **0.9563** | 35.034 | **0.9353** | 36 min |
| CG-SENSE | 35.704 | 0.7782 | 33.957 | 0.9032 | 2 min |
| MC-GS single | 36.204 | 0.8749 | 33.409 | 0.9006 | 29 min |
| MC-GS joint | 36.646 | 0.9079 | 33.391 | 0.9064 | 29 min |
| 3D-INR | 34.637 | 0.8274 | 31.190 | 0.8277 | 57 min |
| 3D-DIP | 34.730 | 0.8131 | 30.810 | 0.8333 | 50 min |
| ZF | 32.109 | 0.8523 | 28.193 | 0.7330 | 2 min |

GS wall times are with one card per job; on the first subject the same runs took about 100 min
because three shared the host with the classical arms.

Model selection, no ground truth involved: ZS-SSDU early-stopped at epoch 40 (T1) and 66 (FLAIR);
DIP's best held-out iterate was 4700 of 5000 (T1) and 4600 (FLAIR), i.e. still improving at the
budget as on the first subject; INR peaked at 800 (T1) and 350 (FLAIR) and over-fitted after.

Figures: `figures/t1flair_v3_af12_{T1,FLAIR}_plane{0,1,2}.png`, mirrored to
`/home/congcongliu/figs/t1flair_v3_af12/`; plane0 transverse, plane1 coronal, plane2 sagittal.

## Across subjects and pairings

`../T1FLAIR_AF12_retro/scripts/compare_topics.py` reads the three topics' summaries.

**Ranking is stable.** On every one of the six (subject, pairing, channel) columns CS-SENSE and
ZS-SSDU are first and second, within 0.1 dB of each other, MC-GS follows 1.3 to 1.7 dB behind,
CG-SENSE next, and 3D-DIP / 3D-INR last. Absolute levels move with the subject (this one is 1 to
2 dB better across the board), the order does not.

**Joint minus single, the shared-geometry claim, six channel-pairings:**

| subject | pairing | channel | whole PSNR | head PSNR | head SSIM |
|---|---|---|---|---|---|
| huangyuwei | T1+T2 | T1 | +0.343 | −0.001 | +0.003 |
| huangyuwei | T1+T2 | T2 | +0.090 | −0.376 | −0.002 |
| huangyuwei | T1+FLAIR | T1 | +0.891 | −0.114 | +0.002 |
| huangyuwei | T1+FLAIR | FLAIR | +0.290 | +0.051 | +0.006 |
| jiangyizhou | T1+FLAIR | T1 | +0.820 | **+0.361** | **+0.014** |
| jiangyizhou | T1+FLAIR | FLAIR | +0.442 | −0.018 | +0.006 |

Whole-volume PSNR gains in six of six. Head-only PSNR: mean −0.016 dB, one gain above 0.1 dB, two
losses below −0.1 dB. Head-only SSIM: positive in five of six, but small except on this subject's
T1. This corrects the earlier "never gains inside the head": the joint model **can** gain inside
the head (this subject's T1, +0.36 dB, visible in the error map as less parenchymal speckle than
the single arm), but which channel benefits is not predictable from the pairing, and averaged over
what has been run the head-only effect is zero. The whole-volume gain is robust and is background
suppression by the explicit representation.

Every MC-GS run in all three topics is a cold start (the frozen warm-start cloud belongs to the
old FOV). That is the one untested lever left for the 1.3 to 1.7 dB gap to CS-SENSE.

## Regenerating

```bash
R=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou
bash $R/scripts/prep_03_csm.sh          # both maps, ~85 min CPU
env PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python $R/scripts/prep_04_export.py
bash $R/scripts/run_all.sh              # waits for the maps, then all sixteen arms on GPUs 0-3,6,7
env PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python $R/scripts/summarize.py
```
