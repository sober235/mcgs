# Retrospective AF 12 on the 3D T1 + FLAIR pair (same subject, controlled swap)

Companion to `../T1T2_AF12_retro/`. **Read that README first** — data provenance, the geometry
adjudication, the mask, the ESPIRiT recipe, the known-zero region, the eight methods and the
scoring gauge are all identical here and are documented there, not repeated.

## What differs

Exactly one thing: the second contrast is `flair_train_huangyuwei_cc` instead of
`t2_train_huangyuwei_cc`. Same subject (`vol1_huangyuwei`), same exam, same grid
`(256, 256, 176, 26)`, same mask, same maps recipe, same frozen working point, same seed, same
gauge. It is a controlled swap, built to ask whether the null joint-versus-single result on
T1+T2 was about the contrast pairing.

**The T1 single-contrast arms are reused, not recomputed.** They depend only on the T1 k-space,
the mask and the T1 maps, all byte-identical to the T1+T2 topic, so re-running them would burn
about six GPU-hours to reproduce the same numbers. `recon/*_T1_*.npy`, `logs/{classical,zsssdu,
dip,inr}_T1.json` and `results_arc/t1flair_af12_single_T1_score.json` are copies of that topic's.
Only the FLAIR arms and the **T1+FLAIR joint run** are new here.

| contrast | PE-plane support | AF vs the full grid | AF vs its own support |
|---|---|---|---|
| T1 | 1.0000 | 11.71 | 11.71 |
| FLAIR | 0.7577 | 12.86 | 9.74 |

## Co-registration: FLAIR is the better-matched partner

| pair | peak cross-correlation shift | centroid offset |
|---|---|---|
| T1 / FLAIR | **(0, 0, 0) voxels** | 0.63 voxels |
| T1 / T2 (other topic) | (-1, 0, 0) voxels | 10.19 voxels |

Shared geometry assumes the two contrasts describe the same anatomy, so it has an easier job on
this pair than on T1+T2.

## Results

`logs/summary.tsv`. FLAIR, sorted by head-only PSNR:

| method | PSNR | SSIM | head PSNR | head SSIM |
|---|---|---|---|---|
| **MC-GS joint v2** (transfer check, see below) | 37.844 | 0.8835 | **33.492** | 0.9023 |
| MC-GS single v2 (ablation) | 37.639 | 0.8708 | 33.367 | 0.8975 |
| ZS-SSDU | 38.012 | 0.9525 | 33.269 | **0.9235** |
| CS-SENSE | 37.897 | 0.9346 | 33.252 | 0.9127 |
| CG-SENSE | 34.616 | 0.7328 | 32.192 | 0.8834 |
| MC-GS joint | 35.466 | 0.8669 | 31.806 | 0.8870 |
| MC-GS single | 35.176 | 0.8448 | 31.755 | 0.8814 |
| 3D-INR | 33.919 | 0.7932 | 29.939 | 0.8070 |
| 3D-DIP | 34.288 | 0.7924 | 29.397 | 0.8048 |
| ZF | 31.677 | 0.8503 | 26.738 | 0.6882 |

The T1 table is in `logs/summary.tsv` and matches the T1+T2 topic except for the joint rows, which
are necessarily different because T1 is now paired with FLAIR (v2: whole 39.253 / head 34.950, head SSIM 0.9379, against CS-SENSE 34.607 / 0.9436).

### The joint-versus-single result, four channel-pairings

| pairing | channel | joint − single, whole | joint − single, head-only |
|---|---|---|---|
| T1+T2 | T1 | +0.343 | −0.002 |
| T1+T2 | T2 | +0.090 | **−0.376** |
| T1+FLAIR | T1 | **+0.891** | −0.115 |
| T1+FLAIR | FLAIR | +0.290 | **+0.051** |

One pattern, four times: the joint model always gains on the whole-volume number and never gains
inside the head. The FLAIR pairing produces the largest whole-volume gain of the four, +0.891 dB
on T1, and still returns nothing head-only. Since whole-volume metrics on this data are dominated
by the background (see the T1+T2 README on why `ecalib -c 0.0` makes them so), the honest reading
is that shared geometry is buying background suppression from the explicit representation, not
better anatomy.

That the better-registered pairing does not rescue the head-only number is the informative part:
it removes co-registration quality as the explanation for the T1+T2 null result. The explanation
turned out to be the cold-start path itself, common to every MC-GS run in both topics — see below.

### MC-GS joint v2 — the T1+T2 recipe transferred unchanged (2026-09-10)

The autoresearch loop run on the T1+T2 pair (framework repo `autoresearch_t1t2.md`, and the
"MC-GS joint v2" section of the T1+T2 README) found three defects in the fresh-cloud path —
uniform init over the render box with 72 % of the points in air, edge-TV kappa calibrated on
the first render's speckle, no capacity growth — and fixed them with four structural changes
(points drawn inside the ZF head support, 3-NN spacing scales, kappa from the ZF image, 1.2M
points). Because those changes were selected on the T1+T2 pair with the ground truth in the
scorer, this topic is a **transfer check: the recipe was applied here without touching a single
knob** (`ar5_tf_hyw_t1flair`, 4000 iterations, volumes in `results_arc/`).

| channel | v2 head PSNR | best other arm | v2 head SSIM | best other arm | v2 head HFEN | best other arm |
|---|---|---|---|---|---|---|
| T1 | **34.950** | CS-SENSE 34.607 | 0.9379 | CS-SENSE 0.9436 | **0.0851** | CS-SENSE 0.1071 |
| FLAIR | **33.492** | ZS-SSDU 33.269 | 0.9023 | ZS-SSDU 0.9235 | **0.1150** | CS-SENSE 0.1302 |

The transfer holds on head PSNR (+0.34 dB on T1, +0.22 dB on FLAIR over the best other arm; +1.77 / +1.69 dB over the
original joint arm) and on head HFEN, and does not hold on head SSIM, where ZS-SSDU keeps a
0.02 lead on FLAIR. Figures regenerated with the tenth column
(`figures/t1flair_af12_{T1,FLAIR}_plane{0,1,2}.png`).

Single-contrast arms of the same recipe (`ar6_single_T1_v2`, shared with the T1+T2 topic, and
`ar9_single_FLAIR_v2_hyw`): T1 34.680 / FLAIR 33.367 head PSNR. The joint model is +0.27 / +0.13 dB
above its own single-contrast ablation on this pairing, i.e. here the shared geometry helps both
channels. The compact figures (`fig_slice.py`, `fig_planes_v2.py`) show only the v2 pair,
labelled "MC-GS joint (ours)" and "MC-GS single (ours)"; the superseded arms stay in the table.

## Regenerating

Same commands as the T1+T2 topic with this topic's scripts. The two long poles are the FLAIR
ESPIRiT map (`scripts/prep_03_csm_flair.sh`, about 45 min, CPU only) and the chain that waits for
it and launches every arm (`scripts/run_all_flair.sh`).

`scripts/setup_subject.sh <vol_dir> <subject> <short_tag>` clones this topic onto another subject
of the CFL set; it retargets every path, the CFL file names, the experiment names and the figure
output directory. Used to create `../T1FLAIR_AF12_v3_jiangyizhou/`.

## Gotcha added here

**A cloned topic must write its figures somewhere else.** The first clone kept the parent's
figure directory and file-name prefix and silently overwrote the parent's panels. Each topic now
writes `figs/<topic>/<topic>_*.png`, and `setup_subject.sh` carries the fix.
