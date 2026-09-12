# Autoresearch on the retrospective AF12 pairs, part 2: a wavelet prior, v3.2 and two held-out subjects — session record, 2026-09-10 / 11

Continues `2026-09-10-autoresearch-t1t2-coldstart.md`, which ended at v2 (the repaired cold start). The goal is
unchanged (set by the user on 2026-09-09): MC-GS above every other arm of the AF12 tables on imaging metrics.
The tuning pair is hyw T1+T2 (`Results_20260908/T1T2_AF12_retro`), the only topic used for selection. hyw T1+FLAIR
checks transfer to another contrast on the same subject. jiangyizhou and tangbing are held out and were never used
to select anything. Every change was a single delta whose decision rule was written into the program before the
run. Program `autoresearch_t1t2.md` (round log with every pre-registration and result), ledger
`autoresearch_t1t2.tsv` (100 rows: 93 training runs and 7 two-seed averages, about 80 GPU-hours for the whole
campaign, part 1 included), runner `autoresearch_t1t2.sh`, branch `autoresearch-t1t2`.

Gauge: head-only unless marked whole-volume (head = |GT| > 0.05 · P99.9; PSNR at the ground-truth maximum; SSIM
with a 7³ window; HFEN with a LoG of σ 1.5). Whole-volume SSIM is dominated by the air noise that `ecalib -c 0.0`
leaves, so claims are made on the head gauges and whole-volume SSIM is reported as it stands.

Naming: MC-GS is the joint arm and GS-single the single-contrast arm. The tables of this line still carry the older
labels "MC-GS joint" and "MC-GS single".

## 1. The bar moved: converged classical arms (09-10 afternoon)

At the user's request, joint CS-SENSE (vectorial TV), L1-wavelet ESPIRiT and P-LORAKS were added and the TV arm was
audited. The 09-09 TV arm had run 35 ADMM iterations; converged (200 outer iterations, λ re-swept on hyw) it is
0.36–0.95 dB higher. L1-ESPIRiT (BART, 100 FISTA iterations, λ per contrast selected on hyw) became the strongest
other arm, 0.4–0.8 dB above v2. Head PSNR:

| arm | hyw T1 | hyw T2 | hyw FLAIR | jyz T1 | jyz FLAIR |
|---|---|---|---|---|---|
| CS-SENSE, TV, converged | 34.966 | 35.168 | 33.873 | 36.144 | 35.581 |
| joint CS-SENSE, vectorial TV | 35.020 | 35.008 | 33.929 | 36.070 | 35.624 |
| L1-ESPIRiT | **35.530** | **35.818** | **34.105** | **36.334** | **35.720** |
| MC-GS v2 | 34.978 | 35.429 | 33.492 | 35.520 | 35.056 |

Against the converged arms v2 was first on 0 full-brain slices. The loop's target became L1-ESPIRiT's
35.530 / 35.818 on the tuning pair. P-LORAKS (rank 40, R = 2, one 2-D problem per readout slice, 6.7–8 h of CPU
per channel) ranks last or near last on the head gauges on every channel (head PSNR 29.1–32.0), with far more
noise inside the head than the ground truth has. It is in the three original tables and was not run on tangbing.

## 2. v3: a wavelet prior on the render, and the capacity it unlocks (Rounds 11–14)

`loss.lambda_wav` adds an L1 penalty on the 3-D db2 detail coefficients of the rendered complex volumes (3 levels,
cycle spinning; `paired_gs/losses/wavelet_l1.py`) on top of edge-TV. Replacing edge-TV by it is worse, and putting
it on the magnitude costs 0.13 dB, so the two priors are complementary. It also lifts the T1 capacity ceiling:
without it 2.4M points cost T1 0.44 dB, with it T1 rises with the cloud. 3.6M points still overfit T1 at the full
budget (35.662 / 36.233, not adopted).

**v3** = the v2 cold-start fixes + 2.4M initial points + a shared complex wavelet strength 0.05, 4000 iterations.
A shared strength was chosen over the per-contrast variant (T1 0.06 / T2 0.03, a slightly higher minimum margin)
because FLAIR has no tuned value of its own. Head PSNR over L1-ESPIRiT: tuning pair +0.241 / +0.294, hyw T1+FLAIR
+0.242 / +0.254, jiangyizhou +0.003 (a tie; seed 43 gives +0.004) / +0.215. Head HFEN was first on five of six
channels. Head SSIM trailed L1-ESPIRiT by 0.0001–0.005, and T1 sharpness (0.867 / 0.869) missed the pre-registered
floor of 0.874 on both hyw topics.

GS-single v3 (the same recipe, one contrast at a time) is also above L1-ESPIRiT on the tuning pair
(35.679 / 36.227). MC-GS minus GS-single: +0.09 / −0.12 on the tuning pair, +0.093 / +0.018 on hyw T1+FLAIR,
+0.080 / −0.074 on jiangyizhou. The lead comes from the representation, the wavelet prior and the capacity; the
coupling moves accuracy between the contrasts.

## 3. Closing the gaps: v3.1 and v3.2 (Rounds 16–19)

A scorer-identical SSIM breakdown placed v3's head-SSIM gap in the mid-gradient quintiles of the head, as missing
local variance (local std 0.86–0.95 of the ground truth's, L1-ESPIRiT 0.91–0.97), not as noise or edges. FLAIR's
gap sits mostly in the darkest decile (suppressed CSF), as a fluctuation of the local mean that the wavelet strength
does not touch. Single deltas from v3, 4000 iterations, on the tuning pair unless noted:

| change against v3 | exp | head PSNR | min margin over L1-ESPIRiT | T1 sharpness | outcome |
|---|---|---|---|---|---|
| T1-channel wavelet 0.05 → 0.06 | `ar16_n24_wav005_T1p06` | 35.855 / 36.110 | +0.292 | 0.862 | qualifies (v3.1) |
| edge-TV weight 0.10 → 0.07 | `ar16_n24_wav005_geo007` | 35.717 / 36.254 | +0.187 | 0.906 | qualifies; the first configuration to meet every gate |
| FLAIR-channel wavelet 0.12 (hyw T1+FLAIR) | `ar17_tf_hyw_wavC012` | 35.771 / 34.575 | +0.241 | 0.868 | kept over 0.08 (34.549) |
| 3.6M points | `ar12_n36_wav005_full4k` | 35.662 / 36.233 | +0.132 | 0.860 | not adopted |
| T1 0.06 + edge-TV 0.07 | `ar19_n24_T1p06_geo007` | **35.827 / 36.254** | **+0.297** | **0.897** | **adopted as v3.2** |

The prediction written down before geo007 ran (less edge-TV would lower head SSIM) was wrong: T2 head SSIM rose by
0.0021. Less edge-TV adds the local variance the diagnosis found missing.

**v3.2** = v3 + T1-channel wavelet 0.06 + edge-TV 0.07; on T1+FLAIR topics also the FLAIR-channel wavelet 0.12.
On the tuning pair, against every other arm (converged TV, joint TV, L1-ESPIRiT, P-LORAKS, ZS-SSDU, 3D-DIP, 3D-INR,
CG-SENSE, the 35-iteration TV, zero-filled):

| column | T1 | T2 | best other arm, T1 / T2 |
|---|---|---|---|
| head PSNR | **35.827**, 1st | **36.254**, 1st | L1-ESPIRiT 35.530 / 35.818 |
| head SSIM | **0.9483**, 1st | **0.9552**, 1st | L1-ESPIRiT 0.9471 / 0.9536 |
| head HFEN | **0.0799**, 1st | **0.0646**, 1st | L1-ESPIRiT 0.0867 / 0.0656 |
| whole-volume PSNR | **39.999**, 1st | **40.893**, 1st | L1-ESPIRiT 39.823 / 40.442 |
| whole-volume NMSE | **0.01205**, 1st | **0.00966**, 1st | L1-ESPIRiT 0.01254 / 0.01072 |
| whole-volume HFEN | 0.5585, 3rd | **0.4756**, 1st | converged TV 0.5549 / 0.5064 |
| whole-volume SSIM | 0.9007, 6th | 0.9294, 6th | L1-ESPIRiT 0.9510 / ZS-SSDU 0.9677 |
| sharpness (floor) | 0.897 (0.874) | 0.854 (0.802) | |
| first on full-brain slices (transverse / coronal / sagittal) | 47/47, 88/88, 93/93 | 44/44, 83/83, 90/90 | |

Every pre-registered gate is met. The seed-43 repeat agrees to 0.002 dB (35.827 / 36.252). Wall time 56 min on
one GPU, 32.6 GB.

## 4. Transfer, and the two held-out subjects

tangbing (MC_Jia vol4) was added in Round 21 as a second held-out subject: jiangyizhou had by then been used to
explain a result and could no longer settle which recipe transfers. Every arm ran with the settings fixed on the
other topics; nothing was tuned on tangbing.

Head PSNR margin over the strongest other arm on each channel (single runs; on the T1+FLAIR topics v3.1 changes
only the T1 channel):

| topic | channel | strongest other arm | v3 | v3.1 | v3.2 |
|---|---|---|---|---|---|
| hyw T1+T2 (tuning pair) | T1 | L1-ESPIRiT 35.530 | +0.241 | +0.325 | +0.297 |
| | T2 | L1-ESPIRiT 35.818 | +0.294 | +0.292 | +0.435 |
| hyw T1+FLAIR | T1 | L1-ESPIRiT 35.530 | +0.242 | +0.320 | +0.286 |
| | FLAIR | L1-ESPIRiT 34.105 | +0.254 | +0.254 | +0.456 |
| jiangyizhou (held out) | T1 | L1-ESPIRiT 36.334 | +0.003 | +0.124 | **−0.035** |
| | FLAIR | L1-ESPIRiT 35.720 | +0.215 | +0.216 | +0.319 |
| tangbing (held out) | T1 | converged TV 33.580 | +0.245 | +0.349 | +0.211 |
| | FLAIR | L1-ESPIRiT 32.351 | +0.052 | +0.052 | +0.240 |

v3.2 on the other columns (rank among the other arms, first contrast / second contrast):

| topic | head SSIM | head HFEN | whole-volume PSNR | first on full-brain T1 slices (tra / cor / sag) | second contrast, per slice |
|---|---|---|---|---|---|
| hyw T1+T2 | 1 / 1 | 1 / 1 | 1 / 1 | 47/47, 88/88, 93/93 | first on every slice |
| hyw T1+FLAIR | 1 / 3 | 1 / 1 | 1 / 1 | 47/47, 86/88, 93/93 | first on every slice |
| jiangyizhou | 5 / 1 | 1 / 2 | 2 / 1 | 9/96, 69/121, 33/118 | first on every slice |
| tangbing | 4 / 4 | 1 / 1 | 2 / 1 | 94/129, 128/128, 104/129 | first on every slice |

Readings:
- The T1-channel wavelet (v3.1 against v3) helps T1 on all four topics, by 0.08–0.12 dB, and leaves the other
  channel unchanged.
- The edge-TV step (v3.2 against v3.1) costs T1 0.14–0.16 dB on both held-out subjects, against 0.03 dB on the two
  hyw topics; together with the FLAIR-channel 0.12 it gains FLAIR 0.10–0.20 dB. On jiangyizhou the T1 cost flips the
  ranking; on tangbing the margin absorbs it. By the reading pre-registered for Round 21, v3.2 is first on head PSNR
  on both tangbing channels, so the jiangyizhou T1 result reads as subject-specific, but the cost itself repeats.
- Head PSNR and head HFEN hold on the held-out subjects for every recipe, except v3.2 on jiangyizhou (T1 head PSNR
  second by 0.035; FLAIR head HFEN second, 0.1025 against L1-ESPIRiT's 0.1002).
- Head SSIM does not hold: of the four held-out channels, only v3.2's jiangyizhou FLAIR leads, by 0.0001. Converged
  TV leads T1 on both held-out subjects, L1-ESPIRiT leads tangbing FLAIR.
- tangbing's numbers sit 2.7–3.4 dB below jiangyizhou's for every arm, mostly because of the PSNR peak (max / P99.9
  1.75 against 2.29); its ground truth is clean. L1-ESPIRiT at the hyw λ is under-regularised on tangbing T1
  (31.865, sharpness 1.10) and keeps its fixed setting, like every arm.

## 5. Two-seed averaging, and the AF16 loop's recipe on hyw

Seed-42 and seed-43 runs agree to 0.003 dB or better on every global score but differ voxel by voxel by 25–43 % of
the error norm (error correlation ρ 0.91–0.97). The voxelwise mean of their magnitudes ("×2", twice the compute)
gains what −10·log10((1 + ρ)/2) predicts. It is reported as its own row; the single run stays the headline arm.

| topic | v3.2 | v3.2 ×2 | gain | ×2 head SSIM |
|---|---|---|---|---|
| hyw T1+T2 | 35.827 / 36.254 | 35.929 / 36.313 | +0.102 / +0.059 | 0.9492 / 0.9557 |
| hyw T1+FLAIR | 35.816 / 34.561 | 35.935 / 34.596 | +0.119 / +0.035 | 0.9494 / 0.9221 |
| jiangyizhou | 36.299 / 36.039 | 36.499 / 36.108 | +0.200 / +0.069 | 0.9445 / 0.9378 |

On jiangyizhou, ×2 turns the T1 deficit into +0.165 dB over L1-ESPIRiT and lifts FLAIR head SSIM above it; T1 head
SSIM stays below converged TV (0.9459) and FLAIR head HFEN below L1-ESPIRiT. tangbing has no seed-43 runs.

Round 20 ran the parallel AF16 loop's recipe (selected on jiangyizhou at AF16: 3.6M points, wavelet T1 0.10 /
contrast 0.07, edge-TV 0.10) unchanged on hyw, the subject it had not seen. Head PSNR is first on all four channels
(35.946 / 36.179 on the tuning pair, 35.952 / 34.557 on hyw T1+FLAIR) and T1 head SSIM first on both topics, but
head HFEN is worse than L1-ESPIRiT's on three of the four channels and T1 sharpness is 0.839–0.840. On jiangyizhou
AF12, its own subject, it reads 36.753 / 36.165. Neither recipe is first on every metric on the subject it was not
selected on.

## 6. Where MC-GS does not lead

- Whole-volume SSIM: fifth or sixth on every channel of every topic (the air-noise gauge).
- Head SSIM on the held-out subjects (section 4).
- jiangyizhou T1 head PSNR with single-run v3.2 (−0.035); v3.1 (+0.124) and v3.2 ×2 (+0.165) lead there.
- Whole-volume HFEN on T1: third on the tuning pair, second on hyw T1+FLAIR, fourth on both held-out subjects.
- AF16: this loop ran only v2 there, and it was second on every channel by 0.2–0.7 dB (ZS-SSDU or CS-SENSE
  leading); v3 and v3.2 were not run at AF16.

## 7. Open decisions (for the user)

1. The headline recipe. v3.2 was selected by the pre-registered rule, meets every gate and is first on every head
   column of the tuning pair; across four topics it is first on head PSNR on seven of eight channels (jiangyizhou T1
   −0.035). v3.1 is first on head PSNR on all eight channels in single runs, but misses the T1 sharpness floor on
   both hyw topics and on tangbing, and trails L1-ESPIRiT on T2 head SSIM and HFEN of the tuning pair. ×2 of either
   costs twice the compute.
2. The T1 sharpness gate: keep the one-sided floor 0.874 (taken from the old cold-start MC-GS arm), or replace it by
   a two-sided band around the ground truth's 1.0 (the metric is a ratio to the ground truth).
3. Whether to fast-forward `master` to this branch as this lineage's main (`main` is an unrelated lineage and must
   not be merged).

## 8. Reproduce

```
cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework
SAVE_VOLUMES=1 ./autoresearch_t1t2.sh <EXP> <GPU> 4000 \
  --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn --set loss.edge_tv_kappa_source=zf \
  --set gaussian.initial_num_points=2400000 --set loss.lambda_wav=0.05 --set loss.lambda_wav_plain=0.06 --lambda_geo 0.07
# T1+FLAIR topics: add --set loss.lambda_wav_contrast=0.12 and CONFIG=configs/t1flair_af12_hyw.yaml (hyw),
# configs/t1flair_v3_af12_hyw.yaml (jiangyizhou) or configs/t1flair_v4_af12_hyw.yaml (tangbing)
```

×2: a second run with `--set training.seed=43`, averaged and scored by `t1t2_af12_retro/scripts/score_average.py`,
which re-scores seed 42 first and must match its logged numbers. Tests: 183 passed (2 min 20 s on an idle GPU,
2026-09-11).

## 9. Archives and figures

Topic archives, volumes included, are in `/data0/congcong/code/GS/Results_20260908/`. The repo mirrors each topic
without volumes: README with the tables, the per-slice sweeps and the MC-GS sections, `logs/summary.tsv`,
`logs/best_slice_v2.log`, figures. In the figures the proposed arm is v3.2, marked "(ours)".

| topic | mirror | three planes | best transverse slice |
|---|---|---|---|
| hyw T1+T2 | [`t1t2_af12_retro/`](../../../t1t2_af12_retro/README.md) | [T1](../../../t1t2_af12_retro/figures/t1t2_af12_T1_best3planes.png) · [T2](../../../t1t2_af12_retro/figures/t1t2_af12_T2_best3planes.png) | [T1 150](../../../t1t2_af12_retro/figures/t1t2_af12_T1_best_axis0_150.png) · [T2 144](../../../t1t2_af12_retro/figures/t1t2_af12_T2_best_axis0_144.png) |
| hyw T1+FLAIR | [`t1flair_af12_retro/`](../../../t1flair_af12_retro/README.md) | [T1](../../../t1flair_af12_retro/figures/t1flair_af12_T1_best3planes.png) · [FLAIR](../../../t1flair_af12_retro/figures/t1flair_af12_FLAIR_best3planes.png) | [T1 150](../../../t1flair_af12_retro/figures/t1flair_af12_T1_best_axis0_150.png) · [FLAIR 150](../../../t1flair_af12_retro/figures/t1flair_af12_FLAIR_best_axis0_150.png) |
| jiangyizhou | [`t1flair_af12_v3_jiangyizhou/`](../../../t1flair_af12_v3_jiangyizhou/README.md) | [T1](../../../t1flair_af12_v3_jiangyizhou/figures/t1flair_v3_af12_T1_best3planes.png) · [FLAIR](../../../t1flair_af12_v3_jiangyizhou/figures/t1flair_v3_af12_FLAIR_best3planes.png) | [T1 173](../../../t1flair_af12_v3_jiangyizhou/figures/t1flair_v3_af12_T1_best_axis0_173.png) · [FLAIR 177](../../../t1flair_af12_v3_jiangyizhou/figures/t1flair_v3_af12_FLAIR_best_axis0_177.png) |
| tangbing | [`t1flair_af12_v4_tangbing/`](../../../t1flair_af12_v4_tangbing/README.md) | [T1](../../../t1flair_af12_v4_tangbing/figures/t1flair_v4_af12_T1_best3planes.png) · [FLAIR](../../../t1flair_af12_v4_tangbing/figures/t1flair_v4_af12_FLAIR_best3planes.png) | [T1 170](../../../t1flair_af12_v4_tangbing/figures/t1flair_v4_af12_T1_best_axis0_170.png) · [FLAIR 71](../../../t1flair_af12_v4_tangbing/figures/t1flair_v4_af12_FLAIR_best_axis0_71.png) |

Helper scripts, in `t1t2_af12_retro/scripts/`: `refresh_topic.sh` (tables, per-slice sweep and figures of a topic),
`mirror_topic.sh` (copy into the repo), `score_average.py` (×2), `ssim_breakdown.py`, `ssim_topic.py`,
`bias_by_intensity.py` (the SSIM diagnosis), `find_best_slice_v2.py`, `fig_slice_v2.py`, `fig_planes_v2.py`.

## Gotchas recorded along the way

- GPUs 6 and 7 carry another user's resident 696 MiB process and train 2–3 times slower. A queue's "GPU is idle"
  check has to ignore processes under 4 GB, or it never fires there. One training process per GPU.
- The session's memory watchdog also kills background wait commands: launch with `setsid nohup`, wait on a PID found
  once at launch (a pgrep pattern matches leftover tool shells), and read logs with synchronous commands.
- zsh: `$VAR:r…` is a modifier (write `${VAR}:…`), and an unquoted `--include=*.py` is a glob.
- Changing a ledger status with awk + mv replaces the file, and a `tail -F` monitor then re-emits every old row.
- A topic cloned with `setup_subject.sh` inherits `summarize.py` / `extract_gs_volumes.py` that read the source
  topic's runs; point them at the new subject's runs before refreshing.
