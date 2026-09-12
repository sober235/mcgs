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
| MC-GS joint v3 ×2 seeds (2× compute) | **39.847** | 0.9219 | **36.539** | 0.9449 | 118 min |
| MC-GS joint v3.2 ×2 seeds (2× compute) | 39.838 | 0.9213 | 36.499 | 0.9445 | 116 min |
| MC-GS joint v3 | 39.662 | 0.9206 | 36.337 | 0.9428 | 63 min |
| L1-ESPIRiT | 39.497 | 0.9420 | 36.334 | 0.9443 | 5 min |
| **MC-GS joint v3.2** | 39.654 | 0.9200 | 36.299 | 0.9423 | 57 min |
| MC-GS single v3 | 39.566 | 0.9251 | 36.257 | 0.9412 | 70 min |
| CS-SENSE (TV, converged) | 39.753 | 0.9522 | 36.144 | **0.9459** | 5 min |
| joint CS-SENSE (JTV) | 39.648 | **0.9587** | 36.070 | 0.9442 | 5 min |
| CS-SENSE (TV, 35 outer, as run 09-09) | 39.395 | 0.9523 | 35.762 | 0.9429 | 2 min |
| ZS-SSDU | 39.357 | 0.9577 | 35.724 | 0.9422 | 22 min |
| MC-GS joint v2 | 38.949 | 0.9194 | 35.520 | 0.9321 | 42 min |
| MC-GS single v2 | 38.601 | 0.9205 | 35.209 | 0.9260 | 42 min |
| MC-GS joint (old cold start) | 36.349 | 0.8703 | 34.160 | 0.9169 |  |
| MC-GS single (old cold start) | 35.529 | 0.8297 | 33.799 | 0.9032 |  |
| CG-SENSE | 35.082 | 0.7562 | 33.514 | 0.8981 | 2 min |
| P-LORAKS | 35.874 | 0.8889 | 31.992 | 0.8528 |  |
| 3D-INR | 34.668 | 0.8218 | 31.638 | 0.8373 | 57 min |
| 3D-DIP | 35.267 | 0.8475 | 31.363 | 0.8634 | 50 min |
| ZF | 30.240 | 0.8353 | 26.130 | 0.7009 | 2 min |

**FLAIR**

| method | PSNR | SSIM | head PSNR | head SSIM | wall |
|---|---|---|---|---|---|
| MC-GS joint v3.2 ×2 seeds (2× compute) | **39.536** | 0.9032 | **36.108** | **0.9378** | 116 min |
| MC-GS joint v3 ×2 seeds (2× compute) | 39.441 | 0.9107 | 36.049 | 0.9360 | 118 min |
| **MC-GS joint v3.2** | 39.473 | 0.9028 | 36.039 | 0.9369 | 57 min |
| MC-GS single v3 | 39.325 | 0.9011 | 36.009 | 0.9346 | 71 min |
| MC-GS joint v3 | 39.334 | 0.9098 | 35.935 | 0.9346 | 63 min |
| L1-ESPIRiT | 39.327 | 0.9525 | 35.720 | 0.9368 | 5 min |
| joint CS-SENSE (JTV) | 39.277 | 0.9523 | 35.624 | 0.9352 | 7 min |
| CS-SENSE (TV, converged) | 39.142 | 0.9497 | 35.581 | 0.9353 | 5 min |
| MC-GS single v2 | 38.503 | 0.8966 | 35.101 | 0.9216 | 39 min |
| CS-SENSE (TV, 35 outer, as run 09-09) | 38.773 | 0.9416 | 35.084 | 0.9307 | 2 min |
| MC-GS joint v2 | 38.584 | 0.9082 | 35.056 | 0.9238 | 42 min |
| ZS-SSDU | 38.807 | **0.9563** | 35.034 | 0.9353 | 36 min |
| CG-SENSE | 35.704 | 0.7782 | 33.957 | 0.9032 | 2 min |
| MC-GS single (old cold start) | 36.204 | 0.8749 | 33.409 | 0.9006 |  |
| MC-GS joint (old cold start) | 36.646 | 0.9079 | 33.391 | 0.9064 |  |
| 3D-INR | 34.637 | 0.8274 | 31.190 | 0.8277 | 57 min |
| P-LORAKS | 35.071 | 0.8731 | 31.176 | 0.8165 |  |
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

## MC-GS joint v2 — the T1+T2 recipe transferred unchanged: second, not first (2026-09-10)

The recipe the autoresearch loop found on huangyuwei's T1+T2 pair (framework repo
`autoresearch_t1t2.md`; "MC-GS joint v2" in the T1+T2 README: points drawn inside the ZF head
support, 3-NN spacing scales, edge-TV kappa from the ZF image, 1.2M points) was applied here
**without changing a knob**, as the held-out-subject check (`ar5_tf_jyz_t1flair`, 4000 iters).

| channel | v2 head PSNR | best other arm | over the original joint arm | v2 head SSIM / best | v2 head HFEN / best |
|---|---|---|---|---|---|
| T1 | 35.520 | CS-SENSE 35.762 | +1.36 dB | 0.9321 / 0.9429 | **0.1056** / 0.1208 |
| FLAIR | 35.056 | CS-SENSE 35.084 | +1.67 dB | 0.9238 / 0.9353 | **0.1063** / 0.1139 |

On this subject the unchanged recipe is **second on head PSNR** (−0.24 dB on T1, −0.03 on
FLAIR against CS-SENSE), first on head HFEN, behind on head SSIM. On huangyuwei it is first on
both pairings. One measurable difference between the subjects is the size of the head in the
FOV: the ZF support covers 42.3 % of the volume here against 32.0 % / 32.7 % on huangyuwei, so
a fixed 1.2M points is a lower Gaussian density (0.246 vs 0.325 per support voxel), and
capacity was the lever the T1 channel was most sensitive to. A points-per-support-voxel rule
fixed on huangyuwei alone (0.325 → 1.59M points here) was pre-registered in the framework
repo's `autoresearch_t1t2.md` before being run. **It failed** (`ar7_tf_jyz_rho`, 1.588M points,
4000 iterations): T1 35.273 / FLAIR 35.071 head PSNR, i.e. T1 0.25 dB *worse* than at 1.2M and
FLAIR flat. The T1 capacity ceiling is not a density per head voxel; the acquired data is the
same size on both subjects (same mask, 3847 lines × 256 × 26 coils), so the number of
Gaussians it can pin down is roughly the same, and an absolute count near 1.2M is the
transferable setting. The honest claim for the recipe is therefore: first on the tuning
subject in both pairings, second on this held-out subject, +1.4 / +1.7 dB over the original
joint arm everywhere.

Single-contrast arms of the same recipe (`ar9_single_{T1,FLAIR}_v2_jyz`): T1 35.209 / FLAIR 35.101
head PSNR. On T1 the joint model is +0.31 dB above its ablation; on FLAIR the single arm is
+0.045 dB above the joint one and edges CS-SENSE by 0.017 dB — the same order as the run-to-run
floor, so it is listed, not claimed. The compact figures show only the v2 pair.

## Classical arms revisited (2026-09-10): convergence, lambda sweeps, two new baselines

Three classical baselines were added or corrected on this topic's data and gauge, after the
"MC-GS joint v2" rows above had been established. Two findings change the reading of the tables.

**1. The CS-SENSE (TV) arm of 09-09 was not converged.** Its 35 outer ADMM iterations were
inherited from the CE study, where no ground truth could check convergence. At 100 and 200 outer
iterations (same lambda) the head PSNR rises by 0.35 to 0.61 dB and then stops moving
(hyw T1 34.607 → 34.955 → 34.966; T2 34.219 → 35.096 → 35.168 at its best lambda; FLAIR
33.252 → 33.873 at 200). The original row is kept, labelled, for traceability; "CS-SENSE (TV,
converged)" is the arm to compare against. Lambda was swept on huangyuwei with the ground truth
in the scorer — the same treatment the v2 recipe received — over {0.5, 0.7, 1.0, 1.4, 2.0}e-3:
the 09-09 value 1e-3 was already optimal on T1, and 0.7e-3 gains +0.07 (T2) / +0.01 (FLAIR).
Those lambdas were transferred unchanged to jiangyizhou.

**2. L1-wavelet ESPIRiT (BART `pics -l1`, 100 FISTA iterations, the BART default) is the
strongest arm on every channel.** Lambda swept on huangyuwei over 0.0005 to 0.02 (per-contrast
optima 0.0015 for T1, 0.001 for T2, 0.005 for FLAIR; both neighbours lower), transferred
unchanged to jiangyizhou. Head PSNR: hyw T1 35.530 / T2 35.818 / FLAIR 34.105, jyz T1 36.334 /
FLAIR 35.720 — above MC-GS joint v2 by 0.4 to 0.8 dB, and above the converged TV arm by
0.15 to 0.65 dB. One caveat belongs next to it: at 300 instead of 100 FISTA iterations the
hyw numbers drop (T1 35.526 → 35.218, T2 35.498 → 35.119 at lambda 0.002), so the arm as run
includes an early-stopping regularisation on top of the wavelet prior; "100 iterations" is part
of its definition, as "4000 iterations" is part of MC-GS's.

**3. Joint CS-SENSE (vectorial TV over both contrasts, 200 outer iterations, lambda 1e-3
selected on hyw over {0.5, 0.7, 1.0, 1.4, 2.0}e-3)** — the classical counterpart of "shared
geometry" — is a wash against the converged single-contrast TV: within ±0.16 dB on every
channel of every topic (hyw T1+T2 35.020 / 35.008; hyw T1+FLAIR 34.851 / 33.929; jyz 36.070 /
35.624). Consistent with the project's earlier finding that a vectorial TV at the same lambda is
just a weaker per-contrast penalty.

**Where MC-GS joint v2 stands after this.** Second or third on head PSNR everywhere: behind
L1-ESPIRiT on all five channels, tied with the converged TV on hyw T1 (34.978 vs 34.966), ahead
of it on hyw T2 (+0.26), behind on the FLAIR channels and on jiangyizhou. It keeps the lowest
head HFEN on hyw T1 (0.0863) and stays +1.4 to +2.3 dB above the cold-start joint arm it
replaced. The claim that survives is the diagnosis and repair of the cold-start path, not
"first among nine".
A per-slice sweep against the converged and new arms (`scripts/find_best_slice_v2.py`, output in
`logs/best_slice_v2.log`) finds v2 first on **none** of the full-brain slices (>= 20000 head
voxels) in any plane of any topic; L1-ESPIRiT leads on the slices where v2 itself peaks as well.
The compact figures (`scripts/fig_slice_v2.py`, `scripts/fig_planes_v2.py`) now carry the
converged TV, joint TV and L1-ESPIRiT columns and show those ranks as they are.

P-LORAKS (LORAKS v2 toolbox, per-readout-slice, S-matrix, R = 2) is the fourth arm of this
revisit; its rank (40) was selected on a six-slice subset of hyw T1 (`../T1T2_AF12_retro/logs/ploraks/ranksweep_R2.log`)
and is applied unchanged here. The T1 run (rank 40, R = 2, 48 MATLAB workers) took 5.7 h and the FLAIR run
3.6 h; both rows are in the tables.
Scripts: `recon_classical.py --suffix` (sweeps), `recon_joint_cssense.py`, `recon_l1espirit.py`,
`ploraks_export.py` / `run_ploraks.m` / `run_ploraks_ranks.m` / `ploraks_import.py`; sweep logs
in `logs/sweeps/`, per-run json in `logs/`.

## MC-GS joint v3 — the T1+T2 recipe with wavelet sparsity, run unchanged on the held-out subject (2026-09-10/11)

v3 is defined and selected on the huangyuwei T1+T2 pair (`../T1T2_AF12_retro/README.md`, section "MC-GS joint
v3"): v2's cold-start fixes + 2.4M initial points + a complex wavelet-sparsity term λ 0.05 on top of edge-TV,
4000 iterations, one strength for both contrasts. Nothing was tuned on this subject; it stays out of every
selection in the campaign (`autoresearch_t1t2.md` in the framework repo).

```bash
cd $F && CONFIG=configs/t1flair_v3_af12_hyw.yaml SAVE_VOLUMES=1 ./autoresearch_t1t2.sh ar13_tf_jyz_t1flair_v3b 5 4000 \
  --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn --set loss.edge_tv_kappa_source=zf \
  --set gaussian.initial_num_points=2400000 --set loss.lambda_wav=0.05
```

| arm (4000 iters) | head PSNR T1 / FLAIR | head SSIM | head HFEN | whole PSNR | sharp |
|---|---|---|---|---|---|
| **MC-GS joint v3** | 36.337 / 35.935 | 0.9428 / 0.9346 | 0.0987 / 0.0994 | 39.662 / 39.334 | 0.892 / 0.847 |
| MC-GS joint v3, seed 43 | 36.338 / 35.937 | 0.9428 / 0.9347 | 0.0987 / 0.0994 | 39.664 / 39.335 | 0.892 / 0.847 |
| MC-GS joint v3 ×2 seeds (mean of the two runs, 2× compute) | 36.539 / 36.049 | 0.9449 / 0.9360 | 0.0974 / 0.0985 | 39.847 / 39.441 | 0.883 / 0.842 |
| MC-GS single v3 (same recipe, one contrast) | 36.257 / 36.009 | 0.9412 / 0.9346 | 0.1019 / 0.0994 | 39.566 / 39.325 | 0.891 / 0.857 |
| L1-ESPIRiT | 36.334 / 35.720 | 0.9443 / 0.9368 | 0.1034 / 0.1002 | 39.497 / 39.327 | 0.906 / 0.809 |
| CS-SENSE (TV, converged) | 36.144 / 35.581 | 0.9459 / 0.9353 | 0.1107 / 0.1073 | 39.753 / 39.142 | 0.838 / 0.783 |
| MC-GS joint v2 | 35.520 / 35.056 | 0.9321 / 0.9238 | 0.1056 / 0.1063 | 38.949 / 38.584 | 0.927 / 0.875 |

- **Held out, v3 is first on head PSNR on FLAIR (+0.216 dB over L1-ESPIRiT) and ties on T1 (+0.003)**; the
  seed-43 repeat gives +0.004 / +0.217. Head HFEN is first on both contrasts. Whole-volume PSNR and NMSE are
  first on FLAIR and second on T1 (converged TV 39.753). Head SSIM trails: fourth on T1 and fifth on FLAIR
  (best other arm: converged TV 0.9459 on T1, L1-ESPIRiT 0.9368 on FLAIR).
- **Two seeds averaged** (v3 ×2, program Round 18): +0.205 / +0.329 dB over L1-ESPIRiT and first on
  whole-volume PSNR on both contrasts, so the tie on T1 becomes a clear lead. Head SSIM rises to 0.9449 / 0.9360,
  still below the best other arm on each channel. It is reported as its own row at twice the compute;
  single-run v3 stays the headline arm.
- **Per slice** (`scripts/find_best_slice_v2.py`, full-brain slices, against every other arm; P-LORAKS not yet
  in): FLAIR first on 94/94 transverse, 87/120 coronal and 118/118 sagittal slices; T1, where the volume-level
  margin is a tie, first on 18/96, 77/121 and 56/118.
- **Joint versus single**: the joint model is +0.080 dB on T1 and −0.074 on FLAIR. The single-contrast FLAIR is
  by itself +0.289 above L1-ESPIRiT, the single-contrast T1 −0.077 below it. The coupling moves accuracy from
  FLAIR to T1, and that is what brings T1 level with L1-ESPIRiT on this subject.
- **Gate**: T1 sharpness 0.892 passes the pre-registered 0.874 floor on this subject.
- **External arm, not held out**: the parallel AF16 loop's recipe, selected on this subject at AF16 (3.6M points,
  wavelet T1 0.10 / FLAIR 0.07), run unchanged on this topic: head PSNR 36.753 / 36.165, head SSIM 0.9480 / 0.9374,
  head HFEN 0.1009 / 0.1002, whole-volume PSNR 40.071 / 39.539 (`mcgs_af16_wt/results_arc/ar17_P6_tf_jyz_af12_full4k_score.json`).
  It is first on head PSNR, head SSIM and whole-volume PSNR on both channels. Because it was selected on this
  subject, it is cross-checked on huangyuwei (framework program, Round 20).
- P-LORAKS for this subject has finished; both rows are in the tables (see "Classical arms revisited").
- Figures (compact panels, v3 as the proposed method): `figures/t1flair_v3_af12_T1_best_axis0_173.png`,
  `figures/t1flair_v3_af12_FLAIR_best_axis0_178.png` (v3's best transverse slices),
  `figures/t1flair_v3_af12_{T1,FLAIR}_best3planes.png`.

## MC-GS joint v3.2 — T1-channel wavelet 0.06 and lighter edge-TV 0.07 (2026-09-11)

v3.2 = v3 + a T1-channel wavelet strength of 0.06 (the second channel keeps 0.05) + edge-TV 0.07 instead of 0.10; on the
T1+FLAIR topics the FLAIR channel takes 0.12. Selected on the tuning pair by the pre-registered rules of the framework
program (Rounds 16–17, "## v3.2 — adopted"). Held out: nothing in v3.2 was selected on this subject.

```bash
cd $F && CONFIG=configs/t1flair_v3_af12_hyw.yaml SAVE_VOLUMES=1 ./autoresearch_t1t2.sh ar19_tf_jyz_full_C012 1 4000 \
  --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn --set loss.edge_tv_kappa_source=zf \
  --set gaussian.initial_num_points=2400000 --set loss.lambda_wav=0.05 --set loss.lambda_wav_plain=0.06 --set loss.lambda_wav_contrast=0.12 --lambda_geo 0.07
```

Standing of v3.2 against every other method in this topic's table (single run, seed 42):

| metric | T1 | FLAIR |
|---|---|---|
| head PSNR | second, 36.299 (best: L1-ESPIRiT 36.334) | **first**, 36.039 (next: L1-ESPIRiT 35.720) |
| head SSIM | fifth, 0.9423 (best: converged TV 0.9459) | **first**, 0.9369 (next: L1-ESPIRiT 0.9368) |
| head HFEN | **first**, 0.0965 (next: L1-ESPIRiT 0.1034) | second, 0.1025 (best: L1-ESPIRiT 0.1002) |
| whole-volume PSNR | second, 39.654 (best: converged TV 39.753) | **first**, 39.473 (next: L1-ESPIRiT 39.327) |
| whole-volume NMSE | second, 0.01411 (best: converged TV 0.01379) | **first**, 0.01325 (next: L1-ESPIRiT 0.01370) |
| whole-volume HFEN | fourth, 0.6449 (best: converged TV 0.6154) | **first**, 0.6013 (next: joint CS-SENSE 0.6188) |
| whole-volume SSIM | sixth, 0.9200 (best: joint CS-SENSE 0.9587) | sixth, 0.9028 (best: ZS-SSDU 0.9563) |

| arm | head PSNR T1 / FLAIR | head SSIM | head HFEN | whole PSNR | sharp |
|---|---|---|---|---|---|
| **MC-GS joint v3.2** | 36.299 / 36.039 | 0.9423 / 0.9369 | 0.0965 / 0.1025 | 39.654 / 39.473 | 0.921 / 0.822 |
| MC-GS joint v3 | 36.337 / 35.935 | 0.9428 / 0.9346 | 0.0987 / 0.0994 | 39.662 / 39.334 | 0.892 / 0.847 |
| L1-ESPIRiT | 36.334 / 35.720 | 0.9443 / 0.9368 | 0.1034 / 0.1002 | 39.497 / 39.327 | 0.906 / 0.809 |
| converged TV (best other head SSIM) | 36.144 / 35.581 | 0.9459 / 0.9353 | 0.1107 / 0.1073 | 39.753 / 39.142 | 0.838 / 0.783 |

- **T1**: first on head HFEN; not first on head PSNR (second), head SSIM (fifth), whole-volume PSNR (second), whole-volume NMSE (second), whole-volume HFEN (fourth), whole-volume SSIM (sixth).
- **FLAIR**: first on head PSNR, head SSIM, whole-volume PSNR, whole-volume NMSE, whole-volume HFEN; not first on head HFEN (second), whole-volume SSIM (sixth).
- **Against v3**: T1 -0.038 dB / head SSIM -0.0005, FLAIR +0.104 dB / head SSIM +0.0023.
- **Gate**: T1 sharpness 0.921 passes the pre-registered 0.874 floor.
- **Two seeds averaged** (v3.2 ×2, Round 18c; its own table row, at twice the compute): head PSNR T1 36.499 (first), FLAIR 36.108 (first); head SSIM T1 0.9445 (second; converged TV 0.9459), FLAIR 0.9378 (first); head HFEN T1 0.0952 (first), FLAIR 0.1015 (second; L1-ESPIRiT 0.1002). Single-run v3.2 stays the headline arm.
- **Per slice** (`scripts/find_best_slice_v2.py`, full-brain slices, against every other arm; P-LORAKS included): T1 first on 9/96, 69/121, 33/118 transverse / coronal / sagittal slices; FLAIR first on 94/94, 120/120, 118/118 transverse / coronal / sagittal slices.
- Figures (compact panels, v3.2 as the proposed method): `figures/t1flair_v3_af12_T1_best_axis0_173.png`, `figures/t1flair_v3_af12_FLAIR_best_axis0_177.png`, `figures/t1flair_v3_af12_{T1,FLAIR}_best3planes.png`.

## Regenerating

```bash
R=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou
bash $R/scripts/prep_03_csm.sh          # both maps, ~85 min CPU
env PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python $R/scripts/prep_04_export.py
bash $R/scripts/run_all.sh              # waits for the maps, then all sixteen arms on GPUs 0-3,6,7
env PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python $R/scripts/summarize.py
```

## Classical λ re-swept on this subject (2026-09-12, for the AF12 autoresearch loop)

Until now the three strongest classical arms ran at the λ selected on huangyuwei. For the jiangyizhou AF12
autoresearch loop (framework repo, `autoresearch_jyz_af12.md`) each λ was swept **on this subject against the ground
truth** — the same privilege the MC-GS loop has — with `scripts/run_classical_resweep_af12.sh` and
`scripts/run_l1_refine_af12.sh`; `summarize.py` now keeps the best λ per family and contrast (head PSNR), so the
tables above are superseded for these three rows:

| arm | T1 λ: head PSNR | FLAIR λ: head PSNR | change vs the hyw λ |
|---|---|---|---|
| L1-ESPIRiT (100 FISTA) | 0.0007 35.226, 0.001 35.917, 0.0015 36.334, **0.002 36.426**, 0.0025 36.374, 0.003 36.255, 0.004 35.934 | 0.003 35.689, **0.004 35.757**, 0.005 35.720, 0.006 35.630, 0.007 35.513, 0.01 35.095 | +0.092 / +0.037 |
| CS-SENSE (TV, 200 outer) | 0.0005 35.970, **0.0007 36.162**, 0.001 36.144, 0.0014 35.931, 0.002 35.501 | 0.0004 35.462, 0.0005 35.575, **0.0007 35.581**, 0.001 35.400, 0.0014 35.075 | +0.018 / 0 |
| joint CS-SENSE (JTV) | 0.0005 35.584, 0.0007 35.936, **0.001 36.070**, 0.0014 35.986, 0.002 35.691 | 0.0005 35.383, 0.0007 35.606, **0.001 35.624**, 0.0014 35.464, 0.002 35.132 | 0 / 0 |

L1-ESPIRiT stays the strongest classical arm on every head cell; its best SSIM (0.9473, λ 0.0025) and best HFEN
(0.1034, λ 0.0015) on T1 sit at other λ than its best PSNR, and the loop's bar takes the best of every swept λ per
cell. Volumes of every λ are in `recon/` (`*_lam<λ>_*_complex.npy`); the sweep logs in `logs/sweeps/`.

## MC-GS after the AF12 autoresearch loop — P10 (2026-09-12)

The jiangyizhou AF12 autoresearch loop (framework repo, branch `autoresearch-jyz-af12`, `autoresearch_jyz_af12.md`) started
from the AF16 loop's recipe P6 (the user's provisional best 12x result) and changed only the L1 wavelet prior: **db4
family (was db2), the finest decomposition level only (was 3 levels), T1 strength 0.12 / FLAIR 0.10** (edge-TV 0.10 / κ_q 0.85,
3.6M points, repaired cold start, 4000 iterations unchanged). Registered here as `gs_joint_ar` ("MC-GS (AF12 loop, P10)"),
with `gs_single_ar` (the same recipe, one contrast at a time) and `gs_joint_ar_x2` (magnitude mean of seeds 42 and 43, 2x compute).
The classical arms are at their λ re-swept on this subject (section above).

| arm | T1 head PSNR / SSIM / HFEN | FLAIR head PSNR / SSIM / HFEN | whole PSNR T1 / FLAIR |
|---|---|---|---|
| **MC-GS (P10)** | **36.944 / 0.9503 / 0.0947** | **36.268 / 0.9397 / 0.0958** | **40.272 / 39.692** |
| MC-GS (P10) ×2 seeds (2x compute) | 36.990 / 0.9507 / 0.0943 | 36.295 / 0.9401 / 0.0954 | 40.319 / 39.723 |
| GS-single (P10) | 36.897 / 0.9495 / 0.0965 | 36.323 / 0.9405 / 0.0944 | 40.241 / 39.693 |
| MC-GS joint v3.2 (hyw-selected, previous proposed arm) | 36.299 / 0.9423 / 0.0965 | 36.039 / 0.9369 / 0.1025 | 39.654 / 39.473 |
| P6 (AF16 loop's recipe, unchanged) | 36.753 / 0.9480 / 0.1009 | 36.165 / 0.9374 / 0.1002 | 40.071 / 39.539 |
| L1-ESPIRiT (best λ per cell) | 36.426 / 0.9473 / 0.1034 | 35.757 / 0.9368 / 0.1000 | 39.893 / 39.205 |

- **Head-only PSNR, SSIM and HFEN: first on both contrasts** (margins over the best other arm +0.518 / +0.0030 / +0.0087 on
  T1, +0.511 / +0.0030 / +0.0042 on FLAIR); whole-volume PSNR and NMSE first; whole-volume SSIM sixth (air-noise gauge).
- **Per slice** (`scripts/find_best_slice_v2.py --ours gs_joint_ar`, full-brain slices, every canonical arm at its best λ):
  first on every slice in every direction — T1 96/96, 121/121, 118/118; FLAIR 94/94, 120/120, 118/118.
- Joint − single: **+0.047 on T1, −0.055 on FLAIR** — the coupling moves quality towards T1 on this subject, as with v3.
- Seed floor at 4000 iterations 0.002 / 0.001 dB (seed 43: 36.942 / 36.267). 6000 iterations: 36.953 / 36.284.
- **Complex gauge** (`scripts/complex_gauge.py`, `logs/complex_gauge.json`): on |rec − GT| L1-ESPIRiT is first (33.449 /
  32.961) and P10 second (33.418 / 32.887); the lead is a magnitude-gauge lead, as the AF16 line found.
- Selected on this subject (recipe and classical λ alike): fair, not held out.
- Figures (compact three-plane panels, orange in-panel slice PSNR / SSIM / NMSE, head-only error maps):
  `figures/t1flair_v3_af12_{T1,FLAIR}_row3planes_compact_174_110_47_headerr.png`, served from
  `http://localhost:8765/t1flair_v3_af12/`. Script `scripts/fig_three_planes_compact.py`.

### Best single slices for MC-GS (P10), transverse (2026-09-12)
Per-slice head-masked PSNR of the P10 volumes against every arm (`logs/best_slice_v2.json`; GS-single re-checked from the
volumes): **T1 slice 173** is MC-GS's best slice and it leads every arm there including its own single (38.266; GS-single
38.216, L1-ESPIRiT 37.845). **FLAIR slice 178** is MC-GS's best FLAIR slice (36.559, +0.493 over L1-ESPIRiT 36.066), but
GS-single is 0.04 dB higher there — and on every transverse FLAIR slice (joint − single < 0 on 92/92 full-brain slices).
One common slice for both contrasts: **177** (T1 38.240 / FLAIR 36.550, each within 0.03 dB of the contrast's own best;
max-sum), or 178 (max-min, T1 38.212 / FLAIR 36.559). Row figures (`scripts/fig_row_final.py`, GT + nine methods, error
maps below, head mask bottom-left): `figures/t1flair_v3_af12_{T1,FLAIR}_row_axis0_{173|178|177}.png`.

### Side experiment: FLAIR mask at AF 11, T1 unchanged, MC-GS only (2026-09-12)
User request. T1 keeps `data/mask_af12_3d.h5`; FLAIR gets a fresh house vd-Poisson mask at nominal AF 11
(`scripts/prep_02b_mask_af11.m`, `prep_04b_export_af11.py` → `data/mask_af11_3d.h5`; 4173 samples, 9.26 % of the PE plane,
**AF 9.05x on FLAIR's own acquired support** against 9.74x for the AF12 mask; only 926 samples coincide with the AF12 mask).
Only MC-GS (P10 recipe, `ar9_P10_flair_af11_full4k`) was re-run; every other arm stays at AF 12. Registered as
`gs_joint_ar_fl11` ("MC-GS (P10, FLAIR mask AF 11)").

| arm | T1 head PSNR / SSIM / HFEN | FLAIR head PSNR / SSIM / HFEN | whole PSNR T1 / FLAIR |
|---|---|---|---|
| MC-GS (P10, FLAIR at AF 11) | 36.945 / 0.9503 / 0.0947 | **36.637 / 0.9439 / 0.0906** | 40.274 / 40.021 |
| MC-GS (P10, both at AF 12) | 36.944 / 0.9503 / 0.0947 | 36.268 / 0.9397 / 0.0958 | 40.272 / 39.692 |
| L1-ESPIRiT (AF 12) | 36.426 / 0.9469 / 0.1036 | 35.757 / 0.9368 / 0.1000 | 39.893 / 39.205 |

The T1 channel is unchanged to 0.001 dB (its mask did not change; the coupling passes nothing measurable back), the FLAIR
channel gains **+0.369 dB** head PSNR, +0.0042 SSIM, 0.0052 HFEN and +0.007 sharpness from the 7 % denser mask. Slice 177
(`figures/t1flair_v3_af12_{T1,FLAIR}_flair_af11_axis0_177.png`, MC-GS column = this run, every other column AF 12): T1 38.24
(GS-single 38.19, L1-ESPIRiT 37.87); FLAIR **36.81** (AF-12 GS-single 36.59, AF-12 MC-GS 36.55, L1-ESPIRiT 36.10) — not a
like-for-like FLAIR comparison, labelled as such in the title.
Three-plane compact panels for this design (MC-GS column = the FLAIR-AF11 run, every other column AF 12, 174 / 110 / 47,
head-only error maps; the FLAIR panel shows the AF 11 mask MC-GS used, `--mask_path data/mask_af11_3d.h5`): `figures/t1flair_v3_af12_{T1,FLAIR}_row3planes_compact_flair_af11_174_110_47_headerr.png` — MC-GS
first on PSNR, SSIM and NMSE in all three planes on both contrasts (`scripts/fig_three_planes_compact.py --mcgs_key gs_joint_ar_fl11`).

### Raw outputs of the AF12 loop (2026-09-12)
`runs_af12_loop/` holds, for the 20 key 4000-iteration runs (P6 reference, promotions P7–P10, seed 43, two-seed mean, 6000
iterations, GS-single arms, design B, marginal variants), the scorer json, the volume npz (magnitude, GT, ZF; complex from
2026-09-11 on) and the log, plus copies of the ledger, program, runner, record and handoff — 14 GB, see its README. The
framework-lineage summary for collaborators is on `main` (`docs/2026-09-12-jyz-af12-mcgs-p10-summary.md`, commit a5ae1e6).
