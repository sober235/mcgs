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

| method | PSNR | SSIM | head PSNR | head SSIM | wall |
|---|---|---|---|---|---|
| **MC-GS joint v3** | 38.588 | 0.8857 | **34.359** | 0.9169 | 60 min |
| MC-GS single v3 | 38.518 | 0.8764 | 34.341 | 0.9158 | 136 min |
| L1-ESPIRiT | **38.607** | 0.9445 | 34.105 | 0.9221 | 4 min |
| joint CS-SENSE (JTV) | 38.544 | 0.9447 | 33.929 | 0.9196 | 7 min |
| CS-SENSE (TV, converged) | 38.333 | 0.9414 | 33.873 | 0.9195 | 4 min |
| MC-GS joint v2 | 37.844 | 0.8835 | 33.492 | 0.9023 | 39 min |
| MC-GS single v2 | 37.639 | 0.8708 | 33.367 | 0.8975 | 38 min |
| ZS-SSDU | 38.012 | **0.9525** | 33.269 | **0.9235** | 35 min |
| CS-SENSE (TV, 35 outer, as run 09-09) | 37.897 | 0.9346 | 33.252 | 0.9127 | 2 min |
| CG-SENSE | 34.616 | 0.7328 | 32.192 | 0.8834 | 2 min |
| MC-GS joint (old cold start) | 35.466 | 0.8669 | 31.806 | 0.8870 |  |
| MC-GS single (old cold start) | 35.176 | 0.8448 | 31.755 | 0.8814 |  |
| 3D-INR | 33.919 | 0.7932 | 29.939 | 0.8070 | 58 min |
| 3D-DIP | 34.288 | 0.7924 | 29.397 | 0.8048 | 52 min |
| ZF | 31.677 | 0.8503 | 26.738 | 0.6882 | 2 min |

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
revisit; its rank (40) was selected on a six-slice subset of hyw T1 (`../T1T2_AF12_retro/logs/ploraks/ranksweep_R2.log`).
The T1 row here is the hyw T1 full run of the T1+T2 topic (the same acquisition; 6.7 h). The FLAIR run (rank 40,
R = 2, 48 MATLAB workers) started 2026-09-11 00:38 and takes 7–8 h; its row enters `logs/summary.tsv` through
`scripts/summarize.py` when it lands.
Scripts: `recon_classical.py --suffix` (sweeps), `recon_joint_cssense.py`, `recon_l1espirit.py`,
`ploraks_export.py` / `run_ploraks.m` / `run_ploraks_ranks.m` / `ploraks_import.py`; sweep logs
in `logs/sweeps/`, per-run json in `logs/`.

## MC-GS joint v3 — the T1+T2 recipe with wavelet sparsity, run unchanged on T1+FLAIR (2026-09-10/11)

v3 is defined and selected on the huangyuwei T1+T2 pair (`../T1T2_AF12_retro/README.md`, section "MC-GS joint
v3"): v2's cold-start fixes + 2.4M initial points + a complex wavelet-sparsity term λ 0.05 on top of edge-TV,
4000 iterations, one strength for both contrasts. This topic shares the subject and the T1 acquisition with the
tuning pair, so it checks the recipe on a contrast it was not tuned on (FLAIR), not on a new subject.

```bash
cd $F && CONFIG=configs/t1flair_af12_hyw.yaml SAVE_VOLUMES=1 ./autoresearch_t1t2.sh ar13_tf_hyw_t1flair_v3b 4 4000 \
  --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn --set loss.edge_tv_kappa_source=zf \
  --set gaussian.initial_num_points=2400000 --set loss.lambda_wav=0.05
```

| arm (4000 iters) | head PSNR T1 / FLAIR | head SSIM | head HFEN | whole PSNR | sharp |
|---|---|---|---|---|---|
| **MC-GS joint v3** | 35.772 / 34.359 | 0.9470 / 0.9169 | 0.0828 / 0.1065 | 39.920 / 38.588 | 0.869 / 0.869 |
| MC-GS joint, 2.4M + wavelet 0.03 (first transfer run) | 35.471 / 34.061 | 0.9426 / 0.9111 | 0.0845 / 0.1102 | 39.597 / 38.267 | 0.882 / 0.892 |
| MC-GS single v3 (same recipe, one contrast) | 35.679 / 34.341 | 0.9455 / 0.9158 | 0.0851 / 0.1069 | 39.840 / 38.518 | 0.864 / 0.876 |
| L1-ESPIRiT | 35.530 / 34.105 | 0.9471 / 0.9221 | 0.0867 / 0.1096 | 39.823 / 38.607 | 0.899 / 0.804 |
| CS-SENSE (TV, converged) | 34.966 / 33.873 | 0.9460 / 0.9195 | 0.0993 / 0.1196 | 39.534 / 38.333 | 0.829 / 0.775 |
| ZS-SSDU | 34.134 / 33.269 | 0.9365 / 0.9235 | 0.1213 / 0.1432 | 38.635 / 38.012 | 1.012 / 1.110 |
| MC-GS joint v2 | 34.950 / 33.492 | 0.9379 / 0.9023 | 0.0851 / 0.1150 | 39.253 / 37.844 | 0.912 / 0.915 |

- **v3 is first on head PSNR on both contrasts**: +0.242 / +0.254 dB over L1-ESPIRiT and
  +0.82 / +0.87 over v2. Head HFEN is first on both. Whole-volume PSNR is first on T1 and
  second on FLAIR (L1-ESPIRiT 38.607 against 38.588). Head SSIM: T1 0.9470, second (best other arm L1-ESPIRiT 0.9471); FLAIR 0.9169, fifth (best other arm ZS-SSDU 0.9235).
- **Per slice** (`scripts/find_best_slice_v2.py`, full-brain slices, against every other arm; P-LORAKS T1 only):
  T1 first on 43/47, 83/88, 90/93 transverse / coronal / sagittal slices; FLAIR first on 55/61, 73/91, 94/94.
- **Joint versus single**: the joint model is +0.093 dB on T1 (the single-contrast T1 is the T1+T2 topic's
  run on the same acquisition) and +0.018 on FLAIR. Unlike the T1+T2 pair and jiangyizhou, the coupling
  costs the second contrast nothing here.
- **Gate**: T1 sharpness 0.869, below the pre-registered 0.874 floor by 0.005; reported as missed.
- **FLAIR head SSIM** (program, "SSIM diagnosis" and Round 17): three quarters of the gap to L1-ESPIRiT sits in
  the flattest fifth of the head, in the darkest decile (suppressed CSF), where v3's local mean fluctuates. The
  wavelet strength does not reach that part; it acts on local correlation. FLAIR-channel strengths 0.08 and 0.12
  are being run on this topic (Round 17), and a second seed for a two-seed average (Round 18).
- P-LORAKS: the T1 row is the T1+T2 topic's run (same acquisition); the FLAIR run is still in progress.
- Figures (compact panels, v3 as the proposed method): `figures/t1flair_af12_T1_best_axis0_147.png`,
  `figures/t1flair_af12_FLAIR_best_axis0_153.png` (v3's best transverse slices),
  `figures/t1flair_af12_{T1,FLAIR}_best3planes.png`.

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
