# Autoresearch — pediatric MTP AF12, stage 4: MC-GS at least 1 dB above every other arm

Asked 2026-09-14 (evening): start a Karpathy-style autoresearch loop so that MC-GS beats every
comparison method by 1 dB. This continues stage 3 (`autoresearch_peds.md`, record
`docs/superpowers/records/2026-09-14-16yo-mtp-stage3.md`, tag `handoff/2026-09-14-16yo-mtp-stage3`),
whose bar was "first in all six cells"; the bar here is stricter.

Ledger: the same `autoresearch_peds.tsv` (the runner writes it; stage-4 runs are named `p4_*`).
Queue: `Results_20260908/Peds_TSC_MTP_AF12_retro/scripts/p4_queue.txt`, scheduler `scripts/p4_sched.sh`
of the topic (one detached instance; log `logs/p4_sched.log`).

## The bar

- **Head PSNR in the primary region** (head inside partitions 12..92, `scripts/score_candidate.py`)
  **at least 1.00 dB above the best other arm, on T1w and on T2s.**
- "Other arm" = all nine other arms of the table, **GS-single included** (user, 2026-09-14), at its best
  available setting. Whatever generic lever MC-GS gets (a weight on one contrast, iterations,
  capacity, optimiser, phase TV), GS-single gets too, and the bar moves with it (stage-3 rule).
- Head SSIM and HFEN are reported beside it; a PSNR margin bought by losing either is said so.

Where it starts (primary region, from `logs/rescore.json` and the ledger):

| | best other arm | target | MC-GS now |
|---|---|---|---|
| T1w head PSNR | 38.848 (GS-single tuned; L1-ESPIRiT 38.830) | **39.848** | 38.516 at 4000 it, 38.588 at 6000 |
| T2s head PSNR | 34.175 (GS-single; L1-ESPIRiT 33.860) | **35.175** | 34.163 at 4000 it, 34.328 at 6000 |

## Why only the joint part can meet it

The margin over a GS-single that receives every generic lever is the joint-minus-single gap, which is
−0.33 dB on T1w and −0.01 dB on T2s at the stage-3 working point. Levers that act on one contrast by
itself move both arms and leave that gap where it is; they are only worth running to stay ahead of
L1-ESPIRiT. The rounds therefore go first to mechanisms that pass information between the contrasts.

## Round 0 — is there a dB left in this data? (CPU, 2026-09-14)

The reference is GRAPPA-completed, SENSE-combined data, so it carries noise, and the noise at the
unsampled positions is unpredictable by any method. Measured on the complex background far from the
head (MAD, per component): σ = 0.0044 (T1w), 0.0057 (T2s), gmax 2.20. The noise at the 17279 unsampled
in-support positions caps head PSNR near **54.6 dB (T1w) and 52.4 dB (T2s)**; the current error is
16–18 dB above that. What is missing is unrecovered signal, so noise does not rule the target out.
(Two other estimators were discarded: the k-space tail is still signal-dominated at the support edge,
and the finest Haar band sits on the measured-zero corners.)
Scripts and numbers: `peds_tsc_mtp_af12_retro/analysis_p4/ceiling_bg.{py,json}`, `ceiling_noise.{py,json}`.

## Round 0b — can T1w carry T2s's detail through a smooth ratio? No (CPU, 2026-09-14)

The one cross-contrast mechanism that could give T2s's unsampled high frequencies real information
from T1w is a signal model: both come from one MTP exam, so T2s = T1w · ρ with ρ a smooth map of the
T1 and T2* weighting. Tested on the ground truth before writing any code: ρ restricted to a low-pass
band on every axis, fitted by CG in the head, and T1w·ρ scored against T2s in the primary region.

| model | band per axis | head PSNR of the approximation |
|---|---|---|
| complex ρ | 1/16 | 20.29 |
| complex ρ | 1/8 | 21.89 |
| complex ρ | 1/4 | 24.52 (24.60 with 120 CG iterations: converged) |
| magnitude ratio, real r | 1/8 | 25.08 |
| magnitude ratio, real r | 1/4 | 26.96 |
| one global scalar | — | 19.09 |
| (current T2s reconstruction from its own data) | | 34.16 |

Even with the ground-truth T1w and a ρ that keeps a quarter of the band, the product explains T2s
10 dB worse than T2s's own undersampled data already do; the reverse direction is worse still
(T1w from T2s: 18.8 / 20.2 / 22.0). The two contrasts differ in fine detail far beyond any smooth
modulation, so the ratio coupling is dropped without a GPU run. What a joint prior can still use
here is where edges sit, not what the intensities are. Script and numbers:
`peds_tsc_mtp_af12_retro/analysis_p4/ratio_fast.{py,json}`, `ratio_check.{py,json}`.

## Levers queued (priority order, `scripts/p4_queue.txt`)

1. `p4_dcw02` — **who drives the shared geometry.** At the end of a c1 run the T2s data term is ~5x
   the T1w one, so the cloud is pulled mostly by the noisier contrast. `loss.dc_weight_contrast` (new,
   `losses/balance.py`, 2 tests, default bit-identical) scales T2s's data term; T2s's own regularisers
   get the same factor, so under Adam T2s's densities see no change and only the geometry split moves.
2. `p4_gwav` — joint (group) wavelet sparsity across the contrasts (new `loss.wav_group`, 6 tests).
3. `p4_dtvswap` — half of T2s's own edge-TV replaced by T1w-guided directional TV.
4. `p4_dcw05`, `p4_dtv005`, `p4_gwav_x14` — the second points of those three scans.

## Round log

### Capacity binds for the joint arm (2026-09-14 20:21, inherited run `p3_n60`)

The stage-3 capacity run finished after that session's handoff. Primary region, head PSNR / SSIM / HFEN:

| run | points | iters | T1w | T2s |
|---|---|---|---|---|
| `p3_c1_wav006_gp003` | 3.6M | 4000 | 38.516 / 0.9668 / 0.0774 | 34.163 / 0.9047 / 0.1857 |
| `p3_i6000` | 3.6M | 6000 | 38.588 / 0.9671 / 0.0771 | 34.176 / 0.9049 / 0.1855 |
| `p3_n60` | 6.0M | 4000 | **38.791** / 0.9680 / 0.0770 | **34.195** / 0.9048 / 0.1863 |
| `p3_single_T1w_c1` (single) | 3.6M | 4000 | 38.848 / 0.9681 / 0.0761 | — |

+0.275 dB on T1w for 0.5 GB more VRAM (38.4 GB) and 79 instead of 67 minutes. The stage-3 record had
written "+0.001" for this lever before the run had finished; corrected in place. One cloud carries both
contrasts in the joint arm, so capacity may relieve exactly the sharing conflict that costs T1w; the
test is whether the single arms gain less at the same capacity (`p4_single_T1w_n60`,
`p4_single_T2s_n60`) and whether the joint keeps gaining (`p4_n100`, 10M). Those three went to the
head of the queue.

### Where the joint arm loses T1w: at T1w's own edges where T2s is flat (CPU, 20:33)

`analysis_p4/where_joint_loses.py`: voxels of the primary head region binned by ground-truth edge
strength of each contrast (Gaussian-derivative gradient, normalised by its head 99th percentile;
flat < 0.1 ≤ mid < 0.3 ≤ edge), stage-3 tuned arms at 3.6M. Joint MSE 6.80e-4 against single 6.30e-4.

| T1w | T2s | voxels | share of the excess | joint vs single in the bin |
|---|---|---|---|---|
| edge | flat | 7.3 % | 13.7 % | +0.50 dB |
| edge | mid | 11.0 % | 22.2 % | +0.50 dB |
| mid | flat | 11.0 % | 17.5 % | +0.42 dB |
| edge | edge | 8.1 % | 11.8 % | +0.37 dB |
| flat | edge | 1.8 % | 0.9 % | +0.17 dB |
| flat | flat | 22.7 % | 8.7 % | +0.20 dB |

The deficit is not T2s-only structure taking T1w's capacity (flat|edge costs least). It is T1w detail
that T2s does not share being smoothed away: the shared cloud also answers to T2s's noisier data and
heavier smoothing, and where T2s is flat that pressure merges what T1w needs separate. Levers that
follow: less T2s pull on the geometry (`p4_dcw02`, moved to second place), more and smaller Gaussians
(`p4_n100`), and, if those confirm the mechanism, keeping T2s's regularisers off the geometry.

## Target revised by the user (2026-09-14, ~20:40): one lesion slice, +1 dB

"只需要一张包含病灶的 slice 超过所有对比方法 1 dB 即可." The whole-volume bar above is dropped. New bar:
**one slice that contains a lesion, on which MC-GS's head PSNR is at least 1.00 dB above every other arm**,
per-slice PSNR exactly as the row figure computes it (`fig_row_final.py`: head = |GT| > 0.05·P99.9 of the volume,
peak = volume max |GT|). GS-single stays a competitor (user, 2026-09-14), and every GS-single volume available
counts toward the bar on a slice. Tool: `scripts/p4_lesion_margin.py --exp <run>` (writes `<run>_lesion.json`).

Lesion slices (T2*w, where the lesions are visible; from the 2026-09-14 lesion check): subependymal nodules on vendor
slices 38 / 46 = transverse SPE 53 / 45 (±1); the calcified-looking focus at RO 166, PE 196, SPE 68–70 = transverse
68–70, coronal 163–169, sagittal 193–199. T1w shows no focal change at the nodules.

Where the existing MC-GS volumes stand (`analysis_p4/lesion_slice_margins.{py,json}`): on **every** lesion slice the
arm to beat is GS-single, and every joint run is within ±0.1 dB of it. Best so far:

| run | best lesion slice | MC-GS | best other | margin | margin over the non-GS arms |
|---|---|---|---|---|---|
| `p3_edge0010` (common edge 0.001) | T2s transverse 69 | 33.545 | 33.450 GS-single | +0.096 | +0.457 |
| `p3_n60` (6.0M points) | T2s transverse 69 | 33.520 | 33.450 GS-single | +0.070 | +0.432 |
| `p3_c1_wav006_gp003` (c1) | T2s transverse 69 | 33.514 | 33.450 GS-single | +0.064 | +0.425 |

On T1w the joint runs trail the tuned single on all lesion slices (−0.03 to −1.1 dB). The queue was rebuilt around
cross-contrast levers that can move T2s: common edge 0.002 / 0.004, T1w-driven geometry (`dc_weight_contrast` 0.2),
group wavelet, T1w-guided directional TV (swap 0.05, strong 0.20), frozen-geometry T2s regularisers, T2s scale offset.
The capacity-fairness and T1w-side runs are parked in the queue file.

### Results against the lesion-slice bar (primary region numbers are volume-level, head PSNR / SSIM / HFEN)

| run | lever | T1w primary | T2s primary | best lesion slice | margin | over non-GS arms | verdict |
|---|---|---|---|---|---|---|---|
| `p4_smoke_newlevers` | all new switches, 60 iters | (smoke) | (smoke) | — | — | — | every switch logs, loss terms add up, volumes and scores written |
| `p4_dtv005` | weighted dTV 0.05 on top of c1 | 38.510 / 0.9668 / 0.0774 | 34.013 / 0.8965 / 0.1965 | coronal 183 | −0.014 | +0.235 | discard: T2s over-smoothed on all three cells |
| `p4_sgtv010` | structure-guided TV 0.10 replacing T2s's edge-TV | 38.520 / 0.9669 / 0.0774 | 33.162 / 0.8754 / 0.2329 | coronal 179 | −0.654 | −0.308 | discard: −1.0 dB on T2s; T2s-only edges (veins, lesions) get the full penalty |

Both guided-smoothing runs say the same thing: T2s loses whenever smoothing is added, including smoothing that spares
T1w's edges. The additive structure-guided run was dropped. What is left of "use T1w's structure" is the opposite
direction: `loss.edge_tv_union_guide` (new, 4 tests) lowers T2s's edge-TV weight also at T1w's edges and never raises
it, so T1w's structure can only remove smoothing from T2s (`p4_union`, now first in the queue).

`p3_single_T1w_i6000` (another session's fairness run): T1w 38.889 / 0.9683 / 0.0760, so iterations give the joint
+0.072 and the single +0.041 (joint − single −0.332 → −0.301); not a T2s lever.

**Coordination (21:30).** The stage-3 session came back to this line and runs one training at a time (its fairness run
`p3_single_T1w_n60`). To stay inside the two-training cap without starving it, `scripts/p4_sched2.sh` replaces
`p4_sched.sh`: the line cap of two still counts every session's runs, and this campaign keeps at most one of its own.
This campaign no longer edits `STATUS.md`, `CLAUDE.md` or the stage-3 record; its notes stay in this file.

`p3_single_T1w_n60` (stage-3 session's fairness run, 22:3x): T1w 38.989 in the primary region, so capacity 3.6M → 6.0M
gives the single +0.141 and the joint +0.275; joint − single at 6.0M is −0.198. T1w context only.

### How far each reading of "every comparison method" is (23:05, all joint volumes so far)

| run | best lesion slice, GS-single counted | margin | best lesion slice, GS-single left out | margin |
|---|---|---|---|---|
| `p3_edge0010` | transverse 69 | +0.096 | coronal 136 | +0.513 |
| `p3_n60` | transverse 69 | +0.070 | coronal 136 | +0.508 |
| `p3_i6000` | transverse 69 | +0.076 | coronal 136 | +0.471 |
| `p3_c1_wav006_gp003` | transverse 69 | +0.064 | coronal 136 | +0.471 |
| `p4_dtv005` | coronal 183 | −0.014 | coronal 136 | +0.367 |
| `p4_sgtv010` | coronal 179 | −0.654 | coronal 179 | −0.308 |

With GS-single counted the gap to +1 dB is 0.9 dB and only a joint-specific T2s gain can close it; without it the gap
is 0.49 dB and generic T2s tuning (its own weights were never scanned), capacity and iterations also count.
The user's words were "所有对比方法" and GS-single has been a comparison method since 2026-09-14, so the campaign keeps
it; the second column is reported so the choice can be revisited.

**GPU state 23:01.** The stage-3 session now runs two trainings on this line (`p3_n96` joint 9.6M, `p3_single_T1w_n96`),
which fills the two-training cap; `p4_union` waits for one of them. Cards 3, 6 and 7 each carry two trainings from
other lines at the moment.

### Capacity is not a T2s lever (2026-09-15 00:45, stage-3 session's `p3_n96`, joint at 9.6M points)

| points | T1w primary | T2s primary | best lesion slice, all arms | best lesion slice, non-GS arms |
|---|---|---|---|---|
| 3.6M (c1) | 38.516 | 34.163 / 0.9047 / 0.1857 | transverse 69 +0.064 | coronal 136 +0.471 |
| 6.0M | 38.791 | 34.195 / 0.9048 / 0.1863 | transverse 69 +0.070 | coronal 136 +0.508 |
| 9.6M | 38.936 | 34.176 / 0.9037 / 0.1871 | coronal 163 / transverse 69 +0.045 | coronal 136 +0.513 |

T1w keeps gaining with points; T2s peaks at 6.0M and then loses SSIM and HFEN (the long-TE contrast has no detail left
for extra Gaussians to carry, so they fit noise). One cloud serves both contrasts, so capacity cannot be spent on T1w
alone. The lesion-slice campaign stays on the 3.6M c1 base.

### The bound, put to the user (2026-09-15 00:50)

Whole-head T2s joint minus single at every setting run on this line: −0.015 (P10), −0.012 (c1), +0.020 (6.0M), +0.001
(9.6M). Best lesion-slice margins over GS-single: +0.096 (common edge 0.001), +0.070 (6.0M), +0.045 (9.6M). A 1 dB margin
over GS-single on one T2s slice would have to be a slice-local effect about thirty times the volume-level difference;
T2s is SNR-limited and the shared-geometry prior adds little information to it (claude-e7 reached the same reading from
the six-cell side). Options given to the user: keep the bar (the queue continues, expected best a few tenths at most);
count GS-single as an ablation (current +0.513 over the other eight arms, coronal 136); or state a smaller margin.
The queue keeps running while the user decides.

`p3_single_T1w_n96` (stage-3 session): T1w 38.993, so T1w joint − single at matched capacity is −0.332 (3.6M), −0.198
(6.0M), −0.057 (9.6M): on T1w the joint deficit is mostly capacity starvation. T2s, the lesion contrast, does not share it.

### Round 1 winner: union-of-edges TV for T2s (`p4_union`, 2026-09-15 03:05)

| | T1w primary | T2s primary | best lesion slice, all arms | best lesion slice, non-GS arms |
|---|---|---|---|---|
| c1 (base) | 38.516 / 0.9668 / 0.0774 | 34.163 / 0.9047 / 0.1857 | transverse 69 +0.064 | coronal 136 +0.471 |
| **`p4_union`** | 38.517 / 0.9668 / 0.0774 | **34.195 / 0.9054 / 0.1842** | **transverse 69 +0.104** | sagittal 171 +0.505 |
| GS-single bars | 38.848 (tuned) | 34.175 / 0.9052 / 0.1855 | | |

The first lever that moves T2s without costing T1w: all three T2s cells now lead GS-single at volume level (+0.020 dB,
+0.0002, −0.0013 HFEN) and T1w is unchanged. Small, but it is the right direction: T1w's structure only ever removes
smoothing from T2s. KEPT; round 2 stacks everything on it (group wavelet, common edge 0.001 / 0.002 / 0.0005, a lower
guide kappa via the new `loss.edge_tv_union_kappa_scale`, T2s's own weights, geometry split, frozen regularisers).

### Round 2 results (all on the union base: c1 + edge_tv_union_guide)

| run | lever | T1w primary | T2s primary | best lesion slice, all arms | non-GS arms | verdict |
|---|---|---|---|---|---|---|
| `p4_u_gwav` | + group wavelet | 38.303 / 0.9652 / 0.0783 | 34.164 / 0.9040 / 0.1856 | sagittal 170 +0.060 | coronal 136 +0.478 | discard: −0.21 T1w, −0.03 T2s against the base; T2s's noisy coefficients relax T1w's shrinkage |
| `p3_n192_union` (stage-3 session) | 19.2M points + union | 38.924 / 0.9685 / 0.0768 | 34.203 / 0.9043 / 0.1854 | transverse 69 +0.073 | coronal 136 +0.522 | capacity still costs T2s SSIM; below the 3.6M union on the lesion bar |
| **`p4_u_edge0010`** | + common edge 0.001 | 38.382 / 0.9659 / 0.0804 | **34.223 / 0.9060 / 0.1826** | **sagittal 172 +0.124** (transverse 69 +0.117) | sagittal 171 +0.531 | **KEEP** for the T2s lesion target: best T2s so far, the two levers stack; T1w −0.135 is irrelevant to this target |

The remaining non-scan probes now carry `--lambda_edge 0.001` as well (`p4_ue_*`). Sagittal 172 runs through the
right-sided subependymal nodule (PE 172 on SPE 45).
| **`p4_u_k05`** | union, guide kappa × 0.5 | 38.514 / 0.9668 / 0.0774 | **34.208 / 0.9063 / 0.1826** | sagittal 170 +0.123 | sagittal 171 +0.527 | **KEEP**: as good as adding common edge on T2s, at zero T1w cost; stacked next (`p4_ue_k05`) and pushed to × 0.25 |
| `p3_n96_union` (stage-3 session) | 9.6M points + union | 38.936 / 0.9685 / 0.0768 | 34.215 / 0.9047 / 0.1852 | sagittal 170 +0.087 | coronal 136 +0.525 | below the 3.6M union runs on the lesion bar; capacity keeps not helping T2s slices |
| `p4_u_edge0020` | union + common edge 0.002 | 38.101 / 0.9633 / 0.0873 | 34.196 / 0.9051 / 0.1841 | sagittal 172 +0.095 | sagittal 171 +0.488 | discard: past the optimum; T2s below edge 0.001 and T1w −0.28 more |
| `p3_n96_union_k05` (stage-3 session) | 9.6M + union + kappa x0.5 | 38.936 / 0.9685 / 0.0768 | 34.227 / 0.9056 / 0.1836 | sagittal 170 +0.108 | sagittal 171 +0.509 | T2s volume cells all above GS-single at 9.6M; still below the 3.6M runs on the lesion bar |
| **`p4_ue_k05`** | union + kappa x0.5 + common edge 0.001 | 38.386 / 0.9659 / 0.0803 | **34.230 / 0.9066 / 0.1815** | **sagittal 172 +0.153** | sagittal 171 +0.550 | **KEEP, round-3 base**: the two T2s levers do stack (+0.12 each alone); best T2s on all three cells |

`scripts/p4_lesion_margin.py` now counts every registered GS-single arm of the contrast (the stage-3 session registered
`gs_single_p3f`, T1w only), so the T2s bars did not move.

### Round 3 results (base: union + kappa x0.5 + common edge 0.001)

| run | lever | T1w primary | T2s primary | best lesion slice, all arms | non-GS arms | verdict |
|---|---|---|---|---|---|---|
| `p4_r3_wavg` | + one-way guided wavelet for T2s | 38.379 / 0.9659 / 0.0803 | 34.047 / 0.9016 / 0.1826 | transverse 52 +0.098 | sagittal 173 +0.493 | discard: −0.18 dB T2s; relaxing TV at T1w edges helps, relaxing wavelet shrinkage there lets T2s noise in |

`p4_ue_k025` was placed on card 2 one minute after a CE-topic job started there but before that job took the GPU, so
two trainings share the card; stopping it was refused by the session's permission rules and it runs at half speed.
| `p4_ue_k025` | kappa x0.25 instead of x0.5 (shared card, 141 min) | 38.385 / 0.9659 / 0.0802 | 34.192 / 0.9066 / 0.1818 | sagittal 171 +0.119 | sagittal 171 +0.536 | discard: past the kappa optimum; x0.5 stays |

**Current best figure (13:50).** `scripts/p4_fig_lesion_slice.py --exp p4_ue_k05 --axis 1 --index 172 --mark "148,45;166,45"`:
T2s sagittal 172 through the two right subependymal nodules. MC-GS 33.84 dB / 0.8800 / 0.00845 against GS-single
33.69 / 0.8765 / 0.00876 and L1-ESPIRiT 33.34 / 0.8753 / 0.0095: first on all three metrics, +0.153 dB over every
arm, +0.50 over the non-GS arms on this slice. Selected slice, labelled as such in the figure.
`/home/congcongliu/figs/peds_tsc_af12_lesion_1db/peds_af12_T2s_sagittal_172_p4_ue_k05.png`
| `p4_r3_geo007` | T2s edge-TV 0.10 → 0.07 | 38.389 / 0.9660 / 0.0802 | 34.167 / 0.9065 / 0.1826 | sagittal 172 +0.101 | sagittal 171 +0.508 | discard: less smoothing everywhere loses; only the relaxation at T1w edges helps |
| **`p4_r3_edge0005`** | common edge 0.001 → 0.0005 | **38.481 / 0.9667 / 0.0779** | **34.228 / 0.9067 / 0.1814** | sagittal 172 +0.150 | sagittal 171 +0.554 | **co-incumbent**: same T2s as edge 0.001 at a third of the T1w cost (−0.035 against c1 instead of −0.13); the preferred recipe for the figure |

Plateau: round 3 has not moved the lesion-slice margin past +0.153 (four runs). Stopping rule for this campaign: once
the remaining round-3 probes (T2s wavelet 0.14, frozen-geometry regularisers, geometry split) are in, stop if none
adds ≥ 0.05 dB, and write up.
