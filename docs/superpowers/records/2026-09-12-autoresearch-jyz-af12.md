# autoresearch on jiangyizhou T1 + FLAIR at AF12 — the MC-GS lead over every arm, widened (2026-09-11 19:40 → 2026-09-12)

Branch `autoresearch-jyz-af12` (worktree `/data0/congcong/code/GS/mcgs_af12_wt`, forked from `autoresearch-jyz-af16`
at 3a44aba, the code that produced P6). Program and round log: `autoresearch_jyz_af12.md`; ledger
`autoresearch_jyz_af12.tsv` (63 rows); every row's score json and log in `results_arc/`. Topic archive:
`Results_20260908/T1FLAIR_AF12_v3_jiangyizhou` (tables, volumes, figures), mirrored here as
`t1flair_af12_v3_jiangyizhou/`. Cards 3, 4, 5 (one trainer per card), card 6 for the classical re-sweep; the
AF16 phase-2 loop held 0–2 throughout.

## 1. The bar

The user's starting point was P6 — the jyz AF16 loop's final recipe run unchanged on the AF12 mask (one vd-Poisson
mask, AF 11.71 on T1 / 9.74 on FLAIR against their own acquired support): head PSNR 36.753 / 36.165 at 4000
iterations, marked "provisional best 12x reconstruction" on 2026-09-11. The objective: raise MC-GS joint and widen
its lead over every comparison arm on every head metric.

Before the first candidate, the three strongest classical arms had their λ **re-swept on this subject against the
ground truth** (the same privilege the loop has; until then their λ came from huangyuwei). L1-ESPIRiT: T1 peak
λ 0.002 = 36.426 (was 0.0015 = 36.334), FLAIR peak 0.004 = 35.757 (was 0.005 = 35.720); converged TV peaks at
0.0007 on both (36.162 / 35.581); joint TV at 0.001 (36.070 / 35.624). The bar per cell is the best of **every**
swept λ, so a competitor tuned for SSIM or HFEN counts for that cell:

| cell | T1 bar | FLAIR bar |
|---|---|---|
| head PSNR | 36.426 (L1-ESPIRiT λ 0.002) | 35.757 (L1-ESPIRiT λ 0.004) |
| head SSIM | 0.9473 (L1-ESPIRiT λ 0.0025) | 0.9368 (L1-ESPIRiT λ 0.004 / 0.005) |
| head HFEN | 0.1034 (L1-ESPIRiT λ 0.0015) | 0.1000 (L1-ESPIRiT λ 0.004) |

Rules, pre-registered before any candidate ran (program, "Decision rules"): screening at 1500 iterations; keep =
the smaller of the two head-PSNR margins rises by more than the seed floor **and** none of the other five head
cells (SSIM ×2, HFEN ×2, the other PSNR) falls beyond its floor **and** sharpness does not drop more than 0.01;
promotion = the same test at 4000 iterations with volumes saved; the promoted run becomes the parent and the
runner bakes it. Seed floors: 0.006 / 0.004 dB at 1500 (P6, P8 both), 0.002 / 0.001 dB at 4000 (P10); SSIM and
HFEN floors 0.0000–0.0002. Two clarifications were added during the loop and are dated in the program: a run whose
cells all hold and at least one improves is a *Pareto keep* even when the smaller margin does not move (the
binding channel swaps); and Pareto keeps at the floor do not change the recipe once the final rows are running.

## 2. Result

**P10 = P6 with the wavelet prior reshaped**: db4 wavelet family (was db2), the L1 penalty on the finest
decomposition level only (was 3 levels), strengths T1 0.12 / FLAIR 0.10 (were 0.10 / 0.07). Everything else is
P6: 3.6M Gaussians, edge-TV λ_geo 0.10 / κ_q 0.85 / w_floor 0.10 / eps 0.04, kspace_high 2.2, repaired cold
start (head-support init, 3-NN scales, κ from ZF), 4000 iterations, seed 42.

| head cell | P6 | **P10** | P10 − P6 | bar | P10 − bar |
|---|---|---|---|---|---|
| T1 PSNR | 36.753 | **36.944** | +0.191 | 36.426 | **+0.518** |
| FLAIR PSNR | 36.165 | **36.268** | +0.103 | 35.757 | **+0.511** |
| T1 SSIM | 0.9480 | **0.9503** | +0.0023 | 0.9473 | +0.0030 |
| FLAIR SSIM | 0.9374 | **0.9397** | +0.0024 | 0.9368 | +0.0030 |
| T1 HFEN | 0.1009 | **0.0947** | −0.0062 | 0.1034 | +0.0087 |
| FLAIR HFEN | 0.1002 | **0.0958** | −0.0044 | 0.1000 | +0.0042 |
| sharpness | 0.857 / 0.832 | 0.858 / 0.823 | +0.001 / −0.009 | L1-ESPIRiT 0.906 / 0.809 | — |
| whole-volume PSNR | 40.071 / 39.539 | **40.272 / 39.692** | +0.201 / +0.153 | 39.893 / 39.205 | +0.379 / +0.487 |

Under P6 the FLAIR HFEN cell was level with L1-ESPIRiT and the T1 SSIM cell led by 0.0007; under P10 every head cell
leads, the thinnest by 0.0030 (SSIM, both contrasts). Seed 43 reproduces P10 to 0.002 / 0.001 dB.

The promotion chain (each step judged at 4000 iterations against the previous parent, all six cells holding or
improving):

| parent | change | head PSNR at 4000 | step |
|---|---|---|---|
| P6 | — | 36.753 / 36.165 | — |
| P7 | wavelet on 2 levels | 36.765 / 36.168 | +0.012 / +0.003 |
| P8 | db4 family | 36.819 / 36.189 | +0.054 / +0.021 |
| P9 | 1 level + FLAIR 0.085 | 36.853 / 36.234 | +0.034 / +0.045 |
| **P10** | FLAIR 0.10 + T1 0.12 | **36.944 / 36.268** | +0.091 / +0.034 |

Extra rows at the final recipe (own rows, not the headline): two-seed average (seeds 42 + 43, 2x compute)
36.990 / 36.295; scale lr 0.005 at 4000 36.955 / 36.316 and 0.008 36.953 / 36.328 (both raise the smaller PSNR
margin but slip T1 HFEN by 0.0002–0.0003, over its floor — recorded as marginal, not adopted); sym4 in place of db4
36.952 / 36.272 (equivalent within the floor); GS-single T1 (same recipe, T1 alone) 36.897 (joint +0.047).
Per slice (`scripts/find_best_slice_ar.py`, head-masked slice PSNR over every full-brain slice, the classical
arms at their best re-swept λ, P-LORAKS included): P10 is first on **every** slice — T1 96/96 transverse, 121/121
coronal, 118/118 sagittal; FLAIR 94/94, 120/120, 118/118 (`logs/best_slice_ar_{T1,FLAIR}.json`). Under v3.2 the T1
count was 9/96, 69/121, 33/118.

Rows still running at the time of writing are appended in section 7.

## 3. What moved and what did not

Levers, single knobs on the parent of their day at 1500 iterations unless stated (full numbers in the ledger):

| lever | verdict |
|---|---|
| wavelet **family** db2 → db4 | keep, +0.048 / +0.020; db6 −0.017; sym4 = db4 |
| wavelet **depth** 3 → 2 → 1 level | 2 levels keep (+0.015 / +0.004, HFEN better); 1 level T1 +0.037 / HFEN better on both but FLAIR −0.014 alone, fixed by more FLAIR strength; 4 levels null |
| wavelet **strength** under the 1-level db4 prior | T1 0.10 → 0.12 +0.073 (0.14 fails sharpness); FLAIR 0.07 → 0.085 → 0.10 → 0.12 = 36.080 → 36.135 → 36.160 → 36.163 (plateau); under the old 3-level db2 prior, up (0.14/0.10) or down (0.07/0.05) were both losses or reshapes |
| capacity 2.4M / 4.2M / 4.8M / 6.0M | at 4000 iterations every step above 3.6M moves quality from T1 to FLAIR (T1 −0.004 … −0.014); settled at 3.6M — the 1500-iteration screen does not see T1's late over-fit |
| scale lr 0.002 → 0.003 → 0.005 → 0.008 → 0.012 | rises at 1500 (+0.063 / +0.129 for 0.008 on P10); at 4000 it is a reshape on P9 (T1 −0.011) and marginal on P10 (T1 HFEN −0.0002/−0.0003) |
| iteration budget 1500 / 2500 / 4000 / 6000 | monotone, diminishing: P6 36.704 → 36.734 → 36.753 → 36.760 (T1); promotion budget kept at 4000 |
| edge-TV λ_geo 0.07 / 0.13, κ_q 0.90, w_floor 0.20, eps 0.03 | 0.07 sharpens but loses 0.06 dB; 0.13 and κ_q 0.90 smooth (HFEN, sharpness); w_floor −0.02/−0.05; eps null |
| k-space band weights (centre 0.15 / 0.5, mid 0.8) | centre null both ways; mid 0.8 −0.16 / −0.06 |
| warm-up 250 / 700, density lr 0.01 / 0.02, wavelet eps 1e-3, init head fraction 0.03 | null, null, null, −0.18 / −0.19 |

The whole gain came from the *shape* of the sparsity prior — which basis, which scales — not from its weight or
from the geometry side. Every smoothing knob traded HFEN and sharpness for PSNR and was refused by the rule.

## 4. Caveats

- **Magnitude gauge.** Every number above is | |rec| − |GT| |. On the strict complex difference |rec − GT|
  (`scripts/complex_gauge.py` in the topic) L1-ESPIRiT is first on both contrasts, 33.449 / 32.961 against P10's
  33.418 / 32.887; MC-GS loses 3.5 dB from magnitude to complex where the classical arms lose 2.8–3.0, and its
  in-head phase agreement is 0.9855 / 0.9821 against L1-ESPIRiT's 0.9862 / 0.9833. Same picture as the AF16 line.
- **Sharpness sits at the edge of its allowance**: P10 is −0.009 on both contrasts against P9 (allowance 0.01),
  and T1 0.14 failed it. Further wavelet strength is not available without paying in sharpness.
- **Screening at 1500 iterations misreads two lever families**: capacity and scale lr both look like clean gains at
  1500 and become T1-for-FLAIR reshapes at 4000. Their decisions were taken at the promotion budget only.
- **One subject, fair but not held out**: the classical λ and the MC-GS knobs were both selected on jiangyizhou's
  ground truth. The recipe has not been run on another subject or another acceleration.
- **FLAIR known zeros**: as in every earlier topic, the GS arms fit FLAIR's measured shutter as zeros; the seven
  comparison arms use the mask only. Unchanged, unmeasured.
- **Whole-volume SSIM** stays sixth (air-noise gauge), reported, not optimised.

## 5. Open decisions (for the user)

1. Accept scale lr 0.005 on top of P10 (36.955 / 36.316, smaller PSNR margin +0.529, FLAIR cells up, T1 HFEN
   −0.0002)? The loop's rule says no; the cost is below anything visible.
2. Whether the topic's headline stays the single seed-42 run (36.944 / 36.268) with the two-seed average as its own
   row (36.990 / 36.295), as in the earlier topics.
3. A second subject at AF12 with the recipe frozen (the CFL subjects vol2/5/6/7, or tangbing).
4. The branch is local only; push `autoresearch-jyz-af12` to origin. `main` shares no history with this lineage.

## 6. Reproduce

```bash
cd /data0/congcong/code/GS/mcgs_af12_wt          # branch autoresearch-jyz-af12; the runner bakes P10
SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh <EXP> <GPU> 4000        # = ar5_best_C10_T112_full4k
# explicit form of P10 on the 3a44aba runner (P6 baked):
#   --set loss.wav_name=db4 --set loss.wav_levels=1 --set loss.lambda_wav=0.10 --set loss.lambda_wav_plain=0.12
python scripts/autoresearch_jyz_af12_view.py     # every ledger row against the current bar, six cells
# topic: register a run as an arm, refresh the table, complex gauge, compact three-plane figure
cd /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou
python scripts/add_gs_arm.py --exp ar5_best_C10_T112_full4k --key gs_joint_ar --label "MC-GS (AF12 loop, P10)"
python scripts/summarize.py && python scripts/complex_gauge.py
python scripts/fig_three_planes_compact.py --contrast T1 --slices 174 110 47 --mask_error
```
