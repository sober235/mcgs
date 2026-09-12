# jiangyizhou AF12 autoresearch — MC-GS joint ahead of every arm on every head cell (2026-09-11 19:40 → 2026-09-12)

Branch `autoresearch-jyz-af12` (from `autoresearch-jyz-af16` at 3a44aba), worktree `/data0/congcong/code/GS/mcgs_af12_wt`.
Program and round log: `autoresearch_jyz_af12.md`; ledger `autoresearch_jyz_af12.tsv` (every row has the commit and the
exact args); topic `Results_20260908/T1FLAIR_AF12_v3_jiangyizhou` (mirrored as `t1flair_af12_v3_jiangyizhou/`).

## 1. The bar

The user marked the AF16 loop's recipe (P6) as the provisional best 12x reconstruction and asked for a loop that widens
MC-GS joint's lead over every comparison arm on every metric, on jiangyizhou AF12 (one vd-Poisson mask, AF 11.71 on T1 /
9.74 on FLAIR against their own support). Six head cells per run: PSNR, SSIM and HFEN margins per contrast over the
best non-MC-GS arm of that cell. Before the loop the classical arms ran at the huangyuwei-selected λ; the loop first
re-swept λ **on this subject against the ground truth** (the same privilege it has itself) for converged TV, joint TV
and L1-ESPIRiT (28 BART/GPU runs, `scripts/run_classical_resweep_af12.sh`, `run_l1_refine_af12.sh`), and the bar
takes the best of every swept λ per cell:

| cell | T1 bar | FLAIR bar |
|---|---|---|
| head PSNR | 36.426 (L1-ESPIRiT λ 0.002) | 35.757 (L1-ESPIRiT λ 0.004) |
| head SSIM | 0.9473 (L1-ESPIRiT λ 0.0025) | 0.9368 (L1-ESPIRiT λ 0.004 / 0.005) |
| head HFEN | 0.1034 (L1-ESPIRiT λ 0.0015) | 0.1000 (L1-ESPIRiT λ 0.004) |

Converged TV peaks at λ 0.0007 on both (36.162 / 35.581), joint TV at 0.001 (36.070 / 35.624); L1-ESPIRiT is the
strongest classical arm on every cell. The re-sweep raised the T1 bar by 0.092 dB and the FLAIR bar by 0.037.

## 2. Rules (pre-registered before the first candidate)

Screen at 1500 iterations against the parent at the same budget; keep = `min_margin` (the smaller of the two head-PSNR
margins) up by more than the seed floor, no other cell down by more than its floor, sharpness not more than 0.01 below
the parent; a PSNR gain paid with SSIM/HFEN is a *reshape*, not a keep. Promote kept knobs at 4000 iterations with
volumes against the 4000-iteration parent by the same rule; the promoted run becomes the parent and the runner bakes
it. Two clarifications recorded during the loop: a run whose cells all hold and at least one improves is a *Pareto
keep* even if `min_margin` is unchanged (the binding channel swaps); the promotion budget stays 4000, 6000 and the
two-seed average are reported as extra-compute rows. Seed floors: 0.006 / 0.004 dB at 1500 (P6 and P8 alike), 0.002 /
0.001 at 4000 (P10); SSIM and HFEN floors 0.0000–0.0002.

## 3. Result

Four promotions, all from the *shape* of the L1 wavelet prior and its per-channel strength; nothing else moved the
binding channel without a trade.

| parent | change | 4000 it, head PSNR T1 / FLAIR | vs previous | six-cell margins on the bar (PSNR / SSIM / HFEN, T1 · FLAIR) |
|---|---|---|---|---|
| P6 (user's mark) | — | 36.753 / 36.165 | — | +0.327 / +0.0007 / +0.0025 · +0.408 / +0.0006 / −0.0002 |
| P7 | wavelet on 2 levels (was 3) | 36.765 / 36.168 | +0.012 / +0.003 | +0.339 / +0.0009 / +0.0035 · +0.411 / +0.0007 / +0.0003 |
| P8 | db4 wavelet (was db2) | 36.819 / 36.189 | +0.054 / +0.021 | +0.394 / +0.0014 / +0.0036 · +0.432 / +0.0011 / +0.0011 |
| P9 | 1 level + FLAIR wavelet 0.085 | 36.853 / 36.234 | +0.034 / +0.045 | +0.427 / +0.0016 / +0.0084 · +0.477 / +0.0022 / +0.0039 |
| **P10** | **T1 wavelet 0.12, FLAIR 0.10** | **36.944 / 36.268** | +0.091 / +0.034 | **+0.518 / +0.0030 / +0.0087 · +0.511 / +0.0030 / +0.0042** |

**P10 = P6 with the wavelet prior changed to db4 on its finest level only, T1 strength 0.12, FLAIR 0.10** (edge-TV
λ_geo 0.10 / κ_q 0.85, 3.6M points, the repaired cold start, 4000 iterations unchanged). Against P6: **+0.191 / +0.103
dB** head PSNR, SSIM +0.0023 / +0.0024, HFEN better by 0.0062 / 0.0044, sharpness 0.858 / 0.823 vs 0.857 / 0.832.
Whole-volume PSNR 40.272 / 39.692 (P6 40.071 / 39.539). Per slice, against every canonical arm at its best λ: **first on
every full-brain slice in every direction on both contrasts** (T1 96/96, 121/121, 118/118; FLAIR 94/94, 120/120,
118/118). Registered in the topic table as `gs_joint_ar` ("MC-GS (AF12 loop, P10)"), with `gs_single_ar` and `gs_joint_ar_x2`.
Figures: `figures/t1flair_v3_af12_{T1,FLAIR}_row3planes_compact_174_110_47_headerr.png` (orange in-panel slice PSNR / SSIM / NMSE, head-only error maps; the layout the user chose on 2026-09-11), served at `http://localhost:8765/t1flair_v3_af12/`.

Extra-compute and reference rows (own rows, not the headline):

| row | head PSNR T1 / FLAIR | note |
|---|---|---|
| P10 seed 43 | 36.942 / 36.267 | 4000-iteration floor 0.002 / 0.001 |
| P10 ×2 seeds (magnitude mean, 2x compute) | 36.990 / 36.295 | `scripts/score_x2.py`; registered as `gs_joint_ar_x2` |
| P10 at 6000 iterations (1.5x) | 36.953 / 36.284 | budget still creeps |
| GS-single T1 (P10 recipe, T1 alone) | 36.897 / — | joint − single **+0.047** on T1 |
| GS-single FLAIR (P10 recipe, FLAIR alone) | — / 36.323 | joint − single **−0.055** on FLAIR: the single is better; coupling helps T1 and costs FLAIR on this subject, as at AF12 v3 (+0.080 / −0.074) and AF16 P6 (+0.037 / −0.003) |
| P10 + scale lr 0.005 / 0.008 | 36.955 / 36.316, 36.953 / 36.328 | marginal: T1 head HFEN slips 0.0002 / 0.0003 (2–3x its floor); not folded in |
| P10 with sym4 instead of db4 | 36.952 / 36.272 | equivalent within 0.008 / 0.004 |

## 4. What did not work (single knobs on the parent of the day; all at 1500 unless noted)

Reshapes (PSNR for SSIM/HFEN/sharpness): wavelet strength up on db2/3 levels (T1 0.14 / FLAIR 0.10), κ_q 0.90, λ_geo 0.07
(sharpens, loses PSNR) and 0.13 (smooths), FLAIR-only 0.085 on 3 levels, T1 0.14 under the 1-level prior (sharpness
−0.017), capacity 4.2M / 4.8M / 6.0M (**T1 falls as FLAIR rises at 4000 iterations, invisible at 1500**), scale lr 0.008
on P9 at 4000 (T1 −0.011). Losses: wavelet strength down (−0.15 / −0.11), 2.4M points (−0.15 / −0.25), w_floor 0.2,
k-space mid-band weight 0.8 (−0.16 / −0.06), init head fraction 0.03 (−0.18 / −0.19), db6 (below db4). Null: wavelet
4 levels, k-space centre weight 0.5 / 0.15, warm-up 250 / 700 (1000 is illegal at 1500 it), tv_eps 0.03, wavelet eps
1e-3, density lr 0.01 / 0.02. Plateaus: FLAIR wavelet 0.12 under 1 level, scale lr 0.012. Budget: 1500 → 2500 → 4000 →
6000 rises with shrinking steps (P6: 36.704 / 36.072 → 36.734 / 36.134 → 36.753 / 36.165 → 36.760 / 36.179).

Two lessons the ledger makes explicit: (1) **capacity and learning-rate knobs must be judged at the promotion budget** —
three times a 1500-iteration keep turned into a T1-for-FLAIR reshape at 4000; (2) **the prior's shape beat its
strength** — level count, wavelet family and per-channel strength under the finer prior gave +0.19 dB on T1 where the
strength alone was bracketed on both sides.

## 5. Caveats

- **Magnitude gauge.** Every table and figure uses | |rec| − |GT| |. On the strict complex gauge |rec − GT|
  (`scripts/complex_gauge.py`, `logs/complex_gauge.json`) L1-ESPIRiT is first on both contrasts (33.449 / 32.961) and
  P10 second (33.418 / 32.887): MC-GS loses 3.5 / 3.4 dB from magnitude to complex against 2.8–3.0 for the classical arms,
  its head phase agreement 0.9855 / 0.9821 against L1-ESPIRiT's 0.9862 / 0.9833. Same finding as the AF16 line. The lead
  is a magnitude-gauge lead.
- **Selected on this subject.** The recipe and the classical λ were both tuned on jiangyizhou's ground truth (fair, not
  held out). Nothing was run on huangyuwei or a second held-out subject in this loop.
- Sharpness of P10 (0.858 / 0.823) sits 0.009 under P9 on both channels, inside the 0.01 allowance at its edge; the T1
  wavelet strength is closed at 0.12 for that reason.
- FLAIR known zeros: as in every MC-GS run of these topics, the GS arms fit the measured shutter as zeros, the
  comparison arms use the mask only (documented asymmetry, unchanged).
- Whole-volume SSIM stays sixth (air-noise gauge), as documented in the topic READMEs.

## 6. Reproduce

```bash
cd /data0/congcong/code/GS/mcgs_af12_wt
# P10 at 4000 iterations with volumes (the runner bakes P10; the extra args are a no-op restatement)
SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh <EXP> <GPU> 4000
# equivalently, from the P6 runner of the AF16 line:
#   --set loss.wav_name=db4 --set loss.wav_levels=1 --set loss.lambda_wav=0.10 --set loss.lambda_wav_plain=0.12
# every row against the current bar, six cells
PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/autoresearch_jyz_af12_view.py
```
Topic refresh: `scripts/add_gs_arm.py` registers a run as an arm, `scripts/summarize.py` rebuilds the table,
`scripts/find_best_slice_v2.py --ours gs_joint_ar` the per-slice ranking, `scripts/fig_three_planes_compact.py
--contrast T1 --slices 174 110 47 --mask_error` the compact figure, `scripts/complex_gauge.py` the caveat table.
