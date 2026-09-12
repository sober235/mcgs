# jiangyizhou AF16, phase 3 — widening the MC-GS lead (session record, 2026-09-12 / 13)

Branch `autoresearch-jyz-af16-phase3` (worktree `/data0/congcong/code/GS/mcgs_af16_wt`), opened from the phase-2 tip
5348815 on the user's instruction to reopen the frozen AF16 line and "拉大" the gap to the comparison methods.
Design `docs/superpowers/specs/2026-09-12-af16-phase3-nonlocal-design.md`; programme, rules and full round log
`autoresearch_af16_p3.md`; ledger `autoresearch_af16_p3.tsv`; probe read-out
`docs/superpowers/records/2026-09-12-af16-probe-c-readout.md`.

## 1. What the user set

Arena jiangyizhou AF16 starting from P6; levers = model-structure changes, the AF12 line's findings (as parameter
values only), acquisition-side design; learned priors excluded; success = head PSNR at least **1.0 dB** above the
strongest comparison arm on both contrasts, all six head cells first, sharpness not below L1-ESPIRiT's. Cards: empty
cards only (a split with the AF12 phase-2 session gave this line cards 4 and 5).

Bar, L1-ESPIRiT at its best λ on this subject: head PSNR 35.209 / 34.503, SSIM 0.9334 / 0.9201, HFEN 0.1286 / 0.1210,
sharpness 0.849 / 0.789.

## 2. Result — the phase-3 recipe

`p3_r3_e0001_ptv003_full4k`, registered in the topic as **`gs_joint_p3`** ("MC-GS (phase 3)"); P6 keeps `gs_joint_ar`.

**Recipe = P6 + three changes**: the wavelet prior becomes **db4 on its finest level only, T1 0.12 / FLAIR 0.10**
(the AF12 line's shape, as a parameter value); **common-edge cross-gradient prior λ 0.001** (`--lambda_edge`);
**phase-only TV λ 0.03** (`--lambda_ptv`). Everything else is P6: repaired cold start (head init, 3-NN scales, ZF-sourced
edge-TV threshold), 3.6M points, edge-TV λ_geo 0.10 / κ_q 0.85 / ε 0.04, kspace_high 2.2, 4000 iterations.

| head-only, 4000 it | T1 | FLAIR |
|---|---|---|
| **MC-GS phase 3** | **35.706** | **34.928** |
| MC-GS P6 (phase 1) | 35.552 | 34.807 |
| L1-ESPIRiT (best other arm) | 35.209 | 34.503 |
| **margin, phase 3** | **+0.497** | **+0.425** |
| margin, P6 | +0.343 | +0.304 |
| head SSIM (phase 3 / P6 / L1) | 0.9396 / 0.9351 / 0.9334 | 0.9261 / 0.9203 / 0.9201 |
| head HFEN (phase 3 / P6 / L1) | 0.1139 / 0.1218 / 0.1286 | 0.1121 / 0.1188 / 0.1210 |
| sharpness (phase 3 / P6 / L1) | **0.855** / 0.830 / 0.849 | **0.804** / 0.788 / 0.789 |

- **Every gate of the pre-registered criterion is met except the size of the margin**: six head cells first, and — for
  the first time in the three phases — **sharpness at or above L1-ESPIRiT's on both contrasts** (P6 was below on both).
- Against P6: +0.154 / +0.121 dB with every structural cell better.
- Seed floor at this recipe, 4000 it: **0.001 / 0.000 dB** (`p3_r3_ptv003_s43_full4k`).
- GS-single at the same recipe (`gs_single_p3`): T1 35.558, FLAIR 34.886 → **MC-GS − GS-single = +0.148 / +0.042 dB**.
  In phase 1 that difference was +0.037 / −0.003: the joint arm now wins on both contrasts, because the common-edge
  term is itself a coupling term.
- Compact three-plane figures 174 / 110 / 47, error inside the head only, orange in-panel metrics:
  `figures/t1flair_v3_af16_{T1,FLAIR}_row3planes_compact_174_110_47_headerr_p3.png` — MC-GS is first on PSNR, SSIM and
  NMSE in **every plane of both contrasts** (at 174/110/47 P6 lost FLAIR coronal SSIM to CS-SENSE).

**The 1.0 dB target was not reached.** The margin roughly doubled (T1 +45 %, FLAIR +40 %) and every quality gate is now
met, but +1.0 dB over a comparison arm whose λ is itself tuned on this subject's ground truth is out of reach of the
lever classes this phase was allowed to use — see §4 and §6.

## 3. What moved it, in order of size

1. **The AF12 wavelet shape on P6** (db4, finest level only, T1 0.12 / FLAIR 0.10): +0.177 / +0.081 dB at 4000 it.
   Phase 2 had tested the same shape stacked on `p6ptv` and found no magnitude gain; on P6 it works — the phase-TV
   term at 0.05 was in the way.
2. **Common-edge cross-gradient prior, λ 0.001**: +0.056 / +0.051 dB at 1500 it, SSIM and HFEN better on both.
   Bracketed: 0.0005 lower, 0.002 level, 0.005 collapses (−0.13 dB).
3. **Phase-only TV, λ 0.03**: the sharpness lever. On the db4 base the trade curve is
   0.01 → sharp 0.839/0.787 at −0.005/+0.019 dB; 0.02 → 0.846/0.792 at −0.033/+0.013; 0.03 → 0.850/0.799;
   0.05 → 0.865/0.815 at −0.182/−0.069. 0.03 is the cheapest point clearing both sharpness gates.

## 4. Lever classes closed in this phase (do not retry)

- **Non-local self-similarity (RED with a lagged BM4D target)** — the phase's main bet, implemented as
  `paired_gs/losses/nonlocal_red.py` (config `loss.lambda_nl`, `nl_refresh`, `nl_start`, `nl_sigma_rel`, `nl_profile`,
  `nl_denoiser`, `nl_threads`, `nl_sharpen`; 14 tests; default 0 leaves the trainer bit-identical). Three settings —
  σ_rel 0.04 at λ 4, σ_rel 0.01 at λ 1, and σ_rel 0.02 at λ 1 with BM4D's transform-domain sharpening α 1.3 — all lose
  sharpness (up to −0.16 / −0.20) and never gain FLAIR PSNR. Mechanism, from the logs: at the refresh the weighted term
  matches the two data terms, within 50 iterations the fit snaps onto the denoised target (R_nl → 1e-3) and the
  k-space residual rises 15 %. The residual the denoiser removes is real high-frequency signal, which is what the
  2026-09-04 data audit said of this data. Cost: BM4D runs on the whole volume (the aliased ZF support fills both PE
  axes, so the head bounding box is the full grid), 156 s per contrast per refresh.
- **Adaptive density control** (never enabled on this line before): gradient score and evidence-energy score, split
  quantile 0.90, cap 5.0M, until iteration 1000 — both null (+0.003 / +0.003 and +0.002 / +0.013 with slightly worse
  HFEN). The alternating split/prune ended at +0.23 % points: at 3.6M the representation is not capacity-limited.
- **Vectorial TV** (λ 0.005 and 0.02): smooths — +0.02 dB of T1 at best, HFEN and sharpness worse. Same signature on
  the probe mask.
- **Contrast-private residual Gaussians** (0.3M): null (+0.005 / +0.015, nothing else moves).

## 5. The acquisition probe (full read-out in its own record)

A complementary FLAIR mask drawn **inside the vendor shutter** (2966 samples, all in-support, AF 11.51 against FLAIR's
own support against 12.49 today, 733 samples in common with the current mask) lifts **every** arm by about the same
amount: L1-ESPIRiT +0.368 dB, MC-GS +0.347. The FLAIR margin goes from +0.222 to +0.201, T1 does not move, and adding
vectorial TV to carry the new samples across contrasts moves T1 by +0.03. Both pre-registered thresholds (+0.3 dB of
margin, +0.15 dB of T1) are missed, so the full re-run of every arm on a two-mask design was not done. The in-shutter
mask is nevertheless a **better FLAIR acquisition for everyone** — 233 of the current mask's 2966 samples fall outside
the shutter and are wasted.

**By-product, never measured before:** the known-zero region (the shutter, used by the GS arms only) is worth
**−0.013 dB** of FLAIR head PSNR on the current mask — i.e. nothing — but **+0.225 dB** on the in-shutter mask, where
no sampled line tells the model the corners are empty. The "GS arms use the shutter, the others do not" asymmetry
carried in the phase-1 and phase-2 records is therefore not a material advantage on the mask the results use.
Enabling it needs `data.<x>.known_zero_support` (new in this phase, `known_zero_from_support` + 5 tests) when the
sampling lies inside the shutter.

## 6. Caveats

- One subject; the MC-GS knobs and the classical λ are both selected on its ground truth — fair, not held out.
- Magnitude gauge. The complex-gauge ranking was not re-measured for the phase-3 recipe; phase 2 showed phase TV lifts
  it, and this recipe carries phase TV at 0.03 (phase 2 used 0.05), so it is likely better than P6 there, but that is
  an expectation, not a measurement.
- Nominal AF 16 (T1 15.19x, FLAIR 12.49x of its acquired k-space).
- Whole-volume SSIM stays sixth (the air-noise gauge), as in every phase.
- The `p3_r0_control` row (the reference for the "db4 base minus control" delta) ran before the non-local module was
  merged; `p3_r5_control_postmerge` re-runs plain P6 after the merge to close that gap. The module is guarded by
  `lambda_nl > 0` and an import-isolation test.

## 7. Open decisions (for the user)

1. **Which recipe is quoted externally.** P6 is the current answer (CLAUDE.md, user decision 2026-09-12). The phase-3
   recipe is better on every head cell and on sharpness, and its figures exist with the `_p3` suffix.
2. **The 1.0 dB target.** Not reached and, under "no learned priors, same data, same mask", not reachable by anything
   this phase found. The options are: accept the doubled margin; re-state the target as a margin ratio; or allow a
   learned prior (and then give the comparison arms one too, which changes what the comparison means).
3. A held-out subject at AF16 with the recipe and λ frozen — still never run, from phase 1 onward.
4. Whether to adopt the in-shutter FLAIR mask for future acquisitions (it helps every method, not the margin).
