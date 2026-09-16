# Record — MC-GS tuned on the Gaussian variable-density random mask (jiangyizhou T1+FLAIR, nominal AF 16), 2026-09-16

Branch `autoresearch-gauss-af16`, worktree `mcgs_gauss_wt` (cut from `jyz-af16-diffmask` 889448a). Program file with the
full round log: `autoresearch_gauss.md`; ledger `autoresearch_gauss.tsv`; every run under `results_arc/g_*`.

## 1. Question, bar and target

The diff-mask study (2026-09-15/16) ran the frozen AF16 phase-3 MC-GS recipe on a 2-D Gaussian i.i.d. random mask with
the same 2964-point budget as the house vd-Poisson mask (T1 15.20x, FLAIR 15.43x, labelled AF 16 nominal). MC-GS stayed
first on all six head cells but every arm lost 0.6–1 dB, and the margin over its own single-contrast arm shrank to
+0.138 / +0.023 dB. The user asked (2026-09-16) for a Karpathy-style autoresearch loop on a new branch to bring MC-GS
**0.5 dB above every comparison arm**, GS-single included, GS-single staying at the frozen recipe (not re-tuned).

Bar (`T1FLAIR_AF16_gauss_v3_jiangyizhou/logs/bar_gauss.json`, `scripts/bar_gauss.py`): the strongest comparison arm on
every one of the six head cells is GS-single at the frozen recipe.

| head cell | T1 bar | FLAIR bar | strongest non-GS arm |
|---|---|---|---|
| PSNR | 34.771 | 34.294 | ZS-SSDU 34.375 / 33.972 |
| SSIM | 0.9305 | 0.9202 | ZS-SSDU 0.9273 / 0.9196 |
| HFEN | 0.1412 | 0.1295 | L1-SPIRiT 0.1653 / 0.1360 |

Targets: T1 >= 35.271, FLAIR >= 34.794. Parent (the frozen recipe on this mask, 4000 it): 34.909 / 34.317.
Forecast written before the first run: the earlier loops of this project gained +0.10 to +0.19 dB each; 0.5 dB over a
frozen GS-single is beyond that record.

## 2. Arena, isolation, protocol

Same data, maps, mask, gauge and scorer as the gauss topic (read-only). Runner `autoresearch_gauss.sh` bakes the parent
(recipe block byte-identical to `run_diffmask_mcgs.sh`); candidates differ only by extra args. Screens at 1500 it,
promotions at 4000 it with volumes, one trainer per empty card, each run holding the shared card lock
(`Cardiac_FreeRunning_MCGS/work/.gpulock.<card>`). Reference rows first: control 34.875 / 34.248 at 1500 it, seed 43
34.865 / 34.240 → **1500-it floor 0.010 / 0.008 dB** (SSIM 0.0002, HFEN 0.0002, sharpness 0.001).
Reading rule (pre-registered, `scripts/gauss_judge.py`): keep = both contrasts up beyond their floor with no cell down
beyond its floor and sharpness not down by more than 0.010; keep-one / reshape / keep-marginal / discard as in the
program file. Losers are not retried; a bracket is queued only for a keep.

## 3. What the screens found (1500 it, deltas over the control, T1 / FLAIR)

Closed — the parent value is the optimum, both directions lose:
- wavelet dose ×0.7 −0.037 / −0.015, ×1.5 −0.038 / −0.043, ×2 −0.130 / −0.124 → 0.12 / 0.10 stays
- edge-TV lambda_geo 0.07 −0.028 / −0.035, 0.14 −0.027 / −0.013 → 0.10 stays
- kspace_high 1.5 −0.247 / −0.284, 3.0 −0.217 / −0.174 → 2.2 stays (a cliff both ways: this weight acts as a step size)
- phase TV 0.01 reshape (T1 +0.040, FLAIR −0.011, structure worse), 0.05 −0.081 / −0.036 → 0.03 stays
- common-edge 0.002 −0.012 / −0.006 → 0.001 stays; edge quantile 0.75 −0.002 / −0.036, 0.90 −0.024 / −0.002 → 0.85 stays
- wavelet 2 levels −0.097 / −0.020; kappa from the render −0.011 / −0.003; tv_eps 0.02 marginal (smooths)
- kspace_center 0.5 null; prewhitening FLAIR-only −0.044, both −0.040 / −0.042
- lr schedule: gamma 0.999 −0.025 / −0.049; 0.9998 +0.009 / +0.018 at 1500 but −0.019 / −0.009 at 4000 (promotion)
- position_lr ×2 −0.006 / −0.012, density_lr ×2 −0.015 / −0.029, scale_lr ×0.5 −0.039 / −0.067
- warm-up 300 / 700: bit-identical to the control — `shared_warmup_iters` is parsed and then `del`eted in the v2 trainer
- directional-TV coupling 0.01: FLAIR +0.028 with sharpness −0.017 (keep-marginal, not stacked)

Kept (each adds on the parent), and how they stack:
- rotation_lr 0.0003 → 0.008: +0.033 / +0.049 (the AF12 phase-3 lever transfers)
- scale_lr 0.002 → 0.004: +0.026 / +0.047; with rot008: **+0.049 / +0.083** (the two lr levers add)
- capacity 4.8M +0.017 / +0.053, 6.0M +0.007 / +0.073 — but on top of either lr lever capacity adds nothing
  (rot008+n60 = n60, scl004+n60 = n60, rs+n48 +0.031 / +0.084 < rs) → 3.6M stays
- edge-TV w_floor 0.10 → 0.05 +0.013 / +0.017, 0.02 +0.016 / +0.023, 0.0 +0.016 / +0.024 (flat below 0.02); rs + w_floor 0.05 **+0.061 / +0.100**
- vector (isotropic) edge-TV: alone a reshape (T1 +0.030, FLAIR −0.012, but SSIM / HFEN / sharpness better on both);
  on rs **+0.091 / +0.077**; rs + w_floor + vector (**rswt**) **+0.097 / +0.085** with the best structure of the loop
- init_mode intensity: keep-marginal alone (T1 +0.046, FLAIR SSIM slips); on rswt T1 +0.108 / FLAIR +0.055 — moves gain
  from FLAIR to T1, so not promoted (FLAIR is the binding side)

## 4. Promotions (4000 it, volumes; parent at 4000 = 34.909 / 34.317, SSIM 0.9333 / 0.9210, HFEN 0.1344 / 0.1278, sharp 0.796 / 0.759)

| recipe | T1 | FLAIR | SSIM T1 / FL | HFEN T1 / FL | sharp | vs parent | margin over the bar |
|---|---|---|---|---|---|---|---|
| rs (rot008 + scl004) | 34.916 | 34.345 | 0.9333 / 0.9213 | 0.1345 / 0.1278 | 0.795 / 0.758 | +0.007 / +0.028 | +0.145 / +0.051 |
| rsw (+ w_floor 0.05) | 34.923 | 34.362 | 0.9337 / 0.9218 | 0.1339 / 0.1274 | 0.806 / 0.769 | +0.014 / +0.045 | +0.152 / +0.068 |
| rsw + gamma 0.9998 | 34.904 | 34.353 | 0.9334 / 0.9217 | 0.1344 / 0.1279 | 0.805 / 0.767 | −0.005 / +0.036 | +0.133 / +0.059 |
| **rswt (+ vector edge-TV)** | **34.976** | 34.353 | **0.9349 / 0.9225** | **0.1305 / 0.1255** | **0.821 / 0.790** | **+0.067 / +0.036** | **+0.205 / +0.059** |

The +0.049 / +0.083 of the lr levers at 1500 it collapses to +0.007 / +0.028 at 4000 it: those levers buy convergence
speed, not a better end point. The prior-shape changes (w_floor, vector edge-TV) are what survives promotion.
All four promotions are first on all six head cells (`six_first` = 1).

## 5. Verdict

**The 0.5 dB target is not met, and the loop closes the parameter space.** Two operating points came out of it, both
first on all six head cells, both far under the target:
- **rsw** = the frozen phase-3 recipe + rotation_lr 0.008 + scale_lr 0.004 + edge-TV w_floor 0.05: 34.923 / 34.362,
  **+0.152 / +0.068 dB over the frozen GS-single** — the formal winner of the pre-registered objective (max of the
  smaller margin: 0.068 against rswt's 0.059; the 4000-it seed floor is 0.001 / 0.001, so the 0.009 is real).
- **rswt** = rsw + vector (isotropic) edge-TV: 34.976 / 34.353, **+0.205 / +0.059** — T1 +0.053 over rsw, and the best
  SSIM / HFEN / sharpness of any MC-GS on this mask (0.9349 / 0.9225, 0.1305 / 0.1255, 0.821 / 0.790); FLAIR PSNR
  0.009 under rsw. Against the strongest non-GS arm (ZS-SSDU) it is +0.601 / +0.381.
The parent was +0.138 / +0.023. Which point is quoted is the user's call (the previous lines left the same choice to
them); the record-only singles of section 6 were run for rswt.
Every prior dose (wavelet, edge-TV, phase TV, common-edge, kspace weights), every shape lever except the two kept, the
lr schedule, capacity, warm-up (inert), prewhitening and the cold start were bracketed or closed with evidence.
What is left is not a parameter: a 0.5 dB gap over the same model run single-contrast needs a coupling mechanism that
the current code does not have (the coupling terms present — common-edge, directional TV, vectorial TV — are all at or
past their optimum here).

## 6. Record-only rows (rule 4) and the 4000-it floor

- `g_r2_rsw_s43_full4k` (rsw, seed 43): 34.925 / 34.363 → **4000-it seed floor 0.001 / 0.001 dB**, SSIM / HFEN /
  sharpness 0.000 — every promotion delta above is far outside it. `g_r2_rswt_s43_full4k` (rswt, seed 43): PENDING.
PENDING at the time of writing (results appended below when they land):
- `g_r3_rswt_singleT1_full4k`, `g_r3_rswt_singleFLAIR_full4k`: rswt's own single-contrast arms at the same recipe —
  the fair coupling number, not part of the bar (the user's decision keeps GS-single frozen in the main table).

## 7. Caveats

One subject; recipe selected on it; magnitude gauge; nominal AF 16 (T1 15.20x); the bar's GS-single is the frozen
recipe by the user's decision (the same-recipe single is reported in section 6 for transparency); classical arms were
lambda-swept on this mask by the diff-mask study, MC-GS was tuned here — the comparison favours MC-GS by construction
of the request.

## 8. Incidents

- 15:02: the tmux server was killed by mistake (pid 1777714 looked like a stray client); five running screens died and
  were re-run from scratch; the scheduler now launches every run detached (`setsid nohup`). ~25 GPU-minutes lost,
  no result affected.
- Two screens crashed on my own inputs (warmup 800 ≥ 0.5 × iters; init_mode "importance" is not a trainer option) and
  were re-queued with legal values.
- The memory watchdog killed one wait loop at 14:30 (another session's 100-GB PDF renders); trainers unaffected.
- 16:45: at the user's request (relayed by session claude-71) the loop was cut to cards 3 / 4 / 7.

## 9. Reproduce

```
cd /data0/congcong/code/GS/mcgs_gauss_wt
PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/bar_gauss.py                    # the bar (reads the gauss topic)
PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/gauss_judge.py                  # every screen against the control
cut -f1,7,8,9,10,11,25 autoresearch_gauss.tsv | grep g_r2_                 # the promotions
SAVE_VOLUMES=1 ./autoresearch_gauss.sh g_r2_rswt_full4k <card> 4000 --set gaussian.rotation_lr=0.008 --set gaussian.scale_lr=0.004 --edge_tv_wfloor 0.05 --tv_vector
```
