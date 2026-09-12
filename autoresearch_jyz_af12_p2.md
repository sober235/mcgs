# autoresearch, phase 2 — MC-GS joint on jiangyizhou T1 + FLAIR in design B: widen the lead further

_Opened 2026-09-12 16:10. Same Karpathy-style loop as `autoresearch_jyz_af12.md` (one change → fixed budget → one
number → keep or discard → record → repeat). Runner `autoresearch_jyz_af12_p2.sh` (P10 baked in, default config =
design B), ledger `autoresearch_jyz_af12.tsv` (continued; phase-2 rows carry the `p2_` prefix and `cfg=…flair_af11`
in the note), branch `autoresearch-jyz-af12-phase2` (from `autoresearch-jyz-af12` at 2887187), worktree
`/data0/congcong/code/GS/mcgs_af12_wt`. A parallel loop (another session, `mcgs_af16_wt`, branch
`autoresearch-jyz-af16-phase3`) runs on this host at the same time; its results are external to this loop and are
never used for selection here. Cards: one trainer per card, never a card that already carries a trainer of any loop;
at launch cards 2, 3 and 5 were free (0/1 the user's other jobs, 4 and 6/7 another user's)._

## Objective (user decisions 2026-09-12, this session)

1. The current AF 12 result is fixed as two figures of P10 (CLAUDE.md, "Current AF 12 result"): T1 with every arm at
   AF 12, FLAIR in **design B** = MC-GS on the AF 11 vd-Poisson mask (`data/mask_af11_3d.h5`), every comparison arm
   at AF 12. **Design B is the setting this loop tunes in** (the user's choice over tuning at AF 12 and re-running
   design B at the end): config `configs/t1flair_v3_af12_hyw_flair_af11.yaml`, T1 mask unchanged.
2. Goal, in the user's words: raise MC-GS's reconstruction quality further above the comparison methods, or widen
   the gap to them. Lever scope: configuration knobs plus small code levers (the per-contrast edge-TV strength,
   `loss.lambda_geo_plain` / `loss.lambda_geo_contrast`, this branch's first commit, unit-tested, default path
   bit-identical). The marginal items the P10 loop left to the user (scale lr 0.005, 6000 iterations) are judged
   together as ONE candidate under the rules below, not folded in by decree.

**The number the loop reads** is unchanged: `min_margin = min(T1 head PSNR − bar_T1, FLAIR head PSNR − bar_FLAIR)`
with the six head cells (PSNR, SSIM, HFEN per contrast) recorded per row. **The bar is the P10 loop's final bar**
(rule 3 there, sharpened 2026-09-11 20:10): per cell the best value over every non-MC-GS row of the topic table and
every λ of the three swept families, all at AF 12 — by the definition of design B the comparison arms stay at AF 12,
so the FLAIR bar does not move. Bar (T1 / FLAIR): PSNR 36.426 / 35.757 (L1-ESPIRiT), SSIM 0.9473 / 0.9368, HFEN
0.1034 / 0.1000 (the row script computes the margins from the topic's logs).

**Where we start** — P10 in design B, 4000 iterations (`ar9_P10_flair_af11_full4k`, side experiment of the P10 loop):

| head cell | T1 | margin | FLAIR | margin |
|---|---|---|---|---|
| PSNR | 36.945 | +0.519 | 36.637 | +0.880 |
| SSIM | 0.9503 | +0.0030 | 0.9439 | +0.0072 |
| HFEN | 0.0947 | +0.0088 | 0.0906 | +0.0093 |

Sharpness 0.858 / 0.830. **T1 is the binding channel** in design B (+0.519 against FLAIR's +0.880), so a keep has
to move T1; FLAIR-only gains show up in the other five cells and are recorded as such.

## Decision rules (pre-registered 2026-09-12 16:10, before the first phase-2 candidate ran)

Rules 1–6 of `autoresearch_jyz_af12.md` apply verbatim, read against the phase-2 reference rows:
1. *Keep* at the screening budget (1500 iterations): `min_margin` above the parent's at the same budget
   (`p2_P10_i1500`) by more than the seed floor at that budget (`p2_P10_i1500` vs `p2_P10_i1500_s43`), AND none of
   the other five cells below the parent's by more than its own floor, AND sharpness of neither contrast more than
   0.01 below the parent's. PSNR bought with SSIM or HFEN = `reshape`, not a keep.
2. *Promote*: kept knobs (singly, then the combination) at 4000 iterations with `SAVE_VOLUMES=1`, rule 1 against the
   4000-iteration parent (`ar9_P10_flair_af11_full4k`) with the 4000-iteration floor (`p2_P10_s43_full4k`; until it
   lands, the P10 loop's 4000-iteration floor at AF 12: PSNR 0.002 / 0.001 dB, SSIM 0.0001, HFEN 0.0001). The
   promoted run becomes the new parent; the runner's baked recipe changes only by a commit that says so.
3. The bar does not move in this phase (design B keeps the comparison arms at AF 12; nothing new is swept).
4. Two-seed average = a separate row at 2x compute, never folded into the recipe.
5. GS-single arms run for the final recipe only.
6. Whole-volume SSIM reported, never optimised.
7. *The bundle* (`p2_bundle_slr005_i6000` = P10 + `gaussian.scale_lr=0.005` at 6000 iterations, volumes saved) is
   judged by rule 2 against the 4000-iteration parent. If kept it becomes the parent, and every later promotion runs
   at 6000 iterations (1.5x compute, accepted by the user). If not, it is a row.
8. A keep at 1500 that becomes a reshape at 4000 is recorded as such and not folded in (the P10 loop's lesson:
   capacity and scale lr traded T1 for FLAIR only at the full budget).

## Fixed — never touched, never used for fitting or selection

Data, both masks (`mask_af12_3d.h5` for T1, `mask_af11_3d.h5` for FLAIR), ESPIRiT maps, ground truth, scorer,
gauge (magnitude), the competitors' numbers. FLAIR known zeros as before. Code changes go in with a unit test and
leave the default path bit-identical (checked for lever A by the iteration-1 loss of `p2_P10_i1500` against
`ar9_P10_flair_af11_full4k`, same config and seed, old code: `loss=1.2281e-01 dc_p=7.1673e-02 dc_c=4.9690e-02
R_geo=6.6043e-03 R_wav=7.8832e-03`).

## Already settled — do not re-run expecting a gain

Everything in the P10 program's "Already settled" list, plus the P10 loop itself: wavelet strength up/down (T1 0.14
fails the sharpness gate; 0.07 loses), levels 4/2/1 (1 is the recipe), db2/db4/db6/sym4 (sym4 ≡ db4), shared λ_geo
0.07/0.13, κ_q 0.80/0.90, points 2.4/4.2/4.8M (4.8M+ reshapes at 4000), 2500/6000 iterations alone, tv_eps 0.03,
w_floor 0.20, warm-up 250/700, k-space centre 0.5/0.15 and mid 0.8, density lr 0.01/0.02, scale lr 0.003/0.008/0.012
(0.008 reshapes at 4000), wavelet eps, seed averaging (2x row). AF16 phase 2 (other loop): phase-only TV 0.05 costs
0.24 / 0.07 dB of magnitude PSNR (helps the complex gauge only) — not a lever for this objective. `loss.loss_type`
other than l2 is a placeholder in the trainer (not implemented) — no data-term family to test.

## Allowed and queued — round 1, single knobs on P10 in design B, 1500 iterations

Never varied on this subject, each with a reason (A = the code lever; B–E configuration only; F = rule 7):

| exp | knob | args | why |
|---|---|---|---|
| `p2_P10_i1500` | parent at the screening budget | (none) | reference; iteration-1 check of lever A's default path |
| `p2_P10_i1500_s43` | seed floor at 1500 | `--set training.seed=43` | reference |
| `p2_P10_s43_full4k` | seed floor at 4000 in design B | `--set training.seed=43` (4000 it) | reference for rule 2 / 7 |
| `p2_geo_T113_F007` | A: per-contrast edge-TV T1 0.13 / FLAIR 0.07 | `--set loss.lambda_geo_plain=0.13 --set loss.lambda_geo_contrast=0.07` | all four P10 promotions were per-contrast prior shape; shared λ_geo failed in both directions, consistent with the channels wanting opposite moves |
| `p2_geo_T107_F013` | A: T1 0.07 / FLAIR 0.13 | `--set loss.lambda_geo_plain=0.07 --set loss.lambda_geo_contrast=0.13` | the other direction |
| `p2_tvvec` | B: isotropic (vector) TV | `--tv_vector` | TV shape never varied; may help the SSIM cells |
| `p2_kfa_off` | C: uniform k-space band weights | `--set loss.kspace_freq_adaptive=false` | 0.3 / 1.0 / 2.2 were tuned on the old bundle; only near-neighbours tried on jyz |
| `p2_khigh20` | C: high band 2.0 | `--kspace_high 2.0` | the old main line's winner (2.0 over 2.2); down never tried on jyz |
| `p2_gradnorm10` | D: gradient clipping 10 (default 1.0) | `--set training.max_grad_norm=10.0` | never varied |
| `p2_rotlr` | D: rotation lr 0.001 (default 0.0003) | `--set gaussian.rotation_lr=0.001` | never varied; anisotropic Gaussians' orientation |
| `p2_vtv002` | E: vectorial joint TV 0.02 | `--lambda_vtv 0.02` | cross-contrast coupling never tried under the wavelet prior (dtv reversed sign there) |
| `p2_edge002` | E: common-edge prior 0.02 | `--lambda_edge 0.02` | same |
| `p2_bundle_slr005_i6000` | F: scale lr 0.005 + 6000 it (rule 7) | `--set gaussian.scale_lr=0.005`, 6000 it, volumes | the two marginal items together, judged once at the promotion budget |

Round 2 (pre-registered in outline, exact args when round 1 is read): combinations of round-1 keeps; if lever A keeps
in one direction, per-contrast κ_q in the same direction (`--set loss.edge_tv_kappa_q_plain/contrast`, a second
small code lever with the same test discipline); if both directions fail, lever A is closed. Guided cross-contrast
edge weights (T1's edge map weighting FLAIR's TV) only if round 1 shows the channels want different smoothing.

## Budget and protocol

Screening 1500 iterations (~26 min on a free A800 at 3.6M points, P10 loop timing), promotion 4000 (~62 min), the
bundle 6000 (~95 min). Chains as bash files under `scripts/p2_*.sh`, launched detached (`setsid nohup … &`); the
runner refuses to overwrite, so a chain can be relaunched safely. Waiting: poll `results_arc/<exp>_score.json`.
Every ledger row carries the commit and the exact args; the cfg is in the note.

## Round log
