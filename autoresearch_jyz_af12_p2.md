# autoresearch, phase 2 — MC-GS joint on jiangyizhou T1 + FLAIR in design B: widen the lead further

_Opened 2026-09-12 15:59. Same Karpathy-style loop as `autoresearch_jyz_af12.md` (one change → fixed budget → one
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

## Decision rules (pre-registered 2026-09-12 15:59 (commit e9c2f40), before the first phase-2 candidate ran)

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

### Round 0 — 2026-09-12 15:59, three chains launched (commit e9c2f40; lever A = 30f8513)
- Card 3, `scripts/p2_chain_a.sh 3`: `p2_P10_i1500` (parent, seed 42) → `p2_geo_T113_F007` → `p2_tvvec` → `p2_kfa_off` → `p2_vtv002`.
- Card 5, `scripts/p2_chain_b.sh 5`: `p2_P10_i1500_s43` (floor) → `p2_geo_T107_F013` → `p2_gradnorm10` → `p2_rotlr` → `p2_edge002`.
- Card 2, `scripts/p2_chain_c.sh 2`: `p2_bundle_slr005_i6000` (rule 7, volumes) → `p2_khigh20` → `p2_P10_s43_full4k`.
- Cards 0/1 carry the user's foundation-model jobs, card 4 another user's job, 6/7 another user's light jobs (the
  AF16 phase-3 loop's smoke on card 5 had ended before launch).
- 16:03 bit-identity check of lever A's default path passed: `p2_P10_i1500` iteration 1 prints
  `loss=1.2281e-01 dc_p=7.1673e-02 dc_c=4.9690e-02 R_geo=6.6043e-03 R_wav=7.8832e-03`, the same digits as
  `ar9_P10_flair_af11_full4k` (old code, same config and seed). Reader for the rules: `scripts/p2_read.py`
  (validated on the P10 loop's rows: reproduces its keep / reshape / null / loses reads).

### Round 0 read — 16:33 (references at 1500 iterations, design B)
Parent P10 at 1500 (`p2_P10_i1500`, card 3, 29 min): head PSNR **36.885 / 36.520**, margins +0.459 / +0.762 (T1 binding),
SSIM 0.9502 / 0.9430 (+0.0029 / +0.0062), HFEN margins +0.0083 / +0.0083, sharp 0.856 / 0.827. T1 reproduces the AF12
loop's P10@1500 (36.885) to the digit, as it must (same mask); FLAIR at 1500 sits 0.117 dB under its 4000-iteration
value (36.637). Seed 43 (`p2_P10_i1500_s43`, card 5, 34 min): 36.875 / 36.515.
**Seed floor at the screening budget: PSNR 0.010 / 0.005 dB; SSIM 0.0001 / 0.0001; HFEN 0.0001 / 0.0000; sharp 0.000 / 0.001.**
Rule-1 threshold for a keep: `min_margin` above +0.459 by more than 0.010 (i.e. T1 head PSNR above 36.895 with FLAIR
still non-binding), no other cell down by more than its floor, sharpness not below 0.846 / 0.817. The T1 floor is
larger than the P10 loop's (0.006): the reads below use the measured 0.010.

### Round 1 read, part 1 — 17:01: lever A (per-contrast edge-TV strength) closed in both directions
Against the parent at 1500 (36.885 / 36.520, T1 floor 0.010):
- `p2_geo_T113_F007` (T1 0.13 / FLAIR 0.07; card 3, 29 min): 36.875 / 36.496 — T1 −0.010 (at the floor), FLAIR −0.024,
  T1 sharpness 0.831 (−0.025, gate fail), FLAIR sharper 0.858. **loses.**
- `p2_geo_T107_F013` (T1 0.07 / FLAIR 0.13; card 5, 28 min): 36.828 / 36.486 — T1 −0.057, FLAIR −0.033, T1 sharpness
  0.885 (+0.029), FLAIR 0.801. **loses.**
Reading: 0.10 is the PSNR optimum of *each* channel, not a compromise between them; moving one channel's TV also moves
the other channel's PSNR down (shared geometry: T1 0.13 costs FLAIR 0.024 dB). Per rule "if both directions fail,
lever A is closed": per-contrast κ_q is not pursued. The sharpness numbers say what the knob does trade: T1 sharpness
follows T1's TV strength (0.831 / 0.856 / 0.885 for 0.13 / 0.10 / 0.07) at a PSNR cost in both directions.

### Round 1 read, part 2 — 17:28
- `p2_tvvec` (isotropic TV; card 3, 27 min): 36.855 / 36.504 — T1 −0.030, FLAIR −0.016; SSIM +0.0001 / +0.0002, HFEN
  +0.0014 / +0.0010, sharper (0.869 / 0.844). **loses** on the read; a sharpness/HFEN-for-PSNR trade, the same shape as
  weaker TV (`p2_geo_T107_F013`, `ar2_geo007`). Anisotropic (per-axis) TV stays.
- `p2_gradnorm10` (gradient clipping 10 instead of 1.0; card 5, 26 min): 36.885 / 36.520 — +0.0004 / +0.0005, every cell
  within its floor. **null**: clipping at 1.0 does not bind at this working point. Settled.
- Running: `p2_kfa_off` (card 3), `p2_rotlr` (card 5); the bundle at 5600/6000 on card 2 (rule 7 read when it lands,
  against `ar9_P10_flair_af11_full4k` 36.945 / 36.637 with the AF12 4000-iteration floor 0.002 / 0.001 dB until
  `p2_P10_s43_full4k` lands).

### Round 2, part 1 — pre-registered 17:35, before any of these ran: the wavelet prior's shape, one step further
Every P10 promotion came from the shape of the wavelet prior (levels 3 → 2 → 1, db2 → db4, per-contrast strength).
Two shape knobs were not expressible until commit d22be4b (`loss.wav_level_weights`, finest first; `loss.wav_band_weights`
per 3-D detail sub-band; default None keeps the old code path — pinned by `tests/test_wavelet_weights.py`, and the
first default-path run after the commit is checked by its iteration-1 loss). Single knobs on P10, 1500 iterations,
rule 1 as above; chain `scripts/p2_chain_d.sh`, launched on the first card free of any trainer:

| exp | knob | args | why |
|---|---|---|---|
| `p2_wav_lev2_w05` | 2 levels, coarse level at half strength | `--set loss.wav_levels=2 --set 'loss.wav_level_weights=[1.0, 0.5]'` | the loop found 1 level > 2 levels at equal weight; a half-strength second level sits between the two |
| `p2_wav_ddd05` | all-detail (diagonal) band at half strength | `--set 'loss.wav_band_weights={ddd: 0.5}'` | the seven sub-bands carry aliasing and anatomy in different proportions; L1-ESPIRiT weights them equally, we need not |
| `p2_wav_ddd20` | diagonal band at double strength | `--set 'loss.wav_band_weights={ddd: 2.0}'` | the other direction |
| `p2_wav_ax05` | the three single-axis bands at half strength | `--set 'loss.wav_band_weights={daa: 0.5, ada: 0.5, aad: 0.5}'` | the complementary split (edges along one axis) |

If one of the band directions keeps, the next step in that direction (0.25 / 4.0, or the two-axis bands) is a round-2
part-2 candidate; if `p2_wav_lev2_w05` keeps, level weights 0.25 / 0.75 bracket it.

### Rule-7 read — 17:38: the bundle (P10 + scale lr 0.005 at 6000 iterations, design B, volumes saved)
`p2_bundle_slr005_i6000` (card 2, 99 min): head PSNR **36.958 / 36.699** against the parent's 36.945 / 36.637 —
**+0.013 / +0.062 dB**; SSIM 0.9501 / 0.9446 (−0.0001 / +0.0006), HFEN 0.0948 / 0.0903 (−0.0002 / +0.0004 in margin
terms), sharpness 0.858 / 0.830 (unchanged). Floor in use (rule 2, the AF12 seed pair `ar5_best_C10_T112_full4k` vs
`ar5_P10_s43_full4k`): PSNR 0.002 / 0.001, SSIM 0.0000 / 0.0000, HFEN 0.0001 / 0.0001.
**By the letter of rule 1 this is a `reshape` (T1 SSIM −0.0001 and T1 HFEN −0.0002 fall beyond floors that are
essentially zero), the same reading the P10 loop gave scale lr 0.005 alone (keep-marginal, deferred).** The user
asked for the bundle to be judged by the rules, so it is not folded in. Recorded before the design-B 4000-iteration
floor exists: when `p2_P10_s43_full4k` lands (queued on card 2), the bundle is re-read once against that floor as
rule 2 prescribes, and that verdict is final; a floor of 0.0001 on T1 SSIM or 0.0002 on T1 HFEN would turn it into a
keep. The FLAIR gain (+0.062) is the sum of the two marginal items measured at AF12 (+0.016 for 6000 iterations,
+0.048 for scale lr 0.005); T1's (+0.013) is sub-additive (+0.009, +0.011).

### Round 1 read, part 3 — 17:56: first keep of phase 2
- `p2_kfa_off` (uniform k-space band weights instead of 0.3 / 1.0 / 2.2; card 3, 27 min): 35.232 / 34.417 — **−1.65 / −2.10 dB**,
  every cell down, sharpness 0.689 / 0.624. **loses** by a mile: the high-band emphasis is load-bearing, not a
  fine-tuning. Settled (with `kspace_high` 2.0 / 2.2 / 3.0 and the centre / mid variants of the P10 loop, the band
  weights are bracketed on this subject).
- `p2_rotlr` (rotation lr 0.001, 3.3x the default 0.0003; card 5, 26 min): **36.907 / 36.545 — +0.023 / +0.025 dB**,
  SSIM +0.0001 / +0.0002, HFEN +0.0001 / +0.0002, sharpness 0.856 / 0.828 (unchanged). **KEEP** (min_margin +0.023 >
  floor 0.010, every other cell up). The Gaussians' orientations were learning too slowly; this is the optimiser
  family the P10 loop found scale lr in.
- Still running: `p2_khigh20` (card 2), `p2_vtv002` (card 3), `p2_edge002` (card 5).

### Round 2, part 2 — pre-registered 18:00, before any of these ran: the optimiser family around the keep
| exp | what | args | budget | read against |
|---|---|---|---|---|
| `p2_rotlr_full4k` | promotion of the keep (rule 2), volumes | `--set gaussian.rotation_lr=0.001` | 4000 | `ar9_P10_flair_af11_full4k`, floor = `p2_P10_s43_full4k` when it lands (AF12 floor until then) |
| `p2_rotlr002` | rotation lr 0.002 (6.7x) | `--set gaussian.rotation_lr=0.002` | 1500 | parent P10@1500 (rule 1); best of 0.0005 / 0.001 / 0.002 by min_margin |
| `p2_rotlr0005` | rotation lr 0.0005 (1.7x) | `--set gaussian.rotation_lr=0.0005` | 1500 | same |
| `p2_rotlr_slr005` | rotation lr 0.001 + scale lr 0.005 (the two optimiser levers together) | `--set gaussian.rotation_lr=0.001 --set gaussian.scale_lr=0.005` | 1500 | `p2_rotlr` as parent (rule 1) |

If a bracket value beats 0.001 by more than the floor at 1500, it is promoted instead of (or after) 0.001; the
promoted rotation lr then becomes part of the parent (P11) by a runner commit. Chain D now carries these three
screens first, then the wavelet-shape screens; card 3 runs the promotion first (wrapper re-armed), card 5 starts
chain D after chain B, and card 3 joins chain D after the promotion (the runner's overwrite refusal shares the work).

### Round 1 read, part 4 — 18:23: round 1 complete
- `p2_khigh20` (high band 2.0 instead of 2.2; card 2, 27 min): 36.912 / 36.463 — T1 +0.028 but FLAIR −0.057, FLAIR SSIM
  −0.0004, HFEN −0.0005 / −0.0004, sharpness 0.832 / 0.800 (−0.024 / −0.027). **reshape** (T1 bought with FLAIR and
  sharpness). With 2.2 / 3.0 the high-band weight is bracketed at 2.2 on this subject.
- `p2_vtv002` (vectorial joint TV 0.02; card 3, 27 min): 36.794 / 36.347 — −0.091 / −0.173, every cell down, sharpness
  0.788 / 0.758. **loses.** `p2_edge002` (common-edge cross-gradient prior 0.02; card 5, 27 min): 35.557 / 35.168 —
  −1.33 / −1.35 dB. **loses.** Both explicit cross-contrast couplings hurt under the wavelet prior, as λ_dtv did at
  AF16: the shared geometry is the only coupling that pays.

**Round 1 tally (12 candidates + the bundle):** one keep (rotation lr 0.001, +0.023 / +0.025), one T1-for-FLAIR
reshape (high band 2.0), one null (gradient clipping), nine losses; the bundle +0.013 / +0.062 at 6000 iterations,
reshape by the letter at the near-zero AF12 floor (re-read pending). Every frozen structural setting of the working
point (band weights, edge-TV strength per channel, TV shape, coupling terms) sits at a local optimum on this
subject; what moved is the optimiser (rotation lr, and scale lr in the P10 loop). Round 2 = the optimiser bracket
and the wavelet-shape screens, as pre-registered.
- 18:23 promotion 1 (`p2_rotlr_full4k`, card 3) and the design-B 4000-iteration floor (`p2_P10_s43_full4k`, card 2,
  since 18:05) are running; chain D starts on card 5 behind chain B.
- 18:24 card collision with the parallel AF16 phase-3 loop (other session, `mcgs_af16_wt`): within a minute of chains A
  and B ending it launched `p3_r0_db4l1` on card 3 and `p3_r1_nl16_s04` on card 5, next to this loop's `p2_rotlr_full4k`
  (card 3) and `p2_rotlr002` (card 5, one minute old). Rule "never a card that already carries a trainer of any loop":
  `p2_rotlr002` was stopped (wrapper E's process group; partial outputs removed, no ledger row) and card 5 ceded; chain D
  now runs on cards 2 and 3 only (behind chain C and behind the promotion). The promotion stays on card 3, shared with
  the other loop's run since both started at 18:23 (2–3x slower; card 3 hung twice under two trainers on 09-11 — watch
  its progress). A card-split proposal (this loop 2 + 3, that loop 4 + 5) was sent to the peer sessions.
- 18:35 the AF16 phase-3 session (claude-64) agreed: it stopped `p3_r0_db4l1` on card 3 and restricted its scheduler to
  cards 4 and 5; this loop keeps 2 and 3. Its correction accepted: its card-3 launch (18:17:53) had found the card
  empty (chain A had just exited) and this loop's wrapper landed beside it a minute later — the wrappers only waited
  for their own chain. The runner now waits, before every launch, while any `arc/cycle.py` trainer sits on the
  target card (one trainer per card, any loop's).

### Design-B 4000-iteration floor and the bundle's final read — 19:11
`p2_P10_s43_full4k` (card 2, 65 min): 36.942 / 36.633 against the parent's 36.945 / 36.637 — **floor at 4000 in
design B: PSNR 0.003 / 0.004 dB, SSIM 0.0000 / 0.0000, HFEN 0.0002 / 0.0002**. Its ledger row first carried
`crash` and its training log was clobbered (see the incident below); the row was relabelled `floor-p2-4000`, its
numbers come from the intact score json.
**Bundle re-read (rule 7, final):** T1 +0.013 / FLAIR +0.062 dB, T1 HFEN −0.0002 now inside its floor, T1 SSIM
−0.0001 against a measured floor of 0.0000 → **`reshape(dsT1)` by the letter of rule 1; not folded in.** The decision
rests on a zero seed floor for T1 SSIM (seeds 42 and 43 agree to four decimals) — the §18 defect of a
near-deterministic repeat making a tolerance vanish; recorded rather than retuned. The user can overrule by hand
(P10 + scale lr 0.005 at 6000 iterations = +0.013 / +0.062 dB for a fourth-decimal SSIM change on T1).

### Incident — 18:35 runner edited in place while two instances of it were executing (fixed 19:14)
The any-trainer guard was written into `autoresearch_jyz_af12_p2.sh` in place (same inode) at 18:35 while the bash
instances launched at 18:05 (`p2_P10_s43_full4k`) and 18:23 (`p2_rotlr_full4k`) were still running it. Bash reads a
script by byte offset, so the 18:05 instance resumed inside the new text when its trainer finished at 19:11: it ran a
garbage line that truncated the run's log (63 bytes, `line 40: exp: command not found`) and passed rc=127 to the row
script (status `crash`). The score json and volumes are intact. Fix: the file was restored in place to its 18:23
content (commit e9c2f40's) so the still-running 18:23 instance resumes correctly; the guarded runner lives under a
new name, `autoresearch_jyz_af12_p2g.sh`, which chain D now calls; chain D on card 2 (45 s old) was stopped and
relaunched with it. Rule from here: **never edit a launched script in place — new behaviour goes in a new file.**

### Promotion 1 — 19:28: rotation lr 0.001 at 4000 iterations — **promoted, P11**
`p2_rotlr_full4k` (card 3, 70 min, the first 5 min shared with the other loop's run; volumes saved): head PSNR
**36.956 / 36.654** against the parent's 36.945 / 36.637 — **+0.011 / +0.017 dB**, SSIM +0.0000 / +0.0002, HFEN
−0.0000 / +0.0001 (inside the 0.0002 floor), sharpness 0.858 / 0.830 unchanged; min_margin +0.011 > floor 0.003.
**KEEP by rule 2.** The 1500-iteration gain (+0.023 / +0.025) shrank at the full budget but did not turn into a trade.
**P11 = P10 + `gaussian.rotation_lr=0.001`** (default was 0.0003) is the parent from here: margins over the bar
+0.530 / +0.897 dB, SSIM +0.0030 / +0.0074, HFEN +0.0088 / +0.0094. Baked into a NEW runner,
`autoresearch_jyz_af12_p2h.sh` (P11 + the any-trainer guard); `p2g` stays for chain D, whose screens were
pre-registered as single knobs on P10 and are read against `p2_P10_i1500`. Combinations of a chain-D keep with P11 go
straight to 4000 iterations through the P11 runner (rule 2), read against `p2_rotlr_full4k` with the design-B floor.
Card 3 went on to chain D (`p2_rotlr0005`; `p2_rotlr002` refused there because card 2 already runs it).

### Round 2 read, part 1 — 19:41: rotation lr 0.002 beats 0.001 at the screening budget
`p2_rotlr002` (card 2, 27 min): **36.923 / 36.564 — +0.038 / +0.045** over P10@1500 (0.001: +0.023 / +0.025), every
other cell up, sharpness 0.857 / 0.828. **KEEP**, and above 0.001 by 0.015 (> floor 0.010): the bracket is still
rising. Pre-registered next step (round 2, part 3, before any of these ran): `p2_rotlr004` (0.004, 13x the default)
at 1500; promotion `p2_rotlr002_full4k` at 4000 with volumes, read against **P11** (`p2_rotlr_full4k`, 36.956 /
36.654) with the design-B floor — if it clears, P12 = P10 + rotation lr 0.002 and 0.004 follows the same route if
its screen beats 0.002's by more than the floor. Chain D's two bash instances were stopped (their running runners
left alone: `p2_rotlr0005` on card 3, `p2_rotlr_slr005` on card 2) and chain E (`scripts/p2_chain_e.sh`: 0.004
screen, 0.002 promotion, then the four wavelet-shape screens) is queued behind each of those runners
(`p2_after_pid_then_e.sh`, kill -0 on the runner pid). Nothing launched in place; nothing edited in place.
- 19:55 `p2_rotlr0005` (rotation lr 0.0005; card 3, 27 min): 36.893 / 36.529 — +0.008 / +0.009, inside the T1 floor:
  **null**. The bracket so far, head PSNR gain over P10 at 1500 (T1 / FLAIR): 0.0005 +0.008 / +0.009; 0.001 +0.023 /
  +0.025; 0.002 +0.038 / +0.045 — monotone, no cell paying. `p2_rotlr004` runs on card 3, `p2_rotlr_slr005` on card 2.

### Round 2 read, part 2 — 20:07: the two optimiser levers stack
`p2_rotlr_slr005` (rotation lr 0.001 + scale lr 0.005; card 2, 27 min): **36.952 / 36.642 — +0.067 / +0.123** over
P10@1500 and **+0.044 / +0.097 over rotation lr 0.001 alone**, SSIM +0.0002 / +0.0011, HFEN +0.0003 / +0.0008,
sharpness 0.858 / 0.830. **KEEP.** Scale lr 0.005 alone was keep-marginal in the P10 loop (T1 HFEN cost at 4000);
on top of the faster rotations it adds cleanly at 1500. The optimiser grid is the live vein of this phase.

### Round 2, part 4 — pre-registered 20:10, before any of these ran: the optimiser grid
1. `p2_rotlr002_slr005` (rotation 0.002 + scale 0.005, 1500; chain F on card 3 behind `p2_rotlr004`): read against
   `p2_rotlr002` and `p2_rotlr_slr005` — is the stacking additive at 0.002 as well?
2. When `p2_rotlr004` is read: if it keeps and beats 0.002 by more than the floor, `p2_rotlr004_slr005` (1500) and
   `p2_rotlr008` (1500) follow; if not, the rotation bracket closes at 0.002.
3. Promotion at 4000 with volumes of the best optimiser row at 1500 (by min_margin over the rotation-lr × scale-lr
   grid), through the P10-parent runner with both values explicit, read against **P11** (`p2_rotlr_full4k`) with the
   design-B floor; a keep becomes **P12**. `p2_rotlr002_full4k` (chain E, card 2, started 20:08) stays as the
   rotation-only reference at 4000 and is read against P11 the same way.
4. The wavelet-shape screens (round 2 part 1) run after the grid on whichever card is free; a keep there is combined
   with the standing parent at 4000.
Chain E's bash on card 3 was stopped (its `p2_rotlr004` runner left alone); chain F is queued behind that runner.
- 20:21 `p2_rotlr004` (rotation lr 0.004; card 3, 26 min): **36.937 / 36.584 — +0.053 / +0.064**, every cell up,
  sharpness 0.857 / 0.829. **KEEP**, above 0.002 by 0.015 (> floor). Bracket: 0.0005 +0.008, 0.001 +0.023, 0.002
  +0.038, 0.004 +0.053 (T1); still monotone. Per item 2: `p2_rotlr008` and `p2_rotlr004_slr005` queued as chain G on
  card 3 behind chain F's `p2_rotlr002_slr005`.
- 20:47 `p2_rotlr002_slr005` (rotation 0.002 + scale lr 0.005; card 3, 26 min): **36.962 / 36.656 — +0.077 / +0.137**
  over P10@1500; +0.039 / +0.092 over rotation 0.002 alone (additive again), +0.010 / +0.014 over rotation 0.001 +
  scale lr (at the T1 floor). Every cell up, sharpness 0.858 / 0.830. **KEEP.** Grid at 1500 (T1 / FLAIR gain over
  P10): rotation only 0.001 +0.023 / +0.025, 0.002 +0.038 / +0.045, 0.004 +0.053 / +0.064; with scale lr 0.005:
  0.001 +0.067 / +0.123, 0.002 +0.077 / +0.137. Running: `p2_rotlr008` (card 3), then `p2_rotlr004_slr005`;
  `p2_rotlr002_full4k` at 2650/4000 on card 2.

### Promotion 2 — 21:12: rotation lr 0.002 at 4000 iterations — **promoted, P12** (rotation-only line)
`p2_rotlr002_full4k` (card 2, 63 min, volumes): **36.962 / 36.665** — +0.006 / +0.011 over P11 (floor 0.003 / 0.004),
+0.017 / +0.028 over P10; SSIM +0.0000 / +0.0001, HFEN −0.0000 / +0.0001 (inside the floor), sharpness 0.858 / 0.831.
**KEEP by rule 2.** **P12 = P10 + `gaussian.rotation_lr=0.002`** (explicit `--set` on the guarded P10 runner; no new
runner file until the optimiser grid settles). The 4000-iteration gain is about half the 1500-iteration one at every
rotation lr so far (0.001: +0.023 → +0.011; 0.002: +0.038 → +0.017 on T1) — the lever is real but shrinks with the
budget. Card 2 moved on to chain E's wavelet screens (`p2_wav_lev2_w05` running); card 3 finishes `p2_rotlr008`, then
`p2_rotlr004_slr005`. Bracket stop rule (stated now): the rotation bracket ends at the first value that does not beat
the previous one by more than the floor at 1500; the grid's best row is then promoted at 4000 against P12.
- 21:14 `p2_rotlr008` (rotation lr 0.008; card 3, 26 min): **36.948 / 36.600 — +0.063 / +0.081**, every cell up,
  sharpness 0.857 / 0.829. **KEEP**, above 0.004 by 0.011 (floor 0.010 — the increments are shrinking: +0.015,
  +0.015, +0.011 per doubling on T1).

### Round 2, part 5 — pre-registered 21:20, before any of these ran
Chain H on card 3 behind `p2_rotlr004_slr005`: `p2_rotlr016` (rotation lr 0.016, 1500) and `p2_rotlr008_slr005`
(rotation 0.008 + scale lr 0.005, 1500); then **the promotion of the best grid row is chosen mechanically**
(`scripts/p2_promote_best_grid.sh`: among the rotation-lr × scale-lr screens with status `keep` at 1500, the largest
min_margin over the bar; run at 4000 with volumes through the guarded P10 runner with both values explicit, named
`p2_best_rot<rl>_slr<sl>_full4k`), read by rule 2 against **P12** (`p2_rotlr002_full4k`) with the design-B floor. If
0.016 does not beat 0.008 by more than the floor, the rotation bracket ends at 0.008 (stop rule above); the selector
does not need that read to run. Card 2 continues chain E's wavelet-shape screens meanwhile.
- 21:22 queued on card 2 behind chain E (`p2_after_pid_then_seed43.sh`): the seed-43 repeat of whichever recipe the
  selector promotes (`scripts/p2_seed43_of_best.sh`, values parsed from the promotion's name, 4000 iterations with
  volumes) — the floor of the promoted recipe and one half of its two-seed row (rule 4). GS-single arms for the final
  recipe (rule 5) are queued by hand once the promotion is read.

### Round 2 read, part 3 — 21:39
- `p2_rotlr004_slr005` (rotation 0.004 + scale lr 0.005; card 3, 26 min): **36.968 / 36.669 — +0.083 / +0.149** over
  P10@1500, +0.006 / +0.013 over rotation 0.002 + scale lr (inside the T1 floor: the stacked rows saturate on T1),
  every cell up, sharpness 0.858 / 0.831. **KEEP**, best grid row so far.
- `p2_wav_lev2_w05` (2 levels, coarse level at half strength; card 2, 27 min): 36.858 / 36.514 — −0.027 / −0.005,
  HFEN −0.0024 / −0.0014, sharpness 0.850 / 0.822. **loses.** A half-weight second level still hurts: the finest level
  alone stays the prior's shape. Running: `p2_wav_ddd05` (card 2), chain H on card 3.
- 21:40 chain H started on card 3 (`p2_rotlr016` running). Note for the record: its wrapper was still inside a 60-s
  poll when its runner exited, a second copy of chain H was launched by hand, and the runner's any-trainer guard held
  the duplicate's first launch until the duplicate was stopped (no double trainer, no duplicate row). The kill -0
  waiters are also exposed to PID reuse on this host (PIDs wrapped past 4194304 tonight); from here chains are launched
  when a card is seen free, with the guard as the safety net.
- 22:05 `p2_wav_ddd05` (diagonal sub-band at half strength; card 2, 26 min): 36.863 / 36.521 — T1 −0.022, FLAIR +0.001,
  T1 sharper (0.859), SSIM −0.0003 / −0.0001. **loses**: weakening the all-detail band costs T1 PSNR. `p2_wav_ddd20`
  (double) running on card 2; `p2_rotlr016` still on card 3.
- 22:07 `p2_rotlr016` (rotation lr 0.016; card 3, 26 min): 36.957 / 36.613 — +0.072 / +0.094 over P10@1500 (a keep on
  its own), but only +0.009 over 0.008 on T1, inside the floor: **the rotation bracket ends at 0.008 by the stop
  rule.** Rotation-only curve at 1500 (T1 gain): 0.0005 +0.008, 0.001 +0.023, 0.002 +0.038, 0.004 +0.053, 0.008 +0.063,
  0.016 +0.072 — concave, flattening. `p2_rotlr008_slr005` running on card 3; the selector then promotes the best
  stacked row (0.004 + scale lr is +0.083 so far).
