# autoresearch phase 3 — jiangyizhou T1+FLAIR at AF16: widen the MC-GS lead to >= 1.0 dB

Started 2026-09-12 (user: "mc-gs的重建质量和对比方法进一步拉大, 并且mc-gs领先"; arena, levers, criterion and route
decided in chat, written down in `docs/superpowers/specs/2026-09-12-af16-phase3-nonlocal-design.md` — read that first).

## Objective

MC-GS joint at least 1.0 dB above the strongest comparison arm on head-only PSNR (magnitude gauge) on BOTH contrasts,
all six head cells (PSNR / SSIM / HFEN x T1 / FLAIR) still first, sharpness not below L1-ESPIRiT's (0.849 / 0.789).
Bar (`Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/logs/bar_complex_gauge.json`): L1-ESPIRiT 35.209 / 34.503 →
targets T1 >= 36.21, FLAIR >= 35.50. Ledger objective: maximise `min_margin_mag`, subject to the six-cell rule.

## Parent

P6 (`ar14_n36_v2tv_full4k`, topic `gs_joint_ar`): 4000 it 35.552 / 34.807 (margin +0.343 / +0.304), SSIM 0.9351 / 0.9203,
HFEN 0.1218 / 0.1188, sharp 0.830 / 0.788. At 1500 it (phase 1, `ar13_n36_final_v2tv`): 35.515 / 34.724.
Baked into `autoresearch_af16_p3.sh` exactly as in phase 2. Candidates differ from the parent only by their extra args.

## Isolation

Same as phase 2: `p3_` names only, no CONFIG path override (`SINGLE=T1|FLAIR`, `MASK_VARIANT=flair_c` select by name),
rows to `autoresearch_af16_p3.tsv`, nothing from the AF12 line's code is merged (its findings enter as parameter values).

## Protocol

Screen 1500 it, promote 4000 it with `SAVE_VOLUMES=1`; one trainer per card (cards 2, 3, 5 main line, 4 probe);
`setsid nohup`; progress from `results_arc/<exp>/<exp>/joint_train.log`. Seed floor <= 0.003 dB at 1500, 0.000 / 0.002 at 4000.
Rule: `keep` = no head cell worse than the parent by more than the floor, at least one better by more than it, sharpness
not down by more than 0.005; `keep-marginal` = PSNR up but a structural cell or sharpness slips (promoted only if the
4000-it run repairs it). Losers are not retried.

## Smoke (2026-09-12 15:50, GPU 5)

`p3_smoke_vtv` = 50 it, `--lambda_vtv 0.02`: runner, scorer and ledger path work. Raw term sizes at iter 50 (the log
prints unweighted terms): dc_p 2.0e-2, dc_c 2.1e-2, R_geo 1.0e-2 (x lambda_geo 0.10 = 1.0e-3), R_vtv 8.0e-2 (x 0.02 =
1.6e-3), R_wav 6.0e-2. So lambda_vtv 0.02 already outweighs the edge-TV term; Round 0 scans 0.005 and 0.02.
R_edge is a normalised direction misfit in [0, 1]; Round 0 scans lambda_edge 0.001 and 0.005.

## Round 0 (existing terms, no code) — queued 2026-09-12 16:10, 1500 it, empty cards only

| exp | extra args | reading |
|---|---|---|
| p3_r0_control | — | P6 on this runner at 1500 (reference, floor) |
| p3_r0_vtv0005 | `--lambda_vtv 0.005` | vectorial TV, gentle |
| p3_r0_vtv002 | `--lambda_vtv 0.02` | vectorial TV, strong |
| p3_r0_edge0001 | `--lambda_edge 0.001` | common-edge direction prior, gentle |
| p3_r0_edge0005 | `--lambda_edge 0.005` | common-edge, strong |
| p3_r0_priv03 | `--set gaussian.num_private_points=300000` | FLAIR-private residual Gaussians |
| p3_r0_db4l1 | `--set loss.wav_name=db4 --set loss.wav_levels=1 --set loss.lambda_wav_plain=0.12 --set loss.lambda_wav=0.10` | the AF12 wavelet shape on P6 |
| p3_r0_ptv002 | `--lambda_ptv 0.02` | phase TV at 1/5 of phase 2's weight, for sharpness |

Cards (user rule 2026-09-12 16:05, after a collision with the AF12 phase-2 chains that took cards 2/3/5 twelve seconds before this line's first launch): **empty cards only** — `scripts/p3_sched.sh scripts/p3_queue.txt` (tmux session `p3_sched`) launches the next queue item on a card with no compute process, no live `p2_chain_*.sh` claim and no trainer of this line; card 4 is shared with the probe. **18:27: card split agreed with the AF12 phase-2 session (claude-ad): AF12 keeps cards 2 and 3, this line keeps 4 and 5 (`ALLOWED="4 5"`); `p3_r0_db4l1`, which had started on card 3 at 18:17 beside their 18:23 launch, was stopped and re-queued.** The three screens that had started on cards 2/3/5 were stopped and their outputs removed; the queue holds the eight items above in priority order.

## Round 1 onward

See the spec, sections 4 and 6: the non-local RED prior (`paired_gs/losses/nonlocal_red.py`, Track A in worktree
`mcgs_af16_nl_wt`) on P6, then stacked on the Round-0 keeps. The acquisition probe (spec section 5, Track B, card 4)
reports into `docs/superpowers/records/2026-09-12-af16-probe-c-readout.md`.

## Probe re-plan (16:20) — the known-zero confound

The first probe run (`p3_probe_c_P6`, in-shutter FLAIR mask, launched 16:14 on card 4 by Track B) printed
`[physics:contrast] ... ellipse=None frac_z_used=0.0000 zero_lines=0/2966`: with every sample inside the shutter the
known-zero estimator has no zero sampled line to learn the shutter from, so that run fits FLAIR WITHOUT the 24 % of the
PE plane that the GS arms treat as measured zeros on the current mask (`frac_z_used=0.2423`, 233 zero lines). Its number
is therefore "complementary mask minus the known-zero advantage", not the probe the spec asked for. Fix (commit e0e1058):
`data.<x>.known_zero_support: <2-D support h5>` bypasses the estimator with the shutter as protocol knowledge
(`known_zero_from_support`, tests in `tests/test_known_zero.py`); the topic's `data/flair_support_2d.h5` (34139 lines,
`scripts/prep_04d_flair_support.py`) is set in `configs/t1flair_v3_af16_jyz_flair_c.yaml`. Queue now: `p3_r0_nokz`
(P6 on the current mask with FLAIR `known_zero=false`: measures the known-zero advantage itself, never measured before),
`p3_probe_c_P6_kz` and `p3_probe_c_P6_kz_vtv0005` (the probe with the shutter given). Read-out, updated:
FLAIR margin of `p3_probe_c_P6_kz` over the best probe-mask L1-ESPIRiT, against P6@1500 − L1 on the current mask
(+0.221); `p3_probe_c_P6` − `p3_r0_nokz` isolates the mask effect without Z; `p3_r0_control` − `p3_r0_nokz` is the
worth of Z on the current mask. The vtv run answers the coupling question (T1 change).

## Track A landed (16:29) — the non-local RED prior is in the trainer (merge 1419aaf)

`paired_gs/losses/nonlocal_red.py` + wiring in `JointTrainer.train` (after the wavelet term; `loss/R_nl` in the
components, `R_nl=` unweighted in the iteration line, one `[nl] iter= contrast= sigma= crop= bm4d=…s` line per refresh).
Keys under `loss`: `lambda_nl` 0 (`lambda_nl_plain` / `lambda_nl_contrast` default to it), `nl_refresh` 500, `nl_start`
500, `nl_sigma_rel` 0.04, `nl_profile` np|lc, `nl_denoiser` bm4d|fake_identity|fake_blur, `nl_threads` 32. Tests: 13 new,
suite 200 passed (`--ignore=tests/smoke_test.py --ignore=tests/test_score_edge.py`). Two facts that change the plan:
- **The head bounding box is the whole volume** here: the union ZF support (0.05 · P99.9 rule) is aliased edge to edge on
  both PE axes, so BM4D runs on all 11.49M voxels: **156 s per contrast per refresh** (profile np; lc 136 s) at host load
  ~106/112. A 1500-it screen with K0 = K = 500 pays 2 refreshes × 2 contrasts ≈ 10 min; a 4000-it promotion ≈ 36 min.
  Tightening the box (morphological opening of the support) is a later optimisation, not done.
- **The term is small**: R_nl(zf | BM4D(zf)) ≈ 1.0e-3 (normalised), against dc ≈ 2e-2 and λ·R_wav ≈ 6e-3 at iter 50.
  The spec's grid {0.5, 1, 2} would be almost inert, so Round 1 screens λ_nl ∈ {1, 4, 16} at σ_rel 0.04 first and reads
  `loss/R_nl` in the log before extending the grid.

## Round 1 (queued 16:30, interleaved with Round 0 in `scripts/p3_queue.txt`)

| exp | extra args | reading |
|---|---|---|
| p3_r1_nl4_s04 | `--set loss.lambda_nl=4 --set loss.nl_sigma_rel=0.04 --set loss.nl_refresh=500 --set loss.nl_start=500` | middle of the grid, first signal |
| p3_r1_nl16_s04 | λ_nl 16, same σ / K | strong |
| p3_r1_nl1_s04 | λ_nl 1, same σ / K | weak |

Queue order now: control, nokz, probe_kz, nl4, db4l1, nl16, probe_kz_vtv, nl1, vtv0005, ptv002, edge0001, priv03,
vtv002, edge0005.

## Round 1b (queued 18:45) — smaller σ, the pull kept gentle

The first non-local screen over-smoothed (above). Round 1b lowers the denoiser strength to the reconstruction's own
error level and below: σ_rel ∈ {0.02, 0.01} × λ_nl ∈ {1, 4}, K0 = K = 500 (`p3_r1b_nl{1,4}_s{02,01}`). Read-out: the
sharpness gate and head HFEN first, then PSNR. If σ_rel 0.01–0.02 still costs sharpness, the next lever class is the
refresh policy (K 250, K0 1000: a target computed from a better-fitted volume, refreshed more often) and the denoiser's
own sharpening option (`BM4DProfile.set_sharpen`), not a larger λ.

## Round 2 (queued 20:35) — the remaining terms on the db4/1-level base

Base B = P6 + `wav_name=db4, wav_levels=1, lambda_wav_plain=0.12, lambda_wav=0.10` (1500-it reference `p3_r0_db4l1` = 35.683 / 34.792,
SSIM 0.9376 / 0.9222, HFEN 0.1171 / 0.1166, sharp 0.830 / 0.779). Screens `p3_r2_*`: phase TV 0.02 and 0.05 (sharpness lever first),
vectorial TV 0.005 / 0.02, common-edge 0.001 / 0.005, FLAIR-private residual 0.3M. Read-out as before; the sharpness gate is now the
binding constraint (B is 0.019 / 0.010 below L1-ESPIRiT's 0.849 / 0.789).

## Round 3 (queued 21:30) — combinations of the Round-2 keeps

`p3_r3_e0001_ptv001` and `p3_r3_e0001_ptv002` = base B + common-edge 0.001 + phase TV 0.01 / 0.02 (1500 it). If common-edge 0.005
beats 0.001, the same pair is repeated with it. The better combination is promoted at 4000 it with volumes, then seed repeat and
the GS-single arms.

## Round log

Head-only, 1500 it unless noted; bar L1-ESPIRiT 35.209 / 34.503 (current mask). Six cells = PSNR / SSIM / HFEN per contrast.

| time | exp | T1 | FLAIR | SSIM T1 / FL | HFEN T1 / FL | sharp | reading |
|---|---|---|---|---|---|---|---|
| 16:43 | p3_probe_c_P6 (in-shutter FLAIR mask, NO known zeros) | 35.526 | 34.847 | 0.9353 / 0.9208 | 0.1217 / 0.1234 | 0.829 / 0.810 | FLAIR +0.12 over the control, T1 +0.01 (no coupling transfer); L1-ESPIRiT on the same mask 34.871 (λ 0.003; 0.002 → 34.677) → FLAIR margin −0.02 vs +0.22 on the current mask. Without Z the mask helps L1 (+0.37) more than MC-GS. Await `_kz`. |
| 17:18 | p3_r0_control (P6 on this runner) | 35.515 | 34.725 | 0.9353 / 0.9198 | 0.1218 / 0.1192 | 0.829 / 0.785 | reproduces phase 1's 1500-it P6 (35.515 / 34.724) to 0.001 dB — the reference row |
| 17:45 | p3_r0_nokz (current mask, FLAIR `known_zero=false`) | 35.525 | 34.738 | 0.9353 / 0.9184 | 0.1218 / 0.1222 | 0.829 / 0.782 | **the known-zero region is worth ~0 dB of FLAIR PSNR** (−0.013, inside noise), +0.0014 SSIM and 0.003 of HFEN with Z. The 'GS-only known zeros' asymmetry noted in the phase-1/2 records is now measured: negligible. |
| 18:12 | p3_probe_c_P6_kz (in-shutter FLAIR mask, shutter given explicitly) | 35.513 | **35.072** | 0.9352 / 0.9246 | 0.1217 / 0.1128 | 0.830 / 0.813 | FLAIR +0.347 over the control (Z is worth +0.225 on THIS mask, against ~0 on the current one: with every sample inside the shutter nothing else tells the model the corners are empty). L1-ESPIRiT on the same mask 34.871 → FLAIR margin **+0.201 vs +0.222** on the current mask; T1 unchanged (35.513). **Probe read-out (pre-registered rule, spec §5): the margin does not grow (−0.02) and T1 does not move → route 2 is not worth a full re-run.** The in-shutter complementary mask is a better FLAIR acquisition for every arm (MC-GS +0.35, L1-ESPIRiT +0.37, both with better SSIM/HFEN/sharpness), not a differential lever. `p3_probe_c_P6_kz_vtv0005` (still queued) remains as the coupling check; unless it moves T1 by ≥ 0.15 the probe is closed. |
| 18:43 | p3_r1_nl4_s04 (λ_nl 4, σ_rel 0.04, K0 = K = 500) | 35.343 | 34.189 | 0.9283 / 0.9047 | 0.1294 / 0.1294 | 0.670 / 0.594 | **discard — over-smoothing**: −0.17 / −0.54 dB, every structural cell worse, sharpness collapses. Dynamics: at the 500-it refresh the weighted term (4 × 5.8e-3 = 0.023) matches the two DC terms; within 50 iterations the fit snaps onto the BM4D target (R_nl → 1e-3) while dc rises 0.0170 → 0.0196 (+15 %). σ = 0.04 · P99(head) ≈ 0.031 is ~1.4× the reconstruction's own error RMS, so the target has lost real detail. `p3_r1_nl16_s04` (a foregone loss in the same direction) was stopped at iteration ~300 and removed; `p3_r1_nl1_s04` dropped from the queue. |
| 19:14 | p3_r0_db4l1 (P6 + the AF12 wavelet shape: db4, 1 level, T1 0.12 / FLAIR 0.10) | **35.683** | **34.792** | 0.9376 / 0.9222 | 0.1171 / 0.1166 | 0.830 / 0.779 | +0.168 / +0.067 dB over the control, SSIM and HFEN better on both contrasts; FLAIR sharpness −0.006 (just past the 0.005 tolerance) → **keep-marginal**, promotion `p3_r0_db4l1_full4k` queued first (4000 it, volumes). Note: stacked on p6ptv in phase 2 this shape gave no magnitude gain; on P6 it does — the phase-TV term was in the way. |
| 19:20 | p3_r1b_nl1_s01 (λ_nl 1, σ_rel 0.01) | 35.510 | 34.602 | 0.9374 / 0.9204 | 0.1222 / 0.1198 | 0.799 / 0.749 | **discard**: T1 flat, FLAIR −0.12, HFEN slightly worse, sharpness −0.03 / −0.04 even at the gentlest setting. With nl4_s04 this brackets the class: the lagged-BM4D pull only smooths here, at no PSNR gain — the residual it removes is real signal, as the 09-04 data audit said of the AF12 residual. `p3_r1b_nl1_s02` (launched 19:20 by the scheduler before the queue edit) was stopped at ~iter 200 and removed; nl4_s02 / nl4_s01 dropped. Last try of the class: `p3_r1c_nl1_s02_sh13` = the same term with BM4D's transform-domain sharpening α 1.3 (`loss.nl_sharpen`, commit below), a detail-boosting rather than smoothing target. If it also loses sharpness, the non-local RED lever is closed. |
| 19:58 | p3_r1c_nl1_s02_sh13 (λ_nl 1, σ_rel 0.02, BM4D sharpen α 1.3) | 35.564 | 34.641 | 0.9382 / 0.9209 | 0.1220 / 0.1194 | 0.788 / 0.736 | **discard — the non-local RED class is closed**: T1 +0.05 but FLAIR −0.08, HFEN flat, sharpness −0.04 / −0.05 even with the sharpened target. Three settings (σ 4 %, 1 %, 2 % sharpened) all lose sharpness and never gain FLAIR; the prior has no information to add on this data. Kept in the trainer (default off) for the record. |
| 20:33 | p3_r0_db4l1_full4k (**4000 it**, volumes) | **35.729** | **34.888** | 0.9376 / 0.9231 | 0.1169 / 0.1159 | 0.831 / 0.781 | vs P6@4000 (35.552 / 34.807): +0.177 / +0.081 dB, SSIM and HFEN better on both; FLAIR sharpness 0.781 vs 0.788 — the −0.007 slip is NOT repaired at 4000, so by the pre-registered rule it stays **keep-marginal**: not promoted on its own, but adopted as the base of Round 2 (every remaining Round-0 term is now screened on top of it, one round instead of two), and the final recipe must pass the sharpness gate anyway (target ≥ 0.849 / 0.789 — no recipe so far does; phase TV is the known sharpness lever). Margin over L1-ESPIRiT +0.520 / +0.385. |
| 20:40 | p3_probe_c_P6_kz_vtv0005 (probe mask, shutter given, + vectorial TV 0.005) | 35.544 | 35.100 | 0.9352 / 0.9247 | 0.1238 / 0.1146 | 0.807 / 0.788 | vs `p3_probe_c_P6_kz`: T1 +0.031, FLAIR +0.028, HFEN worse on both, sharpness −0.023 / −0.025. The coupling term does not carry the complementary FLAIR samples into T1 (+0.03 ≪ the +0.15 threshold). **Probe closed on both counts** (record `docs/superpowers/records/2026-09-12-af16-probe-c-readout.md`). |
| 20:52 | p3_r2_ptv002 (base B + phase TV 0.02) | 35.650 | 34.805 | 0.9387 / 0.9240 | 0.1153 / 0.1145 | **0.846 / 0.795** | vs base B (35.683 / 34.792): T1 −0.033, FLAIR +0.013, SSIM and HFEN better on both, sharpness +0.016 / +0.016 → FLAIR now above L1-ESPIRiT's 0.789, T1 0.003 short of 0.849. **keep-marginal (structural)**: the sharpness lever at a −0.03 T1 cost; `p3_r2_ptv001` added to bracket the cost, `p3_r2_ptv005` still queued. |
| 20:57 | p3_r2_vtv0005 (base B + vectorial TV 0.005) | 35.705 | 34.792 | 0.9376 / 0.9217 | 0.1183 / 0.1179 | 0.809 / 0.756 | **discard**: T1 +0.02 only, HFEN worse on both, sharpness −0.02 / −0.02 — the same smoothing signature as on the probe mask. `p3_r2_vtv002` (stronger) dropped as foregone. |
| 21:18 | p3_r2_ptv001 (base B + phase TV 0.01) | 35.678 | 34.811 | 0.9384 / 0.9234 | 0.1157 / 0.1151 | 0.839 / 0.787 | **keep**: T1 −0.005 (inside the floor), FLAIR +0.019, SSIM / HFEN better on both, sharpness +0.009 / +0.008 — but still below the L1-ESPIRiT sharpness on both (0.849 / 0.789). Phase TV traces a curve: 0.01 → free, 0.02 → FLAIR gate met at −0.03 T1, 0.05 (queued) → both gates met at a larger PSNR cost (phase 2: −0.24 / −0.07 on P6). The operating point on that curve is the user's call. |
| 21:26 | p3_r2_edge0001 (base B + common-edge 0.001) | **35.739** | **34.843** | 0.9384 / 0.9230 | 0.1159 / 0.1149 | 0.828 / 0.776 | **keep**: +0.056 / +0.051 over B, SSIM and HFEN better on both, sharpness −0.002 / −0.003 (inside tolerance). The symmetric cross-gradient prior is the first coupling term that helps both contrasts here. `p3_r2_edge0005` (5×) still queued to bracket it. |
| 21:46 | p3_r2_priv03 (base B + 0.3M FLAIR-private residual Gaussians) | 35.688 | 34.807 | 0.9376 / 0.9223 | 0.1171 / 0.1166 | 0.830 / 0.778 | **null**: +0.005 / +0.015, every other cell unchanged — the shared cloud already carries everything FLAIR needs; a private set adds parameters, not information. |
| 21:53 | p3_r2_edge0005 (base B + common-edge 0.005) | 35.558 | 34.674 | 0.9337 / 0.9177 | 0.1273 / 0.1283 | 0.824 / 0.768 | **discard**: −0.125 / −0.118 vs B, every cell worse — 5× is past the optimum. `p3_r2_edge0002` and `p3_r2_edge00005` queued to bracket 0.001 before the combinations are read. |
| 22:14 | p3_r2_ptv005 (base B + phase TV 0.05) | 35.501 | 34.723 | 0.9380 / 0.9239 | 0.1172 / 0.1148 | **0.865 / 0.815** | both sharpness gates met (L1 0.849 / 0.789) at −0.182 / −0.069 dB vs B. The phase-TV curve on B: 0.01 → 0.839/0.787 at −0.005/+0.019; 0.02 → 0.846/0.795 at −0.033/+0.013; 0.05 → 0.865/0.815 at −0.182/−0.069. T1 sharpness 0.849 sits near ptv ≈ 0.025–0.03 for ≈ −0.05 T1; `p3_r3_e0001_ptv003` added to the combinations. |
| 22:20 | p3_r2_edge0002 (base B + common-edge 0.002) | 35.736 | 34.840 | 0.9380 / 0.9225 | 0.1171 / 0.1162 | 0.826 / 0.773 | PSNR level with 0.001 (−0.003 / −0.003), SSIM / HFEN / sharpness all slightly worse → 0.001 stays; the plateau is 0.001–0.002, 0.005 collapses. |
| 22:44 | p3_r2_edge00005 (base B + common-edge 0.0005) | 35.720 | 34.824 | 0.9382 / 0.9227 | 0.1162 / 0.1155 | 0.829 / 0.777 | −0.019 / −0.019 vs 0.001, structure marginally worse, sharpness +0.001 → **common-edge 0.001 is bracketed as the optimum** (0.0005 < 0.001 ≈ 0.002 ≫ 0.005). |
| 22:48 | p3_r3_e0001_ptv001 (B + common-edge 0.001 + phase TV 0.01) | 35.735 | **34.859** | **0.9392 / 0.9242** | **0.1145 / 0.1136** | 0.836 / 0.784 | **keep**: vs edge0001 alone T1 −0.004 (floor), FLAIR +0.016, SSIM / HFEN best of any row so far, sharpness +0.008 / +0.008 — the two terms add up. Sharpness still 0.013 / 0.005 under the L1 gate; the 0.02 and 0.03 combinations decide the operating point. |
| 23:03 | p3_r3_e0001_ptv002 (B + common-edge 0.001 + phase TV 0.02) | 35.706 | 34.853 | **0.9394 / 0.9248** | **0.1142 / 0.1131** | 0.844 / **0.792** | **keep**: best SSIM and HFEN of the whole loop, FLAIR sharpness gate met (0.792 > 0.789), T1 sharpness 0.844 still 0.005 short; T1 PSNR −0.033 vs edge0001 alone. Phase TV on this base behaves exactly as on B alone (additive), so the operating point is a pure sharpness-vs-T1-PSNR trade. |
| 23:15 | **p3_r3_e0001_ptv003 (B + common-edge 0.001 + phase TV 0.03)** | 35.665 | 34.834 | **0.9394 / 0.9250** | **0.1145 / 0.1131** | **0.850 / 0.799** | **the first recipe of the whole line (all three phases) that passes every pre-registered gate**: six head cells first, and sharpness at or above L1-ESPIRiT's on BOTH contrasts (0.850 ≥ 0.849, 0.799 ≥ 0.789) — P6 never did. Margins +0.456 / +0.331 at 1500 it. Promotion `p3_r3_e0001_ptv003_full4k` (4000 it, volumes) queued at the top. The ptv 0.01 / 0.02 / 0.03 rows are the whole trade curve; 0.03 is the cheapest point that clears both gates. |
| 23:36 | p3_r4_dens_q90 (B + edge 0.001 + adaptive densification, gradient score, q 0.90, cap 5.0M, until iter 1000) | 35.742 | 34.846 | 0.9384 / 0.9230 | 0.1159 / 0.1150 | 0.828 / 0.776 | **null**: +0.003 / +0.003 over the same recipe without densification, every structural cell identical to four decimals. The alternating split/prune ended at 3 608 294 points (+0.23 %), so the machinery ran but barely moved the cloud — at 3.6M the representation is not capacity-limited, which matches the AF12 line's finding that 4.2–6.0M only trades T1 for FLAIR. |
| 23:45 | p3_r4_dens_ee (same, evidence-energy split score) | 35.741 | 34.856 | 0.9384 / 0.9233 | 0.1165 / 0.1156 | 0.826 / 0.772 | **null**: T1 +0.002, FLAIR +0.013, but HFEN and sharpness both a shade worse. **Adaptive density control is closed as a class** on this line: neither split score changes the picture, and the master switch stays off in the final recipe. |
| 00:52 | **p3_r3_e0001_ptv003_full4k (the promotion, 4000 it, volumes)** | **35.706** | **34.928** | **0.9396 / 0.9261** | **0.1139 / 0.1121** | **0.855 / 0.804** | **the phase-3 recipe.** Margins over L1-ESPIRiT **+0.497 / +0.425** (P6: +0.343 / +0.304), SSIM +0.0062 / +0.0060 (P6: +0.0017 / +0.0002), HFEN better by 0.0147 / 0.0089 (P6: 0.0068 / 0.0022), sharpness **0.855 / 0.804 above L1-ESPIRiT's 0.849 / 0.789 on both contrasts** — P6 was below on both. Against P6 itself: +0.154 / +0.121 dB with every structural cell better. Seed repeat and the two GS-single arms are queued behind it. |
| 00:56 | p3_r3_ptv003_s43_full4k (same recipe, seed 43) | 35.707 | 34.928 | 0.9396 / 0.9261 | 0.1140 / 0.1122 | 0.855 / 0.804 | **seed floor 0.001 / 0.000 dB** at 4000 iterations — every margin above is far outside it. |
