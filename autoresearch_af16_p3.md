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

Cards (user rule 2026-09-12 16:05, after a collision with the AF12 phase-2 chains that took cards 2/3/5 twelve seconds before this line's first launch): **empty cards only** — `scripts/p3_sched.sh scripts/p3_queue.txt` (tmux session `p3_sched`) launches the next queue item on a card with no compute process, no live `p2_chain_*.sh` claim and no trainer of this line; card 4 is shared with the probe. The three screens that had started on cards 2/3/5 were stopped and their outputs removed; the queue holds the eight items above in priority order.

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

## Round log

(appended as rounds land)
