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

## Round 0 (existing terms, no code) — launched 2026-09-12 ~16:10, 1500 it, cards 2 / 3 / 5

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

Queues: `scripts/p3_r0_gpu2.sh`, `p3_r0_gpu3.sh`, `p3_r0_gpu5.sh` (sequential per card).

## Round 1 onward

See the spec, sections 4 and 6: the non-local RED prior (`paired_gs/losses/nonlocal_red.py`, Track A in worktree
`mcgs_af16_nl_wt`) on P6, then stacked on the Round-0 keeps. The acquisition probe (spec section 5, Track B, card 4)
reports into `docs/superpowers/records/2026-09-12-af16-probe-c-readout.md`.

## Round log

(appended as rounds land)
