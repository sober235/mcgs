# Handoff 2026-09-13 — jiangyizhou AF 12 phase 3 (branch `autoresearch-jyz-af12-phase3`, worktree `mcgs_af12_wt`)

## 1. Done and verified

- **Final recipe `P14` = P13 + `--lambda_edge 0.001 --lambda_ptv 0.01`** (P13 = P10 + rotation lr 0.008;
  P10 = P6 + db4 wavelet on one level, T1 0.12 / FLAIR 0.10). All at AF 12, 3.6M points, 4000 iterations.
  Runner `autoresearch_jyz_af12_p3.sh` (P13 baked in; the two λ are the extra args).
  ```
  cd /data0/congcong/code/GS/mcgs_af12_wt
  SAVE_VOLUMES=1 ./autoresearch_jyz_af12_p3.sh <EXP> <GPU> 4000 --lambda_edge 0.001 --lambda_ptv 0.01
  ```
- **All six head cells above every arm, GS-single included** — the first recipe of this line that does it:

  | head cell | T1 | margin | FLAIR | margin |
  |---|---|---|---|---|
  | PSNR | 37.015 | +0.150 | 36.360 | +0.036 |
  | SSIM | 0.9516 | +0.0020 | 0.9415 | +0.0005 |
  | HFEN | 0.0928 | +0.0041 | 0.0935 | +0.0008 |

  L1-ESPIRiT trails by +0.589 / +0.602 dB. Joint − single: **+0.066 / −0.038 → +0.150 / +0.036**.
  Seed floor 0.0037 / 0.0026 dB (`p3_e0001_ptv001_s43_full4k`).
- **Held-out check passes** (pre-registered before the run): P14 unchanged on huangyuwei gives FLAIR joint − single
  **+0.030**, and +0.670 / +0.668 dB over that topic's strongest classical arm. First held-out evidence on this line.
- **Figures**, same style / slices / gauge as the AF16 finalised pair so they compare panel by panel, both HTTP 200:
  - `http://localhost:8765/t1flair_v3_af12/t1flair_v3_af12_T1_row3planes_compact_180_124_76_inset48_headerr_p14.png`
  - `http://localhost:8765/t1flair_v3_af12/t1flair_v3_af12_FLAIR_row3planes_compact_180_124_76_inset48_headerr_p14.png`
  - local copies in `/home/congcongliu/figs/t1flair_v3_af12/`, vector PDFs beside each PNG.
- **New code**: `paired_gs/losses/phase_tv.py` ported from the AF16 line, trainer hook, `--lambda_ptv`,
  `tests/test_phase_tv.py` (6 tests). Default path bit-identical, verified against a pre-port run's first iteration.
- **Archive**: `Results_20260908/T1FLAIR_AF12_v3_jiangyizhou/runs_p3_loop/` — 4.3 GB, every run's score json,
  volume npz and log, plus the ledger, program, runner, configs and the new code. The topic README carries the P14
  section; the arms are registered as `gs_joint_p14` / `gs_single_p14`.

## 2. Waiting for the user

- **Capacity 4.8M.** A keep at 1500 (FLAIR +0.051, T1 flat, sharpness unchanged — the AF16 line's shape);
  `p3_e0001_ptv001_n48_full4k` was still running at handoff time. If it holds at 4000 it thickens FLAIR's thin
  margin and P14 gains a capacity variant; read it against the P14 promotion by rule 2.
- Whether P14 replaces P13 as the quoted AF 12 result (the worktree CLAUDE.md still names P13 and its figures).
- Whether to re-measure the complex gauge for P14 (phase 2's caveat: on |rec − GT| L1-ESPIRiT led).

## 3. Next steps (if continued)

- A third subject (tangbing, `T1FLAIR_AF12_v4_tangbing`) would turn two points into a trend; no config exists for it
  yet, one can be cloned as `t1flair_af12_hyw*.yaml` was.
- The AF16 line's remaining open item — capacity at 6.0M — was never tried here.

## 4. Pitfalls and do-not-redo

- **A lever tested at one strength is not tested.** Phase 1 closed the common-edge term at λ 0.02, twenty times its
  optimum; it was worth +0.049 / +0.041 at 0.001. Bracket before closing.
- **Closed next door, so do not re-run**: contrast-private residual Gaussians (null), non-local RED with a BM4D
  target (smooths only), adaptive density control (null), vectorial TV (smooths).
- **Closed here**: FLAIR at a harder mask does not flip joint − single (9.05x / 9.74x / 10.56x all give −0.03).
- `pgrep -f` takes ERE — a `\|` alternation matches nothing, and a kill "verified" with the same pattern reports
  success while the processes keep running (it cost an hour of unwanted competitor runs this session).
- zsh does not word-split `$var` — a launch loop that relies on it silently passes the whole string as one argument;
  four bogus `crash` rows entered the ledger that way and had to be removed by hand.
- Verify with a different tool than the one that just failed: process start times vs log write times vs ledger write
  times caught the above.

## 5. Why the key decisions

- The gauge (GS-single as a comparison arm) is the user's, taken 2026-09-13; it is what made FLAIR an open problem
  rather than a presentational one, and it is the gauge the six-cell objective of this phase was written against.
- The phase-TV operating point is 0.01, not the AF16 line's 0.03, because our sharpness baseline is 0.03 higher and
  the dose curve is correspondingly steeper — measured here, not assumed.
- The held-out subject was chosen as huangyuwei because it is the canonical topic of the earlier phases, with the
  same AF12 mask and gauge, so nothing but the subject changes.
