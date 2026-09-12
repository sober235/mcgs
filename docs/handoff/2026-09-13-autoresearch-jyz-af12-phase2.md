# Handoff 2026-09-13 — jiangyizhou AF 12, phase 2 (branch `autoresearch-jyz-af12-phase2`, worktree `mcgs_af12_wt`)

## 1. Done and verified

- **Final recipe `P13` = P10 + `gaussian.rotation_lr = 0.008`** (default 0.0003; nothing else changed). Runner
  `autoresearch_jyz_af12_P13.sh` has it baked in. At 4000 iterations, head PSNR / SSIM / HFEN:

  | setting | T1 | FLAIR |
  |---|---|---|
  | all AF 12 (`p2_P13_af12_full4k`, arm `gs_joint_p2`) | **36.969 / 0.9504 / 0.0948** | 36.310 / 0.9401 / 0.0956 |
  | design B, FLAIR at AF 11 (`p2_rotlr008_full4k`, arm `gs_joint_p2_fl11`) | 36.968 / 0.9504 / 0.0948 | **36.683 / 0.9444 / 0.0904** |
  | two seeds averaged, design B (`p2_P13_x2_full4k`) | 37.005 / 0.9507 / 0.0945 | 36.703 / 0.9446 / 0.0901 |

  Gain over P10: **T1 +0.025 dB, FLAIR +0.046 dB**, no cell down, sharpness unchanged. Margins over the bar (best
  non-MC-GS value per cell at every swept λ): **+0.543 / +0.926 dB** PSNR, +0.0031 / +0.0076 SSIM, +0.0086 / +0.0096
  HFEN. First on **every** full-brain slice in every direction, both contrasts, both settings.
  ```
  cd /data0/congcong/code/GS/mcgs_af12_wt && PYTHONNOUSERSITE=1 PYTHONPATH= \
    /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/p2_read.py \
    --parent ar9_P10_flair_af11_full4k --floor p2_P10_s43_full4k --iters 4000
  # p2_rotlr008_full4k  4000  36.968  36.683  +0.023  ...  KEEP
  ```
- **The two delivered figures** (the layout the user picked; orange in-panel slice PSNR / SSIM / NMSE, head-only error
  maps, slices 174 / 110 / 47), both HTTP 200:
  - T1, every arm at AF 12:
    `http://localhost:8765/t1flair_v3_af12/t1flair_v3_af12_T1_row3planes_compact_p2_174_110_47_headerr.png`
    `/home/congcongliu/figs/t1flair_v3_af12/t1flair_v3_af12_T1_row3planes_compact_p2_174_110_47_headerr.png`
  - FLAIR, design B (MC-GS at AF 11, every other arm at AF 12; the panel shows the AF 11 mask and the title says so):
    `http://localhost:8765/t1flair_v3_af12/t1flair_v3_af12_FLAIR_row3planes_compact_p2_flair_af11_174_110_47_headerr.png`
    `/home/congcongliu/figs/t1flair_v3_af12/t1flair_v3_af12_FLAIR_row3planes_compact_p2_flair_af11_174_110_47_headerr.png`
- Topic `Results_20260908/T1FLAIR_AF12_v3_jiangyizhou` updated: arms registered (`gs_joint_p2`, `gs_joint_p2_fl11`,
  `gs_joint_p2_x2`, `gs_single_p2`, `gs_single_p2_fl11`), `logs/summary.json` regenerated, per-slice logs written,
  README carries a P13 section, `RESULTS_AF12_T1W_FLAIR.md` regenerated with the new rows.
- Program and round log: `autoresearch_jyz_af12_p2.md` (every candidate, its pre-registration and its verdict).
  Record: `docs/superpowers/records/2026-09-13-autoresearch-jyz-af12-phase2.md`. Ledger rows carry the `p2_` prefix.

## 2. Waiting for the user

- **The bundle the user asked to have judged** (P10 + scale lr 0.005 at 6000 iterations) is a `reshape` by the letter
  of the rule: +0.013 / +0.062 dB head PSNR for −0.0001 head SSIM and −0.0002 head HFEN on T1, against a seed floor
  that is zero to four decimals on T1 SSIM. Not folded in; the user may overrule.
- **`gaussian.scale_lr` is a calibrated FLAIR dial**, deliberately left out of the recipe: 0.003 buys FLAIR +0.023 dB,
  0.005 buys +0.047…+0.052, each for one to five units of T1's fourth decimal. If the FLAIR panel matters more than
  T1's, this is the lever (the record has the dose-response table).
- Whether the worktree `CLAUDE.md` "Current AF 12 result" section should now point at the P13 figures (this handoff
  updates it; say so if the P10 pair should stay the reference).
- No held-out check was run: the recipe, the classical λ and the phase-2 knobs were all selected on jiangyizhou.

## 3. Next steps (if continued)

- Held-out transfer: P13 unchanged on `T1FLAIR_AF12_retro` (huangyuwei) or `T1FLAIR_AF12_v4_tangbing`, with
  `CONFIG=` pointing at that topic, before any claim beyond this subject.
- The knob space is closed on this subject (see the record's last section). The remaining levers are a learned prior,
  a different representation, or acquisition-side changes — none of them a tuning loop.

## 4. Pitfalls and do-not-redo

- **Never edit a launched script in place**: bash reads by byte offset, so a running instance resumes inside the new
  text. It cost one truncated log and a bogus `crash` status tonight. New behaviour goes in a new file.
- **`kill -0` waiters are unsafe here** (PIDs wrapped past 4194304 during the night) and a waiter can still be inside
  its 60-second poll when its runner exits. Use file-triggered waits plus the runner's any-trainer guard.
- **`pgrep -f <pattern>` matches the watcher's own command line** — a monitor waiting for `summarize.py` waited 40
  minutes for itself.
- A selector that filters on hand-written ledger status fires before the human read; `p2_promote_best_grid.sh` picked
  the second-best row that way. Do not reuse it as written.
- Cards: 2 and 3 are this loop's, 4 and 5 the parallel AF16 phase-3 session's (agreed by message 2026-09-12 18:35);
  0 and 1 carry the user's foundation-model jobs for days. The runner waits while any `arc/cycle.py` trainer sits on
  the target card.
- Screening keeps are filters, not rankings: the best 1500-iteration row of this phase (rotation 0.008 + scale lr
  0.005, +0.087 / +0.160) is a T1-for-FLAIR trade at 4000.

## 5. Why the key decisions

- The six-cell rule and the floors came from the P10 phase unchanged, pre-registered before the first candidate, so
  a PSNR gain could not be bought with SSIM or HFEN unnoticed. It rejected the phase's most attractive screening rows.
- The binding channel is T1 in design B (its margin over the bar is the smaller one), so a FLAIR-only gain never
  promotes. That is what closed the scale-lr direction, and it is the rule the user can overrule per channel.
- P13 takes 0.008 rather than 0.016: at 4000 the rotation curve is flat from 0.008 on, and the smallest value on a
  plateau is the safest working point.
