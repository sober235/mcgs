# Handoff 2026-09-12 — jiangyizhou AF12 autoresearch line (branch `autoresearch-jyz-af12`, worktree `mcgs_af12_wt`)

## 1. Done and verified
- The loop (`autoresearch_jyz_af12.md`, ledger `autoresearch_jyz_af12.tsv`) promoted **P10** = P6 + wavelet db4 on 1 level
  + T1 0.12 / FLAIR 0.10. At 4000 iterations: head PSNR **36.944 / 36.268** (P6 36.753 / 36.165), first on every head
  cell (PSNR, SSIM, HFEN, both contrasts) against every arm at its λ re-swept on this subject, first on every full-brain
  slice in every direction. Registered in `Results_20260908/T1FLAIR_AF12_v3_jiangyizhou` as `gs_joint_ar`, with
  `gs_single_ar` (36.897 / 36.323) and `gs_joint_ar_x2` (36.990 / 36.295).
  ```
  cd /data0/congcong/code/GS/mcgs_af12_wt && PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/autoresearch_jyz_af12_view.py | grep -E 'promoted-P10'
  # ar5_best_C10_T112_full4k   promoted-P10  4000  36.944  36.268  +0.518  +0.511  +0.511 +0.0030 +0.0030 +0.0087 +0.0042 0.858 0.823
  ```
- Classical λ re-swept on jiangyizhou AF12 (28 runs): L1-ESPIRiT T1 λ 0.002 = 36.426, FLAIR λ 0.004 = 35.757; TV 0.0007,
  JTV 0.001. Topic `summarize.py` keeps the best λ per family; `recon/` canonical names point at the best-λ volumes.
- Figures: `figures/t1flair_v3_af12_{T1,FLAIR}_row3planes_compact_174_110_47_headerr.png` (orange in-panel metrics,
  head-only error maps, the layout the user picked on 2026-09-11), served at `http://localhost:8765/t1flair_v3_af12/`.
- Complex-gauge caveat measured (`logs/complex_gauge.json`): L1-ESPIRiT first on |rec − GT| (33.449 / 32.961), P10 second
  (33.418 / 32.887).

## 2. Waiting for the user
- Whether to fold in scale lr 0.005 (+0.011 / +0.048 dB at 4000, T1 head HFEN −0.0002 = 2x its floor) — the rule says
  no, the user may say yes. Same for sym4 (+0.008 / +0.004) and the 6000-iteration budget (+0.009 / +0.016).
- Whether the CLAUDE.md "best 12x" note (P6, provisional) should now name P10; this handoff leaves that section as is.
- A held-out check (huangyuwei AF12, or another CFL subject) was not run; the recipe and the λ were tuned on jiangyizhou.
- Branch: pushed to origin as `autoresearch-jyz-af12`, not merged anywhere; `main` shares no history with this line.

## 3. Next steps (if continued)
- Held-out transfer: run P10 unchanged on `T1FLAIR_AF12_retro` (huangyuwei) with `CONFIG=` pointing at that topic, plus the
  classical re-sweep there, before any claim beyond this subject.
- The only positive lever left is the optimiser (scale lr); judge it at 4000 iterations only.

## 4. Pitfalls and do-not-redo
- Capacity above 3.6M and scale lr above 0.005 trade T1 for FLAIR at 4000 iterations, not at 1500 (three 1500-iteration
  keeps became reshapes at 4000). Wavelet strength on db2 / 3 levels is bracketed (0.07 loses, 0.14 reshapes); T1 0.14
  under the 1-level prior fails the sharpness gate. warm-up 1000 is illegal at 1500 iterations (`warmup < 0.5*iters`).
- The tool shell's memory watchdog kills backgrounded CPU scoring (3-D SSIM); run scorers detached on a card
  (`scripts/score_x2.py` uses CUDA when visible).
- Cards: 0–2 were the AF16 phase-2 loop's, 3–5 this loop's, 6/7 shared with another user; `nvidia-smi` before any launch.

## 5. Why the key decisions
- Six-cell rule with floors from a seed repeat, pre-registered before the first candidate, so PSNR could not be bought
  with SSIM/HFEN unnoticed; the Pareto clause was added when the binding channel swapped, not to admit a result.
- Bar = best of every swept λ per cell: a competitor tuned for SSIM or HFEN counts for those cells.
- P10 rather than P10 + scale lr: the letter of the rule; the alternative is written down with its numbers.
