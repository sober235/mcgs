# runs_af12_loop — raw outputs of the jiangyizhou AF12 autoresearch loop (copied 2026-09-12 from mcgs_af12_wt/results_arc)

For every listed run: `<exp>_score.json` (cycle.py's scorer output, both channels), `<exp>_volumes.npz` (magnitude volumes
recT1 / recFL plus the GT and ZF references; from 2026-09-11 on also recT1_complex / recFL_complex), `<exp>.log`.
Ledger `autoresearch_jyz_af12.tsv`, program `autoresearch_jyz_af12.md`, runner `autoresearch_jyz_af12.sh` (bakes P10),
`record_*.md` and `handoff_*.md` as in the framework repo (branch `autoresearch-jyz-af12`).

| exp | what |
|---|---|
| ar0_P6_parent_full4k | P6 (AF16 loop's recipe) unchanged at AF 12 — the user's provisional mark (copy of mcgs_af16_wt ar17) |
| ar5_wavlev2_full4k / ar5_db4_full4k / ar5_db4_lev1_C085_full4k | promotions P7 / P8 / P9 |
| ar5_best_C10_T112_full4k | **P10, the final recipe** (36.944 / 36.268) |
| ar5_P10_s43_full4k, ar5_P10_x2_full4k, ar3_P10_i6000 | seed 43, two-seed magnitude mean (2x), 6000 iterations (1.5x) |
| ar8_P10_singleT1_full4k, ar8_P10_singleFLAIR_full4k | GS-single arms of P10 |
| ar9_P10_flair_af11_full4k | P10 with the FLAIR mask at AF 11 (T1 unchanged), design B |
| ar5_P10_slr005_full4k, ar5_P10_slr008_full4k, ar5_P10_sym4_full4k | marginal / equivalent variants (not folded in) |
| ar5_db4_lev1_C10_full4k, ar5_wavlev1_full4k, ar5_db4_lev1_full4k, ar5_n48_full4k, ar5_n42_full4k, ar5_slr008_full4k | other 4000-iteration reference runs of the loop |

The registered arms' volumes also sit in `../recon/` (`gs_joint_ar_*`, `gs_single_ar_*`, `gs_joint_ar_x2_*`, `gs_joint_ar_fl11_*`, magnitude and complex).
