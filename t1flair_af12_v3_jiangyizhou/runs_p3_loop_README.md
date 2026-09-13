# Phase-3 raw outputs (2026-09-13) — the P14 loop

Self-contained copy of everything the phase-3 autoresearch loop produced, so the result can be re-read without the
worktree. Framework side: branch `autoresearch-jyz-af12-phase3` of `mcgs`, worktree `mcgs_af12_wt`.

## What is here
- `<exp>_score.json` / `<exp>.log` for every run of the phase, and `<exp>_volumes.npz` for the 4000-iteration ones
  (magnitude and complex reconstructions, GT and ZF).
- `autoresearch_jyz_af12.tsv` — the ledger (phase-3 rows carry the `p3_` prefix).
- `autoresearch_jyz_af12_p3.md` — the program: pre-registration, decision rules, every round and its verdict.
- `autoresearch_jyz_af12_p3.sh` — the runner (P13 baked in; P14 = this plus the two λ below).
- `configs/` — the configs used, for jiangyizhou and for the held-out subject huangyuwei.
- `code/` — `phase_tv.py` (ported this phase), its unit test, and `pair_prior.py` (the common-edge prior, unchanged).

## The runs, in the order they decide things
| run | what |
|---|---|
| `p3_P13_i1500`, `p3_P13_i1500_s43` | the parent and the seed floor at the screening budget |
| `p3_edge00005/0001/0002/0005` | the common-edge bracket: single-peaked at 0.001 |
| `p3_ptv001/002/003` | the phase-TV dose curve: 0.01 is this subject's operating point |
| `p3_e0001_ptv001` | the combination at 1500 — the best row of the phase |
| `p3_e0001_ptv001_full4k` | **the P14 promotion** |
| `p3_e0001_ptv001_singleT1_full4k`, `..._singleFLAIR_full4k` | GS-single at P14 — the arms P14 has to beat |
| `p3_e0001_ptv001_s43_full4k` | P14's seed repeat (floor 0.0037 / 0.0026 dB) |
| `p3_hyw_P14_full4k`, `p3_hyw_P14_singleFLAIR_full4k` | **the held-out check on huangyuwei** |
| `p3_e0001_ptv001_n48`, `..._n48_full4k` | capacity at 4.8M (keep at 1500; the 4000 row decides) |

## Reproduce P14
```
cd /data0/congcong/code/GS/mcgs_af12_wt
SAVE_VOLUMES=1 ./autoresearch_jyz_af12_p3.sh <EXP> <GPU> 4000 --lambda_edge 0.001 --lambda_ptv 0.01
```
Held-out: add `CONFIG=configs/t1flair_af12_hyw.yaml`. Single arms: the `*_singleT1` / `*_singleFLAIR` configs.

## Headline
All six head cells above every arm, GS-single included: T1 37.015 / 0.9516 / 0.0928, FLAIR 36.360 / 0.9415 / 0.0935;
margins +0.150 / +0.036 (PSNR) over the binding arm, +0.589 / +0.602 dB over L1-ESPIRiT. Joint − single flipped from
−0.038 to +0.036 on FLAIR, and the held-out subject reproduces it (+0.030). See the topic README's P14 section.
