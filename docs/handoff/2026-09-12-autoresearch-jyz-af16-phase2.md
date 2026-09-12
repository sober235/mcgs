# Handoff 2026-09-12 — jiangyizhou AF16 line, frozen (branch `autoresearch-jyz-af16-phase2`, worktree `mcgs_af16_wt`)

## 1. Done and verified
- Phase 1 (P6, magnitude gauge first on every head cell) and phase 2 (p6ptv: P6 + phase-only TV 0.05, first on both
  gauges) are complete; the record `docs/superpowers/records/2026-09-12-autoresearch-jyz-af16-phase2.md` holds the
  table. Re-run check of the headline numbers:
  ```
  cd /data0/congcong/code/GS/mcgs_af16_wt
  cut -f1,7,8,12,13 autoresearch_af16_p2.tsv | grep -E 'p2_promo_p6ptv_full4k|p2_r4'
  # p2_promo_p6ptv_full4k  35.311  34.74   32.322  31.851
  # p2_r4_db4l1_full4k     35.528  34.805  32.485  31.918
  python3 -c "import json;[print(r['contrast'],r['method'],round(r['head_psnr'],3)) for r in json.load(open('/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/logs/summary.json')) if r['method'] in ('gs_joint_ar','l1espirit')]"
  # T1 gs_joint_ar 35.552 / l1espirit 35.209 ; FLAIR gs_joint_ar 34.807 / l1espirit 34.503
  ```
- The three ledger rows the phase-2 session left uncommitted (p6ptv single arms, the db4 one-level transfer test) are
  committed with their reading (23f8b99).

## 2. Decisions taken and waiting
- **Taken 2026-09-12 (user): the AF16 line is frozen — no more runs; code and parameters stay isolated from the AF12
  line** (P10 is not brought in; the transfer test is an observation only).
- **Taken 2026-09-12, later the same day (user): P6 is the recipe quoted externally (magnitude gauge); p6ptv is
  reported only as the both-gauges note.**
- Waiting: a held-out AF16 subject; nothing else.

## 3. Next steps (only if reopened)
- A clean transfer test of the AF12 wavelet shape without phase TV (P6 + db4 + 1 level + T1 0.12 / FLAIR 0.10, 4000 it)
  would answer whether the magnitude gain transfers; the user declined it on 2026-09-12.

## 4. Pitfalls and do-not-redo
- One trainer per card (two stalls on 2026-09-10 when a card got a second trainer). Cards 0–2 were this line's;
  nothing of this line runs now.
- The runner refuses non-`p2_` names and has no CONFIG override — by design (isolation).
- Do not merge `autoresearch-jyz-af16-phase2` into `autoresearch-jyz-af16` or vice versa without the user: phase 2
  changed `arc/cycle.py` (complex-gauge scorer) and added `paired_gs/losses/phase_tv.py`.
- The magnitude/complex gauge gap is real (MC-GS phase agreement below L1-ESPIRiT's without phase TV); quote the gauge.

## 5. Why
- Freeze: the magnitude objective was met in phase 1, the complex objective in phase 2; the remaining lever
  (the AF12 wavelet shape) belongs to the other line and the user wants the two lines separable.
