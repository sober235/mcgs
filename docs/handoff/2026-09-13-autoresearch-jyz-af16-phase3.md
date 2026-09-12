# Handoff 2026-09-13 — jiangyizhou AF16 phase 3 (branch `autoresearch-jyz-af16-phase3`, worktree `mcgs_af16_wt`)

## 1. Done and verified
- **Phase-3 recipe** = P6 + db4 wavelet on its finest level only (T1 0.12 / FLAIR 0.10) + common-edge 0.001 +
  phase TV 0.03, 4000 it: head PSNR **35.706 / 34.928**, SSIM 0.9396 / 0.9261, HFEN 0.1139 / 0.1121,
  sharpness 0.855 / 0.804. Margins over L1-ESPIRiT **+0.497 / +0.425** (P6: +0.343 / +0.304); first recipe of the
  three phases whose sharpness is at or above L1-ESPIRiT's on both contrasts. Registered as `gs_joint_p3` /
  `gs_single_p3`; P6 keeps `gs_joint_ar`. Record `docs/superpowers/records/2026-09-13-autoresearch-jyz-af16-phase3.md`.
- Re-run check:
  ```
  cd /data0/congcong/code/GS/mcgs_af16_wt
  cut -f1,7,8,21,22,23,24,25,26 autoresearch_af16_p3.tsv | grep -E 'p3_r3_e0001_ptv003_full4k|p3_r3_ptv003_s43_full4k'
  # 35.706  34.928  ... and the seed-43 repeat 35.707 / 34.928  (floor 0.001 / 0.000)
  python3 -c "import json;[print(r['contrast'],r['method'],round(r['head_psnr'],3),round(r['sharp'],3)) for r in json.load(open('/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/logs/summary.json')) if r['method'] in ('gs_joint_p3','gs_single_p3','gs_joint_ar','l1espirit')]"
  ```
- Figures (MC-GS first on PSNR / SSIM / NMSE in every plane of both contrasts):
  `figures/t1flair_v3_af16_{T1,FLAIR}_row3planes_compact_174_110_47_headerr_p3.png`,
  served at `http://localhost:8765/t1flair_v3_af16/`.
- Acquisition probe closed with its own record (`2026-09-12-af16-probe-c-readout.md`); known-zero ablation measured.
- New code, all default-off and unit-tested: `paired_gs/losses/nonlocal_red.py` (non-local RED, 14 tests),
  `known_zero_from_support` + `load_support_map` in `paired_gs/data/known_zero.py` (5 tests),
  `data.<x>.known_zero_support`. Suite 200 passed at the merge; the known-zero and non-local files pass here.

## 2. Decisions waiting for the user
- Which recipe is quoted externally (P6 today; the phase-3 recipe is better on every head cell and on sharpness).
- The 1.0 dB target was not reached (margin roughly doubled instead). Accept, re-state, or allow a learned prior.
- A held-out subject at AF16 (never run since phase 1).
- Whether future FLAIR acquisitions use an in-shutter mask (helps every method by ~0.35 dB, not the margin).

## 3. Next steps if the line stays open
- The complex gauge was not re-measured for the phase-3 recipe (it carries phase TV 0.03; phase 2 showed the term
  lifts that gauge). One `--from_ckpt` scoring pass answers it.
- Everything else this phase tried is closed: non-local RED, adaptive densification, vectorial TV, private residual
  Gaussians, and the two-mask acquisition design.

## 4. Pitfalls and do-not-redo
- One trainer per card; `scripts/p3_sched.sh` launches only on a card with no compute process at all (cards 4 and 5
  are this line's by agreement with the AF12 phase-2 session; 2 and 3 are theirs).
- `pgrep` patterns must be anchored on the python path (`^/home/.../python -u arc/cycle.py --exp p3_`), otherwise they
  match the calling shell and kill it.
- Edit `scripts/p3_queue.txt` by atomic replace (write a temp file, `os.replace`); the scheduler re-reads it every pass
  and a partial write shows up as a "bad queue line".
- A single-arm config feeds the same contrast into both channels, so its score json holds two reconstructions that
  differ only by the per-channel wavelet strength. The topic table and phase 1 quote the **plain** channel; the ledger
  row shows the contrast channel because the two channels share a name.
- The non-local prior is expensive (BM4D over the whole grid, 156 s per contrast per refresh) and loses on this data.

## 5. Why
- The margin was widened by changing the prior's *shape* (db4, finest level), adding the one cross-contrast coupling
  term that helps both contrasts (common-edge), and buying sharpness with phase TV at the cheapest point that clears
  both gates. The big structural bets — a non-local prior and adaptive densification — are closed with evidence.
