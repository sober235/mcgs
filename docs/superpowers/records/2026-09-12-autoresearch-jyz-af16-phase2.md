# jiangyizhou AF16, phase 2 — the complex gauge, and the line frozen (2026-09-11 19:00 → 2026-09-12)

Branch `autoresearch-jyz-af16-phase2` (worktree `/data0/congcong/code/GS/mcgs_af16_wt`), an isolated loop on top of the
first AF16 loop's P6: runner `autoresearch_af16_p2.sh` (AF16 config hard-coded, `p2_` names only), ledger
`autoresearch_af16_p2.tsv` (20 rows), program and round log `autoresearch_af16_p2.md`. Run by the session that owned
the line on 2026-09-11; closed and documented by the session that took it over on 2026-09-12. **User decision
2026-09-12: the AF16 line is frozen — no further runs, and no code or parameter from the AF12 line is brought in.**

## 1. Objective and bar

Phase 1 (`docs/superpowers/records/2026-09-11-autoresearch-jyz-af16.md`) had put P6 first on the magnitude gauge
| |rec| − |GT| | on every head cell. Phase 2 asked for the same on the strict complex gauge |rec − GT|, where P6 sat
0.251 (T1) / 0.517 (FLAIR) dB behind L1-ESPIRiT, subject to the magnitude margin staying non-negative
(`min_margin_cplx` maximised, `min_margin_mag ≥ 0`). Bars (`logs/bar_complex_gauge.json` of the topic): L1-ESPIRiT
magnitude 35.209 / 34.503, complex 32.201 / 31.718, in-head phase agreement 0.9925 / 0.9930.

## 2. What phase 2 found (all at 4000 iterations, head-only, T1 / FLAIR)

| recipe | magnitude PSNR | complex PSNR | head SSIM | head HFEN | sharpness | phase cos |
|---|---|---|---|---|---|---|
| **P6** (phase 1, the magnitude headline) | **35.552 / 34.807** | 31.950 / 31.201 | 0.9351 / 0.9203 | 0.1218 / 0.1188 | 0.830 / 0.788 | 0.9911 / 0.9907 |
| **p6ptv** = P6 + phase-only TV 0.05 | 35.311 / 34.740 | 32.322 / 31.851 | 0.9353 / 0.9231 | 0.1204 / 0.1147 | **0.876 / 0.837** | 0.9928 / 0.9931 |
| ptv05 = wavelet T1 0.14 / FLAIR 0.10 + phase TV 0.05 | 35.364 / 34.768 | 32.413 / 31.922 | 0.9362 / 0.9234 | 0.1265 / 0.1187 | 0.844 / 0.806 | 0.9930 / 0.9932 |
| p6ptv + the AF12 wavelet shape (db4, 1 level, T1 0.12 / FLAIR 0.10) — transfer test only | 35.528 / 34.805 | **32.485** / 31.918 | **0.9382 / 0.9250** | **0.1167 / 0.1140** | 0.871 / 0.821 | 0.9930 / 0.9932 |
| L1-ESPIRiT (bar) | 35.209 / 34.503 | 32.201 / 31.718 | 0.9334 / 0.9201 | 0.1286 / 0.1210 | 0.849 / 0.789 | 0.9925 / 0.9930 |

- The **phase-only TV** term (`paired_gs/losses/phase_tv.py`, `--lambda_ptv`) is what moves the complex gauge:
  +0.37 / +0.65 dB of complex head PSNR over P6 at the same wavelet strength, phase agreement up to L1-ESPIRiT's
  level, sharpness up to 0.876 / 0.837 — at a cost of 0.24 / 0.07 dB of magnitude PSNR. Both p6ptv and ptv05 are
  first against the seven comparison arms on both gauges and on head SSIM and HFEN; neither beats P6 on the
  magnitude gauge.
- Losers (program, "Verdict"): kspace_high 3.0, the frequency curriculum, phase TV 0.10, FLAIR wavelet 0.14 on top of
  phase TV, 5.0M points.
- GS-single at p6ptv: T1 35.132 / FLAIR 34.769 (complex 32.138 / 31.880); joint − single +0.179 / −0.029 magnitude.
  Seed 43 reproduces p6ptv to 0.003 / 0.001 dB.
- The transfer test of the AF12 line's wavelet shape, stacked on p6ptv, gives the best complex T1, SSIM and HFEN of
  any AF16 row but no magnitude gain (T1 −0.024 against P6, inside the phase-TV cost). It was run by the phase-2
  session on its own initiative; it is **not** part of any AF16 recipe (isolation decision below).

## 3. The frozen state (user decision 2026-09-12)

- **Magnitude headline stays P6**: `ar14_n36_v2tv_full4k`, registered in the topic as `gs_joint_ar` ("MC-GS"), with
  `gs_single_ar` — the row figures and the compact three-plane figures 174/110/47 of phase 1 stand.
- **p6ptv is the both-gauges recipe**: registered as `gs_joint_p2` ("MC-GS (P6 + phase TV)"), ptv05 as `gs_joint_p2b`;
  their figures `figures/t1flair_v3_af16_{T1,FLAIR}_row_p2_axis{0,1,2}_*.png` and `_row3planes_p2_174_110_102.png`.
- **Isolation**: the AF16 line keeps its own code (this branch: complex-gauge scorer, phase TV, the isolated runner)
  and its own parameters. Nothing from the AF12 loop (branch `autoresearch-jyz-af12`, recipe P10) is merged or
  adopted here; the db4 / one-level result above is an observation for the transfer question only.
- No further runs on this line unless the user reopens it.

## 4. Caveats (unchanged from phase 1 and 2)

One subject, knobs and classical λ selected on its ground truth; nominal AF 16 (actual T1 15.19x, FLAIR 12.49x of
its acquired k-space); FLAIR known zeros used by the GS arms only; whole-volume SSIM sixth (air-noise gauge);
ZS-SSDU still matches the ground-truth sharpness best.

## 5. Open decisions (for the user)

1. Which AF16 recipe is quoted where: P6 (magnitude, the convention) or p6ptv (both gauges, sharper, −0.24 dB T1 magnitude).
   **Decided 2026-09-12 (user): P6 is quoted externally; p6ptv is the both-gauges note only.**
2. A held-out subject at AF16 with the recipe and λ frozen (never run).
3. Branches: `autoresearch-jyz-af16` (phase 1, 3a44aba) and `autoresearch-jyz-af16-phase2` (this record) are both on
   origin; `main` shares no history with either and is not merged.
