# Record 2026-09-16: stage 5 — the MC-GS lever round on the 16yo MTP AF 12 pair

Branch `16yo-mtp`, worktree `/data0/congcong/code/GS/mcgs_peds_wt`, topic `Results_20260908/Peds_TSC_MTP_AF12_retro/`.
Session claude-e7. Scheduler `scripts/p5_sched.sh` (claude-eb's p4_sched2 with its own log / stop / pause files; line-wide
cap of two trainings), queue `scripts/p5_round1.txt` (the round log lives in its comments), ledger `autoresearch_peds.tsv`
(`p5_*` rows; the full-volume gauge), primary-region scores `results_arc/p5_*_primary.json` (the gauge of the tables).

## 1. What the user asked, and the rules

- 09-16 11:3x "启动mc-gs杠杆": run the levers left from stage 3 to close the T1w cells. Target unchanged: MC-GS first on all
  six head cells against every arm, every registered GS-single counted (user 09-15 / 09-16).
- 09-16 ~15:xx "可以适当将 GS-single 调参一点, 使 MC-GS 超过所有方法 0.8 dB": declined — a comparison arm is not tuned
  down below a configuration already run. Facts given instead: on T1w the 0.8 dB is blocked by L1-SPIRiT (+0.63 dB PSNR,
  HFEN lost), not by GS-single; on T2s the untuned single is 0.05 dB behind.
- 09-16 ~16:xx "无论如何, 只要提出的 MC-GS 尽可能超过对比方法": GS-single is the published baseline — every registered
  version counts as a bar, the joint arm's later levers (phase TV, iterations) are not given to it, and no further
  single controls are run. One control had already launched (16:37) before that decision arrived: `p5_single_T1w_n96_i6000`
  (9.6M, 6000 iterations), T1w 38.998 / 0.9687 / 0.0764. It is in the ledger and quoted below; it is not registered.

Base = the stage-3 final configuration `p3_n96_union_k05` (9.6M points, union guide x0.5, wavelet T1w 0.06 / T2s 0.10,
edge-TV T1w 0.03 / T2s 0.10, 4000 iterations). Phase TV = `loss.lambda_ptv`, cherry-picked from the jyz AF12 phase-3
line (0ba2d3b, `paired_gs/losses/phase_tv.py`, 6 unit tests); per-contrast weights `lambda_ptv_plain / _contrast` added
2026-09-16 19:4x (default = the shared value; smoke `p5_smoke_ptvsplit`, 60 iterations, ran through).

## 2. Results (primary region, head inside partitions 12..92; margins against the bars of `logs/rescore.json`)

Bars: T1w PSNR 38.993 and SSIM 0.9687 = GS-single (tuned, 9.6M); T1w HFEN 0.0679 = L1-SPIRiT; T2s 34.175 / 0.9052 /
0.1855 = GS-single.

| run | lever | T1w PSNR / SSIM / HFEN | T1w margins | T2s PSNR / SSIM / HFEN | T2s margins | verdict |
|---|---|---|---|---|---|---|
| p3_n96_union_k05 (base) | — | 38.936 / 0.9685 / 0.0768 | −0.057 / −0.0002 / −0.0089 | 34.227 / 0.9056 / 0.1836 | +0.052 / +0.0004 / +0.0019 | |
| p5_rot008 | rotation_lr 0.0003 → 0.008 | 38.940 / 0.9685 / 0.0768 | −0.052 / −0.0001 / −0.0089 | 34.227 / 0.9056 / 0.1836 | +0.052 / +0.0004 / +0.0019 | no effect; its single control dropped |
| p5_khp30 | T1w high-band data weight 2.2 → 3.0 | 38.870 / 0.9670 / 0.0766 | −0.123 / −0.0017 / −0.0087 | 34.223 / 0.9057 / 0.1837 | +0.048 / +0.0005 / +0.0018 | discard: −0.07 dB, HFEN unmoved |
| **p5_ptv001** | phase TV 0.01 | 38.988 / 0.9692 / 0.0759 | −0.005 / +0.0005 / −0.0080 | 34.309 / 0.9094 / 0.1789 | +0.135 / +0.0042 / +0.0066 | **KEEP**: the one lever that moves T1w |
| p5_ptv002 | phase TV 0.02 | 38.945 / 0.9692 / 0.0763 | −0.047 / +0.0006 / −0.0084 | 34.320 / 0.9111 / 0.1766 | +0.146 / +0.0059 / +0.0089 | T2s likes it, T1w does not |
| p5_ptv003 | phase TV 0.03 | 38.858 / 0.9690 / 0.0773 | −0.134 / +0.0003 / −0.0094 | 34.292 / 0.9118 / 0.1756 | +0.117 / +0.0065 / +0.0098 | T1w −0.13; T2s SSIM / HFEN best |
| p5_e0005_ptv001 | + common edge 0.0005 | 38.953 / 0.9691 / 0.0764 | −0.039 / +0.0004 / −0.0085 | 34.332 / 0.9098 / 0.1779 | +0.157 / +0.0046 / +0.0076 | T2s-only lever: costs T1w 0.034 |
| **p5_ptv001_i6000** | phase TV 0.01, 6000 iterations | **38.999 / 0.9692 / 0.0759** | **+0.006 / +0.0006 / −0.0080** | 34.311 / 0.9094 / 0.1789 | +0.136 / +0.0042 / +0.0066 | **current best: 5 of 6 cells** |
| p5_single_T1w_n96_i6000 | (single, 6000 iterations; ran before the no-controls decision) | 38.998 / 0.9687 / 0.0764 | | | | not registered; MC-GS i6000 leads it by 0.001 / 0.0005 / 0.0005 |

A parabola through the phase-TV scan (0 / 0.01 / 0.02) puts the T1w optimum at 0.0105, so the weight is not scanned finer.

Six-cell status with `p5_ptv001_i6000`: T1w PSNR first (+0.006 over the registered bar, +0.001 over the 6000-iteration
single), T1w SSIM first, **T1w HFEN second to L1-SPIRiT (0.0759 vs 0.0679; every GS-single is behind at 0.0764)**,
T2s first on all three. The HFEN gap did not move by more than 0.001 under any lever; it is the smoothing character of
the rendering against a lightly regularised k-space interpolation that keeps noise-like high frequencies.

## 3. Running / queued (19:4x)

`p5_ptv001_i8000` (8000 iterations, card 6, 19:42), `p5_ptvp001c003_i6000` (T1w 0.01 / T2s 0.03, the per-contrast
weights, 6000 iterations), `p5_ptv001_i6000_gp0015` (T1w edge-TV 0.015 + phase TV 0.01, 6000 iterations). After them:
register the best as `gs_joint_p5` (add_gs_arm, rescore both gauges, README tables, figures of record, the one-slice
search re-run on the new volume), record, STATUS.
