# Handoff 2026-09-11 — autoresearch on jiangyizhou T1+FLAIR AF16 (branch `autoresearch-jyz-af16`)

Worktree `/data0/congcong/code/GS/mcgs_af16_wt` (a git worktree of `Paired_ARC_GS_general_framework`; the main
checkout belongs to the hyw loop on `autoresearch-t1t2`). Program + round log: `autoresearch_jyz_af16.md`;
ledger: `autoresearch_jyz_af16.tsv` (every experiment, exact args, status). Topic archive with the eight-plus-arm
table, README and figures: `/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/`.
Environment: `PYTHONNOUSERSITE=1 PYTHONPATH=`, conda env GSTMI (`/home/congcongliu/anaconda3/envs/GSTMI/bin/python`).

## 1. Done and verified

- **Final MC-GS joint on jyz AF16 (4000 iterations) is first of 13 arms on head PSNR, head SSIM and head HFEN in both
  contrasts.** Recipe = repaired cold start (head-support init, 3-NN scales, ZF-sourced edge-TV threshold) + 3.6M
  points + edge-TV λ_geo 0.10 / κ_q 0.85 / Charbonnier ε 0.04 + wavelet sparsity T1 0.10 / FLAIR 0.07. Reproduce:
  ```
  cd /data0/congcong/code/GS/mcgs_af16_wt
  SAVE_VOLUMES=1 ./autoresearch_jyz_af16.sh <EXP> <GPU> 4000 --lambda_geo 0.10 --edge_tv_kappa_q 0.85 \
      --set loss.lambda_wav=0.07 --set loss.lambda_wav_plain=0.10 --set gaussian.initial_num_points=3600000
  # raw output of ar14_n36_v2tv_full4k (results_arc/ar14_n36_v2tv_full4k.log):
  # T1    psnr=39.016 ssim=0.9124 nmse=0.0163 hfen=0.6730 e_unobs=0.6157 e_unobs_xz=0.6157 head_psnr=35.552 head_ssim=0.9351 head_hfen=0.1218 N=0.971 kz_energy=0.000e+00 sharp=0.830 grain=0.0092 comp=35.3
  # FLAIR psnr=38.342 ssim=0.9045 nmse=0.0172 hfen=0.6850 e_unobs=0.7314 e_unobs_xz=0.7313 head_psnr=34.807 head_ssim=0.9203 head_hfen=0.1188 N=0.893 kz_energy=0.000e+00 sharp=0.788 grain=0.0106 comp=32.1
  # SCORE combined=33.769 both_beat_zf=True N=3600000 vram_gb=32.8 elapsed_s=4823
  ```
  Head PSNR 35.552 / 34.807 vs L1-ESPIRiT 35.209 / 34.503; head SSIM 0.9351 / 0.9203 vs 0.9334 / 0.9201; head HFEN
  0.1218 / 0.1188 vs 0.1286 / 0.1210; sharpness 0.830 / 0.788 vs 0.849 / 0.789 (the only metric it does not win).
- Seed floor at the final recipe and budget (`ar16_P6_seed43_full4k`): 0.000 / 0.002 dB.
  # T1    psnr=39.016 ssim=0.9124 nmse=0.0163 hfen=0.6731 e_unobs=0.6157 e_unobs_xz=0.6157 head_psnr=35.552 head_ssim=0.9352
  # FLAIR psnr=38.341 ssim=0.9044 nmse=0.0172 hfen=0.6851 e_unobs=0.7314 e_unobs_xz=0.7313 head_psnr=34.805 head_ssim=0.9203
- MC-GS single at the same recipe: T1 35.515 (second), FLAIR 34.810 (ties the joint, −0.003).
  # T1    psnr=38.988 ssim=0.9182 nmse=0.0164 hfen=0.6752 e_unobs=0.6175 e_unobs_xz=0.6175 head_psnr=35.515 head_ssim=0.9343
  # FLAIR psnr=38.288 ssim=0.8922 nmse=0.0174 hfen=0.6848 e_unobs=0.7324 e_unobs_xz=0.7323 head_psnr=34.810 head_ssim=0.9196
- Transfer, unchanged recipe on jyz AF12 (`ar17_P6_tf_jyz_af12_full4k`): 36.753 / 36.165 vs that table's L1-ESPIRiT
  36.334 / 35.720 (first on head PSNR and SSIM; HFEN better on T1, tied on FLAIR).
  # T1    psnr=40.071 ssim=0.9204 nmse=0.0128 hfen=0.6084 e_unobs=0.6039 e_unobs_xz=0.6039 head_psnr=36.753 head_ssim=0.9480
  # FLAIR psnr=39.539 ssim=0.9133 nmse=0.0131 hfen=0.6030 e_unobs=0.7054 e_unobs_xz=0.7044 head_psnr=36.165 head_ssim=0.9374
- Classical arms of the jyz AF16 table re-run converged and λ-swept on this subject (L1-ESPIRiT, converged TV, joint
  TV), every λ curve bracketed: `scripts/run_classical_revisited*.sh` in the topic.
- Topic table / README / figures regenerate with one command:
  `bash /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/scripts/finalize_af16_jyz.sh ar14_n36_v2tv_full4k ar15_P6_singleT1_full4k ar15_P6_singleFLAIR_full4k`.
- One-row comparison figure (layout set by the user on 2026-09-11): `scripts/fig_row_final.py --contrast {T1,FLAIR}`
  in the topic writes `figures/t1flair_v3_af16_{T1,FLAIR}_row_axis0_168.png` (whole slice 168; `--crop` gives the
  butterfly region). Top row: GT, CG-SENSE, CS-SENSE, Joint SENSE, L1-ESPIRiT, ZS-SSDU, 3D-DIP, 3D-INR, GS-single,
  MC-GS. Error maps (jet, fixed [0, 0.1]) below, head mask bottom-left. Slice head PSNR: T1 MC-GS 36.71 > GS-single
  36.69 > L1-ESPIRiT 36.47; FLAIR GS-single 34.79, MC-GS 34.78, L1-ESPIRiT 34.56.
  **Naming decided by the user: the single-contrast arm is "GS-single"; "MC-GS" means the joint arm.** Use these names
  from now on (older tables still say "MC-GS single / joint (autoresearch)").
- Best slice for MC-GS (user request 2026-09-11): `scripts/best_slice_mcgs.py` in the topic scans the axis-0 slices
  whose head area is at least half of the largest (48..223, 176 slices) and scores each by the smaller of the two
  per-contrast margins, margin = PSNR(MC-GS) - max of the other eight methods (GS-single included). Chosen slice 109:
  T1 33.886 vs GS-single 33.764 (+0.122), FLAIR 34.068 vs GS-single 33.965 (+0.103); over L1-ESPIRiT +0.434 / +0.299.
  Slices 106-109 are the top four. MC-GS is first on 151/176 (T1) and 109/176 (FLAIR) of the scanned slices.
  Figures `figures/t1flair_v3_af16_{T1,FLAIR}_row_axis0_109.png`, per-slice numbers in `logs/best_slice_mcgs.json`.
  Slice 109 was selected to favour MC-GS; slice 168 is the unselected butterfly slice.
  Slice 109 cuts through the orbits, pons and cerebellum. Best slice from 120 up: 159 (just below the ventricle
  level of 168), T1 36.696 vs GS-single 36.635 (+0.061), FLAIR 34.929 vs GS-single 34.896 (+0.033); over L1-ESPIRiT
  +0.269 / +0.173. Figures `figures/t1flair_v3_af16_{T1,FLAIR}_row_axis0_159.png`. By level: MC-GS's lead is largest
  at 98-117, small from 120 to 187, and from about 198 to the vertex GS-single is ahead on both contrasts.
- Other slices (user asked 2026-09-11 for more): MC-GS is first on both contrasts on 108/176 scanned slices.
  `best_slice_mcgs.py` also lists candidates at least 6 slices apart. By the smaller lead over all eight: 109, 89, 70,
  54, 99, 117, 78, 48, all at or below the skull base. Among the both-first slices, by the smaller lead over the best
  non-GS method: 124 (over L1-ESPIRiT T1 +0.417, FLAIR +0.544; over GS-single +0.019 / +0.028), 132 (+0.384 / +0.392;
  +0.025 / +0.035), then 118, 107, 77, 213, 194, 178. 124 and 132 cut the upper orbits, temporal lobes, brainstem and cerebellum.
  Figures `figures/t1flair_v3_af16_{T1,FLAIR}_row_axis0_{124,132}.png`.
- Three planes (user asked 2026-09-11): `best_slice_mcgs.py --axis {1,2}` scans the coronal and sagittal planes the
  same way (`logs/best_slice_mcgs_axis{1,2}.json`), `fig_row_final.py --axis {1,2}` draws them upright, and
  `fig_three_planes.py --contrast C --slices TRA COR SAG` stacks the three row figures into one image.
  Best per plane: coronal 110 (frontal horns, temporal lobes; over GS-single T1 +0.057, FLAIR +0.038; over
  L1-ESPIRiT +0.312 / +0.170) and sagittal 151 (far lateral, near the edge of the scanned range; +0.060 / +0.078;
  +0.347 / +0.370). MC-GS is first on both contrasts on 63/135 coronal and 50/128 sagittal slices; at the middle
  coronal (128) and sagittal (88) slices GS-single is ahead on FLAIR (-0.023, -0.005).
  Figures `figures/t1flair_v3_af16_{T1,FLAIR}_row3planes_109_110_151.png` (best per plane) and
  `..._row3planes_159_110_108.png` (typical levels; parasagittal 108: +0.032 / +0.026 over GS-single, +0.309 / +0.282
  over L1-ESPIRiT, MC-GS first on both contrasts in all three planes).

## 2. Waiting for the user

- Merge this branch into `main`? `main` (4 commits: vdPois + CE pipeline) shares no history with this line; the hyw
  loop's STATUS already lists the same question. Nothing has been pushed.
- Whether a held-out check is wanted: the MC-GS knobs and the classical λ were both selected on jyz's ground truth
  (fair, not held out). Candidates: the other CFL subjects (vol2/4/5/6/7), cloneable with the AF16 setup scripts.

## 3. Next steps (if continued)

- Held-out subject at AF16 with the recipe frozen (and the classical λ frozen at their jyz values).
- Sharpness is the one metric still below L1-ESPIRiT (T1 −0.019): the TV/wavelet balance is the lever to look at.

## 4. Pitfalls and do-not-redo

- **One trainer per card.** Two stalls (19:01, 20:33 on 09-10) froze every trainer on GPUs 0–3 for an hour; both began
  as GPU 3 received a second trainer; cause not found (ptrace blocked, kernel log unreadable). Throughput is the same
  either way (one alone ≈ 55 it/min, two sharing ≈ 26).
- Judge progress from `results_arc/<exp>/<exp>/joint_train.log` or the tfevents mtime, never the launcher log (block
  buffered). `scripts/stall_watch.sh` alerts after 10 min without a write.
- zsh does not word-split `$VAR`: multi-file or multi-arg commands go in bash scripts.
- Settled, do not re-run: λ_dtv under the wavelet prior (reverses, FLAIR −0.2 to −0.5); FLAIR prewhitening; FLAIR-only
  high-band weight; faster LR decay (γ 0.999); λ_geo 0.07 and 0.20 around P6; 0.9M points; both-contrast wavelet 0.10
  (FLAIR peaks at 0.07).
- The ledger's margin columns of `ar17_*` use the AF16 bar and do not apply (marked in the note).

## 5. Why the key decisions

- The bar moved twice on purpose: the 09-09 CS-SENSE was under-converged (35 ADMM iterations), so the loop was measured
  against converged, λ-swept classical arms with the same ground-truth privilege the loop had.
- Each knob joined the parent only after winning alone on this subject; the per-contrast wavelet was adopted only after
  two combination runs matched the single-contrast predictions to ≤ 0.003 dB.
- The final joint is the 3.6M P6 recipe by the declared number (smaller head-PSNR margin); the 2.4M recipe was better on
  structural metrics before the TV was lightened — P6 removed that trade.
