# Autoresearch on jiangyizhou T1+FLAIR: MC-GS first on the head metrics at AF16, and at AF12 unchanged — session record, 2026-09-10 / 11

Branch `autoresearch-jyz-af16` (worktree `mcgs_af16_wt`). Goal set by the user on 2026-09-10: MC-GS (the joint
arm) first on head PSNR in the jiangyizhou (jyz) T1+FLAIR AF16 table, with GS-single (the single-contrast arm)
reported alongside. The model may use only the acquired data, never another method's output or the ground truth;
the ground truth only scores. Program and round log `autoresearch_jyz_af16.md`, ledger `autoresearch_jyz_af16.tsv`,
handoff `docs/handoff/2026-09-11-autoresearch-jyz-af16.md`, topic archive
`/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou` (mirrored as `t1flair_af16_v3_jiangyizhou/`).
Scores use the head-only gauge of the AF12 tables (head = |GT| > 0.05 · P99.9, PSNR at the ground-truth maximum).
Naming: MC-GS is the joint arm and GS-single the single-contrast arm; older table labels say "MC-GS joint / single
(autoresearch)".

## 1. The bar

During the loop (09-10, 17:40) every classical arm of the jyz AF16 table was re-run to convergence with its λ swept
on this subject's ground truth (converged TV, joint TV, L1-ESPIRiT; every λ curve bracketed), the same ground-truth
privilege the loop had. L1-ESPIRiT became the strongest other arm on both head-PSNR columns (35.209 / 34.503). The
loop's parent, the huangyuwei loop's v2, started 0.97 / 0.95 dB below it at 1500 iterations; the table's original
cold-start MC-GS arm reads 32.890 / 32.402.

## 2. Result at AF16

AF16 is the nominal setting of the mask generator: the mask samples 2966 of 45056 phase-encode points, which is
15.19x for T1 and 12.49x of FLAIR's acquired k-space (vendor elliptical shutter, 75.8% of the plane). The user
chose on 2026-09-11 to keep these results under that label rather than re-run at an exact 16x.

Final MC-GS (`ar14_n36_v2tv_full4k`, recipe "P6"): the repaired cold start (head-support init, 3-NN scales,
ZF-sourced edge-TV threshold) + 3.6M points + edge-TV λ_geo 0.10 / κ_q 0.85 / ε 0.04 + wavelet sparsity T1 0.10 /
FLAIR 0.07, 4000 iterations. Rank among the nine non-GS arms (ZF, CG-SENSE, CS-SENSE at 35 iterations, converged
TV, joint TV, L1-ESPIRiT, ZS-SSDU, 3D-DIP, 3D-INR):

| column | T1 | FLAIR | best other arm, T1 / FLAIR |
|---|---|---|---|
| head PSNR | **35.552**, 1st | **34.807**, 1st | L1-ESPIRiT 35.209 / 34.503 |
| head SSIM | **0.9351**, 1st | **0.9203**, 1st | L1-ESPIRiT 0.9334 / 0.9201 |
| head HFEN | **0.1218**, 1st | **0.1188**, 1st | L1-ESPIRiT 0.1286 / 0.1210 |
| whole-volume PSNR | **39.016**, 1st | **38.342**, 1st | L1-ESPIRiT 38.758 / joint TV 38.048 |
| whole-volume NMSE | **0.01634**, 1st | **0.01719**, 1st | L1-ESPIRiT 0.01734 / joint TV 0.01840 |
| whole-volume HFEN | **0.6730**, 1st | **0.6850**, 1st | joint TV 0.6806 / 0.7018 |
| whole-volume SSIM | 0.9124, 6th | 0.9045, 6th | joint TV 0.9514 / ZS-SSDU 0.9461 |
| sharpness | 0.830 | 0.788 | L1-ESPIRiT 0.849 / 0.789 |

+2.66 / +2.41 dB over the table's original MC-GS arm. The seed-43 repeat (`ar16_P6_seed43_full4k`) reads
35.552 / 34.805, a floor of 0.000 / 0.002 dB. GS-single at the same recipe: T1 35.515, FLAIR 34.810, above every
non-GS arm on both; MC-GS − GS-single = +0.037 (T1) / −0.003 (FLAIR, a tie at the seed floor).

What moved it, in order of size (program, Verdict): the wavelet-sparsity prior (+0.3–0.5 dB, imported from the
huangyuwei loop); capacity from 1.2M to 3.6M points under that prior (FLAIR +0.4, then +0.15); a per-contrast
wavelet strength (T1 +0.15); the full budget once the T1 wavelet was strong (+0.1); and edge-TV back at its v2
strength once the wavelet carried the regularisation (+0.1 dB, with head HFEN below L1-ESPIRiT's). Lost as single
changes: directional TV under the wavelet prior, FLAIR prewhitening, a FLAIR-only high-band weight, faster LR decay,
stronger edge-TV, λ_geo 0.07, 0.9M points.

Per slice (head-masked slice PSNR on the slices whose head area is at least half of the largest; the other methods
are those of the figures: CG-SENSE, CS-SENSE, Joint SENSE, L1-ESPIRiT, ZS-SSDU, 3D-DIP, 3D-INR, GS-single):
- Against the seven non-GS methods MC-GS is first on every slice of all three planes in both contrasts, except one
  transverse T1 slice where it ties to the third decimal. Its smallest lead elsewhere is +0.023 dB (coronal T1).
- With GS-single included it is first on 151 / 109 of 176 transverse slices (T1 / FLAIR; 108 on both), 115 / 66 of
  135 coronal (63 on both) and 118 / 54 of 128 sagittal (50 on both). Every slice it loses, it loses to GS-single;
  from about transverse slice 198 to the vertex GS-single leads on both contrasts.

Figures (one row: GT, CG-SENSE, CS-SENSE, Joint SENSE, L1-ESPIRiT, ZS-SSDU, 3D-DIP, 3D-INR, GS-single, MC-GS; error
maps below on a fixed jet scale [0, 0.1]; head mask bottom-left):

| figure | T1 | FLAIR |
|---|---|---|
| transverse 168, the unselected butterfly slice | [T1](../../../t1flair_af16_v3_jiangyizhou/figures/t1flair_v3_af16_T1_row_axis0_168.png) | [FLAIR](../../../t1flair_af16_v3_jiangyizhou/figures/t1flair_v3_af16_FLAIR_row_axis0_168.png) |
| three planes 174 / 110 / 102, the current choice (MC-GS first on slice PSNR in both contrasts on all three; on SSIM too at 174 and 102, not at 110 where CS-SENSE leads the FLAIR SSIM, per `slice_quality.py`) | [T1](../../../t1flair_af16_v3_jiangyizhou/figures/t1flair_v3_af16_T1_row3planes_174_110_102.png) | [FLAIR](../../../t1flair_af16_v3_jiangyizhou/figures/t1flair_v3_af16_FLAIR_row3planes_174_110_102.png) |
| three planes 159 / 110 / 108, typical levels | [T1](../../../t1flair_af16_v3_jiangyizhou/figures/t1flair_v3_af16_T1_row3planes_159_110_108.png) | [FLAIR](../../../t1flair_af16_v3_jiangyizhou/figures/t1flair_v3_af16_FLAIR_row3planes_159_110_108.png) |
| three planes 109 / 110 / 151, the best slice per plane, chosen to favour MC-GS (109 is at the skull base) | [T1](../../../t1flair_af16_v3_jiangyizhou/figures/t1flair_v3_af16_T1_row3planes_109_110_151.png) | [FLAIR](../../../t1flair_af16_v3_jiangyizhou/figures/t1flair_v3_af16_FLAIR_row3planes_109_110_151.png) |
| compact three planes 174 / 110 / 47, the final choice: method names on top only, PSNR / SSIM / NMSE in each panel, error shown inside the head only | [T1](../../../t1flair_af16_v3_jiangyizhou/figures/t1flair_v3_af16_T1_row3planes_compact_174_110_47_headerr.png) | [FLAIR](../../../t1flair_af16_v3_jiangyizhou/figures/t1flair_v3_af16_FLAIR_row3planes_compact_174_110_47_headerr.png) |
| compact three planes 174 / 110 / 102, T1 alternative | [T1](../../../t1flair_af16_v3_jiangyizhou/figures/t1flair_v3_af16_T1_row3planes_compact_174_110_102_headerr.png) | - |

Compact three-plane figures (user decisions, 2026-09-11; `t1flair_af16_v3_jiangyizhou/scripts/fig_three_planes_compact.py --contrast C --slices TRA COR SAG --mask_error`): per plane an image row and an error row; the left cell of each error row is that plane's cut of the 3-D sampling mask through the k-space centre (the readout is fully sampled, so the coronal and sagittal cuts are lines); method names only above the top row; minimal gaps; the sagittal grey window matched to the other planes by the in-head median; head-masked slice PSNR / SSIM (7x7 window) / NMSE in orange-yellow inside each panel, the best of each plane in bold. Error maps are shown inside the head only (the filled head mask, opened with a radius-6 disk to drop ghost lines in the air, plus a 2-px rim). The FLAIR ground truth carries phase-encode ghosting and noise in the air that no method reproduces; MC-GS, whose Gaussians sit inside the head, shows it in full, and no sagittal slice where MC-GS leads is free of it. The metrics do not depend on this display choice. At 174 / 110 / 47 MC-GS is first on all three metrics in every plane of both contrasts, except FLAIR coronal SSIM (CS-SENSE 0.9230 against 0.9222).

## 3. The same recipe at AF12, same subject

`ar17_P6_tf_jyz_af12_full4k`: the P6 recipe unchanged with the AF12 config (`configs/t1flair_v3_af12_hyw.yaml`),
4000 iterations, 73 min. Scored against the jyz AF12 table (`Results_20260908/T1FLAIR_AF12_v3_jiangyizhou`, whose
classical λ were selected on huangyuwei); rank among its ten non-GS arms (P-LORAKS included):

| column | T1 | FLAIR | best other arm, T1 / FLAIR |
|---|---|---|---|
| head PSNR | **36.753**, 1st | **36.165**, 1st | L1-ESPIRiT 36.334 / 35.720 |
| head SSIM | **0.9480**, 1st | **0.9374**, 1st | converged TV 0.9459 / L1-ESPIRiT 0.9368 |
| head HFEN | **0.1009**, 1st | 0.1002, 2nd (level to four decimals) | L1-ESPIRiT 0.1034 / 0.1002 |
| whole-volume PSNR | **40.071**, 1st | **39.539**, 1st | converged TV 39.753 / L1-ESPIRiT 39.327 |
| whole-volume NMSE | **0.01282**, 1st | **0.01305**, 1st | converged TV 0.01379 / L1-ESPIRiT 0.01370 |
| whole-volume HFEN | **0.6084**, 1st | **0.6030**, 1st | converged TV 0.6154 / joint TV 0.6188 |
| whole-volume SSIM | 0.9204, 6th | 0.9133, 6th | joint TV 0.9587 / ZS-SSDU 0.9563 |
| sharpness | 0.857 | 0.832 | L1-ESPIRiT 0.906 / 0.809 |

For reference, the recipe selected on huangyuwei (v3.2, branch `autoresearch-t1t2`) reads 36.299 / 36.039 on this
table and its two-seed average 36.499 / 36.108. This is a second acceleration on the same subject, not a held-out
subject. The run is not registered in the AF12 topic table.

## 4. Caveats

- FLAIR known zeros: MC-GS and GS-single estimate FLAIR's shutter from the measured zeros of the undersampled
  data and fit it as zeros; the seven comparison arms use the mask only. The effect is not measured. The user
  decided on 2026-09-11 that any re-run gives every arm the same region.
- One subject. At AF16 the MC-GS knobs and the classical λ were both selected on this subject's ground truth: fair,
  not held out.
- Whole-volume SSIM is sixth at both accelerations (the air noise that `ecalib -c 0.0` leaves dominates it), and T1
  sharpness stays below L1-ESPIRiT's (0.830 against 0.849 at AF16, 0.857 against 0.906 at AF12).
- Once the prior is strong the coupling adds little on this subject: MC-GS − GS-single is +0.037 / −0.003 dB.
- Two unexplained GPU stalls on 09-10 (19:01 and 20:33), both when GPU 3 received a second trainer; one trainer per
  card since.

## 5. Open decisions (for the user)

1. A held-out subject at AF16, with the recipe and the classical λ frozen at their jyz values.
2. Whether to register the AF12 run in the jyz AF12 topic table, where the proposed arm is currently the
   huangyuwei-selected v3.2.
3. The branch is on origin as `autoresearch-jyz-af16`. `main` shares no history with this line and must not be
   merged with it.

## 6. Reproduce

```
cd /data0/congcong/code/GS/mcgs_af16_wt
SAVE_VOLUMES=1 ./autoresearch_jyz_af16.sh <EXP> <GPU> 4000 --lambda_geo 0.10 --edge_tv_kappa_q 0.85 \
    --set loss.lambda_wav=0.07 --set loss.lambda_wav_plain=0.10 --set gaussian.initial_num_points=3600000
# AF12: the same command with CONFIG=configs/t1flair_v3_af12_hyw.yaml in front
# AF16 topic table, README and figures:
bash /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/scripts/finalize_af16_jyz.sh \
    ar14_n36_v2tv_full4k ar15_P6_singleT1_full4k ar15_P6_singleFLAIR_full4k
```

Tests on this branch, 2026-09-11: 183 passed. `tests/test_score_edge.py` reads the gitignored
`results_arc/reg_aligned_4k_volumes.npz`, which exists only in the main checkout; with that file linked in, it loads
and runs without error (it defines no test functions).

Volumes on disk (the topic's `recon/`, not in git): every arm is stored as a complex64 256x256x176 `*_complex.npy`. The GS arms' magnitudes (`*_mag.npy`) come from the training runs. Their complex volumes (`gs_joint_ar_*_complex.npy`, `gs_single_ar_*_complex.npy`) were re-rendered on 2026-09-11 from the iter-4000 checkpoints with `arc/cycle.py <the run's args> --exp <exp>_cplx --from_ckpt results_arc/<exp>/<exp>/checkpoints/latest.pth --save_volumes`, which yields the hard data-consistency image that was scored. The re-rendered scores equal the original runs' exactly; |complex| equals the saved magnitude to 1e-7 of the peak; the phase agrees with the ground truth's (magnitude-weighted cosine of the phase difference inside the head 0.991; CG-SENSE 0.986-0.990). From now on `--save_volumes` stores the complex volumes next to the magnitudes. The metrics and error maps in this record are computed on magnitudes.
