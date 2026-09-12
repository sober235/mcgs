# jiangyizhou, 3D T1W + FLAIR, retrospective AF 12 — consolidated results (2026-09-12)

Topic `Results_20260908/T1FLAIR_AF12_v3_jiangyizhou` (mirrored in the framework repo as `t1flair_af12_v3_jiangyizhou/`, branch `autoresearch-jyz-af12`). Every number below is read from `logs/summary.json` by `scripts/results_af12_md.py`.

## 1. Setup

- Subject vol3_jiangyizhou (MC_Jia CFL), 26 coils, volumes 256 × 256 × 176; T1 = 3D FSP-GRE (fully sampled PE plane), FLAIR = 3D MX3D with the vendor's elliptical shutter (acquired support 0.7577 of the PE plane).
- One house vd-Poisson mask on the (ky, kz) plane for both contrasts (`data/mask_af12_3d.h5`, 3847 samples = 8.54 % of the plane, calib 24, readout fully sampled): nominal AF 12, **T1 11.71x** and **FLAIR 9.74x** against each contrast's own acquired support.
- Ground truth = the fully sampled data through the same ESPIRiT maps (SoS = 1); gauge = magnitude difference, whole-volume PSNR at max|GT|; head-only metrics on |GT| > 0.05 · P99.9. Head-only PSNR / SSIM / HFEN are the numbers to quote (whole-volume SSIM is dominated by air noise).
- The three strongest classical arms have their λ swept on this subject against the ground truth (README, "Classical λ re-swept on this subject"); the table keeps each family's best-λ row. ZS-SSDU, 3D-DIP, 3D-INR: model selection without ground truth.

## 2. Every arm, both contrasts (best value per column in bold)

| arm | T1 whole PSNR / SSIM / NMSE | T1 head PSNR / SSIM / HFEN | T1 sharp | FLAIR whole PSNR / SSIM / NMSE | FLAIR head PSNR / SSIM / HFEN | FLAIR sharp |
|---|---|---|---|---|---|---|
| Zero-filled | 30.240 / 0.8353 / 0.12325 | 26.130 / 0.7009 / 0.6641 | 0.294 | 32.109 / 0.8523 / 0.07221 | 28.193 / 0.7330 / 0.6266 | 0.278 |
| CG-SENSE | 35.082 / 0.7562 / 0.04042 | 33.514 / 0.8981 / 0.1741 | 0.944 | 35.704 / 0.7782 / 0.03156 | 33.957 / 0.9032 / 0.1285 | 0.870 |
| CS-SENSE (35 outer, 09-09, under-converged) | 39.395 / 0.9523 / 0.01497 | 35.762 / 0.9429 / 0.1208 | 0.819 | 38.773 / 0.9416 / 0.01557 | 35.084 / 0.9307 / 0.1139 | 0.728 |
| CS-SENSE (TV, 200 outer, λ swept here) | 39.646 / 0.9571 / 0.01413 | 36.162 / 0.9446 / 0.1102 | 0.874 | 39.142 / 0.9497 / 0.01430 | 35.581 / 0.9353 / 0.1073 | 0.783 |
| Joint SENSE (JTV, 200 outer, λ swept here) | 39.648 / **0.9587** / 0.01413 | 36.070 / 0.9442 / 0.1143 | 0.863 | 39.277 / 0.9523 / 0.01386 | 35.624 / 0.9352 / 0.1073 | 0.802 |
| L1-ESPIRiT (BART, 100 FISTA, λ swept here) | 39.893 / 0.9582 / 0.01335 | 36.426 / 0.9469 / 0.1036 | 0.876 | 39.205 / 0.9452 / 0.01409 | 35.757 / 0.9368 / 0.1000 | 0.832 |
| P-LORAKS | 35.874 / 0.8889 / 0.03368 | 31.992 / 0.8528 / 0.1961 | 1.182 | 35.071 / 0.8731 / 0.03651 | 31.176 / 0.8165 / 0.1253 | **1.446** |
| ZS-SSDU | 39.357 / 0.9577 / 0.01511 | 35.724 / 0.9422 / 0.1330 | 1.000 | 38.807 / **0.9563** / 0.01545 | 35.034 / 0.9353 / 0.1297 | 1.099 |
| 3D-DIP | 35.267 / 0.8475 / 0.03873 | 31.363 / 0.8634 / 0.2750 | 0.645 | 34.730 / 0.8131 / 0.03950 | 30.810 / 0.8333 / 0.2656 | 0.518 |
| 3D-INR | 34.668 / 0.8218 / 0.04446 | 31.638 / 0.8373 / 0.1577 | **1.292** | 34.637 / 0.8274 / 0.04035 | 31.190 / 0.8277 / 0.1690 | 1.069 |
| MC-GS, old cold start | 36.349 / 0.8703 / 0.03019 | 34.160 / 0.9169 / 0.1478 | 0.895 | 36.646 / 0.9079 / 0.02541 | 33.391 / 0.9064 / 0.1481 | 0.801 |
| GS-single, old cold start | 35.529 / 0.8297 / 0.03647 | 33.799 / 0.9032 / 0.1444 | 0.977 | 36.204 / 0.8749 / 0.02813 | 33.409 / 0.9006 / 0.1425 | 0.911 |
| MC-GS v2 | 38.949 / 0.9194 / 0.01659 | 35.520 / 0.9321 / 0.1056 | 0.927 | 38.584 / 0.9082 / 0.01626 | 35.056 / 0.9238 / 0.1063 | 0.875 |
| GS-single v2 | 38.601 / 0.9205 / 0.01798 | 35.209 / 0.9260 / 0.1136 | 0.947 | 38.503 / 0.8966 / 0.01657 | 35.101 / 0.9216 / 0.1077 | 0.909 |
| MC-GS v3 | 39.662 / 0.9206 / 0.01408 | 36.337 / 0.9428 / 0.0987 | 0.892 | 39.334 / 0.9098 / 0.01368 | 35.935 / 0.9346 / 0.0994 | 0.847 |
| GS-single v3 | 39.566 / 0.9251 / 0.01440 | 36.257 / 0.9412 / 0.1019 | 0.891 | 39.325 / 0.9011 / 0.01371 | 36.009 / 0.9346 / 0.0994 | 0.857 |
| MC-GS v3 ×2 seeds | 39.847 / 0.9219 / 0.01349 | 36.539 / 0.9449 / 0.0974 | 0.883 | 39.441 / 0.9107 / 0.01335 | 36.049 / 0.9360 / 0.0985 | 0.842 |
| MC-GS v3.2 (hyw-selected) | 39.654 / 0.9200 / 0.01411 | 36.299 / 0.9423 / 0.0965 | 0.921 | 39.473 / 0.9028 / 0.01325 | 36.039 / 0.9369 / 0.1025 | 0.822 |
| MC-GS v3.2 ×2 seeds | 39.838 / 0.9213 / 0.01352 | 36.499 / 0.9445 / 0.0952 | 0.912 | 39.536 / 0.9032 / 0.01306 | 36.108 / 0.9378 / 0.1015 | 0.818 |
| MC-GS P10 (AF12 loop) | 40.272 / 0.9281 / 0.01224 | 36.944 / 0.9503 / 0.0947 | 0.858 | 39.692 / 0.9186 / 0.01260 | 36.268 / 0.9397 / 0.0958 | 0.823 |
| GS-single P10 | 40.241 / 0.9341 / 0.01232 | 36.897 / 0.9495 / 0.0965 | 0.855 | 39.693 / 0.9063 / 0.01260 | 36.323 / 0.9405 / 0.0944 | 0.819 |
| MC-GS P10 ×2 seeds | **40.319** / 0.9285 / **0.01210** | **36.990** / **0.9507** / **0.0943** | 0.856 | 39.723 / 0.9190 / 0.01251 | 36.295 / 0.9401 / 0.0954 | 0.822 |
| **MC-GS P13 (phase-2 loop)** | 40.286 / 0.9286 / 0.01220 | 36.969 / 0.9504 / 0.0948 | 0.858 | **39.725** / 0.9196 / **0.01250** | 36.310 / 0.9401 / 0.0956 | 0.823 |
| GS-single P13 | 40.236 / 0.9348 / 0.01234 | 36.903 / 0.9495 / 0.0968 | 0.854 | 39.714 / 0.9073 / 0.01254 | **36.347** / **0.9408** / **0.0943** | 0.818 |

Margins of MC-GS P10 over the best non-MC-GS arm of each cell (head-only): T1 PSNR +0.518 dB (l1espirit), SSIM +0.0034 (l1espirit), HFEN +0.0089 (l1espirit); FLAIR PSNR +0.511 dB (l1espirit), SSIM +0.0030 (l1espirit), HFEN +0.0042 (l1espirit).

## 3. MC-GS lineage on this subject (head PSNR T1 / FLAIR at 4000 iterations)

| recipe | selected on | T1 / FLAIR | what changed |
|---|---|---|---|
| MC-GS, old cold start | — | 34.160 / 33.391 | frozen warm start of the old FOV (cold start) |
| MC-GS v2 | huangyuwei T1+T2 | 35.520 / 35.056 | repaired cold start: head-support init, 3-NN scales, κ from ZF, 1.2M points |
| MC-GS v3 | huangyuwei T1+T2 | 36.337 / 35.935 | + 2.4M points + shared complex wavelet L1 0.05 (db2, 3 levels) |
| MC-GS v3.2 (hyw-selected) | huangyuwei T1+T2 | 36.299 / 36.039 | + T1 wavelet 0.06, FLAIR 0.12, edge-TV 0.07 |
| MC-GS P10 (AF12 loop) | jiangyizhou (this subject) | 36.944 / 36.268 | P6 of the AF16 loop (3.6M points, T1 0.10 / FLAIR 0.07, edge-TV 0.10 / κ_q 0.85) + wavelet **db4, 1 level, T1 0.12 / FLAIR 0.10** |

P6 itself (the AF16 loop's recipe run unchanged at AF 12, the user's provisional mark of 2026-09-11): 36.753 / 36.165 — P10 adds +0.191 / +0.103 dB with SSIM +0.0023 / +0.0024 and HFEN better by 0.0062 / 0.0044 (framework repo, `autoresearch_jyz_af12.md`).

## 4. Per slice (P10 against every canonical arm at its best λ, full-brain slices)

- T1 transverse: first on 96 of 96 slices
- T1 coronal: first on 121 of 121 slices
- T1 sagittal: first on 118 of 118 slices
- FLAIR transverse: first on 94 of 94 slices
- FLAIR coronal: first on 120 of 120 slices
- FLAIR sagittal: first on 118 of 118 slices
- Best own transverse slices: T1 173 (38.266 dB, +0.421 over L1-ESPIRiT), FLAIR 178 (36.559, +0.493); a common slice for both: 177 (T1 38.240 / FLAIR 36.550). GS-single (P10, one contrast at a time) is above the joint arm on every FLAIR slice by ~0.04 dB and below it on T1 (joint − single +0.047 / −0.055 dB whole-head).

## 5. Figures (URL line, then path; all served, HTTP 200 at write time)

**Compact three-plane panels, orange in-panel slice PSNR / SSIM / NMSE, head-only error maps, transverse 174 / coronal 110 / sagittal 47 (the layout chosen by the user on 2026-09-11)**

http://localhost:8765/t1flair_v3_af12/t1flair_v3_af12_T1_row3planes_compact_174_110_47_headerr.png
/home/congcongliu/figs/t1flair_v3_af12/t1flair_v3_af12_T1_row3planes_compact_174_110_47_headerr.png

http://localhost:8765/t1flair_v3_af12/t1flair_v3_af12_FLAIR_row3planes_compact_174_110_47_headerr.png
/home/congcongliu/figs/t1flair_v3_af12/t1flair_v3_af12_FLAIR_row3planes_compact_174_110_47_headerr.png

**Best single slices, one-row comparison (GT + nine methods, error maps below, head mask bottom-left): T1 173, FLAIR 178, and the common slice 177**

http://localhost:8765/t1flair_v3_af12/t1flair_v3_af12_T1_row_axis0_173.png
/home/congcongliu/figs/t1flair_v3_af12/t1flair_v3_af12_T1_row_axis0_173.png

http://localhost:8765/t1flair_v3_af12/t1flair_v3_af12_FLAIR_row_axis0_178.png
/home/congcongliu/figs/t1flair_v3_af12/t1flair_v3_af12_FLAIR_row_axis0_178.png

http://localhost:8765/t1flair_v3_af12/t1flair_v3_af12_T1_row_axis0_177.png
/home/congcongliu/figs/t1flair_v3_af12/t1flair_v3_af12_T1_row_axis0_177.png

http://localhost:8765/t1flair_v3_af12/t1flair_v3_af12_FLAIR_row_axis0_177.png
/home/congcongliu/figs/t1flair_v3_af12/t1flair_v3_af12_FLAIR_row_axis0_177.png

**Earlier panels with v3.2 as the proposed arm (per-slice head PSNR and rank in each title): best transverse slice and three planes**

http://localhost:8765/t1flair_v3_af12/t1flair_v3_af12_T1_best_axis0_173.png
/home/congcongliu/figs/t1flair_v3_af12/t1flair_v3_af12_T1_best_axis0_173.png

http://localhost:8765/t1flair_v3_af12/t1flair_v3_af12_FLAIR_best_axis0_177.png
/home/congcongliu/figs/t1flair_v3_af12/t1flair_v3_af12_FLAIR_best_axis0_177.png

http://localhost:8765/t1flair_v3_af12/t1flair_v3_af12_T1_best3planes.png
/home/congcongliu/figs/t1flair_v3_af12/t1flair_v3_af12_T1_best3planes.png

http://localhost:8765/t1flair_v3_af12/t1flair_v3_af12_FLAIR_best3planes.png
/home/congcongliu/figs/t1flair_v3_af12/t1flair_v3_af12_FLAIR_best3planes.png

**Ten-column mid-slice panels of the 09-10 study (whole-volume and head PSNR / SSIM in the titles; MC-GS arms are the old cold start and v2 — superseded)**

http://localhost:8765/t1flair_v3_af12/t1flair_v3_af12_T1_plane0.png
/home/congcongliu/figs/t1flair_v3_af12/t1flair_v3_af12_T1_plane0.png

http://localhost:8765/t1flair_v3_af12/t1flair_v3_af12_T1_plane1.png
/home/congcongliu/figs/t1flair_v3_af12/t1flair_v3_af12_T1_plane1.png

http://localhost:8765/t1flair_v3_af12/t1flair_v3_af12_T1_plane2.png
/home/congcongliu/figs/t1flair_v3_af12/t1flair_v3_af12_T1_plane2.png

http://localhost:8765/t1flair_v3_af12/t1flair_v3_af12_FLAIR_plane0.png
/home/congcongliu/figs/t1flair_v3_af12/t1flair_v3_af12_FLAIR_plane0.png

http://localhost:8765/t1flair_v3_af12/t1flair_v3_af12_FLAIR_plane1.png
/home/congcongliu/figs/t1flair_v3_af12/t1flair_v3_af12_FLAIR_plane1.png

http://localhost:8765/t1flair_v3_af12/t1flair_v3_af12_FLAIR_plane2.png
/home/congcongliu/figs/t1flair_v3_af12/t1flair_v3_af12_FLAIR_plane2.png

## 6. Caveats

- Magnitude gauge. On the strict complex difference |rec − GT| (`logs/complex_gauge.json`) L1-ESPIRiT is first on both contrasts (33.449 / 32.961) and P10 second (33.418 / 32.887); the MC-GS phase is marginally less faithful.
- P10 and the classical λ were selected on this subject's ground truth (fair, not held out); no second subject was run with P10.
- The GS arms fit FLAIR's measured shutter as known zeros; the comparison arms use the mask only (documented asymmetry).
- Whole-volume SSIM is sixth for every MC-GS arm (air-noise gauge).
- Not AF 12, kept apart: the side experiment with FLAIR's mask at nominal AF 11 (9.05x of its support, T1 unchanged), MC-GS only — T1 36.945 / FLAIR 36.637 head PSNR; slice-177 figures `t1flair_v3_af12_{T1,FLAIR}_flair_af11_axis0_177.png` (README, "Side experiment").

## 7. Design B (user, 2026-09-12): every arm at AF 12 on T1W and on FLAIR, except the proposed MC-GS whose FLAIR mask is AF 11

T1W: every method including MC-GS on the AF12 mask (identical to section 2). FLAIR: the comparison methods on the AF12 mask (the same mask as T1), MC-GS on the AF11 mask (`data/mask_af11_3d.h5`, 9.05x of FLAIR's acquired support against 9.74x). Run `ar9_P10_flair_af11_full4k` (P10 recipe), registered as `gs_joint_ar_fl11`.

| arm | T1 head PSNR / SSIM / HFEN | FLAIR head PSNR / SSIM / HFEN | FLAIR whole PSNR / SSIM / NMSE |
|---|---|---|---|
| **MC-GS P10, FLAIR mask AF 11 (T1 mask AF 12)** | **36.945 / 0.9503 / 0.0947** | **36.637 / 0.9439 / 0.0906** | **40.021** / 0.9214 / **0.01168** |
| MC-GS P10 (AF12 loop) (AF 12) | 36.944 / 0.9503 / 0.0947 | 36.268 / 0.9397 / 0.0958 | 39.692 / 0.9186 / 0.01260 |
| GS-single P10 (AF 12) | 36.897 / 0.9495 / 0.0965 | 36.323 / 0.9405 / 0.0944 | 39.693 / 0.9063 / 0.01260 |
| L1-ESPIRiT (BART, 100 FISTA, λ swept here) (AF 12) | 36.426 / 0.9469 / 0.1036 | 35.757 / 0.9368 / 0.1000 | 39.205 / 0.9452 / 0.01409 |
| CS-SENSE (TV, 200 outer, λ swept here) (AF 12) | 36.162 / 0.9446 / 0.1102 | 35.581 / 0.9353 / 0.1073 | 39.142 / 0.9497 / 0.01430 |
| Joint SENSE (JTV, 200 outer, λ swept here) (AF 12) | 36.070 / 0.9442 / 0.1143 | 35.624 / 0.9352 / 0.1073 | 39.277 / 0.9523 / 0.01386 |
| ZS-SSDU (AF 12) | 35.724 / 0.9422 / 0.1330 | 35.034 / 0.9353 / 0.1297 | 38.807 / 0.9563 / 0.01545 |
| CG-SENSE (AF 12) | 33.514 / 0.8981 / 0.1741 | 33.957 / 0.9032 / 0.1285 | 35.704 / 0.7782 / 0.03156 |

Under this design MC-GS is first on every FLAIR head cell: over GS-single (AF 12) +0.314 dB, SSIM +0.0035, HFEN +0.0038; over L1-ESPIRiT (AF 12) +0.880 dB. T1 is unchanged to 0.001 dB. Slice 177 (`figures/t1flair_v3_af12_{T1,FLAIR}_flair_af11_axis0_177.png`): FLAIR MC-GS 36.81 vs GS-single 36.59, L1-ESPIRiT 36.10; T1 MC-GS 38.24 vs GS-single 38.19, L1-ESPIRiT 37.87. The FLAIR column of MC-GS is at a 7 % denser mask than the field; the like-for-like single at AF 11 is a separate footnote (`ar9_P10_singleFLAIR_af11_full4k`).

http://localhost:8765/t1flair_v3_af12/t1flair_v3_af12_T1_flair_af11_axis0_177.png
/home/congcongliu/figs/t1flair_v3_af12/t1flair_v3_af12_T1_flair_af11_axis0_177.png

http://localhost:8765/t1flair_v3_af12/t1flair_v3_af12_FLAIR_flair_af11_axis0_177.png
/home/congcongliu/figs/t1flair_v3_af12/t1flair_v3_af12_FLAIR_flair_af11_axis0_177.png

## 8. Reproduce / refresh

`scripts/summarize.py` (table), `scripts/find_best_slice_v2.py --ours gs_joint_ar` (per slice), `scripts/fig_three_planes_compact.py --contrast T1 --slices 174 110 47 --mask_error`, `scripts/fig_row_final.py --contrast FLAIR --index 178`, `scripts/complex_gauge.py`, then `scripts/results_af12_md.py` for this sheet. P10 itself: framework repo, worktree `mcgs_af12_wt`, `SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh <EXP> <GPU> 4000` (the runner bakes P10).
