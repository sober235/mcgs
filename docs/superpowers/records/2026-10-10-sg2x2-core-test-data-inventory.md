# 2026-10-10 sg2x2 core test: data inventory of the 16 T1 + FLAIR pairs

Status: read-only inventory. CPU only, no GPU, no reconstruction run, nothing committed. Branch `exp/shared-geometry-2x2-ablation-20261009`, worktree `/data0/congcong/code/GS/mcgs_sg2x2_wt`. Spec: `docs/experiments/2026-10-09-mcgs-shared-geometry-2x2-ablation.md` section 4 (initialisation) and section 5.1 item 2 (core test). Nothing below is a reconstruction score: every number is a property of the files, so the inventory does not touch the test set in the spec's sense.

What was written: this file and the scratch directory `/tmp/claude-1002/sg2x2_inventory/` (about 25 GB: three converted test pairs 13 GB, BART intermediates 11 GB, arrays, figures, scripts, outputs). No data or code was changed, moved or removed. Scripts and outputs are listed in appendix A and are re-runnable.

## Summary (what the controller needs first)

1. **Health.** All 16 pairs (32 files) open, both datasets read completely, finite, complex64, no dead coil, no zero readout slice. All 45 h5 files in the two trees are fully allocated (no truncation). The 32 pair files in the `/data0/congcong/data/T1_T2_FLAIR_3D` copy are byte-identical to the `/data1` originals (md5, 32 of 32), so neither copy is "the good one". The `*.baiduyun.uploading.cfg` markers do not predict damage.
2. **The "different grid" is zero padding.** File array `(256, 12, NY, 256)` = (readout, coil, partition SPE, phase PE), NY = 220 for 14 pairs and 264 for the two `790` pairs. Only 176 SPE lines carry data; the other `NY-176` lines (44 or 88) are zero padding at the low-index end. The acquired block is 176 x 256 for all 16 pairs, which is the pilot's `(256, 256, 176)` grid after cropping the SPE axis and swapping the two PE axes. The "+22 px DC offset" of the old huangyuwei bundle is half of that padding (+44 for the 790 pairs); after the crop the DC sits at (88, 128) within one sample, no roll is needed.
3. **Axis 0 of the raw `kspace` is already image domain** (hybrid array). A centred 1-D FFT along axis 0 is required before `ifft3c` (this is what the earlier `*_trainer.mat` and `jd_kfix` conversions did). `ifft3c` of the raw array gives a garbled volume (head fraction 0.12 instead of 0.34, CSM residual 0.35 instead of 0.06).
4. **The stored `csm` should not be used as is.** They are smooth along the two PE axes but jump from readout slice to readout slice (neighbour phase difference 0.55 to 0.84 rad along readout against 0.01 to 0.05 along PE), so the complex ground truth built with them has random phase between readout slices (0.59 to 0.90 rad) and, for the two 790 pairs, visible stripes. Freshly estimated ESPIRiT maps (the pilot recipe `ecalib -t 0.001 -c 0.0 -r 24 -m 1`, 57 s per contrast with 12 coils) fit the data 2 to 6 times better on regular pairs and 16 to 27 times better on the 790 pairs (rank-1 residual in bright voxels, 5 contrast-pairs tested), and are smooth along all three axes.
5. **Subjects.** 16 pairs, 14 distinct persons if the two `790` files are second scans of `jiangdian_20230705` and `tangbing_20230705` (probable, section 2), 15 if tangbin and tangbing are different people. After the development subject (jiangyizhou) that leaves 13 (strict) to 14 test candidates, so the "at least 12" target leaves room for exactly one more development/tuning subject in the strict reading.
6. **Pilot objects carry over.** On the native grid the pilot's frozen support file (`configs/sg2x2/init_support_af16_jyz.npz`, md5 310a6520...) is shape-compatible with every converted pair and covers 99.98 % of jiangyizhou's own bright voxels; its coverage of the other 15 subjects' image energy has median 94.6 %, minimum 87.8 %. The pilot's AF16 mask (2,966 points, 15.19x) loads unchanged through `MRIDataset` for converted pairs (dry run, section 3.4).
7. **FLAIR plane.** The acquired FLAIR PE plane is bit-identical across the 14 regular pairs (34,259 of 45,056 block points, 76.04 %) and across the two 790 pairs (33,035, 73.32 %). The h5 FLAIR plane differs from the CFL export of the same exam in the lowest ky columns (section 5).
8. **790 pairs** differ in protocol (50 % slab oversampling, different FLAIR shutter), have the worst stored CSM, and are probable duplicates of two other subjects.
9. **GPU cost** at the pilot's grid size (voxel factor 1.00, 12 instead of 26 coils): about 4.0 GPU-hours per subject and seed (six runs); 12 test subjects x 3 seeds = 144 GPU-hours.

## 1. Per-pair table

File layout used in all tables: `kspace` and `csm` are `(256, 12, NY, 256)` complex64 = axis 0 readout RO (image domain in the file), axis 1 coil, axis 2 partition SPE (NY lines), axis 3 phase PE. After the conversion of section 3 the volume is `(coil, RO 256, PE 256, SPE 176)`, `readout_axis = 0`, `phase_axes = [1, 2]`.

Definitions. Acquired point = PE-plane amplitude above 1e-12 of the plane maximum (acquired points are at least 0.1 absolute, unacquired ones at most 1e-9, ratio between the two classes at least 2.4e8 in every file). Block = rows `[NY-176, NY)` of the SPE axis, all 256 PE columns. DC = argmax of the energy summed over coils and readout. Head mask = coil-combined magnitude above 0.05 times its P99.9, image from `ifft3c` after the axis-0 FFT, combined with the stored CSM as `sum_c conj(S_c) I_c` (the trainer's rule). Registration shift = integer translation that maximises the cross-correlation (mask xcorr of the two head masks, edge xcorr of Gaussian gradient magnitudes, search window +-12), positive = FLAIR content at a higher index than T1.

### Table A: identity and file health

| # | subject (key) | exam date | start | exam no. | name tag | array (file) | marker T1 / FLAIR | opens, both datasets read, finite, complex64 (T1 / FLAIR) | dead coils, zero readout slices (T1 / FLAIR) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | jiangyizhou_20230705 | 2023-07-05 | 15:28:18 | 1450 | (none) | (256, 12, 220, 256) | yes / no | ok / ok | 0,0 / 0,0 |
| 2 | huangyuwei_20230705 | 2023-07-05 | 16:09:27 | 1451 | (none) | (256, 12, 220, 256) | yes / yes | ok / ok | 0,0 / 0,0 |
| 3 | jiangdian_20230705 | 2023-07-05 | 14:45:38 | 1449 | (none) | (256, 12, 220, 256) | no / no | ok / ok | 0,0 / 0,0 |
| 4 | sunxinyi_20230705 | 2023-07-05 | 16:47:17 | 1452 | (none) | (256, 12, 220, 256) | yes / no | ok / ok | 0,0 / 0,0 |
| 5 | tangbing_20230705 | 2023-07-05 | 13:23:43 | 1448 | (none) | (256, 12, 220, 256) | yes / no | ok / ok | 0,0 / 0,0 |
| 6 | 790_jiangdian_20230706 | 2023-07-06 | 10:53:12 | 3356 | 790 | (256, 12, 264, 256) | yes / no | ok / ok | 0,0 / 0,0 |
| 7 | 790_tangbin_20230706 | 2023-07-06 | 09:06:57 | 3355 | 790 | (256, 12, 264, 256) | yes / no | ok / ok | 0,0 / 0,0 |
| 8 | liaozhenghan_20231030 | 2023-10-30 | 14:16:03 | 1576 | (none) | (256, 12, 220, 256) | yes / no | ok / ok | 0,0 / 0,0 |
| 9 | zhoushuo_20231030 | 2023-10-30 | 16:18:05 | 1579 | (none) | (256, 12, 220, 256) | yes / no | ok / ok | 0,0 / 0,0 |
| 10 | shiyixuan_20231101 | 2023-11-01 | 16:32:20 | 1587 | (none) | (256, 12, 220, 256) | yes / yes | ok / ok | 0,0 / 0,0 |
| 11 | zhangyangming_20231101 | 2023-11-01 | 14:43:41 | 1584 | (none) | (256, 12, 220, 256) | yes / yes | ok / ok | 0,0 / 0,0 |
| 12 | yangkun_20231102 | 2023-11-02 | 13:18:58 | 1588 | (none) | (256, 12, 220, 256) | yes / yes | ok / ok | 0,0 / 0,0 |
| 13 | zhangxiqian_20231102 | 2023-11-02 | 14:21:31 | 1589 | (none) | (256, 12, 220, 256) | yes / no | ok / ok | 0,0 / 0,0 |
| 14 | chenshuaibao_20231106 | 2023-11-06 | 11:41:49 | 1598 | (none) | (256, 12, 220, 256) | yes / no | ok / ok | 0,0 / 0,0 |
| 15 | lishengbo_20231106 | 2023-11-06 | 15:48:03 | 1601 | (none) | (256, 12, 220, 256) | no / no | ok / ok | 0,0 / 0,0 |
| 16 | litianyu_20231107 | 2023-11-07 | 13:10:10 | 1602 | (none) | (256, 12, 220, 256) | no / no | ok / ok | 0,0 / 0,0 |

Notes. T1 and FLAIR files of a pair share one file name but are different files (no two of the 32 files are byte-identical; T1 and FLAIR CSMs differ, relative rms difference 1.33 to 1.57). Marker files: 13 of the 16 T1 files and 4 of the 16 FLAIR files have one; the FLAIR tree also holds one orphan marker (`yexinting_20230531`, no h5 beside it). The T1 tree holds 13 more T1-only files (May 2023, `(256, 12, 166, 256)`, 7 of them with a marker) that have no FLAIR and are not part of the set. All datasets are contiguous and uncompressed, storage size equals nominal size in all 45 files. The markers are binary uploader state (not parseable) from an interrupted cloud upload.

### Table B: layout, sampling, DC

| subject | zero-padded SPE lines (low, high) T1 / FLAIR | acquired block (SPE x PE) | T1 block occupancy | FLAIR block occupancy (points) | DC (ax2, ax3) T1 / FLAIR | DC minus array centre (ax2) | DC minus block centre (ax2, ax3) T1 / FLAIR | readout DC after the axis-0 FFT, T1 / FLAIR (centre 128) |
|---|---|---|---|---|---|---|---|---|
| jiangyizhou_20230705 | [44, 0] / [44, 1] | 176 x 256 | 1.0000 | 0.7604 (34259) | (132, 128) / (132, 128) | +22 | (0, 0) / (0, 0) | 125 / 127 |
| huangyuwei_20230705 | [44, 0] / [44, 1] | 176 x 256 | 1.0000 | 0.7604 (34259) | (132, 128) / (132, 128) | +22 | (0, 0) / (0, 0) | 126 / 127 |
| jiangdian_20230705 | [44, 0] / [44, 1] | 176 x 256 | 1.0000 | 0.7604 (34259) | (132, 128) / (132, 128) | +22 | (0, 0) / (0, 0) | 126 / 127 |
| sunxinyi_20230705 | [44, 0] / [44, 1] | 176 x 256 | 1.0000 | 0.7604 (34259) | (132, 128) / (132, 128) | +22 | (0, 0) / (0, 0) | 125 / 127 |
| tangbing_20230705 | [44, 0] / [44, 1] | 176 x 256 | 1.0000 | 0.7604 (34259) | (132, 128) / (132, 128) | +22 | (0, 0) / (0, 0) | 125 / 127 |
| 790_jiangdian_20230706 | [88, 0] / [88, 1] | 176 x 256 | 1.0000 | 0.7332 (33035) | (176, 129) / (176, 128) | +44 | (0, 1) / (0, 0) | 126 / 128 |
| 790_tangbin_20230706 | [88, 0] / [88, 1] | 176 x 256 | 1.0000 | 0.7332 (33035) | (176, 129) / (176, 127) | +44 | (0, 1) / (0, -1) | 126 / 127 |
| liaozhenghan_20231030 | [44, 0] / [44, 1] | 176 x 256 | 1.0000 | 0.7604 (34259) | (132, 128) / (132, 128) | +22 | (0, 0) / (0, 0) | 126 / 127 |
| zhoushuo_20231030 | [44, 0] / [44, 1] | 176 x 256 | 1.0000 | 0.7604 (34259) | (132, 128) / (132, 128) | +22 | (0, 0) / (0, 0) | 125 / 127 |
| shiyixuan_20231101 | [44, 0] / [44, 1] | 176 x 256 | 1.0000 | 0.7604 (34259) | (132, 128) / (132, 128) | +22 | (0, 0) / (0, 0) | 125 / 127 |
| zhangyangming_20231101 | [44, 0] / [44, 1] | 176 x 256 | 1.0000 | 0.7604 (34259) | (132, 128) / (132, 128) | +22 | (0, 0) / (0, 0) | 125 / 127 |
| yangkun_20231102 | [44, 0] / [44, 1] | 176 x 256 | 1.0000 | 0.7604 (34259) | (132, 128) / (132, 128) | +22 | (0, 0) / (0, 0) | 125 / 127 |
| zhangxiqian_20231102 | [44, 0] / [44, 1] | 176 x 256 | 1.0000 | 0.7604 (34259) | (132, 128) / (132, 128) | +22 | (0, 0) / (0, 0) | 125 / 127 |
| chenshuaibao_20231106 | [44, 0] / [44, 1] | 176 x 256 | 1.0000 | 0.7604 (34259) | (132, 128) / (132, 128) | +22 | (0, 0) / (0, 0) | 125 / 127 |
| lishengbo_20231106 | [44, 0] / [44, 1] | 176 x 256 | 1.0000 | 0.7604 (34259) | (132, 128) / (132, 128) | +22 | (0, 0) / (0, 0) | 125 / 127 |
| litianyu_20231107 | [44, 0] / [44, 1] | 176 x 256 | 1.0000 | 0.7604 (34259) | (132, 128) / (132, 128) | +22 | (0, 0) / (0, 0) | 126 / 127 |

Notes. (i) T1 has no shutter inside the block (all 45,056 points acquired) but 20 % (220-grid) or 33 % (264-grid) of the array plane is zero padding, so "T1 has no shutter" holds only on the native block. (ii) The FLAIR shutter is an ellipse covering 76.04 % (33,035 / 45,056 = 73.32 % for the 790 pairs) of the block; its plane is bit-identical across the subjects of a protocol group (checked against one reference per group). FLAIR data span SPE rows 44..218 (one row short of T1's block at the high end, hence the "1" in the padding column) and PE columns 1..254 (2..253 for the 790 pairs), which is part of the shutter ellipse; no FLAIR point lies outside T1's block. (iii) DC minus block centre is zero on the SPE axis for every pair and within one sample on the PE axis. The readout DC after the axis-0 FFT is 3 to 0 samples below the centre (125 to 128), as in the CFL exports of the pilot (126 / 127 for huangyuwei). (iv) Readout axis decision: the all-or-nothing test of `prep_01_inspect.py` is decisive on FLAIR only: axis 0 has partial-line fraction 0.000 in all 16 FLAIR files (zero-line fraction 0.392 for the 14 regular pairs, 0.511 for the 790 pairs), axis 2 has 0.984 to 0.992 and axis 3 has 0.663 to 0.755. On T1 the test cannot discriminate (the zero padding is a full-width strip), so T1 takes the geometry of its FLAIR and the axis-peak test (peak over median of the energy profile along the axis, all 32 files: axis 0 2.1 to 10.1, axes 2 and 3 169 to 2523, i.e. axis 0 is the image-domain axis, axes 2 and 3 are k-space axes).

### Table C: CSM, image, alignment

| subject | stored CSM: SoS in head, mean (share within 5 % of 1) T1 / FLAIR | stored CSM rank-1 residual, whole head T1 / FLAIR | same, bright voxels (stored-CSM combined magnitude at least 0.35 P99.9; bins .35-.60 and .60-1) T1 / FLAIR | head fraction of the box T1 / FLAIR | T1-FLAIR head-mask Dice raw / cleaned | head centroid offset FLAIR minus T1 (ax0, ax2, ax3), voxels | registration shift of FLAIR rel. T1: mask xcorr / edge xcorr (ax0, ax2, ax3) |
|---|---|---|---|---|---|---|---|
| jiangyizhou_20230705 | 0.998 (0.991) / 0.998 (0.994) | 0.065 / 0.032 | 0.050, 0.029 / 0.025, 0.013 | 0.378 / 0.266 | 0.805 / 0.838 | (+22.1, +0.2, +3.5) | (0, 0, 0) / (0, 0, 0) |
| huangyuwei_20230705 | 0.998 (0.993) / 0.999 (0.999) | 0.041 / 0.024 | 0.025, 0.021 / 0.021, 0.010 | 0.267 / 0.237 | 0.887 / 0.906 | (+9.1, -0.3, -0.2) | (0, 0, 0) / (1, 0, 1) |
| jiangdian_20230705 | 0.996 (0.987) / 0.999 (0.996) | 0.046 / 0.028 | 0.028, 0.025 / 0.021, 0.011 | 0.303 / 0.248 | 0.836 / 0.848 | (+18.7, +1.1, -1.4) | (1, 0, 0) / (1, 0, 1) |
| sunxinyi_20230705 | 0.997 (0.988) / 0.999 (0.997) | 0.039 / 0.037 | 0.026, 0.024 / 0.026, 0.013 | 0.305 / 0.266 | 0.880 / 0.900 | (+9.4, +0.7, +2.0) | (1, 0, 0) / (1, 0, 1) |
| tangbing_20230705 | 0.998 (0.994) / 0.999 (0.997) | 0.056 / 0.041 | 0.039, 0.023 / 0.025, 0.012 | 0.385 / 0.309 | 0.856 / 0.881 | (+15.3, +1.0, +0.0) | (0, 0, 0) / (0, 0, 0) |
| 790_jiangdian_20230706 | 0.999 (1.000) / 1.000 (1.000) | 0.299 / 0.118 | 0.267, 0.165 / 0.112, 0.075 | 0.312 / 0.259 | 0.848 / 0.874 | (+13.3, +2.9, -0.2) | (1, 0, 0) / (2, 0, 1) |
| 790_tangbin_20230706 | 0.999 (1.000) / 0.999 (1.000) | 0.371 / 0.204 | 0.313, 0.182 / 0.187, 0.089 | 0.419 / 0.326 | 0.789 / 0.839 | (+18.4, +0.8, -0.7) | (0, 0, 1) / (0, 0, 1) |
| liaozhenghan_20231030 | 0.995 (0.983) / 0.999 (0.998) | 0.070 / 0.045 | 0.040, 0.023 / 0.029, 0.016 | 0.325 / 0.272 | 0.847 / 0.859 | (+14.6, +1.1, -0.2) | (0, 0, 0) / (0, 0, 1) |
| zhoushuo_20231030 | 0.997 (0.989) / 0.999 (0.997) | 0.064 / 0.034 | 0.041, 0.023 / 0.030, 0.013 | 0.348 / 0.272 | 0.841 / 0.862 | (+15.8, -0.0, +3.1) | (0, 0, 0) / (0, 0, 1) |
| shiyixuan_20231101 | 0.983 (0.931) / 0.999 (0.999) | 0.031 / 0.034 | 0.020, 0.016 / 0.024, 0.011 | 0.306 / 0.258 | 0.859 / 0.872 | (+12.5, +0.8, +1.4) | (0, 0, 0) / (1, 0, 2) |
| zhangyangming_20231101 | 0.997 (0.994) / 0.999 (0.996) | 0.092 / 0.036 | 0.069, 0.041 / 0.026, 0.011 | 0.385 / 0.286 | 0.815 / 0.872 | (+12.5, +0.1, +3.6) | (0, 0, 0) / (0, 0, 1) |
| yangkun_20231102 | 0.996 (0.979) / 0.999 (0.998) | 0.070 / 0.056 | 0.053, 0.030 / 0.041, 0.021 | 0.417 / 0.296 | 0.795 / 0.844 | (+16.1, -0.0, +7.8) | (0, 0, 1) / (0, 0, 1) |
| zhangxiqian_20231102 | 0.998 (0.993) / 0.999 (0.998) | 0.086 / 0.031 | 0.063, 0.036 / 0.026, 0.013 | 0.327 / 0.273 | 0.855 / 0.898 | (+9.4, +0.9, +0.7) | (1, 0, 0) / (1, 0, 1) |
| chenshuaibao_20231106 | 0.997 (0.991) / 0.999 (0.996) | 0.060 / 0.032 | 0.036, 0.022 / 0.024, 0.011 | 0.313 / 0.262 | 0.854 / 0.874 | (+13.5, -0.1, +0.7) | (0, 0, 1) / (1, 0, 1) |
| lishengbo_20231106 | 0.990 (0.958) / 0.999 (0.996) | 0.034 / 0.039 | 0.021, 0.014 / 0.026, 0.011 | 0.370 / 0.299 | 0.848 / 0.857 | (+17.0, +0.9, +2.9) | (1, 0, 1) / (1, 0, 1) |
| litianyu_20231107 | 0.998 (0.995) / 0.998 (0.992) | 0.113 / 0.039 | 0.096, 0.052 / 0.035, 0.016 | 0.438 / 0.276 | 0.751 / 0.823 | (+16.8, +1.0, +9.5) | (1, 0, 1) / (1, 0, 1) |

Ranges over the 14 regular pairs: head fraction T1 0.267 to 0.438 (median 0.337), FLAIR 0.237 to 0.309 (0.272); Dice raw 0.751 to 0.887 (0.847), cleaned (largest component, holes filled) 0.823 to 0.906 (0.867); whole-head rank-1 residual T1 0.031 to 0.113 (0.062), FLAIR 0.024 to 0.056 (0.035); bright-voxel residual (bin .35-.60) T1 0.020 to 0.096 (0.040), FLAIR 0.021 to 0.041 (0.026); background median over P99.9 T1 0.007 to 0.026, FLAIR 0.013 to 0.018.

Notes. (i) "Rank-1 residual" = `sum |I_c - S_c x| ^2 / sum |I_c| ^2` inside the head with `x = sum conj(S_c) I_c / sum |S_c| ^2`; low-signal voxels are noise dominated, hence the bright-voxel columns. A mask-free check with the other contrast's maps gives 0.72 to 0.92, so the maps are contrast-specific. (ii) The head centroid offset along axis 0 (+9 to +22 voxels) is a mask-definition effect (the T1 mask contains more neck), not a misregistration: the cross-correlation shifts are at most 1 voxel (mask) and 2 voxels (edge) in every pair. The edge cross-correlation peak is only 1.04 to 1.17 times its second peak, so single-voxel entries are at the resolution of the estimate. (iii) The head fraction of the box depends on SNR through the 0.05 x P99.9 threshold: T1 of litianyu, yangkun and 790_tangbin (0.42 to 0.44, background 0.022 to 0.026) is dominated by speckle outside the head; largest connected component share of the T1 head mask is 0.865 to 0.995 (median 0.972). This is the spec's head-only definition applied to the stored-CSM image; cropping to the native grid does not change it (jiangyizhou T1: 0.378 on the 220 grid, 0.378 on the native grid). (iv) Direct `ifft3c` of the raw array (no axis-0 FFT) gives head fraction T1 0.105 to 0.148 and rank-1 residual 0.29 to 0.52 in all 16 pairs.

### Table D: usable?

Criteria used: both sequences present, files intact, T1 and FLAIR on the same native grid with the expected geometry, DC at the block centre, CSM obtainable that fits the data, T1-FLAIR shift at most 2 voxels. "Usable" below assumes the CSM is re-estimated from the converted k-space (section 3.4 and 4(f)); with the stored CSM none of the 16 is a good choice, the two 790 pairs are clearly worse.

| subject | usable | reason / caveat |
|---|---|---|
| jiangyizhou_20230705 | yes | development subject; same exam as the pilot's CFL export (RSS NCC 0.942 T1 / 0.950 FLAIR at identity orientation, zero shift) |
| huangyuwei_20230705 | yes | has a CFL export (NCC 0.949 / 0.960); an unpaired T1-only exam of the same person exists (20230531) |
| jiangdian_20230705 | yes | has a CFL export; probable same person as 790_jiangdian_20230706 (they must stay in one split) |
| sunxinyi_20230705 | yes | has a CFL export |
| tangbing_20230705 | yes | has a CFL export; probable same person as 790_tangbin_20230706 and tangbin_20230531 (T1-only) |
| 790_jiangdian_20230706 | yes only with re-estimated CSM; not an independent subject | stored CSM fits badly (bright-voxel residual 0.27 T1 / 0.11 FLAIR against at most 0.10 / 0.04 elsewhere) and stripes the combined image; different protocol (50 % slab oversampling, FLAIR shutter 73.3 %); FLAIR displaced-column energy 8.1e-4 (regular pairs 4e-6); probable duplicate of jiangdian_20230705 |
| 790_tangbin_20230706 | yes only with re-estimated CSM; not an independent subject | same as above (residual 0.31 / 0.19, displaced energy 8.5e-4); T1 and FLAIR DC differ by 2 samples on the PE axis (129 / 127); probable duplicate of tangbing_20230705 |
| liaozhenghan_20231030 | yes | none |
| zhoushuo_20231030 | yes | none |
| shiyixuan_20231101 | yes | stored T1 CSM SoS dips inside the head (share within 5 % of 1 is 0.931, p1 0.625); irrelevant with fresh maps |
| zhangyangming_20231101 | yes | none |
| yangkun_20231102 | yes | T1 head-mask fraction 0.417 (low SNR speckle) |
| zhangxiqian_20231102 | yes | none |
| chenshuaibao_20231106 | yes | none |
| lishengbo_20231106 | yes | stored T1 CSM SoS share within 5 % of 1 is 0.958 |
| litianyu_20231107 | yes, with a caution | noisiest T1 (background 0.026 of P99.9, head-mask fraction 0.438, P99.9 scale 2.6e5), stored-CSM residual highest of the regular pairs (0.113 / 0.096); fresh ESPIRiT residual 0.029 / 0.012 |

## 2. Duplicates and identity

### 2.1 Names, dates, exam counters

Same-name groups in the whole T1 + FLAIR trees (the pair set is bold):

| person (name) | files |
|---|---|
| jiangyizhou | 20230529 exam 1389 17:18 (T1 only, 166 grid); **20230705 exam 1450 15:28 (pair)** |
| huangyuwei | 20230531 exam 1398 15:48 (T1 only, 166 grid); **20230705 exam 1451 16:09 (pair)** |
| jiangdian | **20230705 exam 1449 14:45 (pair)**; **790 20230706 exam 3356 10:53 (pair)** |
| tangbin / tangbing (names differ by one letter) | tangbin 20230531 exam 1395 13:28 (T1 only); **tangbing 20230705 exam 1448 13:23 (pair)**; **790 tangbin 20230706 exam 3355 09:06 (pair)** |

All other 10 pair subjects appear once (7 exam days in the pair set: 2023-07-05 five subjects, 2023-07-06 two, 2023-10-30 two, 2023-11-01 two, 2023-11-02 two, 2023-11-06 two, 2023-11-07 one). Same-name, same-person duplicates inside the pair set: **jiangdian (two files)**. Name-variant duplicate: **tangbing / 790 tangbin**.

### 2.2 Imaging evidence (circumstantial, rigid registration of T1 volumes)

Every 790 volume was registered (6 degrees of freedom, normalised cross-correlation, 7 starts, three resolutions) to the T1 volume of every other subject; the table gives the two best partners of each row (NCC / head-mask Dice after registration):

| 790 volume | best partner | second | third | row mean (sd) |
|---|---|---|---|---|
| 790_jiangdian_20230706 | jiangdian_20230705: 0.625 / 0.931 | sunxinyi_20230705: 0.552 / 0.898 | tangbing_20230705: 0.540 / 0.837 | 0.439 (0.089) |
| 790_tangbin_20230706 | tangbing_20230705: 0.664 / 0.949 | zhangyangming_20231101: 0.575 / 0.898 | yangkun_20231102: 0.517 / 0.918 | 0.481 (0.062) |

In both rows the same-name candidate is first on both NCC and head-outline Dice with the largest margin (about 0.07 to 0.09 NCC, 0.03 Dice), but the absolute NCC is low for a same-person match (different system, different day, striped CSM-combined image), so this supports the name-based reading and does not prove it. Among the 14 regular volumes the closest pairs are different names (zhangxiqian / zhoushuo 0.773, jiangyizhou / litianyu 0.768, tangbing / zhangyangming 0.733; over all 120 volume pairs mean 0.560, sd 0.095), and no two regular volumes are alike enough to be a repeat scan. No two of the 32 pair files are byte-identical.

### 2.3 Scanner / protocol

The h5 files carry no scanner metadata (no attributes). From the names: tag `790` on two files, untagged on 14. The exam counters support two consoles: untagged exams run 1381-1398 (May), 1448-1452 (07-05), 1576-1602 (Oct-Nov); the two tagged exams are 3355 and 3356 (07-06). The tagged pair also has a different acquisition: SPE array 264 instead of 220 (50 % instead of 25 % slab oversampling over 176 partitions), FLAIR shutter 73.3 % instead of 76.0 % of the block. From the project notes (`jyz-t1flair-acquisition-params`, read from the same-day huangyuwei protocol XML) the untagged 2023-07-05 exams were acquired on a UIH uPMR 790 (PET/MR), 3 T, 32-channel head coil, so the name tag `790` is not by itself the scanner model; the tagged exams are a second console or system of unrecorded model. Treat "untagged" and "790" as two acquisition groups.

## 3. What converting a pair takes

### 3.1 Meaning of "trainer", "n4" and "jd_kfix"

* `*_T1.mat`, `*_FLAIR.mat` in `T2_csm_ksp_n4_mat`: MATLAB v7.3 copies of the h5 (`kspace`, `csm`; axes reversed by MATLAB storage), bit-identical to the h5 (coil 0 compared: max difference 0). They are still hybrid and garble under `ifft3c`.
* `*_T1_trainer.mat`, `*_FLAIR_trainer.mat`: the same data after a centred 1-D FFT along axis 0 (ortho), coil last, variables renamed `ksp` / `map`; written by `/data0/congcong/code/GS/3DGS/GS_TMI/extract_S01_t1_flair.py` (docstring: discovered 2026-04-30, the earlier extraction left axis 0 in image domain and `ifft3c` garbled the volume). Checked on coil 0: relative max difference to `fft1c_axis0(h5)` 9.2e-8 (T1) and 5.4e-8 (FLAIR). On disk the trainer file is `(256 PE, 220 SPE, 256 RO, 12)`, which is why the old configs read `readout_axis: 2, phase_axes: [0, 1]`.
* `jd_kfix/jiangdian_{T1,FLAIR}_kfix.h5` (made by `.../3dgsVC/preprocess_jd_kfix.py`): the same axis-0 FFT for jiangdian_20230705 in h5 layout, unnormalised (all values 16 = sqrt(256) times the ortho result, relative misfit 2.5e-8), `csm` unchanged. It is prior art for exactly the conversion step 2 below.
* "n4": no script defines it. Evidence: `T2_csm_ksp_n4/Train_3dBrain_liaozhenghan_...h5` has a `csm` bit-identical to the plain T2 and FLAIR directory files (first two coils compared), but a `kspace` that equals the plain k-space image times a smooth field (ratio 0.65 to 1.69 between the 1st and 99th percentile in the head, a cubic polynomial explains 83.5 % of the log-ratio variance, correlation of magnitudes 0.78): consistent with an N4 bias-field correction applied to the data (inference). Only that one subject exists in `T2_csm_ksp_n4`. The huangyuwei T1/FLAIR trainer files are not N4-corrected; they sit in the `..._n4_mat` directory because the extract script copied that directory's layout. (The plain `T2_csm_ksp/liaozhenghan` file is identical to the FLAIR file on the first two coils compared, as the T1T2 README already noted for 12 subjects by md5.)

### 3.2 DC

In all 16 pairs the DC is at the centre of the acquired block (SPE: 132 for NY = 220, 176 for NY = 264; PE 127 to 129). Relative to the array centre that is +22 (+44): the 44 (88) zero lines are all at the low-index end. This is the mask-misregistration of the old bundle (worktree `.claude/memory.md` section 8: "data DC at array index axis1=132, matrix centre=110"). Cropping the block puts the DC at SPE index 88 = 176 // 2 and PE index 128 within one sample, so `center_kspace_dc` is not needed and no roll is applied. The readout DC after the axis-0 FFT is at index 125 to 128 (centre 128); the pilot's CFL exports show 126 to 127, so this is the same convention.

### 3.3 Conversion recipe (identical for all 16 pairs, both contrasts)

1. Read `kspace` `(256, 12, NY, 256)`, move the coil axis first: `(12, 256, NY, 256)` = (coil, RO, SPE, PE).
2. Centred 1-D FFT along RO (`fftshift(fft(ifftshift(x, 1), 1, norm="ortho"), 1)`): hybrid array to pure k-space.
3. Crop the SPE axis to the acquired rows `[NY - 176, NY)` (44..219 for 220, 88..263 for 264): `(12, 256, 176, 256)`. This is the DC centring; it also removes the known-zero padding.
4. Swap the two PE axes: `(12, 256, 256, 176)` = (coil, RO, PE, SPE) = the pilot's frame (`ky` = PE 256, `kz` = SPE 176). Write `ksp_<T1|FLAIR>.h5`, key `kspace`, chunks `(1, 32, 256, 176)` as `prep_04_export.py` does. Config: `readout_axis: 0`, `phase_axes: [1, 2]`, `presampled: false`, `skip_readout_check: false`.
5. CSM, key `map`, `(12, 256, 256, 176)` coil first, per contrast. Two routes (section 4(f)): (a) fresh ESPIRiT, the pilot recipe, from the converted k-space: write it as CFL `(256, 256, 176, 12)`, `bart ecalib -t 0.001 -c 0.0 -r 24 -m 1`, then `prep_05_csm_to_h5.py` (SoS is 1 everywhere); (b) the stored `csm` resampled from NY to 176 along SPE (centred FFT, central 176 coefficients, inverse FFT, times sqrt(176 / NY)), axes swapped, renormalised to unit SoS where SoS exceeds 0.05. The loader accepts the keys `map` and `csm`; it does not normalise the maps, it only refuses an all-zero map.
6. Mask: any `(256, 256, 176)` readout-constant mask; `MRIDataset` validates that every PE line is all-or-nothing along readout. The pilot's `mask_af16_3d.h5` passes.
7. Normalisation: the loader's own (`normalize_kspace: true`, P99.9 of the ground-truth magnitude); the scale is data dependent (T1 2.9e5 to 3.3e5, FLAIR 3.4e5 to 3.9e5 in the three dry runs). The unnormalised-FFT variant of `jd_kfix` (amplitudes 16 times larger) is cancelled by it.
8. FLAIR only: the known-zero region. With `known_zero: true` the repo's estimator (`paired_gs/data/known_zero.py`, estimated from the sampled lines) gives for the converted FLAIR with the pilot mask: ellipse (124, 88, 1.0), `frac_z` 0.2391, 231 sampled zero lines of which 204 explained, 20 outliers none significant. The pilot (CFL FLAIR) gave a 20-line padding cut plus ellipse (128, 88, 1.0), `frac_z` 0.2423, 233 of 233 explained. An explicit support map (`known_zero_support`) is an alternative because the plane is identical across subjects of a protocol group.

No export script was written into the repo. Working reference code for steps 1 to 5(b) is `convert()` in `/tmp/claude-1002/sg2x2_inventory/x08_loader_dryrun.py`, and for the BART route (step 5(a)) `x23_bart_ecalib_test.py` (CFL writing, `bart ecalib` call, reading the map back).

One export script serves all pairs: the only per-file input is the path; `lo = NY - 176` holds for both array sizes; the script can assert that the T1 block is fully acquired, that nothing is acquired outside it, and that the FLAIR plane equals its group reference. CPU time per pair about 35 s for the conversion plus 2 x 57 s for ESPIRiT (12 coils, 24 threads); disk 4.4 GB per pair for the four h5 files (1.107 GB each), 70.9 GB for 16 pairs, plus 2.2 GB of CFL intermediates per contrast if ESPIRiT is run via BART files.

### 3.4 What the dry run measured (scratch only, CPU)

The recipe was run on jiangyizhou, 790_tangbin and litianyu (`x08_loader_dryrun.py`) and loaded through the repo's `MRIDataset` exactly as `configs/sg2x2/af16_jyz_joint.yaml` does, with the pilot's `mask_af16_3d.h5`. For all three: `volume_shape=(256, 256, 176) readout=0 phase=(1, 2) coils=12`, the mask validates, "actual acceleration 15.19x", 2,966 mask points. Head fractions of the box on the native grid: jiangyizhou 0.384 / 0.266 (T1 / FLAIR), 790_tangbin 0.420 / 0.326, litianyu 0.441 / 0.277. The trainer itself was not run (no free GPU): the steps of `JointTrainer._setup_data` beyond `MRIDataset` and `estimate_known_zero` were not exercised.

Fresh versus stored CSM on the converted data (rank-1 residual in bins of the RSS magnitude, which does not depend on either map set; bright bins .35-.60 / .60-1.5 of the RSS P99.9; `x23_bart_ecalib_test.py`, `x24`, `x25`; the stored-CSM numbers of Table C are binned on the stored-CSM combined magnitude instead, so they differ slightly from the "stored" column here):

| pair, contrast | stored (resampled, renormalised) | fresh ESPIRiT | neighbour phase difference of the complex GT along RO, stored / fresh (rad) |
|---|---|---|---|
| jiangyizhou T1 | 0.050 / 0.034 | 0.021 / 0.010 | 0.607 / 0.140 |
| jiangyizhou FLAIR | 0.026 / 0.014 | 0.007 / 0.003 | 0.587 / 0.063 |
| 790_tangbin T1 | 0.384 / 0.301 | 0.024 / 0.011 | 0.904 / 0.143 |
| 790_tangbin FLAIR | 0.208 / 0.129 | 0.012 / 0.005 | not measured |
| litianyu T1 | 0.100 / 0.068 | 0.029 / 0.012 | not measured |

Phase difference of the complex GT along the two PE axes is 0.06 to 0.19 rad with either map set, so only the readout axis is affected. The stored maps themselves, neighbour to neighbour along RO / PE / SPE: |dphase| 0.55 to 0.84 rad along RO against 0.01 to 0.05 along both PE axes, relative magnitude change 0.16 to 0.36 against 0.01 to 0.04 (5 map sets checked); fresh BART maps: 0.007 to 0.018 rad and 0.007 to 0.013 along all three axes. Reading: the stored maps were estimated independently per readout slice (inference from the slice-to-slice jumps; the file records no provenance), which a 3-D complex ground truth and a phase-TV prior do not tolerate. The stripes of the 790 volumes are in the CSM-combined image, not in the data: the RSS image of the same file is clean and so is the CFL export (figure `figs/rss_vs_csm_790_tangbi_jiangyizho.png`, `figs/cfl_vs_h5_T1_sagittal.png`).

## 4. Facts for the decision (options, not decisions)

### (a) How many usable distinct subjects

Unit of analysis is the person (spec 5.3). Counts, with jiangyizhou as the development subject and the CSM re-estimated:

| reading | pair files | distinct persons | test candidates after dev = jiangyizhou | after 1 more dev/tuning subject | after 2 more |
|---|---|---|---|---|---|
| strict: both 790 files are second scans (jiangdian, tangbing / tangbin) | 16 | 14 | 13 | 12 | 11 |
| literal names: tangbin is not tangbing, 790_jiangdian = jiangdian | 16 | 15 | 14 | 13 | 12 |
| 790 pairs dropped | 14 | 14 | 13 | 12 | 11 |
| every file a subject (ignores the duplicates) | 16 | 16 | 15 | 14 | 13 |

Consequences. The target "at least 12" is reachable with one development subject plus at most one more tuning subject in the strict reading (two more in the literal reading). The two 790 files add no independent person in the strict reading; they can serve as same-person replicates on another system (their person must not be split across dev and test). T1-only files (13) and the May exams are unusable for a pair. With the stored CSM no pair is a good choice (section 3.4), so "usable" is conditional on re-estimating the maps.

### (b) Development subject(s) and the remaining test subjects

* jiangyizhou_20230705 is in the set (row 1) and usable: same exam as the pilot's CFL export, orientation identical (all 8 flip combinations tested, the identity one wins, RSS NCC 0.942 / 0.950, shift (0, 0, 0)), same T1 plane (IoU 1.000), FLAIR plane IoU 0.942 (section 5), 12 instead of 26 coils. The pilot's numbers (A-B +0.006 / -0.022 dB, C-D +1.563 / +1.049 dB) were obtained on the CFL export and are not directly comparable with h5-based subjects; if comparability with the pilot matters, the development subject has to be re-run on the h5 export (about 4 GPU-hours per seed). The development seeds 43 / 44 already queued in `scripts/sg2x2_dev_seeds.txt` run on the pilot data path and are unaffected.
* Frozen support: `configs/sg2x2/init_support_af16_jyz.npz` (256 x 256 x 176, support fraction 0.4273, derived from the development subject's zero-filled union) fits the native frame of every pair when volumes are ordered (RO, PE, SPE). Coverage by it (union head of T1 and FLAIR from the fully sampled ground truth with the stored CSM, a proxy for the zero-filled rule; U = max of the two magnitudes, each normalised by its own P99.9):

| subject | head voxels (U > 0.05) | bright voxels (U > 0.20) | image energy | support voxels that are head |
|---|---|---|---|---|
| jiangyizhou_20230705 (dev, own) | 0.9406 | 0.9998 | 0.9986 | 0.847 |
| huangyuwei_20230705 | 0.9636 | 0.9842 | 0.9909 | 0.632 |
| jiangdian_20230705 | 0.9385 | 0.9822 | 0.9905 | 0.705 |
| sunxinyi_20230705 | 0.9301 | 0.9697 | 0.9776 | 0.696 |
| tangbing_20230705 | 0.8434 | 0.8978 | 0.9038 | 0.783 |
| 790_jiangdian_20230706 | 0.9297 | 0.9748 | 0.9772 | 0.715 |
| 790_tangbin_20230706 | 0.7999 | 0.9235 | 0.9359 | 0.845 |
| liaozhenghan_20231030 | 0.8689 | 0.9175 | 0.9458 | 0.700 |
| zhoushuo_20231030 | 0.8738 | 0.9135 | 0.9332 | 0.734 |
| shiyixuan_20231101 | 0.9199 | 0.9524 | 0.9626 | 0.693 |
| zhangyangming_20231101 | 0.8291 | 0.9156 | 0.9086 | 0.771 |
| yangkun_20231102 | 0.7948 | 0.8763 | 0.8833 | 0.799 |
| zhangxiqian_20231102 | 0.8788 | 0.9433 | 0.9612 | 0.706 |
| chenshuaibao_20231106 | 0.9014 | 0.9308 | 0.9551 | 0.695 |
| lishengbo_20231106 | 0.8407 | 0.8606 | 0.8782 | 0.759 |
| litianyu_20231107 | 0.7624 | 0.9212 | 0.9205 | 0.795 |

  The other 15 subjects: minimum / median voxel coverage 0.762 / 0.874, bright 0.861 / 0.924, energy 0.878 / 0.946. So 5 to 12 % of the image energy of a test subject lies outside the pilot support; with adaptive operations disabled that part has to be reached by position optimisation. The worst cases are lishengbo, yangkun, tangbing, zhangyangming, litianyu. A new support derived from a different development subject would not cover better in general: each subject's own union head as a support covers the other 15 with median 0.70 to 0.92 and minimum 0.59 to 0.79 (full 16 x 16 matrix in appendix A.6); larger heads (790_tangbin, tangbing, yangkun, litianyu) cover most, at the price of a smaller head share of the support. A union of several development subjects' supports is a third option; none of this uses any test-subject data if the sources are the development subjects only.
* Same-batch candidates for extra tuning subjects (same day and console as the development subject, CFL exports exist for all): huangyuwei, jiangdian, sunxinyi, tangbing (2023-07-05). The Oct-Nov batch has 9 subjects on one console. Splitting strictly by batch (dev = the July batch, test = Oct-Nov) gives 9 test subjects, below 12; the test set then needs some July subjects.
* Test candidates in every reading: the 13 regular pairs other than jiangyizhou (rows 2 to 5 and 8 to 16 of Table A): huangyuwei_20230705, jiangdian_20230705, sunxinyi_20230705, tangbing_20230705, liaozhenghan_20231030, zhoushuo_20231030, shiyixuan_20231101, zhangyangming_20231101, yangkun_20231102, zhangxiqian_20231102, chenshuaibao_20231106, lishengbo_20231106, litianyu_20231107. The 790 files are the only ones whose status depends on the readings above.

### (c) Mask generation on this grid: two conventions, the user must pick

Both apply to the native plane (ky = PE 256, kz = SPE 176, 45,056 points).

| | vdPoisMex (skill `mri-vdpois-mask`, house convention, every brain topic including the pilot) | BART standard Poisson disc (user rule 2026-10-09, `knee-mask-bart-corners-rule.md`: whole (ky, kz) plane, no `-e`) |
|---|---|---|
| generator | MATLAB `vdPoisMex(256, 176, 256, 176, accelx, accely = 4, calib = 24, ellipse = 0, pp = 1)`, `accelx = AF / 4` (`/usr/local/bin/matlab`, mex in `/data0/congcong/code/ESPIRiT/utils`) | `bart poisson -Y 256 -Z 176 -y A -z A -C 24 -v -s SEED` (BART at `/home/congcongliu/src/bart`, CPU) |
| points for "AF16" | the pilot file `mask_vdpois_256x176_af16.mat` has 2,966 points (15.19x); it is byte-identical in the two AF16 topics (md5 2d7ed000...); new draws are time-seeded and cannot be regenerated | whole-plane AF16 = 2,816 points; probe on this plane: A = 3.8 gives 2,857 to 2,873, A = 4.0 gives 2,691 to 2,705; an exact count needs a pool search (the knee line used 120 draws) |
| central region | 16 x 16 fully sampled, 24 x 24 only 81.9 % | 24 x 24 fully sampled in every probe draw |
| density | central 64 x 64: 0.311, outer 32 rows: 0.023 | central 64 x 64: 0.19 to 0.20, outer 32 rows: 0.038 to 0.039 |
| FLAIR shutter | 231 (regular) or 255 (790) mask points land on unacquired FLAIR points, i.e. measured zeros | same mechanism; the count depends on the draw |
| state | ready, loads unchanged | a draw and selection script on the (256, 176) plane has to be written |

Also available but for the zero-padded array only: `/data0/congcong/code/ESPIRiT/utils/mask_vdpois_220x256_af{08,10,12,14,16}.mat` (the AF16 one has 3,659 points, 15.39x over the 220 x 256 array, ACS 24 x 24 at 92.7 %). On the native plane they do not apply. Two choices are independent of the generator: one fixed mask for all subjects (the pilot's way, removes mask-draw variance) or one per subject (spec 4 pairs the seed within a subject only). The memory index line for the BART rule in `MEMORY.md` still describes the first reading (with `-e`, four corners fully sampled); the file body records the same-night correction used in the task text.

### (d) AF definition on this grid, and AF16 in points

AF = whole PE plane points / sampled points (user, 2026-09-14). On the native block the plane is 176 x 256 = 45,056, so exactly AF16 is 2,816 points; the pilot's mask has 2,966 (15.19x on T1). FLAIR: of the 2,966 points 2,735 land on acquired FLAIR points in the 14 regular pairs (231 on unacquired points), giving 16.47x of the plane and 12.53x of FLAIR's own acquired points (790 pairs: 2,711 sampled, 255 unacquired, 16.62x and 12.19x). If the zero-padded arrays were used instead, the whole-array plane would be 220 x 256 = 56,320 (AF16 = 3,520 points) or 264 x 256 = 67,584 (4,224 points), but 20 % or 33 % of those plane points are padding, so the same number of samples would be AF 12.8x or 10.7x of the acquired block; the stated definition is only consistent with the native block.

### (e) GPU cost

Pilot run times (record section 4 of the pilot, score jsons; 26 coils, 3.6M points, 32.9 GB for A and C): A 3,810 s, C 3,261 s, B_T1 1,962 s, D_T1 1,871 s, D_FLAIR 1,701 s, B_FLAIR 6,410 s on a shared card (counted at D_FLAIR's 28 min). Six runs per subject and seed = about 238 min = 4.0 GPU-hours on an idle A100 (an estimate, as the project notes say).

* Native grid: 256 x 256 x 176 = 11,534,336 voxels, identical to the pilot: factor 1.00; the 12 instead of 26 coils should lower the data-consistency cost (not measured, no free card).
* Zero-padded grids: 14,417,920 voxels (factor 1.25) or 17,301,504 (1.50); evaluation on interpolated voxels and the zero lines treated as known zeros.
* Totals at factor 1.00: per seed 4.0 h x N subjects; N = 12: 48 GPU-h per seed, 144 for seeds 42 / 43 / 44; N = 13: 52 / 156; N = 14: 56 / 168. With cards 0 to 4 usable and no sharing, 144 GPU-h is about 29 h of wall time; two trainers on one card stall each other, so this needs idle cards. Optional supplements (1.8M capacity, `INIT=head`) add on top. If the development subject is re-run on the h5 export, add 4.0 h per seed.

### (f) Further choices this inventory surfaced

1. **Grid.** Native crop (176 SPE, DC centred, pilot frame, pilot mask / support / AF / configs reusable, no known-zero padding) against keeping the zero-padded array (needs a roll of -22 or -44 lines or `center_kspace_dc`, a block-restricted mask, known-zero padding cuts, 1.25 / 1.5 times the voxels, an AF definition tied to the block anyway). Everything else in this report assumes the native crop.
2. **CSM source.** Stored `csm` (no extra compute, per-readout-slice maps, complex GT phase jumps 0.6 to 0.9 rad along RO, 790 pairs fit badly) against fresh ESPIRiT per contrast (pilot recipe, 2 minutes per pair, SoS 1 everywhere, fit 2 to 27 times better in bright voxels; tested on 5 contrast-pairs, not on all 16). The pilot used fresh maps.
3. **FLAIR plane.** Use the h5 FLAIR as is (the data of the unacquired lowest ky columns stay in the ground truth, energy share 4.0e-6 to 6.8e-6 of the total in the 14 regular pairs, 8.1e-4 and 8.5e-4 in the 790 pairs, relative image-domain norm 0.2 % and 2.8 %), or zero the displaced columns (ky below 20; below 7 for the 790 pairs). The known-zero region can come from the estimator or from an explicit support map; the plane is identical across subjects of a group.
4. **790 files.** Treat as extra scans of jiangdian and tangbing (not independent), drop, or keep as a separate exploratory group on another system.
5. **Registration tolerance.** No pair is shifted by more than 2 voxels (by either estimator; at most 1 voxel by the mask estimator in all 16); 14 of 16 pairs have both estimates within 1 voxel (the exceptions are 790_jiangdian and shiyixuan, 2 voxels on one axis of the edge estimate). The pilot subject is (0, 0, 0) on both.
6. **Head-only scoring mask.** The spec's `|GT| > 0.05 P99.9` mask includes speckle for noisy T1 volumes (head fraction 0.42 to 0.44 for litianyu, yangkun, 790_tangbin against about 0.31 typical). Paired differences are unaffected, absolute PSNR levels differ between subjects for this reason.

## 5. Anything surprising

1. **The 16-pair "other grid" is the pilot grid with padding.** 14 pairs have 44 and two have 88 zero SPE lines at the low-index end; with them removed all 16 pairs are 256 x 256 x 176. The +22 DC offset that once cost a mask misregistration, the 220 vs 176 mismatch and the "different reconstruction grid" of the task brief are one fact.
2. **Axis 0 of the raw h5 is image domain.** The loader would read it as k-space without error (it only checks the coil axis and shapes). `preprocess_jd_kfix.py` and `extract_S01_t1_flair.py` found this twice before; the task brief's phrase "full k-space" holds only after that FFT.
3. **The stored CSM are per-readout-slice maps** (readout-axis neighbour phase 0.55 to 0.84 rad). With them the 790 volumes show stripes in the CSM-combined image that are absent from the RSS image and from the CFL export. This affects the complex ground truth of every pair (random phase between readout slices, 0.59 to 0.90 rad in the three pairs measured); the magnitude image is visibly damaged only in the 790 pairs.
4. **The h5 FLAIR PE plane is not the CFL FLAIR plane.** For jiangyizhou: h5 34,259 acquired points, CFL 34,139, IoU 0.942; 964 CFL-only points (the upper-kz ends of the outer ky columns, up to 14 lines short per column), 1,084 h5-only points. In the h5 the ky columns 1 to 19 hold data displaced by about 30 SPE lines against the shutter's centre line (the CFL has these columns empty, matching the protocol's partial Fourier of 20 unacquired ky lines); the displaced column profiles match the Hermitian-partner columns 256 - j with correlation 0.95 to 0.99, which looks like a conjugate-symmetry fill written at the wrong kz centre (inference). The same pattern appears in all 14 regular FLAIR files (columns 1 to 19) and in a larger form in the two 790 files (columns 2 to 6, jump of about 55 lines, CFL FLAIR occupancy 0.7797 against 0.7332 in the h5). Image-domain consequence: relative norm 0.2 % (regular) and 2.8 % (790). The known-zero estimator copes (no cut found, 27 of 231 sampled zero lines unexplained, no significant outlier).
5. **The two 790 files** differ in several ways at once (system, 50 % oversampling, shutter, stored CSM quality, striping, duplicate status), so excluding them is cheap and removes most of the heterogeneity.
6. **Marker files are noise.** 17 of the 32 pair files have an upload marker and none is damaged; neither copy had a broken file.
7. **The ACS of the pilot's vdPoisMex mask is not fully sampled** (24 x 24 at 81.9 %, 16 x 16 full); BART's `-C 24` makes 24 x 24 full. `calib` in vdPoisMex only raises central density, the skill's own caveat.
8. **SNR varies a lot between T1 volumes** (background median 0.007 to 0.026 of P99.9; P99.9 scale 2.5e5 to 4.3e5) while FLAIR is steady (0.013 to 0.018), so the T1 head-only mask is speckle-laden for some subjects.
9. **The BART rule in the memory index is stale** (section 4(c)). The rule file says "new lines and the knee line" (新线和膝线重做) while the skill `mri-vdpois-mask` says "always vdPoisMex"; whether this brain core test counts as a "new line" is the user's call.
10. **Not measured or not knowable from the files.** Voxel size and FOV are not in the h5 files (the 1 mm isotropic, 256 x 256 mm, 176 mm slab figures exist only for the 2023-07-05 exams, from the huangyuwei protocol XML in the project notes; the Oct-Nov and 790 exams have no protocol record here). Rotation between the T1 and FLAIR scans was not estimated (translation only). Noise covariance / prewhitening was not examined. The fresh-ESPIRiT comparison covers 5 contrast-pairs, the identity registration is circumstantial, and the scanner statement in section 2.3 is an inference from file names, exam counters and one protocol note.

## Appendix A: commands, scripts, raw outputs

Environment for every Python command: `env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= nice -n 10 /home/congcongliu/anaconda3/envs/GSTMI/bin/python <script>`; scripts and outputs in `/tmp/claude-1002/sg2x2_inventory/`.

| script | what it does | output |
|---|---|---|
| `s01_headers.py` | h5 header scan of all 45 files | `s01_headers.out`, `s01_headers.json` |
| `run_md5.sh` | md5 of the 32 pair files in both trees (4 parallel `md5sum`, `ionice -c3`) | `md5/data1.md5`, `md5/data0.md5` |
| `s02_pair.py KEY` (driver `run_all_pairs.sh`) | per pair: read both datasets fully, finiteness, per-coil and per-slice energy, readout-axis test, PE-plane occupancy and block, DC, axis-0 FFT, image, head mask, CSM SoS and rank-1 residual, direct `ifft3c` contrast, native crop, T1-FLAIR Dice / centroid / xcorr | `pairs/<key>.json`, `arrays/`, `figs/<key>.png`, `logs_pair_*.txt` |
| `s03_summary.py`, `s04_tables.py` | tables from the JSONs | `out_s03_summary.txt`, `report_tables.md` |
| `x01..x05, x10, x28` | exploratory: plane histogram, FLAIR shape, energy ridge, CFL plane comparison, origin of displaced columns, `jd_kfix` | `out_*` printed in A.4 |
| `x02_cfl_equiv.py` | h5 native crop vs pilot CFL export (planes and RSS images, 8 flips, FFT cross-correlation) | `out_x02_cfl_equiv.txt` |
| `x06`, `x17` | coverage of head by candidate supports and by the pilot support | `out_x06_support_cover.txt`, `out_x17.txt` |
| `x07`, `x16` | rigid registration of T1 volumes for identity evidence | `out_x07_identity.txt`, `out_x16_identity_790.txt` |
| `x08_loader_dryrun.py` | conversion + `MRIDataset` + `estimate_known_zero` | `out_x08_*.txt` |
| `x09_bart_counts.py` | BART poisson point counts on 256 x 176 | printed in A.9 |
| `x11_af_facts.py` | pilot mask on each FLAIR plane | `out_x11_af_facts.txt` |
| `x12`, `x13`, `x14`, `x21`, `x22` | striping and alignment numbers, zoom views, montages, CFL vs h5 views, RSS vs CSM views | `out_x12.txt`, `figs/*.png` |
| `x15`, `x23`, `x24`, `x25` | CSM residual bins, fresh BART ESPIRiT test, slice-wise smoothness, GT phase continuity | `out_x15*.txt`, `out_x23*.txt`, `out_x24*.txt`, `out_x25_gt_phase.txt` |
| `x18`, `x19` | `.mat` and N4 checks | `out_x18.txt`, `out_x19.txt` |

Figures worth opening (not committed, in scratch): `figs/montage_T1.png`, `figs/montage_FLAIR.png` (mid-sagittal slice of all 16 pairs), `figs/rss_vs_csm_790_tangbi_jiangyizho.png`, `figs/cfl_vs_h5_T1_sagittal.png`, `figs/<key>.png` (per pair: three planes, acquired PE plane, SoS, head masks).

To remove the scratch bulk by hand (not done by the agent): `rm -r /tmp/claude-1002/sg2x2_inventory/convtest /tmp/claude-1002/sg2x2_inventory/bart_test`.

### A.1 Header scan and md5 (condensed raw output, 32 pair files)

`s01_headers.py` printed one line per file; all 45 files: `open=1`, `storage_full=True`; every pair file is `complex64`, contiguous, uncompressed; shapes `kspace == csm`:

```
T1/FLAIR 790_jiangdian_20230706-105312-3356_105518  3321890816 B  (256, 12, 264, 256)   marker T1=1 FLAIR=0
T1/FLAIR 790_tangbin_20230706-090657-3355_090938    3321890816 B  (256, 12, 264, 256)   marker T1=1 FLAIR=0
14 more pair names, each 2768242688 B, (256, 12, 220, 256); markers as in Table A
13 T1-only files (May 2023): (256, 12, 166, 256), 7 with a marker; orphan FLAIR marker: .Train_3dBrain_yexinting_20230531-144037-1397_144349.h5.baiduyun.uploading.cfg
```

```
$ cd md5 && sort -k2 data1.md5 > d1.sorted && sort -k2 data0.md5 > d0.sorted && diff d1.sorted d0.sorted && echo "ALL 32 md5 IDENTICAL between /data1 originals and /data0 copies"
ALL 32 md5 IDENTICAL between /data1 originals and /data0 copies
$ awk '{print $1}' d1.sorted | sort | uniq -d     (no output: no two of the 32 files are byte-identical)
```

### A.2 PE plane and DC of the first pair (`x01_plane_hist.py jiangyizhou_20230705`, key lines)

```
== T1 plane (220, 256) max=5.538e+06 min=1.719e-11
   log10 amp [-11,-10): 11110 (0.1973)   [-10,-9): 154 (0.0027)   [4,5): 40180 (0.7134)  [5,6): 4823 (0.0856)  [6,7): 53
   exact zeros: 0 ; occupancy at abs amp > 1.0: 0.8000 ; ax2 lines first/last with data 44/219, n_full(>0.999)=176
   ax2 indices fully empty: (0, 43, 44)
== FLAIR plane (220, 256) max=8.585e+06 min=0
   exact zeros: 948 ; occupancy at abs amp > 1.0: 0.6080 ; ax2 lines with data 44/218 ; ax3 columns with data 2/254
```

### A.3 Per-pair raw summary

The complete per-pair raw values (about 90 numbers per pair) are in `pairs/<key>.json`; the tables above are generated from them by `s04_tables.py`. `s03_summary.py` output is in `out_s03_summary.txt`.

### A.4 FLAIR displaced columns (jiangyizhou, FLAIR) and the CFL comparison

```
x03_flair_shape.py (acquired range along ax2 per ax3 column, first columns; SPE index uncropped):
ax3=  0: n=  0      ax3=  1: n= 14 range=(97,110)    ax3=  5: n= 40 (80,119)     ax3= 10: n= 60 (68,127)
ax3= 19: n= 86 range=(53,138)                       ax3= 20: n= 85 range=(85,169)   ax3= 21: n= 87 (84,170)
ax3=255: n=  0      ax3=254: n= 13 (117,129)
x04_ridge.py: energy-weighted centroid along ax2 per column: cols 1..19: 101.2, 98.6, ... 93.6 ; cols 20..30: 128.0, 127.8, ... 127.0
              T1 centroid per column: 135.0 ... 136.2 (no jump); column energy relative to the median: FLAIR cols 0..19: 0.0; cols 20..: 0.136 0.151 ...
x10_displaced_origin.py: col 4 vs partner 252: reflect c=220 corr 0.951 | col 10 vs 246: reflect c=220 corr 0.985 | col 19 vs 237: reflect c=220 corr 0.992 | col 20 vs 236 (undisturbed): shift d=0 corr 0.996
x05_flair_cfl_plane.py: CFL plane 34139 points, h5 plane 34259; XOR 2048 (CFL-only 964, h5-only 1084); CFL columns 0..19 empty, h5 columns 1..19 occupied
                        col 20: CFL (41,135) n=95 | h5 (41,125) n=85;  col 248: CFL (58,118) n=61 | h5 (58,104) n=47;  col 255: CFL (78,98) n=21 | h5 empty
790_tangbin FLAIR (x03): columns 2..6 hold n = 5, 12, 18, 24, 29 points at ranges (109,113) ... (93,121); column 7 (148,175) n=28 (jump of ~55 lines)
jd_kfix (x28): csm identical to the original; kspace = fft1c(axis 0, unnormalised) of the original (relative misfit 2.5e-8, scale 16 vs ortho); acquired planes and DC identical
```

CFL equivalence (`x02_cfl_equiv.py`, identity orientation is the best of 8 flips in every case; shift (0, 0, 0); T1 plane IoU 1.0000 in all seven):

```
exam                      RSS NCC T1   RSS NCC FLAIR   FLAIR plane IoU (h5 occupancy / CFL occupancy)
jiangyizhou_20230705      0.9421       0.9496          0.9419 (0.7604 / 0.7577)
huangyuwei_20230705       0.9489       0.9601          0.9419 (0.7604 / 0.7577)
jiangdian_20230705        0.9270       0.9518          0.9419 (0.7604 / 0.7577)
tangbing_20230705         0.9410       0.9519          0.9419 (0.7604 / 0.7577)
sunxinyi_20230705         0.9563       0.9632          0.9419 (0.7604 / 0.7577)
790_jiangdian_20230706    0.9630       0.9642          0.9355 (0.7332 / 0.7797)
790_tangbin_20230706      0.9138       0.9371          0.9355 (0.7332 / 0.7797)
```

### A.5 Alignment and striping numbers (`x12_stripe_and_align.py`, key columns)

```
key                      | T1<->FLAIR edge-xcorr shift (ax0,ax2,ax3) peak NCC (second peak)
790_jiangdian_20230706   | [2, 0, 1] 0.553 (0.513)      790_tangbin_20230706 | [0, 0, 1] 0.544 (0.505)
chenshuaibao_20231106    | [1, 0, 1] 0.688 (0.613)      huangyuwei_20230705  | [1, 0, 1] 0.674 (0.634)
jiangdian_20230705       | [1, 0, 1] 0.677 (0.617)      jiangyizhou_20230705 | [0, 0, 0] 0.730 (0.644)
liaozhenghan_20231030    | [0, 0, 1] 0.765 (0.668)      lishengbo_20231106   | [1, 0, 1] 0.603 (0.566)
litianyu_20231107        | [1, 0, 1] 0.753 (0.646)      shiyixuan_20231101   | [1, 0, 2] 0.599 (0.576)
sunxinyi_20230705        | [1, 0, 1] 0.613 (0.589)      tangbing_20230705    | [0, 0, 0] 0.626 (0.582)
yangkun_20231102         | [0, 0, 1] 0.631 (0.556)      zhangxiqian_20231102 | [1, 0, 1] 0.747 (0.648)
zhangyangming_20231101   | [0, 0, 1] 0.649 (0.596)      zhoushuo_20231030    | [0, 0, 1] 0.732 (0.643)
```

The per-readout-slice head-mean profile (9-slice running median, dips below 0.9 of the median) does not separate the 790 volumes from the others (0 to 4 slices per volume; `out_x20_dips.txt`): the stripes are partial lines inside slices, visible in the images and absent from the RSS image.

### A.6 Support coverage (`x06_support_cover.py`, `x17_support_cover_weighted.py`)

```
pilot support (256, 256, 176) frac=0.4273
other 15 subjects: min / median over [head, bright, energy] cover: [0.7624 0.8606 0.8782] [0.8738 0.9235 0.9458]
union head fraction of the box (GT proxy): 790_jiangdian 0.3287, 790_tangbin 0.4513, chenshuaibao 0.3292, huangyuwei 0.2802, jiangdian 0.3208,
  jiangyizhou 0.3849, liaozhenghan 0.3442, lishengbo 0.3855, litianyu 0.4457, shiyixuan 0.3217, sunxinyi 0.3198, tangbing 0.3968,
  yangkun 0.4296, zhangxiqian 0.3434, zhangyangming 0.3973, zhoushuo 0.3591
candidate support = one subject's own union head (voxel coverage of the other 15: min / median):
  790_jiangdian 0.660/0.782  790_tangbin 0.793/0.917  chenshuaibao 0.690/0.793  huangyuwei 0.590/0.700  jiangdian 0.597/0.728
  jiangyizhou 0.708/0.818  liaozhenghan 0.692/0.813  lishengbo 0.761/0.864  litianyu 0.783/0.895  shiyixuan 0.656/0.775
  sunxinyi 0.662/0.788  tangbing 0.753/0.904  yangkun 0.793/0.901  zhangxiqian 0.696/0.783  zhangyangming 0.761/0.879  zhoushuo 0.732/0.848
```

(The full 16 x 16 matrix is in `out_x06_support_cover.txt`.)

### A.7 Identity registrations (`x07_identity.py`, `x16_identity_790.py`)

```
all 120 pairs: mean 0.560 sd 0.095 min 0.309 max 0.773
top pairs: 0.773 zhangxiqian/zhoushuo | 0.768 jiangyizhou/litianyu | 0.733 tangbing/zhangyangming | 0.730 chenshuaibao/zhangxiqian | 0.717 yangkun/zhoushuo
== 790_jiangdian_20230706 (best of 7 starts): 0.625 Dice 0.931 jiangdian_20230705 | 0.552 / 0.898 sunxinyi | 0.540 / 0.837 tangbing | 0.525 / 0.885 shiyixuan | ... mean 0.439 sd 0.089
== 790_tangbin_20230706:                       0.664 Dice 0.949 tangbing_20230705  | 0.575 / 0.898 zhangyangming | 0.517 / 0.918 yangkun | 0.501 / 0.891 litianyu | ... mean 0.481 sd 0.062
```

### A.8 CSM, dry run and fresh ESPIRiT (raw lines)

```
x08 jiangyizhou: [loader:T1] volume_shape=(256, 256, 176) readout=0 phase=(1, 2) coils=12 normalization_scale=3.252e+05 head_frac=0.3836 mask_points=2966 of 45056 (15.19x)
                 [loader:FLAIR] ... normalization_scale=3.876e+05 head_frac=0.2662 mask_points=2966 of 45056 (15.19x)
                 [known_zero:FLAIR] dc=(127, 128, 88) cuts={axis0_lo: None, ...} ellipse=(124.0, 88.0, 1.0) frac_z=0.2391 n_sampled_lines=2966 n_zero_lines=231 n_zero_explained=204 n_nonzero_in_z=0 n_outliers=20 n_significant_outliers=0
x08 790_tangbin: [convert] T1: lo=88 ; [loader:T1] normalization_scale=3.063e+05 head_frac=0.4197 ; [loader:FLAIR] normalization_scale=3.388e+05 head_frac=0.3263
                 [known_zero:FLAIR] ellipse=(121.5, 88.0, 1.0) frac_z=0.2547 n_zero_lines=255 n_zero_explained=226 n_outliers=17 n_significant_outliers=0
x08 litianyu:    [convert] lo=44 ; [loader:T1] normalization_scale=2.875e+05 head_frac=0.4410 ; [loader:FLAIR] normalization_scale=3.643e+05 head_frac=0.2768
pilot (sg2x2_pilot2_af16_s42_A.log, CFL FLAIR): known_zero cuts={'axis0_lo': 20} ellipse=(128.0, 88.0, 1.0) frac_z_used=0.2423 zero_lines=233/2966 explained=233 outliers=0/0
x23 (bart ecalib rc 0, 55-57 s each) RSS bins [0.35,0.60) and [0.60,1.50): stored / fresh
  790_tangbin T1:   0.384 / 0.024 ; 0.301 / 0.011      per-slice roughness 0.0568 / 0.0225
  jiangyizhou T1:   0.050 / 0.021 ; 0.034 / 0.010      0.0257 / 0.0163
  jiangyizhou FLAIR:0.026 / 0.007 ; 0.014 / 0.003      0.0986 / 0.0149
  790_tangbin FLAIR:0.208 / 0.012 ; 0.129 / 0.005      0.0268 / 0.0117
  litianyu T1:      0.100 / 0.029 ; 0.068 / 0.012      0.0211 / 0.0064
x24 stored maps, neighbour |dphase| (rad) along RO / SPE / PE: 790_tangbin T1 0.844/0.018/0.018 ; FLAIR 0.593/0.012/0.013 ; jiangyizhou T1 0.747/0.034/0.030 ; FLAIR 0.547/0.023/0.021 ; chenshuaibao T1 0.784/0.046/0.040
x24b fresh BART maps, |dphase| along RO / PE / SPE: jiangyizhou T1 0.017/0.015/0.018 ; 790_tangbin T1 0.007/0.013/0.013
x25 complex GT neighbour |dphase| along RO / PE / SPE (stored vs fresh): jiangyizhou T1 0.607/0.148/0.151 vs 0.140/0.146/0.149 ; jiangyizhou FLAIR 0.587/0.060/0.057 vs 0.063/0.059/0.056 ; 790_tangbin T1 0.904/0.173/0.189 vs 0.143/0.159/0.172
```

### A.9 Masks and AF (raw lines)

```
x11: pilot AF16 mask: (256, 176), 2966 points, whole-block AF 15.191; strict AF16 = 2816 points; ACS24x24 fully sampled? False (central 24x24 fraction 0.8194)
     FLAIR: lo=44 block occ 0.7604 | mask points on FLAIR acquired 2735 | AF vs block T1 / FLAIR 15.19 / 16.47 | AF FLAIR vs own acquired 12.53 | mask points on unacquired 231
     FLAIR: lo=88 block occ 0.7332 | 2711 | 15.19 / 16.62 | 12.19 | 255
x09: plane 256x176 = 45056 points; AF16 over the whole plane = 2816 points
     A=3.0: [3752, 3759, 3794] A=3.5: [3170, 3136, 3160] A=3.8: [2857, 2870, 2873] A=4.0: [2705, 2692, 2691] A=4.2: [2542, 2540, 2540] A=4.4: [2403, 2407, 2415] A=4.6: [2298, 2279, 2269]
     (ACS24 sampled 1.00 for every draw)
x27: pilot vdPoisMex: central 64x64 density 0.311, edge band 0.023, central 24x24 0.819, central 16x16 1.000 ; BART A=3.8: 0.197 / 0.039 / 1.000 ; A=4.0: 0.192 / 0.038 / 1.000
x26: ready-made mask_vdpois_220x256_af16.mat: 3659 points, AF over array plane 15.39, ACS24 0.927 (af08 7127, af10 5704, af12 4760, af14 4130)
md5 of the pilot AF16 masks: T1FLAIR_AF16_retro and T1FLAIR_AF16_v3_jiangyizhou 2d7ed0001214aec2af8f6dce5d52ce70 ; init_support_af16_jyz.npz 310a6520631a330e59cb5deb390e9035
```

### A.10 Trainer `.mat` and n4 checks (`x18`, `x19`)

```
T1:            keys [csm, kspace], on-disk (256, 220, 12, 256): mat vs h5 as-is (axes reversed) max|diff|/max = 0.000e+00 ; vs h5 after fft1c(ax0) 9.175e-01
T1_trainer:    keys [ksp, map],    on-disk (256, 220, 256, 12): vs h5 as-is 3.885e+00 ; vs h5 after fft1c(ax0) 9.124e-08
FLAIR:         as-is 0.000e+00 ; after fft1c 8.488e-01        FLAIR_trainer: as-is 4.889e+00 ; after fft1c 5.358e-08
liaozhenghan kspace: n4 vs T2-dir 1.779e-01 ; n4 vs FLAIR-dir 1.779e-01 ; T2-dir vs FLAIR-dir 0.000e+00     csm: all three 0.000e+00
x19 liaozhenghan n4/plain image ratio in the head: mean 1.0127 sd 0.2509 p1 0.6500 p50 0.9525 p99 1.6886 ; corr 0.7832 ;
    log-ratio explained by a cubic field R^2 = 0.835 (residual sd 0.097 of 0.238)
```
