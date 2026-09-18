# Retrospective nominal AF 16 on the 3D T1 + FLAIR pair, jiangyizhou -- Gaussian variable-density random mask (diff-mask study)

Mask-pattern study, 2026-09-15/16. **Read `../T1FLAIR_AF16_v3_jiangyizhou/README.md` first**: same subject, same CFL
k-space, same ESPIRiT maps, same ground truth, same gauge, same arms and the same frozen AF16 phase-3 MC-GS recipe are
documented there and not repeated here.

## What this topic adds

**Exactly one thing differs from `T1FLAIR_AF16_v3_jiangyizhou`: the mask PATTERN.** That topic is the comparison
column of this study and is not re-run -- every number in its column below is the one it already recorded. The
k-space and the ESPIRiT maps are symlinks into `../T1FLAIR_AF12_v3_jiangyizhou/data/` (they do not depend on the
mask); everything else in this topic was reconstructed from scratch, because every arm does depend on the mask.

The question the study answers: does MC-GS's lead survive when the sampling pattern is not the house variable-density
Poisson disc?

### The mask

2-D Gaussian variable density: outside the ACS the points are drawn i.i.d. without replacement with probability proportional to exp(-rn^2 / 2s^2). The width s is not hand-picked -- it is chosen on a 0.005 grid so the draw's median normalised radius matches the house vd-Poisson mask's (0.399 against 0.400), which matches the density curve and leaves the point process as the only difference: i.i.d. random against Poisson disc.

Common to this study's masks: the fully sampled **24 x 16** block of the house vd-Poisson AF16 mask is kept (that is
the block `recon_spirit.py`'s `acs_box()` measures, not the generator's nominal calib 24), so L1-SPIRiT calibrates
from the same ACS in every topic; DC at (128, 88); one shared mask for both contrasts; `numpy` `Generator(seed 0)`, so
unlike `vdPoisMex` (which seeds from `time(0)`) **this mask can be regenerated**.

**Acceleration** follows the user's definition of 2026-09-14: AF = PE-plane points / sampled points that carry that
contrast's data. T1 acquires the whole (ky, kz) plane; FLAIR's elliptical vendor shutter leaves 34139 of 45056 lines,
so the shared mask gives FLAIR the higher AF. The budget is matched to the house mask's own T1 acceleration, and the
topic is **labelled AF 16 (nominal)** exactly as that topic is (its own T1 AF is 15.19).

| mask | PE points | T1 AF | FLAIR AF | FLAIR points outside the shutter | 24x24 centre | median rn |
|---|---|---|---|---|---|---|
| vd-Poisson disc (house, AF16 topic) | 2966 | 15.19 | 16.49 | 233 | 472/576 | 0.400 |
| **Gaussian variable-density random (this topic)** | **2964** | **15.20** | **15.43** | **44** | **442/576** | **0.399** |

Generator `scripts/make_masks_af16_diffmask.py --kinds gauss` (it refuses to overwrite an existing mask), read-out
`logs/make_masks_af16_diffmask.json`, mask `data/mask_gauss_256x176_af16.mat` (md5 `306f4caed5118687c09ad364c63d02d4`) broadcast
along the readout into `data/mask_af16_3d.h5`.

Figures of the masks and of the fully sampled k-space they undersample:
`figures/t1flair_v3_af16_diffmask_masks.png`, `figures/t1flair_v3_af16_diffmask_gauss_width.png`,
`figures/t1flair_v3_jyz_full_kspace.png` (served at `http://localhost:8765/t1flair_v3_af16_diffmask/`).

## Method under test

The **AF16 phase-3 recipe, unchanged**, as at AF 20 and AF 24: P6's repaired cold start (init_mode head, 3-NN scales,
edge-TV threshold from ZF), 3.6M points, edge-TV lambda_geo 0.10 / kappa_q 0.85 / w_floor 0.10 / eps 0.04,
kspace_high 2.2, warmup 500, wavelet db4 on its finest level only (T1 0.12 / FLAIR 0.10), common-edge cross-gradient
prior lambda 0.001, phase-only TV lambda 0.03, 4000 iterations. The argument list lives in one place,
`/data0/congcong/code/GS/mcgs_diffmask_wt/run_diffmask_mcgs.sh`, whose recipe block is byte-identical to
`run_af24_mcgs.sh`'s (checked with `diff`).

The classical arms are re-swept on this mask, grids = the AF16 topic's own, with JTV widened from its two points to
five so the winner can land in the interior. **L1-ESPIRiT is not listed**: the user dropped it from the comparison on
2026-09-15, so it was not run here (`run_all_diffmask.py --with_l1espirit` puts it back). Every job and grid is
defined once, in `scripts/run_all_diffmask.py`.

## Results

<!-- TABLES:BEGIN -->
**T1** (sorted by head-only PSNR; vd-Poisson column = the same arm in `T1FLAIR_AF16_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @vd-Poisson | this − vd-Poisson |
|---|---|---|---|---|---|---|---|---|
| MC-GS | 38.615 | 0.9316 | **34.909** | **0.9333** | 0.1344 | 0.796 | 35.706 | -0.80 |
| GS-single | 38.471 | 0.9347 | 34.771 | 0.9305 | 0.1412 | 0.794 | 35.558 | -0.79 |
| ZS-SSDU | 38.217 | 0.9518 | 34.375 | 0.9273 | 0.1697 | 0.899 | 34.150 | +0.22 |
| CS-SENSE (TV, converged) (λ 0.0007) | 37.652 | 0.9418 | 33.896 | 0.9190 | 0.1814 | 0.788 | 34.945 | -1.05 |
| joint CS-SENSE (JTV) (λ 0.001) | 37.638 | 0.9430 | 33.841 | 0.9188 | 0.1832 | 0.779 | 34.831 | -0.99 |
| CS-SENSE | 37.481 | 0.9389 | 33.637 | 0.9172 | 0.1855 | 0.747 | 34.409 | -0.77 |
| L1-SPIRiT (λ 0.001) | 36.709 | 0.9210 | 32.995 | 0.9010 | 0.1653 | 0.896 | 33.788 | -0.79 |
| CG-SENSE | 34.598 | 0.7887 | 31.899 | 0.8782 | 0.2516 | 0.811 | 32.354 | -0.45 |
| MC-INR (Niessen 2025) | 31.874 | 0.6231 | 31.558 | 0.8463 | 0.2598 | 0.929 | 31.590 | -0.03 |
| 3D-DIP | 35.063 | 0.8570 | 31.141 | 0.8546 | 0.3053 | 0.647 | 30.994 | +0.15 |
| 3D-INR | 32.299 | 0.7073 | 29.443 | 0.7659 | 0.2206 | 1.436 | 30.375 | -0.93 |
| ZF | 29.468 | 0.7953 | 25.412 | 0.6523 | 0.7320 | 0.278 | 25.841 | -0.43 |

**FLAIR** (sorted by head-only PSNR; vd-Poisson column = the same arm in `T1FLAIR_AF16_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @vd-Poisson | this − vd-Poisson |
|---|---|---|---|---|---|---|---|---|
| MC-GS | 38.101 | 0.9263 | **34.317** | **0.9210** | 0.1278 | 0.759 | 34.928 | -0.61 |
| GS-single | 38.019 | 0.9046 | 34.294 | 0.9202 | 0.1295 | 0.767 | 34.886 | -0.59 |
| ZS-SSDU | 37.902 | 0.9471 | 33.972 | 0.9196 | 0.1479 | 0.905 | 34.005 | -0.03 |
| joint CS-SENSE (JTV) (λ 0.001) | 37.588 | 0.9408 | 33.731 | 0.9112 | 0.1481 | 0.719 | 34.355 | -0.62 |
| CS-SENSE (TV, converged) (λ 0.0007) | 37.503 | 0.9395 | 33.683 | 0.9115 | 0.1480 | 0.699 | 34.302 | -0.62 |
| CS-SENSE | 37.314 | 0.9298 | 33.447 | 0.9078 | 0.1477 | 0.655 | 33.806 | -0.36 |
| L1-SPIRiT (λ 0.001) | 36.652 | 0.9187 | 32.849 | 0.8923 | 0.1360 | 0.876 | 33.720 | -0.87 |
| CG-SENSE | 35.275 | 0.8047 | 32.557 | 0.8828 | 0.1800 | 0.757 | 32.826 | -0.27 |
| MC-INR (Niessen 2025) | 33.082 | 0.6739 | 32.102 | 0.8599 | 0.1946 | 0.852 | 32.243 | -0.14 |
| 3D-DIP | 34.850 | 0.8239 | 30.918 | 0.8393 | 0.2613 | 0.545 | 30.650 | +0.27 |
| 3D-INR | 32.917 | 0.7329 | 29.807 | 0.7745 | 0.2053 | 1.212 | 30.257 | -0.45 |
| ZF | 31.332 | 0.8335 | 27.451 | 0.6956 | 0.6888 | 0.260 | 27.879 | -0.43 |

**MC-GS joint minus MC-GS single**

| channel | whole PSNR | head PSNR | head SSIM | (vd-Poisson: whole / head) |
|---|---|---|---|---|
<!-- TABLES:END -->

## Figures

House layouts, at the AF16 figure-of-record slices and box so nothing is re-selected on this mask. Twelve columns:
ground truth, zero-filling, CG-SENSE, CS-SENSE, Joint SENSE, L1-SPIRiT, ZS-SSDU, 3D-DIP, 3D-INR, MC-INR, GS-single,
MC-GS.

| figure | what |
|---|---|
| `http://localhost:8765/t1flair_v3_af16_diffmask_gauss/t1flair_v3_af16g_T1_row3planes_compact_174_110_47_headerr_p3_spirit.png` | T1, transverse 174 / coronal 110 / sagittal 47 |
| `http://localhost:8765/t1flair_v3_af16_diffmask_gauss/t1flair_v3_af16g_FLAIR_row3planes_compact_174_110_47_headerr_p3_spirit.png` | FLAIR, same slices |
| `http://localhost:8765/t1flair_v3_af16_diffmask_gauss/t1flair_v3_af16g_T1_row3planes_compact_180_124_76_inset48_headerr_p3_spirit.png` | T1, figure of record, 48-voxel zoom inset (also `.pdf`, 600 dpi vector) |
| `http://localhost:8765/t1flair_v3_af16_diffmask_gauss/t1flair_v3_af16g_FLAIR_row3planes_compact_180_124_76_inset48_headerr_p3_spirit.png` | FLAIR, same (also `.pdf`) |
| `http://localhost:8765/t1flair_v3_af16_diffmask/t1flair_v3_af16_diffmask_masks.png` | the four candidate masks side by side |
| `http://localhost:8765/t1flair_v3_af16_diffmask/t1flair_v3_jyz_full_kspace.png` | the fully sampled k-space both contrasts are undersampled from |

Files also live in this topic's `figures/`. Orange in-panel numbers are head-masked slice PSNR / SSIM (7x7 window) /
NMSE, the best of each plane in bold; error maps are `|recon - GT| / max(GT)`, jet, fixed [0, 0.1], inside the head
only; left of each error row is that plane's sampling mask.

Zero-filling is the adjoint, so it barely uses the pattern and lands within 0.4 dB of itself on all three masks (head
PSNR 25.4-25.8 on T1, 27.4-27.9 on FLAIR). It is the anchor showing the three studies start from the same place; what
differs in its panel is the *shape* of the aliasing, noise-like on the random patterns and streaky on the radial one.

The served folder and the file-name prefix are derived from the topic path inside `fig_three_planes_compact.py`, not
hardcoded: a hardcoded copy of that script wrote this study's radial renders into the Gaussian folder under Gaussian
names on 2026-09-16 and overwrote them. Both sets were re-rendered after the fix.

## The study's headline: does MC-GS's lead depend on the pattern?

The three topics share everything but the sampling pattern, so this table isolates it. **MC-GS is first on head-only
PSNR on all three patterns and both contrasts** (`scripts/compare_masks.py`, which reads all three topics read-only;
full tables in `logs/compare_masks_head_{psnr,ssim,hfen}.md`):

| contrast | pattern | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | vd-Poisson (house) | 35.706 | GS-single 35.558 | **+0.148** |
| T1 | Gaussian | 34.909 | GS-single 34.771 | **+0.138** |
| T1 | pseudo-radial | 31.113 | GS-single 30.913 | **+0.200** |
| FLAIR | vd-Poisson (house) | 34.928 | GS-single 34.886 | **+0.042** |
| FLAIR | Gaussian | 34.317 | GS-single 34.294 | **+0.023** |
| FLAIR | pseudo-radial | 31.729 | ZS-SSDU 31.635 | **+0.093** |

**GS-single counts as a comparison arm, not as a second version of ours** (user, 2026-09-13, restated 2026-09-16 with
its reason): 3-D reconstruction by Gaussian splatting is already published work, so a single-contrast Gaussian-splatting
reconstruction is prior art that this method has to beat, exactly like CS-SENSE or ZS-SSDU. Every margin quoted in this
topic is therefore over the strongest arm *including* GS-single. A margin computed over "the strongest non-GS arm" is
not a number this study reports.

**Five of the six cells win on all three head metrics; the sixth does not.** On the pseudo-radial mask FLAIR keeps head
PSNR but loses head SSIM (0.8673 against ZS-SSDU's 0.8686) and head HFEN (0.2809 against 0.2768). That is the one cell
where MC-GS is not the outright winner, and it is the cell whose known-zero prior the pattern disables (27 mask points
outside the shutter, the estimator cannot fit the ellipse). It is reported, not smoothed over.

**The pattern itself is worth more than any arm's lead over another.** Against the house vd-Poisson mask, the same arm
loses 0.3-1.1 dB on the Gaussian mask and 1.1-5.0 dB on the pseudo-radial one -- an order of magnitude more than the
margins above. Poisson disc is the better pattern of the three, which is what the 3-D head literature says; the point
of the study is that the ranking of the *methods* does not change with it.

**The coupling gain (MC-GS minus GS-single) is positive on every pattern**: T1 +0.148 / +0.138 / +0.200, FLAIR +0.042 /
+0.023 / +0.150 (vd-Poisson / Gaussian / pseudo-radial). It is largest on the worst pattern, which is consistent with
cross-contrast information being worth more when the single-contrast problem is harder -- an observation, not evidence.

## Known limitations
- **Parameter selection, for the record.** MC-GS and GS-single run one frozen recipe (the AF16 phase-3 one) on all
  three masks -- that is what makes the study single-variable. The classical arms' lambdas were re-selected on this
  mask from the AF16 grids (CS-SENSE converged TV, joint TV, L1-SPIRiT); every winner here landed in the interior of
  its grid, so those arms are at their grid optimum.
- **L1-ESPIRiT is retired from this series, not merely unlisted** (user, 2026-09-16): it is replaced everywhere by
  L1-SPIRiT as the k-space comparison arm. It was not run on either diff-mask topic and will not be; the older topics
  keep their recorded L1-ESPIRiT volumes and summary rows as history. `run_all_diffmask.py --with_l1espirit` still
  exists but is not something this series uses.
- **The 2-D CAIPIRINHA mask was built but not reconstructed** (user, 2026-09-16). Its topic
  `../T1FLAIR_AF16_caipi_v3_jiangyizhou` holds the mask, its read-out and the inspection figures only; the study
  compares vd-Poisson, Gaussian and pseudo-radial.
- Everything the AF16 topic's own README lists as a limitation applies here unchanged: the recipe and the classical
  lambdas were selected on this subject, and the numbers are on the magnitude gauge.

## Regenerating

```
# mask (numpy, seeded, regenerable; it refuses to overwrite)
PYTHONNOUSERSITE=1 PYTHONPATH= python scripts/make_masks_af16_diffmask.py --kinds gauss

# every arm, both topics, queued onto free cards, then finalize
setsid nohup bash scripts/launch_diffmask.sh > /dev/null 2>&1 < /dev/null &
```

## Status

2026-09-16: complete. Eleven arms x two contrasts reconstructed, scored and figured; MC-INR re-scored through this topic's own gauge with a worst disagreement of 1e-7 against the reproduction's own numbers (the gauge is mask-independent, so this is the expected result and it confirms the three topics' numbers are directly comparable).

## 三图案横断位叠放图（2026-09-16 晚，用户要求）

`scripts/fig_masks_transverse_stack.py --contrast T1|FLAIR --pdf`：三种图案从上到下（vd-Poisson / Gaussian / 伪径向）各占一行灰度图
加一行误差图，横断位 180（图 of record 的层，眼睛朝上）；列 = GT、ZF、九个方法、GS-single、MC-GS（p3 配方，各图案自己的 λ）。
每行灰度图右侧一条灰度 colorbar（窗 = GT 该层的 P99.9，三行相同），误差图右侧一条 jet colorbar（固定 0 到 0.1，只显示头内）；
误差行左格 = 该图案的 (ky, kz) 掩膜；面板内橙黄的头内层 PSNR / SSIM / NMSE，每行最好者加粗。
输出 `figures/t1flair_v3_af16_masks_stack_{T1,FLAIR}_tra180_headerr_p3_zf_caps.{png,pdf}`（200 dpi PNG；PDF 面板 300 dpi 栅格、字体 TrueType），
served 于 `http://localhost:8765/t1flair_v3_af16_diffmask/`。
该层读数：vd-Poisson 与 Gaussian 两行 MC-GS 三项都第一；伪径向行 T1 三项第二（ZS-SSDU 第一，32.10 对 31.87 dB），FLAIR 的 PSNR / NMSE 第二、
SSIM 第一。这是单层读数；全体积头内 PSNR 三种图案 MC-GS 仍第一（`logs/compare_masks_head_psnr.md`）。换层用 `--tra`。

### 三种图案全超的层（2026-09-16 晚，用户问"难道没有三个 mask 都超过对比方法的层"）

`scripts/find_slice_all_masks.py`（口径与叠放图相同：头内层 PSNR / SSIM / NMSE，12 列，GS-single 算对比臂；合格层 = 头内面积
不低于最大层一半，两对比度都满足，48 到 223 共 176 层；读数 `logs/find_slice_all_masks.json`）：
- T1 三项、三图案全第一：106 层（48 到 138 连续，加 203 到 223 里的一段），最小余量最大的是 104 层（+0.178 dB）。
- FLAIR 三项、三图案全第一：25 层，最好 148（+0.063）、192（+0.062）、205（+0.050）。
- **两对比度、三项、三图案全第一：只有 203 到 206 四层，205 层余量最大**（最小余量 +0.050 dB；T1 三图案 +0.131 / +0.122 / +0.140，
  FLAIR +0.050 / +0.079 / +0.197）。只看 PSNR 时两对比度全超的有 22 层（含 122 到 132 一段，129 层最小余量 +0.041）。
- 叠放图 205 层版：`figures/t1flair_v3_af16_masks_stack_{T1,FLAIR}_tra205_headerr_p3_zf_caps.{png,pdf}`。180 层版保留作图 of record 层的对照
（那一层伪径向行 T1 是 ZS-SSDU 第一）。

### 叠放图加放大框（2026-09-17，用户"补充 zoom-in view"）

`scripts/fig_masks_transverse_stack.py` 新增三个默认关闭的开关：`--inset W`（W×W 框在每个灰度与误差面板上橙黄描边，框内放大贴到面板一角，
放大区不写数字，与正式图一致；文件名加 `_inset<W>`）、`--inset_center J K`（框在横断层内的中心，体积第 1、2 轴，默认 124 76）、
`--inset_corner`（默认右上）。三行掩膜用同一个框；掩膜格不放大；开放大框时 GT 面板的图案标签分两行写，避开放大区（脚本出图后逐个核对
文字与放大区不重叠）。不加这些开关时出图与原图逐像素相同（`scripts/figs_masks_stack_inset.sh` 第一步：T1 205 层重出后与已提交的图比对，全等）。

框的位置没有另挑：中心 (124, 76) 就是正式图 48³ 框（180 / 124 / 76）的层内中心。`scripts/find_zoom_box_all_masks.py --tra 205|180`
按叠放图的口径（框内头部像素的 PSNR 与 SSIM，12 列，GS-single 算对比臂）查了这个框：
- 205 层：三图案、两对比度六行 MC-GS 框内 PSNR 与 SSIM 全第一，最小 PSNR 领先 +0.154 dB（FLAIR vd-Poisson，对 GS-single）；
  T1 三图案 +0.299 / +0.335 / +0.703，FLAIR +0.154 / +0.220 / +0.719。
- 180 层：六行同样全第一，最小领先 +0.082 dB（FLAIR vd-Poisson）；T1 +0.336 / +0.375 / +0.585，FLAIR +0.082 / +0.140 / +0.484。

同一脚本也搜了全层（步长 2，框内头部占比 ≥ 0.95，避开右上放大区与左上文字）：205 层有 1114 个中心六行全第一，最好的 (138, 76) 最小领先
+0.219 dB；180 层 1911 个，最好 (172, 92) +0.171 dB。没有换用，框与正式图保持一致；读数在 `logs/find_zoom_box_all_masks_tra{205,180}.json`。
框内数字只打印不上图，而且只代表这一块区域，对外仍引全层与全体积余量。

输出 `figures/t1flair_v3_af16_masks_stack_{T1,FLAIR}_tra{205,180}_inset48_headerr_p3_zf_caps.{png,pdf}`（200 dpi PNG；PDF 面板 300 dpi 栅格），
served 于 `http://localhost:8765/t1flair_v3_af16_diffmask/`；无放大框的四张保留。重出：`bash scripts/figs_masks_stack_inset.sh`（CPU，逐张）。

**差距更大的框（2026-09-17，用户"挑一张差别更大的"）。** 在整层全第一的四个层（203 到 206）上逐像素重搜（`scripts/run_zoom_box_search_203_206.sh`，
`--step 1`），最弱一行的领先最大的仍是 205 层，框中心换到 (138, 76)：六行最小领先 +0.219 dB（原框 +0.154），其余五行有升有降，六行平均基本持平
（+0.396 对 +0.405）。对最强非 GS 方法的框内领先，新框 T1 +0.32 / +0.93 / +0.98、FLAIR +0.64 / +0.79 / +1.36（三图案），原框 T1 +0.30 / +0.80 / +0.77、
FLAIR +0.55 / +0.75 / +1.65，六行里五行变大。按"GS-single 算对比臂、六行都第一"的规则，这四层找不到最弱行超过 +0.22 dB 的 48 框。
图 `figures/t1flair_v3_af16_masks_stack_{T1,FLAIR}_tra205_inset48_headerr_p3_zf_caps_box138_76.{png,pdf}`（`scripts/figs_masks_stack_inset_box138.sh`），
原框版本保留；四层读数 `logs/find_zoom_box_all_masks_tra{203,204,205,206}_step1.{json,txt}`。

### 放大行版式（2026-09-17，用户："选的那张差异不够明显，zoom-in view 也不够明显"）

角上的小放大图太小（约面板宽的 0.38），改成**放大行**：`fig_masks_transverse_stack.py --zoom_rows W --inset_center J K`（新开关，默认关闭；
不加时出图仍与原图逐像素相同，`scripts/figs_masks_stack_zoomrows.sh` 第一步再次核对）。每种掩膜占四行：原图行（框橙黄描边）、放大图像行、
误差行（框描边）、放大误差行；放大格与原面板同宽，最近邻放大（40 框约 13 倍、32 框约 16 倍），橙黄边框；每行各有 colorbar，放大误差仍是固定 jet [0, 0.1]；
放大区不写数字。

选框改为按"看得见的差距"：`scripts/find_zoom_box_visible.py --contrast T1|FLAIR --rank nongs`，T1 图与 FLAIR 图各选各的框（两张图本来分开），
范围 203 到 206 层、框边长 32 / 40 / 48、逐像素；条件 = 框内头部占比 ≥ 0.95、避开左上文字、GT 边缘量不低于中位数（框里要有沟回结构）、
三种掩膜下 MC-GS 在框内 PSNR 与 SSIM 都对所有列第一（GS-single 算对比臂）；排序 = 三行里对最强非 GS 列的最小领先。逐框复核用 `scripts/eval_zoom_box.py`。
为了两张图同层，都取 205 层（最好的在 206 层，只比 205 多 0.08 dB）：

| 图 | 框 | 框内对所有列的领先（vd-Poisson / Gaussian / 伪径向） | 对最强非 GS 列 |
|---|---|---|---|
| T1 | 40×40，中心 (160, 131) | +0.233 / +0.230 / +0.443（最近的都是 GS-single） | +0.616 / +0.611 / +0.614 |
| FLAIR | 32×32，中心 (148, 146) | +0.142 / +0.155 / +0.285（同上） | +0.795 / +0.838 / +0.796 |

限制：候选框里 MC-GS 的均方根误差最多只比 GS-single 低 2% 到 9%（比值 0.91 到 0.98），在固定 [0, 0.1] 色标下看不出来；放大行里看得出的差距
是对非 GS 列的。没有一个框能让两个对比度同时拉开对非 GS 列的差距（`eval_zoom_box.py` 试过的共用框，最好的一个 T1 对非 GS 只有 +0.27）。
这些框是在本数据上挑的，框内数字只打印不上图，对外仍引全层与全体积余量。

输出 `figures/t1flair_v3_af16_masks_stack_T1_tra205_zoomrows40_headerr_p3_zf_caps_box160_131.{png,pdf}` 与
`figures/t1flair_v3_af16_masks_stack_FLAIR_tra205_zoomrows32_headerr_p3_zf_caps_box148_146.{png,pdf}`，served 于 `http://localhost:8765/t1flair_v3_af16_diffmask/`；
读数 `logs/find_zoom_box_visible_{T1,FLAIR}{,_nongs}.{json,txt}`。角上小放大图的各版本保留。

### 差距最大的层，两个对比度各一层（2026-09-18，用户："重新给出最大差异的结果图"）

`logs/find_slice_all_masks.json` 的 176 个合格层按几种口径重排（口径 = 头内层 PSNR，MC-GS 对最强对比臂，GS-single 算对比臂）：
- **两对比度合看**：九项（三指标 × 三图案 × 两对比度）全第一的只有 203–206，其中 205 层按最小余量和按平均余量都排第一（最小 +0.050、平均 +0.120）。
  只要求六行 PSNR 第一时 205 仍第一。所以 205 已经是两对比度合看的最大差距层，之前选它是对的。
- **各对比度分开看**（两张图本来分开，这样能再大一点）：T1 最好是 **104 层**（三图案三指标全第一，最小余量 **+0.178**，对比 205 层的 +0.122；
  三图案 +0.252 / +0.178 / +0.200），FLAIR 最好是 **148 层**（最小 **+0.063**，对比 205 层的 +0.050；+0.076 / +0.063 / +0.115）。

据此出的图（放大行版式，框按该层重挑，规则同上一节）：`scripts/figs_masks_stack_maxgap.sh`
- T1：横断 104（颅底层，小脑与颞叶），框 32×32 @ (107, 34)。框内对所有列 +0.316 / +0.100 / +0.181，对最强非 GS 列 +0.743 / +0.830 / +0.714。
- FLAIR：横断 148（基底节层），框 32×32 @ (141, 139)。框内对所有列 +0.149 / +0.252 / +0.474，对最强非 GS 列 +0.586 / +0.499 / +1.086。

文件 `figures/t1flair_v3_af16_masks_stack_T1_tra104_zoomrows32_headerr_p3_zf_caps_maxgap.{png,pdf}` 与
`figures/t1flair_v3_af16_masks_stack_FLAIR_tra148_zoomrows32_headerr_p3_zf_caps_maxgap.{png,pdf}`。205 层的两张（同层同框）保留，
要"一张图同时覆盖两个对比度"时用它。层是在本数据上按余量挑的，这一点在对外写作时要说明。

### 放大区里"看得见"的差距最大的一版（2026-09-18，用户："换个区域让 zoom-in 的差异更明显"）

读者在放大区里能分辨的是 MC-GS 与**非 GS 方法**的差距（对 GS-single 任何框里都只低 2–9% 误差，看不出）。因此改按
"对最强非 GS 列的最小领先"挑层与框，条件不变：MC-GS 在整层九项与框内 PSNR/SSIM 上仍要对所有列（含 GS-single）第一。
- 层：`scripts/find_slice_nongs_gap.py`（178 个合格层逐层重打分，72 个体积；22 层满足"六行对所有列第一"）。按对非 GS 的最小领先，
  T1 最好的层是 126（三图案 +0.68 / +0.58 / +0.62，整层口径），FLAIR 最好的仍是 205（+0.70 / +0.41 / +0.29）。
- 框：`scripts/find_zoom_box_visible.py --rank nongs --edge_q 0.75`（框边长 32 / 40，结构量前 25%）。
  T1 在 126 层取 32×32 @ (146, 29)：框内对最强非 GS 列 **+1.250 / +1.126 / +1.118 dB**（vd-Poisson / Gaussian / 伪径向，
  对手是 Joint SENSE 或收敛 TV），对所有列 +0.225 / +0.093 / +0.314。比 205 层那版（对非 GS +0.61）差不多翻倍。
  FLAIR 留在 205 层的 32×32 @ (148, 146)（对非 GS +0.795 / +0.838 / +0.796），它在 126 层只能到 +0.568。

图：`figures/t1flair_v3_af16_masks_stack_T1_tra126_zoomrows32_headerr_p3_zf_caps_nongsmax.{png,pdf}`（T1）与既有的
`..._FLAIR_tra205_zoomrows32_headerr_p3_zf_caps_box148_146.{png,pdf}`（FLAIR）。两张层号不同，标题各自写明。
读数 `logs/find_slice_nongs_gap.json`、`logs/find_zoom_box_visible_*_nongs*.{json,txt}`。

同一选择也出了"放大贴在图上"的版本（用户 2026-09-18："将这里的 zoom-in view 放在图上"）：`scripts/figs_masks_stack_inset_nongsmax.sh`，
即 `--inset 32 --inset_center … --inset_frac 0.5`。新开关 `--inset_frac` 控制角上放大图的边长占面板宽的比例，默认 0.38（与既有正式图一致），
这里用 0.5 以便看清；贴图变宽后 GT 面板的图案标签会被压住，脚本在 `--inset_frac > 0.42` 且贴图在上方时自动把标签移到左下角（出图后的文字重叠自检从
`['Pseudo-radial']` 变为 `none`）。文件 `figures/t1flair_v3_af16_masks_stack_T1_tra126_inset32_headerr_p3_zf_caps_nongsmax.{png,pdf}` 与
`..._FLAIR_tra205_inset32_headerr_p3_zf_caps_nongsmax.{png,pdf}`；放大行版本同时保留，两种版式数字完全相同。
这两张按用户 2026-09-18 的要求调亮：新开关 `--grey_scale`（灰度窗 = 系数 × 该层 GT 的 P99.9，默认 1.0 与既有图一致）取 **0.8**，
三档试过 1.0 / 0.8 / 0.65，0.65 已经发灰、背景噪声明显。误差图不受影响（仍按 max|GT| 归一、固定 jet [0, 0.1]），指标也不变。
