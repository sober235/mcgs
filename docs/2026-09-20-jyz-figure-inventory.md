# jiangyizhou T1+FLAIR 结果图清单与选图口径（2026-09-17 至 09-20，会话 claude-64）

本文只做两件事：把现在散在多个分支上的结果图列清楚（每套图在哪、是什么版式、几列），以及把这两天定下来的
**选层、选框、亮度**三条口径和支撑数字写下来。图本身不在这个分支里，各自在下面写明的分支上。
所有数字都是本会话在服务器上重算或从课题 `logs/` 里读出来的，不是转述。

---

## 1. 三种采样掩膜叠在一张图（diff-mask 研究）

一张图里三种掩膜各占一行（灰度行 + 误差行），列是 GT、ZF 和九个方法。分支 **`diffmask-zoom-figures`**
（从 `jyz-af16-diffmask` 长出，7 个提交），图在 `t1flair_af16_gauss_v3_jiangyizhou/figures/`，共 17 张：

| 版式 | 张数 | 文件名特征 |
|---|---|---|
| 无放大（原本入库的一版） | 4 | `..._{T1,FLAIR}_tra{180,205}_headerr_p3_zf_caps.png` |
| 角上小放大，48 框 | 6 | `..._inset48_headerr_p3_zf_caps[_box138_76].png` |
| 放大行（每种掩膜下加放大图像行 + 放大误差行） | 5 | `..._zoomrows{32,40}_...` |
| 角上放大占面板一半宽 + 调亮（现行） | 2 | `..._inset32_headerr_p3_zf_caps_nongsmax.png` |

出图脚本 `scripts/fig_masks_transverse_stack.py`，本会话新增四个开关，默认值都保持原行为，不加开关重出与已入库的图
逐像素相同（`scripts/figs_masks_stack_*.sh` 第一步每次都核对）：

- `--inset W` / `--inset_center J K` / `--inset_corner`：把 W×W 框画在每个面板上，框内放大贴到面板一角。
- `--inset_frac`：放大贴图占面板宽的比例，默认 0.38（既有正式图），现行图用 0.5。
- `--zoom_rows W`：改成整行放大，放大格与原面板同宽。
- `--grey_scale`：灰度窗 = 系数 × 该层 GT 的 P99.9，默认 1.0，现行图用 0.8（调亮约四分之一）。误差图不受影响，
  仍按 max|GT| 归一、固定 jet [0, 0.1]。

## 2. 七方法精简版（sel7）

同样的三面版式，只留 **CS-SENSE、Joint SENSE、ZS-SSDU、3D-INR、MC-INR、GS-single、MC-GS** 七列加 GT，
去掉 ZF、CG-SENSE、L1-SPIRiT、3D-DIP。分支 **`sel7-seven-method-figures`**，23 张 PNG + 23 张 PDF，
覆盖 AF12 / AF16 / AF20、gauss 与伪径向掩膜、四档噪声课题、儿童 TSC AF12，以及三掩膜叠放图的七方法版。
分辨率约 4350×4420 px（200 dpi PNG，180 mm 版面折合约 614 dpi）。

面板里的 PSNR / SSIM / NMSE 是各臂对真值算的，**不随列数变化**；变的只是加粗标记谁最好（"所展示列里的最好者"）。
三个加速档在这七个方法里的头内 PSNR：

| 档 | MC-GS T1 / FLAIR | 七列里第二名 | 余量 |
|---|---|---|---|
| AF12（配方 P14） | 37.015 / 36.360 | GS-single 36.865 / 36.324 | +0.150 / +0.036 |
| AF16（三期配方） | 35.706 / 34.928 | GS-single 35.558 / 34.886 | +0.148 / +0.042 |
| AF20（三期配方） | 34.832 / 34.179 | GS-single 34.667 / 34.127 | +0.165 / +0.052 |

## 3. 十二列定稿图（各加速档）

分支 **`autoresearch-jyz-af24`**，`t1flair_af{12,16,20,24}_v3_jiangyizhou/figures/`，文件名含
`_p{3,14}_mcinr_spirit_zf_caps`：GT 后有 ZF 列、方法名大写、含 L1-SPIRiT 与 MC-INR，横断 180 / 冠状 124 / 矢状 76
加 48 放大框。AF12 的 MC-GS 用 P14 配方，AF16/20/24 用三期配方，文件名里的 `p14` / `p3` 就是这个区别。

---

## 4. 选层与选框的三条口径（本会话定，带支撑数字）

### 4.1 层：两对比度合看，205 层已是最大差距层

`scripts/find_slice_all_masks.py` 的 176 个合格层，按"MC-GS 对最强对比臂（GS-single 算对比臂）的余量"重排：
九项（三指标 × 三掩膜 × 两对比度）全第一的只有 203–206；其中 205 层按**最小余量**和**平均余量**都排第一
（最小 +0.050 dB、平均 +0.120）。放宽到"只要求六行 PSNR 第一"，205 仍第一。不设条件时 106 层平均更高（+0.123），
但它的 FLAIR Gaussian 行 MC-GS 输 0.01 dB，不能用。

各对比度分开看可以再大一点：T1 最好是 104 层（三掩膜最小余量 +0.178），FLAIR 最好是 148 层（+0.063）。
对应的图在 `diffmask-zoom-figures` 的 `..._T1_tra104_...maxgap.png` 与 `..._FLAIR_tra148_...maxgap.png`。

### 4.2 框：看得见的差距只在"对非 GS 方法"上

任何候选框里 MC-GS 对 GS-single 的均方根误差只低 **2% 到 9%**（比值 0.91–0.98），在固定 jet [0, 0.1] 下肉眼分不出；
能看出来的是对经典方法与深度学习方法的差距。因此改用 `scripts/find_slice_nongs_gap.py`（178 层逐层重打分，72 个体积）
加 `scripts/find_zoom_box_visible.py --rank nongs --edge_q 0.75` 选层与框，条件不放松：MC-GS 在整层九项与框内
PSNR/SSIM 上仍要对所有列（含 GS-single）第一。结果：

| 图 | 层 | 框 | 框内对最强非 GS 列的领先（vd-Poisson / Gaussian / 伪径向） |
|---|---|---|---|
| T1 | 126 | 32×32 @ (146, 29) | +1.250 / +1.126 / +1.118 dB |
| FLAIR | 205 | 32×32 @ (148, 146) | +0.795 / +0.838 / +0.796 dB |

对照：同样口径下 205 层 T1 的最好框只有 +0.61；FLAIR 在 126 层只有 +0.568，所以两张图层号不同。
框内数字只打印不上图，对外仍引全层与全体积余量；框和层都是在本数据上挑的，写作时要说明。

### 4.3 亮度

灰度窗从 P99.9 改为 0.8 × P99.9。试过 1.0 / 0.8 / 0.65 三档，0.65 整体发灰、背景噪声明显。误差图与指标不变。

---

## 5. AF16 与 AF20 是不是同一批图（核对结果：不是）

用户问过这两档是否重复。三个层面都查了：

- **图文件**：四张定稿图 md5 两两不同。
- **掩膜**：AF16 2966 点（15.19×，md5 `d08179b8e74f…`），AF20 2427 点（18.56×，md5 `9c889b61837e…`）。
- **重建体积逐元素比较**：真值 T1 与 FLAIR **逐位相同**（同一套全采样数据做不同倍数的回顾性欠采样，本该相同）；
  零填充最大逐点差 0.135 / 0.125，MC-GS 0.378 / 0.376，GS-single 0.416 / 0.358，ZS-SSDU 0.558 / 0.510，都不相同。
- **指标**：每一臂都随加速下降（MC-GS 35.706 / 34.928 → 34.832 / 34.179）。

⚠️ 但两个课题**共用同一组展示层与同一个 48 框**，所以并排看解剖完全一样，只有画质和数字不同。放进论文时图注要点明，
免得读者误以为是同一张图。

---

## 6. 其他这两天产出的图（不在上面三类里）

- **零填充三面图**：`t1flair_af12_v3_jiangyizhou/figures/t1flair_v3_af12_ZF_three_planes_180_124_76.png`（合成图）
  与 `figures/zf_panels/` 下六张单面板，脚本 `scripts/fig_zf_three_planes.py`。
- **重建时间与质量**：`t1flair_af12_v3_jiangyizhou/figures/jyz_af12_recon_time_vs_quality.png`（另有 `_polished` 带
  Pareto 前沿的一版），脚本 `scripts/fig_recon_time.py`，数据取自 `logs/bench/recon_time_table.json`。一对 T1+FLAIR
  的求解器时间：ZF 0.09 s、CG-SENSE 11 s、L1-SPIRiT 3.1 min、CS-SENSE 与 Joint SENSE 4.8 min、MC-INR 11.5 min、
  ZS-SSDU 57.9 min、**MC-GS 1.02 h**、3D-DIP 1.67 h、3D-INR 1.91 h、GS-single 2.03 h（两次单跑相加）。
- **图索引页**（只在服务器上，不入库）：`~/figs/t1flair_v3_af16_diffmask/index.html`，带缩略图，分五类列出叠放图与
  单掩膜图，`http://localhost:8765/t1flair_v3_af16_diffmask/index.html`。

## 7. 每套图在哪（一句话版）

| 内容 | 分支 | 目录 |
|---|---|---|
| 三掩膜叠放图全部 17 张 | `diffmask-zoom-figures` | `t1flair_af16_gauss_v3_jiangyizhou/figures/` |
| 七方法精简版 23 张 | `sel7-seven-method-figures` | 各课题 `figures/` |
| 十二列定稿图（AF12/16/20/24） | `autoresearch-jyz-af24` | `t1flair_af{12,16,20,24}_v3_jiangyizhou/figures/` |
| 掩膜图（四倍数系列、四种图案） | `autoresearch-jyz-af24` / `jyz-af16-diffmask` | 同上各 `figures/` |
| 本清单 | `jyz-figure-inventory` | `docs/` |

PDF 与读数 json 按各分支惯例多数不入库，都在 `/data0/congcong/code/GS/Results_20260908/<课题>/` 的
`figures/`、`logs/` 下；服务器看图走 `http://localhost:8765/<课题短名>/`。
