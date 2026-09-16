# T1+FLAIR 回顾性重建 AF 12 / 16 / 20 / 24 —— 高分辨率定稿图

受试者 jiangyizhou（jyz），3D T1 + FLAIR，同一套数据、同一套打分口径，只换欠采掩膜。
本文件夹只放**图**；数字、方法、每条臂的来源，以各课题自己的 README 为准：

```
Results_20260908/T1FLAIR_AF12_v3_jiangyizhou/README.md
Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/README.md
Results_20260908/T1FLAIR_AF20_v3_jiangyizhou/README.md
Results_20260908/T1FLAIR_AF24_v3_jiangyizhou/README.md
```

## 内容

`png/` 与 `pdf/` 各 8 个文件 = 4 个加速档 × 2 个对比度（T1、FLAIR）。
每张图 = 三个正交面（横断 180 / 冠状 124 / 矢状 76）× 十二列方法，每面下方是头内误差图
（jet，固定 [0, 0.1]×max|GT|），每个面板内嵌 48×48 放大框。

列的顺序：GT、ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、
MC-INR、GS-single、MC-GS。

## 分辨率口径（2026-09-16 用户定）

- **PNG：`savefig(dpi=200)`，6453 × 4352 px。** 图幅本身是 32.3 × 21.8 英寸的大横图，
  所以按 180 mm 版面宽折算是 **911 dpi**，按 A4 横向可用宽度（273 mm）折算是 600 dpi。
- **PDF：矢量。** 文字、框线、色条边框是矢量；图像面板按 600 dpi 栅格化
  （`--pdf_dpi 600`，matplotlib 对 `AxesImage` 的 `set_rasterized(True)`）。
- **PDF 字体是 TrueType，不是 matplotlib 默认的 Type 3**（脚本里设了
  `matplotlib.rcParams["pdf.fonttype"] = 42`）。IEEE、Elsevier 的投稿系统常拒收 Type 3 字体；
  `pdffonts` 现在报的是 `CID TrueType`。辅助图同样处理。

**一个必须说清楚的限制**：每个面板的底图只有约 256 体素宽。提高 dpi 让文字、边框、放大框、
色条变锐利，**不会给图像本身增加任何信息**——图像部分是插值放大的。

## 本批相对旧定稿图的三处改动

1. **ZF 列**：真值右边加了零填充列（用户 2026-09-16），四档统一。
2. **caps 排版**：平面名首字母大写、首行方法名大写（用户 2026-09-16）。
3. **SPIRiT 例外**：`L1-SPIRiT` / `L1-ESPIRiT` 的小写 i 是方法名本身的拼法
   （iTerative Self-consistent Parallel Imaging Reconstruction），全大写等于拼错，
   所以对含 `SPIRiT` 的标签跳过大写（用户 2026-09-16）。

其余一律未动：切片、放大框、灰度窗、误差色标、朝向（横断眼睛朝上）、方法、参数、打分口径。
三面的 PSNR/SSIM/NMSE 排名与改动前的 `_mcinr_spirit` 定稿图逐位一致。

## 加速倍数标注

标题里的 AF = **整平面相位编码点数 ÷ 该掩膜采样点数**（用户 2026-09-14 定，两对比度同一数字）：

| 课题 | 标题 actual | 采样点 | T1 / FLAIR（各自采集支撑口径） |
|---|---|---|---|
| AF12 | 11.71× | 3847 | 11.71 / 12.86 |
| AF16 | 15.19× | 2966 | 15.19 / 16.49 |
| AF20 | 18.56× | 2427 | 18.56 / 19.96 |
| AF24 | 24.00× | 1877 | 24.00 / 25.51 |

⚠️ AF24 的 FLAIR 标签（A 案 24.00× 还是 B 案 25.51×）用户尚未拍板，见该课题 README。

## 怎么重出

驱动脚本 `scripts/render_hires_final.sh`（本文件夹内）。四档的差别只有三处：
AF12 用 `fig_three_planes_inset.py` 且键是 `gs_joint_p14 / gs_single_p14`；
AF16 的 MC-INR 列默认开（`--no_mcinr` 才关），AF20/AF24 默认关（要 `--with_mcinr`）。

## 各课题另有的辅助图（不在本文件夹，就地更新）

掩膜检查、掩膜序列、原始 k 空间支撑等小图已就地重出为 **600 dpi PNG + 矢量 PDF**：
AF20 两张、AF24 四张，在各自 `figures/` 下。

AF12 的 `mask_check_af12_af11_af16.png`：原来没有生成脚本（2026-09-13 临时画的），2026-09-16
按掩膜数据重写了 `scripts/fig_mask_check_af12_af11_af16.py`。图上每个数字都是从三张掩膜现算的，
**不是抄旧图**，算出来与旧图逐个吻合（3847 / 4173 / 2966 点，AF 11.71 / 10.80 / 15.19，
中心 56×56 采样率 39.3 / 40.8 / 36.1 %，AF12∩AF16 共享 812 点 = 21 % / 27 %）。

AF16 的 `probe_c_mask_T1_vs_FLAIRc.png`：生成代码一直都在，只是藏在 prep 脚本里
（`prep_04c_export_af16c.py` 的 `figure()`），它自带 `--fig_only` 模式只重画不重导数据。
