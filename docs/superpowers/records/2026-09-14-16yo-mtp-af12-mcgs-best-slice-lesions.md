# 记录 2026-09-14：16 岁 TSC 回顾性 AF12，病灶可见性与 MC-GS 最优层

课题：`Results_20260908/Peds_TSC_MTP_AF12_retro/`（权威数字在它的 README）。本记录的脚本、说明、逐层指标和图在
`peds_tsc_mtp_af12_retro/analysis_2026-09-14_mcgs_best_slice_lesions/`。全部是只读分析，没有改动课题里的任何文件。
病灶判断都是目测加定量辅助，**不是放射科阅片**。图和文档里没有病人姓名。

## 1. 结论

- T2*w（FA4/TE27.41）能看到病灶，T1w（FA16/TE3.25）看不到。
- 12 倍欠采样下，十个方法都保留了 T2*w 上侧脑室壁的结节；3D-DIP 模糊成团，3D-INR 和 MC-INR 被颗粒噪声淹没一部分。
- MC-GS 质量最优的层选定为 T2*w 横断第 69 层（厂商第 22 层）。这是挑出来的层：T1w 上 MC-GS 在 559 层里一层都不领先，T2*w 上 551 层里 67 层三项全第一。
- 这一层有一个钙化样小灶：SWI 与 T2*w 上是黑点，QSM 上强烈抗磁性，纵向约 7 mm 的局灶，不像血管。MC-GS 重建保留了它。

## 2. 用了哪些结果

十个方法：CG-SENSE、CS-SENSE、Joint SENSE、L1-ESPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR、GS-single（`gs_single_ar`）、
MC-GS（`gs_joint_ar`），都是二期登记的版本；经典方法取课题重扫 λ 后的那一行。GS-single 算作对比方法。
调参后的臂（`gs_joint_p3`、`gs_single_p3`，提交 275277c，18:06 登记）晚于本分析开始，不在逐层排名里，第 6 节单独补查了第 69 层。
AF 写法：掩膜 2635 个点全在采集范围内，AF = 31616 / 2635 = 12.00，两个对比度相同。

## 3. 病灶能否看到

一期普查（`docs/superpowers/records/2026-09-12-16yo-mtp-reference-dataset.md` 第 7 节）：侧脑室壁的黑色小灶在厂商第 38、46 层，
与厂商 SWI 对应，外观像钙化的室管膜下结节；皮层结节只在厂商 2D T2-FLAIR 上是亮斑。

本次把参考图和十个方法在第 38、46 层侧脑室附近放大对比，灰度窗统一取参考图（`figures/peds_af12_T2s_lesion_zoom_vendor38_46.png`、
`figures/peds_af12_T1w_lesion_zoom_vendor38_46.png`，脚本 `scripts/lesion_zoom_af12.py`）：

- T2*w：结节在所有方法里都能看到。CG-SENSE、CS-SENSE、Joint SENSE、L1-ESPIRiT、ZS-SSDU、GS-single、MC-GS 清楚；3D-DIP 模糊；3D-INR 与 MC-INR 颗粒重。
- T1w：参考图同位置只有脑室轮廓，没有局灶信号改变；十个方法一样，没有假病灶，也没有丢病灶。
- 皮层结节在这两个对比度上都不明显。

## 4. MC-GS 最优层

口径（`scripts/peds_mcgs_best_slice.py` → `peds_slice_metrics.json`）：三个方向、头部面积不少于该方向最大截面 25% 的所有层；
头部掩膜 = |GT| > 0.05 · P99.9，限在中间 80 个分区（SPE 12..91，与主表一致）；逐层算头内 PSNR（按体积峰值）、7×7 SSIM 图在头内的均值、头内 NMSE。

| 对比度 | 打分层数 | MC-GS 三项全第一 |
|---|---|---|
| T1w | 559 | 0（单项也一次都没有） |
| T2s | 551 | 67（横断 13 / 80，冠状 39 / 249，矢状 15 / 222） |

挑选过程：领先最多的是冠状第 35 层（比 GS-single 高 0.234 dB），但只是头部边缘一条薄弧，不采用；先交的横断第 91 层在颅底，
用户认为脑结构太少；随后在头部面积不少于 90% 的候选（横断 69 / 71，矢状 169，冠状 119）里选了横断第 69 层，
矢状和冠状受 1.35 mm 分区厚度影响画出来是扁条。

横断第 69 层（厂商第 22 层，头部面积 97%，可见颞叶、中脑、额叶眶面、外侧裂），`figures/peds_af12_T2s_mcgs_best_transverse_69.png`：

| 方法 | 头内 PSNR | SSIM | NMSE |
|---|---|---|---|
| MC-GS | 33.50 dB | 0.8723 | 0.01007 |
| GS-single | 33.45 dB | 0.8710 | 0.01019 |
| L1-ESPIRiT | 33.09 dB | 0.8717 | 0.01107 |

比第二名只高 0.05 dB。肉眼可见的差别在额底近鼻窦区域：CG-SENSE、CS-SENSE、Joint SENSE、MC-INR 在误差图上一块红，MC-GS 基本干净。
这是挑选出来的层，不代表整卷：主表上二期 MC-GS 在 T1w 落后 L1-ESPIRiT，在 T2*w 与 GS-single 持平。
重画：`python scripts/peds_fig_row_best.py --contrast T2s --axis 2 --index 69`。

## 5. 第 22 层的钙化样小灶

脚本 `scripts/peds_slice22_lesion_check.py`、`scripts/peds_focus22.py`、`scripts/peds_qsm_sign.py`；图 `figures/lesion_focus_vendor22.png`（红圈标记与放大）、
`figures/lesion_check_vendor22.png`（SWI、参考图、MC-GS、最近的 FLAIR 层）。

- 位置：厂商第 22 层（z = 6.2 mm）320×300 画幅的第 166 行、第 194 列，图像右侧颞叶内侧、中脑外侧。病人左右未核实。
  它低于一期记录的结节（z 27.6 / 38.3 mm）和皮层结节（FLAIR z 40–66 mm），是本次新看到的。

| 检查 | 结果 |
|---|---|
| 参考 T2*w | 中心比周围暗 1.86 个标准差 |
| 厂商 SWI | 黑点 |
| 厂商 QSM | 中心 −730，周围 +22；这一层 P1..P99 为 −78..+62 |
| QSM 符号核实 | 黑点附近 SWI 上最暗的血管样像素均值 +23；整层 6 条细长暗结构中位数 +23.5 |
| 纵向范围 | T2*w 变暗在第 21–23 层，QSM 负值在第 19–25 层，约 7 mm |

同层血管在 QSM 上为正（顺磁为正），该灶强烈负（抗磁），且是局灶而非管状，外观符合钙化，不像静脉。
最近的 FLAIR 层（第 5 层，z 8.3 mm）上看不出亮斑。MC-GS 的 AF12 重建里这个黑点清楚保留。

## 6. 补查：调参后的臂

第 69 层 T2*w，同一口径：`gs_joint_p3`（调参后的 MC-GS）33.514 dB / 0.8728 / 0.01004，略好于二期 MC-GS 的 33.501 / 0.8723 / 0.01007；
T2*w 没有调参后的 GS-single（`gs_single_p3` 只跑了 T1w），GS-single 仍是 33.450。所以把调参后的臂算进来，这一层仍是 MC-GS 第一。
逐层排名没有用调参后的臂重跑；要让最优层对应调参后的表，改 `scripts/peds_mcgs_best_slice.py` 里的 MC-GS 键后重跑即可。

## 7. 注意

- 参考图是 GRAPPA 补齐的前瞻 2 倍数据，不是全采样；只有一个受试者、一次检查。
- 病灶判断是目测和定量辅助，不是阅片；钙化与否需要放射科确认。
- 分析图也在服务器 `/home/congcongliu/figs/peds_tsc_af12_lesion/` 与 `/home/congcongliu/figs/peds_tsc_af12_mcgs_best/`。
