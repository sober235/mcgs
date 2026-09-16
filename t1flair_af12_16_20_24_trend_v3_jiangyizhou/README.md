# jiangyizhou T1+FLAIR：AF12 / AF16 / AF20 / AF24 跨加速倍数复核（2026-09-15，按现行对比口径）

问题：四个回顾性课题的全部重建结果，指标是不是随加速倍数提高而逐级变差？MC-GS 对最强对比方法的余量怎么随加速倍数变化？

本课题是 `T1FLAIR_AF12_16_20_trend_v3_jiangyizhou`（三档）加上 AF24 后的扩展，只读取四个课题，不改它们。09-15 上午的第一版按旧口径写（列 L1-ESPIRiT、把 GS-single 当自己人），在 git 历史里（mcgs 仓库 d79b0bc）；本版按用户定的现行口径重出：

- **GS-single 算对比方法**（用户 09-13），所以主余量是 MC-GS − GS-single。
- **L1-ESPIRiT 不再列出**（用户 09-15）；它的重打分仍留在 `logs/rescore_af*.json` 里（数据），只是表、图、判定不显示。
- **L1-SPIRiT 加入**（k 空间方法，09-14/15 登记进各课题），**MC-INR** 四档齐（AF24 于 09-15 补跑）。

| 课题 | 掩膜点数 | **AF 标签（两对比度同一个数，本文用）** | FLAIR 带数据的采样点数 → 09-14 逐对比度口径 | FLAIR 对它自己的原始采集 |
|---|---|---|---|---|
| `T1FLAIR_AF12_v3_jiangyizhou` | 3847 | **11.71** | 3504 → 12.86 | 9.74 |
| `T1FLAIR_AF16_v3_jiangyizhou` | 2966 | **15.19** | 2733 → 16.49 | 12.49 |
| `T1FLAIR_AF20_v3_jiangyizhou` | 2427 | **18.56** | 2257 → 19.96 | 15.13 |
| `T1FLAIR_AF24_v3_jiangyizhou` | 1877 | **24.00** | 1766 → 25.51 | 19.33 |

AF 标签（用户 09-16 定）= 整个相位编码平面线数 ÷ 掩膜采样线数。每个课题一张掩膜，两个对比度读同一张，所以标签只有一个数；本文的图和"每翻一倍掉多少 dB"都按它算。T1 采满整个平面；FLAIR 的原始扫描只采了 34139/45056 条线（四角椭圆扫描 + ky 0–19 行，原始 CFL 核实，图见 AF24 课题 `figures/t1flair_v3_af24_raw_kspace_support_T1_vs_FLAIR.png`），所以掩膜点里只有第四列的那些带 FLAIR 数据，按它们算就是 09-14 的逐对比度口径；按"比原始 FLAIR 扫描少采多少"算则是最后一列。

## 1. 口径核对

- 四个课题的 GT 逐位相同（md5 T1 af619da6、FLAIR c302014a）；打分代码 `diag_artefact/headonly.py` md5 32a4e4a9。
- 104 个存盘重建体（13 个方法 × 4 档 × 2 对比度，含留作数据的 L1-ESPIRiT）用同一套代码重打分，与各课题原表的最大差异：头内 PSNR 3.1e-5 dB、SSIM 4.2e-7、HFEN 3.9e-7。与上午第一版共有的 282 个数值逐位相同，新增的只有 8 行 L1-SPIRiT 和 AF24 的 2 行 MC-INR。
- 经典方法的 λ（`logs/lambda_grid_check.txt`）：L1-SPIRiT 八条曲线、收敛 TV 八条的最优都在网格内部；Joint SENSE 除 AF16 只扫了两个点（最优贴边）外都在内部。
- MC-GS / GS-single：AF12 用 AF12 那条线的定稿配方 P14，AF16/20/24 用 AF16 三期配方（两套接近但不完全相同）。

## 2. 整个体积：头内三项逐级变差

头内 PSNR（dB），AF12 → AF16 → AF20 → AF24：

| 方法 | T1 | FLAIR | 头内 PSNR、SSIM、HFEN 每一步都变差（T1 / FLAIR） |
|---|---|---|---|
| MC-GS | 37.015 → 35.706 → 34.832 → 33.767 | 36.360 → 34.928 → 34.179 → 33.182 | 是 / 是 |
| GS-single | 36.865 → 35.558 → 34.667 → 33.588 | 36.324 → 34.886 → 34.127 → 33.110 | 是 / 是 |
| CS-SENSE 收敛 TV | 36.162 → 34.945 → 34.003 → 32.820 | 35.581 → 34.302 → 33.494 → 32.573 | 是 / 是 |
| Joint SENSE | 36.070 → 34.831 → 33.885 → 32.730 | 35.624 → 34.355 → 33.544 → 32.605 | 是 / 是 |
| CS-SENSE 35 步 | 35.762 → 34.409 → 33.418 → 32.231 | 35.084 → 33.806 → 33.027 → 32.165 | 是 / 是 |
| ZS-SSDU | 35.724 → 34.150 → 32.892 → 32.578 | 35.034 → 34.005 → 32.660 → 32.107 | 是 / 是 |
| L1-SPIRiT | 35.017 → 33.788 → 32.882 → 31.783 | 35.103 → 33.720 → 32.824 → 31.803 | 是 / 是 |
| CG-SENSE | 33.514 → 32.354 → 31.425 → 30.340 | 33.957 → 32.826 → 32.089 → 31.259 | 是 / 是 |
| MC-INR | 32.192 → 31.590 → 30.969 → 30.076 | 33.042 → 32.243 → 31.666 → 30.947 | 是 / 是 |
| 3D-DIP | 31.363 → 30.994 → 30.996 → 30.500 | 30.810 → 30.650 → 30.637 → 30.360 | PSNR 否 / HFEN 否 |
| 3D-INR | 31.638 → 30.375 → 29.456 → 28.444 | 31.190 → 30.257 → 29.809 → 29.146 | 是 / 是 |
| ZF | 26.130 → 25.841 → 25.639 → 25.333 | 28.193 → 27.879 → 27.636 → 27.301 | 是 / 是 |

SSIM、HFEN 的完整数值和每一步的变化在 `logs/af_trend_tables.md`；图 `figures/jyz_t1flair_af12_16_20_24_metric_steps.png`（点在 0 左边 = 这一步变差，三种颜色和点形 = 三步）。

- 唯一例外是 3D-DIP 的 AF16 → AF20 那一步（T1 头内 PSNR +0.002 dB，FLAIR 头内 HFEN 好 0.0001）；它在 AF20 → AF24 三项都变差。
- MC-INR 的全体积 PSNR 不单调（误差主要在头外背景），头内三项每一步都变差。
- 按 AF 标签每翻一倍折算，除 ZS-SSDU 外的主要方法每步掉 2.2–3.8 dB/倍（MC-INR、3D-INR 的 FLAIR、3D-DIP、ZF 更少）；ZS-SSDU 在 AF16 → AF20 掉得最快（T1 4.35、FLAIR 4.65 dB/倍），AF20 → AF24 这一步只掉 0.85 / 1.49 dB/倍（AF24 时训练到第 73 / 59 轮才早停，AF20 为 53 / 41）。

## 3. 余量（头内 PSNR，同一打分器，`logs/af_trend_margins.json`）

| 档位 | T1：**MC-GS − GS-single** | MC-GS − 最强非 GS | GS-single − 最强非 GS | FLAIR：**MC-GS − GS-single** | MC-GS − 最强非 GS | GS-single − 最强非 GS |
|---|---|---|---|---|---|---|
| AF12 | **+0.150** | +0.853 | +0.702 | **+0.036** | +0.735 | +0.700 |
| AF16 | **+0.148** | +0.761 | +0.613 | **+0.042** | +0.573 | +0.531 |
| AF20 | **+0.165** | +0.829 | +0.664 | **+0.052** | +0.634 | +0.582 |
| AF24 | **+0.179** | +0.947 | +0.768 | **+0.071** | +0.577 | +0.505 |

最强对比方法四档两对比度都是 GS-single；最强的非 GS 方法 T1 四档都是 CS-SENSE 收敛 TV，FLAIR 四档都是 Joint SENSE。

- **主余量 MC-GS − GS-single 随加速变大**：FLAIR 逐档变大（+0.036 → +0.071）；T1 在同一配方的 AF16 → AF20 → AF24 上逐档变大（+0.148 → +0.179），AF12 用的是另一套配方（P14），+0.150 与 AF16 基本持平。
- **对最强非 GS 方法的余量不单调**：T1 AF16 最低、AF24 最高；FLAIR AF12 最高、AF24 与 AF16 接近。
- 每档只有一张掩膜。AF20、AF24 四张掩膜测过的只是对 L1-ESPIRiT 余量的抽样 SD（0.002–0.006 dB）；主余量的抽样波动没测（抽样掩膜上没跑 GS-single）。

## 4. 同一层

单层指标 = 出图脚本 `fig_three_planes_compact.py` 面板上的定义（头内 PSNR、7×7 窗 SSIM 头内平均、头内 NMSE）。向量化 SSIM 与 skimage 在展示层上最多差 7.5e-13；AF20 与 AF24 定稿图上印的 MC-GS 和 L1-SPIRiT 面板数字都能原样算出（`logs/slice_trend_summary.txt`）。

展示层（12 方法 × 3 面 × 3 指标 = 108 格）每一步都变差的格数：

| 层组 | T1 | FLAIR |
|---|---|---|
| 180/124/76（定稿图） | 97/108 | 105/108 |
| 174/110/47 | 99/108 | 108/108 |

没变差的格子只有 3D-DIP 的 AF16 → AF20 那一步，以及 MC-INR 的 3 格 SSIM（AF12 → AF16 略升）。

全部层（头部占该层面积 ≥ 5%：横断 243 层，冠状 187 / 193 层，矢状 167 / 166 层，T1 / FLAIR）上三项每一步都变差的层占比：

- MC-GS、GS-single：T1 三个面 100%；FLAIR 横断、矢状 100%，冠状 97.9%（4 层，都在离覆盖边缘 10 层以内）。
- L1-SPIRiT：T1 三个面 100%；FLAIR 至少 99.2%（横断第 0 层、冠状第 227 层，都在边缘）。
- 两个 CS-SENSE、Joint SENSE、3D-INR、ZF：至少 97.9%；逐层 PSNR 列出的例外层除 3D-INR FLAIR 横断第 37 层（头占 23%，AF16 → AF20 持平）外都在边缘。CG-SENSE：T1 ≥ 99.2%，FLAIR 横断 86.8%。
- ZS-SSDU：T1 横断 86.8%（32 层不单调，都不在边缘，列出的几层是 AF20 → AF24 回升约 0.1 dB），FLAIR 横断 77.0%；与它这一步掉得少一致。
- MC-INR：PSNR 在 T1 为 86–99%、FLAIR 横断 67%，SSIM 在 T1 为 62–78%。3D-DIP：29–71%。

## 5. 复现

```
T=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_16_20_24_trend_v3_jiangyizhou
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
bash $T/scripts/run_rescore.sh                                   # 八个后台作业，CPU，约 35 分钟 -> logs/rescore_af*.json
PYTHONNOUSERSITE=1 PYTHONPATH= $PY $T/scripts/analyze_af_trend.py > $T/logs/af_trend_summary.txt   # 表、判定、余量、图
python3 $T/scripts/lambda_grid_check.py > $T/logs/lambda_grid_check.txt
bash $T/scripts/run_slice_trend.sh                               # 两个后台作业，约 6 分钟 -> logs/slice_trend_*.json
PYTHONNOUSERSITE=1 PYTHONPATH= $PY $T/scripts/slice_summary.py > $T/logs/slice_trend_summary.txt
```

坑：
- 后台作业要 `setsid nohup` 脱离会话（脚本里已这样写）；zsh 里环境变量前缀要直接写出来。
- 显示哪些方法由各脚本的 `ORDER` 决定；L1-ESPIRiT 的重打分留在 json 里，别当成漏算。
- `analyze_af_trend.py` 会覆盖出图目录里的同名图（`/home/congcongliu/figs/t1flair_v3_af_trend/`）。
- AF24 课题的 `logs/` 里有掩膜抽样的 L1 结果（`l1espirit_*_d{b,c,d}_lam*.json`），`lambda_grid_check.py` 按文件名排除。
