# jiangyizhou T1+FLAIR：AF12 / AF16 / AF20 / AF24 跨加速倍数复核（2026-09-15）

问题：四个回顾性课题的全部重建结果，指标是不是随加速倍数提高而逐级变差？MC-GS 对最强对手的余量怎么随加速倍数变化？

这是 `T1FLAIR_AF12_16_20_trend_v3_jiangyizhou`（09-14，三档）加上 AF24 课题后的扩展，脚本从那里泛化而来（档位 12/16/20/24，判定改为"每一步都变差"）。原三档课题不动。本文件夹只读取四个课题，不改它们：

| 课题 | 掩膜点数 | 加速倍数（整平面，T1 / FLAIR） |
|---|---|---|
| `T1FLAIR_AF12_v3_jiangyizhou` | 3847 | 11.71 / 12.86 |
| `T1FLAIR_AF16_v3_jiangyizhou` | 2966 | 15.19 / 16.49 |
| `T1FLAIR_AF20_v3_jiangyizhou` | 2427 | 18.56 / 19.96 |
| `T1FLAIR_AF24_v3_jiangyizhou` | 1877 | 24.00 / 25.51 |

加速倍数口径（用户 2026-09-14 定）：整个相位编码平面的线数 ÷ 该对比度真正带数据的采样线数。AF24 课题按这个口径让 T1 正好 24×。

## 1. 口径核对

- 四个课题的 GT（`recon/gt_{T1,FLAIR}_complex.npy`）逐位相同（md5 T1 af619da6、FLAIR c302014a）。打分代码 `diag_artefact/headonly.py` md5 32a4e4a9，与三档复核相同。
- 每个课题每个方法的存盘重建体用同一套代码重新打分：94 个（12 方法 × 4 档 × 2 对比度，减去 AF24 没跑的 MC-INR 两个）。与各课题原表的最大差异：头内 PSNR 3.1e-5 dB、SSIM 4.2e-7、HFEN 3.9e-7（`logs/af_trend_summary.txt` 开头）。原表可以复现；AF12/16/20 的 72 个与三档复核的重打分逐位相同（216 个数值最大差 0.0）。
- 经典方法的 λ（`logs/lambda_grid_check.txt`）：24 条曲线里 22 条最优在网格内部；唯一例外仍是 AF16 的 Joint SENSE，只扫了 0.0007、0.001 两个点。AF24 的六条全在内部。
- MC-GS / GS-single：AF12 用 AF12 那条线的定稿配方 P14，AF16、AF20、AF24 用 AF16 三期配方（两套接近但不完全相同，三档复核第 2 节有 P6 同配方交叉核对）。

## 2. 整个体积：头内三项逐级变差

头内 PSNR（dB），AF12 → AF16 → AF20 → AF24：

| 方法 | T1 | FLAIR | 头内 PSNR、SSIM、HFEN 每一步都变差（T1 / FLAIR） |
|---|---|---|---|
| MC-GS | 37.015 → 35.706 → 34.832 → 33.767 | 36.360 → 34.928 → 34.179 → 33.182 | 是 / 是 |
| GS-single | 36.865 → 35.558 → 34.667 → 33.588 | 36.324 → 34.886 → 34.127 → 33.110 | 是 / 是 |
| L1-ESPIRiT | 36.426 → 35.209 → 34.371 → 33.401 | 35.757 → 34.503 → 33.688 → 32.782 | 是 / 是 |
| CS-SENSE 收敛 TV | 36.162 → 34.945 → 34.003 → 32.820 | 35.581 → 34.302 → 33.494 → 32.573 | 是 / 是 |
| Joint SENSE | 36.070 → 34.831 → 33.885 → 32.730 | 35.624 → 34.355 → 33.544 → 32.605 | 是 / 是 |
| CS-SENSE 35 步 | 35.762 → 34.409 → 33.418 → 32.231 | 35.084 → 33.806 → 33.027 → 32.165 | 是 / 是 |
| ZS-SSDU | 35.724 → 34.150 → 32.892 → 32.578 | 35.034 → 34.005 → 32.660 → 32.107 | 是 / 是 |
| CG-SENSE | 33.514 → 32.354 → 31.425 → 30.340 | 33.957 → 32.826 → 32.089 → 31.259 | 是 / 是 |
| MC-INR | 32.192 → 31.590 → 30.969 → 未跑 | 33.042 → 32.243 → 31.666 → 未跑 | 是 / 是（只有前两步） |
| 3D-DIP | 31.363 → 30.994 → 30.996 → 30.500 | 30.810 → 30.650 → 30.637 → 30.360 | PSNR 否 / HFEN 否 |
| 3D-INR | 31.638 → 30.375 → 29.456 → 28.444 | 31.190 → 30.257 → 29.809 → 29.146 | 是 / 是 |
| ZF | 26.130 → 25.841 → 25.639 → 25.333 | 28.193 → 27.879 → 27.636 → 27.301 | 是 / 是 |

SSIM、HFEN 的完整数值和每一步的变化在 `logs/af_trend_tables.md`；图 `figures/jyz_t1flair_af12_16_20_24_metric_steps.png`（点在 0 左边 = 这一步变差，三种颜色和点形 = 三步）。

- 例外仍只有 3D-DIP 的 AF16 → AF20 那一步（T1 头内 PSNR +0.002 dB，FLAIR 头内 HFEN 好 0.0001），它在 AF20 → AF24 这一步三项都变差。
- 加进 AF24 这一步没有新的例外；AF20 → AF24 每个方法都变差。
- 按整平面加速倍数每翻一倍折算（`logs/af_trend_summary.txt` 的 compact summary）：大多数方法这一步掉 2.3–3.2 dB/倍，与前两步同一量级；ZF、3D-DIP、ZS-SSDU 和 FLAIR 的 3D-INR 掉得少。ZS-SSDU 这一步只掉 0.85 / 1.56 dB/倍（前一步 4.35 / 4.88）：AF24 时它训练到第 73 / 59 轮才早停（AF20 为 53 / 41），它在不少层上这一步略有回升（第 4 节）。

## 3. MC-GS 对最强对手的余量（头内 PSNR，同一打分器）

**09-15 下午更正**：GS-single 算对比方法（用户 09-13），所以主余量是下表的"MC-GS − GS-single"一列：T1 +0.150 → +0.148 → +0.165 → +0.179，FLAIR +0.036 → +0.042 → +0.052 → +0.071，逐档变大。L1-ESPIRiT 已移出对比（用户 09-15），下表"最强对手"两列是旧口径下对 L1-ESPIRiT 的历史数字，不再作为余量引用。AF24 的 L1-SPIRiT、MC-INR 补跑登记后，本课题会按新口径（不列 L1-ESPIRiT、加 L1-SPIRiT）重出。

| 档位 | T1：MC-GS − 最强对手 | GS-single − 最强对手 | MC-GS − GS-single | FLAIR：MC-GS − 最强对手 | GS-single − 最强对手 | MC-GS − GS-single |
|---|---|---|---|---|---|---|
| AF12 | +0.589 | +0.439 | +0.150 | +0.602 | +0.567 | +0.036 |
| AF16 | +0.497 | +0.349 | +0.148 | +0.425 | +0.383 | +0.042 |
| AF20 | +0.461 | +0.296 | +0.165 | +0.491 | +0.439 | +0.052 |
| AF24 | **+0.366** | +0.187 | +0.179 | **+0.400** | +0.329 | +0.071 |

最强对手在四档两对比度上都是 L1-ESPIRiT（`logs/af_trend_margins.json`）。

- T1 余量逐档变小（+0.59 → +0.50 → +0.46 → +0.37）；FLAIR 不单调（+0.60 → +0.43 → +0.49 → +0.40），AF24 是四档里最低的。靠提高加速倍数拉大余量，这份证据不支持。
- 缩小的是表征本身：GS-single 对 L1-ESPIRiT 的领先 T1 从 +0.44 降到 +0.19；联合带来的增益反而逐档变大（T1 +0.150 → +0.179，FLAIR +0.036 → +0.071）。
- 每档只有一张掩膜。AF20 四张掩膜测过余量的抽样 SD 为 0.002 / 0.004 dB；AF24 的三张额外掩膜正在跑（AF24 课题 `scripts/run_all_af24.py --study draws`，结果进 AF24 课题 `logs/mask_draws_af24_summary.json`）。AF12 的 MC-GS 配方不同，AF12 那一档余量不能与后三档严格同配方比较。

## 4. 同一层

单层指标 = 出图脚本 `fig_three_planes_compact.py` 面板上的定义（头内 PSNR、7×7 窗 SSIM 头内平均、头内 NMSE）。向量化 SSIM 与 skimage 在展示层上最多差 7.5e-13；AF20 与 AF24 定稿图上印的数字都能原样算出（`logs/slice_trend_summary.txt`）。

展示层（12 方法 × 3 面 × 3 指标 = 108 格）每一步都变差的格数：

| 层组 | T1 | FLAIR |
|---|---|---|
| 180/124/76（定稿图） | 97/108 | 105/108 |
| 174/110/47 | 99/108 | 108/108 |

没变差的格子与三档复核完全相同：3D-DIP 的 AF16 → AF20 那一步，以及 MC-INR 的 3 格 SSIM（AF12 → AF16 略升）。AF20 → AF24 这一步在展示层上没有任何例外。

全部层（头部占该层面积 ≥ 5%：横断 243 层，冠状 187 / 193 层，矢状 167 / 166 层，T1 / FLAIR）上三项每一步都变差的层占比：

- MC-GS、GS-single：T1 三个面 100%；FLAIR 横断、矢状 100%，冠状 97.9%（4 层，都在离覆盖边缘 10 层以内；例：第 37 层头占 7%，是三档复核已记的 AF16 → AF20 回升；第 226、227 层头占 8% / 7%，MC-GS 在 AF20 → AF24 回升 0.09 / 0.47 dB）。
- L1-ESPIRiT、两个 CS-SENSE、Joint SENSE、3D-INR、ZF：至少 97.9%。逐层 PSNR 列出的例外层除 3D-INR FLAIR 横断第 37 层（头占 23%，AF16 → AF20 持平 +0.005 dB）外都在离覆盖边缘 10 层以内（CS-SENSE 35 步和 ZF 没有逐层列出）。CG-SENSE：T1 ≥ 99.2%，FLAIR 横断 86.8%（31 层不单调，其中 10 层在边缘，横断第 0–2 层逐档回升）。
- ZS-SSDU：T1 横断 86.8%（32 层不单调，都不在边缘；列出的几层都是 AF20 → AF24 回升约 0.1 dB，如第 30 层 34.026 → 34.135），FLAIR 横断 77.0%。与第 2 节它这一步掉得少一致。
- MC-INR（只有 AF12–20）与 3D-DIP（29–57%）同三档复核。

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
- 后台作业要 `setsid nohup` 脱离会话（脚本里已这样写）；zsh 里环境变量前缀要直接写出来，别用变量拼。
- `analyze_af_trend.py` 会覆盖出图目录里的同名图（`/home/congcongliu/figs/t1flair_v3_af_trend/`，与三档复核的图不同名）。
- AF24 课题的 `logs/` 里有掩膜抽样的 L1 结果（`l1espirit_*_d{b,c,d}_lam*.json`）；`lambda_grid_check.py` 与 AF24 的 `summarize.py` 都按文件名排除它们。
