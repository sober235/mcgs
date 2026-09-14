# jiangyizhou T1+FLAIR：AF12 / AF16 / AF20 跨加速倍数复核（2026-09-14）

问题：三个回顾性课题的全部重建结果，指标是不是随加速倍数提高而逐级变差？对所有对比方法和 MC-GS 都成立吗？在同一层上也成立吗？

本文件夹只读取三个课题，不改它们：

| 课题 | 掩膜点数 | 加速倍数（整平面，T1 / FLAIR） |
|---|---|---|
| `T1FLAIR_AF12_v3_jiangyizhou` | 3847 | 11.71 / 12.86 |
| `T1FLAIR_AF16_v3_jiangyizhou` | 2966 | 15.19 / 16.49 |
| `T1FLAIR_AF20_v3_jiangyizhou` | 2427 | 18.56 / 19.96 |

加速倍数口径（用户 2026-09-14 定）：整个相位编码平面的线数 ÷ 该对比度真正带数据的采样线数，也就是 BART `pics` 打印的 `Acc`。
T1 采满整个平面；FLAIR 有厂商快门（34139/45056 线），掩膜落在快门外的线对 FLAIR 是空的，不计入，所以 FLAIR 的倍数高于 T1。
AF20 课题旧文档里的"标称 AF20，实际 T1 18.56×、FLAIR 15.13×（对采集区）"是旧口径，还没有改（改标签待用户点头）。

## 1. 口径核对

- 三个课题的 GT（`recon/gt_{T1,FLAIR}_complex.npy`）逐位相同。
- 打分代码 `diag_artefact/headonly.py` 在 mcgs_af16_wt、mcgs_af12_wt、mcgs_af16_p3_wt、Paired_ARC_GS_general_framework 四个检出里 md5 相同（32a4e4a9）。
- 每个课题每个方法的存盘重建体（12 方法 × 3 加速倍数 × 2 对比度 = 72 个）用同一套代码重新打分，与各课题原表的最大差异：
  头内 PSNR 3.1e-5 dB、SSIM 2.4e-7、HFEN 3.9e-7（`logs/rescore_af*.json`，`logs/af_trend_summary.txt` 开头）。原表可以复现。
- 经典方法的 λ：三个加速倍数都取各自网格上的最优值，且都在网格内部；唯一例外是 AF16 的 Joint SENSE 只扫了 0.0007 和 0.001 两个点，最优点贴边（`logs/lambda_grid_check.txt`）。

各方法用的体积（`logs/rescore_af*.json` 的 `file` 字段）：经典方法取课题表保留的 λ；MC-GS / GS-single 在 AF12 用 AF12 那条线的定稿配方 P14，在 AF16、AF20 用 AF16 三期配方。两套配方接近（P14 多 rotation_lr 0.008，相位 TV 0.01 对 0.03），但不完全相同。完全相同的 P6 配方只在 AF12 和 AF16 有运行，交叉核对见第 2 节。P-LORAKS 只在 AF12 跑过，不纳入。

## 2. 整个体积：头内三项逐级变差

头内 PSNR（dB），AF12 → AF16 → AF20：

| 方法 | T1 | FLAIR | 头内 PSNR、SSIM、HFEN 两步都变差（T1 / FLAIR） |
|---|---|---|---|
| MC-GS | 37.015 → 35.706 → 34.832 | 36.360 → 34.928 → 34.179 | 是 / 是 |
| GS-single | 36.865 → 35.558 → 34.667 | 36.324 → 34.886 → 34.127 | 是 / 是 |
| L1-ESPIRiT | 36.426 → 35.209 → 34.371 | 35.757 → 34.503 → 33.688 | 是 / 是 |
| CS-SENSE 收敛 TV | 36.162 → 34.945 → 34.003 | 35.581 → 34.302 → 33.494 | 是 / 是 |
| Joint SENSE | 36.070 → 34.831 → 33.885 | 35.624 → 34.355 → 33.544 | 是 / 是 |
| CS-SENSE 35 步 | 35.762 → 34.409 → 33.418 | 35.084 → 33.806 → 33.027 | 是 / 是 |
| ZS-SSDU | 35.724 → 34.150 → 32.892 | 35.034 → 34.005 → 32.660 | 是 / 是 |
| CG-SENSE | 33.514 → 32.354 → 31.425 | 33.957 → 32.826 → 32.089 | 是 / 是 |
| MC-INR | 32.192 → 31.590 → 30.969 | 33.042 → 32.243 → 31.666 | 是 / 是 |
| 3D-DIP | 31.363 → 30.994 → 30.996 | 30.810 → 30.650 → 30.637 | PSNR 否 / HFEN 否 |
| 3D-INR | 31.638 → 30.375 → 29.456 | 31.190 → 30.257 → 29.809 | 是 / 是 |
| ZF | 26.130 → 25.841 → 25.639 | 28.193 → 27.879 → 27.636 | 是 / 是 |

SSIM、HFEN 的完整数值和每一步的变化在 `logs/af_trend_tables.md`；图 `figures/jyz_t1flair_af12_16_20_metric_steps.png`（点在 0 左边 = 这一步变差）。

- 例外只有 3D-DIP 的 AF16 → AF20：T1 头内 PSNR +0.002 dB，FLAIR 头内 HFEN 好了 0.0001，其余指标同一步仍变差。它固定 5000 步、在两个加速倍数下都欠拟合（AF20 记录已写），AF20 换掩膜重抽时绝对 PSNR 波动为 0.007–0.04 dB，这是持平。
- 排名三个加速倍数都不变：MC-GS 第一，GS-single 第二，L1-ESPIRiT 是最强的非 GS 方法。MC-GS 对 L1-ESPIRiT 的余量 T1 +0.59 → +0.50 → +0.46，FLAIR +0.60 → +0.43 → +0.49，没有随加速变大。
- 加速每翻一倍，多数方法掉 2.7–4.0 dB；ZS-SSDU 掉得最快（T1 约 4.2–4.4 dB，FLAIR 第二步 4.9 dB）。
- 同配方交叉核对（P6，各自运行的分数文件）：AF12 → AF16，T1 36.753 → 35.552，FLAIR 36.165 → 34.807，SSIM、HFEN 也都变差，所以 AF12 → AF16 这一步的下降不是配方差异造成的。
- 全体积 PSNR（原表，未重算）也基本单调，只有 MC-INR 例外（T1 30.151 → 30.285 → 30.432）：它的全体积 PSNR 低于头内，误差主要在头外背景，头内三项仍逐级变差。

## 3. 同一层

单层指标 = 出图脚本 `fig_three_planes_compact.py` 面板上的定义（头内 PSNR、7×7 窗 SSIM 头内平均、头内 NMSE）；AF20 定稿图上印的数字能原样算出；全部层用的向量化 SSIM 与 skimage 在展示层上差 1e-13。

展示层（12 方法 × 3 面 × 3 指标 = 108 格）两步都变差的格数：

| 层组 | T1 | FLAIR |
|---|---|---|
| 180/124/76（定稿图） | 97/108 | 105/108 |
| 174/110/47 | 99/108 | 108/108 |

没变差的格子只有 3D-DIP（AF16 → AF20 略好，如 T1 横断 180 PSNR 33.419 → 33.004 → 33.085）和 MC-INR 的 3 格 SSIM（AF12 → AF16 略升，其 PSNR、NMSE 仍变差）。

全部层（头部占该层面积 ≥ 5%：横断 243 层，冠状 187 / 193 层，矢状 167 / 166 层，T1 / FLAIR）上三项都两步变差的层占比：

- MC-GS、GS-single：T1 三个面 100%；FLAIR 除冠状第 37 层（头部覆盖边缘，头只占 7%，AF16 → AF20 回升 0.2 dB）外全部。
- L1-ESPIRiT、两个 CS-SENSE、Joint SENSE、3D-INR、ZF：至少 97.9%，例外层都在覆盖边缘；ZS-SSDU 的 FLAIR 冠状 96%，7 个例外层也在边缘。
- CG-SENSE：T1 ≥ 99%，FLAIR 横断 87%。MC-INR：PSNR 在 T1 为 86–99%、FLAIR 横断 67%，SSIM 在 T1 为 62–78%。3D-DIP：29–71%（AF16 → AF20 的中位变化 −0.06 到 +0.005 dB）。

明细在 `logs/slice_trend_summary.txt`，逐层数值在 `logs/slice_trend_{T1,FLAIR}.json`。

## 4. 同日的 AF20 复查（文件在 AF20 课题里）

- 掩膜复查图 `T1FLAIR_AF20_v3_jiangyizhou/figures/t1flair_v3_af20_mask_T1_FLAIR_reinspect.png`（脚本 `scripts/fig_mask_reinspect.py`）：T1 和 FLAIR 两个通道读的是同一个 `mask_af20_3d.h5`（md5 9c889b61），逐线相同；T1 2427 线全部有数据（18.56×），FLAIR 2257 线有数据、170 线落在快门外（19.96×）。
- 所有 AF20 结果（主表各臂、MC-GS 和 GS-single、MC-INR、定稿图）都来自这张掩膜：各运行日志的掩膜路径与点数、BART 打印的 `Acc`、图中 MC-GS 体积与 `af20_p3_joint_full4k` 保存的体积逐位相同。例外只有掩膜抽样方差的 b/c/d 三次（另外三张掩膜）。
- 定稿图 `..._180_124_76_inset48_headerr_p3.png`（T1、FLAIR）按 `finalize_af20_jyz.sh` 的命令从存盘体积重出，与原图逐像素相同。

## 5. 复现

```
T=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_16_20_trend_v3_jiangyizhou
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
bash $T/scripts/run_rescore.sh                                   # 六个后台作业，CPU，约 25 分钟 -> logs/rescore_af*.json
PYTHONNOUSERSITE=1 $PY $T/scripts/analyze_af_trend.py | tee $T/logs/af_trend_summary.txt
python3 $T/scripts/lambda_grid_check.py | tee $T/logs/lambda_grid_check.txt
bash $T/scripts/run_slice_trend.sh                               # 两个后台作业，约 3 分钟 -> logs/slice_trend_*.json
PYTHONNOUSERSITE=1 $PY $T/scripts/slice_summary.py | tee $T/logs/slice_trend_summary.txt
```

坑：
- 后台作业要 `setsid nohup` 脱离会话，否则会话的内存看门狗会杀掉（脚本里已这样写）。
- `analyze_af_trend.py` 会覆盖出图目录里的同名图；`fig_mask_reinspect.py` 同理。
- 路径检查别用 `"SP" in text` 这种子串判断："L1-ESPIRiT" 里就有 "SP"。
