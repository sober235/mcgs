# 测试集 T1 + FLAIR 不同加速倍数：结果（草稿，部分数据）

> **草稿：只含已完成的结果。** 完整格子（扫描 x 加速倍数；T1 与 FLAIR 两种对比度的 11 个方法都齐全）8 / 24。不完整的格子对两种对比度都不计入名次与统计。全部跑完后由 `autoresearch/t1flair_af_testset_20261121/collect_stats.py --pipeline` 重新生成本文件、table.json、stats.json、table.md 与图。

口径：头部掩膜内（|GT| > 0.05 P99.9）的 PSNR / SSIM / NMSE；评分代码是各课题共用的 `t1t2_common.Pair.score`，真值来自该扫描自己的全采样 k 空间，所有数字都由保存的体积重新计算，再与各运行脚本记录的分数对照（见第 6 节）。NMSE 的差值写成相对降低（%），其余为绝对差；大于 0 表示 MC-GS 更好。

## 1. MC-GS 在完整格子里是否第一

| 对比度 | AF | 完整格子（共 6） | PSNR 第一 | SSIM 第一 | NMSE 第一 |
|---|---|---|---|---|---|
| T1 | 12 | 2 | 1 / 2 | 2 / 2 | 1 / 2 |
| T1 | 16 | 2 | 1 / 2 | 2 / 2 | 1 / 2 |
| T1 | 20 | 2 | 1 / 2 | 2 / 2 | 1 / 2 |
| T1 | 24 | 2 | 2 / 2 | 2 / 2 | 2 / 2 |
| FLAIR | 12 | 2 | 2 / 2 | 2 / 2 | 2 / 2 |
| FLAIR | 16 | 2 | 1 / 2 | 2 / 2 | 1 / 2 |
| FLAIR | 20 | 2 | 1 / 2 | 2 / 2 | 1 / 2 |
| FLAIR | 24 | 2 | 1 / 2 | 1 / 2 | 1 / 2 |

合计：完整格子 8 / 24，即 16 列（T1 与 FLAIR 各一列）；MC-GS 第一的列数：PSNR 10 / 16，SSIM 15 / 16，NMSE 10 / 16。分对比度：T1 PSNR 5 / 8，SSIM 8 / 8，NMSE 5 / 8；FLAIR PSNR 5 / 8，SSIM 7 / 8，NMSE 5 / 8。

### MC-GS 不是第一的列（逐个列出，不省略）

| 扫描 | AF | 对比度 | 指标 | MC-GS 名次 | 第一名 | 第一名数值 | MC-GS 数值 |
|---|---|---|---|---|---|---|---|
| vol7_tangbin_790 | 12 | T1 | PSNR | 2 | GS-single | 37.021 | 37.005 |
| vol7_tangbin_790 | 12 | T1 | NMSE | 2 | GS-single | 0.00874 | 0.00877 |
| vol7_tangbin_790 | 16 | T1 | PSNR | 2 | GS-single | 35.632 | 35.589 |
| vol7_tangbin_790 | 16 | T1 | NMSE | 2 | GS-single | 0.01203 | 0.01215 |
| vol7_tangbin_790 | 16 | FLAIR | PSNR | 2 | GS-single | 33.438 | 33.419 |
| vol7_tangbin_790 | 16 | FLAIR | NMSE | 2 | GS-single | 0.01986 | 0.01995 |
| vol7_tangbin_790 | 20 | T1 | PSNR | 2 | GS-single | 34.689 | 34.672 |
| vol7_tangbin_790 | 20 | T1 | NMSE | 2 | GS-single | 0.01495 | 0.01501 |
| vol7_tangbin_790 | 20 | FLAIR | PSNR | 2 | GS-single | 32.696 | 32.680 |
| vol7_tangbin_790 | 20 | FLAIR | NMSE | 2 | GS-single | 0.02356 | 0.02365 |
| vol7_tangbin_790 | 24 | FLAIR | PSNR | 2 | GS-single | 31.802 | 31.797 |
| vol7_tangbin_790 | 24 | FLAIR | SSIM | 2 | ZS-SSDU | 0.8474 | 0.8472 |
| vol7_tangbin_790 | 24 | FLAIR | NMSE | 2 | GS-single | 0.02894 | 0.02898 |

## 2. 病例间：MC-GS 相对两个对照的差值

对照：GS-single；该格最强的非 GS 方法（按该格头部 PSNR 选，三个指标都与同一个方法比）。“符号一致”= 差值大于 0 的单位数 / 单位总数。p 为精确双侧 Wilcoxon 符号秩检验（零差值舍去，并列用平均秩；单位数 4 时 p 最小 0.125，6 时 0.031，见第 5 节）。

### PSNR差值 dB

| 对比度 | AF | 对照 | 单位 | 均值 | 标准差 | 最小 | 符号一致 | p |
|---|---|---|---|---|---|---|---|---|
| T1 | 12 | GS-single | 2 次扫描 | 0.06545 | 0.116 | -0.01677 | 1 / 2 | 1.000 |
| T1 | 12 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 0.1477 | 0 | 0.1477 | 1 / 1 | 1.000 |
| T1 | 12 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | -0.01677 | 0 | -0.01677 | 0 / 1 | 1.000 |
| T1 | 12 | GS-single | 1 个人（重复扫描先平均） | 0.1477 | 0 | 0.1477 | 1 / 1 | 1.000 |
| T1 | 12 | 最强非 GS | 2 次扫描 | 0.8419 | 0.249 | 0.6658 | 2 / 2 | 0.500 |
| T1 | 12 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 1.018 | 0 | 1.018 | 1 / 1 | 1.000 |
| T1 | 12 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 0.6658 | 0 | 0.6658 | 1 / 1 | 1.000 |
| T1 | 12 | 最强非 GS | 1 个人（重复扫描先平均） | 1.018 | 0 | 1.018 | 1 / 1 | 1.000 |
| T1 | 16 | GS-single | 2 次扫描 | 0.06083 | 0.148 | -0.0437 | 1 / 2 | 1.000 |
| T1 | 16 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 0.1654 | 0 | 0.1654 | 1 / 1 | 1.000 |
| T1 | 16 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | -0.0437 | 0 | -0.0437 | 0 / 1 | 1.000 |
| T1 | 16 | GS-single | 1 个人（重复扫描先平均） | 0.1654 | 0 | 0.1654 | 1 / 1 | 1.000 |
| T1 | 16 | 最强非 GS | 2 次扫描 | 0.8765 | 0.31 | 0.6574 | 2 / 2 | 0.500 |
| T1 | 16 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 1.096 | 0 | 1.096 | 1 / 1 | 1.000 |
| T1 | 16 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 0.6574 | 0 | 0.6574 | 1 / 1 | 1.000 |
| T1 | 16 | 最强非 GS | 1 个人（重复扫描先平均） | 1.096 | 0 | 1.096 | 1 / 1 | 1.000 |
| T1 | 20 | GS-single | 2 次扫描 | 0.08721 | 0.148 | -0.01731 | 1 / 2 | 1.000 |
| T1 | 20 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 0.1917 | 0 | 0.1917 | 1 / 1 | 1.000 |
| T1 | 20 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | -0.01731 | 0 | -0.01731 | 0 / 1 | 1.000 |
| T1 | 20 | GS-single | 1 个人（重复扫描先平均） | 0.1917 | 0 | 0.1917 | 1 / 1 | 1.000 |
| T1 | 20 | 最强非 GS | 2 次扫描 | 1.036 | 0.331 | 0.8019 | 2 / 2 | 0.500 |
| T1 | 20 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 1.27 | 0 | 1.27 | 1 / 1 | 1.000 |
| T1 | 20 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 0.8019 | 0 | 0.8019 | 1 / 1 | 1.000 |
| T1 | 20 | 最强非 GS | 1 个人（重复扫描先平均） | 1.27 | 0 | 1.27 | 1 / 1 | 1.000 |
| T1 | 24 | GS-single | 2 次扫描 | 0.1029 | 0.136 | 0.006867 | 2 / 2 | 0.500 |
| T1 | 24 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 0.199 | 0 | 0.199 | 1 / 1 | 1.000 |
| T1 | 24 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | 0.006867 | 0 | 0.006867 | 1 / 1 | 1.000 |
| T1 | 24 | GS-single | 1 个人（重复扫描先平均） | 0.199 | 0 | 0.199 | 1 / 1 | 1.000 |
| T1 | 24 | 最强非 GS | 2 次扫描 | 1.235 | 0.405 | 0.9484 | 2 / 2 | 0.500 |
| T1 | 24 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 1.521 | 0 | 1.521 | 1 / 1 | 1.000 |
| T1 | 24 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 0.9484 | 0 | 0.9484 | 1 / 1 | 1.000 |
| T1 | 24 | 最强非 GS | 1 个人（重复扫描先平均） | 1.521 | 0 | 1.521 | 1 / 1 | 1.000 |
| FLAIR | 12 | GS-single | 2 次扫描 | 0.01745 | 0.0176 | 0.004973 | 2 / 2 | 0.500 |
| FLAIR | 12 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 0.02993 | 0 | 0.02993 | 1 / 1 | 1.000 |
| FLAIR | 12 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | 0.004973 | 0 | 0.004973 | 1 / 1 | 1.000 |
| FLAIR | 12 | GS-single | 1 个人（重复扫描先平均） | 0.02993 | 0 | 0.02993 | 1 / 1 | 1.000 |
| FLAIR | 12 | 最强非 GS | 2 次扫描 | 0.645 | 0.281 | 0.446 | 2 / 2 | 0.500 |
| FLAIR | 12 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 0.844 | 0 | 0.844 | 1 / 1 | 1.000 |
| FLAIR | 12 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 0.446 | 0 | 0.446 | 1 / 1 | 1.000 |
| FLAIR | 12 | 最强非 GS | 1 个人（重复扫描先平均） | 0.844 | 0 | 0.844 | 1 / 1 | 1.000 |
| FLAIR | 16 | GS-single | 2 次扫描 | 0.01301 | 0.046 | -0.01949 | 1 / 2 | 1.000 |
| FLAIR | 16 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 0.04551 | 0 | 0.04551 | 1 / 1 | 1.000 |
| FLAIR | 16 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | -0.01949 | 0 | -0.01949 | 0 / 1 | 1.000 |
| FLAIR | 16 | GS-single | 1 个人（重复扫描先平均） | 0.04551 | 0 | 0.04551 | 1 / 1 | 1.000 |
| FLAIR | 16 | 最强非 GS | 2 次扫描 | 0.4901 | 0.217 | 0.3367 | 2 / 2 | 0.500 |
| FLAIR | 16 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 0.6436 | 0 | 0.6436 | 1 / 1 | 1.000 |
| FLAIR | 16 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 0.3367 | 0 | 0.3367 | 1 / 1 | 1.000 |
| FLAIR | 16 | 最强非 GS | 1 个人（重复扫描先平均） | 0.6436 | 0 | 0.6436 | 1 / 1 | 1.000 |
| FLAIR | 20 | GS-single | 2 次扫描 | 0.01687 | 0.046 | -0.01565 | 1 / 2 | 1.000 |
| FLAIR | 20 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 0.04939 | 0 | 0.04939 | 1 / 1 | 1.000 |
| FLAIR | 20 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | -0.01565 | 0 | -0.01565 | 0 / 1 | 1.000 |
| FLAIR | 20 | GS-single | 1 个人（重复扫描先平均） | 0.04939 | 0 | 0.04939 | 1 / 1 | 1.000 |
| FLAIR | 20 | 最强非 GS | 2 次扫描 | 0.5342 | 0.304 | 0.3193 | 2 / 2 | 0.500 |
| FLAIR | 20 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 0.7491 | 0 | 0.7491 | 1 / 1 | 1.000 |
| FLAIR | 20 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 0.3193 | 0 | 0.3193 | 1 / 1 | 1.000 |
| FLAIR | 20 | 最强非 GS | 1 个人（重复扫描先平均） | 0.7491 | 0 | 0.7491 | 1 / 1 | 1.000 |
| FLAIR | 24 | GS-single | 2 次扫描 | 0.0302 | 0.0502 | -0.005272 | 1 / 2 | 1.000 |
| FLAIR | 24 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 0.06568 | 0 | 0.06568 | 1 / 1 | 1.000 |
| FLAIR | 24 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | -0.005272 | 0 | -0.005272 | 0 / 1 | 1.000 |
| FLAIR | 24 | GS-single | 1 个人（重复扫描先平均） | 0.06568 | 0 | 0.06568 | 1 / 1 | 1.000 |
| FLAIR | 24 | 最强非 GS | 2 次扫描 | 0.5097 | 0.293 | 0.3028 | 2 / 2 | 0.500 |
| FLAIR | 24 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 0.7167 | 0 | 0.7167 | 1 / 1 | 1.000 |
| FLAIR | 24 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 0.3028 | 0 | 0.3028 | 1 / 1 | 1.000 |
| FLAIR | 24 | 最强非 GS | 1 个人（重复扫描先平均） | 0.7167 | 0 | 0.7167 | 1 / 1 | 1.000 |

### SSIM差值 

| 对比度 | AF | 对照 | 单位 | 均值 | 标准差 | 最小 | 符号一致 | p |
|---|---|---|---|---|---|---|---|---|
| T1 | 12 | GS-single | 2 次扫描 | 0.001384 | 0.00104 | 0.0006468 | 2 / 2 | 0.500 |
| T1 | 12 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 0.00212 | 0 | 0.00212 | 1 / 1 | 1.000 |
| T1 | 12 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | 0.0006468 | 0 | 0.0006468 | 1 / 1 | 1.000 |
| T1 | 12 | GS-single | 1 个人（重复扫描先平均） | 0.00212 | 0 | 0.00212 | 1 / 1 | 1.000 |
| T1 | 12 | 最强非 GS | 2 次扫描 | 0.005463 | 0.00137 | 0.004496 | 2 / 2 | 0.500 |
| T1 | 12 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 0.00643 | 0 | 0.00643 | 1 / 1 | 1.000 |
| T1 | 12 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 0.004496 | 0 | 0.004496 | 1 / 1 | 1.000 |
| T1 | 12 | 最强非 GS | 1 个人（重复扫描先平均） | 0.00643 | 0 | 0.00643 | 1 / 1 | 1.000 |
| T1 | 16 | GS-single | 2 次扫描 | 0.001807 | 0.00145 | 0.000782 | 2 / 2 | 0.500 |
| T1 | 16 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 0.002832 | 0 | 0.002832 | 1 / 1 | 1.000 |
| T1 | 16 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | 0.000782 | 0 | 0.000782 | 1 / 1 | 1.000 |
| T1 | 16 | GS-single | 1 个人（重复扫描先平均） | 0.002832 | 0 | 0.002832 | 1 / 1 | 1.000 |
| T1 | 16 | 最强非 GS | 2 次扫描 | 0.007854 | 0.00236 | 0.006184 | 2 / 2 | 0.500 |
| T1 | 16 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 0.009523 | 0 | 0.009523 | 1 / 1 | 1.000 |
| T1 | 16 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 0.006184 | 0 | 0.006184 | 1 / 1 | 1.000 |
| T1 | 16 | 最强非 GS | 1 个人（重复扫描先平均） | 0.009523 | 0 | 0.009523 | 1 / 1 | 1.000 |
| T1 | 20 | GS-single | 2 次扫描 | 0.002482 | 0.00179 | 0.001217 | 2 / 2 | 0.500 |
| T1 | 20 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 0.003747 | 0 | 0.003747 | 1 / 1 | 1.000 |
| T1 | 20 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | 0.001217 | 0 | 0.001217 | 1 / 1 | 1.000 |
| T1 | 20 | GS-single | 1 个人（重复扫描先平均） | 0.003747 | 0 | 0.003747 | 1 / 1 | 1.000 |
| T1 | 20 | 最强非 GS | 2 次扫描 | 0.01105 | 0.00283 | 0.009047 | 2 / 2 | 0.500 |
| T1 | 20 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 0.01305 | 0 | 0.01305 | 1 / 1 | 1.000 |
| T1 | 20 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 0.009047 | 0 | 0.009047 | 1 / 1 | 1.000 |
| T1 | 20 | 最强非 GS | 1 个人（重复扫描先平均） | 0.01305 | 0 | 0.01305 | 1 / 1 | 1.000 |
| T1 | 24 | GS-single | 2 次扫描 | 0.00329 | 0.00201 | 0.00187 | 2 / 2 | 0.500 |
| T1 | 24 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 0.00471 | 0 | 0.00471 | 1 / 1 | 1.000 |
| T1 | 24 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | 0.00187 | 0 | 0.00187 | 1 / 1 | 1.000 |
| T1 | 24 | GS-single | 1 个人（重复扫描先平均） | 0.00471 | 0 | 0.00471 | 1 / 1 | 1.000 |
| T1 | 24 | 最强非 GS | 2 次扫描 | 0.01736 | 0.00486 | 0.01392 | 2 / 2 | 0.500 |
| T1 | 24 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 0.0208 | 0 | 0.0208 | 1 / 1 | 1.000 |
| T1 | 24 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 0.01392 | 0 | 0.01392 | 1 / 1 | 1.000 |
| T1 | 24 | 最强非 GS | 1 个人（重复扫描先平均） | 0.0208 | 0 | 0.0208 | 1 / 1 | 1.000 |
| FLAIR | 12 | GS-single | 2 次扫描 | 0.0004454 | 0.00029 | 0.0002401 | 2 / 2 | 0.500 |
| FLAIR | 12 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 0.0006506 | 0 | 0.0006506 | 1 / 1 | 1.000 |
| FLAIR | 12 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | 0.0002401 | 0 | 0.0002401 | 1 / 1 | 1.000 |
| FLAIR | 12 | GS-single | 1 个人（重复扫描先平均） | 0.0006506 | 0 | 0.0006506 | 1 / 1 | 1.000 |
| FLAIR | 12 | 最强非 GS | 2 次扫描 | 0.005392 | 0.00147 | 0.004355 | 2 / 2 | 0.500 |
| FLAIR | 12 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 0.00643 | 0 | 0.00643 | 1 / 1 | 1.000 |
| FLAIR | 12 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 0.004355 | 0 | 0.004355 | 1 / 1 | 1.000 |
| FLAIR | 12 | 最强非 GS | 1 个人（重复扫描先平均） | 0.00643 | 0 | 0.00643 | 1 / 1 | 1.000 |
| FLAIR | 16 | GS-single | 2 次扫描 | 0.0007511 | 0.000969 | 6.616e-05 | 2 / 2 | 0.500 |
| FLAIR | 16 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 0.001436 | 0 | 0.001436 | 1 / 1 | 1.000 |
| FLAIR | 16 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | 6.616e-05 | 0 | 6.616e-05 | 1 / 1 | 1.000 |
| FLAIR | 16 | GS-single | 1 个人（重复扫描先平均） | 0.001436 | 0 | 0.001436 | 1 / 1 | 1.000 |
| FLAIR | 16 | 最强非 GS | 2 次扫描 | 0.004206 | 0.00213 | 0.0027 | 2 / 2 | 0.500 |
| FLAIR | 16 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 0.0027 | 0 | 0.0027 | 1 / 1 | 1.000 |
| FLAIR | 16 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 0.005712 | 0 | 0.005712 | 1 / 1 | 1.000 |
| FLAIR | 16 | 最强非 GS | 1 个人（重复扫描先平均） | 0.0027 | 0 | 0.0027 | 1 / 1 | 1.000 |
| FLAIR | 20 | GS-single | 2 次扫描 | 0.001029 | 0.0012 | 0.000181 | 2 / 2 | 0.500 |
| FLAIR | 20 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 0.001878 | 0 | 0.001878 | 1 / 1 | 1.000 |
| FLAIR | 20 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | 0.000181 | 0 | 0.000181 | 1 / 1 | 1.000 |
| FLAIR | 20 | GS-single | 1 个人（重复扫描先平均） | 0.001878 | 0 | 0.001878 | 1 / 1 | 1.000 |
| FLAIR | 20 | 最强非 GS | 2 次扫描 | 0.007568 | 0.0046 | 0.004315 | 2 / 2 | 0.500 |
| FLAIR | 20 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 0.01082 | 0 | 0.01082 | 1 / 1 | 1.000 |
| FLAIR | 20 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 0.004315 | 0 | 0.004315 | 1 / 1 | 1.000 |
| FLAIR | 20 | 最强非 GS | 1 个人（重复扫描先平均） | 0.01082 | 0 | 0.01082 | 1 / 1 | 1.000 |
| FLAIR | 24 | GS-single | 2 次扫描 | 0.001601 | 0.00132 | 0.0006693 | 2 / 2 | 0.500 |
| FLAIR | 24 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 0.002532 | 0 | 0.002532 | 1 / 1 | 1.000 |
| FLAIR | 24 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | 0.0006693 | 0 | 0.0006693 | 1 / 1 | 1.000 |
| FLAIR | 24 | GS-single | 1 个人（重复扫描先平均） | 0.002532 | 0 | 0.002532 | 1 / 1 | 1.000 |
| FLAIR | 24 | 最强非 GS | 2 次扫描 | 0.008136 | 0.00836 | 0.002225 | 2 / 2 | 0.500 |
| FLAIR | 24 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 0.01405 | 0 | 0.01405 | 1 / 1 | 1.000 |
| FLAIR | 24 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 0.002225 | 0 | 0.002225 | 1 / 1 | 1.000 |
| FLAIR | 24 | 最强非 GS | 1 个人（重复扫描先平均） | 0.01405 | 0 | 0.01405 | 1 / 1 | 1.000 |

### NMSE差值 %（相对降低）

| 对比度 | AF | 对照 | 单位 | 均值 | 标准差 | 最小 | 符号一致 | p |
|---|---|---|---|---|---|---|---|---|
| T1 | 12 | GS-single | 2 次扫描 | 1.478 | 2.64 | -0.3868 | 1 / 2 | 1.000 |
| T1 | 12 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 3.343 | 0 | 3.343 | 1 / 1 | 1.000 |
| T1 | 12 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | -0.3868 | 0 | -0.3868 | 0 / 1 | 1.000 |
| T1 | 12 | GS-single | 1 个人（重复扫描先平均） | 3.343 | 0 | 3.343 | 1 / 1 | 1.000 |
| T1 | 12 | 最强非 GS | 2 次扫描 | 17.55 | 4.73 | 14.21 | 2 / 2 | 0.500 |
| T1 | 12 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 20.9 | 0 | 20.9 | 1 / 1 | 1.000 |
| T1 | 12 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 14.21 | 0 | 14.21 | 1 / 1 | 1.000 |
| T1 | 12 | 最强非 GS | 1 个人（重复扫描先平均） | 20.9 | 0 | 20.9 | 1 / 1 | 1.000 |
| T1 | 16 | GS-single | 2 次扫描 | 1.362 | 3.36 | -1.011 | 1 / 2 | 1.000 |
| T1 | 16 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 3.736 | 0 | 3.736 | 1 / 1 | 1.000 |
| T1 | 16 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | -1.011 | 0 | -1.011 | 0 / 1 | 1.000 |
| T1 | 16 | GS-single | 1 个人（重复扫描先平均） | 3.736 | 0 | 3.736 | 1 / 1 | 1.000 |
| T1 | 16 | 最强非 GS | 2 次扫描 | 18.17 | 5.83 | 14.05 | 2 / 2 | 0.500 |
| T1 | 16 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 22.3 | 0 | 22.3 | 1 / 1 | 1.000 |
| T1 | 16 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 14.05 | 0 | 14.05 | 1 / 1 | 1.000 |
| T1 | 16 | 最强非 GS | 1 个人（重复扫描先平均） | 22.3 | 0 | 22.3 | 1 / 1 | 1.000 |
| T1 | 20 | GS-single | 2 次扫描 | 1.96 | 3.34 | -0.3993 | 1 / 2 | 1.000 |
| T1 | 20 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 4.319 | 0 | 4.319 | 1 / 1 | 1.000 |
| T1 | 20 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | -0.3993 | 0 | -0.3993 | 0 / 1 | 1.000 |
| T1 | 20 | GS-single | 1 个人（重复扫描先平均） | 4.319 | 0 | 4.319 | 1 / 1 | 1.000 |
| T1 | 20 | 最强非 GS | 2 次扫描 | 21.11 | 6.01 | 16.86 | 2 / 2 | 0.500 |
| T1 | 20 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 25.36 | 0 | 25.36 | 1 / 1 | 1.000 |
| T1 | 20 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 16.86 | 0 | 16.86 | 1 / 1 | 1.000 |
| T1 | 20 | 最强非 GS | 1 个人（重复扫描先平均） | 25.36 | 0 | 25.36 | 1 / 1 | 1.000 |
| T1 | 24 | GS-single | 2 次扫描 | 2.319 | 3.06 | 0.158 | 2 / 2 | 0.500 |
| T1 | 24 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 4.479 | 0 | 4.479 | 1 / 1 | 1.000 |
| T1 | 24 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | 0.158 | 0 | 0.158 | 1 / 1 | 1.000 |
| T1 | 24 | GS-single | 1 个人（重复扫描先平均） | 4.479 | 0 | 4.479 | 1 / 1 | 1.000 |
| T1 | 24 | 最强非 GS | 2 次扫描 | 24.58 | 7.02 | 19.62 | 2 / 2 | 0.500 |
| T1 | 24 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 29.54 | 0 | 29.54 | 1 / 1 | 1.000 |
| T1 | 24 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 19.62 | 0 | 19.62 | 1 / 1 | 1.000 |
| T1 | 24 | 最强非 GS | 1 个人（重复扫描先平均） | 29.54 | 0 | 29.54 | 1 / 1 | 1.000 |
| FLAIR | 12 | GS-single | 2 次扫描 | 0.4007 | 0.405 | 0.1145 | 2 / 2 | 0.500 |
| FLAIR | 12 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 0.6868 | 0 | 0.6868 | 1 / 1 | 1.000 |
| FLAIR | 12 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | 0.1145 | 0 | 0.1145 | 1 / 1 | 1.000 |
| FLAIR | 12 | GS-single | 1 个人（重复扫描先平均） | 0.6868 | 0 | 0.6868 | 1 / 1 | 1.000 |
| FLAIR | 12 | 最强非 GS | 2 次扫描 | 13.71 | 5.59 | 9.76 | 2 / 2 | 0.500 |
| FLAIR | 12 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 17.66 | 0 | 17.66 | 1 / 1 | 1.000 |
| FLAIR | 12 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 9.76 | 0 | 9.76 | 1 / 1 | 1.000 |
| FLAIR | 12 | 最强非 GS | 1 个人（重复扫描先平均） | 17.66 | 0 | 17.66 | 1 / 1 | 1.000 |
| FLAIR | 16 | GS-single | 2 次扫描 | 0.2963 | 1.06 | -0.4497 | 1 / 2 | 1.000 |
| FLAIR | 16 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 1.042 | 0 | 1.042 | 1 / 1 | 1.000 |
| FLAIR | 16 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | -0.4497 | 0 | -0.4497 | 0 / 1 | 1.000 |
| FLAIR | 16 | GS-single | 1 个人（重复扫描先平均） | 1.042 | 0 | 1.042 | 1 / 1 | 1.000 |
| FLAIR | 16 | 最强非 GS | 2 次扫描 | 10.62 | 4.47 | 7.459 | 2 / 2 | 0.500 |
| FLAIR | 16 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 13.77 | 0 | 13.77 | 1 / 1 | 1.000 |
| FLAIR | 16 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 7.459 | 0 | 7.459 | 1 / 1 | 1.000 |
| FLAIR | 16 | 最强非 GS | 1 个人（重复扫描先平均） | 13.77 | 0 | 13.77 | 1 / 1 | 1.000 |
| FLAIR | 20 | GS-single | 2 次扫描 | 0.3849 | 1.05 | -0.3611 | 1 / 2 | 1.000 |
| FLAIR | 20 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 1.131 | 0 | 1.131 | 1 / 1 | 1.000 |
| FLAIR | 20 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | -0.3611 | 0 | -0.3611 | 0 / 1 | 1.000 |
| FLAIR | 20 | GS-single | 1 个人（重复扫描先平均） | 1.131 | 0 | 1.131 | 1 / 1 | 1.000 |
| FLAIR | 20 | 最强非 GS | 2 次扫描 | 11.47 | 6.19 | 7.088 | 2 / 2 | 0.500 |
| FLAIR | 20 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 15.84 | 0 | 15.84 | 1 / 1 | 1.000 |
| FLAIR | 20 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 7.088 | 0 | 7.088 | 1 / 1 | 1.000 |
| FLAIR | 20 | 最强非 GS | 1 个人（重复扫描先平均） | 15.84 | 0 | 15.84 | 1 / 1 | 1.000 |
| FLAIR | 24 | GS-single | 2 次扫描 | 0.6897 | 1.15 | -0.1215 | 1 / 2 | 1.000 |
| FLAIR | 24 | GS-single | 对齐组（vol1、vol2、vol4）1 次 | 1.501 | 0 | 1.501 | 1 / 1 | 1.000 |
| FLAIR | 24 | GS-single | 偏移组（vol5、vol6、vol7）1 次 | -0.1215 | 0 | -0.1215 | 0 / 1 | 1.000 |
| FLAIR | 24 | GS-single | 1 个人（重复扫描先平均） | 1.501 | 0 | 1.501 | 1 / 1 | 1.000 |
| FLAIR | 24 | 最强非 GS | 2 次扫描 | 10.97 | 5.99 | 6.735 | 2 / 2 | 0.500 |
| FLAIR | 24 | 最强非 GS | 对齐组（vol1、vol2、vol4）1 次 | 15.21 | 0 | 15.21 | 1 / 1 | 1.000 |
| FLAIR | 24 | 最强非 GS | 偏移组（vol5、vol6、vol7）1 次 | 6.735 | 0 | 6.735 | 1 / 1 | 1.000 |
| FLAIR | 24 | 最强非 GS | 1 个人（重复扫描先平均） | 15.21 | 0 | 15.21 | 1 / 1 | 1.000 |

## 3. 病例内：逐层面配对检验

每个完整格子内，三个平面里头部占层面 >= 5 % 的层面合并（规则同 `t1flair_af_record_fig_20261120/fig_af_overview.py` 所读的 slice_trend 记录），逐层面配对 Wilcoxon（双侧）。这里的 p 值用正态近似（每个病例约 600 层），零差值被舍去；第 2 节的扫描 / 人层面检验才是精确检验。下表对每个（对比度，AF，对照，指标）把扫描分成三类：p < 0.05 且中位差为正（显著更好）、p < 0.05 且中位差为负（显著更差）、其余（不显著），并列出后两类的扫描；最后一列是各扫描里 MC-GS 更好的层面占比的范围。层面属于同一个体积，彼此不独立，p 值只说明该病例内的一致性，不是人群推断。逐病例的全部数值在 stats.json 的 `slices`。

| 对比度 | AF | 对照 | 指标 | 显著更好 | 显著更差（扫描） | 不显著（扫描） | 更好层面占比范围 |
|---|---|---|---|---|---|---|---|
| T1 | 12 | GS-single | PSNR | 1 / 2 | 1（vol7_tangbin_790） | 0（无） | 43% - 99% |
| T1 | 12 | GS-single | SSIM | 2 / 2 | 0（无） | 0（无） | 76% - 99% |
| T1 | 12 | GS-single | NMSE | 1 / 2 | 1（vol7_tangbin_790） | 0（无） | 43% - 99% |
| T1 | 12 | 最强非 GS | PSNR | 2 / 2 | 0（无） | 0（无） | 99% - 100% |
| T1 | 12 | 最强非 GS | SSIM | 2 / 2 | 0（无） | 0（无） | 98% - 100% |
| T1 | 12 | 最强非 GS | NMSE | 2 / 2 | 0（无） | 0（无） | 99% - 100% |
| T1 | 16 | GS-single | PSNR | 1 / 2 | 1（vol7_tangbin_790） | 0（无） | 35% - 99% |
| T1 | 16 | GS-single | SSIM | 2 / 2 | 0（无） | 0（无） | 71% - 99% |
| T1 | 16 | GS-single | NMSE | 1 / 2 | 1（vol7_tangbin_790） | 0（无） | 35% - 99% |
| T1 | 16 | 最强非 GS | PSNR | 2 / 2 | 0（无） | 0（无） | 99% - 100% |
| T1 | 16 | 最强非 GS | SSIM | 2 / 2 | 0（无） | 0（无） | 98% - 100% |
| T1 | 16 | 最强非 GS | NMSE | 2 / 2 | 0（无） | 0（无） | 99% - 100% |
| T1 | 20 | GS-single | PSNR | 1 / 2 | 1（vol7_tangbin_790） | 0（无） | 44% - 99% |
| T1 | 20 | GS-single | SSIM | 2 / 2 | 0（无） | 0（无） | 77% - 99% |
| T1 | 20 | GS-single | NMSE | 1 / 2 | 1（vol7_tangbin_790） | 0（无） | 44% - 99% |
| T1 | 20 | 最强非 GS | PSNR | 2 / 2 | 0（无） | 0（无） | 99% - 100% |
| T1 | 20 | 最强非 GS | SSIM | 2 / 2 | 0（无） | 0（无） | 99% - 100% |
| T1 | 20 | 最强非 GS | NMSE | 2 / 2 | 0（无） | 0（无） | 99% - 100% |
| T1 | 24 | GS-single | PSNR | 2 / 2 | 0（无） | 0（无） | 55% - 100% |
| T1 | 24 | GS-single | SSIM | 2 / 2 | 0（无） | 0（无） | 80% - 100% |
| T1 | 24 | GS-single | NMSE | 2 / 2 | 0（无） | 0（无） | 55% - 100% |
| T1 | 24 | 最强非 GS | PSNR | 2 / 2 | 0（无） | 0（无） | 100% - 100% |
| T1 | 24 | 最强非 GS | SSIM | 2 / 2 | 0（无） | 0（无） | 99% - 100% |
| T1 | 24 | 最强非 GS | NMSE | 2 / 2 | 0（无） | 0（无） | 100% - 100% |
| FLAIR | 12 | GS-single | PSNR | 1 / 2 | 0（无） | 1（vol7_tangbin_790） | 49% - 76% |
| FLAIR | 12 | GS-single | SSIM | 1 / 2 | 0（无） | 1（vol7_tangbin_790） | 54% - 56% |
| FLAIR | 12 | GS-single | NMSE | 1 / 2 | 0（无） | 1（vol7_tangbin_790） | 49% - 76% |
| FLAIR | 12 | 最强非 GS | PSNR | 2 / 2 | 0（无） | 0（无） | 97% - 100% |
| FLAIR | 12 | 最强非 GS | SSIM | 2 / 2 | 0（无） | 0（无） | 92% - 98% |
| FLAIR | 12 | 最强非 GS | NMSE | 2 / 2 | 0（无） | 0（无） | 97% - 100% |
| FLAIR | 16 | GS-single | PSNR | 1 / 2 | 1（vol7_tangbin_790） | 0（无） | 34% - 91% |
| FLAIR | 16 | GS-single | SSIM | 1 / 2 | 0（无） | 1（vol7_tangbin_790） | 49% - 86% |
| FLAIR | 16 | GS-single | NMSE | 1 / 2 | 1（vol7_tangbin_790） | 0（无） | 34% - 91% |
| FLAIR | 16 | 最强非 GS | PSNR | 2 / 2 | 0（无） | 0（无） | 92% - 97% |
| FLAIR | 16 | 最强非 GS | SSIM | 2 / 2 | 0（无） | 0（无） | 75% - 94% |
| FLAIR | 16 | 最强非 GS | NMSE | 2 / 2 | 0（无） | 0（无） | 92% - 97% |
| FLAIR | 20 | GS-single | PSNR | 1 / 2 | 1（vol7_tangbin_790） | 0（无） | 37% - 91% |
| FLAIR | 20 | GS-single | SSIM | 1 / 2 | 0（无） | 1（vol7_tangbin_790） | 55% - 87% |
| FLAIR | 20 | GS-single | NMSE | 1 / 2 | 1（vol7_tangbin_790） | 0（无） | 37% - 91% |
| FLAIR | 20 | 最强非 GS | PSNR | 2 / 2 | 0（无） | 0（无） | 94% - 99% |
| FLAIR | 20 | 最强非 GS | SSIM | 2 / 2 | 0（无） | 0（无） | 86% - 99% |
| FLAIR | 20 | 最强非 GS | NMSE | 2 / 2 | 0（无） | 0（无） | 94% - 99% |
| FLAIR | 24 | GS-single | PSNR | 1 / 2 | 1（vol7_tangbin_790） | 0（无） | 47% - 96% |
| FLAIR | 24 | GS-single | SSIM | 2 / 2 | 0（无） | 0（无） | 61% - 91% |
| FLAIR | 24 | GS-single | NMSE | 1 / 2 | 1（vol7_tangbin_790） | 0（无） | 47% - 96% |
| FLAIR | 24 | 最强非 GS | PSNR | 2 / 2 | 0（无） | 0（无） | 84% - 100% |
| FLAIR | 24 | 最强非 GS | SSIM | 2 / 2 | 0（无） | 0（无） | 70% - 99% |
| FLAIR | 24 | 最强非 GS | NMSE | 2 / 2 | 0（无） | 0（无） | 84% - 100% |

## 4. 缺失清单

不完整格子 16 / 24。每个（扫描，AF，对比度）缺的方法（原因见 table.json 的 `missing`）：

- vol2_jiangdian AF12 FLAIR：L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol2_jiangdian AF12 T1：L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol2_jiangdian AF16 FLAIR：L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol2_jiangdian AF16 T1：L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol2_jiangdian AF20 FLAIR：Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol2_jiangdian AF20 T1：Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol2_jiangdian AF24 FLAIR：Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol2_jiangdian AF24 T1：Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol4_tangbing AF12 FLAIR：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol4_tangbing AF12 T1：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol4_tangbing AF16 FLAIR：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol4_tangbing AF16 T1：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol4_tangbing AF20 FLAIR：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol4_tangbing AF20 T1：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol4_tangbing AF24 FLAIR：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol4_tangbing AF24 T1：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol5_sunxinyi AF12 FLAIR：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol5_sunxinyi AF12 T1：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol5_sunxinyi AF16 FLAIR：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol5_sunxinyi AF16 T1：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol5_sunxinyi AF20 FLAIR：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol5_sunxinyi AF20 T1：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol5_sunxinyi AF24 FLAIR：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol5_sunxinyi AF24 T1：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol6_jiangdian_790 AF12 FLAIR：CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol6_jiangdian_790 AF12 T1：CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol6_jiangdian_790 AF16 FLAIR：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol6_jiangdian_790 AF16 T1：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol6_jiangdian_790 AF20 FLAIR：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol6_jiangdian_790 AF20 T1：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol6_jiangdian_790 AF24 FLAIR：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol6_jiangdian_790 AF24 T1：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR

## 5. 必须写明的限制

- 只有 4 个人。以人为单位的 Wilcoxon 检验即使全胜，精确双侧 p 最小为 0.125；以 6 次扫描为单位最小为 0.031，但其中两次是重复扫描（vol2 与 vol6 是同一个人，vol4 与 vol7 是同一个人），不独立。结论写成“几个里几个一致”。
- 层面级检验的层面彼此不独立（同一体积、三个平面共用数据），只说明病例内的一致性。
- 一个种子。
- ZS-SSDU 是 5 个残差块的版本（各课题登记的实现）。
- GS-single 是登记版：同一对比度送入联合训练器的两个通道（登记的 singleT1 / singleFLAIR 配置），取第一个通道的结果。
- 偏移组（vol5、vol6、vol7）按采集到的样子跑，没有做偏移校正；对齐组与偏移组分开列出。
- 所有设置取自调参病例（jiangyizhou）的登记值，没有在测试扫描上调参；真值只用于评分。

## 6. 重算分数与运行脚本记录的分数

每个体积（含不完整格子里的）都用同一段评分代码（AF16 克隆的 `t1t2_common.Pair.score`，CPU）重算，再与运行脚本记录的值比较。任何一个 MC-GS 体积的头部 PSNR 相差超过 0.005 dB 都会使整次运行失败。最大绝对差：

| 量 | 最大绝对差 | 出现在 |
|---|---|---|
| 头部 PSNR（dB） | 0.000167 | vol7_tangbin_790 AF12 T1 MC-GS |
| 头部 SSIM | 1.07e-06 | vol7_tangbin_790 AF24 T1 GS-single |
| 整体 NMSE | 6.2e-07 | vol7_tangbin_790 AF24 T1 MC-GS |

各方法的最大 |头部 PSNR 差|（dB）：ZF 3.1e-06，CG-SENSE 4.1e-06，CS-SENSE 4.1e-06，Joint SENSE 3.2e-06，L1-SPIRiT 2.9e-06，ZS-SSDU 2.6e-06，3D-DIP 2.6e-06，3D-INR 2.7e-06，MC-INR 2.9e-06，GS-single 0.00016，MC-GS 0.00017

## 7. 图

汇总图：`autoresearch/t1flair_af_testset_20261121/figures/testset_summary.png`（PNG / PDF 只放在 git 忽略的 figures 文件夹，不提交）。每个加速倍数一栏，每次扫描一个点，纵轴是头部 PSNR 差（MC-GS 减对照）；零线为界；圆点 = 对齐组，三角 = 偏移组；只画完整格子。
