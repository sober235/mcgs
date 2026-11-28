# 测试集 T1 + FLAIR 不同采样掩膜（AF16，均匀密度 / 伪径向）：结果

> 完整格子（扫描 x 掩膜；T1 与 FLAIR 两种对比度的 11 个方法都齐全）12 / 12。不完整的格子对两种对比度都不计入名次与统计。全部跑完后由 `autoresearch/diffmask_testset_20261122/collect_stats.py --pipeline` 重新生成本文件、table.json、stats.json、table.md 与图。

口径：头部掩膜内（|GT| > 0.05 P99.9）的 PSNR / SSIM / NMSE；评分代码是各课题共用的 `t1t2_common.Pair.score`，真值来自该扫描自己的全采样 k 空间，所有数字都由保存的体积重新计算，再与各运行脚本记录的分数对照（见第 6 节）。NMSE 的差值写成相对降低（%），其余为绝对差；大于 0 表示 MC-GS 更好。

## 1. MC-GS 在完整格子里是否第一

| 对比度 | 掩膜 | 完整格子（共 6） | PSNR 第一 | SSIM 第一 | NMSE 第一 |
|---|---|---|---|---|---|
| T1 | uniform | 6 | 6 / 6 | 6 / 6 | 6 / 6 |
| T1 | radial | 6 | 2 / 6 | 5 / 6 | 2 / 6 |
| FLAIR | uniform | 6 | 5 / 6 | 1 / 6 | 5 / 6 |
| FLAIR | radial | 6 | 6 / 6 | 4 / 6 | 6 / 6 |

合计：完整格子 12 / 12，即 24 列（T1 与 FLAIR 各一列）；MC-GS 第一的列数：PSNR 19 / 24，SSIM 16 / 24，NMSE 19 / 24。分对比度：T1 PSNR 8 / 12，SSIM 11 / 12，NMSE 8 / 12；FLAIR PSNR 11 / 12，SSIM 5 / 12，NMSE 11 / 12。

### MC-GS 不是第一的列（逐个列出，不省略）

| 扫描 | 掩膜 | 对比度 | 指标 | MC-GS 名次 | 第一名 | 第一名数值 | MC-GS 数值 |
|---|---|---|---|---|---|---|---|
| vol1_huangyuwei | radial | T1 | PSNR | 2 | ZS-SSDU | 29.665 | 29.160 |
| vol1_huangyuwei | radial | T1 | SSIM | 2 | ZS-SSDU | 0.8646 | 0.8587 |
| vol1_huangyuwei | radial | T1 | NMSE | 2 | ZS-SSDU | 0.03622 | 0.04069 |
| vol2_jiangdian | uniform | FLAIR | SSIM | 2 | ZS-SSDU | 0.9118 | 0.9109 |
| vol4_tangbing | uniform | FLAIR | SSIM | 2 | ZS-SSDU | 0.8817 | 0.8760 |
| vol5_sunxinyi | uniform | FLAIR | SSIM | 2 | ZS-SSDU | 0.9020 | 0.8947 |
| vol5_sunxinyi | radial | T1 | PSNR | 2 | ZS-SSDU | 28.805 | 28.765 |
| vol5_sunxinyi | radial | T1 | NMSE | 2 | ZS-SSDU | 0.04122 | 0.04160 |
| vol6_jiangdian_790 | uniform | FLAIR | SSIM | 2 | ZS-SSDU | 0.9058 | 0.9019 |
| vol6_jiangdian_790 | radial | T1 | PSNR | 2 | ZS-SSDU | 29.788 | 29.577 |
| vol6_jiangdian_790 | radial | T1 | NMSE | 2 | ZS-SSDU | 0.03806 | 0.03996 |
| vol6_jiangdian_790 | radial | FLAIR | SSIM | 2 | ZS-SSDU | 0.8461 | 0.8453 |
| vol7_tangbin_790 | uniform | FLAIR | PSNR | 2 | ZS-SSDU | 31.993 | 31.887 |
| vol7_tangbin_790 | uniform | FLAIR | SSIM | 2 | ZS-SSDU | 0.8531 | 0.8331 |
| vol7_tangbin_790 | uniform | FLAIR | NMSE | 2 | ZS-SSDU | 0.02770 | 0.02838 |
| vol7_tangbin_790 | radial | T1 | PSNR | 2 | ZS-SSDU | 30.414 | 30.364 |
| vol7_tangbin_790 | radial | T1 | NMSE | 2 | ZS-SSDU | 0.04001 | 0.04047 |
| vol7_tangbin_790 | radial | FLAIR | SSIM | 2 | ZS-SSDU | 0.8216 | 0.8204 |

## 2. 病例间：MC-GS 相对两个对照的差值

对照：GS-single；该格最强的非 GS 方法（按该格头部 PSNR 选，三个指标都与同一个方法比）。“符号一致”= 差值大于 0 的单位数 / 单位总数。p 为精确双侧 Wilcoxon 符号秩检验（零差值舍去，并列用平均秩；单位数 4 时 p 最小 0.125，6 时 0.031，见第 5 节）。

### PSNR差值 dB

| 对比度 | 掩膜 | 对照 | 单位 | 均值 | 标准差 | 最小 | 符号一致 | p |
|---|---|---|---|---|---|---|---|---|
| T1 | uniform | GS-single | 6 次扫描 | 0.5697 | 0.19 | 0.2319 | 6 / 6 | 0.031 |
| T1 | uniform | GS-single | 对齐组（vol1、vol2、vol4）3 次 | 0.6269 | 0.141 | 0.4661 | 3 / 3 | 0.250 |
| T1 | uniform | GS-single | 偏移组（vol5、vol6、vol7）3 次 | 0.5126 | 0.245 | 0.2319 | 3 / 3 | 0.250 |
| T1 | uniform | GS-single | 4 个人（重复扫描先平均） | 0.5903 | 0.165 | 0.349 | 4 / 4 | 0.125 |
| T1 | uniform | 最强非 GS | 6 次扫描 | 2.154 | 0.216 | 1.779 | 6 / 6 | 0.031 |
| T1 | uniform | 最强非 GS | 对齐组（vol1、vol2、vol4）3 次 | 2.311 | 0.0782 | 2.224 | 3 / 3 | 0.250 |
| T1 | uniform | 最强非 GS | 偏移组（vol5、vol6、vol7）3 次 | 1.997 | 0.192 | 1.779 | 3 / 3 | 0.250 |
| T1 | uniform | 最强非 GS | 4 个人（重复扫描先平均） | 2.175 | 0.14 | 2.002 | 4 / 4 | 0.125 |
| T1 | radial | GS-single | 6 次扫描 | 0.2295 | 0.0634 | 0.1502 | 6 / 6 | 0.031 |
| T1 | radial | GS-single | 对齐组（vol1、vol2、vol4）3 次 | 0.2492 | 0.0786 | 0.16 | 3 / 3 | 0.250 |
| T1 | radial | GS-single | 偏移组（vol5、vol6、vol7）3 次 | 0.2097 | 0.0518 | 0.1502 | 3 / 3 | 0.250 |
| T1 | radial | GS-single | 4 个人（重复扫描先平均） | 0.2413 | 0.0637 | 0.1551 | 4 / 4 | 0.125 |
| T1 | radial | 最强非 GS | 6 次扫描 | -0.08467 | 0.255 | -0.5053 | 2 / 6 | 0.562 |
| T1 | radial | 最强非 GS | 对齐组（vol1、vol2、vol4）3 次 | -0.06931 | 0.391 | -0.5053 | 2 / 3 | 1.000 |
| T1 | radial | 最强非 GS | 偏移组（vol5、vol6、vol7）3 次 | -0.1 | 0.0962 | -0.211 | 0 / 3 | 0.250 |
| T1 | radial | 最强非 GS | 4 个人（重复扫描先平均） | -0.1317 | 0.261 | -0.5053 | 1 / 4 | 0.625 |
| FLAIR | uniform | GS-single | 6 次扫描 | 0.08362 | 0.0363 | 0.01995 | 6 / 6 | 0.031 |
| FLAIR | uniform | GS-single | 对齐组（vol1、vol2、vol4）3 次 | 0.1097 | 0.00833 | 0.1033 | 3 / 3 | 0.250 |
| FLAIR | uniform | GS-single | 偏移组（vol5、vol6、vol7）3 次 | 0.05758 | 0.0344 | 0.01995 | 3 / 3 | 0.250 |
| FLAIR | uniform | GS-single | 4 个人（重复扫描先平均） | 0.0838 | 0.0191 | 0.06538 | 4 / 4 | 0.125 |
| FLAIR | uniform | 最强非 GS | 6 次扫描 | 0.5208 | 0.372 | -0.1056 | 5 / 6 | 0.062 |
| FLAIR | uniform | 最强非 GS | 对齐组（vol1、vol2、vol4）3 次 | 0.7454 | 0.266 | 0.5259 | 3 / 3 | 0.250 |
| FLAIR | uniform | 最强非 GS | 偏移组（vol5、vol6、vol7）3 次 | 0.2961 | 0.353 | -0.1056 | 2 / 3 | 0.500 |
| FLAIR | uniform | 最强非 GS | 4 个人（重复扫描先平均） | 0.5758 | 0.351 | 0.2102 | 4 / 4 | 0.125 |
| FLAIR | radial | GS-single | 6 次扫描 | 0.09547 | 0.0305 | 0.03778 | 6 / 6 | 0.031 |
| FLAIR | radial | GS-single | 对齐组（vol1、vol2、vol4）3 次 | 0.1119 | 0.0153 | 0.09423 | 3 / 3 | 0.250 |
| FLAIR | radial | GS-single | 偏移组（vol5、vol6、vol7）3 次 | 0.07905 | 0.0357 | 0.03778 | 3 / 3 | 0.250 |
| FLAIR | radial | GS-single | 4 个人（重复扫描先平均） | 0.09925 | 0.024 | 0.06601 | 4 / 4 | 0.125 |
| FLAIR | radial | 最强非 GS | 6 次扫描 | 0.2096 | 0.124 | 0.08772 | 6 / 6 | 0.031 |
| FLAIR | radial | 最强非 GS | 对齐组（vol1、vol2、vol4）3 次 | 0.2225 | 0.159 | 0.111 | 3 / 3 | 0.250 |
| FLAIR | radial | 最强非 GS | 偏移组（vol5、vol6、vol7）3 次 | 0.1966 | 0.114 | 0.08772 | 3 / 3 | 0.250 |
| FLAIR | radial | 最强非 GS | 4 个人（重复扫描先平均） | 0.2471 | 0.138 | 0.09937 | 4 / 4 | 0.125 |

### SSIM差值 

| 对比度 | 掩膜 | 对照 | 单位 | 均值 | 标准差 | 最小 | 符号一致 | p |
|---|---|---|---|---|---|---|---|---|
| T1 | uniform | GS-single | 6 次扫描 | 0.01302 | 0.00355 | 0.005938 | 6 / 6 | 0.031 |
| T1 | uniform | GS-single | 对齐组（vol1、vol2、vol4）3 次 | 0.01425 | 0.00112 | 0.01296 | 3 / 3 | 0.250 |
| T1 | uniform | GS-single | 偏移组（vol5、vol6、vol7）3 次 | 0.01179 | 0.00507 | 0.005938 | 3 / 3 | 0.250 |
| T1 | uniform | GS-single | 4 个人（重复扫描先平均） | 0.01343 | 0.00266 | 0.009448 | 4 / 4 | 0.125 |
| T1 | uniform | 最强非 GS | 6 次扫描 | 0.04047 | 0.0111 | 0.03199 | 6 / 6 | 0.031 |
| T1 | uniform | 最强非 GS | 对齐组（vol1、vol2、vol4）3 次 | 0.04604 | 0.0144 | 0.03199 | 3 / 3 | 0.250 |
| T1 | uniform | 最强非 GS | 偏移组（vol5、vol6、vol7）3 次 | 0.03491 | 0.00316 | 0.03211 | 3 / 3 | 0.250 |
| T1 | uniform | 最强非 GS | 4 个人（重复扫描先平均） | 0.03864 | 0.00779 | 0.03199 | 4 / 4 | 0.125 |
| T1 | radial | GS-single | 6 次扫描 | 0.009604 | 0.0022 | 0.006351 | 6 / 6 | 0.031 |
| T1 | radial | GS-single | 对齐组（vol1、vol2、vol4）3 次 | 0.01062 | 0.00228 | 0.008254 | 3 / 3 | 0.250 |
| T1 | radial | GS-single | 偏移组（vol5、vol6、vol7）3 次 | 0.008586 | 0.00195 | 0.006351 | 3 / 3 | 0.250 |
| T1 | radial | GS-single | 4 个人（重复扫描先平均） | 0.01004 | 0.00225 | 0.007303 | 4 / 4 | 0.125 |
| T1 | radial | 最强非 GS | 6 次扫描 | 0.003723 | 0.00649 | -0.005887 | 5 / 6 | 0.219 |
| T1 | radial | 最强非 GS | 对齐组（vol1、vol2、vol4）3 次 | 0.00476 | 0.00991 | -0.005887 | 2 / 3 | 0.500 |
| T1 | radial | 最强非 GS | 偏移组（vol5、vol6、vol7）3 次 | 0.002687 | 0.002 | 0.001252 | 3 / 3 | 0.250 |
| T1 | radial | 最强非 GS | 4 个人（重复扫描先平均） | 0.002678 | 0.00594 | -0.005887 | 3 / 4 | 0.625 |
| FLAIR | uniform | GS-single | 6 次扫描 | 0.002722 | 0.00117 | 0.001608 | 6 / 6 | 0.031 |
| FLAIR | uniform | GS-single | 对齐组（vol1、vol2、vol4）3 次 | 0.003526 | 0.00117 | 0.002583 | 3 / 3 | 0.250 |
| FLAIR | uniform | GS-single | 偏移组（vol5、vol6、vol7）3 次 | 0.001919 | 0.000318 | 0.001608 | 3 / 3 | 0.250 |
| FLAIR | uniform | GS-single | 4 个人（重复扫描先平均） | 0.002637 | 0.000799 | 0.001608 | 4 / 4 | 0.125 |
| FLAIR | uniform | 最强非 GS | 6 次扫描 | -0.006119 | 0.00746 | -0.01999 | 1 / 6 | 0.094 |
| FLAIR | uniform | 最强非 GS | 对齐组（vol1、vol2、vol4）3 次 | -0.001852 | 0.00352 | -0.005752 | 1 / 3 | 0.750 |
| FLAIR | uniform | 最强非 GS | 偏移组（vol5、vol6、vol7）3 次 | -0.01039 | 0.00849 | -0.01999 | 0 / 3 | 0.250 |
| FLAIR | uniform | 最强非 GS | 4 个人（重复扫描先平均） | -0.005366 | 0.00607 | -0.01287 | 1 / 4 | 0.250 |
| FLAIR | radial | GS-single | 6 次扫描 | 0.003578 | 0.000942 | 0.002295 | 6 / 6 | 0.031 |
| FLAIR | radial | GS-single | 对齐组（vol1、vol2、vol4）3 次 | 0.004335 | 0.000467 | 0.003797 | 3 / 3 | 0.250 |
| FLAIR | radial | GS-single | 偏移组（vol5、vol6、vol7）3 次 | 0.002821 | 0.000529 | 0.002295 | 3 / 3 | 0.250 |
| FLAIR | radial | GS-single | 4 个人（重复扫描先平均） | 0.003615 | 0.000756 | 0.002816 | 4 / 4 | 0.125 |
| FLAIR | radial | 最强非 GS | 6 次扫描 | 0.002668 | 0.00388 | -0.001175 | 4 / 6 | 0.312 |
| FLAIR | radial | 最强非 GS | 对齐组（vol1、vol2、vol4）3 次 | 0.004449 | 0.00415 | 0.0003359 | 3 / 3 | 0.250 |
| FLAIR | radial | 最强非 GS | 偏移组（vol5、vol6、vol7）3 次 | 0.0008867 | 0.00329 | -0.001175 | 1 / 3 | 1.000 |
| FLAIR | radial | 最强非 GS | 4 个人（重复扫描先平均） | 0.003665 | 0.00389 | -0.0002523 | 3 / 4 | 0.250 |

### NMSE差值 %（相对降低）

| 对比度 | 掩膜 | 对照 | 单位 | 均值 | 标准差 | 最小 | 符号一致 | p |
|---|---|---|---|---|---|---|---|---|
| T1 | uniform | GS-single | 6 次扫描 | 12.22 | 3.91 | 5.199 | 6 / 6 | 0.031 |
| T1 | uniform | GS-single | 对齐组（vol1、vol2、vol4）3 次 | 13.41 | 2.84 | 10.18 | 3 / 3 | 0.250 |
| T1 | uniform | GS-single | 偏移组（vol5、vol6、vol7）3 次 | 11.04 | 5.1 | 5.199 | 3 / 3 | 0.250 |
| T1 | uniform | GS-single | 4 个人（重复扫描先平均） | 12.65 | 3.39 | 7.688 | 4 / 4 | 0.125 |
| T1 | uniform | 最强非 GS | 6 次扫描 | 39.04 | 3.09 | 33.61 | 6 / 6 | 0.031 |
| T1 | uniform | 最强非 GS | 对齐组（vol1、vol2、vol4）3 次 | 41.26 | 1.06 | 40.08 | 3 / 3 | 0.250 |
| T1 | uniform | 最强非 GS | 偏移组（vol5、vol6、vol7）3 次 | 36.82 | 2.82 | 33.61 | 3 / 3 | 0.250 |
| T1 | uniform | 最强非 GS | 4 个人（重复扫描先平均） | 39.34 | 1.99 | 36.85 | 4 / 4 | 0.125 |
| T1 | radial | GS-single | 6 次扫描 | 5.138 | 1.39 | 3.399 | 6 / 6 | 0.031 |
| T1 | radial | GS-single | 对齐组（vol1、vol2、vol4）3 次 | 5.566 | 1.72 | 3.618 | 3 / 3 | 0.250 |
| T1 | radial | GS-single | 偏移组（vol5、vol6、vol7）3 次 | 4.71 | 1.14 | 3.399 | 3 / 3 | 0.250 |
| T1 | radial | GS-single | 4 个人（重复扫描先平均） | 5.396 | 1.39 | 3.509 | 4 / 4 | 0.125 |
| T1 | radial | 最强非 GS | 6 次扫描 | -2.117 | 6.08 | -12.34 | 2 / 6 | 0.562 |
| T1 | radial | 最强非 GS | 对齐组（vol1、vol2、vol4）3 次 | -1.886 | 9.32 | -12.34 | 2 / 3 | 1.000 |
| T1 | radial | 最强非 GS | 偏移组（vol5、vol6、vol7）3 次 | -2.347 | 2.28 | -4.978 | 0 / 3 | 0.250 |
| T1 | radial | 最强非 GS | 4 个人（重复扫描先平均） | -3.245 | 6.32 | -12.34 | 1 / 4 | 0.625 |
| FLAIR | uniform | GS-single | 6 次扫描 | 1.904 | 0.822 | 0.4583 | 6 / 6 | 0.031 |
| FLAIR | uniform | GS-single | 对齐组（vol1、vol2、vol4）3 次 | 2.493 | 0.187 | 2.35 | 3 / 3 | 0.250 |
| FLAIR | uniform | GS-single | 偏移组（vol5、vol6、vol7）3 次 | 1.315 | 0.783 | 0.4583 | 3 / 3 | 0.250 |
| FLAIR | uniform | GS-single | 4 个人（重复扫描先平均） | 1.909 | 0.434 | 1.494 | 4 / 4 | 0.125 |
| FLAIR | uniform | 最强非 GS | 6 次扫描 | 11.03 | 7.77 | -2.462 | 5 / 6 | 0.062 |
| FLAIR | uniform | 最强非 GS | 对齐组（vol1、vol2、vol4）3 次 | 15.67 | 5.11 | 11.41 | 3 / 3 | 0.250 |
| FLAIR | uniform | 最强非 GS | 偏移组（vol5、vol6、vol7）3 次 | 6.383 | 7.75 | -2.462 | 2 / 3 | 0.500 |
| FLAIR | uniform | 最强非 GS | 4 个人（重复扫描先平均） | 12.14 | 7.08 | 4.472 | 4 / 4 | 0.125 |
| FLAIR | radial | GS-single | 6 次扫描 | 2.172 | 0.689 | 0.8662 | 6 / 6 | 0.031 |
| FLAIR | radial | GS-single | 对齐组（vol1、vol2、vol4）3 次 | 2.543 | 0.345 | 2.146 | 3 / 3 | 0.250 |
| FLAIR | radial | GS-single | 偏移组（vol5、vol6、vol7）3 次 | 1.802 | 0.81 | 0.8662 | 3 / 3 | 0.250 |
| FLAIR | radial | GS-single | 4 个人（重复扫描先平均） | 2.258 | 0.543 | 1.506 | 4 / 4 | 0.125 |
| FLAIR | radial | 最强非 GS | 6 次扫描 | 4.679 | 2.71 | 2 | 6 / 6 | 0.031 |
| FLAIR | radial | 最强非 GS | 对齐组（vol1、vol2、vol4）3 次 | 4.953 | 3.44 | 2.524 | 3 / 3 | 0.250 |
| FLAIR | radial | 最强非 GS | 偏移组（vol5、vol6、vol7）3 次 | 4.405 | 2.5 | 2 | 3 / 3 | 0.250 |
| FLAIR | radial | 最强非 GS | 4 个人（重复扫描先平均） | 5.494 | 3 | 2.262 | 4 / 4 | 0.125 |

## 3. 病例内：逐层面配对检验

每个完整格子内，三个平面里头部占层面 >= 5 % 的层面合并（规则同 `t1flair_af_record_fig_20261120/fig_af_overview.py` 所读的 slice_trend 记录），逐层面配对 Wilcoxon（双侧）。这里的 p 值用正态近似（每个病例约 600 层），零差值被舍去；第 2 节的扫描 / 人层面检验才是精确检验。下表对每个（对比度，掩膜，对照，指标）把扫描分成三类：p < 0.05 且中位差为正（显著更好）、p < 0.05 且中位差为负（显著更差）、其余（不显著），并列出后两类的扫描；最后一列是各扫描里 MC-GS 更好的层面占比的范围。层面属于同一个体积，彼此不独立，p 值只说明该病例内的一致性，不是人群推断。逐病例的全部数值在 stats.json 的 `slices`。

| 对比度 | 掩膜 | 对照 | 指标 | 显著更好 | 显著更差（扫描） | 不显著（扫描） | 更好层面占比范围 |
|---|---|---|---|---|---|---|---|
| T1 | uniform | GS-single | PSNR | 6 / 6 | 0（无） | 0（无） | 90% - 100% |
| T1 | uniform | GS-single | SSIM | 6 / 6 | 0（无） | 0（无） | 95% - 100% |
| T1 | uniform | GS-single | NMSE | 6 / 6 | 0（无） | 0（无） | 90% - 100% |
| T1 | uniform | 最强非 GS | PSNR | 6 / 6 | 0（无） | 0（无） | 99% - 100% |
| T1 | uniform | 最强非 GS | SSIM | 6 / 6 | 0（无） | 0（无） | 99% - 100% |
| T1 | uniform | 最强非 GS | NMSE | 6 / 6 | 0（无） | 0（无） | 99% - 100% |
| T1 | radial | GS-single | PSNR | 6 / 6 | 0（无） | 0（无） | 98% - 100% |
| T1 | radial | GS-single | SSIM | 6 / 6 | 0（无） | 0（无） | 98% - 100% |
| T1 | radial | GS-single | NMSE | 6 / 6 | 0（无） | 0（无） | 98% - 100% |
| T1 | radial | 最强非 GS | PSNR | 3 / 6 | 2（vol1_huangyuwei、vol6_jiangdian_790） | 1（vol7_tangbin_790） | 12% - 83% |
| T1 | radial | 最强非 GS | SSIM | 5 / 6 | 1（vol1_huangyuwei） | 0（无） | 31% - 91% |
| T1 | radial | 最强非 GS | NMSE | 3 / 6 | 2（vol1_huangyuwei、vol6_jiangdian_790） | 1（vol7_tangbin_790） | 12% - 83% |
| FLAIR | uniform | GS-single | PSNR | 6 / 6 | 0（无） | 0（无） | 67% - 94% |
| FLAIR | uniform | GS-single | SSIM | 6 / 6 | 0（无） | 0（无） | 72% - 91% |
| FLAIR | uniform | GS-single | NMSE | 6 / 6 | 0（无） | 0（无） | 67% - 94% |
| FLAIR | uniform | 最强非 GS | PSNR | 5 / 6 | 0（无） | 1（vol7_tangbin_790） | 54% - 100% |
| FLAIR | uniform | 最强非 GS | SSIM | 1 / 6 | 5（vol2_jiangdian、vol4_tangbing、vol5_sunxinyi、vol6_jiangdian_790、vol7_tangbin_790） | 0（无） | 12% - 68% |
| FLAIR | uniform | 最强非 GS | NMSE | 5 / 6 | 0（无） | 1（vol7_tangbin_790） | 54% - 100% |
| FLAIR | radial | GS-single | PSNR | 6 / 6 | 0（无） | 0（无） | 77% - 88% |
| FLAIR | radial | GS-single | SSIM | 6 / 6 | 0（无） | 0（无） | 73% - 80% |
| FLAIR | radial | GS-single | NMSE | 6 / 6 | 0（无） | 0（无） | 77% - 88% |
| FLAIR | radial | 最强非 GS | PSNR | 6 / 6 | 0（无） | 0（无） | 73% - 98% |
| FLAIR | radial | 最强非 GS | SSIM | 4 / 6 | 0（无） | 2（vol6_jiangdian_790、vol7_tangbin_790） | 51% - 81% |
| FLAIR | radial | 最强非 GS | NMSE | 6 / 6 | 0（无） | 0（无） | 73% - 98% |

## 4. 缺失清单

不完整格子 0 / 12。每个（扫描，掩膜，对比度）缺的方法（原因见 table.json 的 `missing`）：


## 5. 必须写明的限制

- 只有 4 个人。以人为单位的 Wilcoxon 检验即使全胜，精确双侧 p 最小为 0.125；以 6 次扫描为单位最小为 0.031，但其中两次是重复扫描（vol2 与 vol6 是同一个人，vol4 与 vol7 是同一个人），不独立。结论写成“几个里几个一致”。
- 层面级检验的层面彼此不独立（同一体积、三个平面共用数据），只说明病例内的一致性。
- 一个种子。
- ZS-SSDU 是 5 个残差块的版本（各课题登记的实现）。
- GS-single 是登记版：同一对比度送入联合训练器的两个通道（登记的 singleT1 / singleFLAIR 配置），取第一个通道的结果。
- 偏移组（vol5、vol6、vol7）按采集到的样子跑，没有做偏移校正；对齐组与偏移组分开列出。
- 所有设置取自调参病例（jiangyizhou）各掩膜课题的登记值，没有在测试扫描上调参；真值只用于评分。
- “伪径向”按 9 月的伪径向掩膜理解，尚未得到用户确认。
- 第三种掩膜（变密度泊松 AF16）来自加速倍数测试集研究，本文件第 1–7 节不含；三种掩膜的合并结果见第 8 节和 `table_3patterns.md`。

## 6. 重算分数与运行脚本记录的分数

每个体积（含不完整格子里的）都用同一段评分代码（均匀密度克隆的 `t1t2_common.Pair.score`，CPU）重算，再与运行脚本记录的值比较。任何一个 MC-GS 体积的头部 PSNR 相差超过 0.005 dB 都会使整次运行失败。最大绝对差：

| 量 | 最大绝对差 | 出现在 |
|---|---|---|
| 头部 PSNR（dB） | 0.000208 | vol7_tangbin_790 AFuniform T1 MC-GS |
| 头部 SSIM | 2.32e-06 | vol7_tangbin_790 AFradial T1 MC-GS |
| 整体 NMSE | 1.15e-06 | vol7_tangbin_790 AFradial T1 MC-GS |

各方法的最大 |头部 PSNR 差|（dB）：ZF 3.6e-06，CG-SENSE 3.3e-06，CS-SENSE 3.3e-06，Joint SENSE 3.2e-06，L1-SPIRiT 4.3e-06，ZS-SSDU 4.4e-06，3D-DIP 4.5e-06，3D-INR 4.5e-06，MC-INR 3.6e-06，GS-single 0.0002，MC-GS 0.00021

## 7. 图

汇总图：`autoresearch/diffmask_testset_20261122/figures/testset_summary.png`（PNG / PDF / SVG 与数据源 CSV 都在 figures 文件夹，随结果一起提交）。每种掩膜一栏，每次扫描一个点，纵轴是头部 PSNR 差（MC-GS 减对照）；零线为界；圆点 = 对齐组，三角 = 偏移组；只画完整格子。

## 8. 三种采样掩膜合并（变密度泊松 / 均匀密度 / 伪径向，AF16）

本节由 `merge_patterns.py` 生成的 `table_3patterns.md` 整理而来（2026-11-28 06:22 PST）。变密度泊松的格子取自加速倍数测试集研究的最终统计表（`../t1flair_af_testset_20261121/table.json`，24 / 24 个格子，GitHub 分支 diff-af 提交 118cdc0），没有重新训练。

**量尺核对：** 两项研究的统计脚本评分代码逐行相同。另用 `gauge_check.py` 把变密度泊松 AF16 的 132 个重建体积（6 次扫描 × 2 对比度 × 11 个方法）用本研究的评分（各扫描的均匀密度克隆）重算，与该研究表中的头部 PSNR / SSIM / NMSE 逐项比较，最大差值均为 0.000000（记录在 `gauge_check.json`）。因此三种掩膜的数字在同一把量尺上。

**合计：** 36 个 PSNR 列（3 种掩膜 × 2 对比度 × 6 次扫描）中 MC-GS 第一 29 个，36 个 SSIM 列中第一 28 个，36 个 NMSE 列中第一 29 个；不是第一的列全部排第 2。

### MC-GS 排第一的扫描数

| 采样方式 | 对比度 | 完整扫描 | PSNR | SSIM | NMSE |
|---|---|---|---|---|---|
| 变密度泊松 | T1 | 6 / 6 | 5 / 6 | 6 / 6 | 5 / 6 |
| 变密度泊松 | FLAIR | 6 / 6 | 5 / 6 | 6 / 6 | 5 / 6 |
| 均匀密度 | T1 | 6 / 6 | 6 / 6 | 6 / 6 | 6 / 6 |
| 均匀密度 | FLAIR | 6 / 6 | 5 / 6 | 1 / 6 | 5 / 6 |
| 伪径向 | T1 | 6 / 6 | 2 / 6 | 5 / 6 | 2 / 6 |
| 伪径向 | FLAIR | 6 / 6 | 6 / 6 | 4 / 6 | 6 / 6 |

### 头部 PSNR 差值（MC-GS 减对方，dB）

| 采样方式 | 对比度 | 对 GS-single：均值（最小；正数个数） | 对最强的非 GS 方法：均值（最小；正数个数） |
|---|---|---|---|
| 变密度泊松 | T1 | +0.123（-0.044；5 / 6） | +0.910（+0.657；6 / 6） |
| 变密度泊松 | FLAIR | +0.030（-0.019；5 / 6） | +0.525（+0.337；6 / 6） |
| 均匀密度 | T1 | +0.570（+0.232；6 / 6） | +2.154（+1.779；6 / 6） |
| 均匀密度 | FLAIR | +0.084（+0.020；6 / 6） | +0.521（-0.106；5 / 6） |
| 伪径向 | T1 | +0.229（+0.150；6 / 6） | -0.085（-0.505；2 / 6） |
| 伪径向 | FLAIR | +0.095（+0.038；6 / 6） | +0.210（+0.088；6 / 6） |

### MC-GS 不是第一的列

- 变密度泊松 T1 PSNR：vol7_tangbin_790 第 2
- 变密度泊松 T1 NMSE：vol7_tangbin_790 第 2
- 变密度泊松 FLAIR PSNR：vol7_tangbin_790 第 2
- 变密度泊松 FLAIR NMSE：vol7_tangbin_790 第 2
- 均匀密度 FLAIR PSNR：vol7_tangbin_790 第 2
- 均匀密度 FLAIR SSIM：vol2_jiangdian 第 2
- 均匀密度 FLAIR SSIM：vol4_tangbing 第 2
- 均匀密度 FLAIR SSIM：vol5_sunxinyi 第 2
- 均匀密度 FLAIR SSIM：vol6_jiangdian_790 第 2
- 均匀密度 FLAIR SSIM：vol7_tangbin_790 第 2
- 均匀密度 FLAIR NMSE：vol7_tangbin_790 第 2
- 伪径向 T1 PSNR：vol1_huangyuwei 第 2
- 伪径向 T1 PSNR：vol5_sunxinyi 第 2
- 伪径向 T1 PSNR：vol6_jiangdian_790 第 2
- 伪径向 T1 PSNR：vol7_tangbin_790 第 2
- 伪径向 T1 SSIM：vol1_huangyuwei 第 2
- 伪径向 T1 NMSE：vol1_huangyuwei 第 2
- 伪径向 T1 NMSE：vol5_sunxinyi 第 2
- 伪径向 T1 NMSE：vol6_jiangdian_790 第 2
- 伪径向 T1 NMSE：vol7_tangbin_790 第 2
- 伪径向 FLAIR SSIM：vol6_jiangdian_790 第 2
- 伪径向 FLAIR SSIM：vol7_tangbin_790 第 2

### 本节的限制

- 变密度泊松下 vol7 的 PSNR 和 NMSE 输给 GS-single（T1 差 0.044 dB，FLAIR 差 0.019 dB），其余 5 次扫描 MC-GS 第一。
- 三种掩膜里最弱的是伪径向 T1：对最强的非 GS 方法平均落后 0.085 dB，6 次扫描中只赢 2 次。
- 每个格子只有一个随机种子；设置取自调参病例，没有在测试扫描上调参。
