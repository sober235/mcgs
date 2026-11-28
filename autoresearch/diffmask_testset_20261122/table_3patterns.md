# 三种采样方式（AF16，测试集 6 次扫描）合并表

本表由 `merge_patterns.py` 从两份已存的统计表合并，不重新计算任何分数。变密度泊松来自加速倍数研究（`t1flair_af_testset_20261121/table.json`）的 AF16 格子。

完整扫描数：变密度泊松 6 / 6，均匀密度 6 / 6，伪径向 6 / 6。

## MC-GS 排第一的扫描数

| 采样方式 | 对比度 | 完整扫描 | PSNR | SSIM | NMSE |
|---|---|---|---|---|---|
| 变密度泊松 | T1 | 6 / 6 | 5 / 6 | 6 / 6 | 5 / 6 |
| 变密度泊松 | FLAIR | 6 / 6 | 5 / 6 | 6 / 6 | 5 / 6 |
| 均匀密度 | T1 | 6 / 6 | 6 / 6 | 6 / 6 | 6 / 6 |
| 均匀密度 | FLAIR | 6 / 6 | 5 / 6 | 1 / 6 | 5 / 6 |
| 伪径向 | T1 | 6 / 6 | 2 / 6 | 5 / 6 | 2 / 6 |
| 伪径向 | FLAIR | 6 / 6 | 6 / 6 | 4 / 6 | 6 / 6 |

## 头部 PSNR 差值（MC-GS 减对方，dB）

| 采样方式 | 对比度 | 对 GS-single：均值（最小；正数个数） | 对最强的非 GS 方法：均值（最小；正数个数） |
|---|---|---|---|
| 变密度泊松 | T1 | +0.123（-0.044；5 / 6） | +0.910（+0.657；6 / 6） |
| 变密度泊松 | FLAIR | +0.030（-0.019；5 / 6） | +0.525（+0.337；6 / 6） |
| 均匀密度 | T1 | +0.570（+0.232；6 / 6） | +2.154（+1.779；6 / 6） |
| 均匀密度 | FLAIR | +0.084（+0.020；6 / 6） | +0.521（-0.106；5 / 6） |
| 伪径向 | T1 | +0.229（+0.150；6 / 6） | -0.085（-0.505；2 / 6） |
| 伪径向 | FLAIR | +0.095（+0.038；6 / 6） | +0.210（+0.088；6 / 6） |

## MC-GS 不是第一的列

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

## 数据来源

- 本研究 `table.json` sha256 `16dca012bad06ee0…`
- 加速倍数研究 `table.json` sha256 `b8ee71189eb637a0…`
