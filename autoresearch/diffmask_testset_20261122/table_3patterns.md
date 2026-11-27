# 三种采样方式（AF16，测试集 6 次扫描）合并表

本表由 `merge_patterns.py` 从两份已存的统计表合并，不重新计算任何分数。变密度泊松来自加速倍数研究（`t1flair_af_testset_20261121/table.json`）的 AF16 格子。

完整扫描数：变密度泊松 4 / 6，均匀密度 6 / 6，伪径向 6 / 6。

**中间结果：** 变密度泊松 的扫描还不齐，下面的计数只含已完整的扫描，不能作为最终结论。

## MC-GS 排第一的扫描数

| 采样方式 | 对比度 | 完整扫描 | PSNR | SSIM | NMSE |
|---|---|---|---|---|---|
| 变密度泊松 | T1 | 4 / 6 | 3 / 4 | 4 / 4 | 3 / 4 |
| 变密度泊松 | FLAIR | 4 / 6 | 3 / 4 | 4 / 4 | 3 / 4 |
| 均匀密度 | T1 | 6 / 6 | 6 / 6 | 6 / 6 | 6 / 6 |
| 均匀密度 | FLAIR | 6 / 6 | 5 / 6 | 1 / 6 | 5 / 6 |
| 伪径向 | T1 | 6 / 6 | 2 / 6 | 5 / 6 | 2 / 6 |
| 伪径向 | FLAIR | 6 / 6 | 6 / 6 | 4 / 6 | 6 / 6 |

## 头部 PSNR 差值（MC-GS 减对方，dB）

| 采样方式 | 对比度 | 对 GS-single：均值（最小；正数个数） | 对最强的非 GS 方法：均值（最小；正数个数） |
|---|---|---|---|
| 变密度泊松 | T1 | +0.113（-0.044；3 / 4） | +0.910（+0.657；4 / 4） |
| 变密度泊松 | FLAIR | +0.024（-0.019；3 / 4） | +0.519（+0.337；4 / 4） |
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

## 还缺方法的格子

- 变密度泊松 T1 vol2_jiangdian：缺 3D-INR、MC-INR
- 变密度泊松 T1 vol4_tangbing：缺 ZS-SSDU、3D-DIP、3D-INR
- 变密度泊松 FLAIR vol2_jiangdian：缺 3D-INR、MC-INR
- 变密度泊松 FLAIR vol4_tangbing：缺 ZS-SSDU、3D-DIP、3D-INR

## 数据来源

- 本研究 `table.json` sha256 `16dca012bad06ee0…`
- 加速倍数研究 `table.json` sha256 `7638cbfeade39dc0…`
