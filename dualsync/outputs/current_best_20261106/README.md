# 当前最优结果：Yuantian sharp

2026-11-06 用户指示："Record the "yuantian sharp" as the CURRENT optimal result"。

方法：Yuantian 移植版 sharp 配置（切向 0.25、法向 0.10、每步更新上限 0.04、mu 10、t_jac 150、sigma_y 0.3、50 步），base seed 1234，单次重建。

- MSP06：[完整图](MSP-FSE-T/slice_06_full.png) · [局部图](MSP-FSE-T/slice_06_roi.png) · [复数数据](MSP-FSE-T/slice_06.mat)
- DYF01：[完整图](DYF-FSE-T/slice_01_full.png) · [局部图](DYF-FSE-T/slice_01_roi.png) · [复数数据](DYF-FSE-T/slice_01.mat)

| 切片 | 背景 SNR | 带内 SNR | 与旧最终图粗尺度 SSIM | 与输入粗尺度 SSIM | 强边缘幅度 | 冻结门槛 |
|---|---:|---:|---:|---:|---:|---|
| MSP-FSE-T/slice_06 | ×1.69 | ×1.90 | 0.977 | 0.934（旧最终图 0.986） | 0.988 | 未通过：input coarse ssim、tuned coarse ssim |
| DYF-FSE-T/slice_01 | ×1.59 | ×1.65 | 0.987 | 0.952（旧最终图 0.988） | 0.989 | 未通过：input coarse ssim、tuned coarse ssim |

说明：这是用户选择，不是冻结门槛下的数值最优；sharp 在两张图上都未通过粗尺度 SSIM 门槛。只有这两张切片、一个 seed，尚无 38 张全量结果。MAT 为源文件原字节副本；PNG 使用与旧登记相同的输入 p99.5 窗口和 ROI，未做额外滤波或推理。

上一登记 [final_best_20261106](../final_best_20261106/README.md)（原始 Accepted Tuned）原样保留为历史记录。
