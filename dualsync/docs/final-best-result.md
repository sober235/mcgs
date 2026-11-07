# 当前最优结果图

## 当前：Yuantian sharp（2026-11-06）

用户指示："Record the "yuantian sharp" as the CURRENT optimal result"。

Yuantian 移植版 sharp 配置（切向 0.25、法向 0.10、每步更新上限 0.04、mu 10、t_jac 150、sigma_y 0.3、50 步），base seed 1234，单次重建。代表图为 MSP06、DYF01。

[当前最优图与数据登记](../outputs/current_best_20261106/README.md) · [完整来源清单](../outputs/current_best_20261106/manifest.json)

相对上一登记：背景 SNR ×1.69/×1.59，带内 SNR ×1.90/×1.65。在冻结门槛下两张图都未通过粗尺度 SSIM（与输入、与上一登记），因此这是用户选择，不是门槛内的数值最优。目前只有这两张切片，尚无 38 张全量结果。默认代码和配置未改。

## 上一登记：原始 Accepted Tuned（2026-11-06，保留为历史）

2026-11-06 用户确认：“OK，将最后一列记录为最终最优结果图”。

选择的是最近分步诊断图第4列 **Geometry output（最终Tuned）**，即原始 Accepted Tuned 的最终几何输出，base seed=1234。两张代表图为MSP06、DYF01，实际seed分别1259、1235。输出与历史Matched Accepted Tuned逐点一致。

[最终图与数据登记](../outputs/final_best_20261106/README.md) · [完整来源清单](../outputs/final_best_20261106/manifest.json)

本次仅登记用户选择并导出对应图像，未新增推理、滤波、调参或覆盖默认代码。完整38张同方法结果对应原Accepted Tuned归档，路径在manifest中。历史K2/alpha.5、法向扫描和诊断结果均保留，用户视觉选择不改写其数值结论。
