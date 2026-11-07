# DualSync 实验记录（2026-11-06）

这些文件来自 `sober235/DualSync` 的 `exp/yuantian-repro` 分支，只是记录副本。路径保持原仓库的相对位置。脚本按服务器 `/data0/zhuoxu/code/DualSync` 的目录结构编写，在这里不能直接运行。

## 当前最优结果：Yuantian sharp

- 方法：Yuantian 移植版 sharp 配置。参数为切向 0.25、法向 0.10、每步更新上限 0.04、`mu` 10、`t_jac` 150、`sigma_y` 0.3、50 步。base seed 1234，单次重建。
- 切片：MSP-FSE-T/slice_06、DYF-FSE-T/slice_01。

与上一登记（Accepted Tuned）的对比：

| 切片 | 背景 SNR 增益 | 带内 SNR 增益 |
|---|---|---|
| MSP06 | ×1.69 | ×1.90 |
| DYF01 | ×1.59 | ×1.65 |

- 两张切片在冻结门槛下都没有通过粗尺度 SSIM，因此登记依据是用户选择，不是门槛内的最优。
- 目前还没有 38 张切片的全量结果。
- 登记详情见 [`outputs/current_best_20261106/README.md`](outputs/current_best_20261106/README.md)。

## 内容

- `outputs/autoresearch_sigma_y_20261106/`：`sigma_y` 与 `eta` 的单次重建 SNR 自动实验。按停止规则暂停，见 `SUMMARY.md`。
- `outputs/yuantian_repro_20261106/`：Yuantian 移植版 sharp、balanced、legacy 三个配置在 MSP06、DYF01 上的复现与评估，见 `metrics.json`。
- `outputs/current_best_20261106/`：当前最优结果的登记清单和对比脚本。
- `docs/final-best-result.md`：当前最优与上一登记的说明。

## 未包含的文件

MRI 数据（`.mat`）、图像（`.png`）和交互查看器（`*viewer*.html`）都包含图像像素，只保存在服务器或本地，不放进 git。
