# 病例内幅值先验的历史核对

2026-10-22，独立只读核对了保存的指标、训练日志、实验账本及三指标重评。没有重新训练、重建或把日志条目计为新实验。当前正式 MRI 训练总数为 30。

| 已执行方案 | 事实与结果 | 证据 |
| --- | --- | --- |
| BM4D RED，ar_g8_nl05 | 旧轨迹 4000 步，权重 0.05、sigma_rel 0.04、每 500 步刷新，日志有实际目标刷新和非零先验。平扫/增强 PSNR 34.635554/36.966871 dB；后续三指标切片同时达标 0/59。 | [历史核对](../../mcgs_ce_autoresearch_20261017/history_audit_20261018.md) |
| Patch 低秩 C–G | 五个各 500 步的活跃先验实验，从旧 i1 第 4000 步起，目标实际在 4001、4251 刷新。最好的 E 为权重 0.1、sigma_rel 0.015，平扫/增强 PSNR 34.712135/36.954495 dB，三指标整脑最小裕量 -0.161944207；五个均未胜出。 | [实际 E 指标](../../mcgs_ce_autoresearch_20261017/nonlocal_autoresearch/outputs/E/metrics.json)、[协议](../../mcgs_ce_autoresearch_20261017/nonlocal_autoresearch/autoresearch.md) |
| 病例内残差幅值/增益网络 | 三个各 500 步实验通过 MRI 物理损失优化 Gaussian 系数，最小裕量约 -0.11734、-0.11716、-0.11734，均低于 keeper。这是系数参数化实验，不是 patch 去噪器训练。 | [已完成族审计](../../mcgs_ce_karpathy_20261019/docs/family1_outcome_audit.md) |
| 完整幅值函数 | 两个表示诊断未达到预先规定的原始输出保真界限，未获准进入 MRI 训练，不计作正式 MRI 试验。 | [表示结果](../../mcgs_ce_karpathy_20261022_fullamp/docs/representation_outcome.md) |
| 固定点数重定位 | 本轮普通与重定位两组 500 步均完成，整脑最小裕量 -0.117185586/-0.117187726，均未超过 keeper，双扫描同时达标均为 0/59。 | [本轮结果](relocation_outcome.md) |
| 病例内训练的 patch 去噪器 | 上述记录中未发现执行、已训练去噪器产物或评分结果。当前只开始实现独立新候选。 | [前期建议](../../mcgs_ce_autoresearch_20261017/research_directions_20261019/BRIEF.md)、[本轮设计](superpowers/specs/2026-10-22-case-prior-design.md) |

旧固定去噪器采用 i1 起点，不能当作 G2 上的匹配对照。它们的失败也不证明新病例内学习先验一定失败。新方案仍须实际控制、原 Gaussian 输出和固定评分；不能凭方法名称或网络训练误差宣称 MRI 指标改善。
