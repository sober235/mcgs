# MC-GS 应对轻微运动：三处小改动（第二版）

第一版（`../motion_adapter_sim_20261113/`，结果见其 `RESULTS.md`）让适配器的全部参数通过 k 空间数据一致性去学，
四条判据都没有通过。本版不换框架，只做三处小改动：M1 扫描间变换由配准给出；M2 T2 的数据一致性按回波链加权；
M3 检测到持续偏移时给之后的数据一个状态位姿。

- 设计稿：`docs/superpowers/specs/2026-11-21-motion-mild-adjust-design.md`（用户 2026-11-21 批准）
- 实施计划：`docs/superpowers/plans/2026-11-21-motion-mild-adjust.md`
- 第一版目录在这里当库用（`geom`、`warp`、`lesions`、`schedule`、`simulate`、`adapter`），只做带默认值的追加式修改。
- 服务器根目录：`codex_artifacts/mcgs_motion_mild_adjust_20261121`。
- 采集顺序是假设：`il`（交错，每条回波链跨越整个 k 空间，主）和 `kz`（kz 线性，对照）。结论只对所用的顺序成立。

测试：`/usr/bin/python3 ../motion_adapter_sim_20261113/tools/rrun.py --cpu -- @python autoresearch/motion_mild_adjust_20261121/tests/run_tests.py`

## 状态

进行中（任务 1：采集顺序、事件类型、常量）。
