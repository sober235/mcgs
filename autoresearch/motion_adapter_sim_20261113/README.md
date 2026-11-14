# Motion-Aware MC-GS：hyw 数据上的运动仿真验证

在 hyw 的真实多线圈 k 空间上加入已知的 T2W 运动（扫描间刚体/非刚体、扫描内分段刚体）和对比度特有病灶，
检验给共享高斯加一个轻量运动适配器后，MC-GS 能否找回运动并保住 T2W 质量，零运动时不添乱。

- 设计：`docs/superpowers/specs/2026-11-13-motion-adapter-sim-design.md`
- 实施计划：`docs/superpowers/plans/2026-11-13-motion-adapter-sim.md`
- 分支 `exp/motion-adapter-sim-20261113`，worktree `mcgs/.claude/worktrees/motion-adapter-sim`
- 服务器根目录：`codex_artifacts/mcgs_motion_adapter_20261113/`（`trainer/`、`data/`、`configs/`、`ledger.jsonl`）

## 运行

在 worktree 根目录、本机上执行。`tools/rrun.py` 用 worktree 私有的 gpu.py 配置（`.claude/gpu/config.json`，
`repo` 指向本 worktree）快照并上传本 worktree，在 amax 上运行一条命令，跟随日志，并以作业的退出码退出。

```sh
E=autoresearch/motion_adapter_sim_20261113
/usr/bin/python3 $E/tools/rrun.py --cpu -- @python $E/tests/run_tests.py            # 全部 CPU 测试
/usr/bin/python3 $E/tools/rrun.py --cpu -- @python $E/tests/run_tests.py tests/test_geom.py
/usr/bin/python3 $E/tools/rrun.py --gpu ID --detach -- @python $E/run.py ...        # 训练，立即返回作业号
```

## 规则

- 时间只用本机时钟。服务器上不读任何时钟、文件时间或程序自己量的耗时；`tests/run_tests.py` 在服务器端删去
  pytest 的耗时，本实验的代码不打印、不写入任何时钟值。账本里的启动/结束时间是本机时间。
- GPU：启动前两次检查（间隔约 75 秒）都为空的卡才用；一卡一作业；不动别人的作业。
- GT、ROI、病灶掩模、运动真值不进训练（oracle 组除外，表中标注）。
- 适配器设置只在开发条件 D1–D3 上确定，冻结后才跑测试条件。

## 仿真数据（任务 4，2026-11-13）

`simulate.py --cond X` 在服务器 `data/X/` 下生成；`--check` 重新计算哈希并打印 QC。10 个条件全部生成，
`--check` 全部无哈希差异。QC 图在 `figures/big/qc/qc_X.png`（不进 git），与服务器文件 sha256 一致。

| 条件 | 运动 | 脑内位移 均值 / 最大 (mm) | 高频能量比 | 刚体往返误差（脑内相对 RMS） |
|---|---|---|---|---|
| E1a | 扫描间刚体（小） | 3.41 / 5.99 | 0.9994 | 1.96 % |
| E1b | 扫描间刚体（大） | 6.84 / 12.00 | 0.9990 | 2.88 % |
| E2 | 刚体 + 轻度非刚体 | 3.37 / 5.89 | 0.9993 | — |
| E3 | E2 + 对比度特有病灶（T1、T2 各自的 k 空间） | 3.37 / 5.89 | 0.9993 | — |
| E4a | 扫描内：一次阶跃 | — | — | — |
| E4b | 扫描内：缓慢漂移（111 个不同姿态） | — | — | — |
| E4c | E1a + E4b（参考姿态 GT 单独存 `gt_ksp_T2.h5`） | 3.41 / 5.99 | 0.9994 | 1.96 % |
| D1 | 开发：刚体 | 2.90 / 5.01 | 0.9992 | 1.63 % |
| D2 | 开发：刚体 + 非刚体 | 2.97 / 5.00 | 1.0002 | — |
| D3 | 开发：扫描内漂移 + 阶跃 | — | — | — |

所有条件：线圈灵敏度支撑外的增量能量为 0；快门置零的 k 空间点集合与原数据相同；扫描内条件的参考回波链
等于参考姿态（参考链第 54 条，参考段 3，共 110 条回波链，ETL 320，K = 8）。

两点与设计文档不同，照实记录：

- 设计写 E1a“约 2 mm”，按设计给定的角度和平移，脑内实际位移为均值 3.4 mm、最大 6.0 mm（绕颈部支点转动，
  离支点越远位移越大）。角度和平移没有改，以实测位移为准。
- 设计要求刚体往返误差 < 0.5 %。刚体变换用精确的傅里叶剪切旋转实现，对球形带限的内容误差 < 0.5 %（测试覆盖）；
  真实数据上为 1.6–2.9 %，来自 k 空间立方体角部的内容（主要是噪声），旋转必然把它混叠掉。高频能量比 0.999–1.000
  说明图像细节没有被抹平。第一版用样条插值旋转，往返误差 6.7 %，已作废，旧文件移到
  `data/_superseded_spline_warp/`（未删除）。
