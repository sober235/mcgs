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
