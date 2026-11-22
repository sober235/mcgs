# T1+FLAIR 加速倍数研究：独立版 GS-single（2026-11-21）

用户 2026-11-21 要求：做一个独立的 GS-single，并把结果同步到 AF12 记录图上。

- `indep_single.py`：运行脚本。行 `<af>:<T1|FLAIR>` 是单对比训练器上的独立模型，行 `<af>:full` 是联合配方在同一份代码里的对照。用法见脚本开头。
- `results/`：七次运行的分数文件，`table.md` / `table.json` 是四个加速倍数的对照表（AF16 取自 2026-11-10 框架消融的 `fw_t1flair_nojoint_*`）。
- 图在 `../t1flair_af_record_fig_20261120/figures/af{12,16,20,24}_record_indep_single.*`，由该目录的出图脚本加 `--single indep` 画出。
- 结论和限定写在 `../t1flair_af_record_fig_20261120/RECORD.md` 的补充一节。
- 服务器输出：`/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_t1flair_af_indep_single_20261121/`（训练器副本、体积、图）。
- 运行：GPU 1/2/3 各一个任务，本机时间 2026-11-21 07:24–07:29 PST 启动；七次训练全部退出码 0，卡已释放。
