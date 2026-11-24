# 运动研究（第二版「轻微调整」）：当前上下文（2026-11-24）

这份文件给接手的人（或下一个会话）看：这项研究做到哪里、东西放在哪里、正在跑什么、怎么接着做。
写于 2026-11-24 06:05 PST（本地电脑的时钟；文中所有时间都是本地时钟）。数字照录自记录，没有重新计算。

## 1. 研究是什么

用户的 Motion-Aware MC-GS 设想（2026-11-13）：T1W 与 T2W 两个对比度联合重建时，T2W 扫描中的轻微脑部运动
（扫描间错位 + 扫描内运动）会损失多少，能否收回。第一版（`../../motion_adapter_sim_20261113/`，结果见其
`RESULTS.md`）让适配器的全部参数靠 k 空间数据一致性去学，四条判据 S1–S4 都没有全部通过。

第二版（本目录，用户 2026-11-21 批准）不换框架，只做三处小改动：

- **M1** 扫描间变换由零填充图像的配准给出（对齐步已关，只保留刚体位姿）；
- **M2** T2 数据一致性按回波链加权（每条链的统计量 = 逐线 能量 / 正常能量 的中位数，z0 = 6、z1 = 12）；
- **M3** 检测到持续偏移时，之后的数据用一个状态位姿（检测量 = 每条回波链的线性化位移，配准初始化）。

设置在开发条件（DTa、DTc、D2）上确定并通过零运动关卡后冻结为 `../settings_frozen.json`（2026-11-22），之后不再改。

- 设计稿：`docs/superpowers/specs/2026-11-21-motion-mild-adjust-design.md`
- 实施计划（13 个任务）：`docs/superpowers/plans/2026-11-21-motion-mild-adjust.md`
- 研究说明与全部开发、上限组读数：`../README.md`
- 执行记录（每条裁定 `Ruling:` 及其代价）：本目录 `progress_ledger.md`（原件在 git 忽略的
  `.superpowers/sdd/2026-11-21-motion-mild-adjust/progress.md`，这里是 2026-11-24 06:04 PST 的副本）

## 2. 分支与位置

- 本地工作树：`mcgs/.claude/worktrees/motion-adapter-sim`，工作分支 `exp/motion-mild-adjust-20261121`
  （GitHub 上同名分支停在 2486ac3；其后的提交在本地）。
- 这份上下文所在的新分支：`exp/motion-mild-context-20261124`（从工作分支的当前提交建出并推送）。
- 本地旧分支 `exp/motion-adapter-sim-20261113`（9d51b06）是改写前的备份，提交哈希是旧的，**不要推送**。
  执行记录里 2026-11-22 换分支之前的哈希是旧哈希，按提交标题对应。
- 服务器结果根目录：`codex_artifacts/mcgs_motion_mild_adjust_20261121`（训练结果、`*_mm.json`、体数据留在服务器）。

## 3. 进度

| 任务 | 内容 | 状态 |
|---|---|---|
| 1–5 | 采集顺序与常量、仿真器、回波链统计、运行时补丁、评估核心 | 完成 |
| 6 | 可行性验证（交错顺序 DTa） | 通过：AUC 1.000，真值权重收回 67.7 % |
| 7–9 | M2、M1、M3 实现 | 完成 |
| 10 | 开发三轮（d1 / d2 / d3，共 9 次训练，上限 10）、零运动关卡 7 行、冻结 | 完成（2026-11-22） |
| 11 | 正式矩阵 43 行 | **进行中：34 行完成，1 行在跑，8 行排队**（见 `formal_rows.md`） |
| 12 | 判据 S1–S4 与汇总（`mevaluate.py --all`、`summary.json`、`tools/mtables.py`） | 未开始（S3 的报告字段已提前加入，见第 5 节） |
| 13 | 出图与中文 `RESULTS.md`（`mfigures.py --all`） | 未开始；图像图的格式已由用户确定（见第 6 节） |
| 之后 | 一次全分支审阅、一轮修正、最终说明（列出全部 `Ruling:`）、收尾 | 未开始 |

正式矩阵的组成：可行性行 il:C0:B0 + 零运动关卡 7 行 + 上限组 13 行（队列 `upper`，已全部完成）+ 其余 22 行
（队列 `rest`，13 行完成）。全部用冻结设置、种子 42、AF12。上限组（Ow / Op）的读数和损失已写在 `../README.md`
「正式矩阵」一节：交错顺序损失 Ta 1.898、Tb 1.851、Tc 8.591、E2Tb 1.275 dB；Ow 收回 82.7 / 66.1 / 64.7 / 100.6 %，
Tc 的 Op 收回 86.0 %；线性顺序 Ta 损失 4.060、Tc 6.773 dB，Tc 的 Ow 只收回 10.7 %、Op 80.0 %。

`formal_rows.md` 里 `rest` 各行的 PSNR 是训练结束时打印的头部 PSNR（训练器读数），不是任务 12 的正式评分，
不能直接拿来判 S1–S4。

## 4. 正在跑什么、怎么接着跑

- 队列 `rest` 在本机后台独立运行（父进程是 launchd，关掉编辑器窗口不影响），只用 amax 的 2 号卡。
  2026-11-24 05:41 PST 在 2 号卡上启动 il:C0:I-T2；之后依次是 il:{Ta,Tb,Tc,E2,TbL}:I-T2、il:Tb:I-T2+w、
  kz:{Ta,Tc}:B123。每行约 1 小时（本地墙钟），全部跑完约在 2026-11-24 14:00–15:00 PST（单对比度的 I-T2 行可能更快）。
- 查看：`pgrep -fl "queue.py --study mild"`；日志 `.claude/gpu/queue_rest.log`（本地时钟；本目录 `ops/` 有副本）。
- 队列不在了就从工作树用**同一条命令**重启，它会跳过已完成的行、接上正在跑的行：

  ```sh
  /usr/bin/python3 .claude/gpu/detach.py /dev/null -- "unset SSH_AUTH_SOCK; cd <工作树> && /usr/bin/python3 -u \
    autoresearch/motion_adapter_sim_20261113/tools/queue.py --study mild --rows <22 行，见下> \
    --settings settings_frozen.json --name rest 2>&1 | sed -l -E 's/j-[0-9]{8}T[0-9]{6}-/j-/g' >> .claude/gpu/queue_rest.log"
  ```

  22 行：`af12:il:Ta:B123 af12:il:Tb:B123 af12:il:Tc:B123 af12:il:Ta:B2 af12:il:Tb:B2 af12:il:Tc:B2 af12:il:E2:B0
  af12:il:E2:B1 af12:il:E2:B123 af12:il:E2Tb:B123 af12:il:E2Tb:B12 af12:il:TbL:B0 af12:il:TbL:B123 af12:il:C0:I-T2
  af12:il:Ta:I-T2 af12:il:Tb:I-T2 af12:il:Tc:I-T2 af12:il:E2:I-T2 af12:il:TbL:I-T2 af12:il:Tb:I-T2+w af12:kz:Ta:B123
  af12:kz:Tc:B123`。日志用 `>>` 追加，不要用 `>`。

- **`unset SSH_AUTH_SOCK` 必须保留。** 2026-11-23 20:17 到 2026-11-24 05:33，队列从启动它的登录会话继承的
  SSH 代理失去响应，ssh 卡在向代理要密钥这一步，每次查卡、查作业都失败；il:TbL:B123 约 20:10 就已跑完，
  2 号卡空了约 9 个半小时。连接 amax 的密钥没有口令、配置里写了 IdentityFile，不需要代理。
  日志里超过 10 分钟只有 "gpu check failed" / "state check failed" 时，先手动测 ssh，再按上面重启。
- 卡：`.claude/gpu/avoid.json` = `[5, 0, 6, 3, 7, 1, 4]`（只放开 2 号卡）。
  `.claude/gpu/claims.json` 记录卡与行的对应。不要直接调 `gpu.py run`（它会打印服务器时钟的
  "created" 字段）；CPU 作业用 `../../motion_adapter_sim_20261113/tools/rrun.py --cpu -- @python ...`。
- **项目 GPU 上限：所有 mcgs 会话合计不超过 4 张卡**（用户 2026-11-23 17:05 PST，已写入工作区 CLAUDE.md），
  研究之间不分先后，每个研究一张卡。2026-11-24 05:41 PST 的占用：0 号 diff-af，3 号和 5 号的一半 diff-mask，
  5 号另一半 mcgs-0f，2 号 motion（本研究），正好 4 张。会话名（用户改名）：本会话 `motion`（原 mcgs-c9），
  `diff-mask`（原 mcgs-f1），`plain-enhan`（原 mcgs-32）。
- 2 号卡曾两次独占借给 plain-enhan（2026-11-23 00:50–02:02、03:12–04:32 PST）；第三次（05:59 起）它没有用上，
  2 号卡空到 17:00 PST。之后不再借 2 号卡。

## 5. 留给任务 12 的问题（设置不改，只决定怎么报告）

1. **il:Tb:B123 的误处理率 0.284。** M3 在第 81 条链开出第二个状态；Tb 的真值在 0.74（第 80.7 条链）处有一个
   持续的小台阶（0.05°、0.1 mm，低于判「受污染」的 0.2 mm），所以第 81–109 条链按定义是「干净」的，被放进
   非参考状态就算误处理：29 / 102 = 0.284。找到的位姿与这个台阶相符，M2 没有误降任何链。
   已裁定：预先登记的 S3 项保持原定义，所以这一行**不通过**该项；另加只报告、不参与判定的字段
   `false_rate_unmoved`（只数真实位移 ≤ 0.05 mm 的干净链）、`n_clean_moved`、`n_clean_moved_handled`（提交 c3cf285）。
   Tb、E2Tb、TbL 都有这 29 条有真实 0.144 mm 位移的干净链。
2. **I-T2+w 的零运动关卡**只按权重项判了通过；它的 PSNR 基线是 I-T2（il:C0:I-T2 正在跑），跑完后在 S1 里补上差值。
3. **il:TbL:B123 在 `experiments.jsonl` 里的本地结束时间**是队列恢复后 05:34 PST 才看到的时间，比实际结束晚约 9 小时
   （见第 4 节），它的「墙钟时长」不能用。
4. 线性顺序 C0 两行全程最大 z 为 5.14 / 5.24，离 z0 = 6 的余量只有 0.8（交错顺序 1.3 以上）；线性顺序只报告，不参与判据。

## 6. 剩下的步骤

1. 任务 11 收尾：两个队列都结束后，`mrun.py --dry --all` 应为 43 行；README 补上其余行（含 I-T2+w 的关卡差值）；
   提交前对 `experiments.jsonl` 运行 `../../motion_adapter_sim_20261113/tools/scrub_clock.py`；
   提交 `results(motion-mild): formal matrix, 43 runs`。
2. 任务 12：`mevaluate.py --all`（并行评分，结果缓存在各结果文件旁）、检查所有要提交的文件里没有服务器时钟
   （包括 `experiments.jsonl`）、`summary.json`、`tools/mtables.py`。
3. 任务 13：`mfigures.py --all`，中文 `RESULTS.md`，先给用户看 `fig_recovered.png`。图像图的格式**已由用户确定**
   （2026-11-22「save new type for the figure of motion research」）：沿用不同掩模记录图的版式（图像行 + 放大行、
   橙色指标、inferno 误差图，`mfigures.py::record_figure`，`--all` 画成 `fig_images_record.png`），箭头只放在放大的
   加权图像框里误差最大处。图一律按 Nature 风格（183 mm、5–7 pt 无衬线、600 dpi）。
4. 一次全分支审阅 → 一轮修正 → 最终说明（列出全部 `Ruling:` 和推迟的小问题）。
5. 推送只在用户要求时做。

## 7. 时间规则的陷阱

- 服务器上的作业目录名形如 `j-<服务器日期时间>-<后缀>`：所有 ssh / rrun / 队列输出都先经过
  `sed -E 's/j-[0-9]{8}T[0-9]{6}-/j-/g'` 再看；提交 `experiments.jsonl` 前运行 `scrub_clock.py`（正在跑的队列仍会把
  完整路径写进 `argv`）。
- 不读服务器的 `date`、`uptime`、进程启动时间、文件时间（`ls -l`、`stat` 的时间字段）、带时间戳的日志行。
- amax 上的 `ps` 目前会崩溃（"free(): invalid pointer"）；要看进程属主和命令行，读 `/proc/<pid>/cmdline` 和属主即可。

## 8. 本目录的文件

| 文件 | 内容 |
|---|---|
| `README.md` | 本文件 |
| `formal_rows.md` | 正式矩阵 43 行的状态与训练器读数（由 `../experiments.jsonl` 生成） |
| `progress_ledger.md` | 执行记录副本（全部 `Ruling:` 与 note，本地时钟） |
| `ops/detach.py` | 把命令从 Claude Code 进程完全分离出来运行的小工具（`.claude/gpu/detach.py` 的副本） |
| `ops/queue_rest.log`、`ops/queue_upper.log` | 两个队列的日志副本（本地时钟，作业目录名已去掉服务器时钟） |
