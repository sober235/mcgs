# 不同掩膜实验 · 测试集（T1 + FLAIR，加速倍数 16）

本文件是这项实验的完整上下文：目标、用户决定、设计、文件、当前状态、初步结果、待办、恢复步骤和运行记录。
所有时间都是本机（Mac）时钟，时区 PST；不含任何服务器时间。负责会话：**diff-mask**（原名 mcgs-f1，用户 2026-11-23 改名）。

## 1. 目标与用户决定

| 本机时间 | 用户原话 / 决定 |
|---|---|
| 2026-11-22 ~06:55 | "These are different mask experiments … the following experiments need to be carried out using uniform density and radial masks with af = 16 to verify the robustness and generalization ability of the method using a larger test set." |
| 2026-11-22 ~07:00 | "What I mean is that 'different masks (uniform density, radial) are also applied to the test set'." |
| 2026-11-22 ~07:05 | "Don't use the mcgs-0f. Use a new one instead." → 单独的流程和输出目录，不进 mcgs-0f（现名 diff-af）的队列 |
| 2026-11-22 22:10 | 整个 MCGS 项目所有会话合计最多 3 张卡（经 mcgs-0f 转述）；5、6 号卡的任务合并到一张卡（经 mcgs-32 转述） |
| 2026-11-23 17:05 | 上限改为 4 张卡（经 mcgs-c9，现名 motion，转述）；各研究之间没有优先级 |

目的：在加速倍数研究所用的同一测试集上，用另外两种采样掩膜检验 MC-GS 的鲁棒性和泛化能力。
vd-Poisson AF16 已在加速倍数测试集研究（`autoresearch/t1flair_af_testset_20261121`）里跑过，作为第三种掩膜直接引用，不重跑。

## 2. 设计

- **数据**：加速倍数测试集研究的 6 个扫描、4 名受试者（不含调参受试者 jiangyizhou）：
  vol1_huangyuwei、vol2_jiangdian、vol4_tangbing、vol5_sunxinyi、vol6_jiangdian_790、vol7_tangbin_790。
  只读取那项研究准备好的 k 空间和线圈灵敏度（`mcgs_t1flair_af_testset_20261121/data/<scan>`），不在其目录下写任何东西。
- **掩膜**：
  - 均匀密度 `uniform`：用户 2026-11-11 批准的那张（.mat md5 d0796ac1…，2964 点；h5 md5 c2aee739… 是它沿读出方向的广播）。
  - 伪径向 `radial`：9 月的伪径向掩膜（h5 md5 579d4ac4…）；用户 2026-11-12 决定径向一行沿用它。
    "radial = 9 月伪径向"是我的理解，**用户尚未单独确认**。
- **方法（11 种）**：ZF、CG-SENSE、CS-SENSE、Joint CS-SENSE（JTV）、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR、GS-single、MC-GS。
- **冻结设置**（取自 jiangyizhou 上各掩膜课题的 `logs/summary.json`，测试扫描上不调任何参数）：

  | 掩膜 | CS-SENSE | JTV | L1-SPIRiT ks5 |
  |---|---|---|---|
  | uniform | T1 0.0014 / FLAIR 0.0014 | 0.002 / 0.002 | 0.001 / 0.001 |
  | radial | T1 0.0014 / FLAIR 0.001 | 0.0014 / 0.0014 | 0.0005 / 0.0005 |

  MC-GS / GS-single 用各掩膜登记运行的命令行（`af16{u,r}_p3_{joint,singleT1,singleFLAIR}_full4k`，即 AF16 phase-3 配方），
  训练器副本 `mcgs_sg2x2_wt` @ 54f8a2a。
- **行数**：高斯 36 行（6 扫描 × 2 掩膜 × joint / singleT1 / singleFLAIR）；对比方法 96 行（6 × 2 × 8 组）。
- **对照行**（在 jiangyizhou 均匀密度课题上复现登记值）：
  `jyz:uniform:joint` 应得 34.903 / 34.984 dB；`jyz:uniform:classical`、`jyz:uniform:cssense` 与登记值之差 ≤ 0.005 dB 才放行正式对比方法。
- **占位机制**：每行开始前在 `OUT/claims` 原子创建目录；已完成或已被占的行自动跳过，所以同一行列表可以在多张卡上并行，重启也不会重跑。
- **输出**：服务器 `/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_diffmask_testset_20261122/`
  （`trainer/results_arc/dm_*` 高斯结果，`topics/<scan>_<pattern>` 对比方法课题克隆，`claims/`，`control_check.json`）。

## 3. 文件

| 文件 | 作用 |
|---|---|
| `ts_common.py` | 扫描、掩膜、冻结参数、输出路径；`--selftest` 与两个课题的登记值核对 |
| `make_topics.py` `run_gs.py` `run_arms.py` | 从 `t1flair_af_testset_20261121` 移植：行里的 "af" 改为掩膜名，登记运行和行数相应修改，逻辑不变 |
| `check_all.py` | CPU 自检（四步） |
| `chain_arms.py` | 服务器任务：对比方法。可等高斯任务结束（默认）或立即开始（`--no_wait`）；对照核对结果写入 `OUT/control_check.json`，两份副本共用，不重复核对 |
| `watch_launch.py` | 本机脚本：等空卡后启动（项目卡数上限出台后不再使用） |
| `move_gs_worker.py` | 本机脚本：在行间断点把高斯任务搬到另一张卡（2026-11-22 用于 6 → 5；教训：停机后重启那一步必须在断线时重试） |

## 4. 当前状态（2026-11-24 06:10 PST）

| 任务（尾缀） | 卡 | 内容 | 进度 |
|---|---|---|---|
| …b2b0295f | 5（与 diff-af 的对比方法任务共用） | `run_gs.py`，36 行高斯 | 24 / 36（vol1、vol2、vol4、vol5 完成），无失败；共用卡上每行约 1.5–2 h |
| …b2b8b211 | 3（plain-enhan 让出） | `chain_arms.py --no_wait` | 对照核对**通过**（6 个结果差值全为 0.000000 dB）；正式 13 / 96，约 0.7 h / 行 |
| …71ef6fa7 | 5 | `chain_arms.py`（等待） | 高斯任务结束后加入，读取已有核对结果，和 3 号卡分跑剩余对比方法行 |

项目卡数（2026-11-24 05:40 PST，各会话核对）：0 = diff-af，2 = motion，3 = diff-mask，5 = diff-af + diff-mask，已达 4 张上限；1 号卡不取用。

## 5. 初步结果（仅供参考，正式统计待全部完成后统一重新评分）

**MC-GS 与 GS 单对比度**（全头部；MC-GS 取 joint 的 T1 / FLAIR 通道，GS-single 取各自单对比度训练；训练器评分）：

| 扫描 | 掩膜 | MC-GS T1 / FLAIR (dB) | GS-single T1 / FLAIR (dB) | ΔPSNR T1 / FLAIR | ΔSSIM T1 / FLAIR |
|---|---|---|---|---|---|
| vol1 | uniform | 34.008 / 33.703 | 33.324 / 33.600 | +0.684 / +0.103 | +0.0148 / +0.0032 |
| vol1 | radial | 29.160 / 30.282 | 28.851 / 30.160 | +0.309 / +0.122 | +0.0128 / +0.0046 |
| vol2 | uniform | 34.687 / 34.882 | 33.957 / 34.775 | +0.730 / +0.107 | +0.0150 / +0.0026 |
| vol2 | radial | 29.936 / 31.605 | 29.657 / 31.486 | +0.279 / +0.119 | +0.0108 / +0.0038 |
| vol4 | uniform | 32.306 / 31.605 | 31.840 / 31.486 | +0.466 / +0.119 | +0.0130 / +0.0048 |
| vol4 | radial | 27.854 / 28.633 | 27.694 / 28.538 | +0.160 / +0.094 | +0.0083 / +0.0046 |
| vol5 | uniform | 33.274 / 33.963 | 32.654 / 33.898 | +0.620 / +0.065 | +0.0145 / +0.0016 |
| vol5 | radial | 28.765 / 31.464 | 28.521 / 31.365 | +0.245 / +0.099 | +0.0099 / +0.0028 |

已完成的 8 组中，MC-GS 的 PSNR 和 SSIM 全部高于 GS-single。（vol2 radial 与 vol4 uniform 的 FLAIR 前三位小数相同属巧合，其余指标都不同，已核对。）

**vol1 与对比方法**（全头部 PSNR / SSIM；对比方法为其评估脚本的分数，MC-GS 为训练器分数，两者口径尚未统一）：

- uniform：MC-GS 34.008 / 33.703 dB（SSIM 0.915 / 0.904），最强对比方法 ZS-SSDU 31.674 / 32.661 dB（0.883 / 0.903）→ MC-GS 第一。
- radial：FLAIR 上 MC-GS 30.282 dB（0.845）高于 ZS-SSDU 29.878 dB（0.836）；
  **T1 上 ZS-SSDU 29.665 dB（0.865）高于 MC-GS 29.160 dB（0.859）**。需在统一重新评分后确认，并在更多扫描上观察。

## 6. 待办与待确认

1. 等高斯 36 行与对比方法 96 行全部完成（按目前速度约一天半到两天）。
2. 移植 `t1flair_af_testset_20261121/collect_stats.py`（掩膜名代替 AF 键；从重建体数据统一重新评分；把 diff-af 的 vd-Poisson AF16 行并入作为第三种掩膜），等 diff-af 的统计脚本定稿后再做；图按 Nature 风格。
3. 关注伪径向 T1 上 MC-GS 与 ZS-SSDU 的差距（见第 5 节）。
4. 待用户确认："radial" 是否就是 9 月的伪径向掩膜。

## 7. 恢复步骤（新会话接手）

1. `/usr/bin/python3 ../gpu-tools/gpu.py jobs` 找到上表三个任务；只显示 8 位尾缀（完整 id 含服务器时钟）。
2. 进度（不读时间）：`ls <OUT>/trainer/results_arc | grep '^dm_.*_score.json$' | grep -v jyz | wc -l`（共 36）；对比方法看 `OUT/claims/arm_*` 与任务日志里的 `===== <row> done`。
3. 查日志：任务目录 `/home/congcongliu/.cache/mcgs-vscode/jobs/<id>/output.log`，找 `FAILED`、`Traceback`、`CHAIN DONE`；读之前过滤掉匹配 `[0-9]:[0-9][0-9]:[0-9]` 的行。
   注意：通过 ssh 运行 `pgrep -f <脚本名>` 会匹配到 ssh 自己的 shell，要用 `ps -eo pid,args | grep '[a]naconda3/envs/GSTMI/bin/python.*<脚本>'`。
4. 共用卡上某行因显存不足失败：只释放该行（占位目录 + 部分输出），用同一行列表在同一张卡上重启一次；不算方法失败。
5. 对照核对失败（`control_check.json` 中 `ok: false`，任务退出码 2）：停下汇报，不手动跑正式行。
6. 取卡前先数全项目的卡（ListAgents + 询问各会话 + `nvidia-smi`）；`gpu.py jobs` 不一定显示每个任务的卡号。

## 8. 运行记录

| 本机时间 | 事项 |
|---|---|
| 2026-11-22 07:00 | CPU 自检全部通过（任务 …23ecea99）：冻结参数与课题登记值一致，13 个课题克隆（12 正式 + jiangyizhou 对照），两个运行器自检通过 |
| 2026-11-22 07:02 | 八张卡全部被占用，无法启动；`watch_launch.py --max 4 --skip 4` 在本机等待空卡 |
| 2026-11-22 07:40 | 6 号卡空出（mcgs-81 让出；0、3、5、7 归 mcgs-0f，1、2 归 mcgs-81，均不取用），高斯工作任务 …17a9b829 启动：对照行 `jyz:uniform:joint` + 36 行正式训练 |
| 2026-11-22 17:21 | 对照行 `jyz:uniform:joint` = 34.9026 / 34.9848 dB（登记值 34.903 / 34.984，复现）。正式行完成 7 / 36。对比方法尚未启动：无空卡 |
| 2026-11-22 22:10 | 用户新规则：MCGS 项目合计最多 3 张卡；5、6 号卡的任务合并到一张卡；与 mcgs-0f 商定 5 号卡保留两个任务，本实验的高斯任务搬过去 |
| 2026-11-22 22:59 | `move_gs_worker.py` 在行间断点（对照行 + 13 行正式训练已完成）停掉 6 号卡上的任务；此时与服务器的连接中断，重启命令没有发出 |
| 2026-11-23 00:31 | 高斯工作任务 …b2b0295f 在 5 号卡上重启（与 mcgs-0f 的对比方法任务共用）；停机约 1.5 小时，已完成的行不重跑 |
| 2026-11-23 00:51 | 对比方法链式任务 …9fb40f19 在 5 号卡上排队：高斯任务结束前不占 GPU，之后先核对两个对照行再跑 96 行 |
| 2026-11-23 16:59 | 高斯 19 / 36；vol1、vol2、vol4 的 MC-GS 均高于 GS-single |
| 2026-11-23 17:05 | 用户把项目上限改为 4 张卡 |
| 2026-11-23 19:57 | plain-enhan（原 mcgs-32）确认不需要卡，让出 3 号卡（两次检查为空）。`chain_arms.py` 改写（`--no_wait`、共用核对结果文件）；停掉空等的 …9fb40f19；3 号卡启动 …b2b8b211（立即开始），5 号卡启动 …71ef6fa7（等高斯任务结束） |
| 2026-11-24 05:32 | 对照核对通过（6 个结果差值全为 0）；对比方法正式 13 / 96，高斯 24 / 36，无失败 |
| 2026-11-24 05:40 | 1、2 号卡空着，经 diff-af 和 motion 核对：0 号为 diff-af，motion 将回到 2 号卡，项目已达 4 张上限，1 号卡不取用 |
