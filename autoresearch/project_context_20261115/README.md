# MC-GS 项目：当前上下文与进展（2026-11-15）

这份文件给接手的人（或下一个会话）看：项目现在做到哪里、东西放在哪里、哪些事停着、哪些事等用户决定。
写于 2026-11-15 07:00 PST（本地电脑的时钟）。数字照录自各线的记录，没有重新计算。

相关的两份文件：

- 已完成实验的统计表：[`../summary_completed_20261114/README.md`](../summary_completed_20261114/README.md)
- 完整报告（结论、数字表、600 dpi 的图）的源文件：[`../report_completed_20261114/`](../report_completed_20261114/)；成品 PDF 在服务器上，见第 4 节。

## 1. 现在的状态

- **已完成 17 条实验线**，1 条在进行，1 条暂停，3 项从未执行（明细见统计表）。
- **本会话没有在跑的 GPU 任务**，也没有排队的任务。
- 最近三天做完的事：
  1. T1+T2（tangbing）消融调参第三阶段结束：完整模型对三种消融配置 18 格全胜，种子 7 复核通过（2026-11-14）。
  2. 画了三张图：tangbing 三平面消融图、调参过程图、CE AF12 支撑版三平面图（2026-11-14）。
  3. 已完成实验的统计表（2026-11-14）。
  4. 完成报告的 2026-11-14 版 PDF，存在服务器上（2026-11-15）。

## 2. 各线进展

| 线 | 状态 | 一句话结果 | 记录在哪里 |
|---|---|---|---|
| T1+FLAIR，AF12 / 16 / 20 / 24 | 完成 | 12 个方法里头部 PSNR、SSIM、HFEN 全第一 | 报告第 4 节 |
| 耦合归因、超参数、噪声 | 完成 | 联合增益里共享几何占 +0.046 dB；噪声 ≤ 0.5σ0 时全第一 | 报告第 4 节 |
| 高斯容量消融 | 完成，有记录图 | 超过 360 万后没有一致的提高 | 分支 `ablation`：`autoresearch/hparam_capacity_fig_20261111/` |
| 不同掩膜（均匀密度重做） | 完成，有记录图 | 三种图案下 MC-GS 都最好；横断 205 为 18 / 18 | 分支 `diffmask-af16-masks-20261111` |
| T1+T2，AF12（huangyuwei） | 完成 | 头部指标全第一；对 L1-ESPIRiT +0.30 / +0.44 dB | 报告第 5 节 |
| 儿童 TSC，AF12 | 完成 | PSNR 和 SSIM 第一；联合增益只有 +0.01 dB | 报告第 6 节 |
| CE 前瞻 AF5.33 | 完成 | 无真值，只能做方法间比较 | 报告第 7 节 |
| CE 回顾 AF16（9 月，BART 掩膜） | 完成，有记录图 | 11 个方法全第一；六面板全胜只对登记版 GS-single 成立 | 报告第 8 节、第 14.4 节 |
| CE 回顾 AF16（vdPois）与对比方法审计 | 完成 | 15 块的 ZS-SSDU 超过 MC-GS | 报告第 9 节 |
| CE AF16 autoresearch | 结束，目标未达成 | 冠状、矢状病灶层面胜全部对手；横断面落后 ZS-SSDU 约 0.20 / 0.31 dB | 报告第 10 节 |
| CE AF12 重跑 | 完成 | 11 个方法只换掩膜；MC-GS 全第一，六面板全胜 | 分支 `ablation`：`autoresearch/ce_af12_rerun_20261112/` |
| 膝 SKM-TEA，AF16 | 完成 | 对 GS-single 5 / 5 例更好；15 块 ZS-SSDU 在回波 1 高 1.02 dB | 报告第 11 节（分支不在 GitHub） |
| 心脏，第一阶段 | 完成 | 只验证了读取、轨迹、参考重建、评估口径 | 报告第 12 节 |
| 结构与采集侧探针 | 完成 | 多为负结果 | 报告第 13 节 |
| 框架级消融，15 行 | 完成 | 先验是决定性的；各向异性小；联合建模 ±0.1 dB、方向不一致 | `../ce_af16_ablation_20261110/RESULTS.md` |
| 消融调参：CE | 调参结束，**收尾暂停** | 完整胜独立 +0.203 dB，只有种子 42 | `../ablation_coupling_tune_20261111/program.md` 第 9 节 |
| 消融调参：T1+T2 huangyuwei | 结束，目标未达成 | T1 +0.070，T2 −0.021 dB | 同上，第 10 节 |
| 消融调参：T1+T2 tangbing | 完成，目标达成 | 18 / 18，最小余量 +0.034 dB，种子 7 通过 | 同上，第 11 节；`RESULTS.md` 第 7.1 节 |
| 运动感知 MC-GS 仿真 | **进行中**（另一会话） | 提交记录显示适配器设置已在开发集 D1–D3 上冻结 | 分支 `exp/motion-adapter-sim-20261113`（本地工作树；GitHub 上只有设计文档那一个提交） |

### tangbing 上的最终配置（要复现时用）

V32 配方，加上：

- 对所有配置同样改（`tune.py --sym`）：`loss.lambda_wav=0.08`、`gaussian.rotation_lr=0.008`、`gaussian.scale_lr=0.004`；
- 只属于联合模型（`tune.py --ov`）：`--lambda_edge 0.001 --edge_kappa 100`。

种子 42，4000 步，每个模型 240 万个高斯。完整模型 33.959 / 35.095 dB；最好的独立模型 33.835（`r3_rot008_t1`）/ 35.061（`r4_scl004_t2`）。

## 3. 没有完成的事

- **暂停：** CE AF16 消融调参的收尾第 2 至 6 步（种子 7 复核；用保留配置重跑「去掉各向异性」「去掉对比内先验」）。用户 2026-11-12 指示暂停，没有说重新开始，不要自行启动。
- **未达成：** CE AF16 横断面病灶层面胜过 ZS-SSDU；huangyuwei 的 T1+T2 上完整模型胜独立模型。
- **从未执行：** 心脏第二阶段的方法对比；ZS-SSDU 按 15 块在其余各线上的重跑；2×2 结构消融的 14 例核心测试。
- **没有做的核对：** tangbing 的最终配置没有在别的受试者上验证；AF12 上没有用独立单对比模型核对「六面板全胜」。

## 4. 东西在哪里

### GitHub（`sober235/mcgs`）

| 分支 | 提交 | 内容 |
|---|---|---|
| `exp/ablation-coupling-tune-20261111` | 本提交 | 消融调参的程序、台账、图；框架级消融的 `RESULTS.md`；统计表；报告源文件；本文件 |
| `ablation` | 656e459 | 2026-11-10 版报告的源文件；容量消融记录图；CE AF12 重跑 |
| `diffmask-af16-masks-20261111` | 5467a58 | 不同掩膜的重做和记录图 |
| `exp/motion-adapter-sim-20261113` | 6ebb38b | 运动仿真的设计文档（后续提交只在本地工作树） |

膝 SKM-TEA 这条线的分支没有推到 GitHub。

### 服务器（amax）

所有计算都在这台服务器上做，本地电脑只编辑代码、取回结果、看图。

| 内容 | 路径 |
|---|---|
| 各课题的重建体数据、图、日志 | `/data0/congcong/code/GS/Results_20260908/<课题>/` |
| 近期的消融、调参、出图 | `/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/` |
| 框架级消融与 tangbing 调参的全部训练输出 | `…/codex_artifacts/mcgs_framework_ablation_20261110/`（`trainer/results_arc/`、`tune/experiments.jsonl`、`fig_ablation/`、`fig_process/`） |
| **完成报告，2026-11-14 版** | `…/codex_artifacts/mcgs_completed_report_20261114/MCGS_completed_experiments_report_20261114.pdf` |
| 完成报告，2026-11-10 版 | `…/codex_artifacts/mcgs_completed_report_20261110/` |

2026-11-14 版报告：63 页，44 幅图，全部位图在印刷尺寸下不低于 600 ppi，sha256 `5e81f042…4ed340`（2026-11-15 换入重画的容量图后的版本）。重建命令：

```sh
/usr/bin/python3 ../gpu-tools/gpu.py run --cpu -- @python autoresearch/report_completed_20261114/build_report.py
```

重建体数据、检查点和 600 dpi 的大图只在服务器上有一份；没有查到备份的记录。

### 用户指定的记录图

| 实验 | 记录图 | 指定日期 |
|---|---|---|
| 平扫 / 增强（已被取代） | 9 月 CE AF16，层面 201 / 131 / 48 的三平面图 | 2026-11-11 |
| 平扫 / 增强（现行） | CE AF12 临床验证图，MC-GS 为调参后的配置（见第 8 节） | 2026-11-22，2026-11-24 换成调参版 |
| 高斯容量消融 | `hp_capacity_slice181`（横断 181） | 2026-11-11 |
| 不同掩膜 | 横断 205 的记录格式图 | 2026-11-13 |

## 5. 用户已决定的事（2026-11-15）

写本文件时有四件事待定，用户当天答复如下：

1. **容量消融记录图 e 面板上的训练时长：去掉。** 已重画（e 面板只留高斯个数和峰值显存，a 至 d 面板不变），服务器记录目录、仓库和报告 PDF 都换成了新版；出图脚本不再读取运行的时长。
2. **CE AF16 消融调参的收尾：仍然暂停。** 用户问的是最后两步调参是否有效，没有说恢复。已知的情况：最后两步（边缘 TV 引导 α 0.25 → 0.5，相位 TV 引导 α 0.25 → 0.5）在种子 42 上把余量从 +0.168 提到 +0.188、再到 +0.203 dB，各自超过 0.005 dB 的保留线，但增幅明显变小。别的种子没有测；已测到的种子差是 0.002 dB（种子 7 的平扫独立模型）。收尾的五次训练没有做。
3. **tangbing 的最终配置没有在别的受试者上验证。** 用户确认了这一现状，没有要求安排验证实验；引用结果时作为限定写明。
4. **服务器上的结果没有备份。** 用户确认了这一现状，没有要求安排备份。

## 6. 现行的工作约定

- 计算只在 amax 上做。GPU 任务只通过 `gpu.py run --gpu ID` 提交；提交前隔约 75 秒查两次该卡上没有任何进程；一张卡一个任务；不动别人的任务。
- 时间一律用本地电脑的时钟。不读取、不显示服务器上的任何时间信息（包括程序自己计的时长）。
- 固定的评估器、掩膜、真值和对比方法不改；真值和 ROI 不作为训练输入。
- 看图时先给 PNG，再给数字和说明。结果存在仓库里该实验自己的目录下；大图放在不被跟踪的子目录里。
- 不自动推送；推送要用户明确要求。
- 同一个循环只有一个会话写台账和 `program.md`；动手前先确认别的会话没有在管它。

## 7. 接手时先看什么

1. 本文件和统计表。
2. `../ablation_coupling_tune_20261111/program.md` 第 11.8 节（第三阶段的逐轮记录和交接说明）。
3. `../ce_af16_ablation_20261110/RESULTS.md` 第 7、7.1 节。
4. 服务器上的报告 PDF，或它的源文件 `../report_completed_20261114/report.md`。

常用命令（在本地 `mcgs/` 目录下）：

```sh
# 第三阶段的记分板（只读，CPU）
/usr/bin/python3 ../gpu-tools/gpu.py run --cpu -- @python autoresearch/ablation_coupling_tune_20261111/board.py
# tangbing 三平面消融图
/usr/bin/python3 ../gpu-tools/gpu.py run --cpu -- @python autoresearch/ce_af16_ablation_20261110/abl_fig.py --tasks t1t2tb
# 调参过程图
/usr/bin/python3 ../gpu-tools/gpu.py run --cpu -- @python autoresearch/ablation_coupling_tune_20261111/fig_process.py
```

## 8. 补记（2026-11-24）：平扫 / 增强的最优结果

本节是 2026-11-15 之后补写的；前面各节仍是 2026-11-15 的状态。

- **指定（2026-11-22）。** 用户把 CE AF12 的三平面图（横断 201 / 冠状 131 / 矢状 48）定为平扫 / 增强重建的最优结果，并作为论文的临床验证场景。数据是 UIH 平扫 + 增强，真值为 CS-SENSE，掩膜为 BART Poisson-disc，名义 AF12。该图取代 2026-11-11 指定的 9 月 CE AF16 图。
  - 画法：inferno 误差图，范围 [0, 0.08]。
  - 箭头：红色，只画在放大的加权像框里，指向病灶边界。
  - 输出：600 dpi PDF。
- **调参（2026-11-22 至 11-23）。** 用 autoresearch 在同一数据上调 MC-GS：8 次调参训练，另有 1 次是保存体积的重跑。
  - 最终配置：登记配方 P3，加上 `--lambda_edge 0`、edge-TV 引导 α 0.75、phase-TV 引导 α 0.5、`rotation_lr 0.008`、`scale_lr 0.004`。
  - 训练器记分：36.031 / 38.473 dB，比登记的 9 月运行高 +0.226 / +0.328 dB。
  - 种子 7 复核：36.022 / 38.465 dB。
  - 记录在分支 `exp/ce-af12-tune-20261122` 的 `autoresearch/ce_af12_tune_20261122/program.md`。
- **换图（2026-11-24，用户决定）。** 最优图改为用调参后的 MC-GS 重画的版本。其余 10 种方法、真值、层面、误差范围和箭头规则都不变。
  - MC-GS 在 6 个面板的 PSNR / SSIM / NMSE 上 18 项全第一。
  - 领先第二名 GS-single 0.36–0.63 dB（未调参时为 0.19–0.36 dB）。
- **引用时的限定。**
  - MC-GS 的超参数是按同一扫描对真值的头部 PSNR 选的。对比方法沿用原设置，其中三个扫参方法的 λ 也是按真值选的。
  - 只有一个受试者、一次掩膜抽样。
  - 箭头位置是对照真值挑的，只用于展示。
- **位置。**
  - GitHub 分支 `optimal-for-plain-scan-and-enhancement-phases`（8cfc41f，已推送）：`autoresearch/ce_af12_rerun_20261112/README.md` 的最优结果一节，旧版说明也保留在该节。
  - 服务器：`…/codex_artifacts/mcgs_ce_af12_prepost_tuned_20261123/v1_tuned/`，含 600 dpi PDF、PNG 和所画数据 npz。旧版在 `…/mcgs_ce_af12_prepost_optimal_20261122/v4_clinical_verification_zoom_arrow/`。
  - 本地：`autoresearch/ce_af12_rerun_20261112/figures/results_optimal/`，不进 git。
