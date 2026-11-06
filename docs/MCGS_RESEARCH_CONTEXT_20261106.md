# MC-GS 平扫／增强重建：进展、CSM 核查与上下文交接

日期：2026-11-06（Asia/Shanghai）。本文记录本次科研讨论和实际核验结果，是一个时间点的快照；实时进展以运行目录的状态和账本为准。

## 1. 当前研究任务

在固定 CE AF16 平扫（pre）和增强（post）病例上，使用 MC-GS 联合重建，并通过 Karpathy 风格 autoresearch 调整允许的损失和优化参数。

用户最终明确的成功标准：

| 固定切面 | 数组轴 | 切片索引 |
|---|---:|---:|
| 横断位 | 0 | 201 |
| 冠状位 | 1 | 130 |
| 矢状位 | 2 | 48 |

每个切面的平扫与增强均需在 PSNR、SSIM、NMSE 上严格超过全部十个对照，共18项条件。不得搜索替代切片，不得把不同切面的优胜项拼成一次成功，打平不算胜出。

十个固定对照：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、3D-DIP、3D-INR、MC-INR、ZS-SSDU、GS-single。

**截至本文快照，18/18目标尚未达成。**

## 2. 工作区与运行入口

项目根目录：

```text
/data0/congcong/code/GS/3DGS
```

当前实验工作区（下文简写为 WS）：

```text
/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_losstune_20261101
```

本地训练分支：`autoresearch/losstune-nov01`。本文的 GitHub 文档分支与该本地分支分离，不切换正在运行的训练分支。

数值解释器：`/home/congcongliu/anaconda3/envs/GSTMI/bin/python`。不要根据过时的项目说明改用另一套 Conda 环境。

## 3. 已完成的 C77 与 R1 工作

此前以“最佳同一病灶切面上，平扫／增强六项指标均超过十个对照”的 S6 为目标。C77 配方已冻结，并完成 R1 独立起点重复实验。

| R1项目 | 实测结果 |
|---|---:|
| 主里程碑 | 625，alpha=1 |
| S6为正的独立起点 | 17/17 |
| 平均S6 | +0.027841 |
| 标准差 | 约0.0023 |
| 最小／最大S6 | +0.022473 / +0.033481 |
| 预设半组A均值 | +0.026997 |
| 预设半组B均值 | +0.028790 |
| 预先声明的R1规则 | PASS |
| 全脑分数 | 所有成员仍为负，均值约−0.0655 |

E4为19成员集成的次要结果：S6=+0.042168，全脑分数−0.053411，计算成本为多成员累加，不能当作单次训练结果。

证据文件：`WS/R1_RESULT.md`、`WS/runs/R1_repeat/repeat_decision.json`。

上述结论只证明同一病例上的初始化稳定性，不证明跨病例泛化；也不等于新的固定三切面18项条件已经通过。

## 4. 新的固定三切面 autoresearch

流程：提出假设 → 保存并提交具体配方 → 两个开发起点训练 → 固定评估 → 记录keep/discard → 从当前keeper继续。

- 原数值管线为 `long_chain.py`；原S6评估结果继续保留，但不控制新目标的keep/discard。
- 新评估器为 `fixed_views_score.py`，冻结文件为 `fixed_views_20261104/freeze.json`。
- 指标严格复用 `figs/fig_ce_three_planes_compact.py`：完整切片上的脑部掩膜，阈值为参考幅值体积P99.9的5%；PSNR用体积最大幅值，SSIM为7×7窗口图在掩膜内的均值；NMSE也在同一掩膜内计算。
- 显示时的裁剪和病灶放大框不定义评分区域。
- 每项对该项最强对照计算相对改善；PSNR/SSIM越大越好，NMSE越小越好。
- 候选主分数为两个开发起点42/7全部18项相对改善中的最小值。只有比当前keeper提高超过0.0001才保留。
- 成功要求两个起点的全部18项均为正，之后还需起点201/207确认与最终视觉检查。

### 已测进展

| 阶段／配方 | 最弱项分数 | 两起点中较少的正向项数 | 状态 |
|---|---:|---:|---|
| C77，新目标控制组 | −0.199355621 | 6/18 | 未达标 |
| C69，旧配方重评后起点 | −0.188640078 | 6/18 | 初始keeper |
| FV19 | −0.186470633 | 6/18 | 第一轮最优 |
| FV22 | −0.184633697 | 6/18 | 第二轮接受 |
| FV28 | −0.184197774 | 6/18 | 第二轮接受 |
| FV37 | −0.183496454 | 6/18 | 当前快照keeper |

第一轮FV01–21于11月5日07:02结束；第二轮FV22–37于11月6日01:13结束，均未达到目标。

第三轮FV38–57继续测试 DC强度、先验和优化参数。快照核查时FV38–46均已评分但未替换FV37，FV47正在运行。实时状态文件的候选名称和进度可能随后更新。

### 已知有效或无效的方向

- sigma从0.02到0.0225曾改善最弱项，但0.025/0.03未继续改善；更强平滑会损伤部分SSIM。
- 增强分支非局部权重12、全局LR0.75，以及去除增强几何／共享小波项曾被接受。
- 频繁刷新非局部先验明显有害：刷新间隔100／250曾降到0/18。当前625步路线保留单次冻结先验。
- `frequency_t=1`会把采集点频率权重展平到每个对比的RMS，因此相关权重主要改变每个对比的DC强度，不能解读为独立的频谱强调。
- `lambda_wav=0`会关闭两个对比的小波项；此时只改`lambda_wav_plain`无效。

### 中断与恢复

FV22原始尝试的seed42完成，seed7因同一卡上另一进程占用约52GB导致OOM。该次尝试记录为基础设施／训练失败，未计作科学discard。原目录保留，`FV22_nl_sigma_rel_0.0225_retry1`随后完成并被接受。

当前第三轮恢复命令（先确认没有已有控制器持有`controller.lock`）：

```bash
cd /data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_losstune_20261101
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  MPLCONFIGDIR=/tmp/mcgs-fixed-views-mpl \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -u fixed_views_round3.py
```

该入口自动恢复，无`--resume`参数。不得同时启动第二个控制器，不得在比较进行中改动已绑定的代码、评估器或冻结输入。

## 5. CSM来源与当前实际使用情况

平扫与增强分别使用各自的42通道CSM，形状均为`(42,256,232,176)`，HDF5键为`map`：

```text
平扫：/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois/sim/gs/pre_csm.h5
增强：/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois/sim/gs/post_csm.h5
```

用户在查看CSM图后明确要求用这些地图进行MC-GS重建。已核对运行配置和加载日志：这些就是当前MC-GS已经使用的地图；没有另行替换或重新估计。

**来源不能写成BART/ESPIRiT重新估计。** 原始项目记录说明：地图由MATLAB导出，具体估计算法未记录，未用`bart ecalib`重新计算。预处理脚本只是把原始HDF5地图复制为`csm.npy`。

原始数据为前瞻性欠采样、AF约5.33，没有完全采样真值；当前AF16实验的参考图来自CS-SENSE重建。原始CSM的校准参数和独立物理精度尚未确认。

### 本次核验

1. 固定三切面上，两组训练HDF5地图与对应源`csm.npy`的最大绝对差均为0。
2. 检查值均有限；脑内绝大多数体素满足`sum_c |S_c|²≈1`，偏差量级约5×10⁻⁷；极少支持边界点为0。
3. 当前仿真先用这些地图生成`y=F(S*x_reference)+noise`，再把同一个S交给各重建方法，所以地图／前向模型匹配是构造保证的。
4. 已有无噪声校验记录：平扫PSNR69.19dB，增强75.91dB；`adjoint_rel_err`约3.8×10⁻⁷。这支持实现一致性，不是对真实CSM精度的独立证明。
5. 原始测量相对残差约平扫0.18487、增强0.14969，混合了噪声、参考图误差和可能的地图误差，不能直接当作CSM误差。

因此，不能把当前MC-GS落后直接归因于CSM加载错误或仿真地图失配；也不能宣称这组地图已达到真实采集的高精度标准。若审计真实CSM，需要对原始多线圈数据做地图子空间投影残差与独立校准比较，作为另一个版本的实验。

证据：`CE_pre_post_contrast/README.md`第94行附近；`CE_retro_cssense_af16_vdpois/sim/meta.json`、`logs/check_sim.json`、`scripts/sim_cssense_af16.py`；标准方法参见[ESPIRiT原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4142121/)。

## 6. CSM可视化与图片位置

绘图脚本：`/data0/congcong/code/GS/3DGS/tools/view_ce_csm.py`。

图片目录（服务器本地路径，未随本文上传二进制文件）：

```text
/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_losstune_20261101/figs/csm_20261106
```

三视图总览完整PNG路径，可在支持本地路径识别的终端／编辑器中Ctrl+鼠标左键打开：

```text
/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_losstune_20261101/figs/csm_20261106/csm_pre_post_three_views.png
```

| 文件 | 内容 |
|---|---|
| `csm_pre_post_three_views.png` | 两组同一6个代表线圈、固定三视图、共享幅值色标；右侧为全部42通道功率和 |
| `csm_pre_all42_magnitude_axis0.png` | 平扫42线圈横断位幅值 |
| `csm_post_all42_magnitude_axis0.png` | 增强42线圈横断位幅值 |
| `csm_pre_all42_phase_axis0.png` | 平扫42线圈横断位相位 |
| `csm_post_all42_phase_axis0.png` | 增强42线圈横断位相位 |
| `csm_{pre,post}_all42_magnitude_axis{1,2}.png` | 各组冠状位／矢状位全部线圈幅值 |
| `sources_and_display.json` | 实际源文件、显示范围、线圈选择及数值检查 |

共9张PNG及对应PDF。代表线圈编号为33、36、37、21、23、24，按两组地图三视图的平均功率选择，不使用GT或病灶ROI。共享幅值上限为正值地图的P99.8。冠状／矢状位沿用既有图的脑部slab显示，相位在幅值小于0.001处隐藏。

相位有规范自由度，平扫与增强的绝对相位不能直接逐点作生理差异解释。外观平滑和功率归一化本身也不证明CSM物理准确。

## 7. 重建图的版本区别

- `ce_af16_prepost_row3planes_compact_201_130_48_inset48_headerr_C77s42.png`：11月4日绘制，但MC-GS数据是11月2日的C77 seed42，不是R1新数据。
- `...239_130_68...C77s42_v2.png`：改变了显示切片，不能作为当前固定201/130/48目标的替代。
- `ce_af16_prepost_row3planes_compact_201_130_48_inset48_headerr_R1s201.png`：后续按用户要求，用R1中位成员seed201重新绘制；PNG/PDF均在`WS/figs/`。
- `all_methods_3view_R1_s201_jet.png`和`all_methods_3view_R1_E4_jet.png`：R1单成员／E4集成的另一版三视图。

原图的整切片脑掩膜指标与旧S6病灶切面评估口径不同。不能把“R1 S6通过”解读成横断位和矢状位也全部领先；固定三切面图中仍有实际短板。

## 8. 运行纪律、证据和后续

- 不改动冻结训练器、原评估器、病例、采样mask、CSM及十个对照。
- GT仅用于评分；本实验继承参考图归一化，不宣称为新建的严格GT-free基准。
- 不允许事后去噪、oracle校正、按GT/ROI训练或更换切片。
- 同一用户最多占3张GPU，保留至少1张空GPU，0/4保留；不终止他人任务或用户保留任务。
- 新调度包装器在没有空GPU时拒绝新增分配；外部进程仍可能在分配后占卡，因此OOM必须保留记录并明确重试。
- 不提交`runs/`、`logs/`、`results.tsv`、`ledger.jsonl`中的大体积运行产物，不执行破坏性reset。新目标账本单独保存在`fixed_views_20261104/`。
- 子代理分别负责只读运行核查、评估审查和调参分析；主线程负责协调，不并行修改活跃配方／代码。

恢复上下文时先读：

```text
WS/fixed_views_20261104/status.json
WS/fixed_views_20261104/keeper.json
WS/fixed_views_20261104/results.tsv
WS/fixed_views_20261104/ledger.jsonl
WS/FIXED_VIEWS_PROGRAM.md
WS/FIXED_VIEWS_ROUND2.md
WS/FIXED_VIEWS_ROUND3.md
```

第三轮启动前共12项CPU检查通过，子代理复核未发现阻塞性问题。该检查结论只针对控制器与评估实现，不是18项科学目标达成的证据。

本工作是**单病例、GT评分的自适应开发**。仍缺跨病例／采样条件泛化证据，以及匹配先验、初始化和计算预算的GS-single消融，不能把全部收益归因于联合建模。Bayesian v2存在独立代码线，但不构成本次确定性JointTrainer实测结果。

后续继续以固定18项标准评分；达标后做确认起点和视觉检查。若队列耗尽而未达标，应明确记录失败边界并据实设计下一轮，不修改评估标准制造成功。
