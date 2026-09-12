# 设计规格 — 心脏 3D 径向数据上的 MC-GS 与全部对手臂（第二阶段）

- **日期：** 2026-09-12
- **状态：** 设计（§1–§8）已在对话中经用户确认（"按你的来"），本文待用户审阅；审阅通过后用 writing-plans 出实施计划
- **分支：** `cardiac-phase2`，自 `handoff/2026-09-12-cardiac` 开出，已合并 `autoresearch-t1t2`@39cae91（v3.2 配方与小波先验，`affc488`）
- **worktree：** `/data0/congcong/code/GS/mcgs_cardiac_wt`
- **产生方式：** `superpowers:brainstorming`，前置两轮 grill（决定 Q1–Q31）与一次仓库探索（对手臂清单）
- **前置：** 第一阶段 spec `2026-09-10-cardiac-freerunning-phase1-design.md`（七处修订）、课题 README
  `/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/README.md`

---

## 0. 目标与范围

在 V07、V03、V04、V08 的自由运行多回波 3D 径向数据上，用第一阶段的参考与口径，跑通并比较 MC-GS 项目涉及的全部方法：
ZF、CG-SENSE、CS-SENSE（200 轮收敛版）、Joint SENSE（矢量 TV）、L1-ESPIRiT、ZS-SSDU、3D-DIP、3D-INR、GS-single、MC-GS（v3.2 配方）。

已定的边界（grill 决定）：
- 回波 0（TE 1.25 ms，信噪比最高，水脂 97° 正交相位）与回波 4（TE 9.45 ms，最接近水脂同相，16°）。单对比度方法逐回波各跑一次；Joint SENSE 与 MC-GS 联合两个回波。
- AF 12 四例全做；AF 8 与 16 只在 V07 上做（趋势展示）。所有方法共用第一阶段的子集与留出编号（`V07/gauge_split.npz`）；V03/V04/V08 的 AF 12 子集与留出集用同一规则、同一种子从各自 REF 辐条里抽取（新增 `s09_gauge_split.py <Vxx>` 调用）。
- V07 上调参并冻结，V03、V04、V08 原样迁移作留出受试者（V03 门控弱、V04 冠状 slab，表里带脚注）。
- 心动平均（同 REF）；不做 5D；不做 8 回波推广；P-LORAKS 不做（工具箱是笛卡尔逐层算法，径向无对应物，脑部也只是评审补充）。
- 主口径 = 留出 k 空间 NRMSE；图像指标与误差图为辅，门沿用第一阶段修订 6。

## 1. 隔离与分支（用户要求：不干扰 mcgs 项目其它代码与参数）

1. 全部改动只在 `cardiac-phase2` 分支、同一 worktree。主树 `Paired_ARC_GS_general_framework`、`mcgs_af16_wt`、`mcgs_af12_wt`、脑部课题的 configs 与归档一律不碰。
2. 改动是加法：新文件 + 配置项。训练器新增 `data.<channel>.forward: radial`（默认 `cartesian`），未设置时代码路径逐字节不变。
3. 各方法在脑部课题里的默认超参数不动；心脏用独立配置 `configs/cardiac_*.yaml` 与课题脚本。
4. "没干扰"的验收：合并与改动前后，用 `configs/t1t2_af12_hyw.yaml` 跑 `WARMUP=20`、200 步的短训练（同一 GPU、同一种子、`training.deterministic`），导出体逐位相同（`np.array_equal`）；`tests/` 全部通过（当前 183 + 34）。这项在实施计划的第一个任务和最后一个任务各做一次。
5. 归档：新课题目录 `Results_20260908/Cardiac_FreeRunning_MCGS/`（README、scripts、logs、figures、recon），`data/` 软链接回第一阶段课题的派生数据（`raw.h5`、`common/traj_r.npy`、`sens2x`、`ref_ids`、`gauge_split.npz`），仓库内镜像 `cardiac_freerunning_mcgs/`。

## 2. 公共前向算子、ZF 与一致性测试

- 新模块 `paired_gs/forward/radial.py`（新文件，`paired_gs` 内其它文件不改）：
  ```
  RadialForward(traj_bart: (3, M) float, sens: (C, 224, 224, 224) complex64, device)
    A(x)      -> (C, M)      x: (224, 224, 224) complex; torchkbnufft KbNufft，按线圈分块（默认 4）
    AH(y)     -> (224,)^3    伴随，含线圈合并（conj(sens)）
    normal(x) -> (224,)^3    Toeplitz 正规算子（calc_toeplitz_kernel，一次预计算）
    kabs      -> (M,)        每样本 |k|（BART 单位），供频率加权
  ```
  坐标 = BART 单位 × 2π/224（弧度）。线圈图一律用第一阶段的 `sens2x`（224³，归一化后）。
- 标度约定：以 BART `nufft` 为准。`RadialForward` 内乘一个固定标量 `s_bart`，使 `A_torch(x) ≈ A_bart(x)`；`s_bart` 由一致性测试在 16³ 体模上量出（BART 的 NUFFT 按酉 FFT 归一，见第一阶段修订 5），写死在模块里并有测试守着。这样 λ、`-w`、留出口径与第一阶段同义。
- 一致性测试（`tests/test_radial_forward.py`，CPU 可跑，GPU 可选）：
  1. 16³ 体模、`traj -x 16 -o 2` 轨迹：`A_torch` 对 `A_bart` 的 NRMSE < 1e-2，`AH_torch` 对 `nufft -a` 同样；
  2. `normal(x)` 对 `AH(A(x))` NRMSE < 1e-3（Toeplitz 与直算一致）；
  3. 真实数据（V07 REF 图像、留出轨迹）：两条前向的留出预测 NRMSE < 1e-2，且用 `A_torch` 算出的 REF 样本内留出 NRMSE 与第一阶段的 0.1319（回波 0）差 < 1%。
- ZF 的定义（所有学习型方法的初始化 + 表里的 ZF 臂）：密度补偿伴随。3D 径向解析 DCF w(k) ∝ |k|²（中心样本取相邻值的一半），`ZF = AH(w · y)`，再按 REF 的 P99.9 归一以便显示；留出 NRMSE 直接对 ZF 体算。
- 每样本 |k| 用于训练器的 k 空间频率加权（`kspace_high_weight` 等）：把脑部按笛卡尔 k 半径分段的规则原样套在 `kabs / kmax` 上。

## 3. MC-GS 与 GS-single：训练器改动点与心脏配置

按 09-12 仓库探索列出的笛卡尔绑定点逐处加 `radial` 分支（都在 `paired_gs/trainers/joint_trainer.py` 及其数据类），默认路径不动：

| 位置 | 现状 | radial 分支 |
|---|---|---|
| `_load_dataset`（:60–74） | `MRIDataset`，笛卡尔 h5 + 掩膜 | 新 `RadialDataset`（新文件 `paired_gs/data/radial_dataset.py`）：读 BART cfl 的 `ksp_<set>_e<k>`、`traj_<set>`、`sens2x`；无掩膜；`presampled`/`known_zero`/`prewhiten` 路径跳过 |
| 零填充初始化（:167–168） | `ifft3c(kspace)` | §2 的 ZF |
| `_kspace_loss`（:497–505） | `fft3c(vol×csm)×mask` | `self.A(vol)`，频率加权按 `kabs` |
| 致密化能量反投影（:649–655） | `m·fft3c(...)−y` 再 `ifft3c` | `AH(A(vol) − y)` |
| 导出（:1040–1041，`hard_dc_combine`） | 采样格点用测量覆盖 | 不做数据一致性，导出原始渲染的 224³ 复数体（数据单位），文件 `recon/<方法>_<Vxx>_e<k>_af<AF>.npy` |
| 导出指标（:1068–1070） | 对 GT 的指标 | 只记日志；正式分数由 §5 的统一脚本算 |
| `arc/cycle.py` 的硬 DC 评分（:224–244） | 笛卡尔 | 新驱动 `scripts/run_gs_cardiac.py`，不走 `cycle.py` 的评分 |

- GS-single = 同一训练器两通道喂同一回波（脑部做法）。MC-GS = 回波 0 + 回波 4。
- 起始配方 = v3.2（`autoresearch-t1t2` 记录）：
  ```
  --iters 4000 --warmup 500 --max_radius 20 --kspace_high 2.2 --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04
  --lambda_geo 0.07 --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn --set loss.edge_tv_kappa_source=zf
  --set gaussian.initial_num_points=2400000 --set loss.lambda_wav=0.05 --set loss.lambda_wav_plain=0.06
  ```
  心脏的差别：高斯覆盖 448 mm（体素数 ×8），`initial_num_points` 起点 = 2.4M × (448 mm 网格内的体内体积 / 脑部支撑体积)，在 V07 上按 §4 的调参规则扫描；`init_mode=head` 的支撑取 §2 ZF 幅值的 5% 阈值（第一阶段的 body 掩膜定义）。
- 显存：224³ 体 + 24 线圈的前向约 10 GB（实测），高斯 2.4M 点在脑部约 32 GB；总计以一张 80 GB 卡为限，超出则减点数并记录。

## 4. 对手臂

| 方法 | 实现路线 | 超参数扫描（V07，AF 12，每回波各选） | 预算 |
|---|---|---|---|
| ZF | §2 | 无 | 秒级 |
| CG-SENSE | 第一阶段 BART `pics -l2`，λ = 1e-3 × λmax（共用 REF 的 λmax，第一阶段约定） | 无（已冻结） | 已有 V07 全部；三例 AF 12 各 2 次 |
| L1-ESPIRiT | 第一阶段 BART `pics -l1 -e`，绝对 λ | 网格 {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}（V07 已选 3e-3；回波 4 另选一次） | 每次约 25 分钟 |
| CS-SENSE（TV） | BART `pics -R T:7:0:λ`，ADMM，200 步；V07 上做收敛核对（100 对 200 步体内差 < 1%，不过就加步） | λ 网格与脑部同形：{0.5, 0.7, 1.0, 1.4, 2.0} × λ0，λ0 由 BART 标度下的一次粗扫定 | CPU |
| Joint SENSE | BART：两回波叠在 TE 维、共用轨迹与 `sens2x`，`pics -R T:7:32:λ`（联合阈值跨回波）。**前置核对**：16³ 体模上 BART 的联合 TV 与 `recon_joint_cssense.py` 的矢量 TV 定义一致（同 λ 结果 NRMSE < 1e-2），不一致则改接 torch 算子重写矢量 TV | 同 CS-SENSE 网格，独立扫 | CPU |
| 3D-DIP | 移植 `T1T2_AF12_retro/scripts/recon_dip_inr.py --method dip`，前向换 `RadialForward`，输出 224³；早停用从各子集自身辐条另抽 5%（seed 1），口径留出集（seed 0）永不参与 | 步数：V07 先跑脑部预算 5000 步看留出曲线，再定三例预算；网络参数沿用脑部（base 64、levels 4、in_ch 32） | 约 5 小时/次 |
| 3D-INR | 同上 `--method inr`，hash-grid 沿用脑部（levels 16、log2T 19、feat 2、nmax 512、hidden 64、layers 3） | 步数：V07 先跑 1500 步看曲线 | 约 1.5 小时/次 |
| ZS-SSDU | 只做 V07 两回波、AF 12。3D 网络 + Toeplitz-CG 数据一致性（CG 4 步）、5 次展开；Θ/Λ 按辐条 80/20 重抽、Γ 10% 辐条早停。**前置可行性小试**：224³ 上 ResNet 通道数与梯度检查点，确定放得下 80 GB 的配置并记录 | 步数由小试与曲线定，目标 ≤ 8 小时/次 | GPU |
| GS-single | §3，两通道同一回波 | 与 MC-GS 同一扫描规则 | 约 4 小时/次 |
| MC-GS | §3 | 见下 | 约 4 小时/次 |

调参规则（所有臂一致）：
- 在 V07、AF 12 上做，选择准则 = 中心 112³ 体内 PSNR 对 REF（第一阶段 L1 的做法），每回波各选一次；留出集不参与选择。
- 每臂的扫描范围与次数在实施计划里写死，MC-GS 的扫描预算与对手臂相当（不超过对手臂里最多的那一个），扫描记录进 `logs/sweeps.tsv`。
- MC-GS 的可调项限定为：`initial_num_points`、`lambda_wav`（两通道）、`lambda_geo`、`kspace_high`；其余沿用 v3.2。
- 选好后四例冻结（`params_frozen_phase2.json`），V03/V04/V08 不再调。

## 5. 评分、表与图

- 统一导出：每个方法每次重建写 `recon/<方法>_<Vxx>_e<k>_af<AF>.npy`（224³ 复数，数据单位，BART 标度）。
- `scripts/score_all.py`：对每个文件用 `sens2x` + BART `nufft` 算留出 NRMSE（主口径；与第一阶段 `common.heldout_nrmse` 同一实现），中心 112³ 算体内 PSNR/SSIM/HFEN、心脏框 PSNR/SSIM（V07 用第一阶段的框；其余三例的框在各自 REF 上放好后写进 `gauge.json`），误差图 jet [0, 0.1]。训练器自己算的指标只作日志。
- 表：`logs/summary.tsv`（方法 × 例 × 回波 × AF），按留出 NRMSE 排名，主表 = AF 12 四例。
- 图：`s12_compare_figure.py` 扩到全部方法（每图一例一回波一 AF，三切面 + 误差图）；最终图沿用脑部课题的一行版式（`fig_row_final.py` 的顺序：REF | CG-SENSE | CS-SENSE | Joint SENSE | L1-ESPIRiT | ZS-SSDU | 3D-DIP | 3D-INR | GS-single | MC-GS，误差图在下）。
- 门：只对留出 NRMSE 要求方法内随 AF 单调（V07）；第一阶段修订 6 的说明沿用。

## 6. 顺序与时间

1. 前向算子 + 测试 + ZF + 评分脚本 + "没干扰"验收（1–2 天）。
2. MC-GS 与 GS-single 在 V07 跑通、调参（2–3 天；每次 4000 步约 4 小时）。
3. BART 一族：CS-SENSE、Joint SENSE 的核对与扫描，L1 回波 4 的扫描（1 天 CPU）。
4. 3D-DIP、3D-INR（每次 1.5–5 小时）。
5. ZS-SSDU 可行性小试后定。
6. 三例留出迁移（GPU 约 4–6 天）。
7. 表、图、README、交接。
每档完成即入表出图、提交。总计约 3–4 周。GPU：开跑前查 `nvidia-smi`，只用没有训练进程的卡，一卡一训练（09-11 两次卡死的教训）；grill 时定的 GPU 5 已被 AF16 循环占用，届时重选，没有空卡就报告。CPU 任务沿用 nice 10、32 线程、最多两个并行。

## 7. 验收与停止规则

验收：
1. 默认路径逐位不变（§1.4）且 `tests/` 全过；
2. 前向一致性三项测试通过（§2）；
3. 每个方法在 V07 AF 12 两回波上留出 NRMSE 低于 ZF；MC-GS 在 V07 上留出 NRMSE 随 AF 单调；
4. 四例 AF 12 主表齐全（ZS-SSDU 仅 V07，表里注明），MC-GS 与 GS-single 的对照四例都有；
5. 图、README、交接文档、tag。

停止规则（任一触发就停下来报告，不绕过）：一致性测试不过；脑部配置输出变化；某方法留出 NRMSE 高于 ZF；GPU 上出现第二个训练进程；显存放不下且减配后仍放不下；对手臂的定义核对（Joint SENSE）不一致且 torch 路线也对不上脑部定义。

## 8. 风险与已知限制

- 每步比脑部慢约 30 倍（NUFFT 前向 + 反传 3.2 s），预算要减或等；DIP 在脑部本就未收敛（报下界），心脏同样。
- 448 mm 视野让高斯数与显存翻倍，可能要减点数，与脑部配方不完全同构。
- ZS-SSDU 的 3D 版本显存可能放不下，只在 V07 做，缩减配置要记清楚。
- 对手臂在噪声参考上的图像指标不可靠（第一阶段修订 6），排名靠留出口径。
- V03 门控弱、V04 冠状 slab（自导航辐条沿 AP），作留出例要带脚注；三例的心脏框要各自放。
- 训练器多处笛卡尔绑定，遗漏一处就会在 radial 路径上悄悄用错算子：实施计划要求每处改动配一个最小测试或断言（`radial` 路径下 `mask` 必须为 None）。

## 9. 交给实施计划的接口

第一阶段产物：`raw.h5`、`common/traj_r.npy`、`sens2x`、`ref_ids.npy`、`gauge_split.npz`（V07）、`params_frozen.json`、`gauge.py`、`bart.py`、`common.py`。本阶段新增：`paired_gs/forward/radial.py`、`paired_gs/data/radial_dataset.py`、`configs/cardiac_*.yaml`、课题脚本（`run_gs_cardiac.py`、`recon_*_radial.py`、`score_all.py`、`s09_gauge_split.py <Vxx>`）、测试 `tests/test_radial_forward.py` 与训练器 radial 路径的断言测试。
