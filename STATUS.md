# STATUS — 2026-09-16 21:10（worktree `mcgs_hparam_wt`，分支 `hparam-sensitivity`）— 超参数敏感性研究已完成

## 1. 已完成且已验证
- 六个运行全部 rc=0（`results_arc/hp_runs.log`：n18 19:22、n36 19:33、n09 20:11、p05 20:46、p10 20:49、n72 20:56），
  独占 A100 上每个 47–82 min。汇总表与图：

```
$ env PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/hp_summarize.py | tail -9
Deltas against hp_n36 (head PSNR T1 / FLAIR, dB):
- hp_n09: -0.349 / -0.793
- hp_n18: -0.114 / -0.264
- hp_n72: -0.036 / +0.052
- hp_p05: +0.000 / +0.005
- hp_p10: +0.002 / +0.010

Anchor hp_n36 vs registered gs_joint_p3 35.706 / 34.928: delta 0.000 / 0.000 dB, tolerance 0.02 -> PASS
```
  头内 PSNR T1 / FLAIR：0.9M 35.357 / 34.135，1.8M 35.592 / 34.664，3.6M 35.706 / 34.928，7.2M 35.670 / 34.980；
  5% 35.706 / 34.933，10% 35.708 / 34.938。完整表 `results_arc/hp_summary.md`（课题 `logs/` 有副本）。
- 课题 `Results_20260908/T1FLAIR_AF16_hparam_v3_jiangyizhou/`（README 中文、scripts、logs、mcgs 打分 json、figures），仓库镜像
  `t1flair_af16_hparam_v3_jiangyizhou/`；记录 `docs/superpowers/records/2026-09-16-hparam-sensitivity.md`；根 README 已加一行。
- 图 `figures/hp_sensitivity_psnr_ssim.{png,pdf}`，`http://localhost:8765/t1flair_v3_af16_hparam/hp_sensitivity_psnr_ssim.png`。
- 单元测试 `tests/test_hp_summarize.py` 3 通过。

## 2. 待用户拍板
- 结论怎么写进论文：第一组是"取舍不是饱和"（T1 在 3.6M 最高、FLAIR 随容量继续涨，与三期 4.8M/6.0M 一致）；第二组 null。
  spec §8 的判据照字面得出这两句，措辞由用户定。
- 要不要扩展：第二受试者 huangyuwei（掩膜同一张 md5）、R=6 新掩膜、病灶数据上的 CNR——都不在本 spec 内，没跑。
- 六个 0.71 GB 体积留在 `results_arc/`，要不要打包进课题 `export/`。

## 3. 下一步
无待跑作业。两个运行器在训练全部退出后自行退出（见第 4 节若有例外）。

## 4. 坑与"别重做什么"
- anchor 逐位复现（0.000 / 0.000），说明 5ea6189 与三期尖端的训练/打分路径一致，不必再查漂移。
- `TORCH_EXTENSIONS_DIR` 无效（扩展编进 worktree 的 `paired_gs/gaussian/build/`）；`.gpulock.<卡>` 与其他三个调度器共用；
  两个运行器同时消费一份队列靠"有输出就跳过"，别再起第三个。
- 冒烟 `hp_smoke_*` 不是结果。AF24 日志里 6–7.5 h/run 是共卡数字，别拿来估算。

## 5. 关键决定的"为什么"
- N_s 网格以 3.6M 为中心（设计的 50k–400k 是通用 3DGS 直觉，比本体积的工作点低 9–72 倍，照跑只能看到欠拟合区）。
- 主配方 N_p=0（发表结果全是 0；私有集是测过判 null 的可选杠杆），所以第二组是"往上加"。
- AF16 而非 R=6：图 of record 在 AF16，R=6 各法近饱和且要新掩膜。
- 无 Lesion CNR：三个受试者无病灶；换数据会让两组不在同一数据上。

下面是起跑前后的过程记录（同一天，保留）。

---

# STATUS — 2026-09-16 17:30（worktree `mcgs_hparam_wt`，分支 `hparam-sensitivity`，自 `origin/autoresearch-jyz-af24` 5ea6189 切出）

## 超参数敏感性研究（本分支）：分支已建、设计已对照代码，一步实验没跑，等用户拍板

用户 09-16 要求：基于 GitHub 上的 mcgs 仓库新建分支，理解一份两组实验的设计（共享 Gaussian 数量敏感性 +
私有 Gaussian 比例），千万不要干扰其他实验。本节只有事实与待定项，没有结果。

### 收到的设计（要点）
- 第一组：固定其他一切，只改共享 Gaussian 数 N_s ∈ {50k, 100k, 200k, 400k}；同一加速倍数（原文"例如 R=6"）、
  同一泊松盘掩膜、同一批病例、同一训练设置；报 PSNR / SSIM / NMSE；回答"是否依赖特定数量、性能是否随容量饱和"
  （期望形态 100k < 200k ≈ 400k）。
- 第二组：固定 N_s = 200k，改 N_p/N_s ∈ {0, 5%, 10%}；0% = 完全去掉私有分支（兼消融）；报 PSNR / SSIM / Lesion CNR，
  理由是私有 Gaussian 的作用是保住对比度特异信息与病灶，不是抬全局 PSNR。
- 最终只放两张小曲线：N_s → PSNR/SSIM；N_p/N_s → PSNR/Lesion CNR。不做 5×5 网格，不在所有加速倍数上重复。

### 对照代码（本分支 5ea6189 已核实）
| 设计里的量 | 代码里的旋钮 | 现状 |
|---|---|---|
| N_s 共享 Gaussian 数 | `--set gaussian.initial_num_points=N`。模型 `paired_gs/gaussian/joint_gaussian_model.py`：共享几何（位置/尺度/旋转）+ 每个 Gaussian 的双复数密度 α_plain→T1、α_contrast→FLAIR | 冻结配方 = **3.6M**（`run_af24_mcgs.sh`）。`adaptive_control.enable` 未开（配方 yaml 无该节，`joint_trainer.py:672`），训练中点数不变（AF24 日志 `N=3600000`） |
| N_p 私有 Gaussian 数 | `--set gaussian.num_private_points=N`（`paired_gs/trainers/joint_trainer.py:304` → `add_private_residual`） | 单独一套 Gaussian，**只渲染进 contrast 通道（FLAIR）**，几何与幅值自有，只受 FLAIR 的 DC 项训练；初始化 = 共享云的等步长子集、幅值 ×0.1（或 `gaussian.private_warm_start_checkpoint`）。冻结配方 **N_p = 0** |
| R=6 泊松盘掩膜 | 房屋生成器 `vdPoisMex(256,176,256,176, accelx, 4, 24, 0, 1)`（MATLAB R2021a，`addpath /data0/congcong/code/ESPIRiT/utils`），按 time(0) 播种、不可再生、必须存盘；抽池 + 定点脚本 `t1flair_af24_v3_jiangyizhou/scripts/prep_02_mask_pool_af24.m` / `prep_03_pick_mask_af24.py` | 现有课题掩膜只有 AF12/16/20/24（accelx 3 / 4 / 5 / 6.8656，accely 4）。R=6 要新抽：accelx 1.5，或按 AF24 的办法抽池取 45056/6 = 7509 点。BART `poisson` 也在（`~/src/bart/bart`），但换了生成器家族 |
| PSNR / SSIM / NMSE | `arc/cycle.py` 的打分 json 每个对比度已有 `head_psnr / head_ssim / head_hfen / nmse / psnr / ssim`（主表口径 = 头内） | 直接可用 |
| Lesion CNR | `paired_gs/metrics/lesion_metrics.py`：CNR = \|mean(lesion) − mean(bg)\| / std(bg)；ROI 自动检测按 \|post\| − \|pre\| 差分（增强扫描专用） | **需要病灶 ROI**。T1+FLAIR 的 16 对 raw（`/data1/data1_congcong/data/3D/{T1,FLAIR}_csm_ksp`）全是志愿者，无病灶 |
| 同一批病例 | 已建课题的 T1+FLAIR 受试者：jiangyizhou（AF12/16/20/24）、huangyuwei（`T1FLAIR_AF12_retro` / `AF16_retro`）、tangbing（`T1FLAIR_AF12_v4_tangbing`）；其余 13 对只有 raw，要先做 ESPIRiT 图与课题目录 | 一次 4000 步 3.6M 运行 ≈ 6–7.5 h、33 GB（AF24 日志 elapsed_s 21.4k–27.2k），6.0M 再贵约 50% |

### 已有数据可复用（AF16 课题 jiangyizhou，头内 PSNR T1 / FLAIR）
- 容量（三期冻结配方，4000 步）：3.6M 35.706 / 34.928；4.8M 35.697 / 34.962；6.0M 35.675 / 34.977
  （`docs/superpowers/records/2026-09-13-autoresearch-jyz-af16-phase3.md` 容量表；判定"容量是取舍不是 null"）。
  P6 配方 4000 步：2.4M 35.434 / 34.605 对 3.6M 35.397 / 34.664（`autoresearch_jyz_af16.md` Promotion 3）。一期还有 0.9M / 1.2M 点（旧配方，不可直接并表）。
- 私有：三期 base B（1500 步）+ 0.3M FLAIR 私有（= 3.6M 的 8.3%）：+0.005 / +0.015 dB，其余格不动，判 null
  （`autoresearch_af16_p3.md` 21:46 行 `p3_r2_priv03`）。

### 设计与现状的四处不合（待用户拍板）
1. **N_s 量级**：设计的 50k–400k 比冻结配方 3.6M 低 9–72 倍。体积 256×256×176 = 11.5M 体素，3.6M ≈ 0.31 个/体素，
   200k ≈ 0.017 个/体素；本线从未跑过低于 0.6M（hyw 旧配置的"fresh cloud"）。照字面跑 = 研究严重欠拟合区，
   回答不了"容量—性能平衡"。建议以配方为中心、2 倍步长：{0.9M, 1.8M, 3.6M, 7.2M}（配方在第三档，对应设计里的"200k"）；
   或者沿用已有的 {2.4M, 3.6M, 4.8M, 6.0M}，在 AF16 上只补 2.4M 一次（三期配方）。
2. **私有分支的地位**：发表的 MC-GS（AF12/16/20/24 全部结果）N_p = 0；私有集是一个已测判 null 的可选杠杆，不是模型的组成部分。
   第二组"0% = 去掉私有分支"在现状下是反的：0% 才是主推配方，5% / 10% 是往上加。论文若把私有分支写成方法的一部分，
   主配方得改成带私有集（所有主结果重跑）；若不写，第二组就是"加私有集有没有用"的附加消融。另外私有集只在 FLAIR 侧，不对称。
3. **加速倍数**：设计写的是"例如 R=6"。本线主结果在 AF12–24，图 of record 在 AF16；R=6 下各方法都近饱和，容量差会被压小，
   还要新抽掩膜。建议在 AF16 做：上面的 4 个容量点与 1 个私有点直接复用，经典臂不必重跑（只跑 MC-GS）。
4. **Lesion CNR**：T1+FLAIR 三个受试者无病灶。可选：(a) 换到有病灶的数据 — 儿童 TSC MTP（有参考、有 lesion B 层，
   但对比度是 T1w/T2s，且 MC-GS 在那不领先）或 CE 平扫/增强（有病灶、无 GT，PSNR/SSIM 只能用留出 k 空间）；
   (b) 在 jyz 上换成"对比度特异结构"指标（如 FLAIR 的 CSF/白质 CNR、两对比度不一致区域的误差）；(c) 第二组只报 PSNR/SSIM/NMSE。

### 预算（每个受试者，4000 步）
第一组 4 档 + 第二组 2 档（0% 与第一组共用）= 6 次运行 ≈ 40 GPU·h（7.2M 档更贵、更占显存）；三个受试者 ≈ 120 GPU·h。
09-16 17:00 空卡为 1、2；一卡一训练。

### 别碰的东西（本分支的纪律）
- 其他 worktree 与它们的卡：`mcgs_peds_wt`（卡 0、6）、`mcgs_gauss_wt`（卡 3、4、7，那是 Gaussian **掩膜**调参循环，
  与本分支的 Gaussian **数量**无关，别混）、卡 5 是别的用户。本分支只用空卡，跑前 `nvidia-smi --query-compute-apps`。
- `run_af24_mcgs.sh` 的 `TORCH_EXTENSIONS_DIR` 写死指向 `mcgs_af16_wt.torch_ext`；本分支起跑脚本要改成自己的目录，
  避免与 AF24 worktree 抢编译缓存。
- 本 worktree 的 `results_arc/` 为空（gitignored），任何输出只写这里；不写别的课题目录，不动 `Results_20260908/` 已有课题。

### 用户拍板（2026-09-16，"同意"四条建议）
N_s 取 0.9M / 1.8M / 3.6M / 7.2M；发表的 MC-GS 保持 N_p = 0，第二组在 3.6M 上加 5% / 10% FLAIR 私有集（0% 与 3.6M 档同一次运行）；
在 AF16 现有房屋掩膜上做；不报 Lesion CNR（第二组报 PSNR / SSIM / NMSE）；先跑 jiangyizhou。
spec `docs/superpowers/specs/2026-09-16-hparam-sensitivity-design.md`，plan `docs/superpowers/plans/2026-09-16-hparam-sensitivity.md`。

### 队列已起跑（2026-09-16 18:27，plan Task 4）
- 运行器 `scripts/hp_queue.sh scripts/hp_queue.txt`（`setsid nohup`，日志 `results_arc/hp_queue.log`）；六个运行按
  hp_n36 → hp_n18 → hp_n09 → hp_n72 → hp_p05 → hp_p10 排队（`--set gaussian.initial_num_points` / `gaussian.num_private_points`
  是各运行之间唯一的差别，配方块与 `run_af24_mcgs.sh` 逐字相同，已 diff 核对），只用空卡（默认卡 1、2），一卡一训练，
  持有 `Cardiac_FreeRunning_MCGS/work/.gpulock.<卡>`。实测速度约 1 it/s（hp_n36 150 步用时 3.5 min 含初始化；gauss 循环同规模 4000 步 67 min），每个运行约 1.2–1.5 h（7.2M 更久），两张卡全部约 7–9 h；早先按 AF24 日志估的 6–7.5 h/run 是那批运行与别的任务共卡时的数字。hp_n36 于 18:27 在卡 1 起跑，hp_n18 于 18:28 在卡 2 起跑（两把锁的属主都活着，`nvidia-smi` 上各一个训练进程）。
- 暂停：`touch results_arc/.hp_queue_pause`；停运行器：`pkill -f "scripts/hp_queue.sh"`（训练进程独立，不会跟着死）；停一个训练：
  `pkill -f "cycle.py --exp hp_n72"`（之后删掉它的 `results_arc/hp_n72*` 才会重排）。
- 进度：`tail -2 results_arc/hp_runs.log`；汇总：`env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/hp_summarize.py`
  （写 `results_arc/hp_summary.{md,csv}`），先看 hp_n36 的 anchor 行是否 PASS（对注册的 gs_joint_p3 35.706 / 34.928 容差 0.02 dB）。
- 冒烟 `hp_smoke_n09` / `hp_smoke_p05`（60 步，18:21 / 18:26 都 rc=0；掩膜读到 15.19×，`N=900000`，私有集
  "attached 45000 private residual Gaussians"）只是可行性检查，不是结果，别引用。
- 单元测试 `tests/test_hp_summarize.py` 3 通过（无需 GPU）。
- 坑：`TORCH_EXTENSIONS_DIR` 对这套代码无效，体素化 CUDA 扩展按源码哈希编进各 worktree 自己的 `paired_gs/gaussian/build/`，
  worktree 之间不共享缓存（所以不会互相干扰）；`run_hp_mcgs.sh` 里那个变量留着无害。

### 19:40 进展（用户"继续"）
- **anchor 门 PASS**：hp_n36 = 35.706 / 34.928，对注册 gs_joint_p3 差 0.000 / 0.000 dB（同种子同代码路径，逐位复现）。
- hp_n18 = 35.592 / 34.664（对 3.6M −0.114 / −0.264）。hp_n09（卡 2）、hp_n72（卡 1）在跑。
- 19:39 在空出来的卡 3、4 上又起了一个运行器（`ALLOWED="3 4"`，日志 `results_arc/hp_queue_b.log`），hp_p05 / hp_p10 立即开跑，
  不必等卡 1、2；两个运行器靠同一套"有输出就跳过"规则与卡锁互不重复。起它之前核对：gauss 队列 0 个待跑、未暂停，卡 3/4 无进程无锁。
- 实测每个 3.6M 运行 66 min、1.8M 53 min；预计 21:45 前全部落地（7.2M 最慢）。
- 课题目录骨架已建：`Results_20260908/T1FLAIR_AF16_hparam_v3_jiangyizhou/`（data/ 五个符号链接指向 AF16 课题，掩膜 md5 d08179b8…）。

### 跑完以后（plan Task 5）
汇总表 → anchor 门 → 双面板图（先读 dataviz skill）→ 课题目录 `Results_20260908/T1FLAIR_AF16_hparam_v3_jiangyizhou/` → 记录
`docs/superpowers/records/2026-09-16-hparam-sensitivity.md` → 提交、推送、打 tag `handoff/<日期>-hparam`。

下面是 autoresearch-jyz-af24 分支的 STATUS 原文（本分支的起点）。

---

# STATUS — 2026-09-15 早（worktree `mcgs_af16_wt`，分支 `autoresearch-jyz-af24`，从 `jyz-af20-results` 4d13197 切出）

## AF24（本分支）：已全部跑完（09-14 21:45 启动，09-15 05:30 finalize 完成；09-15 下午补 L1-SPIRiT、MC-INR 与掩膜抽样方差）

先读英文记录 `docs/superpowers/records/2026-09-15-af24-jyz.md`，交接 `docs/handoff/2026-09-15-af24-jyz.md`；
课题 `/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF24_v3_jiangyizhou/`（仓库镜像 `t1flair_af24_v3_jiangyizhou/`，README 有完整表）。

### 0. 09-15 上午追加（用户回复 "continue"，做可选后续的前两项；MC-INR 与改旧标签留给用户）

**① 四档跨加速倍数复核：已完成。** 新课题 `Results_20260908/T1FLAIR_AF12_16_20_24_trend_v3_jiangyizhou/`（仓库镜像 `t1flair_af12_16_20_24_trend_v3_jiangyizhou/`，README 中文）。三档课题不动。
94 个存盘体积同一打分器重打分，原表复现（最大差 头内 PSNR 3.1e-5 dB）；AF12/16/20 的 72 个与三档复核逐位相同。除 3D-DIP 的 AF16→AF20 外，所有方法头内 PSNR/SSIM/HFEN 每一步都变差，AF24 这一步没有新增例外；展示层与全部层同样成立（ZS-SSDU 这一步在不少层略回升）。

```
$ T=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_16_20_24_trend_v3_jiangyizhou
$ sed -n '/MC-GS margin over the strongest/,/^$/p' $T/logs/af_trend_summary.txt
=== MC-GS margin over the strongest competing arm, head PSNR (one scorer) ===
  T1    AF12 (11.71x)  MC-GS 37.015  strongest competing 36.426 (L1-ESPIRiT)  margin +0.589   GS-single +0.439   coupling +0.150
  T1    AF16 (15.19x)  MC-GS 35.706  strongest competing 35.209 (L1-ESPIRiT)  margin +0.497   GS-single +0.349   coupling +0.148
  T1    AF20 (18.56x)  MC-GS 34.832  strongest competing 34.371 (L1-ESPIRiT)  margin +0.461   GS-single +0.296   coupling +0.165
  T1    AF24 (24.00x)  MC-GS 33.767  strongest competing 33.401 (L1-ESPIRiT)  margin +0.366   GS-single +0.187   coupling +0.179
  FLAIR AF12 (12.86x)  MC-GS 36.360  strongest competing 35.757 (L1-ESPIRiT)  margin +0.602   GS-single +0.567   coupling +0.036
  FLAIR AF16 (16.49x)  MC-GS 34.928  strongest competing 34.503 (L1-ESPIRiT)  margin +0.425   GS-single +0.383   coupling +0.042
  FLAIR AF20 (19.96x)  MC-GS 34.179  strongest competing 33.688 (L1-ESPIRiT)  margin +0.491   GS-single +0.439   coupling +0.052
  FLAIR AF24 (25.51x)  MC-GS 33.182  strongest competing 32.782 (L1-ESPIRiT)  margin +0.400   GS-single +0.329   coupling +0.071
```
复现：`bash $T/scripts/run_rescore.sh`（8 个 CPU 作业约 35 分钟）→ `analyze_af_trend.py` → `run_slice_trend.sh` → `slice_summary.py`（README 第 5 节）。图 `figures/jyz_t1flair_af12_16_20_24_metric_steps.png`，`http://localhost:8765/t1flair_v3_af_trend/`。

**② AF24 掩膜抽样方差：09-15 10:21 启动，运行中。** 规则与注册掩膜相同：第二个池子（`prep_02b_mask_pool_draws_af24.m`，67 张）里编号最小的三张恰好 1877 点、未裁剪的抽样 → b/c/d（FLAIR 25.43/25.41/25.50×，与 a 共享 540–572 点；`logs/prep_03b_pick_export_draws_af24.json`）。每张只跑 MC-GS joint（冻结配方，`MASK_DRAW=<d>`，不存体积）与 L1-ESPIRiT 三个 λ（T1 0.0007/0.001/0.0015，FLAIR 0.0014/0.002/0.003）。队列 `scripts/run_all_af24.py --study draws --max_gpu_jobs 3 --max_cpu_jobs 3`（调度器 pid 2796954，卡 6/7/3），跑完自动 `scripts/score_mask_draws_af24.py` → `logs/mask_draws_af24_summary.{txt,json}`。已核对三个 MC-GS 日志读的是各自掩膜（`zero_lines=105/104/110 of 1877`）、L1 用覆盖掩膜。

### 0b. 09-15 下午：补 L1-SPIRiT 与 MC-INR，掩膜抽样方差跑完

用户问 AF24 执行的层是否与 AF12/16/20 一致（一致：同一组层号参数，GT 逐位相同），并要求补齐方法（"needing 补"）。AF12/16/20 在 09-14/15 由 claude-ad 会话加了 L1-SPIRiT、MC-INR，并按用户 09-15 决定去掉 L1-ESPIRiT；已与 claude-ad 发消息协调（AF24 归本会话），其分支 `spirit-arm-jyz-af12-16-20` 已快进合入本分支（13b7bce）。口径见记忆 jyz-comparison-conventions。

- **L1-SPIRiT**（`scripts/run_all_af24.py --study extra`，24 个 GPU 作业，无失败）：核 3/5 × λ{0,5e-4,1e-3,2e-3,4e-3,8e-3}，两对比度都保留核 5、λ 1e-3，全部内部极值。头内 PSNR/SSIM/HFEN：T1 31.783 / 0.8717 / 0.2246，FLAIR 31.803 / 0.8635 / 0.1804（AF20 32.882 / 32.824）。
- **MC-INR**：mcinr_niessen2025 加 AF24（提交 2f0b07f、5ff090a，未推），256 层 707 s，口径复核 OK；头内 T1 30.076 / 0.8103 / 0.3593，FLAIR 30.947 / 0.8303 / 0.2688；`scripts/score_mcinr_arm.py` 登记，最大相对差 1.2e-7。
- **表与图**：summary.json 26 行；README 表按 DROP 不列 L1-ESPIRiT；MC-GS 六格全第一、GS-single 第二。正式图 `figures/t1flair_v3_af24_{T1,FLAIR}_row3planes_compact_180_124_76_inset48_headerr_p3_mcinr_spirit.{png,pdf}`（10 个方法列，标题 `nominal AF 24, actual 24.00x / 25.51x`），`http://localhost:8765/t1flair_v3_af24/`。
- **掩膜抽样方差**（b/c/d 各恰好 1877 点）：对 L1-ESPIRiT 的余量 T1 +0.359 ± 0.006、FLAIR +0.397 ± 0.006（AF20 +0.459 ± 0.002 / +0.488 ± 0.004，缩小 17 / 15 倍 SD）；锐度 4/4 高于 L1。按现行口径这是历史口径的余量，**GS-single 没在 b–d 上跑，主余量的抽样波动未测**。

```
$ cd /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF24_v3_jiangyizhou
$ PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/compare_topics.py | sed -n '/strongest comparison arm/,/older framing/p'
=== MC-GS margin over the strongest comparison arm: GS-single counts, L1-ESPIRiT dropped (the margin to quote), head-only PSNR
  AF20 T1     MC-GS 34.832  strongest 34.667 (gs_single_p3)  margin +0.165   | strongest non-GS 34.003 (cssense_o200)  +0.829
  AF20 FLAIR  MC-GS 34.179  strongest 34.127 (gs_single_p3)  margin +0.052   | strongest non-GS 33.544 (jcssense)  +0.634
  AF24 T1     MC-GS 33.767  strongest 33.588 (gs_single_p3)  margin +0.179   | strongest non-GS 32.820 (cssense_o200)  +0.947
  AF24 FLAIR  MC-GS 33.182  strongest 33.110 (gs_single_p3)  margin +0.071   | strongest non-GS 32.605 (jcssense)  +0.577
$ PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/score_mask_draws_af24.py | tail -2
T1    margin over 4 draws: mean +0.359  SD 0.006  range [+0.354, +0.366]   AF20 (4 draws): +0.459 ± 0.002   shift -0.099 = -17.4 x the larger SD   sharpness gate passed on 4/4
FLAIR margin over 4 draws: mean +0.397  SD 0.006  range [+0.389, +0.403]   AF20 (4 draws): +0.488 ± 0.004   shift -0.092 = -14.7 x the larger SD   sharpness gate passed on 4/4
```

- **数据包与中文总结（claude-ad 按用户直接要求做，76e0d2f）**：`Results_20260908/T1FLAIR_AF24_v3_jiangyizhou/export/af24_p3_mcinr_spirit/`（72 个文件 11.77 GiB，原始 k 空间、线圈图、掩膜 a/b/c/d、各臂重建体含 SPIRiT 与 MC-INR、正式图；README 写明倍数口径与 A/B 未定），本会话独立 `md5sum -c` 72/72 OK；中文一页总结 `docs/2026-09-15-af24-jyz-summary.md`（关键数字已核对）。

### 0c. 09-16：原始数据落盘核实（用户："将上述重建结果对应的原始数据保存到该服务器上"）

已经在服务器上，两份互相独立，**没有再复制**：

- 厂商原始 CFL（采集后未动）`/data1/data1_congcong/data/MC_Jia/Train_SpiritDiffusion_CFL/vol3_jiangyizhou/{t1_fsp,flair}_train_jiangyizhou_cc.{cfl,hdr}`，各 2,399,141,888 B，md5 `c449fdbd…`（T1）/ `ce3627e9…`（FLAIR），头 dims `256 256 176 26 1`。同目录的 t2 文件本课题不用。
- 自包含数据包 `Results_20260908/T1FLAIR_AF24_v3_jiangyizhou/export/af24_p3_mcinr_spirit/`，72 个文件 12 GiB，软链接已解开；`md5sum -c checksums.md5` 09-16 重跑 **72/72 OK，无 FAILED**。

这次真正新做的是溯源核对：把数据包里的 `raw/ksp_{T1,FLAIR}.h5` 按 `prep_04_export.py` 的同一变换（Fortran 序 `(X,Y,Z,C,1)` 去掉末维、`moveaxis(3,0)` 成 `(C,X,Y,Z)`）读回来与厂商 CFL **逐元素**比，256 个读出层全比、不抽样：**每个对比度 299,892,736 个元素，差异 0 个**。所以包里的 k 空间就是厂商原始数据本身，正式图能一路追到采集。

`raw/csm_espirit_*.h5` 不是原始数据，是 BART `ecalib -t 0.001 -c 0.0 -r 24 -m 1` 算出来的，随时可复算；掩膜相反，`vdPoisMex` 按 `time(0)` 取种，不可重生成，数据包和课题 `data/` 是仅有的两份。磁盘 `/data0` 88%（余 5.1T）、`/data1` 76%（余 1.7T），再复制一份 4.8 GiB 没有收益，所以没做。

说明写进了数据包 README（新节 Provenance of the raw data）与课题 README（新节 Raw data on this server），两份都已镜像进本分支；记录见 `docs/superpowers/records/2026-09-15-af24-jyz.md` §12。

### 一句话
AF16 三期配方原样不调，跑在 1877 点的掩膜上（标签 **24.00×**，两个对比度同一个数，用户 09-16 定；FLAIR 有 1766 点带数据）：头内 PSNR **33.767 / 33.182**，九臂里头内 PSNR/SSIM/HFEN 六格全第一（含 GS-single）。**对最强对比方法 GS-single 的余量 +0.179 / +0.071**（AF20 +0.165 / +0.052；GS-single 算对比方法是用户 09-13 定的）。

**09-15 下午更正（claude-ad 会话转告的用户决定）**：① GS-single 算对比方法（用户 09-13），所以主余量是 MC-GS − GS-single，不是对 L1-ESPIRiT；② L1-ESPIRiT 已移出对比（用户 09-15，数据保留、表里不列），最强的非 GS 方法变为收敛 TV（T1 32.820，余量 +0.947）和 JTV（FLAIR 32.605，余量 +0.577）。下文和记录里写的"对 L1-ESPIRiT 余量 +0.366 / +0.400、比 AF20 缩小约 0.1 dB"是旧口径的历史数字，不再作为余量引用；它说明的"单对比度表征相对 L1-ESPIRiT 的领先在变小"仍成立。AF12/16/20 已补 L1-SPIRiT、MC-INR，正式图只有 `_mcinr_spirit` 一种；AF24 的这两个方法在补跑（`run_all_af24.py --study extra`）。

### 1. 已完成且已验证

从归档位置重跑（2026-09-15 05:40，与 finalize 写的 logs/compare_af20_af24.txt 逐字相同）：

```
$ PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
$ cd /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF24_v3_jiangyizhou
$ PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/compare_topics.py | sed -n '/MC-GS margin/,$p'
=== MC-GS margin over the strongest COMPETING arm (ours excluded), head-only PSNR
  AF20 T1     MC-GS 34.832  best other 34.371 (l1espirit)  margin +0.461
  AF20 FLAIR  MC-GS 34.179  best other 33.688 (l1espirit)  margin +0.491
  AF24 T1     MC-GS 33.767  best other 33.401 (l1espirit)  margin +0.366
  AF24 FLAIR  MC-GS 33.182  best other 32.782 (l1espirit)  margin +0.400

=== MC-GS minus GS-single (the coupling), head-only PSNR / SSIM
  AF20 T1     +0.165 dB  +3.2 mSSIM
  AF20 FLAIR  +0.052 dB  +1.6 mSSIM
  AF24 T1     +0.179 dB  +4.1 mSSIM
  AF24 FLAIR  +0.071 dB  +2.3 mSSIM

=== rank of MC-GS among all arms, head-only PSNR (1 = best)
  AF20 T1     rank 1/12   (top: gs_joint_p3 34.832)
  AF20 FLAIR  rank 1/12   (top: gs_joint_p3 34.179)
  AF24 T1     rank 1/11   (top: gs_joint_p3 33.767)
  AF24 FLAIR  rank 1/11   (top: gs_joint_p3 33.182)
```

九臂头内 PSNR（T1 / FLAIR）：MC-GS 33.767 / 33.182，GS-single 33.588 / 33.110，L1-ESPIRiT 33.401 / 32.782，收敛 TV 32.820 / 32.573，
Joint SENSE 32.730 / 32.605，ZS-SSDU 32.578 / 32.107，3D-DIP 30.500 / 30.360，CG-SENSE 30.340 / 31.259，3D-INR 28.444 / 29.146。
头内 HFEN MC-GS 0.1692 / 0.1579 也是两对比度最低；锐度 MC-GS 0.803 / 0.738，L1-ESPIRiT 0.787 / 0.704。

- 掩膜：vdPoisMex 老规矩只改 accelx（6.8656），134 张池子里按事先定的规则取编号最小的恰好 1877 点那张（draw_025，未裁剪）；T1 24.00×，FLAIR 25.51×（快门外 111 点）；md5 c3688d6182ba20df7404519e560dd9b4；三处独立核对一致，训练器日志 `actual acceleration: 24.00x`、`zero_lines=111/1877`。
- 用户 09-14 看过掩膜图回复"没问题"后才上 GPU；冒烟 60 步 rc=0（数字 NOT_EVIDENCE）。
- 经典臂 λ 六条曲线最优全在网格内部：L1-ESPIRiT T1 0.001 / FLAIR 0.002（AF20 为 0.003，往下移，新加的低端 0.0014 才把它夹住），收敛 TV 0.0007 / 0.0005，JTV 0.001 / 0.0007（其余五个与 AF20 相同）。
- 图（九臂，MC-GS 在两对比度、两种版式、三个方向 PSNR/SSIM/NMSE 全第一，标题按用户口径 `AF 24.00` / `AF 25.51`）：
  `figures/t1flair_v3_af24_{T1,FLAIR}_row3planes_compact_180_124_76_inset48_headerr_p3.{png,pdf}`、`..._174_110_47_headerr_p3.png`、三张掩膜图；`http://localhost:8765/t1flair_v3_af24/`。
- 48 框领先（只打印不上图）：T1 +0.371 / +0.330 / **−0.019**（矢状框 JTV 领先 0.019 dB），FLAIR +0.190 / +0.903 / +0.755。框是 AF16 上挑的，不能跨掩膜用，对外只引全体积余量。
- 38 个作业由 `scripts/run_all_af24.py` 排队跑完，无失败无重试；MC-GS joint 454 分钟（卡 6，挨着别人的作业）。

### 2. 待用户拍板
- ~~FLAIR 标法~~ **用户 09-16 定：两个对比度都标 24.00×**（= 整平面点数 ÷ 掩膜点数，同一张掩膜同一个数），FLAIR 图上注明 1766 个采样点带数据。AF24 的结果图、三张掩膜图、原始 k 空间图、README、四档复核已按此重出；09-16 之前写的段落是 09-14 的逐对比度口径（FLAIR 25.51×），保留不改。**AF12/16/20 的 FLAIR 标签（12.86 / 16.49 / 19.96）按同一决定要改成 11.71 / 15.19 / 18.56**：claude-ad 会话已收工，这件事没人做，待用户定谁来做。
- ~~掩膜 b/c/d 上补跑 GS-single~~ 用户 09-16 定：先不跑。
- 09-14 遗留：AF12/16/20 留档的旧定稿图 FLAIR 标签仍是旧口径（claude-ad 会话那边等用户定）。
- ①②③ 已做：四档复核课题（上午）、掩膜抽样方差、MC-INR（下午）。

### 3. 下一步
- 四档复核课题已按现行口径重出（09-15 傍晚）：不列 L1-ESPIRiT、四档都有 L1-SPIRiT 与 MC-INR、主余量 MC-GS − GS-single（T1 +0.150 → +0.148 → +0.165 → +0.179，FLAIR +0.036 → +0.042 → +0.052 → +0.071）；104 行重打分，与上午版共有的 282 个数值逐位相同。复现命令见其 README 第 5 节。
- 等第 2 节两个决定。没有在跑的作业。

### 4. 已知的坑与别重做什么
- 掩膜按秒级 time(0) 取种子、不进 git：别删 `data/mask_pool_af24/`、`data/mask_vdpois_256x176_af24.mat`、`data/mask_af24_3d.h5`；池子与挑选脚本都拒绝覆盖。
- 启动器只有 `scripts/run_all_af24.py`（克隆来的 `sched_af24.py`、`run_classical_af24.sh`、`run_stage2_af24.sh` 已删；前者会用 09-09 的旧 GS 参数和 hyw 课题）。它可以安全重启。
- zsh 里别用 `$E` 装环境变量前缀（不分词），直接写 `PYTHONNOUSERSITE=1 PYTHONPATH= $PY …`。
- Python 改文件失败后同一段命令里的 git 仍会执行：提交一律用 `&&` 串起来。提交 cd7b28b 的说明写的是启动记录，实际只含调度日志，启动记录在下一个提交 779579f。
- README 底部"joint 减 single"小表是空的：`readme_tables.py` 按 `gs_joint` 找，登记键是 `gs_joint_p3`（AF20 同样），耦合看 `compare_topics.py` 输出。
- 工作树目录名 `mcgs_af16_wt` 但检出 AF24 分支；脚本里的这个路径是有意的。
- 出 `_mcinr_spirit` 图：`--with_mcinr` 只加列不加文件名后缀，要 `--tag _p3_mcinr --with_mcinr --l1_arm spirit`（第一次用 `--tag _p3` 出成了 `_p3_spirit`，已删）。
- 余量口径：GS-single 算对比方法（用户 09-13）、L1-ESPIRiT 已移出（09-15）；课题 compare_topics.py 旧节把 GS-single 当 ours，只作历史。
- 调度器 `--study extra` 里 MC-INR 以退出码 3 让卡时自动重排续跑；MC-INR 仓库规定不推远端。

### 5. 为什么
- 冻结 MC-GS 参数、只换掩膜，AF20→AF24 的每个差值才能归因于掩膜；经典臂 λ 重扫，保证对手在网格最优点，比较对 MC-GS 偏保守（仅受网格间距限制，见记录 §6）。
- 让 T1 正好 24×（整平面口径），两个对比度都不低于 24×，不再出现 AF20"标称 20、实际 18.56"要加注释的问题。
- 用排队启动器而不是固定分卡：09-14 21:40 时 8 张卡里 5 张是用户别的会话的作业，固定分卡会饿死。

---

以下是 `jyz-af20-results` 的 STATUS，原样保留。


本分支从 `autoresearch-jyz-af16-phase3` 尖端 f0fe009 切出，承载 jiangyizhou 的**标称 AF20** 研究，**已全部跑完**。
先读中文总结 `docs/2026-09-14-af20-jyz-summary.md`，再读 `docs/handoff/2026-09-14-af20-jyz.md`，英文记录 `docs/superpowers/records/2026-09-14-af20-jyz.md`；
课题目录 `/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF20_v3_jiangyizhou/`（仓库镜像 `t1flair_af20_v3_jiangyizhou/`）。
AF16 三期的 STATUS 在它自己的分支与 tag `handoff/2026-09-13-jyz-af16-phase3` 上。

## L1-SPIRiT 臂（AF12/16/20）、09-15 两条约定、AF 标签修正、AF24 数据包（claude-ad，09-14 至 09-16；提交 13b7bce、76e0d2f）

先读 `docs/handoff/2026-09-16-spirit-arm-jyz-af12-16-20.md`（命令与输出都在里面）。要点：
- 三个课题各补了 k 空间对比臂 L1-SPIRiT（Lustig 工具箱 GPU 移植，双精度与原版对到 1e-12；原版 POCS 在此数据上发散，改 FISTA）；
  72 次扫描最优点全在网格内部，取 5×5 核 / λ 1e-3；头内 PSNR T1 35.017 / 33.788 / 32.882（AF24 31.783），FLAIR 35.103 / 33.720 / 32.824（31.803）。不改变任何余量。
- L1-ESPIRiT 从表和图撤出（数据保留，`DROP` 一处决定）；GS-single 算对比臂且是最强的，头条余量 = MC-GS − GS-single：
  T1 +0.150 / +0.148 / +0.165 / +0.179，FLAIR +0.036 / +0.042 / +0.052 / +0.071（AF12→24）。三课题 README 各有决定说明；`AF12_16_20` 趋势课题同样统一。
- 图标题 FLAIR 加速原取错字段（对自身快门），已改为整平面口径 12.86 / 16.49 / 19.96；定稿图按"旧图移入 `figures/archive_aflabel_pre20260915/`、原名重出"处理，
  T1 逐像素相同、FLAIR 只差标题行。交付图集 = `_spirit`（十列，无 L1-ESPIRiT），索引 `~/figs/spirit_index.html`。
- AF24 数据包 `export/af24_p3_mcinr_spirit/`（72 文件 11.77 GiB，md5 72/72）+ 中文总结 `docs/2026-09-15-af24-jyz-summary.md`。
- 待用户定：FLAIR 标签 A/B；AF24 掩膜 b/c/d 补跑 GS-single；删冗余分支 `spirit-arm-jyz-af12-16-20` 与工作树 `mcgs_spirit_wt`；AF12 用三期配方重跑使四点余量序列同配方（可选）。
- 会话边界：本段与交接文档随 tag `handoff/2026-09-16-spirit-arm-jyz` 一起推 origin（分支 `autoresearch-jyz-af24`，不合 main）。

## 0. 09-14 下午追加（分支 `jyz-af20-results` 的新提交）
- **加速倍数口径（用户 09-14 定）**：整平面线数 ÷ 该对比度带数据的采样线数（即 BART `pics` 打印的 `Acc`）。AF20 掩膜 T1 18.56×、FLAIR 19.96×；AF16 为 15.19 / 16.49；AF12 为 11.71 / 12.86。下文与旧文档、图标题里的“标称 AF20，实际 FLAIR 15.13×”是旧口径，未改。
- **跨加速倍数复核**：新课题 `Results_20260908/T1FLAIR_AF12_16_20_trend_v3_jiangyizhou/`（仓库镜像 `t1flair_af12_16_20_trend_v3_jiangyizhou/`，README 中文）。三个课题 GT 逐位相同、打分代码 md5 相同；72 个存盘体积用同一套代码重打分，原表复现。除 3D-DIP 在 AF16→AF20 持平外，所有方法头内 PSNR/SSIM/HFEN 两步都变差；展示层 180/124/76、174/110/47 与几乎全部层上同样成立；MC-GS 各加速第一，对 L1-ESPIRiT 余量 T1 +0.59/+0.50/+0.46、FLAIR +0.60/+0.43/+0.49。AF12 的 MC-GS 是 P14 配方（AF16/20 为三期配方），P6 同配方核对 AF12→16 同样下降。
- **AF20 掩膜复查**：`scripts/fig_mask_reinspect.py` → `figures/t1flair_v3_af20_mask_T1_FLAIR_reinspect.png`。T1、FLAIR 两个通道读同一文件、逐线相同；所有 AF20 主表结果与图都来自这张掩膜（抽样方差 b/c/d 除外）。定稿图按 finalize 命令重出与原图逐像素相同。
- 验证（从归档位置重跑，原始输出节选）：

```
$ T=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_16_20_trend_v3_jiangyizhou; PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
$ bash $T/scripts/run_rescore.sh      # 六个脱离会话的 CPU 作业，约 25 分钟
$ PYTHONNOUSERSITE=1 $PY $T/scripts/analyze_af_trend.py | head -n 2
rows rescored: 72  missing: []
worst |rescore - recorded|: {'head_psnr': '3.11e-05', 'head_ssim': '2.38e-07', 'head_hfen': '3.91e-07'}
$ bash $T/scripts/run_slice_trend.sh; PYTHONNOUSERSITE=1 $PY $T/scripts/slice_summary.py | grep -E "^=+ |^\[figure slices"
==================== T1
[figure slices 180/124/76] worse at both steps in 97/108 (method x plane x metric) cells; exceptions:
[figure slices 174/110/47] worse at both steps in 99/108 (method x plane x metric) cells; exceptions:
==================== FLAIR
[figure slices 180/124/76] worse at both steps in 105/108 (method x plane x metric) cells; exceptions:
[figure slices 174/110/47] worse at both steps in 108/108 (method x plane x metric) cells; exceptions:
# 2026-09-14 从归档位置重跑 = 会话内首跑：216 个重打分值最大差 0.0，表、判定 json、逐层 json 相同，两张图逐像素相同
```

## 一句话
AF16 三期配方**原样不调**跑在 AF20 掩膜上：头内 PSNR **34.832 / 34.179**，对最强对手 L1-ESPIRiT（λ 在 AF20 重扫）余量
**+0.461 / +0.491**（AF16 +0.497 / +0.425），九臂里头内 PSNR/SSIM/HFEN 六格全第一；**余量没有随加速变大**；
**T1 锐度 0.833 低于 L1-ESPIRiT 的 0.845**（AF16 时高于）。

## 1. 已完成且已验证

重跑核对（原始输出，2026-09-14 01:10）：

```
$ PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
$ cd /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF20_v3_jiangyizhou
$ PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/compare_topics.py | sed -n '/MC-GS margin/,$p'
=== MC-GS margin over the strongest COMPETING arm (ours excluded), head-only PSNR
  AF16 T1     MC-GS 35.706  best other 35.209 (l1espirit)  margin +0.497
  AF16 FLAIR  MC-GS 34.928  best other 34.503 (l1espirit)  margin +0.425
  AF20 T1     MC-GS 34.832  best other 34.371 (l1espirit)  margin +0.461
  AF20 FLAIR  MC-GS 34.179  best other 33.688 (l1espirit)  margin +0.491

=== MC-GS minus GS-single (the coupling), head-only PSNR / SSIM
  AF16 T1     +0.148 dB  +2.5 mSSIM
  AF16 FLAIR  +0.042 dB  +1.3 mSSIM
  AF20 T1     +0.165 dB  +3.2 mSSIM
  AF20 FLAIR  +0.052 dB  +1.6 mSSIM
```

九臂头内 PSNR（T1 / FLAIR）：MC-GS 34.832 / 34.179，GS-single 34.667 / 34.127，L1-ESPIRiT 34.371 / 33.688，
收敛 TV 34.003 / 33.494，Joint SENSE 33.885 / 33.544，ZS-SSDU 32.892 / 32.660，CG-SENSE 31.425 / 32.089，
3D-DIP 30.996 / 30.637，3D-INR 29.456 / 29.809。完整表（含 SSIM/HFEN/锐度/λ）在课题 README 和记录 §3。

- 掩膜 `vdPoisMex(256,176,256,176, 5,4,24,0,1)`：2427 点，T1 18.56×，FLAIR 对快门 15.13×（`logs/prep_04_export.json`）。
- 配方逐行核对过与 AF16 三期运行相同；经典臂 λ 六条曲线全是内部极值。
- 图（九臂，MC-GS 在两对比度、两种版式、三个方向 PSNR/SSIM/NMSE 全第一，MC-GS 面板无伪影）：
  `figures/t1flair_v3_af20_{T1,FLAIR}_row3planes_compact_180_124_76_inset48_headerr_p3.{png,pdf}`（AF16 定稿图同版式同框）、
  `..._174_110_47_headerr_p3.png`、`t1flair_v3_af20_mask_T1_vs_FLAIR.png`；`http://localhost:8765/t1flair_v3_af20/`。
- **48 框领先在 AF20 缩小**（T1 +0.571/+0.487/+0.653，AF16 +0.892/+0.826/+0.836；FLAIR +0.448/+0.900/+0.703，
  AF16 +0.824/+1.102/+1.060）：框是在 AF16 上按差距最大挑的，换掩膜往全体积余量回落。框内数字不能跨掩膜用。

- **掩膜抽样方差已测（09-14 追加，记录 §8）**：再抽 3 张 AF20 掩膜（b/c/d），每张只跑 MC-GS joint 与 L1-ESPIRiT（3 个 λ）。
  四张的余量 T1 **+0.459 ± 0.002**、FLAIR **+0.488 ± 0.004**；两臂绝对 PSNR 随抽样的 SD 是 0.009/0.007（T1）、0.040/0.040（FLAIR），
  两者同涨同跌（相关 +0.97/+0.99），所以配对余量稳 4–10 倍。AF16→AF20 的余量变化（−0.038 / +0.063）是抽样 SD 的 16 / 15 倍，
  是真变化（前提：AF16 那张掩膜的抽样 SD 与 AF20 相当）。**T1 锐度门 4/4 张都没过，FLAIR 4/4 张都过**：不是某张掩膜的偶然。
  复核：`cd <课题> && PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/score_mask_draws.py`

- **MC-INR（Niessen 2025）已登记进 AF20 表（09-14）**：它由另一个会话在 `Results_20260908/T1FLAIR_MCINR_Niessen2025_v3_jiangyizhou/` 跑过，
  AF20 用的正是本课题 `mask_af20_3d.h5`（软链）。用本课题打分代码重打分，与其仓库报的数最大相对差 2.2e-7；头内 PSNR T1 30.969 / FLAIR 31.666
  （AF16 31.590 / 32.243），落后 L1-ESPIRiT 3.4 / 2.0 dB，不改变最强对手与余量。十列图 `..._inset48_headerr_p3_mcinr.{png,pdf}`，九列定稿图保留（出图加 `--with_mcinr` 才有这一列）。

## 2. 待用户拍板
- AF20 上 T1 锐度门失守（0.833 对 0.845）：接受，还是为 AF20 单独调（例如相位 TV 加大）——后者就不再是同参数实验。
- 余量不随加速变大（平均 0.461→0.476）：靠提高加速逼近 1.0 dB 这条路，这份证据不支持。
- **标签**：口径已由用户定（第 0 节），不再需要重做真 20 倍。待定的是：是否把 AF20 文档与图标题里的旧标签改成“同一张掩膜，T1 18.56× / FLAIR 19.96×”（只改文字和重出图，不跑 GPU），以及 AF16 的图（09-11 按旧口径写的 nominal / actual）是否一并改。

**用户已拍板（09-14）**：huangyuwei 数据不考虑（不做留出受试者）；掩膜 `.mat` 暂不进 git（仍只在磁盘上，四张都无法重生成，别覆盖）；
MC-INR 已跑过、不补跑（只做了登记）；加速倍数按整平面口径（第 0 节）；本次新分析放新分支 `jyz-af20-results`（内容 = AF20 现有结果 + 新分析，重建体不进 git）。**剩下待定的是 T1 锐度门和是否按新口径改旧标签（见上）。**

## 3. 下一步
- 没有不需要拍板的事了。开放项两个：T1 锐度门；是否按新口径改旧标签。中文总结 `docs/2026-09-14-af20-jyz-summary.md`，跨加速倍数复核见新课题 README。

## 4. 已知的坑
- **不要在本课题 `data/` 里重跑 `prep_02_mask.m`**：会悄悄换掉所有结果依赖的掩膜（md5 1c90baf850a8c1e64b43f6c64146e2c9）。
- 分数 json 先于 ~750 MB 的体积 npz 落地：等 MC-GS 进程退出再登记臂，否则 `BadZipFile`。
- 卡的选择决定墙上时间：ZS-SSDU 挨着别人的长作业慢 10 倍，损失逐位相同；起跑前看 `nvidia-smi --query-compute-apps`，挑没有计算进程的卡。
- 监视脚本里 `pgrep -f` 会匹配到监视器自己，用 `kill -0 <pid>` 判断存活。
- **空卡检测别写成 `nvidia-smi | grep -q` 再取反**：在 `set -o pipefail` 下 grep 提前退出令 nvidia-smi 收 SIGPIPE，忙卡被判成空卡（09-14 三个训练挤上 5 号卡、一个 OOM）。先存变量再 grep。
- `echo "[$(date)] rc=$?"` 里 `$(date)` 先执行、把 `$?` 冲成 0；先 `rc=$?` 再写日志。
- 别的会话的调度器会在卡空出后几十秒内抢卡；自己的下一个作业要接同一张卡，就在上一个进程退出时直接交接（`handoff_draw_d_af20.sh`），别靠 120 秒轮询。
- `setup_af20_jyz.sh` 首版把自己删了（在自己的 rm 清单里），已补回修正。
- 工作树目录名 `mcgs_af16_wt` 但分支是 AF20；所有脚本里这个路径是有意保留的。

## 5. 会话边界
- `autoresearch-jyz-af20` 已推 origin（私有仓库 sober235/mcgs）并打 tag `handoff/2026-09-14-af20-jyz`。不合 main（跨血脉禁止合并）。
- 09-14 晚：用户要求把 AF20 结果放到新分支 `jyz-af20-results`（1cc612a 切出 + 一次提交），已推 origin，未打 tag；本工作树现在检出这个分支。

## 6. 为什么
- 冻结 MC-GS 参数、只换掩膜，AF16→AF20 的每个差值才能归因于掩膜；经典臂 λ 重扫，保证对手在最佳点。比较对 MC-GS 偏保守。
