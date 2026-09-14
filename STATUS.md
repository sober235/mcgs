# STATUS — 2026-09-14（分支 `16yo-mtp`，worktree `/data0/congcong/code/GS/mcgs_peds_wt`）

本分支 = 儿童 TSC（age16）UIH 3D MTP 数据线。一期、二期已完成，**三期（调参）进行中**。
英文总结：第一阶段 `docs/superpowers/records/2026-09-12-16yo-mtp-reference-dataset.md`，
第二阶段 `docs/superpowers/records/2026-09-13-16yo-mtp-af12.md`。
设计与计划：`docs/superpowers/specs/2026-09-12-peds-tsc-mtp-grappa-design.md`（一期）、
`docs/superpowers/specs/2026-09-12-16yo-mtp-stage2-af12-design.md` 与
`docs/superpowers/plans/2026-09-12-16yo-mtp-stage2-af12.md`（二期）。
课题目录：`Results_20260908/Peds_TSC_MTP_3D/`（参考 k 空间）与
`Results_20260908/Peds_TSC_MTP_AF12_retro/`（回顾性十臂对比，README 是权威数字）。
三期设计：`docs/superpowers/specs/2026-09-14-peds-af12-mcgs-autoresearch-design.md`，
程序与回合日志 `autoresearch_peds.md`，账本 `autoresearch_peds.tsv`。

## 1. 已完成且已验证

### 一期：参考数据集（09-12）
GRAPPA 补齐的多线圈 k 空间 + ESPIRiT 图 + SENSE 合成图，两个对比度（T1w = FA16/TE3.25，
T2s = FA4/TE27.41，脑内相关 0.57，是十个里最不相似的一对），存
`/data1/data1_congcong/data/3D/Peds_TSC_age16_MTP_GT/`（9.8 GB，index.json 的 10 个 md5 于 09-13 复核全部一致）。
藏线自检 0.096 / 0.188，噪声地板 0.067 / 0.144。

### 二期：回顾性 AF 12，十臂（09-13，MC-INR 于 09-14 补入）
```
cd /data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro
./scripts/py scripts/check_gauge.py --contrast T1w --method cgsense --gpu 2 | tail -1
# GAUGE OK -- the shared module reproduces recon_classical exactly   （九项指标差值全为 0.00e+00）
```
- 掩膜：2635 / 31616 点 = **AF 12.00（整格，含 ACS）**，对 19914 个有数据的位置是 **7.56 倍**；
  一张掩膜两个对比度；已知零区 11702 点给**所有**臂（用户拍板）。`logs/prep_mask.json`。
- 26 个排队任务返回码全为 0；六个 λ 扫描的最优**全部落在网格内部**（三个扫描为此外扩过）。
- 主表口径（中间 80 分区）与副表（全 104 分区）一致性：**20 个臂/对比度组合、100 个数值最大差 7e-6**
  （09-14 加入 MC-INR 后两个口径重跑全部十个臂，二期记录的是 18 组 90 个数值、2e-4）。
- **MC-INR（Niessen 2025 复现）09-14 补入**：主表头内 PSNR T1w 33.948 / T2s 29.710，两个对比度都第 8 / 10
  （高于 3D-INR 与 3D-DIP、低于 CG-SENSE）。它不改变任何门槛。跑法与检查见 README 的 "The MC-INR arm" 一节；
  代码在 MC-INR 仓库的 worktree `/data0/congcong/code/GS/mcinr_peds_wt`（分支 `peds-mtp`，未推远端）。

**结论：MC-GS 在这套数据上不领先，且联合不如单臂。**

| 头内 PSNR（中间 80 分区） | T1w | T2s |
|---|---|---|
| L1-ESPIRiT | **38.830** | 33.860 |
| GS-single | 38.086 | **34.175** |
| MC-GS | 37.855 | 34.160 |

联合减单臂：T1w −0.231、T2s −0.015，SSIM/HFEN/锐度两通道也都持平或更差。
经典联合臂独立佐证：Joint SENSE 减收敛的独立 CS-SENSE 是 T1w −0.90、T2s +0.06，方向完全一致。
成人 T1+FLAIR 上这个差值四组全正，这里不复现——决定因素是这对对比度的相似度，不是数据稀疏度。

### 三期：在本受试者上调参（09-14 起，进行中）

目标（用户 09-14 拍板）：MC-GS 在主表六格——头内 PSNR / SSIM / HFEN × T1w / T2s——对**所有**其他臂第一，
**GS-single 算对比方法**。起点门槛：T1w 38.830 / 0.9674 / 0.0756（L1-ESPIRiT），T2s 34.175 / 0.9052 / 0.1855
（GS-single）。基线 `peds_af12_joint` 的 `min_margin = −0.9744`（由 T1w 头内 PSNR 定）。
判分：`scripts/score_candidate.py`（课题），与主表同一函数、同一真值、同一区域；在二期的 joint 运行上逐位复现
已发表行，两分钟一次。停止规则：达标，或杠杆用尽（最后两轮 T1w 头内 PSNR 各自不到 +0.05 dB）。

## 2. 待用户拍板
2. ~~是否用一对"相似"的对比度做对照~~ —— **用户 2026-09-13 拍板：暂时不碰。** 本线的范围守住 T1w 与 T2s
   两个对比度。代价是阴性结论保持为"在这对对比度上联合无益"，不升级成"相似度低于某个水平就无益"的边界条件；
   要升级必须引入第三个对比度（候选 FA16/TE9.29，与 T1w 相关 0.900，一期产出已在 `Peds_TSC_MTP_3D/data/` 里，
   采集支撑与本课题逐位相同 md5 c4624874e099，掩膜可原样复用，约三小时单卡）。想做时不需要重新准备数据。
3. **其余八个对比度的一期输出（23 GB）是否删除**（本次未动，用户此前说先不动）。

## 3. 下一步

三期第一轮（λ 在本受试者上重标定）正在跑：四个运行，每个只动一个旋钮，只跑联合臂
（单臂等决出候选后再跑）。队列 `scripts/p3_round1_{a,b}.txt`，两个调度器实例卡不重叠：
```
cd /data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro
ALLOWED="1 0 2"   TAG=r1a setsid nohup bash scripts/peds_sched.sh scripts/p3_round1_a.txt >> logs/sched_r1a_boot.log 2>&1 &
ALLOWED="3 4 5 7" TAG=r1b setsid nohup bash scripts/peds_sched.sh scripts/p3_round1_b.txt >> logs/sched_r1b_boot.log 2>&1 &
```
会话中断后原样重跑这两条即可续跑：已有结果的队列行会被跳过。进度看 `logs/sched_r1{a,b}.log`
与账本 `autoresearch_peds.tsv` 的 `min_margin` 列。

## 4. 已知的坑与别重做

- **方位**：本课题体积是 (RO 320, PE 304, SPE 104)，**轴 2 = 横断、轴 0 = 冠状、轴 1 = 矢状**。
  已用厂商 DICOM 判定（`gt[:, 2:-2, 52]` 对厂商第 39 层，r = 0.899），不是猜的。从 AF16 课题复制来的脚本
  是反的，已全部修正。体素 0.7 × 0.7 × 1.35 mm 各向异性，画图必须按物理尺寸定面板比例。
- **房规生成器的 `calib` 不产生满采块**：旧课题掩膜也只有 472/576、482/576。本掩膜 366/576。别再为此报警。
- **`known_zero_support` 是 e0e1058 之后才有的**：本 worktree 从 3a44aba 切出，原本没有，导致 GS 臂拿不到
  已知零区（`frac_z_used = 0.0000`）。已移植框架部分。现在两通道都是 0.3701。
- **别让 `recon_classical.py` 改用共享 Pair**：那会让 `check_gauge.py` 变成自己验自己。
- **`grain_ratio` 在 T2s 上不可用**：冻结的白质掩膜只剩 664 个体素（T1w 是 220039）。口径未改，数字别引用。
- **选层别用 `best_slice_mcgs.py` 的判据**：它按 MC-GS 领先幅度排序，而 MC-GS 在 T1w 上从不领先，那样选是自我美化。
  图用的是解剖依据（横断 SPE 53 = 厂商第 38 层，一期病灶普查见室管膜下结节的层面）。
- **重评分必须并行**：冻结的 15³ LoG 卷积单线程，串行两遍口径要六小时。`rescore.py --jobs 9`。
- **GPU 政策 09-14 放宽**（用户拍板）：任意"完全没有计算进程且利用率 ≤10%"的卡，本线同时最多两个训练，
  靠两个 `ALLOWED` 不重叠的调度器实例保证。二期的"只用 2 号卡"已作废。BART 仍走 CPU、nice 10、8 线程。
- **`peds_sched.sh` 的作业日志名曾把 config 路径里的 `/` 原样带入**，导致每次启动立刻 rc=1 并空转；已修
  （`tr ' ./' '___'`）。看到调度器秒退秒启就是这个。
- 测试基线 221（`--ignore=tests/smoke_test.py --ignore=tests/test_score_edge.py`，后者需要只有主检出才有的文件）。

## 5. 关键决定的"为什么"

- **AF 按整格算（2635 点）而非按支撑（1660 点）**：用户原话是"已经采集点除以整体矩阵的大小"。两个口径都在表里标出。
- **已知零区给所有臂**：用户拍板。代价是 BART 必须显式传 `-p`，自监督划分必须与前向掩膜分开，否则会在恒为零的
  目标上训练或验证。
- **主表用中间 80 分区**：外侧各 12 个分区是层向过采样余量，激励剖面已衰减到 6–10%，厂商也不输出。
  全 104 分区的副表保留作交叉核对，两者排名一致。
- **P10 原样迁移不调参**：spec 定的做法，这样"联合是否有增益"这个问题的答案不被调参掩盖。
