# STATUS — 2026-09-13（分支 `16yo-mtp`，worktree `/data0/congcong/code/GS/mcgs_peds_wt`）

本分支 = 儿童 TSC（age16）UIH 3D MTP 数据线。两个阶段都已完成。
英文总结：第一阶段 `docs/superpowers/records/2026-09-12-16yo-mtp-reference-dataset.md`，
第二阶段 `docs/superpowers/records/2026-09-13-16yo-mtp-af12.md`。
设计与计划：`docs/superpowers/specs/2026-09-12-peds-tsc-mtp-grappa-design.md`（一期）、
`docs/superpowers/specs/2026-09-12-16yo-mtp-stage2-af12-design.md` 与
`docs/superpowers/plans/2026-09-12-16yo-mtp-stage2-af12.md`（二期）。
课题目录：`Results_20260908/Peds_TSC_MTP_3D/`（参考 k 空间）与
`Results_20260908/Peds_TSC_MTP_AF12_retro/`（回顾性九臂对比，README 是权威数字）。

## 1. 已完成且已验证

### 一期：参考数据集（09-12）
GRAPPA 补齐的多线圈 k 空间 + ESPIRiT 图 + SENSE 合成图，两个对比度（T1w = FA16/TE3.25，
T2s = FA4/TE27.41，脑内相关 0.57，是十个里最不相似的一对），存
`/data1/data1_congcong/data/3D/Peds_TSC_age16_MTP_GT/`（9.8 GB，index.json 的 10 个 md5 于 09-13 复核全部一致）。
藏线自检 0.096 / 0.188，噪声地板 0.067 / 0.144。

### 二期：回顾性 AF 12，九臂（09-13）
```
cd /data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro
./scripts/py scripts/check_gauge.py --contrast T1w --method cgsense --gpu 2 | tail -1
# GAUGE OK -- the shared module reproduces recon_classical exactly   （九项指标差值全为 0.00e+00）
```
- 掩膜：2635 / 31616 点 = **AF 12.00（整格，含 ACS）**，对 19914 个有数据的位置是 **7.56 倍**；
  一张掩膜两个对比度；已知零区 11702 点给**所有**臂（用户拍板）。`logs/prep_mask.json`。
- 26 个排队任务返回码全为 0；六个 λ 扫描的最优**全部落在网格内部**（三个扫描为此外扩过）。
- 主表口径（中间 80 分区）与副表（全 104 分区）一致性：18 个臂/对比度组合、90 个数值全部在 2e-4 内。

**结论：MC-GS 在这套数据上不领先，且联合不如单臂。**

| 头内 PSNR（中间 80 分区） | T1w | T2s |
|---|---|---|
| L1-ESPIRiT | **38.830** | 33.860 |
| GS-single | 38.086 | **34.175** |
| MC-GS | 37.855 | 34.160 |

联合减单臂：T1w −0.231、T2s −0.015，SSIM/HFEN/锐度两通道也都持平或更差。
经典联合臂独立佐证：Joint SENSE 减收敛的独立 CS-SENSE 是 T1w −0.90、T2s +0.06，方向完全一致。
成人 T1+FLAIR 上这个差值四组全正，这里不复现——决定因素是这对对比度的相似度，不是数据稀疏度。

## 2. 待用户拍板

1. **是否在本受试者上开调参循环**去追 T1w 那 1.0 dB。二期按 spec 原样迁移 P10、未调任何参数；调参属另一次拍板。
2. ~~是否用一对"相似"的对比度做对照~~ —— **用户 2026-09-13 拍板：暂时不碰。** 本线的范围守住 T1w 与 T2s
   两个对比度。代价是阴性结论保持为"在这对对比度上联合无益"，不升级成"相似度低于某个水平就无益"的边界条件；
   要升级必须引入第三个对比度（候选 FA16/TE9.29，与 T1w 相关 0.900，一期产出已在 `Peds_TSC_MTP_3D/data/` 里，
   采集支撑与本课题逐位相同 md5 c4624874e099，掩膜可原样复用，约三小时单卡）。想做时不需要重新准备数据。
3. **其余八个对比度的一期输出（23 GB）是否删除**（本次未动，用户此前说先不动）。

## 3. 下一步

无进行中的计算。二期已收尾并推送。若做上面第 2 条：复用同一掩膜与同一套脚本，只改 `data/contrasts.json`
指向的两个对比度并重跑 `prep_mask.py` 之外的全部队列。

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
- GPU 只用 2 号且必须先看 `nvidia-smi --query-compute-apps`；BART 走 CPU、nice 10、8 线程。
- 测试基线 214（`--ignore=tests/smoke_test.py --ignore=tests/test_score_edge.py`，后者需要只有主检出才有的文件）。

## 5. 关键决定的"为什么"

- **AF 按整格算（2635 点）而非按支撑（1660 点）**：用户原话是"已经采集点除以整体矩阵的大小"。两个口径都在表里标出。
- **已知零区给所有臂**：用户拍板。代价是 BART 必须显式传 `-p`，自监督划分必须与前向掩膜分开，否则会在恒为零的
  目标上训练或验证。
- **主表用中间 80 分区**：外侧各 12 个分区是层向过采样余量，激励剖面已衰减到 6–10%，厂商也不输出。
  全 104 分区的副表保留作交叉核对，两者排名一致。
- **P10 原样迁移不调参**：spec 定的做法，这样"联合是否有增益"这个问题的答案不被调参掩盖。
