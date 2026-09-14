# STATUS — 2026-09-14（分支 `16yo-mtp`，worktree `/data0/congcong/code/GS/mcgs_peds_wt`）

本分支 = 儿童 TSC（age16）UIH 3D MTP 数据线。一期、二期已完成；**三期（在本受试者上调参）进行中，有一个运行在跑**。

英文总结：一期 `docs/superpowers/records/2026-09-12-16yo-mtp-reference-dataset.md`，
二期 `docs/superpowers/records/2026-09-13-16yo-mtp-af12.md`，
**三期 `docs/superpowers/records/2026-09-14-16yo-mtp-stage3.md`（先读这份）**。
设计：一期 `.../specs/2026-09-12-peds-tsc-mtp-grappa-design.md`，二期 `.../specs/2026-09-12-16yo-mtp-stage2-af12-design.md`
与 `.../plans/2026-09-12-16yo-mtp-stage2-af12.md`，**三期 `.../specs/2026-09-14-peds-af12-mcgs-autoresearch-design.md`**。
回合日志 `autoresearch_peds.md`，账本 `autoresearch_peds.tsv`。
课题目录：`Results_20260908/Peds_TSC_MTP_3D/`（参考 k 空间）与 `Results_20260908/Peds_TSC_MTP_AF12_retro/`（十二臂对比表，README 是权威数字）。

## 1. 已完成且已验证

### 一期：参考数据集（09-12）
GRAPPA 补齐的多线圈 k 空间 + ESPIRiT 图 + SENSE 合成图，两个对比度（T1w = FA16/TE3.25，T2s = FA4/TE27.41，
脑内相关 0.57，是十个里最不相似的一对），存 `/data1/data1_congcong/data/3D/Peds_TSC_age16_MTP_GT/`（9.8 GB，
index.json 的 10 个 md5 于 09-13 复核一致）。藏线自检 0.096 / 0.188，噪声地板 0.067 / 0.144。

### 二期：回顾性 AF 12（09-13），十臂（MC-INR 于 09-14 补入）
```
cd /data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro
./scripts/py scripts/check_gauge.py --contrast T1w --method cgsense --gpu <空卡> | tail -1
# GAUGE OK -- the shared module reproduces recon_classical exactly   （九项指标差值全为 0.00e+00）
```
- 掩膜 2635 / 31616 = **AF 12.00**（整格含 ACS），对 19914 个实采位置是 7.56 倍；一张掩膜两个对比度；
  已知零区 11702 点给**所有**臂。
- 两个口径一致性：**20 组、100 个数值最大差 7e-6**（09-14 加入 MC-INR 后两个口径重跑了全部臂）。
- 结论未变：按原样迁移的 P10 配方，MC-GS 不领先且联合不如单臂；Joint SENSE 独立佐证。

### MC-INR 臂（09-14 完成）
主表头内 T1w 33.948 / 0.9027 / 0.1576，T2s 29.710 / 0.7981 / 0.3479，两个对比度都第 8 / 10。
跑法与检查见课题 README 的 “The MC-INR arm”。代码在 MC-INR 仓库的 worktree
`/data0/congcong/code/GS/mcinr_peds_wt`（分支 `peds-mtp`，**无远端、未推**，运行记录 `docs/2026-09-14-peds-mtp-run.md`）。
重跑：`$PY scripts/run_volume.py --topic peds --af 12 --gpu <空卡> --resume --yield_to_others`，再 `$PY scripts/score.py --topic peds --af 12`
（环境 `GR_pt1.10`，本机唯一装 tinycudann 的）。

### 三期到目前为止（09-14）
判据（用户拍板）：MC-GS 在主表六格对**所有**其他臂第一，**GS-single 算对比方法**。判分用
`scripts/score_candidate.py`（与主表同函数、同真值、同区域；在二期 joint 运行上逐位复现已发表行）。
```
cd /data0/congcong/code/GS/mcgs_peds_wt
tail -5 autoresearch_peds.tsv | cut -f1,2,7,8,9,10,11     # 账本：min_margin 为正即达标
```
- 工作点：小波 T1w 0.06 / T2s 0.10，edge-TV 几何 **T1w 0.03 / T2s 0.10**，其余同 P10。
- **逐对比度几何权重是新加的**（`lambda_geo_plain` / `lambda_geo_contrast`，`paired_gs/losses/edge_tv.py::geo_term`，
  4 个测试；默认回退使旧运行逐位不变；全量 **225 测试通过**）。
- T1w 头内 PSNR：37.855（基线）→ **38.516**（拆权重）→ **38.588**（6000 步）。门槛 38.848。
- **调参后的单臂 38.848 / 0.9681 / 0.0761，已超过 L1-ESPIRiT（38.830 / 0.9674 / 0.0756，HFEN 仍是 L1 最好）。**
- **联合减单臂 T1w 从 −0.230 扩大到 −0.332**；共同边缘 0.001 让 T2s 三格全部过线，代价是 T1w −0.133。
- 容量 6.0M：+0.001（无效）。两个调参臂已登记为 `gs_joint_p3` / `gs_single_p3` 并进表。

## 2. 待用户拍板

1. **是否按停止规则收尾**。spec 的“杠杆用尽”还差一步（见第 3 节的在跑运行）。剩余杠杆只有优化器
   （`rotation_lr 0.008`，jyz 线上值 +0.025 / +0.046）和相位 TV（本分支没有，需从 `autoresearch-jyz-af12-phase3`
   cherry-pick `paired_gs/losses/phase_tv.py`，同血脉不涉及跨血脉合并）。
2. **阴性结论怎么写**。当前证据支持的说法是：在每个对比度各自最合适的权重下，GS 单臂是全表最好的 T1w 重建
   （超过 L1-ESPIRiT），而同一方法的联合版落后 0.33 dB，且先验越轻差距越大；耦合项只是把代价从一个对比度挪到另一个。
3. 其余八个对比度的一期输出（23 GB）是否删除（一直未动）。

## 3. 下一步

**在跑（09-14 19:34 起，GPU 5，约 1.5 小时）**：`p3_single_T1w_i6000` —— 单臂的 6000 步。联合臂从这个杠杆拿到 +0.072，
公平起见单臂也要跑；若单臂拿到同等增益，这条杠杆对目标即中性、门槛随之上移。
```
cd /data0/congcong/code/GS/mcgs_peds_wt
grep -E "^\[iter" results_arc/p3_single_T1w_i6000.log | tail -1          # 进度
tail -1 autoresearch_peds.tsv | cut -f1,2,7,9,11                          # 完成后的行
```
进程没了而 `results_arc/p3_single_T1w_i6000_score.json` 也没有，就原样重跑：
```
cd /data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro
ITERS=6000 ALLOWED="5 1 4 7" TAG=r3c setsid nohup bash scripts/peds_sched.sh scripts/p3_round3_c.txt >> logs/sched_r3c_boot.log 2>&1 &
```
之后按第 2 条的拍板走：要么跑优化器那一轮，要么直接收尾（收尾要做的：最终臂重登记、两个口径重算、
`summarize.py` / `readme_tables.py` / `refresh_readme_tables.py`、重出十二臂的横排与三面图、写记录）。

## 4. 已知的坑与别重做

- **保存/打分的重建体带硬数据一致性**：`x_dc = ifft3(mf·y + (1−mf)·k_model)`。所以（a）给它全部可采数据就会原样
  返回真值（`p3_ceiling` 就是这么废掉的，头内 PSNR 150.3、SSIM 1.0，**别再跑**）；（b）“某臂更好地拟合了已采样数据”
  这种话不成立。要问容量用 `--save_render`（新加），它另存纯渲染体 `recT1_render` / `recFL_render`。
- **`peds_sched.sh` 的作业日志名曾把 config 路径里的 `/` 带入**，导致每次启动秒退 rc=1 并空转；已修（`tr ' ./' '___'`）。
- **加给联合臂的杠杆必须同样加给单臂**，否则“联合赢单臂”是靠不等设置赢的。T2s 单臂例外：三期没动过它自己的权重，
  `gs_single_ar` 就是它的调参版本（`rescore.py` 里有注释）。
- **方位**：体积 (RO 320, PE 304, SPE 104)，**轴 2 = 横断、轴 0 = 冠状、轴 1 = 矢状**（厂商 DICOM 判定）。体素各向异性 0.7×0.7×1.35 mm。
- **`grain_ratio` 在 T2s 上不可用**（白质掩膜只剩 664 体素）。
- **选层别用 `best_slice_mcgs.py` 的判据**（按 MC-GS 领先幅度排序 = 自我美化）。
- **重评分必须并行**（`rescore.py --jobs 11`，15³ LoG 单线程）；两个口径各约 40 分钟 CPU。
- GPU：任意“无计算进程且利用率 ≤10%”的卡，本线同时最多两个训练（两个 `ALLOWED` 不重叠的调度器实例）。
- 测试基线 **225**（`--ignore=tests/smoke_test.py --ignore=tests/test_score_edge.py`）。

## 5. 关键决定的“为什么”

- **GS-single 算对比方法**（用户 09-14）：否则“联合最优”可以靠不跟自家单臂比而成立。代价是门槛会随单臂调参上移——
  事实也确实如此（T1w 门槛从 L1 的 38.830 变成调参单臂的 38.848）。
- **逐对比度几何权重**：共享权重下 T1w 越低越好、T2s 越低越差，一个数伺候不了两个对比度；拆开后 T1w +0.661 且 T2s 零代价。
- **MC-INR 用并集掩膜**：本课题规矩是已知零区给所有臂，否则它是唯一可以在部分傅里叶区和椭圆角上凭空造信号的臂。
- **两个调参臂另起 key 进表**（`*_p3`），不覆盖二期的行：二期回答的是“原样迁移的配方能否领先”，三期回答的是
  “在本受试者上调到最优能否领先”，两个问题都要留在表里。
- **AF 按整格算（2635 点）**、**已知零区给所有臂**、**主表用中间 80 分区**：见二期记录，未改动。
