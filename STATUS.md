# STATUS — 2026-09-17（分支 `16yo-mtp`，worktree `/data0/congcong/code/GS/mcgs_peds_wt`）

本分支 = 儿童 TSC（age16）UIH 3D MTP 数据线。一至五期已完成（五期 09-17 收尾，提出的臂 = `gs_joint_p5`）；**本线没有训练在跑；已推 origin 并打 tag `handoff/2026-09-17-16yo-mtp-stage5`；待用户定论文图与 HFEN 写法（第 2 节）**。

英文总结：一期 `docs/superpowers/records/2026-09-12-16yo-mtp-reference-dataset.md`，
二期 `docs/superpowers/records/2026-09-13-16yo-mtp-af12.md`，
三期 `docs/superpowers/records/2026-09-14-16yo-mtp-stage3.md`，
四期（中文，另一会话 claude-eb 写）`docs/superpowers/records/2026-09-15-16yo-mtp-stage4-lesion-slice.md`，
L1-SPIRiT 换臂 `.../2026-09-15-16yo-mtp-l1spirit-arm.md`，选层 `.../2026-09-16-16yo-mtp-one-slice-all-first.md`，
**五期 `docs/superpowers/records/2026-09-16-16yo-mtp-stage5-levers.md`（先读这份）**；课题 README 的 "Stage 5" 一节是给外人看的版本。
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

### L1-SPIRiT 臂（09-15 23:06 登记，替换 L1-ESPIRiT）

用户 16:05 拍板换臂（GS-single 仍算对比方法）。扫描由 claude-eb 跑：50 个点，实采 ACS 24×8 校准与全 k 空间中心 24×24 校准两种
（后者是所有臂的 ESPIRiT 图本来就用的块），ks {3,5} × λ 网格，两个对比度。`summarize.py` 按头内 PSNR 取每对比度最优，两个都是
full24：T1w ks5 λ2.5e-4，T2s ks3 λ5e-4，最优点都在 λ 网格内部。`scripts/rescore_spirit.sh` 只给这一臂重评分并合并进
`rescore{,_full}.json`（整卷口径与扫描自带 json 逐位一致）。主区域：**T1w 38.310 / 0.9607 / 0.0679，T2s 32.017 / 0.8630 / 0.2264**。

- T1w：非 GS 臂里最强（14 个计数行里第 5，比不再计数的 L1-ESPIRiT 低 0.52 dB），**但拿走了 T1w HFEN 一格：0.0679 对
  GS-single 0.0764 / MC-GS 0.0768**——轻正则的 k 空间解把两个 GS 臂抹掉的颗粒（含噪声）留着，锐度 0.982。
- T2s：12 个计数行里第 8（低于 CS-SENSE、Joint SENSE、ZS-SSDU），不改任何门槛。
- L1-ESPIRiT 的行留在 json 里，README 表下打印为 "Not counted"；换臂前的表在 `logs/archive_pre_spirit_20260915/`。
- 图（L1 列已换，看过）：`peds_af12_{T1w,T2s}_row3planes_compact_53_160_152_headerr_p3f_spirit.png` 与
  `..._inset48_headerr_af16style_spirit.{png,pdf}`（`scripts/figs_spirit.sh`；旧版用 `--l1 espirit` 可重画）。
- 未换：`slice_quality.py` / `find_mcgs_first_slice.py` / `best_slice_mcgs.py` 三个逐层分析脚本仍列 L1-ESPIRiT，下次用时再换；
  四期病灶脚本已由 claude-eb 换（43bfe56）。
- 记录：`docs/superpowers/records/2026-09-15-16yo-mtp-l1spirit-arm.md`。

### 五期：杠杆轮（09-16/17，已收尾）——最终配方 `gs_joint_p5`

相位 TV（jyz af12 三期 cherry-pick）是唯一动 T1w 的杠杆；分对比度权重（本线新加）让 T2s 取 0.02、T1w 取 0.01；8000 步。
**主区域：T1w 39.003 / 0.9693 / 0.0758，T2s 34.323 / 0.9111 / 0.1765。六格对所有臂（含所有 GS-single）过五格**：
T1w PSNR +0.010（对已登记 9.6M 单臂）/ +0.005（对未登记的 6000 步单臂）、SSIM +0.0006，T2s +0.148 / +0.0059 / +0.0090；
整卷 PSNR/NMSE 两列也第一。**T1w HFEN 0.0758 输 L1-SPIRiT 0.0679**，十一个运行没有一个让它动过 0.001。
逐层：T1w 234/578、T2s 517/561 三项全第一，**两对比度同层全第一 227 层**（三期只有 10 层）。
**种子检验（09-17，`p5_final_seed1`，seed 1，不登记）**：T1w 39.001 / 0.9692 / 0.0759，T2s 34.322 / 0.9111 / 0.1765，五格照样第一；两种子差 T1w 0.0026 dB、T2s 0.001 dB，T1w PSNR 余量（+0.008 到 +0.010）是种子差的三到四倍。记录 §5。
复核（09-17 21:1x 重跑，写到 scratch 文件、不动登记 json）：
```
cd /data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro
CUDA_VISIBLE_DEVICES= ./scripts/py scripts/rescore.py --only gs_joint_p5 gs_single_p3f spirit --jobs 5 --out /tmp/verify_p5.json
# T1w L1-SPIRiT    head_psnr  38.310 ssim 0.9607 hfen 0.0679 psnr  41.225 nmse 0.00374 sharp 0.982
# T1w GS-single (tuned, 9.6M) head_psnr  38.993 ssim 0.9687 hfen 0.0764 psnr  41.803 nmse 0.00327 sharp 0.880
# T1w MC-GS (phase TV, 8000 it.) head_psnr  39.003 ssim 0.9693 hfen 0.0758 psnr  41.865 nmse 0.00323 sharp 0.884
# T2s L1-SPIRiT    head_psnr  32.017 ssim 0.8630 hfen 0.2264 psnr  34.971 nmse 0.01625 sharp 0.951
# T2s MC-GS (phase TV, 8000 it.) head_psnr  34.323 ssim 0.9111 hfen 0.1765 psnr  37.233 nmse 0.00965 sharp 0.807
# 与 logs/rescore.json 登记值最大差 2.8e-17（约 2 分钟 CPU）
```
T2s 的门槛 GS-single = `gs_single_ar`（34.175 / 0.9052 / 0.1855），见 README 主表。
正式图（ZF 列 + 大写方法名 + 横向放大框，claude-64 加的 `--with_zf --caps`）：
`peds_af12_{T1w,T2s}_row3planes_compact_53_158_150_inset48x48x16_headerr_allfirst_p5_zf_caps.{png,pdf}`（两对比度三平面全第一）、
`..._53_160_152_..._p5_zf_caps`（解剖质心视角，T1w 冠/矢状 PSNR 第二）、
`..._46_149_171_inset48x48x16_box157_171_46_headerr_lesionB_allfirst_p5_zf_caps`（过结节，两对比度三平面全第一）、
`..._46_146_172_..._lesionB_p5_zf_caps`。记录 `docs/superpowers/records/2026-09-16-16yo-mtp-stage5-levers.md`。

### 一张 MC-GS 全超所有对比方法的层（09-16，用户要求；已被五期的 227 层超越）

`scripts/find_slice_all_first.py`：三个平面逐层，图面板同一口径（头内 PSNR / SSIM 7×7 / NMSE），MC-GS = `gs_joint_p3f`，
对手 = 计数表行 + 该对比度**所有** GS-single 体（含 results_arc 里五个 T1w 单臂运行）。计数：T1w 29/578（横断 1 层颅底碎片、
冠状 9 层前后缘帽、**矢状 19 层全是大截面**），T2s 331/561。两个对比度同一层都赢的只有 10 层，全在矢状 137–173。
**选矢状 161**：T1w 39.344 对 GS-single(p3f) 39.326（+0.018 dB / SSIM +0.0001），T2s 33.449 对 GS-single 33.388（+0.061 / +0.0008）。
备选 160（T2s 余量最大 +0.095，但 T1w SSIM 余量四位小数下为 0）。T1w 上没有任何横断或冠状大截面能全超。
图：`peds_af12_{T1w,T2s}_allfirst_sag161_axis1_161.png`（及 sag160），记录 `docs/superpowers/records/2026-09-16-16yo-mtp-one-slice-all-first.md`。
（09-16 上午）用户改口：有一张全超即可；再要"超过的三个轴面 + 病灶 zoom"。T2s 过右侧室管膜下结节的三平面
**横断 46 / 冠状 146 / 矢状 172**，三平面三项全第一（余量 +0.081 / +0.041 / +0.095 dB，对手都是 GS-single）；
定稿图 `peds_af12_T2s_row3planes_compact_46_146_172_inset48x48x16_box157_172_46_headerr_lesionB_allfirst.{png,pdf}`
（用户改：矢状/冠状放大框改成横向矩形 48×48×16 体素、框心移到两个结节中间 RO 157、放大图不压左上角指标；脚本加了 `--inset RO PE SPE` 与 `--inset_center`）。
T1w 同层同框的配套图 `peds_af12_T1w_..._lesionB_allfirst.{png,pdf}` 也出了：三平面都是 GS-single 第一、MC-GS 第二（矢状 39.201 对 39.202），病灶在 T1w 上只是淡灰点。
（09-16 上午末）用户要"T1 和 T2* 都超过对比方法的一层"：十个双赢矢状层里**只有 173 穿过病灶**（右侧两个结节，PE 172）：
T1w +0.007 dB / SSIM +0.00002（面板四位小数下 0.9608 平），T2s +0.058 / +0.00012，三项都严格第一；161 余量都看得见但没病灶。
单平面图（脚本新加 `--planes`）`peds_af12_{T1w,T2s}_row3planes_compact_46_146_173_planes-sag_inset48x48x16_box157_173_46_headerr_bothfirst.{png,pdf}`（另有 161 版）。记录 §6。
钙化样灶的冠状面 SSIM 输 GS-single，凑不齐三平面；T1w 同三层 MC-GS 第二（病灶在 T1w 上本就不可见）。记录 §5。

### 三期：在本受试者上调参（09-14/15，已收尾）

判据（用户 09-14 拍板）：MC-GS 在主表六格对**所有**其他臂第一，GS-single 算对比方法。**没有达成**，
但拆出了比"达成"更有价值的东西。英文全过程见 `docs/superpowers/records/2026-09-14-16yo-mtp-stage3.md`，
课题 README 的 "Stage 3" 一节是给外人看的版本。

最终配置：小波 T1w 0.06 / T2s 0.10，几何 **T1w 0.03 / T2s 0.10**，9.6M 点，4000 步，
`edge_tv_union_guide=true` + `edge_tv_union_kappa_scale=0.5`。登记为 `gs_joint_p3f` / `gs_single_p3f`。

1. **T2s 上联合臂全表第一**：34.227 / 0.9056 / 0.1836，三格全赢 GS-single（34.175 / 0.9052 / 0.1855）。
   靠的是 union guide——用 T1w 的边缘给 T2s 的平滑设上界，**单臂原理上没有第二个对比度可借**，
   是这条线找到的唯一一个"只有联合才有"的机制，且 T1w 零代价。（该开关来自同分支的 stage 4 会话，
   commit a22073f；我在本课题口径上独立复核过它的 p4_union 与 p4_u_k05 才用。）
2. **T1w 上单臂仍赢 0.057 dB**：38.993 / 0.9687 / 0.0764 对 38.936 / 0.9685 / 0.0768。
   两个 GS 臂都已越过 L1-ESPIRiT 的头内 PSNR 与 SSIM，L1 只剩 HFEN（0.0756）。
3. **二期那句"联合不如单臂"只在 3.6M 点云下成立**：匹配容量的差值 −0.332（3.6M）→ −0.198（6.0M）
   → −0.057（9.6M），单臂在 6.0M 封顶而联合一直涨。一个云表示两个对比度需要更多点；这条线
   （以及 jyz 线那次 4.8M 的容量检查）此前所有比较都在 3.6M 上做。饱和后的真实残差是 0.06 dB。
4. **两个对比度对每个旋钮的需求相反，而容量是唯一不能按对比度拆的**（联合臂只有一个云）：
   T1w 表示受限（3.6M→6.0M +0.275），T2s 信噪比受限（+0.032，到 19.2M 其 SSIM/HFEN 反而掉线）。

**为什么停手**：T1w HFEN 差 0.0011 且没有不伤 PSNR 的杠杆，而 PSNR 只剩 0.057 余量；
一个受试者、无留出集，刮第三位小数即拟合测试集。

图（最终配置，最后两列 = MC-GS 与 GS-single）：
`http://localhost:8765/peds_tsc_af12/peds_af12_{T1w,T2s}_row3planes_compact_53_160_152_headerr_p3f.png`

### 并行会话（截至 09-17）

- claude-eb（四期）：已收尾，p4 调度器停（`logs/.p4_stop`），不再起训练；它只写 `autoresearch_peds_p4.md`、`p4_*` 行和
  课题里的 `analysis_2026-09-15_stage4_lesion_slice/`（四期归档，244 MB，带 MANIFEST md5）。
- claude-64：给 `fig_three_planes_inset.py` 加了 `--with_zf`、`--caps`（默认不变）和 `scripts/figs_three_planes_zf_caps.sh`；
  09-17 按用户要求推过本分支。
- claude-e7（本 STATUS 的维护者）：拥有 STATUS / CLAUDE.md / 登记（summarize、rescore、README 表）。
- **本线同时最多两个训练**（计入所有会话）；要起训练先确认没有别的会话在本线跑（`pgrep -fa "cycle.py --exp .*peds_"`）。

## 2. 用户拍板（09-15 起，已定）与待定

1. **四期：1 dB 达不到，接受 0.15 dB。** 即矢状 172 的 +0.153（`p4_ue_k05`）/ +0.150（`p4_r3_edge0005`），
   对所有臂含 GS-single；不算 GS-single 是 +0.55。四期到此收尾，p4 队列里剩下的探针不再跑。
2. **三期剩余杠杆（`rotation_lr`、相位 TV）先不跑。** 用户要先看图，看完再定下一步动什么。
3. **结论方向没有商量余地：推 MC-GS 为最优，且超过所有对比方法（含 GS-single）。** 三期的六格判据不撤，
   上一版第 2 条"阴性结论怎么写"作废。现状离这条差 T1w 三格：PSNR 对调参 9.6M 单臂 −0.057 dB、SSIM −0.0002，
   **HFEN 的门槛换臂后变成 L1-SPIRiT 的 0.0679，MC-GS 0.0768 差 0.0089**（换臂前对 GS-single 只差 0.0004）；T2s 三格已达标。这三格只能靠真实运行补上，不能靠写法（候选杠杆见第 3 节）。
4. **一期其余八个对比度的 23 GB 先不删。**

（09-16 中午）5. **GS-single 的所有版本都算对比方法**（用户重申："gs 重建 for 3D MR 已经有工作了"）。用户提出"适当把 GS-single
   调参一点，让 MC-GS 超所有方法 0.8 dB"——把已跑出的对比配置调差去凑余量，本会话拒绝，理由已答复。事实：T1w 上卡 0.8 dB 的是
   L1-SPIRiT（PSNR 余量 +0.63，HFEN 输），不是 GS-single；T2s 上原样迁移的 GS-single 也只落后 0.05 dB。0.8 dB 对任何 GS-single
   口径都到不了。
6. **"启动 MC-GS 杠杆"**：五期开跑（第 3 节）。目标仍是六格全第一；能动的是 T1w 三格。
7. （15:40）**"无论如何，只要提出的 MC-GS 尽可能超过对比方法"**：GS-single 按已发表基线对待——表里已登记的每个版本都算门槛，
   但联合臂后来的杠杆（相位 TV、步数）不再给单臂，也不再跑新的单臂对照；记录里写明这一点。

另（09-15 16:05，用户）：**L1-SPIRiT 替换 L1-ESPIRiT，GS-single 仍算对比方法**，与 jyz 线同日的口径一致。
L1-ESPIRiT 的行留在表里、标为不计入，所以它的 T1w HFEN 0.0756 不再是门槛；L1-SPIRiT 已登记（第 1 节），它接过了 T1w HFEN 的门槛。
8. （09-17）用户经 claude-64 要求把本分支推到 origin；之后的提交已跟推。

**待用户拍板**：
- ~~A. 论文图用哪组~~ **已定（09-18）：`peds_af12_{T1w,T2s}_row3planes_compact_46_149_171_inset48x48x16_box157_171_46_headerr_lesionB_allfirst_p5_zf_caps.{png,pdf}`**
  ——过右侧两个室管膜下结节的横断 46 / 冠状 149 / 矢状 171，两对比度在三个平面上 MC-GS 三项全第一
  （T1w +0.040 / +0.018 / +0.006 dB，T2s +0.220 / +0.068 / +0.201 dB，每格的第二名都是某个 GS-single），
  放大框 48×48×16 体素、框心 (RO157, PE171, SPE46) 在两个结节中间。论文版的字号/裁切/图注还没定，用户要什么再改。
- B. T1w HFEN 输 L1-SPIRiT（0.0758 对 0.0679）怎么写。数据事实：k 空间插值保留含噪声的颗粒，HFEN 奖励它；MC-GS 仍超过所有 GS-single（0.0764）。
- C. 一期其余八个对比度的 23 GB 仍未删（第 4 条，暂不删）。

## 3. 下一步

**本线没有训练在跑**（p5 调度器已停 `logs/.p5_stop`；claude-eb、claude-64 也没有训练）。五期收尾工作已做完：登记、两个口径重评分、
README 表与 Stage 5 一节、四组正式图、逐层搜索、记录。

1. 新会话先重跑第 1 节五期块的复核命令（2 分钟 CPU），对上登记值再动手。
2. 论文图已定（第 2 节 A）。要出论文版（字号、裁切、单栏/双栏、图注）时在现有 PNG + 600 dpi PDF 上改；
   重画命令见 `scripts/figs_three_planes_zf_caps.sh`（改 `--slices 46 149 171 --inset_center 157 -1 -1` 与 `--mcgs_key gs_joint_p5`）。等第 2 节 B。
3. 杠杆已穷尽（记录 §2 表 + §4），种子检验已做（§5）。没有待跑的训练。
4. 尾巴（下次用到时做）：`slice_quality.py` / `find_mcgs_first_slice.py` / `best_slice_mcgs.py` 仍列 L1-ESPIRiT；README 的
   "MC-GS minus GS-single" 小表仍是二期的 `gs_joint_ar − gs_single_ar`。

## 4. 已知的坑与别重做

- **保存/打分的重建体带硬数据一致性**：`x_dc = ifft3(mf·y + (1−mf)·k_model)`。所以（a）给它全部可采数据就会原样
  返回真值（`p3_ceiling` 就是这么废掉的，头内 PSNR 150.3、SSIM 1.0，**别再跑**）；（b）“某臂更好地拟合了已采样数据”
  这种话不成立。要问容量用 `--save_render`（新加），它另存纯渲染体 `recT1_render` / `recFL_render`。
- **`peds_sched.sh` 的作业日志名曾把 config 路径里的 `/` 带入**，导致每次启动秒退 rc=1 并空转；已修（`tr ' ./' '___'`）。
- **三期的规矩"加给联合臂的杠杆同样加给单臂"在五期被用户改了**（第 2 节第 7 条）：GS-single 按已发表基线对待，已登记的
  每个版本都算门槛，但相位 TV、8000 步不再给单臂，也**不要再跑新的单臂对照**。写文章时要写明这一点。
  （T2s 单臂 `gs_single_ar` 就是它的调参版本，三期没动过它自己的权重。）
- **别把对比方法调差去凑余量**（09-16 用户提过"调 GS-single 让 MC-GS 超 0.8 dB"，已拒；0.8 dB 在 T1w 上卡的是 L1-SPIRiT）。
- **方位**：体积 (RO 320, PE 304, SPE 104)，**轴 2 = 横断、轴 0 = 冠状、轴 1 = 矢状**（厂商 DICOM 判定）。体素各向异性 0.7×0.7×1.35 mm。
- **`grain_ratio` 在 T2s 上不可用**（白质掩膜只剩 664 体素）。
- **选层别用 `best_slice_mcgs.py` 的判据**（按 MC-GS 领先幅度排序 = 自我美化）。
- **重评分必须并行**（`rescore.py --jobs 11`，15³ LoG 单线程）；两个口径各约 40 分钟 CPU。
- GPU：任意“无计算进程且利用率 ≤10%”的卡，本线同时最多两个训练（两个 `ALLOWED` 不重叠的调度器实例）。
- **`fig_row_final.py` 以前在非横断面上比例是错的**：它按物理比例算好了面板框，却没给 `imshow` 传
  `aspect="auto"`，于是按**像素**比例作图，把冠状面画成 1:2.9（真实 1:1.5）。横断面上像素比例与物理比例
  几乎相同（1.05 vs 1.05），所以这个错一直没暴露。已修（09-15）。用它出冠状/矢状图前先确认这条改动在。
- **图必须自己看过再发**：三期在移植 AF16 版式时出过三个只有看图才能发现的错——放大框越界压住下一行、
  放大图与框比例不一致（本课题 0.7 / 1.35 mm 各向异性）、纯重建图把冠矢状面拉伸到与横断同高（解剖比例失真）。
  指标一个都抓不到。已修；脚本 `fig_three_planes_inset.py`（AF16 版式）与 `fig_recon_only.py`。
- 测试基线 **225**（`--ignore=tests/smoke_test.py --ignore=tests/test_score_edge.py`）；`pytest tests -k ...` 会 SystemExit 2（某测试文件解析 argv），按文件跑。
- **调度器已启动的行事后注释无效**（16:37 的单臂对照就是这么跑起来的）；zsh 里未加引号的 `$files` 是一个词，`git add` 用 glob。
- **别的研究正在往卡上读数据的 1–2 分钟里显卡看起来是空的**（claude-71 的 1/2/5 号卡），要改 ALLOWED 只能停掉重起调度器。
- `rescore.py --only KEY` 只算新臂并合并（约 2 分钟），不必全表重跑。

## 5. 关键决定的“为什么”

- **GS-single 算对比方法**（用户 09-14）：否则“联合最优”可以靠不跟自家单臂比而成立。代价是门槛会随单臂调参上移——
  事实也确实如此（T1w 门槛从 L1 的 38.830 变成调参单臂的 38.848）。
- **逐对比度几何权重**：共享权重下 T1w 越低越好、T2s 越低越差，一个数伺候不了两个对比度；拆开后 T1w +0.661 且 T2s 零代价。
- **MC-INR 用并集掩膜**：本课题规矩是已知零区给所有臂，否则它是唯一可以在部分傅里叶区和椭圆角上凭空造信号的臂。
- **两个调参臂另起 key 进表**（`*_p3`），不覆盖二期的行：二期回答的是“原样迁移的配方能否领先”，三期回答的是
  “在本受试者上调到最优能否领先”，两个问题都要留在表里。
- **AF 按整格算（2635 点）**、**已知零区给所有臂**、**主表用中间 80 分区**：见二期记录，未改动。
- **相位 TV 分对比度权重**（五期）：共享权重下 T1w 在 0.01 最好、T2s 在 0.02–0.03 最好，同三期几何权重的道理；拆开后 T1w 不变、
  T2s 三格都涨。
- **最终配方选 T2s 0.02 而不是 0.03**：0.03 的 T2s SSIM/HFEN 更好但 PSNR 少 0.03 dB；0.02 三格都有余量，是共享扫描里 T2s PSNR 的最优。
- **种子检验只跑一个、不登记**：目的是回答"T1w +0.010 dB 是不是单次运行碰巧"，两个种子的差 0.0026 dB，够回答这个问题；
  要报分布需要更多种子，那是新实验。
