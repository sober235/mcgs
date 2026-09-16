# STATUS — 2026-09-16（worktree `mcgs_diffmask_wt`，分支 `jyz-af16-diffmask`，tag `handoff/2026-09-16-diffmask`）

## 结论：MC-GS 的领先不依赖采样图案

三个课题只差采样图案（vd-Poisson 房屋掩膜作对照列未重跑、gauss、radial），点数预算一致（2964 点，T1 15.20×）。
**MC-GS 在三种图案、两个对比度上头内 PSNR 全部第一**，对最强对比臂余量 T1 +0.148/+0.138/+0.200、
FLAIR +0.042/+0.023/+0.093。六格里五格三指标（头内 PSNR/SSIM/HFEN）全胜；唯一例外是 radial 的 FLAIR：
保住 PSNR，SSIM 0.8673 输 ZS-SSDU 0.8686、HFEN 0.2809 输 0.2768。那一列恰好是 known-zero 先验被图案废掉的一列。
图案本身的代价远大于方法间差距：同一臂从 vd-Poisson 换到高斯掉 0.3–1.1 dB、换到径向掉 1.1–5.0 dB。
耦合增益（MC-GS−GS-single）三种图案都为正，径向上最大。
表在 gauss 课题 `logs/compare_masks_head_{psnr,ssim,hfen}.md`（`scripts/compare_masks.py`，只读三个课题）。

## 已完成且已验证

- 两个课题各 11 臂 × 2 对比度，全部重建、打分、出图，零失败。图在
  `http://localhost:8765/t1flair_v3_af16_diffmask_{gauss,radial}/`，定稿版式 180/124/76 + 48 框（含 600dpi PDF），
  另有 174/110/47 无放大版；掩膜与全采样 k 空间图在 `http://localhost:8765/t1flair_v3_af16_diffmask/`。
- MC-INR 在两张掩膜上跑通并注册，本课题打分器重算与 mcinr 仓库自算的最大相对差 1e-7（打分口径与掩膜无关，
  所以三个课题的数字可直接并列）。
- 基线测试 206 通过（`results_arc/tests_baseline.txt`，改动前）。
- 第三张候选掩膜 caipi（4×3 CAIPIRINHA 格点，移位按混叠副本间距判据选）已建，用户没选，未跑。

## 用户已拍板（2026-09-16）

- **caipi 不跑。** 掩膜、读数与检查图留在 `T1FLAIR_AF16_caipi_v3_jiangyizhou`，本研究只比 vd-Poisson / 高斯 / 伪径向。
- **L1-ESPIRiT 彻底退役**，后续一律换成 L1-SPIRiT 作 k 空间对比臂。不是"数据留着不列"，是新课题不再跑它；
  旧课题已有的体积与 summary 行留作历史。`--with_l1espirit` 开关仍在，但本系列不用。
- **不再补调参**（选项三）。MC-GS/GS-single 三张掩膜共用冻结配方，经典臂按 AF16 网格各自重选 λ；
  表与图只呈现结果。radial 的 TV 两臂是边缘赢家，这一条留在「已知限制」，它只限制"对 TV 的余量"这类说法。
- **已推 origin**（sober235/mcgs）：分支 `jyz-af16-diffmask` 与 tag `handoff/2026-09-16-diffmask`。main 未动。

## 已知的坑与别重做

- ⚠️ **调度器的去重曾误杀 46 个 radial 作业**：作业指纹只比对脚本名和参数，两个课题这些完全相同，
  gauss 在跑时 radial 被判为重复，标成 EXTERNAL DONE 而从未启动。已修（指纹加进程工作目录；等待中的作业
  最后没产出就重排一次）。**再驱动多课题前先确认指纹能区分课题。**
- ⚠️ 心脏线 `launch_gs_when_free.sh` 的 `release_lock` 用 `rmdir` 删带 `pid` 文件的目录，永远失败，锁会泄漏。
  本调度器沿用同一套锁但按"属主进程已死即回收"处理。
- ⚠️ radial 掩膜上 FLAIR 的 known_zero 估计器认不出椭圆快门（快门外仅 27 点，`frac_z_used=0`），gauss 是 0.218。
- ⚠️ `logs/summary.json` 存的是臂的 **key** 不是标签；房屋课题里 `gs_joint` 是 2026-09-10 的旧臂，
  三期臂是 `gs_joint_p3`。跨课题比对必须用 key，否则会拿旧臂当对手（写 compare_masks.py 时踩过）。
- ⚠️ **同一类错误今天犯了三次：把为一个课题写死的东西复制给另一个课题。** (1) 调度器作业指纹不含课题→46 个 radial
  作业被当重复跳过；(2) 跨课题比对用 `gs_joint` 而不是 `gs_joint_p3`→拿房屋课题的旧臂当对手；(3) 图脚本的输出目录与
  文件名前缀写死→radial 的渲染盖掉了 gauss 的图。三处都已改成从课题路径/key 推导。**以后在课题间复制脚本前，
  先 grep 有没有写死的课题名、目录名、文件名前缀。**
- **调参口径（用户 09-16 定，选项三）**：不再补跑调参。MC-GS/GS-single 三张掩膜共用一套冻结的 AF16 三期配方；
  经典臂按 AF16 网格在各自掩膜上重选 λ。表和图只呈现结果，不把"对手调了我们没调"当卖点写在头条。
  但 **radial 的收敛 TV 与联合 JTV 赢家落在网格上界**（T1 0.0014、FLAIR 0.001/0.0014），这一条保留在「已知限制」里：
  它限制的是"MC-GS 胜过收敛 TV 多少"这类说法，不影响头条余量（MC-GS−GS-single，两者同配方）。
  记录边缘赢家而不是偷偷延网格，是本项目一贯做法（AF24 与心脏课题都这么记）。
- zsh 不拆分变量里的命令（`E="env A=1"; $E cmd` 报 command not found），长命令一律写成 bash 脚本文件。

---

# STATUS — 2026-09-15 17:30（worktree `mcgs_diffmask_wt`，分支 `jyz-af16-diffmask`，自 `autoresearch-jyz-af24` 13b7bce 切出）

## diff-mask 研究（本分支）：掩膜已建，等用户看图拍板，重建一步没跑

用户要求（09-15）：新分支，Gaussian 掩膜与 radial 掩膜各跑一遍全部方法，加速倍数按 15.2 设、标签记 AF 16；**先看掩膜再决定是否重建**。

- 三个课题目录：`Results_20260908/T1FLAIR_AF16_{gauss,caipi,radial}_v3_jiangyizhou/`，
  data/ 的 k 空间与 ESPIRiT 图是指向 AF12 课题的符号链接（AF16/20/24 课题同样做法）。
- 掩膜（`scripts/make_masks_af16_diffmask.py`，seed 0，可再生；两张都是 2964 点 = T1 15.20×，与 vdPoisson AF16 掩膜的 2966 点/15.19× 对齐；
  都保留 vdPoisson 掩膜实测的 24×16 全采样 ACS，L1-SPIRiT 校准区一致）：
  - gauss：ACS 外按二维高斯密度独立随机抽点，宽度 s=0.355（σ 45.4×31.2 线）由"采样点归一化半径中位数与 vdPoisson 掩膜相同"选出；FLAIR 15.43×，44 点在快门外。
  - caipi：4×3 规则格点 + CAIPI 移位 2（标称 R 12）。移位按 CAIPI 自己的判据选：混叠副本落在对偶格
    {(a/Ry, (b−a·shift/Ry)/Rz)}，4×3 移位 2 是本预算下唯一靠移位把副本最小间距拉开的（0.3005 FOV 对未移位 0.2500）；
    3×4、4×4、3×5 的任何移位都与自身未移位持平。格点加 ACS 超预算 1166 点，按最大半径裁掉 = 椭圆扫描，切在 rn 0.941，
    椭圆内格点完整。FLAIR 15.44×，45 点在快门外。（先试过 4×4 标称 R 16，均匀格点约两成落在平面四角，FLAIR 被推到
    18.14×、480 点在快门外，与另两张对不齐，已弃用。）
  - radial：12 条均匀角度伪径向辐条（每 15°，椭圆视野），按最大半径裁掉 207 个最外点凑到预算；FLAIR 15.34×，27 点在快门外。
    ⚠️ 09-15 文献核查：径向相位编码（ky-kz 平面辐条）是真实家族，但主场是心脏/动态，不算头部三标准之一。
  - 读数 `logs/make_masks_af16_diffmask.json`，图 `figures/t1flair_v3_af16_diffmask_{masks,gauss_width}.png`
    （`http://localhost:8765/t1flair_v3_af16_diffmask/`）。
## 2026-09-15 23:10 起跑（用户看图后拍板：跑第 1、2、4 列）

第 1 列 = 已有的 `T1FLAIR_AF16_v3_jiangyizhou`（vd-Poisson 房屋掩膜），**早已跑完全部臂，不重跑**，当对照列。
第 2 列 gauss 与第 4 列 radial 两个新课题跑全部方法。第 3 列 caipi 用户没选，掩膜留着不跑。

- 调度器 `scripts/run_all_diffmask.py`（两个课题各一份，从 gauss 课题起）：102 个作业，最多 4 个 GPU 作业，
  按层交错两个课题。`logs/run_all_diffmask.{log,_status.json}`，PID 在 `logs/run_all_diffmask.pid`，
  跑完自动 `finalize_diffmask.sh`（注册 MC-GS 臂 → summarize → readme_tables 对照房屋课题 → 紧凑三面图）。
- 臂与网格照 AF16 课题：ZF/CG-SENSE/CS-SENSE（base）、收敛 TV 各 5 个 λ、JTV 5 个 λ（AF16 原来只有 2 个，
  按 AF24 网格补宽，让赢家落在内部）、ZS-SSDU、3D-DIP、3D-INR、L1-SPIRiT（ks 3/5 × 6 个 λ）、MC-INR、
  GS-single、MC-GS。**L1-ESPIRiT 默认不跑**（用户 09-15 已把它移出对比），要跑加 `--with_l1espirit`。
- MC-GS/GS-single 用冻结的 AF16 三期配方，`run_diffmask_mcgs.sh` 的配方块与 `run_af24_mcgs.sh` 逐字相同（已 diff 核对）。
- 冒烟跑（60 步，两张掩膜都读对：15.20×，zero_lines 44 与 27）：gauss 头内 PSNR 33.534/32.634，
  radial 30.191。只是可行性检查，不是结果。
- ⚠️ **GPU 锁**：本调度器沿用心脏线 `Cardiac_FreeRunning_MCGS/work/.gpulock.<卡号>` 的锁，两个调度器不会挑到同一张卡。
  心脏那个 `launch_gs_when_free.sh` 的 `release_lock` 用 `rmdir` 删带 `pid` 文件的目录，永远失败，所以锁会泄漏；
  09-15 卡 1/2/5 就被属主已死的锁封了 5–9 小时。本调度器按"属主进程已死即回收"处理（不等它的 24 小时）。
- ⚠️ radial 掩膜上 FLAIR 的 known_zero 估计器认不出椭圆快门（落在快门外的点只有 27 个，`frac_z_used=0`），
  gauss 上是 0.218。这是掩膜性质不是 bug，写结果时要说明。
- 基线测试（改动前）：206 通过，`results_arc/tests_baseline.txt`。

- 待用户拍板：两张掩膜是否可用（高斯宽度可改 `--gauss_s`；改掩膜先删 data/ 下两个掩膜文件再重跑生成器）；同意后再定跑法
  （沿用 AF24 的 `run_all_af24.py` 作业表：MC-GS/GS-single 冻结 AF16 三期配方、经典臂按 AF24 网格重扫 λ、ZS-SSDU/DIP/INR/L1-SPIRiT/MC-INR）。
- 基线测试在本 worktree 跑过一次（`results_arc/tests_baseline.txt`，改动前）。

下面是 AF24 线的 STATUS 原文（本分支的起点）。

---

# STATUS — 2026-09-15 早（worktree `mcgs_af16_wt`，分支 `autoresearch-jyz-af24`，从 `jyz-af20-results` 4d13197 切出）

## AF24（本分支）：已全部跑完（09-14 21:45 启动，09-15 05:30 finalize 完成）

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

### 一句话
AF16 三期配方原样不调，跑在"T1 正好 24×"的掩膜上（1877 点，FLAIR 25.51×）：头内 PSNR **33.767 / 33.182**，九臂里头内 PSNR/SSIM/HFEN 六格全第一（含 GS-single）。**对最强对比方法 GS-single 的余量 +0.179 / +0.071**（AF20 +0.165 / +0.052；GS-single 算对比方法是用户 09-13 定的）。

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
- 没有阻塞项。可选：① 把 AF24 并进跨加速倍数复核课题 `T1FLAIR_AF12_16_20_trend_v3_jiangyizhou`；② AF24 掩膜抽样方差（余量缩小 0.1 dB 的显著性目前借用 AF20 四张掩膜的 SD 0.002/0.004）；③ AF24 的 MC-INR（没跑）；④ 09-14 遗留：AF16/AF20 图与文档是否按新口径改标签。

### 3. 下一步
- 等用户对第 2 节的选择；没有在跑的作业。

### 4. 已知的坑与别重做什么
- 掩膜按秒级 time(0) 取种子、不进 git：别删 `data/mask_pool_af24/`、`data/mask_vdpois_256x176_af24.mat`、`data/mask_af24_3d.h5`；池子与挑选脚本都拒绝覆盖。
- 启动器只有 `scripts/run_all_af24.py`（克隆来的 `sched_af24.py`、`run_classical_af24.sh`、`run_stage2_af24.sh` 已删；前者会用 09-09 的旧 GS 参数和 hyw 课题）。它可以安全重启。
- zsh 里别用 `$E` 装环境变量前缀（不分词），直接写 `PYTHONNOUSERSITE=1 PYTHONPATH= $PY …`。
- Python 改文件失败后同一段命令里的 git 仍会执行：提交一律用 `&&` 串起来。提交 cd7b28b 的说明写的是启动记录，实际只含调度日志，启动记录在下一个提交 779579f。
- README 底部"joint 减 single"小表是空的：`readme_tables.py` 按 `gs_joint` 找，登记键是 `gs_joint_p3`（AF20 同样），耦合看 `compare_topics.py` 输出。
- 工作树目录名 `mcgs_af16_wt` 但检出 AF24 分支；脚本里的这个路径是有意的。

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
