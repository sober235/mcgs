# STATUS — 2026-09-14 晚（worktree `mcgs_af16_wt`，分支 `autoresearch-jyz-af24`，从 `jyz-af20-results` 4d13197 切出）

## AF24（本分支，09-14 21:00 起）：用户已确认掩膜，21:45 全部作业已启动（跑完自动 finalize）
- 用户要求：在 mcgs 仓库开 AF=24 分支，执行完整的 AF24 实验，开跑前先看准备用的掩膜图。
- 口径按用户 09-14 的定义（整平面点数 ÷ 该对比度带数据的采样点数）。本课题让 T1 正好 24×（1877 点），FLAIR 因快门外 111 点落空为 25.51×。用户没有在"T1 正好 / FLAIR 正好 / 照旧 accelx=6"里明确选，按我推荐的"T1 正好"做了；掩膜图发给用户后，**用户 09-14 回复"没问题"（确认掩膜、同意开跑）**。
- 掩膜：同一生成器、同一设置，只改 accelx。134 张抽样池里按事先定的规则取编号最小的恰好 1877 点那张（draw_025，accelx 6.8656，未裁剪），md5 c3688d6182ba20df7404519e560dd9b4；另一张恰好 1877 点的 FLAIR 为 25.37×。挑选脚本、prep_04_export、按两份配置读回的 fig_mask_reinspect 三处独立核对一致。
- 课题 `Results_20260908/T1FLAIR_AF24_v3_jiangyizhou/`（仓库镜像 `t1flair_af24_v3_jiangyizhou/`，README 有掩膜表）；配置 `configs/t1flair_v3_af24_jyz{,_singleT1,_singleFLAIR}.yaml`（非注释行只改路径、acceleration_factor、实验名；acceleration_factor 只用于阈值 10 的策略分支，16/20/24 同一分支）；MC-GS `run_af24_mcgs.sh`（配方块与 AF20 逐字相同）；全部作业 `scripts/run_all_af24.py`（38 个作业按优先级排队，哪张卡空出来就放哪张：卡上没有本用户任何计算进程、显存够，最多同时 4 个 GPU 作业；12 个 L1-ESPIRiT 占 4 个 CPU 槽；全部结束后自动跑 finalize）。改用排队是因为 21:40 时 8 张卡里 5 张挂着你别的会话的作业（心脏径向、儿童 MC-GS、检测器），固定 6 卡方案会有链一直等；38 个作业的签名匹配已单测，互不误认。
- 经典臂 λ 网格 = AF20 最终网格各往下加一个点（AF16→AF20 唯一移动的最优 L1 T1 是往下移）。
- 基线测试 206 passed（`--ignore=tests/smoke_test.py --ignore=tests/test_score_edge.py`，后者要主检出才有的 npz）。
- 验证（原始输出）：

```
$ cd /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF24_v3_jiangyizhou
$ PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/fig_mask_reinspect.py | head -3
T1 mask /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF24_v3_jiangyizhou/data/mask_af24_3d.h5 md5 3e354e2a: 1877 lines
FLAIR mask /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF24_v3_jiangyizhou/data/mask_af24_3d.h5 md5 3e354e2a: 1877 lines; lines differing from the T1 mask 0
T1 sampled lines with signal 1877 -> AF 24.00; FLAIR sampled lines with signal 1766 -> AF 25.51, without 111; FLAIR acquired region 34139
```

- 冒烟（21:35–21:45，卡 6，60 步，warmup 20；数字只证明流程通，NOT_EVIDENCE）：

```
$ WARMUP=20 ./run_af24_mcgs.sh af24_smoke 6 60        # results_arc/af24_smoke.log
Using external mask with actual acceleration: 24.00x   (T1、FLAIR 各一行)
[physics:contrast] known_zero=True ... zero_lines=111/1877 explained=98 ...
[21:45] af24_smoke rc=0  head_psnr_cplx": 29.176 head_psnr": 32.380
SCORE combined=21.474 both_beat_zf=True N=3600000 vram_gb=32.9 elapsed_s=597
```

- **启动（21:45:49）**：`cd <课题> && setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -u scripts/run_all_af24.py > /dev/null 2>&1 < /dev/null &`，调度器 pid 1098113。首批：gs_joint 卡 6、gs_singleFLAIR 卡 7、L1 T1 四个 λ（数据放在卡 6，每个 5.4 GB：当时还没有经典臂在跑，按规则退到空闲显存最多的卡；峰值估 61/80 GB，没有重启调度器）。其余作业等卡，0–5 号卡上是用户别的会话的作业。
- 看进度：`tail logs/run_all_af24.log`、`logs/run_all_af24_status.json`；全部结束会写 `logs/RUN_ALL_AF24_DONE` 和 `logs/finalize_af24.log`。调度器可以安全重启：结果文件在的跳过，正在跑的按签名等待，不会重复起。
- 提交 cd7b28b 的说明写的是启动记录，但当时 STATUS 的改写没写进去（替换断言失败，后面的 git 命令照样执行），实际只含调度日志；启动记录在它之后的提交里。
- 坑：`prep_02_mask_pool_af24.m` 与 `prep_03_pick_mask_af24.py` 都拒绝覆盖，别删池子或掩膜（按秒级 time(0) 取种子，无法重生成）；克隆来的旧调度器 `sched_af24.py` 已删（会用 09-09 的旧 GS 参数和 hyw 课题）；`compare_topics.py` 首次改写把两个课题都写成了 AF24，已修正并核对。

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
