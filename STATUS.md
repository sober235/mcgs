# STATUS — 2026-09-14 01:10（worktree `mcgs_af16_wt`，分支 `autoresearch-jyz-af20`）

本分支从 `autoresearch-jyz-af16-phase3` 尖端 f0fe009 切出，承载 jiangyizhou 的**标称 AF20** 研究，**已全部跑完**。
先读 `docs/handoff/2026-09-14-af20-jyz.md`，英文记录 `docs/superpowers/records/2026-09-14-af20-jyz.md`；
课题目录 `/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF20_v3_jiangyizhou/`（仓库镜像 `t1flair_af20_v3_jiangyizhou/`）。
AF16 三期的 STATUS 在它自己的分支与 tag `handoff/2026-09-13-jyz-af16-phase3` 上。

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

## 2. 待用户拍板
- AF20 上 T1 锐度门失守（0.833 对 0.845）：接受，还是为 AF20 单独调（例如相位 TV 加大）——后者就不再是同参数实验。
- 余量不随加速变大（平均 0.461→0.476）：靠提高加速逼近 1.0 dB 这条路，这份证据不支持。
- 留出受试者（huangyuwei 数据现成），AF16、AF20 都没做。
- 掩膜抽样方差：每个 AF 只抽了一张掩膜，且 AF16/AF20 掩膜不嵌套（只重合 698/2427 点）；±0.05 dB 的余量变化是否有意义没法判断。
- `vdPoisMex` 按 `time(0)` 取种子，AF20 掩膜**无法重新生成**，而 `*.mat` 被 gitignore：是否 `git add -f` 这个 4.4 KB 文件。
- 另一个会话 09-13 23:19 给 AF16 课题加了 MC-INR（Niessen 2025，AF16 头内 31.590 / 32.243）；AF20 没跑它。

## 3. 下一步
- 若继续：最便宜且最有信息量的是再抽 2–3 张 AF20 掩膜，只跑 MC-GS joint 和 L1-ESPIRiT（λ 固定在 AF20 最优），量余量的抽样方差。

## 4. 已知的坑
- **不要在本课题 `data/` 里重跑 `prep_02_mask.m`**：会悄悄换掉所有结果依赖的掩膜（md5 1c90baf850a8c1e64b43f6c64146e2c9）。
- 分数 json 先于 ~750 MB 的体积 npz 落地：等 MC-GS 进程退出再登记臂，否则 `BadZipFile`。
- 卡的选择决定墙上时间：ZS-SSDU 挨着别人的长作业慢 10 倍，损失逐位相同；起跑前看 `nvidia-smi --query-compute-apps`，挑没有计算进程的卡。
- 监视脚本里 `pgrep -f` 会匹配到监视器自己，用 `kill -0 <pid>` 判断存活。
- `setup_af20_jyz.sh` 首版把自己删了（在自己的 rm 清单里），已补回修正。
- 工作树目录名 `mcgs_af16_wt` 但分支是 AF20；所有脚本里这个路径是有意保留的。

## 5. 会话边界
- 本分支已推 origin（私有仓库 sober235/mcgs）并打 tag `handoff/2026-09-14-af20-jyz`。不合 main（跨血脉禁止合并）。

## 6. 为什么
- 冻结 MC-GS 参数、只换掩膜，AF16→AF20 的每个差值才能归因于掩膜；经典臂 λ 重扫，保证对手在最佳点。比较对 MC-GS 偏保守。
