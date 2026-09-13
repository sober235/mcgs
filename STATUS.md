# STATUS — 2026-09-13 22:15（worktree `mcgs_af16_wt`，分支 `autoresearch-jyz-af20`）

本分支从 `autoresearch-jyz-af16-phase3` 尖端 f0fe009 切出，承载 jiangyizhou 的**标称 AF20** 研究。
AF16 三期的 STATUS 留在它自己的分支与 tag `handoff/2026-09-13-jyz-af16-phase3` 上，这里不再重复。
课题目录 `/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF20_v3_jiangyizhou/`（仓库镜像 `t1flair_af20_v3_jiangyizhou/`），先读它的 README。

## 一句话
AF16 三期配方**原样不调**跑在 AF20 掩膜上：头内 PSNR **34.832 / 34.179**，对最强对手 L1-ESPIRiT（λ 在 AF20 上重扫）
余量 **+0.461 / +0.491**（AF16 是 +0.497 / +0.425），头内 PSNR/SSIM/HFEN 六格全第一（目前 7 臂）；
**T1 锐度 0.833 低于 L1-ESPIRiT 的 0.845**（AF16 时高于），FLAIR 锐度仍高于。余量基本持平，没有随加速变大。

## 1. 已完成且已验证

唯一变量是掩膜。`vdPoisMex(256,176,256,176, 5, 4, 24, 0, 1)`（accelx 5，其余房子惯例不变）：

```
$ PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
$ PYTHONNOUSERSITE=1 PYTHONPATH= $PY -c "
import json;d=json.load(open('/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF20_v3_jiangyizhou/logs/prep_04_export.json'))
print('mask', d['mask2d_samples'], 'samples, AF_rect', round(d['mask2d_AF_rect'],3))
for c,v in d['contrasts'].items(): print(c, 'support', round(v['support_frac'],4), 'AF_own', round(v['AF_vs_own_support'],2), 'acs24_kept', round(v['acs24_kept'],3))"
mask 2427 samples, AF_rect 18.564
T1 support 1.0 AF_own 18.56 acs24_kept 0.823
FLAIR support 0.7577 AF_own 15.13 acs24_kept 0.823
```

结果（重跑核对，2026-09-13 22:15）：

```
$ PYTHONNOUSERSITE=1 PYTHONPATH= $PY -c "
import json;R='/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF20_v3_jiangyizhou/logs/summary.json'
for r in json.load(open(R)):
    if r['method'] in ('gs_joint_p3','gs_single_p3','l1espirit','cssense_o200','jcssense'):
        lam=f\" lam {r['lam']:g}\" if r.get('lam') else ''
        print(f\"{r['contrast']:5s} {r['method']:13s} head {r['head_psnr']:.3f} ssim {r['head_ssim']:.4f} hfen {r['head_hfen']:.4f} sharp {r['sharp']:.3f}{lam}\")"
FLAIR cssense_o200  head 33.494 ssim 0.9049 hfen 0.1523 sharp 0.698 lam 0.0005
FLAIR jcssense      head 33.544 ssim 0.9044 hfen 0.1522 sharp 0.726 lam 0.0007
FLAIR l1espirit     head 33.688 ssim 0.9076 hfen 0.1406 sharp 0.732 lam 0.003
FLAIR gs_joint_p3   head 34.179 ssim 0.9142 hfen 0.1290 sharp 0.776
T1    cssense_o200  head 34.003 ssim 0.9191 hfen 0.1683 sharp 0.781 lam 0.0007
T1    jcssense      head 33.885 ssim 0.9183 hfen 0.1733 sharp 0.768 lam 0.001
T1    l1espirit     head 34.371 ssim 0.9206 hfen 0.1518 sharp 0.845 lam 0.001
T1    gs_joint_p3   head 34.832 ssim 0.9289 hfen 0.1355 sharp 0.833
```

- 配方逐项核对过：AF20 运行日志的 `[phase-tv] 0.03`、`[wavelet] db4 levels=1 plain 0.12 contrast 0.1`、`[tv_shape] eps 0.04`、
  `[edge_tv] kappa_q 0.85 source zf`、`[init_cloud] head 3600000 3nn` 与 AF16 三期 `p3_r3_e0001_ptv003_full4k` 逐行相同。
- 经典臂 21 个运行，六条 λ 曲线全是内部极值（T1 的 L1-ESPIRiT 和 jointTV 首轮贴下边界，已往下扩）。
- AF16→AF20 掉幅：MC-GS −0.874 / −0.749，L1-ESPIRiT −0.838 / −0.815。SSIM 领先 +0.0083 / +0.0066、HFEN 领先 0.0163 / 0.0116，都比 AF16 大。
- 对比脚本：`scripts/compare_topics.py`（AF16 对 AF20，余量、耦合项、名次），输出在 `logs/compare_af16_af20.txt`。
- 复现：见课题 README 的 Regenerating 节；MC-GS 配方只在 `run_af20_mcgs.sh` 一处。

## 2. 待用户拍板
- AF20 上 T1 锐度门失守（0.833 对 0.845）。接受，还是在 AF20 上单独调（例如相位 TV 往上买锐度）——后者就不再是"同参数"实验。
- 余量没有随加速变大（平均 0.461→0.476）。1.0 dB 目标这条路（靠提高加速拉开差距）是否就此放弃。
- 留出受试者（huangyuwei 数据现成）：AF16 一期至今、AF20 都没做。
- 一个只当观察的巧合：AF20 FLAIR（对快门 15.13×）余量 +0.491 ≈ AF16 T1（15.19×）余量 +0.497。是否值得在留出受试者上验证"余量由有效加速决定"。

## 3. 下一步（正在后台跑，不需要人）
- GS-single FLAIR 已完成（22:35，头内 34.127，FLAIR 耦合 +0.052 dB，AF16 是 +0.042）；GS-single T1（7 号卡，约 01:15）→ T1 耦合。
  登记：`scripts/finalize_af20_jyz.sh af20_p3_joint_full4k af20_p3_singleT1_full4k af20_p3_singleFLAIR_full4k`
- 第二段 `scripts/run_stage2_af20.sh` v2（22:39 起）：0 号卡 ZS-SSDU T1→INR T1，2 号卡 ZS-SSDU FLAIR→INR FLAIR，3 号卡 DIP T1→DIP FLAIR，参数与 AF16 课题日志一致；v1 放在 5/6 号卡上慢 10 倍已停（日志在 logs/stage2_aborted/）；
  约明早 8–10 点跑完。九臂齐了 finalize 才会出紧凑三面图（出图脚本断言九臂齐全）。
- 进度：`logs/stage2_af20.log`、`results_arc/af20_runs.log`。

## 4. 已知的坑
- 分数 json 先于体积 npz 落地（npz 约 750 MB，要写一分半钟）；登记臂之前等进程退出，否则 `BadZipFile`。
- 6/7 号卡邻居是别人高算力小显存作业：MC-GS 实测 4.6–5.9 s/步（AF16 独占卡是 1.0 s/步），墙上时间不可比。
- FLAIR 已知零先验在 AF20 上弱一些：零线 170 条只解释 144 条（AF16 233/233）。代码保证不会把实测非零线标成零（`nonzero_in_z=0`）；三期测过该先验约值 −0.013 dB。
- `setup_af20_jyz.sh` 首版把自己删了（在 rm 清单里），已补回并修正；若重跑 setup 用现版本。
- 本工作树目录名是 `mcgs_af16_wt` 但分支是 AF20；所有脚本里的这个路径是有意保留的。

## 5. 会话边界
- 本分支不合 main（跨血脉禁止合并，同 AF16 三条分支）。最新状态在分支尖端。

## 6. 为什么
- 冻结 MC-GS 参数、只换掩膜，是为了让 AF16→AF20 的每一个差值都只归因于加速。代价是比较对 MC-GS 偏保守：经典臂在 AF20 重扫了 λ，MC-GS 没调。
- 经典臂 λ 必须重扫：它们在 AF16 就是在本受试者真值上选的 λ，AF20 不重扫就等于人为削弱对手。
