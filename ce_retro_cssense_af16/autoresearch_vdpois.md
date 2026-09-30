# autoresearch:在 vdPoisMex 口径的平扫/增强 AF16 线上拉大 MC-GS 的余量(2026-09-30 开)

分支 `ce-af16-mask-audit`,worktree `/data0/congcong/code/GS/mcgs_ce_mask_wt`,课题根 `$NEW` =
`/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois`。起点是 `VDPOIS_LINE.md` 的十一臂表。

## 目标(用户 2026-09-30)

用 Karpathy 式 autoresearch 循环把 MC-GS 对对比方法的重建质量余量拉大。余量同时报两个:对最强非 GS 臂,对公平口径的 GS-single。

## 起点

| | 平扫 | 增强 |
|---|---|---|
| MC-GS(冻结配方)| 34.452 | 36.596 |
| CS-SENSE(最强非 GS 臂)| 33.978 | 36.371 |
| 登记 GS-single(带 λ_edge 0.001)| 34.350 | 36.416 |
| 余量对 CS-SENSE / 对登记 GS-single | +0.474 / +0.102 | +0.225 / +0.181 |

增强的 HFEN 已输给 CS-SENSE(0.1246 对 0.1233)。

## 为什么有余量

MC-GS 配方从 jyz T1+FLAIR 三期原样搬来,在本数据上没调过;对手臂的 λ 都在本真值上扫过。三条线索:真值是 TV 正则解
(CS-SENSE 的主场),MC-GS 的 TV 类正则加重应更贴真值;vdPoisMex 外围密度 0.02–0.03,高频数据权重 2.2 在放大噪声
(膝线 2026-09-29 测出降到 1.0 是 E2 的转折);真值相位平滑,相位 TV 可加。跨对比度先验那一族,round 0 在 BART 掩膜下
量过天花板 +0.44 dB,放后面。

## 规则(继承 2026-09-24 / 09-28 的用户决定)

1. 规则 A:GS-single = 任何地方最好的单臂(λ_edge=0 + 自身最优),余量对它报。
2. 展示时十一臂全列;其余九臂沿用 `VDPOIS_LINE.md` 的登记结果,不重跑。
3. 一次筛选只动一个旋钮;读数规则先写后跑。
4. 冻结:数据、掩膜、CSM、噪声、真值、评分器、seed 42;点数 360 万、4000 步除非该轮点名。
5. 调参对着真值(与对手臂的 λ 扫描对称;都是同一受试者样本内,N=1,论文要写)。用户 09-30 以"start"接受了这三条建议:
   对着真值、两个余量都报、第一阶段先跑 P1–P5。

## 读数规则(先登记)

两扫描头内 PSNR 都 ≥ +0.05 dB 且 SSIM / NMSE / HFEN 三项都不退,留;一涨一跌,试一次中点;否则弃。
种子地板 ±0.001 dB(前瞻线 GS 换种子 null),+0.05 是信号。评分口径 = `extract_gs_volumes.py` 的头内口径,与十一臂表同一把尺。

## 第一阶段探针(每个 = 一次 MC-GS 联合训练,`ar_jobs_phase1.tsv`)

| 序 | 实验名 | 旋钮 | 现值 → 试值 | 依据 |
|---|---|---|---|---|
| P1 | ar_p1_kh10 | 高频数据权重 `--kspace_high` | 2.2 → 1.0 | 膝线证据;外围稀疏含噪 |
| P2 | ar_p2_tv20 | TV 权重 `--lambda_geo` | 0.10 → 0.20 | 真值是 TV 解 |
| P3 | ar_p3_wav20 | 小波权重 `loss.lambda_wav_plain / lambda_wav` | 0.12/0.10 → 0.20/0.16 | 同方向换变换 |
| P4 | ar_p4_ptv10 | 相位 TV `--lambda_ptv` | 0.03 → 0.10 | 真值相位平滑 |
| P5 | ar_p5_it6000 | 迭代数 | 4000 → 6000 | 收敛检查 |
| 备 | — | 点数 500 万、lambda_diff 0.05、edge_tv_kappa_q 0.70、中心权重 1.0、lambda_edge 0.003 | | 看完 P1–P5 再定 |

同一队列末尾排了公平基线 `ar_s_pre_e0` / `ar_s_post_e0`(冻结配方 + λ_edge 0 的 GS-single),规则 A 的分母。

## 怎么跑

```bash
N=/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois
S=/data0/congcong/code/GS/mcgs_ce_mask_wt/ce_retro_cssense_af16/scripts
cd $N && setsid nohup env RETRO_TOPIC=$N RETRO_TRAINER=$N/trainer/_trainer_e778bf2 PYTHONNOUSERSITE=1 PYTHONPATH= \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -u $S/ar_queue.py --jobs $S/../ar_jobs_phase1.tsv --lanes 5,0 \
  > logs/ar/queue_phase1.out 2>&1 &
```
`ar_queue.py` 一卡一作业,作业落盘后自动用 `extract_gs_volumes.py --key ar_<exp>` 评到头内口径
(`logs/ar_<exp>_<scan>.json`),台账脚本 `ar_ledger.py` 汇总。起跑时只有 GPU 5 空,0–4 被别的作业占到 96–99%,
所以只开两条车道(5 与 0),一次约 1.2 小时(空卡)到 2.5 小时(挤卡),七个作业约 6–8 小时。

## 轮次记录

### 第一阶段(2026-09-30 起跑)

(待填:每个探针两扫描的头内 PSNR / SSIM / NMSE / HFEN 与对冻结配方的差,按读数规则判留弃。)
