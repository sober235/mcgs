# 掩膜换口径对照:BART poisson -v → vdPoisMex(2026-09-29)

新会话先读本文件,再读 `audit_2026-09-29/ALIASING_AUDIT.md`。分支 `ce-af16-mask-audit`,worktree
`/data0/congcong/code/GS/mcgs_ce_mask_wt`。

## 0. 一句话

审计发现这条线的 AF16 掩膜是 `bart poisson -v`,ACS 外密度几乎均匀,与 jyz 线的 vdPoisMex 不是一个口径。
本对照把掩膜换成 house 口径(vdPoisMex,同 2552 点、同 24×24 全采 ACS),其余一切不动,重跑 ZF、CG-SENSE、
CS-SENSE、GS-single、MC-GS,看中频残差与头内指标各变多少。

## 1. 课题目录(服务器数据,不进 git)

`$NEW` = `/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois`

| 项 | 来源 |
|---|---|
| `sim/gs/`(无掩膜含噪 k 空间 + 42 线圈图)| 软链到原课题 `CE_retro_cssense_af16/sim/gs/`,噪声实现逐位相同 |
| `sim/gt.npz`、`sim/meta.json` | 软链 / 复制原课题 |
| `sim/mask/` | **新**,见 §2 |
| `trainer/_trainer_e778bf2/` | 原课题训练器快照的代码部分(rsync 排除 `results_arc`),同一提交 e778bf2 |
| `scripts` → 本 worktree `ce_retro_cssense_af16/scripts`;`configs` → 本 worktree `configs_vdpois/` | 三份 yaml 只改路径与 exp 名(`ce_retro_af16vd_*`),其余逐字同原配置 |

所有脚本通过 `RETRO_TOPIC=$NEW`(训练器用 `RETRO_TRAINER=$NEW/trainer/_trainer_e778bf2`)切换课题根。

## 2. 掩膜(`scripts/make_mask_af16_vdpois.py`)

MATLAB `vdPoisMex(232,176,232,176,accelx,4,24,0,1)`(与 jyz `prep_02_mask.m` 同设置:accely 4、calib 24、ellipse 0、pp 1),
accelx 扫 3.6–4.8。vdPoisMex 的 calib 只把中心密度抬到 0.87,所以脚本把 24×24 ACS 强制全采,再取"强制后点数 ≥ 2552
中最小者",多余点从最外圈往里剪。选中 accelx 4.45:原始 2477 点 + ACS 补 75 点 = 恰好 2552,**零剪除**。
vdPoisMex 不能定种子,原始抽样 `vdpoismex_raw.mat` 与最终 `mask_af16_pe.npy` 一起存在 `audit_2026-09-29/mask_vdpois/`,
不用 MATLAB 也能逐位复现后续一切。

径向密度:1.00(r<0.1)、0.78(0.12)、0.35(0.18)、0.23(0.23)、0.17(0.28)、0.14(0.33)、0.10(0.38)、…、0.02(边缘)。
原掩膜同位置:1.00、0.61、0.09、0.06、0.06、0.06、0.06、…、0.05。

## 3. 门(`audit_2026-09-29/mask_vdpois/check_sim.json`,GPU 5,已过)

| 扫描 | 伴随相对误差 | 无噪全采 CG-SENSE 头内 PSNR | σ 写入 vs meta | ZF 头内 PSNR(新掩膜)| ZF(原掩膜)|
|---|---|---|---|---|---|
| 平扫 | 3.81e-7 | 69.19 | 4.3e-5 | 25.38 | 24.25 |
| 增强 | 3.84e-7 | 75.91 | 1.6e-5 | 26.93 | 25.83 |

## 4. 四臂(2026-09-29 14:31 起跑,一卡一训练)

```bash
NEW=/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois
S=/data0/congcong/code/GS/mcgs_ce_mask_wt/ce_retro_cssense_af16/scripts
RETRO_TOPIC=$NEW bash $S/run_vdpois_classical.sh 1                                   # ZF+CG-SENSE 两扫描,CS-SENSE o200 七点 λ 网格两扫描,GPU 1 串行
RETRO_TOPIC=$NEW bash $S/run_gs.sh ce_retro_af16vd_joint 4 4000                      # MC-GS,GPU 4
RETRO_TOPIC=$NEW SINGLE=pre  bash $S/run_gs.sh ce_retro_af16vd_single_pre  3 4000    # GS-single 平扫,GPU 3
RETRO_TOPIC=$NEW SINGLE=post bash $S/run_gs.sh ce_retro_af16vd_single_post 0 4000    # GS-single 增强,GPU 0
```
配方与原课题逐字相同(jyz AF16 三期冻结配方,360 万点,4000 步,seed 42),CS-SENSE λ 网格也与原课题相同,由
`summarize.py` 按头内 PSNR 各扫描选各的。日志:`$NEW/logs/classical_chain.log`、`$NEW/trainer/_trainer_e778bf2/results_arc/*.log`,
PID 在 `$NEW/logs/launch/pids.txt`。

收尾命令(训练完成后):
```bash
E="RETRO_TOPIC=$NEW RETRO_TRAINER=$NEW/trainer/_trainer_e778bf2 PYTHONNOUSERSITE=1 PYTHONPATH="
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
env $E $PY $S/extract_gs_volumes.py --exp ce_retro_af16vd_joint       --key gs_joint  --map pre:plain,post:contrast
env $E $PY $S/extract_gs_volumes.py --exp ce_retro_af16vd_single_pre  --key gs_single --map pre:plain
env $E $PY $S/extract_gs_volumes.py --exp ce_retro_af16vd_single_post --key gs_single --map post:plain
env $E $PY $S/summarize.py
env $E $PY $S/compare_masks.py            # 两课题并排:头内指标 + 分带 NMSE + 图 -> $NEW/logs/mask_swap_compare.json
```

## 5. 结果

(训练中,待填。)

## 6. 待用户拍板

- 这条对照只回答"掩膜口径贡献多少",不改正式结果。要不要把整条线(11 臂)迁到 vdPoisMex 口径重做,你定。
- 其余六臂(Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR)本轮没跑。

## 7. 坑

- `run_gs.sh` 拒绝覆盖已有输出;重跑先手动挪走 `results_arc/<exp>*`。
- GPU 是共享的,四张卡起跑时利用率 12–75%;单步耗时若涨几十倍,先怀疑撞卡,换卡重启。
- `sim/work_*/`(掩膜后的 k 空间副本)本课题没建:`Pair` 从 `sim/gs/*.h5` 读无掩膜数据、自己套 `sim/mask/mask_af16_3d.h5`,
  训练器同样;只有 ZS-SSDU / MC-INR 那几条路径读 `work_*`,本轮不跑。
