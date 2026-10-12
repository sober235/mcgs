# CE 平扫/增强 AF16:对比方法逐臂审计与 MC-GS 第六阶段(2026-10-09 至 10-12 会话总结)

分支 `ce-af16-mask-audit`(本总结所在的推送分支见文末),worktree `/data0/congcong/code/GS/mcgs_ce_mask_wt`,
课题根 `N=/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois`。细节文档:`CMP_ARMS_AUDIT.md`(审计全过程)、
`autoresearch_vdpois.md` 末尾"第六阶段"(MC-GS 继续调参)、`STATUS.md` 第一节(当前状态与待拍板)。

## 结论

1. 对比方法配齐之后,ZS-SSDU 按原文网络规模两个扫描四项指标都是第一,MC-GS 第二。
2. MC-GS 对公平 GS-single 仍是 +0.107 / +0.131 dB,对 CS-SENSE 仍是 +0.712 / +0.569 dB,除 ZS-SSDU 外各口径都第一。
3. 登记的 ZS-SSDU 只有原文网络三分之一的规模;本机所有 mcgs 课题线的 ZS-SSDU 都是同一份代码。
4. 用户要求的"MC-GS 对 GS-single 拉开 0.5 dB 以上"试了减影先验,无效。真值分析表明,规则 A 下这在这份数据上做不到。

## 1. 起点(2068603 为止,已在远端)

- 09-29/30 整线十一臂迁到 vdPoisMex 掩膜(`MASK_SWAP.md`、`VDPOIS_LINE.md`)。
- autoresearch 第 1–5 阶段把 MC-GS 定稿为 `ar_i1_k30p90`:头内 PSNR 34.690 / 36.940,对公平单臂 +0.107 / +0.131,
  59 个病灶层四项全第一。对比方法一直沿用 09-30 的登记结果,没有再查过。

## 2. 对比方法逐臂审计(10-09 至 10-10)

规则:每个臂只按它原来的协议选设置。经典臂、L1-SPIRiT、MC-INR 按真值头内 PSNR 选;3D-DIP、3D-INR 按留出 k 空间残差选;
ZS-SSDU 按共同 10 % Γ 上的损失选。不看真值的三个臂没有用真值挑过任何设置。MC-GS 与公平 GS-single 用 `_ar2` 定稿,本轮未动。

| 臂 | 登记 平扫 / 增强 | 审计后 | 做了什么 |
|---|---|---|---|
| CG-SENSE | 31.553 / 33.623 | 不变 | Tikhonov × 步数扫过,登记点就是最优 |
| CS-SENSE | 33.978 / 36.371 | 不变 | 400 外层只差 +0.007 / +0.017,已收敛;μ 0.1 最好 |
| Joint SENSE | 33.948 / 36.257 | 不变 | 同上,+0.008 / +0.025 |
| L1-SPIRiT | 32.340 / 34.617 | 32.414 / 34.632 | 逐层 SVD 线圈压缩 24 / 32 虚拟线圈,增益在噪声量级 |
| MC-INR | 30.098 / 32.660 | 不变 | 论文 200 步、lr 1e-2 是步数 × lr 族的最优点 |
| 3D-INR | 28.223 / 30.528 | 33.139 / 35.130 | 登记网络加头部支撑与 TV 0.03;TV 曲线两扫描都在 0.03 见底 |
| 3D-DIP | 30.093 / 31.236 | 31.485 / 32.899 | base 96、头部支撑、12000 步、TV 0.03;留出规则选中,真值上比无 TV 低 0.16 / 0.10 |
| ZS-SSDU | 33.177 / 35.138 | 35.319 / 37.527 | 改回原文网络:15 个残差块、CG 10,patience 40 |

核验:确认目录 `Results_20260908/CE_retro_cssense_af16_vdpois_cmp` 里 12 个替换体口径复核 `ALL GAUGE OK`;
三面图目测改进后的臂没有凭空结构;课题测试 31 passed(新增 4 个减影先验单测后共 35 个)。

### 最终十一臂表(头内,确认目录 `logs/readme_tables_cmp.md` 原文)

**pre** (sorted by head-only PSNR; truth = CS-SENSE of the prospective pair)

| method | head PSNR | head SSIM | head NMSE | head HFEN | complex PSNR | whole PSNR | λ / kernel |
|---|---|---|---|---|---|---|---|
| ZS-SSDU | **35.319** | **0.9463** | **0.00883** | **0.1045** | 32.441 | 38.476 |  |
| MC-GS | 34.690 | 0.9385 | 0.01021 | 0.1260 | 31.725 | 38.000 |  |
| GS-single | 34.583 | 0.9376 | 0.01046 | 0.1273 | 31.658 | 37.889 |  |
| CS-SENSE | 33.978 | 0.9306 | 0.01203 | 0.1393 | 31.132 | 37.229 | 0.0005 |
| Joint SENSE | 33.948 | 0.9299 | 0.01211 | 0.1401 | 31.107 | 37.224 | 0.0007 |
| 3D-INR | 33.139 | 0.9158 | 0.01459 | 0.1483 | 30.287 | 36.440 |  |
| L1-SPIRiT | 32.414 | 0.8998 | 0.01724 | 0.1544 | 29.606 | 35.478 | 0.001 / ks5 |
| CG-SENSE | 31.553 | 0.8745 | 0.02102 | 0.2127 | 28.831 | 34.074 |  |
| 3D-DIP | 31.485 | 0.8972 | 0.02135 | 0.1994 | 28.769 | 34.872 |  |
| MC-INR | 30.098 | 0.8085 | 0.02938 | 0.2325 | 27.307 | 30.608 |  |
| ZF | 25.382 | 0.7180 | 0.08704 | 0.6868 | 22.639 | 28.656 |  |

**post** (sorted by head-only PSNR; truth = CS-SENSE of the prospective pair)

| method | head PSNR | head SSIM | head NMSE | head HFEN | complex PSNR | whole PSNR | λ / kernel |
|---|---|---|---|---|---|---|---|
| ZS-SSDU | **37.527** | **0.9578** | **0.00882** | **0.0944** | 34.665 | 40.777 |  |
| MC-GS | 36.940 | 0.9509 | 0.01010 | 0.1111 | 33.968 | 40.224 |  |
| GS-single | 36.809 | 0.9495 | 0.01041 | 0.1150 | 33.908 | 40.099 |  |
| CS-SENSE | 36.371 | 0.9453 | 0.01151 | 0.1233 | 33.501 | 39.584 | 0.00035 |
| Joint SENSE | 36.257 | 0.9435 | 0.01181 | 0.1271 | 33.426 | 39.474 | 0.0005 |
| 3D-INR | 35.130 | 0.9280 | 0.01531 | 0.1371 | 32.214 | 38.427 |  |
| L1-SPIRiT | 34.632 | 0.9173 | 0.01718 | 0.1287 | 31.793 | 37.625 | 0.0005 / ks5 |
| CG-SENSE | 33.623 | 0.9013 | 0.02167 | 0.2089 | 30.984 | 36.282 |  |
| 3D-DIP | 32.899 | 0.8902 | 0.02560 | 0.2033 | 30.231 | 36.301 |  |
| MC-INR | 32.660 | 0.8591 | 0.02705 | 0.2060 | 29.685 | 33.506 |  |
| ZF | 26.931 | 0.7140 | 0.10115 | 0.7152 | 24.457 | 30.147 |  |

**Enhancement (post − pre, white-matter units); err = arm − truth**

| arm | vessels | plexus | cortex | WM | lesion core | rim 0–3 mm | background(post) |
|---|---|---|---|---|---|---|---|
| truth (CS-SENSE) | +2.841 | +1.080 | +0.104 | +0.003 | +0.021 | +0.028 | 0.007 |
| ZF | +1.727 (-1.114) | +0.432 (-0.648) | +0.100 (-0.004) | +0.008 (+0.005) | +0.021 (-0.000) | +0.023 (-0.005) | 0.017 |
| CG-SENSE | +2.475 (-0.366) | +0.808 (-0.272) | +0.101 (-0.003) | +0.002 (-0.001) | +0.027 (+0.006) | +0.028 (+0.000) | 0.030 |
| CS-SENSE | +2.726 (-0.115) | +0.917 (-0.163) | +0.104 (-0.001) | +0.003 (-0.000) | +0.023 (+0.002) | +0.027 (-0.001) | 0.012 |
| Joint SENSE | +2.676 (-0.165) | +0.874 (-0.207) | +0.102 (-0.003) | +0.002 (-0.001) | +0.022 (+0.001) | +0.027 (-0.001) | 0.012 |
| L1-SPIRiT | +2.692 (-0.148) | +0.895 (-0.186) | +0.101 (-0.003) | +0.000 (-0.003) | +0.026 (+0.005) | +0.028 (-0.000) | 0.014 |
| ZS-SSDU | +2.860 (+0.019) | +1.046 (-0.034) | +0.105 (+0.001) | +0.003 (-0.000) | +0.022 (+0.001) | +0.027 (-0.001) | 0.021 |
| 3D-DIP | +2.591 (-0.250) | +0.772 (-0.308) | +0.107 (+0.002) | +0.004 (+0.001) | +0.023 (+0.002) | +0.030 (+0.002) | 0.004 |
| 3D-INR | +2.711 (-0.130) | +0.888 (-0.192) | +0.104 (-0.001) | +0.003 (-0.000) | +0.025 (+0.003) | +0.027 (-0.000) | 0.008 |
| MC-INR | +2.576 (-0.265) | +0.851 (-0.229) | +0.101 (-0.004) | +0.003 (-0.001) | +0.024 (+0.003) | +0.025 (-0.003) | 0.151 |
| GS-single | +2.760 (-0.081) | +0.941 (-0.139) | +0.106 (+0.002) | +0.003 (-0.001) | +0.024 (+0.002) | +0.028 (+0.000) | 0.006 |
| MC-GS | +2.761 (-0.080) | +0.948 (-0.132) | +0.106 (+0.002) | +0.003 (-0.001) | +0.024 (+0.003) | +0.029 (+0.001) | 0.006 |

逐层(十一臂,每扫描 604 层):ZS-SSDU 第一 602 层,MC-GS 2 层;MC-GS 在平扫 599 层、增强 601 层高于除 ZS-SSDU 外的所有臂。
横断 190 个合格层:MC-GS 第一 0 层。59 个病灶层四项全第一:MC-GS 0 层(审计前 59 层),挡在前面的都是 ZS-SSDU。

图(服务目录 `http://localhost:8765/ce_retro_af16/`,文件在 `/home/congcongliu/figs/ce_retro_af16/`,不进 git):
`t1flair_v3_af16_{pre,post}_row3planes_compact_201_131_48_inset48_headerr_cmp_spirit_zf_caps.png`、
`ce_retro_af16_lesion_planes_zoom_t204_c118_s44_byplane_cmp.png`、`ce_retro_af16_enhancement_cmp.png`。只出了 PNG。

## 3. 关键发现:登记的 ZS-SSDU 只有原文三分之一

- SSDU(Yaman 等,MRM 2020)正文:展开 10 次,数据一致性的 CG 展开 10 次,ResNet 为输入 / 输出卷积加 15 个残差块。
- ZS-SSL(ICLR 2022,arXiv 2102.07737 v4 §4.2)同一网络;官方代码 `parser_ops.py` 默认 nb_res_blocks 15、CG_Iter 10、lr 5e-4、
  rho_val 0.2、rho_train 0.4、stop_training 25。
- 登记臂是 5 个残差块、CG 6、patience 10,第 77 epoch 就停了。改回 15 块后两扫描各提高 2.1 / 2.4 dB。
- 官方切分比例(Γ 20 %、Λ 40 %)在增强扫描上与我们的比例结果相同(共同 Γ10 损失 0.4464 对 0.4463)。
- 本机 jyz T1+FLAIR AF12–24、t1t2、diffmask / gauss、噪声、hparam、儿童、膝、CE 前瞻各线的 ZS-SSDU 都是这同一份 5 块代码。
  这些线上"MC-GS 胜 ZS-SSDU"的结论,在用 15 块重跑之前都不能用。

## 4. 第六阶段 a:减影先验(10-10 至 10-11)

用户看完审计后要求继续调 MC-GS 到最优,并对 GS-single 拉开 0.5 dB 以上。规则不变:规则 A、不加可训练先验、不改加速倍数、
不削弱任何对比臂。之前试过的跨对比度项都只耦合边缘位置,所以主杠杆选了从未测过的强度耦合:

```
d = |x_post| - s * |x_pre|          (s 由渲染体的头内中位比值每 500 步重估)
R_sub = sum_head rho(d) / N,  rho = Cauchy(delta): delta/2 * log(1 + (d/delta)^2)
```

训练器补丁 `trainer_patches/2026-10-10-subtraction-prior.diff`(`lambda_sub` 为 0 时一行不走),单测 `scripts/tests/test_sub_prior.py`。

| 探针 | λ_sub | 平扫 Δ i1 | 增强 Δ i1 | 对公平单臂 |
|---|---|---|---|---|
| k1 | 0.03 | −0.005 | −0.001 | +0.102 / +0.130 |
| k2 | 0.1 | −0.062 | −0.064 | +0.044 / +0.067 |

λ 越大越差,0.1 时两扫描四项全退。其余 4 个探针按先登记的规则取消,没有放出。

原因在真值:用整体比例对齐后,两份真值在头内仍差组织亮度的 21 %(中位数),75 % 的体素超过 10 %;
允许平滑比例场后仍有 15–20 %;两者高频部分的相关系数只有 0.45;两次扫描之间没有整体错位。
真值是两次独立采集各自做的 CS-SENSE 重建,不是"同一解剖加稀疏强化"。所以绑强度会强加真值里没有的共同纹理,
绑边缘位置最多值 +0.1 dB(本线与 jyz、膝两线一致)。规则 A 下对公平单臂 > 0.5 dB 在这份数据上达不到。

## 5. 待用户拍板

1. 如实报(建议):MC-GS 第二,卖点改为"不带可训练正则器的方法里最好"。
2. 放开"不加可训练先验",给 MC-GS 接一个只在本扫描上训练的去噪器;可能超过 ZS-SSDU,但规则 A 下单臂也要接,余量仍约 0.1 dB。
3. 把"联合优于单独"放到物理上真耦合的数据上(儿童 MTP 多回波)。
4. 放松规则 A(不建议,审稿人会问单臂为什么没调)。
5. 其它课题线的 ZS-SSDU 要不要按 15 块重跑。
6. 正式图要不要按新臂重渲 600 dpi PDF。
7. 请手动删一个误建目录:`Results_20260908/` 下名为 `CE_retro_cssense_af16_vdpois RETRO_TRAINER=` 的目录(名字带空格,只有 68 KB 日志)。

## 6. 复现

```
N=/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois
WT=/data0/congcong/code/GS/mcgs_ce_mask_wt; PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
# 每个臂按自己的规则选赢家
cd $N && env RETRO_TOPIC=$N PYTHONNOUSERSITE=1 PYTHONPATH= $PY $WT/ce_retro_cssense_af16/scripts/cmp_select.py
# 重建确认目录并重跑全部账目与 PNG 图(只用 CPU;输出目录必须不存在)
cd $N && WAIT="dip_sup_b96_tv0.03_pre" OUT=<新目录> bash $WT/ce_retro_cssense_af16/scripts/cmp_final_chain.sh
# 第六阶段读数
cd $N && env RETRO_TOPIC=$N PYTHONNOUSERSITE=1 PYTHONPATH= $PY $WT/ce_retro_cssense_af16/scripts/ar6_ledger.py ar_k1 ar_k2
# 测试
cd $WT/ce_retro_cssense_af16 && env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= $PY -m pytest scripts/tests -q
```

新增或修改的脚本:`cmp_queue.py`、`cmp_ledger.py`、`cmp_select.py`、`cmp_gauge_check.py`、`make_cmp_topic.py`、`confirm_cmp.sh`、
`cmp_final_chain.sh`、`ar_free_queue.py`、`ar6_ledger.py`、`phase6a_chain.sh`;重建脚本 `recon_spirit.py`(`--vcoils`)、
`recon_zsssdu.py`(`--blocks --rho_val --theta_frac`)、`recon_dip_inr.py`(INR 支撑、`--wd --cosine`)、`run_mcinr.py`(`--tag --score_subset`),
登记行为全部不变。mcinr 仓库另有一个提交 25110dc(`--n_epochs --lr`)。交接标签:`handoff/2026-10-10-ce-cmp-arms`、`handoff/2026-10-11-ce-phase6a`。

## 7. 坑

- 共享 GPU:别的会话的作业在读数据时卡看起来是空的,我们两次撞卡。现在 `run_gs.sh` 一启动就导出 `CUDA_VISIBLE_DEVICES`,
  训练器一启动就占住 CUDA 上下文,空卡队列跳过任何进程在环境变量里认领的卡,并要求卡连续空 10 分钟。
- 不同大小的 Γ 上的损失不能直接比,ZS-SSDU 一律比共同 10 % Γ。
- 网络先验臂加头部支撑后,整体 HFEN 被支撑边界的台阶拉高,只看头内指标。
- zsh 不对 `$VAR` 分词;pgrep 模式别包含自己命令里的字符串;队列启动后改清单无效。

## 8. 推送

本总结与 2068603 之后的全部提交(对比方法审计 33 个提交加本总结)推到新分支 `ce-af16-cmp-arms-phase6-2026-10-12`,基于 `ce-af16-mask-audit`,未合进 main。
