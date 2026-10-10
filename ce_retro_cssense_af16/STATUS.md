# STATUS

新会话先读本文件，再读 README，关键数字先重跑。

## ⏸ 2026-10-10 08:31 暂停中(用户要求先暂停、处理完别的事再接着做)——恢复先看这里

**挂起的两个 GPU 作业**(SIGSTOP,进程与进度都在,显存仍占着:卡 1 约 62 GB、卡 6 约 29 GB,别的队列用不了这两张卡):

| PID | 卡 | 作业 | 挂起时进度 | 输出(出分后才有) |
|---|---|---|---|---|
| 3761893 | 1 | 平扫 3D-DIP + TV 0.03(base 96、支撑 6、lr 3e-3、12000 步) | 11800 / 12000 步,留出残差 0.16399 | `logs/dip_sup_b96_tv0.03_pre.json` |
| 3761894 | 6 | 平扫 ZS-SSDU 官方切分比例 z3(15 块、CG 10、Γ 0.2、Θ 0.6、patience 25、≤ 300 epoch) | 第 60 / 300 epoch,自身 Γ 0.4536 | `logs/zsssdu_z3_official_pre.json` |

日志:`$N/logs/cmp/dip_pre_b96_sup_tv0.03_r2.log`、`$N/logs/cmp/zs_pre_z3_official_r2.log`。收尾链与后台等待器已停(没有任何东西会自动启动)。
其余作业全部已出分并落盘(`$N/logs/*.json`),审计代码与文档全部在 git(本分支)。

**恢复步骤**(`N=/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois`,`WT=/data0/congcong/code/GS/mcgs_ce_mask_wt`,
`PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python`):

1. 看进程还在不在:`ps -o pid=,stat=,etime= -p 3761893,3761894`(stat 里有 `T` = 挂起中)。
2. **A. 进程还在**:`kill -CONT 3761893 3761894`;过一两分钟 `tail -1 $N/logs/cmp/dip_pre_b96_sup_tv0.03_r2.log` 应在走(DIP 约 4 分钟后出分),
   `grep "\[ep" $N/logs/cmp/zs_pre_z3_official_r2.log | tail -1` 的 epoch 在涨(约 80 秒一个,预计还要 1.5–3.5 小时,patience 25 会提前停)。
3. **B. 进程没了**(重启、被杀):先 `nvidia-smi` 找两张空卡并 `ps` 确认没有别的会话刚起的作业,然后从头重跑(结果确定性同设置,DIP 约 3.7 小时、ZS ≤ 6.5 小时):
   ```
   cd $N
   setsid nohup env RETRO_TOPIC=$N RETRO_TRAINER=$N/trainer/_trainer_e778bf2 PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID \
     $PY -u $WT/ce_retro_cssense_af16/scripts/recon_dip_inr.py --method dip --scan pre --gpu <卡A> --iters 12000 --lr 3e-3 \
     --eval_every 200 --support 6 --dip_base 96 --tv 0.03 --suffix _sup_b96_tv0.03 > logs/cmp/dip_pre_b96_sup_tv0.03_r3.log 2>&1 < /dev/null &
   setsid nohup env RETRO_TOPIC=$N RETRO_TRAINER=$N/trainer/_trainer_e778bf2 PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID \
     $PY -u $WT/ce_retro_cssense_af16/scripts/recon_zsssdu.py --scan pre --gpu <卡B> --blocks 15 --cg_iters_net 10 --rho_val 0.2 \
     --theta_frac 0.6 --patience 25 --epochs 300 --suffix _z3_official > logs/cmp/zs_pre_z3_official_r3.log 2>&1 < /dev/null &
   ```
   ZS 这一项是可选的(增强已证明官方比例与我们的比例结果相同);跳过它就在第 4 步加 `WAIT="dip_sup_b96_tv0.03_pre"`。
4. 重起收尾链(等上面的结果文件出现,默认最多等 12 小时;然后按各臂规则选赢家、建 `Results_20260908/CE_retro_cssense_af16_vdpois_cmp`、
   口径复核、十一臂表、强化保真、逐层、病灶层、PNG 图):
   `cd $N && setsid nohup bash $WT/ce_retro_cssense_af16/scripts/cmp_final_chain.sh >> logs/cmp/final_chain.log 2>&1 < /dev/null &`
   完成标志:`$N/logs/cmp/final_chain.log` 里出现 `FINAL_CHAIN_DONE`;确认目录里 `logs/gauge_check_cmp.txt` 末行应是 `ALL GAUGE OK`。
5. 收尾后:用确认目录的 `logs/readme_tables_cmp.md`、`find_best_slice_cmp.out`、`best_slice_mcgs_cmp.out`、`lesion_allmetrics_cmp.out`
   替换下面一节的中期表,写进 `CMP_ARMS_AUDIT.md`,再把"待用户拍板"交给用户。

## 2026-10-10 对比方法逐臂审计(本节最新;下面 09-29 起的段落是历史)

用户 10-09:逐个检查对比方法有没有提升空间,有就适当提,别让审稿人一眼看出对比方法实现不对。入口 `CMP_ARMS_AUDIT.md`
(逐臂诊断、出分前写定的选择规则、每个探针的数、全部过程)。worktree 根目录的 `../CLAUDE.md` 是新会话入口。
确认目录 `Results_20260908/CE_retro_cssense_af16_vdpois_cmp` 由收尾链 `scripts/cmp_final_chain.sh` 建
(等平扫三个作业出分,最晚 15:00),**建好之前本节的表是中期数,平扫 DIP 与平扫 ZS 两格可能还会变**。

### 已完成且已验证

规则:每臂只按自己原本的协议选——经典臂、L1-SPIRiT、MC-INR 看真值头内 PSNR(与登记时一样);3D-DIP、3D-INR 看留出 k 空间残差;
ZS-SSDU 看共同 10 % Γ 上的损失。不看真值的三臂没有用真值挑过任何设置。MC-GS 与公平 GS-single 用 `_ar2` 定稿,本轮未动。

| 臂 | 登记(头内 PSNR 平扫 / 增强) | 审计后 | 改了什么 |
|---|---|---|---|
| CG-SENSE | 31.553 / 33.623 | 不变 | Tikhonov × 步数扫过,登记点就是最优 |
| CS-SENSE | 33.978 / 36.371 | 不变 | 400 外层 +0.007 / +0.017(已收敛),μ 0.1 最好 |
| Joint SENSE | 33.948 / 36.257 | 不变 | 同上,+0.008 / +0.025 |
| L1-SPIRiT | 32.340 / 34.617 | 32.414 / 34.632 | 逐层 SVD 线圈压缩 24 / 32 虚拟线圈,增益在噪声量级 |
| MC-INR | 30.098 / 32.660 | 不变 | 论文 200 步 lr 1e-2 是步数 × lr 族的最优点 |
| 3D-INR | 28.223 / 30.528 | 33.139 / 35.130 | 登记网络(nmax 512)+ 头部支撑 + TV 0.03(两扫描留出曲线都在 0.03 见底) |
| 3D-DIP | 30.093 / 31.236 | 31.643(待平扫 + TV 出分) / 32.899 | base 96 + 头部支撑 + 12000 步;增强按留出规则选了 + TV 0.03(真值上比无 TV 低 0.10) |
| ZS-SSDU | 33.177 / 35.138 | 35.319 / 37.527 | 原文网络规模:15 残差块、CG 10、patience 40(两扫描都选中这一版) |
| GS-single / MC-GS | 34.583 / 36.809、34.690 / 36.940 | 不变 | `_ar2` 定稿 |

- ZS-SSDU 的网络规模按一手来源核对过:SSDU(MRM 2020)正文、ZS-SSL(ICLR 2022)§4.2、官方代码 `parser_ops.py` 默认值都是 15 个残差块、
  展开 10、CG 10。官方切分比例(Γ 20 %、Λ 40 %)的增强对照与我们的比例结果相同(共同 Γ10 损失 0.4464 对 0.4463)。
- 中期确认目录(scratchpad)里 12 个替换体 `ALL GAUGE OK`;课题测试 31 passed;mcinr 仓库 27 passed 1 skipped。
- 三面图目测(`~/figs/ce_retro_af16/*_cmp_interim_*`):改进后的臂没有凭空结构,误差图均匀;MC-INR 仍有明显颗粒(论文配置本身)。
- 复现选择:`cd $N && env RETRO_TOPIC=$N PYTHONNOUSERSITE=1 PYTHONPATH= $PY $WT/ce_retro_cssense_af16/scripts/cmp_select.py`
  (`N=/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois`,`WT=/data0/congcong/code/GS/mcgs_ce_mask_wt`)。

### 待用户拍板

1. **ZS-SSDU 配到原文规模后,两扫描四项指标都超过 MC-GS 定稿**(平扫 35.32 对 34.69,增强 37.53 对 36.94;中期逐层第一 598 / 576 层,
   MC-GS 6 / 28 层)。登记的 5 块 ZS-SSDU 是三分之一规模的网络,审稿人对照原文就能看出来,所以不能退回旧臂。可选:
   (a) 如实报新排序,MC-GS 的卖点改为"无训练网络的方法里第一"(对 CS-SENSE +0.71 / +0.57 不变)或其它维度;
   (b) 再给 MC-GS 找杠杆(用户此前限定不加可训练先验,同掩膜下的非学习先验已基本封口);(c) 其它。建议 (a)。
2. **全线 ZS-SSDU 都是同一份 5 块代码**:jyz T1+FLAIR AF12/16/20/24、t1t2、diffmask / gauss、噪声、hparam、儿童、膝、CE 前瞻。
   这些线上"MC-GS 胜 ZS-SSDU"的说法在 15 块重跑之前都不能用。要不要重跑、先跑哪几条,用户定(每条线每对比度约 3–5 GPU 小时)。
3. 论文里两个网络先验臂的名字与说明:"3D-INR + 头部支撑 + TV"、"3D-DIP(base 96)+ 头部支撑(+ TV)",TV 权重按留出 k 空间残差选。
4. 正式图要不要按新臂重渲 600 dpi PDF(每张约 140 GB 内存,串行);目前只出了 PNG。
5. push:本轮没有推任何东西。
6. 请手动删一个误建的目录(名字里带空格,只有 68 KB 队列日志):`Results_20260908/CE_retro_cssense_af16_vdpois RETRO_TRAINER=`。

### 下一步

收尾链出结果后:用最终表替换上面的中期表,写进 `CMP_ARMS_AUDIT.md`;按用户对第 1、2 条的决定再动。

### 已知的坑与别重做什么

- 共享卡:别的会话的队列刚起的作业在读数据时,`nvidia-smi` 看着是空卡;放作业前要 `ps` 看本用户刚起的训练进程。重建脚本现在一启动就占住 CUDA 上下文。
- 不同大小的 Γ 上的损失不能直接比;ZS-SSDU 一律比"共同 10 % Γ"(`val_loss_gamma10`)。
- 3D-INR / 3D-DIP 加头部支撑后整体 HFEN 被支撑边界的台阶拉高,只看头内指标。
- `cmp_queue.py` 启动时读清单,改清单要重起队列;zsh 不对 `$VAR` 分词;pgrep 模式别包含自己命令里的字符串。
- 别重做:CG-SENSE 的 Tikhonov × 步数、CS-SENSE / Joint SENSE 的收敛与 μ、MC-INR 的步数 × lr、L1-SPIRiT 的线圈压缩,都已封口。

### 关键决定的为什么

- 每臂按自己原本的协议选,不看真值的方法不拿真值挑:这样审稿人挑不出"给对手调参不对称"的毛病,也不会被说成替对手作弊。
- CS-SENSE / Joint SENSE 保留 200 外层的登记运行:400 外层只差 ≤ 0.025 dB,改登记要连带改文件名规则,不值。
- 3D-DIP 给与 3D-INR 同样的 TV 项:两臂都是无训练网络先验,只给一个加正则会被问。
- MC-INR 只扫步数与 lr:用户此前定过复现只做方法本身,不做互补掩膜与 PICS。
- 撞卡时只停自己的作业,不动其它会话的作业。

`$TOPIC` = `/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16`，`$WT` =
`/data0/congcong/code/GS/mcgs_ce_retro_wt`，`$PY` = `/home/congcongliu/anaconda3/envs/GSTMI/bin/python`。

> **2026-09-29 补充（分支 `ce-af16-mask-audit`，worktree `mcgs_ce_mask_wt`）**：混叠审计与掩膜换口径对照。
> 先读 `MASK_SWAP.md`，再读 `audit_2026-09-29/ALIASING_AUDIT.md`。要点：没有相干混叠，但 11 臂的非相干残差都重
> （MC-GS 头内 rms 误差 3.1% 峰值、r 0.2–0.4 带 NMSE 7%）；掩膜是 `bart poisson -v`，ACS 外密度几乎均匀，与 jyz 的
> vdPoisMex 不是一个口径。换成 vdPoisMex 掩膜（同 2552 点、同 ACS）重跑五臂：经典臂升 +2.6 到 +3.7 dB，GS 臂只升
> +0.8 到 +1.1 dB；MC-GS 仍第一（34.45 / 36.60），但对 CS-SENSE 的余量从 +2.6 / +2.9 缩到 +0.47 / +0.22 dB，
> 对 GS-single 从 +0.19 / +0.35 缩到 +0.10 / +0.18。**09-30 整线十一臂已迁到 vdPoisMex**(`VDPOIS_LINE.md`;MASK_SWAP.md §9):
> MC-GS 仍 PSNR/SSIM/NMSE 全第一,最强对手回到 CS-SENSE(+0.47 / +0.22),逐层第一 599/604、583/604。下面的正文是 09-18 的状态，未改。

## 已完成且已验证

Task 16 是收尾：口径互验 → 汇总/强化保真/最优层 → 两张图 → README/STATUS 填数 → 导出包 → 落盘提交。全量调度已在
本任务开始前跑完（`logs/orch/ALL_DONE_retro16`，07:58 写出），本任务没有再启动调度器，也没有碰任何 GPU 训练作业。

**口径互验（brief Step 3，CPU，`--gpu -1`）**：对不经 `recon_classical.py` 内部自检的七个臂 × 两个扫描，
用 `scripts/score_volume.py --check` 重评并比对各自记录的 json，容差 2e-4（相对）：

```
cd $TOPIC
for s in pre post; do for k in cgsense zsssdu dip inr mcinr gs_single gs_joint; do
  f=recon/${k}_${s}_complex.npy
  jf=logs/$( [ $k = cgsense ] && echo classical_$s || echo ${k}_$s ).json
  env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/score_volume.py --scan $s --file $f --method $k --check $jf --gpu -1 | tail -1
done; done
```

输出：**14/14 `GAUGE OK`，0 个 MISMATCH**（cgsense/zsssdu/dip/inr/mcinr/gs_single/gs_joint × pre/post）。逐项
`recorded=` 与 `rescored=` 在小数点后 5～6 位内完全一致；完整逐项输出在 `task-16-report.md`。**（whole-branch
review 2026-09-18）** 这份"14/14 GAUGE OK"此前唯一的痕迹是 `task-16-report.md`，而那份文件在
`.superpowers/sdd/.gitignore` 里被通配符排除、不进 git；已重跑同一套 `--gpu -1` 循环（完整 stdout，不只
`tail -1`），落盘到 `logs/gauge_check.txt` 并进了 git（本次结果同样 14/14 `GAUGE OK`、0 MISMATCH，数字与上面
逐字一致，容差没有改动过）。

**汇总 / 强化保真 / 最优层**：

```
cd $TOPIC
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/summarize.py             # -> logs/summary.json (22 行), summary.tsv
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/enhancement_fidelity.py  # -> logs/enhancement_fidelity.json
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/find_best_slice.py       # -> logs/slice_scores.json
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/best_slice_mcgs.py --axis 0   # -> logs/best_slice_mcgs.json
```

输出：`summary.json` 22 行（11 臂 × 2 扫描），与 README §4 表格逐字一致。头条余量 **MC-GS − GS-single：pre
+0.194 dB / post +0.354 dB**（head PSNR）；MC-GS 对最强非 GS 臂：pre 比 ZS-SSDU 高 +1.527 dB，post 比 L1-SPIRiT
高 +1.588 dB。

**（whole-branch review 2026-09-18 修复）** `find_best_slice.py`/`best_slice_mcgs.py` 原来各自的臂列表都只覆盖
8/11 臂（前者缺 Joint SENSE、L1-SPIRiT、MC-INR；后者缺 ZF、L1-SPIRiT、MC-INR），下面这段逐层数字之前其实只跟
8 臂比过，不是全场。两个脚本现在都改成走 `arms.arm_files()`（arms.py 的 ORDER/LABEL，唯一权威）取文件，重跑：
`find_best_slice.py`（全部三轴、逐层、11 法比较）：pre 604 个可评层中 MC-GS 排第一的仍是 602 层（另 2 层第二，
胜者分别是 gs_single 和 zsssdu 各 1 层）；post 仍 604/604 层全部第一——新增的 Joint SENSE/L1-SPIRiT/MC-INR 一层
都没抢到第一，两个计数都不变。`best_slice_mcgs.py --axis 0`（横断面，190 个合格层，头部面积 ≥ 最大值一半）：
MC-GS 在 190/190 层上都超过 GS-single 与其余十臂，两个扫描皆然，也不变；最优层仍是 218，对 GS-single 余量
pre +0.402 / post +0.646 dB 不变。但"对最强非 GS 臂"的余量变了：pre 仍是 ZS-SSDU +1.517 dB（不变）；**post
原来写的 CS-SENSE +2.409 dB 是错的——CS-SENSE 根本没有跟 L1-SPIRiT 比过。11 臂全部算上后，218 层 post 的最强
非 GS 臂是 L1-SPIRiT（35.399 dB），MC-GS 对它的余量是 +1.550 dB。**

**两张图（brief Step 4，纯 CPU，无 `--gpu` 参数可传——三个绘图脚本本来就不含 GPU/torch 代码）**：

```
cd $TOPIC
for s in pre post; do
  env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/fig_three_planes_compact.py --scan $s --slices 201 131 48 \
    --inset 48 --inset_corner tr bl bl --mask_error --gray_cbar --cbar_planes 0 --tra_rot 1 --err_vmax 0.1 \
    --with_zf --caps --l1_arm spirit --pdf --pdf_dpi 600 --tag _201_131_48
done
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/fig_enhancement.py
```

三面图脚本按自己的命名写文件（`t1flair_v3_af16_<scan>_row3planes_compact_..._spirit_zf_caps.{png,pdf}`），已
`cp` 成 README §5 要求的 `figures/ce_retro_af16_three_planes_{pre,post}_201_131_48.{png,pdf}`（两个目录都留了
两个名字）；`fig_enhancement.py` 本来就直接写 `ce_retro_af16_enhancement.{png,pdf}`。三张 PNG 已拷到
`/home/congcongliu/figs/ce_retro_af16/`，`curl` 探活全部 200：

```
$ for f in ce_retro_af16_three_planes_pre_201_131_48.png ce_retro_af16_three_planes_post_201_131_48.png ce_retro_af16_enhancement.png; do
    printf "%-45s " "$f"; curl -s -o /dev/null -w "%{http_code}\n" "http://127.0.0.1:8765/ce_retro_af16/$f"; done
ce_retro_af16_three_planes_pre_201_131_48.png            200
ce_retro_af16_three_planes_post_201_131_48.png           200
ce_retro_af16_enhancement.png                            200
```

MC-GS 在两个扫描 × 三个平面（201/131/48）的全部六个面板上，PSNR/SSIM/NMSE 三项都排第一。渲染 600dpi PDF 时
单进程峰值内存约 140 GB（1 TB 内存机器上安全），两张图串行渲染，没有并行——沿用了之前项目"600dpi 大图限两
并行"的教训，这里更保守，一次一张。

**导出包（brief Step 5，两种都跑，遵照 controller 裁决）**：

```
cd $TOPIC
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/export_bundle.py --no_raw
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/export_bundle.py
```

输出：`--no_raw` 103 files, 2.11 GiB；不带参数（含原始 k 空间）113 files, 28.36 GiB（即原始数据本身约 26.25
GiB，比脚本 docstring 原来写的"~14 GB"几乎多一倍，已在 README §6/§7 和脚本 docstring 里改成实测的 ~26 GB）。
两次都在 `cd .../export/ce_retro_af16_20260918 && md5sum -c md5sums.txt` 下 **全部 OK**（103/103，随后
113/113——两次调用写到同一个按日期命名的目录，第二次的 MANIFEST.tsv/md5sums.txt 覆盖第一次，但因为文件集合
是超集关系，落盘目录本身在任何时刻都和当时最新的清单自洽）。`/data0` 当时可用 4.6 TB，装得下。这个目录本身
是 `$TOPIC` 下的服务器数据，不进 git。

**回归验证**：`find_best_slice.py` 改动后，全量测试仍然通过：

```
cd $WT/ce_retro_cssense_af16 && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest scripts/tests -q
```

输出：`29 passed in 157.78s (0:02:37)`（和 Task 15 记的 29 个一致，没有新增/删除测试——`find_best_slice.py`
本来就没有专门的单测覆盖它）。**（whole-branch review 2026-09-18 修复后重跑）** 同一条命令再跑一次，
`29 passed in 159.08s (0:02:39)`，数量不变，`export_bundle.py` 的假 topic 测试也单独确认过仍然通过
（改动是把 `figures/*.p*` glob 换成显式 `FIGURES` 列表，两个用例都没碰）。

## 待用户拍板

- **push + 打 tag**：brief Step 5 最后一步是 push 到 GitHub 并打 `handoff/YYYY-MM-DD-ce-retro-af16` tag，但
  controller 明确保留这一步给你，本次只提交到本地分支 `ce-retro-cssense-af16`，没有 push、没有打 tag。
- **是否给 MC-GS 一个同规模的小网格做消融**：MC-GS 和 GS-single 目前是同配方同点数（360 万点），头条余量
  +0.194/+0.354 dB 是否会被"联合训练本身多用了一份跨对比度信息、和参数量无关"这个解释完全吸收，还是想再加一组
  控制参数量的对照，你定。
- **是否补第二张掩膜或第二对种子**：现在的结论建立在一张掩膜（seed 0）、一次训练种子上；要不要按 AF16 diff-mask
  或多种子的套路（`docs/superpowers/specs/...` 之前几个课题都做过）再验证一遍稳健性，你定。
- **正文图是否要砍列**：三面图现在是 12 列（GT+ZF+9 个重建臂+MC-GS，`--with_zf` 打开），投稿正文篇幅有限时是否
  要砍掉几个明显弱的臂（如 ZF、3D-DIP、MC-INR）只留补充材料，你定。
- （已解决，不再待拍板）L1-SPIRiT 核 5 是否发散/要不要从表里去掉——见下方"关键决定的为什么"，核 5 没有发散，
  两个扫描都被 `summarize.py` 选中，已保留在表里。
- （已解决，不再待拍板）`_chain.sh`/`_cs_chain_pre.sh` 要不要收进 git——controller 已经在任务指令里裁定删除，
  见下方"关键决定的为什么"。

## 下一步

Task 16 是这个课题目前规划的最后一步（brief 五步全部做完：门检查→全量调度→口径互验/汇总→出图→README/STATUS/
导出/提交）。如果用户看完图和数决定要跑上面"待拍板"的任何一项，建议顺序：

1. 先决定要不要 push + 打 tag（如果只是内部核验完毕，随时可以让下一个会话直接 push 现在这个 commit，不需要
   重新做任何计算）。
2. 如果要砍正文图列数，改 `scripts/fig_three_planes_compact.py` 调用去掉 `--with_zf` 或加一个只挑列的选项，
   不需要重新训练，volumes 都在 `recon/` 里。
3. 如果要第二张掩膜/种子或同规模消融，那是新的训练工作量（MC-GS/GS-single 各 ~1.2 小时/对，见下方效率参考），
   建议另开一个 Task/spec，不要塞进这个课题的 README。
4. 通读一遍 README，确认数字和图都对得上（这次已经全文过了一遍，删掉了所有 `NOT_RUN_YET`）。

## 已知的坑与别重做什么

- **GPU 争用把 3D-DIP/3D-INR 拖慢了两个数量级，这不是脚本 bug，是别的会话占卡的结果。** `dip_post` 第一次尝试
  （`logs/prod/dip_post_driver.log`，21:56 在 GPU 0 上启动）第 1→100 步用了 7598 秒（约 76.7 s/步），而正常
  情况（重启后那次，`logs/orch/dip_post.log`，05:06:07 在 GPU 3 上）前 100 步只要 70～74 秒（约 0.7 s/步）——
  慢了约 **109 倍**，量级上与"120x"的估计一致。第一次尝试最终被判定失败（driver 日志里的 `Terminated` /
  `FAILED`），后来手动在别的卡上重跑才在 10205 秒（约 170 分钟）内正常收尾。`inr_pre` 更极端：第一次尝试
  （21:57 在 GPU 2 上）第 1 步就用了 1956 秒（正常约 4 秒），driver 日志里那次调用连"done"时间戳都没有——大概率
  是挂住后被杀掉，后来在 04:38:23 于 GPU 5 上重启，正常跑完只要 6490 秒（约 108 分钟）。教训：这台机器的 GPU
  是共享的，`nvidia-smi` 显示忙不代表这次作业会慢一点点，可能慢一到两个数量级；如果单步耗时突然涨了几十倍，
  别怀疑脚本本身，先怀疑是不是撞上了别人的重活，果断换卡重启比等它"跑完"更省时间。
- **`find_best_slice.py` 的 "cssense" 键以前从来没真的跑通过（已在本任务修复）。** 脚本自己的 docstring 早就
  写明这是个已知但"未观察到"的风险：`METHODS` 里的 `"cssense"` 用的是扁平文件名 `recon/cssense_<c>_complex.npy`，
  但这个课题的 CS-SENSE 体是按 λ 扫的 `recon/cssense_o200_lam<L>_<c>_complex.npy`，前者永远不存在。之前没触发是
  因为 `gs_joint`/`gs_single` 也还没注册，脚本会先在别处报错；本任务里全部 11 臂都齐了，第一次运行直接
  `KeyError: 'cssense'`，和 docstring 预言的一模一样。修法：让 `"cssense"` 走 `logs/summary.json` 里
  `cssense_o200` 那一行记录的 λ 去拼文件名，和 `arms.py`/`best_slice_mcgs.py` 现成的 `lam_file()` 套路一样，
  别的七个方法没动。改动只有 `scripts/find_best_slice.py` 这一处，改后跑通，全量 pytest（29 个）照样全过。
- **上面这个修法本身没修全：两个逐层脚本各自的臂列表都只有 8/11（whole-branch review 2026-09-18 发现并修复）。**
  `find_best_slice.py` 的 `METHODS` 修完 cssense 之后仍缺 Joint SENSE、L1-SPIRiT、MC-INR；`best_slice_mcgs.py`
  的 `arms()` 是完全独立的另一份硬编码列表，缺 ZF、L1-SPIRiT、MC-INR。两处都不是同一次疏漏，是两份列表各自
  维护、各自漏掉不同的臂——这正是 README/STATUS 敢说"排第一"却从没跟 L1-SPIRiT 比过的根源。改法：两个脚本都
  不再手写文件名列表，统一走 `arms.arm_files()`（`arms.py` 的 `ORDER`/`LABEL`，本课题唯一权威的"某臂某扫描的
  文件在哪"来源），以后两个脚本不会再互相漏成不一样的子集。重跑后 `find_best_slice.py`/`best_slice_mcgs.py`
  的计数本身没变（见上一条"已完成且已验证"），但 `best_slice_mcgs.py` 218 层 post 的"最强非 GS 臂"从
  CS-SENSE 变成了 L1-SPIRiT——这才是真正被吃掉的错误。
- **导出包两次调用写到同一个目录，不是两份独立的包。** `export_bundle.py` 的输出目录只按日期命名
  （`export/ce_retro_af16_<YYYYMMDD>/`），不区分 `--no_raw`；今天先跑 `--no_raw`（103 文件/2.11 GiB）、
  再跑不带参数的全量版（113 文件/28.36 GiB）后，目录里实际是全量版覆盖 + 追加的结果，`--no_raw` 那份"小包"
  单独已经不存在了（如果需要一份纯 `--no_raw` 的包留档，得改天跑或者手动拷走）。这是脚本本来的行为，本任务
  没有改它。
- **`export_bundle.py` docstring 和 README §6 原来写的"~14 GB 原始数据"是猜的，不是量出来的。** 实测原始
  k 空间（`sim/gs/` 4 个 h5 + `sim/work_{pre,post}/` 各 3 个文件）合计约 26.25 GiB，接近两倍于原估计。已经在
  两处文档里改成实测数字（见 README §7 新增条目）；这个数字从来没有被任何门槛用到，只影响"/data0 装不装得下"
  的判断——装得下（当时可用 4.6 TB）。
- **`figures/` 里混进了亮度实验图，`export_bundle.py` 原来的 `figures/*.p*` glob 会连它们一起打进包
  （whole-branch review 2026-09-18 发现并修复）。** `fig_three_planes_compact.py` 不管 `--tag` 传什么都写进
  同一个 `figures/` 目录，亮度扫的 `t1flair_v3_af16_post_row3planes_compact_..._b1.25/_b1.5/_b1.8/_b2.2_...png`
  四个文件（其中 1.25/1.5 已经挑定、拷贝改名进了 git 仓库自己的 `figures/`，1.8/2.2 只是散在 topic 的
  `figures/` 里，没人挑）就跟正式图存在同一层。改法：`export_bundle.py` 不再 glob，改成显式列出这个课题自己
  的 11 个图文件名（`FIGURES` 常量），不存在就跳过——以后不管往 `figures/` 里扔什么标签的一次性渲染图，不主动
  把文件名加进 `FIGURES` 就不会被打包。没有动那 4 个亮度实验 png 本身（server 数据，不进 git），只改了打包
  脚本怎么选文件。
- 以下是 Task 1–15 已经踩过、这次继续遵守、没有重犯的坑（细节见 README §7，这里不重复）：`check_sim.py` 伴随
  恒等式在低线圈覆盖处的奇异性限制、`sigma_from_residual` 的读出轴漏算 bug、trainer 快照的真实路径
  `trainer/_trainer_e778bf2/`、zsh 变量必须加引号。
- 两个 controller 的临时调度脚本 `scripts/_chain.sh`、`scripts/_cs_chain_pre.sh` 已删除，没有提交（见下方
  "为什么"）。

## 关键决定的为什么

- **为什么 L1-SPIRiT 核 5 保留在表里、不单独因为"标定欠定"就撤掉**：`summarize.py` 是按头内 PSNR 在核 3/核 5、
  全部 λ 里选最优的；实测核 5 在两个扫描的几乎每个 λ 上都比核 3 高（唯一例外是 post 在 λ=0 时核 3 反而高
  0.09 dB，但 λ=0 从来不是被选中的工作点），核 5 也确实是两个扫描最终都胜出的那一档。"标定方程欠定"是先验
  风险（design §5 写明的已知担忧），但落到数据上没有发生——没有理由为了一个没兑现的风险去丢掉表现更好的结果。
- **为什么两个脚本重跑的 pytest 没有新增测试**：这次唯一的代码改动是 `find_best_slice.py` 里 "cssense" 的
  文件名解析，改动前就没有专门的单测覆盖这个脚本（`scripts/tests/` 目录下搜不到任何引用
  `find_best_slice` 的测试文件），改动本身也很小、而且已经用真实 topic 数据完整跑通两次（改前复现
  `KeyError`、改后产出合理的 `logs/slice_scores.json`），认为这个验证强度足够，没有额外补单测——补测试意味着
  搭一个新的假 topic fixture，超出了这次任务"收尾"的范围。
- **为什么 `_chain.sh`/`_cs_chain_pre.sh` 直接删除而不是问一遍**：这两个脚本上次（Task 15）STATUS 里还是待
  拍板项，但这次任务的指令里 controller 已经明确认领了这两个文件（"是我的"）并直接下达了删除指令，不再是
  需要用户拍板的事，照办即可。
- **为什么导出包"先小包再全包"两步都跑、而不是只跑一个**：任务指令要求先跑 `--no_raw` 报文件数/大小做核对，
  再基于"`/data0` 有空间、原始输入能让包自包含"这条 controller 裁决去跑全量版；两次调用的输出都已经记在上面
  "已完成且已验证"里，供审计。
- **为什么三面图串行渲染、不并行**：单张 600dpi PDF 渲染时观测到进程 RSS 峰值约 140 GB；机器总内存 1 TB、
  当时可用还有 900 多 GB，两张并行理论上够用，但沿用了这台机器上其它课题"600dpi 大图限两并行"的教训，本来就
  只有两张图，串行也没有明显时间代价，不值得为了省几分钟去冒内存风险。
- **为什么把"~14 GB"改成"~26 GB"却没有改 `sigma_from_residual` 那条坑里提到的"14 GB of study data"**：那句话
  在 README §7 是描述 Task 5 那个 sigma bug 的历史叙述（"如果不修，会把 14 GB 的合成数据写噪声写坏"），和这次
  `export_bundle.py` 的原始数据大小是两件不同的事、写在不同的历史时间点，不属于这次任务改动的范围，没有动它。
