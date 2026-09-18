# STATUS

新会话先读本文件，再读 README，关键数字先重跑。

`$TOPIC` = `/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16`，`$WT` =
`/data0/congcong/code/GS/mcgs_ce_retro_wt`，`$PY` = `/home/congcongliu/anaconda3/envs/GSTMI/bin/python`。

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
