# 病灶保留：AF 16.00 下各重建把病灶保住了多少

新会话读这一份就够了解这条线在做什么。课题根 `$TOPIC` =
`/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16`，分支 `ce-lesion-seg-figure`，
worktree `/data0/congcong/code/GS/mcgs_ce_segfig_wt`。

## 1. 一句话

PSNR 高不等于病灶还在。这条线用**固定 ROI 内的保真指标**（不经过任何分割模型）证明：AF 16.00 下
**MC-GS 把病灶的对比度与可见度保留得最接近真值**，而两个隐式神经表示方法把病灶填亮、损失近一成对比度和四分之一 CNR。

## 2. 为什么不用分割模型

两条分割路线都试过，都不成立，记录在 `SEGMENTATION.md` 与 `../ce_retro_pdgm_af16/FINDINGS.md`：

- 本课题线：真值本身是 CS-SENSE 的 TV 重建，分割器读 CS-SENSE 家族最像真值是结构性的，排名不反映重建质量。
- PDGM 线（真值是真实临床图像 + 专家修正标注）：MedSAM 在干净图上系统性欠分割，退化反而让掩膜外糊、盖住更多标注，
  **零填充 0.675 高过 MC-GS 0.634，病例 0031 上真值自己的掩膜 0.523 还输给 ZF 的 0.620**。

共同教训：把比较塞进一个黑箱，只有当黑箱自身误差远小于臂间差异时才成立。固定 ROI 没有这个黑箱。

## 3. 口径

| | |
|---|---|
| 数据 | UIH 3D T1 GRE 平扫/增强同患者同次检查；真值 = 全部采到的数据（AF 5.33）的 CS-SENSE 重建 |
| 欠采样 | 真值经真实 42 线圈 CSM 合成完整 k 空间后，按确定性泊松盘 AF 16.00（2552/40832 点）重新欠采，两个扫描同一张掩膜 |
| 病灶核心 ROI | 6039 体素，`enhance_rois_cssense.npz::core`。**人工在轴位指认位置 + 阈值 0.32 + 最大连通块**，不是医生勾画 |
| 正常白质 ROI | 486769 体素，铺满全脑。与病灶核心重叠 0 体素，与核心外 0–3 mm 圈重叠 0 体素；到病灶的距离中位 52 mm |
| 单位 | 各体积除以**自己**白质的中位数；对比度与 CNR 都是无量纲比值，全局增益动不了它们 |
| 臂 | 正文图用七个：CS-SENSE、Joint SENSE、ZS-SSDU、3D-INR、MC-INR、GS-single、MC-GS。十一臂全量在 json 里 |

## 4. 指标定义

```
对比度   C    = (白质均值 − 病灶均值) / 白质均值            真值 0.766（平扫）/ 0.747（增强）
CNR          = (白质均值 − 病灶均值) / 白质标准差           真值 2.709 / 2.612
图上画的都是相对误差：(方法 − 真值) / 真值 × 100%，所以原点 = 真值
```

⚠️ CNR 的分母不是纯噪声。真值白质区标准差 0.3058 里只有约 29%（0.0900）来自小尺度起伏，其余是真实解剖结构。
所以**过度平滑的方法分母变小、CNR 反而变高**——CNR 不能单独用，必须和对比度一起看，因为过度平滑会同时压掉对比度。

## 5. 结果

**MC-GS 对 GS-single：九个指标 × 两个扫描，18 项全胜**（`logs/lesion_preservation.json`）。

七臂中 MC-GS 第一的四项：

| | 平扫 | 增强 |
|---|---|---|
| 病灶对比度误差 | **−2.01%** | **−1.98%** |
| 病灶 CNR 误差 | **−2.04%** | **−0.48%** |
| 次优（GS-single） | −2.43% / −2.62% | −2.22% / −1.19% |
| 3D-INR | −7.28% / −20.99% | −9.50% / −24.63% |
| MC-INR | −9.18% / −21.82% | −10.20% / −25.07% |

核心区假强化（增强 − 平扫，真值是地板）：MC-GS −0.0004、GS-single −0.0024、**3D-INR +0.0156、MC-INR +0.0063**。
前两者在地板上（前瞻线测得 GS 两臂自身种子间地板 ±0.001，所以 MC-GS 与 GS-single 之差撑不起结论）；
后两者在**造假强化**，幅度是地板的 6 到 16 倍，这条站得住。

⚠️ 弱指标，别用：**病灶区 PSNR**（余量只有 0.1%–0.4%，而且零填充在这项上能拿 41.4 dB，高过 CG-SENSE 的 36.9 dB，
指标本身不可信）；**壳层 RMSE** 增强上输给 CS-SENSE。

## 6. 三张图怎么读

**剖面图** `figures_seg/ce_retro_af16_lesion_profile.png`（Nature 规范版在 `figures_seg/nature/`）

横轴是到病灶表面的有符号距离，负=病灶内、正=周围组织。**它不是图上的一条线，是一层层壳**：x=−3 那一点，
取的是所有"位于病灶内部、距表面 3 mm"的体素的平均亮度。像剥洋葱，一层一个点。x=0 没有点，因为体素要么在内
（距离 ≥1 mm）要么在外（≥1 mm）。各层体素数 493 到 5087，逐层记在 source-data CSV 里。

纵轴是白质单位的平均亮度。曲线是一道台阶：低端是病灶，高端是周围组织，落差就是对比度，陡峭程度就是边界宽度。
下面的窄带是残差（各方法 − 真值），把上面挤在一起的差异放大。

第三面板横轴=对比度误差、纵轴=CNR 误差，原点是真值，越近越好。标记形状与颜色表示方法，实心=平扫、空心=增强。

**横断位图** `figures_seg/ce_retro_af16_lesion_axial_201_whole_b1.8_cnone_a7.png`

四行：平扫图 / 平扫误差 / 增强图 / 增强误差。每格带右上角放大框。误差行最左边是实际用的 AF16 掩膜。
误差图 jet [0, 0.1]，只显示头内。图像面板印头内层内 PSNR / SSIM / NMSE。

**三维曲面图** `figures_seg/ce_retro_af16_lesion_surface.png`

真值的病灶边界曲面提取一次，七个面板共用，只把各方法的误差涂上去。看的是"错在病灶的哪个部位"。
⚠️ 曲面上的平均误差 MC-GS 不是第一（它是绝对强度误差，而 MC-GS 的优势在对比度与 CNR），所以这张图只做定性展示、不标数字。

## 7. 能说什么 / 不能说什么

能说：误差集中在病灶边界一圈，出边界 2 mm 后所有方法回到零；两种失败方式被区分开（填亮 vs 轻微模糊）；
MC-GS 两个扫描都最接近真值；所有方法的对比度与 CNR 都只会低于真值，没有例外。

不能说：只有一个病灶、一个受试者、一张掩膜、一个种子，没有误差棒，不能讲统计显著；真值不是全采样采集，
AF16 掩膜里 62.7% 的点在原采集中从未测过；病灶 ROI 不是医生勾画；没有读片实验，所以只能说"对比度与可见度
保留得最好"，**不能说提高了病灶检出率**。

另外：这个病灶**不强化**（核心 +0.021 白质单位 = 数据本底）。增强扫描上真正变亮的是血管/静脉窦（+2.841）、
脉络丛（+1.080）。并且**增强行的 PSNR 不能和平扫行直接比**——peak 分别是 1.472 和 1.892，光这一项就带来 +2.18 dB
的系统性差。SSIM 和 NMSE 无此问题。

## 8. 复现

```bash
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
E="PYTHONNOUSERSITE=1 PYTHONPATH="
cd /data0/congcong/code/GS/mcgs_ce_segfig_wt/ce_retro_cssense_af16

env $E $PY scripts/lesion_preservation.py            # -> $TOPIC/logs/lesion_preservation.json
env $E $PY scripts/fig_lesion_profile.py             # 剖面图（--title/--subtitle 可换中文）
env $E $PY scripts/fig_lesion_axial.py --index 201 --whole --brightness 1.8 --contour none \
    --arms cssense_o200,jcssense,zsssdu,inr,mcinr,gs_single,gs_joint
env $E $PY scripts/fig_lesion_surface.py             # 三维曲面
env $E FIGURE_QA_SCRIPTS=<出版 QA 工具目录> $PY scripts/fig_lesion_profile_nature.py   # 183×100 mm 投稿版
env $E $PY -m pytest scripts/tests -q                # 50 passed
```

投稿版的三道强制审计（面板对齐、PDF 字号、渲染碰撞）全部 PASS；碰撞审计需要 PyMuPDF。

## 9. 待办

- 未 push 到 main，也未与 `plain-enhanced-af16` 合并（该分支另有会话在用）。
- 剖面图与横断位图的配色不一致（工作版按方法、Nature 版按方法家族），要统一说一声。
- 折线图 `ce_retro_af16_lesion_metrics.png` 仍是十一臂，与七臂的正文图列序不一致。
- Nature 版只做了剖面图；横断位图若要同规范出，需另跑一次三道审计。
