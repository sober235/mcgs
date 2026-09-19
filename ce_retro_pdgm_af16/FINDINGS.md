# PDGM AF16 上的统一分割 — 结论：这个下游指标在 MedSAM 这个胜任度下不成立

`$PDGM` = `/data0/congcong/code/GS/Results_20260908/CE_retro_af16`，分支 `ce-lesion-seg-figure`，
worktree `/data0/congcong/code/GS/mcgs_ce_segfig_wt`。脚本 `scripts/medsam_seg_pdgm.py`，测试 7 个。

## 1. 为什么换到这条线

UIH 那条线（`../ce_retro_cssense_af16`）的真值是 CS-SENSE 的 TV 重建，分割器读 CS-SENSE 家族最像真值是结构性的，
排名不反映重建质量（见那边的 `SEGMENTATION.md`）。PDGM 线的真值是真实临床 T1/T1c 图像，自带专家肿瘤标注
（标签 1 坏死、2 水肿、4 强化），参考是医生的勾画而不是任何算法的输出。**这个换线目的达到了**：
CS-SENSE 不再是无条件冠军。但暴露了一个更严重的问题。

## 2. 协议（只看真值胜任度定下来，定完才看各臂）

- 目标 = 肿瘤核心（标签 1|4）。真值上 Dice 0.787 / 0.765 / 0.523 / 0.604（0004/0023/0031/0053），
  强于只取强化（0.614/0.437/0.489/0.507）和全肿瘤（0.603/0.520/0.339/0.508），四例全胜。
- 提示 = **逐层框**，从专家标注取该层截面的包围盒外扩 4 体素，只在截面 ≥ 20 体素的层给提示。
  四例全面优于"一个三维框投到每层"（0.773/0.727/0.398/0.589），也是 MedSAM 论文自己的评测协议。
- 对所有臂完全一致：同一模型、同一权重、同一组框（来自标注，不来自任何臂的图像）、
  各自白质中位数归一化、固定窗转 8 位。**不做**对真值的复数标定——除了白质掩膜，真值不该碰臂的图像。
- 后处理只有最大 26-连通块，真值和每个臂一样做。

## 3. 结果（增强扫描，肿瘤核心，四个真实病例均值）

| 臂 | Dice 对专家标注 | Dice 对真值掩膜 |
|---|---|---|
| CS-SENSE | 0.720 | 0.753 |
| L1-SPIRiT | 0.720 | 0.726 |
| Joint SENSE | 0.703 | 0.746 |
| 3D-INR | 0.687 | 0.760 |
| CG-SENSE | 0.678 | 0.737 |
| **ZF** | **0.675** | 0.706 |
| MC-INR | 0.671 | 0.734 |
| ZS-SSDU | 0.667 | 0.772 |
| GS-single | 0.646 | 0.750 |
| 3D-DIP | 0.640 | **0.780** |
| **MC-GS** | **0.634（末位）** | 0.748 |

逐例数字在 `$PDGM/logs/medsam_seg_post_tc.json`，掩膜在 `$PDGM/seg_medsam/<case>/`。

## 4. 这个指标不能用来排重建，证据有三条

**一、它和重建质量反着走。** 同一批数据上的重建质量（`$PDGM/retro_scores_fast.json`，六个臂的快速打分，
增强扫描头内 PSNR）：GS-single 25.17 dB > MC-GS 24.40 > 3D-DIP 23.61 > 3D-INR 22.45 > CG-SENSE 20.51 > ZF 19.13。
分割 Dice 里 ZF 0.675 高过 MC-GS 0.634 和 GS-single 0.646。零填充是毫无争议的最差重建。

**二、完美图像输给零填充。** 病例 0031，真值自己的掩膜对专家标注只有 0.523，ZF 是 0.620。
一个连真值都排不到第一的指标，不能拿来比重建。

**三、机制清楚，不是随机噪声。** MedSAM 在干净图上系统性**欠分割**：真值掩膜体积只有专家标注的
15557/23265、8836/13774、24947/49062、42834/94585。它勾的是亮的强化部分，把暗的坏死核心排除在外。
图像一退化，掩膜边界外糊，反而盖住更多标注 → Dice 升高。退化在这个指标上是加分项。

**不是窗宽的锅。** 窗从 [0,1.5] 到 [0,5] 扫过，真值均值 Dice 只在 0.669–0.679 之间动，欠分割照旧。

## 5. 这条线上 MC-GS 的位置（需要知道的背景）

PDGM AF16 的快速打分里，**MC-GS 在增强扫描上不如 GS-single**（24.40 对 25.17 dB），平扫打平
（29.78 对 29.76）。UIH 线上 MC-GS 领先 GS-single 的结论在这条线上不成立——这与分割无关，是重建本身的事实，
而且那份打分只覆盖六个臂，不是全场。要下结论得先用完整的 `retro_score.py` 重跑十一个臂。

## 6. 下一步的三个选择

1. **换更强的公开分割模型**。需要真值上 Dice 到 0.85 以上，退化才会严格地扣分，指标才成立。候选
   MedSAM2（3D 传播）、SAM-Med3D（3D 原生）。代价是再下一次权重（本机只能走 hf-mirror，分块续传），
   加一套推理代码，半天。不保证成功。
2. **换下游任务**。用 SynthSeg 这种在这个数据上确实可靠的解剖分割，测"重建如何影响解剖结构分割的一致性"。
   不是病灶分割，但指标站得住。
3. **把这份阴性结果定稿**。两条线（UIH 真值同源、PDGM 模型不胜任）都做过了，说明"同一分割模型下我们更准"
   这个claim在现有条件下拿不到，正文改用已有的保真指标，分割作为局限写进讨论。

## 7. 复现

```bash
MEDSAM_PY=/home/congcongliu/.venvs/medsam_probe/bin/python
cd /data0/congcong/code/GS/mcgs_ce_segfig_wt/ce_retro_pdgm_af16
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES=6 $MEDSAM_PY scripts/medsam_seg_pdgm.py --scan post --target tc
env PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest scripts/tests -q   # 7 passed
```
