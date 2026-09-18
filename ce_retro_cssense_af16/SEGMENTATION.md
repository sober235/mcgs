# 公开分割模型下的统一分割 — CE retro AF 16.00

新会话从这里开始读这条线：本文件说清楚这一步做了什么、数字是多少、以及引用时必须一起说的限制。
课题根 `$TOPIC` = `/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16`，
分支 `ce-lesion-seg-figure`，worktree `/data0/congcong/code/GS/mcgs_ce_segfig_wt`。

## 1. 任务链条

1. 原始采集：UIH 3D T1 GRE 平扫/增强同患者同次检查，**前瞻欠采 AF 5.33，没有全采参考**。
2. 真值：把全部采到的数据做 **CS-SENSE 重建**，得到的那张图定义为本课题真值（`sim/gt.npz`，
   与 `recon/gt_{pre,post}_complex.npy` 相差一个全局系数 `scale_*`，实测相对误差 1.6e-7）。
3. 用真实 42 线圈 CSM 从真值正向合成完整 k 空间，按确定性泊松盘 **AF 16.00**（2552/40832 点）重新欠采，
   平扫增强同一张掩膜，噪声 σ 由原采样点残差定。
4. 十一个臂在这份 AF16 数据上重建（`recon/*_complex.npy`）。
5. **本步**：用公开分割模型对这十一个 AF16 重建执行统一分割，与同一模型在真值上的输出比较。

## 2. 分割模型与口径

**模型**：MedSAM（Ma et al., Nat Commun 2024），公开权重 `wanglab/medsam-vit-base`，即 SAM ViT-B 在约 150 万
医学图像-掩膜对上微调的版本，用框提示。权重在 `/home/congcongliu/models/medsam-vit-base`。
本机连不上 HF 的 CDN（`cdn-lfs.huggingface.co` 直接连接失败），权重从 `https://hf-mirror.com` 分块断点续传拉下来，
再转成 safetensors（transformers 5 在 torch < 2.6 下拒绝加载 `.bin`）。

**环境**：`medsam_seg.py` 需要 `transformers`，GSTMI 环境里没有，所以它单独跑在
`/home/congcongliu/.venvs/medsam_probe/bin/python`（用 nvgen 的 python 建的 `--system-site-packages` venv
加 `pip install transformers`）。课题其余脚本仍然跑 GSTMI。

**对所有臂完全一致的处理**，这是比较公平性的全部依据：

- 每个体除以**自己**白质的中位数（SynthSeg 标签 2/41，取自真值那份 `cssense_labels.npy`），所以臂之间的全局
  增益差异不可能影响结果；
- 固定窗 `[0, 2]` 白质单位映射到 8 位，**不是逐层 min-max**（逐层归一化会给每个臂不同的对比度拉伸）；
- 一个固定框提示：真值病灶核心的三维包围盒外扩 4 体素，即 z[186,221] y[101,144] x[29,64]，对每个臂、
  两个扫描、框内每一个横断层都用这一个框；
- 二维掩膜堆成体，取最大 26-连通块（唯一的后处理，真值和每个臂都做）。

**参考**：同一模型在**真值**上的输出。所以这个数回答的是"下游分割器读某个重建时，读出来的东西和读真值时有多一致"，
不是"谁更接近医生的勾画"——这条线上没有医生勾画。

## 3. 结果（Dice 对真值掩膜，HD95 单位 mm）

| 臂 | 平扫 Dice | 平扫 HD95 | 增强 Dice | 增强 HD95 |
|---|---|---|---|---|
| CS-SENSE | **0.869** | **2.24** | **0.846** | **3.00** |
| Joint SENSE | 0.868 | 2.24 | 0.841 | 3.16 |
| GS-single | 0.843 | 3.00 | 0.783 | 6.71 |
| **MC-GS** | 0.835 | 3.00 | 0.782 | 6.40 |
| L1-SPIRiT | 0.813 | 3.00 | 0.794 | 4.12 |
| 3D-DIP | 0.803 | 3.61 | 0.771 | 6.40 |
| CG-SENSE | 0.770 | 4.12 | 0.769 | 5.00 |
| ZS-SSDU | 0.763 | 5.00 | 0.772 | 7.00 |
| MC-INR | 0.724 | 4.97 | 0.725 | 5.00 |
| 3D-INR | 0.718 | 4.24 | 0.723 | 5.92 |
| ZF | 0.704 | 6.00 | 0.714 | 9.54 |

MC-GS 平扫第 4、增强第 5；MC-GS 减 GS-single = −0.008（平扫）/ −0.001（增强）。
真值掩膜 16998（平扫）/ 14752（增强）体素。

## 4. 引用这张表必须一起说的话

**这条线的真值本身是一个 TV 正则重建，CS-SENSE 家族有结构性主场优势。** 真值来自 AF 5.33 的前瞻采集，
PE 支撑只占 18.8%；AF16 掩膜的 2552 个点里 1599 个（62.7%）在原采集中从未测过，那些位置的真值是 TV 先验的外推
而不是数据。病灶边界正是高频信息，最依赖这些从未测过的位置，而分割器判边界时读的就是这个纹理。
PSNR 大体逃过了这一层（CS-SENSE 在头内 PSNR 上只排第五），分割 Dice 逃不过。

换分割口径救不了：把当初生成品红轮廓的那个确定性阈值算子（SynthSeg 脑实质 + 白质归一化 + 高斯 σ=1 +
阈值 0.32 + 最大连通块）同样统一跑一遍，冠军变成 L1-SPIRiT（平扫 0.822 / 增强 0.810），MC-GS 平扫第 2
（0.809，比 GS-single 高 0.004）、增强第 5（0.768，比 GS-single 低 0.023）。两个口径连冠军都不一样。
换更强的分割模型只会更忠实地反映真值的 TV 纹理，方向不会变。

**其余两行不受影响。** 同一张图的图像行和误差行用的是本课题原有的三个指标，在矢状第 41 层上
MC-GS 在 {层内 PSNR，壳层 RMSE，边界梯度 RMSE} × {平扫，增强} 六个条件上全部第一（见 `best_slice_lesion.py`）。

**框提示的性质**：box 提示的 SAM 对被问到的每一层都必须给出一个掩膜，没有"这里什么都没有"这个选项，
所以掩膜填满提示框的整个 z 范围，越过了病灶按阈值定义的范围（真值在 z=186 和 z=221 各有 255–339 体素，
而阈值 core 在那里是 0）。每个臂被问的是同一个问题，臂间比较不受影响，但**绝对体积不是病灶体积**。

**要做"我们分割病灶更准"这个claim，得换一条真值独立的线。** UCSF-PDGM 的两个课题
（`../CE_retro_gt`、`../CE_retro_af16`）真值是真实临床图像，不是任何重建算法的产物，而且自带专家肿瘤标注
（标签 4 强化、标签 1 坏死核心）。

## 5. 复现

```bash
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
MEDSAM_PY=/home/congcongliu/.venvs/medsam_probe/bin/python
E="PYTHONNOUSERSITE=1 PYTHONPATH="
cd /data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16

# 统一分割（GPU，每个扫描约 1 分钟；把 CUDA_VISIBLE_DEVICES 指到一张空卡）
env $E CUDA_VISIBLE_DEVICES=6 $MEDSAM_PY scripts/medsam_seg.py --scan pre
env $E CUDA_VISIBLE_DEVICES=6 $MEDSAM_PY scripts/medsam_seg.py --scan post

# 图（纯 CPU）
env $E $PY scripts/fig_lesion_seg.py --axis 2 --index 41

# 测试
env $E $PY -m pytest scripts/tests -q        # 42 passed
```

产物：`seg/medsam_*.npy`（24 个 bool 体）、`logs/medsam_seg_{pre,post}.json`、
`figures_seg/ce_retro_af16_lesion_seg_2041.png`，图同时拷到 `/home/congcongliu/figs/ce_retro_seg/`
（`http://localhost:8765/ce_retro_seg/ce_retro_af16_lesion_seg_2041.png`）。

## 6. 图怎么读

13 列：整层全脑 | 真值裁剪 | 十一个臂。每个扫描三行：

- **图像行**：灰度 P99.9 窗口（显示亮度 ×1.5，指标在未缩放数据上算），品红 = 已存病灶核心轮廓，
  橙黄框 = 贴在每个面板右上角的放大框（真值那列也有），面板内数字 = 头内层内 PSNR；
- **误差行**：`|rec − GT| / max|GT|`，jet [0, 0.1]，只显示头内，面板内是壳层 RMSE 和边界梯度 RMSE；
- **分割行**：青色 = MedSAM 在真值上的轮廓，红色填充 = 该臂多分出来的，蓝色填充 = 漏掉的，
  白色虚线 = 提示框（模型的视野），面板内 Dice 和 HD95 是三维的。

最左一列每个扫描一张大的整层全脑图，品红框标出右边裁剪的位置，青色是 MedSAM 在真值上的轮廓。
加粗 = 该扫描该指标十一个臂里最好的。

## 7. 待用户拍板

- 分割行是留在正文图里（连同第 4 节的限制一起写），还是撤到补充材料、正文只留图像行和误差行。
- 要不要把同一套流程搬到 UCSF-PDGM 线，用专家标注当金标准重做这张分割图。
- 要不要出 600 dpi PDF（本课题的经验是单张 600 dpi 渲染峰值内存约 140 GB，一次只渲一张）。
