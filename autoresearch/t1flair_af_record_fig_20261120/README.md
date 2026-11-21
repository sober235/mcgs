# T1+FLAIR AF12 / 16 / 20 / 24：平扫 / 增强记录格式的三平面图（jiangyizhou）

用户在 2026-11-20 提了两项要求：
1. "redraw this figure, arranging the figures of different acceleration ratios in the same style as the plain scan and
   enhancement ones. Each acceleration is one picture."
2. "convert the images here into a 600 dpi PDF and save the original data to the GPU server. Then record the optimal
   results of each test stage at different acceleration multiples."

后来又加了两项：
3. 2026-11-20：画一张不用折线的多加速倍数概览图，指标只用 PSNR、SSIM、NMSE。见 `fig_af_overview.py` 和 `figures/af_overview.{png,pdf}`。
4. 2026-11-21：把这里的结果记为本阶段的最优结果。见 `RECORD.md`。

各测试阶段的最优结果见 `STAGES.md`。

## 概览图（`figures/af_overview.png` / `.pdf`）

- **(a)(b) 热图：** T1 和 FLAIR 各一张，11 个方法 × PSNR / SSIM / NMSE × AF12 / 16 / 20 / 24，颜色越深越好，每列最好的一格加框。MC-GS 在 24 列里全部第一。
- **(c) 逐层面配对小提琴图：** 每个对比度、每个加速倍数取 597 个层面，画 MC-GS 相对 GS-single 和相对最强非 GS 方法的提升，上方标出 MC-GS 更好的层面比例。
- **数据来源：** 只读取已登记的结果，即四个课题的 `logs/summary.json`，以及趋势课题的 `head_nmse_*.json` 和 `slice_trend_*.json`。
  出图前核对了两份记录的 PSNR 是否一致、四个课题的真值是否完全相同、选出的层面数是否为 243 / 187 / 167。
- **格式：** PNG 为 200 dpi，PDF 全矢量。

## 图

每个加速倍数一张图。T1 和 FLAIR 分别占平扫和增强的位置，六行依次是：横断 T1、横断 FLAIR、冠状 T1、冠状 FLAIR、矢状 T1、矢状 FLAIR。

- **放大行：** 每个图像行下面有一行，左半是放大框，右半是同一框的误差图。GT 列的右半换成该平面的采样掩膜。
- **数字：** 每个面板标层面 PSNR / SSIM / NMSE，每行最好的加粗。
- **色条：** 全图只有一条灰度条和一条误差条。
- **与原定稿图相同的部分：** 层面 180 / 124 / 76、48 体素放大框、12 列方法、误差色标 [0, 0.1]。
- **使用的臂：** 已发表的登记臂，AF12 用 `p14`，其余用 `p3`。

| AF | 本地图 | PNG sha256 | PDF sha256 |
|---|---|---|---|
| 12（11.71×） | `figures/af12_record.png` / `.pdf` | `85eb5b8a…` | `b3f089c8…` |
| 16（15.19×） | `figures/af16_record.png` / `.pdf` | `a8bf671c…` | `6fa03315…` |
| 20（18.56×） | `figures/af20_record.png` / `.pdf` | `2266056c…` | `6f5e2950…` |
| 24（24.00×） | `figures/af24_record.png` / `.pdf` | `0bfcb8f8…` | `1e5e6699…` |

- **PNG：** 200 dpi，6523 × 6006 像素。
- **PDF：** 文字、框线、色条都是矢量，218 幅图像面板各按 600 dpi 嵌入，即 2.6 英寸宽的面板为 1561 像素。没有 Type 3 字体，每个约 8.8 MB。
- **分辨率的限制：** 每个面板的底图只有约 256 体素宽。600 dpi 让文字和边框更清楚，不会给图像增加信息。
- **git：** `figures/` 下的 PNG 和 PDF 都被 `.gitignore` 排除。
- **四张图的结果：** MC-GS 在六行的 PSNR、SSIM、NMSE 上都是第一（每张 18 / 18），第二名每行都是 GS-single。

## 服务器上的原始文件

`/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_t1flair_af_record_fig_20261120/`

- 四张图的 PNG 和 PDF、预览 JPEG、记录 JSON、出图脚本，以及本 README 和 `STAGES.md`。
- **记录 JSON：** 包含所有面板的数字，以及每个源体积的路径和 SHA-256。
- **源体积：** 没有复制，仍在各课题的 `recon/` 里（`Results_20260908/T1FLAIR_AF{12,16,20,24}_v3_jiangyizhou/`）。

## 复现

```sh
/usr/bin/python3 ../gpu-tools/gpu.py run --cpu -- @python autoresearch/t1flair_af_record_fig_20261120/fig_t1flair_af_record_format.py \
    --af 16 --pdf --pdf_dpi 600 --save_dir <a new server folder>
```

- **运行：** 每个加速倍数一个作业，因为 fetch 每次最多 50 MiB。
- **核对：** 脚本先重算每个体积的头部 PSNR，与课题的 `logs/summary.json` 比较，差超过 0.005 dB 就停。这次四张图的最大差是 0.00003 dB。
  AF16 T1 的面板数字与原定稿图逐个一致。
- **重复运行：** 带 PDF 的第二次运行，PNG 与第一次逐字节相同。
- **覆盖保护：** `--save_dir` 不会用不同内容覆盖已有文件。
