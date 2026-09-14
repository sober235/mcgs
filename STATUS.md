# STATUS — 2026-09-13（worktree `mcgs_af16_wt`，分支 `autoresearch-jyz-af16-phase3`）

本工作树只承载 jiangyizhou AF16 一条线。**三期已跑完**（用户 09-12 下午要求重开并"拉大"与对比方法的差距）。
先读 `docs/handoff/2026-09-13-autoresearch-jyz-af16-phase3.md`，再读 `autoresearch_af16_p3.md`（规则与逐轮日志）
与账本 `autoresearch_af16_p3.tsv`；英文总结在 `docs/superpowers/records/2026-09-13-autoresearch-jyz-af16-phase3.md`。

## 一句话
三期配方 = P6 + 小波改 db4 只用最细一层（T1 0.12 / FLAIR 0.10）+ common-edge 0.001 + 相位 TV 0.03，4000 步头内
PSNR **35.706 / 34.928**，对 L1-ESPIRiT 余量 **+0.497 / +0.425**（P6 是 +0.343 / +0.304），六格全第一，
**锐度 0.855 / 0.804 两个对比度都不低于 L1-ESPIRiT**（三期以来第一次）。课题里登记为 `gs_joint_p3` / `gs_single_p3`，
P6 仍占 `gs_joint_ar`。

## 1. 已完成且已验证

重跑核对（原始输出，2026-09-13）：

```
$ cd /data0/congcong/code/GS/mcgs_af16_wt
$ cut -f1,7,8 autoresearch_af16_p3.tsv | grep -E "ptv003_full4k|s43_full4k"
p3_r3_e0001_ptv003_full4k	35.706	34.928
p3_r3_ptv003_s43_full4k	35.707	34.928

$ PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
$ PYTHONNOUSERSITE=1 PYTHONPATH= $PY -c "
import json;R='/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/logs/summary.json'
for r in json.load(open(R)):
    if r['method'] in ('gs_joint_p3','gs_single_p3','l1espirit'):
        print(f\"{r['contrast']:5s} {r['method']:13s} head {r['head_psnr']:.3f} ssim {r['head_ssim']:.4f} hfen {r['head_hfen']:.4f} sharp {r['sharp']:.3f}\")"
FLAIR l1espirit     head 34.503 ssim 0.9201 hfen 0.1210 sharp 0.789
FLAIR gs_joint_p3   head 34.928 ssim 0.9261 hfen 0.1121 sharp 0.804
FLAIR gs_single_p3  head 34.886 ssim 0.9249 hfen 0.1137 sharp 0.806
T1    l1espirit     head 35.209 ssim 0.9334 hfen 0.1286 sharp 0.849
T1    gs_joint_p3   head 35.706 ssim 0.9396 hfen 0.1139 sharp 0.855
T1    gs_single_p3  head 35.558 ssim 0.9371 hfen 0.1186 sharp 0.859
```

复现终配方（4000 步，约 1 小时一张空卡）：

```
cd /data0/congcong/code/GS/mcgs_af16_wt
SAVE_VOLUMES=1 ./autoresearch_af16_p3.sh p3_repro 4 4000 \
    --set loss.wav_name=db4 --set loss.wav_levels=1 --set loss.lambda_wav_plain=0.12 --set loss.lambda_wav=0.10 \
    --lambda_edge 0.001 --lambda_ptv 0.03
```

- 上面的配方与数字；种子地板 0.001 / 0.000 dB；联合减单臂 +0.148 / +0.042（一期是 +0.037 / −0.003）。
- 紧凑三面图 174/110/47（`_p3` 后缀）：MC-GS 在两个对比度、三个方向的 PSNR/SSIM/NMSE 全部第一。
- **AF16 定稿图（用户 09-13 拍板以 48 为准）**：`..._row3planes_compact_180_124_76_inset48_headerr_p3.png`，
  整层 180/124/76 + 48 体素放大框（横断放右上、其余左下），面板指标保留，色条只在横断行，横断位转到眼睛朝上（`--tra_rot 1`），误差图色标固定 [0, 0.1]（全局约定；`--err_vmax` 可切 0.05/0.08/0.09 等更紧的上限）；
  框内 MC-GS 对最强竞争方法领先 T1 +0.89/+0.83/+0.84、FLAIR +0.82/+1.10/+1.06。64 框版同风格保留为备选。四张图均有 600 dpi 矢量 PDF（`--pdf`，文字线条矢量、图像面板 600 ppi）。
- **数据导出（含原始数据，用户 09-13 要求）**：`Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/export/af16_phase3/`，
  48 文件约 10.9 GiB，全部是真实拷贝（可整目录搬走）：`volumes/` 九个臂两对比度的重建体加真值、
  `raw/` 26 线圈全采 k 空间与 ESPIRiT 灵敏度图（原本是指向 AF12 课题的符号链接，这里已展开）加 FLAIR 快门支撑、
  `mask/` 三维掩膜与二维 vd-Poisson 图案、`logs/` 课题 summary 与四个三期运行的打分、`figures/` 四张图的 PDF 与 PNG、
  `MANIFEST.tsv`（形状/类型/字节/md5）与 `checksums.md5`。脚本 `scripts/export_phase3_bundle.py [--with_raw]`。
  校验：`cd export/af16_phase3 && md5sum -c checksums.md5`。⚠️路径与 README 带受试者真实姓名，未去标识，只留服务器。
- 采集侧探针关闭（快门内互补 FLAIR 掩膜让所有臂各涨约 0.35 dB，余量不变），单独成记录。
- 顺带测出：快门已知零区在现掩膜上值 −0.013 dB（等于不值钱），在快门内掩膜上值 +0.225 dB。
- 新代码默认关且有测试：非局部 RED 先验、显式快门支撑 `data.<x>.known_zero_support`。

## 2. 待用户拍板
- 对外引用哪一个配方（现在是 P6；三期配方在每一格与锐度上都更好）。
- **1.0 dB 判据没达到**（余量约翻倍）。接受现状、改判据、还是放开学习先验（放开就得同时给对比方法也加）。
- **容量**：配方按预注册规则定在 3.6M（T1 最大）；6.0M 让最小余量最大（+0.465 / +0.474 对 +0.497 / +0.425），
  代价是 T1 少 0.031 dB、显存与时间多约一半。两者六格都第一、锐度门都过。
- AF16 留出受试者（一期至今从未跑过）。
- 以后 FLAIR 采集是否改用快门内掩膜（对所有方法都好，对余量无帮助）。

## 3. 下一步
- 若继续：只剩一件便宜事——用 `--from_ckpt` 给三期配方补一次复数口径打分（它带相位 TV 0.03）。
- 本期试过并关闭的：非局部 RED、自适应密化、矢量 TV、私有残差高斯、双掩膜采集设计。

## 4. 已知的坑
- 每卡一个训练；调度器只在完全空闲的卡上起（4/5 是本线的，2/3 归 AF12 二期会话，已与其约定）。
- `pgrep` 模式必须锚定 python 路径，否则会匹配并杀掉自己的 shell。
- 改 `scripts/p3_queue.txt` 要原子替换；调度器每轮重读，半行会被记成 bad queue line。
- 单臂配置把同一对比度喂进两个通道；表与一期取 plain 通道，账本行取 contrast 通道（两通道同名所致）。

## 5. 会话边界
- 本分支 `autoresearch-jyz-af16-phase3` 已推 origin（私有仓库 sober235/mcgs）并打 tag。
- **不合 main**：本仓库 CLAUDE.md 明令禁止跨血脉合并，`main` 与三条 AF16 分支没有共同祖先。
  交接协议里"合回 main"这一条在本仓库不适用，最新状态就在本分支尖端。
- 三条 AF16 分支互不合并：`autoresearch-jyz-af16`（一期）、`-phase2`（二期）、`-phase3`（本期，从 phase2 尖端起）。

## 6. 为什么
- 拉开差距靠的是换先验的"形状"（db4 最细层）、加唯一对两个对比度都有益的耦合项（common-edge）、
  以及用相位 TV 在最便宜的点买回锐度。两个大结构赌注（非局部先验、自适应密化）都有证据地关闭了。

## MC-INR（Niessen et al. 2025 复现）并入 AF16 课题（2026-09-13 晚，用户要求）
与 AF12 课题同一套做法（那边先做，分支 `autoresearch-jyz-af12-phase3`）：MC-INR 用**本课题的 `mask_af16_3d.h5`**、
本受试者的 k 空间与 ESPIRiT 图跑出来，重建体软链进 `recon/`，再用课题自己的 `t1t2_common.Pair.score` 重打分
（`scripts/score_mcinr_arm.py` → `logs/mcinr_{T1,FLAIR}.json`），与对方自报数字九项指标最大相对偏差 **1.4e-7**。
- 头内：T1 31.590 / 0.8355 / 0.2325，FLAIR 32.243 / 0.8590 / 0.1711 —— 两个对比度都排 **8/11**，与它在 AF12 的位次相同；
  高于 3D-INR（+1.215 / +1.986 dB）、3D-DIP、ZF，低于 CG-SENSE（−0.764 / −0.584）、L1-ESPIRiT、MC-GS 三期。
- **不动任何余量**：三期六格的 binding arm 加它前后都是 GS-single，它离成为 bar 还差 4.1 / 2.7 dB。
- 它从 AF12 到 AF16 掉得最少（−0.60 / −0.80 dB，其余臂 −0.93 ~ −1.57）——论文方向唯一能看到的一点，但两个加速下都低于 CG-SENSE。
- 只看头内：整卷 PSNR 两个对比度都是倒数第二（只高于零填充），整卷 SSIM 全表最低。
- `summarize.py` / `readme_tables.py` / 两个 figure 脚本加了这一列；README 的自动表已刷新（自带 AF12 对照列）。
- 图：三期定稿图重渲成十方法版（后缀 `_p3_mcinr`，九方法的 `_p3` 原样保留）；48 框领先与定稿记录逐项一致
  （T1 +0.892/+0.826/+0.836，FLAIR +0.824/+1.102/+1.060），即加这一列没改到别的东西。
- ⚠️同样的两条不对等已写进 README：对手在本受试者上按真值调过参而 MC-INR 没有（但 CG-SENSE 无参可调仍高于它）；
  以及论文**协议**没对齐（缺互补采样等），所以只能读成"在共用掩膜、两次扫描的设置下低于 CG-SENSE"。
  本课题其实有一张同预算的互补 FLAIR 掩膜 `data/mask_af16c_flair_3d.h5`，补救实验可以在它上面做，**没做**。
- **数据包已补齐（2026-09-14，用户要求）**：`export/af16_phase3/` 原地重建，48 文件 10.89 GiB → **68 文件 11.27 GiB**，
  `md5sum -c` **68/68 OK**（`md5check.out`），零软链，旧包没留下游离文件（比对 MANIFEST 与实际文件，只差 README/清单/校验本身）。
  新增：MC-INR 的重建体（两个对比度）、本课题给它打的两行分、`mcinr/` 出处目录（论文单位输出、逐层损失、meta、自查、
  口径复核、层间抖动、三份文档含保真度自审）、十方法版定稿图 PNG+PDF，以及 `README_topic.md`。
  `raw/` 仍是 26 线圈满采 k 空间与 ESPIRiT 图（9.0 GB，解引用）。生成脚本 `scripts/export_phase3_bundle.py --with_raw` 已同步更新。
  与 AF12 的 `export/af12_p14_mcinr/`（66 文件 11.23 GiB）同结构同清单格式，可并排读。
  仓库只进清单三件，数据留盘上；⚠️未去标识，不出服务器。
- ⚠️本工作树 `mcgs_af16_p3_wt` 是这次新建的：`mcgs_af16_wt` 当时在 `autoresearch-jyz-af20` 上跑实验，不能动它的分支。
