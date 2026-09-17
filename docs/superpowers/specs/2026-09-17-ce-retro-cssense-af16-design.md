# CE 回顾性对比实验：CS-SENSE 当 GT，泊松盘 AF16，11 臂（设计）

日期：2026-09-17。课题目录 `/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16/`，分支 `ce-retro-cssense-af16`（从 `Plain_Enhanced` 切出，worktree `/data0/congcong/code/GS/mcgs_ce_retro_wt`）。

本文档是一次全新的对比实验的设计，不是前瞻 CE 课题（`CE_pre_post_contrast/`）的延伸，也与另一会话正在做的 `CE_retro_gt/`（UCSF-PDGM 合成 GT）无关。三者互不写对方目录。

## 1. 目的与范围

把 UIH 3D T1 GRE 平扫/增强这对数据上的 CS-SENSE 重建当作真值，用真实 42 线圈 CSM 合成 k 空间，在变密度泊松盘 AF16 下重跑 11 个重建臂，并用有真值的指标评估。MC-GS 是唯一的提出方法，其余十臂（含 GS-single）都是对比方法。头条余量 = MC-GS − GS-single。

臂：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR、GS-single、MC-GS。模型架构和超参数一比一对齐 jyz T1+FLAIR AF16 定稿（`Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/`），见第 5 节。

不做的事：MC-GS 不做任何调参（配方从 jyz 三期冻结后原样搬）；不跑 L1-ESPIRiT（09-15 已移出正式对比）；平扫/增强不用不同掩膜；不做无噪版本；不碰 PDGM 线。

## 2. 数据合成

**真值 x_GT**（平扫、增强各一个，同一网格）：
`CE_pre_post_contrast/tv_sense/cssense_o200_lam0.002_full_{pre,post}_complex.npy`，形状 (256, 232, 176) = (S-I 读出, A-P, L-R)，complex64，已按测量 k 空间做过 α 复标度（pre α≈1.016 实数）。原样使用，不再缩放。md5 与导出包 `export/ce_l1spirit_20260915/volumes/{pre,post}/cssense_{scan}_complex.npy` 一致。

**线圈图 S**：真实 42 线圈，
`/data0/congcong/code/NV-Generate-CTMR/recon_real/work_ce_t1_16887049370010/csm.npy`（平扫）、`.../work_ce_t1_16887049370060/csm.npy`（增强），形状 (42, 256, 232, 176) complex64。各扫描用自己的 CSM，与当初 CS-SENSE 重建一致。

**正向模型**：y_full = F₃(S · x_GT)，F₃ 为 `ce_common.fft3c` 同款中心化正交三维 FFT（`ifftshift → fftn(norm="ortho") → fftshift`，作用在最后三轴）。读出轴 0 全采，掩膜作用在 PE 面 (232, 176) 上并沿读出轴广播。

**写盘布局**（臂脚本已读的格式）：
```
sim/work_pre/ksp_full.npy   (42,256,232,176) complex64  含噪全 k 空间（第 4 节），掩膜外 = 0
sim/work_pre/csm.npy        (42,256,232,176) complex64
sim/work_pre/mask2d.npy     (256,232,176)    float32     沿读出广播
sim/work_post/...           同上
sim/gt.npz                  pre, post: (256,232,176) complex64；mask2d: (232,176) bool；
                            scale_pre, scale_post: P99.9(|x_GT|)
sim/meta.json               种子、σ、kres 复现值、掩膜统计、脚本版本
```
给 GS 训练器另写 h5（`sim/gs/{pre,post}_kspace.h5`、`sim/gs/{pre,post}_csm.h5`、`sim/mask/mask_af16_3d.h5`）：k 空间存**含噪、未乘掩膜**的全点版本，掩膜由训练器自己乘（jyz 的非 presampled 路径）；键名与轴布局以训练器提交 e778bf2 的 `paired_gs/dataset.py` 为准（键优先级 kspace/ksp，线圈轴按 size ≤ 64 推断，readout_axis 0，phase_axes [1, 2]）；合成脚本写完后在冒烟阶段用训练器加载一次作为验证。

**数据尺度（jyz 口径）**：每个扫描各自的 scale = P99.9(|x_GT|)（`gt.npz` 的 scale_pre / scale_post）。臂的输入 y 与真值都除以该扫描的 scale；联合类臂的两个通道各用各的 scale；臂输出在该尺度下直接与 GT/scale 比，不做幅度复标度。

## 3. 掩膜

- 生成器：BART `poisson`（`/home/congcongliu/src/bart/bart`），命令形如
  `bart poisson -Y 232 -Z 176 -y a -z a -C 24 -v -s <seed> out`：变密度、**不加椭圆快门**（整个 232×176 矩形都可采样，四个角也有采样点；用户 09-17 定，与 jyz 的 vdPoisMex ellipse=0 一致）、24×24 ACS 全采、确定性种子。不用 vdPoisMex（它以系统时间为种子，不可复现）。
- 目标点数：整平面 232×176 = 40832 点 ÷ 16 = **2552** 点（含 ACS），即用户 09-14 定义下 AF = 16.00（= BART pics 打印的 Acc）。
- 找法：固定种子 0，在 a 的细网格上扫，取点数恰为 2552 的第一张；扫不到则取点数最接近且大于 2552 的一张，从距 DC 最远的点起按半径逐个去掉多余点（jyz AF24 的兜底法）。若种子 0 下扫描全程不能得到 ≥ 2552 的可修剪张，换种子 1、2 重复；使用的种子与 a 写进 meta。
- 平扫和增强同一张掩膜。
- 校验：点数 = 2552；ACS 24×24 全为 1；ACS 中心与 DC 一致（(116, 88)）；四个角（归一化半径 > 1 的区域）有采样点；沿读出恒定。
- 产物：`sim/mask/mask_af16_pe.npy`（(232,176) uint8）、`sim/mask/mask_af16_3d.h5`（键 `mask`，(256,232,176) uint8）、`sim/mask/mask_af16.json`（点数、AF、种子、a、bart 版本与命令）、`figures/mask_af16.png`（PE 面图 + 径向密度曲线）。
- **审批门**：掩膜图给用户看过、点头后，才允许开任何 GPU 作业。

## 4. 噪声

每个扫描各自定 σ：
1. 取该扫描的原前瞻测量 y_meas（`work_ce_t1_*/ksp_full.npy`）与原掩膜 M₀（`mask2d.npy`）。
2. 残差 r = M₀ · (y_meas − α · F₃(S · x_GT))，α 为 ce_common 定义的最小二乘复标度 α = Σ conj(A x)·y / Σ |A x|²。
3. σ = sqrt(mean_{M₀ 采样点、全线圈} |r|² / 2)，即实部、虚部各自的标准差。
4. 合成：ksp_full_noisy = F₃(S · x_GT) + σ·(N(0,1) + i·N(0,1))，噪声加在**全部** k 空间点上（种子固定：pre 100、post 200），然后再乘掩膜写盘。臂只看到采样点，所以"全点加噪再掩膜"与"只在采样点加噪"等价；全点含噪版本同时能喂训练器的非 presampled 路径。
5. meta 记录：σ、‖r‖/‖M₀y‖（应复现 CS-SENSE 全掩膜 kres 0.18487 / 0.14969）、σ 折算到像域的白质 SNR：正交 FFT 下每线圈像域噪声标准差仍为 σ，SENSE 伴随 Σ_c conj(S_c)·x_c 的逐体素噪声标准差 σ_img = σ · sqrt(Σ_c |S_c|²)，白质 SNR = median_WM(|x_GT|) ÷ median_WM(σ_img)，白质掩膜取 `enhance_rois_cssense.npz` 的"正常白质"。

口径说明：残差里除热噪声外还含 CSM 误差和被 TV 抹掉的细节，所以 σ 略高于纯热噪声；这是用户选定的口径。

## 5. 各臂：对齐 jyz AF16 定稿

臂脚本从 `Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/scripts/` 搬（权威副本），做三处参数化：对比度名 T1/FLAIR → pre/post；线圈数 26 → 由数据推断（42）；网格 (256,256,176) → (256,232,176)。`t1t2_common.py` 的 ROOT/R 改到本课题，GT 加载改为读 `sim/gt.npz`（不再从全采 k 空间做 SENSE 伴随）。

| 臂 | 设置（照搬） | 本课题差别 / 风险 |
|---|---|---|
| ZF、CG-SENSE | `recon_classical.py`：ZF = SENSE 伴随；CG-SENSE 60 步、Tikhonov 0.01；真 3D，线圈分块 | 42 线圈 |
| CS-SENSE | 同脚本 `--admm_outer 200 --admm_cg 8 --mu 0.1`，各向同性 3D TV；λ 网格 {2.5e-4, 3.5e-4, 5e-4, 7e-4, 1e-3, 1.4e-3, 2e-3} | λ 重扫 |
| Joint SENSE | `recon_joint_cssense.py`：矢量 TV（两对比度 6 分量联合收缩）、外层 200、CG 8、μ 0.1；λ 网格 {5e-4, 7e-4, 1e-3, 1.4e-3, 2e-3} | 通道 pre,post |
| L1-SPIRiT | `recon_spirit.py`（mcgs_spirit_wt 版）：核 {3,5} × λ {0, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3}、Tikhonov 0.01、150 步、db2 四层、FISTA、`--calib acquired`、逐读出层 2D | ACS 真 24×24；42 线圈下核 5 的标定方程 (24−5+1)²=400 个 < 未知数 42×25=1050 个，欠定；核 3 未知数 378 个 < 484 个方程。网格里两种核都跑，按 GT 选；若核 5 发散或明显差，写进 README |
| ZS-SSDU | `recon_zsssdu.py`：ResNet 5 块 64 通道、展开 10、CG 6、batch 4、lr 5e-4、μ 0.05、≤100 epoch、patience 10、Γ 10% 早停、Θ/Λ 80/20 每步重采（seed 1234+step）、seed 0、逐读出层 2D | 42 线圈更慢 |
| 3D-DIP | `recon_dip_inr.py --method dip`：ConvDecoder3D 通道 [256,128,64,64]、固定 N(0,0.1) 噪声输入、BN、零初始化输出头、5000 步、lr 3e-3、每 100 步评、TV 0、留出 5% PE 线（seed 0）选 checkpoint | 低分辨种子网格按 (256,232,176)/16 取整 |
| 3D-INR | 同脚本 `--method inr`：hash 16 级、T=2^19、每级 2 特征、分辨率 16→512、MLP 64×3、1500 步、lr 5e-3、每 50 步评、留出 5% | 同上 |
| MC-INR | `/data0/congcong/code/GS/mcinr_niessen2025`（`run_volume.py`，已有 `--ksp_T1/--ksp_FLAIR` 外部 k 空间入口）：hash 16 级、log2T 20、MLP 32→64→64→2N、lr 1e-2、每层 200 步、加权 k 空间 MSE、最后一步、seed=层号；逐读出层，一网出两对比度 | pre 走 T1 槽、post 走 FLAIR 槽；输出用本课题评分器重算 |
| MC-GS | 训练器提交 e778bf2（`git archive e778bf2 paired_gs` 钉进 `scripts/_trainer_e778bf2/`）。基础配置从 `mcgs_af16_p3_wt/configs/t1flair_v3_af16_jyz.yaml` 派生为 `configs/ce_retro_af16.yaml`：data 段换本课题 h5，plain=pre、contrast=post，网格 (256,232,176)，**known_zero 两通道都关**（合成数据无厂商快门）。CLI 原文：`--iters 4000 --lambda_geo 0.10 --warmup 500 --max_radius 20 --kspace_high 2.2 --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn --set loss.edge_tv_kappa_source=zf --set gaussian.initial_num_points=3600000 --set loss.wav_name=db4 --set loss.wav_levels=1 --set loss.lambda_wav_plain=0.12 --set loss.lambda_wav=0.10 --lambda_edge 0.001 --lambda_ptv 0.03 --save_volumes` | 360 万点 × 42 线圈显存待探（jyz 26 线圈 32.9 GB）；若 OOM，先降 tile，再报告用户决定 |
| GS-single | 同配方同训练器；`configs/ce_retro_af16_single_pre.yaml`、`..._single_post.yaml` 把同一扫描喂进 plain 与 contrast 两通道；引用 plain 通道 | 同上 |

**λ 协议**：CS-SENSE、Joint SENSE、L1-SPIRiT 按本数据 GT 的头内 PSNR 取最优（每个扫描各选各的；Joint SENSE 一次 λ 运行同时产出两个扫描的体，每个扫描从这些运行里各挑自己最优的 λ；jyz 口径）；最优落在网格边缘时向外扩一档再扫，最多扩两次。ZS-SSDU、3D-DIP、3D-INR 不看 GT（留出/早停）；MC-INR 无选择；MC-GS、GS-single 不调。

**不对称说明**：MC-GS 配方在 jyz 上冻结、搬来不调，对手 λ 却在本数据 GT 上扫。这对 MC-GS 偏保守，与 jyz 留出受试者的做法一致，写进 README。

## 6. 评分

评分器 = `t1t2_common.py::Pair.score` + `mcgs_af16_wt/diag_artefact/headonly.py` + `T1FLAIR_AF12_16_20_24_trend_v3_jiangyizhou/scripts/head_nmse.py` 的头内 NMSE，搬进 `scripts/`，GT 换成 x_GT/scale。

- 头掩膜：|GT| > 0.05 · P99.9(|GT|)。
- 引用指标（每臂每扫描）：头内 PSNR（峰值 max|GT|，无幅度复标度）、头内 SSIM（skimage，7³ 窗，data_range = max|GT|）、头内 NMSE、头内 HFEN（LoG σ 1.5、15 抽头、去直流）。
- 同时记录：whole@max 的 PSNR/SSIM/NMSE；复数口径 |rec − GT| 的 PSNR。
- 强化保真：六个参考区沿用 `CE_pre_post_contrast/lesion_candidates/enhance_rois_cssense.npz`（病灶核心、核心外 0–3 mm、正常白质、正常皮层、血管/静脉窦、脉络丛），每臂算 增强−平扫（白质单位，各扫描按自身白质中位数归一），报与 GT 同量的差；外加颅外背景均值（头掩膜膨胀 6 mm 之外，增强图）。
- 逐层头内 PSNR：三个面全部层；最优层规则同 jyz：两扫描取 min 的 MC-GS 余量，GS-single 不算"对手"但 MC-GS 必须超过它。
- 产物：`logs/<arm>_<scan>.json`、`logs/summary.json`（含每臂选定 λ）、`logs/enhancement_fidelity.json`、`logs/slice_scores.json`；README 表由 `readme_tables.py` 生成。

## 7. 图

- 正式三面图：`fig_three_planes_compact.py` 版式（误差只画头内、jet 固定 0–0.1、格内橙黄 PSNR/SSIM/NMSE、放大框无数字、横断眼睛朝上、色条只在横断行、PDF 600 dpi），11 列 ZF…MC-GS，层 z201 / y131 / x48（病灶层）。文件 `figures/ce_retro_af16_three_planes_201_131_48.{png,pdf}`。
- 强化图：CE 课题 `ce_figure_paper.py --diff` 的 9 行版式（每面加 增强−平扫，列底四个参考区数值）。文件 `figures/ce_retro_af16_enhancement.{png,pdf}`。
- 掩膜图：`figures/mask_af16.png`（第 3 节审批门）。

## 8. 调度与算力

- 调度器：`CE_pre_post_contrast/scripts/ce_orchestrate.py` 的副本，加计划 `retro16`；空卡判据不变（利用率 < 30% 且非本用户占用；作业按 need MB 申请）。每卡同时只放一个训练作业（09-11 卡死教训）。不杀别人的作业，冲突交用户。
- 作业图：合成（CPU）→ 掩膜 → [审批门] → 冒烟 → ZF/CG → CS-SENSE 网格 ×2 → Joint 网格 → SPIRiT 网格 ×2 → ZS-SSDU ×2 → DIP ×2 → INR ×2 → MC-INR → GS-single ×2 → MC-GS → 评分 → 图 → README。
- 估算：GPU 12 到 14 小时，两张卡 6 到 7 小时墙钟。GPU 7 此刻被另一 CE 会话的探针占用。
- 长任务一律 `setsid nohup`，日志进 `logs/`，绝不放 scratchpad（09-07 重启教训）。

## 9. 目录、git、命名

```
Results_20260908/CE_retro_cssense_af16/
  README.md  STATUS.md
  scripts/            合成、掩膜、臂、评分、图、调度；_trainer_e778bf2/
  configs/            ce_retro_af16.yaml、ce_retro_af16_single_{pre,post}.yaml
  sim/                work_pre/ work_post/ gs/ mask/ gt.npz meta.json
  recon/              <arm>[_suffix]_<scan>_complex.npy（GS 臂另有 _mag.npy）
  logs/               评分 json、summary.json、作业日志、pid、ALL_DONE_retro16
  figures/            正式图、强化图、掩膜图、cache/
  export/             收尾时的数据包 + MANIFEST + md5
```
git：分支 `ce-retro-cssense-af16`，worktree `mcgs_ce_retro_wt`，仓库内目录 `ce_retro_cssense_af16/` 镜像 scripts、configs、README、STATUS、小 json、图；体数据与 h5 不进 git（.gitignore 已挡 npy/pth）。作者 `Congcong Liu <sober235@users.noreply.github.com>`，无 AI 痕迹。推送用 `git -c credential.helper='!gh auth git-credential' push`。

## 10. 验证门与测试

全量 GPU 作业之前必须全部通过：
1. 合成自检：掩膜 2552 点、ACS 全采、椭圆外为零、沿读出恒定；x_GT 套回原前瞻掩膜的 kres 与 0.18487（pre）/ 0.14969（post）之差 < 1e-3，证明 FFT 与 CSM 约定与 CE 管线一致；加噪后实测 σ 与目标相对误差 < 2%。
2. 算子天花板：无噪、全采 k 空间上，x̂ = Σ_c conj(S_c)·F₃⁻¹(y_c) ÷ Σ_c |S_c|² 在头内与 x_GT 的相对误差 < 1e-4（证明正向/伴随算子自洽）；再用 CG-SENSE（Tikhonov 1e-6，60 步）确认头内 PSNR > 60 dB。
3. 口径互验：任一臂导出的体经共享评分器重算，与臂自报头内 PSNR 之差 < 2e-4 dB（jyz `check_gauge.py` 做法）。
4. 冒烟：每臂短跑（几十步）能跑通并写出正确形状的 `_complex.npy`；MC-GS 用 360 万点 × 42 线圈跑 50 步记录峰值显存。
5. 训练器加载验证：e778bf2 的 dataset.py 能读本课题 h5，报告的掩膜点数 = 2552。
6. 单元测试（pytest，`scripts/tests/`）：掩膜生成（点数、ACS、椭圆、确定性）；合成写盘（形状、掩膜外为零、σ）；评分器（已知答案：GT 自评 PSNR = inf、加已知噪声后 PSNR 与解析值一致）。新代码按 TDD 写。

## 11. 结论措辞约束

- GT 是 TV 正则化解，CS-SENSE 家族在本数据上有主场优势，所有结论都要带这句。
- 噪声按残差定，略高于纯热噪声。
- 同一掩膜下 增强−平扫 的相干伪影部分抵消，差值图不能单独当画质证据。
- 可写：AF16 泊松盘下各臂对同一 GT 的头内指标与强化保真；MC-GS 与 GS-single 的余量。不可写："优于前瞻实验"（口径不同）、任何灌注措辞、病灶检出。
