# jiangyizhou 3D T1W + FLAIR，回顾性 AF 12：MC-GS 自动调参（P10）结果与上下文总结（2026-09-12）

> 这份文档放在 `main` 上只为给冷启动的会话和协作者一个入口。代码、账本、逐轮日志、记录都在框架血脉的分支
> `autoresearch-jyz-af12`（tag `handoff/2026-09-12-jyz-af12`），归档在服务器 `/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou/`。
> `main` 与框架血脉无共同祖先，按仓库 CLAUDE.md 的规矩不合并、不 cherry-pick；本文档是独立提交的说明文件。

## 1. 数据与口径
- 受试者 vol3_jiangyizhou（MC_Jia CFL），26 线圈，256×256×176；T1 = 3D FSP-GRE（PE 平面全采），FLAIR = 3D MX3D（厂商椭圆 shutter，支撑 0.7577）。
- 一张 house vdPoisson 掩膜同时用于两对比度（`data/mask_af12_3d.h5`，3847 点 = 8.54%，calib 24，读出全采）：名义 AF 12，对各自支撑 T1 11.71×、FLAIR 9.74×。
- 真值 = 全采数据经同一 ESPIRiT 图；幅值口径；要引用的是头内 PSNR / SSIM / HFEN（整图 SSIM 被空气噪声主导）。
- 对手线：L1-ESPIRiT、收敛 TV、JTV 三族的 λ 都在 jyz 上按真值扫过（与 MC-GS 同等特权），表取每族最优 λ；循环的门槛取每格所有 λ 的最好值。

## 2. 结果（4000 步，头内 PSNR · SSIM · HFEN，T1 / FLAIR）

| 臂 | T1 | FLAIR |
|---|---|---|
| **MC-GS P10** | **36.944 · 0.9503 · 0.0947** | **36.268 · 0.9397 · 0.0958** |
| MC-GS P10 两种子平均（2 倍算力） | 36.990 · 0.9507 · 0.0943 | 36.295 · 0.9401 · 0.0954 |
| GS-single P10（单对比度） | 36.897 · 0.9495 · 0.0965 | 36.323 · 0.9405 · 0.0944 |
| P6（AF16 循环终配方原样跑 AF12，此前的暂定最优） | 36.753 · 0.9480 · 0.1009 | 36.165 · 0.9374 · 0.1002 |
| MC-GS v3.2（在 hyw 上选的旧主推） | 36.299 · 0.9423 · 0.0965 | 36.039 · 0.9369 · 0.1025 |
| L1-ESPIRiT | 36.426 · 0.9469 · 0.1036 | 35.757 · 0.9368 · 0.1000 |
| CS-SENSE 收敛 TV | 36.162 · 0.9446 · 0.1102 | 35.581 · 0.9353 · 0.1073 |
| Joint SENSE（JTV） | 36.070 · 0.9442 · 0.1143 | 35.624 · 0.9352 · 0.1073 |
| ZS-SSDU | 35.724 · 0.9422 · 0.1330 | 35.034 · 0.9353 · 0.1297 |
| CG-SENSE | 33.514 · 0.8981 · 0.1741 | 33.957 · 0.9032 · 0.1285 |
| P-LORAKS / 3D-INR / 3D-DIP | 31.99 / 31.64 / 31.36 | 31.18 / 31.19 / 30.81 |

- P10 对非 GS 最强臂（全部是 L1-ESPIRiT）的头内余量：PSNR +0.518 / +0.511 dB，SSIM +0.0030 / +0.0030，HFEN +0.0087 / +0.0042；整图 PSNR 40.272 / 39.692 第一；对 P6 +0.191 / +0.103 dB。
- 逐层（完整脑层，三方向，对全部经典臂）：P10 全部第一（T1 96/96、121/121、118/118；FLAIR 94/94、120/120、118/118）。
- 联合 − 单臂：T1 +0.047，FLAIR −0.055（耦合把质量往 T1 挪，FLAIR 每一层单臂略高）。

**P10 配方** = P6 只改小波先验三处：db4 基（原 db2）、只用最细一层（原 3 层）、T1 强度 0.12 / FLAIR 0.10（原 0.10 / 0.07）；其余不变（edge-TV 0.10 / κ_q 0.85，3.6M 点，冷启动修复，4000 步）。
四次晋级（P7 两层 → P8 db4 → P9 一层 + FLAIR 0.085 → P10）全部来自小波先验的"形状"；小波强度、edge-TV、κ_q、容量、预算、warm-up、频带权重、学习率、初始支撑等单旋钮全部证否或为 reshape（详见分支上的程序文件）。
教训：容量与尺度学习率在 1500 步筛选是 keep、到 4000 步变成 T1 换 FLAIR 的 reshape，此类旋钮只能在晋级预算判。

## 3. 设计 B（用户 2026-09-12）：T1 全 AF12；FLAIR 对比方法 AF12，MC-GS 的 FLAIR 用 AF11 掩膜
新掩膜同一生成器重画，名义 AF 11，4173 点（9.26%），对 FLAIR 支撑 9.05×（AF12 掩膜 9.74×）。只重跑 MC-GS（P10）：
T1 36.945（不变），FLAIR **36.637 · 0.9439 · 0.0906**，对 AF12 的 GS-single +0.314 dB、对 L1-ESPIRiT +0.880 dB，三个轴面全部第一。
这不是同口径对比（MC-GS 的 FLAIR 多 7% 采样），图题里注明。

## 4. 保留意见
- 复数口径 |rec − GT| 下 L1-ESPIRiT 第一（33.449 / 32.961），P10 第二（33.418 / 32.887）；领先只在幅值口径。
- 配方与对手 λ 都在 jyz 自身真值上选，未做留出检验。
- P10 锐度 0.858 / 0.823，比 P9 低 0.009（允许边缘）；T1 小波强度封在 0.12。
- 未纳入的边际项：尺度学习率 0.005（+0.011 / +0.048，T1 HFEN −0.0002）、sym4（≈ db4）、6000 步（+0.009 / +0.016）。

## 5. 图（服务器 http://localhost:8765/t1flair_v3_af12/，同名文件在 /home/congcongliu/figs/t1flair_v3_af12/）
- 紧凑三面图（橙黄标注 PSNR/SSIM/NMSE，误差只画头内，174/110/47）：`t1flair_v3_af12_{T1,FLAIR}_row3planes_compact_174_110_47_headerr.png`；设计 B 版本：`..._compact_flair_af11_174_110_47_headerr.png`（FLAIR 图左侧显示 AF11 掩膜）。
- 最佳单层排图：T1 173、FLAIR 178、共用 177：`t1flair_v3_af12_{T1,FLAIR}_row_axis0_{173|178|177}.png`；设计 B 的 177：`..._flair_af11_axis0_177.png`。

## 6. 在哪里找
- 分支 `autoresearch-jyz-af12`（worktree `/data0/congcong/code/GS/mcgs_af12_wt`）：`autoresearch_jyz_af12.md`（规则、逐轮日志、verdict）、账本 `.tsv`、`docs/superpowers/records/2026-09-12-autoresearch-jyz-af12.md`、`docs/handoff/2026-09-12-autoresearch-jyz-af12.md`、课题镜像 `t1flair_af12_v3_jiangyizhou/`（README、`RESULTS_AF12_T1W_FLAIR.md`、脚本、图）。
- 归档：`/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou/`，原始重建体在 `recon/`（各臂幅值与复数 npy）和 `runs_af12_loop/`（20 次关键运行的分数 json + 体积 npz + 日志，另有账本、程序、记录、交接的副本）。
- 复现 P10：`cd mcgs_af12_wt && SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh <EXP> <GPU> 4000`（运行器已烤入 P10）。
