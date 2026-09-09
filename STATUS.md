# STATUS — 2026-09-10 03:10（分支 `autoresearch-t1t2`，tag `handoff/2026-09-10`）

## 1. 已完成且已验证
- **MC-GS joint v2 在 hyw T1+T2 八法表上头内 PSNR 第一**：T1 34.978 / T2 35.429（CS-SENSE 34.607 / 34.219），整图 PSNR、NMSE、头内 HFEN 也第一，头内 SSIM 第二（差 0.006 / 0.002）。
  ```
  cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework
  SAVE_VOLUMES=1 ./autoresearch_t1t2.sh ar3_P2_full4k 4 4000 \
    --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn \
    --set loss.edge_tv_kappa_source=zf --set gaussian.initial_num_points=1200000
  # 原始输出（results_arc/ar3_P2_full4k.log 末尾）
  # T1    psnr=39.249 ... head_psnr=34.978 head_ssim=0.9373 head_hfen=0.0863 ... sharp=0.912
  # T2    psnr=40.174 ... head_psnr=35.429 head_ssim=0.9474 head_hfen=0.0665 ... sharp=0.857
  ```
- 配方只有四项改动，逐项单独验证（见 `autoresearch_t1t2.md` Round 0–3）；默认路径逐位不变，174 测试通过。
- 迁移检验：hyw T1+FLAIR 同配方第一（34.950 / 33.492）；jiangyizhou 第二（35.520 vs 35.762，35.056 vs 35.084），预注册密度规则失败（1.59M 点 T1 反降到 35.273）。
- joint vs single（修复后）：T1 +0.30、T2 −0.25。三个归档课题的表、图、README 已加 v2 臂并镜像进仓库。
- 账本 `autoresearch_t1t2.tsv` 34 行，每行有精确参数与 commit。

## 2. 待用户拍板
- 对外表述选哪句："调参受试者两种配对第一、留出受试者第二"（诚实版）还是只报 hyw。
- 头内 SSIM 缺口（0.006 / 0.002）是否继续追：拟合侧平滑已证明够不着，剩下的路是评分口径或方法侧。
- 分支：本仓库 `main` 与本线无共同祖先（CE 那条线），本次没有强行合并；是否把 `master` 快进到本分支当作本线的 main。
- 待判视觉门：v2 的十列图（`http://localhost:8765/t1t2_af12/t1t2_af12_{T1,T2}_plane{0,1,2}.png`）。

## 3. 下一步
- 若要 jiangyizhou 也第一：不是容量（已证否），候选 = 逐受试者的 lambda_geo（jyz 头大、噪声不同）或 kappa 分位；先在 jyz 做 1500 步单旋钮筛选，别碰 hyw 的配方。
- 其余四个 CFL 受试者（vol2/4/5/6/7）一条命令可克隆：`T1FLAIR_AF12_retro/scripts/setup_subject.sh`，然后同配方跑 `ar5_tf_*`。
- AF16 T1+FLAIR（并行会话 claude-06，worktree `mcgs_af16_wt`，分支 `t1flair-af16-retro`）完成后，把 v2 配方也跑一遍。

## 4. 已知的坑与别重做
- 别再试：intensity 初始化、密化（fixed 尺度下 split 不触发；3NN 下被初始点数支配）、position_lr、lambda_geo 0.07、T1 高频权重 1.0/3.0、kappa 300 步刷新、分位 0.75、1.5 倍尺度、tv_eps 0.06；平滑侧三旋钮（λ0.12/分位 0.90/w_floor 0.20）= T1 换 T2 的重塑。
- 容量：T1 在 120 万点封顶，再加就过拟合采样 k（e_unobs 单调升）；T2 到 240 万还涨；几何共享跟 T1 走。密度规则（点/支撑体素）在 jyz 证否。
- 会话看门狗会按 MemFree 杀后台任务（主机其实有 900+ GB 可用）：长跑一律 `setsid nohup` 脱离；工具 shell 是 zsh，`$VAR` 不分词，多参数启动写成 bash 脚本。
- GPU 与他人共卡会慢 20 倍（09-10 01:00 GPU 2/3），读时间前先看 `nvidia-smi --query-compute-apps`。
- 密化 `scale_threshold` 必须低于初始尺度才会 split。

## 5. 关键决定的"为什么"
- 竞技场选 hyw T1+T2：README 定的 canonical 课题，且 T2 有真值；T1+FLAIR 留作迁移检验。
- 读数用头内 PSNR 的最小余量：整图 SSIM 被 `ecalib -c 0.0` 留下的空气噪声支配（见 T1T2 README）。
- 平滑旋钮不晋级：预注册规则要求两对比度 SSIM/HFEN 不劣于父配置，越过它就是"重塑"，与 §18 的教训一致。
- 密度规则先预注册再跑：迁移受试者上任何调参都会污染"留出"的含义，失败也如实记录。
