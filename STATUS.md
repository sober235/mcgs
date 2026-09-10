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
- AF16 迁移（09-10 12:10，同配方不改）：hyw T1 33.465 / FLAIR 31.943（ZS-SSDU 33.665 / 32.685，−0.20 / −0.74）；jyz T1 33.982 / FLAIR 33.640（CS-SENSE 34.410 / ZS-SSDU 34.005，−0.43 / −0.36）→ 全部第二，仍比冷启动 joint +1.1~1.4 dB；并行会话正在 jyz AF16 上跑自己的循环（分支 autoresearch-jyz-af16）。
- 出图：三课题的十列图灰度窗改为头部 P95；紧凑版式（掩膜在左下、等大面板、单 colorbar、旧 joint 臂删除、v2 标为 ours）的单层最优图与三轴面图已出（`fig_slice.py` / `fig_planes_v2.py`，逐层扫描 `find_best_slice_v2.py`）。
- 账本 `autoresearch_t1t2.tsv` 36 行，每行有精确参数与 commit。

- **经典基线复核（09-10 下午）改变了名次**：原 TV 版 CS-SENSE 只跑 35 轮未收敛（沿用 CE 设置），200 轮后高 0.35~0.6 dB；L1 小波 ESPIRiT（BART，100 步 FISTA，λ 在 hyw 上按对比度选定后原样迁移）五个通道全部第一（hyw 35.530/35.818/34.105，jyz 36.334/35.720），比 v2 高 0.4~0.8 dB；联合矢量 TV 与单 TV 持平。v2 变为第二/第三，保留 hyw T1 最低头内 HFEN 与对冷启动臂 +1.4~2.3 dB。三个课题的表与 README 已更新（`## Classical arms revisited`）。P-LORAKS 秩扫描进行中（R=2）。
- **逐层复核（09-10 16:40）**：`find_best_slice_v2.py` 改为只对收敛/规范臂取余量（收敛 TV、联合 TV、L1-ESPIRiT、ZS-SSDU、DIP、INR、CG-SENSE、ZF），v2 在三个课题、三个方向的完整脑层（头体素 ≥20000）上**第一的层数为 0**；紧凑版图（`fig_slice_v2.py` 通用版、`fig_planes_v2.py`）已加入新臂并按真实名次重出（9fd8fa8）。

## 2. 待用户拍板
- 复核后对外表述：v2 不再是 AF12 第一（L1-ESPIRiT 领先），主张改为"冷启动路径的诊断与修复 +1.4~2.3 dB"，还是继续把循环目标改成追 L1-ESPIRiT（hyw 35.53/35.82）。
- 对外表述选哪句："调参受试者两种配对第一、留出受试者第二"（诚实版）还是只报 hyw。
- 头内 SSIM 缺口（0.006 / 0.002）是否继续追：拟合侧平滑已证明够不着，剩下的路是评分口径或方法侧。
- 分支：本仓库 `main` 与本线无共同祖先（CE 那条线），本次没有强行合并；是否把 `master` 快进到本分支当作本线的 main。
- 待判视觉门：v2 的十列图（`http://localhost:8765/t1t2_af12/t1t2_af12_{T1,T2}_plane{0,1,2}.png`）。

## 3. 下一步
- **第 11 轮（小波稀疏进 MC-GS，`loss.lambda_wav`，`paired_gs/losses/wavelet_l1.py`）已见效**：1500 步时加在 TV 之上 λ 0.03 给 T1 35.373 / T2 35.120（P2 为 35.007 / 35.022），λ 0.05 给 T1 35.447；替代 TV 更差。**240 万点 + 小波 0.03 + TV = T1 35.513 / T2 35.878，已到 L1-ESPIRiT 全预算的 35.530 / 35.818（差 −0.02 / +0.06），头内 HFEN 与 SSIM 两对比度都好于 P2，没有模糊代价**：小波先验解除了 T1 的容量上限。进行中：两个 4000 步晋级（`ar12_wav003_full4k` 在 GPU 4，`ar12_n24_wav003_full4k` 在 GPU 5）、240 万点 + 小波 0.05 筛选（GPU 7 排队）、0.1 替代 TV 与幅值模式两个筛选。晋级若两对比度都超 L1-ESPIRiT，再做三课题迁移检验（同配方不改）。
- **进行中**：P-LORAKS 全量（秩 40、R=2，每读出层一个 2D 问题，MATLAB 96 worker，三波五通道，预计到夜里）；`scratchpad/ploraks_finish.sh` 会逐个导入打分并自动刷新三个课题的表、逐层扫描和 12 张紧凑图。子集上它只有 28–30 dB（无校准低秩在面内 AF 11.7、16×16 校准区下结构性落后），入表只为完整。
- 若要 jiangyizhou 也第一：不是容量（已证否），候选 = 逐受试者的 lambda_geo（jyz 头大、噪声不同）或 kappa 分位；先在 jyz 做 1500 步单旋钮筛选，别碰 hyw 的配方。
- 其余四个 CFL 受试者（vol2/4/5/6/7）一条命令可克隆：`T1FLAIR_AF12_retro/scripts/setup_subject.sh`，然后同配方跑 `ar5_tf_*`。
- AF16 T1+FLAIR（并行会话 claude-06，worktree `mcgs_af16_wt`，分支 `t1flair-af16-retro`）完成后，把 v2 配方也跑一遍。

## 4. 已知的坑与别重做
- 经典臂必须查收敛：CS-SENSE 的 35 轮 ADMM 未收敛（200 轮高 0.35–0.6 dB）；L1-ESPIRiT 反而 100 步好于 300 步（早停正则），报告时写明步数。
- 本机 BART 无 GPU（`-g` 报错），CPU 32 线程约 4 分钟一次；LORAKS 工具箱在 /data0/congcong/code/LORAKS_v2，P-LORAKS 在 R=3 下单层超 1 小时不可用；SENSE-LORAKS 输出是 MATLAB 1/N 逆 FFT 尺度（乘 √N 回到口径），λ≤1e-2 不起作用、λ=10 退化出 NaN，已放弃。
- `pkill -f 模式` 会匹配到工具 shell 自己并把它杀掉（exit 144）；MATLAB 长跑与 GS 训练都要 setsid 脱离会话。
- CPU 被 MATLAB 占满时，pytest 会因 torch 默认开满线程而近乎卡死（09-10 一小时只过 15 个用例）：跑测试加 `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`。
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
