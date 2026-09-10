# STATUS — 2026-09-11 01:05（分支 `autoresearch-t1t2`，上次 tag `handoff/2026-09-10`）

## 1. 已完成且已验证
- **v3 是当前配方（09-11 00:25 定）**：v2 的冷启动修复 + 240 万初始点 + 共用复数小波稀疏 λ 0.05（加在 edge-TV 之上），4000 步。
  ```
  cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework
  SAVE_VOLUMES=1 ./autoresearch_t1t2.sh <EXP> <GPU> 4000 \
    --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn --set loss.edge_tv_kappa_source=zf \
    --set gaussian.initial_num_points=2400000 --set loss.lambda_wav=0.05
  # results_arc/ar12_n24_wav005_full4k_score.json（hyw T1+T2）
  # T1 head_psnr=35.771 head_ssim=0.9468 head_hfen=0.0831 psnr=39.912 sharp=0.867
  # T2 head_psnr=36.112 head_ssim=0.9531 head_hfen=0.0679 psnr=40.766 sharp=0.826
  ```
  头内 PSNR，对手是每张表最强的 L1-ESPIRiT：

  | 课题 | v3 | L1-ESPIRiT | 余量 |
  |---|---|---|---|
  | hyw T1+T2（调参对，`ar12_n24_wav005_full4k`） | 35.771 / 36.112 | 35.530 / 35.818 | +0.241 / +0.294 |
  | hyw T1+FLAIR（`ar13_tf_hyw_t1flair_v3b`） | 35.772 / 34.359 | 35.530 / 34.105 | +0.242 / +0.254 |
  | jiangyizhou T1+FLAIR，留出（`ar13_tf_jyz_t1flair_v3b`） | 36.337 / 35.935 | 36.334 / 35.720 | +0.003（打平，seed 43 为 +0.004）/ +0.215 |

  六个通道五个第一、一个打平；头内 HFEN 五个第一；头内 SSIM 全部略低于 L1-ESPIRiT（0.0001–0.005）；hyw 两个课题 T1 锐度 0.867 / 0.869，低于预注册下限 0.874。
- T1+T2 逐层（完整脑层，对手含 L1-ESPIRiT 与 P-LORAKS，单对比度消融不算对手）：T1 横断 / 冠状 / 矢状第一 42/47、83/88、90/93 层，T2 所有层第一。
- 单对比度 v3（同配方，单独重建）：T1 35.679、T2 36.227，单独也超 L1-ESPIRiT。joint 减 single = T1 +0.09、T2 −0.12。领先来自表示 + 小波先验 + 容量，不来自耦合。
- P-LORAKS（秩 40，R=2，每个读出层一个 2D 问题）hyw T1、T2 已完成（6.7 h / 8.0 h）：头内 PSNR 30.623 / 31.343、整图 35.512 / 36.462，锐度比 1.24 / 1.32，头内噪声远大于 GT。
- SSIM 诊断（`t1t2_af12_retro/scripts/ssim_breakdown.py`）：v3 在最平坦的五分之一头内领先，在中等梯度的三个五分位落后，原因是局部方差不足（局部 std 为 GT 的 0.86–0.95，L1-ESPIRiT 为 0.91–0.97），T1 另有局部相关略低。缺的是中尺度纹理，不是噪声，也不是边缘。
- 三个课题 README 的 v3 段落都已写好并镜像进仓库：T1+T2（c89bf4b）、jiangyizhou（d4b73f7）、hyw T1+FLAIR（409df68）。hyw T1+FLAIR：头内 PSNR 两对比度第一（+0.242 / +0.254），逐层 T1 第一 43/47、83/88、90/93，FLAIR 55/61、73/91、94/94，joint−single = +0.093 / +0.018（耦合不伤 FLAIR）。jiangyizhou：FLAIR 第一（+0.216），T1 打平，逐层 FLAIR 94/94、87/120、118/118，T1 18/96、77/121、56/118，joint−single = +0.080 / −0.074。三个课题的图都已探活（HTTP 200）。
- 已提交：v3 定稿与三表 v3 行（0eaa964）、紧凑图改用 v3（db673f0）、T1+T2 README v3 段落与逐层（c89bf4b）、P-LORAKS T2 入表（2580721）、第 16 轮预注册（12d13d9）、SSIM 诊断与归档辅助脚本（a5dbb8a）。
- 第 16 轮 T1 通道小波 0.06（`ar16_n24_wav005_T1p06`）按预注册规则通过：T1 35.855 / T2 36.110，对 L1-ESPIRiT 最小余量 +0.292（v3 +0.241），T1 头内 SSIM 0.9481 超过 L1-ESPIRiT 的 0.9471；T1 锐度 0.862，比 v3 离下限更远。暂定为 v3.1，等 geo007 与第 17 轮结果决定是否合并。
- v3.1（T1 通道 0.06）的迁移：jyz（留出）T1 36.458 / FLAIR 35.936，对 L1-ESPIRiT +0.124 / +0.216，T1 由打平变领先（单次运行），T1 头内 SSIM 0.9445 超过 L1、低于收敛 TV 0.9459，T1 锐度 0.884 过下限；hyw T1+FLAIR T1 35.850 / FLAIR 34.359（T1 +0.078，FLAIR 不变）。
- 第 16 轮 edge-TV 0.07（`ar16_n24_wav005_geo007`）按规则通过，是第一个满足全部预注册门的配置：T1 35.717 / T2 36.254（对 L1-ESPIRiT +0.187 / +0.436），T1 锐度 0.906 过 0.874 下限；T2 头内 SSIM 0.9552、两对比度头内 HFEN 0.0797 / 0.0646 都第一，T1 头内 SSIM 差 L1 0.0004。赛前记录的预测（SSIM 会降）错了。T1 0.06 与 edge-TV 0.07 都通过，按规则先在调参对上合并跑一次（`ar19_n24_T1p06_geo007`，`scratchpad/queue23.sh`），过锐度门且最小余量 > +0.187 就成为 v3.2，否则 edge-TV 0.07 单独成为 v3.2。
- 第 17 轮 FLAIR 通道小波 0.08（`ar17_tf_hyw_wavC008`，hyw T1+FLAIR）：FLAIR 34.549（v3 34.359，+0.190；对 L1-ESPIRiT +0.444），T1 不变 35.771；FLAIR 头内 SSIM 0.9205（L1 0.9221、ZS-SSDU 0.9235 仍略高）。按规则等 0.12 的结果二选一。
- 外部结果（并行 AF16 会话，**不是留出**）：它在 jyz AF16 上选出的配方（360 万点、小波 T1 0.10 / FLAIR 0.07、edge-TV 0.10/0.85）原样跑本循环的 jyz AF12：头内 PSNR 36.753 / 36.165、SSIM 0.9480 / 0.9374、HFEN 0.1009 / 0.1002、整图 PSNR 40.071 / 39.539，头内 PSNR、SSIM、整图 PSNR 两通道第一。jyz 是它的调参受试者，所以在 hyw 上交叉检验（第 20 轮）。
- 两个种子平均（seed 42 与 43 的幅值平均，2 倍算力）：调参对分对比度变体 T1 35.856 → 35.951、T2 36.079 → 36.148；jiangyizhou v3 T1 36.337 → 36.539（对 L1-ESPIRiT 由打平变成 +0.205）、FLAIR 35.935 → 36.049；头内 SSIM 升 0.0006–0.0021。增益与两次运行误差相关系数的预测一致。
- 仍成立的旧结论：冷启动修复在所有课题 +1.4~2.3 dB；v2 = 冷启动修复 + 120 万点（`ar3_P2_full4k`，34.978 / 35.429）。

## 2. 待用户拍板
- 对外表述：v3 在调参对与 hyw T1+FLAIR 上头内 PSNR 两对比度第一，留出受试者 T1 打平、FLAIR 第一；SSIM 全部略低；hyw 两课题 T1 锐度门未过。是否按这个口径写。
- T1 锐度下限 0.874 取自旧冷启动 joint 臂：v3 的颗粒度比与头内 HFEN 都更好，读作去噪。这条门是保持、改成对 GT 的双边带，还是看第 16 轮结果再定。
- v3 ×2（两种子平均，2 倍算力）是否作为主推结果。目前主推仍是单次 v3，v3 ×2 单列一行。
- 主推配方的表述（等第 20 轮）：v3 / v3.1 在 hyw 上选、在 jyz 上留出检验；AF16 会话的配方在 jyz 上选、在 hyw 上检验。若后者在 hyw 两个课题上也第一，可以写成"两个受试者互为留出，同一配方族都第一"。
- 分支：本仓库 `main` 与本线无共同祖先（CE 那条线），是否把 `master` 快进到本分支当本线的 main。
- 待看的图：`http://localhost:8765/t1t2_af12/t1t2_af12_T1_best_axis0_150.png`、`..._T2_best_axis0_144.png`、`..._T1_best3planes.png`、`..._T2_best3planes.png`。

## 3. 下一步
- 第 16 轮（预注册与判定规则见 `autoresearch_t1t2.md` Round 16）：`ar16_n24_wav005_T1p06`（T1 通道小波 0.06，另一通道保持 0.05）、`ar16_n24_wav005_geo007`（edge-TV 0.10 → 0.07），4000 步，`scratchpad/queue16.sh` 在 GPU 4、6 空出后自动起跑（约 01:32、01:53）。替代 v3 的配置要再原样跑两个 T1+FLAIR 课题。
- 第 17 轮（预注册见程序 Round 17）：FLAIR 通道小波 0.08 / 0.12，只在 hyw T1+FLAIR 上选（`ar17_tf_hyw_wavC008` / `wavC012`），`scratchpad/queue17.sh` 在 GPU 5、7 空出后起跑（约 02:15、02:25）；选中的值再原样跑 jiangyizhou。
- GPU 分布（03:58）：GPU 0 `ar20_xv_P6_hyw_t1flair`（第 20 轮，约 05:10）；GPU 1 `ar16_tf_jyz_T1p06`、GPU 2 `ar16_tf_hyw_T1p06`、GPU 3 `ar18_tf_hyw_t1flair_v3b_s43`（约 04:45）；GPU 4 调参对 v3 seed 43（约 04:00）；GPU 5 `ar18_n24_wav005_T1p06_s43`（第 18b 轮，约 04:52）；GPU 6 geo007（将落账）；GPU 7 `ar17_tf_hyw_wavC012`（约 04:20）。GPU 4/6 空出后 `scratchpad/queue21.sh` 起跑 T1 0.06 + FLAIR 0.08 组合（`ar19_tf_hyw_T1p06_wavC008`），`scratchpad/queue22.sh` 在剩下的 GPU 上起跑第 20 轮调参对（`ar20_xv_P6_t1t2`）。三个 x2_after 等 seed 43 落账后打两种子平均，`scratchpad/after_x2.sh` 随后刷新 T1+T2 与 hyw T1+FLAIR 的表。0.12 落账后：按第 17 轮规则选 FLAIR 强度，选中的值原样跑 jyz。
- 在跑：单对比度消融 `ar15_single_FLAIR_jyz_v3`（GPU 5，约 02:26）、`ar15_single_FLAIR_hyw_v3`（GPU 7，约 02:31）。已落账：分对比度变体 seed 43（与 seed 42 相同到 0.001 dB，但体素级差 25–29%）、jyz T1 单对比度 36.257（joint 比它高 0.08）、360 万点晋级（T1 35.662 / T2 36.233，按规则未采纳）。
- 某个 T1+FLAIR 课题的两个单对比度落账后：跑该课题 `scripts/extract_gs_volumes.py`，再 `bash $S/refresh_topic.sh <课题> <前缀> T1,FLAIR`；README 的 v3 段落数字取 `python $S/v3_facts.py <课题>`；最后 `bash $S/mirror_topic.sh <课题> <镜像目录> <前缀> T1,FLAIR` 并提交（`S=/data0/congcong/code/GS/Results_20260908/T1T2_AF12_retro/scripts`）。
- P-LORAKS 剩三个通道：hyw FLAIR 与 jyz T1 在跑（00:38 起，各 7–8 h），jyz FLAIR 随后。`scratchpad/ploraks_finish2.sh` 落一个导入一个，课题齐了自动刷新表、逐层与图（v3 最优层）。刷新后只剩镜像和提交：
  ```
  S=/data0/congcong/code/GS/Results_20260908/T1T2_AF12_retro/scripts
  bash $S/mirror_topic.sh T1FLAIR_AF12_retro t1flair_af12_retro t1flair_af12 T1,FLAIR
  bash $S/mirror_topic.sh T1FLAIR_AF12_v3_jiangyizhou t1flair_af12_v3_jiangyizhou t1flair_v3_af12 T1,FLAIR
  git commit -m "archive: P-LORAKS rows for the two T1+FLAIR topics"
  ```
  scratchpad = `/tmp/claude-1002/-home-congcongliu--claude/02e916fd-fbec-4d76-a42c-a38bb99a9681/scratchpad`，日志 `ploraks_finish2.log`。
- 其余四个 CFL 受试者（vol2/4/5/6/7）一条命令可克隆：`T1FLAIR_AF12_retro/scripts/setup_subject.sh`，然后同配方跑。

## 4. 已知的坑与别重做
- 经典臂必须查收敛：CS-SENSE 的 35 轮 ADMM 未收敛（200 轮高 0.35–0.6 dB）；L1-ESPIRiT 100 步好于 300 步（早停正则），报告时写明步数。
- 本机 BART 无 GPU，CPU 32 线程约 4 分钟一次；LORAKS 工具箱在 /data0/congcong/code/LORAKS_v2，P-LORAKS 在 R=3 下单层超 1 小时不可用；SENSE-LORAKS 输出是 1/N 逆 FFT 尺度，λ≤1e-2 不起作用、λ=10 出 NaN，已放弃。
- `pkill -f 模式` 会匹配到工具 shell 自己（exit 144）；等待脚本用"启动时找一次 PID，再 kill -0 轮询"。MATLAB 长跑与 GS 训练都要 `setsid nohup` 脱离会话（会话看门狗按 MemFree 杀后台任务）。
- 工具 shell 是 zsh：`$VAR` 不分词，多参数启动写成 bash 脚本；未加引号的 `--include=*.py` 会被当成通配符报错。`$VAR:r…` 会被 zsh 当成参数修饰符（去扩展名），拼路径写 `${VAR}:…`。工具 shell 的 cwd 会被重置到 `~/.claude`，写文件一律用绝对路径。
- CPU 被 MATLAB 占满时跑 pytest 要加 `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`。
- GPU 6、7 上有别人的进程（09-10 18:25 起），同卡训练慢 2–3 倍：v3 本身在 GPU 7 跑了 189 min，同尺寸的其他运行 51–69 min。读时间前先看 `nvidia-smi --query-compute-apps`。
- GPU 6、7 上常驻别人的 696 MiB 进程：排队脚本判断"卡上没有进程"在这两张卡上永远不成立（queue22 因此卡住），要改成只看 4 GB 以上的进程。
- 旧 `ploraks_finish.sh` 按 v2 的最优层出图，会覆盖 v3 的图，已停掉，换成 `ploraks_finish2.sh`。
- 仓库的 `*/logs/` 被 .gitignore，镜像 summary 与逐层结果要 `git add -f`（`mirror_topic.sh` 已处理）。
- 别再试：intensity 初始化、密化、position_lr、T1 高频权重 1.0/3.0、kappa 300 步刷新、分位 0.75、1.5 倍尺度、tv_eps 0.06；小波替代 TV（λ 0.03 / 0.1 都差）、小波作用在幅值上（差 0.13 dB）、360 万点 + 小波 0.03（T1 −0.16）、360 万点 + 小波 0.05（4000 步 T1 −0.11、T2 +0.12，未采纳）。lambda_geo 0.07 在无小波时（第 1 轮）证否，第 16 轮在有小波的前提下重测。
- 容量：无小波时 T1 在 120 万点封顶（过拟合采样 k）；加小波 0.05 后 240 万点 T1 也涨。密度规则（点/支撑体素）在 jyz 证否。

## 5. 关键决定的"为什么"
- 竞技场选 hyw T1+T2：README 定的 canonical 课题，且 T2 有真值；T1+FLAIR 留作迁移检验，jiangyizhou 不参与任何选择。
- 读数用头内 PSNR 的最小余量：整图 SSIM 被 `ecalib -c 0.0` 留下的空气噪声支配（见 T1T2 README）。
- v3 用共用小波强度而不用分对比度（T1 0.06 / T2 0.03 在调参对上最小余量更高，+0.261 对 +0.241）：FLAIR 没有自己的调参值，共用强度才能原样迁移。第 16 轮的 T1 通道变体只改三个课题都有的 T1，另一通道仍用共用值，同样可迁移。
- 第 16 轮的两个改动由 SSIM 诊断决定：缺口在中等梯度区的局部方差，所以试"少平滑"，而不是"多正则"。
- 平滑旋钮不晋级、预注册规则先写后跑：越过预注册规则就是重塑结果；留出受试者上的任何调参都会污染"留出"的含义，失败也如实记录。
