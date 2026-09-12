# STATUS — 2026-09-12（分支 `cardiac-freerunning`，tag `handoff/2026-09-12-cardiac`）

心脏自由运行多回波 3D 径向数据（Mackowiak 2023，V07/V03/V04/V08）的第一阶段：读取、BART 参考重建、评估口径。已完成。
第二阶段（MC-GS 与全部对手臂接 NUFFT 前向）设计已在对话中分节提出，spec 未写、代码未动。
先读 `CLAUDE.md` 顶部一节，再读本文件、spec、plan、课题 README（`/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/README.md`）。

## 1. 已完成且已验证

命令都在课题脚本目录下跑：`cd /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts`。原始输出摘自 `logs/`。

- 读取：四例 twix 读成 `raw.h5`；V07 抽 1,000 条与作者 MATLAB 读取器逐位相同。
  ```
  ./py xcheck_reader_compare.py V07
  # V07: 1000 lines, ours (1000, 24, 224) vs theirs (1000, 24, 224), bitwise equal = True, max |diff| = 0
  ```
- 读出首尾：四例只有采样 0–2 异常（V07 为正常水平的 18.28、4.39、1.52 倍），保留 3–220。`./py s02_edge_check.py` → 四例 PASS。
- 轨迹：与作者 `computePhyllotaxis` 最大差 2.22e-16。`./py xcheck_traj_compare.py` → PASS。
- 生理：四例 PhysioInfo 时间戳与 MDH 逐行相等（`timestamp_max_abs_diff_ms` 0.0）。
- 方位：`orientation.json` 按例存放。V07/V03/V08 横断 slab（axis2 = 上→下，头在低下标）；**V04 冠状 slab**（axis0 = 下→上，头在高下标；twix `sNormal.dCor = 1`）。
- 极性（`logs/polarity_<Vxx>.json`，高三分组相对低三分组的位移，体素，+ = 脚侧）：V07 −3.80/−2.55、V04 +2.17/+1.97（自身坐标系）、V08 −2.67/−1.34，都与直方图一致；V03 +0.67/+0.62 低于 1 体素门槛，按直方图（停留不对称 2.54）取高端，标"门控弱"。四例 REF 均为高端三分组 6,813 条。
- 线圈图：112³ 体内平方和门 1.000（四例）；224³ 插值前 0.983/0.982/0.982/0.978（门 0.90）。
- REF（224³，λ = 1e-3 × λmax，100 步 CG）：λmax V07 2,185 / V03 2,829 / V04 2,488 / V08 2,266；60 对 100 步体内差 0.25%/0.10%/0.14%/0.21%；`-w 2X` 比值 1.0000。
  ```
  cat logs/checks_V07.json
  # "iter_rel_diff_60_vs_100": 0.00264, "iter_ok": true, "scale_ratio_w2x_over_w1x": 1.0, "scale_ok": true
  ```
- 口径自检（V07，`logs/selfcheck_V07.tsv`）：留出 NRMSE 回波 0 CG 0.1408/0.1467/0.1526、L1 0.1325/0.1332/0.1344（AF 8/12/16），回波 4 CG 0.3669/0.3817/0.3990、L1 0.3387/0.3389/0.3408；REF 样本内底 0.1319/0.3406。必需判定 12/12 单调，PASS（修订 6 的门）。L1 λ = 3e-3（AF 12 回波 0 体内 PSNR 42.72 dB）。
  ```
  ./py s10_selfcheck.py --rescore   # 只重评分不重建
  # SELF-CHECK PASS
  ```
- 冻结参数：`params_frozen.json`（recon_grid 224，eval_crop 56:168，ref_iters 100，ref_lam_rel 1e-3，l1_lambda 3e-3）。
- 测试：
  ```
  cd /data0/congcong/code/GS/mcgs_cardiac_wt && /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/py -m pytest tests/test_cardiac_fr_*.py -q
  # 34 passed
  ```
- 图：全部在 `/home/congcongliu/figs/cardiac_fr/`（40 张），课题 README 的 Figures 节逐张列了 URL 与路径。

## 2. 待用户拍板

- 是否把分支 `cardiac-freerunning` 与 tag 推到 GitHub（sober235/mcgs）。
- 第二阶段设计 §1–§8（对话 2026-09-12 下午提出）逐节确认后写 spec。
- 第二阶段用哪块 GPU：grill 时定的 GPU 5 已被 AF16 第三阶段循环占用（其它会话消息），开跑时按"一卡一训练"重选。

## 3. 下一步

- 第二阶段 spec（brainstorming 分节 → spec 文件 → plan）：公共 torchkbnufft 前向（`paired_gs/forward/radial.py`，与 BART nufft 一致性测试）、JointTrainer 的 `forward: radial` 分支（数据加载、损失、初始化、致密化、导出）、对手臂逐个接入（BART `-R T` 表达 CS-SENSE 与 Joint SENSE；DIP/INR 换前向；ZS-SSDU 3D + Toeplitz CG 仅 V07 缩减版；P-LORAKS 排除）、统一评分脚本（`sens2x` + BART nufft 留出 NRMSE）。
- 第二阶段分支 `cardiac-phase2` 从本 tag 开出，先合并 `autoresearch-t1t2`@39cae91（v3.2 小波配方，试合并零冲突）。
- 新课题目录 `Results_20260908/Cardiac_FreeRunning_MCGS/`，数据软链接回本课题。

## 4. 已知的坑与别重做

- `bart pics -l1` 在径向上必须加 `-e`；一律加 `-S`；BART 只在 `-d5` 下打印 scaling。
- 径向 AᴴA 的 λmax 是几千：`-l2` 的 λ 乘 λmax（λ_rel 1e-3），`-l1` 的 λ 保持绝对值。两个方向都失败过一次。
- 220 mm 视野折叠 62% 信号：REF/子集必须在 224³ 上重建，中心裁剪评估。R0 与极性三分组是 112³ 折叠图，只看方位、量位移。
- ecalib 在 224³ 上要几小时：112³/448 mm 算完插值。
- 方位逐例看 twix slab 法向，V04 是冠状；别凭折叠的 R0 目视。
- 极性判据 (b) 只能用局部块、门槛 1 体素；全视野剖面和中心大块都错过。
- FISTA 100 步没收敛，L1 臂是早停解；对噪声参考的 PSNR/HFEN 分不开去噪方法的 AF（修订 6）。
- 成像 measurement 的第 0 条 ADC 是噪声；每个 .dat 两个 measurement 取最后一个；读出采样 0–2 丢弃。
- BART NUFFT 按酉 FFT 归一：224³ 上 `-S` 图比 112³ 亮 2.83 倍，指标无量纲。`bart traj -x 32` 坐标到 ±15.5，匹配 16 网格要 `-o 2`。
- 机器共享：nice 10、32 线程、两个并行 BART 已获批。长任务 `setsid nohup` + PID 文件，等待用 `kill -0`；会话内存看门狗会杀后台等待循环（页缓存占满 MemFree），脱离的任务不受影响。zsh 不分词，`=` 开头的词报错。
- 数据不用再找（`docs/data_surveys_2026-09-08/`）。

## 5. 关键决定的"为什么"

- 走 BART 而不是旧代码换参数：MC-GS 与自写 CG/CS 的前向是笛卡尔 `fft3c × 掩膜`，需要全采样真值；本数据是 22,000 条 3D 径向辐条。网格化到笛卡尔再喂旧代码在 V07 上量过：k 空间误差 33–43%，远大于各臂之间 0.005–0.02 的留出差别，只能当冒烟测试。
- 参考用 CG-SENSE：线性、无稀疏先验，迭代数和缩放能核对，不把某种正则写进"真值"。
- 排名以留出 k 空间 NRMSE 为主：REF 有噪声和条纹，图像指标偏向同类方法并对去噪方法饱和（修订 6 的证据）。
- REF 用相对 λ、L1 用绝对 λ；REF/子集在 224³ 上重建：证据在课题 README 的 Seven corrections。
- 极性要两条判据，图像判据有分辨率下限：V03 的两端只差不到 2 mm。
