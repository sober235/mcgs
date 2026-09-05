# 设计规格 — AF=12 达到 AF=10 的重建质量（无可学习先验）

- **日期：** 2026-09-05
- **状态：** 三段设计已逐段获用户确认（范围/验收、Phase 1、Phase 2–3/执行），待用户审阅本文
- **分支：** `af12-to-af10`（自 `af10-data-audit` @ `d6188a1` 开出）
- **产生方式：** `superpowers:brainstorming`
- **范围：** 本文描述整个研究计划（Phase 0–3）。实现计划分三份：计划 1 = Phase 0 + Phase 1；计划 2 = Phase 2（在 Phase 1 出结果后立）；计划 3 = Phase 3（在 Phase 2 出结果后立）。
- **前置记录：** `docs/superpowers/records/2026-09-05-data-audit-and-af10-rerun.md`（数据信息审计与 AF10 复跑）、`.claude/memory.md` §0/§8b/§12/§14/§17–19（已证否清单）

---

## 1. 目标

在 HYW 同一受试者的 T1+FLAIR 数据上，**ALIGNED AF=12 掩膜（净 11.83）下的 MC-GS 重建，达到 AF=10（净 9.87）旧配方的质量**。

要补的差距（`tvs4k_e040` → `af10_joint_edgetv_e040_rerun20260904`）：

| 口径 | T1 | FLAIR |
|---|---|---|
| PSNR whole@max | +0.57 dB | +0.65 dB |
| PSNR head-only@max | +0.68 dB | +0.78 dB |

### 1.1 用户约束（2026-09-05 确认）

1. **方法里不能有任何可学习先验。** 不训练网络、不用预训练/生成模型、不用外部数据拟合任何参数。
2. **外部公开数据只能当新的测试集**（Calgary-Campinas 12 通道 3D T1，正在下载到 `/data2/congcong/data/3D/Calgary-Campinas/`），不得用于训练或调参。
3. 方法身份仍是 MC-GS：只往目标函数或前向模型里加项。

### 1.2 不变量（整个项目里不动）

- 数据：HYW `161015` 的 T1（plain）+ FLAIR（contrast）`_trainer.mat`；ESPIRiT 1-map CSM；GT 定义（`Σ_c conj(S_c)·ifft3c(y_full_c)`，P99.9 归一）。
- 掩膜：`mask3d_vdpois_af12_transverse_ALIGNED.mat`。不重设计采样。
- 工作点：edge-TV + Charbonnier，`λ_geo 0.10 / tv_eps 0.04 / κ_q 0.85 / w_floor 0.1 / kspace_high 2.2 / warmup 500 / max_radius 20 / 4000 步 / 576902 高斯热启动`。这些数不扫（`.claude/memory.md` §18 已封）。
- 所有新增项配置门控；默认关闭时 `tests/smoke_test.py` 的参考 JSON 逐位不变。

### 1.3 已证否、本设计不再碰的路线

edge-TV 工作点调参（AF12 两轮、AF10 四轮全拒）；密化/容量；TV 形状家族（dtv/vtv/edge/Charbonnier 之外）；2-map CSM 与重标定；掩膜密度重设计与对比度自适应采样；共享响应解码器与瓶颈；P1（高斯模型上的 Θ/Λ 自监督外推）与 P3（局部两列 Casorati 低秩）；冻结 MAISI VAE；T1↔FLAIR 空间配准；支撑约束；相位/共轭对称约束（T1 相位粗糙）。

---

## 2. 验收门（预注册，按对比度各自判）

T1 不能替 FLAIR 补，反之亦然。

| 指标 | 定义（§2.1） | 通过条件 |
|---|---|---|
| head-only PSNR@max | 头掩膜内幅值 MSE，data_range = max\|GT\| | ≥ AF10 参考值 − tol_psnr |
| head-only SSIM | SSIM 图在头掩膜内取均值 | ≥ AF10 参考值 − tol_ssim |
| head-only HFEN | LoG 误差能量比，只在头掩膜内 | ≤ AF10 参考值 + tol_hfen |
| e_unobs | 未观测 k 空间复数多线圈相对误差（oracle） | ≤ AF12 基线 `tvs4k_e040` 的值（防 reshape） |
| 颗粒 N | §2.1 白质高通 rms 比 GT | ≤ 1.00 |
| 视觉门 | 蝴蝶层 z=160 + 三视图，误差 jet 固定 [0, 0.1] | 用户判：无新伪影、无片状/斑块、无相干条带 |

**参考值**：AF10 复跑 `af10_joint_edgetv_e040_rerun20260904` 的 head-only 三指标，由 Phase 0 的新评分器从 `results_arc/af10_joint_edgetv_e040_rerun20260904_volumes.npz` 算出（PSNR 预计 T1 ≈ 34.605 / FLAIR ≈ 34.401 dB）。AF12 基线 `tvs4k_e040` 的 e_unobs 由 `scripts/autoresearch_kspace_score.py` 回放 `results_arc/tvs4k_e040/tvs4k_e040/checkpoints/best.pth` 得到。

**容差**：tol = 2 × 两次 AF10 全预算运行（`af10_joint_edgetv_e040` 与其 `rerun20260904`）之间的实测差，并设下限 **0.005 dB / 0.0005 SSIM / 0.001 HFEN**。下限是为了避开 `.claude/memory.md` §18 记录的"训练近似确定 → 容差退化为零"缺陷。数值在 Phase 0 写入 `gates.json` 后不再改。

**whole@max PSNR/SSIM/NMSE/HFEN** 继续报告以便与旧记录衔接，不作判据。

### 2.1 指标精确定义（Phase 0 实现，参考值与候选用同一份代码）

```
rec  = |x_dc|                    # hard-DC 输出的幅值；不做幅度重标定
gt   = |GT|
p999 = quantile(gt, 0.999)
head = gt > 0.05 * p999          # 与 09-04 审计 probe1 相同
gmax = max(gt)

head_psnr = 20*log10(gmax) - 10*log10( mean( (rec[head]-gt[head])^2 ) )
head_ssim = mean( ssim_map(rec, gt, data_range=gmax)[head] )
            # ssim_map = skimage.metrics.structural_similarity(..., full=True) 的 3D 图，
            # 默认 7^3 均匀窗、K1=0.01、K2=0.03
head_hfen = || (LoG(rec) - LoG(gt))[head] ||_2 / || LoG(gt)[head] ||_2
            # LoG = 3D 高斯拉普拉斯，σ=1.5，核 15 taps（若现有 hfen 辅助函数暴露 σ 则复用）
N         = rms( hp(rec)[wm] ) / rms( hp(gt)[wm] ),  hp(v) = v - gaussian_filter(v, 4)
            # wm = .claude/memory.md §14c 的白质掩膜（diag_artefact/volumes.py::wm_mask）
```

---

## 3. Phase 0 — 仪表与参考表（约 1 天）

1. `arc/cycle.py::score_contrast` 增加 `head_psnr / head_ssim / head_hfen / grain_ratio`，写入评分 JSON 与 TSV。
2. 新脚本 `scripts/score_volumes_headonly.py`：只读 `*_volumes.npz`，输出同一组指标，用于给历史运行算参考值（npz 无相位，算不了 e_unobs，e_unobs 只来自训练时评分或 checkpoint 回放）。
3. 参考表 `docs/superpowers/records/af12_to_af10/reference_table.md`：`tvs4k_e040`（AF12）、`af10_joint_edgetv_e040`、`af10_joint_edgetv_e040_rerun20260904`（AF10）三行 × 两对比度，含 whole 与 head-only 全部指标、两次 AF10 之差、由此得到的容差。
4. 门文件 `docs/superpowers/records/af12_to_af10/gates.json`（预注册，含参考值、容差、e_unobs 基线、N 上限）。**任何 Phase 1/2 的实验都在门文件提交之后才允许跑。**
5. 决策账本 `docs/superpowers/records/af12_to_af10/decisions.jsonl`（追加式，每条：run、phase、指标、判决、依据）。

---

## 4. Phase 1 — 物理杠杆：已知零区 Z 与噪声预白化 W（2–3 天）

### 4.1 已知零区 Z

**物理事实（09-04 审计）**：厂商滤波使 FLAIR 在相位编码面 ρ ≥ 1.0 处 k 空间 ≈ 0（0.8–0.9 → 0.9–1.0 能量掉 27 倍）；T1 有椭圆快门，ρ ≈ 1.28–1.35 之外精确为零。这些位置无论采没采，值都已知为 0。

**做法**：把 Z 当作"测到值为 0 的样本"。

```
mask_fit = max(mask, Z)                # 拟合与 hard-DC 用
y_fit    = y ;  y_fit[Z] = 0           # Z 内即使采到也置 0（按定义只是噪声）
DC 损失分母 = |Ω| = mask.sum()          # 不是 |Ω ∪ Z|
hard-DC   = pred·(1 − mask_fit) + y_fit # 输出 k 在 Z 上精确为零
e_unobs / 零填充锚 仍用原始 mask 与原始 y   # 与旧基线可比
```

分母必须保持 \|Ω\|：FLAIR 的 Z 约占 PE 面 20%，若用 \|Ω ∪ Z\|，DC 项会缩到约 0.3 倍，等于把 λ_geo 悄悄放大 3 倍以上。

**Z 的估计只用本次扫描的已采样点**（不看 GT、不看未采样点），`paired_gs/data/known_zero.py::estimate_known_zero`：

1. DC = 已采样点上 Σ_coils Σ_ro \|y\|² 在 PE 面的峰（ACS 峰）。
2. 归一化半径 ρ(i,j) = sqrt(((i−dc0)/(D/2))² + ((j−dc1)/(H/2))²)；读出轴 κ(k) = \|k−dc2\|/(W/2)。
3. 每个已采样 PE 点的能量 e = 对线圈与读出取平均的 \|y\|²；按 ρ 分 0.05 宽的环，环能量 E_r = 环内已采样点 e 的中位数；少于 30 个采样点的环并入内侧相邻环。
4. **悬崖规则**：n0 = 所有环 E_r 的最小值（平台）；Z_PE = 从最外环向内、连续满足 E_r ≤ 1.5·n0 的环集合，且要求该集合内侧相邻 0.15 宽度内存在 ≥ 5 倍的能量下降（悬崖）、集合至少含 2 个环。无悬崖、或最外环本身不满足阈值（n0 来自内侧的偶然低谷）则 Z_PE = ∅，宁缺毋滥。快门（精确零，E_r = 0）自然满足。
5. 读出轴按 0.05 宽的 κ 带做同一规则得 Z_RO。
6. Z = Z_PE 沿读出轴广播 ∪ Z_RO 沿 PE 面广播。

margin 1.5、悬崖 5 倍、环宽 0.05 全部先验固定，**不扫**。估计结果（ρ_c、κ_c、n0、Z 占比）写入运行日志与评分 JSON。允许把估计的 Z 与审计的 oracle 径向剖面**对照作为诊断**，但拟合用的 Z 只能来自采样点。

0.9–1.0 过渡带（非零但衰减）v1 不建模；只有当 FLAIR 差一个容差量级过门时，才允许作为唯一后续项加入"按环能量比加权的软零惩罚"，并且要单独立记录。

### 4.2 噪声预白化 W

- Σ（12×12 复 Hermitian）= Z ∩ Ω 内已采样点的样本协方差（FLAIR 约十万样本/线圈）。若 Z ∩ Ω 样本全为精确零（T1 快门）或每线圈少于 2000 个，退回用悬崖内侧 0.1 宽的最外采样环并在日志标记 `sigma_source=outer_ring`；若仍不足则 W = I 并记录。
- W = Σ^(−1/2) · sqrt(mean(diag Σ))，使 Σ = σ²I 时 W = I、损失逐位不变。
- **只改拟合**：`csm_fit = W·csm`（线圈维混合）、`y_fit ← W·y_fit`，进 `_kspace_loss`。hard-DC、评分、零填充、导出仍用原始线圈基与原始 csm，GT 与指标不动。

### 4.3 改动文件

| 文件 | 改动 |
|---|---|
| `paired_gs/data/known_zero.py`（新） | `estimate_known_zero(y, mask, ro_axis) -> Z, info`；`estimate_noise_cov(y, mask, Z) -> Σ, info`；`whitening_matrix(Σ) -> W`。纯函数，numpy/torch 均可 |
| `paired_gs/trainers/joint_trainer.py` | `_setup_data`：配置门 `data.plain.known_zero` / `data.contrast.known_zero`、`loss.prewhiten`，产出 `mask_fit_*`、`kspace_fit_*`、`csm_fit_*`（默认即原张量）；`_kspace_loss` 分母改用 `mask.sum()` 的原始值；`export_results` 的 hard-DC 用 `mask_fit/kspace_fit` |
| `arc/cycle.py` | `--known_zero`、`--prewhiten`；`score_contrast` 的 hard-DC 用 fit 张量，e_unobs/零填充用原始张量；JSON 记录 Z/Σ 诊断 |
| `configs/af12_kz.yaml`、`configs/af12_kz_pw.yaml`（新） | 基于 `winner_v2_edgeTV.yaml`，只开对应开关 |

### 4.4 测试（先写测试）

- `tests/test_known_zero.py`：合成 12 线圈 k 空间 + 椭圆带限（已知 ρ_c）+ 白噪声，估计的 ρ_c 落在真值相邻一环内；无带限合成数据返回 Z = ∅；快门（精确零）被识别；Z 开启时 hard-DC 输出在 Z 上恰为 0；DC 分母等于原始 mask.sum()；Z 关闭时张量逐位等于原张量。
- `tests/test_prewhiten.py`：已知 Σ 的合成相关噪声 → W Σ Wᴴ 为单位阵倍数（相对误差 < 1e-4）；等方差对角 Σ → W = I，损失逐位相同；`score_contrast` 的 e_unobs 与零填充值不随 `prewhiten` 变化。
- 回归：两开关全关，`tests/smoke_test.py` 参考 JSON 不变。

### 4.5 运行与门

| run | 设置 | GPU | 预算 |
|---|---|---|---|
| `af12_kz` | 两对比度 known_zero | 0 | 4000 步 |
| `af12_kz_pw` | + prewhiten | 1 | 4000 步 |

门：FLAIR 的 head-only 三指标对 AF10 参考值 + e_unobs + N + 视觉门。T1 只记录不判。预期 FLAIR +0.3~0.6 dB，T1 ≤ +0.4 dB。

**Phase 2 基座 `base1` 的选取规则**（先验固定）：FLAIR 过门项数多者；平手取 FLAIR e_unobs 低者；再平手取 `af12_kz_pw`。若两者都让 T1 或 FLAIR 的任一指标比 `tvs4k_e040` 倒退超过容差，该开关不进基座并记录原因。

---

## 5. Phase 2 — 非局部组低秩先验（1–2 周）

### 5.1 机制

解剖结构在体内大量重复。把每个参考块的 M 个最相似 p³ 块按列拼成矩阵 A_g，真实解剖使它近似低秩，欠采残差非相干、满秩；惩罚 Σ_g ‖A_g‖_* 压残差、留结构。它约束的是块内容对其非局部副本的一致性，不是梯度，与已封的 TV 家族正交。无可学习部件。

### 5.2 多对比度耦合

T1/FLAIR 共享几何，所以用两对比度拼接的幅值（各自按当前重建的 P99.9 归一）联合匹配，同一组索引同时用于两对比度，A_g = [P_T1; P_FLAIR] ∈ R^(2p³ × M)。只用幅值：T1 相位粗糙（24×24 低分辨相位虚部残差 18.9%），复数块匹配不成立。

### 5.3 实现（`paired_gs/losses/nonlocal_lowrank.py`）

**块匹配**（无梯度，`nl_start`=500 步起每 `nl_every`=100 步一次）：

- 头掩膜 = 两对比度归一幅值的逐体素最大值 > 0.05（来自当前重建，不用 GT）。
- 在头掩膜内随机取 N_ref = 200 000 个参考块角点（窗口与块须落在体内）；搜索窗 s³ = 11³，步长 1；距离 d = ‖P_T1−Q_T1‖²/(p³σ_T1²) + ‖P_FL−Q_FL‖²/(p³σ_FL²)，σ_m = 头内 (x − gaussian_filter(x,1)) 的 MAD 标准差。
- 取 top-M（含自身）；d > τ = 4.0 的邻居截掉；有效成员 < 8 的组丢弃。分块 gather 计算（每块 2000 个参考，显存约 2 GB）。
- 输出 `idx (N_g, M, 3)` int32、`valid (N_g, M)` bool。随机数生成器由训练种子 42 + 迭代号决定。

**每步损失**（有梯度）：从已存组中抽 B = 8192 组，可微 gather 当前渲染幅值 → (B, 2p³, M)，每组减去列均值，`torch.linalg.svdvals`，R_NL = 组内奇异值之和的均值（仅 valid 成员）。

**λ_nl 按比值定**：在第 500 步的首次匹配时，用同一份 detached 体计算 R_NL 与 R_geo，令 λ_nl = r · λ_geo·R_geo / R_NL，之后固定；r 与得到的 λ_nl 写入日志。

### 5.4 配置键与 CLI

`loss.lambda_nl_ratio`（0 = 关）、`nl_patch` 6、`nl_group` 16、`nl_search` 11、`nl_refs` 200000、`nl_batch` 8192、`nl_every` 100、`nl_start` 500、`nl_tau` 4.0、`nl_min_members` 8、`nl_match` joint|separate。CLI：`--lambda_nl_ratio --nl_patch --nl_group --nl_match`。

### 5.5 测试（先写测试）

- 合成"重复纹理瓦片 + 噪声"体：块匹配找到的 top-M 中 ≥ 90% 是真副本；同种子两次结果逐位相同。
- 完全相同的块组减均值后核范数为 0，加噪后 > 0；autograd 只在被分组的体素上产生非零梯度。
- 全尺寸预算测试（需 GPU，默认跳过标记）：N_ref 200k / s 11 / p 6 的匹配与每步损失的耗时、峰值显存。目标 < 60 s / < 20 GB / < 50 ms；这是预算不是判据，超出就记录并下调 `nl_batch` 或 `nl_refs`，不改门。
- `lambda_nl_ratio = 0` 时 `tests/smoke_test.py` 参考 JSON 不变。

### 5.6 预注册搜索（≤ 12 格 1500 步 + ≤ 3 次 4000 步）

基线：`base1_1500`（Phase 1 胜者的 1500 步版本，1 格，不计入 12）。筛选时 §2 全表的比较对象都是 `base1_1500`（同预算、同物理开关，量的只是非局部项自己的效果）；`tvs_e040`（1500 步，已有）只作衔接报告。

| 批次 | 格 | 变量 |
|---|---|---|
| 主网格 | 6 | r ∈ {0.25, 0.5, 1.0} × p ∈ {4, 6}，M 16，joint |
| 最佳格上 | 2 | M ∈ {8, 32} |
| 最佳格上 | 1 | `nl_match = separate`（逐对比度单独匹配） |
| 预留 | 3 | 唯一允许的后续项：核范数出现模糊时改加权核范数 w_i = 1/(σ_i+ε) |

筛选判据 = §2 全表相对 1500 步基线；晋级到 4000 步的候选 ≤ 3 个；4000 步的判据 = §2 全表相对 AF10 参考值，**T1 为主判**。

### 5.7 止损

预算用完仍过不了 T1 的门：报最好结果与差距，停，不扩搜索、不重开工作点。

---

## 6. Phase 3 — Calgary-Campinas 泛化测试（3–4 天，Phase 2 结束后单独立计划）

- 数据：CC-359 多通道 12 通道测试集，预注册取文件名排序后前 3 个受试者。沿层轴 FFT 回全 3D k 空间；读出轴为全采样轴，PE 面为 (ky, kz)；BART `ecalib -t 0.001 -c 0.0 -r 24 -m 1` 估 CSM；GT = conj-SENSE 合成；`vdPoisMex` 生成该矩阵尺寸的 AF12/AF10 vd-Poisson 掩膜（ACS 落在 DC）。
- 单对比度：edge-TV/Charbonnier、Z/W、非局部项需接到单对比度训练路径（`paired_gs/trainers/trainer.py` 或以 JointTrainer 单通道模式），在该阶段的计划里决定。
- 每个受试者三项：AF12 新配方、AF10 旧配方、AF12 旧配方。判定 = AF12 新配方的 head-only 三指标 ≥ AF10 旧配方（同容差）。同时报 AF10 新配方。
- CC 若无厂商带限，Z = ∅，由非局部先验单独承担，如实报。
- 只用于评估，任何参数不得因 CC 结果回调。

---

## 7. 执行

- 分支 `af12-to-af10`；TDD；每格实验的评分 JSON、`decisions.jsonl` 随做随提交；图沿用 `scripts/fig_3views_jet.py` 与 `scripts/fig_butterfly_z160.py`，放 `docs/figures/af12_to_af10/`，同时拷到 `/home/congcongliu/figs/af12_to_af10/` 并给 `http://localhost:8765/af12_to_af10/...` URL 行。
- 环境：`/home/congcongliu/anaconda3/envs/GSTMI/bin/python`，`PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID`。8 张 80 GB 卡全部空闲，1500 步网格一波并行约 30 分钟，4000 步约 1 小时。
- 时间：Phase 0 一天；Phase 1 两三天；Phase 2 一到两周；Phase 3 三四天。
- 提交身份沿用仓库本地配置，commit 不带任何 AI 痕迹。

## 8. 风险

| 风险 | 应对 |
|---|---|
| Z 估计在稀疏的外环找不到悬崖 | 外环每 0.05 宽约 300 个采样 PE 点 × 256 读出，足够；找不到即 Z = ∅，如实记录，与审计 oracle 剖面对照作诊断 |
| 非局部先验把孤立结构拉向"相似"组（斑块/抹平） | τ 截断、最少成员数、N ≤ 1、HFEN、e_unobs、视觉门 |
| 联合匹配压制 T1 私有细节 | `nl_match = separate` 消融格 |
| 块匹配显存/耗时 | 分块 gather，全尺寸预算测试 |
| CC 数据格式、单对比度路径 | 独立计划，不影响 HYW 验收 |

## 9. 非目标

不引入任何可学习部件；不重设计掩膜；不做配准；不重调冻结工作点；不改 GT/CSM；v1 不建模过渡带；不超出预注册预算。
