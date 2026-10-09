# MC-GS 共享几何结构的 2×2 因子消融实验方案

> 日期：2026-10-09  
> 仓库：sober235/mcgs  
> 设计分支：exp/shared-geometry-2x2-ablation-20261009  
> 基础分支：widen-mcgs-af12-16-20-24  
> 状态：**实验设计；尚未新增模型开关、训练或产生本方案的四组结果。**

## 1. 科学问题和可证伪假设

核心问题：**在两套单对比度 MRI 重建使用相同前向物理模型、显式正则、初始化策略、训练迭代和每对比度 Gaussian 表达能力时，仅共享 T1/FLAIR 的 Gaussian 几何参数，是否带来可重复的重建增益？**

研究对象是原始 MC-GS（共享几何 + 双通道独立复数密度），**不是** Paired-ARC-GS 的共享 latent decoder / change-audit 架构。

待检验的假设：

- **H1（结构独立贡献）**：关闭全部显式正则后，共享几何仍优于两套完全独立的 GS（C > D）。
- **H2（完整单通道正则条件下）**：使用相同单通道正则后，共享几何仍优于独立 GS（A > B）。
- **H3（互作）**：正则改变共享几何的收益，即 (A − B) 与 (C − D) 存在交互作用。
- **H4（完整方法增量；扩展对照）**：跨对比度正则可在共享几何基础上产生额外收益（E > A）。

这些是假设而不是预设结果。任一对比度上增益消失或变负，都必须如实报告。

## 2. 模型定义

对于对比度 m ∈ {T1, FLAIR}，以多线圈 k-space 数据一致性为所有实验共同的优化目标：

\[
L_{\rm DC}=\sum_m\|M_mFC_m x_m-y_m\|^2.
\]

统一 Gaussian 渲染表达：

\[
x_m(\mathbf r)=\sum_{n=1}^{N_m}a_{m,n}
G(\mathbf r;\boldsymbol\mu_{m,n},\boldsymbol\Sigma_{m,n}),
\]

其中复数密度 \(a_{m,n}\) 对每个对比度始终独立。

- **共享几何（S=1）**：T1 与 FLAIR 使用同一套 Gaussian 位置、尺度与旋转，仅密度独立；对应代码 `paired_gs/gaussian/joint_gaussian_model.py`。
- **独立几何（S=0）**：每个对比度分别拥有位置、尺度、旋转、复数密度；对应两套独立 `GaussianModel3D`。两个序列的 k-space 损失不能通过共享几何反向传播相互影响。
- **有正则（R=1）**：每个序列使用相同种类、各自预设权重的 self-guided edge-preserving TV、wavelet-L1 和 phase-TV；不允许使用另一个序列作为正则的 guide。
- **无正则（R=0）**：显式 TV、wavelet、phase-TV、跨对比度先验等权重全部为零；**保留相同的数据一致性与必要的物理/数值约束**。`kspace_freq_adaptive` 属于 DC 权重配置，不属于本次消融的显式图像先验；四组必须一致。

## 3. 核心四组：严格 2×2 对照

| 实验 ID | 模型 | 共享几何 S | 单通道正则 R | 跨对比度正则 | 输入与输出 |
|---|---|---:|---:|---:|---|
| **A / S1R1** | MC-GS + Reg | 1 | 1 | 关 | T1+FLAIR 联合训练，输出双序列 |
| **B / S0R1** | GS-single + Reg | 0 | 1 | 关 | T1、FLAIR 各自独立训练，输出双序列 |
| **C / S1R0** | MC-GS − Reg | 1 | 0 | 关 | T1+FLAIR 联合训练，输出双序列 |
| **D / S0R0** | GS-single − Reg | 0 | 0 | 关 | T1、FLAIR 各自独立训练，输出双序列 |

主表**不能**在 A 中保留 `lambda_edge > 0` 而在 B 中置零，否则 A−B 混入 common-edge 正则效应，无法证明共享几何本身的贡献。

### 扩展组 E：完整 MC-GS

| 实验 ID | 模型 | 共享几何 | 单通道正则 | 跨对比度正则 | 目的 |
|---|---|---:|---:|---:|---|
| **E / S1R1X1** | Full MC-GS | 1 | 1 | 开 | 测量完整方法相对 A 的跨模态正则增量 |

E 为附加模型行，不纳入四组因子主效应估计；优先使用既有 `widen-mcgs-af12-16-20-24` 的完整正则配方，并记录 E 与 A 的**单通道正则配方是否完全一致**。若不一致，应重新运行匹配的 E，不能把配方差异归因于跨对比度正则。

## 4. 正则开关和默认配方

首轮使用 **AF16 phase-3 冻结配方**，避免边运行边追求有利结论。正则相关权重设置如下（最终以解析后的 trainer 配置为准）：

| 配置键 | A、B | C、D | E（扩展） |
|---|---:|---:|---|
| `loss.tv_weight` / CLI `--lambda_geo` | 0.10 | 0 | 同 A |
| `loss.edge_tv` | true | false | 同 A |
| `loss.tv_eps` | 0.04 | 0.04（不生效） | 同 A |
| `loss.edge_tv_kappa_q` | 0.85 | 不生效 | 同 A |
| `loss.edge_tv_wfloor` | 0.10 | 不生效 | 同 A |
| `loss.lambda_wav`（FLAIR） | 0.10 | 0 | 同 A |
| `loss.lambda_wav_plain`（T1） | 0.12 | 0 | 同 A |
| `loss.lambda_ptv` | 0.03 | 0 | 同 A |
| `loss.lambda_edge` | 0 | 0 | 0.001（可作为起点） |
| `loss.lambda_dtv` | 0 | 0 | 根据冻结完整配方 |
| `loss.lambda_vtv` | 0 | 0 | 根据冻结完整配方 |
| `loss.lambda_dtv_proj` | 0 | 0 | 根据冻结完整配方 |
| `loss.edge_tv_guide_alpha` | 0 | 0 | 根据冻结完整配方 |
| `loss.ptv_guide_alpha` | 0 | 0 | 根据冻结完整配方 |
| `loss.lambda_nl` | 0 | 0 | 0，除非完整模型明确使用 |
| `gaussian.num_private_points` | 0 | 0 | 0（主归因） |

特别注意：

1. **FLAIR 单模型必须使用 FLAIR 自己的 wavelet λ=0.10。** 现有双槽 `singleFLAIR.yaml` 将 FLAIR 同时作为 plain / contrast，评分 plain 通道时可能误用 T1 的 `lambda_wav_plain=0.12`；这是历史 Stage 4 记录过的混杂因素。
2. `lambda_ptv > 0` 要求 `edge_tv=true`；无正则组同时置 `lambda_ptv=0`。
3. 确保 `guide_path_plain`、`guide_path_contrast` 均未设置；不使用真实图像、未采样 k-space 或对方重建作为 GS-single 的隐性辅助输入。
4. 所有开关必须在模型实际训练日志中输出；检查 `R_geo`、`R_wav`、`R_ptv`、`R_edge` 等**加权损失**而不仅是配置文件，C/D 应全为 0。

## 5. 必须控制的公平性问题

### 5.1 Gaussian 数量与容量：两种对照同时给出

**主比较——每对比度容量匹配：**

- A/C：共 3.6M 个 Gaussian 几何实体，T1 和 FLAIR 各有自己的复数密度。
- B/D：每个对比度独立使用 3.6M 个 Gaussian，共 7.2M 个几何实体。
- 理由：对每个输出序列，独立方法的表达能力不低于共享方法；若共享仍胜出，不能简单归咎于 GS-single 容量不足。
- 同时报告总参数量、训练时间和峰值显存，承认 B/D 总预算更高。

**容量补充——两方法总 Gaussian 数匹配：**

- A/C 共享 3.6M；B/D 改为每对比度 1.8M。
- 此比较控制总几何实体数量，但独立组每个输出序列容量变少；作为资源公平实验，**不能替代**上面的每对比度容量匹配主比较。

### 5.2 初始化：主实验不能把额外测量信息提前给 GS-single

现有 `joint_trainer.py` 在 `gaussian.init_mode=head` 下基于 `union_magnitude(zf_T1,zf_FLAIR)` 初始化联合 Gaussian；现有双槽 single 则基于同序列 ZF 初始化。**两种几何初始分布并不相同**，若直接比较，收益混合了联合初始化效应。

主归因实验应实现 `init_mode=shared_fixed_prior`：使用预先规定的、不依赖当前受试者 T1/FLAIR 额外测量或 GT 的固定 FOV / 头部空间范围，以相同 seed 生成初始位置和尺度，A/B/C/D 完全一致；各模型的初始复数密度仅来自自己对应序列的可用 ZF（T1 不得读取 FLAIR 密度初始化，反之亦然）。若固定先验的空间范围在开发集上制定，必须冻结并在独立受试者上评估。

另做实用场景对照：联合模型使用实际的双序列 ZF 初始化，独立模型只能使用自己的 ZF 初始化；标注为 **end-to-end gain**，不与“仅共享几何”主归因混用。

### 5.3 优化与物理模型

- 完全一致的 k-space、CSM、有效采样支持、known-zero / prewhitening 处理、FFT 规范、归一化与最终 hard-DC 后处理。
- 4000 次 optimizer step；基础 `max_radius=20`、高频 DC 权重 `kspace_high_weight=2.2`，相同优化器、学习率、调度和梯度裁剪。
- 主实验关闭 Gaussian private branch、densification / cloning / pruning，避免动态容量差异。
- GS-single **必须真正独立**：推荐新增 `shared_geometry=false` 路径，分别实例化两套 `GaussianModel3D`；可由两次独立单对比度训练交叉校验。现有复制 T1→T1 或 FLAIR→FLAIR 的双槽 joint-trainer 是历史 proxy，**不能不经等价性检查直接当作严格独立结构主对照**。
- 检查梯度隔离：B/D 的 T1 loss 不得给 FLAIR 几何产生梯度，FLAIR loss 反之；A/C 两个 DC loss 应共同作用于同一几何张量。对完全独立目标，合并训练和两次分开训练应能在数值容差内复现（控制随机数、wavelet cycle spin、optimizer 顺序）。
- 所有模型在**同一训练预算口径**下评估；单对比度 λ 允许各自选择，但有同等搜索预算、搜索范围和独立验证集；不得用测试受试者/GT 选最好超参数。

## 6. 数据集、实验矩阵和执行顺序

**Pilot（工程检查，不作论文机制证据）**

- 数据：现有 jiangyizhou 配对 T1+FLAIR，vd-Poisson retrospective AF16，seed=42。
- 每组都产出 raw Gaussian volume、hard-DC volume、head-only metrics 和未采样 k-space 误差。
- 对照现有配置：`configs/t1flair_v3_af16_jyz.yaml` 和 `configs/t1flair_v3_af16_jyz_single{T1,FLAIR}.yaml`。
- 输出新 run ID：`sg2x2_pilot_af16_s42_{A,B,C,D}`，其中 B/D 分别生成 T1、FLAIR 子运行。
- Pilot 验收：所有配置可解析、无 NaN/OOM、单模型确实独立、正则开关有日志证据、头掩膜/单位一致。

**核心实验（统计证据）**

- 从 MRI-148 现有配对数据中核验并选取**至少 12 对**符合双序列完整 k-space、空间配准和可用 CSM 要求的不同受试者；旧文档提及 16 对，仅作为候选池，具体可用数必须重新盘点。
- AF12 和 AF16 为主；AF20、AF24 用于高欠采样趋势验证。记录名义与**各对比度实际**加速倍率，不把 T1、FLAIR 的实际 AF 错写为相同。
- 3 seeds：42、43、44。不同方法对同一个受试者 / AF 使用相同 seed 与掩膜；若使用多个掩膜重复采样，先规定掩膜集合。
- 以**受试者为单位**划分开发/调参和最终锁定测试集；测试集不可再次决定超参数或方法选择。
- 若 Gaussian 数 3.6M × 4 组 × 多病例的总训练成本过高，优先完成 AF16 的 4 组与 ≥12 对受试者，再补其它 AF；不能以一个受试者上的大规模参数搜索代替跨受试者统计。
- 严格的四组主结果不得直接挪用历史只关掉 `lambda_edge` 的运行作为“无正则组”。

**建议执行次序：**

1. 新增共享/独立几何显式开关与四组配置生成器；
2. 单元测试：共享张量身份、梯度隔离、loss 开关、single 评分通道、数值匹配；
3. AF16 Pilot 的 A/B/C/D；
4. 冻结超参数，跨受试者 AF16 核心表；
5. AF12/20/24、容量匹配与固定/实用初始化对照；
6. E 组完整 MC-GS、汇总图与统计分析。

## 7. 指标、机制归因与统计

逐受试者、逐序列、逐 AF、逐 seed 输出：

- **主要终点**：head-only magnitude PSNR（dB），T1 和 FLAIR 分别报告。
- **次要图像指标**：head SSIM、NMSE、HFEN、误差图及组织边缘局部 ROI（如灰白质边界）；可补复杂值 PSNR 与 phase agreement。
- **测量域指标**：raw 模型在未采样 k-space 的 NRMSE / `e_unobs`；仅在 retrospective 有完整 k-space 的病例评估，避免用测试测量信息训练模型。
- **审计指标**：raw 与 hard-DC 后图像双口径、Gaussian 数 / 参数数、VRAM、训练 wall-time、失败率。

对任一序列 m，令指标 Q 越大越好：

\[
\Delta_{\rm shared}^{\rm reg}(m)=Q_A(m)-Q_B(m)
\]

\[
\Delta_{\rm shared}^{\rm no-reg}(m)=Q_C(m)-Q_D(m)
\]

\[
I(m)=[Q_A(m)-Q_B(m)]-[Q_C(m)-Q_D(m)]
\]

\[
\Delta_{\rm cross}(m)=Q_E(m)-Q_A(m).
\]

这里 A−B 和 C−D 是**共享几何效应**，I 是共享几何与单通道正则的交互作用，E−A 是额外跨对比度正则收益。不能用 E−B 直接证明共享几何单独有效。

- 先对同一受试者多 seed 求平均，再做受试者层面配对统计；报告 N、均值差、受试者间 SD、95% CI 和效应方向占比。可以使用受试者级配对 bootstrap / permutation，必要时用包含 subject 随机效应的 2×2 模型复核交互项。
- 两个序列分别检验，不用“平均增益为正”掩盖一个序列的负结果；跨 AF 比较明确多重检验处理。
- 历史文档使用 **0.02 dB** 作为单受试者重复容差附近的判断带，**不是跨受试者统计显著性或临床重要性的阈值**。本研究应报告置信区间和绝对增益，谨慎解释非常小的 dB 差异。
- 不能以 GT 最优 checkpoint 或测试 PSNR 最优超参数作为最终报告的模型选择规则；使用预先确定的最终步或独立验证准则。

**预期结论判读：**

| 结果 | 可支持的结论 |
|---|---|
| A>B 且 C>D，并在两个对比度/多病例稳定 | 最有力：共享几何独立贡献存在，单通道正则下依然有效 |
| A>B 但 C≈D | 共享几何收益依赖显式正则；不能声称无正则时仍有效 |
| A≈B 但 C>D | 无正则时的共享先验有效，有正则时额外收益被替代 |
| A≤B 且 C≤D | 不支持当前配方的共享几何重建增益；需要讨论容量、对齐和观测互补性 |
| T1、FLAIR 结果方向相反 | 分序列陈述，不能选择性强调获益序列 |

## 8. 与已有实验的关系及风险

已检查的历史记录：

- `docs/superpowers/specs/2026-09-20-coupling-attribution-design.md`（服务器 `mcgs_hparam_wt`）曾在 AF16 比较 joint / duplicate-single 于 `lambda_edge=0` 和 `0.001`，但**只改变了跨对比度 edge 项**，TV、wavelet、phase-TV 仍存在；因此它不是本次 A/B/C/D 的无正则组。
- `docs/superpowers/specs/2026-09-24-coupling-attribution-stage4-design.md` 记录了 GS-single 需要独立、等预算调参，以及 FLAIR scored-channel 使用 wavelet 权重不一致的问题。
- `docs/2026-09-28-widen-mcgs-summary.md` 指出已有各 AF 的联合优势较小（约 0.02–0.12 dB，随对比度和 AF 改变），且仅一个受试者；**本方案不能预设更强的结构优势。**

因此，**本文能否声称 shared geometry creates measurable reconstruction gain，取决于独立受试者上新的 C−D 与 A−B，而不能靠既有完整 MC-GS 对 GS-single 的微小领先代替。**

## 9. 代码实现接口及交付物（待开发）

建议在原始 MC-GS 目录内最小改造，不触碰 `paired_arc_gs/`：

- `paired_gs/gaussian/joint_gaussian_model.py`：保留共享路径；
- `paired_gs/gaussian/gaussian_model.py`：作为真正 GS-single 单序列模型；
- `paired_gs/trainers/joint_trainer.py`：增加清晰的 `shared_geometry` 分支或新的匹配 single trainer，统一评估和损失配置；
- `arc/cycle.py`：增加/透传结构开关，统一生成 run manifest；
- `configs/abl_shared_geometry/`：四份配置模板，以及共同冻结参数；
- `tests/`：结构身份、梯度隔离、所有正则为零、FLAIR 通道权重、初始化无对方数据泄漏；
- `results_arc/sg2x2_*/`（不提交重建大文件）：包含配置快照、日志、数值结果；
- `docs/experiments/`：保存最终受试者配对统计、表格、失败和负结果。

**最低验收清单**

- [ ] A/B/C/D 的区别严格只落在两个开关 S、R；不存在偷加跨对比度正则
- [ ] 对每个序列的输入 k-space、掩膜、CSM 和 DC 处理相同
- [ ] B/D 的两个独立模型没有几何参数或梯度共享
- [ ] A/C 的共享位置、尺度、旋转由 T1/FLAIR 两个 DC 项共同更新
- [ ] A/B 的通道内正则完全一致，C/D 的显式正则损失严格为零
- [ ] 主归因采用不依赖测试受试者另一序列的匹配初始化
- [ ] 每对比度 N=3.6M 的主容量比较与总 N 匹配的补充比较分开报告
- [ ] 不用测试 GT 决定超参数、checkpoint、最佳方法
- [ ] 至少完成一轮跨受试者、两序列分别统计的 A−B / C−D / 交互项结果
- [ ] 完整 MC-GS（E）相对 A 的增量单独报告，不替代结构归因

---

**范围说明：** 本提交仅固化实验设计，不声称上述新四组已经运行；所有参数为待验证的冻结初始配方，必要改动先通过 Pilot 和代码测试。