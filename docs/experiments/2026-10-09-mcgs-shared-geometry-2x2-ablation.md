# MC-GS 共享结构有效性的四组消融实验（2×2 Factorial Ablation）

> 日期：2026-10-09  
> 目标仓库：sober235/mcgs  
> 分支：exp/shared-geometry-2x2-ablation-20261009  
> 状态：**仅为实验设计；不包含算法修改、实验脚本或新实验结果。**

## 1. 研究目的

**核心科学问题：** 在相同 MRI 物理前向模型、Gaussian 表达能力、训练配置和单对比度显式正则下，T1 与 FLAIR **共享 Gaussian 结构**，相比各自独立的 Gaussian 表达，能否稳定改善欠采样 MRI 重建质量？

本次只检验原始 **MC-GS** 的共享几何机制，不涉及 Paired-ARC-GS。

- 共享结构具体指：T1、FLAIR **共用位置 μ、尺度 s、旋转 q**，各自保留独立的复值 Gaussian 密度 a_T1 和 a_FLAIR。
- GS-single 指：分别用互不共享任何几何参数的 Gaussian 模型重建 T1、FLAIR；**必须是真正的独立模型**，不能未经检验把现有同序列复制到双输入槽的 proxy 当成独立 GS。
- 正则指 **单对比度显式正则**（各自的 edge-preserving TV、wavelet L1、phase-TV），其函数形式和对应对比度的权重在共享组和独立组一致。
- **四组全部禁用跨对比度显式正则**（例如 common-edge、cross-contrast directional TV 和跨对比度 edge/phase guide），使两维因子仅为“共享结构 S”与“单对比度正则 R”。

> **关键边界：** 如果原先提出的 *full MC-GS* 包括 λ_edge 等跨对比度正则，A 组在这份严格结构归因实验中应表述为 **MC-GS (shared geometry + matched within-contrast regularization)**，而不是宣称包含原论文的全部耦合损失。否则 A−B 无法证明增益来自共享结构本身。

## 2. 核心四组设计

| 实验组 | 方法 | 共享几何结构 S | 单对比度正则 R | 跨对比度正则 | 验证目的 |
|---|---|:---:|:---:|:---:|---|
| **A** | **MC-GS + Reg** | ✓ | ✓ | 全关 | 有正则时共享结构的重建效果 |
| **B** | **GS-single + Reg** | ✗ | ✓ | 全关 | A 的严格独立结构对照 |
| **C** | **MC-GS − Reg** | ✓ | ✗ | 全关 | 无显式正则时共享结构的独立作用 |
| **D** | **GS-single − Reg** | ✗ | ✗ | 全关 | 无共享、无显式正则的基础对照 |

所有组都使用 T1、FLAIR 各自的相同多线圈 k-space 测量、采样掩膜、CSM 和数据一致性算子。**“− Reg”仅关闭显式图像先验，不关闭 k-space 数据一致性。**

### 2.1 数学定义

每个序列 m ∈ {T1, FLAIR} 的 Gaussian 图像为：

\[
x_m(\mathbf r)=\sum_{n=1}^{N_m}a_{m,n}G(\mathbf r;\boldsymbol\mu_{m,n},\boldsymbol\Sigma_{m,n}),
\]

其中 a 为复值密度，Σ 由 Gaussian 尺度和旋转共同确定。

- **A/C（共享）**：两个序列使用相同的 μ、Σ；密度 a_T1、a_FLAIR 始终分开学习。每个序列自己的 k-space 残差共同更新共享几何。
- **B/D（独立）**：两个序列分别学习 μ_m、Σ_m、a_m；一个序列的损失不得通过共享参数影响另一个序列。

\[
L_{\mathrm{DC}}=\sum_{m\in\{\mathrm{T1,FLAIR}\}}
\|M_mFC_mx_m-y_m\|_2^2
\]

\[
L_{\mathrm{Reg}}=\sum_m
\left(\lambda_{\mathrm{geo}}R_{\mathrm{edgeTV}}(x_m)+
\lambda_{\mathrm{wav},m}R_{\mathrm{wav}}(x_m)+
\lambda_{\mathrm{ptv}}R_{\mathrm{phaseTV}}(x_m)\right)
\]

A/B 使用 L_DC + L_Reg，C/D 只使用 L_DC。为保持现有代码一致性，可以在四组固定相同的 k-space 频率加权方案；它属于统一的数据保真项，而非本实验切换的显式图像正则。

## 3. 正则参数控制

初次验证固定现有 AF16 phase-3 单对比度配方（Pilot 可检查数值稳定性，但不以测试 PSNR 选择有利参数）：

| 配置 / 损失项 | A、B：+Reg | C、D：−Reg |
|---|---|---|
| edge-preserving TV 权重 λ_geo | 0.10 | **0** |
| edge-TV 开关 | on | off |
| edge-TV κ_q / w_floor / Charbonnier ε | 0.85 / 0.10 / 0.04 | 不生效 |
| wavelet-L1，T1 | 0.12 | **0** |
| wavelet-L1，FLAIR | 0.10 | **0** |
| phase-TV 权重 λ_ptv | 0.03 | **0** |
| common-edge，λ_edge | **0** | **0** |
| directional-TV，λ_dtv | **0** | **0** |
| vectorial-TV，λ_vtv | **0** | **0** |
| 其他跨对比度引导 / 非局部正则 | **0** | **0** |

执行前需要冻结并审核 **解析后的实际 loss 配置**，尤其是 wavelet 的 T1/FLAIR 权重。旧的 `singleFLAIR` 双槽代理有“FLAIR 作为 plain 评分通道时可能拿到 T1 的 wavelet 权重”的历史问题，新对照绝不能重复该不公平性。

### 禁止的比较

- **禁止** A 带 λ_edge = 0.001 而 B 不带：此时 A−B 同时包含结构共享和跨对比度先验两种作用。
- **禁止** C/D 只将 λ_edge 置零却保留 TV、wavelet、phase-TV：这不是“without regularization”。
- **禁止**只挑 T1 中获益的一组而忽略 FLAIR 中可能出现的负结果。

## 4. 公平性控制（核心实验的必要条件）

| 条件 | A/C 共享组 | B/D 独立组 | 控制原则 |
|---|---|---|---|
| 输入 | T1+FLAIR | T1、FLAIR 独立使用 | 各序列训练只使用自身的 k-space 数据；共享组通过共用几何间接融合信息 |
| Gaussian 数量（主实验） | 共享 N=3.6M，双密度 | **每序列** N=3.6M | 每对比度表达容量匹配；报告独立组总参数更多的事实 |
| 初始几何 | 预先规定的匹配分布 | 同一分布、同一随机种子 | **不能**共享组联合 ZF 初始化、独立组单序列 ZF 初始化后直接把差异归因于“共享参数” |
| 初始幅值 | T1/FLAIR 各自 ZF | 分别使用自身 ZF | 避免额外通道信息从幅值初始化泄漏 |
| 优化 | 4000 steps，原配方学习率 | 每个单模型 4000 steps | 每序列 optimizer steps 相同；另报合计 GPU-hours/显存 |
| 自适应 Gaussian 操作 | 禁用 | 禁用 | 固定容量，排除 densify/clone/prune 混杂 |
| 输出评估 | T1、FLAIR 分别评估 | T1、FLAIR 分别评估 | 相同 head-mask、raw/hard-DC 及评估代码 |
| 其他 | seed、mask、CSM、归一化、DC 频率加权固定 | 完全一致 | 无跨序列 guide/GT 引用 |

- **主容量口径**：每个独立模型 N=3.6M。另可追加**资源容量匹配的同一四组实验**：B/D 每序列 N=1.8M，与 A/C 的共享 N=3.6M 比较；两种口径分别报告，不能混用。
- **初始化控制**：优先使用固定 FOV 采样或仅从开发集确定的、**不依赖测试病例任一序列 ZF 的公共初始位置分布**。如进一步测试联合 ZF 引导初始化，须单独标为 end-to-end 附加实验。
- **独立性审核**：B/D 应分别优化两套 Gaussian；T1 损失对 FLAIR 几何梯度为零，反之亦然。
- **调参公平性**：相同的单通道超参数搜索空间、等预算和锁定验证集；不允许只充分调优共享组。测试病例 GT 不用于选择最佳参数或停止时刻。
- **随机性**：所有组同病例/采样掩膜使用配对 seed，记录训练失败/OOM；不同 seed 不视为不同受试者。

## 5. 数据、顺序与评价指标

### 5.1 数据选择

1. **Pilot**：使用现有 jiangyizhou T1+FLAIR retrospective AF16 数据及已验证 CSM、vd-Poisson mask，先检验四组比较能否严格执行。Pilot **不作为泛化或统计显著性的证据**。
2. **核心测试**：使用 MRI-148 上其他具备双序列、配准、完整 k-space 与可用 CSM 的受试者；**目标不少于 12 对**，但实际样本数需要在盘点后决定。预先分出调参/验证和最终锁定测试病例。
3. **AF 扩展**：AF12、AF16 为主要对照；AF20、AF24 验证更高加速倍率。对 T1/FLAIR 分别记录实际 AF，注意 FLAIR 的椭圆有效支持。
4. **重复**：预设 seeds 42/43/44；先优先完成 AF16 的四组、不同受试者，再向其他 AF 扩展。

### 5.2 指标

- 主要：**head-only PSNR (dB)**，T1 与 FLAIR **分别报告**。
- 次要：head SSIM、NMSE、HFEN、edge ROI 误差、视觉差异及伪影。
- 物理一致性：未采样 k-space 预测 NRMSE/e_unobs（仅在 retrospective 全采真值可获得时进行离线评分）。
- 两种图像：**raw rendered volume** 和 **hard-DC reconstructed volume** 两套指标，以免后处理掩盖模型结构差异。
- 资源：Gaussian 个数、参数量、训练时间、显存峰值与失败次数。

### 5.3 结构增益与 2×2 因子归因

对任一序列 m 和指标 Q（如 PSNR，越高越好）：

\[
\Delta_{\mathrm{shared}|R=1}=Q_A-Q_B
\]

\[
\Delta_{\mathrm{shared}|R=0}=Q_C-Q_D
\]

\[
\Delta_{\mathrm{interaction}}=(Q_A-Q_B)-(Q_C-Q_D)
\]

其中：

- **A−B**：正则存在时共享结构的贡献；
- **C−D**：无显式正则时共享结构本身的贡献（**最直接的结构独立性证据**）；
- **(A−B)−(C−D)**：单通道正则是否增强或减弱结构收益。

| 观察结果 | 可以支持的科学结论 |
|---|---|
| A > B **且** C > D，T1/FLAIR、多病例均稳定 | **最强证据**：共享几何自身有效，加入单序列正则后仍有效 |
| A > B，但 C ≈ D | 共享几何的增益依赖或受益于显式正则；不能声称独立正则外仍有明显优势 |
| A ≈ B，但 C > D | 无正则时共享结构有作用，但强单序列正则可能替代了部分共享收益 |
| A ≤ B 且 C ≤ D | 当前数据/配置不支持共享结构提高重建质量 |
| T1 增益为正但 FLAIR 为负 | 必须分别报告并解释；不能用平均 PSNR 掩盖不利结果 |

受试者为独立统计单位：先合并同受试者各 seed 的估计，再分析配对 PSNR 差，输出均值、95% CI、效果方向和病例比例；必要时用配对 bootstrap / permutation 或包含 subject 随机效应的因子模型评估交互。多个对比度/AF 的显著性分析应有事前指定的主终点和多重检验处理。

**判定原则：** 不是只要某个 seed 上 A 或 C 略胜就证明共享有效。论文主张应由锁定测试病例的**受试者级稳定正向差值和置信区间**支持；对微小 dB 差异还需审查实际效应量、图像和伪影。

## 6. 结果记录模板（待填；没有实际结果）

| Method | T1 PSNR ↑ | T1 SSIM ↑ | FLAIR PSNR ↑ | FLAIR SSIM ↑ |
|---|---:|---:|---:|---:|
| **A. MC-GS + Reg** | — | — | — | — |
| **B. GS-single + Reg** | — | — | — | — |
| **C. MC-GS − Reg** | — | — | — | — |
| **D. GS-single − Reg** | — | — | — | — |
| **A−B：共享几何（有正则）** | — | — | — | — |
| **C−D：共享几何（无正则）** | — | — | — | — |

建议配图：四组对应的 T1/FLAIR 中心切面+残差图（同窗宽）、跨受试者配对 PSNR 差值点图，以及按 AF 分层的结构增益与 CI 图。

## 7. 先前工作的注意事项

- 2026-09-20 的 coupling-attribution 实验比较过 joint/single 与 λ_edge on/off，但 **λ_geo、wavelet、phase-TV 仍在**，不能直接作为本方案 C、D 的“无正则”结果。
- 历史 GS-single 使用过相同序列填入 joint trainer 的两个输入槽，且 cross-edge 先验会额外惩罚这样的单序列代理；本方案要求真正的 GS-single。
- 2026-09-24 的 Stage 4 设计明确提示：GS-single 未获与 MC-GS 完全对等的超参数搜索预算时，之前的收益不足以用于严格的结构归因。
- 2026-09-28 的 AF12/16/20/24 记录显示 MC-GS 对独立 GS 的优势相对较小，且当时主要是**单个受试者**；新的主张须经多病例检验。
- 这四组的价值是**区分共享参数的结构先验与显式图像正则的收益**，而不是保证共享方法获胜。负结果也是有效的机制证据。

---

### 本文档的明确范围

**只包括上述 A/B/C/D 四个消融实验的设计与判定口径。** 不添加第五组、不修改 MC-GS 源码、不生成脚本、不声称已运行实验。后续实际训练需根据本方案另行实施。
