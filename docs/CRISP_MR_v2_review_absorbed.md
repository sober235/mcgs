# CRISP-MR v2：评审意见吸收后的无监督三维多对比 MRI 重建方案

## Cross-Fitted Predictive Structure Routing for Unsupervised 3D Multi-Contrast MRI Reconstruction

> **版本说明**
>
> 本版本在原始 `CRISP-GS` 方案基础上，系统吸收了对 MC-GS edge-TV 的独立评审意见。  
> 主要修订包括：
>
> 1. 不再将 3D Gaussian 表示本身作为主要创新；
> 2. 明确将 edge-TV 降级为已验证的负结果；
> 3. 将方法提升为表示无关的 `CRISP-MR`，Gaussian 仅作为一种实现 `CRISP-GS`；
> 4. 将原始 in-sample MDL 改为 **cross-fitted predictive MDL**；
> 5. 将普通梯度余弦改为 Fisher/Gauss–Newton 预条件后的测量梯度一致性；
> 6. 增加充分调参的 CS-SENSE、结构引导 TV、低秩和单对比 3DGS 等强基线；
> 7. 引入表示能力天花板实验、冻结开发集/测试集协议、多 mask/多初始化方差分析；
> 8. 修正 CSM 循环评价、幅值域“零空间误差”、背景主导 SSIM 和数据去标识问题；
> 9. 将病灶安全性从未经验证的主张改为必须通过仿真和真实病理队列检验的研究假设。

---

# 1. 评审结论与方案调整

## 1.1 对原始 MC-GS edge-TV 的最终判断

原始 MC-GS edge-TV 不应继续作为论文主方法，原因包括：

- 复杂值 3D Gaussian MRI 重建的核心表示已被已有 3DGSMR 工作覆盖；
- 共享几何与对比度独立幅值的直接组合，创新增量有限；
- edge-TV 本质上是固定权重的 Perona–Malik 型 lagged diffusivity；
- edge-TV 未通过项目预先设定的 edge-NMSE、HFEN 和 whole-brain NMSE 判据；
- edge-TV 相对 isotropic-TV 的 whole-volume SSIM 优势来自脑外背景，而脑内指标更差；
- 充分调参的单对比 CS-SENSE(TV) 在 FLAIR 上可超过当前 MC-GS，并在 T1W 上基本打平；
- 当前结果来自单受试者、单 mask、单加速倍数、单初始化和同一数据上的反复调参。

因此：

\[
\boxed{
\text{MC-GS edge-TV = 负结果与基线，不再是新方法。}
}
\]

原始 MC-GS isotropic-TV 可保留为 legacy baseline，但也不能作为 MedIA 级创新。

---

## 1.2 对原 CRISP-GS 方案的保留部分

以下思想仍具有研究价值：

1. 不预设所有多对比结构均应共享；
2. 在局部三维 primitive 层面区分 shared 与 contrast-specific information；
3. 使用真实多线圈测量，而不是图像相关性或 cross-gradient，决定结构归属；
4. 动态控制表示容量，而不是固定 Gaussian 数量；
5. 输出对比度特异结构证据，而不只输出最终图像。

---

## 1.3 对原 CRISP-GS 方案必须修改的部分

### 问题 A：Gaussian 表示不能继续充当主要 novelty

3DGSMR 已完成复杂值 3D Gaussian、欠采样 k-space 和 scan-specific MRI 重建；AtlasGS 也已研究多模态共享 Gaussian geometry。

因此，新方案必须从：

> Gaussian-based multi-contrast reconstruction

转向：

> **measurement-audited shared/private structure selection**

Gaussian 只是实现结构路由的一种 atom representation。

### 问题 B：原始 in-sample MDL 存在选择偏差

原方案在同一批测量上：

- 生成 candidate；
- 优化 candidate；
- 计算 residual reduction；
- 选择 shared/private 状态。

即使增加 BIC/MDL penalty，也容易因动态搜索、非线性参数化、多个候选比较和未知有效自由度而过拟合。

因此必须使用独立测量子集审计候选结构。

### 问题 C：普通梯度余弦依赖参数坐标

Gaussian center、scale 和 rotation 的量纲与曲率不同。直接计算：

\[
\frac{\langle g_m,g_\ell\rangle}
{\|g_m\|\|g_\ell\|}
\]

会受参数重标度影响，不能作为稳定的几何一致性指标。

因此应使用局部 Fisher 或 Gauss–Newton 度量进行预条件。

### 问题 D：病灶保护尚无真实证据

当前可用 T1W–FLAIR 数据主要用于技术验证，不能自动证明：

- FLAIR 病灶不会被压制；
- 增强区域不会被平扫污染；
- 无监督模型不会产生假病灶。

病灶安全性必须通过：

1. 可控注入实验；
2. 真实病理或平扫–增强队列；
3. 盲法阅片或下游任务；

进行验证。

---

# 2. 新的方法定位

## 2.1 总方法名称

# CRISP-MR

## Cross-Fitted Predictive Structure Routing for Unsupervised Multi-Contrast MRI

中文名称：

> **基于交叉拟合预测证据的无监督多对比 MRI 共享–私有结构路由方法**

Gaussian 实现称为：

# CRISP-GS

## Gaussian-Primitive Instantiation of CRISP-MR

这种命名区分：

- `CRISP-MR`：方法学贡献；
- `CRISP-GS`：具体三维显式表示实现。

---

## 2.2 核心科学问题

\[
\boxed{
\text{在没有全采标签和外部训练数据时，}
\atop
\text{如何利用未参与候选拟合的真实 k-space，}
\atop
\text{决定一个局部三维结构应共享、应私有，还是应被拒绝？}
}
\]

该问题不同于：

- 将多个对比度输入同一网络；
- 假设同一 latent anatomy；
- 使用 reference image 强制引导；
- 用 cross-gradient 对齐边缘；
- 用外部数据学习 content/style decomposition。

---

# 3. “无监督”的严格定义

CRISP-MR 属于：

> **scan-specific unsupervised inverse reconstruction with internal measurement audit**

具体含义：

- 不使用全采图像作为训练标签；
- 不使用外部受试者训练网络；
- 不使用预训练 CNN、扩散模型或生成模型；
- 不使用病灶标注参与重建；
- 每个受试者单独优化；
- acquired k-space 在结构选择阶段进行内部交叉拟合；
- 状态与模型复杂度确定后，再使用全部 acquired k-space 重新拟合最终模型。

内部 measurement split 仅用于**模型审计和结构选择**，不训练 population network，因此不等同于传统 SSDU 网络训练。

---

# 4. 成像模型

对同一受试者的 \(M\) 个对比度：

\[
y_m=A_mx_m+n_m,
\qquad
m=1,\ldots,M,
\]

其中：

\[
A_m=P_{\Omega_m}F S_m.
\]

符号定义：

- \(x_m\in\mathbb C^{D\times H\times W}\)：第 \(m\) 个复数 MR 图像；
- \(y_m\in\mathbb C^{C\times|\Omega_m|}\)：已采集多线圈 k-space；
- \(S_m\)：coil sensitivity operator；
- \(F\)：中心化三维 FFT 或一般 Fourier operator；
- \(P_{\Omega_m}\)：已采样位置选择算子；
- \(n_m\sim\mathcal{CN}(0,\Sigma_{n,m})\)。

首先使用噪声协方差执行 prewhitening，使后续 residual 近似服从单位方差复高斯模型。

---

# 5. 表示无关的共享–私有结构模型

CRISP-MR 将每个对比度写为：

\[
x_m
=
\Phi^{S}(\eta^{S})a_m^{S}
+
\Phi_m^{P}(\eta_m^{P})a_m^{P}.
\]

其中：

- \(\Phi^{S}\)：共享空间 primitive 字典；
- \(\eta^{S}\)：共享几何参数；
- \(a_m^{S}\)：第 \(m\) 个对比度的复系数；
- \(\Phi_m^{P}\)：第 \(m\) 个对比度的私有 primitive 字典；
- \(\eta_m^{P}\)：私有几何参数；
- \(a_m^{P}\)：私有复系数。

可选择的 primitive 表示包括：

- 3D anisotropic Gaussian；
- 局部 spline atom；
- voxel block；
- compact-support INR component。

本项目首选 Gaussian，是因为其：

- 几何参数显式；
- 适合 primitive-level split/merge；
- 可连续表示三维体积；
- 便于定义 shared/private state。

但论文必须通过 voxel-grid 或另一种表示的控制实验，证明收益来自 routing，而不只是 Gaussian 表示。

---

# 6. CRISP-GS：Gaussian 实现

共享场：

\[
x_m^{S}(\mathbf r)
=
\sum_{j\in\mathcal S}
a_{m,j}^{S}
\phi(\mathbf r;\eta_j^{S}),
\]

私有场：

\[
x_m^{P}(\mathbf r)
=
\sum_{j\in\mathcal P_m}
a_{m,j}^{P}
\phi(\mathbf r;\eta_{m,j}^{P}),
\]

其中：

\[
\eta_j=(\boldsymbol\mu_j,\mathbf s_j,\mathbf q_j).
\]

每个 candidate primitive 状态为：

\[
z_j\in\{0,S,P_1,\ldots,P_M,\mathrm{Sep}\}.
\]

- \(0\)：拒绝；
- \(S\)：多个对比度共享一个几何 primitive；
- \(P_m\)：只属于第 \(m\) 个对比度；
- \(\mathrm{Sep}\)：多个对比度均需要该区域，但要求不同几何，分别建立 private primitives。

---

# 7. 交叉拟合测量审计

## 7.1 acquired k-space 划分

对每个对比度，将已采样位置划分为 \(K\) 个审计 fold：

\[
\Omega_m
=
\bigcup_{k=1}^{K}
\Lambda_m^{(k)},
\qquad
\Lambda_m^{(k)}
\cap
\Lambda_m^{(\ell)}
=
\varnothing.
\]

第 \(k\) 个 fold 中：

\[
\Theta_m^{(-k)}
=
\Omega_m\setminus\Lambda_m^{(k)}.
\]

- \(\Theta_m^{(-k)}\)：候选拟合集；
- \(\Lambda_m^{(k)}\)：独立预测审计集。

ACS 需要特殊处理：

- CSM 估计使用固定 ACS；
- 结构审计 folds 主要在非 ACS 区域进行；
- 各 fold 按径向频率 shell 分层，避免审计集只覆盖单一频率区域。

推荐：

\[
K=3\text{ 或 }4.
\]

---

## 7.2 审计原则

对 candidate \(j\) 和状态模型 \(\mathcal M\)：

1. 仅使用 \(\Theta_m^{(-k)}\) 优化 candidate 参数；
2. 不允许使用 \(\Lambda_m^{(k)}\) 更新 candidate；
3. 在 \(\Lambda_m^{(k)}\) 上计算预测负对数似然；
4. 在所有 folds 上聚合；
5. 状态确定后，才用全部 \(\Omega_m\) 重新拟合。

这样可避免同一批测量既提出结构又证明结构。

---

# 8. 候选结构生成

对当前估计 \(\widehat x_m\)，训练子集 residual 为：

\[
r_m^{(-k)}
=
P_{\Theta_m^{(-k)}}
\left(
F S_m\widehat x_m-y_m
\right).
\]

使用 coil-和采样密度归一化的伴随残差：

\[
b_m^{(-k)}
=
\frac{
\left|
A_{m,-k}^{H}
W_m r_m^{(-k)}
\right|
}{
\sqrt{
\operatorname{diag}
\left(
A_{m,-k}^{H}W_mA_{m,-k}
\right)
}
+\epsilon
}.
\]

其中 \(W_m\) 为 prewhitening 后的噪声权重或密度补偿。

稳健标准化：

\[
\widetilde b_m
=
\frac{
b_m-\operatorname{median}(b_m)
}{
\operatorname{MAD}(b_m)+\epsilon
}.
\]

对于两个对比度，可定义：

\[
b^{S}
=
\sqrt{
[\widetilde b_1]_+
[\widetilde b_2]_+
},
\]

\[
b_m^{P}
=
\left[
\widetilde b_m-
\max_{\ell\neq m}\widetilde b_\ell
\right]_+.
\]

- \(b^S\)：共享 candidate proposal；
- \(b_m^P\)：私有 candidate proposal。

proposal map 只负责生成候选，不负责最终接受。

---

# 9. Fisher 预条件的测量梯度一致性

## 9.1 局部梯度

对 candidate \(j\) 的几何参数 \(\eta_j\)，第 \(m\) 个对比度在拟合集上的梯度为：

\[
g_{m,j}
=
J_{m,j}^{H}
W_m r_m,
\]

其中：

\[
J_{m,j}
=
\frac{\partial A_mx_m}{\partial\eta_j}.
\]

## 9.2 局部曲率预条件

定义局部 Gauss–Newton/Fisher 矩阵：

\[
H_{m,j}
=
J_{m,j}^{H}W_mJ_{m,j}
+
\lambda_H I.
\]

预条件梯度：

\[
\widetilde g_{m,j}
=
H_{m,j}^{-1/2}g_{m,j}.
\]

这使 center、scale 和 rotation 的不同单位与局部曲率得到归一化。

## 9.3 一致性分数

\[
C_{m\ell,j}
=
\frac{
\operatorname{Re}
\left\langle
\widetilde g_{m,j},
\widetilde g_{\ell,j}
\right\rangle
}{
\|\widetilde g_{m,j}\|_2
\|\widetilde g_{\ell,j}\|_2+\epsilon
}.
\]

同时定义 activity：

\[
A_{m,j}
=
\|\widetilde g_{m,j}\|_2.
\]

状态 proposal 规则：

- 多个对比度均 active，且 \(C_{m\ell,j}>0\)：允许 shared；
- 仅一个对比度 active：优先 private；
- 多个对比度 active，但 \(C_{m\ell,j}<0\)：优先 separate-private；
- 所有对比度均不 active：拒绝 candidate。

activity threshold 不手工固定，而由噪声 residual 的 parametric bootstrap 校准。

梯度一致性只是 admissibility gate，最终状态由独立审计数据决定。

---

# 10. Cross-Fitted Predictive MDL

## 10.1 预测负对数似然

第 \(k\) 个审计 fold 上：

\[
\mathcal L_{\mathrm{pred}}^{(k)}(\mathcal M)
=
\sum_m
\frac{
\left\|
P_{\Lambda_m^{(k)}}
\left(
F S_mx_m^{\mathcal M,(-k)}-y_m
\right)
\right\|_2^2
}{
2\widehat\sigma_m^2
}.
\]

其中 \(x_m^{\mathcal M,(-k)}\) 只使用 \(\Theta_m^{(-k)}\) 拟合。

总预测损失：

\[
\mathcal L_{\mathrm{CF}}(\mathcal M)
=
\sum_{k=1}^{K}
\mathcal L_{\mathrm{pred}}^{(k)}(\mathcal M).
\]

---

## 10.2 复杂度编码

由于非线性 Gaussian 模型存在参数重叠和不可辨识性，不能直接把参数个数当作真实自由度。

定义局部有效秩：

\[
d_{\mathrm{eff}}(\mathcal M)
=
\operatorname{rank}_{\tau}
\left(
J_{\mathcal M}^{H}WJ_{\mathcal M}
\right).
\]

复杂度编码为：

\[
L_{\mathrm{comp}}(\mathcal M)
=
\frac{
d_{\mathrm{eff}}(\mathcal M)
}{2}
\log N_{\mathrm{fit}}
+
L_{\mathrm{state}}(\mathcal M)
+
L_{\mathrm{geom}}(\mathcal M).
\]

其中：

- \(L_{\mathrm{state}}\)：shared/private/absent 状态编码；
- \(L_{\mathrm{geom}}\)：量化后 center、scale、rotation 的几何编码长度。

最终：

\[
\operatorname{CF\text{-}PMDL}(\mathcal M)
=
\mathcal L_{\mathrm{CF}}(\mathcal M)
+
L_{\mathrm{comp}}(\mathcal M).
\]

---

## 10.3 Candidate 接受判据

候选状态：

\[
\mathcal M_j^\star
=
\arg\min_{\mathcal M\in\mathcal A_j}
\operatorname{CF\text{-}PMDL}(\mathcal M),
\]

其中 \(\mathcal A_j\) 为 gradient gate 允许的状态集合。

candidate 仅在以下条件同时满足时接受：

1. 相对 absent 模型：

\[
\Delta_j
=
\operatorname{CF\text{-}PMDL}(\mathcal M_0)
-
\operatorname{CF\text{-}PMDL}(\mathcal M_j^\star)
>0;
\]

2. 至少 \(K-1\) 个 folds 的预测误差方向一致；
3. 改善超过 null-candidate bootstrap threshold；
4. primitive-level false discovery rate 低于预设水平。

这比原始单次 in-sample MDL 更能抵抗动态候选搜索造成的过拟合。

---

# 11. 动态结构操作

## 11.1 Birth

从 \(b^S\) 和 \(b_m^P\) 的非极大值峰值产生候选。

## 11.2 Route

使用 Fisher-preconditioned gradient gate 和 CF-PMDL 决定：

- shared；
- private；
- separate-private；
- absent。

## 11.3 Split

若现有 shared primitive 在多个优化轮次中持续满足：

\[
C_{m\ell,j}<0
\]

且 separate-private 的 CF-PMDL 更低，则拆分。

## 11.4 Merge

两个 primitive 合并后，若 cross-fitted predictive loss 不升高且复杂度下降，则合并。

## 11.5 Prune

删除 primitive 后，若 CF-PMDL 不升高，则删除。

## 11.6 Stop

连续若干 structural rounds 中不存在显著 CF-PMDL 改善时停止。

停止点不能由 GT PSNR 或同一报告集上的最佳指标决定。

---

# 12. 频率 continuation 与数据一致性

## 12.1 频率 continuation

使用：

\[
0<\rho_1<\rho_2<\cdots<\rho_T=1
\]

逐步开放已采样 k-space：

\[
\Omega_m^{(t)}
=
\{
\mathbf k\in\Omega_m:
\|\mathbf k\|\leq\rho_t
\}.
\]

该策略替代固定的 \(0.3/1.0/2.2\) 频率权重。

## 12.2 周期性复数数据一致性

\[
x_m^{\mathrm{DC}}
=
\arg\min_x
\|A_mx-y_m\|_2^2
+
\mu
\|x-x_m^{\mathrm{prior}}\|_2^2.
\]

使用 complex CG-SENSE。

所有方法必须使用相同 data-consistency operator，不能只对提出方法进行额外 hard-DC。

---

# 13. 算法流程

```text
Algorithm: CRISP-MR / CRISP-GS

Input:
    Acquired multi-coil k-space {y_m}
    Sampling sets {Omega_m}
    Coil sensitivities {S_m}
    Noise covariance estimates
    Number of audit folds K

Output:
    Reconstructed contrasts {x_m}
    Shared primitive set S
    Private primitive sets {P_m}
    Predictive evidence maps {E_m}

1. Prewhiten k-space and coil sensitivities
2. Construct frequency-stratified audit folds
3. Initialize a low-capacity shared representation
4. Run frequency continuation

5. At each structural round:
6.     For each fold k:
7.         Fit current model on Omega \ Lambda_k
8.         Compute coil-normalized adjoint residual maps
9.         Propose shared/private candidates

10.    For each candidate:
11.        Compute local Fisher-preconditioned gradients
12.        Build admissible states
13.        Fit each admissible state on fold training measurements
14.        Evaluate predictive likelihood on held-out measurements
15.        Compute CF-PMDL
16.        Aggregate fold consistency
17.        Accept shared/private/separate state or reject

18.    Refit accepted structure on all acquired measurements
19.    Apply periodic complex CG-SENSE data consistency
20.    Perform split, merge and prune using cross-fitted evidence
21.    Stop when no operation yields significant CF-PMDL improvement

22. Final refit using all acquired k-space
23. Export reconstructions, routing states and evidence maps
```

---

# 14. 表示能力与方法可行性的前置闸门

在大规模开发 CRISP-GS 前，必须完成以下 go/no-go experiments。

## Gate G0：Gaussian full-data representation ceiling

使用满采样 k-space 拟合同一 Gaussian renderer，比较：

- voxel-grid full-data CG-SENSE；
- Gaussian full-data fit；
- Gaussian 数量与 residual saturation。

主要指标：

\[
E_{\mathrm{coil,full}}
=
\frac{
\|FS\widehat x-y^{\mathrm{full}}\|_2
}{
\|y^{\mathrm{full}}\|_2
}.
\]

若 Gaussian 在满采样条件下仍明显达不到 voxel-grid reference，说明表示本身是瓶颈，CRISP-GS 不应继续作为主要实现。

## Gate G1：充分调参的单对比基线

必须重新实现并冻结：

- CS-SENSE(TV)；
- CS-SENSE wavelet；
- single-contrast 3DGSMR；
- locally low-rank 或 structured low-rank。

所有超参数只能在 development subjects 上调节，并要求最优值不能落在搜索网格边界；否则继续扩展网格。

如果 Gaussian 单对比表示持续不优于充分调参的经典 CS，论文不能宣称 Gaussian 本身提升重建质量。

## Gate G2：多对比共享是否真实有益

在完全相同表示与复杂度下比较：

- independent reconstruction；
- all-shared；
- all-private；
- CRISP routing。

只有当 all-shared 或 CRISP 在 complementary sampling 下相对 independent 产生稳定收益，才说明多对比联合建模有价值。

## Gate G3：选择性共享是否优于强制共享

在对比度特异注入实验中，CRISP 必须同时满足：

\[
\Delta\text{background fidelity}\geq0
\]

和：

\[
\Delta\text{innovation preservation}>0,
\]

且不能增加 cross-contrast leakage。

---

# 15. 数据与冻结实验协议

## 15.1 当前可用 T1W–FLAIR 数据

评审清点显示已有 16 例配对三维 T1W–FLAIR 数据。

该信息说明多受试者验证具有工程可行性，但以下事项仍需确认：

- 伦理审批是否覆盖当前二次分析；
- 是否允许发表聚合结果；
- 数据是否可匿名化后内部使用；
- 是否存在重复扫描或非独立样本；
- 每例 CSM、矩阵和扫描协议是否一致。

不能仅因数据已存在就自动推断“不需要新的伦理审查”。

## 15.2 去标识

所有原始姓名、拼音和私有绝对路径必须替换为：

```text
S001
S002
...
```

配置文件只保存 token，不保存姓名或可识别路径。

论文必须包含：

- IRB/伦理审批声明；
- 知情同意或豁免说明；
- 数据共享限制；
- 去标识流程。

## 15.3 推荐 subject split

首版建议：

- development：5 例；
- locked test：11 例。

development subjects 用于：

- baseline hyperparameter selection；
- CF-PMDL numerical thresholds；
- frequency schedule；
- candidate budget；
- bootstrap/FDR calibration。

test subjects 在冻结配置后只运行一次主分析。

若需要更稳健估计，可补充 subject-level cross-validation，但不能替代 locked test。

## 15.4 mask 与初始化

test protocol 至少包括：

- 3 个独立 mask realizations；
- 对随机初始化敏感的方法使用 3 个初始化；
- AF \(\in\{6,8,10,12\}\)；
- same-mask 与 complementary-mask 两种情况。

统计分析使用 subject、mask 和 initialization 的层级结构，而不是把所有运行当成独立受试者。

---

# 16. 公平基线协议

## 16.1 必须纳入的无训练基线

1. Zero-filled / adjoint SENSE；
2. CS-SENSE(TV)；
3. CS-SENSE(\(\ell_1\)-wavelet)；
4. structure-guided TV / parallel level sets；
5. joint or vectorial TV；
6. locally low-rank；
7. structured low-rank，如适用；
8. single-contrast 3DGSMR；
9. independent scan-specific INR；
10. joint multi-contrast INR；
11. original all-shared MC-GS isotropic-TV；
12. CRISP-GS。

edge-TV 仅作为 documented negative ablation，不作为 strongest baseline。

## 16.2 可选外部训练方法

可另外报告：

- VarNet；
- MoDL；
- supervised multi-contrast network；
- self-supervised population network；
- PnP-CoSMo。

但必须与无训练方法分组，避免混淆 training regime。

## 16.3 调参规则

- 所有 baseline 代码进入同一仓库；
- 保存 config、日志、输出 JSON 和版本哈希；
- 仅 development subjects 调参；
- 搜索最优值不得位于网格边界；
- 所有方法使用相同 CSM、mask、normalization 和 DC；
- 运行失败和负结果同样记录；
- test set 结果不得反向修改超参数。

---

# 17. 评价体系修正

## 17.1 Primary：coil-domain full-k-space error

对于 retrospective full k-space：

\[
E_{\mathrm{coil}}
=
\frac{
\|FS\widehat x-y^{\mathrm{full}}\|_2
}{
\|y^{\mathrm{full}}\|_2
}.
\]

未采样区误差：

\[
E_{\mathrm{unobs}}
=
\frac{
\|
P_{\Omega^c}
(FS\widehat x-y^{\mathrm{full}})
\|_2
}{
\|
P_{\Omega^c}y^{\mathrm{full}}
\|_2
}.
\]

这是复数、多线圈、与前向模型一致的指标。

不得再把：

\[
(1-M)
\left(
F(x)-F(|x_{\mathrm{GT}}|)
\right)
\]

称为严格的 null-space error。

## 17.2 Image-domain reference

为避免同一 1-map CSM 同时定义 GT 与前向模型产生循环评价，至少报告两种 reference：

1. fully sampled root-sum-of-squares 或 adaptive coil-combined magnitude；
2. independently configured full-data CG-SENSE reference。

可进一步进行：

- 1-map ESPIRiT；
- 2-map ESPIRiT；
- JSENSE；

敏感性分析。

1-map 与 2-map 方法应在匹配或模型无关 reference 下比较。

## 17.3 ROI 与动态范围

Primary image metrics 使用：

- 固定 brain/body mask；
- mask 内统一动态范围；
- brain-restricted PSNR；
- brain-restricted 3D SSIM；
- NMSE；
- HFEN。

whole-volume SSIM 仅作为补充指标，因为大体积空气背景可主导结果。

## 17.4 Data consistency

每个候选方法都报告：

- raw output；
- identical DC-projected output。

论文必须明确主表使用哪个版本，不能只对某一方法追加后处理。

---

# 18. 对比度特异信号实验

## 18.1 当前健康 T1W–FLAIR 数据可完成的实验

在真实 fully sampled 数据上进行物理一致的信号注入：

1. 在 complex image domain 构造局部 FLAIR-only lesion；
2. 通过真实 CSM 和 FFT 生成多线圈 k-space；
3. 保持 T1W 不变；
4. 测试不同 lesion size、contrast、location；
5. 重建后评价 private routing 和 leakage。

## 18.2 指标

Innovation preservation：

\[
P_{\mathrm{keep}}
=
\frac{
\|
\widehat x_m^{\mathrm{lesion}}
-
\widehat x_m^{\mathrm{control}}
\|_{\mathrm{ROI}}
}{
\|
x_m^{\mathrm{lesion}}
-
x_m^{\mathrm{control}}
\|_{\mathrm{ROI}}
}.
\]

Cross-contrast leakage：

\[
L_{\mathrm{leak}}
=
\frac{
\|
\widehat x_\ell^{\mathrm{lesion}}
-
\widehat x_\ell^{\mathrm{control}}
\|_{\mathrm{ROI}}
}{
\|
x_m^{\mathrm{lesion}}
-
x_m^{\mathrm{control}}
\|_{\mathrm{ROI}}
},
\qquad
\ell\neq m.
\]

还需评价：

- lesion CNR；
- lesion volume bias；
- small-lesion sensitivity；
- private-state detection sensitivity；
- false private primitive rate。

## 18.3 真实临床验证

要对标 MedIA，仍需要至少一种真实临床队列：

- MS / white-matter lesion T1W–FLAIR；
- glioma 或 metastasis pre/post contrast；
- vessel-wall pre/post contrast；
- 其他具有明确 contrast-specific findings 的三维 raw k-space。

没有真实病理数据时，只能主张“可控仿真中的对比度特异信号保持”，不能主张临床病灶安全性。

---

# 19. 理论范围修正

原方案中的：

\[
O(N)
\text{ measurement mismatch}
\quad\text{vs}\quad
O(\log N)
\text{ complexity}
\]

可作为理论直觉，但不能直接表述为一般 nonlinear dynamic Gaussian model 的既定结论。

可证明的首版理论应限定在：

1. 固定有限 candidate set；
2. whitening 后独立复高斯噪声；
3. candidate 的局部线性化成立；
4. forward operator 已知；
5. shared 与 private 模型局部可辨识；
6. audit folds 与 candidate fitting 独立；
7. 不考虑无限次 adaptive search。

在这些条件下，研究：

### 命题 1：predictive model-selection consistency

真实 shared candidate 的 cross-fitted predictive loss 不因增加独立几何参数而持续改善，因此复杂度更低的 shared model 被选择。

### 命题 2：private structure detectability

当 contrast-specific candidate 在审计 k-space 中产生的信号能量超过噪声与复杂度阈值时，private model 相对 absent/shared model 被一致选择。

### 命题 3：false discovery control

在 null candidate 下，bootstrap-calibrated CF-PMDL threshold 控制错误 primitive 接受概率。

对于完整 birth–split–merge 动态系统，只主张经验机制，不应过度声称全局收敛或普适选择一致性。

---

# 20. 消融实验

必须包括：

1. independent reconstruction；
2. all-shared；
3. all-private；
4. shared/private but no measurement audit；
5. in-sample MDL；
6. cross-fitted predictive MDL；
7. raw gradient cosine；
8. Fisher-preconditioned concordance；
9. no complexity code；
10. no bootstrap/FDR calibration；
11. fixed Gaussian count；
12. conventional densification；
13. birth only；
14. birth + prune；
15. complete birth–route–split–merge–prune；
16. no frequency continuation；
17. no in-loop DC；
18. voxel-grid CRISP；
19. Gaussian CRISP；
20. original MC-GS isotropic-TV；
21. documented edge-TV negative result。

---

# 21. 代码改造

## 21.1 保留

- complex 3D Gaussian renderer；
- tiled CUDA voxelizer；
- multi-coil forward operator；
- 3D FFT；
- ESPIRiT CSM pipeline；
- config snapshot；
- run hash；
- negative-result logging。

## 21.2 删除或降级

- edge-TV 作为主方法；
- 固定 high-frequency weight；
- 固定 576k warm-start cloud；
- 单受试者 checkpoint 初始化；
- test-volume hyperparameter search；
- whole-volume SSIM 主判据；
- magnitude-domain pseudo-nullspace metric。

## 21.3 新增

```text
crisp/
    representations/
        gaussian_atoms.py
        voxel_atoms.py

    audit/
        fold_builder.py
        predictive_likelihood.py
        bootstrap_null.py
        primitive_fdr.py

    routing/
        normalized_residual.py
        fisher_gradient.py
        state_models.py
        cf_pmdl.py
        router.py

    control/
        birth.py
        split.py
        merge.py
        prune.py

    optimization/
        frequency_continuation.py
        complex_cg_sense.py
        refit_all_measurements.py

    evaluation/
        coilspace_metrics.py
        independent_reference.py
        brain_metrics.py
        innovation_leakage.py
```

---

# 22. 分阶段执行计划

## Phase 0：先结算现有方法

在开发 CRISP 前完成：

1. 仓库化、重新运行充分调参的 CS-SENSE；
2. iso-TV 与 edge-TV 的公平同权重对照；
3. full-data Gaussian representation ceiling；
4. coil-domain complex metric；
5. 现有 16 例匿名化和数据质量检查；
6. 伦理覆盖确认。

## Phase 1：最小 CRISP 原型

只实现：

- shared/private Gaussian banks；
- 3-fold audit；
- candidate birth；
- Fisher gradient gate；
- shared/private/absent 三状态 CF-PMDL；
- final refit。

暂不实现 split、merge 和多对比 \(M>2\)。

## Phase 2：机制验证

完成：

- synthetic shared/private identifiability；
- lesion injection；
- all-shared/all-private comparison；
- 5-development/11-test；
- 3 masks；
- AF 6/8/10/12。

## Phase 3：完整结构控制

加入：

- separate-private；
- split；
- merge；
- prune；
- FDR calibration；
- voxel-grid CRISP control。

## Phase 4：MedIA 级验证

增加：

- 真实病理或 pre/post cohort；
- prospective acceleration；
- reader study 或下游临床任务；
- 外部扫描协议；
- failure-case analysis。

---

# 23. Go / No-Go 判据

## Go 条件

只有同时满足以下条件，才继续以 MedIA 为目标：

1. Gaussian full-data fit 不构成明显表示瓶颈；
2. tuned baselines 可复现且参数冻结；
3. CRISP 相对 all-shared 在 test subjects 上显著降低 leakage；
4. CRISP 相对 independent/all-private 提高共享解剖恢复；
5. coil-domain unobserved-k error 不恶化；
6. 结果在多 mask 和多初始化下稳定；
7. 真实临床特异信号实验成立；
8. 所有数据完成匿名化和伦理确认。

## No-Go / Pivot 条件

若出现以下情况，应及时转向：

- Gaussian full-data ceiling显著低于 voxel-grid；
- tuned CS/LLR 持续优于 Gaussian；
- routing 只改变图像外观，不改善审计 k-space；
- private evidence map 与真实注入位置不对应；
- shared/private 状态对 folds 极不稳定；
- 计算成本远高于收益；
- 真实病理信号仍被压制或泄漏。

对应 pivot：

1. 保留 CRISP-MR，放弃 CRISP-GS；
2. 使用 voxel-grid、INR 或 compact local basis；
3. 将当前资产转为公平 baseline benchmark / reproduction study；
4. 降低目标期刊至 MRM、TMI 技术验证或会议论文。

---

# 24. 修订后的创新点

## 创新点 1：cross-fitted predictive structure routing

在无外部训练条件下，用未参与 candidate fitting 的真实 k-space 对 shared/private/absent 状态进行预测审计。

## 创新点 2：Fisher-normalized measurement agreement

在局部测量曲率度量下比较不同对比度对 primitive geometry 的要求，避免图像域相似性和参数坐标尺度造成错误共享。

## 创新点 3：audited shared–private volumetric decomposition

将稳定解剖恢复与 contrast-specific innovation recovery 分离，并要求每一个 private primitive 均具有独立测量预测收益。

## 创新点 4：representation-falsifiable framework

明确区分 routing 贡献和 Gaussian 表示贡献，并通过 full-data ceiling、voxel control 和强经典基线使方法可被证伪。

---

# 25. 与相关工作的区别

## 3DGSMR

3DGSMR 已建立复杂值 3D Gaussian 的 scan-specific MRI 重建。

CRISP-MR 不主张 Gaussian 首创，其增量必须来自：

- 多对比结构状态；
- 独立测量审计；
- predictive routing；
- contrast-specific leakage control。

## AtlasGS

AtlasGS 使用一个结构扫描学习 shared Gaussian scaffold，再用于其他模态的空间分辨率 harmonization。

CRISP-MR 不预设某一对比度为可信几何参考，而在多个欠采样对比度间对每个局部 primitive 做对称状态选择。

## Joint multi-contrast INR

现有 joint INR 通过共享隐式表示和 complementary sampling 联合恢复多个对比度。

CRISP-MR 的核心问题是：

> 哪些局部结构不应共享，以及如何用独立测量证据拒绝共享。

## Structure-guided TV

结构引导 TV 已通过梯度位置或方向共享利用另一对比度。

CRISP-MR 不把所有边缘关系写入固定正则，而对局部结构状态进行预测模型选择。

## PnP-CoSMo

PnP-CoSMo 使用外部图像数据学习 content/style decomposition。

CRISP-MR 不使用外部图像训练集，shared/private 状态由当前扫描测量决定。

## Multi-contrast SSDU

SSDU 类方法通过 k-space partition 训练 population network。

CRISP-MR 的 partition 仅用于当前扫描的结构审计，状态确定后使用全部 acquired measurements 重新拟合。

---

# 26. 论文定位

本项目不能再表述为：

> 3D Gaussian 比经典 CS 更强。

也不能表述为：

> edge-preserving TV 提高了重建质量。

更准确的论文主线为：

> **Multi-contrast MRI reconstruction should not assume a globally shared anatomical representation. CRISP-MR uses cross-fitted acquired-measurement prediction to determine whether each local volumetric structure is shared, contrast-specific, or unsupported.**

MedIA 级论文需要同时满足：

\[
\text{方法学新问题}
+
\text{严格无偏评价}
+
\text{临床特异信号验证}
+
\text{强基线}
+
\text{真实多受试者/前瞻数据}.
\]

仅依靠当前 16 例健康 T1W–FLAIR retrospective 数据，可以完成：

- 方法原型；
- 多受试者技术验证；
- 仿真病灶；
- 公平基线 benchmark。

但不足以单独支撑：

- 临床病灶安全性；
- 平扫–增强 generality；
- MedIA 级临床结论。

---

# 27. 最终建议

评审意见应当**实质性吸收，而不是只增加几个基线或重写文字**。

修订后的最优路线是：

\[
\boxed{
\text{MC-GS edge-TV 负结果}
\rightarrow
\text{公平基线与表示天花板}
\rightarrow
\text{CRISP-MR 测量审计结构路由}
\rightarrow
\text{CRISP-GS 作为可证伪实例}
\rightarrow
\text{病理/增强临床验证}
}
\]

当前最优的近期执行顺序：

1. 重建并冻结强 CS-SENSE 基线；
2. 完成 full-data Gaussian ceiling；
3. 匿名化 16 例并完成 5/11 split；
4. 实现最小三状态 cross-fitted routing；
5. 先用可控注入证明 private structure 可辨识；
6. 只有通过上述闸门后，再投入完整 split–merge–prune 和 MedIA 级病理实验。

---

# 28. 参考文献与需重点比较的工作

1. Peng T, Zha R, Li Z, Liu X, Zou Q. **Three-Dimensional MRI Reconstruction With 3D Gaussian Representations: Tackling the Undersampling Problem.** IEEE Transactions on Medical Imaging, 2026; arXiv:2502.06510.
2. Gao Y, Xu P, He Y, et al. **AtlasGS: Brain MRI Spatial Resolution Harmonization With Shared Gaussian Geometry.** arXiv:2606.02961, 2026.
3. Niessen N, Pirkl CM, Solana AB, et al. **INR Meets Multi-Contrast MRI Reconstruction.** arXiv:2509.04888, 2025.
4. Rao C, van Osch M, Pezzotti N, et al. **A Plug-and-Play Method for Guided Multi-Contrast MRI Reconstruction Based on Content/Style Modeling.** Medical Image Analysis, 113:104160, 2026.
5. Ehrhardt MJ, Betcke MM. **Multi-Contrast MRI Reconstruction With Structure-Guided Total Variation.** SIAM Journal on Imaging Sciences, 2016.
6. Kadota B, Millard C, Chiew M. **Optimized Multi-Contrast Self-Supervised MRI Reconstruction Using Learned k-space Partitioning.** arXiv:2606.19182, 2026.

> 参考文献在正式投稿前仍需核对卷期、页码、DOI 和最终发表状态。
