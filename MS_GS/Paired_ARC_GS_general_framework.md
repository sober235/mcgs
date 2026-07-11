# Paired-ARC-GS：面向显著对比差异的通用成对 3D MRI 重建框架

## 1. 框架定位

对于 T1W–FLAIR、平扫–增强这类成对 MRI，最合理的统一建模不是要求两幅图像共享强度、纹理或全部结构，而是分离三类信息：

$$
\boxed{
\text{共同解剖支撑}
+
\text{成像条件响应}
+
\text{对比特异变化}
}
$$

据此，将原 MC-GS 重构为：

> **Paired-ARC-GS：Paired Anatomy–Response–Change Gaussian Reconstruction**

它是一个 scan-specific、physics-constrained、measurement-audited 的成对 3D MRI 重建框架。

---

## 2. 统一问题定义

给定同一受试者的两个三维 MRI 对比度：

$$
y_1=A_1x_1+n_1,
\qquad
y_2=A_2x_2+n_2,
$$

其中：

$$
A_m=M_m\mathcal F_{3D}S_m,
\qquad m\in\{1,2\}.
$$

符号含义如下：

- $x_1,x_2\in\mathbb C^{H\times W\times D}$：待重建复图像；
- $M_1,M_2$：两个对比度各自的欠采样 mask；
- $S_1,S_2$：两个对比度各自的 ESPIRiT coil sensitivity maps；
- $\mathcal F_{3D}$：三维 Fourier 变换；
- $y_1,y_2$：多线圈欠采样 k-space；
- $n_1,n_2$：复噪声。

该框架适用于：

| 成对场景 | 对比度 1 | 对比度 2 |
|---|---|---|
| 常规脑多对比 | T1W | FLAIR |
| 造影成像 | 平扫 | 增强 |
| 血管壁成像 | T1W | 增强 T1W |
| 多期增强 | 动脉期 | 静脉期 |
| 多参数成像 | T1W | T2W |
| 随访成像 | 基线 | 随访 |

框架的核心假设不是 $x_1\approx x_2$，而是：

> 两个对比度共享主要空间解剖支撑，但其组织响应、病灶可见性和局部状态变化可以显著不同。

---

## 3. 原 Shared-Geometry MC-GS 的局限

原始模型为：

$$
x_1=\mathcal V(\mu,s,q,\alpha_1),
$$

$$
x_2=\mathcal V(\mu,s,q,\alpha_2).
$$

虽然共享 $\mu,s,q$，但存在以下关键问题。

### 3.1 共享 geometry 可能退化

若某个 Gaussian 满足：

$$
|\alpha_{1,n}|\approx0,
\qquad
|\alpha_{2,n}|>0,
$$

那么它实际上是对比度 2 的私有 Gaussian，只是被存放在 shared parameter table 中。

因此，共享参数容器并不等同于真正的共享解剖表示。

### 3.2 独立复密度不是“由共享模型派生”

$\alpha_1$ 和 $\alpha_2$ 是完全独立的可训练参数，并没有共同组织状态变量。因此，原模型属于：

$$
\text{共享空间字典}
+
\text{独立对比度系数},
$$

而不是真正的：

$$
\text{共享解剖模型}
\rightarrow
\text{对比度特异表示}.
$$

### 3.3 真实变化和欠采样伪影无法区分

平扫–增强中的强化、FLAIR 独有病灶均可能被：

- shared geometry 强行吸收；
- TV 平滑；
- 当作 aliasing 去除；
- 错误传播到另一个对比度。

因此，需要显式引入对比特异变化分支，并要求该分支得到独立 k-space 测量证据支持。

---

## 4. Paired-ARC-GS 的完整表示

对第 $m$ 个对比度，定义：

$$
x_m(\mathbf r)
=
e^{j\phi_m(\mathbf r)}
\left[
x_m^{\mathrm{base}}(\mathbf r)
+
x_m^{\mathrm{change}}(\mathbf r)
\right].
$$

其中：

$$
x_m^{\mathrm{base}}
=
\mathcal V
\left(
T_m(\mathcal G_A),
\alpha_m^A
\right),
$$

$$
x_m^{\mathrm{change}}
=
\mathcal V
\left(
\mathcal G_{\Delta,m},
\alpha_{\Delta,m}
\right).
$$

进一步得到统一模型：

$$
\boxed{
x_m
=
e^{j\phi_m}
\left[
\mathcal V
\left(
T_m(\mathcal G_A),
D_m(z^A)
\right)
+
\mathcal V
\left(
\mathcal G_{\Delta,m},
\alpha_{\Delta,m}
\right)
\right]
}
$$

该模型包含四个模块：

1. 共享解剖 Gaussian 模型 $\mathcal G_A$；
2. 对比度响应模型 $D_m$；
3. 对比特异变化模型 $\mathcal G_{\Delta,m}$；
4. 对比度独立的形变 $T_m$ 和相位 $\phi_m$。

---

## 5. 共享解剖 Gaussian 模型

定义共享解剖集合：

$$
\mathcal G_A
=
\left\{
\mu_n,\Sigma_n,z_n^A,\pi_n^A
\right\}_{n=1}^{N_A},
$$

其中：

- $\mu_n$：三维解剖位置；
- $\Sigma_n$：三维空间形状；
- $z_n^A$：局部组织 latent code；
- $\pi_n^A$：该 Gaussian 属于公共解剖的置信度。

协方差为：

$$
\Sigma_n
=
R(q_n)
\operatorname{diag}
\left(
s_{n,x}^2,s_{n,y}^2,s_{n,z}^2
\right)
R(q_n)^T.
$$

第 $n$ 个 Gaussian kernel 为：

$$
G_n(\mathbf r)
=
\exp
\left[
-\frac12
(\mathbf r-\mu_n)^T
\Sigma_n^{-1}
(\mathbf r-\mu_n)
\right].
$$

$\mathcal G_A$ 不直接存储 T1W、FLAIR、平扫和增强图像的绝对强度，只表示稳定存在的局部解剖支撑。

---

## 6. 条件化对比度响应模型

### 6.1 从共同组织状态派生对比度信号

对同一个 shared Gaussian，其在两个成像条件下的复密度由不同的轻量 decoder 生成：

$$
\alpha_{1,n}^{A}=D_1(z_n^A),
$$

$$
\alpha_{2,n}^{A}=D_2(z_n^A).
$$

为使模型能够在不同成对任务之间复用，可以统一写为：

$$
\alpha_{m,n}^{A}
=
D_\psi(z_n^A,e_m),
$$

其中 $e_m$ 是成像条件编码：

$$
e_m
=
[
\text{sequence type},
\text{contrast state},
\text{pre/post},
\text{echo/inversion category}
].
$$

该条件编码不要求输入精确的 TR、TE、TI，也不宣称建立完整 Bloch 定量模型。它只使用低维条件变量控制同一解剖 Gaussian 在不同成像条件下的复响应。

### 6.2 限制 decoder 容量

Decoder 应保持轻量，例如：

$$
h_n=\sigma(W_1z_n^A+b_1),
$$

$$
[\Re(\alpha_{m,n}^{A}),\Im(\alpha_{m,n}^{A})]
=
W_{2,m}h_n+b_{2,m}.
$$

推荐设置：

- anatomy latent dimension：8–16；
- hidden dimension：16–32；
- 使用两层 MLP；
- T1W/FLAIR 使用独立输出头；
- 平扫/增强使用独立输出头；
- geometry 不由 decoder 生成。

如果 decoder 容量过大，它会重新记忆每幅图像，使共享解剖失去约束作用。

---

## 7. 对比特异变化 Gaussian 模型

定义：

$$
\mathcal G_{\Delta,m}
=
\left\{
\mu_{\Delta,m,l},
\Sigma_{\Delta,m,l},
\alpha_{\Delta,m,l},
p_{\Delta,m,l}
\right\}_{l=1}^{N_{\Delta,m}}.
$$

该分支只表示不能由共同解剖和条件响应解释的局部信号。

### 7.1 T1W–FLAIR

模型表示为：

$$
x_{\mathrm{T1W}}
=
x_{\mathrm{T1W}}^A
+
x_{\mathrm{T1W}}^\Delta,
$$

$$
x_{\mathrm{FLAIR}}
=
x_{\mathrm{FLAIR}}^A
+
x_{\mathrm{FLAIR}}^\Delta.
$$

其中：

- 灰质、白质、脑室和脑沟等主要解剖由 $\mathcal G_A$ 表示；
- T1W 与 FLAIR 的全局组织强度差异由 $D_{\mathrm{T1W}}$ 和 $D_{\mathrm{FLAIR}}$ 表示；
- FLAIR 独有白质高信号主要进入 $\mathcal G_{\Delta,\mathrm{FLAIR}}$。

这样不会要求 T1W 中必须出现与 FLAIR 病灶对应的可见信号。

### 7.2 平扫–增强

模型表示为：

$$
x_{\mathrm{pre}}
=
x_{\mathrm{pre}}^A
+
x_{\mathrm{pre}}^\Delta,
$$

$$
x_{\mathrm{post}}
=
x_{\mathrm{post}}^A
+
x_{\mathrm{post}}^\Delta.
$$

增强分量主要表示为：

$$
E_{\mathrm{enh}}
=
\left|
x_{\mathrm{post}}^\Delta
\right|.
$$

但是 $x_{\mathrm{post}}^\Delta$ 不能简单等同于：

$$
x_{\mathrm{post}}-x_{\mathrm{pre}},
$$

因为该差值同时包含：

- 运动；
- 相位变化；
- 扫描增益差异；
- 噪声；
- CSM 差异；
- 真正强化。

Paired-ARC-GS 先通过形变和条件响应解释系统性差异，再由 change Gaussian 表示局部强化。

---

## 8. 显式建模跨对比几何错位

T1W–FLAIR 和平扫–增强通常无法达到严格体素级对齐，因此对每个对比度引入：

$$
T_m(\mathbf r)
=
R_m\mathbf r+t_m+u_m(\mathbf r),
$$

其中：

- $R_m,t_m$：刚性变换；
- $u_m$：低频、低自由度非刚性位移。

Gaussian 中心变换为：

$$
\mu_{m,n}=T_m(\mu_n).
$$

局部协方差近似变换为：

$$
\Sigma_{m,n}
=
J_{T_m}(\mu_n)
\Sigma_n
J_{T_m}(\mu_n)^T.
$$

形变正则为：

$$
\mathcal R_{\mathrm{def}}
=
\sum_m
\left[
\|\nabla u_m\|_2^2
+
\lambda_J
\|\det J_{T_m}-1\|_2^2
\right].
$$

必须严格限制 $u_m$ 的空间频率和幅度，否则形变模型可能吸收真实病灶和强化。

---

## 9. 相位建模

对每个对比度分别表示：

$$
x_m(\mathbf r)
=
m_m(\mathbf r)e^{j\phi_m(\mathbf r)}.
$$

相位场使用低频模型：

$$
\phi_m(\mathbf r)=P_{\omega_m}(\mathbf r),
$$

其中 $P_{\omega_m}$ 可以由三维 B-spline 或低频坐标网络实现。

相位正则采用 wrap-aware 形式：

$$
\mathcal R_{\phi}
=
\sum_m
\left\|
w_m
\odot
\nabla e^{j\phi_m}
\right\|_1.
$$

不建议直接对 angle map 使用普通 TV，因为 $-\pi$ 与 $\pi$ 附近存在相位跳变。

---

## 10. 测量证据驱动的 Change Routing

引入 private branch 后，最大的风险是它拟合欠采样 aliasing。因此，不能仅根据图像 residual 生成 change Gaussian。

### 10.1 Fit–audit k-space 划分

将每个对比度已经采集的 k-space 分为：

$$
\Omega_m
=
\Omega_m^{\mathrm{fit}}
\cup
\Omega_m^{\mathrm{audit}},
$$

且：

$$
\Omega_m^{\mathrm{fit}}
\cap
\Omega_m^{\mathrm{audit}}
=
\varnothing.
$$

- fit set 用于更新参数；
- audit set 用于判断新增结构能否预测独立测量。

### 10.2 从测量 residual 提出 change 候选

首先仅使用 anatomy–response branch 得到：

$$
x_m^A.
$$

计算 fit k-space residual：

$$
r_m^{\mathrm{fit}}
=
M_m^{\mathrm{fit}}
\left[
\mathcal F(S_mx_m^A)-y_m
\right].
$$

将其反投影至图像空间：

$$
e_m
=
\left|
A_{m,\mathrm{fit}}^H
r_m^{\mathrm{fit}}
\right|.
$$

高 evidence 区域用于提出 private Gaussian 候选，但不能直接接受。

### 10.3 使用独立测量验证候选 Gaussian

设加入候选变化 Gaussian 前后的模型分别为 $x_m$ 和 $x_m^+$。定义 audit gain：

$$
\Delta E_{m,l}
=
\left\|
A_{m,\mathrm{audit}}x_m
-y_{m,\mathrm{audit}}
\right\|_2^2
-
\left\|
A_{m,\mathrm{audit}}x_m^+
-y_{m,\mathrm{audit}}
\right\|_2^2.
$$

仅当：

$$
\Delta E_{m,l}>\tau_m
$$

时，才将候选 Gaussian 加入 $\mathcal G_{\Delta,m}$。

这相当于询问：

> 该局部 FLAIR 信号或强化信号是否能够解释一组没有参与生成它的真实 k-space 测量？

只有答案为肯定时，才将其保留为对比特异变化。

---

## 11. Gaussian–Gabor 混合表示

普通 Gaussian 的 Fourier 频谱为：

$$
\widehat G_n(\mathbf k)
\propto
\exp
\left(
-2\pi^2
\mathbf k^T\Sigma_n\mathbf k
\right)
e^{-j2\pi\mathbf k^T\mu_n}.
$$

其能量天然集中于低频。为了表达清晰边缘，原 MC-GS 往往需要大量极小 Gaussian 和较大的高频 DC 权重。

因此，将 shared representation 分为：

$$
x_m^A
=
\mathcal V_G
(\mathcal G_A,\alpha_m^G)
+
\mathcal V_B
(\mathcal B_A,\alpha_m^B),
$$

其中：

- Gaussian primitives 表示平滑组织和低中频信息；
- Gabor primitives 表示局部边缘和高频细节。

Gabor primitive 定义为：

$$
B_l(\mathbf r)
=
G_l(\mathbf r)
\exp
\left[
j2\pi
\xi_l^T
(\mathbf r-\mu_l)
\right],
$$

其中 $\xi_l$ 是局部载频。

Gabor branch 不应从训练开始全面开放。它应在 shared Gaussian 稳定后，依据 held-out k-space 高频误差生成，并严格限制 primitive 数量。

---

## 12. 最终优化目标

整体目标函数定义为：

$$
\mathcal L
=
\mathcal L_{\mathrm{DC}}
+
\lambda_\Delta\mathcal R_\Delta
+
\lambda_{\mathrm{route}}\mathcal R_{\mathrm{route}}
+
\lambda_{\mathrm{geo}}\mathcal R_{\mathrm{geo}}
+
\lambda_{\mathrm{def}}\mathcal R_{\mathrm{def}}
+
\lambda_\phi\mathcal R_\phi.
$$

### 12.1 噪声归一化数据一致性

对 coil noise covariance $\Psi_m$ 进行 pre-whitening：

$$
\tilde y_m=\Psi_m^{-1/2}y_m,
$$

$$
\tilde S_m=\Psi_m^{-1/2}S_m.
$$

数据项为：

$$
\mathcal L_{\mathrm{DC}}
=
\sum_{m=1}^{2}
\frac{
\left\|
M_m^{\mathrm{fit}}
\left[
\mathcal F_{3D}
(\tilde S_mx_m)-\tilde y_m
\right]
\right\|_2^2
}{
|\Omega_m^{\mathrm{fit}}|
}.
$$

不再使用固定的中心频率 0.3 和高频 2.2 加权，因为该方法会改变测量噪声的统计含义。

### 12.2 Change branch 稀疏约束

$$
\mathcal R_\Delta
=
\sum_{m,l}
p_{\Delta,m,l}
\|\alpha_{\Delta,m,l}\|_2.
$$

该约束限制 change branch 只表示必要的局部特异信号。

### 12.3 Shared–change 排他约束

$$
\mathcal R_{\mathrm{route}}
=
\sum_m
\int
a_m^A(\mathbf r)
a_m^\Delta(\mathbf r)
\,d\mathbf r.
$$

该约束抑制同一空间内容同时被 shared branch 和 change branch 重复解释。

### 12.4 直接 Gaussian geometry regularization

$$
\mathcal R_{\mathrm{geo}}
=
\mathcal R_{\mathrm{scale}}
+
\lambda_a\mathcal R_{\mathrm{aniso}}
+
\lambda_o\mathcal R_{\mathrm{overlap}}.
$$

尺度约束为：

$$
\mathcal R_{\mathrm{scale}}
=
\sum_n
\left[
\max(0,s_n-s_{\max})^2
+
\max(0,s_{\min}-s_n)^2
\right].
$$

这才是真正作用于 Gaussian 参数的 geometry regularization。原算法中的：

$$
TV(|x_1|)+TV(|x_2|)
$$

只能称为 image-space TV。

---

## 13. 稳定训练流程

### Stage 1：ACS 初始化

分别从两个对比度的 ACS 重建低分辨率复图像：

$$
x_m^{\mathrm{LR}}
=
A_{m,\mathrm{ACS}}^\dagger
y_{m,\mathrm{ACS}}.
$$

计算经过对比度归一化的三维结构图：

$$
E_m
=
\frac{
\|\nabla|x_m^{\mathrm{LR}}|\|
}{
\operatorname{median}
(\|\nabla|x_m^{\mathrm{LR}}|\|)+\epsilon
}.
$$

构造联合结构：

$$
E_A
=
\sqrt{E_1^2+E_2^2}.
$$

根据 $E_A$ 初始化 shared Gaussian 的位置、尺度和方向。

### Stage 2：固定 geometry，拟合对比度响应

固定：

$$
\mu_n,\Sigma_n,
$$

仅优化：

$$
z_n^A,
\quad D_1,
\quad D_2,
\quad \phi_1,
\quad \phi_2.
$$

该阶段避免尚未稳定的图像 residual 直接扭曲 shared geometry。

### Stage 3：估计低自由度跨扫描形变

基于低分辨率 anatomy branch 优化：

$$
T_1,T_2.
$$

此阶段保持 change branch 关闭，避免将病灶差异用于配准。

### Stage 4：低学习率优化 shared geometry

开放 $\mu,\Sigma$，但满足：

$$
\eta_{\mu,\Sigma}
\ll
\eta_{z,\alpha,\phi}.
$$

建议 geometry learning rate 为 signal parameter learning rate 的 $1/10$ 至 $1/50$。

### Stage 5：启动 measurement-audited change birth

使用 fit residual 提出 change Gaussian，使用 audit k-space gain 判断是否接受。

### Stage 6：启动高频 detail primitives

仅在 audit 高频 residual 持续存在的区域加入少量 Gabor primitive。

### Stage 7：联合优化和 pruning

删除：

- 长期低密度 Gaussian；
- audit gain 不显著的 private Gaussian；
- 高度重叠且响应近似相同的 Gaussian；
- 只降低 fit loss、不改善 audit loss 的 Gaussian。

### Stage 8：稳定性选择

重复 $B$ 次 fit–audit 划分，计算每个 change 区域的选择概率：

$$
P_{\mathrm{stable}}(\mathbf r)
=
\frac1B
\sum_{b=1}^{B}
\mathbf 1
\left[
\mathbf r
\text{ 在第 }b\text{ 次被选择}
\right].
$$

最终仅保留：

$$
P_{\mathrm{stable}}(\mathbf r)
>
\tau_{\mathrm{stable}}
$$

的变化成分。

### Stage 9：完整测量微调

模型 topology 确定后，将 audit k-space 放回数据项。冻结 Gaussian birth、death 和大幅 geometry 更新，仅优化 signal parameters。

### Stage 10：CG 数据一致性投影

对每个对比度求解：

$$
x_m^*
=
\arg\min_x
\frac12\|x-\tilde x_m\|_2^2
+
\frac{\eta_m}{2}
\|A_mx-y_m\|_2^2.
$$

对应正规方程：

$$
(I+\eta_mA_m^HA_m)x_m^*
=
\tilde x_m
+
\eta_mA_m^Hy_m.
$$

通过 conjugate gradient 求解，替代原算法的 per-coil k-space replacement。

---

## 14. 完整算法伪代码

```text
Algorithm: Paired-ARC-GS

Input:
    Paired undersampled multi-coil k-space y1, y2
    Sampling masks M1, M2
    ACS regions C1, C2
    Coil-noise covariance Psi1, Psi2

1. Pre-whiten k-space and coils
2. Estimate ESPIRiT maps S1, S2 from corresponding ACS
3. Reconstruct low-resolution complex volumes x1_LR, x2_LR
4. Estimate initial rigid alignment T1, T2
5. Construct contrast-normalized joint structural map
6. Initialize shared anatomy Gaussian set G_A
7. Initialize anatomy codes z_A and contrast decoders D1, D2
8. Initialize smooth phase fields phi1, phi2

9. Split measured k-space:
       Omega_m = Omega_fit_m union Omega_audit_m

10. Freeze Gaussian geometry
11. Optimize z_A, D1, D2, phi1, phi2 using fit k-space

12. Estimate low-frequency transforms T1, T2
13. Unfreeze shared geometry with a small learning rate
14. Optimize shared anatomy-response reconstruction

15. Compute fit-k-space residuals
16. Backproject residuals to propose private/change Gaussians

17. For every candidate change Gaussian:
       insert candidate temporarily
       calculate audit-k-space error reduction
       retain candidate only when audit gain > threshold

18. Add audit-supported Gabor primitives for persistent
    high-frequency residuals

19. Jointly optimize:
       shared anatomy
       contrast responses
       accepted change Gaussians
       phase and deformation

20. Prune unsupported and redundant primitives
21. Repeat fit/audit splits for stability selection
22. Restore all measured k-space and refine signal parameters
23. Apply conjugate-gradient data-consistency projection

Output:
    Reconstructed complex volumes x1*, x2*
    Shared anatomy model G_A
    Contrast-specific change maps Delta1, Delta2
    Measurement-support confidence maps P1, P2
```

---

## 15. 两类主要任务中的输出

### 15.1 T1W–FLAIR

最终输出：

$$
\hat x_{\mathrm{T1W}},
\qquad
\hat x_{\mathrm{FLAIR}},
$$

以及：

$$
\Delta_{\mathrm{FLAIR}}
=
\left|
\mathcal V
(\mathcal G_{\Delta,\mathrm{FLAIR}})
\right|.
$$

$\Delta_{\mathrm{FLAIR}}$ 不自动等同于病灶分割，而是表示不能由共享解剖和正常 FLAIR 响应解释、且受到独立测量支持的 FLAIR 特异信号。

### 15.2 平扫–增强

最终输出：

$$
\hat x_{\mathrm{pre}},
\qquad
\hat x_{\mathrm{post}},
$$

以及测量支持的增强图：

$$
E_{\mathrm{enh}}(\mathbf r)
=
\left|
\mathcal V
(\mathcal G_{\Delta,\mathrm{post}})
\right|
P_{\mathrm{stable}}(\mathbf r).
$$

这比直接计算：

$$
|\hat x_{\mathrm{post}}|
-
|\hat x_{\mathrm{pre}}|
$$

更稳健，因为框架已经显式处理配准、条件响应、相位和测量支持。

---

## 16. 如何证明框架具有通用性

必须在同一框架、同一组主要超参数和同一训练机制下同时验证：

| 数据场景 | 必须回答的问题 |
|---|---|
| T1W–FLAIR | 是否减少混叠且保留 FLAIR-only 病灶 |
| 平扫–增强 | 是否减少混叠且保留真实强化 |
| 轻度错位 | 是否将运动与真实变化正确区分 |
| 相同 mask | 是否仍能利用解剖共享 |
| 互补 mask | 是否进一步利用频率互补性 |
| 不同加速率 | shared branch 是否帮助更严重欠采样对比度 |
| 不同 scanner | scan-specific 机制是否保持稳定 |

最关键的验证目标不是单独比较 PSNR，而是：

$$
\text{去伪影能力}
+
\text{对比特异信号保持}
+
\text{跨对比泄漏控制}.
$$

---

## 17. 核心消融实验

| 模型 | Shared anatomy | Response decoder | Change Gaussian | Audit routing |
|---|---:|---:|---:|---:|
| Independent GS | 否 | 否 | 否 | 否 |
| 原 MC-GS | 是 | 否 | 否 | 否 |
| Paired-ARC-GS w/o change | 是 | 是 | 否 | 否 |
| Paired-ARC-GS w/o audit | 是 | 是 | 是 | 否 |
| Paired-ARC-GS | 是 | 是 | 是 | 是 |

预期应证明：

1. 原 MC-GS 相对 independent GS 减少部分伪影；
2. response decoder 减少两个独立 density tables 的退化；
3. change branch 恢复 FLAIR 病灶和增强信号；
4. audit routing 防止 change branch 拟合 aliasing；
5. 完整模型取得最好的 artifact–lesion trade-off。

---

## 18. 评价指标

### 18.1 全局重建

- PSNR；
- SSIM；
- NMSE；
- 3D HFEN。

### 18.2 去伪影

- aliasing residual energy；
- flat-region grain；
- background high-pass variance；
- held-out k-space prediction error。

### 18.3 T1W–FLAIR

- FLAIR lesion contrast recovery；
- lesion volume error；
- small-lesion sensitivity；
- FLAIR 病灶向 T1W 的错误泄漏；
- T1W 结构向 FLAIR 的错误复制。

### 18.4 平扫–增强

- enhancement contrast-to-noise ratio；
- enhancing volume recovery；
- small-enhancement sensitivity；
- enhancement hallucination rate；
- post-to-pre enhancement leakage。

### 18.5 表示机制

- shared Gaussian utilization；
- private Gaussian fraction；
- fit–audit generalization gap；
- change selection stability；
- Gaussian 数量与性能关系。

---

## 19. 对框架的最终审核

### 19.1 设计成立的部分

1. 共享解剖而不共享绝对强度，符合显著多对比差异场景；
2. 条件响应可以表达 T1W–FLAIR 的系统性组织对比差异；
3. change Gaussian 可以表示 FLAIR-only 病灶和真实强化；
4. 显式形变避免把扫描错位误认为病灶；
5. 独立相位模型保持复数 MRI forward consistency；
6. fit–audit 测量审计可以抑制 private branch 拟合 aliasing；
7. Gaussian–Gabor 混合表示可以减少对大量微小 Gaussian 的依赖。

### 19.2 仍需警惕的限制

#### 19.2.1 分解不是天然可辨识

$$
x_m=x_m^A+x_m^\Delta
$$

本身存在无限种分解。可辨识性依赖低容量 response decoder、稀疏 change branch、测量审计和分阶段优化共同提供，不能只依靠网络结构命名。

#### 19.2.2 Audit k-space 不能提供绝对真实性证明

Audit improvement 只能说明某个成分得到未参与生成的测量支持，不能完全排除结构化噪声和模型偏差。因此还需要多划分稳定性和病灶级验证。

#### 19.2.3 形变模型不能太强

高容量形变可能吸收强化和病变变化。应仅使用低频、平滑、小幅度变形，并从低分辨率共有结构估计。

#### 19.2.4 Private branch 必须显著小于 shared branch

若 $N_{\Delta,m}$ 接近 $N_A$，模型会重新退化为两个独立 GS。应通过 audit-supported birth 控制 private primitive 数量，而不是预先分配大容量 branch。

---

## 20. 最终定位

Paired-ARC-GS 最准确的定位是：

> 一种面向显著对比差异成对 3D MRI 的测量约束、scan-specific Gaussian 重建框架。它将共同解剖、条件化组织响应、局部真实变化和扫描不一致显式分离，在利用跨对比冗余去除混叠的同时，通过独立 k-space 审计保护 FLAIR 病灶和造影强化。

真正的核心创新不是简单的 shared Gaussian，而是：

$$
\boxed{
\text{测量支持的 shared–response–change Gaussian routing}
}
$$

它决定：

1. 哪些信息可以跨对比共享；
2. 哪些信息必须保留在单一对比度中；
3. 哪些局部结构只是欠采样伪影。

这一机制使同一个框架能够同时适用于 T1W–FLAIR 和平扫–增强，并将“减少混叠伪影”和“保持真实对比特异病灶”纳入同一个可审计的重建过程。
