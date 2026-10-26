# 新路线调研与来源，2026-10-19

本轮只做文献、源码与历史实验核对；没有训练、安装依赖或修改正式方法。目标是在现有 AF16、同一 mask、同一平扫/增强数据、同一十个对手下提高 PSNR、SSIM、NMSE。外部大数据预训练先验排除。

## 当前证据与检索增量

复用：25 次正式实验、65 个历史重建的三指标重评；既有正则/优化器/非局部先验失败史；共享几何与独立复系数的源码审计；完整 ZS-SSDU 配置核对。旧 HFEN 决策不再作为当前质量目标。

新增检索：2025–2026 Gaussian MRI、隐式 Gaussian 幅值、Gabor 频率调制、同病例自校准先验；补查 2024 重定位与更早的结构化 k-space 先验。只依赖论文原文与作者项目/仓库作技术依据。搜索结果中的媒体摘要与第三方复刻未作为结论依据。当前工具不提供 quest memory/artifact 接口，以上本地记录充当项目记忆，未假称调用那些接口。

## 与本项目最相关的来源

| 来源 | 已核对的机制与适用范围 | 对当前 MC-GS 的意义和限制 |
|---|---|---|
| [1] MIGA，2026-09-21，预印本 | 共享 Gaussian 几何，坐标编码网络生成各回波幅值，相位独立；只用采集 k-space。方法及消融见 §§II-C/E、III-C、IV-C。 | 最接近当前薄弱环节：独立系数缺少空间与跨通道的幅值结构。研究的是多回波，不能直接保证平扫/增强迁移；部分 MTP 消融中独立网络 SSIM 更好。论文没有给出已核验代码入口。 |
| [2] Gabor primitives，2026-03 预印本；MICCAI 2026 有官方代码 | 对 Gaussian 包络施加复指数调制，使基元的主频可移动；原实验是二维动态心脏、多线圈模型。 | 可研究细节表示效率；不是证明当前 Gaussian 没有表达能力。需三维复数渲染/梯度扩展，不能直接套用其时间低秩模型。官方代码已发布，不能沿用三月预印本的“待开源”状态。 |
| [3] DCE primitive representation，2026-08，预印本 | 区分基础解剖、运动和对比剂动态信号，采用独立时间基。 | 支持保护增强特异变化这一设计原则。当前仅两个对比，不能照搬多时间点低秩分解，更不能把增强压成平扫的固定比例。 |
| [4] 3DGSMR，2025，预印本原文 | 已在 MRI 中研究长轴分裂与 coarse-to-fine Gaussian 密度控制。 | 分裂和增点本身不是新颖贡献；现有源码已有相关函数。需要与固定预算重定位区分。 |
| [5] 3DGS-MCMC，NeurIPS 2024 | 场景表示中的随机更新与重定位，关注初始化和有限基元利用。 | 可借鉴重定位，而非照搬 RGB opacity 概率公式；本项目是有符号复系数线性叠加。不能未经推导称新方案为正确的 MCMC 后验采样。 |
| [6] ReSiDe，2025 期刊版 | ReSiDe-S 从当前重建的 patch 自训练去噪器并与物理模型交替；不需要外部训练图像。 | 一个可用的内部学习备选，不等于已失败的固定 BM4D/patch-low-rank。其脑部实验主要是 R4，且每轮训练成本高；不能挪用其效果数字预测当前 R16。 |
| [7] HICU，2021 | 多线圈 k-space Hankel 结构低秩与矩阵自由运算。 | 与历史图像 patch 低秩不同，但当前已有 L1-SPIRiT 对照，且三维 42 线圈代价大，暂列后备。带外部训练 DNN 的变体不符合本轮约束。 |
| [8] ZS-SSL，ICLR 2022 | 从当前采集内分出训练、损失与验证测量；无需外部预训练。 | 当前最强对照的关键范式参照；不能把随机遮掉固定 Gaussian 拟合的残差点，直接当作等价的 SSDU 网络训练。 |

## 已确认的反例与迁移风险

MIGA 表 III 中，双回波 CMRA AF16 的 joint explicit 与 joint implicit PSNR 分别为 40.89 和 41.77 dB，说明幅值参数化可能有价值。该差值是异数据的论文消融，不是本项目的预期收益或承诺。同表六回波 MTP AF16 的 independent implicit SSIM 高于 joint implicit，直接反驳“共享越强越好”的泛化。

FastGaborMRI 原仓针对二维 cine，README 的默认环境是 Torch2.6/CUDA12.4，并包含首次运行的 JIT 构建。本项目使用 GSTMI Torch2.7.1/cu118 与严格冻结扩展：只能参考算法/内核，不能直接执行其 demo、下载其他数据或替换当前环境。

ReSiDe 的残差噪声校准需要正确的噪声与算子定义。本项目已知的是合成 k-space 噪声；不能把经 SENSE/欠采样后相关的图像误差直接当作独立白噪声，也不能把噪声期望残差当作严格的优化损失下界。

## 检索账本

1. `MRI Gaussian reconstruction 2025/2026`：补齐直接 MRI 表示文献，找到 [1][2][4]。
2. `Gaussian MRI densification`：确认长轴分裂已有先例，并回查本地已实现路径。
3. `3D Gaussian Splatting as Markov Chain Monte Carlo`：核对作者项目及 NeurIPS 正文，确定仅转移重定位原则。
4. `MRI scan-specific denoiser/ReSiDe`：找到 [6] 原文与作者仓库，区分单病例版本和多数据集训练版本。
5. `Gabor MRI code`：找到 MICCAI 2026 论文页及刚发布的官方仓库，更新旧预印本中的代码状态。
6. `MIGA 2609.24468 github`：未找到可核验作者实现；不据此声称实现不存在。
7. `HICU MRI structured low rank`：确认可作为信号处理后备，但不优先复制传统基线。

未解决：MIGA 的具体优化率/训练预算未完整公开；Gabor 三维内核对当前冻结环境的兼容性未测试；所有迁移路线在本病例上的实际收益均未知。

## 参考文献与实现入口

[1] J. Xu, Y. Liu, Y. Zhu. *MIGA: Shared-Geometry Gaussian Representation with Implicit Amplitude Modeling for Accelerated 3D Multi-Echo MRI*. arXiv:2609.24468v1, 2026. https://arxiv.org/html/2609.24468v1

[2] W. Huang et al. *Gabor Primitives for Accelerated Cardiac Cine MRI Reconstruction*. MICCAI, 2026; arXiv:2603.05681. https://arxiv.org/html/2603.05681v1 ; accepted paper/code link: https://papers.miccai.org/miccai-2026/0413-Paper0972.html ; author code: https://github.com/wenqihuang/FastGaborMRI

[3] V. Spieker et al. *Primitive Representation Learning for Unsupervised Dynamic Contrast Enhanced MRI Reconstruction*. arXiv:2608.18055v1, 2026. https://arxiv.org/html/2608.18055v1

[4] T. Peng et al. *Three-Dimensional MRI Reconstruction with Gaussian Representations: Tackling the Undersampling Problem*. arXiv:2502.06510v1, 2025. https://arxiv.org/html/2502.06510v1

[5] S. Kheradmand et al. *3D Gaussian Splatting as Markov Chain Monte Carlo*. NeurIPS, 2024. https://ubc-vision.github.io/3dgs-mcmc/ ; https://proceedings.neurips.cc/paper_files/paper/2024/file/93be245fce00a9bb2333c17ceae4b732-Paper-Conference.pdf

[6] S. Liu et al. *MRI recovery with self-calibrated denoisers without fully-sampled data*. MAGMA 38, 53–66, 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC11790797/ ; author code: https://github.com/OSU-MR/reside

[7] S. Zhao, L. C. Potter, R. Ahmad. *High-dimensional Fast Convolutional Framework (HICU) for Calibrationless MRI*. Magnetic Resonance in Medicine, 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC8184615/ ; author code: https://github.com/OSU-CMR/HICU

[8] B. Yaman, S. A. H. Hosseini, M. Akcakaya. *Zero-Shot Self-Supervised Learning for MRI Reconstruction*. ICLR, 2022. https://arxiv.org/abs/2102.07737 ; author code: https://github.com/byaman14/ZS-SSL
