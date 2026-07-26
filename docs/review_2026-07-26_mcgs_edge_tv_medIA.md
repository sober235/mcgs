# Review — MC-GS edge-TV 作为 Medical Image Analysis 投稿的可行性评审

**日期:** 2026-07-26
**评审对象:** `docs/mcgs_edge_tv_method.md`(MC-GS edge-TV 完整方法与参数说明)
**评审问题(用户提出):** ① novelty 是否适合 Medical Image Analysis ② 是否可行 ③ 是否达到最优重建质量
**产生方式:** 按 `superpowers:requesting-code-review`,派发 **3 名互相独立的评审 subagent**(novelty / 可行性 / 重建质量),每人只拿到裁剪过的上下文,**不给会话历史**,并被明确要求把 `memory.md` 与 `.claude/memory.md` 视为"作者自述、未经核实",不得引用其结论。三份报告返回后,由我对其中最致命的断言逐条独立复核。

> **重要:本文件独立于项目既有记忆写成。** 用户明确要求"撇开记忆",故本评审不采信 `memory.md` / `.claude/memory.md` 中任何未经重新验证的结论。凡与既有记忆冲突之处,以本文件为准。

## 核验图例

| 标记 | 含义 |
|---|---|
| ✅ | **我亲手复核过**(重跑脚本 / 读代码 / 抓取原始文献页面),数字为我自己算出 |
| 🟡 | 评审员的断言,**我未独立核实**,进论文前必须自查 |
| 📄 | 直接引用仓库内文件原文 |

---

## 0. 总判决

| 用户的问题 | 判决 | 一句话理由 |
|---|---|---|
| ① novelty 适合 MedIA? | **否** | 核心表示已作为 3DGSMR 发表于 IEEE TMI ✅;两个"增量"各有前作;剩下的只有最显然的交集一格 |
| ② 可行? | **目前不可行,但补救便宜** | 真实 N=1 受试者、零留出集、超参在报告集上选 ✅;但 `/data1` 上已有 **16 例配对 T1+FLAIR** ✅,约 120 GPU-h 可补齐 |
| ③ 最优重建质量? | **否,且被本项目自己的数据反驳** | 调参后的**单对比度** CS-SENSE 在 FLAIR 上六项指标全胜、T1 上打平 ✅;且 edge-TV 是自家两个变体中较差的那个 ✅ |

---

## 1. Novelty(问题 ①)

### 1.1 决定性事实 ✅

**3DGSMR** — Peng, Zha, Li, Liu, Zou,*Three-Dimensional MRI Reconstruction with 3D Gaussian Representations: Tackling the Undersampling Problem*,arXiv:2502.06510(v1 2025-02,最新 2026-03),**已刊于 IEEE Transactions on Medical Imaging**(IEEE Xplore 文档号 11289574)。

我抓取了 arXiv 摘要页与 IEEE Xplore 检索结果,确认其自述包含:**用 3D 高斯分布作为 MR 体积的显式表示** + **复数值** MR 信号 + 从**欠采样 k 空间**重建 + **自监督框架、无需训练数据集**。

这与本方法 §2 / §4 的表示层逐条重合。换言之:**我们的 iso-TV 版本在方法层面约等于一篇已发表 TMI 论文的复现(且关闭了它引入的 densification)**,而 edge-TV 版本是在其上加一个 1990 年的正则项。

### 1.2 前作对照表

| 我们主张的贡献 | 最接近的前作 | 核验状态 |
|---|---|---|
| 3D 高斯显式表示 + 复数逐基元幅值 + 多线圈 SENSE k 空间 DC + TV(=iso-TV 版本全部) | **3DGSMR, IEEE TMI**(arXiv:2502.06510) | ✅ **完全被占** |
| 可微分 CUDA 体素化光栅器、3σ 截断 | R²-Gaussian, NeurIPS 2024(同组) | 🟡 |
| 共享几何 + 每对比度独立幅值 | AtlasGS(arXiv:2606.02961,脑 MRI,T1 学几何 → T2/FLAIR/DWI 各自 appearance);多光谱 GS 的 "Joint geometry" 策略(arXiv:2509.00989) | 🟡 |
| 联合多对比度先验(共享解剖) | Ehrhardt & Betcke SGTV(SIIMS 9(3):1084-1106, 2016);Chatnuntawech VTGV(MRI 34(8), 2016);Polak jVN(MRM 84(3), 2020);Bilgic(MRM 2011) | 🟡 |
| edge-weighted TV | Perona & Malik(TPAMI 12(7):629-639, 1990)的第二扩散系数 —— `paired_gs/losses/edge_tv.py:12` 即 `1/(1+(g/κ)²)` 原式 | ✅ 代码即原式 |
| 逐受试者、免训练的连续表示重建 | NeRP(TNNLS 2022);IMJENSE(TMI 2023) | 🟡 |
| 联合多对比度、逐受试者、从多线圈欠采样 k 空间、R=12 | Niessen et al., MICCAI 2025(arXiv:2509.04888),R=4/8/12,胜过 PICS | 🟡 |
| 频率自适应 k 空间损失加权 | 同上(其 `W=√(k_y²+k_z²)+1`) | 🟡 |

### 1.3 剩下的空白

唯一未被明确占据的是**交集**:*共享几何 + 双复数幅值的高斯,直接拟合**联合欠采样**的多线圈 k 空间*。评审 1 的判断(我同意):这是上表第 1 行与第 7 行的显然交叉,作为 MedIA 的方法学贡献不够。

**敌意评审会问的三句话**(需预先备好答案):
1. edge-weighted TV 是不是 Perona-Malik 换皮?—— 更糟:因为权重 detach 且 κ 在第 0 步冻结(`edge_tv.py:28-41`),它是**单步 IRLS / lagged-diffusivity**,严格弱于 1990 年代已定论的迭代重加权版本。
2. 共享几何+双幅值是不是就是 joint sparsity?—— 代码层面确实:除"完全相同的支撑集"外,两个幅值集之间**没有任何耦合项**(`joint_gaussian_model.py:49-62` ✅ 我读过)。需要论证"硬共享支撑"为何优于 SGTV 式的软梯度耦合,并做实验。
3. 逐受试者高斯拟合是不是比训练好的展开网络又慢又差?—— 唯一站得住的反驳是"无训练分布 ⇒ 无分布漂移、无幻觉",但这需要 OOD/病理实验来证明,目前只是断言。

### 1.4 投稿去向建议

- **现状(n=1,且主打变体输给自家消融)**:不具备任何同行评审期刊的投稿条件。
- **最现实的即期产出**:ISMRM abstract,或 MICCAI workshop(RIME —— Niessen 那篇可比工作就发在那里)。
- **补足多受试者验证 + 真实基线后**:MRM 或 IEEE TMI(注意 TMI 已发过 3DGSMR,delta 必须锋利且正面 benchmark)。
- **MedIA**:只有在下面 §4 的重构定位 + 病理/临床终点实验完成后才谈得上。

---

## 2. 可行性与方法学严谨性(问题 ②)

### 2.1 真实的 N ✅

| 维度 | 真实值 | 证据 |
|---|---|---|
| 受试者 | **1**(healthy volunteer) | 全部配置与 62 次运行快照都指向 `…huangyuwei_20230705-…_{T1,FLAIR}_trainer.mat` |
| 对比度 | 2(T1 + FLAIR) | — |
| mask realization | 1 | 仅 `mask3d_vdpois_af12_transverse_ALIGNED.mat`(实测采样率 0.084517,net AF 11.832 ✅) |
| 加速因子 | 1(AF≈12) | 所有 config `acceleration_factor: 12` |
| seed | 1(42,硬编码,无 CLI 开关) | `arc/cycle.py:213` |
| 留出集 / 测试集 | **0** | 62 个 `*_score.json` 全在同一体积上 |
| 前瞻采样 | **0**(100% 回顾式) | `dataset.py:332-350`, `:754` |
| 病理 | **0** | 📄 `docs/validation_ablation17.md:37`「Not testable on this **healthy** pair」 |

### 2.2 Critical 级问题

**C1 — 预注册判据被事后替换 ✅(最致命)**

📄 `docs/superpowers/specs/2026-07-23-edge-preserving-tv-csf-design.md:39-45` 预先写死了三项**合取**判据:

> - edge-region `NMSE`(或 `HFEN`)**down by > run-to-run noise floor**;AND
> - whole-brain `SSIM` **not down**;AND
> - whole-brain `NMSE` **not up**。
> 「edge NMSE down but SSIM down」= **RESHAPE = FAIL**

📄 同文件 `:119` 的决策门:「reshape-only → **documented negative result**,then consider Approach B or stop」。

实测(`results_arc/edge_scores.tsv` ✅,edgeTV_best vs baseline_4k):

| | T1 nmse_edge | T1 hfen_edge | T1 nmse_whole | FLAIR nmse_edge | FLAIR hfen_edge | FLAIR nmse_whole |
|---|---|---|---|---|---|---|
| baseline_4k (iso-TV) | 0.01417 | 0.12148 | 0.01323 | 0.00584 | 0.10482 | 0.00565 |
| **edgeTV_best** | **0.01503 ↑6.1%** | **0.12700 ↑4.5%** | **0.01365 ↑** | **0.00628 ↑7.5%** | **0.10891 ↑3.9%** | **0.00593 ↑** |

三项判据败其二(第一项反向、第三项反向),两个对比度都是。项目自己的 commit `cbe5a47` 记录为 “P2 4000-iter verdict = NO-GAIN/FAIL (reshape)” ✅,配置文件头 📄 `configs/winner_v2_edgeTV.yaml:2` 也写着「NOT the shipped baseline; edge_tv negative result」。而 `mcgs_edge_tv_method.md` §8 用事后选定的 SSIM 判据替换了它。

**有预注册 + 事后换判据**,是评审中最难辩护的组合。要么把 edge-TV 降级为负结果、以 iso-TV 为方法;要么明说判据是事后选的,并用**盲法多读者研究**支撑感知主张(而那需要不止一个受试者)。

**C2 — 被裁决的差异小于流程自身的扰动 ✅**

- 真·重复运行噪声地板(`p0_noise_4k` vs `baseline_4k`,配置等价):nmse_edge **完全相同**(0.01417 / 0.00584),ΔSSIM 仅 0.00019(T1)/ 0.00032(FLAIR)。
- 但这只是**逐位重跑**的地板(warm start 使初始化确定性)。🟡 评审 2 指出:三个 warm-start 变体(同为 iso-TV,3000 iters)之间 PSNR 跨度 **0.077 dB**、SSIM 跨度 0.0035–0.0058。
- 而 edge-TV vs iso-TV 的差是 ΔPSNRb −0.135/−0.210 dB、ΔSSIMb −0.0007/−0.0022。**PSNR 差落在初始化扰动之内**。
- 结论:本研究现有的方差估计**不足以裁决 0.05–0.2 dB 级别的差异**,而这正是 edge-TV/iso-TV、DC-centring、vTV 三个决定的量级。

**C3 — 超参在报告集上选(选择偏差)✅**

κ_q × λ 的 9 格扫描、λ_geo / kspace_high 的历史扫描,全部在**同一体积、同一 mask、同一指标**上进行,而该指标即 §8 报告的最终结果。有效自由参数约 9 个、运行约 62 次,报告值实为 ~62 个相关抽样的最大值。仓库中不存在任何无偏性能估计。

**C4 — GT 与前向模型共用同一套 CSM ✅(循环性)**

`dataset.py` 中 `ground_truth_image = Σ_c conj(csm_gt)·ifft3c(y_full)`,而 `sensitivity_maps` 就是同一个 `csm_gt`。SoS≡1 时 `Sᴴ S = I`,GT 因此**按构造**就是满数据前向模型的精确极小点;真实线圈信号中落在 `range(S)` 之外的成分,对**目标与模型同时不可见**。
直接后果:§9 中「2-map ESPIRiT 零收益」是**循环论证**——2-map 重建被 1-map 导出的 GT 打分,必输。该结论需在各自匹配的 GT 下重做。

**C5 — 唯一的竞争基线没有代码 ✅**

CS-SENSE(TV)在仓库内**不存在实现、不存在参数、不存在 score 文件**;它只以 PNG(`results_arc/three_way_cssense_mcgs_edgeTV.png` 等)和 `.claude/memory.md` 中的一行文字存在,重建代码位于会话临时目录。外部评审无法检查基线是否被调过参 —— 而 §3 将证明它**没有**。

**C6 — 数据伦理与去标识 ✅**

数据文件名内含**受试者姓名拼音**(`Train_3dBrain_huangyuwei_…`,`/data1/data1_congcong/data/3D/{T1,FLAIR}_csm_ksp/` 下另有 15 例同样命名);所有 config 硬编码私有绝对路径;无伦理/IRB 声明。`paired_gs/data/private_paths.py` 已实现 `S01_plain.mat` 式的匿名 token 接缝,但**没有任何 config 在用**。MedIA 要求伦理声明;带姓名的原始数据无法发布。

### 2.3 好消息:补救成本远低于预期 ✅

`/data1/data1_congcong/data/3D/` 下 `T1_csm_ksp`(29 个文件)与 `FLAIR_csm_ksp`(16 个文件)取交集 = **16 例配对 T1+FLAIR**(含当前这例),`.h5` 格式,`paired_gs/data/dataset.py` 本来就能读:

```
jiangdian(×2) tangbin/tangbing chenshuaibao huangyuwei jiangyizhou liaozhenghan
lishengbo litianyu shiyixuan sunxinyi yangkun zhangxiqian zhangyangming zhoushuo
```

即**跨受试者验证不需要新采数据、不需要新伦理审批**。

### 2.4 最小可投稿实验计划(优先级序)

> **前置决定:先解决 C1。** 在 edge-TV 配置上花 120 GPU-h,等于为一个未通过自身预注册判据的变体做验证。建议在 iso-TV(`reg_aligned_4k`)上跑。

| # | 实验 | 解决 | 成本 | 需新数据? |
|---|---|---|---|---|
| P0 | 把调参后的 CS-SENSE 做成仓库内可复现基线(代码 + JSON + λ 扫描,且**最优必须落在网格内部**) | C5、§3 全部 | 数小时 | 否 |
| P1 | 冻结配置,在其余 15 例上跑 AF12,报告逐例指标 + Wilcoxon | C3(提供留出集)、n=1 | 🟡 40–60 GPU-h(含每例 CSM 标定与 warm start;冷启动分支**零运行记录**,有风险) | 否 |
| P2 | 5 例调参 / 11 例报告的分折,重选超参后冻结 | C3 选择偏差 | ~15 GPU-h | 否 |
| P3 | 方差协议:每例 3 个 mask realization × 3 个初始化,报告 mean±SD | C2 | 🟡 ~48 GPU-h | 否(mask 需做同款 DC 对齐) |
| P4 | AF 扫描 {6,8,10,12,16} | 把单点变成工作曲线 | 🟡 ~80 GPU-h + mask 准备 | 否 |
| P5 | 在各自匹配 GT 下重做 1-map vs 2-map;逐条复核 §9 消融确实改变了参数 | C4 | ~10 GPU-h | 否 |
| P6 | 匿名化 + 伦理声明,全面切到 `private_paths` token | C6 | ~1 天 | 否 |
| P7 | 病理队列(MS / 肿瘤,FLAIR+T1 原始 k 空间) | 临床相关性、共享几何是否压制对比度特异病灶 | 新采集 + IRB,数月 | **是** |

**最小可投稿集 = P0+P1+P2+P3+P6 ≈ 120 GPU-h + 约两周工程**,全部基于已在盘数据。

---

## 3. 重建质量是否最优(问题 ③)

### 3.1 决定性结果 ✅(我自己重算,非转述)

把上游会话扫参得到的 **调参后 CS-SENSE(TV, λ=0.0015)** 与我们的方法放在同一把尺子上——同一 mask、同一 ESPIRiT CSM、同一 GT,并对基线施加**同样的 hard-DC 投影**(`dc` 数组):

| | PSNR_whole | SSIM_whole | **PSNR_brain** | **SSIM_brain** | **NMSE_brain** |
|---|---|---|---|---|---|
| T1 MC-GS **edge-TV(本方法)** | 38.464 | 0.8980 | 33.896 | 0.9104 | 0.01365 |
| T1 MC-GS iso-TV | 38.512 | 0.8936 | **34.031** | **0.9111** | **0.01323** |
| T1 **CS-SENSE λ=0.0015 +DC** | **38.516** | **0.9195** | 33.641 | 0.9099 | 0.01448 |
| FLAIR MC-GS **edge-TV(本方法)** | 38.184 | 0.9093 | 33.315 | 0.9272 | 0.00593 |
| FLAIR MC-GS iso-TV | 38.291 | 0.9017 | 33.525 | 0.9294 | 0.00565 |
| FLAIR **CS-SENSE λ=0.0015 +DC** | **38.775** | **0.9424** | **33.689** | **0.9348** | **0.00544** |

- **FLAIR:一个单对比度、体素网格、普通 TV 的凸重建,在全部六项指标上击败 577k 高斯的联合模型。**
- **T1:** MC-GS(iso)保住 +0.39 dB 脑内 PSNR 与更优 NMSE,脑内 SSIM 基本打平(0.9111 vs 0.9099),但整体 SSIM 输 0.026。
- 🟡 评审 3 另报告 `e_null`(未采样 k 误差)同样是 CS-SENSE 更低(0.1913 vs 0.1958;0.1198 vs 0.1278)。我未复核该项。

**根因**:旧 CS-SENSE 基线的 λ 只扫了 3 个点 `{0.003, 0.01, 0.03}`,且**最优落在网格边界**——基线未调参的典型特征。真实最优在 λ≈0.0012–0.0015,小 5–8 倍。因此 `.claude/memory.md` 中「every MC-GS variant ≫ CS-SENSE」以及由此推出的「自包含手段已穷尽,只剩学习先验」这条主线,**建立在一个欠调的基线之上**,必须撤回重做。

> ⚠️ 我复核的是**已保存的输出体积**的重新打分,未重跑 FISTA 本身。其算子(fft3c / CSM / mask / P99.9 / hard-DC)与 `arc/cycle.py` 逐行一致,但正式发表前应重跑一次。

### 3.2 edge-TV 是自家两个变体中较差的那个 ✅

SSIM 拆解(`mcgs_brainsplit.json`,我读取原值):

| | SSIM_brain | SSIM_background | SSIM_whole |
|---|---|---|---|
| T1 iso-TV | **0.91110** | 0.88814 | 0.89358 |
| T1 edge-TV | 0.91034 | **0.89439** | **0.89799** |
| FLAIR iso-TV | **0.92933** | 0.89291 | 0.90174 |
| FLAIR edge-TV | 0.92715 | **0.90339** | **0.90934** |

**edge-TV 唯一的胜项(整体 SSIM)100% 来自脑外背景** —— 占 77–78% 体素、<2% GT 能量的空气区。脑内两个对比度都更差。加上 §2 C1 的 edge-NMSE / HFEN 全面上升,以及打分器自带的抗模糊诊断 `sharp`(edge-TV 0.768/0.810 vs 🟡 iso 0.804/0.845,更**模糊**),"edge-preserving" 这个名字与其实测行为相反。

另外这个 A/B **不干净**:`edgeTV_best` 同时改了 `edge_tv=true` 与 λ_geo 0.06→0.10,而 🟡 对齐 mask 上没有 iso-TV λ=0.10 的对照。背景平滑度的收益甚至无法归因于 edge-TV 本身。

### 3.3 基线覆盖的缺口

| 基线类别 | 仓库内存在? | 是否充分 |
|---|---|---|
| Zero-filled / conj-SENSE | ✅ 有(每个 score JSON 的 `zf`) | 是(正确的地板) |
| CS-SENSE(TV) | ❌ 无代码/无 JSON | 否(见 §3.1) |
| L1-wavelet CS-SENSE(本数据) | ❌ | 缺 |
| 结构化低秩(LORAKS/SAKE/ALOHA) | ❌ 全树零代码 | **缺(关键类别)** |
| 结构引导 TV / parallel level sets(Ehrhardt) | ❌ | **缺(方法机制的正面对手)** |
| 联合多对比度经典法(VTGV/jVN) | ❌ | 缺 |
| 展开网络(VarNet/MoDL) | ❌ | 缺(需按"免训练/已训练"两栏分开呈现) |
| 单对比度 3DGS(=3DGSMR) | ❌ 仓库内无 | **缺(最直接的消融:双幅值到底值多少)** |

### 3.4 天花板归因(我的独立判断)

不是优化问题(重复运行差 <0.001 dB;1500→4000 iters 仅约 +0.09 dB),也不像是表示容量问题(🟡 900k 高斯比 577k 更差)。**决定性证据是:一个普通体素网格 + 普通 TV,落在 577k 各向异性高斯的 0.2–0.4 dB 以内,并在未采样 k 的恢复上更好。** 两种截然不同的表示在同一先验类下收敛到同一处 ⇒ **约束来自先验类(手工 TV 类)+ 采样,而非表示**。

结算这个问题的两个实验:
1. 用**满采样** k 空间拟合同一高斯模型 —— 若远早于数值地板饱和,则表示是瓶颈,架构需重想。
2. 在同一算子上跑一个**真正不同的先验类**(SLR 或 joint SG-TV)—— 若显著超出 TV 家族的 ~0.3 dB 带宽,则残差是先验受限,"AF=12 物理天花板"的叙事应当撤回。

---

## 4. 我对评审意见的保留与反驳

1. **`e_null` 指标的批评成立,但不推翻 MAISI 结论。** 评审 3 指出 `e_null` 在幅值域定义,`|·|` 的非线性本身会在未采样区产生频谱内容(故 zero-fill 得 0.430 而非 1.0),因此它不是复数意义上的纯零空间误差 —— **这条我接受**,论文中必须改名并重新定义。但所有候选用的是同一把尺子,MAISI 那次 49%/89% 的**排序**结论不受影响。
2. **`center_kspace_dc` 的"无收益"记载不准确 ✅。** `centered_4k` 拿到全部运行的最高 PSNR(38.588/38.420,vs `reg_aligned_4k` 38.512/38.291),但 SSIM 大幅下跌(0.8632/0.8809 vs 0.8936/0.9017)。它是又一次 PSNR↔SSIM reshape,应如实记载为 reshape 而非"无收益"。
3. **文献类断言需自查。** §1.2 表中标 🟡 的条目(AtlasGS、Niessen MICCAI 2025、Ehrhardt、IMJENSE、R²-Gaussian 等)是评审员检索所得,我只核实了 3DGSMR 一条。**引用前必须逐条自行确认** —— 编造引用是比负结论更严重的后果。
4. **工程与记录质量应被公正承认。** 三名评审独立指出:文档中每一个可复算的数字都精确复现(我自己也复现了脑内 PSNR/SSIM/NMSE、zero-fill 锚点、mask 采样率);逐次运行的 config 快照 + `run_info.json`(含模型源码 SHA256)是优于多数投稿的溯源实践;`§7.2/§7.3` 主动披露自身陷阱;`§9` 的负结果目录本身有社区价值。

---

## 5. 建议的下一步(优先级序)

1. **撤回"胜过 CS-SENSE"的记载**,把调参后的 CS-SENSE 做成仓库内一等公民(代码 + λ 扫描 + JSON),接受"最优不得落在网格边界"这一硬性验收条件。**数小时即可完成,且是其余一切的前提。**
2. **把主打变体换回 iso-TV**,或在对齐 mask 上补跑 iso-TV λ=0.10 @4000 的对照,以脑内口径重新裁决。一次运行即可定案。
3. **统一口径**:脑内掩膜 + 脑内峰值,一种约定贯穿全文,hard-DC 施加于每一个候选,并以表格形式给出各口径换算。
4. **补齐缺失的基线类别**,按价值排序:joint SG-TV(方法机制的正面对手)→ SLR(LORAKS/SAKE)→ L1-wavelet CS → 单对比度 3DGS 消融。
5. **跑其余 15 例**(§2.3),给所有结论装上误差棒。
6. **重新定位选题。** 本项目已无意中取得的最强证据是:*在同一算子、同一 GT、同一 hard-DC 下,充分调参的经典 CS 与 577k 高斯的差距接近于零* —— 这直接质疑已发表文献中"高斯 ≫ CS-TV"的巨大 margin(3DGSMR 报告 AF=8 时约 9.5 dB 的差距 🟡)是否是基线调参不足的产物。**一篇诚实的、公平基线的 benchmark/复现研究,比一篇"我们又加了一项正则"的方法论文更有发表价值,也更符合本项目的现有资产。**

---

## 附录

**我的复核脚本**(会话 scratchpad `6d63c3af-570b-444e-acd6-95a021b6e8e5/scratchpad/`):
`nullspace_gate.py` / `nullspace_gate2.py` / `nullspace_gate3.py` / `nullspace_gate_fig.py`(MAISI 准入闸)、`n4_errormap.py` / `n4_errormap_fig.py`(N4 误差图)。CS-SENSE 重打分与配对受试者清点为本次会话内联执行。

**评审 subagent**(可用 SendMessage 继续追问):
`a535b6f6113c072d1`(novelty / MedIA fit)、`ae40185720bc31d5d`(feasibility / rigour)、`a4f50ee5554513c0c`(reconstruction quality)。

**相关文件**:被评审对象 `docs/mcgs_edge_tv_method.md`;预注册规格 `docs/superpowers/specs/2026-07-23-edge-preserving-tv-csf-design.md`;判据实测 `results_arc/edge_scores.tsv`;图 `results_arc/{maisi_nullspace_gate,n4_errormap_edgeTV_T1_FLAIR}.png`。
