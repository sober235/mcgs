# 公开的"多对比 3D cine + k-space"心脏数据集调研

日期：2026-09-08。范围：挑战赛/基准集、Zenodo/DataCite/Figshare/OSF/IEEE DataPort/scidb/OpenDataLab、课题组论文的数据可用性声明与代码仓库、工具箱样例、合成幻影、中文来源。所有"已核实"条目都用官方记录页（Zenodo API、GitHub、PMC、IEEE DataPort）核对过；未能核实的地方明确标注。

## 判据

- k-space：多线圈原始数据（.dat/.mat/.h5/MRD），不是 DICOM/NIfTI。
- 3D：两个相位编码方向（ky 与 kz），全心 3D 径向、3D 笛卡尔、stack-of-stars 都算；2D 多层堆叠不算；挑战赛文档里的"3D k-t 掩膜"指 (kx,ky,t)，也不算。
- cine：心动周期分辨（ECG/自门控/实时）；3D+t、4D、5D（含呼吸维）都算；4D flow 视为近邻。
- 多对比：同一受试者的 3D cine 有多种对比（亮血/黑血、多回波 Dixon、不同脉冲/序列），或 3D cine 之外还有同人的其它 3D 对比。

分级：T0 = 3D cine + 多对比（目标）；T1 = 3D cine 单对比；T2 = 3D 动态非 cine（4D flow）；T3 = 多对比但 2D；T4 = 合成；T5 = 3D 静态心脏 k-space。

## 结论

全网符合 T0 的公开数据集只有两个，都来自伯尔尼/洛桑（Bastiaansen、Stuber、Piccini）团队的自由运行（free-running）3D 径向全心采集，且都是 CC-BY-4.0 直接下载：

1. Zenodo 13868462（2024-10）：5 名志愿者，每人 5 种序列：bSSFP、FISS、BORR、LIBRE、LIBOR。
2. Zenodo 7621356 + 7615780（2023-02）：10 名志愿者，自由运行 8 回波 GRE 加 ECG 触发 5 回波 GRE，Pilot Tone 生理信号。

两者都没有全采样参考，因为自由运行径向本身就是欠采样的，"真值"只能是全数据的 CS/XD-GRASP 重建。矩阵都是 2 mm 各向同性、112 左右立方，样本 5 到 10 人。

## T0：3D cine + 多对比（已核实）

| 数据集 | 内容 | 采集 | 格式与体量 | 许可/入口 | 论文与代码 |
|---|---|---|---|---|---|
| Evaluating rapid water excitation techniques for 5D whole-heart fat-suppressed free-running cardiac MRI at 1.5T | 5 名健康志愿者，每人 bSSFP、FISS、BORR、LIBRE、LIBOR 五次自由运行全心采集 | Siemens Sola 1.5T，32+18 通道，3D 径向螺旋叶序，112×112×112，FOV 220 mm，2 mm iso，45,048 条读出，自门控分 25 心动 × 4 呼吸相位，扫描 1:51 到 3:41 | inVivo.zip 34.3 GB，扫描仪原生原始格式 | CC-BY-4.0，https://zenodo.org/records/13868462 | Safarkhanlo et al., MAGMA 2025, doi 10.1007/s10334-025-01273-z；自门控代码 github.com/QIS-MRI/SG_RespiratoryCardiac_3DRadial |
| V1–V5 与 V6–V10 Raw Data: Motion-resolved fat fraction mapping with whole-heart free-running multiecho GRE | 10 名健康志愿者；每人自由运行 3D 径向 8 回波 GRE 加 ECG 触发 5 回波 GRE，PhysioInfo.mat 含 Pilot Tone 心动/呼吸信号 | 1.5T，2 mm iso，6 min，心动+呼吸分相，产出水/脂/脂肪分数/R2*/B0 | 10 个 7z，每个 4.2 到 6.2 GB，合计约 53 GB，Siemens .dat（仓库内含 mapVBVD 读取） | CC-BY-4.0，https://zenodo.org/records/7621356 与 https://zenodo.org/records/7615780 | Mackowiak et al., MRM 2023, doi 10.1002/mrm.29680；读取与轨迹代码 github.com/QIS-MRI/ReadDataAndTrajectory_FreeRunningFatFractionHeart |

边缘 T0：A demo imaging dataset for simultaneous 5D cardiopulmonary MRI（Zenodo 15033956，2025-03，上海交大 Ding 等）。1 名受试者，约 10 min 自由呼吸，1.5T，1.5 mm iso，双回波 3D 径向 bSSFP（Echo1/Echo2 的 k-space、轨迹、DCF、SI 导航、序列参数、5D 重建），37.7 GB .mat，CC-BY-4.0。论文 Ding et al., MRM 2026, doi 10.1002/mrm.70334；代码 github.com/CardiopulmonaryMRI/SubSpaceMoCo5D。双回波算弱多对比，且只有 1 例。

## T1：3D cine 单对比（已核实）

| 数据集 | 内容 | 格式与体量 | 许可/入口 | 来源 |
|---|---|---|---|---|
| CMR 3D Cine, 4D Flow and Exercise Stress 4D flow datasets（OSU） | 3D cine 笛卡尔 bSSFP 自门控，呼气相 k-space 按 20 心动相位排好，前瞻欠采 R 7 到 34，无全采样；Siemens 1.5T 28 通道与 3T 48 通道；另含静息 4D flow 与负荷 4D flow 各 1 例 | 3 个 .mat：5.14 GB、807 MB、696 MB | CC-BY-4.0，https://zenodo.org/records/12515230 | Arshad et al., MRM 2024, doi 10.1002/mrm.30123；github.com/OSU-MR/motion-robust-CMR |
| 5D whole-heart MRI data with self-gating for EMORe & CS（OSU） | 1 名志愿者，5 min 自由呼吸自门控 5D 全心，末分钟含模拟咳嗽 | volunteer1.mat 7.4 GB | CC-BY-4.0，https://zenodo.org/records/15769665 | EMORe，Arshad 等，IEEE TMI 2026（arXiv 2507.23224）；github.com/OSU-MR/Motion-Robust-5D-MRI-EMORe |
| Fast ungated 5D cardiac MRI on a 1.5T MR-linac, public k-space dataset | 10 名健康志愿者，无门控可变密度 3D 笛卡尔 bSSFP，60 s，2 mm iso，8 通道；HDF5 内含欠采 k-space、ESPIRiT 灵敏度图、运动校正参考、5D 重建、相位编码索引、人工分割 | 10 个 .h5，7.8 到 13.3 GB，合计约 115 GB，附 Python 示例 | CC-BY-4.0，https://zenodo.org/records/21278895 | Terpstra 等，UMC Utrecht，arXiv 2607.16033（2026-07；Zenodo 记录上作者暂隐名） |
| SSA-FARY 数据（哥廷根 Uecker 组） | 8 名志愿者 stack-of-stars bSSFP 自由运行 3 min，心动+呼吸自门控；3T Skyra，30 通道；只有 14 个分区、层厚 7 mm，中心 6 层全采样、外 8 层 R=4，kz 方向很粗；另有 2D 单层 bSSFP 与 SMS FLASH | SoS_Vol1 到 8 各 4.0 到 5.1 GB，BART cfl/hdr | CC-BY-4.0，https://zenodo.org/records/3822451 | Rosenzweig et al., IEEE TMI 2020, doi 10.1109/TMI.2020.2985994 |

## T2：3D 动态非 cine

- CMRx4DFlow2026：约 400 例，10+ 中心，1.5/3/5T，主动脉为主；训练集给 kdata_full (Nv=4, Nt, Nc, SPE, PE, FE) 加 coilmap 加 segmask，验证集 kt-Gaussian R 10 到 50；需批准参赛，数据禁运到 2026-12。Synapse syn64545434。
- OSU 静息与负荷 4D flow 各 1 例（同上 Zenodo 12515230）。
- ETH FlowVN 训练/测试数据（Zenodo 19565271，2026-04）：受限访问。

## T5：3D 静态多对比（已核实）

- Fat-free noncontrast whole-heart CMR with LIBOR/BORR/LIBRE（Zenodo 8338079 与 10715769）：3T，自由呼吸呼吸自导航 3D 径向全心，10 名志愿者，每人 LIBOR 1 ms、BORR 1 ms、LIBRE 1 ms、LIBRE 2.2 ms 四脉冲，原始 Siemens 加 DICOM，约 19 GB，CC-BY-4.0。论文 Mackowiak et al., JOCMR 2024, doi 10.1016/j.jocmr.2024.101096；代码 github.com/QIS-MRI/LIBOR_LIBRE_BORR_SimulationCode。静态全心，非 cine。
- Pole-to-pole Phyllotaxis MRI Dataset（Zenodo 17428583，2025-12）：3 名志愿者 ECG 触发全心 3D 径向 GRE 3.6 GB，另有球形与 NIST 幻影 bSSFP 37 GB，MATLAB 格式，CC-BY-4.0。论文 Peper et al., MRM 2026, doi 10.1002/mrm.70237。

## T3：多对比但 2D（已知，简列）

- CMRxRecon2023/2024/2025：2D 多层 5D 数组 (nx,ny,nc,nz,nt)，无 3D 采集。
- OCMR v1.2（2023-08）：165 组全采样加 212 组前瞻欠采 2D cine，含 0.55T 数据；仍无 3D。
- Harvard CMR Dataverse：唯一 k-space 条目是 2D 径向 cine。
- PTB Open-Source Cardiac MRF（Zenodo 15831511，2025-07）：cMRF 加笛卡尔/径向 cine，3 名志愿者，MRD 格式，1.8 GB。
- MMCMR-427K（IEEE DataPort，2026-03）：427,465 组多线圈 k-space，1,504 人，6,120 次扫描，12 种模态，13 中心 15 台设备 4 厂商，约 3.5 TB 两部分；需 IEEE DataPort 订阅登录；页面未说明是否含 3D 采集。论文 arXiv 2512.21652；代码 github.com/wangziblake/CardioMM_MMCMR-427K。

## T4：合成

- MRXCAT（ETH 2014，cmr.ethz.ch/research/software.html，GitHub mrxcat-cmr/mrxcat）与 MRXCAT2.0（gitlab.ethz.ch/ibt-cmr-public/mrxcat-2.0 与 GitHub sbuoso/mrxcat-2.0，MIT）：可生成 2D/3D cine 与灌注的多线圈 k-space，任意对比与轨迹；需要 XCAT 幻影许可，非商业；只有模拟器和示例，没有现成大数据集。
- CMRsim（ETH，Python）：含运动与流动的心血管 MR 模拟。

## 动物

- Dryad 10.5061/dryad.7b73j（2017）：小鼠 3T 3D 自门控 stack-of-stars T1w GRE 原始 .dat，1.3 GB，CC0。
- Zenodo 4607426（2021）：7 只小鼠 3D UTE 4D 自门控原始，5.8 GB。
- Zenodo 18809241（Cedars，2026-02）：猪自由运行运动分辨 3D 采集 Siemens TWIX 原始加重建代码，3.27 GB。
- Zenodo 19005977（USC/NHLBI，2026-03）：猪 0.55T 容积实时 stack-of-spirals 3D+t 原始，ISMRMRD 格式，4.8 GB，CC-BY-4.0；代码在 NHLBI/lit_gadgetron 的 volumetric_rt_mri 分支。

## 边缘条目

- Huttinga LowRank_MR-MOTUS 示例数据（UMC Utrecht，GitHub nrfhuttinga/LowRank_MRMOTUS，surfdrive 链接）：1 名受试者，1.5T Philips 24 通道，3D 黄金角径向胸腹自由呼吸采集，DREME-MR 用它同时解出心动与呼吸运动；格式与体量未标明。

## 明确未公开（按请求或无声明）

- KCL Qi 等 2022 自由运行 3D 全心 T1/T2 mapping 加 cine（正是多对比 3D cine，11 例，Philips 1.5T）：全文无公开数据声明。
- Free-Breathing 3D Whole Heart and Aorta Cine（2026，24 例，Philips 1.5T CASPR bSSFP）：声明"available from the corresponding author by request"。
- CINENet（Kuestner，20 健康加 15 患者 3D cine）：未公开。
- FAST-CMR 自由运行多中心多厂商研究：2026-06 才开始招募。
- Cedars 铁剂增强 3D/5D cine（Zhao 2025 ROCK 5D 无声明；Generative Multitasking 为 2D 且按请求）：未公开。
- KCL Bustin PROST（JCMR 2020）："available from the corresponding author on reasonable request"；AurelienBustin 的 GitHub 只有 PROST 代码。
- CHUV Roy 2022 铁剂 5D CMRA 按请求；5D flow 先心病 2026 与 VPAL 无声明；Ogier JCMR 2026 写明多数数据不能共享，GitLab 样例只有图像；Monalisa 工具箱无演示原始数据。
- OSU ML-DIP 3D 实时（JOCMR 2026）承诺代码与样例在 github.com/OSU-CMR/ml-dip，但仓库仍是占位；MOSAIC 3D LGE 待发表后公开。
- NHLBI 0.55T 3D cine 4D-iMoCo 论文引用的示例数据集，lit_gadgetron 对应分支只有 README。
- PTB mrpro 示例数据只有肝 GRPE、2D 黑血 TSE、呼吸 3D GRPE 胸部与 2D cMRF 幻影，无 3D 心脏 cine。
- Stanford Extreme MRI 的 Zenodo 数据是 3D cones DCE 与肺，Sandino 数据为 2D。
- Nezafat 的 Harvard Dataverse 只有 2D 径向 cine 与化学位移编码数据，无 3D LGE 原始。
- Feng 的 XD-GRASP/GRASP 演示数据是肝 DCE，无心脏 3D。
- UMC Utrecht CMR-MOTUS 3D 实时（Continuous Ventricular Volumetric Quantification）按请求。

## 未能访问

mridata.org 从服务器与联网工具两侧均连不上；scidb.cn 与 OpenDataLab 搜索页为 JS 渲染无法抓取；Zenodo API 多次触发限流，宽泛查询只成功一部分。中文社区（CSDN/知乎）帖子只转述 CMRxRecon/ACDC，没有新的原始数据集。

## 建议

1. 要"多对比 3D cine + k-space"，直接下 Zenodo 13868462（5 种序列）和 7621356/7615780（8 回波），用 QIS-MRI 的读取与自门控代码起步，重建走 XD-GRASP/CS。
2. 要更多样本或笛卡尔采样，用 MR-linac 5D（10 例，单对比）或 CMRx4DFlow2026（400 例，4 个速度编码）。
3. 要全采样参考，公开界没有 3D cine 能给；只能用本机 5T 站点 2026-08-14 那次 MRCA_tra 加 MRCA_tra_full 的原始数据，或用 MRXCAT2.0 合成。
