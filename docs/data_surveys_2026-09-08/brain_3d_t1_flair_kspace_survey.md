# 公开的"3D T1 + FLAIR 多对比 k-space"脑部数据调研

日期：2026-09-08。判据同心脏调研：多线圈原始 k-space、真 3D（ky 与 kz 两个相位编码）、同一受试者同时有 3D T1 与 3D FLAIR。

## 结论

公开渠道只有一套同人配对的 3D T1 加 3D FLAIR 原始 k-space，其余都是 2D、单对比或只有图像。本机自采的 UIH 3D T1/T2/FLAIR 数据（16 对 T1+FLAIR）在规模和采样完整性上都超过公开集。

## 直接命中（已核实）

| 数据集 | 内容 | 采集 | 格式与获取 | 论文 |
|---|---|---|---|---|
| Markerless Prospective Motion Correction Data（哥本哈根 Rigshospitalet / DRCMR） | 22 名健康成人，完整临床协议：3D T1 MPRAGE（22 人）、3D T2 FLAIR（仅 10 人）、2D T1 STIR、2D T2 TSE、2D T2* GRE、2D DWI；每个序列在静止/点头两种状态下各采 PMC 开与关，MPRAGE 另有摇头 | 3T Prisma，64 通道；MPRAGE 0.93 mm iso 矢状，GRAPPA 2，5:12；FLAIR 1.03 mm iso 矢状，加速 3，4:12。注意是前瞻加速，不是全采样 | 原始 ISMRMRD 放在 PublicnEUro（DOI 10.70883/QBVI3432），需注册并签 DUA，审批 1 到 2 周；去面 NIfTI 在 OpenNeuro；CC BY-NC-ND 4.0 | Scientific Data 2026，doi 10.1038/s41597-026-07144-z |

## 近邻（已核实）

- M4Raw（Zenodo 7523691，V1.6 加测试集 8056074）：183 名健康人，0.3T，4 通道，T1w/T2w/FLAIR 同人配对且各 2 到 3 次重复，但是 2D 多层（18 层，5 mm），H5 类 ISMRMRD 格式。Scientific Data 2023。
- fastMRI brain：6970 次 2D 轴位扫描，T1/T1 增强/T2/FLAIR 各自独立文件，同人不配对，非 3D。
- Calgary-Campinas CC-359：3D T1 GRE 1 mm，67 人 12 通道原始，无 FLAIR。本机已有 tar 包（/data1/data1_congcong/data/3D/calgary-campinas_version-1.0.tar.gz）。
- Diff5T（SIAT，Scientific Data 2025）：5T 脑部扩散 k-space 为主，摘要未提 FLAIR。
- T2*-MOVE（Helmholtz Munich）：多回波 GRE 运动 k-space，ISMRMRD，单对比族；页面本次未能打开。
- 哥本哈根 2021 挑战（Zenodo 4572640）：2D 轴位 T1/T2 含运动 k-space，25 人，非 3D。
- Bilgic 组软件页只有 QSM 挑战与多 shot EPI 数据，没有 MPRAGE/SPACE/FLAIR 原始。
- AHEAD 7T MP2RAGEME：Philips PAR/REC 图像，无 k-space。7T pTx（Zenodo 21409149）只是 PDF。
- 只有图像的 3D T1 加 FLAIR：MNI FCD 多中心（Zenodo 5193400）、MGH 血管区模板等，不能当 k-space 用。

## 本机自有（UIH 3D，1 mm 矢状，12 通道 256×264×256）

- `/data1/data1_congcong/data/3D/{T1_csm_ksp,FLAIR_csm_ksp,T2_csm_ksp}`：h5 内含 csm 与 kspace，(256,12,264,256) complex64。T1 29 例、FLAIR 16 例、T2 16 例；T1 与 FLAIR 同人配对 16 例（chenshuaibao、huangyuwei、jiangdian、jiangyizhou、liaozhenghan、lishengbo、litianyu、shiyixuan、sunxinyi、tangbing、yangkun、zhangxiqian、zhangyangming、zhoushuo，另有 790 机型的 jiangdian 与 tangbin），11 例只有 T1。`/data0/congcong/data/T1_T2_FLAIR_3D` 是同样计数的副本。
- T2 目录名不副实：同日另一次逐字节比对发现 `T2_csm_ksp` 15 个同名文件里 12 个与 FLAIR 完全相同，只有 790_jiangdian、790_tangbin、tangbing 三例是真 T2。真正的 3D T1+T2+FLAIR 三对比原始 k-space 在 `/data1/data1_congcong/data/MC_Jia/Train_SpiritDiffusion_CFL/vol1..7`，BART CFL，每例 (256,256,176,26)，26 线圈，DC 居中，T1 相位编码面满采，T2 与 FLAIR 有椭圆厂商遮挡（支撑 0.777 与 0.758）。
- 注意：两处目录里残留 `.baiduyun.uploading.cfg` 标记（T1 20 个、FLAIR 5 个、T2 9 个），说明当时网盘传输可能未完成，用前逐个校验 h5 能否打开、尺寸是否一致。
- 原始 .raw 形式：`/data0/congcong/data/MultiContrast3D`（tangbin 790 与 tangbing，t1_fsp3d、t2_mx3d、t2flair_mx3d 以及 0.68 mm 4 倍加速，72 GB）；`/data1/data1_congcong/data/MC_Jia`（huangyuwei 原始、luobaoyu 2023-08 前瞻 ACS 的 t1/t2/flair 1 mm iso、SpiritDiffusion 7 卷 cfl，85 GB）；`/data0/congcong/data/3D_data/3D_MC`（1 人 t1/t2/t2flair 原始加 UIH 读取代码，37 GB）。
- 已打包的实验集：`/data1/data1_congcong/data/3D/T1W_FLAIR_AF12_bundle`（huangyuwei T1/FLAIR trainer .mat、ESPIRiT csm、对齐后的 AF12 掩膜，11 GB）。

## 建议

- 需要外部验证集时，用哥本哈根集的 10 例 3D T1 加 FLAIR 配对做跨厂商泛化测试，前提是接受前瞻加速与 DUA。
- 需要大样本时，本机 16 对仍是主力；先清理网盘残留标记，再考虑把 11 例只有 T1 的受试者补扫 FLAIR。
