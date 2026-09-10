# 本机心脏 k-space 数据盘点（2026-09-08）

范围：`/home/congcongliu`、`/data0/congcong`、`/data1/data1_congcong`、`/data1/dataset`、`/data2/congcong`，先按目录名全盘搜索，再逐目录核实文件类型、数组维度、DICOM 头与协议文件，关键样本目视确认解剖部位。

## 结论

本机唯一同时满足"3D、多对比、心脏、前瞻欠采 k-space"的数据是 `Corony_data/1`，只有 1 例，且没有全采样参考。其余带 k-space 的心脏数据都是 2D cine。大量心脏影像只有 DICOM，不能做原始域重建。

## 有 k-space 的数据

### 1. `/data0/congcong/data/Corony_data/1/`：UIH 5T 全心 MRCA，1 例

- 设备 UIH uMR Jupiter 5.0T；序列 `MRCA_1ISO_ACS11`；扫描日期 2024-01-14；男，60 岁。DICOM 的 BodyPartExamined 为 CARDIAC，目视为轴位胸部全心（`figures/case1_ACS11_IP.png`）。
- 协议文件要点：3D 双回波 GRE，TE 1.85 与 2.68 ms，TR 5.04 ms，翻转角 10 度，FOV 288 乘 300 mm，ReadResolution 288，SlicePerSlab 120，Acceleration2d 2 乘 Acceleration3d 2，ACS 11，T2Prep 关，NavigatorMode 0，RespAcqWindowRatio 0.6，ECG 触发（`vsm.ecg1.dat`、`vsm.ecg2.dat`）。
- 输出对比：Water、Fat、IP、OP 四套 DICOM，各 240 层，576 乘 600，0.5 mm 插值。
- 文件：`raw/7_MRCA_1ISO_ACS11/*.raw.part1` 250 MB，顶层 `*_combine.raw` 7.4 GB。

### 2. `Corony_data` 其余 20 次扫描是大腿和盆腔外周 MRA，不是心脏

- `0.8mm`、`0.9mm`、`1.0mm` 下的 `pelvis` 与 `thigh`，2023-11-12 到 16，序列 `MRCA_ISO0.8/0.9/1.0`，BodyPartExamined 为 ANGIOGRAPHY，目视为双侧大腿或盆腔（`figures/thigh_1p0_IP.png`、`figures/pelvis_1p0_IP.png`）。每例 `.raw` 15 到 44 GB，`mat_file_rawdata` 317 GB，整个目录约 1.2 TB。
- `3D/csm.mat` (576,380,144,34) 与 `3D/mask.mat` (576,380,144,24) 属于 0.9 mm 盆腔例 UID_7302732128884904667；`crop_*.mat` (100,380,144,34) 属于大腿例 UID_7300447768923914363；`mask_320_128_R10/15/19/29.mat` 是回顾性掩膜。这些都不是心脏。
- 读取链路 `read_raw_data/`：`Merge_5T_Raw.m` 合并分卷，`Read_UIH_RawCMRA_v5_2` 读原始，`proprocess_CMRA.m` 做斜坡采样重网格，`rearrange_shotrawdata.m` 按 shot 重排（180 shot 乘 140 段，nRO 640，nCH 16，nPE 416，nSPE 233）。

### 3. CMRxRecon2023 本地副本，两份

- `/data1/data1_congcong/data/realtimecine/MICCAIChallenge2023/ChallengeData`，`/data0/congcong/data/realtimecine` 有同名副本，94 GB，只有 `MultiCoil/Cine`。
- 训练集只有 `AccFactor10`，72 例；验证集 `AccFactor04/08/10` 各 60 例；另有 `Local_GRAPPA/MultiCoil/Cine/ValidationSet`。
- `cine_sax.mat` 内 `kspace_sub10` (12, 9, 10, 246, 512)，即 (nt, nz, nc, ny, nx)，`mask10` (246, 512)，MATLAB v7.3。没有 FullSample，没有 Mapping。

### 4. `RTcine_tfast_2022`，`/data0` 与 `/data1` 各一份，2.9 GB

- UIH 实时 cine `.raw` 7 个，23 到 81 MB；`CScine_sax.raw` 272 MB 为前瞻压缩感知采集，配 `.prot`；`tfast_6x/7x/8x.cfl` 各 92.6 MB。
- `AICine/data/`：`fullsample/cine_sax.mat` 936 MB 加 `AccFactor04/08/10`，1 例。MATLAB 脚本 CSCine_Spirit、CSCine_ScoRe、CSCine_ktSpirit、TSNMs。全部 2D。

### 5. `/data1/data1_congcong/data/cine_lax_4ch.mat`

- `kspace` (12,1,10,198,448)，v7.3，配套 `cine_lax_4ch_im_448x198x12.mat`。单例 2D 四腔长轴 cine。

## 只有影像的心脏数据

- `/data1/data1_congcong/data/5T/Body`，104 GB，251,558 个 dcm。`Archived/BOD001` 到 `BOD112` 是 2026-09-02 去重集。BOD001 为 2021-12-14 心脏 cine 2ch、4ch、sax 加黑血 `t2_fse_db(_fs)_ACS`。BOD081 为 2026-08-14 `zz_CMRA_wukerui`，含 `MRCA_tra_{Water,Fat,IP,OP,IP_LocalP,OP_LocalP}` 与 `MRCA_tra_full_*` 成对，另有 `gre_cine_4ch_fb`。第四批含 `gre_cine_sax_3sl/9sl`、`gre_cine_4ch`、`psir_sax_9sl_NIR`、`t2_fse_db_spair_sax`、`epi_dti_sax`、`Cine_CEST`。
- `5T/Public Data`：30 例，patient1 到 9 与 volunteer1 到 21，`native/post_s1..3.mat` 内是 `image` (256,228) uint16 幅值，T1 mapping；`Body/Cine_5T` 是其 DICOM 版。
- `/data0/congcong/data/CMR_PETMR_full63`，68 GB，与 `/data1/dataset/Cine_MRI_NMG`，76 GB，71 例，Siemens Biograph mMR：cine、T1/T2 map、LGE、灌注、NH3 PET。数据卡写明无 raw。
- `/data1/dataset/NMG_rawdata`，357 GB，名不副实。`chaizhuren/心脑血管医院/CMR合作项目导出文件` 98 次检查、志愿者 rar、2026 心梗 zip；`wangzhuren` 下是 Cine_MRI_NMG 与 PET_body。149 个 `.dat` 都是 500 字节的 StudyInfo。
- 空目录：`/data0` 与 `/data2` 两处 `FM_data/CMRxRecon_ty`，`5T/MC_data`，`Multicontrast_3D`。

## 图

`figures/case1_ACS11_IP.png`、`figures/thigh_1p0_IP.png`、`figures/pelvis_1p0_IP.png`，每张为中层、正交重排、层向 MIP。本机原图在 `/home/congcongliu/figs/cardiac_survey/`。

## 建议

要"3D 多对比心脏 k-space 且有全采样参考"，最短路径是向 5T 站点索取 BOD081（2026-08-14）那次 `MRCA_tra` 与 `MRCA_tra_full` 的原始数据，加速与全采样成对、Dixon 四对比，读取链路可直接复用 `Corony_data/read_raw_data/`。
