# 数据调研 2026-09-08：带欠采样 k-space 的 3D 多对比数据

起因：为多对比 3D 重建找"3D、多对比、原始 k-space、欠采样"四合一的数据，先心脏，再脑部 T1 加 FLAIR。三份文档都以官方记录页（Zenodo API、GitHub、PMC、IEEE DataPort）逐条核实，未能核实处在文内标明。

| 文件 | 范围 | 一句话结论 |
|---|---|---|
| `cardiac_local_kspace_inventory.md` | 本机五个数据根的心脏 k-space 盘点 | 唯一"3D 加多对比加心脏加前瞻欠采"原始数据是 `Corony_data/1`，UIH 5T 全心 Dixon 双回波 1 例；同目录其余 20 次扫描是大腿和盆腔；其余心脏 k-space 都是 2D cine，大量心脏影像只有 DICOM |
| `3d_cine_kspace_public_survey.md` | 全网公开的多对比 3D cine k-space | 只有伯尔尼/洛桑两套自由运行 3D 径向全心数据（Zenodo 13868462：5 序列乘 5 人；7621356 加 7615780：8 回波乘 10 人），CC-BY 但无全采样参考；真 3D 笛卡尔只有 MR-linac 5D 10 例与 CMRx4DFlow2026；CMRxRecon 全系列没有 3D 采集 |
| `brain_3d_t1_flair_kspace_survey.md` | 全网公开的脑 3D T1 加 3D FLAIR 同人 k-space | 只有哥本哈根 DRCMR 运动校正集，MPRAGE 22 人、FLAIR 仅 10 人，前瞻加速，PublicnEUro 需 DUA；M4Raw 与 fastMRI 是 2D，Calgary-Campinas 无 FLAIR；本机 UIH 自采 16 对 T1 加 FLAIR 更大，但 T2 目录多为 FLAIR 副本 |

`figures/` 是三张目视核实图：`case1_ACS11_IP.png` 证明 `Corony_data/1` 是心脏，`thigh_1p0_IP.png` 与 `pelvis_1p0_IP.png` 证明其余是外周血管。

## 判据与分级

- k-space：多线圈原始数据，不是 DICOM 或 NIfTI。
- 3D：两个相位编码方向；2D 多层堆叠与挑战赛的"3D k-t 掩膜"都不算。
- cine：心动周期分辨；3D 加 t、4D、5D 都算。
- 多对比：同一受试者的多种对比或多回波。
- T0 目标；T1 3D cine 单对比；T2 3D 动态非 cine；T3 多对比但 2D；T4 合成；T5 3D 静态。

## 相关本机目录速查

- 心脏 3D 原始：`/data0/congcong/data/Corony_data/1/`
- CMRxRecon2023 本地副本：`/data1/data1_congcong/data/realtimecine/MICCAIChallenge2023/ChallengeData`，仅 cine，训练集只有 AccFactor10
- 脑 T1 加 FLAIR 配对 h5：`/data1/data1_congcong/data/3D/{T1_csm_ksp,FLAIR_csm_ksp}`
- 脑真 T1 加 T2 加 FLAIR 三对比 CFL：`/data1/data1_congcong/data/MC_Jia/Train_SpiritDiffusion_CFL/vol1..7`
