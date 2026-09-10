# 交接 2026-09-10：心脏 3D 多对比 k-space 数据线

本次会话只做了数据：找、下、验、选、归档。没有动重建代码，没有跑任何实验。

## 1. 已完成且已验证

**全网调研。** 公开的多对比 3D cine 原始 k-space 只有两套，都是伯尔尼与洛桑团队的自由运行 3D 径向全心数据，CC-BY-4.0。CMRxRecon 全系列没有 3D 采集，OCMR 只有 2D。详见 `../data_surveys_2026-09-08/3d_cine_kspace_public_survey.md`。

**下载与校验。** 两套共 11 个压缩包全部下到 `/data2/congcong/data/cardiac_3d_cine_public/`，MD5 与 Zenodo API 清单逐个一致。

```
cat /data2/congcong/data/cardiac_3d_cine_public/logs/md5_check.txt   # 11 行全部 OK
cat /data2/congcong/data/cardiac_3d_cine_public/download_status.txt
```

**发现水激发那套在源头损坏。** `inVivo.zip` 的 MD5 与清单一致，但两个独立解码器在同样 7 个条目上失败，24 个 .dat 只剩 17 个可用。Zenodo 上该记录只有一个版本，重下无用。证据 `logs/unzip_test_inVivo.log`。

**选型与归档。** 选多回波那套的 V03、V04、V07、V08，已归档进项目 3D 数据根。

```
ls -l /data1/data1_congcong/data/3D/cardiac_freerunning_multiecho_5D/V07/
# freerunning_8echo.dat -> /data2/.../V7/V7/meas_MID00613_..._8XC_..._22x1000.dat
# ecgtrig_5echo.dat     -> /data2/.../V7/V7/meas_MID00612_..._5XC_..._12x437.dat
# PhysioInfo.mat        -> /data2/.../V7/V7/PhysioInfo.mat
```

理由与全部参数见 `../data_surveys_2026-09-08/downloaded_data_and_selection.md`。

## 2. 待用户拍板

**要不要向作者索要完整的水激发压缩包。** 那 7 个坏条目让 BORR 只剩 2 例、FISS 只剩 3 例。若这条线还要做，需要给 Safarkhanlo 或 Bastiaansen 发信，附文件名与 CRC 证据。若不做，这套就只当备用。

**归档用软链接还是实体拷贝。** 现在是软链接，`/data1` 实占 64 KB。若担心 `/data2` 上的原件被清理，改成 `cp -rL` 即可，代价是 43 GB。

**是否扩到 10 例。** 现在只接了 4 例 24 线圈的。V6、V9、V10 是 20 线圈，同协议，随时可加。

## 3. 下一步

1. 先写 twix 读取与轨迹计算，验证能从 `freerunning_8echo.dat` 复原出 22,000 条辐条乘 8 回波乘 24 线圈的数组，并与 `PhysioInfo.mat` 的 176,000 条时间戳对上。
2. 接 NUFFT，用全部辐条做一次 CS 或 XD-GRASP 参考重建，确定 PSNR 口径。
3. 再谈回顾性欠采与联合重建。

## 4. 已知的坑，别重做

- **别再全网找数据。** 心脏与脑部两轮调研都已落在 `../data_surveys_2026-09-08/`，包括所有查过但没有结果的仓库与课题组。
- **别再试直连下载。** 这台机器没有国际直连，全部流量走 clash-verge 的 TUN，global 模式，无 root 改不了路由。114 个节点里只有 20 个能通 Zenodo。
- **别把 `T2_csm_ksp` 当 T2。** 那批 15 个文件里 12 个与 FLAIR 逐字节相同。真正的 3D T1 加 T2 加 FLAIR 三对比原始数据在 `MC_Jia/Train_SpiritDiffusion_CFL`。
- **读 twix 时别取第一个 measurement。** 每个 `.dat` 含 2 个，第一个是 `AdjCoilSens` 预扫。
- **别把文件尾部的多余字节当截断。** twix 按 512 字节对齐补零。
- **长任务要脱离会话跑。** 解压期间后台任务被内存看门狗误杀过，当时机器实际有 938 GB 空闲。

## 5. 关键决定的为什么

**为什么不选水激发那套。** 它的 5 个序列里 4 个都是 bSSFP 家族，只差脂肪抑制脉冲，对比度差异本质是脂肪信号而不是组织加权，共享结构假设近乎恒等，拿它论证联合重建站不住。加上源头损坏，唯一"齐全"的 V05 其 FISS 还是另一协议变体。

**为什么选 V03、V04、V07、V08。** 它们是全组里唯一同时满足统一 2P0MM 协议、24 线圈、两次扫描线圈数一致这三条的。

**为什么用软链接。** `/data1` 已用到 76%，`/data2` 还有 22 TB，且原件在 `/data2` 有 MD5 清单。不值得为 43 GB 做重复。

**为什么这些文档只加到 main 而不合并整个工作分支。** main 是精简的发布分支，与工作分支已分叉 139 个提交。合并会把整棵工作树倒进 main，改变它的性质。这次只挑了文档提交。
