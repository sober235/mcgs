# 下载、校验、选型与归档（2026-09-09 到 09-10）

承接同目录 `3d_cine_kspace_public_survey.md` 的调研结论：全网公开的多对比 3D cine 原始 k-space 只有伯尔尼与洛桑团队的两套自由运行 3D 径向全心数据。本文记录把它们取到本机 amax（172.20.114.148）之后发生的事，以及为什么最终只把其中一套的 4 个人接进项目。

## 1. 下载与校验

下载根目录 `/data2/congcong/data/cardiac_3d_cine_public/`，脚本 `scripts/download_zenodo.sh` 与 `scripts/run_until_complete.sh`，校验清单取自 Zenodo API。

| 数据集 | Zenodo | 文件 | 体量 | MD5 |
|---|---|---|---|---|
| 水激发 5D 全心 | 13868462 | inVivo.zip | 32 GiB | 通过 |
| 多回波 GRE 5D 全心 | 7621356 与 7615780 | V1 到 V10 共 10 个 7z | 49.7 GiB | 全部通过 |

耗时约 10 小时。瓶颈是网络：这台机器全部出站流量被 clash-verge 的 TUN 以 global 模式接管，直连到 Zenodo 三次全部超时，说明实验室本身没有到 CERN 的国际出口。代理节点对 Zenodo 的 TLS 握手成功率约十分之一，而 aria2c 把握手失败当致命错误不重试，所以必须外层循环重试。用控制器对全部 114 个节点各做两次真实 HTTPS 测试，只有 20 个能通，唯一两次都通的是香港 BGP-08。

## 2. 水激发那套在发布方源头就损坏

`inVivo.zip` 的 MD5 与 Zenodo 清单一致，说明本地字节与 Zenodo 存储完全相同。但 Info-ZIP 的 `unzip -t` 与 Python zlib 两个独立解码器在同样 7 个条目上失败，6 个 CRC 错误，V02 的 FISS 在约 268 MB 处 deflate 流无效。该记录在 Zenodo 上只有一个版本，没有修正上传，重新下载无用。

24 个 .dat 里只有 17 个可用，按序列统计：

| 志愿者 | bSSFP | FISS | LIBOR | LIBRE | BORR |
|---|---|---|---|---|---|
| V01 | 有 | 有 | 有 | 有 | 坏 |
| V02 | 坏 | 坏 | 有 | 有 | 有 |
| V03 | 有 | 坏 | 有 | 未采集 | 坏 |
| V04 | 有 | 有 | 有 | 坏 | 坏 |
| V05 | 有 | 有 | 有 | 有 | 有 |

残次文件已隔离到 `inVivo/_CORRUPT_IN_SOURCE_ZIP/`，证据在 `logs/unzip_test_inVivo.log`。

## 3. 两套里选哪一套

结论是选多回波那套。

| | 水激发 13868462 | 多回波 GRE 7621356 加 7615780 |
|---|---|---|
| 完好数据 | 24 个里 17 个 | 20 个 .dat 全部，外加 10 份生理信号 |
| 对比度来源 | 5 次独立扫描，换脂肪抑制脉冲 | 1 次扫描 8 个回波，ΔTE 2.05 ms |
| 对比度之间是否配准 | 否，两次扫描间隔 2 到 4 分钟有体位漂移 | 是，同一读出串天然逐体素对齐 |
| 对比度差异本质 | 5 个里 4 个都是 bSSFP 家族，只差脂肪抑制 | 水脂相位加 R2* 衰减，有物理模型 |
| 自带参考 | 无 | 每人另有 ECG 触发 5 回波扫描 |
| 门控信号 | 需自己做自门控 | 自带 pilot tone、心动、呼吸、逐线时间戳 |

关键判断是水激发那套的多对比名不副实。bSSFP、LIBOR、LIBRE、BORR 全属 bSSFP 家族，差别只在脂肪抑制脉冲，共享结构假设近乎恒等，用它论证联合重建站不住。而且唯一号称五种齐全的 V05，其 FISS 实为另一协议变体，翻转角 55 度而非 50 度，扫描 186 秒而非 175 秒，跨对比度不可比。

## 4. 多回波那套里选哪几个人

全部 10 人的头部参数实测如下，取自 twix 的 ASCCONV 段。

| 受试者 | 线圈（自由运行 / ECG） | 分区 | 协议 | 判定 |
|---|---|---|---|---|
| V1 | 16 / 16 | 128 | 早期，ECG 只有 4 回波 | 排除 |
| V2 | 20 / 16 | 128 | 早期，两次扫描线圈数不一致 | 排除 |
| V3 | 24 / 24 | 112 | 2P0MM | 主力 |
| V4 | 24 / 24 | 112 | 2P0MM | 主力 |
| V5 | 16 / 16 | 112 | 2P0MM | 低线圈备用 |
| V6 | 20 / 20 | 112 | 2P0MM | 扩样本 |
| V7 | 24 / 24 | 112 | 2P0MM | 主力 |
| V8 | 24 / 24 | 112 | 2P0MM | 主力 |
| V9 | 20 / 20 | 112 | 2P0MM | 扩样本 |
| V10 | 20 / 20 | 112 | 2P0MM | 扩样本 |

统一采集参数：自由运行 8 回波，ΔTE 2.05 ms，首个回波 1.25 ms，22 段乘 1000 交错约 22,000 条辐条，每段 TR 374.44 ms，翻转角 12 度，375 秒；ECG 触发 5 回波，12 乘 437，每段 TR 130.44 ms，325 到 427 秒。矩阵 112 的三次方，FOV 220 mm，2 mm 各向同性，1.5T Siemens。

## 5. 归档进项目 3D 数据根

项目的 3D 数据根经本仓库 configs 与 scripts 实测确认是 `/data1/data1_congcong/data/3D/`，那里对 `T2_csm_ksp_n4_mat` 有 48 处引用，对 `T1W_FLAIR_AF12_bundle` 有 30 处。

```
/data1/data1_congcong/data/3D/cardiac_freerunning_multiecho_5D/
  README.md      选型理由、参数表、读取要点
  params.json    四例完整采集参数
  V03/ V04/ V07/ V08/
      freerunning_8echo.dat    自由运行 8 回波
      ecgtrig_5echo.dat        同人 ECG 触发 5 回波
      PhysioInfo.mat           pilot tone 与心动、呼吸信号
      SOURCE.txt               各自指向的原始路径与 Zenodo 记录号
```

三个数据文件都是软链接，指向 `/data2` 上已过 MD5 校验的原件。链接目标 43.2 GB，`/data1` 实占 64 KB。这样做是因为 `/data1` 已用到 76%，而 `/data2` 还有 22 TB。要实体拷贝，`cp -rL V07 V07_real`。

## 6. 动手前必须解决的两件事

1. **非笛卡尔。** 这是 3D 径向叶序采集，现有笛卡尔流水线要接 NUFFT 才能用，回顾性欠采变成按辐条抽取。这是切到心脏的主要工程成本，与选哪一套无关。
2. **没有全采样参考。** 自由运行径向本身就是欠采样，参考只能是用全部辐条做的 CS 或 XD-GRASP 重建。PSNR 口径必须重新定义，不能沿用脑部那套以全采样为真值的门限。

## 7. 读取坑

- 每个 `.dat` 内含 2 个 measurement，第一个是 `AdjCoilSens` 预扫，成像数据是最后一个，读取时取 last。
- `PhysioInfo.mat` 的 `TimeStamp_ms` 与 `respSignal` 长度都是 176,000，等于 22,000 辐条乘 8 回波，与采集线一一对应；`PTsignals` 是 748,871 乘 16 的每线圈 pilot tone，采样率更高。
- twix 文件末尾按 512 字节对齐补零，实际大小比头部 off 加 len 大不到 512 字节，属正常，不要当成截断。
- 读取与轨迹代码 `github.com/QIS-MRI/ReadDataAndTrajectory_FreeRunningFatFractionHeart`，自门控提取 `github.com/QIS-MRI/SG_RespiratoryCardiac_3DRadial`。

## 引用与许可

两套均为 CC-BY-4.0。

- Mackowiak ALC, Roy CWR, Yerly J, et al. Motion-resolved fat-fraction mapping with whole-heart free-running multiecho GRE and pilot tone. Magn Reson Med. 2023. doi 10.1002/mrm.29680
- Safarkhanlo Y, Yerly J, Falcão MBL, et al. Comparison between fast-interrupted steady-state and rapid water-excitation pulses for fat signal suppression in free-running whole-heart MRI at 1.5 T. MAGMA. 2025. doi 10.1007/s10334-025-01273-z
