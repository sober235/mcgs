# STATUS — 2026-09-12 16:20（分支 `peds-tsc-mtp`，worktree `/data0/congcong/code/GS/mcgs_peds_wt`，未推送）

本分支 = 儿童 TSC（age16）UIH 3D MTP 数据线。课题目录 `/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_3D/`
（README 是结果的权威记录，仓库里 `peds_tsc_mtp_3d/` 是它的镜像）。设计：
`docs/superpowers/specs/2026-09-12-peds-tsc-mtp-grappa-design.md`。

## 1. 已完成且已验证
- `uih_mtp/uih_raw.py`：UIH RAWDATA-V004 读取器（头索引、按 (set, echo, u1) 选读、去读出过采样）；
  `uih_mtp/grappa.py`：3D GRAPPA（PE 2 倍、24 条 ACS、椭圆包络、部分傅里叶区留空）+ 藏线自检 + 留一自检。
  测试 `tests/test_uih_mtp_raw.py`（真实文件：135084 成像行、11257 采样位置、DC 在 (152, 52)）与
  `tests/test_uih_mtp_grappa.py`（8 线圈体模：图像 NRMSE < 5 %、采样线逐位不变）共 8 个，全过；基线 183 个测试全过。
- 第一阶段跑完（09-12 12:47，GPU 2）：FA16/TE3.25 与 FA4/TE27.41 各得补齐 k 空间 `data/<contrast>/kspace_grappa.h5`
  （coil, RO, PE, SPE = 32 x 320 x 304 x 104）、掩膜、RSS 图、图两张。
  ```
  cd /data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_3D/scripts && bash run_stage1.sh --overwrite
  # logs/run_stage1.log 摘录
  # 12:45:57 GRAPPA: 8657 positions filled (284400 calibration equations, 3200 unknowns), 8 s
  # 12:45:58 ACS hold-out check: relative k-space error on the hidden odd lines 0.0964 (2 s)
  # 12:46:59 ACS hold-out check: relative k-space error on the hidden odd lines 0.1880 (2 s)
  # FA16_TE3.25 corr_grappa 0.9099 (zero-filled 0.9056); FA4_TE27.41 corr_grappa 0.9638 (zero-filled 0.9413)
  ```
- 1b 跑完（16:08）：ESPIRiT 图（房规 `ecalib -t 0.001 -c 0.0 -r 24 -m 1`，全分辨率，SoS=1）存 `data/<contrast>/csm_espirit.h5`（key `map`）；
  SENSE 合成 `sense_grappa.npy`；CG-SENSE（40 步，GPU）`cgsense.npy`。两者头内复数 NRMSE 0.078 / 0.196，幅值差在 1 体素高斯平滑后降到
  0.010 / 0.046，差异图无解剖结构 → 两种经典重建一致到噪声量级。厂商相关 0.912 / 0.962（与 RSS 相同：单图 ESPIRiT 的 SENSE 合成保留线圈亮度分布，
  差距来自厂商的强度归一化与部分傅里叶重建，不是并行成像步骤）。
- 噪声地板（重复的 echo 0，`logs/noise_floor.json`）：藏线自检的地板 FA16 0.067、FA4 echo4 0.144，GRAPPA 在地板的 1.3–1.4 倍。
- 核大小留一验证（`logs/grappa_kernel_loo_FA16_TE3.25.json`）：8 种设置 0.094–0.102，默认 (4, 2, 2, 1e-3) = 0.097，核不是瓶颈，不改。

## 2. 待用户拍板（第二阶段设计）
- 参考 = GRAPPA 补齐的多线圈 k 空间 + ESPIRiT SENSE 合成图（建议），还是 CG-SENSE 图。
- 回顾性掩膜相对什么定义：相对采集支撑（椭圆包络内的 227×104 带，AF 按支撑计，与 AF16 课题的 FLAIR 口径一致）还是相对整格 304×104。
- 加速倍数档位（例如 4 / 8 / 12 相对支撑）、掩膜类型（vdPoisson 同 AF12/AF16 课题、calib 24）、两对比度同掩膜还是各自掩膜。
- 部分傅里叶区与椭圆角作为已知零区：只给 GS 臂（沿用旧课题的不对称）还是所有方法都用。
- 方法表范围：全套（CG-SENSE、CS-SENSE、Joint SENSE、L1-ESPIRiT、ZS-SSDU、3D-DIP、3D-INR、GS-single、MC-GS）还是先做经典 + GS。

## 3. 下一步
- 第二阶段（回顾性欠采样 + 方法表）：用户拍板上面五条后写 spec，再执行；数据入口 = `data/<contrast>/kspace_grappa.h5` + `csm_espirit.h5` + 回顾性掩膜，
  和 CE / AF12 课题相同的 (coil, RO, PE, SPE) 布局。
- 其余 8 个对比度一条命令补跑（`prep_mtp_grappa.py --fa F --echo E`，每个 ecalib 约 25 分钟 CPU）。

## 4. 已知的坑与别重做
- raw 的翻转角在 user index 1（u1），不在 set；set 0 是 echo 0 的重复读出；ROCenter 160 是按 320 网格计的。
- raw 图像轴到厂商 DICOM：SPE 要翻转并取中间 80 层，PE 两侧各裁 2 像素，RO 不动。
- `pydicom` 只在 nvgen 环境有；GSTMI 环境跑重建，DICOM 先用 `vendor_dicom_to_npy.py` 导成 npy。
- 只用 GPU 2 且必须先看 `nvidia-smi --query-compute-apps`；GPU 1 是用户自己的 nnUNet 训练，0 与 3–7 是其他重建线。
- 长任务 `setsid nohup` 脱离，等待用 `kill -0 PID`；工具 shell 是 zsh，cwd 会被重置，路径一律写绝对路径。

## 5. 关键决定的"为什么"
- 主重建选 GRAPPA 而非 ESPIRiT-SENSE：要看的是补齐的 k 空间，采样线不改动，输出直接是第二阶段要的多线圈 k 空间。
- 部分傅里叶区不填：参考 k 空间里只留测量值和并行成像插值，不掺合成高频。
- 核不调：留一误差随核几乎不变，且离噪声地板只有 1.4 倍。
