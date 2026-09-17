# jiangyizhou AF16 测量加噪鲁棒性研究总结（2026-09-16 至 09-17）

分支 `jyz-af16-noise`（从 `jyz-af16-diffmask` bbba53b 切出），工作树 `mcgs_noise_wt`。
课题目录 `Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou/`（2σ0）和 `Results_20260908/T1FLAIR_AF16_noise1x_v3_jiangyizhou/`（1σ0），
仓库镜像 `t1flair_af16_noise2x_v3_jiangyizhou/` 和 `t1flair_af16_noise1x_v3_jiangyizhou/`。
零噪声一列直接用 `T1FLAIR_AF16_v3_jiangyizhou` 的记录，没有重跑。
英文详细记录 `docs/superpowers/records/2026-09-17-jyz-af16-noise.md`，交接 `docs/handoff/2026-09-17-jyz-af16-noise.md`，现状 `STATUS.md`。

## 1. 结论

- 冻结的 AF16 三期配方扛不住 2σ0 的测量噪声。头内 PSNR 掉 5.9 dB（T1）和 5.0 dB（FLAIR）。十二个方法里，MC-GS 从六格全第一掉到 T1 第 8、7、5 名，FLAIR 第 9、9、7 名（依次是 PSNR、SSIM、HFEN）。
- 1σ0 下，PSNR 和 HFEN 基本守住，SSIM 守不住。FLAIR 头内 PSNR 仍是第一，比 GS-single 高 0.007 dB。T1 排第二，比收敛 TV 低 0.003 dB，可以看作持平。HFEN 两个对比度都第一，SSIM 两个对比度都第五。
- 主因是 GS 渲染把噪声拟合进了图像。MC-GS 的锐度比从零噪声的 0.855 / 0.804，升到 1σ0 的 0.958 / 0.899，再到 2σ0 的 1.331 / 1.227。经典方法则对着真值重选 λ，选中的 λ 是零噪声时的 1.4 到 4 倍。
- 硬数据一致性不是主因。同一批体积去掉这一步再打分，头内 PSNR 只多 0.02 到 0.06 dB。
- 假如经典方法也冻结在零噪声的 λ，1σ0 下 MC-GS 的 T1 头内 PSNR 会领先 0.066 dB（对 ZS-SSDU），FLAIR 领先 0.007 dB，但 SSIM 仍落后 ZS-SSDU。2σ0 下仍落后 ZS-SSDU 3.3 / 2.4 dB。这几条是从现有表格读出来的，没有另跑。
- 联合增益（MC-GS 减 GS-single 的头内 PSNR）在 T1 上随噪声变大（+0.148、+0.217、+0.232），在 FLAIR 上变小并转负（+0.042、+0.007、−0.116）。

## 2. 实验设计

唯一变量是测量数据。受试者、k 空间、ESPIRiT 灵敏度图、泊松盘掩膜（2966 个 PE 点，T1 15.19×）、真值、尺度、打分代码、各方法设置和 MC-GS 配方都与 AF16 课题相同。真值和尺度取自干净 k 空间，各方法拟合的是带噪测量。

噪声模型：

```
每个实际采集的样本加复高斯噪声，线圈间、样本间独立；FLAIR 厂商视野外保持为 0
sigma      = k × sigma0_k            k = 1 或 2
sigma0_k   = sigma0_img / sqrt(f)    f = 该对比度采集的 PE 面比例
sigma0_img = 干净真值空气区的噪声（P99.9 = 1 的口径）

T1     sigma0_img 0.01408   f 1.0000   sigma0_k 0.01408
FLAIR  sigma0_img 0.01282   f 0.7577   sigma0_k 0.01472
```

只抽一次噪声，种子 20260916。1σ0 的噪声正好是 2σ0 的一半，所以两档只差噪声幅度。生成器的六项检查两档都通过，1σ0 另加一项嵌套检查（最大比值 0.29，限值 1）。

代码改动：
- 训练器加可选项 `data.<channel>.measurement_path`。测试改动前 206 个通过，改动后 211 个通过。
- `arc/cycle.py --save_render`，额外保存未做硬数据一致性的渲染体。
- MC-INR 仓库的 `run_volume.py` 加 `--ksp_T1 / --ksp_FLAIR`（本地提交 726a0b6，不推送）。
- 课题脚本把真值和测量分开读。口径检查：把测量指回干净文件，CG-SENSE 九项指标与 AF16 课题记录的差都是 0。

每档 57 个作业。经典方法的 λ 对着真值重选（整个系列一直这样做），MC-GS 配方不动。2σ0 下有三个最优 λ 落在网格上界，上扩两档后最优值没变。1σ0 下所有最优 λ 都在网格内部。

## 3. 主结果

头内 PSNR（dB），数字取自 `t1flair_af16_noise1x_v3_jiangyizhou/logs/compare_noise_levels_head_psnr.md`。

T1

| arm | no noise | 1 sigma0 | 2 sigma0 | 1 sigma0 − none | 2 sigma0 − none |
|---|---|---|---|---|---|
| MC-GS | 35.706 | 34.016 | 29.802 | -1.690 | -5.904 |
| GS-single | 35.558 | 33.799 | 29.571 | -1.759 | -5.987 |
| ZS-SSDU | 34.150 | 33.950 | 33.131 | -0.200 | -1.019 |
| CS-SENSE (TV, converged) | 34.945 | 34.018 | 32.766 | -0.927 | -2.179 |
| Joint SENSE (JTV) | 34.831 | 33.935 | 32.697 | -0.896 | -2.134 |
| CS-SENSE (35 outer) | 34.409 | 33.730 | 32.227 | -0.679 | -2.183 |
| L1-SPIRiT | 33.788 | 32.772 | 31.453 | -1.016 | -2.335 |
| CG-SENSE | 32.354 | 31.613 | 29.924 | -0.741 | -2.430 |
| MC-INR | 31.590 | 29.713 | 25.034 | -1.877 | -6.555 |
| 3D-DIP | 30.994 | 30.944 | 30.951 | -0.050 | -0.043 |
| 3D-INR | 30.375 | 29.538 | 28.698 | -0.837 | -1.677 |
| ZF | 25.841 | 25.839 | 25.835 | -0.001 | -0.005 |

FLAIR

| arm | no noise | 1 sigma0 | 2 sigma0 | 1 sigma0 − none | 2 sigma0 − none |
|---|---|---|---|---|---|
| MC-GS | 34.928 | 33.635 | 29.946 | -1.293 | -4.982 |
| GS-single | 34.886 | 33.628 | 30.062 | -1.258 | -4.824 |
| ZS-SSDU | 34.005 | 33.289 | 32.348 | -0.715 | -1.657 |
| CS-SENSE (TV, converged) | 34.302 | 33.556 | 32.526 | -0.746 | -1.776 |
| Joint SENSE (JTV) | 34.355 | 33.602 | 32.553 | -0.753 | -1.802 |
| CS-SENSE (35 outer) | 33.806 | 33.345 | 32.172 | -0.461 | -1.635 |
| L1-SPIRiT | 33.720 | 32.848 | 31.636 | -0.872 | -2.084 |
| CG-SENSE | 32.826 | 32.055 | 30.297 | -0.771 | -2.530 |
| MC-INR | 32.243 | 30.734 | 26.507 | -1.509 | -5.735 |
| 3D-DIP | 30.650 | 30.637 | 30.611 | -0.013 | -0.039 |
| 3D-INR | 30.257 | 29.900 | 29.389 | -0.356 | -0.868 |
| ZF | 27.879 | 27.876 | 27.868 | -0.002 | -0.010 |

MC-GS 与最强对比方法的头内 PSNR 余量（GS-single 算对比方法）：

| contrast | level | MC-GS | strongest comparison arm | margin |
|---|---|---|---|---|
| T1 | no noise | 35.706 | GS-single 35.558 | +0.1481 |
| T1 | 1 sigma0 | 34.016 | CS-SENSE (TV, converged) 34.018 | -0.0026 |
| T1 | 2 sigma0 | 29.802 | ZS-SSDU 33.131 | -3.3283 |
| FLAIR | no noise | 34.928 | GS-single 34.886 | +0.0417 |
| FLAIR | 1 sigma0 | 33.635 | GS-single 33.628 | +0.0073 |
| FLAIR | 2 sigma0 | 29.946 | Joint SENSE (JTV) 32.553 | -2.6063 |

MC-GS 在十二个方法中的名次：

| contrast | level | head PSNR | head SSIM | head HFEN |
|---|---|---|---|---|
| T1 | no noise | 1/12 | 1/12 | 1/12 |
| T1 | 1 sigma0 | 2/12 | 5/12 | 1/12 |
| T1 | 2 sigma0 | 8/12 | 7/12 | 5/12 |
| FLAIR | no noise | 1/12 | 1/12 | 1/12 |
| FLAIR | 1 sigma0 | 1/12 | 5/12 | 1/12 |
| FLAIR | 2 sigma0 | 9/12 | 9/12 | 7/12 |

头内 SSIM、HFEN 的完整表格见英文记录第 3 节。

## 4. 图

两档各五张图，1σ0 的文件名前缀是 `t1flair_v3_af16n1_`，2σ0 是 `t1flair_v3_af16n_`。

- 噪声示意图（干净全采图、带噪全采图、两者之差）：
  - `http://localhost:8765/t1flair_v3_af16_noise1x/t1flair_v3_af16n1_noise_data.png`
  - `http://localhost:8765/t1flair_v3_af16_noise2x/t1flair_v3_af16n_noise_data.png`
- 正式三面图（180 / 124 / 76 加 48 体素放大框，另有 600 dpi PDF）：
  - `http://localhost:8765/t1flair_v3_af16_noise1x/t1flair_v3_af16n1_T1_row3planes_compact_180_124_76_inset48_headerr_p3_mcinr_spirit_zf_caps.png`
  - `http://localhost:8765/t1flair_v3_af16_noise1x/t1flair_v3_af16n1_FLAIR_row3planes_compact_180_124_76_inset48_headerr_p3_mcinr_spirit_zf_caps.png`
  - `http://localhost:8765/t1flair_v3_af16_noise2x/t1flair_v3_af16n_T1_row3planes_compact_180_124_76_inset48_headerr_p3_mcinr_spirit_zf_caps.png`
  - `http://localhost:8765/t1flair_v3_af16_noise2x/t1flair_v3_af16n_FLAIR_row3planes_compact_180_124_76_inset48_headerr_p3_mcinr_spirit_zf_caps.png`
- 174 / 110 / 47 版本：把上面文件名里的 `180_124_76_inset48` 换成 `174_110_47`。

图片文件在 `/home/congcongliu/figs/t1flair_v3_af16_noise1x/` 和 `/home/congcongliu/figs/t1flair_v3_af16_noise2x/`，各课题的 `figures/` 里也有一份。误差图一律是 `|recon − GT| / max(GT)`，jet，范围 [0, 0.1]，只显示头内。

## 5. 局限

- 只有一个受试者，只抽了一次噪声，没有量抽样间的波动。
- 加的是白噪声，线圈间独立；扫描仪自身的噪声在线圈间是相关的。
- 灵敏度图和真值都是干净的，只有 L1-SPIRiT 的核用到了带噪的校准区。真值本身带着扫描仪的本底噪声。
- 经典方法对着真值选 λ，MC-GS 用冻结配方，这对经典方法有利。第 1 节“假如经典方法也冻结”那一条给出了这部分的大小。
- 零噪声一列的 JTV 只扫了两个 λ（0.0007 和 0.001），它的零噪声值是下界。零噪声下它在任何一格都不是最强对比方法。
- 1σ0 下 T1 头内 PSNR 的名次只差 0.003 dB。配方在干净数据上的种子间差异是 0.001 dB，带噪数据上没有测。

## 6. 待定

- 在带噪数据上重选 GS 配方的正则强度（与经典方法同等待遇），是最直接的下一步。09-16 用户选的是先加 1σ0 一档，没有做这一臂。
- AF16 课题 `_zf_caps` 正式图的一行说明文字超出了图宽，两个噪声课题已改成三行，AF16 那边没动，是否改由用户定。
- 多抽几次噪声或加一个受试者，可以给 1σ0 的名次加上误差范围。
