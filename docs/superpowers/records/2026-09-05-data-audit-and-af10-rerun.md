# 2026-09-04/05 — HYW T1+FLAIR 数据信息审计与 AF=10 复跑记录

分支 `diag-t1-artefact`。所有数字为实测，脚本与日志随本记录一起版本化。

## 1. 问题

AF≈12 的 MC-GS 重建（edge-TV 工作点 λ_geo 0.10 / tv_eps 0.04 / kspace_high 2.2 / κ_q 0.85 / w_floor 0.10，
ALIGNED vd-Poisson 掩膜）在 whole@max 口径已过 38 dB，但视觉上仍有颗粒与边缘不实。用户约束：只用欠采样数据，
不加外部数据。要回答两件事：(a) 剩余误差来自哪里；(b) 欠采 k 空间之外，数据里还有什么可用的辅助信息。

## 2. 审计方法

`scripts/diag_data_information_probe1.py`、`scripts/diag_data_information_probe2.py`（CPU，numpy 直接读
trainer `.mat` + ESPIRiT CSM + ALIGNED 掩膜；日志在 `probe_output/`（目录不能叫 logs，会被 .gitignore 的 logs*/ 吞掉））。自检：零填充锚 31.112 / 30.699 dB 逐位复现，
GT 与 `results_arc/*_volumes.npz` 中的 gt 相对差 1e-7。

## 3. 残差去向（`edgeTV_best_bboxfix`，AF 11.83，whole@max 38.48 / 38.19）

| 指标 | T1 | FLAIR |
|---|---|---|
| PSNR whole@max / head-only@max | 38.48 / 34.03 | 38.19 / 33.56 |
| NMSE whole / head | 2.03% / 1.54% | 0.86% / 0.64% |
| 残差在 3D 归一化半径 r>0.6 的份额 | 69% | 67% |
| 掩膜外 GT 能量已恢复 | 92.6% | 96.5% |
| 头内残差中不可避免噪声 | 5% | 6.5% |
| 白噪声天花板 whole / head | 44.6 / 46.9 dB | 43.0 / 45.4 dB |
| 背景残差中 GT 自身噪声占比 | 92% | ≥100% |

结论：不是噪声限，不是 bug；约 70% 的残差是 2 体素以下尺度的真实高频信号（该处 GT 能量为噪声 5–50 倍，
采样密度 3–8%，每线圈每样本 SNR 1–3），通用先验已恢复掩膜外能量的 92–97%，剩余部分不由数据决定。
whole@max 口径被 73–75% 的背景零体素抬高约 4.5 dB，报数应给 head-only。

## 4. 数据物理事实

- 原始 h5 只有 `kspace` 与厂商 `csm`，无噪声预扫、导航、多回波、头信息。
- **FLAIR 在相位编码面被厂商带限**：ρ 0.8–0.9 → 0.9–1.0 能量掉 27×，ρ≥1.0 ≈ 0；GT 噪声沿两个 PE 轴 lag-1
  自相关 0.45/0.49（读出轴 0.005）。重建在 ρ≥0.9 放的能量（0.103%）比 GT（0.095%）还多；ρ≥1.0 处残差占
  FLAIR 总残差 7.4%，ρ≥0.9 占 13.9%。把带外当已知零预计 +0.3~0.6 dB，且滤波形状可由已采样点的径向能量估计。
  同时说明 FLAIR 没有 ρ>0.9 的高频，跨对比度共享在原理上补不了 T1 的细节。
- T1 椭圆快门在 ρ≈1.28–1.35（PE 面 4.6%），收益 <0.2 dB。
- **T1 相位粗糙**：24×24 ACS 低分辨相位的虚部残差能量 18.9%（强信号体素 17.6%），48×48 仍 10–12% ⇒
  实值 / 共轭对称 / VCC 约束对 T1 不可行；FLAIR 0.7–1.3%，与当前误差同量级，也不值。
- T1 线圈噪声相关：|corr| 均值 0.17、最大 0.51，协方差特征值跨 10× ⇒ 预白化有理但 ≤0.2 dB。
- 头部支撑约束：背景残差已等于 GT 噪声，T1 +0.09 dB，FLAIR 0，无收益。
- 同次检查存在 T2（同扫描号 161015）：PE 面不带限，读出轴被强滤波，DC 在 (158,132,128)，用前需配准。

辅助数据可行性总表：FLAIR 带限已知零（值得做）> 噪声协方差预白化（小）> T1 快门（小）> 支撑约束（无）>
相位/共轭对称（不可行）。T1 细节缺口只能靠降 AF、换线圈或外部学习先验。

## 5. AF=10 复跑

命令（GPU 2，3001 s）：

```
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python -u arc/cycle.py --exp af10_joint_edgetv_e040_rerun20260904 --gpu 2 --iters 4000 \
  --lambda_geo 0.10 --warmup 500 --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 \
  --base_cfg configs/af10_joint_edgetv.yaml --save_volumes
```

| 口径 | AF12 tvs4k_e040 | AF10 2026-08-11 | AF10 复跑 2026-09-04 |
|---|---|---|---|
| T1 PSNR whole@max | 38.379 | 38.944 | 38.945 |
| T1 PSNR head@max | 33.922 | 34.604 | 34.605 |
| T1 SSIM / NMSE / HFEN | 0.906 / 0.0209 / 0.778 | 0.912 / 0.0182 / 0.731 | 0.912 / 0.0182 / 0.731 |
| FLAIR PSNR whole@max | 38.221 | 38.866 | 38.867 |
| FLAIR PSNR head@max | 33.624 | 34.400 | 34.401 |
| FLAIR SSIM / NMSE / HFEN | 0.910 / 0.0089 / 0.693 | 0.916 / 0.0074 / 0.631 | 0.916 / 0.0074 / 0.631 |

两次 AF10 重建体素级互 PSNR 59.7 dB（cuDNN 非确定性），指标层面可复现。AF12→AF10 两对比度 whole +0.56/+0.65 dB，
head-only +0.68/+0.78 dB；分区 rms 误差（% GT 峰值）：T1 深部 1.64→1.54、外壳 2.10→1.93；FLAIR 深部 1.90→1.74、
外壳 2.12→1.93；误差 >5% 峰值的体素 T1 2.43%→1.64%、FLAIR 2.84%→1.82%。评分 JSON：
`af10_joint_edgetv_e040_rerun20260904_score.json`。

## 6. 蝴蝶层（横断位 z=160，ROI 行 55:150 列 80:185，同 `scripts/fig_af12_butterfly_roi.py`）

| | 对比度 | 本层头内 PSNR 零填充 / 重建 | ROI PSNR | ROI NMSE | ROI 误差 >5% 体素 |
|---|---|---|---|---|---|
| AF12 | T1 | 25.83 / 34.55 | 35.45 | 0.68% | 1.11% |
| AF10 | T1 | 26.10 / 35.22 | 36.08 | 0.59% | 0.69% |
| AF12 | FLAIR | 24.49 / 33.36 | 33.13 | 0.45% | 3.70% |
| AF10 | FLAIR | 24.93 / 34.21 | 34.04 | 0.36% | 2.37% |

ROI 内误差集中在脑室壁与尾状核边缘（CSF–组织交界），腔内与白质为均匀低幅颗粒，无相位编码方向条带或重影。

## 7. 图与脚本

- `docs/figures/af10_rerun_20260905/butterfly_z160_{T1,FLAIR}.png`：`scripts/fig_butterfly_z160.py`
- `docs/figures/af10_rerun_20260905/AF10_rerun20260904_{T1,FLAIR}_3views.png`：`scripts/fig_3views_jet.py`
  （GT | 零填充 | 重建 | 误差 jet 固定 [0,0.1]；transverse v[:,:,128]、sagittal flipud(v[:,110,:].T)、
  coronal flipud(v[128,:,:].T)）
- 体数据（gitignored，本地）：`results_arc/af10_joint_edgetv_e040_rerun20260904_volumes.npz`
