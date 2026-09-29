# 混叠伪影审计(2026-09-29)

起因:用户看过平扫/增强 AF16 的重建图后质疑"MC-GS 有很严重的混叠伪影"。本目录是从重建体重新做的审计,不是看图说话。
课题根 `$TOPIC` = `/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16`,重建体在 `$TOPIC/recon/`,
真值 `recon/gt_{pre,post}_complex.npy`,掩膜 `sim/mask/mask_af16_pe.npy`。数字全部在本目录的 `audit_*.json` 里。

## 1. 结论

1. **没有相干混叠。** 误差在 PE 平面上的自相关没有离心峰(最大离心值 0.08 在 lag 4,只是主瓣宽度);掩膜 PSF 没有远处旁瓣;
   头外背景 rms 是峰值的 1.6%,真值自己的噪声 1.0%,CS-SENSE / L1-SPIRiT 是 2.1%。
2. **但非相干混叠残差很重,11 个臂都重,MC-GS 只是最轻的。** "全部第一"不等于"干净"。README 只报了排名,没报残差绝对水平。
3. **根因之一是掩膜口径。** 这条线的掩膜由 `bart poisson -v` 生成,ACS 外密度几乎均匀(6%),r 0.10–0.18 处一道悬崖;
   jyz T1+FLAIR 线用的 vdPoisMex 掩膜密度平滑衰减。两条线的"AF16"不是一个实验。换掩膜能拉回多少,见 `../MASK_SWAP.md`。

## 2. 数字(头内口径,MC-GS)

| | 平扫 | 增强 |
|---|---|---|
| 头内 PSNR | 33.62 | 35.66 |
| rms 误差 / 峰值 | 3.1% | 3.1% |
| p99 误差 / 峰值 | 9.3% | 9.8% |
| 头内 NMSE | 1.3% | 1.3% |
| 采样点带 NMSE | 5.5e-4 | — |
| 残差能量落在未采样点的比例 | 98.6% | — |
| 残差与 ZF 残差的相关系数 | 0.49 | 0.53 |

分带 NMSE(平扫,归一化半径):r 0.1–0.2 为 1.6%,0.2–0.4 为 7%,0.4–0.7 为 15%,0.7–1.0 为 32%。
(`audit_spectrum.json`、`audit_bg.json`、`audit_residual_structure_{pre,post}.json`)

肉眼表现:白质里 4–8 体素尺度的斑块纹理,颅骨边缘一圈涟漪。涟漪在 ZF 残差里是同样的环纹,来源是 24×24 ACS 方块的截断振铃。
正式图的 1.5× 提亮会把这一层放大。

## 3. 掩膜口径对比(`audit_mask_compare.json`、`audit_mask_energy.json`)

| 归一化半径 | 0.03 | 0.12 | 0.18 | 0.23 | 0.33 | 0.43 | 0.6–1.0 |
|---|---|---|---|---|---|---|---|
| CE 掩膜(BART poisson -v) | 1.00 | 0.61 | 0.09 | 0.06 | 0.06 | 0.06 | 0.06 |
| jyz 掩膜(vdPoisMex 256×176) | 1.00 | 0.53 | 0.33 | 0.24 | 0.14 | 0.10 | 0.06→0.03 |

在这份真值上,r 0.2–0.3 带未测能量:CE 掩膜 93%,jyz 密度 79%;r 0.3–0.4:94% 对 87%。总未测能量 25.8% 对 24.3%
(真值 66% 的能量在 r<0.1,所以总量差得少,中频差得多)。

## 4. 图(`figures/`,看图 `http://localhost:8765/ce_retro_af16_audit/<名>.png`)

- `F_zoom_texture_{pre,post}_z201.png` 横断 201 放大:GT / ZF / L1-SPIRiT / GS-single / MC-GS,原窗、高通、带符号误差。
- `G_mask_compare_ce_vs_jyz.png` 两张掩膜、径向密度、PSF。
- `B_error_spectrum_pre.png` 各臂误差的 k 空间分布与 PE 平面自相关。
- `E_perslice_psnr.png` 三轴逐层头内 PSNR。
- `C_mask_psf.png` CE 掩膜 PSF。
- 全视野轴位与矢状/冠状 gamma 图(A_*、D_*,各 2.5 MB)没进 git,在 `/home/congcongliu/figs/ce_retro_af16_audit/`。

## 5. 复现

```bash
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
cd ce_retro_cssense_af16/audit_2026-09-29/scripts
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY audit2.py   # 全视野轴位、误差谱、自相关 -> audit_spectrum.json
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY audit3.py   # 矢状/冠状 gamma 图、背景与头内误差统计 -> audit_bg.json
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY audit4.py   # 逐层曲线、残差低通/高通与 ZF 相关 -> audit_residual_structure_*.json
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY audit5.py   # 放大纹理图
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY audit6.py   # 掩膜密度/PSF 对比 -> audit_mask_compare.json
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY audit7.py   # 掩膜未测能量 -> audit_mask_energy.json
```
脚本把图写到 `~/figs/ce_retro_af16_audit/`;`audit2/3/4/5` 通过 `arms.py` 定位重建体(尊重 `RETRO_TOPIC`),
`audit6/7` 里的路径是写死的。

## 6. 不能说的

只有一个受试者、一张掩膜、一个种子;真值是 TV 正则的 CS-SENSE 解,不是全采样。上面的相关系数 0.49/0.53 同时含
"没去掉的混叠"和"没补回来的高频",这个口径分不开两者。
