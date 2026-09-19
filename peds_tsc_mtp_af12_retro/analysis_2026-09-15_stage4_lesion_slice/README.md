# 四期：病灶层面的 MC-GS 结果（2026-09-15 / 16）

儿童 TSC（16 岁）UIH 3D MTP，回顾性 AF 12.00（31616 点里采 2635 点），两个对比度 T1w = FA16/TE3.25、T2s = FA4/TE27.41。
本目录是四期（病灶层面）的全部产物：图、脚本、逐次运行的指标、以及出图用的 MC-GS 重建体。
课题主表、README 与打分流水线在上一级目录，由三期会话维护；本目录只做加法，不改动它们。

## 1. 用户定的目标与判据

- 09-14：先要求 MC-GS 超过所有对比方法 1 dB；当晚改为**只需一张包含病灶的层面超过所有对比方法 1 dB**。
- 判据：逐层头内 PSNR，算法同课题的 `fig_row_final.py`（头 = |GT| > 0.05·P99.9，峰值 = 整卷 max|GT|）。
  **GS-single 算对比方法**，课题登记的每个 GS-single 版本都计入门槛。
- 09-15：**L1-SPIRiT 替换 L1-ESPIRiT**（与 jyz 线同日口径）；GS-single 仍算对比方法。
- 09-15 收尾：1 dB 达不到，**用户接受 +0.15 dB**，四期结束，不再跑新的调参。
- 09-16：图里加**放大框**，并在 GT 之后加**零填充**列。

## 2. 结果

推荐做法（运行名 `p4_r3_edge0005`）：三期工作点 + 并集边缘 TV（`loss.edge_tv_union_guide`，引导阈值 ×0.5）+ 共同边缘 0.0005。
并集边缘 TV 是这条线上唯一单对比度臂无法使用的机制：它只在 T1w 有边缘处**减少** T2s 的平滑，别处不加。

T2s 病灶层面，对全部对比方法（含 GS-single）的余量，括号内为不算 GS-single：

| 层面 | 病灶 | MC-GS | GS-single | 余量 |
|---|---|---|---|---|
| 矢状 172 | 右侧脑室壁结节 | 33.840 | 33.690 | +0.150（+1.19） |
| 横断 69 | 钙化样灶 | 33.574 | 33.450 | +0.124（+1.18） |
| 矢状 196 | 钙化样灶 | 34.397 | 34.292 | +0.105（+1.26） |
| 横断 45 | 结节 | 34.702 | 34.634 | +0.068（+0.82） |
| 冠状 165 | 钙化样灶 | — | — | +0.024（+1.55） |
| 冠状 148 | 结节 | 33.810 | 33.801 | +0.009（+1.31） |

T1w 上这两个病灶没有局灶改变；T1w 最强的联合运行是 `p3_n96_union_k05`（9.6M 点 + 并集边缘 TV，课题登记为 `gs_joint_p3f`），
在同样的层面上比同容量单臂低 0.001–0.05 dB，均排第二。

L1-SPIRiT（本次新建，脚本在 `scripts/recon_spirit.py`，50 个参数点）：两个对比度都是全 k 空间中心 24×24 校准更好，
T1w ks5 λ2.5e-4、T2s ks3 λ5e-4；主区域 T1w 38.310 / 0.9607 / 0.0679，T2s 32.017 / 0.8630 / 0.2264。
它接管了 T1w 的头内 HFEN 门槛（0.0679，低于 GS-single 的 0.0764）。

## 3. 目录内容

- `figures/`：九张图的 png，带放大框的四张另有 600 dpi 矢量 pdf。命名含层面、放大框、MC-GS 运行名。
  每张图：GT、零填充、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR、GS-single、MC-GS 一行排开，
  下排是误差图（jet，[0, 0.1]，只画头内），左下角是该平面的采样掩膜，病灶用红圈或橙色放大框标出。
- `scripts/`：出图与评分脚本（`p4_fig_*`、`p4_lesion_margin.py`、`p4_summary.py`）、调度（`p4_sched2.sh`、`p4_queue.txt`）、
  L1-SPIRiT 重建与扫描（`recon_spirit.py`、`spirit_sweep*.sh`）、四期的诊断脚本（噪声天花板、比值模型、误差按边缘分类）。
- `metrics/`：每次运行的主区域指标（`*_primary.json`）、病灶层面余量（`*_lesion.json`）、账本中的 p4 行、L1-SPIRiT 扫描日志。（目录名不用 `logs`，仓库的 .gitignore 有 `logs*/` 规则会把它挡掉。）
- `volumes/`：推荐做法的复数重建体（T1w、T2s 各一，complex64，320×304×104）。其余臂的重建体在课题的 `recon/`。
- `docs/`：四期记录与回合日志（每次运行做了什么、保留还是淘汰、为什么）。

## 4. 重画

```
cd /data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro
./scripts/py analysis_2026-09-15_stage4_lesion_slice/scripts/p4_fig_three_planes_inset.py \
    --contrast T2s --slices 69 165 196 --inset 48 48 16 --inset_center 166 196 69 \
    --mask_error --single_key gs_single_ar --mcgs_npz p4_r3_edge0005 --tag _focus --pdf
```
脚本默认从 `/data0/congcong/code/GS/mcgs_peds_wt/results_arc/<run>_volumes.npz` 读 MC-GS；该 worktree 若已清理，
把 `volumes/` 里的两个 npy 作为 MC-GS 输入即可（需改脚本的读取路径）。

## 5. 必须随数字一起说明

- 层面是按余量挑出来的，不代表整卷；图注里已写明。
- 单受试者，在本受试者上调参，没有留出检验。
- 参考图是 GRAPPA 补齐的 2 倍采集，不是全采样。
- 经典臂（CS-SENSE、Joint SENSE、L1-SPIRiT）的 λ 都在本受试者上扫过，与调参后的 GS 臂可比；ZS-SSDU、3D-DIP、3D-INR、MC-INR 没有调参。
- T2s 自身的正则权重两个 GS 臂都用的是迁移值 0.10/0.10，四期试过 0.07 更差。

代码与完整历史在 git 分支 `16yo-mtp`（worktree `/data0/congcong/code/GS/mcgs_peds_wt`），本目录的脚本是其副本。

---
**仓库镜像说明。** 这是服务器归档目录的镜像，不含 `figures/`（89 MB）与 `volumes/`（155 MB）。
两者在 `/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro/analysis_2026-09-15_stage4_lesion_slice/`，
校验值见 `MANIFEST.tsv`。图也在 `/home/congcongliu/figs/peds_tsc_af12_lesion_1db/`（本机 http://localhost:8765/peds_tsc_af12_lesion_1db/ 可看）。
