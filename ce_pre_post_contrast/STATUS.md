# 平扫/增强（CE pre/post）课题交接 — 2026-09-16

新会话从这里开始：先读本文件，再读 `README.md`（完整协议与历史）。关键数字先重跑再动手，重跑结果与记录不符以重跑为准。

分支 `Plain_Enhanced` 自带脚本与归档（结果体数据不在 git，在服务器）。数据与派生结果的服务器位置见文末"文件地图"。

---

## 1. 已完成且已验证

命令都能直接粘贴重跑；图缓存命中时不碰 k 空间也不用 GPU。

```bash
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
E="PYTHONNOUSERSITE=1 PYTHONPATH="
T=/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast
cd $T
env $E $PY scripts/ce_lesion_candidates.py slices     # 选层 + 三面图
env $E $PY scripts/ce_lesion_candidates.py enhance    # 参考区数值 + 掩膜 npz + 教学图
env $E $PY scripts/ce_figure_paper.py --gpu -1        # 正式三面对比图（6 行）
env $E $PY scripts/ce_figure_paper.py --gpu -1 --diff # 强化版（9 行，每面加 增强−平扫）
env $E $PY scripts/ce_figure_paper.py --gpu -1 --zoom # 放大对比（病灶/血管/背景）
env $E $PY scripts/ce_figure.py --gpu -1              # 误差图，三个面各一张 + 5× 放大版
env $E $PY scripts/ce_edge_profile.py --lesion_thr 0.65   # 病灶边界
env $E $PY scripts/export_ce_derived.py               # 派生结果落盘 + md5
```

**病灶与选层。** 本例无标注、无报告。自动找强化（`analyze`/`blobs`）只抓到血管、静脉窦、脉络丛；病灶是在横断位拼图上看出来的：图像左侧上部半球，T1 低信号、分叶状，中心有更暗的核心。`slices` 取框内最大暗连通块（6039 mm³，质心 z202/y125/x48，范围 z190–217、y105–140、x33–60），按核心截面最大选层：**横断 z=201、冠状 y=131、矢状 x=48**。可替换层：z 199–203、y 130–135（另一叶 y=122）、x 47–54。NIfTI 体素坐标 (i,j,k) = (48,131,201)。

**所有对比图已统一到这三层**（用户 09-15/16 要求）。`ce_figure_paper.py --slices` 与 `ce_figure.py --slices` 默认 `201,131,48`；旧的 170/162/86 版本保留为 `*_l1spirit_z170_20260915`，其冠状/矢状来自"最强 post−pre 体素"，不过病灶。误差图现按面各出一张。

**病灶不强化**（白质=1，`enhance` 阶段）：

| 区域 | 增强 − 平扫 |
|---|---|
| 正常白质 | +0.004 |
| 血管/静脉窦 | +2.858 |
| 脉络丛 | +1.087 |
| 正常皮层 | +0.105（皮层小血管与部分容积，非漏出）|
| 病灶核心 | +0.021 |
| 核心外 0–3 mm | +0.028 |

对侧镜像区反而更高（0.046/0.061/0.035），病灶对周围的对比度平扫 0.303、增强 0.291（略降）。全脑按"平扫像正常组织、增强后变亮"筛：218 个峰里团块状且深在的只剩 1 个，放大看是血管；这类信号最多的一层 z=129（761 mm²）全是颅底血管与静脉窦。**所以不存在"平扫不明显、增强明显"的层。**

**各方法的强化保真**（白质单位，`--diff` 图每列下方也有）：

```
              血管   脉络丛  皮层     白质   病灶核心
ZF           +1.81  +0.52  +0.099  +0.007  +0.019
CG-SENSE     +2.80  +1.10  +0.103  +0.002  +0.022
CS-SENSE     +2.84  +1.08  +0.105  +0.003  +0.021
Joint SENSE  +2.81  +1.04  +0.103  +0.003  +0.021
L1-SPIRiT    +2.86  +1.11  +0.101  -0.001  +0.021
ZS-SSDU      +3.02  +1.21  +0.110  +0.003  +0.023
3D-DIP       +2.60  +0.77  +0.109  +0.002  +0.057
3D-INR       +2.88  +1.17  +0.106  +0.004  +0.029
MC-INR       +2.86  +1.13  +0.102  +0.001  +0.022
GS-single    +2.92  +1.16  +0.108  +0.004  +0.021
MC-GS        +2.92  +1.15  +0.106  +0.003  +0.024
```

颅外背景均值（增强图，头掩膜膨胀 6 mm 之外，白质单位）：3D-DIP 0.878、3D-INR 0.427、MC-INR 0.122、ZS-SSDU 0.024、CG 0.013、ZF 0.010、L1-SPIRiT 0.009、GS-single 0.009、**MC-GS 0.008**、CS/Joint 0.005。

**病灶边界：没有方法更好。** 固定外缘（CS-SENSE 阈值 0.65，37.7 mL）后，等效宽度除 ZF 3.13 mm、3D-DIP 2.71 mm 外全是 2.57–2.61 mm，20–80% 宽 1.55–1.69（ZF 2.72），平扫/增强 50% 穿越点只差 0.12–0.19 mm。MC-GS 2.59/1.61/0.18 与 CS-SENSE、GS-single 同级。看起来更锐的臂都是更噪的臂：边界梯度÷白质高频全在 1.02–1.15。

**落盘与推送。** 派生结果包 `export/ce_enhancement_20260916/`（64 文件 329 MiB，md5 64/64）。GitHub `sober235/mcgs`：`main`=5616333（脚本）、`ce-pre-post-contrast`=581a578（归档）、tag `handoff/2026-09-16-ce-enhancement`、本分支 `Plain_Enhanced`（脚本+归档+本文件）。

---

## 2. 待用户拍板

1. 9 行的强化版（`ce_pair_recon_3planes_enhancement`）要不要取代 6 行版成为正式图？现在两版并存，6 行版仍是 figures of record。
2. 正文图是否砍列（建议 ZF / CS-SENSE / 3D-DIP / 3D-INR / MC-INR / GS-single / MC-GS 七列，其余进补充材料）。
3. 要不要跑随机种子零假设实验（见下一步第 1 条），这是把"3D-DIP 造假强化"写成结论的前提。
4. 是否开回顾性验证线（BMSR/PDGM），这决定论文能不能写"在 R 倍加速下保住诊断对比度"。

## 3. 下一步

1. **种子零假设**：同一份平扫数据，换随机种子各跑两次 DIP、INR、GS-single、MC-GS，量 pre−pre 在病灶核心与血管区的量级，得到"方法自身随机性"的地板。DIP 的 +0.057 是否真是假强化，取决于这个地板。占 GPU，需用户点头。
2. **回顾性验证**：`/data2/congcong/data/FM_data` 的 UCSF-BMSR（461 例 T1pre/T1post + 分割，病灶强化约 2.1×）与 UCSF-PDGM（501 例 T1/T1c + 肿瘤分割，强化区约 6.7×）。幅值 NIfTI，需要自己做回顾性欠采仿真。指标定成强化对比度误差、假强化率、边界 Dice/Hausdorff。
3. 论文写法已拟好四条可写、四条不可写，见 README 与本文件第 5 节。

## 4. 已知的坑（别重做）

- **自动找强化在本例无效**：`analyze`（k·MAD 阈值）和 `blobs`（Hessian）返回的全是血管、静脉窦、脉络丛。别再调阈值，病灶不强化。
- **阈值分割会漏**：`lesion_mask_thr0.55/0.6.npy`、`lesion_measure_cssense.json` 漏进脑脊液并跨中线（82–199 mL），已作废，留档不要用。
- **核心边界不能当病灶边界**：它落在一段渐变里，所有方法都会量出同样的 ~8 mm。
- **阈值不加脑掩膜**会把整个头（800 万体素）当成病灶。
- **`distance_transform_edt` 在 |d|<1 内没有体素**，直方图中间几个 bin 天生为空，界面要挪 0.5。
- **`tight_layout` 遇到定宽高比图像会让标签压到下一行**，出图用 `layout="compressed"`。
- **MC-INR 在冠状/矢状残差图上有横条纹**：它沿读出方向逐层独立拟合，不是数据问题。
- **两次扫描共用同一掩膜**：相干欠采伪影在相减时部分抵消，所以差值图不能用来判断单幅图像质量。
- **NIfTI 含面部**，不能推 GitHub；只在服务器包里。

## 5. 关键决定的"为什么"

- **为什么所有图统一到病灶层**：原来的 170/162/86 来自"最强 post−pre 体素"，那是血管，不过病灶。对比方法的图应当落在临床上要看的地方。
- **为什么差值行要显示但数值不平滑**：差值 = 真强化 + 两次扫描的生理差 + 两次重建噪声之和（噪声按 √2 放大），不平滑就被噪声盖住；数值用未平滑的全体素统计，避免平滑影响结论。
- **为什么用白质单位**：两次扫描各按自身白质中位数归一，去掉全局尺度差，使"某区强化了多少"可跨方法比较。代价是全脑整体的轻微强化被吃掉，所以读作"相对白质的强化"。
- **为什么静态增强只能说漏出和血池**：单个时间点一个方程，三个未知量（vp、Ktrans、ve），且强化程度还依赖注药到扫描的延迟（本例未记录）。定量灌注/通透性要 DSC、DCE 或 ASL。
- **MC-GS 能写什么**：在"无训练先验"这一类里，它同时做到保住血管强化（+2.92，仅次于 ZS-SSDU 的 +3.02）、不给不强化的病灶造假（+0.024，对 3D-DIP +0.057、3D-INR +0.029）、背景与经典方法同级（0.008，对 3D-DIP 0.878）。**不能写**"重建质量最好"（留出 kres 第七，落后 GS-single 0.0017）、"联合优于单扫"（只能写 not worse）、"提高病灶检出"（无标注无读片实验）、任何灌注措辞。

## 6. 文件地图

```
服务器课题根   /data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast
  README.md                      完整协议、结果表、坑
  scripts/                       全部脚本（与本分支同步）
  lesion_candidates/             SynthSeg 标签、pn/qn、强化图、掩膜、候选表
  figures/                       所有图（与 ~/figs/ce_pair 同步）
  export/ce_l1spirit_20260915/   11 臂重建体 + 原始 k 空间 + CSM（15.05 GiB）
  export/ce_enhancement_20260916/ 本轮派生结果（329 MiB，md5 全过）
看图          http://localhost:8765/ce_pair/<图名>.png       （服务 ~/figs）
GitHub        sober235/mcgs（私有）：main / ce-pre-post-contrast / Plain_Enhanced
```
