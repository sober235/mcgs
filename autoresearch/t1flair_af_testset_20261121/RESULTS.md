# 测试集 T1 + FLAIR 不同加速倍数：结果（草稿，部分数据）

> **草稿：只含已完成的结果。** 完整格子（扫描 x 加速倍数 x 对比度，11 个方法齐全）0 / 48；两种对比度都完整的（扫描 x 加速倍数）0 / 24。不完整格子不计入名次与统计。全部跑完后由 `autoresearch/t1flair_af_testset_20261121/collect_stats.py --pipeline` 重新生成本文件、table.json、stats.json、table.md 与图。

口径：头部掩膜内（|GT| > 0.05 P99.9）的 PSNR / SSIM / NMSE；评分代码是各课题共用的 `t1t2_common.Pair.score`，真值来自该扫描自己的全采样 k 空间，所有数字都由保存的体积重新计算，再与各运行脚本记录的分数对照（见第 6 节）。NMSE 的差值写成相对降低（%），其余为绝对差；大于 0 表示 MC-GS 更好。

## 1. MC-GS 在完整格子里是否第一

| 对比度 | AF | 完整格子 | PSNR 第一 | SSIM 第一 | NMSE 第一 |
|---|---|---|---|---|---|
| T1 | 12 | 0 / 6 | 0 / 0 | 0 / 0 | 0 / 0 |
| T1 | 16 | 0 / 6 | 0 / 0 | 0 / 0 | 0 / 0 |
| T1 | 20 | 0 / 6 | 0 / 0 | 0 / 0 | 0 / 0 |
| T1 | 24 | 0 / 6 | 0 / 0 | 0 / 0 | 0 / 0 |
| FLAIR | 12 | 0 / 6 | 0 / 0 | 0 / 0 | 0 / 0 |
| FLAIR | 16 | 0 / 6 | 0 / 0 | 0 / 0 | 0 / 0 |
| FLAIR | 20 | 0 / 6 | 0 / 0 | 0 / 0 | 0 / 0 |
| FLAIR | 24 | 0 / 6 | 0 / 0 | 0 / 0 | 0 / 0 |

合计（完整格子 0）：PSNR 第一 0，SSIM 第一 0，NMSE 第一 0。

### MC-GS 不是第一的格子（逐个列出，不省略）

（目前的完整格子里没有。）

## 2. 病例间：MC-GS 相对两个对照的差值

对照：GS-single；该格最强的非 GS 方法（按该格头部 PSNR 选，三个指标都与同一个方法比）。“符号一致”= 差值大于 0 的单位数 / 单位总数。p 为精确双侧 Wilcoxon 符号秩检验（单位数 4 时 p 最小 0.125，6 时 0.031，见第 5 节）。

### PSNR差值 dB

| 对比度 | AF | 对照 | 单位 | 均值 | 标准差 | 最小 | 符号一致 | p |
|---|---|---|---|---|---|---|---|---|

### SSIM差值 

| 对比度 | AF | 对照 | 单位 | 均值 | 标准差 | 最小 | 符号一致 | p |
|---|---|---|---|---|---|---|---|---|

### NMSE差值 %（相对降低）

| 对比度 | AF | 对照 | 单位 | 均值 | 标准差 | 最小 | 符号一致 | p |
|---|---|---|---|---|---|---|---|---|

## 3. 病例内：逐层面配对检验

每个完整格子内，三个平面里头部占层面 >= 5 % 的层面合并（规则同 `t1flair_af_record_fig_20261120/fig_af_overview.py` 所读的 slice_trend 记录），逐层面配对 Wilcoxon（双侧）。下表给每个（对比度，AF，对照，指标）下 p < 0.05 且 MC-GS 的中位差为正的扫描数，以及各扫描里 MC-GS 更好的层面占比的范围。层面属于同一个体积，彼此不独立，p 值只说明该病例内的一致性，不是人群推断。逐病例的全部数值在 stats.json 的 `slices`。

| 对比度 | AF | 对照 | 指标 | p<0.05 且为正 / 扫描数 | 更好层面占比范围 |
|---|---|---|---|---|---|

## 4. 缺失清单

不完整格子 48 / 48。每个（扫描，AF）缺的方法（两种对比度合并；逐格原因见 table.json 的 `missing`）：

- vol1_huangyuwei AF12：ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol1_huangyuwei AF16：ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol1_huangyuwei AF20：L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol1_huangyuwei AF24：L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol2_jiangdian AF12：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol2_jiangdian AF16：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR
- vol2_jiangdian AF20：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR、GS-single、MC-GS
- vol2_jiangdian AF24：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR、GS-single、MC-GS
- vol4_tangbing AF12：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR、GS-single、MC-GS
- vol4_tangbing AF16：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR、GS-single、MC-GS
- vol4_tangbing AF20：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR、GS-single、MC-GS
- vol4_tangbing AF24：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR、GS-single、MC-GS
- vol5_sunxinyi AF12：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR、GS-single、MC-GS
- vol5_sunxinyi AF16：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR、GS-single、MC-GS
- vol5_sunxinyi AF20：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR、GS-single、MC-GS
- vol5_sunxinyi AF24：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR、GS-single、MC-GS
- vol6_jiangdian_790 AF12：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR、GS-single、MC-GS
- vol6_jiangdian_790 AF16：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR、GS-single、MC-GS
- vol6_jiangdian_790 AF20：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR、GS-single、MC-GS
- vol6_jiangdian_790 AF24：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR、GS-single、MC-GS
- vol7_tangbin_790 AF12：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR、GS-single、MC-GS
- vol7_tangbin_790 AF16：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR、GS-single、MC-GS
- vol7_tangbin_790 AF20：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR、GS-single、MC-GS
- vol7_tangbin_790 AF24：ZF、CG-SENSE、CS-SENSE、Joint SENSE、L1-SPIRiT、ZS-SSDU、3D-DIP、3D-INR、MC-INR、GS-single、MC-GS

## 5. 必须写明的限制

- 只有 4 个人。以人为单位的 Wilcoxon 检验即使全胜，精确双侧 p 最小为 0.125；以 6 次扫描为单位最小为 0.031，但其中两次是重复扫描（vol2 与 vol6 是同一个人，vol4 与 vol7 是同一个人），不独立。结论写成“几个里几个一致”。
- 层面级检验的层面彼此不独立（同一体积、三个平面共用数据），只说明病例内的一致性。
- 一个种子。
- ZS-SSDU 是 5 个残差块的版本（各课题登记的实现）。
- GS-single 是登记版：同一对比度送入联合训练器的两个通道（登记的 singleT1 / singleFLAIR 配置），取第一个通道的结果。
- 偏移组（vol5、vol6、vol7）按采集到的样子跑，没有做偏移校正；对齐组与偏移组分开列出。
- 所有设置取自调参病例（jiangyizhou）的登记值，没有在测试扫描上调参；真值只用于评分。

## 6. 重算分数与运行脚本记录的分数

每个体积都用同一段评分代码（AF16 克隆的 `t1t2_common.Pair.score`，CPU）重算，再与运行脚本记录的值比较。最大绝对差：

| 量 | 最大绝对差 | 出现在 |
|---|---|---|
| 头部 PSNR（dB） | 0.00011 | vol2_jiangdian AF12 T1 GS-single |
| 头部 SSIM | 5.36e-07 | vol1_huangyuwei AF24 FLAIR Joint SENSE |
| 整体 NMSE | 2.44e-07 | vol2_jiangdian AF16 T1 GS-single |

各方法的最大 |头部 PSNR 差|（dB）：ZF 3.1e-06，CG-SENSE 3.4e-06，CS-SENSE 3.2e-06，Joint SENSE 3.2e-06，L1-SPIRiT 2.8e-06，GS-single 0.00011，MC-GS 0.00011

## 7. 图

汇总图：`autoresearch/t1flair_af_testset_20261121/figures/testset_summary.png`（PNG / PDF 只放在 git 忽略的 figures 文件夹，不提交）。每个加速倍数一栏，每次扫描一个点，纵轴是头部 PSNR 差（MC-GS 减对照）；零线为界；圆点 = 对齐组，三角 = 偏移组。
