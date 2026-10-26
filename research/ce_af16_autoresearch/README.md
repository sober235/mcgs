# MC-GS 平扫／增强 AF16：研究归档（2026-10-26）

本目录归档官方 [Karpathy autoresearch](https://github.com/karpathy/autoresearch) 实验循环在 proposed MC-GS 上的项目适配。固定现有病例数据、AF16、mask、CSM、42线圈、十个完整对照，只用 PSNR、SSIM、NMSE 决策。

当前累计 **32 次正式 MRI 训练、65 次历史重评分**；整脑六项全面领先尚未实现，双扫描同时达标仍为 **0/59**。当前keeper仍为 `lowreg_f075_solve_v1`，最小六项裕量 -0.111181285329。旧弱对照的“59/59领先”已撤回。

| 方法 | 平扫PSNR | 平扫SSIM | 平扫NMSE | 增强PSNR | 增强SSIM | 增强NMSE |
|---|---:|---:|---:|---:|---:|---:|
| 当前MC-GS keeper |34.836644|0.941252470|0.009868666|37.143680|0.953854978|0.009632337|
| CNN512候选，已拒绝 |34.804084|0.940984786|0.009942933|37.195325|0.954201937|0.009518470|
| ZS-SSDU完整对照 |35.319498|0.946334600|0.008830251|37.527185|0.957845211|0.008818226|

## 最新可核实进展

- CNN128和CNN512正式候选均已完成固定评分及独立结果审计，未超过keeper，全部负证据保留。
- 复频率载波算子真实GPU验证通过：14个数值案例及非法输入检查完整，非连续输入和非默认流均通过；独立审计0项问题。
- 载波纯二进制加载器和原训练循环接入runtime已实现，30项合成CPU检查通过，独立源码/质量审查0项问题。
- **临床协议/子进程适配、真实G2零步与两类短程控制、正式carrier500尚未完成。** 算子与CPU通过不代表重建质量改善。

## 导航

- [最新CNN512正式结果](codex_artifacts/mcgs_ce_karpathy_20261022_relocation/docs/caseprior512_outcome.md) 与 [独立结果审计](codex_artifacts/mcgs_ce_karpathy_20261022_relocation/docs/caseprior512_formal_outcome_audit.md)。
- [载波GPU验收](codex_artifacts/mcgs_ce_karpathy_20261022_relocation/docs/carrier_operator_outcome_20261026.md) 与 [实际通过报告](codex_artifacts/mcgs_ce_karpathy_20261022_relocation/carrier_verification/20261026T063900Z-7a66d90c3cf1/microgate.json)。[首轮失败报告](codex_artifacts/mcgs_ce_karpathy_20261022_relocation/carrier_verification/20261023T011403Z-bc86c383e901/microgate.json) 保持failed。
- [已审查runtime实现](codex_artifacts/mcgs_ce_karpathy_20261022_relocation/docs/carrier_mri_runtime_implementation.md)、[独立审查](codex_artifacts/mcgs_ce_karpathy_20261022_relocation/docs/carrier_mri_runtime_review.md)、[后续实验设计](codex_artifacts/mcgs_ce_karpathy_20261022_relocation/docs/superpowers/specs/2026-10-26-carrier-mri-design.md)。
- [完整进展记录](codex_artifacts/mcgs_ce_karpathy_20261019/docs/progress.md)、[三指标合同](codex_artifacts/mcgs_ce_autoresearch_20261017/reassessment_without_hfen_20261019/metric_contract_v1.json)、[文件哈希清单](MANIFEST.json)。

## 归档范围与复现边界

`codex_artifacts/`按原研究目录名保存代码、历史数值源码以及小型评分、完成记录和日志，以保留报告间的相对链接；`evidence/development/`保存开发与审查证据。清单记录每个文件的来源、版本和SHA256，路径映射保留原研究目录身份。

MRI体数据、重建数组、模型检查点、训练目标和本地.so二进制为外部运行产物；本目录记录其已有哈希/位置，不含这些载荷。代码为来源保持的研究快照，包含原机器绝对路径和既有环境假设，不是独立数据集或一键安装包。执行命令须在对应病例数据、G2、原binary和GSTMI环境就绪且通过来源校验后使用；重编译的新binary需重新验证。

本病例体积256×232×176，沿用历史固定vdPoisMex AF16 mask，readout沿axis0重复，中心24×24 ACS全采；这是既有固定mask的已披露例外，不能冒充BART新生成mask。参考为CS-SENSE AF5.33，沿用参考派生共享归一化，属于单病例自适应开发。训练未使用评分参考、ROI或外部预训练权重。

原仓库主示例的T1/FLAIR设置与本CE平扫/增强归档有各自数据和轴向约定，复现时以本归档合同与实际输入为准。
