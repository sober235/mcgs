# T1+FLAIR 加速倍数研究：测试集全量运行 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 6 次测试扫描 × 4 个加速倍数上用冻结设置跑 11 个方法，产出逐病例结果表、统计汇总和一张汇总图。

**Architecture:** 仓库目录 `autoresearch/t1flair_af_testset_20261121/` 里放五个脚本，全部经 `gpu.py run` 在服务器执行。服务器输出只写到 `OUT`；每个（扫描，加速倍数）是 `OUT/topics/<scan>_af<AF>/` 下的一个课题副本，脚本从 jiangyizhou 同一加速倍数的课题克隆，数据用符号链接。每一行运行用认领目录防止重复，可断点续跑。

**Tech Stack:** 服务器 GSTMI Python（torch、h5py、numpy、scipy、matplotlib）、BART `ecalib`、MC-INR 用 GR_pt1.10 环境；本机只用 `/usr/bin/python3 ../gpu-tools/gpu.py`。

**Spec:** `docs/superpowers/specs/2026-11-21-t1flair-af-testset-design.md`

## Global Constraints

- 测试集：`vol1_huangyuwei`、`vol2_jiangdian`、`vol4_tangbing`、`vol5_sunxinyi`、`vol6_jiangdian_790`、`vol7_tangbin_790`；不含 `vol3_jiangyizhou`。
- 加速倍数：标称 12 / 16 / 20 / 24（净 11.71 / 15.19 / 18.56 / 24.00×）。
- `OUT = /data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_t1flair_af_testset_20261121`。不写 `Results_20260908`，不覆盖已有文件，不改任何源码工作树。
- 测试病例上不调参：λ、掩膜、配方、早停规则全部取自 jiangyizhou 的登记运行（数值见设计文档"冻结的设置"）。
- 真值只用于评分；偏移不校正。
- GPU：只经 `gpu.py run --gpu ID` 启动，每卡一个作业；启动前查两次 `nvidia-smi --query-compute-apps`，间隔约 75 秒；不动别人的作业。
- 时间：不读取、不记录服务器上的任何时间或耗时；脚本日志不打印时钟；各方法 json 里的 `seconds` 字段不进入任何表或报告。
- 回复和文档用中文；提交只在用户要求时做，不自动推送。
- 大的 PNG 放在 `figures/`（带 `.gitignore`），否则 gpu.py 的快照会超限。

## Review Focus

1. **克隆脚本里残留的 jiangyizhou 路径**：某个脚本仍读或写原课题。期望：克隆后对全部脚本 grep，除 `mcgs_af16_wt` 代码路径和只读的掩膜来源外，不许出现 `Results_20260908`；Task 3 的测试覆盖。
2. **同一行被两个作业同时跑**：两张卡拿到同一行。期望：认领目录 `mkdir` 原子创建，后到者跳过；Task 2 的自检覆盖。
3. **作业中途被杀**：认领留下、结果没有。期望：`--status` 列出"已认领、无结果"的行，`--release ROW` 手动释放后重跑，绝不自动覆盖；Task 2 覆盖。
4. **FLAIR 采样范围不同的扫描（vol6、vol7 为 0.780）**：`known_zero` 和 FLAIR 的有效加速倍数与其他扫描不同。期望：数据准备把每次扫描的采样范围和每张掩膜下的有效采样点数写进 `prep.json`，统计表原样列出；Task 1 覆盖。
5. **某个方法在某一格失败或发散**：期望：该格记为缺失并在表里标出，名次和胜负只在 11 个方法齐全的格子上算，另列缺失清单；Task 5 的自检覆盖。

---

### Task 1: 公共定义与数据准备

**Files:**
- Create: `autoresearch/t1flair_af_testset_20261121/ts_common.py`
- Create: `autoresearch/t1flair_af_testset_20261121/prep_cases.py`

**Interfaces:**
- Produces（`ts_common.py`）：
  - `OUT: Path`，`SCANS: dict[str, dict]`（键为上面 6 个扫描名；值含 `person`、`t1`、`flair` 文件名、`group` = `"aligned"` 或 `"offset"`、`reuse` = 已有课题数据目录或 `None`；vol1 → `T1FLAIR_AF12_retro/data`，vol4 → `T1FLAIR_AF12_v4_tangbing/data`）。
  - `AFS = (12, 16, 20, 24)`；`JYZ_TOPIC(af) -> Path`；`MASK(af) -> Path`（jiangyizhou 课题的 `data/mask_af{af}_3d.h5`）。
  - `FROZEN: dict[int, dict]`：每个加速倍数的 `cssense = {"T1": λ, "FLAIR": λ}`、`jcssense = {"T1": λ, "FLAIR": λ}`、`spirit = {"T1": (5, λ), "FLAIR": (5, λ)}`，数值照设计文档的表。
  - `sha256(path) -> str`；`claim(root: Path, name: str) -> bool`（`mkdir` 原子认领）；`data_dir(scan) -> Path` = `OUT/data/<scan>`。
- Produces（`prep_cases.py`）：`OUT/data/<scan>/{ksp_T1.h5, ksp_FLAIR.h5, csm_espirit_T1.h5, csm_espirit_FLAIR.h5, prep.json}`；h5 的键和轴顺序与 jiangyizhou 课题相同（`kspace` / `map`，线圈在前）。

- [ ] **Step 1:** 写 `ts_common.py`，带 `--selftest`：断言 6 个扫描、4 个人、对齐组 3 个和偏移组 3 个；`FROZEN` 的每个值与 jiangyizhou 各课题 `logs/summary.json` 里 `cssense_o200` / `jcssense` / `spirit` 行的 `lam`、`ksize` 逐项相等；四张掩膜存在。
- [ ] **Step 2:** `gpu.py run --cpu -- @python autoresearch/t1flair_af_testset_20261121/ts_common.py --selftest`，期望末行 `SELFTEST_OK`。
- [ ] **Step 3:** 写 `prep_cases.py --scans ...`：`reuse` 不为空的扫描建符号链接，并核对其 k 空间与源 CFL 在抽查的 3 个线圈上逐值相等；其余扫描从 CFL 导出 k 空间、跑 `bart ecalib -t 0.001 -c 0.0 -r 24 -m 1`（T1、FLAIR 各一次，可并行）、转 h5。写 `prep.json`：每个对比度的采样范围、DC 位置、灵敏度图 SoS 的最小 / 最大值、四张掩膜下的有效采样点数与净加速倍数、各文件 sha256。已有目录时拒绝覆盖。
- [ ] **Step 4:** 运行 `gpu.py run --cpu -- @python .../prep_cases.py`；期望每个扫描一行 `PREP_OK <scan>`，SoS 在 [0.99, 1.01]，T1 采样范围 1.0，AF12 掩膜下 T1 净加速 11.71。
- [ ] **Step 5:** 取回 `prep.json` 存到 `autoresearch/t1flair_af_testset_20261121/prep/`。

### Task 2: MC-GS 与 GS-single 的运行脚本

**Files:**
- Create: `autoresearch/t1flair_af_testset_20261121/run_gs.py`

**Interfaces:**
- Consumes: `ts_common` 的 `OUT`、`SCANS`、`AFS`、`MASK`、`data_dir`、`claim`、`sha256`。
- Produces: 行名 `<scan>:<af>:<joint|singleT1|singleFLAIR>`，实验名 `ts_<scan>_af<af>_<kind>`；结果在 `OUT/trainer/results_arc/<exp>_score.json` 和 `<exp>_volumes.npz`（训练器自己的格式，键 `plain` / `contrast` 下有 `head_psnr`、`head_ssim`、`nmse`）。
- 做法沿用 `autoresearch/t1flair_af_indep_single_20261121/indep_single.py`：训练器副本取自 `mcgs_sg2x2_wt`（HEAD `54f8a2a3bef402e785314b748f730a9623ca9d43`）并逐文件核对；每个加速倍数的配方从 jiangyizhou 登记运行的 `<exp>_score.json` 的 `argv` 读出（AF12：`mcgs_af12_wt/results_arc/p3_e0001_ptv001_full4k` 与 `p3_e0001_ptv001_single{c}_full4k`；AF16：`mcgs_af16_wt/results_arc/p3_r3_e0001_ptv003_full4k` 与 `p3_r3_single{c}_full4k`；AF20 / AF24：`af{af}_p3_joint_full4k` 与 `af{af}_p3_single{c}_full4k`），去掉 `--exp`、`--gpu`、`--base_cfg` 后原样重放。
- 配置：读登记运行所用的 yaml（joint 和两个 single 各自的 `--base_cfg`），只把 `data.*.data_path`、`csm_path` 换成 `data_dir(scan)` 下的文件，`mask_path` 保持 `MASK(af)`，其余字段不动；写到 `OUT/configs/`。

- [ ] **Step 1:** 写 `run_gs.py`，参数 `--rows ...`、`--smoke`（30 次迭代，写到 `OUT/smoke`）、`--status`、`--release ROW`、`--selftest`。`--selftest`（CPU）断言：24 × 3 = 72 行；每个加速倍数重建出的三个配置与登记 yaml 只在数据路径上不同；认领同一名字第二次返回 `False`；`--status` 把"有认领、无 score"的行列为 `stale`。
- [ ] **Step 2:** `gpu.py run --cpu -- @python .../run_gs.py --selftest`，期望 `SELFTEST_OK`。
- [ ] **Step 3:** 冒烟：查空卡后 `gpu.py run --gpu N -- @python .../run_gs.py --smoke --rows vol1_huangyuwei:20:joint vol1_huangyuwei:20:singleFLAIR`，期望两行都以退出码 0 结束并写出 score。
- [ ] **Step 4:** 口径核对：`--rows` 支持一个只读对照行 `jyz:20:joint`（数据取 jiangyizhou 课题），正式跑一次，期望头部 PSNR 与登记值 34.832 / 34.179 相差小于 0.005 dB。不满足就停下报告，不继续 Task 4。

### Task 3: 课题副本与对比方法的运行脚本

**Files:**
- Create: `autoresearch/t1flair_af_testset_20261121/make_topics.py`
- Create: `autoresearch/t1flair_af_testset_20261121/run_arms.py`

**Interfaces:**
- Consumes: `ts_common` 全部；Task 1 的数据目录。
- Produces（`make_topics.py`）：`OUT/topics/<scan>_af<af>/{scripts,data,logs,recon}`；`scripts/` 是 `JYZ_TOPIC(af)/scripts` 里 `t1t2_common.py`、`recon_classical.py`、`recon_joint_cssense.py`、`recon_spirit.py`、`recon_zsssdu.py`、`recon_dip_inr.py` 的副本，只把课题路径字符串替换成副本路径；`data/` 是指向 `data_dir(scan)` 四个文件和 `MASK(af)` 的符号链接。另建只读对照 `OUT/topics/jyz_af20/`（数据链接到 jiangyizhou 课题）。
- Produces（`run_arms.py`）：行名 `<scan>:<af>:<arm>`，`arm` ∈ `classical`（ZF、CG-SENSE，T1 和 FLAIR）、`cssense`、`jcssense`、`spirit`、`zsssdu`、`dip`、`inr`、`mcinr`；每个方法的结果是副本 `logs/<method>_<contrast>*.json`（字段 `head_psnr`、`head_ssim`、`nmse`）和 `recon/<method>_<contrast>_complex.npy`。命令行参数照 `run_classical_af20.sh`、`run_spirit_sweep.sh`、`run_stage2_af20.sh` 里的写法，λ 取 `FROZEN[af]`；`jcssense` 在 T1 与 FLAIR 的 λ 不同时跑两次，各取对应对比度。
- MC-INR：入口 `/data0/congcong/code/GS/mcinr_niessen2025/scripts/run_volume.py`，环境 `/home/congcongliu/anaconda3/envs/GR_pt1.10/bin/python`；先读该仓库的 `scripts/run_volume.py` 和 `score.py` 的参数，输出目录指到副本下的 `mcinr/`，再用副本的 `Pair.score` 评分（做法照 `score_mcinr_arm.py`）。该仓库只读。

- [ ] **Step 1:** 写 `make_topics.py`，带检查：副本脚本里 `grep Results_20260908` 的命中数为 0；每个符号链接可解析；掩膜 sha256 与 `MASK(af)` 相同。已存在的副本内容不同则报错。
- [ ] **Step 2:** `gpu.py run --cpu -- @python .../make_topics.py`，期望 `TOPICS_OK 25`（24 个加对照）。
- [ ] **Step 3:** 写 `run_arms.py`，参数与 `run_gs.py` 相同（`--rows`、`--status`、`--release`、`--selftest`）；`--selftest` 断言 24 × 8 = 192 行、每行的命令里 λ 等于 `FROZEN`。
- [ ] **Step 4:** 口径核对：查空卡后跑 `jyz:20:classical`，期望 CG-SENSE 的头部 PSNR 与 jiangyizhou AF20 课题 `logs/summary.json` 的登记值相差小于 0.005 dB（FLAIR 登记值 32.089）。不满足就停下报告。
- [ ] **Step 5:** MC-INR 冒烟：`vol1_huangyuwei:12:mcinr` 只跑 2 个层面（若入口支持）或整卷跑一次，确认评分路径可用。

### Task 4: 正式运行

**Files:**
- Create: `autoresearch/t1flair_af_testset_20261121/RUNLOG.md`（本机时钟记录每次启动的作业号、卡号、行）

**Interfaces:**
- Consumes: Task 2、Task 3 的行名。

- [ ] **Step 1:** 用 ListAgents 看其他会话；空卡按规则查两次。每张空卡启动一个作业，`--rows` 给它一串行；顺序为扫描 vol1 → vol2 → vol4 → vol5 → vol6 → vol7，每次扫描内先 `joint`、`singleT1`、`singleFLAIR`，再 `classical`、`cssense`、`jcssense`、`spirit`，最后 `zsssdu`、`dip`、`inr`、`mcinr`。
- [ ] **Step 2:** 用 `--status`（CPU 作业）跟进；有 `stale` 行先查原因再 `--release`。每完成一次扫描，跑一次 Task 5 的汇总，把中间表给用户看。
- [ ] **Step 3:** 全部 72 + 192 行有结果后进入 Task 5。失败且重试一次仍失败的行记入缺失清单，不改设置。

### Task 5: 汇总、统计与图

**Files:**
- Create: `autoresearch/t1flair_af_testset_20261121/collect_stats.py`
- Create: `autoresearch/t1flair_af_testset_20261121/figures/.gitignore`（内容 `*.png` 和 `*.pdf`）
- Create: `autoresearch/t1flair_af_testset_20261121/RESULTS.md`

**Interfaces:**
- Consumes: Task 2 的 `_volumes.npz` 与 score，Task 3 的 `recon/*.npy` 与 json，副本的 `t1t2_common.Pair`（取真值和头部掩膜）。
- Produces: `table.json`（每格 11 个方法 × PSNR / SSIM / NMSE，全部用同一段评分代码从体积重算）、`stats.json`、`table.md`、`figures/testset_summary.png`。
- 统计函数（纯函数，放在同一文件里）：
  - `rank_of_mcgs(cell: dict[str, float], higher_is_better: bool) -> int`
  - `paired_summary(diffs: list[float]) -> dict`，返回 `mean`、`sd`、`min`、`n_positive`、`n`、`wilcoxon_p`（精确双侧）
  - `per_person(values: dict[str, float]) -> dict[str, float]`，把同一人的两次扫描取平均
  - `slice_tests(mcgs, other, gt, head) -> dict`，层面选取规则照 `fig_af_overview.py` 所读的 `slice_trend_*.json` 的生成脚本（先读 `T1FLAIR_AF12_16_20_24_trend_v3_jiangyizhou/scripts/` 确认规则，原样采用）

- [ ] **Step 1:** 写统计函数和 `--selftest`：`paired_summary([0.1]*6)` 给 `n_positive == 6`、`wilcoxon_p == 0.03125`；`paired_summary([0.1]*4)` 给 `0.125`；`rank_of_mcgs` 在并列和缺失方法时的行为（缺失 → 该格标 `incomplete`，不计名次）；`per_person` 把 vol2 与 vol6、vol4 与 vol7 合并成 4 个值。
- [ ] **Step 2:** `gpu.py run --cpu -- @python .../collect_stats.py --selftest`，期望 `SELFTEST_OK`。
- [ ] **Step 3:** 写汇总主体：重算的 MC-GS 头部 PSNR 与训练器 score 相差超过 0.005 dB 就报错；输出按 6 次扫描、按 4 个人、对齐组、偏移组四种口径的统计；缺失清单。
- [ ] **Step 4:** 画汇总图（先按 dataviz 技能定配色和形式）：每个加速倍数一栏，每次扫描一个点，纵轴是 MC-GS 减 GS-single 和 MC-GS 减最强非 GS 方法的头部 PSNR 差，T1 与 FLAIR 分面，零线明显；对齐组与偏移组用不同标记。
- [ ] **Step 5:** 取回表和图，写 `RESULTS.md`（中文）：结果表、统计、限制（4 个人、1 个种子、ZS-SSDU 5 块、GS-single 登记版、偏移未校正、层面不独立）。MC-GS 没有第一的格子逐个列出，不省略。
- [ ] **Step 6:** 把 PNG 在 VS Code 里打开，按"先图后表"的规则向用户报告；提交与否问用户。
