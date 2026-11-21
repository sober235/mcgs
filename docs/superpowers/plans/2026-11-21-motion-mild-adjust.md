# MC-GS 应对轻微运动（三处小改动）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改 MC-GS 主框架的前提下加三处小改动（M1 配准给出扫描间变换、M2 T2 数据项按回波链加权、M3 持续偏移给一个状态位姿），并用 hyw 仿真数据检验它们能否应对扫描间位移和扫描内偶发的轻微运动。

**Architecture:** 新目录 `autoresearch/motion_mild_adjust_20261121/`（下称 `M`）放本版的常量、仿真入口、权重与状态算法、运行时补丁、启动器、评估和出图；第一版目录 `autoresearch/motion_adapter_sim_20261113/`（下称 `E1`）当库用（`geom`、`warp`、`lesions`、`schedule`、`simulate.build_custom`、`adapter`、`probe_reginit.register`、`evaluate` 的数组级函数），只做带默认值的追加式修改。训练仍是子进程在内存里给私有训练器副本打补丁后原样运行 `arc/cycle.py`。

**Tech Stack:** 服务器 GSTMI 环境（torch 2.7.1+cu118、numpy 1.26、scipy 1.15、h5py、pytest 9），现有编译好的 CUDA 体素化扩展，`gpu.py` 远程执行，本机队列 `E1/tools/queue.py`。

**Spec:** `docs/superpowers/specs/2026-11-21-motion-mild-adjust-design.md`（下称「设计稿」，用户 2026-11-21 批准）。第一版的设计稿、计划和 `E1/RESULTS.md` 是背景。本计划只写设计稿没定的东西。

## Global Constraints

- 工作树 `mcgs/.claude/worktrees/motion-adapter-sim`，分支 `exp/motion-adapter-sim-20261113`；不推送，除非用户要求。
- 服务器新根目录 `ROOT2=/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_motion_mild_adjust_20261121/`（`trainer/`、`data/<顺序>/<条件>/`、`configs/`、`reg/`、`eval/`、`figures/`、`ledger.jsonl`）。第一版根目录和课题数据只读。
- 第一版的文件只做追加式修改（新增带默认值的参数、新增函数），默认行为逐位不变；`E1` 的测试必须仍是 `84 passed, 2 deselected`。
- 训练器：`SRC=/data0/congcong/code/GS/mcgs_sg2x2_wt`，提交 `54f8a2a3bef402e785314b748f730a9623ca9d43`，复制到 `ROOT2/trainer` 并逐文件核对；不改、不重编译。配方：联合组 V32G，单对比度组 V32 去掉联合专属项（取 `E1/common.py` 的 `V32`、`V32G`）。种子 42。数据 hyw T1 + T2W，AF12。
- 运行：`/usr/bin/python3 E1/tools/rrun.py {--cpu | --gpu ID} -- @python <脚本>`（内部走 `gpu.py`，工作树私有配置）。GPU 只经它启动；启动前两次查空卡、间隔约 75 秒；一卡一作业；不动别人的作业；`cycle.py` 自己设 `CUDA_VISIBLE_DEVICES`，传物理卡号。
- 时间只用本机时钟。服务器上不运行 `date`、不读文件时间、不读 `run_info.json` 的 `timestamp_utc`、不读 `*_score.json` 的 `elapsed_s`。本版代码不打印、不写入任何时钟值或耗时。作业号内嵌服务器时钟，只记随机后缀。
- 真值图像、真值运动、ROI、病灶掩模不进训练和运动估计（真值对照组 Ow、Op 除外，表中标注）。评分器、掩模、对比方法不改。
- 设置只在开发条件（DTa、DTc、D2、C0）上定；调参训练不超过 10 次；每个组先过 C0 零运动关卡再冻结；冻结后才跑测试条件。
- 每个任务结束提交一次。大图放 `M/figures/big/`（自带 `.gitignore`）；提交的 PNG 为 200 dpi。

## 已核实的事实（2026-11-21 读源码）

1. 数据项 `JointTrainer._kspace_loss`：`diff = (fft3c(vol·csm) − ksp_meas)·mask`（再乘 `freq_weight`），返回 `Σ|diff|² / norm`。所以按点加权 w 等价于把掩模换成 `mask·sqrt(w)`，分母不变。
2. T2 密度初值只有一处：`d_c = self.zf_c[ix, iy, iz] * 0.1`，在未变换的高斯下标上取值。`zf_c` 另外只用于头部支撑和 `edge_kappa` 分位数。
3. `SingleContrastTrainer` 继承 `JointTrainer`，唯一通道叫 `plain`（属性后缀 `_p`）。
4. `probe_reginit.register(a=T1, b=T2, mask)` 返回 `x=(theta_deg[3], t_mm[3])`，`rigid(x)` 把标准帧（T1）的点送到它在 T2 图像里的位置，与适配器同向。C0 上它给出 hyw 自带的偏移（约 0.5 mm），不是 0。
5. 第一版的 `patch.kloss` 每次迭代调用 `apply_stage` 把适配器参数全部设为可训练，所以本版**不调用** `patch.install`，自己装挂钩。
6. `simulate.build_custom` 把同一位姿的回波链归为一组，每组做一次形变；非刚体场由 `warp.rbf_field(seed, peak_mm, brain, brain_max_mm, shape, sigma_mm)` 生成。

## 约定

- 坐标、变换方向、旋转命名与第一版计划相同（`T` 把标准帧的点 `r` 送到 T2 图像里的位置 `r′ = R(r + u(r)) + t`）。
- 顺序代号：`il` = 交错，`kz` = kz 线性。实验名 `mm_<af>_<顺序>_<条件>_<组>[_s<种子>][_<标签>]`。不涉及回波链的行（C0/E2 的 B0、B1、I-T2）也带 `il`，只是标签。
- 回波链按编号先后采集，第 n 条的时间比例 `n/(链数−1)`。参考链 = 含 (ky, kz) 中心点的那条；真值图像 = 参考链状态下的图像。
- 「受污染的回波链」：相对参考链，脑内平均位移 > `CONTAM_MM = 0.2`。

## 文件结构

| 文件 | 职责 |
|---|---|
| `E1/schedule.py`（改） | `build(..., order="linear")` 增加 `"interleaved"`；事件增加 `"pulse"`；新增 `soft_amp` |
| `E1/simulate.py`（改） | `build_custom(..., order="linear", soft_u=None)` |
| `E1/adapter.py`（改） | 新增 `set_inter`、`set_segment_pose` |
| `E1/evaluate.py`、`E1/run.py`、`E1/tools/queue.py`（改） | 路径相关函数加可选参数；队列加课题档案 |
| `M/mcommon.py` | 路径、条件、组、矩阵、阈值、命名；无第三方依赖 |
| `M/msim.py` | CLI：按 (条件, 顺序) 生成数据、真值、质检、清单 |
| `M/weights.py` | 回波链残差统计、稳健 z、权重、变化点、AUC（numpy，纯函数） |
| `M/align.py` | `canonical_zf`、`ngf_loss`（torch，纯函数） |
| `M/reg.py` | CLI：零填充 T1/T2 的刚体配准；状态间配准函数 |
| `M/mpatch.py` | 子进程入口：装挂钩，运行 `arc/cycle.py`，保存 `<exp>_mm.json` 与 `<exp>_canon.npz` |
| `M/mrun.py` | 启动器：核对训练器副本、写配置、起子进程、记账本；`--dry`、`--selftest` |
| `M/mevaluate.py` | 表、收回比例、权重检出、状态误差、判据；`--feas`、`--dev`、`--all` |
| `M/mfigures.py`、`M/tools/mtables.py` | 出图；由 `summary.json` 生成 Markdown 表 |
| `M/tests/` | `conftest.py`（把 `M` 和 `E1` 加入 `sys.path`）、`run_tests.py`（调用 `E1/tests/run_tests.py`，默认参数是本目录）、`test_*.py` |
| `M/README.md`、`M/experiments.jsonl`、`M/settings_dev.json`、`M/settings_frozen.json`、`M/summary.json`、`M/RESULTS.md` | 说明、本地账本、设置、结果 |

测试命令（下文写作 `PYTEST2 <文件>`）：`/usr/bin/python3 E1/tools/rrun.py --cpu -- @python M/tests/run_tests.py [M/tests/<文件>]`。第一版回归（写作 `PYTEST1`）：`... -- @python E1/tests/run_tests.py`，期望 `84 passed, 2 deselected`。

## 数值表（我定的；设计稿只给了范围）

**事件**（`theta_deg` 分量 = 转头、侧倾、点头；`soft` = 软组织场的幅度倍数）：

| 条件 | 扫描间 | 扫描内事件 | 病灶 |
|---|---|---|---|
| C0 | — | — | — |
| E2 | 第一版 E2 的数据（只读引用） | — | — |
| Ta | — | pulse at 0.40 dur 0.05，θ (0, 0, 1.0)，t (0.5, 0, 0)，soft 1 | — |
| Tb | — | pulse at 0.25 dur 0.04，θ (0.8, 0, 0)，t (0, 0.4, 0)，soft 1；pulse at 0.70 dur 0.04，θ (0, 0, −1.0)，t (−0.5, 0, 0)，soft 1；step at 0.74，θ (0, 0, 0.05)，t (0.1, 0, 0)，soft 0.1 | — |
| Tc | — | step at 0.60，θ (0, 0, 1.0)，t (0.5, 0, 0)，soft 1 | — |
| E2Tb | `_E1A` + `_NR_E2`（重新生成，因扫描内事件要叠在上面） | 同 Tb | — |
| TbL | — | 同 Tb | 第一版的病灶选点 |
| DTa（开发） | — | pulse at 0.55 dur 0.06，θ (0.9, 0, 0)，t (0, −0.5, 0)，soft 1 | — |
| DTc（开发） | — | step at 0.45，θ (0, 0.7, −0.6)，t (0, 0.4, 0.3)，soft 1 | — |
| D2（开发） | 第一版 D2 的数据（只读引用） | — | — |

软组织场：`warp.rbf_field(seed=23, peak_mm=2.0, brain, brain_max_mm=0.5, ...)`。

**组**：

| 组 | 训练器 | M1 | M2 | M3 |
|---|---|---|---|---|
| B0 | 联合 | — | monitor（只记录，不改损失） | — |
| B1 | 联合 | 开 | monitor | — |
| B2 | 联合 | — | adaptive | — |
| B12 | 联合 | 开 | adaptive | — |
| B123 | 联合 | 开 | adaptive | detect |
| Ow | 联合 | — | truth | — |
| Op | 联合 | 开 | truth（只对状态位姿解释不了的污染链） | truth |
| I-T2 | 单对比度 T2 | — | monitor | — |
| I-T2+w | 单对比度 T2 | — | adaptive | — |

**矩阵**（AF12，行写作 `顺序:条件:组`）：

- `feas`（3）：`il:DTa:B0`、`il:DTa:Ow`、`il:C0:B0`。
- `dev`（基础 6，标签 `d1`；其余 4 次留给重调）：`il:DTa:B2`、`il:DTc:B0`、`il:DTc:B123`、`il:DTc:Op`、`il:D2:B1`（对齐项开）、`il:D2:B1`（对齐项关，标签 `d1r`）。
- `gate`（7，同时是正式行）：`il:C0:{B1, B2, B12, B123, I-T2+w}`、`kz:C0:{B2, B123}`。
- `upper`（先跑）：`il:{Ta, Tb, Tc, E2Tb}:{B0, Ow}`、`il:Tc:Op`、`kz:{Ta, Tc}:B0`、`kz:Tc:{Ow, Op}`。
- `rest`：`il:C0:I-T2`；`il:E2:{B0, B1, B123, I-T2}`；`il:Ta:{B2, B123, I-T2}`；`il:Tb:{B2, B123, I-T2, I-T2+w}`；`il:Tc:{B2, B123, I-T2}`；`il:E2Tb:{B12, B123}`；`il:TbL:{B0, B123, I-T2}`；`kz:Ta:B123`；`kz:Tc:B123`。
- 正式行合计 43（含 `feas` 的 `il:C0:B0`）。

**起始设置** `M/settings_dev.json`：

```json
{"m1": {"align": {"start": 300, "end": 1500, "lr": 5e-3, "weight": 1.0}},
 "m2": {"z0": 4.0, "z1": 8.0, "cap": 0.25, "warmup": 600, "every": 100, "n_shells": 24},
 "m3": {"at": 1000, "min_frac": 0.10, "cp_t": 8.0, "max_states": 3, "init": "reg",
        "lrs": {"seg_theta": 5e-4, "seg_t": 1e-2}},
 "reg": {"ctrl_l2": 1e-3, "ctrl_smooth": 1e-2, "seg_l2": 0.0, "seg_smooth": 0.0}}
```

**阈值** `mcommon.THRESH2`：`feas_auc 0.95`、`feas_recover 0.5`、`min_loss_db 0.1`、`s1_psnr −0.05`、`s1_w1_frac 0.99`、`s2_tre_rigid 0.2`、`s2_tre_nonrigid 0.3`、`s2_dT2 −0.1`、`s2_dT1 −0.05`、`s3_recover 0.5`、`s3_recall 0.9`、`s3_false 0.02`、`s3_pose_deg 0.3`、`s3_pose_mm 0.3`、`s4_recovery_drop 0.05`、`kz_harm −0.05`。

## Review Focus

1. **监视与全 1 权重必须逐位不改变损失**：否则 B0 不再是「不改动的 MC-GS」，零运动关卡也失去意义。（任务 4 `test_weighted_mask_identity_bitexact`）
2. **没有采样点的回波链、全部回波链残差相同（MAD = 0）**：权重必须是 1，不能出现 NaN 或除零。（任务 3 `test_degenerate_trains_keep_weight_one`）
3. **零运动下误报「持续偏移」**：重尾噪声序列上变化点检测必须返回空。（任务 3 `test_changepoints_none_on_stationary_noise`）
4. **加权组的评分口径**：权重全 1、单状态时，`rec_w_dc` 的头部 PSNR 必须等于评分器自己的数值，否则加权组与 B0 不可比。（任务 4 自检 (b)）
5. **配准结果进入模型的方向**：`canonical_zf` 和 `set_inter` 的方向错了，T2 密度初值会偏两倍位移，却不报错。（任务 8 `test_canonical_zf_undoes_known_shift`、`test_set_inter_matches_set_truth`）

---

### Task 1: 采集顺序、事件类型、本版常量

**Files:**
- Modify: `E1/schedule.py`
- Create: `M/mcommon.py`、`M/tests/conftest.py`、`M/tests/run_tests.py`、`M/tests/test_order.py`、`M/tests/test_mcommon.py`、`M/README.md`（一段话说明 + 指向设计稿）

**Interfaces:**
- Produces:
  - `schedule.build(support, k_seg=K_SEG, etl=ETL, order="linear") -> dict`：`order="interleaved"` 时，支撑内第 i 个点（kz 外循环、ky 内循环）属于第 `i % n_trains` 条链；返回的键与原来相同。
  - `schedule._raw_pose` 接受 `{"kind": "pulse", "at", "dur", "theta_deg", "t_mm"}`：`at <= f < at + dur` 时权重 1。
  - `schedule.soft_amp(events, train_time_frac) -> np.ndarray`：每条链 `Σ_e w_e(f)·e.get("soft", 0.0)`。
  - `mcommon`：`ROOT2, TR2, RES2, DATA2, CFG2, REG2, LEDGER2, ORDERS=("il", "kz"), ORDER_NAME={"il": "interleaved", "kz": "linear"}, CONDITIONS2, ARMS2, MATRIX2, THRESH2, CONTAM_MM, SOFT, exp_name(af, order, cond, arm, seed=42, tag="")`。`CONDITIONS2[c]` 的键：`inter, nonrigid, lesions, intra, v1_data`；`ARMS2[a]` 的键：`trainer, slot, m1, m2, m3`（`m2 ∈ {"monitor", "adaptive", "truth"}`，`m3 ∈ {None, "detect", "truth"}`）。

- [ ] **Step 1: 写失败的测试** `M/tests/test_order.py`、`test_mcommon.py`

```python
def test_linear_is_unchanged():            # 默认参数与显式 "linear" 的所有键逐位相等
def test_interleaved_spreads_every_train():
    s = schedule.build(sup, order="interleaved")
    kz = np.argwhere(s["train_of"] == 3)[:, 1]
    assert kz.max() - kz.min() > 0.8 * (np.argwhere(sup)[:, 1].ptp())      # 一条链跨过整个 kz 范围
    assert np.bincount(s["train_of"][sup]).ptp() <= 1                      # 各链点数相差不超过 1
def test_pulse_returns_to_reference():      # pulse 之外的链位姿为恒等，之内为给定值
def test_soft_amp():                        # Tb：两个 pulse 内为 1，step 之后为 0.1，其余为 0
def test_matrix_rows_are_defined():         # MATRIX2 每一行的条件、组、顺序都在表里；正式行去重后 43
def test_exp_name():  assert exp_name("af12", "il", "Ta", "B2") == "mm_af12_il_Ta_B2"
```

- [ ] **Step 2: 运行，确认失败** — `PYTEST2 test_order.py test_mcommon.py`，期望：`order` 参数、`soft_amp`、`mcommon` 不存在而失败。
- [ ] **Step 3: 实现**上面 Interfaces 里的签名；数值照「数值表」。
- [ ] **Step 4: 运行，确认通过** — `PYTEST2` 全部通过；`PYTEST1` 仍是 `84 passed, 2 deselected`。
- [ ] **Step 5: 提交** — `feat(motion-mild): acquisition orders, pulse events, constants of the second version`

### Task 2: 仿真器扩展与数据生成入口

**Files:**
- Modify: `E1/simulate.py`（`build_custom`）
- Create: `M/msim.py`、`M/tests/test_msim.py`

**Interfaces:**
- Consumes: Task 1 的 `schedule.build(order=)`、`soft_amp`、`mcommon`。
- Produces:
  - `simulate.build_custom(src, inter=None, u=None, intra_events=None, lesion_sites=None, small=False, meta=None, order="linear", soft_u=None)`。第 n 条链的形变是 `rel_n ∘ g ∘ (id + u + (a_n − a_ref)·soft_u)`，`a = soft_amp`；分组键改为 (位姿, 幅度差)。`truth` 新增：`order`、`soft_amp`、`train_brain_disp_mm`、`contaminated`（> `CONTAM_MM`，需要 `src["brain"]`）、`state_of_train`、`state_poses`（相对参考状态；只有 step 且脑内平均位移 > `CONTAM_MM` 才开新状态）、`ref_state`。
  - `msim.build(cond, order, src, small=False) -> dict`；CLI `msim.py --cond Ta --order il [--check]` 写 `DATA2/<order>/<cond>/{ksp_T2.h5, gt_ksp_T2.h5, schedule.npz, motion_truth.json, brain.npy, qc.json, manifest.json}`（有病灶时加 `ksp_T1.h5` 和病灶掩模，有扫描间时加 `disp_T2.npy`），拒绝覆盖。`C0` 只写 `schedule.npz`、`brain.npy` 和全干净的真值。带 `v1_data` 的条件（E2、D2）不复制数据，清单记录第一版文件的路径和 sha256，另写 `schedule.npz`。

- [ ] **Step 1: 写失败的测试**（用 `E1/tests/helpers.small_src`，`small=True`）

```python
def test_defaults_reproduce_v1():           # 不传新参数时，E4a 小样本的 ksp_T2 与改动前逐位相同（先存参考哈希）
def test_reference_train_untouched():       # 两种顺序：参考链的线等于 gt_ksp_T2 的线
def test_pulse_changes_only_its_trains():   # Ta：ksp_T2 != gt 的 (ky, kz) 点集 == contaminated 链的点集
def test_reference_train_outside_pulses():  # 全尺寸支撑、两种顺序、每个条件：参考链不在任何 pulse 内
def test_tb_residual_below_contamination_line():   # Tb：最后一个 pulse 之后的链 contaminated 全为 False
def test_truth_states():                    # Tc：两个状态，边界在 0.60；Ta、Tb：一个状态
def test_soft_field_scales_with_amp():      # soft_u 非零时，pulse 内的线与只有刚体时不同；幅度差为 0 的链不变
```

- [ ] **Step 2: 运行，确认失败** — `PYTEST2 test_msim.py`。
- [ ] **Step 3: 实现**。`test_defaults_reproduce_v1` 的参考哈希在改 `simulate.py` 之前先算好写进测试。
- [ ] **Step 4: 运行，确认通过** — `PYTEST2`、`PYTEST1` 都通过。
- [ ] **Step 5: 检查磁盘并生成可行性验证用的数据**

Run: `ssh -p 2222 ... alias 'df -BG --output=avail /data0 | tail -1'`，期望 ≥ 80G（9 个条件各约 5 GB）。然后 `rrun.py --cpu -- @python M/msim.py --cond C0 --order il`、`--cond DTa --order il`，再各跑一次 `--check`。
Expected: `qc.json` 中 `reference_train_equals_reference_pose: true`、`shutter_zero_set_identical: true`；DTa 的 `contaminated` 约占 6 %。

- [ ] **Step 6: 提交** — `feat(motion-mild): simulator takes an acquisition order and a soft-tissue field; msim builds a condition per order`

### Task 3: 回波链统计、权重、变化点（纯函数）

**Files:**
- Create: `M/weights.py`、`M/tests/test_weights.py`

**Interfaces:**
- Produces（全部 numpy，无状态）：
  - `shell_index(shape2d, n_shells) -> int ndarray (ky, kz)`：按归一化半径分壳层。
  - `train_stat(e2d, sampled2d, train_of, n_trains, shells) -> rho (n_trains,)`：`e2d` = 每个 (ky, kz) 点的残差能量；每个壳层取采样点能量的中位数作「正常残差」；`rho_n = Σ e / Σ 正常残差`（只对该链的采样点）；没有采样点的链为 NaN。
  - `robust_z(rho, groups=None) -> z`：`(rho − 中位数) / (1.4826·MAD)`，按组（状态）分别算；NaN 和 MAD = 0 时 z = 0。
  - `weights_from_z(z, z0, z1, cap) -> w`：`z <= z0` 为 1，线性降到 `z1` 处为 0；若 w < 1 的链多于 `cap·n`，只保留 z 最大的 `floor(cap·n)` 条的降权，其余恢复 1。
  - `changepoints(rho, min_frac, cp_t, max_states) -> list[int]`：对时间序列做二分分割；候选分割点取使两段中位数之差的统计量 `|Δ中位数| / (1.4826·合并MAD·sqrt(1/n1 + 1/n2))` 最大的位置；统计量 ≥ `cp_t` 且两段都 ≥ `min_frac·n` 才接受；返回新状态起始的链号（升序），最多 `max_states − 1` 个。
  - `auc(score, truth) -> float`。

- [ ] **Step 1: 写失败的测试**

```python
def test_clean_trains_all_weight_one():      # rho ~ 1 + 0.03·N(0,1)，n = 110：w 全为 1（20 个种子）
def test_contaminated_trains_are_found():    # 6 条链 rho 乘 1.5：这 6 条 w == 0，其余 == 1；auc == 1.0
def test_ramp_and_cap():                     # z = z0、(z0+z1)/2、z1 → w = 1、0.5、0；40 % 超线时 w < 1 的不超过 cap·n
def test_degenerate_trains_keep_weight_one():# rho 全相等 → w 全 1；含 NaN 的链 w == 1，且不影响其他链的 z
def test_shell_normalisation():              # 能量随半径衰减 100 倍、各链覆盖不同半径（kz 线性）：干净数据 w 全 1
def test_changepoint_found():                # 110 条链，后 40 % 的 rho 高 20 %，噪声 3 %：返回 [66]（±1）
def test_changepoints_none_on_stationary_noise():   # t 分布（自由度 3）噪声，200 个种子：全部返回 []
def test_pulses_are_not_states():            # 两段各 5 条链的尖峰：返回 []
```

- [ ] **Step 2: 运行，确认失败** — `PYTEST2 test_weights.py`。
- [ ] **Step 3: 实现**。
- [ ] **Step 4: 运行，确认通过**。
- [ ] **Step 5: 提交** — `feat(motion-mild): per-echo-train residual statistic, robust weights, change points`

### Task 4: 运行时补丁（监视、真值权重、加权数据项）与启动器

**Files:**
- Create: `M/mpatch.py`、`M/mrun.py`、`M/settings_dev.json`、`M/tests/test_mpatch.py`、`M/tests/test_mrun.py`
- Modify: `E1/run.py`（`ensure_trainer(tr=TR)`）、`E1/tools/queue.py`（课题档案）

**Interfaces:**
- Consumes: Task 1–3；`E1/patch.py` 的 `load_with_reference_gt`、`segment_masks`、`guide_frame`；`E1/run.py` 的 `strip_durations`、`sha`、`ledger_row`、`_source_digest`、`ensure_trainer`。
- Produces:
  - 运动配置（`CFG2/motion/<exp>.json`）的键：`exp, arm, cond, order, af, seed, channel, out_dir, schedule_path, truth_path, brain_path, m1, m2, m3, reg, native_shift_mm`。`m2 = {"mode", z0, z1, cap, warmup, every, n_shells}`；`m1`、`m3` 在本任务为 `None`。
  - `mpatch.validate(cfg)`；`mpatch.weight_map(w, train_of) -> (ky, kz) float`（支撑外为 1）；`mpatch.weighted_mask(mask_fit, wmap) -> mask_fit * sqrt(wmap)`（`wmap` 全 1 时返回 `mask_fit` 本身，同一对象）；`mpatch.install(cfg)`；`mpatch.main()`：`mpatch.py --motion FILE -- <cycle.py 参数>`。
  - 挂钩：数据集加载（`load_with_reference_gt`）；`_kspace_loss`（识别 T2 调用的条件与第一版相同：`ksp_meas is self.kspace_fit_<tag>`）；`train` 结束后保存。`monitor`：到 `warmup` 后每 `every` 次在 `no_grad` 下算 `rho`、`z` 并记入历史，损失不变。`truth`：从第 1 次迭代起用 `1 − contaminated`。
  - 保存：`RES2/<exp>_mm.json` = `{exp, arm, order, n_trains, m2: {mode, rho, z, w, history: [{it, n_down, w_sum}]}, m3: null, reg: null}`；`RES2/<exp>_canon.npz` = `rec_raw`，加权组另有 `rec_w_dc`（用评分器 `arc/cycle.py score_contrast` 的同一个硬数据一致性函数，只插回参考状态中 `w >= 0.5` 的采样线）。
  - `mrun.py --af af12 --order il --cond DTa --arm Ow --gpu N [--settings F] [--seed 42] [--tag T] [--smoke]`；`--dry --all`；`--selftest --gpu N`。账本行 = `ledger_row(...)` 加 `order`。扫描内条件：`data_path = gt_ksp_T2.h5`，`motion_ksp_path = ksp_T2.h5`。
  - `queue.py --study mild`：行格式 `af:order:cond:arm`，调用 `M/mrun.py`，读 `ROOT2/ledger.jsonl`，写 `M/experiments.jsonl`，代码哈希覆盖 `M/*.py` 和 `E1` 的库文件；不带 `--study` 时行为不变。

- [ ] **Step 1: 写失败的测试**

```python
def test_weighted_mask_identity_bitexact():
    assert mpatch.weighted_mask(mask, torch.ones_like(wmap)) is mask
    # 假损失函数（与训练器同式）：全 1 权重的损失与原损失逐位相等
def test_weighted_loss_equals_weighted_sum():   # w ∈ {0, 0.25, 1}：损失 == Σ_n w_n·r_n / norm（相对误差 < 1e-6）
def test_truth_weights_zero_contaminated():     # 真值模式：contaminated 链 w == 0，其余 == 1
def test_monitor_never_touches_the_loss():      # monitor 模式下挂钩返回的张量就是原函数的返回值
def test_validate_rejects_missing_schedule():   # m2 非 None 而没有 schedule_path → AssertionError
def test_command_and_config():                  # mrun.command 的实验名、配方、--base_cfg；单对比度组去掉联合专属项
def test_intra_condition_uses_reference_gt():   # data_config：data_path 是 gt_ksp_T2.h5，motion_ksp_path 是 ksp_T2.h5
def test_queue_default_study_unchanged():       # 不带 --study 时解析出的命令与改动前相同
```

- [ ] **Step 2: 运行，确认失败** — `PYTEST2 test_mpatch.py test_mrun.py`。
- [ ] **Step 3: 实现**。
- [ ] **Step 4: 运行，确认通过** — `PYTEST2`、`PYTEST1`。
- [ ] **Step 5: GPU 自检** — 查空卡两次后 `rrun.py --gpu N -- @python M/mrun.py --selftest --gpu N`（30 次迭代，真实数据）。自检内容：(a) `il:C0:B0`（monitor，`warmup=10, every=10`）每次迭代打印的损失与不经补丁直接跑 `arc/cycle.py` 的逐字符相同；(b) 同一次运行里 `rec_w_dc` 的头部 PSNR 与评分器的 `head_psnr` 相差 < 1e-4 dB；(c) `il:DTa:Ow` 跑完并写出两个文件。
Expected: 三行 `SELFTEST ok`。
- [ ] **Step 6: 提交** — `feat(motion-mild): patch with monitored and truth weights in the T2 data term; launcher and queue profile`

### Task 5: 评估核心

**Files:**
- Create: `M/mevaluate.py`、`M/tests/test_mevaluate.py`
- Modify: `E1/evaluate.py`（`_state`、`_brain`、`_vols`、`motion_errors`、`lesion_metrics` 加可选的 `res=RES, data=DATA`）

**Interfaces:**
- Consumes: 账本行、`<exp>_mm.json`、`<exp>_canon.npz`、`<exp>_volumes.npz`、`motion_truth.json`；`diag_artefact.headonly.score_headonly`。
- Produces:
  - `load_ledger(path=LEDGER2)`；`table(ledger) -> list[dict]`：每行有 `exp, af, order, cond, arm, seed, tag, T1, T2`；T2（单对比度组为唯一通道）含 `head_psnr`（评分器，插回全部采样线）、`head_psnr_raw`、`head_psnr_wdc`（有 `rec_w_dc` 时）、`head_psnr_eval`（M2 为 adaptive 或 truth 的组取 `wdc`，其余取评分器值）。
  - `val(table, order, cond, arm, c="T2", key="head_psnr_eval", tag="")`。
  - `recovered(table, order, cond, arm, tag="") -> dict(loss, value, b0_c0, b0, arm_psnr)`：`loss = B0(C0) − B0(cond)`；`loss < min_loss_db` 时 `value = None`。
  - `weight_detection(mm, truth) -> dict(recall, false_rate, frac_w1, auc)`：`recall` = 受污染链中 `w < 0.5` 的比例；`false_rate` = 干净链中 `w < 0.5` 的比例。
  - `feasibility(table) -> dict(auc, auc_history, recover, loss, pass_auc, pass_recover, decision)`；CLI `--feas` 打印它，`--dev TAG` 打印开发报告。

- [ ] **Step 1: 写失败的测试**（合成账本和小数组）

```python
def test_eval_key_by_arm():          # B0 取评分器值；B2、Ow 取 wdc
def test_recovered_fraction():       # B0(C0) 36.0，B0(Ta) 34.0，B2(Ta) 35.5 → loss 2.0，value 0.75
def test_recovered_none_when_loss_small():   # loss 0.05 → value None
def test_weight_detection_counts():  # 10 条受污染链 9 条 w < 0.5，100 条干净链 1 条 w < 0.5 → 0.9、0.01
def test_feasibility_decision():     # auc 0.97、recover 0.6 → "go"；任一不满足 → "stop"；loss < 0.1 → "stop: motion too small to measure"
```

- [ ] **Step 2–4:** 运行确认失败；实现；`PYTEST2`、`PYTEST1` 通过。
- [ ] **Step 5: 提交** — `feat(motion-mild): evaluation core -- evaluation score per arm, recovered loss, weight detection, feasibility report`

### Task 6: 可行性验证（设计稿第 10 节第 1 步；不通过就停）

**Files:** Modify: `M/experiments.jsonl`、`M/README.md`

- [ ] **Step 1: 启动 3 次训练** — `E1/tools/queue.py --study mild --rows af12:il:DTa:B0 af12:il:DTa:Ow af12:il:C0:B0 --settings M/settings_dev.json --name feas`。队列自己做两次空卡检查。
- [ ] **Step 2: 读结果** — 三行都进了 `M/experiments.jsonl` 后：`rrun.py --cpu -- @python M/mevaluate.py --feas`。
Expected: 打印 `loss`、各次重算的 `auc`、最终 `auc`、`recover`、`decision`。另核对 `il:C0:B0` 的 T1/T2 头部 PSNR 与第一版 C0/A0（35.833 / 36.305）相差 ≤ 0.01 dB；超出则先查原因，不往下走。
- [ ] **Step 3: 判定** — `decision == "go"`（最终 `auc >= 0.95` 且 `recover >= 0.5`）才继续任务 7。否则**停下**：把三行结果、AUC 随迭代的变化、每条链的 z 分数图写进 `M/README.md`，提交，报告用户。这是设计稿规定的停点，不自行换方案继续。
- [ ] **Step 4: 提交** — `results(motion-mild): feasibility check on DTa (interleaved order)`

### Task 7: M2 自适应权重

**Files:** Modify: `M/mpatch.py`、`M/tests/test_mpatch.py`

**Interfaces:**
- Produces: `m2.mode == "adaptive"`：迭代数 ≥ `warmup` 且 `(it − warmup) % every == 0` 时重算 `rho → z → w`（`weights.py`），之后的迭代用新 `w`；`warmup` 之前 `w ≡ 1`。单对比度训练器同样生效（通道 `plain`）。`history` 每次重算记一条。

- [ ] **Step 1: 写失败的测试**

```python
def test_adaptive_schedule():        # 假训练循环 1000 次，warmup 600、every 100：重算发生在 600、700、800、900；之前 w 全 1
def test_adaptive_downweights_planted_trains():  # 给 6 条链的测量值加偏移：第一次重算后这 6 条 w == 0，损失下降
def test_weights_carry_no_gradient():           # w 不在计算图里（requires_grad False），残差统计在 no_grad 下
def test_single_trainer_channel():              # channel "plain" 时挂钩识别 kspace_fit_p
```

- [ ] **Step 2–4:** 确认失败；实现；`PYTEST2` 通过。
- [ ] **Step 5: GPU 冒烟** — `mrun.py --af af12 --order il --cond DTa --arm B2 --gpu N --smoke`（`warmup=10, every=10`）。Expected: 跑完，`_mm.json` 的 `history` 非空。
- [ ] **Step 6: 提交** — `feat(motion-mild): adaptive per-echo-train weights (M2)`

### Task 8: M1 配准驱动的对齐

**Files:**
- Create: `M/align.py`、`M/reg.py`、`M/tests/test_align.py`
- Modify: `E1/adapter.py`、`M/mpatch.py`、`M/mrun.py`、`M/tests/test_mpatch.py`

**Interfaces:**
- Produces:
  - `MotionAdapter.set_inter(inter: geom.Rigid)`、`set_segment_pose(seg: int, pose: geom.Rigid)`：把 `set_truth` 里已有的换算拆成这两个方法，`set_truth` 的结果不变；不改 `requires_grad`。
  - `align.canonical_zf(zf: complex Tensor (N0, N1, N2), inter: geom.Rigid) -> Tensor`：标准帧体素 `r` 处取 `zf(T r)`，三线性，实部虚部分别插值。
  - `align.ngf_loss(a, b, mask, eps_q=0.1) -> Tensor`：`1 − mean_mask[(∇a·∇b)² / ((|∇a|² + εa²)(|∇b|² + εb²))]`，幅值图，中心差分，`ε = eps_q × 掩模内梯度模的中位数`。
  - `reg.py --af af12 --order il --conds C0 E2 ... [--check-truth]`：零填充 T1、T2（欠采样数据和训练用的掩模）→ `probe_reginit.register` → `REG2/<af>_<order>_<cond>.json` = `{theta_deg, t_mm, cost, brain_in_voxels}`。脑掩模取自零填充 T1。`--check-truth` 事后打印与真值的差分配准误差，不写进该文件。
  - 运动配置 `m1 = {"reg_path", "align": {start, end, lr, weight} | None}`。挂钩：(i) 建适配器（刚体开，非刚体随 `align` 开），`set_inter(配准结果)`，刚体参数始终冻结；(ii) 包住训练器里含 `d_c = self.zf_c[ix, iy, iz]` 的那个方法，调用期间把 `self.zf_c` 换成 `canonical_zf(self.zf_c, inter)`，结束后换回；(iii) T2 渲染走 `render_transformed`，引导图的坐标系处理与第一版 `patch.install` 里的 `ept` 相同（照抄那十几行并注明来源）；(iv) 迭代数在 `[start, end)` 内每次做一步对齐：移动图 = 用 `.detach()` 后的高斯参数和 T1 密度经适配器渲染，参考图 = 当前 T2 渲染 `.detach()`，损失 `weight · ngf_loss(|移动图|, |参考图|, 头部支撑) + adapter.regulariser(reg)`，只对控制网格求梯度，用一个独立的 `Adam([ctrl], lr)` 更新；这一步之外控制网格 `requires_grad=False`，主损失不给它梯度；`end` 之后不再更新。
  - `_mm.json` 的 `reg = {theta_deg, t_mm}`；适配器终态仍按第一版的格式写 `<exp>_motion.json`（`mevaluate` 用 `E1/evaluate.motion_errors(..., res=RES2, data=...)` 读）。

- [ ] **Step 1: 写失败的测试**

```python
def test_canonical_zf_undoes_known_shift():
    # 团块在 c，T2 图像里在 c + (3, 0, 0)；inter = 平移 (3, 0, 0) → canonical_zf 的质心回到 c（误差 < 0.05 体素）
def test_set_inter_matches_set_truth():     # set_inter(g) 后的 state() 与 set_truth(inter=g) 的刚体部分相同
def test_set_segment_pose_matches_set_truth():
def test_ngf_loss_minimum_at_alignment():   # b = a 平移 0、0.5、1、2 体素：损失单调增，0 处最小
def test_ngf_gradient_points_to_alignment():# 用 grid_sample 的可微平移，梯度下降 200 步把 1 体素的偏移降到 < 0.1
def test_ngf_ignores_contrast_inversion():  # b = max(a) − a：损失与 b = a 相同（1e-6）
def test_ctrl_gets_no_gradient_from_main_loss():   # 假训练器：主损失 backward 后 ctrl.grad 为 None；对齐步之后 ctrl 变了
def test_adapter_frozen_after_end():        # it >= end：ctrl 不再变化
```

- [ ] **Step 2–4:** 确认失败；实现；`PYTEST2`、`PYTEST1`（含 `E1/tests/run_tests.py --gpu` 的 2 个 GPU 测试，因为动了 `adapter.py`）通过。
- [ ] **Step 5: 配准** — `rrun.py --cpu -- @python M/reg.py --af af12 --order il --conds C0 D2 --check-truth`。
Expected: C0 的平移与 `NATIVE_SHIFT_MM` 相差 ≤ 0.3 mm、旋转 ≤ 0.2°；D2 的差分配准误差 ≤ 0.2 mm（探针里是 0.08）。不满足时按 systematic-debugging 查，不放宽阈值。
- [ ] **Step 6: GPU 冒烟** — `mrun.py ... --cond D2 --arm B1 --gpu N --smoke`。Expected: 日志有 `[mm] M1: pose from registration ...` 和 T2 密度初值换帧的一行；跑完。
- [ ] **Step 7: 提交** — `feat(motion-mild): registration-driven inter-scan alignment (M1)`

### Task 9: M3 持续偏移的状态位姿

**Files:** Modify: `M/mpatch.py`、`M/reg.py`、`M/mevaluate.py`、`M/tests/test_mpatch.py`、`M/tests/test_mevaluate.py`

**Interfaces:**
- Produces:
  - 运动配置 `m3 = {"mode": "detect" | "truth", at, min_frac, cp_t, max_states, init: "reg" | "zero", lrs}`。适配器建成 `n_segments = max_states`、`ref_segment = 0`，状态位姿参数从一开始就在优化器里（末尾两个组，学习率 `lrs`），`requires_grad=False` 直到有状态被启用。
  - `detect`：迭代数 == `at` 时，用当时的 `rho`（按时间顺序）调 `weights.changepoints`。无变化点 → M3 之后不起作用，记 `states.n = 1`。有 → `state_of_train`；含参考链的状态编号 0、位姿固定为恒等；其余状态取初值（`init="reg"`：`reg.register_states(und_ksp, csm, mask, train_of, state_of_train, brain_mask) -> list[geom.Rigid]`，两个状态各自零填充后配准；`"zero"`：恒等），`set_segment_pose`，打开 `requires_grad`。之后数据项 = `Σ_s loss_fn(render_s, mask_s)`，`mask_s = segment_masks(mask_fit, state2d, 0, n) × sqrt(w)`，每次迭代渲染全部已启用的状态，不抽样；`robust_z` 按状态分组。
  - `truth`：从第 1 次迭代起用真值的 `state_of_train` 和 `state_poses`，位姿冻结；M2 的真值权重只把「相对本状态位姿仍受污染」的链置 0。
  - `_mm.json` 的 `m3 = {mode, n, detected_at, boundaries, state_of_train, init_poses, final_poses}`。
  - `mevaluate.state_errors(mm, truth, brain) -> dict(n_states, n_true, boundary_err_trains, rot_err_deg_max, t_err_mm_max)`（位姿误差用 `E1/evaluate.segment_errors`）。

- [ ] **Step 1: 写失败的测试**

```python
def test_no_changepoint_means_no_states():   # rho 平稳：at 之后仍只渲染一次，seg 参数 requires_grad 仍为 False
def test_states_partition_the_mask():        # 检出后 Σ_s mask_s² == mask_fit · w（逐点）
def test_reference_state_is_zero_and_frozen():   # 含参考链的状态编号 0，位姿恒等，不在可训练参数里
def test_all_states_rendered_every_iteration():  # 两个状态：每次迭代 render 被调用 2 次（状态 0、1）
def test_truth_mode_uses_truth_states():
def test_seg_groups_are_last_and_present_from_start():
def test_state_errors():                     # 边界差 2 条链、位姿差 0.1°/0.2 mm 的合成例子
```

- [ ] **Step 2–4:** 确认失败；实现；`PYTEST2`、`PYTEST1` 通过。
- [ ] **Step 5: 生成 DTc 并冒烟** — `msim.py --cond DTc --order il`（加 `--check`）；`reg.py --af af12 --order il --conds DTc`；`mrun.py ... --cond DTc --arm B123 --gpu N --smoke`（`at=20, warmup=10, every=10`）和 `--arm Op --smoke`。Expected: 两次都跑完；`Op` 的 `_mm.json` 里 `m3.n == 2`。
- [ ] **Step 6: 提交** — `feat(motion-mild): a pose for a detected persistent shift (M3) and its truth control`

### Task 10: 开发条件定设置、零运动关卡、冻结

**Files:** Create: `M/settings_frozen.json`；Modify: `M/experiments.jsonl`、`M/README.md`（「开发记录」）

- [ ] **Step 1: 基础开发运行**（标签 `d1`）— 先 `reg.py --af af12 --order il --conds DTa`；然后队列跑 `dev` 的 6 行（「对齐项关」那行用 `settings_dev_noalign.json`，只把 `m1.align` 设为 `null`，标签 `d1r`）。
- [ ] **Step 2: 读开发报告** — `mevaluate.py --dev d1`。逐项记录到 README：DTa/B2 的收回比例、检出率、误降率；DTc/B123 的状态数、边界误差、位姿误差、收回比例，对照 DTc/Op；D2/B1 两种设置的差分配准误差（刚体 / 含非刚体）和 T1、T2 PSNR。
- [ ] **Step 3: 定设置** — 规则（都只看开发条件）：
  - 对齐项：D2 上含非刚体的配准误差不比只用刚体小，或 T2 PSNR 低 0.02 dB 以上 → `m1.align = null`（设计稿的退路），写明。
  - `z0/z1/cap/warmup/every`：DTa 上检出率 < 0.9 或误降率 > 0.02 时调整。
  - `m3.init`：DTc 上若 `"reg"` 的初值误差大于恒等初值的误差，改 `"zero"`。
  - 每次调整重跑相关的开发行，新标签 `d2`、`d3`…；开发条件上的训练（含基础 6 次）合计不超过 10 次。到 10 次仍不达标就按现状冻结并如实写明。
- [ ] **Step 4: 零运动关卡** — 先 `msim.py --cond C0 --order kz`、`reg.py` 补齐 C0；队列跑 `gate` 的 7 行（候选设置）。`mevaluate.py --dev gate` 打印每个组相对 `il:C0:B0` 的 T1、T2 差、`frac_w1`、状态数。
Expected: 每个组 T1、T2 差 ≥ −0.05 dB；`frac_w1 >= 0.99`；状态数 1。任一不满足 → 回到 Step 3（重跑计入 10 次），不带着已知的零运动损失往下走。
- [ ] **Step 5: 冻结** — 把通过关卡的设置写成 `M/settings_frozen.json`；关卡各行若就是用这份设置跑的，即为正式行。
- [ ] **Step 6: 提交** — `results(motion-mild): development rounds, zero-motion gate for every arm, settings frozen`

### Task 11: 测试条件的数据与正式运行

**Files:** Modify: `M/experiments.jsonl`

- [ ] **Step 1: 生成数据** — `msim.py` 依次生成 `il:{Ta, Tb, Tc, E2Tb, TbL, E2}`、`kz:{Ta, Tc}`，各跑 `--check`。Expected: 每个 `qc.json` 的两项检查为 true；`contaminated` 比例：Ta 约 5 %、Tb 约 8 %、Tc 约 40 % 或 60 %（取决于参考链在哪一侧）。
- [ ] **Step 2: 配准** — `reg.py --af af12 --order il --conds E2 Ta Tb Tc E2Tb TbL --check-truth`、`--order kz --conds Ta Tc`。
- [ ] **Step 3: 先跑上限组** — 队列 `upper`（`--settings M/settings_frozen.json --name upper`）。跑完 `mevaluate.py --dev ""` 看每个条件的 `loss` 和 Ow / Op 的收回比例，记入 README。`loss < 0.1 dB` 的条件按设计稿写明「该幅度下问题不显著」，其余行照跑。
- [ ] **Step 4: 跑其余行** — 队列 `rest`。有卡就并行，一卡一作业。
- [ ] **Step 5: 核对** — `mrun.py --dry --all` 列出 43 个正式行都有分数文件；`M/experiments.jsonl` 行数相符；失败的作业查明原因后重跑并在 README 记一行。
- [ ] **Step 6: 提交** — `results(motion-mild): formal matrix, 43 runs`

### Task 12: 判据与汇总

**Files:** Modify: `M/mevaluate.py`、`M/tests/test_mevaluate.py`；Create: `M/summary.json`、`M/tools/mtables.py`

**Interfaces:**
- Produces: `criteria(summary) -> dict`，每一项 `{value, threshold, sense, pass}`：
  - S1（两种顺序的 C0）：每个组相对 B0 的 T1、T2 差；`frac_w1`；状态数；另报 B1 的平移与 `NATIVE_SHIFT_MM` 之差、旋转（不计通过与否）。
  - S2（E2）：B1、B123 的差分配准误差（刚体 / 含非刚体）；归一化 Δ：`PSNR(组, E2) − PSNR(组, C0) −`（T2 再减 I-T2 的同一差值）。
  - S3（`il` 的 Ta、Tb、Tc、E2Tb）：B123 的收回比例；检出率；误降率；Tc 的状态位姿误差。`loss < 0.1` 的条件 `pass = None`。
  - S4（TbL 对 Tb）：B123 的病灶对比度恢复 ≥ B0 的值 − 0.05；串扰不超容差（`E1/evaluate.lesion_metrics(..., cond="TbL", ref_cond="Tb", res=RES2, data=...)`）。
  - `kz`：只报告；B123 相对 B0 的差 ≥ −0.05 记为「无损害」。
  - `build_summary()`、CLI `--all` 写 `ROOT2/eval/summary.json`（不含任何时钟键）。`mtables.py [summary.json]` 打印 Markdown 表。

- [ ] **Step 1: 写失败的测试** — 合成 `summary`：`test_s1_fails_on_downweighted_clean_data`、`test_s3_skips_small_loss`、`test_s3_pass_and_fail`、`test_kz_reported_not_judged`、`test_summary_has_no_clock_keys`（键名不含 `time`、`elapsed`、`timestamp`、`date`）。
- [ ] **Step 2–4:** 确认失败；实现；`PYTEST2`、`PYTEST1` 通过。
- [ ] **Step 5: 汇总** — `rrun.py --cpu -- @python M/mevaluate.py --all`；在服务器端先扫一遍键名确认无时钟字段，记 sha256，再取回为 `M/summary.json`；`M/tools/mtables.py > 表`。
- [ ] **Step 6: 提交** — `results(motion-mild): criteria S1-S4 and summary`

### Task 13: 出图与结果文件

**Files:** Create: `M/mfigures.py`、`M/figures/*.png`、`M/RESULTS.md`；Modify: `M/README.md`

**Interfaces:**
- Produces: `mfigures.py --all` 写 `ROOT2/figures/`（600 dpi PDF + 200 dpi PNG）：
  - `fig_images_<cond>.png`（Ta、Tb、Tc、E2Tb，`il`）：一行图像（参考、B0、B123、Ow）+ 一行误差图（同一色标），横断位一层，标 PSNR。
  - `fig_trains_<order>_<cond>.png`：横轴回波链时间；三条曲线：真值脑内平均位移、z 分数、权重；状态边界用竖线。
  - `fig_recovered.png`：各条件 B2、B123、Ow、Op 的收回比例，`il` 与 `kz` 分面。
  - `fig_align_E2.png`：B0、B1、B123 的配准误差和归一化 Δ。

- [ ] **Step 1: 出图** — `rrun.py --cpu -- @python M/mfigures.py --all`，取回 PNG 到 `M/figures/`，逐张打开看：标题不重叠、色标一致、数字与 `summary.json` 一致。
- [ ] **Step 2: 写 `M/RESULTS.md`**（中文，结构同 `E1/RESULTS.md`）：一段话结论；做了什么；判据表；分项结果（零运动、扫描间、扫描内 `il`、`kz` 对照、病灶、联合与独立的比较）；与第一版的对比；局限（采集顺序是假设、单受试者、单种子、仿真）；没做的；来源与命令。表格全部由 `mtables.py` 生成，正文不手写数字以外的结论性数字。
- [ ] **Step 3: 更新 `M/README.md`** 的状态，并在 `E1/README.md` 末尾加一行指向本版。
- [ ] **Step 4: 全部测试** — `PYTEST2`、`PYTEST1` 通过。
- [ ] **Step 5: 提交** — `results(motion-mild): figures and RESULTS.md`
- [ ] **Step 6: 给用户看结果** — 先在 VS Code 打开 `fig_recovered.png` 并单独一行给路径，再给简短结论（项目规则「先图后文」）。

## 与设计稿的出入（执行时按此处理，并在结果里写明）

1. 设计稿说「Tc 在 kz 线性下就是第一版的 E4a」。本版在动作期间叠加了软组织场，所以 `kz:Tc` 重新生成，与 E4a 的刚体部分相同、数据不逐位相同。
2. 设计稿的开发条件是 DTa、DTc、C0。M1 的非刚体细化需要一个扫描间的开发条件，沿用第一版的 D2（只读引用它的数据）。
3. 设计稿的对齐项写的是「T1 按当前变换渲染到 T2 坐标系，与 T2 的渲染比较」。两幅图若都随适配器移动，这一项对适配器没有梯度；本计划把 T2 渲染当作固定参考（`.detach()`），控制网格只从这一项得到梯度。
4. 持续偏移的检出用回波链统计量时间序列上的变化点（中位数之差），而不是「被降权的链连成一段」。原因：受污染的链占 40 % 时，稳健 z 分数的中位数本身已被拉高，按 z 阈值连段会漏检。
5. Op = B123 换成真值的状态、位姿和权重（含 M1）；Ow = 只有 M2 的真值权重（不含 M1）。
