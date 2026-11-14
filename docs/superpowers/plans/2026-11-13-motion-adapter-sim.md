# Motion-Aware MC-GS 运动仿真 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 hyw 的真实多线圈 k 空间上加入已知的 T2W 运动（扫描间刚体/非刚体、扫描内分段刚体）和对比度特有病灶，验证给共享高斯加一个轻量运动适配器后，MC-GS 能找回运动并保住 T2W 质量，零运动时不添乱。

**Architecture:** 所有代码放在 `autoresearch/motion_adapter_sim_20261113/`。仿真器在 CPU 上由 hyw 数据生成各条件的数据集；适配器是一个独立的 `nn.Module`，由子进程在内存里给冻结训练器副本打补丁（`_render`、`_kspace_loss`、`_optimizer_param_groups`、数据集加载），然后原样运行 `arc/cycle.py`；评估与出图读训练器保存的体数据和适配器保存的运动估计。

**Tech Stack:** Python 3（服务器 GSTMI 环境：torch 2.7.1+cu118、numpy 1.26、scipy 1.15、h5py、pytest 9），现有编译好的 CUDA 体素化扩展，`gpu.py` 远程执行。

**Spec:** `docs/superpowers/specs/2026-11-13-motion-adapter-sim-design.md`（下称“设计稿”）。本计划只写设计稿没定的东西；数值、判据、矩阵以设计稿为准。

## Global Constraints

- 代码只写在 worktree `mcgs/.claude/worktrees/motion-adapter-sim`、分支 `exp/motion-adapter-sim-20261113`；共享工作区不切分支、不改文件。
- 冻结训练器：`SRC=/data0/congcong/code/GS/mcgs_sg2x2_wt`，提交 `54f8a2a3bef402e785314b748f730a9623ca9d43`，目录 `arc/ paired_gs/ diag_artefact/`；只复制、逐文件核对 sha256，不改、不重编译。
- 服务器根目录 `ROOT=/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_motion_adapter_20261113/`（`trainer/`、`data/<af>/<条件>/`、`configs/`、`ledger.jsonl`）。课题数据和其他根目录只读。
- hyw 数据：`/data0/congcong/code/GS/Results_20260908/T1T2_AF12_retro/data/{ksp_T1,ksp_T2,csm_espirit_T1,csm_espirit_T2,mask_af12_3d}.h5`；AF16 掩模 `/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_retro/data/mask_af16_3d.h5`。
- 配方：联合组 V32G，单对比度组 V32 去掉联合专属项，取自 `autoresearch/ce_af16_ablation_20261110/abl_framework.py`，不重调。种子 42。
- 运行命令一律为（`W` = worktree 绝对路径，`E` = `autoresearch/motion_adapter_sim_20261113`）：
  `/usr/bin/python3 <workspace>/gpu-tools/gpu.py --config W/.claude/gpu/config.json run {--cpu | --gpu ID} -- @python E/...`
- GPU：启动前两次 `nvidia-smi --query-compute-apps`、间隔约 75 s 都为空才用；一卡一作业；`cycle.py` 自己设 `CUDA_VISIBLE_DEVICES`，所以传物理卡号给 `--gpu`（与 `abl_framework.py` 相同）；不动别人的作业。
- 时间：只用本机时钟。服务器上不运行 `date`、不读文件时间、不读 `run_info.json` 的 `timestamp_utc`、不读 `*_score.json` 的 `elapsed_s`；账本记本机启动/结束时间。本实验的代码不打印、不写入任何时钟值或耗时。
- GT、ROI、病灶掩模、运动真值不进训练（oracle 组除外，并在表中标注）。适配器设置只在 D1–D3 上定，冻结后才跑测试条件。
- 每个任务结束提交一次；不推送，除非用户要求。大图放 `E/figures/big/`（自带 `.gitignore`）。

## 已核实的训练器事实（设计稿第 14 节，2026-11-13 读源码）

1. `shared_warmup_iters` 在 `JointTrainer.train` 里被 `del warmup`，不起作用：T2 的数据项从第 1 次迭代就有梯度。阶段 1 可以直接估计刚体。
2. 优化器由 `JointTrainer._optimizer_param_groups()` 给出参数组，`_setup_optimizer` 和 `_rebuild_optimizer` 都调用它；重建时按组序恢复学习率。配方里 `adaptive_control.enable` 未打开，不会重建。运动参数组追加在末尾即可。
3. 学习率调度 `ExponentialLR(gamma=0.9995)` 作用于所有组，4000 次后为初值的 0.135 倍，这就是阶段 3 的衰减。
4. 梯度裁剪只对 `self.model.parameters()`。适配器不注册到 `model` 上，不参与裁剪，也不改变高斯参数的裁剪。
5. `train()` 是单个大函数：每次迭代调用 `_render("plain")`、`_render("contrast")` 各一次，再调用 `_kspace_loss(vol, csm_fit, kspace_fit, mask_fit, freq_w, norm)`。跨对比度引导图在 `edge_preserving_tv` 内部已 `detach`。
6. `TileVoxelizer.forward` 由四元数和尺度构造 `cov_inv`、由 `_bbox_radius_vox` 算包围盒、`build_tile_pairs` 建瓦片表，再调用 `_TileVoxelizeCUDA.apply(center_vox, cov_inv, den_r, den_i, tile_ranges, pair_ids, half_shape, tile_size, D, H, W)`；CUDA 反向对 `center_vox` 和 `cov_inv` 都给梯度。
7. GT 由数据集类从 `data_path` 的全 k 空间算出，并按 GT 的 P99.9 归一化；`score_contrast` 的 `e_unobs` 用 `ds.kspace_multicoil`。所以扫描内条件必须在数据集加载处打补丁（任务 6）。
8. `SingleContrastTrainer` 的唯一通道叫 `plain`，并拒绝联合专属配置项。

## 坐标与方向约定（全计划通用）

- 体素下标 `i`（轴 0/1/2 = S-I/A-P/L-R，1 mm），FOV 中心 `c = (N−1)/2`，`N = (256, 256, 176)`。相对中心的 mm 坐标 `r = i − c`；训练器的归一化坐标 `p = r / (N/2)`，即 `r = D p`，`D = diag(N/2)`。
- **变换方向**：`T` 把标准帧（T1 帧）里的物质点 `r` 送到它在 T2W 图像里的位置 `r′ = R(r + u(r)) + t`。仿真器生成的图像满足 `m_T2(r′) = m(r)`，即 `m_T2 = m ∘ T⁻¹`。
- 旋转用轴角向量 `θ`（弧度，分量顺序 = 轴 0/1/2）。命名：转头 = 绕轴 0，侧倾 = 绕轴 1，点头 = 绕轴 2。
- 仿真器的旋转支点 `q = c + (70, 20, 0)` mm（下方 70、后方 20）。换算到以 `c` 为中心：`t = t_sim + (I − R)(q − c)`。
- 扫描内：第 s 段的总变换 `T_s = Δ_s ∘ T`，`Δ_s(r) = R_s r + t_s`，参考段 `Δ = I`。

## 文件结构（都在 `E/` 下）

| 文件 | 职责 |
|---|---|
| `common.py` | 路径、配方、条件表、对照组表、运行矩阵；无第三方依赖，可在本机导入 |
| `geom.py` | 约定：轴角↔旋转矩阵、支点换算、变换合成与求逆、稠密位移场、TRE（numpy） |
| `schedule.py` | 相位编码排序、回波链、K 段划分、参考段（numpy） |
| `warp.py` | 仿真用图像形变：刚体（傅里叶平移 + 三次剪切旋转）、非刚体（三次样条）、RBF 位移场（numpy/scipy） |
| `lesions.py` | 病灶位置选取与插入、掩模（numpy/scipy） |
| `simulate.py` | CLI：生成一个条件的数据集、真值、质检、清单 |
| `adapter.py` | `MotionAdapter(nn.Module)` 与 `render_transformed`（torch） |
| `patch.py` | 子进程入口：给训练器打补丁，运行 `arc/cycle.py`，保存运动估计和标准帧渲染 |
| `run.py` | 启动器：核对训练器副本、写配置、起子进程、记账本；`--dry` 只打印命令 |
| `evaluate.py` | 读结果：图像指标表、归一化 Δ、运动误差、病灶指标、判据 S1–S4 |
| `figures.py` | 记录格式图、病灶放大图、运动恢复图、ΔPSNR 汇总图 |
| `tests/test_*.py` | 单元测试（服务器 CPU 上用 pytest 跑） |
| `README.md`、`experiments.jsonl`、`RESULTS.md`、`settings_frozen.json` | 实验说明、本地账本、结果、冻结的适配器设置 |

测试统一这样跑（下文写作 `PYTEST <文件>`）：
`gpu.py --config W/.claude/gpu/config.json run --cpu -- @python -m pytest E/tests/<文件> -q -p no:cacheprovider`，随后 `logs latest` 读结果。

## Review Focus

1. **零运动必须严格退回原模型**：适配器全零时 `render_transformed` 的输出与训练器 `_render` 逐位相同；否则 S1 和 C0 回归无从谈起。（任务 5 测试 `test_identity_bitexact`）
2. **方向符号**：适配器、仿真器、真值、oracle 四处对 `T` 的方向必须一致；错一处会让 TRE 看起来是两倍运动。（任务 5 `test_recovers_simulated_shift_direction`，任务 4 `test_integer_shift_equals_roll`）
3. **T2 椭圆遮挡外不严格为零**：会让 `estimate_known_zero` 判断出错，训练悄悄换了数据一致性集合。（任务 4 `test_shutter_zero_set_identical`）
4. **扫描内条件的 GT 取错**：若 GT 来自被运动污染的全 k 空间，所有组都被错误的参照评分。（任务 6 `test_intra_dataset_uses_reference_gt`）
5. **分段损失在零运动时的期望不等于原损失**：会让 A3 在 C0 上系统性偏离 A0。（任务 6 `test_segment_loss_matches_full_when_static`）

---

### Task 1: 实验骨架、私有 gpu.py 配置、常量表

**Files:**
- Create: `E/common.py`、`E/README.md`、`E/tests/test_common.py`、`E/figures/big/.gitignore`（内容 `*`\n`!.gitignore`）
- Create（不入库，`.claude/` 已被忽略）: `W/.claude/gpu/config.json`

**Interfaces — Produces（`common.py`）:**
- `SRC, SRC_HEAD, CODE_DIRS, ROOT, TR=ROOT/"trainer", RES=TR/"results_arc", HYW, MASKS={"af12": ..., "af16": ...}`
- `V32, V32G, JOINT_ONLY, CROSS_OFF`：与 `abl_framework.py` 逐项相同的列表。
- `N=(256,256,176)`、`PIVOT_OFFSET_MM=(70.0,20.0,0.0)`、`NATIVE_SHIFT_MM=(0.59,0.12,-0.13)`、`TR_S=2.5`、`ETL=320`、`K_SEG=8`。
- `CONDITIONS: dict[str, dict]`，键 `C0,E1a,E1b,E2,E3,E4a,E4b,E4c,D1,D2,D3`，每项含 `inter`（`theta_deg`、`t_sim_mm` 或 `None`）、`nonrigid`（`seed`、`peak_mm` 或 `None`）、`lesions`（bool）、`intra`（事件列表或 `None`）。数值：
  - E1a：`theta_deg=(0.5,0.5,1.5)`，`t_sim_mm=(0.8,-0.5,0.3)`；E1b 为两倍。
  - E2/E3：E1a + `nonrigid=dict(seed=7, peak_mm=1.5, brain_max_mm=0.8)`；E3 另 `lesions=True`。
  - E4a：`intra=[dict(kind="step", at=0.60, theta_deg=(0,0,1.0), t_mm=(0.5,0,0))]`。
  - E4b：`intra=[dict(kind="drift", theta_deg=(0,0.3,1.0), t_mm=(0.8,0,0))]`；E4c = E1a + E4b。
  - D1：`theta_deg=(-0.8,0.9,-1.0)`，`t_sim_mm=(-0.6,0.7,-0.4)`；D2 = D1 + `nonrigid=dict(seed=101, peak_mm=1.2, brain_max_mm=0.6)`；D3：`intra=[drift theta_deg=(0,0.4,-0.8), t_mm=(0,0.6,0); step at=0.35 theta_deg=(0.7,0,0)]`。
- `ARMS: dict[str, dict]`，键 `A0,A1,A2,A3,A23,O,I-T1,I-T2,I-T2+S`，每项含 `trainer`（`joint`/`single`）、`slot`、`rigid`、`nonrigid`、`segments`、`oracle`。
- `MATRIX: dict[str, list[tuple[str, str]]]`，键 `af12_core, af12_rest, af16, dev`，值为（条件, 组）列表，按设计稿第 10 节。
- `exp_name(af: str, cond: str, arm: str, seed: int = 42) -> str`，形如 `ma_af12_E2_A2`（seed≠42 加 `_s7`）。

- [ ] **Step 1: 写失败的测试** `tests/test_common.py`

```python
import ast, pathlib
import common as c

def _const(name):
    src = pathlib.Path(c.__file__).parents[1] / "ce_af16_ablation_20261110/abl_framework.py"
    for node in ast.parse(src.read_text()).body:
        if isinstance(node, ast.Assign) and node.targets[0].id == name:
            return node
    raise KeyError(name)

def test_v32_matches_framework_ablation():
    assert c.V32 == ast.literal_eval(_const("V32").value)
    assert c.V32G == [*c.V32, "--set", "loss.edge_tv_guide_alpha=0.5"]
    assert c.SRC_HEAD == ast.literal_eval(_const("SRC_HEAD").value)

def test_matrix_counts_and_names():
    assert len(c.MATRIX["af12_core"]) + len(c.MATRIX["af12_rest"]) == 40   # 设计稿第 10 节 AF12，不含开发条件
    assert len(c.MATRIX["af16"]) == 21
    for part in c.MATRIX.values():
        for cond, arm in part:
            assert cond in c.CONDITIONS and arm in c.ARMS
    assert c.exp_name("af12", "E2", "A2") == "ma_af12_E2_A2"
    assert c.CONDITIONS["E1b"]["inter"]["theta_deg"] == (1.0, 1.0, 3.0)
```

- [ ] **Step 2: 写 `W/.claude/gpu/config.json`**：复制 `<workspace>/gpu-tools/config.json`，把 `repo` 改成 worktree 的绝对路径，其余不变。运行 `gpu.py --config ... sync`，确认返回的快照里有 `E/` 而没有共享工作区的未跟踪文件。
- [ ] **Step 3: `PYTEST test_common.py`**，预期失败（`common` 不存在）。
- [ ] **Step 4: 写 `common.py` 和 `README.md`**（README：一段话目的、指向设计稿和本计划、运行命令模板、时间规则）。
- [ ] **Step 5: `PYTEST test_common.py`**，预期 2 passed。
- [ ] **Step 6: 提交** `exp(motion): scaffold, constants and private gpu config`。

### Task 2: 几何约定与采集时间表

**Files:** Create `E/geom.py`、`E/schedule.py`、`E/tests/test_geom.py`、`E/tests/test_schedule.py`

**Interfaces — Produces:**
- `geom.rotmat(theta: np.ndarray) -> np.ndarray`（3×3，Rodrigues）；`geom.axis_angle(R) -> np.ndarray`。
- `geom.Rigid`：数据类 `R: (3,3)`, `t: (3,)`（以 FOV 中心为原点的 mm）；方法 `apply(r)`, `inverse()`, `compose(other)`（`a.compose(b)` = 先 b 后 a）。
- `geom.rigid_from_pivot(theta_deg, t_sim_mm, pivot_offset_mm=PIVOT_OFFSET_MM) -> Rigid`。
- `geom.grid_mm(shape=N) -> np.ndarray`（`(*shape, 3)`，`r = i − c`）。
- `geom.tre(est_disp: np.ndarray, true_disp: np.ndarray, mask: np.ndarray) -> dict`（`mean_mm`, `max_mm`）；位移场形状 `(*shape, 3)`。
- `schedule.build(support: np.ndarray, k_seg: int = 8, etl: int = 320) -> dict`：`support` 为 (256,176) 布尔（椭圆内的相位编码点）。返回 `train_of`（(256,176) int32，支撑外为 −1）、`seg_of`（同形）、`n_trains`、`ref_train`、`ref_seg`、`train_time_frac`（每条链在 [0,1] 的时刻）。排序：kz（轴 2）外循环、ky（轴 1）内循环，只数支撑内的点；每 `etl` 个点一条链；段 = `train * k_seg // n_trains`；参考链 = 含点 (128, 88) 的链。
- `schedule.poses(events: list, train_time_frac: np.ndarray) -> list[Rigid]`：每条链的 `Δ`，已减去参考链的姿态使参考链为恒等；`step` 在 `at` 之后取全幅，`drift` 线性到全幅；多个事件相加（小角度下 θ、t 直接相加）。
- `schedule.segment_mean_poses(train_poses, seg_of_train, k_seg) -> list[Rigid]`（oracle 用，轴角与平移取段内平均）。

- [ ] **Step 1: 写失败的测试**

```python
# test_geom.py
def test_pivot_conversion_matches_direct():
    g = geom.rigid_from_pivot((0.5, 0.5, 1.5), (0.8, -0.5, 0.3))
    q = np.array(PIVOT_OFFSET_MM); r = np.array([10.0, -20.0, 30.0])
    R = geom.rotmat(np.deg2rad([0.5, 0.5, 1.5]))
    np.testing.assert_allclose(g.apply(r), R @ (r - q) + q + [0.8, -0.5, 0.3], atol=1e-9)

def test_inverse_and_compose():
    g = geom.rigid_from_pivot((1, -2, 3), (1, 2, 3)); r = np.random.default_rng(0).normal(size=(5, 3)) * 50
    np.testing.assert_allclose(g.inverse().apply(g.apply(r)), r, atol=1e-9)
    np.testing.assert_allclose(g.compose(g.inverse()).apply(r), r, atol=1e-9)

def test_e1a_brain_displacement_is_about_2mm():
    g = geom.rigid_from_pivot((0.5, 0.5, 1.5), (0.8, -0.5, 0.3))
    d = np.linalg.norm(g.apply(np.array([[-30.0, 0, 0], [0, 0, 0], [20.0, 40.0, 0]])) - [[-30, 0, 0], [0, 0, 0], [20, 40, 0]], axis=1)
    assert 1.0 < d.min() and d.max() < 4.5

# test_schedule.py
def test_counts_and_reference():
    sup = ellipse_support()                      # (256,176) 内切椭圆，占比约 0.785
    s = schedule.build(sup)
    assert s["n_trains"] == -(-int(sup.sum()) // 320)
    assert (s["train_of"][~sup] == -1).all() and s["seg_of"].max() == 7
    assert s["train_of"][128, 88] == s["ref_train"] and s["seg_of"][128, 88] == s["ref_seg"]

def test_poses_zero_at_reference_and_drift_endpoint():
    s = schedule.build(ellipse_support())
    p = schedule.poses([dict(kind="drift", theta_deg=(0, 0.3, 1.0), t_mm=(0.8, 0, 0))], s["train_time_frac"])
    np.testing.assert_allclose(p[s["ref_train"]].t, 0, atol=1e-12)
    total = p[-1].compose(p[0].inverse())
    np.testing.assert_allclose(np.rad2deg(geom.axis_angle(total.R)), (0, 0.3, 1.0), atol=0.02)
```

- [ ] **Step 2: `PYTEST test_geom.py test_schedule.py`**，预期失败。
- [ ] **Step 3: 实现 `geom.py`、`schedule.py`**。
- [ ] **Step 4: 再跑**，预期 5 passed。
- [ ] **Step 5: 提交** `exp(motion): geometry conventions and acquisition schedule`。

### Task 3: 仿真用形变与病灶

**Files:** Create `E/warp.py`、`E/lesions.py`、`E/tests/test_warp.py`、`E/tests/test_lesions.py`

**Interfaces — Produces:**
- `warp.fourier_shift(vol: np.ndarray, shift_mm) -> np.ndarray`：复数体，沿三轴线性相位；内容向 `+shift` 方向移动。
- `warp.rigid(vol, g: Rigid) -> np.ndarray`：返回 `vol ∘ g⁻¹`（内容按 `g` 前移）。旋转用 `scipy.ndimage.map_coordinates(order=3, mode="constant")` 分别作用于实部和虚部，平移部分用 `fourier_shift`；先把体零填充 16 体素再裁回。
- `warp.rbf_field(seed: int, peak_mm: float, brain: np.ndarray, brain_max_mm: float, shape=N) -> np.ndarray`：`(*shape,3)` 位移场 `u`。4 个高斯径向基，σ 在 30–50 mm；3 个中心在下部（轴 0 下标 > 2/3·N₀），1 个在脑内；整体缩放到峰值 `peak_mm`，再把脑内那个基的幅度缩到脑内最大值 ≤ `brain_max_mm`；最后减去全场均值。
- `warp.nonrigid_then_rigid(vol, u, g) -> np.ndarray`：返回 `vol ∘ T⁻¹`，`T(r) = g(r + u(r))`。`T⁻¹` 用 5 次定点迭代求 `u` 的逆，再接 `g⁻¹`，一次三次样条重采样。
- `warp.displacement(u, g, shape=N) -> np.ndarray`：`T(r) − r`，评估用。
- `lesions.pick_sites(t1_mag, brain, rng_seed=3) -> list[dict]`：5 个病灶（T2 独有 4 个，直径 4/6/8/12 mm，脑室旁/深部白质/近皮层/后颅窝各一；T1 独有 1 个，8 mm）。白质 = 脑内 T1 幅度高于脑内 P60 的体素；每个中心离脑边界 ≥ 半径 + 3 mm，彼此间距 ≥ 25 mm。返回 `center_vox`、`diameter_mm`、`contrast`（`"T2"`/`"T1"`）、`gain`。
- `lesions.insert(vol, sites, contrast) -> (vol2, mask)`：`vol2 = vol · (1 + gain · w)`，`w` 为半径处边缘 σ=1 mm 的平滑球；T2 病灶 `gain` ∈ [0.6, 1.0]（按直径从小到大取 1.0/0.9/0.75/0.6），T1 病灶 `gain = −0.4`；`mask = w > 0.5`，`uint8` 标号 1–5。
- `lesions.brain_mask(mag) -> np.ndarray`：移植 `results/motion_adapter_hyw_align/scripts/hyw_align_probe.py` 的 `head_mask` + `brain_mask`（同一规则，评估也用它）。

- [ ] **Step 1: 写失败的测试**

```python
# test_warp.py
def test_integer_shift_equals_roll():
    v = phantom()                                            # 带限复数椭球体 (64,64,48)
    np.testing.assert_allclose(warp.fourier_shift(v, (3, -2, 1)), np.roll(v, (3, -2, 1), (0, 1, 2)), atol=1e-5)

def test_rigid_roundtrip_under_half_percent():
    v = phantom(); g = geom.rigid_from_pivot((1.0, 1.0, 3.0), (1.6, -1.0, 0.6), pivot_offset_mm=(15, 5, 0))
    back = warp.rigid(warp.rigid(v, g), g.inverse()); m = np.abs(v) > 0.2 * np.abs(v).max()
    assert np.sqrt((np.abs(back - v)[m] ** 2).mean()) < 0.005 * np.sqrt((np.abs(v)[m] ** 2).mean())

def test_rigid_moves_content_forward():
    v = blob_at((32, 32, 24)); g = geom.Rigid(np.eye(3), np.array([4.0, 0, 0]))
    assert centroid(np.abs(warp.rigid(v, g)))[0] == pytest.approx(36.0, abs=0.05)

def test_rbf_field_limits_and_zero_mean():
    brain = ball_mask(); u = warp.rbf_field(7, 1.5, brain, 0.8, shape=(64, 64, 48))
    n = np.linalg.norm(u, axis=-1)
    assert n.max() == pytest.approx(1.5, rel=0.15) and n[brain].max() <= 0.8 + 1e-6
    assert np.abs(u.reshape(-1, 3).mean(0)).max() < 1e-6

# test_lesions.py
def test_sites_and_contrast():
    t1 = wm_phantom(); brain = lesions.brain_mask(t1); sites = lesions.pick_sites(t1, brain)
    assert [s["contrast"] for s in sites].count("T2") == 4 and len(sites) == 5
    v2, mask = lesions.insert(t1.astype(complex), sites, "T1")
    core = mask == 5
    assert np.abs(v2)[core].mean() / np.abs(t1)[core].mean() == pytest.approx(0.6, abs=0.05)
    assert (np.abs(v2)[(mask > 0) & (mask < 5)] == np.abs(t1)[(mask > 0) & (mask < 5)]).all()   # T2 病灶不改 T1
```

- [ ] **Step 2: `PYTEST test_warp.py test_lesions.py`**，预期失败。
- [ ] **Step 3: 实现**。若 `test_rigid_roundtrip_under_half_percent` 用三次样条达不到 0.5 %，改用三次剪切傅里叶旋转（设计稿 7.2 允许），测试不变。
- [ ] **Step 4: 再跑**，预期 5 passed。
- [ ] **Step 5: 提交** `exp(motion): simulation warps and lesion insertion`。

### Task 4: 仿真器与全部条件的数据

**Files:** Create `E/simulate.py`、`E/tests/test_simulate.py`

**Interfaces:**
- Consumes: 任务 1–3。
- Produces: CLI `simulate.py --cond <名> [--check] [--small]`，写 `ROOT/data/<条件>/`（与 AF 无关，掩模在配置里选）：
  - `ksp_T2.h5`（键 `kspace`，(26,256,256,176) complex64，布局与原文件相同）；E3 另有 `ksp_T1.h5`；有 `intra` 的条件另有 `gt_ksp_T2.h5`（参考位姿的全 k 空间）。C0 不写数据文件，配置直接指向原文件。
  - `motion_truth.json`：`inter`（`R`、`t`，以 FOV 中心为原点）、`pivot_offset_mm`、`nonrigid` 生成参数、`train_poses`、`segment_mean_poses`、`ref_train`、`ref_seg`；`disp_T2.npy`（float32，`T(r) − r`，只含加性运动）；`schedule.npz`（`train_of`、`seg_of`）；`lesion_mask_canon.npy`、`lesion_mask_T2.npy`、`lesion_sites.json`；`manifest.json`（每个文件的 sha256、源文件 sha256、快照 sha）；`qc_<条件>.png`（移动后 T2 与原 T2 之差，三平面）。
  - 函数 `simulate.build(cond: str, src: dict, small: bool = False) -> dict`（`src` 含 `ksp_T1, ksp_T2, csm_T1, csm_T2` 数组；`small=True` 时用测试给的小数组，不读盘）。
- 算法（设计稿 7.1）：`x_c = ifft3c(y_c)`；`m = Σ_c conj(S_c) x_c`；`m′ = insert(m)`（仅 E3）；对每个不同的姿态 `T`：`y_c^T = F[x_c + S_c·(W_T(m′) − m)]`；无扫描内运动时输出 `shutter ⊙ y^T`；有扫描内运动时对每条回波链取 `T = Δ_train ∘ T_inter`，按 `train_of` 把各链的相位编码线拼起来（沿读出轴整条线），再乘遮挡。`shutter` = 原 T2 k 空间所有线圈都为零的点的补集。FFT 用与训练器 `data/transforms.py` 相同的 `fftshift → fftn → ifftshift` 约定（numpy 实现，测试与 torch 版核对）。E3 的 T1：`y_c = F[x_c + S_c·(m_T1′ − m_T1)]`，无运动。
- 内存：一次只处理一个姿态，按线圈循环；110 条链的条件逐链生成并只保留该链的线。

- [ ] **Step 1: 写失败的测试**（`--small` 数组：8 线圈、(48,48,32)，合成线圈图与椭圆遮挡）

```python
def test_zero_motion_is_bit_identical():
    src = small_src(); out = simulate.build("C0", src, small=True)
    assert out["ksp_T2"] is src["ksp_T2"] or np.array_equal(out["ksp_T2"], src["ksp_T2"])

def test_shutter_zero_set_identical():
    src = small_src(); out = simulate.build("E1a", src, small=True)
    assert np.array_equal(np.all(out["ksp_T2"] == 0, axis=0), np.all(src["ksp_T2"] == 0, axis=0))

def test_integer_shift_matches_roll_of_coil_combined():
    src = small_src(shutter=False)
    out = simulate.build_custom(src, inter=geom.Rigid(np.eye(3), np.array([2.0, 0, 0])))
    m0, m1 = combine(src["ksp_T2"], src["csm_T2"]), combine(out["ksp_T2"], src["csm_T2"])
    np.testing.assert_allclose(m1, np.roll(m0, 2, 0), atol=2e-3 * np.abs(m0).max())

def test_fft_convention_matches_trainer():
    x = rng_complex((4, 16, 16, 8))
    np.testing.assert_allclose(simulate.fft3c(x), trainer_fft3c(x), atol=1e-5)     # trainer_fft3c 从 TR/paired_gs 导入

def test_intra_lines_come_from_their_train_pose():
    src = small_src(); out = simulate.build("E4a", src, small=True)
    sch = out["schedule"]; pre = sch["train_of"] < int(0.6 * sch["n_trains"])
    ref_pose_k = out["gt_ksp_T2"]
    late = (~pre) & (sch["train_of"] >= 0)
    assert not np.allclose(out["ksp_T2"][:, :, late], ref_pose_k[:, :, late])      # 突动之后的线来自另一个姿态
    seg_ref = sch["train_of"] == sch["ref_train"]
    np.testing.assert_allclose(out["ksp_T2"][:, :, seg_ref], ref_pose_k[:, :, seg_ref], atol=1e-6)

def test_truth_json_has_no_clock_fields():
    out = simulate.build("E4b", small_src(), small=True)
    assert not any(k in json.dumps(out["truth"]).lower() for k in ("time_utc", "timestamp", "elapsed", "date"))
```

- [ ] **Step 2: `PYTEST test_simulate.py`**，预期失败。
- [ ] **Step 3: 实现 `simulate.py`**（含 `build_custom` 供测试和 D 条件用）。
- [ ] **Step 4: 再跑**，预期 6 passed。
- [ ] **Step 5: 生成真实数据**：对 `E1a E1b E2 E3 E4a E4b E4c D1 D2 D3` 各起一个 `--cpu` 作业（可并行，注意每个约占 30 GB 内存，同时不超过 4 个）；每个结束后 `simulate.py --cond X --check` 重算 manifest 并打印四项质检（遮挡零点集合相同、参考链与 `gt_ksp_T2` 一致、E1a 的脑内平均位移 mm、往返误差）。把各条件的脑内平均/最大位移记入 `README.md`。
- [ ] **Step 6: 取回并查看质检图**：`fetch` 各作业的 `qc_*.png` 到 `E/figures/big/qc/`，逐张看：差图只在边缘处有结构、遮挡外无能量、病灶在预定位置。
- [ ] **Step 7: 提交** `exp(motion): simulator, QC and data manifests for all conditions`。

### Task 5: 运动适配器

**Files:** Create `E/adapter.py`、`E/tests/test_adapter.py`

**Interfaces:**
- Consumes: 训练器副本的 `gaussian.tile_voxelizer`（`TileVoxelizer`、`_TileVoxelizeCUDA`、`_quaternion_to_rotation_matrix`、`build_tile_pairs`）。
- Produces:
  - `class MotionAdapter(nn.Module)`，`__init__(volume_shape, rigid: bool, nonrigid: bool, n_segments: int, ref_segment: int, u_max_mm: float = 3.0, grid: int = 5)`。参数（全部零初始化）：`theta (3,)`、`t_mm (3,)`、`ctrl (grid,grid,grid,3)`、`seg_theta (K,3)`、`seg_t_mm (K,3)`。关掉的部分不建参数。
  - `transform(self, p: Tensor, seg: int | None = None) -> tuple[Tensor, Tensor]`：输入归一化位置 `(N,3)`，返回变换后的归一化位置和 `A (3,3)`（`A = D⁻¹ R_total D`）。非刚体 `u = u_max·tanh(ctrl)` 经 `grid_sample` 三线性插值（`align_corners=True`，控制点铺满 FOV），再减去控制点的均值位移；`seg` 为 `None` 或 `ref_segment` 时不加分段增量。
  - `is_identity(self) -> bool`：所有参数恰为零。
  - `set_truth(self, truth: dict) -> None`：oracle，用真值填参数并 `requires_grad_(False)`（非刚体真值把稠密场在控制点上最小二乘拟合后填入，拟合残差的脑内最大值写进返回的 `self.oracle_fit_mm`）。
  - `param_groups(self, lrs: dict) -> list[dict]`：键 `rigid_theta`、`rigid_t`、`nonrigid`、`seg_theta`、`seg_t`。
  - `regulariser(self, w: dict) -> Tensor`：`w["ctrl_l2"]·‖u‖² + w["ctrl_smooth"]·‖一阶差分‖² + w["seg_l2"]·‖seg‖² + w["seg_smooth"]·‖相邻段差‖²`。
  - `state(self) -> dict`：可 JSON 化的当前估计（`theta_deg`、`t_mm`、`ctrl` 列表、各段位姿）。
  - `render_transformed(voxelizer, positions, scales, rotations, density, adapter, seg=None) -> Tensor`（复数体）。`adapter.is_identity()` 且 `seg` 为参考段时直接返回 `voxelizer(positions, scales, rotations, density)`（保证逐位相同）；否则照 `TileVoxelizer.forward` 的步骤：`cov_inv = A⁻ᵀ (R_q S⁻² R_qᵀ) A⁻¹`，半径 `3·sqrt(diag(A Σ Aᵀ))·half_shape`，`build_tile_pairs`，`_TileVoxelizeCUDA.apply`。
  - `dense_displacement(adapter, shape, seg=None) -> np.ndarray`：`T(r) − r` 的稠密场（评估用）。

- [ ] **Step 1: 写失败的测试**（需要 GPU 的测试标 `@pytest.mark.gpu`，在任务 7 的冒烟里跑；其余 CPU 上用 `_tile_forward_pytorch` 后端的小体积）

```python
def test_identity_bitexact():
    vox, g = small_voxelizer_and_cloud()              # (32,32,24)，300 个各向异性高斯
    ad = MotionAdapter(vox.volume_shape, True, True, 4, 1)
    a = render_transformed(vox, *g, ad); b = vox(*g)
    assert torch.equal(a, b if not isinstance(b, tuple) else torch.complex(*b))

def test_translation_matches_shifted_render():
    vox, g = small_voxelizer_and_cloud(); ad = MotionAdapter(vox.volume_shape, True, False, 1, 0)
    with torch.no_grad(): ad.t_mm.copy_(torch.tensor([2.0, 0.0, -1.0]))
    moved = render_transformed(vox, *g, ad); ref = as_complex(vox(*g))
    torch.testing.assert_close(moved[4:-4, 4:-4, 4:-4], torch.roll(ref, (2, 0, -1), (0, 1, 2))[4:-4, 4:-4, 4:-4], atol=1e-4, rtol=0)

def test_rotation_matches_resampled_render():
    vox, g = small_voxelizer_and_cloud(smooth=True); ad = MotionAdapter(vox.volume_shape, True, False, 1, 0)
    with torch.no_grad(): ad.theta.copy_(torch.deg2rad(torch.tensor([0.0, 0.0, 3.0])))
    moved = render_transformed(vox, *g, ad).abs().numpy()
    ref = np.abs(warp.rigid(as_complex(vox(*g)).numpy(), geom.Rigid(geom.rotmat(np.deg2rad([0, 0, 3.0])), np.zeros(3))))
    assert np.abs(moved - ref)[4:-4, 4:-4, 4:-4].max() < 0.02 * ref.max()     # “先变换再渲染” = “先渲染再重采样”

def test_gradients_match_finite_differences():
    vox, g = small_voxelizer_and_cloud(); ad = MotionAdapter(vox.volume_shape, True, True, 3, 0).double()
    target = as_complex(vox(*g)).detach()
    f = lambda: (render_transformed(vox, *g, ad, seg=2) - target).abs().pow(2).sum()
    for name in ("theta", "t_mm", "seg_theta", "seg_t_mm"):
        assert_fd_close(f, getattr(ad, name), eps=1e-3, rtol=0.05)

def test_recovers_simulated_shift_direction():
    vox, g = small_voxelizer_and_cloud()
    target = torch.from_numpy(warp.rigid(as_complex(vox(*g)).numpy(), geom.Rigid(np.eye(3), np.array([1.5, 0, 0]))))
    ad = MotionAdapter(vox.volume_shape, True, False, 1, 0); opt = torch.optim.Adam(ad.parameters(), lr=0.05)
    for _ in range(200):
        opt.zero_grad(); (render_transformed(vox, *g, ad) - target).abs().pow(2).sum().backward(); opt.step()
    assert ad.t_mm[0].item() == pytest.approx(1.5, abs=0.1) and abs(ad.t_mm[1].item()) < 0.1   # 同号：仿真的 T 与适配器的 T 同向

def test_nonrigid_has_zero_mean_and_bounded():
    ad = MotionAdapter((32, 32, 24), False, True, 1, 0)
    with torch.no_grad(): ad.ctrl.fill_(5.0)
    d = dense_displacement(ad, (32, 32, 24))
    assert np.abs(d).max() <= 3.0 + 1e-5 and np.abs(d.reshape(-1, 3).mean(0)).max() < 0.05
```

- [ ] **Step 2: `PYTEST test_adapter.py`**，预期失败。
- [ ] **Step 3: 实现 `adapter.py`**。
- [ ] **Step 4: 再跑**，预期 6 passed。
- [ ] **Step 5: 提交** `exp(motion): motion adapter and transformed render`。

### Task 6: 训练器运行时补丁

**Files:** Create `E/patch.py`、`E/tests/test_patch.py`

**Interfaces:**
- Consumes: 任务 5；训练器副本 `TR`。
- Produces: 子进程入口 `python E/patch.py --motion <motion.json> -- <cycle.py 参数...>`，工作目录为 `TR`。`motion.json` 字段：`arm`、`channel`（`"contrast"`，单对比度 + 分段时为 `"plain"`）、`rigid`、`nonrigid`、`segments`（0 或 8）、`n_extra`（每次迭代除参考段外随机抽的段数）、`schedule_path`、`gt_ksp_path`（或 `null`）、`truth_path`（oracle 用）、`native_shift_mm`（oracle 用）、`lrs`、`reg`、`stage_iters`（`[s1, s2]`）、`out_dir`、`seed`。
- 补丁（都是在内存里替换属性，`runpy.run_path("arc/cycle.py")` 之前完成，方式同 `abl_framework.py` 的 `--child iso`）：
  1. **数据集**（仅 `gt_ksp_path` 非空）：包住 `trainers.joint_trainer._load_dataset`。对带 `gt_data_path` 标记的那个通道，用 `gt_ksp_path` 作为 `data_path` 构造数据集 `ds`（于是 GT、`kspace_multicoil`、归一化系数都来自参考位姿），再把 `ds.kspace_undersampled_complex` 换成 `(被污染的全 k 空间 / ds.normalization_scale) · mask`。标记通过 `--set data.contrast.motion_ksp_path=<污染文件>` 传入，`data_path` 指向 `gt_ksp_T2.h5`。
  2. **适配器的创建**：包住 `JointTrainer._setup_optimizer`，在其开头创建 `self.motion = MotionAdapter(...)`（放到 `self.device`；不注册到 `self.model`）；oracle 时调用 `set_truth`（真值 = 加性运动 ∘ 平移 `native_shift_mm`）。
  3. **参数组**：包住 `_optimizer_param_groups`，在返回的列表末尾追加 `self.motion.param_groups(lrs)`（oracle 不追加）。
  4. **渲染**：替换 `_render`。通道等于 `channel` 时返回 `render_transformed(..., self.motion, seg=None)`（私有高斯同样变换）；另一通道原样。分阶段：迭代数 < `stage_iters[0]` 时非刚体和分段参数 `requires_grad=False`，到点后打开（用一个在 `_kspace_loss` 里自增的迭代计数器，每次迭代该函数对本通道恰好调用一次）。
  5. **数据项**：包住 `_kspace_loss`。当 `ksp_meas is self.kspace_fit_<通道>` 且 `segments > 0`：`loss = L(vol, mask_fit ∩ seg_ref⁺) + (K−1)/n_extra · Σ_{s∈抽样} L(render(seg=s), mask_fit ∩ seg_s)`，其中 `seg_ref⁺` = 参考段的点 ∪ 已知零区 Z（`mask_fit` 中不属于采样掩模的点）∪ 不属于任何段的采样点；各项都除以同一个 `norm`。抽样用独立的 `torch.Generator(seed + 4243)`，不动全局随机流。任何情况下都把 `self.motion.regulariser(reg)` 加进本通道的返回值。
  6. **引导图的坐标系**（设计稿 6.3，仅当联合且适配器非恒等）：包住模块属性 `trainers.joint_trainer.edge_preserving_tv`。`guide is vol_p` 时换成 `no_grad` 下用 T2 变换渲染的 plain 密度；`guide is vol_c` 时换成 `no_grad` 下用恒等变换渲染的 contrast 密度。`vol_p`/`vol_c` 的身份由 `_render` 补丁记录最近一次返回的张量得到。
  7. **保存**：包住 `JointTrainer.train`，结束后写 `out_dir/<exp>_motion.json`（`self.motion.state()`、每 200 次迭代记录一次的轨迹、`oracle_fit_mm`）和 `out_dir/<exp>_canon.npz`（标准帧渲染的 contrast 复数体，`segments > 0` 时另存“只插回参考段采样线”的硬数据一致性结果 `rec_ref_dc`，见设计稿 11.1）。
  8. **不输出时间**：把 `arc/cycle.py` 模块级的 `log` 所打印的 `elapsed_s=...` 留在服务器日志里不读即可；`patch.py` 自己不打印任何时钟值。
- 另有 `patch.py --selftest`：在 `TR` 下用 30 次迭代的真实配置跑 `A0` 与“适配器全零且学习率全 0 的 A2”，断言两者的 `*_score.json` 里 `plain` 与 `contrast` 的 `psnr`、`ssim`、`nmse` 完全相等（任务 7 用）。

- [ ] **Step 1: 写失败的测试**（CPU，小数组，用假训练器对象验证补丁函数本身；这些函数在 `patch.py` 里是模块级的，可单独调用）

```python
def test_segment_loss_matches_full_when_static():
    tr = fake_trainer(K=4)                              # 静止物体，mask_fit、seg_of、Z 齐全
    full = patch.dc_loss_plain(tr, tr.vol_c)
    est = torch.stack([patch.dc_loss_segmented(tr, tr.vol_c, n_extra=1, gen=g) for g in gens(600)]).mean()
    assert est.item() == pytest.approx(full.item(), rel=0.02)
    assert patch.dc_loss_segmented(tr, tr.vol_c, n_extra=3, gen=gens(1)[0]).item() == pytest.approx(full.item(), rel=1e-5)

def test_known_zero_points_stay_in_reference_term():
    tr = fake_trainer(K=4, with_z=True); masks = patch.segment_masks(tr.mask_fit_c, tr.mask_c, tr.seg_of, ref=1, K=4)
    assert torch.equal(sum(masks), tr.mask_fit_c) and (masks[1] >= (tr.mask_fit_c - tr.mask_c).clamp_min(0)).all()

def test_param_groups_appended_last_and_survive_rebuild():
    tr = fake_trainer(K=4); groups = patch.wrap_param_groups(tr, base=lambda: [{"params": [tr.w], "lr": 1.0}])()
    assert [g.get("name") for g in groups][-5:] == ["rigid_theta", "rigid_t", "nonrigid", "seg_theta", "seg_t"]

def test_intra_dataset_uses_reference_gt(tmp_path):
    files = write_small_h5(tmp_path)                    # gt_ksp、污染 ksp、csm、mask
    ds = patch.load_dataset_with_reference_gt(files["cfg"])
    gt_only = patch._plain_load(dict(files["cfg"], data_path=files["gt"]))
    assert torch.equal(ds.ground_truth_image, gt_only.ground_truth_image)
    assert torch.equal(ds.kspace_multicoil, gt_only.kspace_multicoil)
    exp = torch.from_numpy(files["corrupt_arr"]) / gt_only.normalization_scale * ds.mask.unsqueeze(0)
    torch.testing.assert_close(ds.kspace_undersampled_complex, exp.to(torch.complex64))

def test_stage_gating():
    ad = MotionAdapter((32, 32, 24), True, True, 4, 1)
    patch.apply_stage(ad, it=10, stage_iters=(500, 1500));  assert ad.theta.requires_grad and not ad.ctrl.requires_grad
    patch.apply_stage(ad, it=500, stage_iters=(500, 1500)); assert ad.ctrl.requires_grad and ad.seg_theta.requires_grad
```

- [ ] **Step 2: `PYTEST test_patch.py`**，预期失败。
- [ ] **Step 3: 实现 `patch.py`**。
- [ ] **Step 4: 再跑**，预期 5 passed。
- [ ] **Step 5: 提交** `exp(motion): runtime patches for the frozen trainer`。

### Task 7: 启动器、账本、GPU 冒烟与 C0 回归

**Files:** Create `E/run.py`、`E/experiments.jsonl`、`E/tests/test_run.py`

**Interfaces:**
- Consumes: 任务 1、4、6。
- Produces: `run.py --af {af12,af16} --cond X --arm Y --gpu ID [--seed 42] [--smoke] [--dry] [--settings settings_frozen.json]`。
  - `ensure_trainer()`：与 `abl_framework.py` 同逻辑（SRC 在 `SRC_HEAD` 且干净；复制到 `TR`；逐个受控文件核对 sha256）。
  - `write_config(af, cond, arm) -> Path`：由 `SRC/configs/t1t2_af12_hyw.yaml` 改 `data_path`/`mask_path` 得到 `ROOT/configs/<af>_<条件>[_single_<名>].yaml`；已存在且内容不同则报错。
  - `command(af, cond, arm, gpu, seed, smoke) -> tuple[str, list[str]]`：联合组 `V32G`，单对比度组按 `abl_framework.row_cmd` 的 nojoint 规则；A0 和 I-* 直接 `arc/cycle.py`，其余经 `patch.py --motion ...`；一律 `--save_volumes`（冒烟除外）；`--set training.seed=<seed>`。
  - 运行结束后向 `ROOT/ledger.jsonl` 追加一行：`exp, af, cond, arm, seed, config_sha256, snapshot_sha256, argv, plain{head_psnr,head_ssim,nmse,head_hfen,psnr,ssim,e_unobs}, contrast{同}, motion_file`。不含任何时间或耗时字段。
  - 本地 `experiments.jsonl` = 服务器行 + `local_start_pst`、`local_end_pst`、`job_id`、`gpu`（取回时由执行者补上；本机时钟）。

- [ ] **Step 1: 写失败的测试**

```python
def test_dry_commands():
    exp, cmd = run.command("af12", "E2", "A2", "3", 42, False)
    assert exp == "ma_af12_E2_A2" and "patch.py" in " ".join(cmd) and "loss.edge_tv_guide_alpha=0.5" in cmd
    exp, cmd = run.command("af12", "E2", "I-T2", "3", 42, False)
    assert "--trainer" in cmd and "single" in cmd and not any(a.startswith("loss.edge_tv_guide_alpha") for a in cmd)
    exp, cmd = run.command("af12", "C0", "A0", "3", 42, False)
    assert cmd[2].endswith("arc/cycle.py") and "patch.py" not in " ".join(cmd)

def test_ledger_row_has_no_clock_fields(tmp_path):
    row = run.ledger_row(fake_score_json(with_elapsed=True), meta=fake_meta())
    assert not any(k in json.dumps(row) for k in ("elapsed", "utc", "timestamp", "seconds"))
```

- [ ] **Step 2: `PYTEST test_run.py`**，预期失败；实现 `run.py`；再跑，预期 2 passed。
- [ ] **Step 3: CPU 干跑全矩阵**：`run.py --dry --all`（打印 `MATRIX` 每一行的实验名与命令，并核对每个条件的数据文件存在、sha 与 manifest 一致）。预期 61 + 开发条件行，全部 `ok`。
- [ ] **Step 4: GPU 冒烟（一张空卡）**：`@python -m pytest E/tests/test_adapter.py -m gpu -q`（CUDA 后端下的 `test_identity_bitexact`、`test_gradients_match_finite_differences`、`test_rotation_matches_resampled_render`），随后 `patch.py --selftest`。预期：全部通过；自检打印 `selftest: A0 == identity-A2 (6/6 metrics equal)`。同时记下一次 `n_extra=2` 迭代的显存峰值，写进 README。
- [ ] **Step 5: C0 回归**：`run.py --af af12 --cond C0 --arm A0 --gpu ID`。预期头部 PSNR T1 35.834 ± 0.01、T2 36.305 ± 0.01。不在范围内就停下，按 superpowers:systematic-debugging 排查，不往下做。
- [ ] **Step 6: 提交** `exp(motion): launcher, ledger, GPU smoke and C0 regression`。

### Task 8: 在 D1–D3 上定适配器设置并冻结

**Files:** Create `E/settings_frozen.json`；Modify `E/README.md`（开发记录小节）

**Interfaces — Produces:** `settings_frozen.json`：`lrs{rigid_theta, rigid_t, nonrigid, seg_theta, seg_t}`、`reg{ctrl_l2, ctrl_smooth, seg_l2, seg_smooth}`、`stage_iters`、`n_extra`、`K`。之后所有带适配器的运行都传 `--settings settings_frozen.json`。

- [ ] **Step 1: 起始设置**：`rigid_theta 1e-3`（弧度）、`rigid_t 2e-2`（mm）、`nonrigid 5e-3`、`seg_theta 5e-4`、`seg_t 1e-2`；`ctrl_l2 1e-3`、`ctrl_smooth 1e-2`、`seg_l2 1e-3`、`seg_smooth 1e-2`；`stage_iters [500, 1500]`；`n_extra 2`；`K 8`。
- [ ] **Step 2: 跑 D1/A1、D2/A2、D3/A3 和 C0/A2（AF12）**，各一张空卡。用 `evaluate.py --dev`（任务 9 的函数，先实现其运动误差部分）读脑内 TRE、分段位姿误差，以及 C0/A2 相对 C0/A0 的 PSNR 差和脑内非刚体最大位移。
- [ ] **Step 3: 至多再调两轮**（总训练数 ≤ 10，含 Step 2 的 4 次）：每轮只改一组量（学习率组或正则组或 `n_extra`），依据是 D 条件的 TRE 和 C0/A2 的零运动检查。不看任何 E 条件。
- [ ] **Step 4: 关卡**：若 C0 下 A2 相对 A0 的 T1 或 T2 头部 PSNR 差 < −0.05 dB，或脑内非刚体最大位移 > 0.3 mm（S1 不成立），停下，向用户报告现象和已试过的设置，不进入任务 10。
- [ ] **Step 5: 写 `settings_frozen.json`**，在 README 记录每次开发运行的设置和结果，提交 `exp(motion): adapter settings frozen on D1-D3`。此后不再改该文件。

### Task 9: 评估

**Files:** Create `E/evaluate.py`、`E/tests/test_evaluate.py`

**Interfaces:**
- Consumes: `ROOT/ledger.jsonl`、`RES/<exp>_volumes.npz`、`<exp>_motion.json`、`<exp>_canon.npz`、各条件的真值文件。
- Produces:
  - `image_table(ledger) -> list[dict]`：每个（AF, 条件, 组）的 T1/T2 头部 PSNR、SSIM、NMSE、HFEN、`e_unobs`。
  - `normalised_delta(table, af, cond, arm) -> dict`：`Δ = PSNR(组, X) − PSNR(组, C0)`，再减 I-T2（T1 减 I-T1，无 I-T1 的条件只报未归一化值并注明）的同一差值；C0 没有该组时用 A0 的 C0 值并注明。
  - `motion_errors(exp, cond) -> dict`：脑内 `tre_mean_mm`、`tre_max_mm`（绝对：对“加性真值 ∘ 自带偏移 `NATIVE_SHIFT_MM`”；差分：对“加性真值 ∘ 同组 C0 的估计”）、`rot_err_deg`、`nonrigid_rmse_mm`、各段 `seg_rot_err_deg`/`seg_t_err_mm`。
  - `lesion_metrics(exp, cond="E3") -> dict`：每个病灶的 `contrast_recovery`（核心均值 − 厚 3 体素壳层均值，与 GT 同量之比）、`centroid_shift_mm`；串扰 = 对方对比度重建在该位置的“重建 − GT”均值，与 E2 同组同位置之差。
  - `zero_motion_checks(af) -> dict`：C0 下各适配器组与 A0 的 PSNR 差、估计平移与 `NATIVE_SHIFT_MM` 之差、脑内非刚体最大位移、各段位姿最大值。
  - `criteria(...) -> dict`：按设计稿第 12 节的阈值给出 S1–S4 各子项的 `pass`/`fail` 和数值。
  - CLI：`evaluate.py --all` 写 `ROOT/eval/summary.json` 并复制到 `MCGS_ARTIFACT_DIR`；`--dev` 只算开发条件的运动误差。

- [ ] **Step 1: 写失败的测试**

```python
def test_normalised_delta():
    t = fake_table({("C0", "A2"): 36.30, ("E2", "A2"): 36.25, ("C0", "I-T2"): 36.33, ("E2", "I-T2"): 36.31})
    assert evaluate.normalised_delta(t, "af12", "E2", "A2")["T2"] == pytest.approx(-0.03, abs=1e-6)

def test_tre_zero_for_exact_estimate_and_differential():
    truth = geom.rigid_from_pivot((0.5, 0.5, 1.5), (0.8, -0.5, 0.3)); native = geom.Rigid(np.eye(3), np.array([0.6, 0.1, -0.1]))
    est = truth.compose(native)
    r = evaluate.rigid_motion_errors(est, truth, c0_est=native, brain=ball_mask())
    assert r["tre_mean_mm_differential"] < 1e-6 and r["rot_err_deg"] < 1e-6

def test_contrast_recovery():
    gt, mask = lesion_phantom(gain=0.8); rec = gt.copy(); rec[mask] = gt[mask] * (1.4 / 1.8)     # 病灶恢复一半
    assert evaluate.contrast_recovery(rec, gt, mask) == pytest.approx(0.5, abs=0.05)

def test_criteria_thresholds():
    c = evaluate.criteria(fake_summary(s1_dpsnr=-0.06))
    assert c["S1"]["psnr"]["pass"] is False and c["S1"]["psnr"]["threshold"] == -0.05
```

- [ ] **Step 2: `PYTEST test_evaluate.py`**，预期失败；实现；再跑，预期 4 passed。
- [ ] **Step 3: 提交** `exp(motion): evaluation of images, motion, lesions and criteria`。

### Task 10: 正式运行

**Files:** Modify `E/experiments.jsonl`、`E/README.md`（进展表）

- [ ] **Step 1: AF12 核心**（`MATRIX["af12_core"]`：C0、E1a、E2、E3、E4b 的各组，C0/A0 已在任务 7 完成）。有几张空卡就并行几个，每次启动前按 Global Constraints 检查卡；每个作业结束后取回账本行、补本机时间、追加 `experiments.jsonl`。
- [ ] **Step 2: 中途检查**：核心条件跑完后运行 `evaluate.py --all`，把 S1、S2（E1a/E2）、S3、S4（E4b）的初步结论写进 README 进展表并提交。不因结果改任何设置。
- [ ] **Step 3: AF12 其余**（`MATRIX["af12_rest"]`：E1b、E4a、E4c）。
- [ ] **Step 4: AF16 子集**（`MATRIX["af16"]`）。
- [ ] **Step 5: 可选项，只在有空卡时做，按此顺序**：C0 与 E2 的 A0、A2 补 seed 7；E4b 的 I-T2+S；交错时间表变体（A0、A3、O）。没做的在 RESULTS.md 里写明“未做”。
- [ ] **Step 6: 任何一次训练失败**：停下该条件，读日志（过滤掉含时间的行），按 superpowers:systematic-debugging 处理；是代码缺陷则修复、补测试、重跑受影响的全部运行，并在账本里把旧行标 `superseded`。
- [ ] **Step 7: 提交** `results(motion): AF12 and AF16 runs recorded`。

### Task 11: 图、RESULTS.md、收尾

**Files:** Create `E/figures.py`、`E/RESULTS.md`、`E/figures/*.png`（小图入库，600 dpi 大图放 `figures/big/`）

**Interfaces — Consumes:** `ROOT/eval/summary.json`、各运行的体数据；记录格式脚本 `autoresearch/ce_af16_lesion_figs_20261107` 一线的现有三平面出图函数（按 `autoresearch/ce_af12_rerun_20261112/fig_af12_prepost.py` 的做法调用，不改原脚本）。

- [ ] **Step 1: `figures.py`** 生成设计稿 11.4 的五类图：`fig_E2_T2_threeplane.png`（GT、A0、A2、O、I-T2）、`fig_E4b_T2_threeplane.png`（GT、A0、A3、O、I-T2）、`fig_E3_lesions.png`（T1、T2 各一行放大）、`fig_motion_recovery.png`（估计与真值；各段位姿随链号变化）、`fig_delta_psnr.png`。图上不出现任何时间。
- [ ] **Step 2: 在服务器 CPU 上出图并取回**，用 VS Code 打开每张 PNG 逐一查看（切面是否落在病灶上、色标是否统一、标签是否正确）。
- [ ] **Step 3: 写 `RESULTS.md`**：一段话结论；S1–S4 判据表（数值、阈值、通过与否）；AF12、AF16 图像指标表；运动恢复表；病灶表；注意事项（单被试、单种子、扫描内时间表是假设、“全采样”是厂商补全数据、oracle 的自带偏移来自影像法估计、联合与独立的差距不是本工作的判据、未做的可选项）。不达标的判据如实写。
- [ ] **Step 4: 核对**：按 superpowers:verification-before-completion，重新运行 `PYTEST`（全部测试文件）和 `evaluate.py --all`，确认 RESULTS.md 的每个数字都能在 `summary.json` 里找到。
- [ ] **Step 5: 提交** `results(motion): figures and RESULTS.md`；按用户的展示规则先给 PNG 路径，再给简短结论。是否推送由用户决定。

---

## 自查记录

- **设计稿覆盖**：G1 → 任务 5、6；G2 → 任务 2–4；G3 → 任务 7、8、10；G4 → 任务 9、11。设计稿第 14 节的 7 个风险：1–3 已由读源码核实（上文“已核实的训练器事实”），3 另有有限差分测试；4 由去均值和固定参考段处理并有测试；5 在任务 7 Step 4 实测；6 用差分 TRE 处理（任务 9）；7 写进 RESULTS.md 的注意事项。
- **与设计稿的一处出入**：设计稿第 10 节写 AF12 “约 50 次”，逐格数是 40 次测试条件运行加约 10 次开发运行；AF16 逐格数是 21 次。本计划按逐格数。
- **设计稿 6.3 的取舍**：跨对比度引导图不做重采样，改为在对方坐标系里多渲染一次（`no_grad`），避免插值模糊；每次迭代多两次无梯度渲染。
