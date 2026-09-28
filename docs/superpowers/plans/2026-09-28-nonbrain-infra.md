# nonbrain-infra(公共分支)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `nonbrain-infra` 分支上加一层共享设施:修掉大体积下的 torch.quantile 崩溃,新增严格入口 `arc/run_spec.py`(缺键即中止、只收 `--spec/--exp/--gpu`),并用测试和重打分证明脑的流程和数字一个不变。

**Architecture:** 现有文件只改 `paired_gs/data/dataset.py` 的一处归一化分位数。其余全是新增文件:`arc/strict_config.py`(记录"读到代码默认值"的字典)、`arc/spec_io.py`(spec 读取、校验、溯源、补默认值)、`arc/spec_scoring.py`(cycle.py 打分函数的参数化副本)、`arc/run_spec.py`(进程内训练加打分的薄入口)、`arc/spec_expand.py`(模板按受试者展开)。脑的等价性用同一个 hp_n36 checkpoint,分别经 cycle.py 和 run_spec.py 打分,两边逐项比对。

**Tech Stack:** Python 3.11(GSTMI 环境,torch 2.7.1+cu118)、PyYAML、numpy、pytest;GPU 只用于第 5 步和第 8 步。

**Spec:** `docs/superpowers/specs/2026-09-28-nonbrain-af16-infra-knee-design.md`(第 2、3 节)

## Global Constraints

- 工作树:`/data0/congcong/code/GS/mcgs_nb_infra_wt`,分支 `nonbrain-infra`(起点 2eef592)。所有命令在这里执行。
- Python:`PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python`,一律 `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY ...`;用 GPU 时加 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。
- Bash 工具跑的是 zsh:未加引号的 `$VAR` 不分词,`"$X:path"` 会被吃掉。带循环或变量的命令写成 bash 脚本再 `bash script.sh`。
- 不得修改:`arc/cycle.py`、`paired_gs/trainers/joint_trainer.py`、`diag_artefact/headonly.py`、`diag_artefact/volumes.py`、`paired_gs/metrics/metrics.py`、`configs/` 下任何文件、`run_*_mcgs.sh`。唯一改动的既有文件是 `paired_gs/data/dataset.py`。
- 不删除任何文件或数据(rm、git clean、git reset --hard、git checkout -- 等都被钩子拦截)。
- 提交:`git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit`,提交信息写成普通人话,不加任何 AI 署名行;只 `git add` 本任务点名的文件(工作树里有很多历史未跟踪文件)。
- GPU:启动前用 `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader` 找已用显存 < 1 GB 的卡;每卡只放一个任务;不停别人的进程。
- 分位数门槛:`16_000_000`(与 `dataset.py:306` 已有写法一致;脑体积 11,534,336 走原分支)。
- 脑重打分基准:hp_n36 的 head PSNR 为 35.706153059238126 / 34.92807957136013(plain / contrast),容差 1e-4 dB。
- 不推送到 GitHub,推送前问用户。
- 位置说明(相对 spec 的小调整):`paired_gs` 下没有 `utils` 包,所以 `StrictConfig` 放在 `arc/strict_config.py`,和入口放在一起。
- 这个工作树是新检出的:第一次用到 tile 体素器时会在本工作树里编译 CUDA 扩展(几分钟);少数依赖被 gitignore 的本地数据的旧测试可能 skip,全量测试数量以实际为准并写明原因。

## 文件结构

| 文件 | 职责 |
|---|---|
| `paired_gs/data/dataset.py`(改 :372 附近) | 回顾性真值归一化:>16M 元素时改用 numpy 分位数 |
| `arc/strict_config.py`(新) | `StrictConfig`:嵌套字典;缺键的 `.get/.setdefault` 在严格模式抛错、记录模式下记下来 |
| `arc/spec_io.py`(新) | spec 读取与校验、运行名检查、运行目录、训练器配置、溯源(sha256、git、md5)、按审计结果补默认值(含命令行 `fill-defaults`) |
| `arc/spec_scoring.py`(新) | cycle.py 打分函数的副本:解剖掩膜阈值与白质颗粒指标可配,大体积分位数改 numpy |
| `arc/run_spec.py`(新) | 严格入口:`--spec --exp --gpu [--audit N]`,进程内训练、打分、存 json 与体积 |
| `arc/spec_expand.py`(新) | 把含 `{subject}` 的模板展开成每个受试者一份 spec;校验"只有路径不同" |
| `scripts/make_brain_equiv_spec.py`(新) | 从 hp_n36 的配置快照生成脑等价 spec |
| `scripts/regress_brain_pin.sh`(新) | 用不变的 cycle.py 对 hp_n36 checkpoint 重打分并比对基准 |
| `specs/brain_hp_n36_equiv.yaml`(新,生成后入库) | 脑等价 spec(只打分,不训练) |
| `tests/data/brain_pinned_md5.txt`(新,生成) | 脑相关共享文件在 2eef592 的 md5 |
| `tests/test_dataset_quantile_gate.py`、`tests/test_strict_config.py`、`tests/test_spec_io.py`、`tests/test_spec_scoring.py`、`tests/test_scoring_equivalence.py`、`tests/test_spec_expand.py`、`tests/test_brain_files_pinned.py`(新) | 各自对应上面的单元 |

---

### Task 1: dataset.py 的分位数分支

**Files:**
- Modify: `paired_gs/data/dataset.py:364-374`(`_normalize_intensity_scale`)
- Test: `tests/test_dataset_quantile_gate.py`

**Interfaces:**
- Consumes: 无
- Produces: `MRIDataset._normalize_intensity_scale` 对 >16,000,000 元素的真值不再报错;≤16,000,000 时行为逐位不变。

- [ ] **Step 1: 写失败的测试**

`tests/test_dataset_quantile_gate.py`:

```python
"""dataset.py retrospective normalisation: torch.quantile caps at 2**24 elements. Volumes above
16,000,000 elements must use numpy (same fallback as the presampled branch, dataset.py:306-313);
volumes at or below it keep the torch path bit for bit (every brain topic is 11,534,336 voxels).

Run: env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_dataset_quantile_gate.py -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
from data.dataset import MRIDataset  # noqa: E402


def _stub(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    gt = torch.complex(torch.rand(n, generator=g), torch.rand(n, generator=g))
    return SimpleNamespace(normalize_kspace=True, normalization_percentile=99.9,
                           ground_truth_image=gt.clone(), kspace_multicoil=torch.ones(2, 4, dtype=torch.complex64))


def test_brain_sized_volume_keeps_torch_quantile_bit_for_bit():
    s = _stub(1_000_000)
    expected = float(torch.clamp(torch.quantile(torch.abs(s.ground_truth_image), 0.999), min=1e-8).item())
    MRIDataset._normalize_intensity_scale(s)
    assert s.normalization_scale == expected


def test_volume_above_16m_elements_uses_numpy_quantile():
    s = _stub(17_000_000)                      # > 2**24 = 16,777,216: torch.quantile raises on this
    expected = float(np.quantile(torch.abs(s.ground_truth_image).numpy(), 0.999))
    MRIDataset._normalize_intensity_scale(s)
    assert s.normalization_scale == pytest.approx(expected, rel=1e-6)
```

- [ ] **Step 2: 运行,确认失败**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_dataset_quantile_gate.py -q`
Expected: 第一个 PASS;第二个 FAIL,报 `RuntimeError: quantile() input tensor is too large`。

- [ ] **Step 3: 最小实现**

把 `paired_gs/data/dataset.py` 的 `_normalize_intensity_scale` 中这三行:

```python
        if self.normalization_percentile >= 100.0:
            scale = ground_truth_mag.max()
        else:
            scale = torch.quantile(ground_truth_mag, self.normalization_percentile / 100.0)
```

替换为:

```python
        if self.normalization_percentile >= 100.0:
            scale = ground_truth_mag.max()
        elif ground_truth_mag.numel() <= 16_000_000:
            scale = torch.quantile(ground_truth_mag, self.normalization_percentile / 100.0)
        else:
            # torch.quantile caps at ~16M elements; use numpy for larger tensors (same fallback as the
            # presampled branch above). Volumes of 16M elements or fewer keep the torch path unchanged.
            scale = torch.tensor(
                float(np.quantile(ground_truth_mag.cpu().numpy(), self.normalization_percentile / 100.0)),
                dtype=ground_truth_mag.dtype, device=ground_truth_mag.device,
            )
```

- [ ] **Step 4: 运行,确认通过**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_dataset_quantile_gate.py -q`
Expected: `2 passed`

- [ ] **Step 5: 提交**

```bash
git add paired_gs/data/dataset.py tests/test_dataset_quantile_gate.py
git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit -m "dataset: numpy quantile for ground truths above 16M elements (brain-sized volumes keep the torch path)"
```

---

### Task 2: StrictConfig

**Files:**
- Create: `arc/strict_config.py`
- Test: `tests/test_strict_config.py`

**Interfaces:**
- Consumes: 无
- Produces: `StrictConfig(data: dict, *, strict: bool = True)`,`dict` 子类;属性 `fallbacks -> list[tuple[str, Any]]`(点分路径与默认值);方法 `to_plain() -> dict`;异常 `StrictConfigError(KeyError)`。导入即为 `yaml.Dumper` 与 `yaml.SafeDumper` 注册表示器,`yaml.dump` 输出普通映射。

- [ ] **Step 1: 写失败的测试**

`tests/test_strict_config.py`:

```python
"""StrictConfig: reads that fall back to a code default are recorded (record mode) or refused (strict mode).

Run: env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_strict_config.py -q
"""
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "arc"))
from strict_config import StrictConfig, StrictConfigError  # noqa: E402

CFG = {"loss": {"tv_weight": 0.1, "edge_tv": True}, "training": {"max_iterations": 4000}, "name": "x"}


def test_present_keys_read_like_a_dict():
    s = StrictConfig(CFG)
    assert s.get("name", "d") == "x"
    assert s["loss"].get("tv_weight", 0.0) == 0.1
    assert s.get("training", {}).get("max_iterations", 8000) == 4000
    assert "edge_tv" in s["loss"]
    assert s.fallbacks == []


def test_missing_key_raises_in_strict_mode():
    s = StrictConfig(CFG)
    with pytest.raises(StrictConfigError, match="loss.lambda_ptv"):
        s["loss"].get("lambda_ptv", 0.0)


def test_missing_keys_recorded_in_record_mode_including_children_of_missing_sections():
    s = StrictConfig(CFG, strict=False)
    assert s["loss"].get("lambda_ptv", 0.0) == 0.0
    assert s.get("mode", {}).get("strict_cuda", False) is False
    assert s.fallbacks == [("loss.lambda_ptv", 0.0), ("mode", {}), ("mode.strict_cuda", False)]


def test_setdefault_is_tracked_and_stores_a_wrapped_value():
    s = StrictConfig(CFG, strict=False)
    v = s.setdefault("voxelizer", {})
    v["type"] = "tile_cuda"
    assert s["voxelizer"]["type"] == "tile_cuda"
    assert isinstance(s["voxelizer"], StrictConfig)
    assert ("voxelizer", {}) in s.fallbacks


def test_assignment_wraps_nested_dicts_and_keeps_paths():
    s = StrictConfig(CFG)
    s["output"] = {"output_dir": "/tmp/x"}
    assert isinstance(s["output"], StrictConfig)
    with pytest.raises(StrictConfigError, match="output.exp_name"):
        s["output"].get("exp_name", "d")


def test_dumps_as_plain_mapping():
    s = StrictConfig(CFG)
    assert yaml.safe_load(yaml.dump(s)) == CFG
    assert json.loads(json.dumps(s)) == CFG
    plain = s.to_plain()
    assert plain == CFG and type(plain) is dict and type(plain["loss"]) is dict
```

- [ ] **Step 2: 运行,确认失败**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_strict_config.py -q`
Expected: 收集阶段报 `ModuleNotFoundError: No module named 'strict_config'`。

- [ ] **Step 3: 实现**

`arc/strict_config.py`:

```python
"""StrictConfig: a nested dict that records every read that falls back to a code default.

JointTrainer and the loader read their settings with ``cfg.get(key, default)``; a key missing from the
spec silently takes the code default. run_spec.py hands the trainer a StrictConfig instead of a dict:
in strict mode the first fallback raises StrictConfigError (the run stops before it can use a value
nobody wrote down); in record mode (``run_spec.py --audit N``) fallbacks are collected so the defaults
can be written into the spec. Spec 2026-09-28 §2.2 / §3.2.
"""
from __future__ import annotations

import copy
from typing import Any, List, Tuple

import yaml


class StrictConfigError(KeyError):
    """A config key was read that is not in the spec."""


class StrictConfig(dict):
    """dict subclass; every nested dict value is itself a StrictConfig sharing one fallback log."""

    def __init__(self, data=None, *, strict: bool = True, _path: str = "", _log: list | None = None):
        super().__init__()
        self._strict = strict
        self._path = _path
        self._log: List[Tuple[str, Any]] = [] if _log is None else _log
        for key, value in (data or {}).items():
            super().__setitem__(key, self._wrap(key, value))

    def _child_path(self, key) -> str:
        return f"{self._path}.{key}" if self._path else str(key)

    def _wrap(self, key, value):
        if isinstance(value, dict) and not isinstance(value, StrictConfig):
            return StrictConfig(value, strict=self._strict, _path=self._child_path(key), _log=self._log)
        return value

    def _fallback(self, key, default) -> None:
        path = self._child_path(key)
        try:
            recorded = copy.deepcopy(default)
        except Exception:  # noqa: BLE001 - an uncopyable default is still worth recording by reference
            recorded = default
        self._log.append((path, recorded))
        if self._strict:
            raise StrictConfigError(
                f"config key '{path}' is not in the spec; the code default {default!r} would be used")

    def get(self, key, default=None):
        if key in self:
            return super().__getitem__(key)
        self._fallback(key, default)
        if isinstance(default, dict):          # record mode: keep tracking reads below a missing section
            return StrictConfig(default, strict=False, _path=self._child_path(key), _log=self._log)
        return default

    def setdefault(self, key, default=None):
        if key in self:
            return super().__getitem__(key)
        self._fallback(key, default)
        value = self._wrap(key, default)
        super().__setitem__(key, value)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, self._wrap(key, value))

    @property
    def fallbacks(self) -> List[Tuple[str, Any]]:
        return list(self._log)

    def to_plain(self) -> dict:
        return {k: (v.to_plain() if isinstance(v, StrictConfig) else v) for k, v in self.items()}


def _represent(dumper, data):
    return dumper.represent_dict(data)


yaml.add_representer(StrictConfig, _represent, Dumper=yaml.Dumper)
yaml.add_representer(StrictConfig, _represent, Dumper=yaml.SafeDumper)
```

- [ ] **Step 4: 运行,确认通过**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_strict_config.py -q`
Expected: `6 passed`

- [ ] **Step 5: 提交**

```bash
git add arc/strict_config.py tests/test_strict_config.py
git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit -m "arc: StrictConfig, a config dict that refuses (or records) reads that fall back to a code default"
```

---

### Task 3: spec_io(spec 读取、校验、溯源、补默认值)

**Files:**
- Create: `arc/spec_io.py`
- Test: `tests/test_spec_io.py`

**Interfaces:**
- Consumes: 无(纯 Python,不 import torch)
- Produces:
  - `SpecError(ValueError)`
  - `load_spec(path) -> dict`(读 yaml 并调用 `validate_spec`)
  - `validate_spec(spec: dict) -> None`
  - `check_exp(exp: str, spec: dict) -> None`
  - `run_dir(exp: str, spec: dict) -> pathlib.Path`(`results_root/exp`,已存在则 `SpecError`)
  - `trainer_config(spec: dict, exp: str) -> dict`(深拷贝 `spec["trainer"]`,填 `output.output_dir = results_root/exp`、`output.exp_name = exp`)
  - `md5_file(path) -> str`
  - `provenance(spec_path, spec: dict, repo_root) -> dict`(键:`spec_sha256`、`git_commit`、`git_dirty`、`data_md5`)
  - `fill_defaults(spec: dict, fallbacks: list[tuple[str, Any]]) -> dict`
  - 命令行:`python arc/spec_io.py fill-defaults SPEC AUDIT_JSON --out OUT`

spec 文件格式(三节,缺一不可,多出的键报错):

```yaml
spec:
  dataset: <id>            # 数据集标识
  exp_prefix: <prefix>     # 运行名必须以 "<prefix>_" 开头;只允许 [a-z0-9]
  results_root: /abs/path  # 运行目录 = results_root/<exp>,不许已存在
  save_volumes: false
  from_ckpt: null          # 路径:跳过训练,载入 model_state 后打分
scoring:
  anatomy_rel: 0.05        # 解剖掩膜 = |GT| > anatomy_rel * P99.9(|GT|)
  grain_ratio: true        # 是否计算脑白质颗粒指标 N
trainer:                   # 原样交给 JointTrainer(经 StrictConfig 包装)
  data: {plain: {...}, contrast: {...}}
  training: {seed: 42, deterministic: false, max_iterations: ..., shared_warmup_iters: ..., ...}
  ...
```

- [ ] **Step 1: 写失败的测试**

`tests/test_spec_io.py`:

```python
"""spec_io: spec validation, run naming, provenance and default filling (no torch, no GPU).

Run: env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_spec_io.py -q
"""
import copy
import hashlib
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "arc"))
import spec_io  # noqa: E402
from spec_io import SpecError  # noqa: E402


def _spec(tmp_path):
    paths = {}
    for n in ("ksp.h5", "csm.h5", "mask.h5"):
        p = tmp_path / n
        p.write_bytes(n.encode())
        paths[n] = str(p)
    ch = {"name": "E1", "data_path": paths["ksp.h5"], "csm_path": paths["csm.h5"], "mask_path": paths["mask.h5"]}
    return {"spec": {"dataset": "toy", "exp_prefix": "toy", "results_root": str(tmp_path / "res"),
                     "save_volumes": False, "from_ckpt": None},
            "scoring": {"anatomy_rel": 0.05, "grain_ratio": False},
            "trainer": {"data": {"plain": dict(ch), "contrast": dict(ch, name="E2")},
                        "training": {"seed": 42, "deterministic": False, "max_iterations": 10,
                                     "shared_warmup_iters": 2},
                        "loss": {"tv_weight": 0.1}, "output": {"use_tensorboard": False}}}


def test_valid_spec_passes_and_loads_from_file(tmp_path):
    s = _spec(tmp_path)
    spec_io.validate_spec(s)
    f = tmp_path / "s.yaml"
    f.write_text(yaml.safe_dump(s))
    assert spec_io.load_spec(f) == s


@pytest.mark.parametrize("mutate, message", [
    (lambda s: s.pop("scoring"), "top-level"),
    (lambda s: s["spec"].update(extra=1), "spec block"),
    (lambda s: s["scoring"].update(extra=1), "scoring block"),
    (lambda s: s["spec"].update(results_root="rel/path"), "absolute"),
    (lambda s: s["spec"].update(exp_prefix="Knee"), "exp_prefix"),
    (lambda s: s["trainer"]["output"].update(output_dir="/x"), "output_dir"),
    (lambda s: s["trainer"]["loss"].update(lambda_geo=0.1), "lambda_geo"),
    (lambda s: s["trainer"]["training"].pop("seed"), "training.seed"),
    (lambda s: s["trainer"]["data"]["plain"].update(csm_path="rel.h5"), "absolute"),
])
def test_invalid_specs_are_refused(tmp_path, mutate, message):
    s = _spec(tmp_path)
    mutate(s)
    with pytest.raises(SpecError, match=message):
        spec_io.validate_spec(s)


def test_exp_name_must_carry_the_prefix(tmp_path):
    s = _spec(tmp_path)
    spec_io.check_exp("toy_joint_01", s)
    for bad in ("joint_01", "toyjoint", "toy_a/b"):
        with pytest.raises(SpecError):
            spec_io.check_exp(bad, s)


def test_run_dir_refuses_an_existing_run(tmp_path):
    s = _spec(tmp_path)
    d = spec_io.run_dir("toy_a", s)
    assert d == Path(s["spec"]["results_root"]) / "toy_a"
    d.mkdir(parents=True)
    with pytest.raises(SpecError, match="exists"):
        spec_io.run_dir("toy_a", s)


def test_trainer_config_fills_output_and_leaves_the_spec_alone(tmp_path):
    s = _spec(tmp_path)
    before = copy.deepcopy(s)
    cfg = spec_io.trainer_config(s, "toy_a")
    assert cfg["output"]["output_dir"] == str(Path(s["spec"]["results_root"]) / "toy_a")
    assert cfg["output"]["exp_name"] == "toy_a"
    assert s == before


def test_provenance_hashes(tmp_path):
    s = _spec(tmp_path)
    f = tmp_path / "s.yaml"
    f.write_text(yaml.safe_dump(s))
    p = spec_io.provenance(f, s, ROOT)
    assert p["spec_sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()
    ksp = s["trainer"]["data"]["plain"]["data_path"]
    assert p["data_md5"][ksp] == hashlib.md5(Path(ksp).read_bytes()).hexdigest()
    assert len(p["git_commit"]) == 40 and isinstance(p["git_dirty"], bool)


def test_fill_defaults_inserts_missing_keys_only(tmp_path):
    s = _spec(tmp_path)
    out = spec_io.fill_defaults(s, [("loss.tv_weight", 9.9), ("loss.lambda_ptv", 0.0),
                                    ("mode", {}), ("mode.strict_cuda", False),
                                    ("gaussian.init.scale", 0.008)])
    assert out["trainer"]["loss"] == {"tv_weight": 0.1, "lambda_ptv": 0.0}
    assert out["trainer"]["mode"] == {"strict_cuda": False}
    assert out["trainer"]["gaussian"] == {"init": {"scale": 0.008}}
    assert "mode" not in s["trainer"]
```

- [ ] **Step 2: 运行,确认失败**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_spec_io.py -q`
Expected: 收集阶段 `ModuleNotFoundError: No module named 'spec_io'`。

- [ ] **Step 3: 实现**

`arc/spec_io.py`:

```python
"""spec_io: read, check and trace the spec files that drive arc/run_spec.py (spec 2026-09-28 §2.2, §3.2).

A spec has exactly three blocks: `spec` (run bookkeeping), `scoring` (the two scoring settings that are
constants in cycle.py) and `trainer` (the complete JointTrainer config). Nothing here imports torch.

CLI:  python arc/spec_io.py fill-defaults SPEC AUDIT_JSON --out OUT
      writes SPEC with every fallback recorded by `run_spec.py --audit` inserted under `trainer`.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, Tuple

import yaml

SPEC_KEYS = {"dataset", "exp_prefix", "results_root", "save_volumes", "from_ckpt"}
SCORING_KEYS = {"anatomy_rel", "grain_ratio"}
CHANNELS = ("plain", "contrast")
PATH_KEYS = ("data_path", "csm_path", "mask_path")


class SpecError(ValueError):
    """The spec file breaks a rule of spec 2026-09-28 §2.2."""


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise SpecError(msg)


def validate_spec(spec: dict) -> None:
    _need(isinstance(spec, dict) and set(spec) == {"spec", "scoring", "trainer"},
          f"top-level blocks must be exactly spec/scoring/trainer, got {sorted(spec) if isinstance(spec, dict) else spec!r}")
    s, sc, tr = spec["spec"], spec["scoring"], spec["trainer"]
    _need(set(s) == SPEC_KEYS, f"spec block keys must be {sorted(SPEC_KEYS)}, got {sorted(s)}")
    _need(set(sc) == SCORING_KEYS, f"scoring block keys must be {sorted(SCORING_KEYS)}, got {sorted(sc)}")
    _need(re.fullmatch(r"[a-z0-9]+", str(s["exp_prefix"])) is not None, "exp_prefix must match [a-z0-9]+")
    _need(Path(str(s["results_root"])).is_absolute(), "spec.results_root must be an absolute path")
    _need(isinstance(s["save_volumes"], bool), "spec.save_volumes must be a bool")
    _need(s["from_ckpt"] is None or isinstance(s["from_ckpt"], str), "spec.from_ckpt must be null or a path")
    _need(isinstance(sc["anatomy_rel"], (int, float)) and 0.0 < float(sc["anatomy_rel"]) < 1.0,
          "scoring.anatomy_rel must be in (0, 1)")
    _need(isinstance(sc["grain_ratio"], bool), "scoring.grain_ratio must be a bool")
    data = tr.get("data", {})
    for ch in CHANNELS:
        _need(ch in data, f"trainer.data.{ch} is missing")
        _need("name" in data[ch], f"trainer.data.{ch}.name is missing")
        for k in PATH_KEYS:
            _need(k in data[ch] and Path(str(data[ch][k])).is_absolute(),
                  f"trainer.data.{ch}.{k} must be an absolute path")
    out = tr.get("output", {})
    for k in ("output_dir", "exp_name"):
        _need(k not in out, f"trainer.output.{k} is set by run_spec.py; remove it from the spec")
    loss = tr.get("loss", {})
    _need(not ("lambda_geo" in loss and "tv_weight" in loss),
          "trainer.loss has both lambda_geo and tv_weight; JointTrainer would silently use lambda_geo "
          "(joint_trainer.py:781-791); keep tv_weight only")
    training = tr.get("training", {})
    for k in ("seed", "deterministic", "max_iterations", "shared_warmup_iters"):
        _need(k in training, f"trainer.training.{k} is missing")


def load_spec(path) -> dict:
    spec = yaml.safe_load(Path(path).read_text())
    validate_spec(spec)
    return spec


def check_exp(exp: str, spec: dict) -> None:
    prefix = spec["spec"]["exp_prefix"] + "_"
    _need(exp.startswith(prefix) and len(exp) > len(prefix), f"run name must start with '{prefix}', got {exp!r}")
    _need(re.fullmatch(r"[A-Za-z0-9_.-]+", exp) is not None, f"run name has characters outside [A-Za-z0-9_.-]: {exp!r}")


def run_dir(exp: str, spec: dict) -> Path:
    d = Path(spec["spec"]["results_root"]) / exp
    _need(not d.exists(), f"run directory {d} exists; pick a new run name (runs are never overwritten)")
    return d


def trainer_config(spec: dict, exp: str) -> dict:
    cfg = copy.deepcopy(spec["trainer"])
    out = cfg.setdefault("output", {})
    out["output_dir"] = str(Path(spec["spec"]["results_root"]) / exp)
    out["exp_name"] = exp
    return cfg


def md5_file(path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def provenance(spec_path, spec: dict, repo_root) -> dict:
    spec_path, repo_root = Path(spec_path), Path(repo_root)
    commit = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip() != ""
    paths = sorted({str(spec["trainer"]["data"][ch][k]) for ch in CHANNELS for k in PATH_KEYS})
    return {"spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            "git_commit": commit, "git_dirty": dirty,
            "data_md5": {p: md5_file(p) for p in paths}}


def fill_defaults(spec: dict, fallbacks: Iterable[Tuple[str, Any]]) -> dict:
    out = copy.deepcopy(spec)
    for path, default in fallbacks:
        node = out["trainer"]
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            _need(isinstance(node, dict), f"cannot insert {path}: '{part}' is not a mapping")
        if parts[-1] not in node:
            node[parts[-1]] = copy.deepcopy(default)
    return out


def _main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="spec utilities")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fill-defaults", help="insert the fallbacks recorded by run_spec.py --audit")
    f.add_argument("spec")
    f.add_argument("audit_json")
    f.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = yaml.safe_load(Path(a.spec).read_text())
    entries = json.loads(Path(a.audit_json).read_text())["fallbacks"]
    bad = [e["key"] for e in entries if isinstance(e["default"], dict) and "__repr__" in e["default"]]
    if bad:
        raise SystemExit(f"defaults that are not YAML-serialisable, set them by hand: {bad}")
    out = fill_defaults(spec, [(e["key"], e["default"]) for e in entries])
    validate_spec(out)
    Path(a.out).write_text(yaml.safe_dump(out, sort_keys=True))
    print(f"filled {len(entries)} fallback(s) -> {a.out}")


if __name__ == "__main__":
    _main()
```

- [ ] **Step 4: 运行,确认通过**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_spec_io.py -q`
Expected: `15 passed`(参数化的 9 个非法用例 + 6 个其他)

- [ ] **Step 5: 提交**

```bash
git add arc/spec_io.py tests/test_spec_io.py
git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit -m "arc: spec_io -- spec file rules (three blocks, absolute paths, prefixed run names, no overwrite), provenance and default filling"
```

---

### Task 4: spec_scoring(cycle.py 打分函数的参数化副本)

**Files:**
- Create: `arc/spec_scoring.py`
- Test: `tests/test_spec_scoring.py`

**Interfaces:**
- Consumes: `paired_gs/data/transforms.py` 的 `fft3c, ifft3c`;`paired_gs/metrics` 的 `evaluate_reconstruction`;`diag_artefact/headonly.py` 的 `head_psnr, head_ssim, head_hfen, grain_ratio`(只 import,不改)。
- Produces:
  - `QUANTILE_TORCH_MAX = 16_000_000`
  - `quantile(t: torch.Tensor, q: float) -> torch.Tensor`
  - `hfen(rec, gt) -> float`、`sharp_ratio(rec, gt) -> float`、`grain(rec, gt) -> float`(与 cycle.py 同名同义)
  - `score_anatomy(rec: np.ndarray, gt: np.ndarray, anatomy_rel: float, with_grain_ratio: bool) -> dict`(键:`head_psnr, head_ssim, head_hfen, grain_ratio, head_frac, gmax`;不算颗粒时 `grain_ratio` 为 `None`)
  - `score_channel(tr, which: str, dev, anatomy_rel: float, with_grain_ratio: bool) -> dict`(键与 `cycle.score_contrast` 返回的完全相同)

- [ ] **Step 1: 写失败的测试**

`tests/test_spec_scoring.py`:

```python
"""spec_scoring: the scoring copy used by run_spec.py. The copied helpers must stay textually identical to
arc/cycle.py (checked without importing cycle.py, whose argparse runs at import), the anatomy metrics must
equal diag_artefact.headonly.score_headonly at the brain settings, and the >16M-element path must work.
The full score_channel == cycle.score_contrast check is tests/test_scoring_equivalence.py (GPU runs).

Run: env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_spec_scoring.py -q
"""
import ast
import inspect
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "arc"))
sys.path.insert(0, str(ROOT))
import spec_scoring  # noqa: E402
from diag_artefact.headonly import score_headonly  # noqa: E402


def _cycle_function_source(name):
    src = (ROOT / "arc/cycle.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise AssertionError(f"{name} not found in cycle.py")


def test_copied_helpers_match_cycle_source():
    for name in ("hfen", "sharp_ratio"):
        assert inspect.getsource(getattr(spec_scoring, name)).strip() == _cycle_function_source(name).strip()
    cyc = _cycle_function_source("grain")
    ours = inspect.getsource(spec_scoring.grain).strip()
    assert ours == cyc.replace("thr = torch.quantile(gg[brain], 0.30)", "thr = quantile(gg[brain], 0.30)").strip()


def test_quantile_torch_path_below_the_cap_and_numpy_above():
    small = torch.rand(1_000_000, generator=torch.Generator().manual_seed(0))
    assert torch.equal(spec_scoring.quantile(small, 0.3), torch.quantile(small, 0.3))
    big = torch.rand(17_000_000, generator=torch.Generator().manual_seed(1))
    assert float(spec_scoring.quantile(big, 0.3)) == float(np.float32(np.quantile(big.numpy(), 0.3)))


def _volume(seed=0, shape=(48, 56, 40)):
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij")
    obj = (zz ** 2 + yy ** 2 + xx ** 2 < 0.7).astype(np.float32)
    gt = ndimage.gaussian_filter(obj * (1 + 0.3 * rng.standard_normal(shape)), 1.0).astype(np.float32)
    rec = (gt + 0.02 * rng.standard_normal(shape)).astype(np.float32)
    return rec, gt


def test_anatomy_metrics_equal_headonly_at_brain_settings():
    rec, gt = _volume()
    ours = spec_scoring.score_anatomy(rec, gt, 0.05, True)
    ref = score_headonly(rec, gt)
    assert set(ours) == set(ref)
    for k in ref:
        a, b = ours[k], ref[k]
        assert (math.isnan(a) and math.isnan(b)) or a == b, k


def test_anatomy_threshold_and_grain_switch():
    rec, gt = _volume(1)
    loose = spec_scoring.score_anatomy(rec, gt, 0.05, False)
    tight = spec_scoring.score_anatomy(rec, gt, 0.40, False)
    assert tight["head_frac"] < loose["head_frac"]
    assert loose["grain_ratio"] is None
```

- [ ] **Step 2: 运行,确认失败**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_spec_scoring.py -q`
Expected: 收集阶段 `ModuleNotFoundError: No module named 'spec_scoring'`。

- [ ] **Step 3: 实现(先写文件,三个复制函数用 Step 3b 的脚本从 cycle.py 原样插入)**

`arc/spec_scoring.py`。`hfen`、`sharp_ratio`、`grain` 三个函数不手抄:文件里先放一行标记 `# <<CYCLE_HELPERS>>`,Step 3b 的脚本用 `ast` 从 `arc/cycle.py` 取出原文插入,`grain` 只把 `thr = torch.quantile(gg[brain], 0.30)` 改成 `thr = quantile(gg[brain], 0.30)`(行尾注释与空格保持不变)。测试逐字比对源码。

```python
"""Scoring for arc/run_spec.py (spec 2026-09-28 §3.2).

score_channel() is arc/cycle.py's score_contrast() with two constants taken from the spec's `scoring`
block -- the anatomy-mask threshold (0.05 * P99.9 in cycle.py and diag_artefact/headonly.py) and whether
the brain white-matter grain metric N is computed -- and with torch.quantile replaced by numpy above
16M elements. hfen/sharp_ratio/grain are verbatim copies (tests/test_spec_scoring.py compares the source).
With anatomy_rel=0.05 and grain_ratio=True the numbers equal cycle.py's (tests/test_scoring_equivalence.py).
arc/cycle.py and diag_artefact/headonly.py are not modified.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "paired_gs", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from data.transforms import fft3c, ifft3c  # noqa: E402
from metrics import evaluate_reconstruction  # noqa: E402
from diag_artefact.headonly import grain_ratio as _grain_ratio  # noqa: E402
from diag_artefact.headonly import head_hfen, head_psnr, head_ssim  # noqa: E402

QUANTILE_TORCH_MAX = 16_000_000


def quantile(t: torch.Tensor, q: float) -> torch.Tensor:
    """torch.quantile below 16M elements (bit-identical to cycle.py); numpy above (torch caps at 2**24)."""
    flat = t.reshape(-1)
    if flat.numel() <= QUANTILE_TORCH_MAX:
        return torch.quantile(flat, q)
    return torch.tensor(float(np.quantile(flat.detach().cpu().numpy(), q)), dtype=flat.dtype, device=flat.device)


# --- verbatim copies of arc/cycle.py hfen / sharp_ratio / grain (grain: quantile() instead of torch.quantile)
# <<CYCLE_HELPERS>>
# --------------------------------------------------------------------------------------------------------


def score_anatomy(rec: np.ndarray, gt: np.ndarray, anatomy_rel: float, with_grain_ratio: bool) -> dict:
    """diag_artefact.headonly.score_headonly with the mask threshold as a parameter (brain: 0.05)."""
    rec = np.asarray(rec, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    head = gt > anatomy_rel * float(np.quantile(gt, 0.999))
    gmax = float(gt.max())
    return {
        "head_psnr": head_psnr(rec, gt, head, gmax),
        "head_ssim": head_ssim(rec, gt, head, gmax),
        "head_hfen": head_hfen(rec, gt, head, gmax),
        "grain_ratio": _grain_ratio(rec, gt) if with_grain_ratio else None,
        "head_frac": float(head.mean()),
        "gmax": gmax,
    }


def score_channel(tr, which, dev, anatomy_rel, with_grain_ratio):
    """arc/cycle.py score_contrast (lines 212-267) with the anatomy threshold and grain switch as parameters."""
    if which == "plain":
        csm, y, mask, ds = tr.csm_p, tr.kspace_p, tr.mask_p, tr.ds_p
        mask_fit, y_dc, tag = tr.mask_fit_p, tr.kspace_dc_p, "p"
    else:
        csm, y, mask, ds = tr.csm_c, tr.kspace_c, tr.mask_c, tr.ds_c
        mask_fit, y_dc, tag = tr.mask_fit_c, tr.kspace_dc_c, "c"
    gt = ds.get_data()["ground_truth"].to(dev)
    m = (mask if mask.ndim == 3 else mask[0]).to(dev).float()
    mf = (mask_fit if mask_fit.ndim == 3 else mask_fit[0]).to(dev).float()
    with torch.no_grad():
        x = tr._render(which)
        k_model = fft3c(csm * x.unsqueeze(0))
        y_full = ds.kspace_multicoil.to(dev)
        unobserved = (1.0 - m).unsqueeze(0)
        z = (mf - m).clamp_min(0.0).unsqueeze(0)
        unobserved_error = ((k_model - y_full) * unobserved).abs().square().sum()
        unobserved_energy = (y_full * unobserved).abs().square().sum().clamp_min(1e-12)
        nmse_unobs = float(unobserved_error / unobserved_energy)
        e_unobs = float(torch.sqrt(unobserved_error / unobserved_energy))
        unobs_x = unobserved * (1.0 - z)
        unobs_error_xz = ((k_model - y_full) * unobs_x).abs().square().sum()
        unobs_energy_xz = (y_full * unobs_x).abs().square().sum().clamp_min(1e-12)
        e_unobs_excl_z = float(torch.sqrt(unobs_error_xz / unobs_energy_xz))
        k_dc = mf.unsqueeze(0) * y_dc + (1.0 - mf).unsqueeze(0) * k_model
        kz_energy = float((k_dc * z).abs().square().sum())
        x_dc = (ifft3c(k_dc) * csm.conj()).sum(0)
        gm = gt.abs()
        hm = gm > anatomy_rel * quantile(gm.reshape(-1).float(), 0.999)
        dz = (x_dc - gt)[hm]
        head_psnr_cplx = float(20 * torch.log10(gm.max()) - 10 * torch.log10(dz.abs().square().mean()))
        wph = (x_dc.abs() * gm)[hm]
        cos_phase = float((wph * torch.cos(torch.angle(x_dc[hm]) - torch.angle(gt[hm]))).sum()
                          / wph.sum().clamp_min(1e-12))
        met = evaluate_reconstruction(x_dc, gt, compute_extended=False)
        hf = hfen(x_dc.abs(), gt.abs())
        sr = sharp_ratio(x_dc.abs(), gt.abs())
        gr = grain(x_dc.abs(), gt.abs())
        ho = score_anatomy(x_dc.abs().cpu().numpy(), gt.abs().cpu().numpy(), anatomy_rel, with_grain_ratio)
        x_zf = (ifft3c(m.unsqueeze(0) * y) * csm.conj()).sum(0)
        mz = evaluate_reconstruction(x_zf, gt, compute_extended=False)
        zf_hf = hfen(x_zf.abs(), gt.abs())
    psnr, ssim, nmse = float(met["psnr"]), float(met["ssim"]), float(met["nmse"])
    comp = psnr + 25.0 * ssim - 60.0 * nmse - 25.0 * hf - 40.0 * max(0.0, 1.0 - sr) - 200.0 * gr
    beat = (ssim >= float(mz["ssim"])) and (nmse <= float(mz["nmse"])) and (hf <= zf_hf)
    return dict(psnr=psnr, ssim=ssim, nmse=nmse, hfen=hf, sharp=sr, grain=gr,
                head_psnr_cplx=head_psnr_cplx, cos_phase=cos_phase,
                e_unobs=e_unobs, e_unobs_excl_z=e_unobs_excl_z, nmse_unobs=nmse_unobs, comp=comp, beat=beat,
                zf=(float(mz["psnr"]), float(mz["ssim"]), float(mz["nmse"]), zf_hf),
                kz_energy=kz_energy, physics=tr.physics_info.get(tag, {}), **ho)
```

- [ ] **Step 3b: 从 cycle.py 原样插入三个函数**

`<scratchpad>/insert_cycle_helpers.py`:

```python
import ast
from pathlib import Path

root = Path("/data0/congcong/code/GS/mcgs_nb_infra_wt")
src = (root / "arc/cycle.py").read_text()
segs = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef) and n.name in ("hfen", "sharp_ratio", "grain")}
assert set(segs) == {"hfen", "sharp_ratio", "grain"}
old = "thr = torch.quantile(gg[brain], 0.30)"
assert segs["grain"].count(old) == 1
segs["grain"] = segs["grain"].replace(old, "thr = quantile(gg[brain], 0.30)")
block = "\n\n\n".join(segs[k] for k in ("hfen", "sharp_ratio", "grain"))
p = root / "arc/spec_scoring.py"
text = p.read_text()
assert text.count("# <<CYCLE_HELPERS>>") == 1
p.write_text(text.replace("# <<CYCLE_HELPERS>>", block))
print("inserted:", ", ".join(segs))
```

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY <scratchpad>/insert_cycle_helpers.py`
Expected: `inserted: hfen, sharp_ratio, grain`;之后 `grep -c "<<CYCLE_HELPERS>>" arc/spec_scoring.py` 输出 `0`。

- [ ] **Step 4: 运行,确认通过**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_spec_scoring.py -q`
Expected: `4 passed`

- [ ] **Step 5: 提交**

```bash
git add arc/spec_scoring.py tests/test_spec_scoring.py
git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit -m "arc: spec_scoring -- cycle.py's scorer with the anatomy threshold and the grain metric taken from the spec, numpy quantile above 16M elements"
```

---

### Task 5: run_spec.py,以及与 cycle.py 的打分等价(GPU)

**Files:**
- Create: `arc/run_spec.py`、`scripts/make_brain_equiv_spec.py`、`scripts/regress_brain_pin.sh`、`specs/brain_hp_n36_equiv.yaml`(生成)、`tests/test_scoring_equivalence.py`
- 运行产物(不入库,被 `results*/` 忽略):`results_spec/`、`results_arc/pin_hp_n36_*`

**Interfaces:**
- Consumes: Task 2 的 `StrictConfig`;Task 3 的 `load_spec, check_exp, run_dir, trainer_config, provenance`;Task 4 的 `score_channel`;`paired_gs/seed.py` 的 `set_seed(seed, deterministic)`;`JointTrainer(config, device)`、`tr.train()`、`tr.model.load_state_dict`、`tr.model.num_points`。
- Produces:
  - 命令 `python arc/run_spec.py --spec PATH --exp NAME --gpu ID [--audit N]`
  - 正常运行写 `<results_root>/<exp>_score.json`(键:`exp, spec, spec_sha256, git_commit, git_dirty, data_md5, argv, elapsed_s, npts, vram_gb, plain_name, contrast_name, plain, contrast, combined, both_beat, fallbacks, resolved_config`),`save_volumes` 为真时写 `<results_root>/<exp>_volumes.npz`(键 `rec_<name>, gt_<name>, zf_<name>` 及 `_complex` 版本,`<name>` 取 `trainer.data.<ch>.name`)
  - 审计运行写 `<results_root>/<exp>_audit.json`(键:`exp, spec, spec_sha256, iters, fallbacks`,`fallbacks` 为 `[{"key": 点分路径, "default": 值}]`)

- [ ] **Step 1: 写入口**

`arc/run_spec.py`:

```python
#!/usr/bin/env python3
"""arc/run_spec.py -- strict entry for the multi-anatomy studies (spec 2026-09-28 §3.2).

One run = one spec file; the command line carries no tuning parameter:
    python arc/run_spec.py --spec specs/<ds>/<run>.yaml --exp <prefix>_<name> --gpu <id>
    python arc/run_spec.py --spec ... --exp ... --gpu ... --audit N
The trainer sees spec["trainer"] through StrictConfig. A normal run stops at the first key that is not in
the spec. --audit N records every fallback instead, trains N iterations (warm-up shortened to N//4 so both
branches run), writes <exp>_audit.json and scores nothing; merge the result with
`python arc/spec_io.py fill-defaults`. arc/cycle.py is not used and not modified.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        return {"__repr__": repr(v)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="strict spec-driven MC-GS run")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--audit", type=int, default=0, help="record config fallbacks over N iterations; no scoring")
    a = ap.parse_args(argv)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    import numpy as np
    import torch

    here = Path(__file__).resolve().parent
    root = here.parent
    for p in (root / "paired_gs", here):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from seed import set_seed
    from trainers.joint_trainer import JointTrainer
    from data.transforms import fft3c, ifft3c
    import spec_io
    import spec_scoring
    from strict_config import StrictConfig

    log = lambda *m: print(*m, flush=True)  # noqa: E731
    t0 = time.time()
    spec_path = Path(a.spec).resolve()
    spec = spec_io.load_spec(spec_path)
    spec_io.check_exp(a.exp, spec)
    spec_io.run_dir(a.exp, spec)
    res = Path(spec["spec"]["results_root"])
    res.mkdir(parents=True, exist_ok=True)
    prov = spec_io.provenance(spec_path, spec, root)
    plain_cfg = spec_io.trainer_config(spec, a.exp)
    if a.audit:
        plain_cfg["training"]["max_iterations"] = a.audit
        plain_cfg["training"]["shared_warmup_iters"] = max(1, min(int(plain_cfg["training"]["shared_warmup_iters"]),
                                                                  a.audit // 4))
    cfg = StrictConfig(plain_cfg, strict=not a.audit)
    dev = torch.device("cuda", 0)
    set_seed(int(cfg["training"]["seed"]), deterministic=bool(cfg["training"]["deterministic"]))
    log(f"[run_spec {a.exp}] spec={spec_path} sha256={prov['spec_sha256'][:12]} commit={prov['git_commit'][:10]}"
        f"{' DIRTY' if prov['git_dirty'] else ''} gpu={a.gpu} audit={a.audit}")
    tr = JointTrainer(cfg, device=dev)

    if a.audit:
        tr.train()
        seen, entries = set(), []
        for key, default in cfg.fallbacks:
            if key not in seen:
                seen.add(key)
                entries.append({"key": key, "default": _jsonable(default)})
        out = {"exp": a.exp, "spec": str(spec_path), "spec_sha256": prov["spec_sha256"], "iters": a.audit,
               "fallbacks": entries}
        (res / f"{a.exp}_audit.json").write_text(json.dumps(out, indent=2))
        log(f"[run_spec {a.exp}] audit: {len(entries)} fallback(s) -> {res / (a.exp + '_audit.json')}")
        return

    if spec["spec"]["from_ckpt"]:
        ck = torch.load(spec["spec"]["from_ckpt"], map_location="cpu", weights_only=False)
        tr.model.load_state_dict(ck["model_state"])
        log(f"[run_spec {a.exp}] loaded {spec['spec']['from_ckpt']} (iter {ck.get('iter')}); training skipped")
    else:
        tr.train()

    rel, gr = float(spec["scoring"]["anatomy_rel"]), bool(spec["scoring"]["grain_ratio"])
    names = {ch: cfg["data"][ch]["name"] for ch in ("plain", "contrast")}
    scores = {ch: spec_scoring.score_channel(tr, ch, dev, rel, gr) for ch in ("plain", "contrast")}
    combined = 0.5 * (scores["plain"]["comp"] + scores["contrast"]["comp"])
    both_beat = scores["plain"]["beat"] and scores["contrast"]["beat"]
    vram = torch.cuda.max_memory_allocated() / 1e9
    for ch in ("plain", "contrast"):
        s = scores[ch]
        n = "off" if s["grain_ratio"] is None else f"{s['grain_ratio']:.3f}"
        log(f"{names[ch]:5s} psnr={s['psnr']:.3f} ssim={s['ssim']:.4f} nmse={s['nmse']:.4f} hfen={s['hfen']:.4f} "
            f"head_psnr={s['head_psnr']:.3f} head_ssim={s['head_ssim']:.4f} head_hfen={s['head_hfen']:.4f} "
            f"N={n} sharp={s['sharp']:.3f} grain={s['grain']:.4f} head_psnr_cplx={s['head_psnr_cplx']:.3f} "
            f"comp={s['comp']:.3f} beat={s['beat']}")
    row = dict(exp=a.exp, spec=str(spec_path), **prov, argv=sys.argv[1:], elapsed_s=time.time() - t0,
               npts=int(tr.model.num_points), vram_gb=vram, plain_name=names["plain"],
               contrast_name=names["contrast"], plain=scores["plain"], contrast=scores["contrast"],
               combined=combined, both_beat=both_beat, fallbacks=[list(f) for f in cfg.fallbacks],
               resolved_config=cfg.to_plain())
    (res / f"{a.exp}_score.json").write_text(json.dumps(row, indent=2, default=repr))
    log(f"SCORE combined={combined:.3f} both_beat_zf={both_beat} vram_gb={vram:.1f} elapsed_s={time.time() - t0:.0f}")

    if spec["spec"]["save_volumes"]:
        vols = {}
        with torch.no_grad():
            for ch, tag in (("plain", "p"), ("contrast", "c")):
                csm, y = getattr(tr, f"csm_{tag}"), getattr(tr, f"kspace_{tag}")
                mask, ds = getattr(tr, f"mask_{tag}"), getattr(tr, f"ds_{tag}")
                mask_fit, y_dc = getattr(tr, f"mask_fit_{tag}"), getattr(tr, f"kspace_dc_{tag}")
                gt = ds.get_data()["ground_truth"].to(dev)
                m = (mask if mask.ndim == 3 else mask[0]).to(dev).float().unsqueeze(0)
                mf = (mask_fit if mask_fit.ndim == 3 else mask_fit[0]).to(dev).float().unsqueeze(0)
                x = tr._render(ch)
                k_model = fft3c(csm * x.unsqueeze(0))
                x_dc = (ifft3c(mf * y_dc + (1.0 - mf) * k_model) * csm.conj()).sum(0)
                x_zf = (ifft3c(m * y) * csm.conj()).sum(0)
                n = names[ch]
                for key, t in ((f"rec_{n}", x_dc), (f"gt_{n}", gt), (f"zf_{n}", x_zf)):
                    vols[key] = t.abs().cpu().numpy().astype("float32")
                    vols[key + "_complex"] = t.cpu().numpy().astype("complex64")
        np.savez_compressed(res / f"{a.exp}_volumes.npz", **vols)
        log(f"  saved volumes -> {res / (a.exp + '_volumes.npz')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 生成脑等价 spec 的脚本**

`scripts/make_brain_equiv_spec.py`:

```python
"""Write specs/brain_hp_n36_equiv.yaml: the hp_n36 run (jyz AF16 gs_joint_p3 anchor, head PSNR 35.706 / 34.928)
as a run_spec spec that only scores the saved checkpoint. Used to show spec_scoring == cycle.py (plan Task 5).
Source = the resolved config JointTrainer wrote for that run. Output section and seed handling follow cycle.py:
output_dir / exp_name are set by run_spec.py, and cycle.py seeds with set_seed(42, deterministic=False)."""
from pathlib import Path

import yaml

HP = Path("/data0/congcong/code/GS/mcgs_hparam_wt/results_arc/hp_n36/hp_n36")
ROOT = Path(__file__).resolve().parents[1]

trainer = yaml.safe_load((HP / "config.yaml").read_text())
trainer.setdefault("output", {})
trainer["output"].pop("output_dir", None)
trainer["output"].pop("exp_name", None)
trainer["training"].setdefault("seed", 42)
trainer["training"].setdefault("deterministic", False)
spec = {"spec": {"dataset": "brain_jyz_af16_hp_n36", "exp_prefix": "brainpin",
                 "results_root": str(ROOT / "results_spec"), "save_volumes": False,
                 "from_ckpt": str(HP / "checkpoints/latest.pth")},
        "scoring": {"anatomy_rel": 0.05, "grain_ratio": True},
        "trainer": trainer}
out = ROOT / "specs/brain_hp_n36_equiv.yaml"
out.parent.mkdir(exist_ok=True)
out.write_text(yaml.safe_dump(spec, sort_keys=True))
print(f"wrote {out}")
```

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/make_brain_equiv_spec.py`
Expected: `wrote /data0/congcong/code/GS/mcgs_nb_infra_wt/specs/brain_hp_n36_equiv.yaml`

- [ ] **Step 3: 审计并补齐默认值,直到零回退**

先找一张空卡(已用显存 < 1 GB),记作 `G`。写 `/tmp/claude-1002/…/scratchpad/audit_brain.sh`(zsh 不分词,所以用 bash 脚本):

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /data0/congcong/code/GS/mcgs_nb_infra_wt
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
G=$1; ROUND=$2
mkdir -p results_spec
LOG="results_spec/audit${ROUND}.log"
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID "$PY" arc/run_spec.py \
  --spec specs/brain_hp_n36_equiv.yaml --exp "brainpin_audit${ROUND}" --gpu "$G" --audit 40 \
  > "$LOG" 2>&1 || { tail -20 "$LOG"; exit 1; }
tail -1 "$LOG"
env PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" arc/spec_io.py fill-defaults specs/brain_hp_n36_equiv.yaml \
  "results_spec/brainpin_audit${ROUND}_audit.json" --out specs/brain_hp_n36_equiv.yaml
```

Run: `bash <scratchpad>/audit_brain.sh G 1`,然后 `bash <scratchpad>/audit_brain.sh G 2`,依次加一,直到某一轮打印 `audit: 0 fallback(s)`。
Expected: 第 1 轮有几十个回退;通常第 2 或第 3 轮归零。每轮约 3 分钟(载入数据 + 40 步;这个工作树第一次用到体素器时还要先编译 CUDA 扩展,多几分钟)。日志在 `results_spec/`,被 `results*/` 忽略,不入库。

- [ ] **Step 4: 脑重打分脚本(用不变的 cycle.py)**

`scripts/regress_brain_pin.sh`:

```bash
#!/usr/bin/env bash
# Rescore the hp_n36 checkpoint (jyz AF16 gs_joint_p3 anchor) with the unchanged arc/cycle.py and compare
# with the registered head PSNR 35.706153 / 34.928080 (tolerance 1e-4 dB). About 4 minutes on one GPU.
# Usage: scripts/regress_brain_pin.sh GPU
set -euo pipefail
GPU=${1:?usage: scripts/regress_brain_pin.sh GPU}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
CKPT=/data0/congcong/code/GS/mcgs_hparam_wt/results_arc/hp_n36/hp_n36/checkpoints/latest.pth
EXP=pin_hp_n36_$(date +%Y%m%d_%H%M%S)
cd "$ROOT"
mkdir -p results_arc
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID "$PY" arc/cycle.py --exp "$EXP" --gpu "$GPU" \
  --iters 4000 --lambda_geo 0.10 --warmup 500 --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 \
  --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn --set loss.edge_tv_kappa_source=zf \
  --set gaussian.initial_num_points=3600000 --set loss.wav_name=db4 --set loss.wav_levels=1 \
  --set loss.lambda_wav_plain=0.12 --set loss.lambda_wav=0.10 --lambda_edge 0.001 --lambda_ptv 0.03 \
  --base_cfg configs/t1flair_v3_af16_jyz.yaml --from_ckpt "$CKPT" > "results_arc/$EXP.log" 2>&1
env PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" - "$ROOT/results_arc/${EXP}_score.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
ref = {"plain": 35.706153059238126, "contrast": 34.92807957136013}
ok = True
for ch, r in ref.items():
    v = d[ch]["head_psnr"]
    print(f"{ch}: head_psnr {v:.6f} vs {r:.6f} (diff {v - r:+.2e})")
    ok = ok and abs(v - r) <= 1e-4
print("PIN PASS" if ok else "PIN FAIL")
sys.exit(0 if ok else 1)
EOF
echo "score json: $ROOT/results_arc/${EXP}_score.json"
```

Run: `chmod +x scripts/regress_brain_pin.sh && bash scripts/regress_brain_pin.sh G`
Expected: 两行差值都在 1e-4 以内,最后打印 `PIN PASS`。若 `PIN FAIL`:停下,用 superpowers:systematic-debugging 查,不要改容差。

- [ ] **Step 5: 用 run_spec 对同一 checkpoint 打分**

Run(在 bash 脚本里或直接一行,变量都已展开):
`env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID /home/congcongliu/anaconda3/envs/GSTMI/bin/python arc/run_spec.py --spec specs/brain_hp_n36_equiv.yaml --exp brainpin_equiv1 --gpu G`
Expected: 日志中 T1 行 `head_psnr=35.706`、FLAIR 行 `head_psnr=34.928`;`results_spec/brainpin_equiv1_score.json` 中 `fallbacks` 为 `[]`。

- [ ] **Step 6: 等价测试**

`tests/test_scoring_equivalence.py`:

```python
"""spec_scoring (arc/run_spec.py) and arc/cycle.py give the same numbers on the same checkpoint (hp_n36).
Needs the score jsons of plan Task 5 steps 4-5; skipped when they are absent.

Run: env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_scoring_equivalence.py -q
"""
import glob
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["psnr", "ssim", "nmse", "hfen", "sharp", "grain", "head_psnr", "head_ssim", "head_hfen",
        "grain_ratio", "head_psnr_cplx", "cos_phase", "e_unobs", "e_unobs_excl_z", "nmse_unobs", "comp"]


def _latest(pattern):
    files = sorted(glob.glob(str(ROOT / pattern)))
    return json.loads(Path(files[-1]).read_text()) if files else None


CYC = _latest("results_arc/pin_hp_n36_*_score.json")
SPC = _latest("results_spec/brainpin_equiv*_score.json")
pytestmark = pytest.mark.skipif(CYC is None or SPC is None,
                                reason="run scripts/regress_brain_pin.sh and the brainpin_equiv run first")


@pytest.mark.parametrize("channel", ["plain", "contrast"])
def test_spec_scoring_equals_cycle(channel):
    for k in KEYS:
        a, b = CYC[channel][k], SPC[channel][k]
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), f"{channel}.{k}: cycle {a!r} vs run_spec {b!r}"


def test_run_spec_run_had_no_fallbacks():
    assert SPC["fallbacks"] == []
```

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_scoring_equivalence.py -q`
Expected: `3 passed`。不等:停下查原因(先看两边的体积和掩膜是否一致),不放宽容差。

- [ ] **Step 7: 提交**

```bash
git add arc/run_spec.py scripts/make_brain_equiv_spec.py scripts/regress_brain_pin.sh specs/brain_hp_n36_equiv.yaml tests/test_scoring_equivalence.py
git status --short | grep -v '^??' ; git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit -m "arc: run_spec.py, the strict entry; hp_n36 scored through it equals cycle.py on every metric, and cycle.py itself still reproduces 35.706 / 34.928"
```

---

### Task 6: spec_expand(模板按受试者展开)

**Files:**
- Create: `arc/spec_expand.py`
- Test: `tests/test_spec_expand.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `PLACEHOLDER = "{subject}"`
  - `expand(template: dict, subject: str) -> dict`(所有字符串值里的 `{subject}` 替换为受试者号;模板里一处都没有时 `ValueError`)
  - `differs_only_by_subject(template: dict, expanded: dict, subject: str) -> bool`
  - 命令行:`python arc/spec_expand.py TEMPLATE OUT_DIR STEM SUBJECT [SUBJECT ...]`,写 `OUT_DIR/STEM_<SUBJECT>.yaml`

- [ ] **Step 1: 写失败的测试**

`tests/test_spec_expand.py`:

```python
"""spec_expand: one spec per subject from a template; only strings carrying {subject} may differ.

Run: env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_spec_expand.py -q
"""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "arc"))
import spec_expand  # noqa: E402

T = {"spec": {"dataset": "knee", "results_root": "/r"},
     "trainer": {"data": {"plain": {"data_path": "/d/{subject}/ksp_E1.h5", "name": "E1"}},
                 "loss": {"tv_weight": 0.1, "list": [1, "{subject}"]}}}


def test_expand_replaces_every_placeholder_and_nothing_else():
    e = spec_expand.expand(T, "MTR_013")
    assert e["trainer"]["data"]["plain"]["data_path"] == "/d/MTR_013/ksp_E1.h5"
    assert e["trainer"]["loss"] == {"tv_weight": 0.1, "list": [1, "MTR_013"]}
    assert spec_expand.differs_only_by_subject(T, e, "MTR_013")
    assert "{subject}" in T["trainer"]["data"]["plain"]["data_path"]


def test_tampered_expansion_is_detected():
    e = spec_expand.expand(T, "MTR_013")
    e["trainer"]["loss"]["tv_weight"] = 0.2
    assert not spec_expand.differs_only_by_subject(T, e, "MTR_013")


def test_template_without_placeholder_is_refused():
    with pytest.raises(ValueError, match="placeholder"):
        spec_expand.expand({"a": "/no/subject"}, "MTR_013")


def test_cli_writes_one_file_per_subject(tmp_path):
    tpl = tmp_path / "t.yaml"
    tpl.write_text(yaml.safe_dump(T))
    spec_expand.main([str(tpl), str(tmp_path / "out"), "joint", "MTR_013", "MTR_045"])
    for s in ("MTR_013", "MTR_045"):
        e = yaml.safe_load((tmp_path / "out" / f"joint_{s}.yaml").read_text())
        assert spec_expand.differs_only_by_subject(T, e, s)
```

- [ ] **Step 2: 运行,确认失败**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_spec_expand.py -q`
Expected: `ModuleNotFoundError: No module named 'spec_expand'`

- [ ] **Step 3: 实现**

`arc/spec_expand.py`:

```python
"""spec_expand: write one spec per subject from a template (spec 2026-09-28 §2.2).

Every string value of the template that contains "{subject}" gets the subject id; nothing else changes,
so all per-subject specs of one arm carry the same hyper-parameters. differs_only_by_subject() is what the
dataset branches' tests use to prove it.

CLI:  python arc/spec_expand.py TEMPLATE OUT_DIR STEM SUBJECT [SUBJECT ...]   -> OUT_DIR/STEM_<SUBJECT>.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

PLACEHOLDER = "{subject}"


def _count(node) -> int:
    if isinstance(node, dict):
        return sum(_count(v) for v in node.values())
    if isinstance(node, list):
        return sum(_count(v) for v in node)
    return int(isinstance(node, str) and PLACEHOLDER in node)


def _sub(node, subject: str):
    if isinstance(node, dict):
        return {k: _sub(v, subject) for k, v in node.items()}
    if isinstance(node, list):
        return [_sub(v, subject) for v in node]
    if isinstance(node, str):
        return node.replace(PLACEHOLDER, subject)
    return node


def expand(template: dict, subject: str) -> dict:
    if _count(template) == 0:
        raise ValueError(f"template has no {PLACEHOLDER} placeholder")
    return _sub(template, subject)


def differs_only_by_subject(template, expanded, subject: str) -> bool:
    if isinstance(template, dict):
        return (isinstance(expanded, dict) and set(template) == set(expanded)
                and all(differs_only_by_subject(template[k], expanded[k], subject) for k in template))
    if isinstance(template, list):
        return (isinstance(expanded, list) and len(template) == len(expanded)
                and all(differs_only_by_subject(t, e, subject) for t, e in zip(template, expanded)))
    if isinstance(template, str):
        return expanded == template.replace(PLACEHOLDER, subject)
    return type(template) is type(expanded) and template == expanded


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="expand a spec template per subject")
    ap.add_argument("template")
    ap.add_argument("out_dir")
    ap.add_argument("stem")
    ap.add_argument("subjects", nargs="+")
    a = ap.parse_args(argv)
    template = yaml.safe_load(Path(a.template).read_text())
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for s in a.subjects:
        (out / f"{a.stem}_{s}.yaml").write_text(yaml.safe_dump(expand(template, s), sort_keys=True))
        print(f"wrote {out / f'{a.stem}_{s}.yaml'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行,确认通过**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_spec_expand.py -q`
Expected: `4 passed`

- [ ] **Step 5: 提交**

```bash
git add arc/spec_expand.py tests/test_spec_expand.py
git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit -m "arc: spec_expand, one spec per subject from a template, with a check that only {subject} strings differ"
```

---

### Task 7: 脑相关共享文件的 md5 钉

**Files:**
- Create: `tests/data/brain_pinned_md5.txt`(由 2eef592 的内容生成)、`tests/test_brain_files_pinned.py`

**Interfaces:**
- Consumes: 无
- Produces: 清单格式为每行 `<md5>  <repo 相对路径>`;测试逐个比对工作树文件。

- [ ] **Step 1: 生成清单**

`<scratchpad>/make_pin_manifest.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /data0/congcong/code/GS/mcgs_nb_infra_wt
mkdir -p tests/data
{
  git ls-tree -r --name-only 2eef592 -- configs
  git ls-tree --name-only 2eef592 | grep -E '^run_.*_mcgs\.sh$'
  printf '%s\n' arc/cycle.py paired_gs/trainers/joint_trainer.py diag_artefact/headonly.py \
                diag_artefact/volumes.py paired_gs/metrics/metrics.py
} | sort -u | while read -r f; do
  printf '%s  %s\n' "$(git show "2eef592:$f" | md5sum | cut -d' ' -f1)" "$f"
done > tests/data/brain_pinned_md5.txt
wc -l tests/data/brain_pinned_md5.txt
```

Run: `bash <scratchpad>/make_pin_manifest.sh`
Expected: `67 tests/data/brain_pinned_md5.txt`(57 个 configs + 5 个 run 脚本 + 5 个文件)。

- [ ] **Step 2: 写测试**

`tests/test_brain_files_pinned.py`:

```python
"""The brain line's files must stay byte-identical to 2eef592 on every branch cut from nonbrain-infra
(spec 2026-09-28 §2.2): configs/, run_*_mcgs.sh, arc/cycle.py, the joint trainer and the scoring modules.

Run: env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_brain_files_pinned.py -q
"""
import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = [line.split("  ", 1) for line in (ROOT / "tests/data/brain_pinned_md5.txt").read_text().splitlines()
           if line.strip()]


def test_manifest_covers_the_pinned_set():
    names = {p for _, p in ENTRIES}
    for f in ("arc/cycle.py", "paired_gs/trainers/joint_trainer.py", "diag_artefact/headonly.py",
              "diag_artefact/volumes.py", "paired_gs/metrics/metrics.py", "run_wd_mcgs.sh"):
        assert f in names
    assert sum(1 for n in names if n.startswith("configs/")) == 57


@pytest.mark.parametrize("md5,path", ENTRIES, ids=[p for _, p in ENTRIES])
def test_brain_file_unchanged(md5, path):
    assert hashlib.md5((ROOT / path).read_bytes()).hexdigest() == md5, f"{path} differs from 2eef592"
```

- [ ] **Step 3: 运行,确认通过**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_brain_files_pinned.py -q`
Expected: `68 passed`

- [ ] **Step 4: 提交**

```bash
git add tests/data/brain_pinned_md5.txt tests/test_brain_files_pinned.py
git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit -m "tests: pin the brain line's configs, launchers, cycle.py, trainer and scorers to their 2eef592 bytes"
```

---

### Task 8: 全量测试、交接文档、打 tag

**Files:**
- Modify: `CLAUDE.md`(顶部加一段)、`STATUS.md`(顶部加一节)

**Interfaces:**
- Consumes: Task 1–7 的全部产物
- Produces: 交接入口;tag `handoff/<执行日期>-nonbrain-infra`

- [ ] **Step 1: 全量测试**

找空卡 `G`,运行:
`env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES=G /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests -q --ignore=tests/smoke_test.py --ignore=tests/test_score_edge.py -p no:cacheprovider`
Expected: 全部通过,数量 = 223(原有)+ 2 + 6 + 15 + 4 + 3 + 4 + 68 = 325。数量不符时先弄清原因再继续。

- [ ] **Step 2: CLAUDE.md 顶部加入口段**

在 `CLAUDE.md` 第一行标题之后、"Read in this order" 之前插入:

```markdown
## Branch `nonbrain-infra` (worktree `mcgs_nb_infra_wt`) — shared layer for the multi-anatomy AF16 studies
Read `docs/superpowers/specs/2026-09-28-nonbrain-af16-infra-knee-design.md`, then
`docs/superpowers/plans/2026-09-28-nonbrain-infra.md`, then `STATUS.md` (top section). Studies (knee, liver,
breast, thorax) each live on their own branch cut from here; merges go only nonbrain-infra -> dataset branch.
Entry: `arc/run_spec.py --spec SPEC --exp <prefix>_<name> --gpu G [--audit N]` (no tuning flags; a key missing
from the spec stops the run). Brain check after any shared change: `bash scripts/regress_brain_pin.sh G`
(expects 35.706153 / 34.928080) and `pytest tests/test_brain_files_pinned.py`.
```

- [ ] **Step 3: STATUS.md 顶部加一节**

在 `STATUS.md` 最前面插入一节,五段齐全。标题与段落结构如下,方括号处贴本次实际命令的原始输出(不是编的数):

```markdown
# STATUS — nonbrain-infra (worktree `mcgs_nb_infra_wt`) — shared layer done, knee next

## 1. Done and verified
(贴:Step 1 的 pytest 末行;`bash scripts/regress_brain_pin.sh G` 的最后三行;
 `pytest tests/test_scoring_equivalence.py -q` 的末行;`git log --oneline 2eef592..HEAD` 的输出)

## 2. Waiting for the user
- Push `nonbrain-infra` to GitHub? (not pushed)
- Plan 1 (knee) execution go-ahead.

## 3. Next
Write/execute plan 1 (knee, `nb-knee-skmtea`, worktree `mcgs_nb_knee_wt`) per spec §4–§6.

## 4. Pitfalls, and what not to redo
- `run_spec.py --audit N` shortens warm-up to N//4 and ignores `from_ckpt`; audit runs score nothing.
- An audit must reach every periodic event the real run hits (eval_every, save_every, densification windows),
  otherwise a strict run can stop late on a key read only there; audit dataset specs with N >= max(eval_every, save_every).
- A spec may not hold both `loss.tv_weight` and `loss.lambda_geo` (JointTrainer silently prefers lambda_geo).
- Runs are never overwritten: a new run needs a new `--exp`.

## 5. Why
- StrictConfig: every value a study uses is written in its own spec (spec §2.2), so one study cannot move another.
- spec_scoring is a copy, not an edit, of cycle.py's scorer: the brain line's files stay byte-identical (md5 test).
```

- [ ] **Step 4: 提交并打 tag(不推送)**

```bash
git add CLAUDE.md STATUS.md
git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit -m "nonbrain-infra: hand-off entry in CLAUDE.md and STATUS.md"
git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" tag -a "handoff/$(date +%Y-%m-%d)-nonbrain-infra" -m "nonbrain-infra: strict entry and shared layer, brain unchanged"
git log --oneline -3; git tag --points-at HEAD
```

Expected: tag 指向交接提交;`git status --short` 中没有已跟踪文件的改动。推送前问用户。
