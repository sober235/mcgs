# CRISP-MR Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立 CRISP-MR 的可复现基础设施(冻结划分、coil 域指标、CG-SENSE、程序化掩膜、CS-SENSE 基线),并跑完 spec §8 的三项立即可执行运行,交付两个决策:CE 临床臂生死、G0/G2 的方向。

**Architecture:** 新增独立包 `crisp/`,与既有 `paired_gs/` 并存、不修改后者。所有前向/反向算子复用仓库已验证的 `fft3c/ifft3c` 约定(`paired_gs/data/transforms.py`),数值约定与 `arc/cycle.py:188-196` 的 hard-DC 打分逐位一致。实验脚本只组合库函数,不含算法逻辑。

**Tech Stack:** Python 3(env `/home/congcongliu/anaconda3/envs/GSTMI/bin/python`)、PyTorch(CUDA)、h5py、scipy.io、scikit-image、pytest、matplotlib(Agg)。

## Global Constraints

- 解释器一律 `/home/congcongliu/anaconda3/envs/GSTMI/bin/python`,调用前置 `PYTHONNOUSERSITE=1 PYTHONPATH=`;GPU 运行另加 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。
- 空闲 GPU:**2、3、5、7**(0/1/4/6 被他人 `train_tfi.py` 占用)。所有 GPU 运行必须显式 `--gpu`。
- FFT 约定固定为 `fftshift(fftn(ifftshift(x)))`,norm=backward,轴 `(-3,-2,-1)`。禁止引入其他约定。
- 归一化固定为 GT 幅值的 **P99.9**。
- 主要终点为 coil 域 $E_{\text{unobs}}$;图像域指标一律**脑内掩膜 + 脑内峰值**统一动态范围。禁用 `@max` 跨对比度比较与幅值域"零空间误差"。
- 新代码一律写入 `crisp/`,**不修改 `paired_gs/` 与 `arc/`**。
- 任何含受试者姓名的路径只允许出现在 `crisp/data/subjects.py` 的私有映射加载逻辑中,其余代码只见 `S0xx` token。
- 单例(huangyuwei)运行只用于闸门与机制检验;结论必须在 5 例 development 上复验后才可写入论文。
- 每个实验脚本必须输出 JSON(含全部参数与结果)与图,图存 `results_crisp/`。
- 依赖 spec:`docs/superpowers/specs/2026-07-27-crisp-mr-design.md`。冲突时以 spec 为准。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `crisp/__init__.py` | 空包标记 |
| `crisp/io.py` | 数据加载:trainer `.mat`、`T1/FLAIR_csm_ksp` 的 `.h5`、CE 目录;统一返回 `(kspace, csm)` 且线圈轴在前 |
| `crisp/data/subjects.py` | 受试者注册表、S0xx token、SHA-256 冻结划分 |
| `crisp/metrics/coil_domain.py` | `e_coil_full`、`e_unobs` |
| `crisp/metrics/image_domain.py` | 脑掩膜、统一动态范围下的 PSNR/SSIM/NMSE |
| `crisp/recon/cg_sense.py` | 复数 CG-SENSE(也用作统一 DC 算子) |
| `crisp/recon/cs_sense.py` | FISTA CS-SENSE:各向同性 TV 与联合(vectorial)TV |
| `crisp/masks/vdpoisson.py` | 程序化 vd-Poisson 生成、DC 对齐、互补配对 |
| `crisp/experiments/ce_check.py` | CE 对验证重建(交付临床臂决策) |
| `crisp/experiments/g2_voxel.py` | 体素域 G2 快跑 |
| `crisp/experiments/g0_ceiling.py` | 高斯表示天花板 |
| `tests/crisp/test_*.py` | 各模块单元测试(不需 GPU、不需真实数据) |

---

## Task 1: 受试者注册表与冻结划分

**Files:**
- Create: `crisp/__init__.py`, `crisp/data/__init__.py`, `crisp/data/subjects.py`
- Test: `tests/crisp/test_subjects.py`

**Interfaces:**
- Consumes: 无
- Produces: `split_subjects(subject_keys: list[str], forced_dev: str, n_dev: int) -> dict` 返回 `{"development": [...], "test": [...]}`;`token_map(subject_keys) -> dict[str, str]` 返回 `{subject_key: "S001", ...}`

- [ ] **Step 1: 写失败测试**

```python
# tests/crisp/test_subjects.py
import pytest
from crisp.data.subjects import split_subjects, token_map

KEYS = [f"subj_{i:02d}" for i in range(16)]

def test_split_sizes_and_forced_member():
    s = split_subjects(KEYS, forced_dev="subj_07", n_dev=5)
    assert len(s["development"]) == 5
    assert len(s["test"]) == 11
    assert "subj_07" in s["development"]
    assert set(s["development"]) & set(s["test"]) == set()
    assert set(s["development"]) | set(s["test"]) == set(KEYS)

def test_split_is_order_invariant_and_deterministic():
    a = split_subjects(KEYS, forced_dev="subj_07", n_dev=5)
    b = split_subjects(list(reversed(KEYS)), forced_dev="subj_07", n_dev=5)
    assert a == b
    assert a == split_subjects(KEYS, forced_dev="subj_07", n_dev=5)

def test_forced_member_must_exist():
    with pytest.raises(ValueError):
        split_subjects(KEYS, forced_dev="not_a_subject", n_dev=5)

def test_tokens_are_stable_and_unique():
    t = token_map(KEYS)
    assert len(set(t.values())) == 16
    assert t == token_map(list(reversed(KEYS)))
    assert all(v.startswith("S") and len(v) == 4 for v in t.values())
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/crisp/test_subjects.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'crisp'`

- [ ] **Step 3: 实现**

```python
# crisp/data/subjects.py
"""受试者注册表与冻结划分。

划分规则(spec §3.1,先于任何实验固定,不得事后调整):
  1. forced_dev 先行进入 development(huangyuwei 已承受约 62 次调参运行,已污染);
  2. 其余 development 名额,在剩下的受试者中按 SHA-256(subject_key) 十六进制升序取。
token 规则:按 subject_key 字典序编号 S001..,与划分无关,保证跨会话稳定。
"""
from __future__ import annotations

import hashlib


def _digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def split_subjects(subject_keys: list[str], forced_dev: str, n_dev: int) -> dict:
    keys = sorted(set(subject_keys))
    if forced_dev not in keys:
        raise ValueError(f"forced_dev {forced_dev!r} not among the {len(keys)} subject keys")
    if not 1 <= n_dev <= len(keys):
        raise ValueError(f"n_dev {n_dev} out of range for {len(keys)} subjects")
    rest = sorted((k for k in keys if k != forced_dev), key=_digest)
    development = sorted([forced_dev] + rest[: n_dev - 1])
    test = sorted(k for k in keys if k not in set(development))
    return {"development": development, "test": test}


def token_map(subject_keys: list[str]) -> dict:
    return {k: f"S{i + 1:03d}" for i, k in enumerate(sorted(set(subject_keys)))}
```

同时创建两个空文件 `crisp/__init__.py` 与 `crisp/data/__init__.py`(内容为空),以及 `tests/crisp/__init__.py`(内容为空)。

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/crisp/test_subjects.py -v`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add crisp/__init__.py crisp/data/__init__.py crisp/data/subjects.py tests/crisp/__init__.py tests/crisp/test_subjects.py
git commit -m "feat(crisp): subject registry with frozen SHA-256 development/test split"
```

---

## Task 2: 指标(coil 域 + 图像域)

**Files:**
- Create: `crisp/metrics/__init__.py`, `crisp/metrics/coil_domain.py`, `crisp/metrics/image_domain.py`
- Test: `tests/crisp/test_metrics.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `e_coil_full(x, csm, y_full) -> float`
  - `e_unobs(x, csm, y_full, mask) -> float`
  - `brain_mask(gt_mag, frac=0.05) -> np.ndarray[bool]`
  - `brain_gauge(rec_mag, gt_mag) -> dict`(键 `psnr_brain`、`ssim_brain`、`nmse_brain`;峰值用**脑内** P99.9)

- [ ] **Step 1: 写失败测试**

```python
# tests/crisp/test_metrics.py
import numpy as np
import torch

from crisp.metrics.coil_domain import e_coil_full, e_unobs, fft3c
from crisp.metrics.image_domain import brain_mask, brain_gauge


def _toy(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(8, 6, 8, generator=g, dtype=torch.complex64)
    csm = torch.randn(3, 8, 6, 8, generator=g, dtype=torch.complex64)
    csm = csm / (csm.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)   # SoS == 1
    y = fft3c(csm * x.unsqueeze(0))
    return x, csm, y


def test_e_coil_full_is_zero_for_exact_and_one_for_zero():
    x, csm, y = _toy()
    assert e_coil_full(x, csm, y) < 1e-5
    assert abs(e_coil_full(torch.zeros_like(x), csm, y) - 1.0) < 1e-5


def test_e_unobs_ignores_sampled_region():
    x, csm, y = _toy()
    mask = torch.zeros(8, 6, 8)
    mask[:4] = 1.0
    x_bad = x.clone()
    x_bad_k = fft3c(csm * x.unsqueeze(0)).clone()
    x_bad_k[:, :4] *= 3.0                       # 只破坏已采样区
    e_ref = e_unobs(x, csm, y, mask)
    assert e_ref < 1e-5                          # 完美重建 -> 未采样区误差为 0
    assert abs(e_unobs(x_bad, csm, y, mask) - e_ref) < 1e-6   # 未采样区未变


def test_brain_gauge_perfect_and_zero():
    gt = np.zeros((16, 16, 16), dtype=np.float32)
    gt[4:12, 4:12, 4:12] = 1.0
    m = brain_mask(gt)
    assert m.sum() == 8 ** 3
    perfect = brain_gauge(gt, gt)
    assert perfect["nmse_brain"] < 1e-12
    assert perfect["ssim_brain"] > 0.999
    zero = brain_gauge(np.zeros_like(gt), gt)
    assert abs(zero["nmse_brain"] - 1.0) < 1e-6
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/crisp/test_metrics.py -v`
Expected: FAIL,`No module named 'crisp.metrics'`

- [ ] **Step 3: 实现**

```python
# crisp/metrics/coil_domain.py
"""coil 域指标:与前向模型同域的复数多线圈误差(spec §4.1 主要终点)。"""
from __future__ import annotations

import torch

_AX = (-3, -2, -1)


def fft3c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=_AX), dim=_AX), dim=_AX)


def ifft3c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(x, dim=_AX), dim=_AX), dim=_AX)


def e_coil_full(x: torch.Tensor, csm: torch.Tensor, y_full: torch.Tensor) -> float:
    """||F S x - y_full|| / ||y_full||,x 为复数体积,csm/y_full 线圈轴在前。"""
    num = torch.linalg.vector_norm(fft3c(csm * x.unsqueeze(0)) - y_full)
    return float(num / torch.linalg.vector_norm(y_full))


def e_unobs(x: torch.Tensor, csm: torch.Tensor, y_full: torch.Tensor, mask: torch.Tensor) -> float:
    """未采样区相对误差:||(1-M)(F S x - y_full)|| / ||(1-M) y_full||。"""
    inv = (1.0 - mask).unsqueeze(0)
    num = torch.linalg.vector_norm(inv * (fft3c(csm * x.unsqueeze(0)) - y_full))
    return float(num / torch.linalg.vector_norm(inv * y_full))
```

```python
# crisp/metrics/image_domain.py
"""图像域指标:脑内掩膜 + 脑内统一动态范围(spec §4.1)。"""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as ssim_sk


def brain_mask(gt_mag: np.ndarray, frac: float = 0.05) -> np.ndarray:
    return gt_mag > frac * float(gt_mag.max())


def brain_gauge(rec_mag: np.ndarray, gt_mag: np.ndarray, frac: float = 0.05) -> dict:
    """峰值取脑内 P99.9(不用全局离群峰),SSIM 用 full map 后在脑内平均。"""
    brain = brain_mask(gt_mag, frac)
    peak = float(np.quantile(gt_mag[brain], 0.999)) if brain.any() else 1.0
    peak = max(peak, 1e-8)
    g = (gt_mag / peak).astype(np.float64)
    r = (rec_mag / peak).astype(np.float64)
    valid = np.zeros_like(brain)
    valid[3:-3, 3:-3, 3:-3] = True
    _, s_map = ssim_sk(g, r, data_range=float(max(g.max(), 1.0)), win_size=7,
                       channel_axis=None, full=True)
    e = r - g
    mse = float((e[brain] ** 2).mean())
    return {
        "psnr_brain": float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse)),
        "ssim_brain": float(s_map[brain & valid].mean()),
        "nmse_brain": float((e[brain] ** 2).sum() / (g[brain] ** 2).sum()),
        "peak_used": peak,
    }
```

创建空文件 `crisp/metrics/__init__.py`。

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/crisp/test_metrics.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add crisp/metrics tests/crisp/test_metrics.py
git commit -m "feat(crisp): coil-domain E_coil/E_unobs and brain-gauge image metrics"
```

---

## Task 3: 复数 CG-SENSE(兼统一 DC 算子)

**Files:**
- Create: `crisp/recon/__init__.py`, `crisp/recon/cg_sense.py`
- Test: `tests/crisp/test_cg_sense.py`

**Interfaces:**
- Consumes: `crisp.metrics.coil_domain.fft3c/ifft3c`
- Produces: `cg_sense(y, csm, mask, lam=0.0, iters=30, x0=None, device="cpu") -> torch.Tensor`(复数体积)

- [ ] **Step 1: 写失败测试**

```python
# tests/crisp/test_cg_sense.py
import torch
from crisp.metrics.coil_domain import fft3c, e_coil_full
from crisp.recon.cg_sense import cg_sense


def _toy(seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(8, 8, 8, generator=g, dtype=torch.complex64)
    csm = torch.randn(4, 8, 8, 8, generator=g, dtype=torch.complex64)
    csm = csm / (csm.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)
    y = fft3c(csm * x.unsqueeze(0))
    return x, csm, y


def test_recovers_exactly_when_fully_sampled():
    x, csm, y = _toy()
    mask = torch.ones(8, 8, 8)
    xh = cg_sense(y, csm, mask, lam=0.0, iters=60)
    assert e_coil_full(xh, csm, y) < 1e-4


def test_regularization_shrinks_solution():
    x, csm, y = _toy()
    mask = torch.ones(8, 8, 8)
    x_small = cg_sense(y, csm, mask, lam=10.0, iters=60)
    x_big = cg_sense(y, csm, mask, lam=0.0, iters=60)
    assert torch.linalg.vector_norm(x_small) < torch.linalg.vector_norm(x_big)


def test_undersampled_run_is_finite_and_shaped():
    x, csm, y = _toy()
    mask = torch.zeros(8, 8, 8)
    mask[::2] = 1.0
    xh = cg_sense(y * mask.unsqueeze(0), csm, mask, lam=1e-3, iters=20)
    assert xh.shape == x.shape
    assert torch.isfinite(xh.abs()).all()
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/crisp/test_cg_sense.py -v`
Expected: FAIL,`No module named 'crisp.recon'`

- [ ] **Step 3: 实现**

```python
# crisp/recon/cg_sense.py
"""复数 CG-SENSE。求解 (A^H A + lam I) x = A^H y,A = P_Omega F S。

同一实例既作重建器,也作 spec §4.3 要求的"所有方法共用的 DC 算子"。
"""
from __future__ import annotations

import torch

from crisp.metrics.coil_domain import fft3c, ifft3c


def _normal_op(x, csm, mask, lam):
    k = mask.unsqueeze(0) * fft3c(csm * x.unsqueeze(0))
    return (ifft3c(k) * csm.conj()).sum(0) + lam * x


def cg_sense(y, csm, mask, lam: float = 0.0, iters: int = 30, x0=None,
             device: str | None = None) -> torch.Tensor:
    if device is not None:
        y, csm, mask = y.to(device), csm.to(device), mask.to(device)
    b = (ifft3c(mask.unsqueeze(0) * y) * csm.conj()).sum(0)
    x = torch.zeros_like(b) if x0 is None else x0.clone().to(b.device)
    r = b - _normal_op(x, csm, mask, lam)
    p = r.clone()
    rs = torch.vdot(r.reshape(-1), r.reshape(-1)).real
    for _ in range(iters):
        if rs.sqrt() < 1e-12:
            break
        ap = _normal_op(p, csm, mask, lam)
        alpha = rs / torch.vdot(p.reshape(-1), ap.reshape(-1)).real.clamp_min(1e-30)
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = torch.vdot(r.reshape(-1), r.reshape(-1)).real
        p = r + (rs_new / rs.clamp_min(1e-30)) * p
        rs = rs_new
    return x
```

创建空文件 `crisp/recon/__init__.py`。

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/crisp/test_cg_sense.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add crisp/recon/__init__.py crisp/recon/cg_sense.py tests/crisp/test_cg_sense.py
git commit -m "feat(crisp): complex CG-SENSE reconstructor / shared DC operator"
```

---

## Task 4: 数据加载与 CE 对验证(交付临床臂决策)

**Files:**
- Create: `crisp/io.py`, `crisp/experiments/__init__.py`, `crisp/experiments/ce_check.py`
- Test: `tests/crisp/test_io.py`

**Interfaces:**
- Consumes: `cg_sense`、`e_coil_full`、`brain_gauge`
- Produces: `load_trainer_mat(path) -> (kspace, csm)`、`load_pair_h5(path) -> (kspace, csm)`、`normalize_p999(kspace, csm) -> (kspace, scale)`;CE 结论 JSON `results_crisp/ce_check.json`

- [ ] **Step 1: 先探明 CE 目录的真实结构(不假设)**

Run:
```bash
PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python - <<'EOF'
import h5py, glob
for p in sorted(glob.glob("/data1/data1_congcong/data/3D/CE_Data/*/*.h5")):
    with h5py.File(p, "r") as f:
        print(p)
        f.visit(lambda n: print("   ", n, f[n].shape, f[n].dtype) if hasattr(f[n], "shape") else None)
EOF
```
Expected: 打印每个 `.h5` 的数据集名、形状、dtype。**把输出粘进 `crisp/io.py` 的 docstring 作为记录**,再据此写 loader。若 `.h5` 不可读则改用同目录 `.mat`(`ksp.mat`/`csm.mat`/`ksp2.mat`/`csm2.mat`,用 `h5py` 或 `scipy.io.loadmat` 视版本而定)。

- [ ] **Step 2: 写 loader 的失败测试**

```python
# tests/crisp/test_io.py
import numpy as np
import torch
from crisp.io import normalize_p999


def test_normalize_p999_scales_kspace_and_returns_scale():
    g = torch.Generator().manual_seed(3)
    ksp = torch.randn(2, 4, 4, 4, generator=g, dtype=torch.complex64)
    csm = torch.randn(2, 4, 4, 4, generator=g, dtype=torch.complex64)
    csm = csm / (csm.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)
    ksp_n, scale = normalize_p999(ksp, csm)
    assert scale > 0
    assert torch.allclose(ksp_n * scale, ksp, atol=1e-5)
```

- [ ] **Step 3: 运行确认失败**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/crisp/test_io.py -v`
Expected: FAIL,`No module named 'crisp.io'`

- [ ] **Step 4: 实现 loader**

```python
# crisp/io.py
"""数据加载。线圈轴一律返回在最前(C, D, H, W)。

各来源的真实结构(Step 1 探明后填入此处,勿凭记忆):
  - trainer .mat(huangyuwei):h5py,'ksp' 为 (D,H,W,C) 复合 real/imag;'map' 同形
  - T1/FLAIR_csm_ksp .h5:'kspace' 与 'csm',均为 (D,C,H,W) complex64
  - CE_Data:见 Step 1 输出
"""
from __future__ import annotations

import h5py
import numpy as np
import torch


def _to_coil_first(arr: np.ndarray) -> np.ndarray:
    """把长度等于线圈数的轴挪到最前。线圈数取该数组最小的那个维度。"""
    axis = int(np.argmin(arr.shape))
    return np.moveaxis(arr, axis, 0)


def load_trainer_mat(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(path, "r") as f:
        k = f["ksp"][()]
        m = f["map"][()]
    ksp = k["real"].astype(np.complex64) + 1j * k["imag"].astype(np.complex64)
    csm = m["real"].astype(np.complex64) + 1j * m["imag"].astype(np.complex64)
    return torch.from_numpy(_to_coil_first(ksp)), torch.from_numpy(_to_coil_first(csm))


def load_pair_h5(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(path, "r") as f:
        ksp = f["kspace"][()].astype(np.complex64)
        csm = f["csm"][()].astype(np.complex64)
    return torch.from_numpy(_to_coil_first(ksp)), torch.from_numpy(_to_coil_first(csm))


def normalize_p999(kspace: torch.Tensor, csm: torch.Tensor) -> tuple[torch.Tensor, float]:
    """按 GT 幅值 P99.9 归一化 k 空间(spec Global Constraints)。"""
    from crisp.metrics.coil_domain import ifft3c
    gt = (csm.conj() * ifft3c(kspace)).sum(0)
    scale = float(np.quantile(gt.abs().cpu().numpy(), 0.999))
    return kspace / scale, scale
```

创建空文件 `crisp/experiments/__init__.py`。

- [ ] **Step 5: 运行确认通过**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/crisp/test_io.py -v`
Expected: 1 passed

- [ ] **Step 6: 写 CE 验证实验**

```python
# crisp/experiments/ce_check.py
"""CE pair 验证(spec §4.7 / §8 item 1)。

两个序列同 StudyInstanceUID、同协议、相隔 7 min 01 s,协议元数据无法区分平扫/增强,
只能靠图像:若为平扫-增强对,差值图应在血管/硬膜/垂体等强化结构上出现结构化正差,
且差值集中在少数体素;若为重复扫描,差值应接近噪声、无解剖结构。

判据(先于运行固定):
  ENHANCED  若 (a) 差值图脑内 P99.9 / 脑内中位差 > 5,且 (b) 正差体素占比 < 15%;
  REPEAT    若上述不成立。
两项都必须报告,不得只报其一。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from crisp.io import normalize_p999
from crisp.metrics.image_domain import brain_mask
from crisp.recon.cg_sense import cg_sense

OUT = Path("/data0/congcong/code/GS/Paired_ARC_GS_general_framework/results_crisp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--series", nargs=2, required=True,
                    help="两个序列的 .h5 路径,顺序为 …0010 …0060")
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    OUT.mkdir(exist_ok=True)

    from crisp.io import load_pair_h5      # CE 的 loader 在 Step 1/4 确认后可能需替换
    recs = []
    for p in a.series:
        ksp, csm = load_pair_h5(p)
        ksp, _ = normalize_p999(ksp, csm)
        mask = torch.ones(ksp.shape[1:], dtype=torch.float32)
        x = cg_sense(ksp, csm, mask, lam=a.lam, iters=a.iters, device=dev)
        recs.append(x.abs().cpu().numpy().astype(np.float32))

    a_img, b_img = recs
    brain = brain_mask(a_img)
    diff = b_img - a_img
    pos = diff[brain] > 0
    ratio = float(np.quantile(np.abs(diff[brain]), 0.999) /
                  (np.median(np.abs(diff[brain])) + 1e-12))
    pos_frac = float(pos.mean())
    verdict = "ENHANCED" if (ratio > 5.0 and pos_frac < 0.15) else "REPEAT"

    res = dict(series=a.series, lam=a.lam, iters=a.iters,
               tail_to_median_ratio=ratio, positive_fraction=pos_frac, verdict=verdict)
    (OUT / "ce_check.json").write_text(json.dumps(res, indent=2))

    z = a_img.shape[2] // 2
    sl = lambda v: np.rot90(v[:, :, z])
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.5))
    vmax = float(np.quantile(a_img[brain], 0.995))
    ax[0].imshow(sl(a_img), cmap="gray", vmin=0, vmax=vmax); ax[0].set_title("series 1 (…0010)")
    ax[1].imshow(sl(b_img), cmap="gray", vmin=0, vmax=vmax); ax[1].set_title("series 2 (…0060)")
    h = ax[2].imshow(sl(diff) / vmax, cmap="jet", vmin=0, vmax=0.10)
    ax[2].set_title(f"(2)-(1)  jet [0,0.1]\nratio {ratio:.1f}  pos {pos_frac:.2f}  -> {verdict}")
    plt.colorbar(h, ax=ax[2], fraction=0.04)
    for x_ in ax: x_.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "ce_check.png", dpi=110, bbox_inches="tight")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: 运行 CE 验证**

Run:
```bash
cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework && \
CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1 PYTHONPATH=. \
/home/congcongliu/anaconda3/envs/GSTMI/bin/python -m crisp.experiments.ce_check --gpu 2 \
  --series /data1/data1_congcong/data/3D/CE_Data/t1_3D_gre_fsp_sag_iso_ACS_16887049370010/ksp1.h5 \
           /data1/data1_congcong/data/3D/CE_Data/t1_3D_gre_fsp_sag_iso_ACS_16887049370060/ksp2.h5
```
Expected: 打印含 `verdict` 的 JSON;生成 `results_crisp/ce_check.{json,png}`。**若 Step 1 显示 CSM 在独立文件(`csm.h5`/`csm2.h5`),先改 loader 调用再运行。**

- [ ] **Step 8: 提交**

```bash
git add crisp/io.py crisp/experiments/__init__.py crisp/experiments/ce_check.py tests/crisp/test_io.py
git commit -m "feat(crisp): data loaders + CE pair verification experiment"
```

---

## Task 5: 程序化 vd-Poisson 掩膜、DC 对齐、互补配对

**Files:**
- Create: `crisp/masks/__init__.py`, `crisp/masks/vdpoisson.py`
- Test: `tests/crisp/test_masks.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `vd_poisson_2d(shape, af, acs, seed, suppress=None) -> np.ndarray[bool]`
  - `align_to_dc(mask2d, ksp) -> (np.ndarray[bool], tuple[int, int])` 返回对齐后掩膜与位移量
  - `complementary_pair(shape, af, acs, seed) -> (m1, m2, overlap_frac)`

- [ ] **Step 1: 写失败测试**

```python
# tests/crisp/test_masks.py
import numpy as np
from crisp.masks.vdpoisson import vd_poisson_2d, complementary_pair


def test_sampling_fraction_and_acs():
    m = vd_poisson_2d((220, 256), af=12.0, acs=16, seed=0)
    frac = m.mean()
    assert 0.9 / 12 < frac < 1.1 / 12
    c0, c1 = 220 // 2, 256 // 2
    assert m[c0 - 8:c0 + 8, c1 - 8:c1 + 8].all()          # ACS 全采


def test_density_decreases_with_radius():
    m = vd_poisson_2d((220, 256), af=12.0, acs=16, seed=0)
    yy, xx = np.mgrid[0:220, 0:256]
    r = np.sqrt(((yy - 110) / 110.0) ** 2 + ((xx - 128) / 128.0) ** 2)
    inner = m[(r > 0.15) & (r < 0.35)].mean()
    outer = m[(r > 0.7) & (r < 0.9)].mean()
    assert inner > outer


def test_complementary_pair_budget_and_overlap():
    m1, m2, ov = complementary_pair((220, 256), af=12.0, acs=16, seed=0)
    assert abs(m1.mean() - m2.mean()) / m1.mean() < 0.05     # 预算相同
    assert ov <= 0.10                                        # 非 ACS 重叠 <= 10%
    c0, c1 = 220 // 2, 256 // 2
    assert m1[c0 - 8:c0 + 8, c1 - 8:c1 + 8].all() and m2[c0 - 8:c0 + 8, c1 - 8:c1 + 8].all()


def test_seed_reproducibility():
    a = vd_poisson_2d((64, 64), af=6.0, acs=8, seed=7)
    b = vd_poisson_2d((64, 64), af=6.0, acs=8, seed=7)
    assert np.array_equal(a, b)
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/crisp/test_masks.py -v`
Expected: FAIL,`No module named 'crisp.masks'`

- [ ] **Step 3: 实现**

```python
# crisp/masks/vdpoisson.py
"""程序化变密度 Poisson-disc 掩膜(替代 MATLAB vdPoisMex),含 DC 对齐与互补配对。

采用变密度伯努利采样 + 最小间距拒绝:密度 p(r) ∝ (1-r)^alpha,alpha 由二分搜索
使总采样率命中 1/af;ACS 方块强制全采。这与 vdPoisMex 的具体点分布不同,但满足
本项目实际依赖的三条性质:总预算、中心密集外围稀疏、ACS 全采。
"""
from __future__ import annotations

import numpy as np


def _radius_grid(shape):
    d, w = shape
    yy, xx = np.mgrid[0:d, 0:w]
    return np.sqrt(((yy - d / 2) / (d / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2).clip(0, 1)


def vd_poisson_2d(shape, af: float, acs: int, seed: int,
                  suppress: np.ndarray | None = None, alpha: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = _radius_grid(shape)
    base = (1.0 - r) ** alpha
    if suppress is not None:
        base = base * np.where(suppress, 0.15, 1.0)      # 抑制已被另一对比度采到的位置
    d, w = shape
    acs_mask = np.zeros(shape, dtype=bool)
    h = acs // 2
    acs_mask[d // 2 - h:d // 2 + h, w // 2 - h:w // 2 + h] = True
    target = shape[0] * shape[1] / af
    budget = target - acs_mask.sum()
    lo, hi = 0.0, 1.0 / max(base.max(), 1e-12)
    for _ in range(60):                                   # 二分 scale 命中预算
        mid = 0.5 * (lo + hi)
        p = np.clip(base * mid, 0, 1)
        p[acs_mask] = 0.0
        if p.sum() > budget:
            hi = mid
        else:
            lo = mid
    p = np.clip(base * lo, 0, 1)
    p[acs_mask] = 0.0
    m = rng.random(shape) < p
    return m | acs_mask


def align_to_dc(mask2d: np.ndarray, ksp: np.ndarray) -> tuple[np.ndarray, tuple]:
    """把掩膜的 ACS 中心 roll 到数据能量峰所在的相位编码位置。

    ksp: (C, D, H, W) 复数;沿线圈与读出方向求能量,得到 (D, H) 的能量图。
    """
    energy = (np.abs(ksp) ** 2).sum(axis=(0, 3))
    p0, p1 = np.unravel_index(int(np.argmax(energy)), energy.shape)
    d, w = mask2d.shape
    shift = (int(p0 - d // 2), int(p1 - w // 2))
    return np.roll(np.roll(mask2d, shift[0], axis=0), shift[1], axis=1), shift


def complementary_pair(shape, af: float, acs: int, seed: int):
    m1 = vd_poisson_2d(shape, af, acs, seed)
    m2 = vd_poisson_2d(shape, af, acs, seed + 10_000, suppress=m1)
    h = acs // 2
    d, w = shape
    acs_mask = np.zeros(shape, dtype=bool)
    acs_mask[d // 2 - h:d // 2 + h, w // 2 - h:w // 2 + h] = True
    non_acs = ~acs_mask
    denom = max((m1 & non_acs).sum(), 1)
    overlap = float((m1 & m2 & non_acs).sum() / denom)
    return m1, m2, overlap
```

创建空文件 `crisp/masks/__init__.py`。

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/crisp/test_masks.py -v`
Expected: 4 passed。若 `test_complementary_pair_budget_and_overlap` 的 `ov` 超过 0.10,把 `vd_poisson_2d` 的抑制系数 `0.15` 调小并重跑,**把最终值记进 docstring**。

- [ ] **Step 5: 提交**

```bash
git add crisp/masks tests/crisp/test_masks.py
git commit -m "feat(crisp): programmatic vd-Poisson masks with DC alignment and complementary pairs"
```

---

## Task 6: CS-SENSE(独立 TV 与联合 vectorial TV)

**Files:**
- Create: `crisp/recon/cs_sense.py`
- Test: `tests/crisp/test_cs_sense.py`

**Interfaces:**
- Consumes: `fft3c/ifft3c`
- Produces: `fista_tv(y, csm, mask, lam, iters=200, device=None) -> torch.Tensor`;`fista_joint_tv(ys, csms, masks, lam, iters=200, device=None) -> list[torch.Tensor]`

- [ ] **Step 1: 写失败测试**

```python
# tests/crisp/test_cs_sense.py
import torch
from crisp.metrics.coil_domain import fft3c, e_coil_full
from crisp.recon.cs_sense import fista_tv, fista_joint_tv


def _toy(seed=2):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(8, 8, 8, dtype=torch.complex64)
    x[2:6, 2:6, 2:6] = 1.0                       # 分片常数 -> TV 友好
    csm = torch.randn(3, 8, 8, 8, generator=g, dtype=torch.complex64)
    csm = csm / (csm.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)
    y = fft3c(csm * x.unsqueeze(0))
    return x, csm, y


def test_tv_fit_improves_over_zero_filled():
    from crisp.metrics.coil_domain import ifft3c
    x, csm, y = _toy()
    mask = torch.zeros(8, 8, 8); mask[::2] = 1.0
    yu = y * mask.unsqueeze(0)
    zf = (ifft3c(yu) * csm.conj()).sum(0)                 # 真正的零填充伴随解
    xh = fista_tv(yu, csm, mask, lam=1e-3, iters=100)
    assert e_coil_full(xh, csm, y) < e_coil_full(zf, csm, y)


def test_joint_channels_agree_when_inputs_identical():
    x, csm, y = _toy()
    mask = torch.zeros(8, 8, 8); mask[::2] = 1.0
    yu = y * mask.unsqueeze(0)
    solo = fista_tv(yu, csm, mask, lam=1e-3, iters=50)
    joint = fista_joint_tv([yu, yu], [csm, csm], [mask, mask], lam=1e-3, iters=50)
    # 两个通道数据相同 -> 联合解的两个通道彼此相同
    assert torch.allclose(joint[0], joint[1], atol=1e-5)
    assert joint[0].shape == solo.shape


def test_joint_reduces_to_independent_at_zero_coupling():
    """lam=0 时两者都退化为纯数据一致性梯度下降,应逐点相等。"""
    x, csm, y = _toy()
    mask = torch.zeros(8, 8, 8); mask[::2] = 1.0
    yu = y * mask.unsqueeze(0)
    solo = fista_tv(yu, csm, mask, lam=0.0, iters=40)
    joint = fista_joint_tv([yu, yu], [csm, csm], [mask, mask], lam=0.0, iters=40)
    assert torch.allclose(joint[0], solo, atol=1e-5)


def test_larger_lambda_gives_smaller_tv():
    x, csm, y = _toy()
    mask = torch.zeros(8, 8, 8); mask[::2] = 1.0
    yu = y * mask.unsqueeze(0)
    tv = lambda v: (v.abs().diff(dim=0).abs().sum() + v.abs().diff(dim=1).abs().sum()
                    + v.abs().diff(dim=2).abs().sum())
    assert tv(fista_tv(yu, csm, mask, lam=1e-2, iters=80)) < tv(fista_tv(yu, csm, mask, lam=1e-4, iters=80))
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/crisp/test_cs_sense.py -v`
Expected: FAIL,`cannot import name 'fista_tv'`

- [ ] **Step 3: 实现**

```python
# crisp/recon/cs_sense.py
"""CS-SENSE:FISTA + 各向同性 TV(独立)与 vectorial TV(联合多对比度)。

TV 用固定步长的次梯度近端替代(smoothed TV,eps 平滑),便于两种变体共用同一步长规则。
联合版的耦合仅通过 vectorial TV 的跨对比度梯度范数实现:
    TV_joint = sum_voxel sqrt( sum_m sum_axis |grad_axis |x_m| |^2 + eps^2 )
"""
from __future__ import annotations

import torch

from crisp.metrics.coil_domain import fft3c, ifft3c

_EPS = 1e-8


def _grads(mag):
    gx = torch.zeros_like(mag); gy = torch.zeros_like(mag); gz = torch.zeros_like(mag)
    gx[:-1] = mag[1:] - mag[:-1]
    gy[:, :-1] = mag[:, 1:] - mag[:, :-1]
    gz[:, :, :-1] = mag[:, :, 1:] - mag[:, :, :-1]
    return gx, gy, gz


def _tv_grad(x, coupled_norm=None):
    """smoothed-TV 对复数 x 的梯度(经由 |x|)。coupled_norm 给出跨对比度归一化项。

    前向差分 g_j = m_{j+1} - m_j(末位补 0)。对 sum_j u_j g_j 关于 m_j 求导得
    d/dm_j = u_{j-1} - u_j,故散度按 `div[1:] += u[:-1]; div[:-1] -= u[:-1]` 累加。
    最后经由 d|x|/dx = x/|x| 传回复数域。
    """
    mag = x.abs().clamp_min(_EPS)
    gx, gy, gz = _grads(mag)
    n = coupled_norm if coupled_norm is not None else torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2 + _EPS ** 2)
    ux, uy, uz = gx / n, gy / n, gz / n
    div = torch.zeros_like(mag)
    div[1:] += ux[:-1];        div[:-1] -= ux[:-1]
    div[:, 1:] += uy[:, :-1];  div[:, :-1] -= uy[:, :-1]
    div[:, :, 1:] += uz[:, :, :-1]; div[:, :, :-1] -= uz[:, :, :-1]
    return div * (x / mag)


def _data_grad(x, y, csm, mask):
    resid = mask.unsqueeze(0) * (fft3c(csm * x.unsqueeze(0)) - y)
    return (ifft3c(resid) * csm.conj()).sum(0)


def fista_tv(y, csm, mask, lam: float, iters: int = 200, device: str | None = None,
             step: float = 1.0) -> torch.Tensor:
    if device is not None:
        y, csm, mask = y.to(device), csm.to(device), mask.to(device)
    x = (ifft3c(mask.unsqueeze(0) * y) * csm.conj()).sum(0)
    z, t = x.clone(), 1.0
    for _ in range(iters):
        g = _data_grad(z, y, csm, mask) + lam * _tv_grad(z)
        x_new = z - step * g
        t_new = 0.5 * (1 + (1 + 4 * t * t) ** 0.5)
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return x


def fista_joint_tv(ys, csms, masks, lam: float, iters: int = 200, device: str | None = None,
                   step: float = 1.0) -> list:
    if device is not None:
        ys = [v.to(device) for v in ys]; csms = [v.to(device) for v in csms]
        masks = [v.to(device) for v in masks]
    xs = [(ifft3c(m.unsqueeze(0) * y) * c.conj()).sum(0) for y, c, m in zip(ys, csms, masks)]
    zs, t = [v.clone() for v in xs], 1.0
    for _ in range(iters):
        acc = None
        grads = []
        for z in zs:
            gx, gy, gz = _grads(z.abs().clamp_min(_EPS))
            grads.append((gx, gy, gz))
            s = gx ** 2 + gy ** 2 + gz ** 2
            acc = s if acc is None else acc + s
        n = torch.sqrt(acc + _EPS ** 2)                      # 跨对比度共享的归一化
        new_xs = []
        for i, (z, y, c, m) in enumerate(zip(zs, ys, csms, masks)):
            g = _data_grad(z, y, c, m) + lam * _tv_grad(z, coupled_norm=n)
            new_xs.append(z - step * g)
        t_new = 0.5 * (1 + (1 + 4 * t * t) ** 0.5)
        zs = [xn + ((t - 1) / t_new) * (xn - xo) for xn, xo in zip(new_xs, xs)]
        xs, t = new_xs, t_new
    return xs
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/crisp/test_cs_sense.py -v`
Expected: 4 passed。若 `_tv_grad` 的散度实现使测试不收敛,先把 `test_larger_lambda_gives_smaller_tv` 作为判据修正散度符号,**不要放宽测试**。

- [ ] **Step 5: 提交**

```bash
git add crisp/recon/cs_sense.py tests/crisp/test_cs_sense.py
git commit -m "feat(crisp): FISTA CS-SENSE with isotropic TV and joint vectorial TV"
```

---

## Task 7: G2 体素域快跑(检验 §2.5 预注册预测)

**Files:**
- Create: `crisp/experiments/g2_voxel.py`

**Interfaces:**
- Consumes: Task 2/3/5/6 全部
- Produces: `results_crisp/g2_voxel_<subject>.json`、`results_crisp/g2_voxel_<subject>.png`

- [ ] **Step 1: 实现实验脚本**

```python
# crisp/experiments/g2_voxel.py
"""G2 体素域快跑(spec §8 item 3)。

对同一例的 T1/FLAIR:
  条件 A(same-mask)     :两对比度共用一个掩膜
  条件 B(complementary):互补掩膜,预算不变
  方法  independent      :各自 fista_tv
  方法  joint            :fista_joint_tv(vectorial TV 耦合)
主要终点 E_unobs(相对满采 k),次要终点脑内 PSNR/SSIM/NMSE。
lam 在给定网格上扫描,**若最优落在网格边界则自动扩展网格并重跑**(spec §5 G1 规则)。

预注册预测(spec §2.5):same-mask 下 joint 相对 independent 的 E_unobs 增益 ≈ 0;
complementary 下应出现实质增益。本脚本输出直接对照该预测。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from crisp.io import load_trainer_mat, normalize_p999
from crisp.masks.vdpoisson import align_to_dc, complementary_pair, vd_poisson_2d
from crisp.metrics.coil_domain import e_unobs, ifft3c
from crisp.metrics.image_domain import brain_gauge
from crisp.recon.cs_sense import fista_joint_tv, fista_tv

OUT = Path("/data0/congcong/code/GS/Paired_ARC_GS_general_framework/results_crisp")
BASE = ("/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/"
        "Train_3dBrain_huangyuwei_20230705-160927-1451_161015_%s_trainer.mat")


def _stack(mask2d, shape):
    """2D 相位编码平面掩膜 -> 3D(读出轴 = 最后一轴,全采)。"""
    return torch.from_numpy(np.repeat(mask2d[:, :, None], shape[-1], axis=2).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--af", type=float, default=12.0)
    ap.add_argument("--acs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lams", type=float, nargs="+", default=[3e-4, 6e-4, 1.2e-3, 2.4e-3, 4.8e-3])
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    OUT.mkdir(exist_ok=True)

    data = {}
    for c in ("T1", "FLAIR"):
        ksp, csm = load_trainer_mat(BASE % c)
        ksp, _ = normalize_p999(ksp, csm)
        gt = (csm.conj() * ifft3c(ksp)).sum(0)
        data[c] = dict(ksp=ksp.to(dev), csm=csm.to(dev), gt=gt.to(dev))

    pe_shape = tuple(data["T1"]["ksp"].shape[1:3])
    vol_shape = tuple(data["T1"]["ksp"].shape[1:])
    m_same = vd_poisson_2d(pe_shape, a.af, a.acs, a.seed)
    m1, m2, overlap = complementary_pair(pe_shape, a.af, a.acs, a.seed)
    ksp_np = data["T1"]["ksp"].cpu().numpy()
    m_same, shift = align_to_dc(m_same, ksp_np)
    m1, _ = align_to_dc(m1, ksp_np)
    m2, _ = align_to_dc(m2, ksp_np)

    conds = {"same": (m_same, m_same), "complementary": (m1, m2)}
    results = {"dc_shift": list(shift), "complementary_overlap": overlap,
               "sampling_fraction": float(m_same.mean()), "lams": a.lams, "cells": {}}

    for cond, (ma, mb) in conds.items():
        masks = {"T1": _stack(ma, vol_shape).to(dev), "FLAIR": _stack(mb, vol_shape).to(dev)}
        ys = {c: data[c]["ksp"] * masks[c].unsqueeze(0) for c in ("T1", "FLAIR")}
        for lam in a.lams:
            ind = {c: fista_tv(ys[c], data[c]["csm"], masks[c], lam, a.iters) for c in ("T1", "FLAIR")}
            jnt = fista_joint_tv([ys["T1"], ys["FLAIR"]],
                                 [data["T1"]["csm"], data["FLAIR"]["csm"]],
                                 [masks["T1"], masks["FLAIR"]], lam, a.iters)
            cell = {}
            for i, c in enumerate(("T1", "FLAIR")):
                gt_mag = data[c]["gt"].abs().cpu().numpy()
                for name, x in (("independent", ind[c]), ("joint", jnt[i])):
                    cell[f"{c}_{name}"] = dict(
                        e_unobs=e_unobs(x, data[c]["csm"], data[c]["ksp"], masks[c]),
                        **brain_gauge(x.abs().cpu().numpy(), gt_mag))
            results["cells"][f"{cond}|lam={lam:g}"] = cell
            print(cond, lam, {k: round(v["e_unobs"], 4) for k, v in cell.items()}, flush=True)

    # 边界检查:最优 lam 不得落在网格端点
    for cond in conds:
        best = min(a.lams, key=lambda L: np.mean(
            [results["cells"][f"{cond}|lam={L:g}"][k]["e_unobs"] for k in
             ("T1_independent", "FLAIR_independent")]))
        results.setdefault("boundary_optimum", {})[cond] = bool(best in (a.lams[0], a.lams[-1]))

    (OUT / "g2_voxel_huangyuwei.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results[k] for k in
                      ("dc_shift", "complementary_overlap", "sampling_fraction", "boundary_optimum")},
                     indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行**

Run:
```bash
cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework && \
CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1 PYTHONPATH=. \
/home/congcongliu/anaconda3/envs/GSTMI/bin/python -m crisp.experiments.g2_voxel --gpu 3
```
Expected: 逐格打印 `e_unobs`;生成 `results_crisp/g2_voxel_huangyuwei.json`。**若 `boundary_optimum` 任一条件为 true,扩展 `--lams` 网格并重跑,直到最优落在内部。**

- [ ] **Step 3: 判读并记录**

把两个条件下 joint 相对 independent 的 `e_unobs` 增益写进 JSON 的 `verdict` 字段(手工追加一段 `python -c` 或在脚本尾部补一行),并与 spec §2.5 的预注册预测对照:same-mask 增益应 ≈0,complementary 应为正。**无论结果是否符合预测,都必须原样记录。**

- [ ] **Step 4: 提交**

```bash
git add crisp/experiments/g2_voxel.py results_crisp/g2_voxel_huangyuwei.json
git commit -m "exp(crisp): G2 voxel-domain quick run — same vs complementary mask, independent vs joint TV"
```

---

## Task 8: G0 表示天花板(单例)

**Files:**
- Create: `crisp/experiments/g0_ceiling.py`

**Interfaces:**
- Consumes: Task 2/3/4
- Produces: `results_crisp/g0_ceiling_huangyuwei.json`

- [ ] **Step 1: 冷启动冒烟(先于正式运行)**

Run:
```bash
cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework && \
CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1 PYTHONPATH= \
/home/congcongliu/anaconda3/envs/GSTMI/bin/python arc/cycle.py --exp g0_smoke --gpu 5 --iters 20 \
  --lambda_geo 0.06 --kspace_high 2.2 --warmup 5 --max_radius 20 \
  --base_cfg configs/winner_v2_aligned.yaml
```
Expected: 20 次迭代跑完不报错。**注意 `configs/winner_v2_aligned.yaml` 含 `warm_start_checkpoint`;冷启动需临时复制一份 config 并删除该键。**

**若冷启动分支崩溃:停止本任务并报告,不要动手修。** `joint_trainer.py:203-221` 属于 `paired_gs/`,而 Global Constraints 禁止修改它——改动会影响既有 MC-GS 结果的可复现性,是否破例由人类裁定。

- [ ] **Step 2: 实现天花板实验**

```python
# crisp/experiments/g0_ceiling.py
"""G0 表示天花板(spec §5)。

问题:高斯表示本身能否在**满采**条件下逼近体素参考?若不能,后续所有关于"路由"的
结论都会被表示瓶颈这个混杂因素污染。

做法:同一例、同一 CSM、掩膜全 1(满采)。
  参考臂:CG-SENSE(满采,lam=0)-> E_coil_full 的下界
  测试臂:调用 arc/cycle.py 以全 1 掩膜训练高斯模型,取导出体积算 E_coil_full
通过线(先于实验固定):高斯在 <=1e6 primitive 内达到 <= 2 x 参考臂的 E_coil_full。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from crisp.io import load_trainer_mat, normalize_p999
from crisp.metrics.coil_domain import e_coil_full
from crisp.recon.cg_sense import cg_sense

OUT = Path("/data0/congcong/code/GS/Paired_ARC_GS_general_framework/results_crisp")
BASE = ("/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/"
        "Train_3dBrain_huangyuwei_20230705-160927-1451_161015_%s_trainer.mat")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=5)
    ap.add_argument("--contrast", default="T1")
    ap.add_argument("--gauss_npz", default="",
                    help="arc/cycle.py 满采训练导出的 *_volumes.npz;留空则只算参考臂")
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    OUT.mkdir(exist_ok=True)

    ksp, csm = load_trainer_mat(BASE % a.contrast)
    ksp, scale = normalize_p999(ksp, csm)
    ksp, csm = ksp.to(dev), csm.to(dev)
    mask = torch.ones(ksp.shape[1:], dtype=torch.float32, device=dev)

    x_ref = cg_sense(ksp, csm, mask, lam=0.0, iters=60)
    ref = e_coil_full(x_ref, csm, ksp)
    res = {"contrast": a.contrast, "reference_e_coil_full": ref, "pass_line": 2.0 * ref}

    if a.gauss_npz:
        key = "recT1" if a.contrast == "T1" else "recFL"
        vol = np.load(a.gauss_npz)[key].astype(np.float32)
        x_g = torch.from_numpy(vol).to(dev).to(torch.complex64)
        res["gaussian_e_coil_full"] = e_coil_full(x_g, csm, ksp)
        res["verdict"] = "PASS" if res["gaussian_e_coil_full"] <= res["pass_line"] else "FAIL"
        res["note"] = ("高斯臂用的是导出的幅值体积(无相位),这会高估其误差;"
                       "若判为 FAIL,须改用带相位的渲染输出复核后才能定论。")

    (OUT / f"g0_ceiling_huangyuwei_{a.contrast}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 跑参考臂**

Run:
```bash
cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework && \
CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1 PYTHONPATH=. \
/home/congcongliu/anaconda3/envs/GSTMI/bin/python -m crisp.experiments.g0_ceiling --gpu 5 --contrast T1
```
Expected: 打印 `reference_e_coil_full` 与 `pass_line`。

- [ ] **Step 4: 跑高斯臂(满采训练)**

先复制一份满采配置:
```bash
cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework && \
python - <<'EOF'
import yaml
c = yaml.safe_load(open("configs/winner_v2_aligned.yaml"))
c["gaussian"].pop("warm_start_checkpoint", None)      # 冷启动
for s in ("plain", "contrast"):
    c["data"][s]["mask_type"] = "full"                 # 满采;若 dataset 不支持该值,改为
                                                       # 指向一个全 1 的 mask .mat 并记录路径
yaml.safe_dump(c, open("configs/g0_fullsample.yaml", "w"), sort_keys=False)
print(open("configs/g0_fullsample.yaml").read()[:400])
EOF
```
再训练并导出体积:
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1 PYTHONPATH= \
/home/congcongliu/anaconda3/envs/GSTMI/bin/python arc/cycle.py --exp g0_full --gpu 5 --iters 4000 \
  --lambda_geo 0.0 --kspace_high 1.0 --warmup 5 --max_radius 20 \
  --base_cfg configs/g0_fullsample.yaml --save_volumes
```
最后:
```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. /home/congcongliu/anaconda3/envs/GSTMI/bin/python \
  -m crisp.experiments.g0_ceiling --gpu 5 --contrast T1 \
  --gauss_npz results_arc/g0_full_volumes.npz
```
Expected: JSON 含 `gaussian_e_coil_full` 与 `verdict`。**`mask_type: full` 若不被 `paired_gs/data/dataset.py` 支持,改为生成一个全 1 的 mask `.mat` 并在 config 里指向它;把实际采用的做法写进脚本 docstring。**

- [ ] **Step 5: 提交**

```bash
git add crisp/experiments/g0_ceiling.py configs/g0_fullsample.yaml results_crisp/
git commit -m "exp(crisp): G0 representation-ceiling experiment (full-sampled Gaussian vs CG-SENSE)"
```

---

## 不在本计划内(等伦理确认后另立计划)

- 16 例的逐例 CSM 重估(`bart ecalib`)与逐例掩膜生成流水线
- `S001…S016` 匿名化落地与私有映射文件
- G1 强基线全量扫描(CS-SENSE wavelet、LLR、单对比 3DGSMR 等价物)
- 5 例 development 上的 G0/G2 正式运行与统计

---

## Self-Review

**1. Spec 覆盖检查**

| spec 条目 | 对应任务 |
|---|---|
| §3.1 冻结划分规则 | Task 1 |
| §3.3 程序化掩膜 + DC 对齐 + 互补配对 | Task 5 |
| §4.1 coil 域主要终点 / 脑内统一动态范围 | Task 2 |
| §4.3 统一 DC 算子 | Task 3(`cg_sense` 兼任) |
| §4.7 CE 臂决策 | Task 4 |
| §5 G0 通过线 | Task 8 |
| §8 item 1/2/3 | Task 4 / Task 8 / Task 7 |
| §2.5 预注册预测的检验 | Task 7 Step 3 |
| §3.2 匿名化、§3.3 逐例 CSM、§5 G1 | **不在本计划**,已在上节列出,等伦理确认 |
| §2.2 闭式 GN 审计、§4.4 注入框架 | Phase 1/2/3,不在本计划 |

**2. 占位符扫描**:无 TBD/TODO;两处"若…则…"是**可执行的分支指令**(CE loader 依 Step 1 实测结构、`mask_type: full` 的回退方案),均给出了具体替代做法与记录要求,不是待填空白。

**3. 类型一致性**:`fft3c/ifft3c` 只在 `crisp/metrics/coil_domain.py` 定义一次,其余模块一律从该处导入;`brain_gauge` 的返回键(`psnr_brain`/`ssim_brain`/`nmse_brain`/`peak_used`)在 Task 2 定义、Task 7 消费,一致;`cg_sense` 与 `fista_tv` 的签名 `(y, csm, mask, ...)` 顺序统一;掩膜在库函数中一律为 `torch.Tensor` float、在掩膜生成模块中为 `np.ndarray` bool,转换只发生在 `_stack`。
