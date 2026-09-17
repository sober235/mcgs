# CE 回顾性对比实验（CS-SENSE 当 GT，泊松盘 AF16，11 臂）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 UIH 平扫/增强对的 CS-SENSE 重建当真值，用真实 42 线圈 CSM 合成加噪 k 空间，在确定性泊松盘 AF16 掩膜下重跑 11 个臂，并用有真值的 jyz 口径评分、出图、落盘。

**Architecture:** 三块拼起来：合成骨架（新写 `sim_cssense_af16.py` + `make_mask_af16.py`，产出臂脚本已读的 h5/npy 布局）；臂脚本从 jyz AF16 课题搬（它们都经 `t1t2_common.Pair` 拿数据和评分，所以只需重写 `t1t2_common.py` 一处，把 T1/FLAIR 改成 pre/post）；MC-GS 用钉住的训练器快照 e778bf2 跑三期冻结配方，MC-INR 用 mcinr_niessen2025 仓库并补三个参数。调度沿用 CE 课题的 `ce_orchestrate.py` 骨架加一个 `retro16` 计划。

**Tech Stack:** Python 3 / torch 2.7（GSTMI 环境）、torch 1.13 + tinycudann（GR_pt1.10 环境，仅 MC-INR）、BART 1.0 `poisson`、h5py、numpy、scipy、scikit-image、matplotlib、pytest。

**Spec:** `/data0/congcong/code/GS/mcgs_ce_retro_wt/docs/superpowers/specs/2026-09-17-ce-retro-cssense-af16-design.md`

## Global Constraints

- 课题根 `TOPIC=/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16`；git worktree `WT=/data0/congcong/code/GS/mcgs_ce_retro_wt`（分支 `ce-retro-cssense-af16`）；仓库内目录 `WT/ce_retro_cssense_af16/`。`TOPIC/scripts` 和 `TOPIC/configs` 是指向 `WT/ce_retro_cssense_af16/{scripts,configs}` 的符号链接，脚本只有一份。
- 脚本里课题根一律 `R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))`，绝不从 `__file__` 推（`__file__` 解析到 worktree）。
- 解释器：`PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python`（所有臂、评分、图）；`PY_TCNN=/home/congcongliu/anaconda3/envs/GR_pt1.10/bin/python`（只用于 MC-INR）。每条 python 命令前置 `env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID`。
- BART：`/home/congcongliu/src/bart/bart`（v1.0.00）。
- 测试：`cd $WT/ce_retro_cssense_af16 && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest scripts/tests -q`。测试不碰 GPU、不读 5 GB 数据；需要课题布局的测试用 `RETRO_TOPIC` 指到 tmp 目录里造的小数据。
- 输入数据（只读）：GT `/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/tv_sense/cssense_o200_lam0.002_full_{pre,post}_complex.npy`；原采集 `/data0/congcong/code/NV-Generate-CTMR/recon_real/work_ce_t1_1688704937{0010,0060}/{ksp_full.npy,csm.npy,mask2d.npy}`（0010=pre，0060=post）；ROI `/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/lesion_candidates/enhance_rois_cssense.npz`（键 core, roi0..roi5, names；names 顺序 病灶核心/核心外 0-3 mm/正常白质/正常皮层/血管/静脉窦/脉络丛）。
- 网格 (256, 232, 176) = (读出 S-I, A-P, L-R)，线圈 42，掩膜 PE 面 (232, 176)，DC 在 (116, 88)，ACS 24×24，目标 2552 点。
- git：作者 `Congcong Liu <sober235@users.noreply.github.com>`，每次 commit 用 `git -c user.name="Congcong Liu" -c user.email="sober235@users.noreply.github.com" commit`；commit 信息里不出现 AI、Claude、Co-Authored-By、Generated。体数据、h5、npz、trainer/ 目录不进 git。
- 算力规矩：每张卡同时只放一个训练作业；空卡 = 利用率 < 30% 且本用户无进程且显存够；不杀别人的作业；长任务 `setsid nohup … &`，日志进 `TOPIC/logs/`，不放 scratchpad。**掩膜图经用户点头之前不开任何 GPU 全量作业**（冒烟允许）。
- 另一个会话正在 `Results_20260908/CE_retro_gt/` 和 `CE_pre_post_contrast/` 工作：本计划只读那两个目录，不写。

---

## 文件结构

```
WT/ce_retro_cssense_af16/                 （git）
  scripts/
    t1t2_common.py          路径、Pair（数据+评分口径），jyz 同名以便臂脚本原样 import
    diag_artefact/          从 mcgs_af16_wt 逐字拷贝：__init__.py headonly.py volumes.py
    make_mask_af16.py       BART 泊松盘、恰好 2552 点、h5/npy/json/图
    sim_cssense_af16.py     合成 k 空间（σ 估计、加噪、写盘、kres 复现门）
    check_sim.py            GPU 门：伴随自洽、CG-SENSE 天花板、σ 实测
    recon_classical.py      ZF / CG-SENSE / CS-SENSE（围绕 Pair 重写）
    recon_joint_cssense.py  jyz 原样拷贝
    recon_spirit.py         jyz(spirit_wt) 拷贝，改对比度名
    recon_zsssdu.py         jyz 拷贝，改对比度名
    recon_dip_inr.py        jyz 拷贝，改对比度名
    run_mcinr.py            调 mcinr_niessen2025，换算到口径并评分
    run_gs.sh               MC-GS / GS-single 三期配方原文
    extract_gs_volumes.py   npz → recon/ + 单位换算 + 评分
    score_volume.py         任意体经 Pair 重评（口径互验、GS/MC-INR 行）
    arms.py                 每个扫描 11 臂的 (key,label,path)，从 summary.json 取选定 λ
    summarize.py            汇总 logs/*.json → logs/summary.{json,tsv}
    enhancement_fidelity.py 六参考区 增强−平扫 对 GT 的误差 + 颅外背景
    find_best_slice.py best_slice_mcgs.py   jyz 拷贝，改名字
    retro_orchestrate.py    ce_orchestrate.py 骨架 + 计划 retro16
    fig_three_planes_compact.py  jyz 拷贝，改臂表与标签
    fig_enhancement.py      ce_figure_paper.py --diff 的移植
    readme_tables.py        summary.json → README 表
    export_bundle.py        收尾数据包 + MANIFEST + md5
    tests/                  pytest
  configs/
    ce_retro_af16.yaml  ce_retro_af16_single_pre.yaml  ce_retro_af16_single_post.yaml
  README.md  STATUS.md
  logs/*.json  figures/*.png（提交时从 TOPIC 拷入）

TOPIC/                                   （服务器，不进 git）
  scripts -> WT/ce_retro_cssense_af16/scripts     configs -> …/configs
  sim/  work_pre/ work_post/ gs/ mask/ gt.npz meta.json
  trainer/_trainer_e778bf2/   git archive e778bf2 (paired_gs arc configs diag_artefact) + results_arc/
  recon/  logs/  figures/  export/
```

---

### Task 1: 课题骨架与路径约定

**Files:**
- Create: `WT/ce_retro_cssense_af16/scripts/tests/conftest.py`
- Create: `WT/ce_retro_cssense_af16/scripts/tests/test_layout.py`
- Create: `WT/ce_retro_cssense_af16/.gitignore`
- Create: `TOPIC/{sim,recon,logs,figures,export,trainer}`，符号链接 `TOPIC/scripts`、`TOPIC/configs`

**Interfaces:**
- Produces: 目录布局；`conftest.py` 提供 fixture `tiny_topic(tmp_path)` 供后面所有需要课题布局的测试使用（Task 4 起）。

- [ ] **Step 1: 建目录与符号链接**

```bash
WT=/data0/congcong/code/GS/mcgs_ce_retro_wt; TOPIC=/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16
mkdir -p $WT/ce_retro_cssense_af16/scripts/tests $WT/ce_retro_cssense_af16/configs $WT/ce_retro_cssense_af16/logs $WT/ce_retro_cssense_af16/figures
mkdir -p $TOPIC/sim/mask $TOPIC/sim/gs $TOPIC/recon $TOPIC/logs $TOPIC/figures $TOPIC/export $TOPIC/trainer
ln -s $WT/ce_retro_cssense_af16/scripts $TOPIC/scripts
ln -s $WT/ce_retro_cssense_af16/configs $TOPIC/configs
ls -la $TOPIC
```
Expected: `scripts -> /data0/congcong/code/GS/mcgs_ce_retro_wt/ce_retro_cssense_af16/scripts`，`configs` 同理。

- [ ] **Step 2: .gitignore 与 conftest**

`WT/ce_retro_cssense_af16/.gitignore`:
```
__pycache__/
*.pyc
```
（仓库根 .gitignore 已挡 *.npy/*.pth；本目录内不会出现 h5，trainer/ 在 TOPIC 下不在 git。）

`scripts/tests/conftest.py`:
```python
"""Shared fixtures. tiny_topic builds a miniature topic (3 coils, 16x16x16 grid, one Poisson-like mask) under tmp_path
and points RETRO_TOPIC at it, so Pair / scorers / extractors can be tested without the 5 GB data and without a GPU."""
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]          # scripts/: the modules under test live here
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def fft3c(x):
    ax = (-3, -2, -1)
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)


@pytest.fixture
def tiny_topic(tmp_path, monkeypatch):
    G, NC = (16, 16, 16), 3
    rng = np.random.default_rng(0)
    zz, yy, xx = np.meshgrid(*[np.linspace(-1, 1, g) for g in G], indexing="ij")
    head = (zz ** 2 + yy ** 2 + xx ** 2) < 0.8
    S = np.stack([np.exp(-((xx - c) ** 2 + yy ** 2)) * np.exp(1j * 0.3 * c * zz) for c in (-0.5, 0.0, 0.5)]).astype(np.complex64)
    S /= np.sqrt((np.abs(S) ** 2).sum(0, keepdims=True))
    top = tmp_path / "topic"
    for d in ("sim/gs", "sim/mask", "recon", "logs", "figures"):
        (top / d).mkdir(parents=True)
    m2 = rng.random((16, 16)) < 0.35
    m2[6:10, 6:10] = True
    m3 = np.broadcast_to(m2[None], G).astype(np.uint8)
    with h5py.File(top / "sim/mask/mask_af16_3d.h5", "w") as f:
        f.create_dataset("mask", data=m3)
    gts, scales = {}, {}
    for scan, amp in (("pre", 1.0), ("post", 1.3)):
        x = (amp * head * (1.0 + 0.2 * rng.random(G)) * np.exp(1j * 0.1 * yy)).astype(np.complex64)
        y = fft3c(S * x[None]).astype(np.complex64)
        with h5py.File(top / f"sim/gs/{scan}_kspace.h5", "w") as f:
            f.create_dataset("kspace", data=y)
        with h5py.File(top / f"sim/gs/{scan}_csm.h5", "w") as f:
            f.create_dataset("map", data=S)
        gts[scan] = x
        scales[f"scale_{scan}"] = float(np.quantile(np.abs(x), 0.999))
    np.savez(top / "sim/gt.npz", mask2d=m2, **gts, **scales)
    monkeypatch.setenv("RETRO_TOPIC", str(top))
    return dict(root=top, G=G, NC=NC, S=S, m2=m2, gts=gts, scales=scales)
```

`scripts/tests/test_layout.py`:
```python
from pathlib import Path


def test_topic_symlinks_point_into_worktree():
    topic = Path("/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16")
    wt = Path("/data0/congcong/code/GS/mcgs_ce_retro_wt/ce_retro_cssense_af16")
    for name in ("scripts", "configs"):
        assert (topic / name).is_symlink()
        assert (topic / name).resolve() == (wt / name).resolve()
```

- [ ] **Step 3: 跑测试**

Run: `cd $WT/ce_retro_cssense_af16 && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest scripts/tests -q`
Expected: `1 passed`

- [ ] **Step 4: Commit**

```bash
cd $WT && git add ce_retro_cssense_af16 && git -c user.name="Congcong Liu" -c user.email="sober235@users.noreply.github.com" commit -m "ce retro af16: topic skeleton, symlinked scripts, test fixtures"
```

---

### Task 2: 掩膜 `make_mask_af16.py`

**Files:**
- Create: `scripts/make_mask_af16.py`
- Test: `scripts/tests/test_mask.py`

**Interfaces:**
- Produces: `sim/mask/mask_af16_pe.npy`（(232,176) uint8）、`sim/mask/mask_af16_3d.h5`（键 `mask`，(256,232,176) uint8）、`sim/mask/mask_af16.json`、`figures/mask_af16.png`。函数 `poisson(acc, seed, workdir) -> np.ndarray[bool] (232,176)`、`trim_to_target(m, rn, target) -> np.ndarray[bool]`、`find_mask(seeds=(0,1,2)) -> (mask, info)`。

- [ ] **Step 1: 写失败的测试**

`scripts/tests/test_mask.py`:
```python
import numpy as np
import pytest

import make_mask_af16 as M


def test_poisson_is_deterministic_elliptical_with_full_acs(tmp_path):
    a = M.poisson(4.0, 1, tmp_path)
    b = M.poisson(4.0, 1, tmp_path / "again")
    assert a.shape == (M.NY, M.NZ) and a.dtype == bool
    assert np.array_equal(a, b)
    assert M.acs_block(a).all()
    assert a[M.radius() > 1.0].sum() == 0


def test_trim_keeps_acs_and_hits_target():
    rng = np.random.default_rng(0)
    m = rng.random((M.NY, M.NZ)) < 0.08
    m[M.radius() > 1.0] = False
    y0, z0 = M.NY // 2 - M.ACS // 2, M.NZ // 2 - M.ACS // 2
    m[y0:y0 + M.ACS, z0:z0 + M.ACS] = True
    t = M.trim_to_target(m, M.radius(), M.TARGET)
    assert t.sum() == M.TARGET
    assert M.acs_block(t).all()
    assert not (t & ~m).any()                      # trimming only removes points


def test_find_mask_exact_af16(tmp_path):
    m, info = M.find_mask(workdir=tmp_path)
    assert m.sum() == M.TARGET == 2552
    assert M.acs_block(m).all()
    assert info["af"] == pytest.approx(16.0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd $WT/ce_retro_cssense_af16/scripts && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_mask.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'make_mask_af16'`

- [ ] **Step 3: 实现**

`scripts/make_mask_af16.py`:
```python
#!/usr/bin/env python3
"""make_mask_af16.py -- one variable-density Poisson-disc mask on the (232, 176) phase-encode plane with exactly
2552 = 40832 / 16 sampled points (AF 16.00 under the 2026-09-14 definition: plane points / sampled points), a fully
sampled 24 x 24 ACS, elliptical scanning, generated by BART `poisson` with a fixed seed so it can be regenerated
bit for bit (vdPoisMex seeds from the wall clock and cannot).

Search: seed 0, acceleration a = -y = -z swept on a 0.01 grid from 2.5 to 8.0; the first a whose count is exactly
2552 wins. If no a hits, the smallest count above 2552 is trimmed from the outside in (largest normalised radius
first, ACS never touched); if seed 0 never reaches 2552 the same is tried with seeds 1 and 2.

  python scripts/make_mask_af16.py            # writes sim/mask/* and figures/mask_af16.png
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
BART = os.environ.get("BART", "/home/congcongliu/src/bart/bart")
NX, NY, NZ, ACS = 256, 232, 176, 24
TARGET = NY * NZ // 16                                 # 2552


def read_cfl(base):
    with open(str(base) + ".hdr") as f:
        dims = [int(v) for v in f.read().splitlines()[1].split()]
    data = np.fromfile(str(base) + ".cfl", dtype=np.complex64)
    return data.reshape(dims, order="F").squeeze()


def poisson(acc, seed, workdir):
    """BART poisson -Y 232 -Z 176 -y a -z a -C 24 -v -e -s seed -> boolean (232, 176)"""
    Path(workdir).mkdir(parents=True, exist_ok=True)
    base = Path(workdir) / f"p_{acc:.3f}_{seed}"
    subprocess.run([BART, "poisson", "-Y", str(NY), "-Z", str(NZ), "-y", f"{acc:.3f}", "-z", f"{acc:.3f}",
                    "-C", str(ACS), "-v", "-e", "-s", str(seed), str(base)], check=True, capture_output=True)
    m = np.abs(read_cfl(base)) > 0.5
    assert m.shape == (NY, NZ), m.shape
    return m


def radius():
    u = (np.arange(NY) - NY // 2) / (NY / 2)
    v = (np.arange(NZ) - NZ // 2) / (NZ / 2)
    return np.sqrt(u[:, None] ** 2 + v[None, :] ** 2)


def acs_slice():
    return slice(NY // 2 - ACS // 2, NY // 2 + ACS // 2), slice(NZ // 2 - ACS // 2, NZ // 2 + ACS // 2)


def acs_block(m):
    sy, sz = acs_slice()
    return m[sy, sz]


def trim_to_target(m, rn, target):
    m = m.copy()
    acs = np.zeros_like(m); sy, sz = acs_slice(); acs[sy, sz] = True
    extra = int(m.sum()) - target
    if extra <= 0:
        return m
    cand = np.argwhere(m & ~acs)
    order = np.argsort(-rn[cand[:, 0], cand[:, 1]], kind="stable")    # farthest from DC first, ties by index
    for i in order[:extra]:
        m[cand[i, 0], cand[i, 1]] = False
    return m


def find_mask(seeds=(0, 1, 2), workdir=None):
    workdir = Path(workdir or tempfile.mkdtemp(prefix="poisson_"))
    rn = radius()
    for seed in seeds:
        best = None
        for acc in np.round(np.arange(2.5, 8.0001, 0.01), 2):
            m = poisson(float(acc), seed, workdir)
            n = int(m.sum())
            if n == TARGET:
                return m, dict(seed=seed, acc=float(acc), count_raw=n, trimmed=0, af=NY * NZ / n)
            if n > TARGET and (best is None or n < best[1]):
                best = (float(acc), n, m)
        if best is not None:
            acc, n, m = best
            t = trim_to_target(m, rn, TARGET)
            return t, dict(seed=seed, acc=acc, count_raw=n, trimmed=n - TARGET, af=NY * NZ / int(t.sum()))
    raise SystemExit("no Poisson-disc draw reached 2552 points with seeds " + str(seeds))


def density_profile(m, rn, nb=20):
    edges = np.linspace(0, 1, nb + 1)
    prof = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (rn >= lo) & (rn < hi)
        prof.append(float(m[sel].mean()) if sel.any() else 0.0)
    return edges, prof


def main():
    m, info = find_mask()
    assert int(m.sum()) == TARGET and acs_block(m).all() and m[radius() > 1.0].sum() == 0
    out = R / "sim/mask"; out.mkdir(parents=True, exist_ok=True)
    np.save(out / "mask_af16_pe.npy", m.astype(np.uint8))
    with h5py.File(out / "mask_af16_3d.h5", "w") as f:
        f.create_dataset("mask", data=np.broadcast_to(m[None], (NX, NY, NZ)).astype(np.uint8))
    ver = subprocess.run([BART, "version"], capture_output=True, text=True).stdout.strip()
    info.update(plane=[NY, NZ], target=TARGET, acs=ACS, dc=[NY // 2, NZ // 2], bart=ver, bart_path=BART,
                cmd=f"{BART} poisson -Y {NY} -Z {NZ} -y {info['acc']:.3f} -z {info['acc']:.3f} -C {ACS} -v -e -s {info['seed']}")
    json.dump(info, open(out / "mask_af16.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rn = radius(); edges, prof = density_profile(m, rn)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), layout="compressed")
    ax[0].imshow(m.T, cmap="gray", origin="lower", interpolation="nearest"); ax[0].set_title(f"PE plane {NY}x{NZ}: {TARGET} points, AF {info['af']:.2f}, ACS {ACS}x{ACS}")
    ax[0].set_xlabel("ky (A-P)"); ax[0].set_ylabel("kz (L-R)")
    ax[1].step(edges[:-1], prof, where="post"); ax[1].set_xlabel("normalised radius"); ax[1].set_ylabel("sampled fraction"); ax[1].set_title("variable density profile")
    (R / "figures").mkdir(exist_ok=True)
    fig.savefig(R / "figures/mask_af16.png", dpi=150)
    print(json.dumps(info, indent=1)); print(f"wrote {out} and figures/mask_af16.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd $WT/ce_retro_cssense_af16/scripts && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_mask.py -q`
Expected: `3 passed`（约 1 到 3 分钟，因为 find_mask 扫 550 次 BART 调用）。若 `test_poisson_is_deterministic_elliptical_with_full_acs` 在 `acs_block(a).all()` 失败，说明 BART 的 -C 在短轴上不是 24 行：把 `poisson()` 改为生成后强制 `m[acs_slice()] = True`，并在 docstring 记一句，然后重跑。

- [ ] **Step 5: 生成正式掩膜并出图**

Run: `cd $TOPIC && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/make_mask_af16.py`
Expected: 打印 json 含 `"af": 16.0`、`"count_raw"`、`"trimmed"`；`figures/mask_af16.png` 生成。把图拷到 `/home/congcongliu/figs/ce_retro_af16/mask_af16.png`，用 `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8765/ce_retro_af16/mask_af16.png` 探活（200），在汇报里给 URL 和路径两行。**这是审批门：用户点头前不开全量 GPU。**

- [ ] **Step 6: Commit**

```bash
cd $WT && cp $TOPIC/figures/mask_af16.png ce_retro_cssense_af16/figures/ && cp $TOPIC/sim/mask/mask_af16.json ce_retro_cssense_af16/logs/ && git add ce_retro_cssense_af16 && git -c user.name="Congcong Liu" -c user.email="sober235@users.noreply.github.com" commit -m "ce retro af16: deterministic Poisson-disc mask, exactly 2552 points, 24x24 ACS"
```

---

### Task 3: 训练器快照与评分依赖

**Files:**
- Create: `TOPIC/trainer/_trainer_e778bf2/`（git archive，不进 git）+ `SOURCE.txt`
- Create: `scripts/diag_artefact/{__init__.py,headonly.py,volumes.py}`（逐字拷贝）
- Test: `scripts/tests/test_deps.py`

**Interfaces:**
- Produces: `from metrics import evaluate_reconstruction` 可用（sys.path 加 `TRAINER/paired_gs`）；`from diag_artefact.headonly import score_headonly, head_mask` 可用（sys.path 加 scripts）。环境变量 `RETRO_TRAINER` 可覆盖快照位置（默认 `TOPIC/trainer/_trainer_e778bf2`）。

- [ ] **Step 1: 抽快照并复用 CUDA 扩展的编译产物**

```bash
G=/data0/congcong/code/GS/Paired_ARC_GS_general_framework; TR=$TOPIC/trainer/_trainer_e778bf2
mkdir -p $TR && git -C $G archive e778bf2 paired_gs arc configs diag_artefact | tar -x -C $TR
echo "git archive e778bf2 (sober235/mcgs framework lineage): paired_gs arc configs diag_artefact; the trainer the jyz AF16 phase-3 runs used" > $TR/SOURCE.txt
mkdir -p $TR/paired_gs/gaussian/cuda_ext/build && cp -a /data0/congcong/code/GS/mcgs_af16_p3_wt/paired_gs/gaussian/cuda_ext/build/. $TR/paired_gs/gaussian/cuda_ext/build/
md5sum $TR/paired_gs/gaussian/cuda_ext/tile_voxelize_cuda.cpp /data0/congcong/code/GS/mcgs_af16_p3_wt/paired_gs/gaussian/cuda_ext/tile_voxelize_cuda.cpp
ls $TR/arc/cycle.py $TR/paired_gs/metrics/metrics.py $TR/diag_artefact/headonly.py
```
Expected: 两个 md5 相同（编译产物的名字含源码 md5，相同才会被复用；不同也无妨，首次运行会自动重编译）；三个文件都在。

- [ ] **Step 2: 拷贝 diag_artefact 三个文件**

```bash
mkdir -p $WT/ce_retro_cssense_af16/scripts/diag_artefact
for f in __init__.py headonly.py volumes.py; do cp /data0/congcong/code/GS/mcgs_af16_wt/diag_artefact/$f $WT/ce_retro_cssense_af16/scripts/diag_artefact/$f; done
echo "verbatim copies of mcgs_af16_wt/diag_artefact/{__init__,headonly,volumes}.py (the frozen head-only gauge); do not edit" > $WT/ce_retro_cssense_af16/scripts/diag_artefact/SOURCE.txt
grep -n "^from\|^import" $WT/ce_retro_cssense_af16/scripts/diag_artefact/__init__.py
```
Expected: `__init__.py` 不 import 本包外的模块（若它 import 了 `.metrics`/`.report` 等未拷贝的模块，就把 `__init__.py` 换成空文件，并在 SOURCE.txt 记一句）。

- [ ] **Step 3: 写测试**

`scripts/tests/test_deps.py`:
```python
import os
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
TRAINER = Path(os.environ.get("RETRO_TRAINER", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16/trainer/_trainer_e778bf2"))


def test_trainer_snapshot_and_gauge_imports():
    assert (TRAINER / "arc/cycle.py").exists() and (TRAINER / "SOURCE.txt").exists()
    for p in (str(TRAINER / "paired_gs"), str(SCRIPTS)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from metrics import evaluate_reconstruction          # noqa: F401
    from diag_artefact.headonly import score_headonly, head_mask
    gt = np.zeros((16, 16, 16), np.float32); gt[4:12, 4:12, 4:12] = 1.0
    s = score_headonly(gt, gt)
    assert s["head_psnr"] == float("inf") and abs(s["head_ssim"] - 1.0) < 1e-6
    assert head_mask(gt).sum() == 8 ** 3
```

- [ ] **Step 4: 跑测试**

Run: `cd $WT/ce_retro_cssense_af16/scripts && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_deps.py -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
cd $WT && git add ce_retro_cssense_af16/scripts/diag_artefact ce_retro_cssense_af16/scripts/tests/test_deps.py && git -c user.name="Congcong Liu" -c user.email="sober235@users.noreply.github.com" commit -m "ce retro af16: frozen head-only gauge copied in, trainer snapshot e778bf2 documented"
```

---

### Task 4: `t1t2_common.py`：数据与评分口径

**Files:**
- Create: `scripts/t1t2_common.py`
- Test: `scripts/tests/test_gauge.py`

**Interfaces:**
- Produces（后面所有臂和评分脚本依赖，名字必须一字不差）：`R`（Path）、`TRAINER`、`SCANS = ("pre","post")`、`fft3c/ifft3c/fft2c/ifft2c/ifft1c(x, d)`、`hfen/sharp_ratio/grain`、`class Pair(contrast, dev, chunk=4)` 属性 `contrast dev chunk y_full S m3 m2 NC G scale gt_c gt y`，方法 `sense_adj(k)`、`AHA(x, mask=None)`、`score(x_complex, method, seconds=0.0, **extra) -> dict`（键 method contrast psnr ssim nmse hfen sharp grain seconds head_psnr head_ssim head_hfen grain_ratio head_frac gmax head_nmse psnr_cplx + extra）。

- [ ] **Step 1: 写失败的测试**

`scripts/tests/test_gauge.py`:
```python
import math

import numpy as np
import torch


def test_pair_loads_tiny_topic_and_scores_truth_perfectly(tiny_topic):
    import t1t2_common as C
    P = C.Pair("pre", torch.device("cpu"))
    assert P.NC == 3 and P.G == (16, 16, 16) and P.contrast == "pre"
    assert abs(P.scale - tiny_topic["scales"]["scale_pre"]) < 1e-6
    assert bool((P.y[:, :, ~torch.from_numpy(tiny_topic["m2"])] == 0).all())
    row = P.score(P.gt_c, "truth")
    assert row["head_psnr"] == math.inf and row["psnr_cplx"] == math.inf
    assert row["head_nmse"] == 0.0 and row["nmse"] < 1e-10 and abs(row["ssim"] - 1) < 1e-6


def test_forward_and_adjoint_agree(tiny_topic):
    import t1t2_common as C
    P = C.Pair("post", torch.device("cpu"))
    g = torch.Generator().manual_seed(1)
    x = torch.randn(P.G, generator=g, dtype=torch.complex64)
    k = torch.randn((P.NC,) + P.G, generator=g, dtype=torch.complex64)
    Ax = C.fft3c(P.S * x) * P.m3
    AHk = P.sense_adj(k * P.m3)
    lhs = torch.vdot(Ax.flatten(), k.flatten()); rhs = torch.vdot(x.flatten(), AHk.flatten())
    assert abs(lhs - rhs) < 1e-3 * abs(lhs)
    assert torch.allclose(P.AHA(x), P.sense_adj(Ax), atol=1e-5)


def test_head_nmse_matches_head_psnr(tiny_topic):
    import t1t2_common as C
    P = C.Pair("pre", torch.device("cpu"))
    x = P.gt_c * 0.9
    row = P.score(x, "scaled")
    head = torch.from_numpy(C.head_mask(P.gt.numpy()))
    mse = float(((x.abs() - P.gt)[head] ** 2).mean())
    assert abs(row["head_psnr"] - (20 * np.log10(row["gmax"]) - 10 * np.log10(mse))) < 1e-4
    assert abs(row["head_nmse"] - mse * head.sum().item() / float((P.gt[head] ** 2).sum())) < 1e-7
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd $WT/ce_retro_cssense_af16/scripts && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_gauge.py -q`
Expected: FAIL，`No module named 't1t2_common'`

- [ ] **Step 3: 实现**

`scripts/t1t2_common.py`:
```python
"""t1t2_common.py -- paths, data loading and the scoring gauge shared by every arm of this topic.

Kept under the jyz file name so recon_joint_cssense.py / recon_spirit.py / recon_zsssdu.py / recon_dip_inr.py import
it unchanged. Differences from the jyz AF16 module (Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/scripts):
  * the ground truth is the CS-SENSE volume of the prospective pair (sim/gt.npz), not a SENSE adjoint of full k-space;
  * the two scans are called pre / post; the noisy synthetic k-space is sim/gs/<scan>_kspace.h5 (unmasked, the mask
    is applied here), maps sim/gs/<scan>_csm.h5 (key map), mask sim/mask/mask_af16_3d.h5 (key mask);
  * two extra scores: head_nmse (head-masked NMSE on magnitudes) and psnr_cplx (head-masked complex PSNR);
  * RETRO_TOPIC / RETRO_TRAINER override the topic root and the trainer snapshot (tests).
Gauge: scale = P99.9 of |GT| per scan (stored in gt.npz), applied to GT and to y; metrics =
paired_gs.metrics.evaluate_reconstruction (whole@max) + diag_artefact.headonly.score_headonly; hfen / sharp_ratio /
grain transcribed verbatim from arc/cycle.py (do not "improve").
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.fft as tfft
import torch.nn.functional as F

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
TRAINER = Path(os.environ.get("RETRO_TRAINER", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16/trainer/_trainer_e778bf2"))
HERE = Path(__file__).resolve().parent
for _p in (str(TRAINER / "paired_gs"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from metrics import evaluate_reconstruction                      # noqa: E402
from diag_artefact.headonly import score_headonly, head_mask      # noqa: E402

SCANS = ("pre", "post")


# ---- centred orthonormal FFTs ---------------------------------------------------------------
def _c(fn, x, dims):
    return tfft.fftshift(fn(tfft.ifftshift(x, dim=dims), dim=dims, norm="ortho"), dim=dims)


def fft3c(x):  return _c(tfft.fftn,  x, (-3, -2, -1))
def ifft3c(x): return _c(tfft.ifftn, x, (-3, -2, -1))
def fft2c(x):  return _c(tfft.fftn,  x, (-2, -1))
def ifft2c(x): return _c(tfft.ifftn, x, (-2, -1))
def ifft1c(x, d): return _c(tfft.ifftn, x, (d,))


# ---- metrics transcribed verbatim from arc/cycle.py (do not "improve") ----------------------
def hfen(rec, gt):
    def lap(v):
        return (6.0 * v - torch.roll(v, 1, -3) - torch.roll(v, -1, -3)
                - torch.roll(v, 1, -2) - torch.roll(v, -1, -2)
                - torch.roll(v, 1, -1) - torch.roll(v, -1, -1))
    g = gt / (gt.max() + 1e-9); r = rec / (gt.max() + 1e-9); lg = lap(g)
    return float(torch.linalg.norm(lap(r) - lg) / (torch.linalg.norm(lg) + 1e-9))


def _gmag(v):
    gx = torch.zeros_like(v); gy = torch.zeros_like(v); gz = torch.zeros_like(v)
    gx[:-1] = v[1:] - v[:-1]; gy[:, :-1] = v[:, 1:] - v[:, :-1]; gz[:, :, :-1] = v[:, :, 1:] - v[:, :, :-1]
    return torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)


def sharp_ratio(rec, gt):
    brain = gt > 0.05 * gt.max()
    return float(_gmag(rec)[brain].mean() / (_gmag(gt)[brain].mean() + 1e-9))


def _quantile(t, q):
    if t.numel() <= 16_000_000:
        return torch.quantile(t, q)
    return torch.tensor(float(np.quantile(t.detach().cpu().numpy(), q)), device=t.device, dtype=t.dtype)


def grain(rec, gt):
    g = gt / (gt.max() + 1e-9); r = rec / (gt.max() + 1e-9)
    sm = F.avg_pool3d(r[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    hp = r - sm
    brain = g > 0.05 * g.max()
    gg = _gmag(g)
    thr = _quantile(gg[brain], 0.30)
    flat = brain & (gg <= thr)
    return float(hp[flat].std())


# ---- data -----------------------------------------------------------------------------------
class Pair:
    """Everything a reconstruction needs, on the device and in the normalised gauge."""

    def __init__(self, contrast: str, dev, chunk: int = 4):
        assert contrast in SCANS, contrast
        self.contrast, self.dev, self.chunk = contrast, dev, chunk
        with h5py.File(R / f"sim/gs/{contrast}_kspace.h5", "r") as f:
            y_full = torch.from_numpy(f["kspace"][:]).to(torch.complex64)
        with h5py.File(R / f"sim/gs/{contrast}_csm.h5", "r") as f:
            S = torch.from_numpy(f["map"][:]).to(torch.complex64)
        with h5py.File(R / "sim/mask/mask_af16_3d.h5", "r") as f:
            m3 = torch.from_numpy(f["mask"][:]).to(torch.float32)
        with np.load(R / "sim/gt.npz") as g:
            gt_c = torch.from_numpy(np.asarray(g[contrast])).to(torch.complex64)
            self.scale = float(g[f"scale_{contrast}"])
        assert y_full.shape == S.shape, (y_full.shape, S.shape)
        assert tuple(gt_c.shape) == tuple(m3.shape) == tuple(y_full.shape[1:]), (gt_c.shape, m3.shape, y_full.shape)
        self.y_full = (y_full / self.scale).to(dev); self.S = S.to(dev)
        self.m3 = (m3 > 0.5).to(dev)
        self.NC = self.y_full.shape[0]
        self.G = tuple(self.m3.shape)
        assert bool((self.m3 == self.m3[0:1]).all()), "mask is not constant along the readout axis"
        self.m2 = self.m3[0]                                   # (ky, kz) = (232, 176)
        self.gt_c = (gt_c / self.scale).to(dev)
        self.gt = self.gt_c.abs()
        self.y = self.y_full * self.m3.unsqueeze(0)             # the retrospectively acquired data

    def sense_adj(self, k):
        acc = torch.zeros(self.m3.shape, dtype=torch.complex64, device=self.dev)
        for i in range(0, self.NC, self.chunk):
            acc += (self.S[i:i + self.chunk].conj() * ifft3c(k[i:i + self.chunk])).sum(0)
        return acc

    def AHA(self, x, mask=None):
        m = self.m3 if mask is None else mask
        acc = torch.zeros_like(x)
        for i in range(0, self.NC, self.chunk):
            acc += (self.S[i:i + self.chunk].conj()
                    * ifft3c(fft3c(self.S[i:i + self.chunk] * x) * m)).sum(0)
        return acc

    def score(self, x_complex, method: str, seconds: float = 0.0, **extra) -> dict:
        rec = x_complex.abs()
        met = evaluate_reconstruction(x_complex, self.gt_c, compute_extended=False)
        gt_np = self.gt.cpu().numpy()
        ho = score_headonly(rec.cpu().numpy(), gt_np)
        head = torch.from_numpy(head_mask(gt_np)).to(self.dev)
        e_gt = float((self.gt[head] ** 2).sum())
        d_mag = float(((rec - self.gt)[head] ** 2).sum())
        mse_c = float((x_complex - self.gt_c)[head].abs().square().mean())
        gmax = float(self.gt.max())
        psnr_cplx = float("inf") if mse_c == 0.0 else float(20 * np.log10(gmax) - 10 * np.log10(mse_c))
        return dict(method=method, contrast=self.contrast,
                    psnr=float(met["psnr"]), ssim=float(met["ssim"]), nmse=float(met["nmse"]),
                    hfen=hfen(rec, self.gt), sharp=sharp_ratio(rec, self.gt),
                    grain=grain(rec, self.gt), seconds=seconds,
                    **{k: float(v) for k, v in ho.items()},
                    head_nmse=d_mag / e_gt, psnr_cplx=psnr_cplx, **extra)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd $WT/ce_retro_cssense_af16/scripts && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_gauge.py -q`
Expected: `3 passed`。若 `evaluate_reconstruction` 对 16³ 体的 SSIM 报窗口错误，把 fixture 网格改成 (24, 24, 24) 再跑。

- [ ] **Step 5: Commit**

```bash
cd $WT && git add ce_retro_cssense_af16/scripts/t1t2_common.py ce_retro_cssense_af16/scripts/tests/test_gauge.py && git -c user.name="Congcong Liu" -c user.email="sober235@users.noreply.github.com" commit -m "ce retro af16: t1t2_common gauge on the CS-SENSE truth (pre/post, head NMSE, complex PSNR)"
```

---

### Task 5: 合成 k 空间 `sim_cssense_af16.py` 与 GPU 门 `check_sim.py`

**Files:**
- Create: `scripts/sim_cssense_af16.py`
- Create: `scripts/check_sim.py`
- Test: `scripts/tests/test_sim.py`

**Interfaces:**
- Consumes: Task 2 的 `sim/mask/mask_af16_pe.npy`；Task 4 的 `t1t2_common.fft3c`（同一 FFT 约定）。
- Produces: `sim/work_{pre,post}/{ksp_full.npy,csm.npy,mask2d.npy}`（已掩膜，供 CE 风格加载）、`sim/gs/{pre,post}_kspace.h5`（键 `kspace`，含噪未掩膜，(42,256,232,176) complex64）、`sim/gs/{pre,post}_csm.h5`（键 `map`）、`sim/gt.npz`（pre, post, mask2d, scale_pre, scale_post）、`sim/meta.json`、`logs/prep_04_export.json`。函数 `sigma_from_residual(S, x, y_meas, m0, dev, chunk=6) -> (alpha, kres, sigma)`、`synth_kspace(S, x, sigma, rng, dev, chunk=6) -> np.ndarray complex64 (NC,*G)`。

- [ ] **Step 1: 写失败的测试**

`scripts/tests/test_sim.py`:
```python
import numpy as np
import torch

import sim_cssense_af16 as SIM
from conftest import fft3c


def _toy(seed=0, G=(8, 8, 8), NC=2):
    rng = np.random.default_rng(seed)
    S = (rng.random((NC,) + G) + 1j * rng.random((NC,) + G)).astype(np.complex64)
    x = (rng.random(G) + 1j * rng.random(G)).astype(np.complex64)
    return rng, S, x


def test_sigma_from_residual_recovers_known_noise():
    rng, S, x = _toy(G=(16, 16, 16), NC=4)
    m0 = rng.random((16, 16)) < 0.5
    sigma0 = 0.05
    y = fft3c(S * x[None]) + sigma0 * (rng.standard_normal(S.shape) + 1j * rng.standard_normal(S.shape))
    y = (y * m0[None, None]).astype(np.complex64)
    alpha, kres, sigma = SIM.sigma_from_residual(torch.from_numpy(S), torch.from_numpy(x), torch.from_numpy(y),
                                                 torch.from_numpy(m0), torch.device("cpu"), chunk=3)
    assert abs(alpha - 1) < 0.02
    assert abs(sigma / sigma0 - 1) < 0.05
    n = m0.sum() * S.shape[0]
    expect_kres = np.sqrt(2 * n) * sigma0 / np.linalg.norm(y[:, :, m0])
    assert abs(kres / expect_kres - 1) < 0.1


def test_synth_kspace_noise_level_and_seed():
    rng, S, x = _toy()
    k1 = SIM.synth_kspace(torch.from_numpy(S), torch.from_numpy(x), 0.1, np.random.default_rng(7), torch.device("cpu"), chunk=1)
    k2 = SIM.synth_kspace(torch.from_numpy(S), torch.from_numpy(x), 0.1, np.random.default_rng(7), torch.device("cpu"), chunk=1)
    clean = fft3c(S * x[None])
    assert k1.dtype == np.complex64 and np.array_equal(k1, k2)
    noise = k1 - clean
    assert abs(noise.real.std() / 0.1 - 1) < 0.1 and abs(noise.imag.std() / 0.1 - 1) < 0.1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd $WT/ce_retro_cssense_af16/scripts && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_sim.py -q`
Expected: FAIL，`No module named 'sim_cssense_af16'`

- [ ] **Step 3: 实现合成脚本**

`scripts/sim_cssense_af16.py`:
```python
#!/usr/bin/env python3
"""sim_cssense_af16.py -- synthetic pre/post k-space: the CS-SENSE volumes as truth, the real 42-coil maps,
the Poisson-disc AF16 mask, complex Gaussian noise at the level the real acquisition showed.

Per scan:  sigma  from r = M0 (y_meas - alpha F(S x_GT)) on the prospective mask M0 (alpha = least-squares complex
           scale), sigma = sqrt(mean|r|^2 / 2); the relative residual ||r||/||M0 y|| must reproduce the archived
           CS-SENSE kres (0.18487 pre / 0.14969 post) or the run stops -- that is the FFT/CSM convention gate.
           y_noisy = F(S x_GT) + sigma (N + iN) on every k-space point (seed 100 pre / 200 post), then
           work_<scan>/ksp_full.npy = mask . y_noisy   and   gs/<scan>_kspace.h5 = y_noisy (the trainer masks itself).
  python scripts/sim_cssense_af16.py --gpu 0        (about 10 min; --gpu -1 for CPU, slower)
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import torch

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
CE = Path("/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast")
GT = {s: CE / f"tv_sense/cssense_o200_lam0.002_full_{s}_complex.npy" for s in ("pre", "post")}
MEAS = {"pre": Path("/data0/congcong/code/NV-Generate-CTMR/recon_real/work_ce_t1_16887049370010"),
        "post": Path("/data0/congcong/code/NV-Generate-CTMR/recon_real/work_ce_t1_16887049370060")}
ROI = CE / "lesion_candidates/enhance_rois_cssense.npz"
KRES_REF = {"pre": 0.18487, "post": 0.14969}
NOISE_SEED = {"pre": 100, "post": 200}
NX, NY, NZ = 256, 232, 176


def _c(fn, x, dims):
    return torch.fft.fftshift(fn(torch.fft.ifftshift(x, dim=dims), dim=dims, norm="ortho"), dim=dims)


def fft3c(x):  return _c(torch.fft.fftn, x, (-3, -2, -1))
def ifft3c(x): return _c(torch.fft.ifftn, x, (-3, -2, -1))


def sigma_from_residual(S, x, y_meas, m0, dev, chunk=6):
    """alpha = <A x, y>_M0 / ||A x||_M0^2 ; kres = ||M0 (alpha A x - y)|| / ||M0 y|| ; sigma per real component."""
    x = x.to(dev); m0 = m0.to(dev).bool()
    num = 0j; den = 0.0; yy = 0.0
    for i in range(0, S.shape[0], chunk):
        k = fft3c(S[i:i + chunk].to(dev) * x)[..., m0].to(torch.complex128)
        ym = y_meas[i:i + chunk].to(dev)[..., m0].to(torch.complex128)
        num += complex((k.conj() * ym).sum().item()); den += float((k.abs() ** 2).sum().item()); yy += float((ym.abs() ** 2).sum().item())
    alpha = num / den
    res = 0.0
    for i in range(0, S.shape[0], chunk):
        k = fft3c(S[i:i + chunk].to(dev) * x)[..., m0].to(torch.complex128)
        ym = y_meas[i:i + chunk].to(dev)[..., m0].to(torch.complex128)
        res += float(((alpha * k - ym).abs() ** 2).sum().item())
    n = int(m0.sum().item()) * S.shape[0]
    return alpha, float(np.sqrt(res / yy)), float(np.sqrt(res / (2.0 * n)))


def synth_kspace(S, x, sigma, rng, dev, chunk=6):
    """F(S x) + sigma (N + iN) on every point, coil by coil in a fixed order so the seed fixes the noise."""
    x = x.to(dev)
    out = np.empty(tuple(S.shape), np.complex64)
    for i in range(0, S.shape[0], chunk):
        k = fft3c(S[i:i + chunk].to(dev) * x).cpu().numpy()
        n = rng.standard_normal(k.shape, dtype=np.float32) + 1j * rng.standard_normal(k.shape, dtype=np.float32)
        out[i:i + chunk] = (k + np.float32(sigma) * n).astype(np.complex64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cpu") if a.gpu < 0 else torch.device(f"cuda:{a.gpu}")
    if a.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu); dev = torch.device("cuda:0")
    t0 = time.time()
    mask2d = np.load(R / "sim/mask/mask_af16_pe.npy") > 0
    assert mask2d.shape == (NY, NZ) and int(mask2d.sum()) == NY * NZ // 16
    with np.load(ROI) as r:
        wm = r["roi2"]
    (R / "sim/gs").mkdir(parents=True, exist_ok=True)
    gt_out, meta = {"mask2d": mask2d}, dict(grid=[NX, NY, NZ], pe_points=int(mask2d.sum()), pe_total=NY * NZ,
                                           af=NY * NZ / int(mask2d.sum()), scans={}, gt=GT_str(), fft="centred orthonormal (ce_common.fft3c)")
    for scan in ("pre", "post"):
        x = torch.from_numpy(np.load(GT[scan])).to(torch.complex64)
        S = torch.from_numpy(np.load(MEAS[scan] / "csm.npy")).to(torch.complex64)
        y_meas = torch.from_numpy(np.load(MEAS[scan] / "ksp_full.npy")).to(torch.complex64)
        m0 = torch.from_numpy(np.load(MEAS[scan] / "mask2d.npy")[0] > 0.5)
        assert x.shape == (NX, NY, NZ) and S.shape == y_meas.shape == (42, NX, NY, NZ)
        alpha, kres, sigma = sigma_from_residual(S, x, y_meas, m0, dev)
        print(f"[{scan}] alpha={alpha:.5f} kres={kres:.5f} (ref {KRES_REF[scan]}) sigma={sigma:.5e}  {time.time() - t0:.0f}s", flush=True)
        if abs(kres - KRES_REF[scan]) >= 1e-3:
            raise SystemExit(f"[{scan}] kres {kres:.5f} does not reproduce the archived {KRES_REF[scan]}: FFT/CSM convention mismatch")
        del y_meas
        y = synth_kspace(S, x, sigma, np.random.default_rng(NOISE_SEED[scan]), dev)
        clean_sample = fft3c(S[:2].to(dev) * x.to(dev)).cpu().numpy()
        sigma_meas = float((y[:2] - clean_sample).real.std())
        assert abs(sigma_meas / sigma - 1) < 0.02, (sigma_meas, sigma)
        with h5py.File(R / f"sim/gs/{scan}_kspace.h5", "w") as f:
            f.create_dataset("kspace", data=y)
        with h5py.File(R / f"sim/gs/{scan}_csm.h5", "w") as f:
            f.create_dataset("map", data=S.numpy())
        w = R / f"sim/work_{scan}"; w.mkdir(exist_ok=True)
        np.save(w / "ksp_full.npy", (y * mask2d[None, None]).astype(np.complex64))
        shutil.copyfile(MEAS[scan] / "csm.npy", w / "csm.npy")
        np.save(w / "mask2d.npy", np.broadcast_to(mask2d[None], (NX, NY, NZ)).astype(np.float32))
        xa = x.abs().numpy()
        sos = torch.sqrt((S.abs() ** 2).sum(0)).numpy()
        snr_wm = float(np.median(xa[wm]) / np.median(sigma * sos[wm]))
        gt_out[scan] = x.numpy(); gt_out[f"scale_{scan}"] = float(np.quantile(xa, 0.999))
        meta["scans"][scan] = dict(alpha_re=alpha.real, alpha_im=alpha.imag, kres_on_prospective_mask=kres, kres_ref=KRES_REF[scan],
                                   sigma=sigma, sigma_measured_first_two_coils=sigma_meas, noise_seed=NOISE_SEED[scan],
                                   wm_snr_sense_adjoint=snr_wm, scale_p999=gt_out[f"scale_{scan}"], csm=str(MEAS[scan] / "csm.npy"))
        print(f"[{scan}] wrote gs/{scan}_kspace.h5 work_{scan}/  WM SNR {snr_wm:.1f}  {time.time() - t0:.0f}s", flush=True)
        del S, y, x
    np.savez(R / "sim/gt.npz", **gt_out)
    json.dump(meta, open(R / "sim/meta.json", "w"), indent=1)
    (R / "logs").mkdir(exist_ok=True)
    af = meta["af"]
    json.dump(dict(mask2d_shape=[NY, NZ], mask2d_samples=int(mask2d.sum()), mask2d_AF_rect=af,
                   contrasts={s: dict(AF_vs_own_support=af, AF_vs_full_rect=af) for s in ("pre", "post")}),
              open(R / "logs/prep_04_export.json", "w"), indent=1)
    print(f"DONE {time.time() - t0:.0f}s -> sim/gt.npz sim/meta.json logs/prep_04_export.json")


def GT_str():
    return {s: str(p) for s, p in GT.items()}


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd $WT/ce_retro_cssense_af16/scripts && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_sim.py -q`
Expected: `2 passed`

- [ ] **Step 5: 生成正式合成数据（门 1）**

先看空卡：`nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv`，选一张利用率 < 30% 的。
Run: `cd $TOPIC && env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID $PY scripts/sim_cssense_af16.py --gpu <G> 2>&1 | tee logs/sim.log`
Expected: `[pre] … kres=0.1849 (ref 0.18487)`、`[post] … kres=0.1497 (ref 0.14969)`（差 < 1e-3，否则脚本自己 SystemExit），两个 `wrote`，最后 `DONE`。`ls -la sim/gs sim/work_pre sim/work_post sim/gt.npz sim/meta.json` 共约 14 GB。把 `sim/meta.json` 的 σ、kres、WM SNR 记进汇报。

- [ ] **Step 6: 写 GPU 门脚本 check_sim.py（门 2）**

`scripts/check_sim.py`:
```python
#!/usr/bin/env python3
"""check_sim.py -- the operator gates of the spec (section 10, items 1-2), on the real synthetic data, one GPU.
  (a) noiseless full k-space: x_hat = sum_c conj(S_c) ifft3c(F(S_c x_GT)) / sum_c |S_c|^2 == x_GT inside the head (rel err < 1e-4)
  (b) CG-SENSE (Tikhonov 1e-6, 60 iters) on that noiseless full k-space: head PSNR > 60 dB
  (c) the noise actually written: std of (gs/<scan>_kspace.h5 - F(S x_GT)) within 2 % of meta sigma
  (d) for the record: ZF head PSNR of the AF16 noisy data
Writes logs/check_sim.json; exits non-zero on any failed gate.
  python scripts/check_sim.py --gpu 0
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, required=True); a = ap.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1t2_common import Pair, R, fft3c, ifft3c, head_mask   # noqa: E402

dev = torch.device("cuda:0")
meta = json.load(open(R / "sim/meta.json"))
out, ok = {}, True


def cg(applyA, b, iters):
    x = torch.zeros_like(b); r = b.clone(); p = r.clone(); rs = (r.conj() * r).sum().real
    for _ in range(iters):
        Ap = applyA(p); al = rs / (p.conj() * Ap).sum().real.clamp_min(1e-30)
        x = x + al * p; r = r - al * Ap; rs_new = (r.conj() * r).sum().real
        p = r + (rs_new / rs.clamp_min(1e-30)) * p; rs = rs_new
    return x


for scan in ("pre", "post"):
    P = Pair(scan, dev)
    head = torch.from_numpy(head_mask(P.gt.cpu().numpy())).to(dev)
    sos2 = (P.S.abs() ** 2).sum(0)
    adj = torch.zeros_like(P.gt_c); noise2 = 0.0; npts = 0
    for i in range(0, P.NC, P.chunk):
        k = fft3c(P.S[i:i + P.chunk] * P.gt_c)                        # noiseless, gauge units
        adj += (P.S[i:i + P.chunk].conj() * ifft3c(k)).sum(0)
        d = (P.y_full[i:i + P.chunk] - k)                             # the written noise (unmasked h5), gauge units
        noise2 += float(d.real.square().sum() + d.imag.square().sum()); npts += 2 * d.numel()
    xhat = adj / sos2.clamp_min(1e-6)
    rel = float(((xhat - P.gt_c)[head].abs().norm() / P.gt_c[head].abs().norm()))
    ones = torch.ones_like(P.m3)
    xcg = cg(lambda v: P.AHA(v, ones) + 1e-6 * v, adj, 60)
    mse = float(((xcg.abs() - P.gt)[head] ** 2).mean()); psnr_cg = float(20 * np.log10(P.gt.max().item()) - 10 * np.log10(mse))
    sigma_w = float(np.sqrt(noise2 / npts)) * P.scale; sigma_ref = meta["scans"][scan]["sigma"]
    zf = P.score(P.sense_adj(P.y), "zf")["head_psnr"]
    row = dict(adjoint_rel_err=rel, cg_noiseless_head_psnr=psnr_cg, sigma_written=sigma_w, sigma_meta=sigma_ref,
               sigma_rel_dev=abs(sigma_w / sigma_ref - 1), zf_af16_head_psnr=zf,
               gates=dict(adjoint=rel < 1e-4, cg=psnr_cg > 60, sigma=abs(sigma_w / sigma_ref - 1) < 0.02))
    ok &= all(row["gates"].values()); out[scan] = row
    print(f"[{scan}] adjoint rel err {rel:.2e}  CG noiseless head PSNR {psnr_cg:.2f} dB  sigma {sigma_w:.4e} vs {sigma_ref:.4e}  ZF AF16 {zf:.2f} dB  gates {row['gates']}", flush=True)
    del P, adj, xhat, xcg; torch.cuda.empty_cache()
json.dump(out, open(R / "logs/check_sim.json", "w"), indent=1)
print("CHECK_SIM " + ("OK" if ok else "FAILED")); sys.exit(0 if ok else 1)
```

- [ ] **Step 7: 跑门 2**

Run: `cd $TOPIC && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/check_sim.py --gpu <G> 2>&1 | tee logs/check_sim.log`
Expected: 两个扫描都 `gates {'adjoint': True, 'cg': True, 'sigma': True}`，末行 `CHECK_SIM OK`。若 adjoint 失败（相对误差 ~1e-2 量级），说明 CSM 的 Σ|S|² 在头内有接近零的体素：把 `clamp_min(1e-6)` 处的比较改为只在 `sos2 > 0.25` 的头内体素上算，并把这个条件写进 json 与 README。

- [ ] **Step 8: Commit**

```bash
cd $WT && cp $TOPIC/sim/meta.json ce_retro_cssense_af16/logs/sim_meta.json && cp $TOPIC/logs/check_sim.json $TOPIC/logs/prep_04_export.json ce_retro_cssense_af16/logs/ && git add ce_retro_cssense_af16 && git -c user.name="Congcong Liu" -c user.email="sober235@users.noreply.github.com" commit -m "ce retro af16: synthetic k-space from the CS-SENSE truth with measured-level noise, operator gates pass"
```

---

### Task 6: 经典臂：`recon_classical.py`（围绕 Pair 重写）与 `recon_joint_cssense.py`（原样拷贝）

**Files:**
- Create: `scripts/recon_classical.py`
- Copy: `/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/scripts/recon_joint_cssense.py` → `scripts/recon_joint_cssense.py`（零改动）
- Test: `scripts/tests/test_classical.py`

**Interfaces:**
- Produces: `recon/{zf,cgsense}_<scan>_complex.npy`、`recon/cssense<suffix>_<scan>_complex.npy`、`recon/gt_<scan>_complex.npy`（口径单位的 GT，供图脚本）、`logs/classical_<scan><suffix>.json`（`{"contrast","scale_p999","coils","mask_frac","rows":[…]}`，每行 = `Pair.score` 的字典 + `lam mu tik`）；Joint：`recon/jcssense<suffix>_<scan>_complex.npy`、`logs/jcssense<suffix>.json`。CLI：`recon_classical.py --scan {pre,post} --methods zf,cgsense,cssense --gpu G [--cg_iters 60 --tik 0.01 --admm_outer 35 --admm_cg 8 --lam 1e-3 --mu 0.1 --suffix ""]`；`recon_joint_cssense.py --topic $TOPIC --contrasts pre,post --gpu G --admm_outer 200 --lam L --suffix _o200_lamL`。

- [ ] **Step 1: 写失败的测试（求解器在 tiny_topic 上的行为）**

`scripts/tests/test_classical.py`:
```python
import subprocess, sys, json
from pathlib import Path

import numpy as np
import torch


def test_cg_solves_normal_equations(tiny_topic):
    import t1t2_common as C, recon_classical as RC
    P = C.Pair("pre", torch.device("cpu"))
    b = P.sense_adj(P.y)
    x = RC.cg(lambda v: P.AHA(v) + 0.01 * v, b, 60)
    r = P.AHA(x) + 0.01 * x - b
    assert float(r.abs().norm() / b.abs().norm()) < 1e-3


def test_div3_is_adjoint_of_grad3():
    import recon_classical as RC
    g = torch.Generator().manual_seed(0)
    x = torch.randn(6, 5, 4, generator=g, dtype=torch.complex64)
    k = torch.randn(3, 6, 5, 4, generator=g, dtype=torch.complex64)
    lhs = torch.vdot(RC.grad3(x).flatten(), k.flatten()); rhs = torch.vdot(x.flatten(), RC.div3(k).flatten())
    assert abs(lhs - rhs) < 1e-4 * abs(lhs)


def test_cli_runs_zf_on_tiny_topic_cpu(tiny_topic):
    top = tiny_topic["root"]
    py = sys.executable
    r = subprocess.run([py, str(Path(__file__).resolve().parents[1] / "recon_classical.py"), "--scan", "pre", "--methods", "zf",
                        "--gpu", "-1"], capture_output=True, text=True, env={**dict(__import__("os").environ), "RETRO_TOPIC": str(top)})
    assert r.returncode == 0, r.stderr
    assert (top / "recon/zf_pre_complex.npy").exists() and (top / "recon/gt_pre_complex.npy").exists()
    rows = json.load(open(top / "logs/classical_pre.json"))["rows"]
    assert rows[0]["method"] == "zf" and rows[0]["head_psnr"] > 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd $WT/ce_retro_cssense_af16/scripts && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_classical.py -q`
Expected: FAIL，`No module named 'recon_classical'`

- [ ] **Step 3: 实现 recon_classical.py**

```python
#!/usr/bin/env python3
"""recon_classical.py -- ZF / CG-SENSE / CS-SENSE (ADMM isotropic 3-D TV) on the synthetic AF16 pre/post pair.

Solvers and defaults are the jyz AF16 ones (T1FLAIR_AF16_v3_jiangyizhou/scripts/recon_classical.py), the data and
gauge come from t1t2_common.Pair (the CS-SENSE truth). --gpu -1 runs on the CPU (tests, tiny data).
  cgsense : CG on (A^H A + tik I) x = A^H y,  tik 0.01, 60 iters
  cssense : ADMM isotropic 3-D TV SENSE,      lam, mu 0.1, admm_outer x admm_cg CG   (the registered arm: --admm_outer 200 with a lambda sweep)
"""
import argparse, json, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--scan", required=True, choices=["pre", "post"])
ap.add_argument("--methods", default="zf,cgsense,cssense")
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--cg_iters", type=int, default=60)
ap.add_argument("--tik", type=float, default=0.01)
ap.add_argument("--admm_outer", type=int, default=35)
ap.add_argument("--admm_cg", type=int, default=8)
ap.add_argument("--lam", type=float, default=1e-3)
ap.add_argument("--mu", type=float, default=0.1)
ap.add_argument("--suffix", default="", help="appended to the method key and file names (lambda sweeps); empty = the canonical arm")
args = ap.parse_args() if __name__ == "__main__" else ap.parse_args(["--scan", "pre", "--gpu", "-1"])

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("PYTHONNOUSERSITE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch                                             # noqa: E402
from t1t2_common import Pair, R                                       # noqa: E402

log = lambda *a: print(*a, flush=True)


def cg(applyA, b, iters, tol=1e-10):
    x = torch.zeros_like(b); r = b.clone(); p = r.clone()
    rs = (r.conj() * r).sum().real
    for _ in range(iters):
        Ap = applyA(p)
        a = rs / (p.conj() * Ap).sum().real.clamp_min(1e-30)
        x = x + a * p; r = r - a * Ap
        rs_new = (r.conj() * r).sum().real
        if rs_new.sqrt() < tol:
            break
        p = r + (rs_new / rs.clamp_min(1e-30)) * p
        rs = rs_new
    return x


def grad3(x):
    return torch.stack([torch.roll(x, -1, 0) - x, torch.roll(x, -1, 1) - x, torch.roll(x, -1, 2) - x], 0)


def div3(g):                            # already the adjoint D^H (CE gotcha: do not negate)
    gx, gy, gz = g[0], g[1], g[2]
    return ((torch.roll(gx, 1, 0) - gx) + (torch.roll(gy, 1, 1) - gy) + (torch.roll(gz, 1, 2) - gz))


def run(P, method):
    t = time.time()
    if method == "zf":
        x = P.sense_adj(P.y)
    elif method == "cgsense":
        x = cg(lambda v: P.AHA(v) + args.tik * v, P.sense_adj(P.y), args.cg_iters)
    elif method == "cssense":
        b0 = P.sense_adj(P.y)
        x = torch.zeros_like(b0); z = grad3(x); u = torch.zeros_like(z)
        for _ in range(args.admm_outer):
            rhs = b0 + args.mu * div3(z - u)
            x = cg(lambda v: P.AHA(v) + args.mu * div3(grad3(v)), rhs, args.admm_cg)
            v = grad3(x) + u
            nv = torch.sqrt((v.abs() ** 2).sum(0, keepdim=True) + 1e-12)
            z = v * torch.clamp(1.0 - args.lam / (args.mu * nv), min=0.0)
            u = v - z
    else:
        raise ValueError(method)
    row = P.score(x, method + args.suffix, seconds=time.time() - t, lam=args.lam, mu=args.mu, tik=args.tik)
    np.save(R / f"recon/{method}{args.suffix}_{P.contrast}_complex.npy", x.cpu().numpy().astype(np.complex64))
    log(f"[{P.contrast}] {method + args.suffix:18s} head_psnr={row['head_psnr']:.3f} head_ssim={row['head_ssim']:.4f} "
        f"head_nmse={row['head_nmse']:.5f} head_hfen={row['head_hfen']:.4f} psnr={row['psnr']:.3f} ({row['seconds']:.0f} s)")
    return row


def main():
    t0 = time.time()
    dev = torch.device("cpu") if args.gpu < 0 else torch.device("cuda:0")
    (R / "recon").mkdir(exist_ok=True); (R / "logs").mkdir(exist_ok=True)
    P = Pair(args.scan, dev)
    log(f"[{args.scan}] coils={P.NC} vol={P.G} mask frac={P.m3.float().mean():.5f} AF_rect={1 / P.m3.float().mean():.3f} scale={P.scale:.4e}")
    rows = [run(P, m) for m in args.methods.split(",")]
    if not args.suffix:
        np.save(R / f"recon/gt_{args.scan}_complex.npy", P.gt_c.cpu().numpy().astype(np.complex64))
    p = R / f"logs/classical_{args.scan}{args.suffix}.json"
    json.dump({"contrast": args.scan, "scale_p999": P.scale, "coils": int(P.NC), "mask_frac": float(P.m3.float().mean()),
               "rows": rows, "total_seconds": time.time() - t0}, open(p, "w"), indent=2)
    log(f"WROTE {p}  total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
```
注意：`args = ap.parse_args() if __name__ == "__main__" else …` 让测试能 import `cg/grad3/div3` 而不解析 pytest 的 argv；`recon/` 与 `logs/` 在 `main()` 开头建好，`run()` 里的 `np.save` 才不会因目录不存在而失败。

- [ ] **Step 4: 拷贝 joint 脚本**

```bash
cp /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/scripts/recon_joint_cssense.py $WT/ce_retro_cssense_af16/scripts/
grep -n "T1\|FLAIR" $WT/ce_retro_cssense_af16/scripts/recon_joint_cssense.py
```
Expected: 只在 docstring 的 usage 里出现 `T1,T2`；代码通过 `--contrasts` 与 `--topic` 完全参数化，不改。

- [ ] **Step 5: 跑测试确认通过**

Run: `cd $WT/ce_retro_cssense_af16/scripts && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_classical.py -q`
Expected: `3 passed`

- [ ] **Step 6: 真数据冒烟（一张空卡，几分钟）**

```bash
cd $TOPIC && env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID $PY scripts/recon_classical.py --scan pre --methods zf,cgsense --gpu <G>
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID $PY scripts/recon_classical.py --scan pre --methods cssense --admm_outer 2 --lam 1e-3 --suffix _smoke --gpu <G>
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID $PY scripts/recon_joint_cssense.py --topic $TOPIC --contrasts pre,post --admm_outer 2 --lam 1e-3 --suffix _smoke --gpu <G>
```
Expected: 三条都 `WROTE …`；CG-SENSE 的 head_psnr 高于 ZF；`recon/gt_pre_complex.npy` 出现。然后删掉冒烟产物：`rm $TOPIC/recon/*_smoke_* $TOPIC/logs/*_smoke*`。

- [ ] **Step 7: Commit**

```bash
cd $WT && git add ce_retro_cssense_af16/scripts/recon_classical.py ce_retro_cssense_af16/scripts/recon_joint_cssense.py ce_retro_cssense_af16/scripts/tests/test_classical.py && git -c user.name="Congcong Liu" -c user.email="sober235@users.noreply.github.com" commit -m "ce retro af16: ZF / CG-SENSE / CS-SENSE and joint CS-SENSE arms on the synthetic pair"
```

---

### Task 7: L1-SPIRiT 臂移植

**Files:**
- Copy+edit: `/data0/congcong/code/GS/mcgs_spirit_wt/t1flair_af16_v3_jiangyizhou/scripts/recon_spirit.py` → `scripts/recon_spirit.py`

**Interfaces:**
- Produces: `recon/spirit<suffix>_<scan>_complex.npy`、`logs/spirit_<scan><suffix>.json`（rows 含 `ksize lam`）。CLI：`recon_spirit.py --scan {pre,post} --gpu G --ksize {3,5} --lam L --tyk 0.01 --iters 150 --levels 4 --solver fista --chunk 24 --calib acquired --suffix _ks<K>_lam<L>`。

- [ ] **Step 1: 拷贝并做三处改动**

```bash
S=/data0/congcong/code/GS/mcgs_spirit_wt/t1flair_af16_v3_jiangyizhou/scripts/recon_spirit.py; D=$WT/ce_retro_cssense_af16/scripts/recon_spirit.py
cp $S $D
sed -i 's/ap.add_argument("--contrast", required=True, choices=\["T1", "FLAIR"\])/ap.add_argument("--scan", dest="contrast", required=True, choices=["pre", "post"])/' $D
grep -n 'choices=\["pre", "post"\]' $D && grep -n "from t1t2_common import" $D && grep -n "sys.path.insert" $D
```
Expected: 第一条 grep 命中一行（参数改名为 `--scan`，内部仍叫 `args.contrast`，其余代码不用动）；第二条命中 `from t1t2_common import Pair, R, fft2c, ifft2c, ifft1c`；第三条显示它把 `Path(__file__).resolve().parent` 加进了 sys.path（若没有，在 import 之前加一行 `sys.path.insert(0, str(Path(__file__).resolve().parent))`）。再把 docstring 里 `--contrast T1` 的 usage 行改成 `--scan pre`。

- [ ] **Step 2: 语法与 CLI 自检**

Run: `cd $WT/ce_retro_cssense_af16/scripts && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -c "import ast,sys; ast.parse(open('recon_spirit.py').read()); print('syntax ok')" && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY recon_spirit.py --help | grep -- --scan`
Expected: `syntax ok` 和 `--scan {pre,post}`。

- [ ] **Step 3: 真数据冒烟**

Run: `cd $TOPIC && env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID $PY scripts/recon_spirit.py --scan pre --gpu <G> --ksize 5 --lam 1e-3 --iters 3 --suffix _smoke --save 0`
Expected: 日志有 `ACS=24x24 at (104,76)`（24×24 块，左上角 116−12=104、88−12=76）和 `calibration matrix 400 x 1050  (under-determined, Tikhonov carries it)`，结束时打印 head_psnr，写出 `logs/spirit_pre_smoke.json`。再跑 `--ksize 3` 同样参数，Expected `calibration matrix 484 x 378`。删冒烟日志。

- [ ] **Step 4: Commit**

```bash
cd $WT && git add ce_retro_cssense_af16/scripts/recon_spirit.py && git -c user.name="Congcong Liu" -c user.email="sober235@users.noreply.github.com" commit -m "ce retro af16: L1-SPIRiT arm (jyz transcription, pre/post names)"
```

---

### Task 8: ZS-SSDU 臂移植

**Files:**
- Copy+edit: `/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/scripts/recon_zsssdu.py` → `scripts/recon_zsssdu.py`

**Interfaces:**
- Produces: `recon/zsssdu_<scan>_complex.npy`、`logs/zsssdu_<scan>.json`。CLI：`recon_zsssdu.py --scan {pre,post} --gpu G [--epochs 100 --unrolls 10 --cg_iters_net 6 --batch 4 --lr 5e-4 --mu 0.05 --patience 10]`。

- [ ] **Step 1: 拷贝并改对比度参数**

```bash
S=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/scripts/recon_zsssdu.py; D=$WT/ce_retro_cssense_af16/scripts/recon_zsssdu.py
cp $S $D
sed -i 's/ap.add_argument("--contrast", required=True, choices=\["T1", "FLAIR"\])/ap.add_argument("--scan", dest="contrast", required=True, choices=["pre", "post"])/' $D
grep -n 'dest="contrast"' $D; grep -n "sys.path.insert\|from t1t2_common" $D; grep -n "\b26\b\|256\b" $D
```
Expected: 改名命中；t1t2_common 从脚本目录 import；最后一条 grep 只应命中 docstring 里的说明（"26 coils"），没有把 26 或 256 写死在代码里的行。若有写死的 `256`（读出层数），改成 `NX = yh.shape[1]` 已有的变量。

- [ ] **Step 2: 自检**

Run: `cd $WT/ce_retro_cssense_af16/scripts && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY recon_zsssdu.py --help | grep -- --scan`
Expected: `--scan {pre,post}`

- [ ] **Step 3: 真数据冒烟（1 epoch，约 10 分钟，显存约 30 GB）**

Run: `cd $TOPIC && env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID $PY scripts/recon_zsssdu.py --scan post --gpu <G> --epochs 1`
Expected: 日志 `coils=42 grid=(256, 232, 176)`，跑完写 `logs/zsssdu_post.json` 与 `recon/zsssdu_post_complex.npy`。删掉这两个冒烟产物（全量会重写，但别让 orchestrator 把它当已完成）。

- [ ] **Step 4: Commit**

```bash
cd $WT && git add ce_retro_cssense_af16/scripts/recon_zsssdu.py && git -c user.name="Congcong Liu" -c user.email="sober235@users.noreply.github.com" commit -m "ce retro af16: ZS-SSDU arm (jyz settings, 42 coils)"
```

---

### Task 9: 3D-DIP / 3D-INR 臂移植

**Files:**
- Copy+edit: `/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/scripts/recon_dip_inr.py` → `scripts/recon_dip_inr.py`

**Interfaces:**
- Produces: `recon/{dip,inr}_<scan>_complex.npy`、`logs/{dip,inr}_<scan>.json`（rows 含 `kres_ho best_iter holdout_lines total_lines`）。CLI：`recon_dip_inr.py --method {dip,inr} --scan {pre,post} --gpu G --iters N --eval_every E [--lr -1 --holdout 0.05 --tv 0]`；正式参数 DIP `--iters 5000 --lr 3e-3 --eval_every 100`，INR `--iters 1500 --eval_every 50`。

- [ ] **Step 1: 拷贝并改对比度参数**

```bash
S=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/scripts/recon_dip_inr.py; D=$WT/ce_retro_cssense_af16/scripts/recon_dip_inr.py
cp $S $D
sed -i 's/ap.add_argument("--contrast", required=True, choices=\["T1", "FLAIR"\])/ap.add_argument("--scan", dest="contrast", required=True, choices=["pre", "post"])/' $D
grep -n 'dest="contrast"' $D; grep -n "self.low = \|np.ceil(s / div)" $D
```
Expected: 改名命中；`self.low = tuple(max(2, int(np.ceil(s / div))) for s in self.shape)` 存在，说明 DIP 的低分辨种子网格由体形状推出（232 → 15），不用改。

- [ ] **Step 2: 自检**

Run: `cd $WT/ce_retro_cssense_af16/scripts && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY recon_dip_inr.py --help | grep -- --scan`
Expected: `--scan {pre,post}`

- [ ] **Step 3: 真数据冒烟**

```bash
cd $TOPIC && env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID $PY scripts/recon_dip_inr.py --method dip --scan pre --gpu <G> --iters 20 --eval_every 10
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID $PY scripts/recon_dip_inr.py --method inr --scan pre --gpu <G> --iters 20 --eval_every 10
```
Expected: 日志 `lines: total 2552  fit 2424  held-out 128 (5.0 %)`（2552 的 5% 取整 = 128），DIP 打印参数量 3.87M 量级，INR 16.79M 量级，两个 json 与 npy 写出。删掉这四个冒烟产物。

- [ ] **Step 4: Commit**

```bash
cd $WT && git add ce_retro_cssense_af16/scripts/recon_dip_inr.py && git -c user.name="Congcong Liu" -c user.email="sober235@users.noreply.github.com" commit -m "ce retro af16: 3D-DIP and 3D-INR arms (jyz settings)"
```

---

### Task 10: MC-INR 臂（mcinr_niessen2025 仓库补三个参数 + 本课题包装脚本）

**Files:**
- Modify: `/data0/congcong/code/GS/mcinr_niessen2025/scripts/run_volume.py:30-35`（参数区）与 `:57-63`（加载区）与 `:86`（输出目录）
- Create: `scripts/run_mcinr.py`
- Test: `scripts/tests/test_mcinr_units.py`

**Interfaces:**
- Consumes: `sim/gs/{pre,post}_kspace.h5`、`sim/gs/{pre,post}_csm.h5`、`sim/mask/mask_af16_3d.h5`；mcinr 的 `mcinr.data.to_gauge_units(x_norm, M, gauge_scale)`。
- Produces: `recon/mcinr_run/{mcinr_norm_T1_complex.npy, mcinr_norm_FLAIR_complex.npy, meta.json}`（mcinr 原生输出，T1 槽 = pre，FLAIR 槽 = post）、`recon/mcinr_<scan>_complex.npy`（口径单位）、`logs/mcinr_<scan>.json`。CLI：`run_mcinr.py --gpu G [--slices all|a:b] [--score_only]`。

- [ ] **Step 1: 在 mcinr 仓库加 `--csm_T1/--csm_FLAIR/--out`**

对 `/data0/congcong/code/GS/mcinr_niessen2025/scripts/run_volume.py` 做三处编辑：
1. 参数区（`ap.add_argument("--ksp_FLAIR", …)` 之后）加：
```python
ap.add_argument("--csm_T1", default=None, help="h5 coil maps for T1 (key map; default: the topic's ESPIRiT maps)")
ap.add_argument("--csm_FLAIR", default=None, help="h5 coil maps for FLAIR (default: the topic's ESPIRiT maps)")
ap.add_argument("--out", default=None, help="output directory (default: the topic's recon/af<AF><tag>[_smoke])")
```
2. 加载区：把 `csms.append(torch.from_numpy(C.read_h5(C.csm_path(c), "map")))` 改成
```python
    csms.append(torch.from_numpy(C.read_h5(csm_paths[c], "map")))
```
并在 `ksp_paths = {...}` 那行后面加 `csm_paths = {"T1": args.csm_T1 or C.csm_path("T1"), "FLAIR": args.csm_FLAIR or C.csm_path("FLAIR")}`。
3. 输出目录：把 `out_dir = C.recon_dir(args.af, args.tag + ("_smoke" if smoke else ""))` 改成
```python
out_dir = Path(args.out) if args.out else C.recon_dir(args.af, args.tag + ("_smoke" if smoke else ""))
```
并在 `meta = dict(…)` 里加一项 `csm={c: str(Path(csm_paths[c]).resolve()) for c in C.CONTRASTS},`。

```bash
cd /data0/congcong/code/GS/mcinr_niessen2025 && git diff --stat && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY_TCNN scripts/run_volume.py --help | grep -E "csm_T1|--out"
git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit -am "run_volume: external coil maps and output directory (retrospective CE study)"
```
Expected: help 里出现两项；提交作者与该仓库现有提交一致（`git log -1 --format='%an <%ae>'` 之前显示 `Congcong Liu <liuc23539@gmail.com>`）。

- [ ] **Step 2: 写失败的测试（单位换算与形状）**

`scripts/tests/test_mcinr_units.py`:
```python
import numpy as np
import torch


def test_to_gauge_units_roundtrip(tiny_topic):
    import run_mcinr as RM
    x_raw = torch.from_numpy((np.random.default_rng(0).random((4, 16, 16)) + 0j).astype(np.complex64))
    M, scale = 3.0, 0.5
    x_norm = x_raw / M * 100                       # the paper's normalisation of the data the net fitted
    back = RM.to_gauge(x_norm, M, scale)
    assert torch.allclose(back, x_raw / scale, atol=1e-6)


def test_assemble_volume_from_slices():
    import run_mcinr as RM
    sl = np.zeros((3, 5, 4), np.complex64); sl[1] = 1
    v = RM.assemble(sl, [10, 11, 12], nx=16)
    assert v.shape == (16, 5, 4) and v[11].real.sum() == 20 and v[0].sum() == 0
```

- [ ] **Step 3: 跑测试确认失败**

Run: `cd $WT/ce_retro_cssense_af16/scripts && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_mcinr_units.py -q`
Expected: FAIL，`No module named 'run_mcinr'`

- [ ] **Step 4: 实现 run_mcinr.py**

```python
#!/usr/bin/env python3
"""run_mcinr.py -- MC-INR (Niessen 2025 reproduction, /data0/congcong/code/GS/mcinr_niessen2025) on the synthetic pair.

Runs scripts/run_volume.py of that repository under its own environment (GR_pt1.10, tinycudann) with this topic's
files: pre in the T1 slot, post in the FLAIR slot, one AF16 mask for both, each scan's own maps. Then converts the
network images (paper units: data / M * 100) to this topic's gauge (x * M/100 / scale, no fit to the truth) and scores
them with t1t2_common.Pair as method 'mcinr'.
  python scripts/run_mcinr.py --gpu 0                 # all 256 readout slices (~15-25 min) -> recon/mcinr_<scan>_complex.npy
  python scripts/run_mcinr.py --gpu 0 --slices 120:122  # smoke: writes recon/mcinr_run_smoke/, scores nothing
  python scripts/run_mcinr.py --gpu 0 --score_only    # convert + score an existing full run
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1t2_common import Pair, R, SCANS   # noqa: E402

MCINR = Path("/data0/congcong/code/GS/mcinr_niessen2025")
PY_TCNN = "/home/congcongliu/anaconda3/envs/GR_pt1.10/bin/python"
SLOT = {"pre": "T1", "post": "FLAIR"}
NX = 256


def to_gauge(x_norm, M, scale):
    return x_norm * (M / 100.0) / scale


def assemble(slices, xs, nx=NX):
    v = np.zeros((nx,) + tuple(slices.shape[1:]), np.complex64)
    v[np.asarray(xs)] = slices
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--slices", default="all")
    ap.add_argument("--score_only", action="store_true")
    a = ap.parse_args()
    smoke = a.slices != "all"
    out = R / ("recon/mcinr_run_smoke" if smoke else "recon/mcinr_run")
    if not a.score_only:
        cmd = [PY_TCNN, "-u", str(MCINR / "scripts/run_volume.py"), "--af", "16", "--gpu", str(a.gpu), "--slices", a.slices, "--resume",
               "--ksp_T1", str(R / "sim/gs/pre_kspace.h5"), "--ksp_FLAIR", str(R / "sim/gs/post_kspace.h5"),
               "--csm_T1", str(R / "sim/gs/pre_csm.h5"), "--csm_FLAIR", str(R / "sim/gs/post_csm.h5"),
               "--mask_T1", str(R / "sim/mask/mask_af16_3d.h5"), "--mask_FLAIR", str(R / "sim/mask/mask_af16_3d.h5"),
               "--out", str(out)]
        env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID")
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env=env)
    if smoke:
        print("smoke run done, nothing scored"); return
    meta = json.load(open(out / "meta.json"))
    assert meta["slices"] == "all", "score needs a full run"
    xs = list(range(NX))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""                       # scoring on the CPU in double precision
    for scan in SCANS:
        P = Pair(scan, torch.device("cpu"))
        P.gt_c = P.gt_c.to(torch.complex128); P.gt = P.gt.to(torch.float64)
        x_norm = torch.from_numpy(np.load(out / f"mcinr_norm_{SLOT[scan]}_complex.npy"))
        x = to_gauge(x_norm.to(torch.complex128), meta["M_acquired"], P.scale)
        assert tuple(x.shape) == P.G, (x.shape, P.G)
        row = P.score(x, "mcinr", seconds=meta["seconds_loop"])
        np.save(R / f"recon/mcinr_{scan}_complex.npy", x.numpy().astype(np.complex64))
        json.dump({"contrast": scan, "rows": [row], "run_meta": str(out / "meta.json"), "M_acquired": meta["M_acquired"],
                   "gauge_scale": P.scale}, open(R / f"logs/mcinr_{scan}.json", "w"), indent=2)
        print(f"[{scan}] mcinr head_psnr={row['head_psnr']:.3f} head_ssim={row['head_ssim']:.4f} head_nmse={row['head_nmse']:.5f}", flush=True)


if __name__ == "__main__":
    main()
```
注意 `M_acquired` 在 mcinr 的 meta 里是 acquired 峰值（含噪、掩膜内），run_volume 自己会断言它等于 full 峰值；含噪的合成数据两者相等（同一数组），无需处理。

- [ ] **Step 5: 跑测试确认通过**

Run: `cd $WT/ce_retro_cssense_af16/scripts && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_mcinr_units.py -q`
Expected: `2 passed`

- [ ] **Step 6: 真数据冒烟（2 层，约 2 分钟）**

Run: `cd $TOPIC && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/run_mcinr.py --gpu <G> --slices 120:122`
Expected: mcinr 日志 `AF16: samples T1 2552/40832 FLAIR 2552/40832`，两层各打印 loss 与秒数，`recon/mcinr_run_smoke/meta.json` 生成，末行 `smoke run done`。之后 `rm -r $TOPIC/recon/mcinr_run_smoke`。

- [ ] **Step 7: Commit**

```bash
cd $WT && git add ce_retro_cssense_af16/scripts/run_mcinr.py ce_retro_cssense_af16/scripts/tests/test_mcinr_units.py && git -c user.name="Congcong Liu" -c user.email="sober235@users.noreply.github.com" commit -m "ce retro af16: MC-INR arm through the Niessen reproduction, converted to the topic gauge"
```

---

### Task 11: MC-GS / GS-single（三期冻结配方，训练器 e778bf2）

**Files:**
- Create: `configs/ce_retro_af16.yaml`、`configs/ce_retro_af16_single_pre.yaml`、`configs/ce_retro_af16_single_post.yaml`
- Create: `scripts/run_gs.sh`
- Create: `scripts/extract_gs_volumes.py`
- Test: `scripts/tests/test_extract_gs.py`

**Interfaces:**
- Consumes: `TOPIC/trainer/_trainer_e778bf2/arc/cycle.py`（Task 3）；`sim/gs/*.h5`、`sim/mask/mask_af16_3d.h5`。
- Produces: `trainer/_trainer_e778bf2/results_arc/<EXP>{.log,_score.json,_volumes.npz}`；`recon/gs_joint_{pre,post}_{complex,mag}.npy`、`recon/gs_single_{pre,post}_{complex,mag}.npy`；`logs/gs_joint_<scan>.json`、`logs/gs_single_<scan>.json`（rows 由本课题 `Pair.score` 算，键 `exp`、`factor_scale`）。CLI：`run_gs.sh EXP GPU ITERS [extra]`（环境变量 `SINGLE=pre|post`，`WARMUP`）；`extract_gs_volumes.py --exp EXP --key gs_joint --map pre:plain,post:contrast` / `--key gs_single --map pre:plain`。函数 `parse_scales(log_text) -> list[float]`、`convert(rec_complex, factor) -> np.ndarray`。

- [ ] **Step 1: 写三个配置**

`configs/ce_retro_af16.yaml`（从 `/data0/congcong/code/GS/mcgs_af16_p3_wt/configs/t1flair_v3_af16_jyz.yaml` 派生：只换 data 段，其余段逐字拷贝）:
```yaml
# ce_retro_af16.yaml -- MC-GS on the synthetic pre/post pair (CS-SENSE truth, Poisson-disc AF16, 42 real coils).
# data: the trainer reads the noisy UNMASKED k-space (sim/gs/*_kspace.h5), the real maps and the external 3-D mask,
# builds its own reference from the full k-space (presampled: false) and masks -- the jyz retrospective path.
# No known_zero anywhere: the synthetic data has no vendor shutter. Everything below data: is the jyz AF16 base
# config verbatim; the phase-3 recipe is applied on the command line by scripts/run_gs.sh.
data:
  plain:
    name: pre
    acceleration_factor: 16
    data_path: /data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16/sim/gs/pre_kspace.h5
    csm_path:  /data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16/sim/gs/pre_csm.h5
    mask_path: /data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16/sim/mask/mask_af16_3d.h5
    mask_type: external
    normalization_percentile: 99.9
    normalize_kspace: true
    phase_axes: [1, 2]
    readout_axis: 0
    presampled: false
    skip_readout_check: false
  contrast:
    name: post
    acceleration_factor: 16
    data_path: /data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16/sim/gs/post_kspace.h5
    csm_path:  /data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16/sim/gs/post_csm.h5
    mask_path: /data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16/sim/mask/mask_af16_3d.h5
    mask_type: external
    normalization_percentile: 99.9
    normalize_kspace: true
    phase_axes: [1, 2]
    readout_axis: 0
    presampled: false
    skip_readout_check: false
gaussian:
  density_lr: 0.015
  init_scale: 0.008
  initial_num_points: 600000
  position_lr: 0.0003
  rotation_lr: 0.0003
  scale_lr: 0.002
hardware:
  device: cuda:0
loss:
  loss_type: l2
  tv_weight: 0.10
  tv_eps: 0.04
  kspace_freq_adaptive: true
  kspace_center_fraction: 0.05
  kspace_center_weight: 0.3
  kspace_mid_weight: 1.0
  kspace_high_weight: 2.2
  lambda_diff: 0.01
  lambda_enh: 0
  edge_tv: true
  edge_tv_kappa_q: 0.85
  edge_tv_wfloor: 0.1
mode:
  experimental_reproduction: false
  paper_faithful: false
  self_supervised_deploy: false
output:
  exp_name: ce_retro_af16_joint
  output_dir: results_arc/ce_retro_af16_joint
training:
  max_iterations: 4000
  shared_warmup_iters: 500
  eval_every: 1000
  log_every: 50
  save_every: 1000
  seed: 42
  max_grad_norm: 1.0
  lr_scheduler:
    type: exponential
    gamma: 0.9995
voxelizer:
  max_radius: 20
  tile_size: 8
```
`configs/ce_retro_af16_single_pre.yaml`：同上，但 `contrast:` 段的 `name: pre` 且三个路径全指向 pre 的文件（两通道装同一扫描，jyz 的 GS-single 定义）；`output.exp_name: ce_retro_af16_single_pre`。`_single_post.yaml`：`plain:` 与 `contrast:` 都指向 post。用下面的命令生成而不是手抄：
```bash
C=$WT/ce_retro_cssense_af16/configs
sed -e '/^  contrast:/,/^gaussian:/{s/name: post/name: pre/;s#/post_kspace.h5#/pre_kspace.h5#;s#/post_csm.h5#/pre_csm.h5#}' -e 's/ce_retro_af16_joint/ce_retro_af16_single_pre/' $C/ce_retro_af16.yaml > $C/ce_retro_af16_single_pre.yaml
sed -e '/^  plain:/,/^  contrast:/{s/name: pre/name: post/;s#/pre_kspace.h5#/post_kspace.h5#;s#/pre_csm.h5#/post_csm.h5#}' -e 's/ce_retro_af16_joint/ce_retro_af16_single_post/' $C/ce_retro_af16.yaml > $C/ce_retro_af16_single_post.yaml
grep -c "pre_kspace" $C/ce_retro_af16_single_pre.yaml; grep -c "post_kspace" $C/ce_retro_af16_single_post.yaml; grep -n "name:" $C/*.yaml
```
Expected: `2`、`2`；single_pre 两个 `name: pre`，single_post 两个 `name: post`，joint 一个 pre 一个 post。

- [ ] **Step 2: 写 run_gs.sh**

```bash
#!/usr/bin/env bash
# run_gs.sh EXP GPU ITERS [extra arc/cycle.py args]
# The jyz AF16 phase-3 frozen recipe on the pinned trainer snapshot (e778bf2). SINGLE=pre|post selects the
# single-scan config (both channels = that scan, the GS-single arm); unset = the joint config (MC-GS).
# WARMUP=20 for smoke runs. Outputs: trainer/_trainer_e778bf2/results_arc/EXP{.log,_score.json,_volumes.npz}.
set -uo pipefail
if [[ $# -lt 3 ]]; then echo "usage: $0 EXP GPU ITERS [extra cycle.py args...]" >&2; exit 2; fi
EXP=$1; GPU=$2; ITERS=$3; shift 3
T=${RETRO_TOPIC:-/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16}
TR=$T/trainer/_trainer_e778bf2
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
WARMUP=${WARMUP:-500}
case "${SINGLE:-joint}" in
  joint) CFG=$T/configs/ce_retro_af16.yaml ;;
  pre)   CFG=$T/configs/ce_retro_af16_single_pre.yaml ;;
  post)  CFG=$T/configs/ce_retro_af16_single_post.yaml ;;
  *)     echo "SINGLE must be pre, post or unset: ${SINGLE}" >&2; exit 2 ;;
esac
mkdir -p "$TR/results_arc"
LOG=$TR/results_arc/$EXP.log
for p in "$LOG" "$TR/results_arc/${EXP}_score.json" "$TR/results_arc/$EXP"; do
  if [[ -e "$p" ]]; then echo "refusing to overwrite existing output: $p" >&2; exit 1; fi
done
cd "$TR"
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID \
  "$PY" -u arc/cycle.py --exp "$EXP" --gpu "$GPU" --iters "$ITERS" \
  --lambda_geo 0.10 --warmup "$WARMUP" --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 \
  --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn \
  --set loss.edge_tv_kappa_source=zf \
  --set gaussian.initial_num_points=3600000 \
  --set loss.wav_name=db4 --set loss.wav_levels=1 \
  --set loss.lambda_wav_plain=0.12 --set loss.lambda_wav=0.10 \
  --lambda_edge 0.001 --lambda_ptv 0.03 \
  --base_cfg "$CFG" --save_volumes "$@" > "$LOG" 2>&1
RC=$?
echo "rc=$RC log=$LOG"
exit $RC
```
`chmod +x scripts/run_gs.sh`。

- [ ] **Step 3: 写失败的测试（日志解析与单位换算）**

`scripts/tests/test_extract_gs.py`:
```python
import numpy as np


def test_parse_scales_in_channel_order():
    import extract_gs_volumes as X
    log = "x\nNormalized MRI intensity scale by P99.9 magnitude = 1.2500e+02 (before)\nfoo\nNormalized MRI intensity scale by P99.9 magnitude = 130.0\n"
    assert X.parse_scales(log) == [125.0, 130.0]


def test_convert_scales_complex_volume():
    import extract_gs_volumes as X
    v = np.array([[1 + 1j]], np.complex64)
    out = X.convert(v, 2.0)
    assert out.dtype == np.complex64 and out[0, 0] == 2 + 2j
```

- [ ] **Step 4: 跑测试确认失败**

Run: `cd $WT/ce_retro_cssense_af16/scripts && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_extract_gs.py -q`
Expected: FAIL，`No module named 'extract_gs_volumes'`

- [ ] **Step 5: 实现 extract_gs_volumes.py**

```python
#!/usr/bin/env python3
"""extract_gs_volumes.py -- pull an MC-GS run's volumes out of cycle.py's npz, put them in this topic's gauge, score.

cycle.py normalises each channel's data by P99.9 of ITS OWN reference (a SENSE adjoint of the noisy full k-space), so
its volumes are in trainer units; this topic's gauge divides by P99.9 of the CS-SENSE truth. The conversion is
x_gauge = x_trainer * scale_trainer / scale_gauge, with scale_trainer parsed from the run log (one line per channel,
plain first) and cross-checked against the zero-filled volumes: the npz carries the trainer's ZF of the same data,
Pair.sense_adj(Pair.y) is ours, and their norm ratio must equal the factor (<1e-3) or the run is not on our data.
npz keys are fixed by channel: recT1/recT1_complex/zfT1_complex = plain, recFL/recFL_complex/zfFL_complex = contrast.
  extract_gs_volumes.py --exp ce_retro_af16_joint --key gs_joint --map pre:plain,post:contrast
  extract_gs_volumes.py --exp ce_retro_af16_single_pre --key gs_single --map pre:plain
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1t2_common import Pair, R, TRAINER   # noqa: E402

NPZ_C = {"plain": "recT1_complex", "contrast": "recFL_complex"}
NPZ_ZF = {"plain": "zfT1_complex", "contrast": "zfFL_complex"}
CH_IDX = {"plain": 0, "contrast": 1}
PAT = re.compile(r"Normalized MRI intensity scale by P99\.9 magnitude = ([0-9.eE+-]+)")


def parse_scales(text):
    return [float(v) for v in PAT.findall(text)]


def convert(rec_complex, factor):
    return (np.asarray(rec_complex) * np.float32(factor)).astype(np.complex64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True); ap.add_argument("--key", required=True)
    ap.add_argument("--map", default="pre:plain,post:contrast"); ap.add_argument("--gpu", type=int, default=-1)
    a = ap.parse_args()
    arc = TRAINER / "results_arc"
    m = dict(kv.split(":") for kv in a.map.split(","))
    score_json = json.load(open(arc / f"{a.exp}_score.json"))
    for scan, ch in m.items():
        assert score_json.get(f"{ch}_name") == scan, (ch, score_json.get(f"{ch}_name"), scan)
    scales = parse_scales(open(arc / f"{a.exp}.log").read())
    assert len(scales) >= 2, f"expected two normalisation lines in {a.exp}.log, found {scales}"
    dev = torch.device("cpu") if a.gpu < 0 else torch.device(f"cuda:{a.gpu}")
    with np.load(arc / f"{a.exp}_volumes.npz") as z:
        for scan, ch in m.items():
            P = Pair(scan, dev)
            factor = scales[CH_IDX[ch]] / P.scale
            zf_tr = torch.from_numpy(np.asarray(z[NPZ_ZF[ch]])).to(torch.complex64)
            zf_us = P.sense_adj(P.y).cpu()
            ratio = float(zf_us.abs().norm() / zf_tr.abs().norm())
            if abs(ratio / factor - 1) > 1e-3:
                raise SystemExit(f"[{scan}] ZF norm ratio {ratio:.6f} != scale factor {factor:.6f}: the run is not on this topic's data or the FFT convention differs")
            x = convert(z[NPZ_C[ch]], factor)
            np.save(R / f"recon/{a.key}_{scan}_complex.npy", x)
            np.save(R / f"recon/{a.key}_{scan}_mag.npy", np.abs(x).astype(np.float32))
            row = P.score(torch.from_numpy(x).to(dev), a.key, seconds=float(score_json.get("elapsed_s", 0)), exp=a.exp, factor_scale=factor)
            json.dump({"contrast": scan, "rows": [row], "exp": a.exp, "npz": str(arc / f"{a.exp}_volumes.npz"),
                       "scale_trainer": scales[CH_IDX[ch]], "scale_gauge": P.scale, "zf_ratio_check": ratio,
                       "vram_gb": score_json.get("vram_gb"), "npts": score_json.get("npts")},
                      open(R / f"logs/{a.key}_{scan}.json", "w"), indent=2)
            print(f"[{scan}] {a.key} <- {a.exp}:{ch}  factor {factor:.5f} (zf ratio {ratio:.5f})  head_psnr={row['head_psnr']:.3f} head_ssim={row['head_ssim']:.4f}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 跑测试确认通过**

Run: `cd $WT/ce_retro_cssense_af16/scripts && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_extract_gs.py -q`
Expected: `2 passed`

- [ ] **Step 7: 真数据冒烟：50 步、看显存、走通抽取（门 4/5）**

```bash
cd $TOPIC && WARMUP=20 scripts/run_gs.sh smoke_joint <G> 50
grep -E "Normalized MRI intensity scale|mask|Loading external|points|vram|SCORE|Traceback|Error" trainer/_trainer_e778bf2/results_arc/smoke_joint.log | head -40
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/extract_gs_volumes.py --exp smoke_joint --key smoke_gs --map pre:plain,post:contrast
```
Expected: 日志里两条 `Normalized MRI intensity scale by P99.9 magnitude = …`（plain 先）、掩膜加载报 2552 点或 6.25 % 采样、无 Traceback；`SCORE … vram_gb=…`（记下峰值显存，写进 STATUS）；抽取脚本打印 `factor` 与 `zf ratio` 一致，两个 `logs/smoke_gs_*.json` 写出。若 OOM：先 `--set voxelizer.tile_size=4` 重试，仍 OOM 则停下汇报用户（spec 第 5 节的约定）。冒烟完成后删 `recon/smoke_gs_*`、`logs/smoke_gs_*`（trainer/results_arc 里的 smoke 产物可留）。

- [ ] **Step 8: Commit**

```bash
cd $WT && git add ce_retro_cssense_af16/configs ce_retro_cssense_af16/scripts/run_gs.sh ce_retro_cssense_af16/scripts/extract_gs_volumes.py ce_retro_cssense_af16/scripts/tests/test_extract_gs.py && git -c user.name="Congcong Liu" -c user.email="sober235@users.noreply.github.com" commit -m "ce retro af16: MC-GS and GS-single on the pinned trainer with the jyz phase-3 recipe; volumes rescaled into the topic gauge"
```

---

### Task 12: 评分与汇总：`arms.py`、`score_volume.py`、`summarize.py`、`enhancement_fidelity.py`、最优层脚本

**Files:**
- Create: `scripts/arms.py`、`scripts/score_volume.py`、`scripts/summarize.py`、`scripts/enhancement_fidelity.py`
- Copy+edit: jyz `find_best_slice.py`、`best_slice_mcgs.py` → `scripts/`
- Test: `scripts/tests/test_summary.py`

**Interfaces:**
- Consumes: `logs/classical_<scan>[_o200_lam*].json`、`logs/jcssense_o200_lam*.json`、`logs/spirit_<scan>_ks*_lam*.json`、`logs/{zsssdu,dip,inr,mcinr,gs_single,gs_joint}_<scan>.json`。
- Produces: `logs/summary.json`（list of rows：`contrast method psnr ssim nmse hfen head_psnr head_ssim head_nmse head_hfen psnr_cplx sharp grain grain_ratio seconds [lam ksize n_lambda lams]`）、`logs/summary.tsv`；`arms.arm_files(scan) -> list[(key,label,path)]`（11 臂固定顺序 zf cgsense cssense_o200 jcssense spirit zsssdu dip inr mcinr gs_single gs_joint；`arms.LABEL` 与 `arms.ORDER`）；`logs/enhancement_fidelity.json`；`logs/slice_scores.json`、`logs/best_slice_mcgs.json`；`score_volume.py --scan s --file recon/x.npy --method m [--gpu -1]` 打印并可 `--check logs/<stem>.json` 与记录值比对（门 3）。

- [ ] **Step 1: 写失败的测试**

`scripts/tests/test_summary.py`:
```python
import json


def _row(c, m, hp, **kw):
    r = dict(contrast=c, method=m, psnr=30.0, ssim=0.9, nmse=0.01, hfen=0.3, head_psnr=hp, head_ssim=0.9, head_nmse=0.01,
             head_hfen=0.3, psnr_cplx=28.0, sharp=1.0, grain=0.01, grain_ratio=1.0, seconds=1.0)
    r.update(kw); return r


def test_summarize_picks_best_lambda_per_scan_and_orders_arms(tiny_topic):
    import summarize as S, arms as A
    top = tiny_topic["root"]; L = top / "logs"
    json.dump({"contrast": "pre", "rows": [_row("pre", "zf", 20.0), _row("pre", "cgsense", 25.0)]}, open(L / "classical_pre.json", "w"))
    for lam, hp in ((0.0005, 30.0), (0.001, 31.5), (0.002, 31.0)):
        json.dump({"contrast": "pre", "rows": [_row("pre", f"cssense_o200_lam{lam:g}", hp, lam=lam)]}, open(L / f"classical_pre_o200_lam{lam:g}.json", "w"))
    json.dump({"contrasts": ["pre", "post"], "lam": 0.001, "rows": [_row("pre", "jcssense_o200_lam0.001", 30.5, lam=0.001), _row("post", "jcssense_o200_lam0.001", 29.0, lam=0.001)]},
              open(L / "jcssense_o200_lam0.001.json", "w"))
    json.dump({"contrast": "pre", "rows": [_row("pre", "spirit_ks5_lam0.001", 29.5, lam=0.001, ksize=5)]}, open(L / "spirit_pre_ks5_lam0.001.json", "w"))
    json.dump({"contrast": "pre", "rows": [_row("pre", "gs_joint", 33.0)]}, open(L / "gs_joint_pre.json", "w"))
    rows = S.summarize()
    pre = {r["method"]: r for r in rows if r["contrast"] == "pre"}
    assert pre["cssense_o200"]["lam"] == 0.001 and pre["cssense_o200"]["n_lambda"] == 3
    assert pre["jcssense"]["head_psnr"] == 30.5 and pre["spirit"]["ksize"] == 5
    order = [r["method"] for r in rows if r["contrast"] == "pre"]
    assert order == [m for m in A.ORDER if m in order]
    assert (top / "logs/summary.json").exists() and (top / "logs/summary.tsv").exists()
    files = dict((k, p) for k, _, p in A.arm_files("pre"))
    assert files["cssense_o200"].name == "cssense_o200_lam0.001_pre_complex.npy"
    assert files["spirit"].name == "spirit_ks5_lam0.001_pre_complex.npy" and files["gs_joint"].name == "gs_joint_pre_complex.npy"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd $WT/ce_retro_cssense_af16/scripts && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_summary.py -q`
Expected: FAIL，`No module named 'summarize'`

- [ ] **Step 3: 实现 arms.py 与 summarize.py**

`scripts/arms.py`:
```python
"""arms.py -- the eleven arms of this topic in figure order, and where each scan's registered volume lives.
Swept families (CS-SENSE o200, joint CS-SENSE, L1-SPIRiT) resolve to the lambda / kernel kept in logs/summary.json."""
import json
import os
from pathlib import Path

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
ORDER = ["zf", "cgsense", "cssense_o200", "jcssense", "spirit", "zsssdu", "dip", "inr", "mcinr", "gs_single", "gs_joint"]
LABEL = {"zf": "ZF", "cgsense": "CG-SENSE", "cssense_o200": "CS-SENSE", "jcssense": "Joint SENSE", "spirit": "L1-SPIRiT",
         "zsssdu": "ZS-SSDU", "dip": "3D-DIP", "inr": "3D-INR", "mcinr": "MC-INR", "gs_single": "GS-single", "gs_joint": "MC-GS"}


def arm_files(scan, summary=None):
    rows = summary if summary is not None else json.load(open(R / "logs/summary.json"))
    sel = {r["method"]: r for r in rows if r["contrast"] == scan}
    out = []
    for key in ORDER:
        if key == "cssense_o200" and key in sel:
            p = R / f"recon/cssense_o200_lam{sel[key]['lam']:g}_{scan}_complex.npy"
        elif key == "jcssense" and key in sel:
            p = R / f"recon/jcssense_o200_lam{sel[key]['lam']:g}_{scan}_complex.npy"
        elif key == "spirit" and key in sel:
            p = R / f"recon/spirit_ks{sel[key]['ksize']}_lam{sel[key]['lam']:g}_{scan}_complex.npy"
        else:
            p = R / f"recon/{key}_{scan}_complex.npy"
        out.append((key, LABEL[key], p))
    return out
```

`scripts/summarize.py`:
```python
#!/usr/bin/env python3
"""summarize.py -- one table of every arm on both scans, from the per-arm score jsons (all written by t1t2_common.Pair.score).
Swept families keep the lambda (and kernel) with the highest head-only PSNR per scan, the jyz protocol.
  python scripts/summarize.py   -> logs/summary.json, logs/summary.tsv, a printout per scan"""
import json
import os
from pathlib import Path

from arms import LABEL, ORDER

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
COLS = ["psnr", "ssim", "nmse", "hfen", "head_psnr", "head_ssim", "head_nmse", "head_hfen", "psnr_cplx", "sharp", "grain", "grain_ratio"]
STEMS = ["classical", "zsssdu", "dip", "inr", "mcinr", "gs_single", "gs_joint"]
FAM = [("cssense_o200", "logs/classical_{c}_o200_lam*.json"), ("jcssense", "logs/jcssense_o200_lam*.json"), ("spirit", "logs/spirit_{c}_ks*_lam*.json")]
SCANS = ("pre", "post")


def summarize():
    rows = []
    for c in SCANS:
        for stem in STEMS:
            p = R / f"logs/{stem}_{c}.json"
            if p.exists():
                for r in json.load(open(p)).get("rows", []):
                    if r.get("contrast", c) == c:
                        rows.append({"contrast": c, "method": r["method"], **{k: r.get(k) for k in COLS}, "seconds": r.get("seconds")})
        for key, pat in FAM:
            cands = []
            for f in sorted(R.glob(pat.format(c=c))):
                for r in json.load(open(f)).get("rows", []):
                    if r.get("contrast", c) == c and r.get("head_psnr") is not None:
                        cands.append(r)
            if cands:
                b = max(cands, key=lambda r: r["head_psnr"])
                rows.append({"contrast": c, "method": key, **{k: b.get(k) for k in COLS}, "seconds": b.get("seconds"),
                             "lam": b.get("lam"), "n_lambda": len(cands), **({"ksize": b["ksize"]} if b.get("ksize") else {}),
                             "lams": sorted({float(r["lam"]) for r in cands if r.get("lam") is not None})})
    rows.sort(key=lambda r: (SCANS.index(r["contrast"]), ORDER.index(r["method"]) if r["method"] in ORDER else 99))
    hdr = ["contrast", "method"] + COLS
    lines = ["\t".join(hdr)]
    for r in rows:
        lines.append("\t".join([r["contrast"], LABEL.get(r["method"], r["method"])] + ["" if r.get(k) is None else f"{r[k]:.5f}" for k in COLS]))
    (R / "logs/summary.tsv").write_text("\n".join(lines) + "\n")
    json.dump(rows, open(R / "logs/summary.json", "w"), indent=2)
    return rows


def main():
    rows = summarize()
    for c in SCANS:
        print(f"\n=== {c}  (CS-SENSE truth, Poisson-disc AF16)")
        print(f"{'method':12s} {'headPSNR':>9s} {'headSSIM':>9s} {'headNMSE':>9s} {'headHFEN':>9s} {'PSNRcplx':>9s} {'PSNR':>8s} {'lam':>8s}")
        for r in [r for r in rows if r["contrast"] == c]:
            f = lambda k, w, p: (f"{r[k]:{w}.{p}f}" if isinstance(r.get(k), float) else " " * w)
            print(f"{LABEL.get(r['method'], r['method']):12s} {f('head_psnr',9,3)} {f('head_ssim',9,4)} {f('head_nmse',9,5)} {f('head_hfen',9,4)} {f('psnr_cplx',9,3)} {f('psnr',8,3)} {f('lam',8,5)}")
    print(f"\nwrote {R / 'logs/summary.json'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd $WT/ce_retro_cssense_af16/scripts && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_summary.py -q`
Expected: `1 passed`

- [ ] **Step 5: 写 score_volume.py（口径互验工具）**

```python
#!/usr/bin/env python3
"""score_volume.py -- re-score any saved volume through t1t2_common.Pair (the one gauge) and, with --check, compare against
the row an arm wrote itself. Every arm here scores through the same Pair.score, so a mismatch means a volume was written in
the wrong units or layout (the GS and MC-INR conversions are the places that can go wrong).
  python scripts/score_volume.py --scan pre --file recon/gs_joint_pre_complex.npy --method gs_joint --check logs/gs_joint_pre.json"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--scan", required=True, choices=["pre", "post"]); ap.add_argument("--file", required=True)
ap.add_argument("--method", required=True); ap.add_argument("--check", default=None); ap.add_argument("--gpu", type=int, default=-1)
ap.add_argument("--tol", type=float, default=2e-4)
a = ap.parse_args()
if a.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1t2_common import Pair, R   # noqa: E402

dev = torch.device("cpu") if a.gpu < 0 else torch.device("cuda:0")
P = Pair(a.scan, dev)
if a.gpu < 0:
    P.gt_c = P.gt_c.to(torch.complex128); P.gt = P.gt.to(torch.float64)     # CPU float32 norms drift ~1e-3 on 10M voxels
v = np.load(R / a.file if not Path(a.file).is_absolute() else a.file)
x = torch.from_numpy(v if np.iscomplexobj(v) else v.astype(np.complex64)).to(dev)
if a.gpu < 0:
    x = x.to(torch.complex128)
got = P.score(x, a.method)
print(json.dumps({k: got[k] for k in ("head_psnr", "head_ssim", "head_nmse", "head_hfen", "psnr_cplx", "psnr", "ssim", "nmse")}, indent=1))
if a.check:
    ref = next(r for r in json.load(open(R / a.check))["rows"] if r["method"] == a.method)
    bad = [k for k in ("psnr", "ssim", "nmse", "hfen", "head_psnr", "head_ssim", "head_nmse", "head_hfen") if abs(got[k] - ref[k]) > a.tol * max(1.0, abs(ref[k]))]
    for k in ("head_psnr", "head_ssim", "head_nmse", "head_hfen", "psnr"):
        print(f"  {'MISMATCH' if k in bad else 'OK      '} {k:10s} recorded={ref[k]:.6f} rescored={got[k]:.6f}")
    if bad:
        raise SystemExit(f"GAUGE MISMATCH on {bad}")
    print("GAUGE OK")
```

- [ ] **Step 6: 写 enhancement_fidelity.py**

```python
#!/usr/bin/env python3
"""enhancement_fidelity.py -- what each arm does to the enhancement, against the truth.
post - pre in white-matter units (each scan divided by its own white-matter median) averaged over the six reference regions
the prospective study defined on CS-SENSE (enhance_rois_cssense.npz: 病灶核心 / 核心外 0-3 mm / 正常白质 / 正常皮层 / 血管/静脉窦 /
脉络丛), plus the extracranial background of the post image (mean outside the head mask dilated by 6 voxels = 6 mm).
The truth row is the CS-SENSE pair itself; err_* = arm - truth.   -> logs/enhancement_fidelity.json"""
import json, os
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from arms import arm_files, LABEL

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
ROI_NPZ = Path(os.environ.get("RETRO_ROI", "/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/lesion_candidates/enhance_rois_cssense.npz"))
KEYS = {"病灶核心": "lesion_core", "核心外 0-3 mm": "rim_0_3mm", "正常白质": "white_matter", "正常皮层": "cortex", "血管/静脉窦": "vessels", "脉络丛": "plexus"}


def wm_units(v, wm):
    return v / np.median(v[wm])


def main():
    with np.load(ROI_NPZ) as r:
        names = [str(n) for n in r["names"]]; roi = {KEYS[n]: r[f"roi{i}"] for i, n in enumerate(names)}
    wm = roi["white_matter"]
    with np.load(R / "sim/gt.npz") as g:
        gt = {s: np.abs(g[s]).astype(np.float32) for s in ("pre", "post")}
    p999 = float(np.quantile(gt["post"], 0.999))
    head = ndi.binary_dilation(gt["post"] > 0.05 * p999, iterations=6)
    vols = {"truth": gt}
    for key, label, p in arm_files("pre"):
        q = arm_files("post"); pp = dict((k, f) for k, _, f in q)[key]
        if p.exists() and pp.exists():
            vols[key] = {"pre": np.abs(np.load(p)).astype(np.float32), "post": np.abs(np.load(pp)).astype(np.float32)}
    rows = []
    for key, v in vols.items():
        d = wm_units(v["post"], wm) - wm_units(v["pre"], wm)
        row = {"method": key, "label": "truth (CS-SENSE)" if key == "truth" else LABEL[key],
               **{k: float(d[m].mean()) for k, m in roi.items()}, "background_post": float(wm_units(v["post"], wm)[~head].mean())}
        rows.append(row)
    truth = rows[0]
    for row in rows[1:]:
        for k in list(roi) + ["background_post"]:
            row["err_" + k] = row[k] - truth[k]
    json.dump({"units": "white-matter units (each scan / its own WM median); post - pre", "rows": rows}, open(R / "logs/enhancement_fidelity.json", "w"), indent=1)
    print(f"{'arm':18s}{'vessels':>9s}{'plexus':>9s}{'cortex':>9s}{'WM':>8s}{'core':>8s}{'rim':>8s}{'bg':>8s}")
    for r in rows:
        print(f"{r['label']:18s}{r['vessels']:>+9.3f}{r['plexus']:>+9.3f}{r['cortex']:>+9.3f}{r['white_matter']:>+8.3f}{r['lesion_core']:>+8.3f}{r['rim_0_3mm']:>+8.3f}{r['background_post']:>8.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: 移植最优层脚本**

```bash
J=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/scripts; D=$WT/ce_retro_cssense_af16/scripts
cp $J/find_best_slice.py $J/best_slice_mcgs.py $D/
sed -i 's#^R = Path(.*#R = Path(__import__("os").environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))#' $D/find_best_slice.py $D/best_slice_mcgs.py
sed -i 's/("T1", "FLAIR")/("pre", "post")/g; s/"gs_joint_ar"/"gs_joint"/g; s/"gs_single_ar"/"gs_single"/g' $D/find_best_slice.py $D/best_slice_mcgs.py
grep -n "T1\|FLAIR\|l1espirit\|espirit" $D/find_best_slice.py $D/best_slice_mcgs.py
```
Expected: 余下的 T1/FLAIR 只在注释/docstring 里；若 `arms(c)` 列表里有 L1-ESPIRiT 行（`l1espirit`），删掉那一行，并确认列表里 CS-SENSE / joint / spirit 的文件名规则与 `arms.py` 一致（`lam_file` 用 `recon/<prefix>_lam*_<c>_complex.npy` 通配，与我们的命名相同）。把 `best_slice_mcgs.py` 里的 `("MC-GS", …_mag.npy)` 和 `("GS-single", …_mag.npy)` 保留（Task 11 写了 `_mag.npy`）。

- [ ] **Step 8: 全部测试 + Commit**

Run: `cd $WT/ce_retro_cssense_af16 && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest scripts/tests -q`
Expected: 全部 passed（此时 test_layout 1 + mask 3 + deps 1 + gauge 3 + sim 2 + classical 3 + mcinr 2 + extract 2 + summary 1 = 18）。
```bash
cd $WT && git add ce_retro_cssense_af16/scripts && git -c user.name="Congcong Liu" -c user.email="sober235@users.noreply.github.com" commit -m "ce retro af16: arm registry, summary, gauge re-check, enhancement fidelity, best-slice scripts"
```

---

### Task 13: 调度器 `retro_orchestrate.py`（计划 retro16）

**Files:**
- Create: `scripts/retro_orchestrate.py`（骨架取自 `/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/scripts/ce_orchestrate.py` 的 `log/alive/add/my_busy_gpus/主循环/launch` 部分）
- Test: `scripts/tests/test_orchestrate_plan.py`

**Interfaces:**
- Consumes: Tasks 6 到 12 的所有 CLI。
- Produces: `logs/orch/<job>.log`、`logs/orch/jobs/<job>.pid`、`logs/orch/ALL_DONE_retro16`；函数 `build_jobs(T) -> list[dict]`（每个 dict：`name lane prio need deps cmd(g) done()`），`plan_extensions(T, jobs) -> list[dict]`（λ 边缘外扩），`--dry` 只打印作业图不启动。

- [ ] **Step 1: 写失败的测试**

`scripts/tests/test_orchestrate_plan.py`:
```python
import json


def test_job_graph_covers_all_arms_and_extends_edge_lambda(tiny_topic):
    import retro_orchestrate as O
    top = tiny_topic["root"]
    jobs = O.build_jobs(top)
    names = {j["name"] for j in jobs}
    for n in ("cls_pre", "cls_post", "cs_pre_lam0.001", "joint_lam0.001", "spirit_post_ks3_lam0.002", "zs_pre", "dip_post", "inr_pre",
              "mcinr", "gs_single_pre", "gs_single_post", "gs_joint", "summarize"):
        assert n in names, n
    cmd = next(j for j in jobs if j["name"] == "cs_pre_lam0.001")["cmd"](3)
    assert "--admm_outer" in cmd and cmd[cmd.index("--admm_outer") + 1] == "200" and "--gpu" in cmd and cmd[cmd.index("--gpu") + 1] == "3"
    assert next(j for j in jobs if j["name"] == "summarize")["deps"]
    # edge extension: best lambda sits at the top of the grid -> one more job at 1.4x
    (top / "logs").mkdir(exist_ok=True)
    for lam in O.GRID["cs"]:
        json.dump({"rows": [dict(method=f"cssense_o200_lam{lam:g}", contrast="pre", head_psnr=30 + lam * 1e3, lam=lam)]},
                  open(top / f"logs/classical_pre_o200_lam{lam:g}.json", "w"))
    ext = O.plan_extensions(top, {j["name"]: dict(j, state="done") for j in jobs})
    assert [j["name"] for j in ext] == ["cs_pre_lam0.0028"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd $WT/ce_retro_cssense_af16/scripts && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_orchestrate_plan.py -q`
Expected: FAIL，`No module named 'retro_orchestrate'`

- [ ] **Step 3: 实现**

`scripts/retro_orchestrate.py`（骨架：把 ce_orchestrate.py 的 `log`、`alive`、`add`、`my_busy_gpus`、主 `while True` 循环和 launch 段原样搬来，删掉 `validation_ok`、`plan_family`、`plan_spirit`、`GRIDS` 文件、所有旧计划；下面给出必须新写的部分）:
```python
#!/usr/bin/env python3
"""retro_orchestrate.py -- run every arm of the retrospective AF16 study, then summarise. Plan name: retro16.
GPU lane: a card is taken when a poll finds no compute process of this user on it, utilisation < 30 % and >= need MB free;
at most MAX_GPU jobs at once, one per card. Restartable: finished work is recognised from its output files; pids in logs/orch/jobs/.
Lambda families extend one step (x1.4 / /1.4, at most 2 times per scan) when the best value sits on a grid edge.
  cd $TOPIC && setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID python scripts/retro_orchestrate.py > logs/orch/orchestrate.log 2>&1 &
  python scripts/retro_orchestrate.py --dry      # print the job graph and exit"""
import json, os, subprocess, sys, time
from pathlib import Path

PY = "/home/congcongliu/anaconda3/envs/GSTMI/bin/python"
GRID = {"cs": [2.5e-4, 3.5e-4, 5e-4, 7e-4, 1e-3, 1.4e-3, 2e-3], "joint": [5e-4, 7e-4, 1e-3, 1.4e-3, 2e-3],
        "spirit_lam": [0.0, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3], "spirit_ks": [3, 5]}
NEED = dict(cls=30000, cs=30000, joint=45000, spirit=30000, zs=40000, dip=40000, inr=40000, mcinr=20000, gs=60000)
MAX_GPU, POLL, MAX_EXT = 3, 20, 2
SCANS = ("pre", "post")


def lam_s(v):
    return f"{v:g}"


def build_jobs(T):
    T = Path(T); S = T / "scripts"
    J = []
    def job(name, kind, cmd, done, deps=(), prio=5):
        J.append(dict(name=name, lane="gpu", prio=prio, need=NEED[kind], deps=list(deps), cmd=cmd, done=done))
    for s in SCANS:
        job(f"cls_{s}", "cls", lambda g, s=s: [PY, "-u", str(S / "recon_classical.py"), "--scan", s, "--methods", "zf,cgsense", "--gpu", str(g)],
            lambda s=s: (T / f"logs/classical_{s}.json").exists(), prio=0)
        for lam in GRID["cs"]:
            job(f"cs_{s}_lam{lam_s(lam)}", "cs", lambda g, s=s, lam=lam: [PY, "-u", str(S / "recon_classical.py"), "--scan", s, "--methods", "cssense",
                "--admm_outer", "200", "--lam", lam_s(lam), "--suffix", f"_o200_lam{lam_s(lam)}", "--gpu", str(g)],
                lambda s=s, lam=lam: (T / f"logs/classical_{s}_o200_lam{lam_s(lam)}.json").exists(), prio=1)
        for ks in GRID["spirit_ks"]:
            for lam in GRID["spirit_lam"]:
                job(f"spirit_{s}_ks{ks}_lam{lam_s(lam)}", "spirit", lambda g, s=s, ks=ks, lam=lam: [PY, "-u", str(S / "recon_spirit.py"), "--scan", s, "--gpu", str(g),
                    "--ksize", str(ks), "--lam", lam_s(lam), "--suffix", f"_ks{ks}_lam{lam_s(lam)}"],
                    lambda s=s, ks=ks, lam=lam: (T / f"logs/spirit_{s}_ks{ks}_lam{lam_s(lam)}.json").exists(), prio=2)
        job(f"zs_{s}", "zs", lambda g, s=s: [PY, "-u", str(S / "recon_zsssdu.py"), "--scan", s, "--gpu", str(g), "--epochs", "100"],
            lambda s=s: (T / f"logs/zsssdu_{s}.json").exists(), prio=3)
        job(f"dip_{s}", "dip", lambda g, s=s: [PY, "-u", str(S / "recon_dip_inr.py"), "--method", "dip", "--scan", s, "--gpu", str(g), "--iters", "5000", "--lr", "3e-3", "--eval_every", "100"],
            lambda s=s: (T / f"logs/dip_{s}.json").exists(), prio=3)
        job(f"inr_{s}", "inr", lambda g, s=s: [PY, "-u", str(S / "recon_dip_inr.py"), "--method", "inr", "--scan", s, "--gpu", str(g), "--iters", "1500", "--eval_every", "50"],
            lambda s=s: (T / f"logs/inr_{s}.json").exists(), prio=3)
        job(f"gs_single_{s}", "gs", lambda g, s=s: ["env", f"SINGLE={s}", "bash", str(S / "run_gs.sh"), f"ce_retro_af16_single_{s}", str(g), "4000"],
            lambda s=s: (T / f"trainer/_trainer_e778bf2/results_arc/ce_retro_af16_single_{s}_volumes.npz").exists(), prio=4)
        job(f"extract_single_{s}", "cls", lambda g, s=s: [PY, "-u", str(S / "extract_gs_volumes.py"), "--exp", f"ce_retro_af16_single_{s}", "--key", "gs_single", "--map", f"{s}:plain", "--gpu", str(g)],
            lambda s=s: (T / f"logs/gs_single_{s}.json").exists(), deps=[f"gs_single_{s}"], prio=4)
    for lam in GRID["joint"]:
        job(f"joint_lam{lam_s(lam)}", "joint", lambda g, lam=lam: [PY, "-u", str(S / "recon_joint_cssense.py"), "--topic", str(T), "--contrasts", "pre,post",
            "--admm_outer", "200", "--lam", lam_s(lam), "--suffix", f"_o200_lam{lam_s(lam)}", "--gpu", str(g)],
            lambda lam=lam: (T / f"logs/jcssense_o200_lam{lam_s(lam)}.json").exists(), prio=1)
    job("mcinr", "mcinr", lambda g: [PY, "-u", str(S / "run_mcinr.py"), "--gpu", str(g)],
        lambda: all((T / f"logs/mcinr_{s}.json").exists() for s in SCANS), prio=3)
    job("gs_joint", "gs", lambda g: ["bash", str(S / "run_gs.sh"), "ce_retro_af16_joint", str(g), "4000"],
        lambda: (T / "trainer/_trainer_e778bf2/results_arc/ce_retro_af16_joint_volumes.npz").exists(), prio=4)
    job("extract_joint", "cls", lambda g: [PY, "-u", str(S / "extract_gs_volumes.py"), "--exp", "ce_retro_af16_joint", "--key", "gs_joint", "--map", "pre:plain,post:contrast", "--gpu", str(g)],
        lambda: all((T / f"logs/gs_joint_{s}.json").exists() for s in SCANS), deps=["gs_joint"], prio=4)
    arms = [j["name"] for j in J]
    J.append(dict(name="summarize", lane="cpu", prio=9, need=0, deps=arms, cmd=lambda g: [PY, "-u", str(S / "summarize.py")],
                  done=lambda: (T / "logs/orch/summarized").exists()))
    return J


def _best_lam(T, fam, scan):
    """(best lambda, sorted lambdas seen) of a swept family from its jsons; None if nothing finished."""
    pat = {"cs": f"logs/classical_{scan}_o200_lam*.json", "joint": "logs/jcssense_o200_lam*.json"}[fam]
    cands = []
    for f in sorted(Path(T).glob(pat)):
        for r in json.load(open(f)).get("rows", []):
            if r.get("contrast", scan) == scan and r.get("head_psnr") is not None:
                cands.append((float(r["lam"]), float(r["head_psnr"])))
    if not cands:
        return None, []
    lams = sorted({l for l, _ in cands})
    return max(cands, key=lambda t: t[1])[0], lams


def plan_extensions(T, jobs):
    """new jobs for lambda families whose best sits on the grid edge (x1.4 up / /1.4 down, MAX_EXT per family+scan)."""
    T = Path(T); S = T / "scripts"; new = []
    for fam, scans in (("cs", SCANS), ("joint", ("pre", "post"))):
        base = set(GRID[fam])
        for scan in scans:
            prefix = f"cs_{scan}_lam" if fam == "cs" else "joint_lam"
            pending = [n for n, j in jobs.items() if n.startswith(prefix) and j["state"] != "done"]
            if pending:
                continue
            best, lams = _best_lam(T, fam, scan)
            if best is None or len(set(lams) - base) >= MAX_EXT * (2 if fam == "joint" else 1):
                continue
            if best == max(lams):
                lam = float(f"{best * 1.4:.2g}")
            elif best == min(lams):
                lam = float(f"{best / 1.4:.2g}")
            else:
                continue
            name = f"{prefix}{lam_s(lam)}"
            if name in jobs or any(j["name"] == name for j in new):
                continue
            if fam == "cs":
                cmd = lambda g, s=scan, lam=lam: [PY, "-u", str(S / "recon_classical.py"), "--scan", s, "--methods", "cssense", "--admm_outer", "200",
                                                  "--lam", lam_s(lam), "--suffix", f"_o200_lam{lam_s(lam)}", "--gpu", str(g)]
                done = lambda s=scan, lam=lam: (T / f"logs/classical_{s}_o200_lam{lam_s(lam)}.json").exists()
            else:
                cmd = lambda g, lam=lam: [PY, "-u", str(S / "recon_joint_cssense.py"), "--topic", str(T), "--contrasts", "pre,post", "--admm_outer", "200",
                                          "--lam", lam_s(lam), "--suffix", f"_o200_lam{lam_s(lam)}", "--gpu", str(g)]
                done = lambda lam=lam: (T / f"logs/jcssense_o200_lam{lam_s(lam)}.json").exists()
            new.append(dict(name=name, lane="gpu", prio=1, need=NEED[fam], deps=[], cmd=cmd, done=done))
    return new
```
主循环要点（照 ce_orchestrate.py 改）：`T = Path(os.environ.get("RETRO_TOPIC", …))`；`LOGS = T/"logs/orch"`，`JOBS = LOGS/"jobs"`；启动时 `for j in build_jobs(T): add(j)`；每轮刷新状态后 `for j in plan_extensions(T, jobs): add(j)`，并把 `summarize` 的 `deps` 更新为所有非 summarize 作业名（`jobs["summarize"]["deps"] = [n for n in jobs if n != "summarize"]`）；`summarize` 作业成功后 `touch LOGS/"summarized"`（在 launch 时给 cpu lane 的命令后面接 `&& touch …` 不方便，改为：主循环看到 `summarize` 进程 rc==0 且 `logs/summary.json` 存在就写 marker）；全部 done 后写 `LOGS/"ALL_DONE_retro16"`。`--dry`：`if "--dry" in sys.argv: for j in build_jobs(T): print(j["name"], j["lane"], j["need"], j["deps"], " ".join(j["cmd"](0))); sys.exit(0)`。GPU 选择沿用 `my_busy_gpus()`（利用率 < 30% 才算空，`free[c] >= j["need"]`），`MAX_GPU = 3`，launch 用 `subprocess.Popen(j["cmd"](g), stdout=lf, stderr=subprocess.STDOUT, env=ENV, start_new_session=True)` 并写 pid 文件。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd $WT/ce_retro_cssense_af16/scripts && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_orchestrate_plan.py -q`
Expected: `1 passed`

- [ ] **Step 5: 干跑**

Run: `cd $TOPIC && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/retro_orchestrate.py --dry | tee logs/orch_dry.txt | wc -l`
Expected: 59 行作业（2 cls + 14 cs + 24 spirit + 2 zs + 2 dip + 2 inr + 2 gs_single + 2 extract_single + 5 joint + 1 mcinr + 1 gs_joint + 1 extract_joint + 1 summarize），每行含完整命令；把实际行数写进 STATUS。

- [ ] **Step 6: Commit**

```bash
cd $WT && git add ce_retro_cssense_af16/scripts/retro_orchestrate.py ce_retro_cssense_af16/scripts/tests/test_orchestrate_plan.py && git -c user.name="Congcong Liu" -c user.email="sober235@users.noreply.github.com" commit -m "ce retro af16: scheduler plan retro16 with lambda edge extension"
```

---

### Task 14: 图：三面对比图移植、强化图移植

**Files:**
- Copy+edit: jyz `fig_three_planes_compact.py` → `scripts/fig_three_planes_compact.py`
- Create: `scripts/fig_enhancement.py`（以 `/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/scripts/ce_figure_paper.py` 的 `--diff` 分支为底）
- Test: `scripts/tests/test_fig_inputs.py`

**Interfaces:**
- Consumes: `logs/summary.json`、`arms.arm_files`、`recon/gt_<scan>_complex.npy`（Task 6 写）、`logs/prep_04_export.json`（Task 5 写）、ROI npz。
- Produces: `figures/ce_retro_af16_three_planes_<scan>_201_131_48.{png,pdf}`（每扫描一张，jyz 版式 11 列）、`figures/ce_retro_af16_enhancement.{png,pdf}`（9 行 × 11 列 + GT 列）。

- [ ] **Step 1: 移植三面图脚本**

```bash
J=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/scripts; D=$WT/ce_retro_cssense_af16/scripts/fig_three_planes_compact.py
cp $J/fig_three_planes_compact.py $D
sed -i 's#^R = Path(.*#R = Path(__import__("os").environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))#' $D
sed -i 's/ap.add_argument("--contrast", required=True, choices=\["T1", "FLAIR"\])/ap.add_argument("--scan", dest="contrast", required=True, choices=["pre", "post"])/' $D
sed -i 's/default="gs_joint_ar"/default="gs_joint"/; s/default="gs_single_ar"/default="gs_single"/; s/default="espirit", choices=\["espirit", "spirit", "both"\]/default="spirit", choices=["espirit", "spirit", "both"]/' $D
grep -n "jiangyizhou\|nominal AF 16\|ARMS = \|l1espirit\|_mag.npy\|jyz" $D
```
再手改四处：(1) `fig.suptitle(f"{c}  jiangyizhou  {AF_LABEL} …` 里的 `jiangyizhou` 改成 `UIH pre/post, CS-SENSE truth`；(2) `ARMS` 列表：确认顺序为 zf(`--with_zf`)、cgsense、cssense_o200、jcssense、spirit、zsssdu、dip、inr、mcinr、gs_single、gs_joint，GS 两臂读 `recon/{key}_{c}_mag.npy`（Task 11 已写）；若列表里 CS-SENSE 用的是 `lam_file("cssense_o200", …)` 之类的通配函数，保持不动（命名一致）；(3) 标签 `"MC-INR"`、`"L1-SPIRiT"` 已在；(4) `_ex["contrasts"][c]` 与 `mask2d_AF_rect / mask2d_shape / mask2d_samples` 都由 Task 5 的 `logs/prep_04_export.json` 提供，不改。

- [ ] **Step 2: 写强化图脚本**

`scripts/fig_enhancement.py`：从 `ce_figure_paper.py` 拷贝后，只保留 `--diff` 路径，并把"体从 k 空间 α 复标度 + 缓存"的加载块（原文件 `W = …` 到 `log(f"cached magnitudes …")` 之间，约 68 到 150 行）整体换成：
```python
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arms import arm_files, LABEL                                            # noqa: E402
R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
ROI_NPZ = "/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/lesion_candidates/enhance_rois_cssense.npz"
mag = {}
with np.load(R / "sim/gt.npz") as g:
    for scan in ("pre", "post"):
        mag[(scan, "GT")] = (np.abs(g[scan]) / float(g[f"scale_{scan}"])).astype(np.float32)
for scan in ("pre", "post"):
    for key, label, p in arm_files(scan):
        v = np.load(p)
        mag[(scan, label)] = (np.abs(v) if np.iscomplexobj(v) else v).astype(np.float32)
COLS = [("GT", None)] + [(label, None) for _, label, _ in arm_files("pre")]
```
其余保留：`planes()`、`span()`、`enh_units()`、差值行的 σ=1 平滑显示、列底四个参考区数值、`--slices 201,131,48` 默认、`--dvmax 0.20`、字体设置。删掉 `--gpu`、`--cache`、`--recompute`、`--drop`、`--zoom` 及其分支；`--out` 默认改为 `R / "figures"`，输出名 `ce_retro_af16_enhancement{suffix}.png/.pdf`；ROI 读 `ROI_NPZ`。所有引用 `WORK`、`fft3c`、`torch` 的行随加载块一起删除（脚本不再需要 torch）。
标题第一行写 `UIH pre/post pair, CS-SENSE truth, Poisson-disc AF 16.00, 11 arms + truth`。

- [ ] **Step 3: 写测试（输入契约）**

`scripts/tests/test_fig_inputs.py`:
```python
import ast
from pathlib import Path

S = Path(__file__).resolve().parents[1]


def test_figure_scripts_parse_and_have_no_jyz_paths():
    for name in ("fig_three_planes_compact.py", "fig_enhancement.py"):
        src = (S / name).read_text()
        ast.parse(src)
        assert "T1FLAIR_AF16_v3_jiangyizhou" not in src and "mcgs_af16_wt/results_arc" not in src
        assert "RETRO_TOPIC" in src


def test_enhancement_script_reads_truth_and_arms():
    src = (S / "fig_enhancement.py").read_text()
    assert "arm_files(" in src and 'sim/gt.npz' in src and "enhance_rois_cssense.npz" in src
    assert "--gpu" not in src and "WORK" not in src
```

- [ ] **Step 4: 跑测试**

Run: `cd $WT/ce_retro_cssense_af16/scripts && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_fig_inputs.py -q`
Expected: `2 passed`

- [ ] **Step 5: 出图（全量跑完、summarize 之后执行；此时只做 --help 冒烟）**

Run: `cd $TOPIC && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/fig_three_planes_compact.py --help | head -5 && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/fig_enhancement.py --help | head -5`
Expected: 两个 usage。正式出图命令（写进 Task 16 手册）：
```bash
for s in pre post; do env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/fig_three_planes_compact.py --scan $s --slices 201 131 48 --inset 48 --inset_corner tr bl bl --mask_error --gray_cbar --cbar_planes 0 --tra_rot 1 --err_vmax 0.1 --with_zf --caps --l1_arm spirit --pdf --pdf_dpi 600 --tag _201_131_48; done
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/fig_enhancement.py
```
（三面图脚本自己命名输出；把两张 png 复制/改名为 `figures/ce_retro_af16_three_planes_<scan>_201_131_48.png`，并拷到 `/home/congcongliu/figs/ce_retro_af16/`。）

- [ ] **Step 6: Commit**

```bash
cd $WT && git add ce_retro_cssense_af16/scripts/fig_three_planes_compact.py ce_retro_cssense_af16/scripts/fig_enhancement.py ce_retro_cssense_af16/scripts/tests/test_fig_inputs.py && git -c user.name="Congcong Liu" -c user.email="sober235@users.noreply.github.com" commit -m "ce retro af16: three-plane comparison and enhancement figures on the truth gauge"
```

---

### Task 15: README、STATUS、`readme_tables.py`、`export_bundle.py`

**Files:**
- Create: `scripts/readme_tables.py`、`scripts/export_bundle.py`、`README.md`、`STATUS.md`（都在 `WT/ce_retro_cssense_af16/`；README/STATUS 同时复制到 `TOPIC/`）

**Interfaces:**
- Consumes: `logs/summary.json`、`logs/enhancement_fidelity.json`、`sim/meta.json`、`sim/mask/mask_af16.json`、`logs/check_sim.json`。
- Produces: `readme_tables.py` 打印 markdown 表（每扫描按头内 PSNR 排序的 11 行 + `MC-GS − GS-single` 行 + 强化保真表）；`export_bundle.py` 写 `export/ce_retro_af16_<date>/{volumes/<scan>/<key>_<scan>_complex.npy, sim/(mask, gt.npz, meta), logs/, figures/, MANIFEST.tsv, md5sums.txt}`（`--no_raw` 不拷 h5）。

- [ ] **Step 1: readme_tables.py**

```python
#!/usr/bin/env python3
"""readme_tables.py -- markdown tables for README.md from logs/summary.json and logs/enhancement_fidelity.json.
Per scan: the eleven arms sorted by head-only PSNR (bold = best per column), the MC-GS - GS-single line (the headline
margin, user rule 2026-09-13), then the enhancement-fidelity table (post - pre in white-matter units, err vs truth)."""
import json, os
from pathlib import Path
from arms import LABEL

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
rows = json.load(open(R / "logs/summary.json"))
for c in ("pre", "post"):
    sub = sorted([r for r in rows if r["contrast"] == c], key=lambda r: -r["head_psnr"])
    best = {k: (max if k != "head_nmse" and k != "head_hfen" else min)(r[k] for r in sub) for k in ("head_psnr", "head_ssim", "head_nmse", "head_hfen")}
    print(f"\n**{c}** (sorted by head-only PSNR; truth = CS-SENSE of the prospective pair)\n")
    print("| method | head PSNR | head SSIM | head NMSE | head HFEN | complex PSNR | whole PSNR | λ / kernel |")
    print("|---|---|---|---|---|---|---|---|")
    for r in sub:
        f = lambda k, p: (f"**{r[k]:.{p}f}**" if r[k] == best[k] else f"{r[k]:.{p}f}")
        lam = (f"{r['lam']:g}" if r.get("lam") is not None else "") + (f" / ks{r['ksize']}" if r.get("ksize") else "")
        print(f"| {LABEL.get(r['method'], r['method'])} | {f('head_psnr',3)} | {f('head_ssim',4)} | {f('head_nmse',5)} | {f('head_hfen',4)} | {r['psnr_cplx']:.3f} | {r['psnr']:.3f} | {lam} |")
print("\n**MC-GS minus GS-single**\n\n| scan | head PSNR | head SSIM | head NMSE | head HFEN |\n|---|---|---|---|---|")
for c in ("pre", "post"):
    j = next(r for r in rows if r["contrast"] == c and r["method"] == "gs_joint"); s = next(r for r in rows if r["contrast"] == c and r["method"] == "gs_single")
    print(f"| {c} | {j['head_psnr']-s['head_psnr']:+.3f} | {j['head_ssim']-s['head_ssim']:+.4f} | {j['head_nmse']-s['head_nmse']:+.5f} | {j['head_hfen']-s['head_hfen']:+.4f} |")
ef = R / "logs/enhancement_fidelity.json"
if ef.exists():
    e = json.load(open(ef))["rows"]
    print("\n**Enhancement (post − pre, white-matter units); err = arm − truth**\n\n| arm | vessels | plexus | cortex | WM | lesion core | rim 0–3 mm | background(post) |\n|---|---|---|---|---|---|---|---|")
    for r in e:
        cell = lambda k: (f"{r[k]:+.3f}" if r["method"] == "truth" else f"{r[k]:+.3f} ({r['err_' + k]:+.3f})")
        print(f"| {r['label']} | {cell('vessels')} | {cell('plexus')} | {cell('cortex')} | {cell('white_matter')} | {cell('lesion_core')} | {cell('rim_0_3mm')} | {r['background_post']:.3f} |")
```

- [ ] **Step 2: export_bundle.py**

```python
#!/usr/bin/env python3
"""export_bundle.py -- the results bundle of this study: every registered volume (both scans), the simulation inputs
(mask, truth, meta), logs and figures, with MANIFEST.tsv and md5sums. --no_raw skips the 14 GB of h5 / work_* k-space.
  python scripts/export_bundle.py [--no_raw]   -> export/ce_retro_af16_<YYYYMMDD>/"""
import argparse, hashlib, os, shutil, time
from pathlib import Path
from arms import arm_files, LABEL

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
ap = argparse.ArgumentParser(); ap.add_argument("--no_raw", action="store_true"); a = ap.parse_args()
out = R / f"export/ce_retro_af16_{time.strftime('%Y%m%d')}"; out.mkdir(parents=True, exist_ok=True)
rows = []


def put(src, rel, note):
    dst = out / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    rows.append((rel, note, dst.stat().st_size, hashlib.md5(dst.read_bytes()).hexdigest(), str(src)))


for scan in ("pre", "post"):
    for key, label, p in arm_files(scan):
        put(p, f"volumes/{scan}/{key}_{scan}_complex.npy", f"{label} reconstruction, gauge units (scale_{scan} in sim/gt.npz), layout (S-I 256, A-P 232, L-R 176)")
    put(R / f"recon/gt_{scan}_complex.npy", f"volumes/{scan}/gt_{scan}_complex.npy", "CS-SENSE truth in gauge units")
put(R / "sim/gt.npz", "sim/gt.npz", "truth volumes (raw units), mask2d, scale_pre/scale_post"); put(R / "sim/meta.json", "sim/meta.json", "sigma, kres, seeds")
for f in (R / "sim/mask").iterdir():
    put(f, f"sim/mask/{f.name}", "Poisson-disc AF16 mask")
if not a.no_raw:
    for f in list((R / "sim/gs").iterdir()) + [q for s in ("pre", "post") for q in (R / f"sim/work_{s}").iterdir()]:
        put(f, f"raw/{f.parent.name}/{f.name}", "synthetic k-space / maps / mask as the arms read them")
for f in sorted((R / "logs").glob("*.json")):
    put(f, f"logs/{f.name}", "score json")
for f in sorted((R / "figures").glob("*.p*")):
    put(f, f"figures/{f.name}", "figure")
with open(out / "MANIFEST.tsv", "w") as m:
    m.write("path\tnote\tbytes\tmd5\tsource\n"); m.writelines("\t".join(map(str, r)) + "\n" for r in rows)
with open(out / "md5sums.txt", "w") as m:
    m.writelines(f"{r[3]}  {r[0]}\n" for r in rows)
print(f"{len(rows)} files, {sum(r[2] for r in rows) / 2**30:.2f} GiB -> {out}"); print("verify: cd", out, "&& md5sum -c md5sums.txt | grep -c OK")
```

- [ ] **Step 3: README.md 与 STATUS.md**

`README.md`（`WT/ce_retro_cssense_af16/README.md`，同步复制到 `TOPIC/README.md`）固定节：① 一句话目的与设计文档路径；② 数据与合成（GT 文件、CSM、掩膜 json 的 seed/acc/trimmed、σ 与 kres 复现值、WM SNR，从 `sim/meta.json`、`sim/mask/mask_af16.json`、`logs/check_sim.json` 抄数）；③ 各臂设置表（spec 第 5 节的表照抄）与 λ 协议；④ 结果表（`readme_tables.py` 的输出粘贴）；⑤ 图（文件名 + `http://localhost:8765/ce_retro_af16/<name>.png` 与绝对路径两行）；⑥ 复现命令（Task 16 手册的命令）；⑦ 坑；⑧ 口径与措辞约束（spec 第 11 节）。数字没跑出来之前对应位置写 `NOT_RUN_YET`，跑完必须替换。
`STATUS.md` 五段（用户交接协议）：已完成且已验证（含可粘贴命令与原始输出）、待用户拍板、下一步、已知的坑与别重做什么、关键决定的为什么；开头写"新会话先读本文件，再读 README，关键数字先重跑"。

- [ ] **Step 4: 跑全部测试 + Commit**

Run: `cd $WT/ce_retro_cssense_af16 && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest scripts/tests -q`
Expected: 全部 passed。
```bash
cd $WT && cp ce_retro_cssense_af16/README.md ce_retro_cssense_af16/STATUS.md $TOPIC/ && git add ce_retro_cssense_af16 && git -c user.name="Congcong Liu" -c user.email="sober235@users.noreply.github.com" commit -m "ce retro af16: README, STATUS, README tables and export bundle"
```

---

### Task 16: 执行手册：门 → 冒烟 → 全量 → 评分 → 图 → 落盘

**Files:**
- Modify: `STATUS.md`、`README.md`（填数）；`TOPIC/logs/*`、`figures/*`、`export/*`

**Interfaces:**
- Consumes: 全部前序任务。
- Produces: 完整结果、GitHub 分支推送、handoff tag。

- [ ] **Step 1: 门检查清单（全部为真才往下）**

```bash
cd $TOPIC
test -f sim/mask/mask_af16.json && grep -q '"af": 16.0' sim/mask/mask_af16.json && echo MASK_OK
grep -q '"cg": true' logs/check_sim.json && grep -c true logs/check_sim.json && echo CHECK_SIM_OK
ls trainer/_trainer_e778bf2/results_arc/smoke_joint_score.json && echo GS_SMOKE_OK
```
并确认用户已对 `figures/mask_af16.png` 点头（Task 2 Step 5 的汇报回复）。缺任何一项都不开全量。

- [ ] **Step 2: 启动全量调度（约 6 到 8 小时墙钟）**

```bash
cd $TOPIC && mkdir -p logs/orch && setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID $PY -u scripts/retro_orchestrate.py > logs/orch/orchestrate.log 2>&1 &
sleep 60; tail -n 20 logs/orch/orchestrate.log; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```
Expected: 日志出现 `queued …` 若干与 `launched cls_pre on gpu N pid …`。监控用 `tail -f logs/orch/orchestrate.log`；单个作业日志在 `logs/orch/<job>.log`；`RETRY`/`FAILED` 行要看对应作业日志并修复后重启调度器（它会跳过已完成作业）。结束标志 `logs/orch/ALL_DONE_retro16`。GPU 被别的会话占满时作业会排队，不要去杀别人的进程。

- [ ] **Step 3: 口径互验（门 3）与汇总**

```bash
cd $TOPIC && for s in pre post; do for k in cgsense zsssdu dip inr mcinr gs_single gs_joint; do f=recon/${k}_${s}_complex.npy; [ -f $f ] || continue; env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/score_volume.py --scan $s --file $f --method $k --check logs/$( [ $k = cgsense ] && echo classical_$s || echo ${k}_$s ).json | tail -1; done; done
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/summarize.py
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/enhancement_fidelity.py
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/find_best_slice.py && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/best_slice_mcgs.py --axis 0
```
Expected: 每个 `GAUGE OK`（CPU 双精度重评与 GPU 单精度记录差 < 2e-4）；summary 打印两个扫描各 11 行；强化保真表打印；最优层 json 写出。若 GS/MC-INR 的某项 MISMATCH，回到 Task 11/10 的单位换算查因，不要改容差。

- [ ] **Step 4: 出图并挂到图片服务**

```bash
cd $TOPIC && for s in pre post; do env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/fig_three_planes_compact.py --scan $s --slices 201 131 48 --inset 48 --inset_corner tr bl bl --mask_error --gray_cbar --cbar_planes 0 --tra_rot 1 --err_vmax 0.1 --with_zf --caps --l1_arm spirit --pdf --pdf_dpi 600 --tag _201_131_48; done
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/fig_enhancement.py
mkdir -p /home/congcongliu/figs/ce_retro_af16 && cp figures/*.png /home/congcongliu/figs/ce_retro_af16/ && for f in /home/congcongliu/figs/ce_retro_af16/*.png; do printf "%s " $(basename $f); curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8765/ce_retro_af16/$(basename $f); done
```
Expected: 全部 200。汇报里每张图两行（URL、绝对路径）。

- [ ] **Step 5: 填 README/STATUS，记忆，落盘**

```bash
cd $TOPIC && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/readme_tables.py > /tmp/readme_tables.md   # 粘进 README 第④节，替换 NOT_RUN_YET
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/export_bundle.py --no_raw && env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/export_bundle.py   # 先小包核对，再全包
cd $WT && cp $TOPIC/logs/*.json ce_retro_cssense_af16/logs/ && cp $TOPIC/figures/*.png ce_retro_cssense_af16/figures/ && cp $TOPIC/README.md $TOPIC/STATUS.md ce_retro_cssense_af16/
git add ce_retro_cssense_af16 && git -c user.name="Congcong Liu" -c user.email="sober235@users.noreply.github.com" commit -m "ce retro af16: results, figures, README and STATUS"
git tag handoff/$(date +%Y-%m-%d)-ce-retro-af16 && git -c credential.helper='!gh auth git-credential' push -u origin ce-retro-cssense-af16 --tags
```
Expected: push 成功（`gh` 已登录 sober235）；GitHub 上分支 `ce-retro-cssense-af16` 有 spec、plan、脚本、日志 json、图、README、STATUS。STATUS 第一段贴上 summarize 的原始输出与 check_sim / gauge OK 的原文；第二段列待拍板：是否给 MC-GS 同规模小网格、是否补第二张掩膜/第二对种子、正文图砍列。最后在汇报里给用户结论：头条余量 MC-GS − GS-single（两扫描），MC-GS 对最强对比臂的余量，强化保真三个数，以及 spec 第 11 节的措辞约束。
