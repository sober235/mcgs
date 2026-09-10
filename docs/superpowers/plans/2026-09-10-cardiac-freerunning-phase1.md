# 心脏自由运行多回波第一阶段 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 V07、V03、V04、V08 的自由运行 8 回波 3D 径向数据读成可重建形式，用 BART 做运动平均（R0）与呼气末参考（REF）重建，并在 V07 上建立、自检评估口径。

**Architecture:** worktree 里新增纯库 `cardiac_fr/`（twix 读取、轨迹、生理、BART 封装、口径），每个模块有 CPU 单元测试。课题目录的 `scripts/` 放薄驱动脚本，按步骤调用库和 BART 命令行；大中间文件放 /data2。重建全走 BART `pics -t`（CPU），不碰 MC-GS，不占 GPU。

**Tech Stack:** Python（GSTMI 环境：numpy 1.26、scipy 1.15、h5py 3.15、torch 2.7 只在 CPU 上算指标、scikit-image 0.26、matplotlib 3.10、pytest 9）；BART v1.0.00（`/home/congcongliu/src/bart/bart`，CPU 构建）与它的 `python/cfl.py`；MATLAB R2021a（只做两次交叉核对）。

**Spec:** `docs/superpowers/specs/2026-09-10-cardiac-freerunning-phase1-design.md`（本 worktree，提交 8b25ad5）

## Global Constraints

- 主树 `/data0/congcong/code/GS/Paired_ARC_GS_general_framework` 与 `/data0/congcong/code/GS/mcgs_af16_wt` 一行不动。本分支已有文件不改，只新增；例外只有本分支的 `CLAUDE.md`、`STATUS.md`，以及 `/data0/congcong/code/GS/Results_20260908/README.md` 课题表加一行。
- 不占 GPU：所有 Python 都经课题的 `scripts/py` 包装运行（`CUDA_VISIBLE_DEVICES=` 置空）；BART 是 CPU 构建。
- CPU 礼让：`scripts/py` 以 `nice -n 10` 启动，BART 线程 `OMP_NUM_THREADS=32`（环境变量 `CARDIAC_FR_THREADS`，默认 32），同一时间只跑一个 BART 重建（`--jobs 1`）。依据：2026-09-10 18:57 负载约 200/112 核，大头是本用户的其他重建与测试。
- 超过几分钟的任务写成 bash 脚本，用 `setsid nohup bash <脚本> > <日志> 2>&1 < /dev/null &` 启动，再看日志。工具 shell 是 zsh，命令里不要依赖 `$VAR` 分词，路径一律写全。
- 原始数据只读：`/data1/data1_congcong/data/3D/cardiac_freerunning_multiecho_5D/` 与 `/data2/congcong/data/cardiac_3d_cine_public/` 下，除新建的 `derived/` 外一概不写。
- 提交：仓库本地身份 `Congcong Liu <liuc23539@gmail.com>`（已配置，不用改），消息写成普通人工提交，不加任何 AI trailer；只 `git add` 点名的文件；推送到 GitHub 前先问用户。
- 数值（取自 spec）：保留读出采样 3–220（`KEEP = slice(3, 221)`）；稳态阈值 10 s；呼气末取 1/3；AF = 19,704 / 辐条数；留出 5%，seed 0；子集 AF 8、12、16，按整组交错抽；R0 与 REF 用 `pics -l2 -r 1e-3 -i 100`；线圈图 `ecalib -t 0.001 -c 0.0 -r 24 -m 1`；L1 λ 候选 {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}。
- BART：所有 `pics` 加 `-S`；`-l1` 必须加 `-e`（径向下不加会发散，2026-09-10 在 16³ 体模上实测数据残差 6.5e10）；REF 与所有子集共用该回波 REF 的 `-w`。
- 误差图：|rec − ref| / P99.9(|ref|)，jet，固定 [0, 0.1]。
- 图存 `/home/congcongliu/figs/cardiac_fr/` 并复制到课题 `figures/`；报给用户时先写 `http://localhost:8765/cardiac_fr/<图>.png`，再写绝对路径。
- 停止条件：5.1 读取交叉核对、5.2 采样点范围、5.3 轨迹核对、5.4 极性判据、5.6 迭代与缩放核对、5.7 口径自检，任一不过就停，报告，不绕过。

## 路径速查（命令里都写全路径）

| 名称 | 路径 |
|---|---|
| worktree | `/data0/congcong/code/GS/mcgs_cardiac_wt` |
| 课题目录 | `/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE` |
| Python 包装 | `/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/py` |
| 派生数据 | `/data2/congcong/data/cardiac_3d_cine_public/derived/megre` |
| 原始数据 | `/data1/data1_congcong/data/3D/cardiac_freerunning_multiecho_5D/<Vxx>/{freerunning_8echo.dat,PhysioInfo.mat}` |

## 文件结构

worktree（进 git）：
- `cardiac_fr/__init__.py`：路径与协议常数。
- `cardiac_fr/twix.py`：Siemens VD/VE multi-raid 读取。
- `cardiac_fr/trajectory.py`：phyllotaxis 轨迹（移植作者 MATLAB）。
- `cardiac_fr/physio.py`：PhysioInfo 与 ADC 的映射、稳态、呼吸门控、心动相位。
- `cardiac_fr/bart.py`：cfl 读写与 BART 命令封装。
- `cardiac_fr/gauge.py`：AF、留出、整组子集、图像指标、k 空间 NRMSE。
- `tests/test_cardiac_fr_{twix,trajectory,physio,bart,gauge}.py`：只用 CPU 与小合成数据。
- `cardiac_freerunning_megre/`：课题镜像（scripts 每次提交同步；README、logs、figures、json 在 Task 13 同步）。

课题目录（按归档约定，不进 git）：
- `scripts/py`、`scripts/common.py`、`scripts/s01_read.py` … `scripts/s11_figures.py`、`scripts/xcheck_*.m`、`scripts/xcheck_*_compare.py`、`scripts/run_*.sh`。
- `third_party/`（作者代码，只做交叉核对）、`data/<Vxx>`（软链接到派生数据）、`recon/<Vxx>/*.npy`、`logs/`、`figures/`、`orientation.json`、`gauge.json`、`params_frozen.json`、`README.md`。

派生数据（/data2）：
- `common/traj_r.npy`、`common/traj_matlab.mat`。
- `<Vxx>/raw.h5`、`noise.npy`、`meta.json`、`spokes.npz`、`physio.json`、`sens.{cfl,hdr}`、`ref_ids.npy`、`gauge_split.npz`、`bart/`（BART 的输入与输出）。

## 镜像同步命令（Task 6 起每个提交步骤先跑它）

```bash
mkdir -p /data0/congcong/code/GS/mcgs_cardiac_wt/cardiac_freerunning_megre && rsync -a --exclude '__pycache__' /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts /data0/congcong/code/GS/mcgs_cardiac_wt/cardiac_freerunning_megre/
```

## 预计耗时（2026-09-10 实测，负载约 200，正常优先级；nice 10 下可能慢到两倍）

112³、24 线圈、6,800 条辐条：`pics` 100 步 CG 约 8.5 分钟（启动约 56 s，每步约 4.5 s），内存约 5.4 GB；`nufft -i` 逐线圈图约 5 分钟。R0（约 20,400 条）每回波约 10–12 分钟。

---

### Task 1: 课题骨架、Python 包装与 twix 读取器

**Files:**
- Create: `/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/py` 与课题子目录、`third_party/`
- Create: `cardiac_fr/__init__.py`、`cardiac_fr/twix.py`
- Test: `tests/test_cardiac_fr_twix.py`

**Interfaces:**
- Produces: `cardiac_fr` 常数 `RAW_ROOT, DERIVED_ROOT, TOPIC, FIGS, SUBJECTS, N_ECHO=8, N_SHOT=1000, N_SEG=22, N_SPOKE=22000, N_SAMP=224, N_COIL=24, GRID=112, KEEP=slice(3,221), T_STEADY_MS=10000.0`。
- Produces: `cardiac_fr.twix.load_freerunning(path, n_echo) -> FreeRunningData`，字段 `ksp (n_echo, n_spoke, n_coil, n_samp) complex64`、`noise (n_coil, n_samp) complex64`、`lin, seg (n_spoke,) int64`、`mask (n_spoke*n_echo,) uint64`、`eco, ts_ticks (n_spoke*n_echo,) int64`、`protocol: str`、`ascconv: dict[str, str]`；另有 `read_raid_header, read_meas_header, parse_ascconv, index_scans, read_adc` 与 `MASK_ACQEND, MASK_SYNCDATA, MASK_NOISEADJ`。

- [ ] **Step 1: 建课题骨架、包装脚本与第三方代码**

```bash
mkdir -p /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/{scripts,logs,figures,recon,data,third_party} /data2/congcong/data/cardiac_3d_cine_public/derived/megre/common /home/congcongliu/figs/cardiac_fr
cat > /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/py <<'EOF'
#!/usr/bin/env bash
# GSTMI python for this topic: house environment, CPU only (spec 1.1), nice 10, 32 BART threads.
export PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= CARDIAC_FR_THREADS=${CARDIAC_FR_THREADS:-32}
exec nice -n 10 /home/congcongliu/anaconda3/envs/GSTMI/bin/python "$@"
EOF
chmod +x /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/py
cd /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/third_party && git clone -q https://github.com/QIS-MRI/ReadDataAndTrajectory_FreeRunningFatFractionHeart.git && git -C ReadDataAndTrajectory_FreeRunningFatFractionHeart checkout -q a78f6f2 && git clone -q https://github.com/QIS-MRI/SG_RespiratoryCardiac_3DRadial.git && git -C SG_RespiratoryCardiac_3DRadial checkout -q 4a9a484
printf 'Cloned 2026-09-10 for cross-checks only (spec section 4). Upstream has no LICENSE file; do not redistribute.\nReadDataAndTrajectory_FreeRunningFatFractionHeart @ a78f6f2\nSG_RespiratoryCardiac_3DRadial @ 4a9a484\n' > /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/third_party/SOURCE.txt
git -C /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/third_party/ReadDataAndTrajectory_FreeRunningFatFractionHeart log -1 --format=%h
```

Expected: 最后一行打印 `a78f6f2`。

- [ ] **Step 2: 写失败的测试** `tests/test_cardiac_fr_twix.py`

```python
import struct

import numpy as np
import pytest

from cardiac_fr.twix import (MASK_ACQEND, MASK_NOISEADJ, MASK_SYNCDATA, index_scans,
                             load_freerunning, read_meas_header, read_raid_header)

ONLINE = 1 << 3
N_SPOKE, N_ECHO, NCH, NS = 5, 3, 2, 6
YAPS = (b"junk\n### ASCCONV BEGIN ###\nsKSpace.lBaseResolution = 112\n"
        b"tProtocolName = \"FRFF_test\"\n### ASCCONV END ###\n")


def _scan_header(mask, ns, nch, lin=0, eco=0, seg=0, ts=0, dma=None):
    h = bytearray(192)
    struct.pack_into("<I", h, 0, (192 + nch * (32 + 8 * ns) if dma is None else dma) & 0x1FFFFFF)
    struct.pack_into("<I", h, 12, ts)
    struct.pack_into("<Q", h, 40, mask)
    struct.pack_into("<HH", h, 48, ns, nch)
    lc = [0] * 14
    lc[0], lc[4], lc[8] = lin, eco, seg
    struct.pack_into("<14H", h, 52, *lc)
    return bytes(h)


def _adc(mask, data, **kw):
    nch, ns = data.shape
    out = bytearray(_scan_header(mask, ns, nch, **kw))
    for c in range(nch):
        out += bytes(32) + data[c].astype("<c8").tobytes()
    return bytes(out)


def _sync(payload=100):
    return _scan_header(MASK_SYNCDATA, 0, 0, dma=192 + payload) + bytes(payload)


def _acqend():
    return _scan_header(MASK_ACQEND, 16, 0, dma=192)


def _meas(buffers, packets):
    body = bytearray()
    for name, b in buffers.items():
        body += name.encode() + b"\0" + struct.pack("<I", len(b)) + b
    return struct.pack("<II", 8 + len(body), len(buffers)) + bytes(body) + b"".join(packets)


def _write_twix(path, measurements):
    off = 8 + 152 * len(measurements)
    entries, blobs = b"", b""
    for i, (prot, payload) in enumerate(measurements):
        entries += (struct.pack("<IIQQ", i + 1, i + 1, off + len(blobs), len(payload))
                    + bytes(64) + prot.encode().ljust(64, b"\0"))
        blobs += payload
    path.write_bytes(struct.pack("<II", 0, len(measurements)) + entries + blobs)


def _value(s, e, c, j):
    return s * 1000 + e * 100 + c * 10 + j + 1j * (s + 0.5)


def _fake_file(tmp_path, echo_order=None, with_acqend=True):
    adj = _meas({"MeasYaps": b""}, [_adc(ONLINE, np.ones((NCH, NS), np.complex64)), _acqend()])
    packets = [_adc(MASK_NOISEADJ | ONLINE, np.full((NCH, NS), 7 - 7j, np.complex64))]
    t = 100
    for s in range(N_SPOKE):
        for e in (echo_order or range(N_ECHO)):
            d = np.array([[_value(s, e, c, j) for j in range(NS)] for c in range(NCH)], np.complex64)
            packets.append(_adc(ONLINE, d, lin=s, eco=e, seg=s // 2, ts=t))
            t += 1
        if s == 2:
            packets.append(_sync())
    if with_acqend:
        packets.append(_acqend())
    p = tmp_path / "fake.dat"
    _write_twix(p, [("AdjCoilSens", adj),
                    ("FRFF_test", _meas({"Config": b"x", "MeasYaps": YAPS}, packets))])
    return p


def test_raid_header_lists_both_measurements(tmp_path):
    with open(_fake_file(tmp_path), "rb") as f:
        assert [m.protocol for m in read_raid_header(f)] == ["AdjCoilSens", "FRFF_test"]


def test_index_skips_syncdata_and_stops_at_acqend(tmp_path):
    with open(_fake_file(tmp_path), "rb") as f:
        meas = read_raid_header(f)[-1]
        hdr_len, bufs = read_meas_header(f, meas.offset)
        idx = index_scans(f, meas, hdr_len)
    assert set(bufs) == {"Config", "MeasYaps"}
    assert idx.n_sync == 1 and idx.acqend
    assert idx.pos.size == 1 + N_SPOKE * N_ECHO


def test_load_places_every_sample(tmp_path):
    d = load_freerunning(_fake_file(tmp_path), n_echo=N_ECHO)
    assert d.ksp.shape == (N_ECHO, N_SPOKE, NCH, NS) and d.ksp.dtype == np.complex64
    for s in range(N_SPOKE):
        for e in range(N_ECHO):
            for c in range(NCH):
                np.testing.assert_array_equal(d.ksp[e, s, c], [_value(s, e, c, j) for j in range(NS)])
    np.testing.assert_array_equal(d.noise, np.full((NCH, NS), 7 - 7j, np.complex64))
    np.testing.assert_array_equal(d.lin, np.arange(N_SPOKE))
    np.testing.assert_array_equal(d.seg, np.arange(N_SPOKE) // 2)
    np.testing.assert_array_equal(d.ts_ticks, 100 + np.arange(N_SPOKE * N_ECHO))
    assert d.protocol == "FRFF_test" and d.ascconv["sKSpace.lBaseResolution"] == "112"


def test_rejects_truncated_measurement(tmp_path):
    with pytest.raises(ValueError, match="ACQEND"):
        load_freerunning(_fake_file(tmp_path, with_acqend=False), n_echo=N_ECHO)


def test_rejects_wrong_echo_cycle(tmp_path):
    with pytest.raises(ValueError, match="echo counter"):
        load_freerunning(_fake_file(tmp_path, echo_order=[0, 2, 1]), n_echo=N_ECHO)
```

- [ ] **Step 3: 跑测试，确认失败**

Run: `cd /data0/congcong/code/GS/mcgs_cardiac_wt && /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/py -m pytest tests/test_cardiac_fr_twix.py -q`
Expected: 收集阶段报 `ModuleNotFoundError: No module named 'cardiac_fr'`。

- [ ] **Step 4: 写 `cardiac_fr/__init__.py`**

```python
"""Free-running multi-echo 3D radial whole-heart data (Mackowiak et al., MRM 2023).

Phase 1 of docs/superpowers/specs/2026-09-10-cardiac-freerunning-phase1-design.md: reading the
Siemens twix files, the phyllotaxis trajectory, physiological gating, BART reconstruction and the
evaluation gauge. Nothing here touches the MC-GS code paths.
"""
from pathlib import Path

RAW_ROOT = Path("/data1/data1_congcong/data/3D/cardiac_freerunning_multiecho_5D")
DERIVED_ROOT = Path("/data2/congcong/data/cardiac_3d_cine_public/derived/megre")
TOPIC = Path("/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE")
FIGS = Path("/home/congcongliu/figs/cardiac_fr")
SUBJECTS = ("V07", "V03", "V04", "V08")

N_ECHO = 8            # free-running multi-echo GRE
N_SHOT = 1000         # interleaves
N_SEG = 22            # spokes per interleave; seg 0 is the SI self-navigation spoke
N_SPOKE = N_SHOT * N_SEG
N_SAMP = 224          # readout samples, 2x oversampled
N_COIL = 24
GRID = 112            # image matrix, FOV 220 mm
KEEP = slice(3, 221)  # readout samples kept (spec 5.2): 0-2 and 221-223 dropped
T_STEADY_MS = 10_000.0
```

- [ ] **Step 5: 写 `cardiac_fr/twix.py`**

```python
"""Minimal reader for Siemens VD/VE multi-raid twix files (what the free-running data needs).

Layout (verified on the four subjects, 2026-09-10): an 8-byte raid header (0, count) followed by
152-byte entries (measId, fileId, offset, length, patient, protocol). Each measurement starts with a
header (length, n_buffers, then name/length/bytes buffers such as MeasYaps) followed by ADC packets:
a 192-byte scan header plus, per channel, a 32-byte channel header and complex64 samples. SYNCDATA
packets are skipped by their DMA length; ACQEND ends the measurement.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCAN_HDR = 192
CHAN_HDR = 32
MASK_ACQEND = 1 << 0
MASK_SYNCDATA = 1 << 5
MASK_NOISEADJ = 1 << 25


@dataclass
class Measurement:
    meas_id: int
    offset: int
    length: int
    protocol: str


@dataclass
class ScanIndex:
    pos: np.ndarray       # file offset of each non-sync ADC packet
    mask: np.ndarray      # evalInfoMask (uint64)
    ns: np.ndarray
    nch: np.ndarray
    lin: np.ndarray
    eco: np.ndarray
    seg: np.ndarray
    ts: np.ndarray        # timestamp ticks (2.5 ms)
    n_sync: int
    acqend: bool


@dataclass
class FreeRunningData:
    ksp: np.ndarray       # (n_echo, n_spoke, n_coil, n_samp) complex64
    noise: np.ndarray     # (n_coil, n_samp) complex64
    lin: np.ndarray       # (n_spoke,)
    seg: np.ndarray       # (n_spoke,) Seg counter (= interleave for this protocol)
    mask: np.ndarray      # (n_spoke*n_echo,) uint64, acquisition order
    eco: np.ndarray       # (n_spoke*n_echo,)
    ts_ticks: np.ndarray  # (n_spoke*n_echo,)
    protocol: str
    ascconv: dict = field(default_factory=dict)


def read_raid_header(f) -> list[Measurement]:
    f.seek(0)
    hd_size, count = struct.unpack("<II", f.read(8))
    if hd_size != 0:
        raise ValueError("not a VD/VE multi-raid twix file")
    out = []
    for _ in range(count):
        raw = f.read(152)
        meas_id, _file_id, off, length = struct.unpack("<IIQQ", raw[:24])
        out.append(Measurement(meas_id, off, length, raw[88:152].split(b"\0")[0].decode("latin1")))
    return out


def read_meas_header(f, offset: int) -> tuple[int, dict[str, bytes]]:
    f.seek(offset)
    hdr_len, n_buf = struct.unpack("<II", f.read(8))
    bufs = {}
    for _ in range(n_buf):
        name = bytearray()
        while (c := f.read(1)) != b"\0":
            name += c
        (blen,) = struct.unpack("<I", f.read(4))
        bufs[name.decode("latin1")] = f.read(blen)
    return hdr_len, bufs


def parse_ascconv(meas_yaps: bytes) -> dict[str, str]:
    text = meas_yaps.decode("latin1")
    m = re.search(r"### ASCCONV BEGIN.*?\n(.*?)### ASCCONV END", text, re.S)
    kv = {}
    for line in (m.group(1) if m else text).splitlines():
        mm = re.match(r"\s*(\S+)\s*=\s*(.+?)\s*(#.*)?$", line)
        if mm:
            kv[mm.group(1)] = mm.group(2).strip().strip('"')
    return kv


def index_scans(f, meas: Measurement, hdr_len: int) -> ScanIndex:
    pos, end = meas.offset + hdr_len, meas.offset + meas.length
    rows, n_sync, acqend = [], 0, False
    while pos + SCAN_HDR <= end:
        f.seek(pos)
        h = f.read(SCAN_HDR)
        dma = struct.unpack_from("<I", h, 0)[0] & 0x1FFFFFF
        mask = struct.unpack_from("<Q", h, 40)[0]
        if mask & MASK_ACQEND:
            acqend = True
            break
        if mask & MASK_SYNCDATA:
            n_sync += 1
            pos += dma
            continue
        ns, nch = struct.unpack_from("<HH", h, 48)
        lc = struct.unpack_from("<14H", h, 52)
        rows.append((pos, mask, ns, nch, lc[0], lc[4], lc[8], struct.unpack_from("<I", h, 12)[0]))
        pos += SCAN_HDR + nch * (CHAN_HDR + 8 * ns)
    a = np.array(rows, dtype=np.int64).reshape(-1, 8)
    return ScanIndex(pos=a[:, 0], mask=a[:, 1].astype(np.uint64), ns=a[:, 2], nch=a[:, 3],
                     lin=a[:, 4], eco=a[:, 5], seg=a[:, 6], ts=a[:, 7], n_sync=n_sync, acqend=acqend)


def read_adc(f, pos: int, ns: int, nch: int) -> np.ndarray:
    """(nch, ns) complex64 samples of the ADC packet whose scan header starts at `pos`."""
    f.seek(pos + SCAN_HDR)
    dt = np.dtype([("hdr", "V32"), ("data", "<c8", (ns,))])
    return np.frombuffer(f.read(nch * dt.itemsize), dtype=dt)["data"].copy()


def load_freerunning(path: str | Path, n_echo: int) -> FreeRunningData:
    """Last measurement of `path`: one leading noise ADC, then n_spoke x n_echo imaging ADCs in
    acquisition order (all echoes of one spoke consecutively)."""
    with open(path, "rb") as f:
        meas = read_raid_header(f)[-1]
        hdr_len, bufs = read_meas_header(f, meas.offset)
        idx = index_scans(f, meas, hdr_len)
        if not idx.acqend:
            raise ValueError("no ACQEND before the end of the measurement (truncated file?)")
        noise_rows = np.flatnonzero(idx.mask & np.uint64(MASK_NOISEADJ))
        if noise_rows.tolist() != [0]:
            raise ValueError(f"expected one noise ADC at row 0, got rows {noise_rows[:5].tolist()}")
        img = np.arange(1, idx.pos.size)
        ns, nch = int(idx.ns[1]), int(idx.nch[1])
        if not ((idx.ns[img] == ns).all() and (idx.nch[img] == nch).all()):
            raise ValueError("imaging ADCs do not all have the same size")
        if img.size % n_echo:
            raise ValueError(f"{img.size} imaging ADCs is not a multiple of {n_echo} echoes")
        n_spoke = img.size // n_echo
        eco = idx.eco[img]
        if not (eco.reshape(n_spoke, n_echo) == np.arange(n_echo)).all():
            raise ValueError("echo counter does not cycle 0..n_echo-1")
        lin = idx.lin[img].reshape(n_spoke, n_echo)
        if not (lin == lin[:, :1]).all():
            raise ValueError("Lin changes between the echoes of one spoke")
        ksp = np.empty((n_echo, n_spoke, nch, ns), np.complex64)
        for k, r in enumerate(img):
            ksp[k % n_echo, k // n_echo] = read_adc(f, int(idx.pos[r]), ns, nch)
        noise = read_adc(f, int(idx.pos[0]), int(idx.ns[0]), int(idx.nch[0]))
    return FreeRunningData(ksp=ksp, noise=noise, lin=lin[:, 0].copy(),
                           seg=idx.seg[img].reshape(n_spoke, n_echo)[:, 0].copy(),
                           mask=idx.mask[img], eco=eco, ts_ticks=idx.ts[img],
                           protocol=meas.protocol, ascconv=parse_ascconv(bufs.get("MeasYaps", b"")))
```

- [ ] **Step 6: 跑测试，确认通过**

Run: `cd /data0/congcong/code/GS/mcgs_cardiac_wt && /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/py -m pytest tests/test_cardiac_fr_twix.py -q`
Expected: `5 passed`

- [ ] **Step 7: 提交**

```bash
cd /data0/congcong/code/GS/mcgs_cardiac_wt && git add cardiac_fr/__init__.py cardiac_fr/twix.py tests/test_cardiac_fr_twix.py && git commit -m "cardiac_fr: twix VD/VE multi-raid reader for the free-running multi-echo data"
```

### Task 2: 轨迹（phyllotaxis 移植）

**Files:**
- Create: `cardiac_fr/trajectory.py`
- Test: `tests/test_cardiac_fr_trajectory.py`

**Interfaces:**
- Produces: `GOLDEN_ANGLE`；`phyllotaxis3d(n_shot, n_seg, self_nav=True) -> (polar, azim)`，两个 `(n_shot*n_seg,)` 数组，下标 = `Lin = shot*n_seg + seg`；`radial_positions(n_samp) -> (n_samp,)`，r = −0.5 + j/n_samp；`spoke_coordinates(n_shot, n_seg, n_samp, self_nav=True) -> (3, n_samp, n_shot*n_seg)`，r 单位，行 = x, y, z。BART 坐标 = 该数组 × `GRID`（由驱动脚本乘）。

- [ ] **Step 1: 写失败的测试** `tests/test_cardiac_fr_trajectory.py`

```python
import numpy as np

from cardiac_fr.trajectory import GOLDEN_ANGLE, phyllotaxis3d, radial_positions, spoke_coordinates


def test_small_case_matches_the_matlab_loop_by_hand():
    # phyllotaxis3D.m with 2 shots x 3 segments: kost = pi / (2 sqrt(6 - 2)) = pi / 4.
    # The loop runs segment-outer, shot-inner; index = seg + shot * 3; seg 0 is the SI spoke.
    polar, azim = phyllotaxis3d(n_shot=2, n_seg=3, self_nav=True)
    k = np.pi / 4
    np.testing.assert_allclose(polar, [0, k, k * np.sqrt(3), 0, k * np.sqrt(2), 2 * k])
    tp = 2 * np.pi
    np.testing.assert_allclose(azim, [0, GOLDEN_ANGLE % tp, (3 * GOLDEN_ANGLE) % tp,
                                      0, (2 * GOLDEN_ANGLE) % tp, (4 * GOLDEN_ANGLE) % tp])


def test_protocol_size_si_spokes_and_ranges():
    polar, azim = phyllotaxis3d(1000, 22, True)
    si = np.arange(22000) % 22 == 0
    assert (polar[si] == 0).all() and (azim[si] == 0).all()
    assert (polar[~si] > 0).all() and polar.max() <= np.pi / 2 + 1e-12
    assert (azim >= 0).all() and (azim < 2 * np.pi).all()


def test_radial_positions_center_and_spacing():
    r = radial_positions(224)
    assert r[0] == -0.5 and r[112] == 0.0 and np.isclose(r[-1], 0.5 - 1 / 224)
    np.testing.assert_allclose(np.diff(r), 1 / 224)


def test_coordinates_lie_on_spokes_through_the_center():
    k = spoke_coordinates(n_shot=10, n_seg=22, n_samp=224)
    assert k.shape == (3, 224, 220)
    r = radial_positions(224)
    np.testing.assert_allclose(np.linalg.norm(k, axis=0), np.repeat(np.abs(r)[:, None], 220, 1), atol=1e-12)
    np.testing.assert_allclose(k[:, :, 0], np.stack([0 * r, 0 * r, r]), atol=1e-15)   # SI spoke on z
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `cd /data0/congcong/code/GS/mcgs_cardiac_wt && /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/py -m pytest tests/test_cardiac_fr_trajectory.py -q`
Expected: `ModuleNotFoundError: No module named 'cardiac_fr.trajectory'`

- [ ] **Step 3: 写 `cardiac_fr/trajectory.py`**

```python
"""3D radial spiral-phyllotaxis trajectory (Piccini et al., MRM 2011), ported line by line from the
authors' phyllotaxis3D.m and computePhyllotaxis.m (QIS-MRI/ReadDataAndTrajectory_FreeRunningFatFractionHeart
@ a78f6f2). Index = acquisition line Lin = shot * n_seg + seg; seg 0 of every shot is the SI spoke."""
from __future__ import annotations

import numpy as np

GOLDEN = (1 + np.sqrt(5)) / 2
GOLDEN_ANGLE = 2 * np.pi - 2 * np.pi / GOLDEN


def phyllotaxis3d(n_shot: int, n_seg: int, self_nav: bool = True) -> tuple[np.ndarray, np.ndarray]:
    n_proj = n_shot * n_seg
    kost = np.pi / (2 * np.sqrt(n_proj - n_shot)) if self_nav else np.pi / (2 * np.sqrt(n_proj))
    polar = np.zeros(n_proj)
    azim = np.zeros(n_proj)
    count = 1
    for lk in range(1, n_seg + 1):              # MATLAB: for lk = 1:m_lProjectionsPerFrame
        for frame in range(1, n_shot + 1):      # MATLAB: for lFrame = 1:m_lNumberOfFrames
            i = (lk - 1) + (frame - 1) * n_seg
            if self_nav and lk == 1:
                continue                        # SI spoke: polar = azimuth = 0, count not advanced
            polar[i] = kost * np.sqrt(count)
            azim[i] = np.mod(count * GOLDEN_ANGLE, 2 * np.pi)
            count += 1
    return polar, azim


def radial_positions(n_samp: int) -> np.ndarray:
    """computePhyllotaxis.m: r = -0.5 : 1/N : 0.5 - 1/N (sample n_samp/2 is the k-space centre)."""
    return -0.5 + np.arange(n_samp) / n_samp


def spoke_coordinates(n_shot: int = 1000, n_seg: int = 22, n_samp: int = 224,
                      self_nav: bool = True) -> np.ndarray:
    """(3, n_samp, n_shot*n_seg) in r units; rows x, y, z = MATLAB sph2cart(azimuth, pi/2 - polar, r)."""
    polar, azim = phyllotaxis3d(n_shot, n_seg, self_nav)
    r = radial_positions(n_samp)[:, None]
    elev = np.pi / 2 - polar
    return np.stack([r * np.cos(elev) * np.cos(azim),
                     r * np.cos(elev) * np.sin(azim),
                     r * np.sin(elev) + 0 * azim])
```

- [ ] **Step 4: 跑测试，确认通过**

Run: 同 Step 2。Expected: `4 passed`

- [ ] **Step 5: 提交**

```bash
cd /data0/congcong/code/GS/mcgs_cardiac_wt && git add cardiac_fr/trajectory.py tests/test_cardiac_fr_trajectory.py && git commit -m "cardiac_fr: phyllotaxis trajectory ported from the authors' MATLAB"
```

### Task 3: 生理信号、稳态与呼吸门控

**Files:**
- Create: `cardiac_fr/physio.py`
- Test: `tests/test_cardiac_fr_physio.py`

**Interfaces:**
- Produces: `load_physio(path) -> dict[str, np.ndarray]`（float64）；`timestamp_mismatch(ts_ms, ts_ticks) -> float`（逐行 |TimeStamp_ms − (ticks − ticks[0])×2.5| 的最大值，行数不同抛 `ValueError`）；`per_spoke(values, n_echo)`；`spoke_time_ms(ts_ms, n_echo)`；`cardiac_phase(t_ms, triggers_ms)`（RR 内相位，首个触发前与末个触发后为 NaN）；`expiration_end_by_histogram(resp, bins=50) -> "high" | "low"`；`third_at_end(resp, end, frac=1/3) -> bool 掩膜`；`spoke_table(physio, lin, n_echo, n_seg, t_steady_ms) -> dict`，键 `lin, shot, seg_in_shot, time_ms, resp, card_phase, is_si, steady`。

- [ ] **Step 1: 写失败的测试** `tests/test_cardiac_fr_physio.py`

```python
import numpy as np
import pytest

from cardiac_fr.physio import (cardiac_phase, expiration_end_by_histogram, per_spoke, spoke_table,
                               third_at_end, timestamp_mismatch)


def test_timestamp_mismatch_zero_and_shape_check():
    ticks = np.array([1000, 1001, 1001, 1003], dtype=np.int64)
    assert timestamp_mismatch(np.array([0, 2.5, 2.5, 7.5]), ticks) == 0.0
    assert timestamp_mismatch(np.array([0, 2.5, 5.0, 7.5]), ticks) == 2.5
    with pytest.raises(ValueError):
        timestamp_mismatch(np.zeros(3), ticks)


def test_per_spoke_mean_over_echoes():
    np.testing.assert_allclose(per_spoke(np.arange(6.0), n_echo=3), [1.0, 4.0])


def test_cardiac_phase():
    ph = cardiac_phase(np.array([50, 100, 150, 300, 400, 500.0]), np.array([100, 200, 400.0]))
    np.testing.assert_allclose(ph, [np.nan, 0.0, 0.5, 0.5, np.nan, np.nan])


def test_expiration_end_by_histogram():
    rng = np.random.default_rng(0)
    plateau_high = np.r_[1 + 0.01 * rng.standard_normal(700), np.linspace(0, 1, 300)]
    assert expiration_end_by_histogram(plateau_high) == "high"
    assert expiration_end_by_histogram(-plateau_high) == "low"


def test_third_at_end_picks_extremes():
    v = np.array([5, 1, 4, 2, 6, 3.0])
    assert third_at_end(v, "high").nonzero()[0].tolist() == [0, 4]
    assert third_at_end(v, "low").nonzero()[0].tolist() == [1, 3]


def test_spoke_table_flags():
    ph = {"TimeStamp_ms": np.repeat(np.arange(6) * 5000.0, 2),
          "respSignal": np.repeat(np.arange(6.0), 2), "cardTriggers": np.array([0.0, 100000.0])}
    t = spoke_table(ph, np.arange(6), n_echo=2, n_seg=3, t_steady_ms=10000)
    assert t["is_si"].tolist() == [True, False, False, True, False, False]
    assert t["steady"].tolist() == [False, False, True, True, True, True]
    assert t["shot"].tolist() == [0, 0, 0, 1, 1, 1]
    np.testing.assert_allclose(t["resp"], np.arange(6.0))
    np.testing.assert_allclose(t["card_phase"], np.arange(6) * 0.05)
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `cd /data0/congcong/code/GS/mcgs_cardiac_wt && /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/py -m pytest tests/test_cardiac_fr_physio.py -q`
Expected: `ModuleNotFoundError: No module named 'cardiac_fr.physio'`

- [ ] **Step 3: 写 `cardiac_fr/physio.py`**

```python
"""PhysioInfo.mat (pilot tone) <-> imaging ADCs, steady state, respiratory gating (spec 5.4).

Row i of TimeStamp_ms / respSignal is imaging ADC i after the noise ADC is removed; rows are
spoke-major, echo-minor (verified on V07 against the MDH timestamps, 2.5 ms ticks)."""
from __future__ import annotations

import numpy as np
import scipy.io as sio


def load_physio(path) -> dict[str, np.ndarray]:
    m = sio.loadmat(str(path), squeeze_me=True)
    return {k: np.asarray(v, dtype=np.float64) for k, v in m.items() if not k.startswith("__")}


def timestamp_mismatch(ts_ms: np.ndarray, ts_ticks: np.ndarray) -> float:
    t = (np.asarray(ts_ticks, dtype=np.int64) - int(ts_ticks[0])) * 2.5
    if t.shape != np.shape(ts_ms):
        raise ValueError(f"row count differs: {np.shape(ts_ms)} vs {t.shape}")
    return float(np.max(np.abs(np.asarray(ts_ms) - t)))


def per_spoke(values: np.ndarray, n_echo: int) -> np.ndarray:
    return np.asarray(values).reshape(-1, n_echo).mean(axis=1)


def spoke_time_ms(ts_ms: np.ndarray, n_echo: int) -> np.ndarray:
    return np.asarray(ts_ms).reshape(-1, n_echo)[:, 0].copy()


def cardiac_phase(t_ms: np.ndarray, triggers_ms: np.ndarray) -> np.ndarray:
    trig = np.sort(np.asarray(triggers_ms, dtype=np.float64))
    k = np.searchsorted(trig, t_ms, side="right") - 1
    ph = np.full(np.shape(t_ms), np.nan)
    ok = (k >= 0) & (k < trig.size - 1)
    ph[ok] = (t_ms[ok] - trig[k[ok]]) / (trig[k[ok] + 1] - trig[k[ok]])
    return ph


def expiration_end_by_histogram(resp: np.ndarray, bins: int = 50) -> str:
    """Criterion (a): end-expiration is the longest plateau, so the histogram mode sits at that end.
    'high' if the mode lies above the robust mid-range (P2 + P98) / 2, else 'low'."""
    hist, edges = np.histogram(resp, bins=bins)
    k = int(np.argmax(hist))
    mode = 0.5 * (edges[k] + edges[k + 1])
    mid = 0.5 * (np.percentile(resp, 2) + np.percentile(resp, 98))
    return "high" if mode > mid else "low"


def third_at_end(resp: np.ndarray, end: str, frac: float = 1 / 3) -> np.ndarray:
    n_keep = int(round(frac * resp.size))
    order = np.argsort(resp, kind="stable")
    keep = order[-n_keep:] if end == "high" else order[:n_keep]
    m = np.zeros(resp.size, dtype=bool)
    m[keep] = True
    return m


def spoke_table(physio: dict, lin: np.ndarray, n_echo: int, n_seg: int, t_steady_ms: float) -> dict:
    t = spoke_time_ms(physio["TimeStamp_ms"], n_echo)
    lin = np.asarray(lin)
    return {"lin": lin, "shot": lin // n_seg, "seg_in_shot": lin % n_seg, "time_ms": t,
            "resp": per_spoke(physio["respSignal"], n_echo),
            "card_phase": cardiac_phase(t, physio["cardTriggers"]),
            "is_si": (lin % n_seg) == 0, "steady": t >= t_steady_ms}
```

- [ ] **Step 4: 跑测试，确认通过**

Run: 同 Step 2。Expected: `6 passed`

- [ ] **Step 5: 提交**

```bash
cd /data0/congcong/code/GS/mcgs_cardiac_wt && git add cardiac_fr/physio.py tests/test_cardiac_fr_physio.py && git commit -m "cardiac_fr: PhysioInfo mapping, steady state and respiratory gating"
```

### Task 4: BART 封装

**Files:**
- Create: `cardiac_fr/bart.py`
- Test: `tests/test_cardiac_fr_bart.py`

**Interfaces:**
- Produces: `write_cfl(base, arr)`、`read_cfl(base) -> ndarray`（BART 自带 `cfl.py`，列主序，去掉末尾长度 1 的维）；`run(args, log=None) -> str`（线程取 `CARDIAC_FR_THREADS`，默认 32；返回码非 0 抛 `RuntimeError`）；`bart_ksp(spokes)`：`(n_spoke, n_coil, n_samp) -> (1, n_samp, n_spoke, n_coil)`；`bart_traj(k)`：`(3, n_samp, n_spoke)` 网格单位 → complex64；`parse_scaling(text) -> float`；`pics(ksp, sens, out, traj, reg, lam, iters, w=None, debug=False, log=None) -> {"seconds": float, "scaling": float | None}`；`coil_maps(traj, ksp, work, out_sens, grid, calib, log=None)`；`forward(traj, img, sens, out, work, log=None)`（中间文件名随 `out` 走，可并发）。

- [ ] **Step 1: 写失败的测试** `tests/test_cardiac_fr_bart.py`

```python
import os

import numpy as np
import pytest

os.environ.setdefault("CARDIAC_FR_THREADS", "4")
from cardiac_fr import bart  # noqa: E402


@pytest.fixture(scope="module")
def phantom(tmp_path_factory):
    d = tmp_path_factory.mktemp("bart")
    bart.run(["traj", "-x", 32, "-y", 400, "-r", "-3", "-G", d / "traj"])
    bart.run(["phantom", "-3", "-x", 16, "-s", 4, "-S", 4, d / "sens_true"])
    bart.run(["phantom", "-3", "-x", 16, d / "img_true"])
    bart.run(["fmac", d / "img_true", d / "sens_true", d / "cimg_true"])
    bart.run(["nufft", d / "traj", d / "cimg_true", d / "ksp"])
    bart.coil_maps(d / "traj", d / "ksp", d, d / "sens", grid=16, calib=12)
    return d


def _nrmse(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def test_parse_scaling():
    txt = "Scaling: 91008.4 (max = 1/p90 = 2/median = 3)\nInverse scaling of the data: 91008.421875\n"
    assert bart.parse_scaling(txt) == pytest.approx(91008.421875)
    with pytest.raises(ValueError):
        bart.parse_scaling("conjugate gradients\n")


def test_ksp_layout_roundtrip(tmp_path):
    y = (np.arange(5 * 3 * 7) + 1j).reshape(5, 3, 7).astype(np.complex64)   # (spoke, coil, sample)
    bart.write_cfl(tmp_path / "k", bart.bart_ksp(y))
    assert (tmp_path / "k.hdr").read_text().split("\n")[1].split()[:4] == ["1", "7", "5", "3"]
    np.testing.assert_array_equal(bart.read_cfl(tmp_path / "k")[0, :, 2, 1], y[2, 1, :])


def test_pics_S_output_is_in_data_units(phantom):
    d = phantom
    bart.pics(d / "ksp", d / "sens", d / "x1", d / "traj", reg="l2", lam=1e-3, iters=30, w=1.0)
    bart.pics(d / "ksp", d / "sens", d / "x2", d / "traj", reg="l2", lam=1e-3, iters=30, w=2.0)
    assert _nrmse(bart.read_cfl(d / "x2"), bart.read_cfl(d / "x1")) < 1e-3
    bart.forward(d / "traj", d / "x1", d / "sens", d / "y1", d)
    assert _nrmse(bart.read_cfl(d / "y1"), bart.read_cfl(d / "ksp")) < 0.05


def test_pics_debug_reports_scaling(phantom):
    d = phantom
    r = bart.pics(d / "ksp", d / "sens", d / "x3", d / "traj", reg="l2", lam=1e-3, iters=5, debug=True)
    assert r["scaling"] > 0 and r["seconds"] > 0


def test_pics_l1_is_stable_on_radial(phantom):
    d = phantom
    bart.pics(d / "ksp", d / "sens", d / "x4", d / "traj", reg="l1", lam=1e-2, iters=100)
    bart.forward(d / "traj", d / "x4", d / "sens", d / "y4", d)
    y4 = bart.read_cfl(d / "y4")
    assert np.isfinite(y4).all() and _nrmse(y4, bart.read_cfl(d / "ksp")) < 0.1
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `cd /data0/congcong/code/GS/mcgs_cardiac_wt && /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/py -m pytest tests/test_cardiac_fr_bart.py -q`
Expected: `ImportError: cannot import name 'bart' from 'cardiac_fr'`

- [ ] **Step 3: 写 `cardiac_fr/bart.py`**

```python
"""Thin wrappers around the BART command line (CPU build v1.0.00) and BART's own cfl reader.

Verified on a 16^3 phantom (2026-09-10): with -S the pics image is in data units (independent of -w;
bart nufft of sens * image reproduces the measured k-space with scale 1); without -S it is divided by
BART's internal scaling. Non-Cartesian -l1 (FISTA) diverges unless -e sets the step size.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BART = "/home/congcongliu/src/bart/bart"
sys.path.insert(0, "/home/congcongliu/src/bart/python")
import cfl  # noqa: E402  (column-major .cfl/.hdr)

_SCALING = re.compile(r"Inverse scaling of the data:\s*([0-9.eE+-]+)")


def write_cfl(base, arr) -> None:
    cfl.writecfl(str(base), np.asarray(arr, dtype=np.complex64))


def read_cfl(base) -> np.ndarray:
    return cfl.readcfl(str(base))


def run(args, log=None) -> str:
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("CARDIAC_FR_THREADS", "32"))
    t0 = time.time()
    res = subprocess.run([BART, *map(str, args)], capture_output=True, text=True, env=env)
    out = res.stdout + res.stderr
    if log is not None:
        with open(log, "a") as fh:
            fh.write(f"$ bart {' '.join(map(str, args))}\n{out}[{time.time() - t0:.1f} s, rc={res.returncode}]\n")
    if res.returncode != 0:
        raise RuntimeError(f"bart {args[0]} failed (rc={res.returncode}):\n{out[-2000:]}")
    return out


def bart_ksp(spokes: np.ndarray) -> np.ndarray:
    """(n_spoke, n_coil, n_samp) -> BART non-Cartesian layout (1, n_samp, n_spoke, n_coil)."""
    return np.ascontiguousarray(np.asarray(spokes).transpose(2, 0, 1))[None]


def bart_traj(k: np.ndarray) -> np.ndarray:
    """(3, n_samp, n_spoke) coordinates in grid units -> complex64 (BART stores trajectories as cfl)."""
    return np.asarray(k).astype(np.complex64)


def parse_scaling(output: str) -> float:
    m = _SCALING.search(output)
    if m is None:
        raise ValueError("no 'Inverse scaling of the data' line (run pics with -d5)")
    return float(m.group(1))


def pics(ksp, sens, out, traj, reg: str, lam: float, iters: int, w: float | None = None,
         debug: bool = False, log=None) -> dict:
    args = ["pics", "-t", traj, "-l1" if reg == "l1" else "-l2", "-r", lam, "-i", iters, "-S"]
    if reg == "l1":
        args.append("-e")
    if w is not None:
        args += ["-w", repr(float(w))]
    if debug:
        args.append("-d5")
    t0 = time.time()
    txt = run([*args, ksp, sens, out], log=log)
    return {"seconds": time.time() - t0, "scaling": parse_scaling(txt) if debug else None}


def coil_maps(traj, ksp, work, out_sens, grid: int, calib: int, log=None) -> None:
    """Spec 5.5: inverse NUFFT to coil images, unitary FFT, ESPIRiT with the brain-topic flags."""
    work = Path(work)
    run(["nufft", "-i", "-x", f"{grid}:{grid}:{grid}", traj, ksp, work / "coilimg"], log=log)
    run(["fft", "-u", "7", work / "coilimg", work / "kgrid"], log=log)
    run(["ecalib", "-t", "0.001", "-c", "0.0", "-r", calib, "-m", "1", work / "kgrid", out_sens], log=log)


def forward(traj, img, sens, out, work, log=None) -> None:
    """k-space predicted by image `img`: sens * img, then NUFFT onto `traj`."""
    tmp = Path(work) / f"{Path(out).name}_coilimg"
    run(["fmac", img, sens, tmp], log=log)
    run(["nufft", traj, tmp, out], log=log)
```

- [ ] **Step 4: 跑测试，确认通过**

Run: 同 Step 2。Expected: `5 passed`（BART 小体模，约半分钟）

- [ ] **Step 5: 提交**

```bash
cd /data0/congcong/code/GS/mcgs_cardiac_wt && git add cardiac_fr/bart.py tests/test_cardiac_fr_bart.py && git commit -m "cardiac_fr: BART wrappers (cfl io, pics with -S/-w/-e, coil maps, forward model)"
```

### Task 5: 评估口径库

**Files:**
- Create: `cardiac_fr/gauge.py`
- Test: `tests/test_cardiac_fr_gauge.py`

**Interfaces:**
- Consumes（仓库已有，不改）：`paired_gs.metrics.metrics.evaluate_reconstruction(pred, target, compute_extended=False)`；`diag_artefact.headonly.head_mask, head_psnr, head_ssim, head_hfen`。
- Produces: `N_NYQ = 19704`；`af(n) -> float`；`heldout_split(ids, frac=0.05, seed=0) -> (heldout, pool)`，都已排序；`shot_subset(pool, target_af, n_seg=22) -> ids`；`norm_scale(ref_mag) -> float`（P99.9）；`box_mask(shape, box)`；`image_metrics(rec_mag, ref_mag, heart_box=None) -> dict`，键 `psnr, ssim, nmse, body_psnr, body_ssim, body_hfen, body_frac`，给了框再加 `heart_psnr, heart_ssim`；`kspace_nrmse(pred, meas)`；`error_map(rec_mag, ref_mag)`；`monotone_worse(values_by_af, higher_is_better) -> bool`。

- [ ] **Step 1: 写失败的测试** `tests/test_cardiac_fr_gauge.py`

```python
import numpy as np
import pytest

from cardiac_fr.gauge import (N_NYQ, af, error_map, heldout_split, image_metrics, kspace_nrmse,
                              monotone_worse, norm_scale, shot_subset)


def test_nyquist_and_af():
    assert N_NYQ == 19704
    assert af(19704) == 1.0 and af(9852) == 2.0


def test_heldout_split_is_disjoint_and_deterministic():
    ids = np.arange(0, 3000, 3)
    ho, pool = heldout_split(ids, 0.05, seed=0)
    assert ho.size == 50 and pool.size == 950
    assert np.intersect1d(ho, pool).size == 0 and np.array_equal(np.union1d(ho, pool), ids)
    assert np.array_equal(ho, heldout_split(ids, 0.05, seed=0)[0])
    assert not np.array_equal(ho, heldout_split(ids, 0.05, seed=1)[0])


def test_shot_subset_takes_whole_shots_close_to_target():
    lin = np.arange(22000)
    pool = lin[(lin % 22 != 0) & (lin % 7 != 3)]          # non-SI, some spokes missing
    sub = shot_subset(pool, target_af=12)
    assert abs(sub.size - N_NYQ / 12) <= 21
    assert np.isin(sub, pool).all()
    for s in np.unique(sub // 22):                         # every pool spoke of a chosen shot is in
        np.testing.assert_array_equal(np.sort(sub[sub // 22 == s]), pool[pool // 22 == s])


def _blob(n=24):
    x = np.linspace(-1, 1, n)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return np.exp(-3 * (X ** 2 + Y ** 2 + Z ** 2)).astype(np.float32)


def test_image_metrics_order_with_noise():
    ref, rng = _blob(), np.random.default_rng(0)
    small = ref + 0.01 * rng.standard_normal(ref.shape).astype(np.float32)
    large = ref + 0.05 * rng.standard_normal(ref.shape).astype(np.float32)
    box = ((6, 18), (6, 18), (6, 18))
    ms, ml = image_metrics(small, ref, heart_box=box), image_metrics(large, ref, heart_box=box)
    assert ms["body_psnr"] > ml["body_psnr"] and ms["body_ssim"] > ml["body_ssim"]
    assert ms["body_hfen"] < ml["body_hfen"] and ms["nmse"] < ml["nmse"]
    assert ms["heart_psnr"] > ml["heart_psnr"] and 0 < ms["body_frac"] <= 1


def test_kspace_nrmse_and_error_map():
    y = np.ones(10, np.complex64)
    assert kspace_nrmse(y, y) == 0.0 and kspace_nrmse(2 * y, y) == pytest.approx(1.0)
    ref = _blob()
    assert error_map(ref * 1.1, ref).max() == pytest.approx(0.1 * ref.max() / norm_scale(ref), rel=1e-5)


def test_monotone_worse():
    assert monotone_worse({8: 30.0, 12: 28.0, 16: 27.5}, higher_is_better=True)
    assert not monotone_worse({8: 30.0, 12: 30.5, 16: 27.5}, higher_is_better=True)
    assert monotone_worse({8: 0.1, 12: 0.2, 16: 0.3}, higher_is_better=False)
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `cd /data0/congcong/code/GS/mcgs_cardiac_wt && /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/py -m pytest tests/test_cardiac_fr_gauge.py -q`
Expected: `ModuleNotFoundError: No module named 'cardiac_fr.gauge'`

- [ ] **Step 3: 写 `cardiac_fr/gauge.py`**

```python
"""Evaluation gauge for the free-running data (spec 5.7).

AF is relative to the 3D radial Nyquist count pi/2 * 112^2 = 19,704 spokes. Image metrics reuse the
brain-topic implementations unchanged: whole-volume PSNR/SSIM/NMSE from paired_gs.metrics, and the
masked PSNR/SSIM/HFEN from diag_artefact.headonly with its |ref| > 0.05 P99.9 mask (called "body"
here). The ranking metric is the held-out k-space NRMSE; the caller computes the prediction with BART.
"""
from __future__ import annotations

import numpy as np
import torch

from diag_artefact.headonly import head_hfen, head_mask, head_psnr, head_ssim
from paired_gs.metrics.metrics import evaluate_reconstruction

N_NYQ = int(round(np.pi / 2 * 112 ** 2))


def af(n_spokes: int) -> float:
    return N_NYQ / n_spokes


def heldout_split(ids, frac: float = 0.05, seed: int = 0):
    ids = np.sort(np.asarray(ids))
    ho = np.sort(np.random.default_rng(seed).choice(ids, size=int(round(frac * ids.size)), replace=False))
    return ho, np.setdiff1d(ids, ho)


def shot_subset(pool, target_af: float, n_seg: int = 22) -> np.ndarray:
    """Whole interleaves from `pool`: the j-th chosen interleave is shots[round(j * step)], with the
    step (grid 1e-3) whose spoke count is closest to N_NYQ / target_af; ties go to the smaller step."""
    pool = np.sort(np.asarray(pool))
    shots, counts = np.unique(pool // n_seg, return_counts=True)
    target = N_NYQ / target_af
    s0 = pool.size / target
    best = None
    for step in np.arange(max(1.0, 0.7 * s0), max(1.0, 1.3 * s0) + 1e-9, 1e-3):
        pick = np.unique(np.round(np.arange(0.0, shots.size / step) * step).astype(np.int64))
        pick = pick[pick < shots.size]
        err = abs(int(counts[pick].sum()) - target)
        if best is None or err < best[0]:
            best = (err, pick)
    return pool[np.isin(pool // n_seg, shots[best[1]])]


def norm_scale(ref_mag) -> float:
    return float(np.quantile(np.asarray(ref_mag), 0.999))


def box_mask(shape, box) -> np.ndarray:
    (x0, x1), (y0, y1), (z0, z1) = box
    m = np.zeros(shape, dtype=bool)
    m[x0:x1, y0:y1, z0:z1] = True
    return m


def image_metrics(rec_mag, ref_mag, heart_box=None) -> dict:
    """Both inputs already divided by the reference P99.9 (spec 5.7)."""
    rec = np.ascontiguousarray(rec_mag, dtype=np.float32)
    ref = np.ascontiguousarray(ref_mag, dtype=np.float32)
    met = evaluate_reconstruction(torch.from_numpy(rec), torch.from_numpy(ref), compute_extended=False)
    body, gmax = head_mask(ref), float(ref.max())
    out = {"psnr": float(met["psnr"]), "ssim": float(met["ssim"]), "nmse": float(met["nmse"]),
           "body_psnr": head_psnr(rec, ref, body, gmax), "body_ssim": head_ssim(rec, ref, body, gmax),
           "body_hfen": head_hfen(rec, ref, body, gmax), "body_frac": float(body.mean())}
    if heart_box is not None:
        hb = box_mask(ref.shape, heart_box)
        out["heart_psnr"] = head_psnr(rec, ref, hb, gmax)
        out["heart_ssim"] = head_ssim(rec, ref, hb, gmax)
    return out


def kspace_nrmse(pred, meas) -> float:
    meas = np.asarray(meas)
    return float(np.linalg.norm(np.asarray(pred) - meas) / np.linalg.norm(meas))


def error_map(rec_mag, ref_mag) -> np.ndarray:
    """|rec - ref| / P99.9(|ref|); display with jet on a fixed [0, 0.1] range."""
    return np.abs(np.asarray(rec_mag) - np.asarray(ref_mag)) / norm_scale(ref_mag)


def monotone_worse(values_by_af: dict, higher_is_better: bool) -> bool:
    v = [values_by_af[a] for a in sorted(values_by_af)]
    return all((b < a) if higher_is_better else (b > a) for a, b in zip(v, v[1:]))
```

- [ ] **Step 4: 跑测试，确认通过**

Run: 同 Step 2。Expected: `6 passed`

- [ ] **Step 5: 跑全部心脏测试并提交**

Run: `cd /data0/congcong/code/GS/mcgs_cardiac_wt && /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/py -m pytest tests/test_cardiac_fr_twix.py tests/test_cardiac_fr_trajectory.py tests/test_cardiac_fr_physio.py tests/test_cardiac_fr_bart.py tests/test_cardiac_fr_gauge.py -q`
Expected: `26 passed`

```bash
cd /data0/congcong/code/GS/mcgs_cardiac_wt && git add cardiac_fr/gauge.py tests/test_cardiac_fr_gauge.py && git commit -m "cardiac_fr: evaluation gauge (AF, held-out split, whole-shot subsets, metrics)"
```

### Task 6: 读四例原始数据、读出首尾检查、与作者读取器逐位核对

**Files:**
- Create（课题 `scripts/`）：`s01_read.py`、`run_read_all.sh`、`s02_edge_check.py`、`xcheck_reader.m`、`xcheck_reader_compare.py`
- Mirror: `cardiac_freerunning_megre/scripts/`

**Interfaces:**
- Consumes: `cardiac_fr.twix.load_freerunning`；`cardiac_fr` 常数。
- Produces: 派生数据 `<Vxx>/raw.h5`（`ksp` (8, 22000, 24, 224) complex64，分块 (1, 256, 24, 224)；`lin`、`seg` (22000,)；`mask`、`eco`、`ts_ticks` (176000,)）、`<Vxx>/noise.npy` (24, 224)、`<Vxx>/meta.json`；课题 `data/<Vxx>` 软链接；日志 `logs/s01_read_all.log`、`logs/meta_<Vxx>.json`、`logs/s02_edge_check.txt`、`logs/xcheck_reader_V07.txt`。

- [ ] **Step 1: 写 `s01_read.py`**

```python
"""Spec 5.1: read one subject's free-running 8-echo twix file into raw.h5 (+ noise.npy, meta.json)."""
import json
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr import DERIVED_ROOT, N_COIL, N_ECHO, N_SAMP, N_SEG, N_SPOKE, RAW_ROOT, TOPIC  # noqa: E402
from cardiac_fr.twix import load_freerunning  # noqa: E402

KEYS = ("tProtocolName", "sKSpace.lBaseResolution", "sKSpace.lPartitions", "sKSpace.lRadialViews",
        "lContrasts", "alTR[0]", "alTE[0]", "alTE[1]", "alTE[7]", "adFlipAngleDegree[0]",
        "sRXSPEC.alDwellTime[0]")

s = sys.argv[1]
src = RAW_ROOT / s / "freerunning_8echo.dat"
out = DERIVED_ROOT / s
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
d = load_freerunning(src, n_echo=N_ECHO)
assert d.ksp.shape == (N_ECHO, N_SPOKE, N_COIL, N_SAMP), d.ksp.shape
assert d.noise.shape == (N_COIL, N_SAMP), d.noise.shape
assert (d.lin == np.arange(N_SPOKE)).all(), "Lin is not 0..21999 in acquisition order"
assert (d.seg == d.lin // N_SEG).all(), "Seg counter is not the interleave index"
si_flag = ((d.mask >> np.uint64(34)) & np.uint64(1)).astype(bool)
assert (si_flag == (np.repeat(d.lin, N_ECHO) % N_SEG == 0)).all(), "MDH bit 34 is not exactly the seg-0 spokes"
with h5py.File(out / "raw.h5", "w") as h:
    h.create_dataset("ksp", data=d.ksp, chunks=(1, 256, N_COIL, N_SAMP))
    for k in ("lin", "seg", "mask", "eco", "ts_ticks"):
        h.create_dataset(k, data=getattr(d, k))
np.save(out / "noise.npy", d.noise)
meta = {"subject": s, "source": str(src.resolve()), "protocol": d.protocol,
        "ksp_shape": list(d.ksp.shape), "n_adc_imaging": int(d.eco.size),
        "seconds": round(time.time() - t0, 1), "ascconv": {k: d.ascconv.get(k) for k in KEYS}}
txt = json.dumps(meta, indent=1)
(out / "meta.json").write_text(txt)
(TOPIC / "logs" / f"meta_{s}.json").write_text(txt)
print(txt, flush=True)
```

- [ ] **Step 2: 写 `run_read_all.sh` 并脱离会话启动**

```bash
cat > /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/run_read_all.sh <<'EOF'
#!/usr/bin/env bash
# Spec 5.1: raw.h5 for the four subjects, one after the other (a few minutes each).
set -euo pipefail
T=/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts
for S in V07 V03 V04 V08; do
  "$T/py" "$T/s01_read.py" "$S"
done
echo ALL_READ_DONE
EOF
setsid nohup bash /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/run_read_all.sh > /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/logs/s01_read_all.log 2>&1 < /dev/null &
```

- [ ] **Step 3: 等它跑完并核对**

Run（每隔几分钟看一次，直到最后一行是 `ALL_READ_DONE`）：`tail -4 /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/logs/s01_read_all.log`
Expected: 日志里四段 JSON，每段 `"ksp_shape": [8, 22000, 24, 224]`、`"n_adc_imaging": 176000`、`"protocol": "FRFF_MEGRE_2P0MM_8XC_BW893_FA12_DTE2P05_MONO_22x1000"`、`"sKSpace.lRadialViews": "22000"`、`"lContrasts": "8"`；没有 `AssertionError`/`Traceback`。再跑 `ls -l /data2/congcong/data/cardiac_3d_cine_public/derived/megre/V0*/raw.h5`，四个文件各约 7.57 GB。

- [ ] **Step 4: 写 `s02_edge_check.py` 并运行（spec 5.2）**

```python
"""Spec 5.2: mean |s| per readout sample over 250 mid-scan spokes x 8 echoes x 24 coils. Samples above
1.2x the kept-edge level must all lie in the dropped set {0, 1, 2, 221, 222, 223}."""
import sys

import h5py
import numpy as np

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr import DERIVED_ROOT, SUBJECTS  # noqa: E402

DROPPED = {0, 1, 2, 221, 222, 223}
ok_all = True
for s in sys.argv[1:] or SUBJECTS:
    with h5py.File(DERIVED_ROOT / s / "raw.h5", "r") as h:
        m = np.abs(h["ksp"][:, 10000:10250]).mean(axis=(0, 1, 2))
    level = float(np.median(np.r_[m[3:9], m[215:221]]))
    flagged = [j for j in [*range(0, 8), *range(216, 224)] if m[j] > 1.2 * level]
    ok = set(flagged) <= DROPPED
    ok_all &= ok
    print(f"{s}: head/level {np.round(m[:5] / level, 2).tolist()}  tail/level "
          f"{np.round(m[219:] / level, 2).tolist()}  flagged {flagged}  {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok_all else 1)
```

Run: `/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/py /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/s02_edge_check.py > /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/logs/s02_edge_check.txt; echo exit=$?; cat /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/logs/s02_edge_check.txt`
Expected: `exit=0`，四行都是 `PASS`；V07 的 `flagged [0, 1, 2]`（2026-09-10 实测前三点为正常水平的 18、4.4、1.5 倍）。任何一行 `FAIL` 就停，报告（停止条件 5.2）。

- [ ] **Step 5: 写作者读取器的交叉核对（spec 5.1）**

`xcheck_reader.m`：

```matlab
function xcheck_reader(datfile, idxfile, outfile)
% Spec 5.1: read the imaging lines listed (1-based, acquisition order) in idxfile with the authors'
% reader (mapVBVD_JH, unsorted, flagIgnoreSeg) and save them for a bitwise comparison.
addpath('/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/third_party/ReadDataAndTrajectory_FreeRunningFatFractionHeart/Code_Dependencies');
idx = readmatrix(idxfile);
twix = mapVBVD_JH(datfile);
if iscell(twix), twix = twix{end}; end
img = twix.image;
img.flagIgnoreSeg = true;
fprintf('image NAcq = %d, NCol = %d, NCha = %d\n', img.NAcq, img.NCol, img.NCha);
lines = img.unsorted(idx(:).');           % [NCol, NCha, numel(idx)] single complex
save(outfile, 'lines', '-v7');
fprintf('saved lines %s\n', mat2str(size(lines)));
end
```

`xcheck_reader_compare.py`：

```python
"""Spec 5.1: 1,000 imaging lines (seed 0) read by the authors' MATLAB reader must equal ours bitwise."""
import subprocess
import sys

import h5py
import numpy as np
import scipy.io as sio

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr import DERIVED_ROOT, N_ECHO, RAW_ROOT, TOPIC  # noqa: E402

S = sys.argv[1] if len(sys.argv) > 1 else "V07"
d = DERIVED_ROOT / S
rows = np.sort(np.random.default_rng(0).choice(176000, size=1000, replace=False))
idx_file, out = d / "xcheck_idx.txt", d / "xcheck_matlab_lines.mat"
np.savetxt(idx_file, rows + 1, fmt="%d")                       # MATLAB is 1-based
dat = (RAW_ROOT / S / "freerunning_8echo.dat").resolve()
subprocess.run(["matlab", "-batch", f"addpath('{TOPIC / 'scripts'}'); xcheck_reader('{dat}', '{idx_file}', '{out}')"],
               check=True)
theirs = sio.loadmat(out)["lines"].transpose(2, 1, 0)         # (1000, coil, sample)
with h5py.File(d / "raw.h5", "r") as h:
    ours = np.empty((rows.size, *h["ksp"].shape[2:]), np.complex64)
    for e in range(N_ECHO):
        sel = np.flatnonzero(rows % N_ECHO == e)
        ours[sel] = h["ksp"][e][rows[sel] // N_ECHO]
equal = bool(np.array_equal(ours, theirs))
print(f"{S}: {rows.size} lines, ours {ours.shape} vs theirs {theirs.shape}, bitwise equal = {equal}, "
      f"max |diff| = {float(np.abs(ours.astype(np.complex128) - theirs).max()):.3g}")
sys.exit(0 if equal else 1)
```

- [ ] **Step 6: 运行交叉核对**

Run: `/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/py /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/xcheck_reader_compare.py V07 > /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/logs/xcheck_reader_V07.txt 2>&1; echo exit=$?; tail -4 /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/logs/xcheck_reader_V07.txt`
Expected: `image NAcq = 176000, NCol = 224, NCha = 24`，最后一行 `bitwise equal = True`，`exit=0`（MATLAB 要几分钟）。不相等就停，报告（停止条件 5.1）。

- [ ] **Step 7: 课题 data 软链接**

```bash
for S in V07 V03 V04 V08; do ln -sfn /data2/congcong/data/cardiac_3d_cine_public/derived/megre/$S /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/data/$S; done; ls -l /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/data/
```

- [ ] **Step 8: 同步镜像并提交**

```bash
mkdir -p /data0/congcong/code/GS/mcgs_cardiac_wt/cardiac_freerunning_megre && rsync -a --exclude '__pycache__' /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts /data0/congcong/code/GS/mcgs_cardiac_wt/cardiac_freerunning_megre/
cd /data0/congcong/code/GS/mcgs_cardiac_wt && git add cardiac_freerunning_megre/scripts/{py,s01_read.py,run_read_all.sh,s02_edge_check.py,xcheck_reader.m,xcheck_reader_compare.py} && git commit -m "cardiac: read the four subjects into raw.h5; readout-edge check; bitwise cross-check against the authors' reader"
```

### Task 7: 轨迹落盘并与作者 MATLAB 逐点核对

**Files:**
- Create（课题 `scripts/`）：`s03_traj.py`、`xcheck_traj.m`、`xcheck_traj_compare.py`

**Interfaces:**
- Consumes: `cardiac_fr.trajectory.spoke_coordinates`。
- Produces: 派生数据 `common/traj_r.npy`（(3, 224, 22000) float64，r 单位，第三维下标 = `Lin`）；`common/traj_matlab.mat`；日志 `logs/xcheck_traj.txt`。

- [ ] **Step 1: 写 `s03_traj.py` 并运行**

```python
"""Spec 5.3: the phyllotaxis trajectory, shared by all four subjects (same 22 x 1000 protocol)."""
import sys

import numpy as np

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr import DERIVED_ROOT, N_SAMP, N_SEG, N_SHOT  # noqa: E402
from cardiac_fr.trajectory import spoke_coordinates  # noqa: E402

out = DERIVED_ROOT / "common" / "traj_r.npy"
k = spoke_coordinates(N_SHOT, N_SEG, N_SAMP, self_nav=True)
np.save(out, k)
print(f"saved {out} {k.shape}, max |k| = {float(np.abs(k).max()):.4f}")
```

Run: `/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/py /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/s03_traj.py`
Expected: `saved /data2/congcong/data/cardiac_3d_cine_public/derived/megre/common/traj_r.npy (3, 224, 22000), max |k| = 0.5000`

- [ ] **Step 2: 写 `xcheck_traj.m` 与 `xcheck_traj_compare.py`**

```matlab
function xcheck_traj(outfile)
% Spec 5.3: the authors' trajectory for this protocol (N 224, 22 segments, 1000 shots, self-navigation).
addpath('/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/third_party/ReadDataAndTrajectory_FreeRunningFatFractionHeart/Code_Dependencies');
[kx, ky, kz] = computePhyllotaxis(224, 22, 1000, false);
save(outfile, 'kx', 'ky', 'kz', '-v7');
fprintf('saved kx %s\n', mat2str(size(kx)));
end
```

```python
"""Spec 5.3: our phyllotaxis port vs the authors' computePhyllotaxis (r units); max |diff| < 1e-9."""
import subprocess
import sys

import numpy as np
import scipy.io as sio

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr import DERIVED_ROOT, N_SAMP, N_SEG, N_SHOT, TOPIC  # noqa: E402

out = DERIVED_ROOT / "common" / "traj_matlab.mat"
subprocess.run(["matlab", "-batch", f"addpath('{TOPIC / 'scripts'}'); xcheck_traj('{out}')"], check=True)
m = sio.loadmat(out)
k = np.load(DERIVED_ROOT / "common" / "traj_r.npy")
worst = 0.0
for i, name in enumerate(("kx", "ky", "kz")):
    ref = m[name].reshape(N_SAMP, N_SEG * N_SHOT, order="F")      # MATLAB (N, nseg, nshot) -> column = Lin
    diff = float(np.abs(ref - k[i]).max())
    worst = max(worst, diff)
    print(f"{name}: max |diff| = {diff:.3g}")
print("PASS" if worst < 1e-9 else "FAIL")
sys.exit(0 if worst < 1e-9 else 1)
```

- [ ] **Step 3: 运行核对**

Run: `/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/py /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/xcheck_traj_compare.py > /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/logs/xcheck_traj.txt 2>&1; echo exit=$?; tail -4 /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/logs/xcheck_traj.txt`
Expected: 三行 `max |diff|` 都 < 1e-9，最后一行 `PASS`，`exit=0`。不过就停，报告（停止条件 5.3）。

- [ ] **Step 4: 同步镜像并提交**

```bash
rsync -a --exclude '__pycache__' /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts /data0/congcong/code/GS/mcgs_cardiac_wt/cardiac_freerunning_megre/
cd /data0/congcong/code/GS/mcgs_cardiac_wt && git add cardiac_freerunning_megre/scripts/{s03_traj.py,xcheck_traj.m,xcheck_traj_compare.py} && git commit -m "cardiac: trajectory on disk; matches the authors' computePhyllotaxis"
```

### Task 8: 生理信号：映射核对、稳态、呼吸三分组

**Files:**
- Create（课题 `scripts/`）：`s04_physio.py`

**Interfaces:**
- Consumes: `cardiac_fr.physio` 全部函数；`raw.h5` 的 `ts_ticks`、`lin`。
- Produces: 派生数据 `<Vxx>/spokes.npz`（`spoke_table` 的全部键，加 `r0_ids`、`low_ids`、`high_ids`）、`<Vxx>/physio.json`（含 `expiration_end_by_histogram`）；日志 `logs/physio_<Vxx>.json`。Task 9–12 用 `r0_ids`、`low_ids`、`high_ids`。

- [ ] **Step 1: 写 `s04_physio.py`**

```python
"""Spec 5.4: PhysioInfo <-> ADC mapping (must be exact), steady state, respiratory thirds, polarity
criterion (a), cardiac phase. Writes spokes.npz and physio.json per subject."""
import json
import sys

import h5py
import numpy as np

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr import DERIVED_ROOT, N_ECHO, N_SEG, RAW_ROOT, SUBJECTS, T_STEADY_MS, TOPIC  # noqa: E402
from cardiac_fr.physio import (expiration_end_by_histogram, load_physio, spoke_table,  # noqa: E402
                               third_at_end, timestamp_mismatch)

ok_all = True
for s in sys.argv[1:] or SUBJECTS:
    ph = load_physio(RAW_ROOT / s / "PhysioInfo.mat")
    with h5py.File(DERIVED_ROOT / s / "raw.h5", "r") as h:
        ts_ticks, lin = h["ts_ticks"][:], h["lin"][:]
    mis = timestamp_mismatch(ph["TimeStamp_ms"], ts_ticks)
    tab = spoke_table(ph, lin, N_ECHO, N_SEG, T_STEADY_MS)
    use = tab["steady"] & ~tab["is_si"]
    r0, resp = tab["lin"][use], tab["resp"][use]
    low, high = r0[third_at_end(resp, "low")], r0[third_at_end(resp, "high")]
    np.savez(DERIVED_ROOT / s / "spokes.npz", **tab, r0_ids=r0, low_ids=low, high_ids=high)
    rep = {"subject": s, "timestamp_max_abs_diff_ms": mis, "n_spoke": int(lin.size),
           "n_si": int(tab["is_si"].sum()), "n_before_steady": int((~tab["steady"]).sum()),
           "n_r0": int(r0.size), "n_third": int(low.size),
           "expiration_end_by_histogram": expiration_end_by_histogram(resp),
           "resp_p2_p50_p98": [float(np.percentile(resp, q)) for q in (2, 50, 98)],
           "rr_ms_median": float(np.median(np.diff(ph["cardTriggers"]))),
           "card_phase_nan": int(np.isnan(tab["card_phase"]).sum())}
    txt = json.dumps(rep, indent=1)
    (DERIVED_ROOT / s / "physio.json").write_text(txt)
    (TOPIC / "logs" / f"physio_{s}.json").write_text(txt)
    print(json.dumps(rep), flush=True)
    ok_all &= mis == 0.0
sys.exit(0 if ok_all else 1)
```

- [ ] **Step 2: 运行**

Run: `/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/py /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/s04_physio.py; echo exit=$?`
Expected: 四行 JSON，每行 `"timestamp_max_abs_diff_ms": 0.0`、`"n_spoke": 22000`、`"n_si": 1000`、`"n_r0"` 约 20,400、`"n_third"` 约 6,800；`exit=0`。记下每例的 `expiration_end_by_histogram`（Task 10 要和图像判据比）。

- [ ] **Step 3: 同步镜像并提交**

```bash
rsync -a --exclude '__pycache__' /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts /data0/congcong/code/GS/mcgs_cardiac_wt/cardiac_freerunning_megre/
cd /data0/congcong/code/GS/mcgs_cardiac_wt && git add cardiac_freerunning_megre/scripts/s04_physio.py && git commit -m "cardiac: PhysioInfo mapping verified row by row; steady state and respiratory thirds"
```

### Task 9: 驱动共用函数、线圈图、V07 运动平均重建（R0）与方位判定

**Files:**
- Create（课题 `scripts/`）：`common.py`、`s05_coilmaps.py`、`s06_recon.py`、`s11_figures.py`、`run_v07_r0.sh`
- Create（课题根）：`orientation.json`（看图后手写）

**Interfaces:**
- Consumes: `cardiac_fr.bart`、`cardiac_fr.gauge`；`spokes.npz`、`traj_r.npy`、`raw.h5`。
- Produces（`common.py`，Task 10–12 都用）：`LOGS`、`work_dir(s)`、`recon_dir(s)`、`set_ids(s, name)`（name ∈ r0, low, high, ref, heldout, pool, af8, af12, af16）、`prepare_inputs(s, name, echoes)`（写 `bart/traj_<name>` 与 `bart/ksp_<name>_e<k>`）、`recon(s, name, echo, reg, lam, iters, tag, w=None, debug=False) -> dict`（存 `recon/<s>/<tag>_e<echo>.npy`，追加 `logs/recon_<s>.jsonl`）、`run_many(jobs, n_parallel=1)`、`heldout_nrmse(s, tag, echo)`、`score(s, tag, echo, heart_box=None) -> dict`、`save_fig(fig, name)`、`three_plane(vols, name, center=None, title="", box=None)`。
- Produces: 派生数据 `<Vxx>/sens.{cfl,hdr}`（112, 112, 112, 24）；`recon/V07/r0_e0.npy … r0_e7.npy`；`orientation.json` 键 `decided_from, si_axis, superior_is_high_index, axis0, axis1, axis2, note`。

- [ ] **Step 1: 写 `common.py`**

```python
"""Shared helpers for the phase-1 drivers (spec 5.5-5.7): spoke sets, BART inputs, reconstruction,
scoring and figures. prepare_inputs() writes every input before any job starts, so parallel jobs never
write the same file."""
from __future__ import annotations

import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr import DERIVED_ROOT, FIGS, GRID, KEEP, TOPIC  # noqa: E402
from cardiac_fr import bart  # noqa: E402
from cardiac_fr.gauge import image_metrics, kspace_nrmse, norm_scale  # noqa: E402

TRAJ_R = DERIVED_ROOT / "common" / "traj_r.npy"
LOGS = TOPIC / "logs"


def work_dir(subject: str) -> Path:
    d = DERIVED_ROOT / subject / "bart"
    d.mkdir(parents=True, exist_ok=True)
    return d


def recon_dir(subject: str) -> Path:
    d = TOPIC / "recon" / subject
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_ids(subject: str, name: str) -> np.ndarray:
    d = DERIVED_ROOT / subject
    if name in ("r0", "low", "high"):
        return np.load(d / "spokes.npz")[f"{name}_ids"]
    if name == "ref":
        return np.load(d / "ref_ids.npy")
    return np.load(d / "gauge_split.npz")[name]


def prepare_inputs(subject: str, name: str, echoes) -> None:
    ids = set_ids(subject, name)
    k = np.load(TRAJ_R, mmap_mode="r")[:, KEEP][:, :, ids] * GRID
    bart.write_cfl(work_dir(subject) / f"traj_{name}", bart.bart_traj(np.asarray(k)))
    with h5py.File(DERIVED_ROOT / subject / "raw.h5", "r") as h:
        for e in echoes:
            y = h["ksp"][e][ids][:, :, KEEP]
            bart.write_cfl(work_dir(subject) / f"ksp_{name}_e{e}", bart.bart_ksp(y))


def recon(subject: str, name: str, echo: int, reg: str, lam: float, iters: int, tag: str,
          w: float | None = None, debug: bool = False) -> dict:
    wd = work_dir(subject)
    out = wd / f"rec_{tag}_e{echo}"
    info = bart.pics(wd / f"ksp_{name}_e{echo}", DERIVED_ROOT / subject / "sens", out, wd / f"traj_{name}",
                     reg=reg, lam=lam, iters=iters, w=w, debug=debug,
                     log=LOGS / f"bart_{subject}_{tag}_e{echo}.log")
    np.save(recon_dir(subject) / f"{tag}_e{echo}.npy", bart.read_cfl(out).astype(np.complex64))
    info.update(subject=subject, set=name, echo=echo, reg=reg, lam=lam, iters=iters, w=w, tag=tag,
                n_spokes=int(set_ids(subject, name).size))
    with open(LOGS / f"recon_{subject}.jsonl", "a") as fh:
        fh.write(json.dumps(info) + "\n")
    return info


def run_many(jobs: list, n_parallel: int = 1) -> list:
    with ThreadPoolExecutor(max_workers=n_parallel) as ex:
        return list(ex.map(lambda j: recon(**j), jobs))


def heldout_nrmse(subject: str, tag: str, echo: int) -> float:
    """||A_ho x - y_ho|| / ||y_ho|| with A_ho = coil maps x BART NUFFT on the held-out spokes
    (prepare_inputs(subject, "heldout", ...) must have run first)."""
    wd = work_dir(subject)
    pred = wd / f"pred_{tag}_e{echo}"
    bart.forward(wd / "traj_heldout", wd / f"rec_{tag}_e{echo}", DERIVED_ROOT / subject / "sens", pred, wd)
    return kspace_nrmse(bart.read_cfl(pred), bart.read_cfl(wd / f"ksp_heldout_e{echo}"))


def score(subject: str, tag: str, echo: int, heart_box=None) -> dict:
    ref = np.abs(np.load(recon_dir(subject) / f"ref_e{echo}.npy"))
    sc = norm_scale(ref)
    rec = np.abs(np.load(recon_dir(subject) / f"{tag}_e{echo}.npy"))
    m = image_metrics(rec / sc, ref / sc, heart_box=heart_box)
    m["heldout_nrmse"] = heldout_nrmse(subject, tag, echo)
    return m


def save_fig(fig, name: str) -> Path:
    FIGS.mkdir(parents=True, exist_ok=True)
    (TOPIC / "figures").mkdir(exist_ok=True)
    p = FIGS / name
    fig.savefig(p, dpi=110)
    shutil.copy2(p, TOPIC / "figures" / name)
    print(f"http://localhost:8765/cardiac_fr/{name}\n{p}", flush=True)
    return p


def three_plane(vols: dict, name: str, center=None, title: str = "", box=None) -> Path:
    """One row per volume, three orthogonal planes through `center`, grey window [0, P99.5] per row;
    optional box ((x0, x1), (y0, y1), (z0, z1)) drawn in yellow."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    first = np.abs(next(iter(vols.values())))
    c = center or tuple(n // 2 for n in first.shape)
    fig, ax = plt.subplots(len(vols), 3, figsize=(9, 3 * len(vols)), squeeze=False)
    for i, (label, v) in enumerate(vols.items()):
        v = np.abs(v)
        vmax = float(np.quantile(v, 0.995))
        for j, (sl, (a, b)) in enumerate(((v[c[0]], (1, 2)), (v[:, c[1]], (0, 2)), (v[:, :, c[2]], (0, 1)))):
            ax[i, j].imshow(sl.T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            if box is not None:
                ax[i, j].add_patch(Rectangle((box[a][0], box[b][0]), box[a][1] - box[a][0],
                                             box[b][1] - box[b][0], fill=False, ec="yellow", lw=1))
        ax[i, 0].set_ylabel(label, fontsize=8)
    for j, t in enumerate((f"axis0={c[0]}  (x: axis1, y: axis2)", f"axis1={c[1]}  (x: axis0, y: axis2)",
                           f"axis2={c[2]}  (x: axis0, y: axis1)")):
        ax[0, j].set_title(t, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    p = save_fig(fig, name)
    plt.close(fig)
    return p
```

- [ ] **Step 2: 写 `s05_coilmaps.py`**

```python
"""Spec 5.5: ESPIRiT coil maps from echo 0 on all steady non-SI spokes (brain-topic flags)."""
import json
import sys

import numpy as np

from common import DERIVED_ROOT, LOGS, bart, prepare_inputs, set_ids, three_plane, work_dir

s = sys.argv[1]
prepare_inputs(s, "r0", [0])
wd = work_dir(s)
bart.coil_maps(wd / "traj_r0", wd / "ksp_r0_e0", wd, DERIVED_ROOT / s / "sens", grid=112, calib=24,
               log=LOGS / f"bart_{s}_coilmaps.log")
sens = bart.read_cfl(DERIVED_ROOT / s / "sens")
rss = np.sqrt((np.abs(bart.read_cfl(wd / "coilimg")) ** 2).sum(-1))
support = rss > 0.05 * np.quantile(rss, 0.999)
sos = np.sqrt((np.abs(sens) ** 2).sum(-1))
rep = {"subject": s, "n_spokes": int(set_ids(s, "r0").size), "sens_shape": list(sens.shape),
       "support_frac": float(support.mean()),
       "sos_within_5pct_in_support": float(np.mean(np.abs(sos[support] - 1) < 0.05))}
(LOGS / f"coilmaps_{s}.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1), flush=True)
three_plane({"RSS coil images": rss, "|map 0|": sens[..., 0], "|map 8|": sens[..., 8],
             "|map 16|": sens[..., 16]}, f"{s}_coilmaps.png",
            title=f"{s} ESPIRiT maps (echo 0, {rep['n_spokes']} spokes)")
sys.exit(0 if rep["sos_within_5pct_in_support"] >= 0.95 else 1)
```

- [ ] **Step 3: 写 `s06_recon.py`**

```python
"""Spec 5.6: BART pics per echo on a named spoke set (r0, low, high, ref, af8, af12, af16)."""
import argparse
import json

from common import prepare_inputs, recon_dir, run_many

ap = argparse.ArgumentParser()
ap.add_argument("subject")
ap.add_argument("--set", required=True)
ap.add_argument("--echoes", default="0,1,2,3,4,5,6,7")
ap.add_argument("--reg", default="l2", choices=["l2", "l1"])
ap.add_argument("--lam", type=float, default=1e-3)
ap.add_argument("--iters", type=int, default=100)
ap.add_argument("--tag", default=None)
ap.add_argument("--w-from-ref", action="store_true", help="-w = REF scaling of the same echo")
ap.add_argument("--debug", action="store_true", help="-d5; a REF run records BART's scaling")
ap.add_argument("--jobs", type=int, default=1)
a = ap.parse_args()
echoes = [int(e) for e in a.echoes.split(",")]
tag = a.tag or a.set
prepare_inputs(a.subject, a.set, echoes)
scal = json.loads((recon_dir(a.subject) / "ref_scaling.json").read_text()) if a.w_from_ref else {}
jobs = [dict(subject=a.subject, name=a.set, echo=e, reg=a.reg, lam=a.lam, iters=a.iters, tag=tag,
             w=scal[str(e)] if a.w_from_ref else None, debug=a.debug) for e in echoes]
infos = run_many(jobs, a.jobs)
if a.debug and a.set == "ref" and tag == "ref":
    p = recon_dir(a.subject) / "ref_scaling.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update({str(i["echo"]): i["scaling"] for i in infos})
    p.write_text(json.dumps(old, indent=1))
for i in infos:
    print(f"{i['tag']} e{i['echo']}: {i['n_spokes']} spokes, {i['seconds']:.0f} s, scaling={i['scaling']}", flush=True)
```

- [ ] **Step 4: 写 `s11_figures.py`**

```python
"""Three-plane figure of one reconstructed set, one row per echo (optionally with the heart box)."""
import argparse
import json

import numpy as np

from common import TOPIC, recon_dir, three_plane

ap = argparse.ArgumentParser()
ap.add_argument("subject")
ap.add_argument("--tag", required=True)
ap.add_argument("--echoes", default="0,1,2,3,4,5,6,7")
ap.add_argument("--center", default=None, help="i,j,k voxel the three planes pass through")
ap.add_argument("--box", action="store_true", help="overlay gauge.json[subject]['heart_box']")
ap.add_argument("--name", default=None)
a = ap.parse_args()
vols = {f"e{e}  TE {1.25 + 2.05 * e:.2f} ms": np.load(recon_dir(a.subject) / f"{a.tag}_e{e}.npy")
        for e in map(int, a.echoes.split(","))}
center = tuple(map(int, a.center.split(","))) if a.center else None
box = json.loads((TOPIC / "gauge.json").read_text())[a.subject]["heart_box"] if a.box else None
three_plane(vols, a.name or f"{a.subject}_{a.tag}.png", center=center, title=f"{a.subject} {a.tag}", box=box)
```

- [ ] **Step 5: 脱离会话跑 V07 的线圈图与 R0**

```bash
cat > /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/run_v07_r0.sh <<'EOF'
#!/usr/bin/env bash
# Spec 5.5-5.6 on V07: coil maps, R0 (8 echoes), R0 figure.
set -euo pipefail
cd /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts
./py s05_coilmaps.py V07
./py s06_recon.py V07 --set r0
./py s11_figures.py V07 --tag r0
echo V07_R0_DONE
EOF
setsid nohup bash /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/run_v07_r0.sh > /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/logs/run_v07_r0.log 2>&1 < /dev/null &
```

Run（每 15–20 分钟看一次）：`tail -5 /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/logs/run_v07_r0.log`
Expected: 先打印线圈图 JSON，`"sens_shape": [112, 112, 112, 24]`、`"sos_within_5pct_in_support"` ≥ 0.95；再是 8 行 `r0 eK: 2xxxx spokes, NNN s`；最后 `V07_R0_DONE`。总时长约 2–3.5 小时。任何 `Traceback`/`RuntimeError` 就停，按 superpowers:systematic-debugging 查。

- [ ] **Step 6: 看图，判定方位，写 `orientation.json`**

用 Read 工具打开 `/home/congcongliu/figs/cardiac_fr/V07_coilmaps.png` 与 `/home/congcongliu/figs/cardiac_fr/V07_r0.png`。要能认出心脏、肝顶、脊柱与胸壁；8 个回波的整体亮度应随 TE 下降，水脂同相/反相在脂肪边界处交替变化。认不出解剖就停，按 systematic-debugging 查（先查轨迹是否乘了 112、`KEEP` 是否对齐、回波与线圈维的顺序）。

SI 自门控辐条沿轨迹 z（数组 axis 2），预期 SI 轴是 axis 2；以图为准。按看到的写：

```bash
cat > /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/orientation.json <<'EOF'
{
 "decided_from": "figures/V07_r0.png",
 "si_axis": 2,
 "superior_is_high_index": true,
 "axis0": "patient direction of increasing axis-0 index, as seen",
 "axis1": "patient direction of increasing axis-1 index, as seen",
 "axis2": "patient direction of increasing axis-2 index, as seen",
 "note": "which structures decided it (e.g. heart above the liver dome at high axis-2 index)"
}
EOF
```

`si_axis` 与 `superior_is_high_index` 两个值必须按图改对（上面是预期值，不是结论）；三个 `axisK` 与 `note` 写实际看到的方向和依据。

- [ ] **Step 7: 同步镜像并提交**

```bash
rsync -a --exclude '__pycache__' /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts /data0/congcong/code/GS/mcgs_cardiac_wt/cardiac_freerunning_megre/
cp /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/orientation.json /data0/congcong/code/GS/mcgs_cardiac_wt/cardiac_freerunning_megre/
cd /data0/congcong/code/GS/mcgs_cardiac_wt && git add cardiac_freerunning_megre/scripts/{common.py,s05_coilmaps.py,s06_recon.py,s11_figures.py,run_v07_r0.sh} cardiac_freerunning_megre/orientation.json && git commit -m "cardiac: coil maps and the motion-averaged reconstruction of V07; image orientation"
```

### Task 10: 呼吸极性（判据 b）、V07 参考重建（REF）与两项核对

**Files:**
- Create（课题 `scripts/`）：`s07_polarity.py`、`s08_checks.py`、`run_v07_ref.sh`

**Interfaces:**
- Consumes: `common.py` 全部；`orientation.json`；`physio.json` 的 `expiration_end_by_histogram`；`spokes.npz` 的 `low_ids`、`high_ids`。
- Produces: 派生数据 `<Vxx>/ref_ids.npy`（只在两条判据一致时写）；`recon/<Vxx>/ref_e0..7.npy`；`recon/<Vxx>/ref_scaling.json`（`{"0": X0, …, "7": X7}`）；日志 `polarity_<Vxx>.json`、`checks_<Vxx>.json`（含 `ref_iters`）；图 `<Vxx>_polarity.png`、`<Vxx>_ref.png`。

- [ ] **Step 1: 写 `s07_polarity.py`**

```python
"""Spec 5.4 criterion (b): the respiratory third whose anatomy sits more superior is end-expiration.
It must agree with criterion (a) (physio.json); only then is ref_ids.npy written."""
import json
import sys

import numpy as np

from common import DERIVED_ROOT, LOGS, TOPIC, recon_dir, save_fig, set_ids

s = sys.argv[1]
ori = json.loads((TOPIC / "orientation.json").read_text())
si, sup_high = int(ori["si_axis"]), bool(ori["superior_is_high_index"])
low = np.abs(np.load(recon_dir(s) / "low_e0.npy"))
high = np.abs(np.load(recon_dir(s) / "high_e0.npy"))
other = [i for i in range(3) if i != si]


def profile(v):
    sl = [slice(None)] * 3
    for i in other:
        sl[i] = slice(v.shape[i] // 4, 3 * v.shape[i] // 4)
    p = v[tuple(sl)].mean(axis=tuple(other))
    return p - p.mean()


pl, ph = profile(low), profile(high)
n = pl.size
lags = np.arange(-20, 21)
corr = [float(np.sum(pl[max(0, -L):n - max(0, L)] * ph[max(0, L):n - max(0, -L)])) for L in lags]
lag = int(lags[int(np.argmax(corr))])            # high(z + lag) matches low(z)
image_end = "ambiguous" if lag == 0 else ("high" if (lag > 0) == sup_high else "low")
hist_end = json.loads((DERIVED_ROOT / s / "physio.json").read_text())["expiration_end_by_histogram"]
agreed = image_end == hist_end
rep = {"subject": s, "lag_vox_high_vs_low": lag, "image_end": image_end, "histogram_end": hist_end,
       "agreed": agreed}
if agreed:
    np.save(DERIVED_ROOT / s / "ref_ids.npy", set_ids(s, hist_end))
    rep["n_ref"] = int(set_ids(s, hist_end).size)
(LOGS / f"polarity_{s}.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1), flush=True)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].plot(pl, label="low third")
axs[0].plot(ph, label="high third")
axs[0].legend()
axs[0].set_xlabel(f"axis {si} (superior = {'high' if sup_high else 'low'} index)")
cut = other[0]
rest = [i for i in range(3) if i != cut]          # this plane keeps the SI axis
for k, (v, t) in enumerate(((low, "low third"), (high, "high third"))):
    sl = np.take(v, v.shape[cut] // 2, axis=cut)
    axs[k + 1].imshow(sl if rest.index(si) == 0 else sl.T, cmap="gray", origin="lower", vmin=0,
                      vmax=float(np.quantile(v, 0.995)))
    axs[k + 1].set_title(t)
fig.suptitle(f"{s}: lag {lag} vox; image says end-expiration = {image_end}, histogram says {hist_end}")
save_fig(fig, f"{s}_polarity.png")
sys.exit(0 if agreed else 1)
```

- [ ] **Step 2: 写 `s08_checks.py`**

```python
"""Spec 5.6 checks on REF echo 0: iterations (0.6N vs N steps, body relative difference < 1 %) and -S
scaling (-w 2X vs -w X, body norm ratio within 1 %)."""
import argparse
import json
import sys

import numpy as np

from common import LOGS, prepare_inputs, recon, recon_dir

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from diag_artefact.headonly import head_mask  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("subject")
ap.add_argument("--ref-iters", type=int, default=100)
a = ap.parse_args()
s, n = a.subject, a.ref_iters
rd = recon_dir(s)
w = json.loads((rd / "ref_scaling.json").read_text())["0"]
prepare_inputs(s, "ref", [0])
short = int(round(0.6 * n))
recon(s, "ref", 0, "l2", 1e-3, short, f"ref_it{short}", w=w)
recon(s, "ref", 0, "l2", 1e-3, n, "ref_w2x", w=2 * w)
xn, xs, x2 = (np.load(rd / f"{t}_e0.npy") for t in ("ref", f"ref_it{short}", "ref_w2x"))
mag = np.abs(xn)
body = head_mask(mag / np.quantile(mag, 0.999))
it = float(np.linalg.norm((xn - xs)[body]) / np.linalg.norm(xn[body]))
ratio = float(np.linalg.norm(x2[body]) / np.linalg.norm(xn[body]))
rep = {"subject": s, "ref_iters": n, f"iter_rel_diff_{short}_vs_{n}": it, "iter_ok": it < 0.01,
       "scale_ratio_w2x_over_w1x": ratio, "scale_ok": abs(ratio - 1) < 0.01}
(LOGS / f"checks_{s}.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1), flush=True)
sys.exit(0 if rep["iter_ok"] and rep["scale_ok"] else 1)
```

- [ ] **Step 3: 脱离会话跑 V07 的极性、REF 与核对**

```bash
cat > /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/run_v07_ref.sh <<'EOF'
#!/usr/bin/env bash
# Spec 5.4(b) and 5.6 on V07: respiratory thirds, polarity, REF (8 echoes), checks, REF figure.
set -euo pipefail
cd /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts
./py s06_recon.py V07 --set low --echoes 0
./py s06_recon.py V07 --set high --echoes 0
./py s07_polarity.py V07
./py s06_recon.py V07 --set ref --debug
./py s08_checks.py V07
./py s11_figures.py V07 --tag ref
echo V07_REF_DONE
EOF
setsid nohup bash /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/run_v07_ref.sh > /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/logs/run_v07_ref.log 2>&1 < /dev/null &
```

Run（每 15–20 分钟看一次）：`tail -8 /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/logs/run_v07_ref.log`
Expected: 极性 JSON 里 `"agreed": true`，`"n_ref"` 约 6,800；8 行 `ref eK: … scaling=<正数>`；核对 JSON 里 `"iter_ok": true`、`"scale_ok": true`；最后 `V07_REF_DONE`。约 2.5–4.5 小时。另查 `logs/recon_V07.jsonl` 里 `tag` 为 `ref` 的 8 行：`seconds` 都应小于 3600；有超过的就先报告，再决定是否减迭代或加 `-U`（spec 第 7 节）。

- [ ] **Step 4: 处理不通过的情况**

- 日志停在 `s07_polarity.py`（`"agreed": false` 或 `"image_end": "ambiguous"`）：停，把 `/home/congcongliu/figs/cardiac_fr/V07_polarity.png` 和 `logs/polarity_V07.json` 报给用户（停止条件 5.4）。
- `"scale_ok": false`：停，报告（停止条件 5.6；单元测试已证明 `-S` 还原数据单位，出现即说明流程别处有错）。
- 只有 `"iter_ok": false`：按 spec 改 200 步，脱离会话跑 `cd /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts && ./py s06_recon.py V07 --set ref --debug --iters 200 && ./py s08_checks.py V07 --ref-iters 200 && ./py s11_figures.py V07 --tag ref`（写成 bash 脚本再 `setsid nohup`）。仍不过就停，报告。

- [ ] **Step 5: 看 REF 图**

用 Read 打开 `/home/congcongliu/figs/cardiac_fr/V07_ref.png` 与 `/home/congcongliu/figs/cardiac_fr/V07_polarity.png`。REF 的膈顶与肝边界应比 R0 清楚，心脏仍有心动模糊（spec 7）。条纹明显强于 R0 时记下来，写进 README 的已知限制。

- [ ] **Step 6: 同步镜像并提交**

```bash
rsync -a --exclude '__pycache__' /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts /data0/congcong/code/GS/mcgs_cardiac_wt/cardiac_freerunning_megre/
cd /data0/congcong/code/GS/mcgs_cardiac_wt && git add cardiac_freerunning_megre/scripts/{s07_polarity.py,s08_checks.py,run_v07_ref.sh} && git commit -m "cardiac: respiratory polarity from two criteria, end-expiration reference of V07, iteration and scaling checks"
```

### Task 11: V07 的评估口径：留出与子集、心脏框、自检

**Files:**
- Create（课题 `scripts/`）：`s09_gauge_split.py`、`s10_selfcheck.py`、`run_v07_selfcheck.sh`
- Create（课题根）：`gauge.json`（s09 写入，心脏框手工补）

**Interfaces:**
- Consumes: `cardiac_fr.gauge`；`common.py`；`ref_ids.npy`；`ref_scaling.json`；`logs/checks_V07.json` 的 `ref_iters`；`orientation.json`。
- Produces: 派生数据 `V07/gauge_split.npz`（`heldout, pool, af8, af12, af16`）；`gauge.json` 结构 `{"V07": {n_ref, af_ref, n_heldout, n_pool, seed, subsets: {af8|af12|af16: {n, af_actual}}, heart_box: [[x0,x1],[y0,y1],[z0,z1]]}, "l1_lambda": λ, "l1_lambda_sweep_body_psnr": {...}}`；日志 `selfcheck_V07.tsv`、`selfcheck_V07.json`；图 `V07_ref_heart.png`、`V07_selfcheck_e0.png`。

- [ ] **Step 1: 写 `s09_gauge_split.py` 并运行**

```python
"""Spec 5.7: held-out 5 % of REF (seed 0) and whole-shot subsets at AF 8, 12, 16 from the rest."""
import json
import sys

import numpy as np

from common import DERIVED_ROOT, TOPIC

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr.gauge import af, heldout_split, shot_subset  # noqa: E402

s = sys.argv[1]
ref = np.load(DERIVED_ROOT / s / "ref_ids.npy")
ho, pool = heldout_split(ref, 0.05, seed=0)
subs = {f"af{t}": shot_subset(pool, t) for t in (8, 12, 16)}
np.savez(DERIVED_ROOT / s / "gauge_split.npz", heldout=ho, pool=pool, **subs)
g = {"n_ref": int(ref.size), "af_ref": af(ref.size), "n_heldout": int(ho.size), "n_pool": int(pool.size),
     "seed": 0, "subsets": {k: {"n": int(v.size), "af_actual": af(v.size)} for k, v in subs.items()}}
p = TOPIC / "gauge.json"
gj = json.loads(p.read_text()) if p.exists() else {}
gj[s] = {**gj.get(s, {}), **g}
p.write_text(json.dumps(gj, indent=1))
print(json.dumps({s: g}, indent=1))
```

Run: `cd /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts && ./py s09_gauge_split.py V07`
Expected: `n_ref` 约 6,800、`af_ref` 约 2.9、`n_heldout` 约 340；三个子集 `af_actual` 分别接近 8、12、16（与目标相差不超过一组交错，即 21 条辐条的量级）。

- [ ] **Step 2: 定心脏框**

先看 `/home/congcongliu/figs/cardiac_fr/V07_ref.png` 找到心脏大致位置，再让三个切面穿过心脏中心出一张图（`i,j,k` 换成估计的心脏中心体素）：

`cd /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts && ./py s11_figures.py V07 --tag ref --echoes 0 --center i,j,k --name V07_ref_heart.png`

用 Read 看图，反复调 `--center` 直到三个切面都过心脏。框取包住四个心腔的最小长方体，每边再留 3 个体素（心脏约 9–13 cm，每边约 45–65 个体素）。把框写进 gauge.json（数字按看到的填）：

`/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/py -c "import json; p='/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/gauge.json'; g=json.load(open(p)); g['V07']['heart_box']=[[x0,x1],[y0,y1],[z0,z1]]; json.dump(g, open(p,'w'), indent=1)"`

再出一张带框的图核对：`./py s11_figures.py V07 --tag ref --echoes 0 --center i,j,k --box --name V07_ref_heart.png`，黄框应在三个切面上都包住心脏。

- [ ] **Step 3: 写 `s10_selfcheck.py`**

```python
"""Spec 5.7 self-check on V07: choose the L1 lambda on AF12 echo 0 (body PSNR vs REF), then CG-SENSE and
L1 at AF 8/12/16 on echoes 0 and 4. Every metric must get monotonically worse with AF."""
import csv
import json
import sys

import numpy as np

from common import LOGS, TOPIC, prepare_inputs, recon_dir, run_many, save_fig, score

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr.gauge import error_map, monotone_worse  # noqa: E402

S, ECHOES, AFS = "V07", (0, 4), ("af8", "af12", "af16")
LAMS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
KEYS = (("body_psnr", True), ("body_ssim", True), ("body_hfen", False), ("heldout_nrmse", False))
ITERS = json.loads((LOGS / "checks_V07.json").read_text())["ref_iters"]
gj = json.loads((TOPIC / "gauge.json").read_text())
box = gj[S]["heart_box"]
scal = json.loads((recon_dir(S) / "ref_scaling.json").read_text())
for name in (*AFS, "heldout"):
    prepare_inputs(S, name, ECHOES)

run_many([dict(subject=S, name="af12", echo=0, reg="l1", lam=l, iters=ITERS, tag=f"l1sweep_{l:g}",
               w=scal["0"]) for l in LAMS])
sweep = {l: score(S, f"l1sweep_{l:g}", 0, box)["body_psnr"] for l in LAMS}
best = max(LAMS, key=sweep.get)
gj["l1_lambda"] = best
gj["l1_lambda_sweep_body_psnr"] = {f"{l:g}": round(v, 3) for l, v in sweep.items()}
(TOPIC / "gauge.json").write_text(json.dumps(gj, indent=1))
print("L1 lambda sweep (body PSNR, AF12, echo 0):", gj["l1_lambda_sweep_body_psnr"], "-> best", best, flush=True)
if best in (LAMS[0], LAMS[-1]):
    print("NOTE: the best lambda is at the edge of the grid", flush=True)

run_many([dict(subject=S, name=a, echo=e, reg=r, lam=l, iters=ITERS, tag=f"{m}_{a}", w=scal[str(e)])
          for e in ECHOES for a in AFS for (m, r, l) in (("cg", "l2", 1e-3), ("l1", "l1", best))])
rows, verdicts = [], []
for e in ECHOES:
    rows.append({"echo": e, "method": "ref_in_sample", "set": "ref", "af": round(gj[S]["af_ref"], 3),
                 **score(S, "ref", e, box)})
    for m in ("cg", "l1"):
        per = {}
        for a in AFS:
            af_a = gj[S]["subsets"][a]["af_actual"]
            per[af_a] = score(S, f"{m}_{a}", e, box)
            rows.append({"echo": e, "method": m, "set": a, "af": round(af_a, 3), **per[af_a]})
        for k, hib in KEYS:
            verdicts.append({"echo": e, "method": m, "metric": k,
                             "monotone": monotone_worse({x: per[x][k] for x in per}, hib)})
with open(LOGS / "selfcheck_V07.tsv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
    wr.writeheader()
    wr.writerows(rows)
passed = all(v["monotone"] for v in verdicts)
(LOGS / "selfcheck_V07.json").write_text(json.dumps({"l1_lambda": best, "iters": ITERS, "verdicts": verdicts,
                                                     "pass": passed}, indent=1))
print((LOGS / "selfcheck_V07.tsv").read_text(), flush=True)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

si = int(json.loads((TOPIC / "orientation.json").read_text())["si_axis"])
fix = [i for i in range(3) if i != si][0]
c = (box[fix][0] + box[fix][1]) // 2
ref = np.abs(np.load(recon_dir(S) / "ref_e0.npy"))
sc = float(np.quantile(ref, 0.999))
tags = ["ref"] + [f"{m}_{a}" for m in ("cg", "l1") for a in AFS]
fig, ax = plt.subplots(2, len(tags), figsize=(2.2 * len(tags), 4.8), squeeze=False)
for j, t in enumerate(tags):
    v = np.abs(np.load(recon_dir(S) / f"{t}_e0.npy"))
    ax[0, j].imshow(np.take(v / sc, c, axis=fix).T, cmap="gray", vmin=0, vmax=1, origin="lower")
    im = ax[1, j].imshow(np.take(error_map(v, ref), c, axis=fix).T, cmap="jet", vmin=0, vmax=0.1, origin="lower")
    ax[0, j].set_title(t, fontsize=8)
    for a_ in ax[:, j]:
        a_.set_xticks([])
        a_.set_yticks([])
fig.colorbar(im, ax=ax[1, :].tolist(), shrink=0.8)
fig.suptitle(f"V07 echo 0 (TE 1.25 ms), plane axis{fix} = {c}; error = |rec - ref| / P99.9(ref), jet [0, 0.1]")
save_fig(fig, "V07_selfcheck_e0.png")
print("SELF-CHECK", "PASS" if passed else "FAIL", flush=True)
sys.exit(0 if passed else 1)
```

- [ ] **Step 4: 脱离会话跑自检**

```bash
cat > /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/run_v07_selfcheck.sh <<'EOF'
#!/usr/bin/env bash
# Spec 5.7 self-check on V07 (5 lambda runs + 12 grid runs, then scoring).
set -euo pipefail
cd /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts
./py s10_selfcheck.py
echo V07_SELFCHECK_DONE
EOF
setsid nohup bash /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/run_v07_selfcheck.sh > /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/logs/run_v07_selfcheck.log 2>&1 < /dev/null &
```

Run（每 20–30 分钟看一次）：`tail -25 /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/logs/run_v07_selfcheck.log`
Expected: 先打印 λ 扫描结果与选中的 λ；再打印 TSV（echo 0 与 4，每个方法三行 AF）；倒数第二行 `SELF-CHECK PASS`，最后 `V07_SELFCHECK_DONE`。约 3–5 小时。`SELF-CHECK FAIL` 就停：把 `logs/selfcheck_V07.json` 里不单调的项与 `V07_selfcheck_e0.png` 报给用户（停止条件 5.7）。

- [ ] **Step 5: 同步镜像并提交**

```bash
rsync -a --exclude '__pycache__' /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts /data0/congcong/code/GS/mcgs_cardiac_wt/cardiac_freerunning_megre/
cp /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/gauge.json /data0/congcong/code/GS/mcgs_cardiac_wt/cardiac_freerunning_megre/
cd /data0/congcong/code/GS/mcgs_cardiac_wt && git add cardiac_freerunning_megre/scripts/{s09_gauge_split.py,s10_selfcheck.py,run_v07_selfcheck.sh} cardiac_freerunning_megre/gauge.json && git commit -m "cardiac: evaluation gauge on V07 (held-out spokes, whole-shot subsets, heart box) and its self-check"
```

### Task 12: 冻结参数，跑 V03、V04、V08

**Files:**
- Create（课题根）：`params_frozen.json`
- Create（课题 `scripts/`）：`run_subject.sh`、`run_rest.sh`

**Interfaces:**
- Consumes: V07 的 `logs/checks_V07.json`（`ref_iters`）与 `gauge.json`（`l1_lambda`）；Task 6、8 已为四例写好的 `raw.h5`、`spokes.npz`、`physio.json`。
- Produces: V03、V04、V08 各自的 `sens`、`recon/<Vxx>/r0_e*.npy`、`ref_e*.npy`、`ref_scaling.json`、`ref_ids.npy`，日志 `coilmaps_/polarity_/checks_<Vxx>.json`、`pipeline_<Vxx>.log`，图 `<Vxx>_{coilmaps,r0,polarity,ref}.png`。

- [ ] **Step 1: 从 V07 的结果生成 `params_frozen.json`**

```bash
/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/py - <<'EOF'
import json
T = "/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE"
it = json.load(open(f"{T}/logs/checks_V07.json"))["ref_iters"]
lam = json.load(open(f"{T}/gauge.json"))["l1_lambda"]
p = {"frozen_on": "V07",
     "readout_samples_kept": [3, 220], "t_steady_ms": 10000,
     "gating": "end-expiration third of the steady non-SI spokes; polarity = histogram mode and image SI shift, must agree",
     "coil_maps": "echo 0, r0 spokes: bart nufft -i -x 112:112:112 -> bart fft -u 7 -> bart ecalib -t 0.001 -c 0.0 -r 24 -m 1",
     "r0_ref": f"bart pics -t <traj> -l2 -r 1e-3 -i {it} -S",
     "ref_scaling": "BART's own (-d5 'Inverse scaling of the data'), reused as -w by every subset of the same echo",
     "l1": f"bart pics -t <traj> -l1 -r {lam} -i {it} -e -S -w <ref scaling>",
     "heldout": "5 % of REF, seed 0", "subsets": "whole interleaves, target AF 8/12/16, AF = 19704 / spokes",
     "threads": 32, "parallel_jobs": 1, "nice": 10}
json.dump(p, open(f"{T}/params_frozen.json", "w"), indent=1)
print(json.dumps(p, indent=1))
EOF
```

- [ ] **Step 2: 写 `run_subject.sh` 与 `run_rest.sh`，脱离会话启动**

```bash
cat > /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/run_subject.sh <<'EOF'
#!/usr/bin/env bash
# Spec 5.8: phase-1 pipeline for one subject whose raw.h5 and spokes.npz exist (Tasks 6 and 8).
# Usage: run_subject.sh <Vxx> <iterations>
set -euo pipefail
S=$1
IT=$2
cd /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts
./py s05_coilmaps.py "$S"
./py s06_recon.py "$S" --set r0 --iters "$IT"
./py s11_figures.py "$S" --tag r0
./py s06_recon.py "$S" --set low --echoes 0 --iters "$IT"
./py s06_recon.py "$S" --set high --echoes 0 --iters "$IT"
./py s07_polarity.py "$S"
./py s06_recon.py "$S" --set ref --debug --iters "$IT"
./py s08_checks.py "$S" --ref-iters "$IT"
./py s11_figures.py "$S" --tag ref
echo "PIPELINE_DONE $S"
EOF
cat > /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/run_rest.sh <<'EOF'
#!/usr/bin/env bash
# Spec 5.8: V03, V04, V08 one after the other with the parameters frozen on V07; stops at the first failure.
set -uo pipefail
T=/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE
IT=$("$T/scripts/py" -c "import json; print(json.load(open('$T/logs/checks_V07.json'))['ref_iters'])")
for S in V03 V04 V08; do
  if bash "$T/scripts/run_subject.sh" "$S" "$IT" > "$T/logs/pipeline_$S.log" 2>&1; then
    echo "OK $S"
  else
    echo "FAILED $S (see logs/pipeline_$S.log)"
    exit 1
  fi
done
echo ALL_DONE
EOF
setsid nohup bash /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/run_rest.sh > /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/logs/run_rest.log 2>&1 < /dev/null &
```

- [ ] **Step 3: 等它跑完**

Run（每 30–60 分钟看一次）：`cat /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/logs/run_rest.log; tail -3 /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/logs/pipeline_V0*.log`
Expected: `OK V03`、`OK V04`、`OK V08`、`ALL_DONE`。总共约 7–12 小时。出现 `FAILED <Vxx>` 就停：看对应 `pipeline_<Vxx>.log` 的最后一步，属于停止条件的（极性不一致、核对不过）照实报告，不绕过。

- [ ] **Step 4: 逐例核对**

Run: `cat /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/logs/{coilmaps,polarity,checks}_V0{3,4,8}.json`
Expected: 每例 `sos_within_5pct_in_support` ≥ 0.95、`"agreed": true`、`"iter_ok": true`、`"scale_ok": true`。再用 Read 看 `/home/congcongliu/figs/cardiac_fr/V0{3,4,8}_r0.png` 与 `_ref.png`：解剖方位必须与 V07 一致（心在肝顶上方、同一个轴向）。有一例方位不同就停：它的极性判据用的是 V07 的 `orientation.json`，结论不能用，报告给用户。

- [ ] **Step 5: 同步镜像并提交**

```bash
rsync -a --exclude '__pycache__' /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts /data0/congcong/code/GS/mcgs_cardiac_wt/cardiac_freerunning_megre/
cp /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/params_frozen.json /data0/congcong/code/GS/mcgs_cardiac_wt/cardiac_freerunning_megre/
cd /data0/congcong/code/GS/mcgs_cardiac_wt && git add cardiac_freerunning_megre/scripts/{run_subject.sh,run_rest.sh} cardiac_freerunning_megre/params_frozen.json && git commit -m "cardiac: parameters frozen on V07; coil maps, R0 and REF for V03, V04, V08"
```

### Task 13: 记录与交接

**Files:**
- Create: 课题 `README.md`
- Modify: `/data0/congcong/code/GS/Results_20260908/README.md`（课题表加一行）
- Create: `cardiac_freerunning_megre/{README.md,logs/,figures/}`（镜像）
- Modify: worktree 的 `CLAUDE.md`（顶部加本分支一节）、`STATUS.md`（改写为这条线的交接）

**Interfaces:**
- Consumes: 前面各任务的日志、json 与图。
- Produces: 可冷启动的交接：新会话读 `CLAUDE.md` → `STATUS.md` → spec → 本计划 → 课题 README 就能接着做第二阶段。

- [ ] **Step 1: 写课题 README**

按下面的结构写 `/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/README.md`，每节的内容从括号里指定的文件原样摘取，数字不许凭记忆写：

```markdown
# Cardiac_FreeRunning_MEGRE — free-running multi-echo 3D radial whole heart, phase 1

Reading, BART reference reconstructions and the evaluation gauge for the free-running 8-echo GRE data of
Mackowiak et al. (MRM 2023; Zenodo 7621356 / 7615780, CC-BY-4.0), subjects V07, V03, V04, V08.
No MC-GS yet (phase 2). No fully sampled ground truth exists for this data.
Spec: /data0/congcong/code/GS/mcgs_cardiac_wt/docs/superpowers/specs/2026-09-10-cardiac-freerunning-phase1-design.md
Plan: /data0/congcong/code/GS/mcgs_cardiac_wt/docs/superpowers/plans/2026-09-10-cardiac-freerunning-phase1.md
Code: worktree /data0/congcong/code/GS/mcgs_cardiac_wt, branch cardiac-freerunning, package cardiac_fr/;
drivers in scripts/, always run through scripts/py (GSTMI python, CPU only, nice 10).

## Data                       (spec section 3 table + logs/meta_<Vxx>.json)
## Pipeline and how to rerun  (one row per script s01..s11 with the exact command and its outputs)
## Verification               (5.1 logs/xcheck_reader_V07.txt; 5.2 logs/s02_edge_check.txt;
                               5.3 logs/xcheck_traj.txt; 5.4 logs/physio_*.json + logs/polarity_*.json;
                               5.5 logs/coilmaps_*.json; 5.6 logs/checks_*.json;
                               5.7 gauge.json + logs/selfcheck_V07.tsv + logs/selfcheck_V07.json)
## Orientation                (orientation.json and what decided it)
## Timings                    (median seconds per run type from logs/recon_<Vxx>.jsonl)
## Figures                    (one per line: URL http://localhost:8765/cardiac_fr/<name>.png, then the path)
## Known limitations          (spec section 7, plus what the images showed)
## Gotchas that cost time     (-e for radial l1, -S, the noise ADC, readout edges, zsh, nice/one job)
```

- [ ] **Step 2: 在归档根 README 课题表加一行**

在 `/data0/congcong/code/GS/Results_20260908/README.md` 的 `| folder | what | status |` 表末尾加：

```markdown
| `Cardiac_FreeRunning_MEGRE/` | 心脏自由运行多回波 3D 径向（Mackowiak 2023，V07/V03/V04/V08）第一阶段：twix 读取、BART 运动平均与呼气末参考重建、留出辐条口径（V07 自检）。无全采样真值；MC-GS 的径向前向留第二阶段。代码在 worktree mcgs_cardiac_wt，分支 cardiac-freerunning。 | <收尾日期>, phase 1 complete: 4 subjects R0 + REF, gauge self-check PASS on V07 |
```

- [ ] **Step 3: 同步镜像（README、日志、图、json）**

```bash
M=/data0/congcong/code/GS/mcgs_cardiac_wt/cardiac_freerunning_megre
T=/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE
rsync -a --exclude '__pycache__' $T/scripts $M/
rsync -a --include '*.json' --include '*.jsonl' --include '*.txt' --include '*.tsv' --include 'run_*.log' --include 'pipeline_*.log' --include 's01_read_all.log' --exclude '*' $T/logs/ $M/logs/
rsync -a --include '*.png' --exclude '*' $T/figures/ $M/figures/
cp $T/README.md $T/orientation.json $T/gauge.json $T/params_frozen.json $M/
du -sh $M
```

Expected: 镜像目录几十 MB 以内（BART 的 `-d5` 日志没有同步）。

- [ ] **Step 4: 改 worktree 的 `CLAUDE.md`：在文件最前面加这一节**

```markdown
# This branch: cardiac-freerunning (cardiac line, phase 1)

Read in this order: this section -> `STATUS.md` -> the spec
`docs/superpowers/specs/2026-09-10-cardiac-freerunning-phase1-design.md` -> the plan
`docs/superpowers/plans/2026-09-10-cardiac-freerunning-phase1.md` -> the topic README
`/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/README.md`.

- Code: package `cardiac_fr/` with CPU tests `tests/test_cardiac_fr_*.py`; drivers live in the topic's
  `scripts/` and always run through `scripts/py` (GSTMI python, CPU only, nice 10, 32 BART threads).
- Isolation: never edit the main checkout (`Paired_ARC_GS_general_framework`) or `mcgs_af16_wt`; phase 1
  uses no GPU. Raw data is read-only; derived data lives in `/data2/congcong/data/cardiac_3d_cine_public/derived/megre/`.
- Tests: `cd /data0/congcong/code/GS/mcgs_cardiac_wt && /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/py -m pytest tests/test_cardiac_fr_*.py -q` (26 tests).

The framework notes below come from `autoresearch-t1t2` and still describe the MC-GS code.

---

```

- [ ] **Step 5: 改写 `STATUS.md`（交接五段）**

整份替换为这条线的状态，固定五段，每条结论后面贴可粘贴的命令与原始输出（来自 Step 1 用到的日志）：

```markdown
# STATUS — <收尾日期>（分支 `cardiac-freerunning`，tag `handoff/<收尾日期>-cardiac`）

## 1. 已完成且已验证
（读取、首尾、轨迹、生理、线圈图、R0/REF、迭代与缩放、口径自检，各一条：命令 + 原始输出摘录）
## 2. 待用户拍板
（至少：第二阶段选哪两个回波还是推广到 8 个；是否推送分支与 tag；运行中新发现的问题）
## 3. 下一步
（第二阶段 spec：torchkbnufft 前向替换 fft3c×掩膜、与 BART nufft 的一致性测试、回波选择、GPU 用前查 nvidia-smi）
## 4. 已知的坑与别重做
（-l1 必须 -e；-S；第 0 条是噪声；读出首三点污染；zsh 不分词；负载高时 nice + 单任务；别再全网找数据）
## 5. 关键决定的"为什么"
（为什么走 BART 而不是旧重建代码只换参数；为什么参考用 CG-SENSE；为什么排名以留出 NRMSE 为主）
```

- [ ] **Step 6: 最后一次测试、提交、打 tag**

Run: `cd /data0/congcong/code/GS/mcgs_cardiac_wt && /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/py -m pytest tests/test_cardiac_fr_twix.py tests/test_cardiac_fr_trajectory.py tests/test_cardiac_fr_physio.py tests/test_cardiac_fr_bart.py tests/test_cardiac_fr_gauge.py -q`
Expected: `26 passed`

```bash
cd /data0/congcong/code/GS/mcgs_cardiac_wt && git add -f cardiac_freerunning_megre CLAUDE.md STATUS.md && git status --short && git commit -m "cardiac: phase 1 records - topic README, logs, figures, handoff (CLAUDE.md, STATUS.md)" && git tag "handoff/$(date +%F)-cardiac" && git log --oneline -3 && git tag --list 'handoff/*cardiac'
```

Expected: `git status --short` 只列出镜像目录、`CLAUDE.md`、`STATUS.md`；提交作者是 `Congcong Liu`，消息里没有 trailer；tag 已建立。确认主树没被碰：`git -C /data0/congcong/code/GS/Paired_ARC_GS_general_framework status --short | head` 的输出与开工前一致（开工前有 2 个已修改文件与若干未跟踪文件，都不是本线产生的）。

- [ ] **Step 7: 更新记忆并向用户汇报**

更新 `/home/congcongliu/.claude/projects/-home-congcongliu/memory/cardiac-fr-phase1.md`（第一阶段结果、关键数字、tag）与 `MEMORY.md` 对应行。给用户的汇报：先结论，再列关键数字与图（每张图两行：URL，再绝对路径），最后问两件事：是否推送 `cardiac-freerunning` 与 tag 到 GitHub；第二阶段选哪两个回波（或推广到 8 个）。

