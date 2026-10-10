# sg2x2 core test on the 16-pair set — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the four ablation cells runnable on the 13 test subjects and the development subject of the local UIH T1 + FLAIR set (h5 export), and summarise them at subject level, without changing anything the pilot ran on.

**Architecture:** A one-shot export (`scripts/sg2x2_core_export.py`) turns each subject's h5 pair into the pilot's frame (readout FFT, crop of the zero-padded partition axis, axis swap) and computes fresh BART ESPIRiT maps with the pilot's recipe; outputs live on /data2. A subjects table (`scripts/sg2x2_core_subjects.py`) is the single list of subjects, roles and paths. Per-subject base configs are generated copies of the three development configs with only the two data paths changed; `scripts/sg2x2_cells.py`, `run_sg2x2.sh` and `scripts/sg2x2_manifest.py` gain an optional subject (absent = today's behaviour, byte-for-byte the same arguments). `scripts/sg2x2_core_table.py` averages seeds within a subject and then reports paired A−B, C−D and the interaction across test subjects with bootstrap intervals.

**Tech Stack:** Python 3 (numpy, h5py, PyYAML, pytest) in `/home/congcongliu/anaconda3/envs/GSTMI`, BART v1.0.00 at `/home/congcongliu/src/bart/bart`, bash.

**Spec:** `docs/experiments/2026-10-09-mcgs-shared-geometry-2x2-ablation.md` (§4 initialisation and fairness, §5.1 data, §5.3 statistics). Data facts: `docs/superpowers/records/2026-10-10-sg2x2-core-test-data-inventory.md` (the conversion recipe is its §3.3, the subjects its Table D and §4(b)).

## Global Constraints

- Environment: `PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python`; every Python command as `env PYTHONNOUSERSITE=1 PYTHONPATH= ... $PY ...`; tests `env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= $PY -m pytest <files> -q -p no:cacheprovider` (every test in this plan runs on the CPU; never use a GPU in Tasks 1–4).
- Worktree `/data0/congcong/code/GS/mcgs_sg2x2_wt`, branch `exp/shared-geometry-2x2-ablation-20261009`. The zsh tool shell does not word-split unquoted variables: quote them, or put multi-line logic in a script file.
- User decisions (2026-10-10, "continue" on the recommendations): mask = the pilot's vdPoisMex AF16 mask `/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/data/mask_af16_3d.h5` (2,966 points, T1 15.19×) for every subject; development subject = `jiangyizhou_20230705` only (h5 export, re-run for comparability); init support = the pilot's `configs/sg2x2/init_support_af16_jyz.npz` for every subject; the two `790` files excluded; coil maps = fresh BART ESPIRiT `ecalib -t 0.001 -c 0.0 -r 24 -m 1` (the pilot's recipe); FLAIR h5 used as it is (the displaced low-ky columns stay); FLAIR `known_zero: true` (the repository's estimator, as in the pilot); seed 42 first.
- Existing files that may change: `scripts/sg2x2_cells.py`, `run_sg2x2.sh`, `scripts/sg2x2_manifest.py` (this branch's own tools), additively — without a subject they must produce exactly today's output, and every existing test must pass unchanged. Nothing under `paired_gs/`, `arc/`, `configs/sg2x2/af16_jyz_*.yaml`, `configs/sg2x2/init_support_af16_jyz.npz` changes.
- Data: the sources under `/data1/data1_congcong/data/3D/{T1,FLAIR}_csm_ksp/` are read only. Outputs go only to `/data2/congcong/data/sg2x2_core_h5/<subject>/`. Nothing is ever overwritten or removed — not data, not scratch, not test files; open files with mode `"x"` or check existence first; no self-cleaning temporary directories (use pytest's `tmp_path`); the CFL intermediates stay. Write files with the Write/Edit tools: a hook rejects shell commands whose text contains removal-related words, even inside heredocs.
- Commits: author is the repository identity (already configured, `Congcong Liu <liuc23539@gmail.com>`), plain one-line messages, no `Co-Authored-By`, no `Generated with`, no AI trailer of any kind; stage named files only.
- Run names: core cells `sg2x2_core_af16_<subject>_s<seed>_<CELL>`; smoke runs `sg2x2_core_smoke_<subject>_s<seed>_<CELL>`.

## Review Focus

1. A subject whose export stopped half-way (T1 written, FLAIR not) must never be queued: the export writes `export.json` only after both contrasts pass, and Task 4's verification requires that file for every subject before it writes the queue. Pinned in Task 1 (`test_export_pair_refuses_a_second_export`, `test_export_pair_stops_on_an_offcentre_dc`) and Task 4 Step 1.
2. A run whose resolved config points at another subject's data (a wrong `SUBJECT`, a stale config) must fail its audit. Pinned in Task 2 (`test_audit_rejects_data_of_another_subject`).
3. An unknown subject must stop the launcher before any training: `cell_config` raises, `sg2x2_cells.py` prints nothing, `run_sg2x2.sh` exits 2. Pinned in Task 2 (`test_unknown_subject_raises`, `test_cli_subject_and_unknown_subject`).
4. The development subject must never enter the test statistics, and a subject with a missing cell must leave only the rows that need that cell. Pinned in Task 3 (`test_dev_subject_never_enters_the_statistics`, `test_missing_cell_drops_the_subject_from_that_row_only`).
5. Without a subject, the cell arguments and the manifest's `config` are byte-for-byte today's. Pinned in Task 2 (`test_no_subject_is_todays_behaviour`).

---

### Task 1: Subjects table and the export (code and tests; no real export yet)

**Files:**
- Create: `scripts/sg2x2_core_subjects.py`
- Create: `scripts/sg2x2_core_export.py`
- Test: `tests/test_sg2x2_core_export.py`

**Interfaces:**
- Produces (`sg2x2_core_subjects`): `SRC: dict[str, str]`, `OUT_ROOT: str`, `CONTRASTS = ("T1", "FLAIR")`, `SUBJECTS: dict[str, tuple[str, str]]` (key → (h5 file name, "dev" | "test")), `EXCLUDED: dict[str, str]`, `core_test_subjects() -> list[str]`, `subject_paths(subject: str, out_root: str = OUT_ROOT) -> dict` with keys `role`, `src` (`{C: path}`), `out`, `ksp` (`{C: path}`), `csm` (`{C: path}`), `bart`, `report`.
- Produces (`sg2x2_core_export`): `fft1c(x, axis)`, `convert(raw, n_spe=176) -> (k, info)`, `write_cfl(base, arr)`, `read_cfl(base)`, `espirit(k, base, bart=BART, ecalib=ECALIB) -> (maps, info)`, `write_h5(path, key, arr) -> md5`, `export_pair(src: dict, out: str, n_spe=176, bart=BART, ecalib=ECALIB) -> dict`; CLI `scripts/sg2x2_core_export.py SUBJECT [SUBJECT ...]`.

- [ ] **Step 1: Write the failing tests**

```python
"""The sg2x2 core-test subjects table and the h5 -> trainer-format export (scripts/sg2x2_core_subjects.py,
scripts/sg2x2_core_export.py) on synthetic data: the conversion recovers a known k-space, the checks stop a wrong
crop, BART ESPIRiT maps have unit sum-of-squares, nothing is ever overwritten.
Run: PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_sg2x2_core_export.py -q
"""
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sg2x2_core_export import (BART, convert, espirit, export_pair, fft1c, read_cfl,  # noqa: E402
                               write_cfl, write_h5)
from sg2x2_core_subjects import EXCLUDED, OUT_ROOT, SUBJECTS, core_test_subjects, subject_paths  # noqa: E402

needs_bart = pytest.mark.skipif(not os.path.exists(BART), reason="BART binary not found")
SMALL_ECALIB = ("ecalib", "-t", "0.001", "-c", "0.0", "-r", "12", "-m", "1")


def _fft3c(x):
    ax = (-3, -2, -1)
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)


def _ifft1c(x, axis):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis, norm="ortho"), axes=axis)


def _synthetic(ro=24, nc=4, ny=20, npe=24, n_spe=16, shutter=False, seed=0):
    """a phantom k-space (coil, RO, PE, SPE) and its raw h5 layout (RO in image domain, SPE padded at the low end).
    A small noise floor keeps every k-space sample far above the export's 1e-12 'acquired' threshold (a pure
    Gaussian phantom's outer k-space would fall below it and fail the T1 check)."""
    rng = np.random.default_rng(seed)
    x, y, z = np.meshgrid(np.linspace(-1, 1, ro), np.linspace(-1, 1, npe), np.linspace(-1, 1, n_spe), indexing="ij")
    img = np.exp(-(x ** 2 + y ** 2 + z ** 2) / 0.3) + 0.01 * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    coils = np.stack([np.exp(1j * 0.5 * c * x) * (1.0 + 0.3 * c * y) for c in range(nc)])
    k = _fft3c(coils * img[None]).astype(np.complex64)                        # (coil, RO, PE, SPE)
    if shutter:
        ky, kz = np.meshgrid(np.linspace(-1, 1, npe), np.linspace(-1, 1, n_spe), indexing="ij")
        k = k * ((ky ** 2 + kz ** 2) <= 1.0)[None, None]
    hybrid = _ifft1c(k, axis=1)                                               # readout back to image domain
    raw = np.zeros((ro, nc, ny, npe), np.complex64)
    raw[:, :, ny - n_spe:, :] = np.transpose(hybrid, (1, 0, 3, 2))            # (RO, coil, SPE, PE)
    return raw, k


def test_subjects_table():
    assert len(SUBJECTS) == 14
    assert [s for s, (_, r) in SUBJECTS.items() if r == "dev"] == ["jiangyizhou_20230705"]
    assert len(core_test_subjects()) == 13 and "jiangyizhou_20230705" not in core_test_subjects()
    assert not set(EXCLUDED) & set(SUBJECTS)
    for s in SUBJECTS:
        p = subject_paths(s)
        assert p["out"] == f"{OUT_ROOT}/{s}" and p["report"] == f"{OUT_ROOT}/{s}/export.json"
        assert p["ksp"]["FLAIR"] == f"{OUT_ROOT}/{s}/ksp_FLAIR.h5" and p["csm"]["T1"] == f"{OUT_ROOT}/{s}/csm_espirit_T1.h5"
        for c in ("T1", "FLAIR"):
            assert os.path.exists(p["src"][c]), p["src"][c]


def test_convert_recovers_the_kspace():
    raw, k_true = _synthetic()
    k, info = convert(raw, n_spe=16)
    assert k.shape == (4, 24, 24, 16) and k.dtype == np.complex64
    assert np.allclose(k, k_true, atol=1e-5)
    assert info["lo"] == 4 and info["ny"] == 20 and info["outside_rel_max"] == 0.0
    assert info["dc_pe_spe"] == info["centre_pe_spe"] == [12, 8]
    assert info["acquired_frac"] == 1.0


def test_convert_rejects_bad_shapes():
    with pytest.raises(ValueError):
        convert(np.zeros((4, 4, 4), np.complex64), n_spe=2)
    with pytest.raises(ValueError):
        convert(np.zeros((8, 2, 6, 8), np.complex64), n_spe=7)


def test_fft1c_is_unitary_and_centred():
    x = np.zeros((1, 8, 1), np.complex64)
    x[0, 4, 0] = 1.0                                  # the centre sample
    X = fft1c(x, axis=1)
    assert np.allclose(np.abs(X), 1 / np.sqrt(8)) and np.isclose(np.sum(np.abs(X) ** 2), 1.0)


def test_cfl_round_trip_and_refusal(tmp_path):
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((5, 4, 3, 2)) + 1j * rng.standard_normal((5, 4, 3, 2))).astype(np.complex64)
    base = str(tmp_path / "a")
    write_cfl(base, a)
    assert np.array_equal(read_cfl(base), a)
    with pytest.raises(FileExistsError):
        write_cfl(base, a)


def test_write_h5_refuses_to_overwrite(tmp_path):
    p = str(tmp_path / "x.h5")
    a = np.ones((2, 3, 4, 5), np.complex64)
    m = write_h5(p, "kspace", a)
    assert len(m) == 32
    with h5py.File(p, "r") as h:
        assert np.array_equal(h["kspace"][...], a)
    with pytest.raises((FileExistsError, OSError)):
        write_h5(p, "kspace", a)


@needs_bart
def test_espirit_maps_have_unit_sos(tmp_path):
    _, k = _synthetic()
    maps, info = espirit(k, str(tmp_path / "T1"), ecalib=SMALL_ECALIB)
    assert maps.shape == k.shape and maps.dtype == np.complex64
    sos = np.sqrt((np.abs(maps) ** 2).sum(axis=0))
    assert info["sos_frac_within_1e-3"] >= 0.999 and np.isclose(float(np.median(sos)), 1.0, atol=1e-3)
    assert os.path.exists(str(tmp_path / "T1_ksp.cfl")) and os.path.exists(str(tmp_path / "T1_map.cfl"))
    with pytest.raises(FileExistsError):
        espirit(k, str(tmp_path / "T1"), ecalib=SMALL_ECALIB)


def _write_src(path, raw):
    with h5py.File(path, "x") as h:
        h.create_dataset("kspace", data=raw)


@needs_bart
def test_export_pair_end_to_end(tmp_path):
    raw_t1, k_t1 = _synthetic()
    raw_fl, _ = _synthetic(shutter=True)
    src = {"T1": str(tmp_path / "t1.h5"), "FLAIR": str(tmp_path / "fl.h5")}
    _write_src(src["T1"], raw_t1)
    _write_src(src["FLAIR"], raw_fl)
    out = str(tmp_path / "subj")
    rep = export_pair(src, out, n_spe=16, ecalib=SMALL_ECALIB)
    for c in ("T1", "FLAIR"):
        for f in (f"ksp_{c}.h5", f"csm_espirit_{c}.h5"):
            assert os.path.exists(os.path.join(out, f))
    with h5py.File(os.path.join(out, "ksp_T1.h5"), "r") as h:
        assert np.allclose(h["kspace"][...], k_t1, atol=1e-5)
    with h5py.File(os.path.join(out, "csm_espirit_FLAIR.h5"), "r") as h:
        assert h["map"].shape == (4, 24, 24, 16)
    saved = json.load(open(os.path.join(out, "export.json")))
    assert saved == json.loads(json.dumps(rep))
    assert saved["contrasts"]["T1"]["acquired_frac"] == 1.0 and saved["contrasts"]["FLAIR"]["acquired_frac"] < 1.0
    assert len(saved["contrasts"]["FLAIR"]["csm_md5"]) == 32


@needs_bart
def test_export_pair_refuses_a_second_export(tmp_path):
    raw, _ = _synthetic()
    src = {"T1": str(tmp_path / "t1.h5"), "FLAIR": str(tmp_path / "fl.h5")}
    _write_src(src["T1"], raw)
    _write_src(src["FLAIR"], raw)
    out = str(tmp_path / "subj")
    export_pair(src, out, n_spe=16, ecalib=SMALL_ECALIB)
    with pytest.raises(FileExistsError):
        export_pair(src, out, n_spe=16, ecalib=SMALL_ECALIB)


def test_export_pair_stops_on_an_offcentre_dc(tmp_path):
    raw, _ = _synthetic()
    raw = np.roll(raw, 3, axis=3)                     # the PE axis shifted by 3 samples
    src = {"T1": str(tmp_path / "t1.h5"), "FLAIR": str(tmp_path / "fl.h5")}
    _write_src(src["T1"], raw)
    _write_src(src["FLAIR"], raw)
    out = str(tmp_path / "subj")
    with pytest.raises(RuntimeError, match="DC"):
        export_pair(src, out, n_spe=16, ecalib=SMALL_ECALIB)
    assert not os.path.exists(os.path.join(out, "export.json"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= $PY -m pytest tests/test_sg2x2_core_export.py -q -p no:cacheprovider`
Expected: collection error `ModuleNotFoundError: No module named 'sg2x2_core_export'`.

- [ ] **Step 3: Write the subjects table**

```python
#!/usr/bin/env python3
"""The sg2x2 core-test subjects and their file paths.

Source: the data inventory of 2026-10-10 (docs/superpowers/records/2026-10-10-sg2x2-core-test-data-inventory.md).
Fourteen regular T1 + FLAIR pairs of the UIH 3D brain set: jiangyizhou_20230705 is the development subject (the
pilot's subject, here in its h5 export), the other thirteen are test subjects. The two `790` files are excluded: they
are second scans of jiangdian_20230705 and tangbing_20230705 on another protocol (inventory sections 2 and 5).
"""
from __future__ import annotations

SRC = {"T1": "/data1/data1_congcong/data/3D/T1_csm_ksp", "FLAIR": "/data1/data1_congcong/data/3D/FLAIR_csm_ksp"}
OUT_ROOT = "/data2/congcong/data/sg2x2_core_h5"
CONTRASTS = ("T1", "FLAIR")

SUBJECTS = {
    "jiangyizhou_20230705": ("Train_3dBrain_jiangyizhou_20230705-152818-1450_152931.h5", "dev"),
    "huangyuwei_20230705": ("Train_3dBrain_huangyuwei_20230705-160927-1451_161015.h5", "test"),
    "jiangdian_20230705": ("Train_3dBrain_jiangdian_20230705-144538-1449_144607.h5", "test"),
    "sunxinyi_20230705": ("Train_3dBrain_sunxinyi_20230705-164717-1452_164752.h5", "test"),
    "tangbing_20230705": ("Train_3dBrain_tangbing_20230705-132343-1448_132439.h5", "test"),
    "liaozhenghan_20231030": ("Train_3dBrain_liaozhenghan_20231030-141603-1576_141649.h5", "test"),
    "zhoushuo_20231030": ("Train_3dBrain_zhoushuo_20231030-161805-1579_161837.h5", "test"),
    "shiyixuan_20231101": ("Train_3dBrain_shiyixuan_20231101-163220-1587_163304.h5", "test"),
    "zhangyangming_20231101": ("Train_3dBrain_zhangyangming_20231101-144341-1584_144414.h5", "test"),
    "yangkun_20231102": ("Train_3dBrain_yangkun_20231102-131858-1588_131925.h5", "test"),
    "zhangxiqian_20231102": ("Train_3dBrain_zhangxiqian_20231102-142131-1589_143120.h5", "test"),
    "chenshuaibao_20231106": ("Train_3dBrain_chenshuaibao_20231106-114149-1598_114223.h5", "test"),
    "lishengbo_20231106": ("Train_3dBrain_lishengbo_20231106-154803-1601_154834.h5", "test"),
    "litianyu_20231107": ("Train_3dBrain_litianyu_20231107-131010-1602_131057.h5", "test"),
}
EXCLUDED = {
    "790_jiangdian_20230706": "second scan of jiangdian_20230705 on another protocol (50 % slab oversampling)",
    "790_tangbin_20230706": "second scan of tangbing_20230705 on another protocol (50 % slab oversampling)",
}


def core_test_subjects() -> list[str]:
    return [s for s, (_, role) in SUBJECTS.items() if role == "test"]


def subject_paths(subject: str, out_root: str = OUT_ROOT) -> dict:
    """the subject's two source h5 files and the export's output files"""
    fname, role = SUBJECTS[subject]
    out = f"{out_root}/{subject}"
    return {"role": role, "src": {c: f"{SRC[c]}/{fname}" for c in CONTRASTS}, "out": out,
            "ksp": {c: f"{out}/ksp_{c}.h5" for c in CONTRASTS},
            "csm": {c: f"{out}/csm_espirit_{c}.h5" for c in CONTRASTS},
            "bart": f"{out}/bart", "report": f"{out}/export.json"}
```

- [ ] **Step 4: Write the export**

```python
#!/usr/bin/env python3
"""Export core-test subjects from the UIH h5 set into the trainer's format (inventory section 3.3).

    env PYTHONNOUSERSITE=1 PYTHONPATH= OMP_NUM_THREADS=24 nice -n 10 $PY scripts/sg2x2_core_export.py SUBJECT [SUBJECT ...]

Per contrast: read `kspace` (RO, coil, SPE, PE) -- the readout axis is stored in image domain -- move the coil axis
first, centred orthonormal 1-D FFT along the readout, crop the SPE axis to its acquired rows [NY - 176, NY), swap the
two phase-encode axes -> (coil, RO, PE, SPE) = (12, 256, 256, 176) pure k-space, the pilot's frame; write ksp_<C>.h5
(key `kspace`). Then the pilot's ESPIRiT recipe: write the k-space as a BART CFL (RO, PE, SPE, coil) under <out>/bart/,
run `bart ecalib -t 0.001 -c 0.0 -r 24 -m 1`, write the maps coil first as csm_espirit_<C>.h5 (key `map`).

Checks (the export stops on a failure and writes no export.json): nothing acquired outside the cropped rows; the
k-space DC within one sample of the centre of the (PE, SPE) plane; T1 acquires the whole cropped plane; ESPIRiT
sum-of-squares within 1e-3 of 1 on at least 99.9 % of the voxels. Every number goes to <out>/export.json.

Never overwrites and never removes a file. The CFL intermediates under <out>/bart/ (2.2 GB per contrast) are kept;
delete them by hand when the maps are no longer needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sg2x2_core_subjects import CONTRASTS, SUBJECTS, subject_paths  # noqa: E402

N_SPE = 176
BART = "/home/congcongliu/src/bart/bart"
ECALIB = ("ecalib", "-t", "0.001", "-c", "0.0", "-r", "24", "-m", "1")


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fft1c(x: np.ndarray, axis: int) -> np.ndarray:
    """centred orthonormal 1-D FFT (the one-axis form of paired_gs.data.transforms.fft3c)"""
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis, norm="ortho"), axes=axis)


def convert(raw: np.ndarray, n_spe: int = N_SPE) -> tuple[np.ndarray, dict]:
    """raw (RO, coil, NY, PE) with the readout in image domain -> k-space (coil, RO, PE, SPE), and the numbers checked"""
    if raw.ndim != 4:
        raise ValueError(f"expected a 4-D (RO, coil, NY, PE) array, got shape {raw.shape}")
    ny, npe = raw.shape[2], raw.shape[3]
    lo = ny - n_spe
    if lo < 0:
        raise ValueError(f"NY {ny} is smaller than the {n_spe} acquired SPE rows")
    k = fft1c(np.moveaxis(raw, 1, 0), axis=1).astype(np.complex64)      # (coil, RO, NY, PE)
    plane = np.sqrt((np.abs(k) ** 2).sum(axis=(0, 1)))                   # (NY, PE)
    pmax = float(plane.max())
    outside = float(plane[:lo].max()) / pmax if lo > 0 else 0.0
    k = np.ascontiguousarray(np.swapaxes(k[:, :, lo:, :], 2, 3))          # (coil, RO, PE, SPE)
    block = np.swapaxes(plane[lo:], 0, 1)                                 # (PE, SPE)
    dc = np.unravel_index(int(np.argmax(block)), block.shape)
    info = {"raw_shape": list(raw.shape), "ny": int(ny), "lo": int(lo), "shape": list(k.shape),
            "outside_rel_max": outside, "acquired_frac": float((block > 1e-12 * pmax).mean()),
            "dc_pe_spe": [int(dc[0]), int(dc[1])], "centre_pe_spe": [int(npe // 2), int(n_spe // 2)]}
    return k, info


def write_cfl(base: str, arr: np.ndarray) -> None:
    """BART CFL (column-major complex64) plus header; refuses to overwrite"""
    for ext in (".cfl", ".hdr"):
        if os.path.exists(base + ext):
            raise FileExistsError(base + ext)
    np.asarray(arr, dtype=np.complex64).ravel(order="F").tofile(base + ".cfl")
    with open(base + ".hdr", "x") as f:
        f.write("# Dimensions\n" + " ".join(str(d) for d in arr.shape) + " \n")


def read_cfl(base: str) -> np.ndarray:
    with open(base + ".hdr") as f:
        dims = [int(v) for v in f.read().splitlines()[1].split()]
    return np.fromfile(base + ".cfl", dtype=np.complex64).reshape(dims, order="F")


def espirit(k: np.ndarray, base: str, bart: str = BART, ecalib: tuple = ECALIB) -> tuple[np.ndarray, dict]:
    """ESPIRiT maps (coil, RO, PE, SPE) of k (coil, RO, PE, SPE); the BART files <base>_ksp and <base>_map are kept"""
    for ext in (".cfl", ".hdr"):
        if os.path.exists(base + "_map" + ext):
            raise FileExistsError(base + "_map" + ext)
    write_cfl(base + "_ksp", np.moveaxis(k, 0, 3))                       # (RO, PE, SPE, coil)
    t0 = time.time()
    env = dict(os.environ, TOOLBOX_PATH=str(Path(bart).parent))
    r = subprocess.run([bart, *ecalib, base + "_ksp", base + "_map"], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"bart ecalib failed (rc {r.returncode}): {r.stderr[-800:]}")
    maps = read_cfl(base + "_map")
    maps = np.ascontiguousarray(np.moveaxis(maps.reshape(maps.shape[:4]), 3, 0)).astype(np.complex64)
    sos = np.sqrt((np.abs(maps) ** 2).sum(axis=0))
    info = {"seconds": round(time.time() - t0, 1), "cmd": " ".join([bart, *ecalib]),
            "sos_min": float(sos.min()), "sos_max": float(sos.max()),
            "sos_frac_within_1e-3": float((np.abs(sos - 1.0) < 1e-3).mean())}
    return maps, info


def write_h5(path: str, key: str, arr: np.ndarray) -> str:
    """write arr under key into a NEW h5 file (mode "x" refuses an existing one); returns the file's md5"""
    chunks = (1, min(32, arr.shape[1]), arr.shape[2], arr.shape[3])
    with h5py.File(path, "x") as h:
        h.create_dataset(key, data=arr, chunks=chunks)
    return md5(path)


def export_pair(src: dict, out: str, n_spe: int = N_SPE, bart: str = BART, ecalib: tuple = ECALIB) -> dict:
    """src {"T1": h5, "FLAIR": h5} -> <out>/ksp_<C>.h5, <out>/csm_espirit_<C>.h5, <out>/bart/, <out>/export.json"""
    report = os.path.join(out, "export.json")
    if os.path.exists(report):
        raise FileExistsError(report)
    os.makedirs(os.path.join(out, "bart"), exist_ok=True)
    rep = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "n_spe": n_spe, "contrasts": {}}
    for c in CONTRASTS:
        with h5py.File(src[c], "r") as f:
            raw = f["kspace"][...]
        k, info = convert(raw, n_spe)
        del raw
        problems = []
        if info["outside_rel_max"] > 1e-6:
            problems.append(f"data outside the cropped rows (relative max {info['outside_rel_max']:.3g})")
        if any(abs(a - b) > 1 for a, b in zip(info["dc_pe_spe"], info["centre_pe_spe"])):
            problems.append(f"k-space DC {info['dc_pe_spe']} is not at the centre {info['centre_pe_spe']}")
        if c == "T1" and info["acquired_frac"] < 0.9999:
            problems.append(f"T1 acquires only {info['acquired_frac']:.4f} of the cropped plane")
        if problems:
            raise RuntimeError(f"{src[c]} ({c}): " + "; ".join(problems))
        info["src"], info["src_md5"] = src[c], md5(src[c])
        info["ksp"] = os.path.join(out, f"ksp_{c}.h5")
        info["ksp_md5"] = write_h5(info["ksp"], "kspace", k)
        maps, einfo = espirit(k, os.path.join(out, "bart", c), bart, ecalib)
        if einfo["sos_frac_within_1e-3"] < 0.999:
            raise RuntimeError(f"{c}: ESPIRiT sum-of-squares off 1 on {1 - einfo['sos_frac_within_1e-3']:.4f} of voxels")
        info["csm"] = os.path.join(out, f"csm_espirit_{c}.h5")
        info["csm_md5"] = write_h5(info["csm"], "map", maps)
        info["espirit"] = einfo
        rep["contrasts"][c] = info
        print(f"[export] {out} {c}: lo={info['lo']} dc={info['dc_pe_spe']} acquired={info['acquired_frac']:.4f} "
              f"ecalib {einfo['seconds']} s, SoS within 1e-3 on {einfo['sos_frac_within_1e-3']:.4f}", flush=True)
        del k, maps
    rep["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(report, "x") as f:
        json.dump(rep, f, indent=2)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subjects", nargs="+", choices=sorted(SUBJECTS))
    a = ap.parse_args()
    for s in a.subjects:
        p = subject_paths(s)
        rep = export_pair(p["src"], p["out"])
        print(f"[export] {s} ({p['role']}) done: {p['report']} "
              f"({rep['started']} -> {rep['finished']})", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= $PY -m pytest tests/test_sg2x2_core_export.py -q -p no:cacheprovider`
Expected: `10 passed`. If a BART test fails only because the synthetic grid is too small for `-r 12` (a BART message about the calibration region), enlarge `_synthetic`'s sizes (keep `n_spe` < `ny`) rather than loosening any check; if the maps' sum-of-squares is not within 1e-3 of 1 on the synthetic data, report it (DONE_WITH_CONCERNS) — do not loosen the export's 1e-3 / 99.9 % check.

- [ ] **Step 6: Commit**

```bash
git add scripts/sg2x2_core_subjects.py scripts/sg2x2_core_export.py tests/test_sg2x2_core_export.py
git commit -m "sg2x2 core test: subjects table and the h5 export (readout FFT, partition crop, fresh BART ESPIRiT), with tests"
```

**Controller step after Task 1 (not an implementer step): start the real export in the background (CPU only, ~45 min):**
```bash
mkdir -p /data2/congcong/data/sg2x2_core_h5
cd /data0/congcong/code/GS/mcgs_sg2x2_wt && setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH= OMP_NUM_THREADS=24 nice -n 10 \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/sg2x2_core_export.py jiangyizhou_20230705 huangyuwei_20230705 \
  jiangdian_20230705 sunxinyi_20230705 tangbing_20230705 liaozhenghan_20231030 zhoushuo_20231030 shiyixuan_20231101 \
  zhangyangming_20231101 yangkun_20231102 zhangxiqian_20231102 chenshuaibao_20231106 lishengbo_20231106 litianyu_20231107 \
  > /data2/congcong/data/sg2x2_core_h5/export_run_20261010.log 2>&1 < /dev/null &
```

---

### Task 2: The subject dimension (configs, cell table, launcher, manifest)

**Files:**
- Create: `scripts/sg2x2_core_configs.py`
- Create: `configs/sg2x2/core/<subject>/{joint,single_T1,single_FLAIR}.yaml` (42 files, generated by the script)
- Modify: `scripts/sg2x2_cells.py` (new `CORE_KIND`, `cell_config`; `cell_args(..., subject=None)`; CLI `--subject`)
- Modify: `run_sg2x2.sh` (env `SUBJECT`)
- Modify: `scripts/sg2x2_manifest.py` (`--subject`; `config` / `config_md5` via `cell_config`; `subject` field; data-path identity check)
- Test: `tests/test_sg2x2_core_configs.py`; append to `tests/test_sg2x2_cells.py` and `tests/test_sg2x2_manifest.py`

**Interfaces:**
- Consumes: `SUBJECTS`, `subject_paths` (Task 1).
- Produces: `cell_config(cell: str, subject: str | None = None) -> str`; `cell_args(cell, npts, seed, support, init="support_file", iters=4000, warmup=500, subject=None)`; `scripts/sg2x2_cells.py CELL ... [--subject S]`; `run_sg2x2.sh` reads `SUBJECT`; `scripts/sg2x2_manifest.py ... [--subject S]`; `audit(cell, resolved_cfg, log_text, args)` checks `args["subject"]` when present; `subject_config(template: dict, subject: str) -> dict`, `write_subject(subject: str, root: Path = ROOT) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_sg2x2_core_configs.py`:
```python
"""Per-subject base configs of the sg2x2 core test (scripts/sg2x2_core_configs.py): copies of the development
configs with only data_path / csm_path changed; the committed files match the generator; nothing is overwritten.
Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_sg2x2_core_configs.py -q
"""
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sg2x2_core_configs import CORE_DIR, TEMPLATES, subject_config, write_subject  # noqa: E402
from sg2x2_core_subjects import SUBJECTS, subject_paths  # noqa: E402


def _strip_paths(cfg):
    for slot in cfg["data"].values():
        slot.pop("data_path")
        slot.pop("csm_path")
    return cfg


@pytest.mark.parametrize("kind", sorted(TEMPLATES))
def test_only_the_two_data_paths_change(kind):
    tmpl = yaml.safe_load(open(ROOT / TEMPLATES[kind]))
    for s in SUBJECTS:
        cfg = subject_config(tmpl, s)
        assert _strip_paths(cfg) == _strip_paths(yaml.safe_load(open(ROOT / TEMPLATES[kind])))
        cfg = subject_config(tmpl, s)
        p = subject_paths(s)
        for slot in cfg["data"].values():
            assert slot["data_path"] == p["ksp"][slot["name"]] and slot["csm_path"] == p["csm"][slot["name"]]
            assert slot["known_zero"] is True if slot["name"] == "FLAIR" else not slot.get("known_zero", False)


def test_committed_configs_match_the_generator():
    for s in SUBJECTS:
        for kind, tpath in TEMPLATES.items():
            got = yaml.safe_load(open(ROOT / CORE_DIR / s / f"{kind}.yaml"))
            assert got == subject_config(yaml.safe_load(open(ROOT / tpath)), s), (s, kind)


def test_write_subject_refuses_to_overwrite(tmp_path):
    for tpath in TEMPLATES.values():
        (tmp_path / tpath).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / tpath, tmp_path / tpath)
    written = write_subject("huangyuwei_20230705", root=tmp_path)
    assert sorted(written) == sorted(f"{CORE_DIR}/huangyuwei_20230705/{k}.yaml" for k in TEMPLATES)
    first = (tmp_path / written[0]).read_text()
    assert first.startswith("# sg2x2 core test, subject huangyuwei_20230705")
    with pytest.raises(FileExistsError):
        write_subject("huangyuwei_20230705", root=tmp_path)
```

Append to `tests/test_sg2x2_cells.py`:
```python
def test_no_subject_is_todays_behaviour():
    from sg2x2_cells import cell_config
    for cell in CELLS:
        assert cell_config(cell) == CELLS[cell].config
        assert cell_args(cell, 3_600_000, 42, SUP) == cell_args(cell, 3_600_000, 42, SUP, subject=None)
        assert _flag_value(cell_args(cell, 3_600_000, 42, SUP), "--base_cfg") == CELLS[cell].config


def test_subject_selects_the_core_config():
    from sg2x2_cells import cell_config
    assert cell_config("A", "huangyuwei_20230705") == "configs/sg2x2/core/huangyuwei_20230705/joint.yaml"
    assert cell_config("C", "huangyuwei_20230705") == "configs/sg2x2/core/huangyuwei_20230705/joint.yaml"
    assert cell_config("B_FLAIR", "litianyu_20231107") == "configs/sg2x2/core/litianyu_20231107/single_FLAIR.yaml"
    assert cell_config("D_T1", "jiangyizhou_20230705") == "configs/sg2x2/core/jiangyizhou_20230705/single_T1.yaml"
    a = cell_args("D_FLAIR", 3_600_000, 42, SUP, subject="yangkun_20231102")
    assert _flag_value(a, "--base_cfg") == "configs/sg2x2/core/yangkun_20231102/single_FLAIR.yaml"
    assert a[:a.index("--base_cfg")] == cell_args("D_FLAIR", 3_600_000, 42, SUP)[:a.index("--base_cfg")]


def test_unknown_subject_raises():
    from sg2x2_cells import cell_config
    with pytest.raises(ValueError):
        cell_config("A", "nobody_20990101")
    with pytest.raises(ValueError):
        cell_args("A", 1, 42, SUP, subject="790_tangbin_20230706")


def test_cli_subject_and_unknown_subject():
    script = str(Path(__file__).resolve().parents[1] / "scripts/sg2x2_cells.py")
    ok = subprocess.run([sys.executable, script, "B_T1", "--npts", "100", "--seed", "42", "--support", SUP,
                         "--subject", "zhoushuo_20231030"], capture_output=True, text=True, check=True).stdout
    assert ok.strip().split("\n") == cell_args("B_T1", 100, 42, SUP, subject="zhoushuo_20231030")
    bad = subprocess.run([sys.executable, script, "B_T1", "--npts", "100", "--seed", "42", "--support", SUP,
                          "--subject", "nobody_20990101"], capture_output=True, text=True)
    assert bad.returncode != 0 and bad.stdout == ""
```
(`pytest` must be imported at the top of `tests/test_sg2x2_cells.py`; add `import pytest` beside the existing imports if it is not there.)

Append to `tests/test_sg2x2_manifest.py`:
```python
def _core_cfg(subject):
    from sg2x2_core_subjects import subject_paths
    p = subject_paths(subject)
    cfg = {"loss": {"tv_weight": 0.10, **{k: v for k, v in RES_ON.items() if k != "lambda_geo"},
                    "kspace_freq_adaptive": True, "kspace_high_weight": 2.2},
           "data": {"plain": {"name": "T1", "data_path": p["ksp"]["T1"], "csm_path": p["csm"]["T1"]},
                    "contrast": {"name": "FLAIR", "data_path": p["ksp"]["FLAIR"], "csm_path": p["csm"]["FLAIR"],
                                 "known_zero": True}}}
    return cfg


def test_audit_without_subject_adds_no_identity_check():
    cfg = _core_cfg("huangyuwei_20230705")
    res = audit("A", cfg, LINE_ON + "\n" + LINE_ON + "\n")
    assert "data_subject" not in res["got"] and "data_subject" not in res["want"]


def test_audit_rejects_data_of_another_subject():
    from sg2x2_manifest import _subject_ok
    cfg = _core_cfg("huangyuwei_20230705")
    assert _subject_ok(cfg, "huangyuwei_20230705") is True
    assert _subject_ok(cfg, "litianyu_20231107") is False
    cfg["data"]["contrast"]["csm_path"] = _core_cfg("litianyu_20231107")["data"]["contrast"]["csm_path"]
    assert _subject_ok(cfg, "huangyuwei_20230705") is False
```
and a full-args check in the style of the existing `test_audit_accepts_full_config_with_matching_args` (read that test first and reuse its fully specified config and log): copy it into `test_audit_subject_check_folds_into_resolved_ok`, replace its data block with `_core_cfg("huangyuwei_20230705")["data"]` (keeping its known_zero values), pass `args={**<its args>, "subject": "huangyuwei_20230705"}` and assert `ok`; then pass `"subject": "litianyu_20231107"` and assert `resolved_ok` is False.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= $PY -m pytest tests/test_sg2x2_core_configs.py tests/test_sg2x2_cells.py tests/test_sg2x2_manifest.py -q -p no:cacheprovider`
Expected: `test_sg2x2_core_configs.py` errors on collection (`No module named 'sg2x2_core_configs'`); the new cell and manifest tests fail (`cell_config` / `_subject_ok` not defined); every pre-existing test still passes.

- [ ] **Step 3: Write the config generator and generate the 42 configs**

```python
#!/usr/bin/env python3
"""Per-subject base configs of the sg2x2 core test.

Copies of the three development configs (configs/sg2x2/af16_jyz_{joint,single_T1,single_FLAIR}.yaml) in which only
data.<slot>.data_path and data.<slot>.csm_path change, to the subject's export (scripts/sg2x2_core_export.py). The
mask (the pilot's vdPoisMex AF16 mask), known_zero, the loss block and everything else keep the development values.
Writes configs/sg2x2/core/<subject>/{joint,single_T1,single_FLAIR}.yaml and refuses to overwrite.

    env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/sg2x2_core_configs.py [SUBJECT ...]     (default: every subject)
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sg2x2_core_subjects import SUBJECTS, subject_paths  # noqa: E402

TEMPLATES = {"joint": "configs/sg2x2/af16_jyz_joint.yaml",
             "single_T1": "configs/sg2x2/af16_jyz_single_T1.yaml",
             "single_FLAIR": "configs/sg2x2/af16_jyz_single_FLAIR.yaml"}
CORE_DIR = "configs/sg2x2/core"


def subject_config(template: dict, subject: str) -> dict:
    cfg = copy.deepcopy(template)
    p = subject_paths(subject)
    for slot in cfg["data"].values():
        slot["data_path"] = p["ksp"][slot["name"]]
        slot["csm_path"] = p["csm"][slot["name"]]
    return cfg


def write_subject(subject: str, root: Path = ROOT) -> list[str]:
    out_dir = Path(root) / CORE_DIR / subject
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for kind, tpath in TEMPLATES.items():
        dst = out_dir / f"{kind}.yaml"
        with open(Path(root) / tpath) as f:
            cfg = subject_config(yaml.safe_load(f), subject)
        header = (f"# sg2x2 core test, subject {subject} ({SUBJECTS[subject][1]}), cell kind {kind}: generated by "
                  f"scripts/sg2x2_core_configs.py from {tpath}; only data_path / csm_path differ. Do not edit.\n")
        with open(dst, "x") as f:
            f.write(header + yaml.safe_dump(cfg, sort_keys=False))
        written.append(str(dst.relative_to(root)))
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subjects", nargs="*", choices=sorted(SUBJECTS))
    a = ap.parse_args()
    for s in a.subjects or list(SUBJECTS):
        for w in write_subject(s):
            print(w)


if __name__ == "__main__":
    main()
```
Run once: `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/sg2x2_core_configs.py` → 42 lines printed, 42 files under `configs/sg2x2/core/`.

- [ ] **Step 4: Add the subject to the cell table**

In `scripts/sg2x2_cells.py`: below `CELLS`, add

```python
CORE_KIND = {"configs/sg2x2/af16_jyz_joint.yaml": "joint", "configs/sg2x2/af16_jyz_single_T1.yaml": "single_T1",
             "configs/sg2x2/af16_jyz_single_FLAIR.yaml": "single_FLAIR"}


def cell_config(cell: str, subject: str | None = None) -> str:
    """The cell's base config: the development subject's CFL-export config without a subject, else the core-test
    subject's generated copy (scripts/sg2x2_core_configs.py)."""
    c = CELLS[cell]
    if subject is None:
        return c.config
    if subject not in CORE_SUBJECTS:
        raise ValueError(f"unknown core-test subject {subject!r}")
    return f"configs/sg2x2/core/{subject}/{CORE_KIND[c.config]}.yaml"
```
with `from sg2x2_core_subjects import SUBJECTS as CORE_SUBJECTS` beside the other imports; change the signature to `def cell_args(cell: str, npts: int, seed: int, support: str, init: str = "support_file", iters: int = 4000, warmup: int = 500, subject: str | None = None) -> list[str]:` and its `"--base_cfg", c.config` to `"--base_cfg", cell_config(cell, subject)`; in `main()` add `ap.add_argument("--subject", default=None)` and pass `subject=a.subject` to `cell_args`. Add one line to the module docstring: "With --subject the base config is that core-test subject's copy (scripts/sg2x2_core_configs.py)."

- [ ] **Step 5: Add the subject to the launcher**

In `run_sg2x2.sh`: extend the env line of the header comment with `SUBJECT (empty = the development subject's CFL export; else a core-test subject of scripts/sg2x2_core_subjects.py)`; after `WARMUP=${WARMUP:-500}` add

```bash
SUBJECT=${SUBJECT:-}
SUBJ=(); [[ -n "$SUBJECT" ]] && SUBJ=(--subject "$SUBJECT")
```
append `"${SUBJ[@]}"` to the `scripts/sg2x2_cells.py` call (after `--warmup "$WARMUP"`) and to the `scripts/sg2x2_manifest.py` call (after `--iters "$ITERS"`); in the `sg2x2_runs.log` line insert `subject=${SUBJECT:-dev}` after `cell=$CELL`. Check: `bash -n run_sg2x2.sh`.

- [ ] **Step 6: Add the subject to the manifest**

In `scripts/sg2x2_manifest.py`: import `cell_config` beside `CELLS, resolved_switches`, and `from sg2x2_core_subjects import subject_paths`; add

```python
def _subject_ok(cfg: dict, subject: str) -> bool:
    """every data_path / csm_path of the resolved config lies under the subject's export directory"""
    out = subject_paths(subject)["out"] + "/"
    return all(str(blk.get(k, "")).startswith(out) for blk in cfg.get("data", {}).values()
               for k in ("data_path", "csm_path"))
```
In `audit()`, after the `if args is not None:` block's existing updates, add (inside that block)

```python
        if args.get("subject"):
            want["data_subject"] = True
            got["data_subject"] = _subject_ok(resolved_cfg, args["subject"])
```
In `main()`: `ap.add_argument("--subject", default=None)`; add `"subject": a.subject` to the args dict passed to `audit`; compute `config = cell_config(a.cell, a.subject)` and use it for the manifest's `"config": config, "config_md5": md5(ROOT / config)`; add `"subject": a.subject` to the manifest dict. Extend the module docstring by one sentence: "With --subject the resolved data and coil-map paths must lie in that subject's export directory."

- [ ] **Step 7: Run the tests to verify they pass**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= $PY -m pytest tests/test_sg2x2_core_configs.py tests/test_sg2x2_cells.py tests/test_sg2x2_manifest.py tests/test_sg2x2_configs.py -q -p no:cacheprovider` and `bash -n run_sg2x2.sh`.
Expected: all pass (the three config tests count as 5 with the parametrisation; the four cell tests and the three manifest tests are new; every pre-existing test unchanged and passing).

- [ ] **Step 8: Commit**

```bash
git add scripts/sg2x2_core_configs.py configs/sg2x2/core scripts/sg2x2_cells.py run_sg2x2.sh scripts/sg2x2_manifest.py \
  tests/test_sg2x2_core_configs.py tests/test_sg2x2_cells.py tests/test_sg2x2_manifest.py
git commit -m "sg2x2 core test: per-subject configs and an optional subject in the cell table, launcher and manifest"
```

---

### Task 3: The subject-level table

**Files:**
- Create: `scripts/sg2x2_core_table.py`
- Test: `tests/test_sg2x2_core_table.py`

**Interfaces:**
- Consumes: `collect`, `CELLS`, `CONTRASTS`, `METRICS` from `scripts/sg2x2_table.py`; `SUBJECTS`, `core_test_subjects` (Task 1).
- Produces: `subject_means(arc, prefix, subject, seeds) -> dict[(cell, contrast) -> dict]`, `differences(means, contrast, metric) -> dict[str, float]`, `summary(values, higher_better) -> dict`, `table(arc, prefix, seeds) -> str`; CLI `--prefix P --seeds S... [--arc results_arc] [--out PATH]`.

- [ ] **Step 1: Write the failing tests**

```python
"""scripts/sg2x2_core_table.py: per-subject seed means, then paired A-B / C-D / interaction across the TEST subjects
with bootstrap intervals; the development subject never enters the statistics; a missing cell removes that subject
from the rows that need it only.
Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_sg2x2_core_table.py -q
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sg2x2_core_table import differences, subject_means, summary, table  # noqa: E402

DEV = "jiangyizhou_20230705"
TESTS = ["huangyuwei_20230705", "jiangdian_20230705", "sunxinyi_20230705"]


def _score(psnr):
    return {"head_psnr": psnr, "head_ssim": 0.9, "head_hfen": 0.1}


def _write(arc, exp, joint, t1=None, fl=None):
    row = ({"plain": _score(t1), "contrast": _score(fl)} if joint else
           {"plain": _score(t1 if t1 is not None else fl), "contrast": None})
    (arc / f"{exp}_score.json").write_text(json.dumps(row))


def _subject(arc, subj, seed, a, b, c, d, skip=()):
    """a, b, c, d = (T1, FLAIR) head PSNR of the four cells"""
    pre = f"p_{subj}_s{seed}"
    _write(arc, f"{pre}_A", True, *a)
    _write(arc, f"{pre}_C", True, *c)
    for cell, (t1, fl) in (("B", b), ("D", d)):
        if f"{cell}_T1" not in skip:
            _write(arc, f"{pre}_{cell}_T1", False, t1=t1)
        if f"{cell}_FLAIR" not in skip:
            _write(arc, f"{pre}_{cell}_FLAIR", False, fl=fl)


def _fill(arc):
    # A-B on T1 = 0.1, 0.2, 0.3 for the three test subjects; C-D on T1 = 1.0 for all
    for i, s in enumerate(TESTS):
        ab = 0.1 * (i + 1)
        _subject(arc, s, 42, a=(35.0 + ab, 34.0), b=(35.0, 34.05), c=(24.0, 25.0), d=(23.0, 24.0))
    _subject(arc, DEV, 42, a=(99.0, 99.0), b=(0.0, 0.0), c=(0.0, 0.0), d=(0.0, 0.0))   # absurd: must not count


def test_subject_means_average_the_seeds(tmp_path):
    s = TESTS[0]
    _subject(tmp_path, s, 42, a=(35.0, 34.0), b=(35.0, 34.0), c=(24.0, 25.0), d=(23.0, 24.0))
    _subject(tmp_path, s, 43, a=(36.0, 34.0), b=(35.0, 34.0), c=(24.0, 25.0), d=(23.0, 24.0))
    m = subject_means(tmp_path, "p", s, [42, 43])
    assert m[("A", "T1")]["head_psnr"] == 35.5 and m[("A", "T1")]["n_seeds"] == 2
    assert math.isnan(m[("A", "T1")]["head_nmse"])                     # no volumes written


def test_differences_and_summary():
    m = {("A", "T1"): {"head_psnr": 35.2}, ("B", "T1"): {"head_psnr": 35.0},
         ("C", "T1"): {"head_psnr": 24.0}, ("D", "T1"): {"head_psnr": 23.0}}
    d = differences(m, "T1", "head_psnr")
    assert np.isclose(d["A-B"], 0.2) and np.isclose(d["C-D"], 1.0) and np.isclose(d["(A-B)-(C-D)"], -0.8)
    s = summary([0.1, 0.2, 0.3], higher_better=True)
    assert s["n"] == 3 and np.isclose(s["mean"], 0.2) and s["k_better"] == 3
    assert s["ci"][0] <= 0.2 <= s["ci"][1] and summary([0.1, 0.2, 0.3], True) == s     # deterministic
    assert summary([0.01, -0.02], higher_better=False)["k_better"] == 1
    one = summary([0.5], True)
    assert one["n"] == 1 and math.isnan(one["sd"]) and math.isnan(one["ci"][0])


def test_dev_subject_never_enters_the_statistics(tmp_path):
    _fill(tmp_path)
    md = table(tmp_path, "p", [42])
    row = next(l for l in md.splitlines() if l.startswith("| A−B | T1 | PSNR |"))
    assert "| 3 |" in row and "+0.200" in row and "3/3" in row
    assert "jiangyizhou_20230705" in md and "development subject" in md


def test_missing_cell_drops_the_subject_from_that_row_only(tmp_path):
    _fill(tmp_path)
    _subject(tmp_path, "sunxinyi_20230705", 43, a=(1, 1), b=(1, 1), c=(1, 1), d=(1, 1))   # a second seed, harmless
    s = "zhoushuo_20231030"
    _subject(tmp_path, s, 42, a=(35.5, 34.0), b=(35.0, 34.0), c=(24.0, 25.0), d=(23.0, 24.0), skip=("B_T1",))
    md = table(tmp_path, "p", [42])
    t1 = next(l for l in md.splitlines() if l.startswith("| A−B | T1 | PSNR |"))
    fl = next(l for l in md.splitlines() if l.startswith("| A−B | FLAIR | PSNR |"))
    assert "| 3 |" in t1 and "| 4 |" in fl
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= $PY -m pytest tests/test_sg2x2_core_table.py -q -p no:cacheprovider`
Expected: collection error `No module named 'sg2x2_core_table'`.

- [ ] **Step 3: Write the table**

```python
#!/usr/bin/env python3
"""Subject-level 2x2 table of the sg2x2 core test (spec section 5.3).

Runs are named <prefix>_<subject>_s<seed>_<CELL> (the cells of scripts/sg2x2_table.collect). Per subject and contrast
the seeds are averaged first; then, over the TEST subjects only (scripts/sg2x2_core_subjects.py), the paired
differences A−B (shared geometry with the per-channel priors), C−D (shared geometry without explicit priors) and
(A−B)−(C−D) are summarised as N, mean, a 95 % subject-level bootstrap interval (10,000 resamples, generator seed 0),
SD and k/N: the number of subjects in which the difference points the "better" way for that metric (higher PSNR /
SSIM, lower NMSE / HFEN). A subject enters a row only when every cell the row needs exists for it. The development
subject is listed separately and never enters the statistics.

    env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/sg2x2_core_table.py --prefix sg2x2_core_af16 --seeds 42 [43 44]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sg2x2_core_subjects import SUBJECTS, core_test_subjects  # noqa: E402
from sg2x2_table import CELLS, CONTRASTS, METRICS, collect     # noqa: E402

HIGHER_BETTER = {"head_psnr": True, "head_ssim": True, "head_nmse": False, "head_hfen": False}
ROWS = ("A−B", "C−D", "(A−B)−(C−D)")
N_BOOT = 10_000


def _mean(vals: list[float]) -> float:
    v = [x for x in vals if not math.isnan(x)]
    return float(sum(v) / len(v)) if v else float("nan")


def subject_means(arc: Path, prefix: str, subject: str, seeds: list[int]) -> dict:
    """(cell, contrast) -> {metric: mean over the seeds present, "n_seeds": k}"""
    got = collect(Path(arc), f"{prefix}_{subject}", seeds)
    out = {}
    for cell in CELLS:
        for c in CONTRASTS:
            rows = [got[(cell, c, s)] for s in seeds if (cell, c, s) in got]
            if rows:
                out[(cell, c)] = {k: _mean([r[k] for r in rows]) for k, _, _ in METRICS}
                out[(cell, c)]["n_seeds"] = len(rows)
    return out


def differences(means: dict, contrast: str, metric: str) -> dict:
    v = {cell: means.get((cell, contrast), {}).get(metric, float("nan")) for cell in CELLS}
    ab, cd = v["A"] - v["B"], v["C"] - v["D"]
    return {"A−B": ab, "C−D": cd, "(A−B)−(C−D)": ab - cd,
            "A-B": ab, "C-D": cd, "(A-B)-(C-D)": ab - cd}


def summary(values: list[float], higher_better: bool) -> dict:
    x = np.array([v for v in values if not math.isnan(v)], dtype=float)
    n = int(x.size)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "ci": (float("nan"), float("nan")), "k_better": 0}
    if n >= 2:
        boots = np.random.default_rng(0).choice(x, size=(N_BOOT, n), replace=True).mean(axis=1)
        ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        sd = float(x.std(ddof=1))
    else:
        ci, sd = (float("nan"), float("nan")), float("nan")
    k = int((x > 0).sum()) if higher_better else int((x < 0).sum())
    return {"n": n, "mean": float(x.mean()), "sd": sd, "ci": ci, "k_better": k}


def _f(v: float, fmt: str, signed: bool = True) -> str:
    if math.isnan(v):
        return "n/a"
    return ("+" if signed and v >= 0 else "") + fmt.format(v)


def table(arc: Path, prefix: str, seeds: list[int]) -> str:
    tests = core_test_subjects()
    dev = [s for s, (_, r) in SUBJECTS.items() if r == "dev"]
    means = {s: subject_means(arc, prefix, s, seeds) for s in tests + dev}
    have = [s for s in tests if means[s]]
    lines = [f"CORE TEST: {len(have)} of {len(tests)} test subjects with results, seeds {seeds}. Head-only metrics "
             f"(|GT| > 0.05 P99.9). Per subject the seeds are averaged first; intervals are 95 % subject-level bootstrap "
             f"({N_BOOT} resamples, seed 0); k/N = subjects in which the shared-geometry cell is better (interaction: the "
             f"difference points the better way). The development subject ({', '.join(dev)}) is listed separately and "
             f"never enters the statistics.", "",
             "| row | contrast | metric | N | mean | 95 % CI | SD | k/N |", "|---|---|---|---|---|---|---|---|"]
    for row in ROWS:
        for c in CONTRASTS:
            for key, name, fmt in METRICS:
                vals = [differences(means[s], c, key)[row] for s in tests]
                sm = summary(vals, HIGHER_BETTER[key])
                lines.append(f"| {row} | {c} | {name} | {sm['n']} | {_f(sm['mean'], fmt)} | "
                             f"[{_f(sm['ci'][0], fmt)}, {_f(sm['ci'][1], fmt)}] | {_f(sm['sd'], fmt, False)} | "
                             f"{sm['k_better']}/{sm['n']} |")
    lines += ["", "Cell means over the test subjects (mean ± SD, N):", "",
              "| cell | contrast | " + " | ".join(n for _, n, _ in METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for cell in CELLS:
        for c in CONTRASTS:
            cells = []
            for key, _, fmt in METRICS:
                v = np.array([means[s][(cell, c)][key] for s in tests if (cell, c) in means[s]
                              and not math.isnan(means[s][(cell, c)][key])])
                cells.append("n/a" if v.size == 0 else
                             f"{fmt.format(v.mean())} ± {fmt.format(v.std(ddof=1)) if v.size > 1 else 'n/a'} ({v.size})")
            lines.append(f"| {cell} | {c} | " + " | ".join(cells) + " |")
    lines += ["", "Per subject, head PSNR (T1 / FLAIR) of A, B, C, D and A−B, C−D:", "",
              "| subject | role | A | B | C | D | A−B | C−D |", "|---|---|---|---|---|---|---|---|"]
    for s in tests + dev:
        if not means[s]:
            lines.append(f"| {s} | {SUBJECTS[s][1]} | no runs |  |  |  |  |  |")
            continue
        def pair(cell):
            return " / ".join(_f(means[s].get((cell, c), {}).get("head_psnr", float("nan")), "{:.3f}", False)
                              for c in CONTRASTS)
        dd = {c: differences(means[s], c, "head_psnr") for c in CONTRASTS}
        lines.append(f"| {s} | {SUBJECTS[s][1]} | {pair('A')} | {pair('B')} | {pair('C')} | {pair('D')} | "
                     f"{' / '.join(_f(dd[c]['A−B'], '{:.3f}') for c in CONTRASTS)} | "
                     f"{' / '.join(_f(dd[c]['C−D'], '{:.3f}') for c in CONTRASTS)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--arc", default="results_arc")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    md = table(ROOT / a.arc, a.prefix, a.seeds)
    out = Path(a.out) if a.out else ROOT / a.arc / f"{a.prefix}_core_table.md"
    with open(out, "x") as f:
        f.write(md)
    print(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```
Note on `differences`: it returns both the Unicode-minus keys used in the printed rows and ASCII aliases (`A-B`, `C-D`, `(A-B)-(C-D)`) so tests and callers can use either; keep both.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= $PY -m pytest tests/test_sg2x2_core_table.py tests/test_sg2x2_table.py -q -p no:cacheprovider`
Expected: `5 passed` for the new file, the existing table tests unchanged and passing, no warnings.

- [ ] **Step 5: Commit**

```bash
git add scripts/sg2x2_core_table.py tests/test_sg2x2_core_table.py
git commit -m "sg2x2 core test: subject-level table (seed means, paired differences, bootstrap intervals, development subject apart)"
```

---

### Task 4: Verify the export, write the queue files and the docs (no GPU)

**Files:**
- Create: `scripts/sg2x2_core_verify.py` (read-only verification; CPU)
- Create: `scripts/sg2x2_core_smoke.txt`, `scripts/sg2x2_queue_main.txt`
- Create: `docs/superpowers/records/2026-10-10-sg2x2-core-test-setup.md`
- Modify: `STATUS.md` (top section only: a new item in §1 and the launch steps in §3)

**Interfaces:**
- Consumes: Task 1's export outputs under `/data2/congcong/data/sg2x2_core_h5/<subject>/` (the controller has run the export), Task 2's configs, `paired_gs/data/dataset.py` `MRIDataset`, `paired_gs/gaussian/init_cloud.py` `head_support`, `union_magnitude`, `load_init_support`.

- [ ] **Step 1: Verification script and run**

Write `scripts/sg2x2_core_verify.py`:

```python
#!/usr/bin/env python3
"""Read-only verification of the sg2x2 core-test export (plan 2026-10-10, Task 4). Per subject: the export's own
checks re-read from export.json; the four output files re-hashed against the recorded md5; the joint config's data
loaded through the trainer's MRIDataset on the CPU (ground truth shape, finiteness, mask acceleration); the zero-filled
union head support as the trainer computes it, and two overlaps with the frozen init support: `in_own` = share of the
support's voxels inside the subject's own support (what the trainer prints as in_own_zf_support for the joint cells)
and `cover` = share of the subject's own support covered by the init support.
Writes results_arc/sg2x2_core_verify_<YYYYMMDD>.json (mode "x").

    env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= nice -n 10 $PY scripts/sg2x2_core_verify.py [SUBJECT ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "paired_gs"))
from sg2x2_core_export import md5  # noqa: E402
from sg2x2_core_subjects import SUBJECTS, subject_paths  # noqa: E402
from data.transforms import ifft3c  # noqa: E402
from gaussian.init_cloud import head_support, load_init_support, union_magnitude  # noqa: E402
from trainers.joint_trainer import _load_dataset  # noqa: E402

SUPPORT = ROOT / "configs/sg2x2/init_support_af16_jyz.npz"


def check_export(subject: str) -> dict:
    rep = json.load(open(subject_paths(subject)["report"]))
    out = {"problems": []}
    for c in ("T1", "FLAIR"):
        i = rep["contrasts"][c]
        if i["outside_rel_max"] > 1e-6:
            out["problems"].append(f"{c}: data outside the cropped rows")
        if any(abs(a - b) > 1 for a, b in zip(i["dc_pe_spe"], i["centre_pe_spe"])):
            out["problems"].append(f"{c}: DC {i['dc_pe_spe']} off centre {i['centre_pe_spe']}")
        if c == "T1" and i["acquired_frac"] < 0.9999:
            out["problems"].append(f"T1 acquires {i['acquired_frac']:.4f} of the plane")
        if i["espirit"]["sos_frac_within_1e-3"] < 0.999:
            out["problems"].append(f"{c}: ESPIRiT SoS off 1")
        for key, rec in (("ksp", "ksp_md5"), ("csm", "csm_md5")):
            if md5(i[key]) != i[rec]:
                out["problems"].append(f"{c}: {key} md5 differs from export.json")
        out[c] = {"lo": i["lo"], "dc": i["dc_pe_spe"], "acquired_frac": i["acquired_frac"],
                  "ecalib_s": i["espirit"]["seconds"], "sos_within_1e-3": i["espirit"]["sos_frac_within_1e-3"]}
    return out


def load_and_cover(subject: str, support: torch.Tensor) -> dict:
    cfg = yaml.safe_load(open(ROOT / f"configs/sg2x2/core/{subject}/joint.yaml"))
    zf, out = {}, {}
    for slot, blk in cfg["data"].items():
        ds = _load_dataset(blk)
        gt = ds.get_data()["ground_truth"]
        zf[slot] = torch.sum(torch.conj(ds.sensitivity_maps) * ifft3c(ds.kspace_undersampled_complex), dim=0)
        out[blk["name"]] = {"gt_shape": list(gt.shape), "gt_finite": bool(torch.isfinite(gt.abs()).all()),
                            "mask_acc": float(ds.mask.numel() / ds.mask.sum())}
    own = head_support(union_magnitude(zf["plain"], zf["contrast"]), 0.05)
    both = float((own & support).sum())
    out["in_own"] = both / float(support.sum())
    out["cover"] = both / float(own.sum())
    out["own_frac"] = float(own.float().mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subjects", nargs="*", choices=sorted(SUBJECTS))
    a = ap.parse_args()
    support = load_init_support(str(SUPPORT), (256, 256, 176), "cpu")
    res, ok = {}, True
    print("| subject | role | export checks | GT shape T1 / FLAIR | mask acc | in_own | cover | own_frac |")
    for s in a.subjects or list(SUBJECTS):
        e = check_export(s)
        lc = load_and_cover(s, support)
        res[s] = {"role": SUBJECTS[s][1], "export": e, "load": lc}
        good = (not e["problems"] and all(lc[c]["gt_shape"] == [256, 256, 176] and lc[c]["gt_finite"] for c in ("T1", "FLAIR")))
        ok &= good
        print(f"| {s} | {SUBJECTS[s][1]} | {'ok' if not e['problems'] else '; '.join(e['problems'])} | "
              f"{lc['T1']['gt_shape']} / {lc['FLAIR']['gt_shape']} | {lc['T1']['mask_acc']:.2f} | {lc['in_own']:.4f} | "
              f"{lc['cover']:.4f} | {lc['own_frac']:.4f} |", flush=True)
    out = ROOT / "results_arc" / f"sg2x2_core_verify_{time.strftime('%Y%m%d')}.json"
    with open(out, "x") as f:
        json.dump(res, f, indent=2)
    print(f"{'ALL OK' if ok else 'PROBLEMS FOUND'} -> {out}")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
```
Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= nice -n 10 $PY scripts/sg2x2_core_verify.py`
Expected: 14 rows, `mask acc` 15.19, `ALL OK`; `in_own` near 1 for jiangyizhou and lower for the others (the inventory's coverage figures put `cover` at median about 0.87 and the bright-voxel coverage higher). If any subject has problems, stop and report it; its queue lines must not be written.

- [ ] **Step 2: Queue files**

`scripts/sg2x2_core_smoke.txt` (two-iteration end-to-end runs through the launcher and the manifest; 5 lines):
```
# sg2x2 core test, smoke: two iterations per cell on one test subject and the development subject's h5 export
SUBJECT=huangyuwei_20230705 WARMUP=0 sg2x2_core_smoke_huangyuwei_20230705_s42_A A 2 42
SUBJECT=huangyuwei_20230705 WARMUP=0 sg2x2_core_smoke_huangyuwei_20230705_s42_C C 2 42
SUBJECT=huangyuwei_20230705 WARMUP=0 sg2x2_core_smoke_huangyuwei_20230705_s42_B_FLAIR B_FLAIR 2 42
SUBJECT=huangyuwei_20230705 WARMUP=0 sg2x2_core_smoke_huangyuwei_20230705_s42_D_T1 D_T1 2 42
SUBJECT=jiangyizhou_20230705 WARMUP=0 sg2x2_core_smoke_jiangyizhou_20230705_s42_A A 2 42
```
`scripts/sg2x2_queue_main.txt`: generate it (do not type the 84 lines by hand) with this snippet, run from the worktree root as `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY <snippet file>`:

```python
import sys
sys.path.insert(0, "scripts")
from sg2x2_core_subjects import SUBJECTS

CELL_ORDER = ["A", "C", "B_T1", "B_FLAIR", "D_T1", "D_FLAIR"]
lines = ["# sg2x2 core test: seed 42 of the six cells for the 14 subjects (development subject first), then the",
         "# development subject's CFL-path seeds 43 / 44 and the 1.8M supplement (scripts/sg2x2_dev_seeds.txt).",
         "# Lines: [KEY=VAL ...] EXP CELL ITERS SEED"]
for s in SUBJECTS:
    for c in CELL_ORDER:
        lines.append(f"SUBJECT={s} sg2x2_core_af16_{s}_s42_{c} {c} 4000 42")
lines.append("# development subject, CFL export: seeds 43 / 44 and the 1.8M capacity supplement")
lines += [l.rstrip("\n") for l in open("scripts/sg2x2_dev_seeds.txt") if l.strip() and not l.startswith("#")]
with open("scripts/sg2x2_queue_main.txt", "x") as f:
    f.write("\n".join(lines) + "\n")
```
Check: `grep -vc '^#' scripts/sg2x2_queue_main.txt` → 108; `grep -v '^#' scripts/sg2x2_queue_main.txt | awk '{for(i=1;i<=NF;i++) if($i ~ /^sg2x2_/) print $i}' | sort | uniq -d` → empty; `bash -n scripts/sg2x2_queue.sh` ok.

- [ ] **Step 3: Record and STATUS**

`docs/superpowers/records/2026-10-10-sg2x2-core-test-setup.md` (English): the decisions (copy the Global Constraints' "User decisions" line), what was built (Tasks 1-3, files), the export run (start/finish time from `/data2/congcong/data/sg2x2_core_h5/export_run_20261010.log`, per-subject `lo`, DC, acquired fractions, ESPIRiT seconds and SoS — from the verification JSON), the support-coverage column, disk use (`du -sh /data2/congcong/data/sg2x2_core_h5` and of the `bart/` subfolders, which the user may delete by hand later), and the launch procedure of the controller (below). STATUS top section: one item in §1 ("core-test setup done and verified", the verification command and its summary line) and in §3 the launch steps.

- [ ] **Step 4: Commit**

```bash
git add scripts/sg2x2_core_verify.py scripts/sg2x2_core_smoke.txt scripts/sg2x2_queue_main.txt \
  docs/superpowers/records/2026-10-10-sg2x2-core-test-setup.md STATUS.md
git commit -m "sg2x2 core test: export verified for the 14 subjects; smoke and main queue files; setup record"
```

**Controller steps after Task 4 (GPU; not implementer steps):**
1. Stop the dev-seeds queue runner (only the runner's own bash process, `pgrep -f 'scripts/sg2x2_queue.sh scripts/sg2x2_dev_seeds.txt'`; launched trainers keep running).
2. `setsid nohup env ALLOWED="0 1 2 3 4 5 6 7" MAXRUN=4 bash scripts/sg2x2_queue.sh scripts/sg2x2_core_smoke.txt > results_arc/sg2x2_queue_core_smoke.log 2>&1 < /dev/null &`; wait for `queue drained`; all five smoke manifests must audit ok (with `data_subject` true), every `[init_cloud]` line `n=3600000 support_frac=0.4273`, no NaN.
3. `setsid nohup env ALLOWED="0 1 2 3 4 5 6 7" MAXRUN=4 bash scripts/sg2x2_queue.sh scripts/sg2x2_queue_main.txt > results_arc/sg2x2_queue_main.log 2>&1 < /dev/null &`, with the stall watcher re-armed on the new queue log.
4. When subjects complete: `scripts/sg2x2_core_table.py --prefix sg2x2_core_af16 --seeds 42` (partial tables are allowed; N per row says how many subjects entered).

---

## Self-review notes

- Spec coverage: §5.1.2 (≥ 12 other subjects with both sequences, registration, full k-space, usable coil maps; development / test split fixed first) → Task 1 subjects table (13 test + 1 dev, 790 excluded) and Task 4 verification; §4 (identical initialisation from a development-derived support, own-ZF densities, same mask and seed) → the pilot support and mask via the generated configs; §5.3 (subject-level paired statistics, CI, direction counts, two contrasts separately) → Task 3; §3 (audited resolved configuration) → the existing manifest plus the subject identity check.
- Not covered by design: seeds 43 / 44 of the core test (append to the main queue later), other AFs, the equal-budget own-λ search, a union support from several development subjects.
- Type consistency: `subject_paths` keys (`ksp`, `csm`, `out`, `report`, `src`, `role`, `bart`) are used identically by Tasks 1, 2 and 4; `cell_config` (Task 2) is used by the manifest; `collect(arc, f"{prefix}_{subject}", seeds)` matches `scripts/sg2x2_table.collect`'s name builder `<prefix>_s<seed>_<cell>`.
