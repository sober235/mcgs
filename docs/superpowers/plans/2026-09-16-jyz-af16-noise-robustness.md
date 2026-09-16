# jiangyizhou AF16 measurement-noise robustness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct the jiangyizhou T1+FLAIR pair at AF16 (house Poisson-disc mask) with every arm of the series on measurements that carry 2 sigma0 of added complex Gaussian noise, score everything against the clean ground truth, and report MC-GS's standing against the recorded zero-noise topic.

**Architecture:** A new results topic `T1FLAIR_AF16_noise2x_v3_jiangyizhou` holds clean reference k-space (symlinks, ground truth only) and new noisy measurement files `data/meas_<c>.h5`. Every loader is split so the ground truth and scale come from the clean file and the fit comes from the measurement file: the topic's `t1t2_common.Pair` and `recon_classical.py`, the trainer (`MRIDataset.measurement_path`) and MC-INR (`--ksp_T1/--ksp_FLAIR`). A single-topic clone of the diff-mask scheduler runs the 57 jobs on cards 1, 2 and 5, and a finalize script registers, scores, tabulates and draws.

**Tech Stack:** Python 3.11 (GSTMI env: torch 2.7.1+cu118, h5py, numpy, scipy, matplotlib), GR_pt1.10 env for MC-INR (tinycudann), bash, git, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-jyz-af16-noise-robustness-design.md` (commits fbff825, de311fb).

## Global Constraints

- Worktree `/data0/congcong/code/GS/mcgs_noise_wt`, branch `jyz-af16-noise`. Topic `/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou` (below: `N`). Zero-noise topic `/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou` (below: `H`, read-only).
- Python: `PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python`, always with `PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID`. MC-INR: `PY_TCNN=/home/congcongliu/anaconda3/envs/GR_pt1.10/bin/python`.
- The tool shell is zsh: it does not word-split `$VAR`, and unquoted globs such as `--include=*.py` fail. Anything with more than one variable argument, and anything that runs longer than a few minutes, goes into a bash script file launched with `setsid nohup ... > log 2>&1 < /dev/null &`.
- GPUs: only cards **1, 2 and 5** (card 5 also carries another user's job). At most one job of this user per card. Check `nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv` before any GPU step; the scheduler also takes `Cardiac_FreeRunning_MCGS/work/.gpulock.<card>`.
- Noise: per-sample complex std `sigma = 2 * sigma0_k`; `SeedSequence(20260916)`, children spawned in the order T1, FLAIR; noise only on the acquired support; ground truth and scale from the clean file.
- Mask: `H/data/mask_af16_3d.h5`, md5 `d08179b8e74f35846455983dbe299294`, 2966 PE points.
- MC-GS / GS-single recipe block byte-identical to `run_diffmask_mcgs.sh` (checked with `diff`). L1-ESPIRiT is not run. No retuning of MC-GS.
- Error maps: jet, fixed `[0, 0.1]`, `|x - GT| / max(GT)`.
- Figures are served from `/home/congcongliu/figs/t1flair_v3_af16_noise2x/` at `http://localhost:8765/t1flair_v3_af16_noise2x/`; every figure is looked at before it is reported, and reported with its URL and its absolute path.
- Commits: author is the repo identity `Congcong Liu <liuc23539@gmail.com>` (already in `.git/config`), plain messages, **no AI trailers**, stage named files only. Never merge into `main`. The MC-INR repository is never pushed.
- Do not modify any other worktree or topic (`mcgs_gauss_wt`, `mcgs_peds_wt`, `mcgs_diffmask_wt`, `mcgs_af16_wt`, the AF12/AF16/gauss topics, ...). Reading is fine.
- Numbers from smoke runs are NOT_EVIDENCE and are labelled so wherever they are written.

---

### Task 1: Worktree prerequisites and the baseline test suite

**Files:**
- Create (untracked, gitignored): `results_arc/reg_aligned_4k_volumes.npz` (copy), `results_arc/tests_baseline.txt`
- Create: `/tmp/claude-1002/-home-congcongliu--claude/9236a300-298a-44d0-bb49-0901a6adc910/scratchpad/run_tests.sh`

**Interfaces:**
- Consumes: nothing.
- Produces: the recorded baseline test count; the CUDA voxelizer extension compiled once in this worktree (`paired_gs/gaussian/cuda_ext/build/`), so later parallel GS runs do not race to compile it.

- [ ] **Step 1: Copy the fixture that `tests/test_score_edge.py` needs**

```bash
cp /data0/congcong/code/GS/Paired_ARC_GS_general_framework/results_arc/reg_aligned_4k_volumes.npz \
   /data0/congcong/code/GS/mcgs_noise_wt/results_arc/
ls -la /data0/congcong/code/GS/mcgs_noise_wt/results_arc/reg_aligned_4k_volumes.npz   # 312100147 bytes
```

- [ ] **Step 2: Write the test runner**

`/tmp/claude-1002/-home-congcongliu--claude/9236a300-298a-44d0-bb49-0901a6adc910/scratchpad/run_tests.sh`:

```bash
#!/bin/bash
# run_tests.sh GPU OUTFILE -- the full suite of the noise worktree on one card
GPU=$1; OUT=$2
cd /data0/congcong/code/GS/mcgs_noise_wt || exit 1
env CUDA_VISIBLE_DEVICES="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1 PYTHONPATH= \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests -q --ignore=tests/smoke_test.py > "$OUT" 2>&1
echo "TESTS_RC=$?" >> "$OUT"
```

- [ ] **Step 3: Pick a card and run the suite detached**

Run `nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv` and pick one of 1, 2, 5 that carries no process of this user. Then:

```bash
setsid nohup bash /tmp/claude-1002/-home-congcongliu--claude/9236a300-298a-44d0-bb49-0901a6adc910/scratchpad/run_tests.sh <CARD> \
  /data0/congcong/code/GS/mcgs_noise_wt/results_arc/tests_baseline.txt > /dev/null 2>&1 < /dev/null &
```

Wait for `TESTS_RC=` to appear in the file (about 11 minutes).
Expected tail: `206 passed, 1 warning` and `TESTS_RC=0` (the diff-mask tip bbba53b recorded exactly this). If the count differs, stop and report the difference before changing any code.

- [ ] **Step 4: Confirm the extension was built in this worktree**

```bash
ls /data0/congcong/code/GS/mcgs_noise_wt/paired_gs/gaussian/cuda_ext/build/
```
Expected: `tile_voxelize_cuda_<hash>_<hash>.so` (a flat file next to `build.ninja`; on 2026-09-16 `tile_voxelize_cuda_0141c525_7aedd3b0.so`). No commit (both files are gitignored).

---

### Task 2: Trainer — optional `measurement_path`

**Files:**
- Create: `tests/test_measurement_path.py`
- Modify: `paired_gs/data/dataset.py` (constructor signature, `_load_h5_kspace`, `_load_mat_kspace`, `_apply_undersampling`, new `_load_measurement`)
- Modify: `paired_gs/trainers/joint_trainer.py` (`_load_dataset`)
- Modify: `arc/cycle.py` (`build_cfg` path resolution)

**Interfaces:**
- Consumes: nothing.
- Produces: config key `data.<plain|contrast>.measurement_path` (absolute h5/mat path). When set, `MRIDataset.kspace_undersampled_complex` (and everything derived from it: `kspace_undersampled`, `zero_filled_complex`, `zero_filled`) come from that file divided by the clean `normalization_scale`; `ground_truth`, `kspace_full`, `kspace_multicoil`, `normalization_scale` are unchanged. The trainer log prints `[Measurement] fitting to <path> (ground truth and scale from <data_path>)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_measurement_path.py`:

```python
"""data.<channel>.measurement_path (noise-robustness study, spec 2026-09-16 section 4): the reconstruction fits the
k-space in that file, while the ground truth, the intensity scale and the reference k-space stay on data_path.

Run: PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_measurement_path.py -v
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
from data.dataset import MRIDataset  # noqa: E402
from data.transforms import ifft3c  # noqa: E402

NC, X, Y, Z = 3, 4, 8, 6


def _write(path, key, arr):
    with h5py.File(path, "w") as f:
        f.create_dataset(key, data=arr)
    return str(path)


@pytest.fixture()
def files(tmp_path):
    rng = np.random.default_rng(0)
    cplx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    k = cplx(NC, X, Y, Z).astype(np.complex64)
    s = cplx(NC, X, Y, Z)
    s = (s / np.sqrt((np.abs(s) ** 2).sum(0, keepdims=True))).astype(np.complex64)
    plane = rng.random((Y, Z)) < 0.4
    plane[Y // 2 - 1:Y // 2 + 1, Z // 2 - 1:Z // 2 + 1] = True
    plane[0, 0] = False                                   # at least one PE point unsampled
    m = np.ascontiguousarray(np.broadcast_to(plane, (X, Y, Z))).astype(np.uint8)
    noisy = (k + 0.05 * cplx(NC, X, Y, Z)).astype(np.complex64)
    return dict(data=_write(tmp_path / "ksp.h5", "kspace", k), csm=_write(tmp_path / "csm.h5", "map", s),
                mask=_write(tmp_path / "mask.h5", "mask", m), noisy=_write(tmp_path / "meas.h5", "kspace", noisy),
                bad=_write(tmp_path / "bad.h5", "kspace", np.ascontiguousarray(k[..., :-1])), noisy_arr=noisy)


def _ds(f, **kw):
    return MRIDataset(data_path=f["data"], csm_path=f["csm"], mask_path=f["mask"], mask_type="external",
                      acceleration_factor=2, readout_axis=0, phase_axes=(1, 2), use_acs=False, acs_lines=0,
                      normalize_kspace=True, normalization_percentile=99.9, device="cpu", presampled=False,
                      skip_readout_check=False, **kw)


FIELDS = ("ground_truth", "kspace_full", "kspace_undersampled_complex", "kspace_undersampled",
          "zero_filled_complex", "zero_filled", "mask", "sensitivity_maps")


def test_explicit_none_is_the_default_path(files):
    a, b = _ds(files).get_data(), _ds(files, measurement_path=None).get_data()
    for key in FIELDS:
        assert torch.equal(a[key], b[key]), key


def test_measurement_equal_to_data_path_is_bit_identical(files):
    a, b = _ds(files), _ds(files, measurement_path=files["data"])
    da, db = a.get_data(), b.get_data()
    for key in FIELDS:
        assert torch.equal(da[key], db[key]), key
    assert da["normalization_scale"] == db["normalization_scale"]
    assert torch.equal(a.kspace_multicoil, b.kspace_multicoil)


def test_noisy_measurement_changes_only_the_measured_data(files):
    a, b = _ds(files), _ds(files, measurement_path=files["noisy"])
    da, db = a.get_data(), b.get_data()
    for key in ("ground_truth", "kspace_full", "mask", "sensitivity_maps"):
        assert torch.equal(da[key], db[key]), key
    assert da["normalization_scale"] == db["normalization_scale"]
    assert torch.equal(a.kspace_multicoil, b.kspace_multicoil)          # the clean reference is untouched
    want = torch.from_numpy(files["noisy_arr"]).to(torch.complex64) / db["normalization_scale"] * db["mask"].unsqueeze(0)
    assert torch.allclose(db["kspace_undersampled_complex"], want, rtol=0, atol=1e-6)
    assert not torch.allclose(db["kspace_undersampled_complex"], da["kspace_undersampled_complex"])
    zf = torch.sum(torch.conj(db["sensitivity_maps"]) * ifft3c(want), dim=0)
    assert torch.allclose(db["zero_filled_complex"], zf, rtol=0, atol=1e-5)


def test_measurement_with_another_shape_raises(files):
    with pytest.raises(ValueError, match="measurement"):
        _ds(files, measurement_path=files["bad"])


def test_joint_trainer_forwards_measurement_path(monkeypatch):
    import trainers.joint_trainer as jt
    seen = {}
    monkeypatch.setattr(jt, "MRIDataset", lambda **kw: seen.update(kw) or "ds")
    assert jt._load_dataset({"data_path": "a.h5", "measurement_path": "b.h5"}) == "ds"
    assert seen["measurement_path"] == "b.h5"
    jt._load_dataset({"data_path": "a.h5"})
    assert seen["measurement_path"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /data0/congcong/code/GS/mcgs_noise_wt && PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_measurement_path.py -v
```
Expected: `test_explicit_none_is_the_default_path`, `test_measurement_equal_to_data_path_is_bit_identical`, `test_noisy_measurement_changes_only_the_measured_data` and `test_measurement_with_another_shape_raises` FAIL with `TypeError: ... unexpected keyword argument 'measurement_path'`; `test_joint_trainer_forwards_measurement_path` FAILS with `KeyError: 'measurement_path'`.

- [ ] **Step 3: Implement in `paired_gs/data/dataset.py`**

(a) Constructor: add the keyword after `mask_split_out_dir`:

```python
        mask_split_out_dir: Optional[str] = None, # 若提供则 dump split_indices.npz
        measurement_path: Optional[str] = None,   # noise study 2026-09-16: fit THIS k-space; GT/scale stay on data_path
    ):
```

and store it before `self._load_data()` (next to the other v2 fields):

```python
        self.mask_split: Dict[str, torch.Tensor] = {}
        self.measurement_path = measurement_path if measurement_path not in (None, "", "None") else None
        self._load_data()
```

(b) Let the two readers take an explicit path (the default keeps every existing call unchanged):

```python
    def _load_h5_kspace(self, path: Optional[str] = None) -> np.ndarray:
        path = path or self.data_path
        if h5py is None:
            raise ImportError("Reading .h5 k-space files requires h5py to be installed.")
        with h5py.File(path, "r") as handle:
```

(the rest of the method body is unchanged), and in `_load_mat_kspace`:

```python
    def _load_mat_kspace(self, path: Optional[str] = None) -> np.ndarray:
        path = path or self.data_path
        try:
            variables = scipy.io.whosmat(path)
        except NotImplementedError:
            print("MATLAB v7.3 file detected, falling back to h5py reader.")
            return self._load_h5_kspace(path)
        if len(variables) == 0:
            raise ValueError(f"No variables found in MATLAB file: {path}")
```

and further down in the same method `mat = scipy.io.loadmat(path, variable_names=[key])`.

(c) Replace the first line of `_apply_undersampling` and add the loader right above it:

```python
    def _load_measurement(self) -> torch.Tensor:
        """The k-space the reconstruction fits when it is not the reference that defines the ground truth
        (data.<channel>.measurement_path; noise-robustness study, spec 2026-09-16 section 4). Read and laid out like
        data_path, rolled like it, and divided by the SAME normalization_scale (the clean ground truth's), so a
        measurement file identical to data_path reproduces the default path bit for bit."""
        path = self.measurement_path
        print(f"[Measurement] fitting to {path} (ground truth and scale from {self.data_path})")
        ext = os.path.splitext(path)[1].lower()
        arr = self._load_mat_kspace(path) if ext == ".mat" else self._load_h5_kspace(path)
        meas = torch.from_numpy(self._normalize_kspace_layout(arr)).to(torch.complex64)
        if tuple(meas.shape) != tuple(self.kspace_multicoil.shape):
            raise ValueError(f"measurement k-space {tuple(meas.shape)} from {path} does not match the reference "
                             f"{tuple(self.kspace_multicoil.shape)} from {self.data_path}")
        if self.center_kspace_dc and any(self._dc_roll):
            meas = torch.roll(meas, shifts=self._dc_roll, dims=(1, 2, 3)).contiguous()
        return meas / self.normalization_scale

    def _apply_undersampling(self):
        measured = self._load_measurement() if self.measurement_path else self.kspace_multicoil
        kspace_undersampled_complex = measured * self.mask.unsqueeze(0)
```

(the rest of `_apply_undersampling` is unchanged).

- [ ] **Step 4: Implement the plumbing**

`paired_gs/trainers/joint_trainer.py`, in `_load_dataset`, add one line after `presampled=...`:

```python
        presampled=data_cfg.get("presampled", True),
        measurement_path=data_cfg.get("measurement_path"),
    )
```

`arc/cycle.py`, in `build_cfg`:

```python
        for k in ("data_path", "csm_path", "mask_path", "measurement_path"):
```

- [ ] **Step 5: Run the new tests to verify they pass**

```bash
cd /data0/congcong/code/GS/mcgs_noise_wt && PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_measurement_path.py -v
```
Expected: 5 passed.

- [ ] **Step 6: Run the full suite (Task 1's runner) and compare with the baseline**

```bash
setsid nohup bash /tmp/claude-1002/-home-congcongliu--claude/9236a300-298a-44d0-bb49-0901a6adc910/scratchpad/run_tests.sh <CARD> \
  /data0/congcong/code/GS/mcgs_noise_wt/results_arc/tests_after_task2.txt > /dev/null 2>&1 < /dev/null &
```
Expected: baseline count + 5 passed (211 if the baseline was 206), `TESTS_RC=0`.

- [ ] **Step 7: Commit**

```bash
cd /data0/congcong/code/GS/mcgs_noise_wt && git add tests/test_measurement_path.py paired_gs/data/dataset.py \
  paired_gs/trainers/joint_trainer.py arc/cycle.py && git commit -q -m "trainer: optional data.<channel>.measurement_path - fit that k-space while the ground truth, the scale and the reference k-space stay on data_path (noise study); unset is unchanged"
```

---

### Task 3: `cycle.py --save_render` (the `arc/cycle.py` part of 02e53dd)

**Files:**
- Modify: `arc/cycle.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `--save_render` flag; with `--save_volumes` the npz also holds `recT1_render`, `recFL_render`, `recT1_render_complex`, `recFL_render_complex` (the render before data consistency, same units as `recT1_complex`).

- [ ] **Step 1: Apply the patch** (02e53dd also touched `autoresearch_peds.tsv`, which does not exist on this branch, so only the `arc/cycle.py` hunk is taken)

```bash
cd /data0/congcong/code/GS/mcgs_noise_wt && git -C /data0/congcong/code/GS/Paired_ARC_GS_general_framework \
  show 02e53dd -- arc/cycle.py | git apply && git diff --stat
```
Expected: `arc/cycle.py | 15 ++++++++++++---`.

- [ ] **Step 2: Verify the flag parses and the npz keys are written**

```bash
cd /data0/congcong/code/GS/mcgs_noise_wt && grep -n "save_render" arc/cycle.py
```
Expected: the `add_argument("--save_render"` line and the `if args.save_render:` block. The keys are checked for real in Task 7's smoke run.

- [ ] **Step 3: Commit**

```bash
cd /data0/congcong/code/GS/mcgs_noise_wt && git add arc/cycle.py && git commit -q -m "cycle.py: --save_render (the arc/cycle.py part of 02e53dd from 16yo-mtp): also store the render before data consistency"
```

---

### Task 4: Topic scaffold and the noisy measurements

**Files:**
- Create: `N/` with `data/ logs/ recon/ figures/ scripts/`
- Create (symlinks): `N/data/ksp_{T1,FLAIR}.h5`, `N/data/csm_espirit_{T1,FLAIR}.h5` (into the AF12 topic), `N/data/mask_af16_3d.h5`, `N/data/flair_support_2d.h5` (into `H`)
- Create: `N/logs/prep_04_export.json` (from `H`, paths updated)
- Create: `N/scripts/make_noisy_kspace.py`
- Create (by the generator): `N/data/meas_{T1,FLAIR}.h5`, `N/recon/fullnoisy_{T1,FLAIR}_complex.npy`, `N/logs/make_noisy_kspace.json`

**Interfaces:**
- Consumes: nothing.
- Produces: `N/data/meas_<c>.h5` (key `kspace`, complex64, shape (26, 256, 256, 176)); `N/logs/make_noisy_kspace.json` with, per contrast: `scale`, `house_scale`, `f_acq`, `support_lines`, `sigma0_img`, `sigma0_k`, `sigma`, `sigma_raw`, `air_noise_noisy`, `air_noise_expected`, `checks` (dict of name -> {value, limit, ok}), `ok`, `md5_src`, `md5_meas`, `seconds`; top level `seed`, `multiplier`, `spec`.

- [ ] **Step 1: Create the folders and links**

`/tmp/claude-1002/-home-congcongliu--claude/9236a300-298a-44d0-bb49-0901a6adc910/scratchpad/scaffold_noise_topic.sh`:

```bash
#!/bin/bash
set -euo pipefail
RES=/data0/congcong/code/GS/Results_20260908
N=$RES/T1FLAIR_AF16_noise2x_v3_jiangyizhou
H=$RES/T1FLAIR_AF16_v3_jiangyizhou
A=$RES/T1FLAIR_AF12_v3_jiangyizhou
[ -e "$N" ] && { echo "refusing: $N exists"; exit 1; }
mkdir -p "$N"/{data,logs,recon,figures,scripts}
for c in T1 FLAIR; do
  ln -s "$A/data/ksp_$c.h5" "$N/data/ksp_$c.h5"
  ln -s "$A/data/csm_espirit_$c.h5" "$N/data/csm_espirit_$c.h5"
done
ln -s "$H/data/mask_af16_3d.h5" "$N/data/mask_af16_3d.h5"
ln -s "$H/data/flair_support_2d.h5" "$N/data/flair_support_2d.h5"
md5sum "$N/data/mask_af16_3d.h5"
mkdir -p /home/congcongliu/figs/t1flair_v3_af16_noise2x
ls -la "$N/data"
```

```bash
bash /tmp/claude-1002/-home-congcongliu--claude/9236a300-298a-44d0-bb49-0901a6adc910/scratchpad/scaffold_noise_topic.sh
```
Expected: md5 `d08179b8e74f35846455983dbe299294`, six symlinks listed.

- [ ] **Step 2: Copy the export metadata the figure script reads**

```bash
cd /tmp && PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python - <<'EOF'
import json
from pathlib import Path
RES = Path("/data0/congcong/code/GS/Results_20260908")
H, N = RES / "T1FLAIR_AF16_v3_jiangyizhou", RES / "T1FLAIR_AF16_noise2x_v3_jiangyizhou"
d = json.load(open(H / "logs/prep_04_export.json"))
for c, e in d["contrasts"].items():
    e["ksp_h5"] = str(N / f"data/ksp_{c}.h5")                     # clean reference (ground truth only)
    e["meas_h5"] = str(N / f"data/meas_{c}.h5")                   # what every arm fits
d["mask3d_path"] = str(N / "data/mask_af16_3d.h5")
d["note"] = ("copied from T1FLAIR_AF16_v3_jiangyizhou/logs/prep_04_export.json on 2026-09-16 for the noise study: same "
             "k-space, same mask; the arms fit data/meas_<c>.h5 (see logs/make_noisy_kspace.json)")
json.dump(d, open(N / "logs/prep_04_export.json", "w"), indent=1)
print(json.dumps({k: d[k] for k in ("mask2d_samples", "mask2d_AF_rect")}))
EOF
```
Expected: `{"mask2d_samples": 2966, "mask2d_AF_rect": 15.190830025067218}`.

- [ ] **Step 3: Write the generator**

`N/scripts/make_noisy_kspace.py`:

```python
#!/usr/bin/env python3
"""make_noisy_kspace.py -- the noise study's measurements: the clean jiangyizhou k-space plus complex Gaussian noise.

Spec: /data0/congcong/code/GS/mcgs_noise_wt/docs/superpowers/specs/2026-09-16-jyz-af16-noise-robustness-design.md,
section 3. Per contrast:
  1. ground truth as t1t2_common.Pair builds it (sum_c conj(S_c) ifft3c(y_c), scale = P99.9 of |GT|), here in double
     precision on the CPU;
  2. sigma0_img = sqrt(var Re + var Im) of GT/scale over air (|GT| > 0.05 P99.9, closed 3x, dilated 12x, complement,
     8-voxel border excluded); sigma0_k = sigma0_img / sqrt(f_acq), f_acq = acquired fraction of the PE plane;
  3. n ~ CN(0, sigma^2) with sigma = MULT * sigma0_k (gauge units), added only on the acquired support (the samples
     that are non-zero in the clean file); PCG64 from SeedSequence(SEED), one spawned child per contrast in the order
     T1, FLAIR, drawn coil by coil;
  4. data/meas_<c>.h5 (key kspace, complex64, the source chunking), re-read from disk for the checks:
       support_exact      meas - clean is exactly 0 outside the support
       mean_z             every coil's mean (Re and Im) within 4 standard errors of 0 (52 tests; 4 SE keeps the
                          chance of a false alarm below 0.5 %, 3 SE would raise one 13 % of the time)
       std_rel            every coil's std of Re and Im equals sigma_raw / sqrt(2) within 0.5 %
       air_noise_rel      air noise of the SENSE-combined noisy full image = sqrt(sigma0_img^2 + sigma^2 f_acq), 2 %
       gt_vs_house        max |GT/scale - house recon/gt_<c>_complex.npy| <= 1e-4 (the house volume is single
                          precision; a wrong map or a flipped array differs by 0.1 or more)
       scale_vs_house     |scale - house scale_p999| / house <= 1e-5 (same reason)
  5. recon/fullnoisy_<c>_complex.npy: the SENSE-combined noisy full image in gauge units (for the noise figure).
Refuses to overwrite. A failed check renames the outputs to *.failed and exits 1.
usage: make_noisy_kspace.py [--contrasts T1,FLAIR] [--threads 16]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import ndimage

R = Path(__file__).resolve().parents[1]
HOUSE = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou")
SPEC = "/data0/congcong/code/GS/mcgs_noise_wt/docs/superpowers/specs/2026-09-16-jyz-af16-noise-robustness-design.md"
SEED, MULT = 20260916, 2.0
ORDER = ("T1", "FLAIR")

ap = argparse.ArgumentParser()
ap.add_argument("--contrasts", default="T1,FLAIR")
ap.add_argument("--threads", type=int, default=16)
a = ap.parse_args()
torch.set_num_threads(a.threads)


def log(*x):
    print(*x, flush=True)


def ifft3c(x):
    d = (-3, -2, -1)
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(x, dim=d), dim=d, norm="ortho"), dim=d)


def sense_adj(y, S):
    """sum_c conj(S_c) ifft3c(y_c), coil by coil, double precision"""
    acc = torch.zeros(y.shape[1:], dtype=torch.complex128)
    for i in range(y.shape[0]):
        acc += S[i].conj() * ifft3c(torch.from_numpy(y[i]).to(torch.complex128))
    return acc


def md5(p, bs=1 << 24):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def air_mask(mag):
    """mag in gauge units (P99.9 = 1)"""
    head = ndimage.binary_closing(mag > 0.05, iterations=3)
    air = ~ndimage.binary_dilation(head, iterations=12)
    keep = np.zeros_like(air)
    keep[8:-8, 8:-8, 8:-8] = True
    return air & keep


def noise_std(v, air):
    return float(np.sqrt(v.real[air].var() + v.imag[air].var()))


out_json = R / "logs/make_noisy_kspace.json"
report = json.load(open(out_json)) if out_json.exists() else {"spec": SPEC, "seed": SEED, "multiplier": MULT,
                                                              "contrasts": {}}
children = dict(zip(ORDER, np.random.SeedSequence(SEED).spawn(len(ORDER))))
failed = False
for c in [x for x in a.contrasts.split(",") if x]:
    dst, img_out = R / f"data/meas_{c}.h5", R / f"recon/fullnoisy_{c}_complex.npy"
    for p in (dst, img_out):
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")
    t0 = time.time()
    src = R / f"data/ksp_{c}.h5"
    with h5py.File(src, "r") as f:
        chunks = f["kspace"].chunks
        y = f["kspace"][:]                                      # (26, 256, 256, 176) complex64
    with h5py.File(R / f"data/csm_espirit_{c}.h5", "r") as f:
        S = torch.from_numpy(f["map"][:]).to(torch.complex128)
    log(f"[{c}] loaded {y.shape} {y.dtype} chunks={chunks} ({time.time() - t0:.0f} s)")

    gt = sense_adj(y, S)
    scale = float(np.quantile(gt.abs().numpy().reshape(-1), 0.999))
    gtn = (gt / scale).numpy()
    del gt
    air = air_mask(np.abs(gtn))
    sigma0_img = noise_std(gtn, air)
    support = np.abs(y).max(axis=0) > 0                        # (X, Y, Z): any coil non-zero
    plane = support.any(axis=0)
    if not np.array_equal(support, np.broadcast_to(plane, support.shape)):
        sys.exit(f"[{c}] the acquired support is not constant along the readout")
    f_acq = float(plane.mean())
    sigma0_k = sigma0_img / np.sqrt(f_acq)
    sigma = MULT * sigma0_k
    sigma_raw = sigma * scale
    log(f"[{c}] scale {scale:.6e}  f_acq {f_acq:.6f} ({int(plane.sum())} lines)  sigma0_img {sigma0_img:.6f}  "
        f"sigma0_k {sigma0_k:.6f}  sigma {sigma:.6f} (raw {sigma_raw:.6e})")

    rng = np.random.Generator(np.random.PCG64(children[c]))
    iy, iz = np.nonzero(plane)
    meas = np.empty_like(y)
    for i in range(y.shape[0]):
        n = (rng.standard_normal((y.shape[1], iy.size)) + 1j * rng.standard_normal((y.shape[1], iy.size))) \
            * (sigma_raw / np.sqrt(2.0))
        yi = y[i].astype(np.complex128)
        yi[:, iy, iz] += n
        meas[i] = yi.astype(np.complex64)
    tmp = dst.with_name(dst.name + ".partial")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("kspace", data=meas, chunks=chunks)
        f.attrs.update(dict(source=str(src.resolve()), seed=SEED, child=ORDER.index(c), contrast=c, multiplier=MULT,
                            sigma_gauge=sigma, sigma_raw=sigma_raw, scale=scale, spec=SPEC))
    tmp.rename(dst)
    del meas
    log(f"[{c}] wrote {dst} ({time.time() - t0:.0f} s)")

    with h5py.File(dst, "r") as f:                             # every check reads what is on disk
        mz = f["kspace"][:]
    checks = {}
    outside = ~support
    exact = all(np.array_equal(mz[i][outside], y[i][outside]) for i in range(y.shape[0]))
    checks["support_exact"] = dict(value=bool(exact), limit=True, ok=bool(exact))
    worst_z, worst_std = 0.0, 0.0
    for i in range(y.shape[0]):
        d = mz[i][:, iy, iz].astype(np.complex128) - y[i][:, iy, iz].astype(np.complex128)
        se = float(sigma_raw / np.sqrt(2.0) / np.sqrt(d.size))
        worst_z = max(worst_z, float(abs(d.real.mean())) / se, float(abs(d.imag.mean())) / se)
        for part in (d.real, d.imag):
            worst_std = max(worst_std, abs(float(part.std()) / float(sigma_raw / np.sqrt(2.0)) - 1.0))
    checks["mean_z"] = dict(value=worst_z, limit=4.0, ok=worst_z <= 4.0)
    checks["std_rel"] = dict(value=worst_std, limit=0.005, ok=worst_std <= 0.005)
    noisy = (sense_adj(mz, S) / scale).numpy()
    air_noisy = noise_std(noisy, air)
    expected = float(np.sqrt(sigma0_img ** 2 + sigma ** 2 * f_acq))
    rel = abs(air_noisy / expected - 1.0)
    checks["air_noise_rel"] = dict(value=rel, limit=0.02, ok=rel <= 0.02)
    house_gt = np.load(HOUSE / f"recon/gt_{c}_complex.npy")
    gdiff = float(np.abs(gtn - house_gt).max())
    checks["gt_vs_house"] = dict(value=gdiff, limit=1e-4, ok=gdiff <= 1e-4)
    house_scale = float(json.load(open(HOUSE / f"logs/classical_{c}.json"))["scale_p999"])
    srel = abs(scale - house_scale) / house_scale
    checks["scale_vs_house"] = dict(value=srel, limit=1e-5, ok=srel <= 1e-5)
    np.save(img_out, noisy.astype(np.complex64))
    ok = all(v["ok"] for v in checks.values())
    for k, v in checks.items():
        log(f"[{c}]   {'OK ' if v['ok'] else 'BAD'} {k:15s} {v['value']} (limit {v['limit']})")
    report["contrasts"][c] = dict(
        shape=list(y.shape), dtype=str(y.dtype), chunks=list(chunks), support_lines=int(plane.sum()), f_acq=f_acq,
        scale=scale, house_scale=house_scale, sigma0_img=sigma0_img, sigma0_k=float(sigma0_k), sigma=float(sigma),
        sigma_raw=float(sigma_raw), air_noise_noisy=air_noisy, air_noise_expected=expected, air_voxels=int(air.sum()),
        checks=checks, ok=ok, md5_src=md5(src.resolve()), md5_meas=md5(dst), seconds=time.time() - t0)
    json.dump(report, open(out_json, "w"), indent=1)
    if not ok:
        failed = True
        for p in (dst, img_out):
            p.rename(p.with_name(p.name + ".failed"))
        log(f"[{c}] CHECKS FAILED: outputs renamed to *.failed")
    del mz, noisy, y, S
log("MAKE_NOISY_KSPACE " + ("FAILED" if failed else "DONE"))
sys.exit(1 if failed else 0)
```

- [ ] **Step 4: Run it detached and wait**

`/tmp/claude-1002/-home-congcongliu--claude/9236a300-298a-44d0-bb49-0901a6adc910/scratchpad/run_make_noisy.sh`:

```bash
#!/bin/bash
cd /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= nice -n 10 \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -u scripts/make_noisy_kspace.py --threads 16 \
  > logs/make_noisy_kspace.log 2>&1
echo "RC=$?" >> logs/make_noisy_kspace.log
```

```bash
setsid nohup bash /tmp/claude-1002/-home-congcongliu--claude/9236a300-298a-44d0-bb49-0901a6adc910/scratchpad/run_make_noisy.sh > /dev/null 2>&1 < /dev/null &
```
Wait for `RC=` in `N/logs/make_noisy_kspace.log` (expected 10 to 30 minutes).
Expected: every check `OK`, `MAKE_NOISY_KSPACE DONE`, `RC=0`; `sigma0_img` about 0.0141 (T1) and 0.0128 (FLAIR); `f_acq` 1.0 and 0.757702 (34139 lines); `air_noise_noisy` about 0.0315 and 0.0287. If a check fails, stop: report the failing check and value to the user; do not loosen a limit without their decision.

- [ ] **Step 5: Look at the disk and the report**

```bash
ls -la /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou/data/ \
       /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou/recon/
tail -20 /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou/logs/make_noisy_kspace.log
```
Expected: `meas_T1.h5` and `meas_FLAIR.h5` about 2.4 GB each; `fullnoisy_{T1,FLAIR}_complex.npy` 92 MB each. No commit yet (the topic is mirrored into the repo in Task 11).

---

### Task 5: Topic scripts with the measurement split, and the gauge check

**Files:**
- Create (copies of `T1FLAIR_AF16_gauss_v3_jiangyizhou/scripts/`): `N/scripts/{t1t2_common,recon_classical,recon_joint_cssense,recon_spirit,recon_zsssdu,recon_dip_inr,check_gauge,add_gs_arm,summarize,readme_tables,refresh_readme_tables,score_mcinr_arm}.py`
- Modify (in those copies): `t1t2_common.py`, `recon_classical.py`, `summarize.py`, `readme_tables.py`, `score_mcinr_arm.py`
- Create: `N/scripts/gauge_check_clean.py`, `/tmp/claude-1002/-home-congcongliu--claude/9236a300-298a-44d0-bb49-0901a6adc910/scratchpad/run_gauge_check.sh`

**Interfaces:**
- Consumes: Task 4's `N/data/meas_<c>.h5`.
- Produces: `t1t2_common.Pair(contrast, dev)` with attributes `meas_path` (str), `y_full` (noisy, gauge units), `y`, `gt_c` / `gt` (clean), `scale` (clean P99.9); env override `T1T2_MEAS_<C>`. `recon_classical.py` honours the same override and records `"measurements"` in its json. `R` in every copied script is `N`; `ROOT` is `/data0/congcong/code/GS/mcgs_noise_wt`.

- [ ] **Step 1: Copy and repoint**

```bash
cd /tmp && PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python - <<'EOF'
from pathlib import Path
RES = Path("/data0/congcong/code/GS/Results_20260908")
G, N = RES / "T1FLAIR_AF16_gauss_v3_jiangyizhou/scripts", RES / "T1FLAIR_AF16_noise2x_v3_jiangyizhou/scripts"
names = ["t1t2_common", "recon_classical", "recon_joint_cssense", "recon_spirit", "recon_zsssdu", "recon_dip_inr",
         "check_gauge", "add_gs_arm", "summarize", "readme_tables", "refresh_readme_tables", "score_mcinr_arm"]
for n in names:
    t = (G / f"{n}.py").read_text()
    t = t.replace("/data0/congcong/code/GS/mcgs_diffmask_wt", "/data0/congcong/code/GS/mcgs_noise_wt")
    t = t.replace("T1FLAIR_AF16_gauss_v3_jiangyizhou", "T1FLAIR_AF16_noise2x_v3_jiangyizhou")
    (N / f"{n}.py").write_text(t)
    print("copied", n)
EOF
cd /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou/scripts && \
  grep -n -E "mcgs_diffmask_wt|T1FLAIR_AF16_gauss|_gauss\b" -- *.py
```
Expected: 12 `copied` lines; the grep prints only the `--tag` default `_gauss` of `score_mcinr_arm.py` (fixed in Step 5).

- [ ] **Step 2: Split `Pair.__init__` in `N/scripts/t1t2_common.py`**

Replace the whole `__init__` of `class Pair` with:

```python
    def __init__(self, contrast: str, dev, chunk: int = 4):
        self.contrast, self.dev, self.chunk = contrast, dev, chunk
        # noise study (spec 2026-09-16, section 4): data/ksp_<c>.h5 is the CLEAN reference and defines only the ground
        # truth and the scale; every arm fits the measurements in data/meas_<c>.h5 (clean k-space + 2 sigma0 of complex
        # Gaussian noise). T1T2_MEAS_<CONTRAST>=<h5 path> overrides the measurement file (the gauge check points it at
        # the clean file and must then reproduce T1FLAIR_AF16_v3_jiangyizhou).
        with h5py.File(R / f"data/ksp_{contrast}.h5", "r") as f:
            y_ref = torch.from_numpy(f["kspace"][:]).to(torch.complex64)
        self.meas_path = os.environ.get(f"T1T2_MEAS_{contrast}") or str(R / f"data/meas_{contrast}.h5")
        if os.environ.get(f"T1T2_MEAS_{contrast}"):
            print(f"[Pair] {contrast}: measurement override T1T2_MEAS_{contrast}={self.meas_path}", flush=True)
        with h5py.File(self.meas_path, "r") as f:
            y_meas = torch.from_numpy(f["kspace"][:]).to(torch.complex64)
        with h5py.File(R / f"data/csm_espirit_{contrast}.h5", "r") as f:
            S = torch.from_numpy(f["map"][:]).to(torch.complex64)
        # phase-3 acquisition probe (mcgs_af16_wt spec 2026-09-12, section 5): T1T2_MASK_<CONTRAST>=<h5 path>
        # swaps the mask for that contrast only. Unset (the default) reads the registered mask, unchanged.
        self.mask_path = os.environ.get(f"T1T2_MASK_{contrast}") or str(R / "data/mask_af16_3d.h5")
        if os.environ.get(f"T1T2_MASK_{contrast}"):
            print(f"[Pair] {contrast}: mask override T1T2_MASK_{contrast}={self.mask_path}", flush=True)
        with h5py.File(self.mask_path, "r") as f:
            m3 = torch.from_numpy(f["mask"][:]).to(torch.float32)
        assert y_ref.shape == S.shape == y_meas.shape, (y_ref.shape, S.shape, y_meas.shape)
        self.S = S.to(dev)
        self.m3 = (m3 > 0.5).to(dev)
        self.NC = y_ref.shape[0]
        self.G = tuple(self.m3.shape)
        # the mask is broadcast along the readout (axis 0); every 2-D method below relies on it
        assert bool((self.m3 == self.m3[0:1]).all()), "mask is not constant along the readout axis"
        self.m2 = self.m3[0]                                   # (ky, kz) = (256, 176)

        gt_c = self.sense_adj(y_ref.to(dev))                    # GT from the CLEAN full k-space
        del y_ref
        self.scale = float(np.quantile(gt_c.abs().reshape(-1).cpu().numpy(), 0.999))
        self.gt_c = gt_c / self.scale
        self.y_full = y_meas.to(dev) / self.scale               # the measurements, fully sampled (noisy by default)
        self.gt = self.gt_c.abs()
        self.y = self.y_full * self.m3.unsqueeze(0)             # the retrospectively acquired data
```

Also change the module docstring's gauge lines to:

```
Gauge:
    GT      = sum_c conj(S_c) . ifft3c(y_ref_c)        from the CLEAN full k-space (data/ksp_<c>.h5)
    scale   = P99.9 of |GT|, applied to GT and to the measurements
    y       = data/meas_<c>.h5 / scale x mask          (noise study: clean + 2 sigma0 of complex Gaussian noise)
```

- [ ] **Step 3: Split the loader in `N/scripts/recon_classical.py`**

Replace

```python
with h5py.File(R / f"data/ksp_{args.contrast}.h5", "r") as f:
    y_full = torch.from_numpy(f["kspace"][:]).to(torch.complex64)
with h5py.File(R / f"data/csm_espirit_{args.contrast}.h5", "r") as f:
    S = torch.from_numpy(f["map"][:]).to(torch.complex64)
with h5py.File(R / "data/mask_af16_3d.h5", "r") as f:
    m3 = torch.from_numpy(f["mask"][:]).to(torch.float32)
assert y_full.shape == S.shape, (y_full.shape, S.shape)
NC = y_full.shape[0]
y_full = y_full.to(dev); S = S.to(dev); m3 = (m3 > 0.5).to(dev)
log(f"[{args.contrast}] coils={NC} vol={tuple(m3.shape)} "
    f"mask frac={m3.float().mean():.5f} AF_rect={1/m3.float().mean():.3f}")
```

with

```python
# noise study (spec 2026-09-16, section 4): ksp_<c>.h5 is the CLEAN reference (ground truth and scale only);
# the arms fit data/meas_<c>.h5, or the file named by T1T2_MEAS_<CONTRAST> (the gauge check's clean override)
MEAS = os.environ.get(f"T1T2_MEAS_{args.contrast}") or str(R / f"data/meas_{args.contrast}.h5")
with h5py.File(R / f"data/ksp_{args.contrast}.h5", "r") as f:
    y_ref = torch.from_numpy(f["kspace"][:]).to(torch.complex64)
with h5py.File(MEAS, "r") as f:
    y_meas = torch.from_numpy(f["kspace"][:]).to(torch.complex64)
with h5py.File(R / f"data/csm_espirit_{args.contrast}.h5", "r") as f:
    S = torch.from_numpy(f["map"][:]).to(torch.complex64)
with h5py.File(R / "data/mask_af16_3d.h5", "r") as f:
    m3 = torch.from_numpy(f["mask"][:]).to(torch.float32)
assert y_ref.shape == S.shape == y_meas.shape, (y_ref.shape, S.shape, y_meas.shape)
NC = y_ref.shape[0]
S = S.to(dev); m3 = (m3 > 0.5).to(dev)
log(f"[{args.contrast}] coils={NC} vol={tuple(m3.shape)} "
    f"mask frac={m3.float().mean():.5f} AF_rect={1/m3.float().mean():.3f}  measurements={MEAS}")
```

replace

```python
gt_c = sense_adj(y_full)                                       # GT from the FULL k-space
mag = gt_c.abs().reshape(-1)
scale = float(np.quantile(mag.cpu().numpy(), 0.999))
gt_c = gt_c / scale; y_full = y_full / scale
```

with

```python
gt_c = sense_adj(y_ref.to(dev))                                # GT from the CLEAN full k-space
del y_ref
mag = gt_c.abs().reshape(-1)
scale = float(np.quantile(mag.cpu().numpy(), 0.999))
gt_c = gt_c / scale; y_full = y_meas.to(dev) / scale          # the measurements (noisy by default)
```

and in the final `json.dump` replace `"coils": int(NC),` with `"coils": int(NC), "measurements": MEAS,`.

- [ ] **Step 4: Point `summarize.py` and `readme_tables.py` at this study**

`N/scripts/summarize.py`:
- delete the two blocks that read `ARC / "t1flair_v3_af16g_joint_score.json"` and `ARC / f"t1flair_v3_af16g_single_{c}_score.json"` (the GS arms of this topic come only through `logs/extra_gs_arms.json`);
- in the docstring replace the `results_arc/t1flair_v3_af16g_{joint,single_T1,single_FLAIR}_score.json   (arc/cycle.py)` line with `results_arc/af16n_p3_*_full4k_score.json via logs/extra_gs_arms.json   (arc/cycle.py, scripts/add_gs_arm.py)`;
- replace `print(f"\n=== {c}   (nominal AF 16, diff-mask study)")` with `print(f"\n=== {c}   (nominal AF 16, noise study: measurements + 2 sigma0)")`.

`N/scripts/readme_tables.py`:
- replace the docstring sentence starting `In the diff-mask study the comparison topic is` (to the end of that sentence) with `In the noise study the comparison topic is T1FLAIR_AF16_v3_jiangyizhou: the same arms on the clean measurements, so the difference column is the effect of 2 sigma0 of added noise.`;
- `REF = "vd-Poisson"` becomes `REF = "no noise"`;
- `j, s = find(rows, c, "gs_joint"), find(rows, c, "gs_single")` becomes `j, s = find(rows, c, "gs_joint_p3"), find(rows, c, "gs_single_p3")`;
- `j20, s20 = find(rows20, c, "gs_joint"), find(rows20, c, "gs_single")` becomes `j20, s20 = find(rows20, c, "gs_joint_p3"), find(rows20, c, "gs_single_p3")`.

- [ ] **Step 5: Point `score_mcinr_arm.py` at the noise run**

In `N/scripts/score_mcinr_arm.py`:
- `ap.add_argument("--tag", default="_gauss", ...` becomes `ap.add_argument("--tag", default="_noise2x", help="the MC-INR run directory suffix: recon/af<AF><tag>/")`;
- replace the docstring paragraph starting `Ported for the diff-mask study (2026-09-16)` with:
  `Noise study (2026-09-16): the MC-INR run fitted this topic's noisy measurements (run_volume.py --af 16 --tag _noise2x --ksp_T1/--ksp_FLAIR data/meas_<c>.h5) with this topic's mask, and mcinr_niessen2025's score.py --af 16 --tag _noise2x put its volumes in gauge units against the clean AF16 topic. This topic's ground truth and scale are the same clean ones, so the two gauges agree by construction.`

- [ ] **Step 6: Write the gauge check**

`N/scripts/gauge_check_clean.py`:

```python
#!/usr/bin/env python3
"""gauge_check_clean.py -- spec section 7 item 4: with the measurements pointed at the CLEAN file, this topic's code
reproduces the zero-noise AF16 topic; without the override it fits data/meas_<c>.h5.

Run after
    T1T2_MEAS_<C>=<clean ksp> recon_classical.py --contrast <C> --gpu G --methods cgsense --suffix _gaugecheck_clean
with the same override set. Checks
  1. that recon_classical row against the AF16 topic's recorded cgsense row, and the scale;
  2. the saved check volume re-scored through this topic's Pair against the same row;
  3. without the override, Pair reads data/meas_<C>.h5, those measurements differ from the clean ones, and the
     ground truth and scale do not change.
Then moves logs/classical_<C>_gaugecheck_clean.json to logs/gauge_check/ and deletes the check volume, so nothing from
the check can be taken for a result.
usage: T1T2_MEAS_<C>=<clean ksp> gauge_check_clean.py --contrast <C> --gpu G
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--contrast", required=True, choices=["T1", "FLAIR"])
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--tol", type=float, default=2e-4)
a = ap.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
c = a.contrast
ENV = f"T1T2_MEAS_{c}"
if not os.environ.get(ENV):
    sys.exit(f"set {ENV} to the clean k-space for this check")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1t2_common as TC  # noqa: E402

H = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou")
KEYS = ("psnr", "ssim", "nmse", "hfen", "head_psnr", "head_ssim", "head_hfen", "sharp", "grain")
house = json.load(open(H / f"logs/classical_{c}.json"))
ref = next(r for r in house["rows"] if r["method"] == "cgsense")
mine_p = TC.R / f"logs/classical_{c}_gaugecheck_clean.json"
mine = json.load(open(mine_p))
row = next(r for r in mine["rows"] if r["method"] == "cgsense_gaugecheck_clean")
bad = []


def compare(tag, got):
    for k in KEYS:
        d = abs(got[k] - ref[k])
        ok = d <= a.tol * max(1.0, abs(ref[k]))
        if not ok:
            bad.append(f"{tag}:{k}")
        print(f"  {'OK ' if ok else 'BAD'} {tag:15s} {k:10s} house={ref[k]:.6f} here={got[k]:.6f} d={d:.2e}", flush=True)


compare("recon_classical", row)
srel = abs(mine["scale_p999"] - house["scale_p999"]) / house["scale_p999"]
print(f"  scale house={house['scale_p999']:.6e} here={mine['scale_p999']:.6e} rel={srel:.2e}")
if srel > 1e-5:
    bad.append("scale")
if mine.get("measurements") != os.environ[ENV]:
    bad.append("recon_classical did not read the override")

dev = torch.device("cuda:0")
vol_p = TC.R / f"recon/cgsense_gaugecheck_clean_{c}_complex.npy"
P = TC.Pair(c, dev)                                            # override active: the measurements are the clean file
compare("Pair.score", P.score(torch.from_numpy(np.load(vol_p)).to(dev), "cgsense"))
y_clean = P.y_full
os.environ.pop(ENV)
Q = TC.Pair(c, dev)                                            # the default: data/meas_<c>.h5
diff = float((Q.y_full - y_clean).abs().max())
same = bool(torch.allclose(Q.gt_c, P.gt_c, rtol=0, atol=1e-6)) and abs(Q.scale - P.scale) <= 1e-9 * P.scale
print(f"  default measurements {Q.meas_path}: max |noisy - clean| = {diff:.4e}; ground truth and scale unchanged: {same}")
if not (Q.meas_path.endswith(f"data/meas_{c}.h5") and diff > 0 and same):
    bad.append("default measurements")

(TC.R / "logs/gauge_check").mkdir(exist_ok=True)
shutil.move(str(mine_p), str(TC.R / "logs/gauge_check" / mine_p.name))
vol_p.unlink()
if bad:
    sys.exit(f"GAUGE CHECK FAILED ({c}): {bad}")
print(f"GAUGE CHECK OK ({c}): clean measurements reproduce T1FLAIR_AF16_v3_jiangyizhou; the default fits data/meas_{c}.h5")
```

- [ ] **Step 7: Run the gauge check for both contrasts on one free card**

`/tmp/claude-1002/-home-congcongliu--claude/9236a300-298a-44d0-bb49-0901a6adc910/scratchpad/run_gauge_check.sh`:

```bash
#!/bin/bash
# run_gauge_check.sh GPU -- spec section 7 item 4, both contrasts, one after the other on one card
GPU=$1
N=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
cd "$N" || exit 1
for c in T1 FLAIR; do
  export T1T2_MEAS_$c="$N/data/ksp_$c.h5"
  env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID "$PY" -u scripts/recon_classical.py \
      --contrast $c --gpu "$GPU" --methods cgsense --suffix _gaugecheck_clean || { echo "RC=1 recon $c"; exit 1; }
  env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID "$PY" -u scripts/gauge_check_clean.py \
      --contrast $c --gpu "$GPU" || { echo "RC=1 check $c"; exit 1; }
  unset T1T2_MEAS_$c
done
echo "RC=0"
```

```bash
setsid nohup bash /tmp/claude-1002/-home-congcongliu--claude/9236a300-298a-44d0-bb49-0901a6adc910/scratchpad/run_gauge_check.sh <CARD> \
  > /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou/logs/gauge_check_clean.log 2>&1 < /dev/null &
```
Expected (about 5 minutes): every line `OK`, `GAUGE CHECK OK (T1)`, `GAUGE CHECK OK (FLAIR)`, `RC=0`; `N/recon/` holds no `*_gaugecheck_clean_*` file and `N/logs/gauge_check/` holds the two jsons. On any `BAD`, stop and report.

- [ ] **Step 8: Checkpoint** — nothing to commit in the repo yet (topic files are mirrored in Task 11).

---

### Task 6: MC-INR reads the noisy measurements

**Files:**
- Modify: `/data0/congcong/code/GS/mcinr_niessen2025/scripts/run_volume.py`

**Interfaces:**
- Consumes: Task 4's measurement files.
- Produces: `run_volume.py --ksp_T1 PATH --ksp_FLAIR PATH` (default: the MC-INR topic's own clean files); `meta.json` gains `ksp` = {contrast: resolved path}. Output directory `T1FLAIR_MCINR_Niessen2025_v3_jiangyizhou/recon/af16_noise2x/` for `--tag _noise2x`.

- [ ] **Step 0: Record the repository's test result before the change** (the tests use the GPU; pick a free card among 1, 2, 5)

```bash
cd /data0/congcong/code/GS/mcinr_niessen2025 && CUDA_VISIBLE_DEVICES=<CARD> CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1 PYTHONPATH= \
  /home/congcongliu/anaconda3/envs/GR_pt1.10/bin/python -m pytest -q tests 2>&1 | tail -2
```
Write the summary line down (for example `NN passed in ...s`).

- [ ] **Step 1: Add the flags**

After the `--mask_FLAIR` argument:

```python
ap.add_argument("--ksp_T1", default=None, help="h5 k-space for T1 (default: the topic's clean file); the noise study passes its noisy measurements")
ap.add_argument("--ksp_FLAIR", default=None, help="h5 k-space for FLAIR (default: the topic's clean file)")
```

Replace

```python
for c in C.CONTRASTS:
    k = torch.from_numpy(C.read_h5(C.ksp_path(c), "kspace"))
```

with

```python
ksp_paths = {"T1": args.ksp_T1 or C.ksp_path("T1"), "FLAIR": args.ksp_FLAIR or C.ksp_path("FLAIR")}
for c in C.CONTRASTS:
    k = torch.from_numpy(C.read_h5(ksp_paths[c], "kspace"))
```

and in the `meta = dict(...)` call add, after `masks={...},`:

```python
            ksp={c: str(Path(ksp_paths[c]).resolve()) for c in C.CONTRASTS},
```

- [ ] **Step 2: Run the repository's tests**

```bash
cd /data0/congcong/code/GS/mcinr_niessen2025 && CUDA_VISIBLE_DEVICES=<CARD> CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1 PYTHONPATH= \
  /home/congcongliu/anaconda3/envs/GR_pt1.10/bin/python -m pytest -q tests 2>&1 | tail -2
```
Expected: the same summary line as Step 0.

- [ ] **Step 3: One-slice smoke on the noisy data** (on a free card among 1, 2, 5; about a minute)

`/tmp/claude-1002/-home-congcongliu--claude/9236a300-298a-44d0-bb49-0901a6adc910/scratchpad/mcinr_smoke.sh`:

```bash
#!/bin/bash
GPU=$1
N=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou
cd /data0/congcong/code/GS/mcinr_niessen2025 || exit 1
env PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GR_pt1.10/bin/python -u scripts/run_volume.py \
  --af 16 --tag _noise2x --gpu "$GPU" --slices 128 \
  --mask_T1 "$N/data/mask_af16_3d.h5" --mask_FLAIR "$N/data/mask_af16_3d.h5" \
  --ksp_T1 "$N/data/meas_T1.h5" --ksp_FLAIR "$N/data/meas_FLAIR.h5"
echo "RC=$?"
```

```bash
bash /tmp/claude-1002/-home-congcongliu--claude/9236a300-298a-44d0-bb49-0901a6adc910/scratchpad/mcinr_smoke.sh <CARD> 2>&1 | tail -5
cat /data0/congcong/code/GS/Results_20260908/T1FLAIR_MCINR_Niessen2025_v3_jiangyizhou/recon/af16_noise2x_smoke/meta.json | head -30
```
Expected: `DONE  1 slices`, `RC=0`, and `meta.json` with `"ksp"` pointing at `.../meas_T1.h5` and `.../meas_FLAIR.h5`. The loss numbers are NOT_EVIDENCE.

- [ ] **Step 4: Commit in the MC-INR repository (local only, never pushed)**

```bash
cd /data0/congcong/code/GS/mcinr_niessen2025 && git add scripts/run_volume.py && git commit -q -m "run_volume.py: --ksp_T1/--ksp_FLAIR, the k-space to fit (default: the topic's clean files), recorded in meta.json; used by the jyz AF16 noise study (tag _noise2x)"
```

---

### Task 7: GS configs, the runner, and a GS smoke run

**Files:**
- Create: `configs/t1flair_v3_af16n_jyz.yaml`, `configs/t1flair_v3_af16n_jyz_singleT1.yaml`, `configs/t1flair_v3_af16n_jyz_singleFLAIR.yaml`
- Create: `run_noise_mcgs.sh`

**Interfaces:**
- Consumes: Task 2 (`measurement_path`), Task 3 (`--save_render`), Task 4 (measurement files).
- Produces: `./run_noise_mcgs.sh EXP GPU ITERS [cycle.py args]` with `EXP` starting `af16n_`, `SINGLE=T1|FLAIR`, `SAVE_VOLUMES=1`; outputs `results_arc/<EXP>{.log,_score.json,_volumes.npz}`.

- [ ] **Step 1: Write the three configs**

```bash
cd /data0/congcong/code/GS/mcgs_noise_wt && PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python - <<'EOF'
import re
from pathlib import Path
import yaml
C = Path("configs")
N = "/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou"
HEADER = """# ============================================================================
# t1flair_v3_af16n_jyz{kind}.yaml -- jiangyizhou T1+FLAIR, noise study, nominal AF 16 (2026-09-16).
# Cloned from the diff-mask Gaussian configs; spec docs/superpowers/specs/2026-09-16-jyz-af16-noise-robustness-design.md.
# ONE thing differs from T1FLAIR_AF16_v3_jiangyizhou: the reconstruction fits data/meas_<c>.h5 = the clean k-space
# plus complex Gaussian noise with per-sample std 2 sigma0 (logs/make_noisy_kspace.json). data_path stays the CLEAN
# k-space (symlink into T1FLAIR_AF12_v3_jiangyizhou/data) and defines the ground truth and the scale only.
# Mask: the house vd-Poisson disc AF16 mask (symlink into T1FLAIR_AF16_v3_jiangyizhou/data, md5 d08179b8...), 2966 PE
# points, T1 15.19x. Recipe: the AF16 phase-3 recipe, frozen; its argument list lives in run_noise_mcgs.sh only.
# ============================================================================
"""
for src, dst, kind in (("t1flair_v3_af16g_jyz.yaml", "t1flair_v3_af16n_jyz.yaml", ""),
                       ("t1flair_v3_af16g_jyz_singleT1.yaml", "t1flair_v3_af16n_jyz_singleT1.yaml", "_singleT1"),
                       ("t1flair_v3_af16g_jyz_singleFLAIR.yaml", "t1flair_v3_af16n_jyz_singleFLAIR.yaml", "_singleFLAIR")):
    t = (C / src).read_text()
    body = "\n".join(l for l in t.splitlines() if not l.startswith("#")).lstrip("\n") + "\n"
    body = body.replace("T1FLAIR_AF16_gauss_v3_jiangyizhou", "T1FLAIR_AF16_noise2x_v3_jiangyizhou")
    body = body.replace("t1flair_v3_af16g_", "t1flair_v3_af16n_")
    body = re.sub(r"^(\s+)data_path: (\S+)/data/ksp_(T1|FLAIR)\.h5[^\n]*$",
                  lambda m: f"{m.group(1)}data_path: {m.group(2)}/data/ksp_{m.group(3)}.h5\n"
                            f"{m.group(1)}measurement_path: {m.group(2)}/data/meas_{m.group(3)}.h5",
                  body, flags=re.M)
    (C / dst).write_text(HEADER.format(kind=kind) + body)
    cfg = yaml.safe_load((C / dst).read_text())
    for ch in ("plain", "contrast"):
        d = cfg["data"][ch]
        name = d["name"]
        assert d["data_path"] == f"{N}/data/ksp_{name}.h5", d["data_path"]
        assert d["measurement_path"] == f"{N}/data/meas_{name}.h5", d["measurement_path"]
        assert d["csm_path"] == f"{N}/data/csm_espirit_{name}.h5", d["csm_path"]
        assert d["mask_path"] == f"{N}/data/mask_af16_3d.h5", d["mask_path"]
        assert d.get("known_zero", False) == (name == "FLAIR"), (dst, ch, d.get("known_zero"))
    print(dst, "ok", cfg["output"]["exp_name"])
EOF
```
Expected: three `ok` lines with exp names `t1flair_v3_af16n_joint`, `t1flair_v3_af16n_single_T1`, `t1flair_v3_af16n_single_FLAIR`.

- [ ] **Step 2: Write the runner**

`run_noise_mcgs.sh` (mode 755):

```bash
#!/usr/bin/env bash
# One MC-GS run of the noise study on the jiangyizhou T1+FLAIR pair (2026-09-16).
#   run_noise_mcgs.sh EXP GPU ITERS [extra arc/cycle.py args...]
#   EXP must start with af16n_.  SINGLE=T1|FLAIR picks that contrast's single-arm config.  SAVE_VOLUMES=1 dumps the npz.
#
# The study varies ONE thing against T1FLAIR_AF16_v3_jiangyizhou: the measurements carry 2 sigma0 of added complex
# Gaussian noise (data.<channel>.measurement_path = data/meas_<c>.h5); ground truth and scale stay on the clean
# k-space; the mask is the house vd-Poisson AF16 mask (2966 points, T1 15.19x). Spec:
# docs/superpowers/specs/2026-09-16-jyz-af16-noise-robustness-design.md.
#
# The recipe is the AF16 PHASE-3 recipe, frozen, byte-identical to run_diffmask_mcgs.sh's recipe block:
#   P6            repaired cold start (init_mode head, 3-NN scales, edge-TV threshold from ZF),
#                 3.6M points, edge-TV lambda_geo 0.10 / kappa_q 0.85 / wfloor 0.10 / eps 0.04,
#                 kspace_high 2.2, warmup 500
#   + phase 3     wavelet db4 on its finest level only, lambda_wav T1 0.12 / FLAIR 0.10
#                 common-edge cross-gradient prior --lambda_edge 0.001
#                 phase-only TV --lambda_ptv 0.03
# Extra args are appended last, so they override (cycle.py: repeated flags -> last wins; --set applies in
# argv order).
set -uo pipefail
if [[ $# -lt 3 ]]; then echo "usage: $0 EXP GPU ITERS [extra cycle.py args...]" >&2; exit 2; fi
EXP=$1; GPU=$2; ITERS=$3; shift 3
case "$EXP" in
  af16n_*) SHORT=af16n ;;
  *) echo "EXP must start with af16n_ (keeps the noise study apart from every other line): $EXP" >&2; exit 2 ;;
esac
WARMUP=${WARMUP:-500}
ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
case "${SINGLE:-joint}" in
  joint) CONFIG=configs/t1flair_v3_${SHORT}_jyz.yaml ;;
  T1)    CONFIG=configs/t1flair_v3_${SHORT}_jyz_singleT1.yaml ;;
  FLAIR) CONFIG=configs/t1flair_v3_${SHORT}_jyz_singleFLAIR.yaml ;;
  *)     echo "SINGLE must be T1, FLAIR or unset: ${SINGLE}" >&2; exit 2 ;;
esac
LOG="$ROOT/results_arc/${EXP}.log"
for path in "$LOG" "$ROOT/results_arc/${EXP}_score.json" "$ROOT/results_arc/${EXP}"; do
  if [[ -e "$path" ]]; then echo "refusing to overwrite existing output: $path" >&2; exit 1; fi
done
VOL=()
if [[ "${SAVE_VOLUMES:-0}" == "1" ]]; then VOL=(--save_volumes); fi
mkdir -p "$ROOT/results_arc"
cd "$ROOT"
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_EXTENSIONS_DIR=/data0/congcong/code/GS/mcgs_noise_wt.torch_ext \
  "$PY" -u arc/cycle.py --exp "$EXP" --gpu "$GPU" --iters "$ITERS" \
  --lambda_geo 0.10 --warmup "$WARMUP" --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 \
  --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn \
  --set loss.edge_tv_kappa_source=zf \
  --set gaussian.initial_num_points=3600000 \
  --set loss.wav_name=db4 --set loss.wav_levels=1 \
  --set loss.lambda_wav_plain=0.12 --set loss.lambda_wav=0.10 \
  --lambda_edge 0.001 --lambda_ptv 0.03 \
  --base_cfg "$CONFIG" "${VOL[@]}" "$@" > "$LOG" 2>&1
RC=$?
echo "[$(date +%H:%M)] $EXP rc=$RC  $(grep -o 'head_psnr[^,]*' "$ROOT/results_arc/${EXP}_score.json" 2>/dev/null | head -2 | tr '\n' ' ')" >> "$ROOT/results_arc/noise_runs.log"
exit $RC
```

- [ ] **Step 3: Prove the recipe block is byte-identical**

```bash
cd /data0/congcong/code/GS/mcgs_noise_wt && chmod 755 run_noise_mcgs.sh && \
diff <(sed -n '/--lambda_geo 0.10 --warmup/,/--lambda_edge 0.001 --lambda_ptv 0.03/p' run_diffmask_mcgs.sh) \
     <(sed -n '/--lambda_geo 0.10 --warmup/,/--lambda_edge 0.001 --lambda_ptv 0.03/p' run_noise_mcgs.sh) && echo RECIPE_IDENTICAL
```
Expected: `RECIPE_IDENTICAL`.

- [ ] **Step 4: 60-iteration smoke on the noisy data** (a free card among 1, 2, 5; a few minutes)

`/tmp/claude-1002/-home-congcongliu--claude/9236a300-298a-44d0-bb49-0901a6adc910/scratchpad/gs_smoke.sh`:

```bash
#!/bin/bash
GPU=$1
cd /data0/congcong/code/GS/mcgs_noise_wt || exit 1
WARMUP=20 SAVE_VOLUMES=1 ./run_noise_mcgs.sh af16n_smoke60_joint "$GPU" 60 --save_render
echo "RC=$?"
```

```bash
bash /tmp/claude-1002/-home-congcongliu--claude/9236a300-298a-44d0-bb49-0901a6adc910/scratchpad/gs_smoke.sh <CARD> | tail -2
cd /data0/congcong/code/GS/mcgs_noise_wt && grep -E "\[Measurement\]|zero_lines|actual acceleration|saved volumes" results_arc/af16n_smoke60_joint.log
PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python -c "import numpy as np; print(sorted(np.load('results_arc/af16n_smoke60_joint_volumes.npz').files))"
```
Expected: `RC=0`; two `[Measurement] fitting to .../data/meas_T1.h5` / `meas_FLAIR.h5` lines; FLAIR `zero_lines=233/2966` (the shutter found as in the clean runs); `actual acceleration: 15.19x`; npz keys include `recT1_render_complex` and `recFL_render_complex`. The PSNR numbers are NOT_EVIDENCE.

- [ ] **Step 5: Commit**

```bash
cd /data0/congcong/code/GS/mcgs_noise_wt && git add configs/t1flair_v3_af16n_jyz.yaml configs/t1flair_v3_af16n_jyz_singleT1.yaml \
  configs/t1flair_v3_af16n_jyz_singleFLAIR.yaml run_noise_mcgs.sh && git commit -q -m "noise study: GS configs (measurement_path = data/meas_<c>.h5, ground truth on the clean k-space) and run_noise_mcgs.sh with the frozen AF16 phase-3 recipe block"
```

---

### Task 8: Scheduler, finalize, diagnostics, comparison and figure scripts

**Files:**
- Create: `N/scripts/run_all_noise.py`, `N/scripts/launch_noise.sh`, `N/scripts/finalize_noise.sh`, `N/scripts/score_gs_render.py`, `N/scripts/compare_noise.py`, `N/scripts/fig_noise_data.py`
- Create: `N/scripts/fig_three_planes_compact.py` (copy of `H/scripts/fig_three_planes_compact.py`, repointed)

**Interfaces:**
- Consumes: Tasks 4–7.
- Produces: the 57-job queue; `N/logs/run_all_noise.log`, `N/logs/run_all_noise_status.json`, `N/logs/RUN_ALL_NOISE_DONE`; `N/logs/gs_render_<c>.json`; `N/logs/compare_noise_<metric>.md`; figures `N/figures/t1flair_v3_af16n_*` mirrored to `/home/congcongliu/figs/t1flair_v3_af16_noise2x/`.

- [ ] **Step 1: The scheduler**

`N/scripts/run_all_noise.py`:

```python
#!/usr/bin/env python3
"""run_all_noise.py -- every arm of the jiangyizhou AF16 noise study, queued onto free cards (2026-09-16).

The study varies ONE thing against T1FLAIR_AF16_v3_jiangyizhou (the zero-noise column, not re-run): the measurements
carry 2 sigma0 of added complex Gaussian noise (data/meas_<c>.h5); ground truth and scale stay on the clean k-space.
Spec: /data0/congcong/code/GS/mcgs_noise_wt/docs/superpowers/specs/2026-09-16-jyz-af16-noise-robustness-design.md.

Arms and grids are the AF16 topic's (as in run_all_diffmask.py), with the TV and JTV grids extended two steps upward
because noise moves the regularisation optimum up; --extend adds more steps to one family if a winner still sits on
the upper edge. MC-GS / GS-single run the frozen AF16 phase-3 recipe (run_noise_mcgs.sh) with --save_render.
L1-ESPIRiT is retired (2026-09-16) and not queued.

Scheduling rules, as in run_all_diffmask.py: one GPU job of this user per card; a card is eligible when no compute
process of this uid is on it, no job of this queue is on it, it has the job's memory free, and its
Cardiac_FreeRunning_MCGS/work/.gpulock.<idx> lock can be taken (a lock whose recorded owner is dead is reaped). A job
whose result exists is skipped; a job already running outside the queue (same argv AND working directory) is waited
for; a failed non-GS job is retried once; a failed GS run is only logged; MC-INR yields with rc 3 and is re-queued.
When nothing is queued or running: finalize_noise.sh.

Launch detached through scripts/launch_noise.sh. --dry_run prints the queue and the card eligibility and exits.
usage: run_all_noise.py [--cards 1,2,5] [--max_gpu_jobs 3] [--extend tv_T1=0.004,0.0056] [--dry_run]
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

RES = Path("/data0/congcong/code/GS/Results_20260908")
N = RES / "T1FLAIR_AF16_noise2x_v3_jiangyizhou"
F = Path("/data0/congcong/code/GS/mcgs_noise_wt")
LOCK_ROOT = RES / "Cardiac_FreeRunning_MCGS/work"
PY = "/home/congcongliu/anaconda3/envs/GSTMI/bin/python"
PY_TCNN = "/home/congcongliu/anaconda3/envs/GR_pt1.10/bin/python"
MCINR = Path("/data0/congcong/code/GS/mcinr_niessen2025")
MCINR_TOPIC = RES / "T1FLAIR_MCINR_Niessen2025_v3_jiangyizhou"
ENV = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID")
UID = os.getuid()
SHORT, TAG = "af16n", "_noise2x"
GS_EXPS = {arm: f"{SHORT}_p3_{arm}_full4k" for arm in ("joint", "singleT1", "singleFLAIR")}

# the AF16 grids plus two upward steps (spec section 5)
GRIDS = {"tv_T1": ["0.00035", "0.0005", "0.0007", "0.001", "0.0014", "0.002", "0.0028"],
         "tv_FLAIR": ["0.00025", "0.00035", "0.0005", "0.0007", "0.001", "0.0014", "0.002"],
         "jtv": ["0.00035", "0.0005", "0.0007", "0.001", "0.0014", "0.002", "0.0028"],
         "spirit": ["0", "0.0005", "0.001", "0.002", "0.004", "0.008"]}
SPIRIT_KS = ("3", "5")


def jobs(grids):
    L, q = N / "logs", []

    def add(tier, name, need, cwd, cmd, done, log, script, args, absent=(), **extra):
        q.append((tier, dict(name=name, need=need, cwd=str(cwd), cmd=cmd, done=str(done), log=str(log),
                             script=script, args=args, absent=list(absent), **extra)))

    for tier, arm, single in ((0, "joint", ""), (1, "singleFLAIR", "FLAIR"), (3, "singleT1", "T1")):
        exp = GS_EXPS[arm]
        add(tier, f"gs_{arm}", 38, F,
            f"SINGLE={single} SAVE_VOLUMES=1 ./run_noise_mcgs.sh {exp} {{gpu}} 4000 --save_render",
            F / f"results_arc/{exp}_score.json", L / f"sched_gs_{arm}.log", "arc/cycle.py", {"--exp": exp})
    for c in ("T1", "FLAIR"):
        add(2, f"base_{c}", 12, N, f"{PY} -u scripts/recon_classical.py --contrast {c} --gpu {{gpu}}",
            L / f"classical_{c}.json", L / f"classical_{c}.log", "recon_classical.py", {"--contrast": c},
            absent=["--suffix"])
    for c in ("T1", "FLAIR"):
        for lam in grids[f"tv_{c}"]:
            add(4, f"tv_{c}_{lam}", 12, N,
                f"{PY} -u scripts/recon_classical.py --contrast {c} --gpu {{gpu}} --methods cssense --admm_outer 200 "
                f"--lam {lam} --suffix _o200_lam{lam}",
                L / f"classical_{c}_o200_lam{lam}.json", L / f"classical_{c}_o200_lam{lam}.log",
                "recon_classical.py", {"--contrast": c, "--suffix": f"_o200_lam{lam}"})
    for lam in grids["jtv"]:
        add(5, f"jtv_{lam}", 20, N,
            f"{PY} -u scripts/recon_joint_cssense.py --topic {N} --contrasts T1,FLAIR --gpu {{gpu}} --admm_outer 200 "
            f"--lam {lam} --suffix _o200_lam{lam}",
            L / f"jcssense_o200_lam{lam}.json", L / f"jcssense_o200_lam{lam}.log",
            "recon_joint_cssense.py", {"--suffix": f"_o200_lam{lam}"})
    for c in ("T1", "FLAIR"):
        add(6, f"zsssdu_{c}", 18, N, f"{PY} -u scripts/recon_zsssdu.py --epochs 100 --contrast {c} --gpu {{gpu}}",
            L / f"zsssdu_{c}.json", L / f"zsssdu_{c}.log", "recon_zsssdu.py", {"--contrast": c})
    for c in ("T1", "FLAIR"):
        add(7, f"inr_{c}", 18, N,
            f"{PY} -u scripts/recon_dip_inr.py --method inr --iters 1500 --eval_every 50 --contrast {c} --gpu {{gpu}}",
            L / f"inr_{c}.json", L / f"inr_{c}.log", "recon_dip_inr.py", {"--method": "inr", "--contrast": c})
    for c in ("T1", "FLAIR"):
        add(8, f"dip_{c}", 40, N,
            f"{PY} -u scripts/recon_dip_inr.py --method dip --iters 5000 --lr 3e-3 --eval_every 100 --contrast {c} "
            f"--gpu {{gpu}}",
            L / f"dip_{c}.json", L / f"dip_{c}.log", "recon_dip_inr.py", {"--method": "dip", "--contrast": c})
    for c in ("T1", "FLAIR"):
        for ks in SPIRIT_KS:
            for lam in grids["spirit"]:
                sfx = f"_ks{ks}_lam{lam}"
                add(9, f"spirit_{c}{sfx}", 24, N,
                    f"{PY} -u scripts/recon_spirit.py --contrast {c} --gpu {{gpu}} --ksize {ks} --lam {lam} "
                    f"--iters 150 --chunk 24 --suffix {sfx}",
                    L / f"spirit_{c}{sfx}.json", L / f"spirit/{c}{sfx}.log",
                    "recon_spirit.py", {"--contrast": c, "--suffix": sfx})
    add(10, "mcinr", 10, MCINR,
        f"{PY_TCNN} -u scripts/run_volume.py --af 16 --tag {TAG} --gpu {{gpu}} --resume --yield_to_others "
        f"--mask_T1 {N}/data/mask_af16_3d.h5 --mask_FLAIR {N}/data/mask_af16_3d.h5 "
        f"--ksp_T1 {N}/data/meas_T1.h5 --ksp_FLAIR {N}/data/meas_FLAIR.h5",
        MCINR_TOPIC / f"recon/af16{TAG}/meta.json", L / "mcinr_noise2x.log",
        "run_volume.py", {"--tag": TAG}, requeue_rc=3)
    q.sort(key=lambda tj: tj[0])
    return [j for _, j in q]


# ---- the shared per-card lock (unchanged from run_all_diffmask.py) -----------------------------
def lock_path(g):
    return LOCK_ROOT / f".gpulock.{g}"


def owner_alive(d):
    try:
        pid = int((d / "pid").read_text().strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def acquire(g, say):
    d = lock_path(g)
    if d.is_dir() and not owner_alive(d):
        say(f"reaping stale lock {d} (recorded owner not running)")
        shutil.rmtree(d, ignore_errors=True)
    try:
        d.mkdir()
    except (FileExistsError, OSError):
        return False
    return True


def release(g, say):
    d = lock_path(g)
    if d.is_dir():
        shutil.rmtree(d, ignore_errors=True)
        say(f"released lock {d}")


def my_procs():
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != UID:
                continue
            toks = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
            cwd = os.path.realpath(os.readlink(f"/proc/{pid}/cwd"))
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            continue
        out.append(([t.decode(errors="replace") for t in toks if t], cwd))
    return out


def matches(proc, j):
    argv, cwd = proc
    if cwd != os.path.realpath(j["cwd"]):
        return False
    if not any(t.endswith(j["script"]) for t in argv):
        return False
    for k, v in j["args"].items():
        if not any(argv[i] == k and argv[i + 1] == v for i in range(len(argv) - 1)):
            return False
    return not any(k in argv for k in j["absent"])


def gpu_state():
    st, bus2i = {}, {}
    for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,pci.bus_id,memory.free",
                                         "--format=csv,noheader,nounits"], text=True).strip().splitlines():
        i, b, fr = [x.strip() for x in line.split(",")]
        st[int(i)] = dict(free=float(fr) / 1024, mine=False)
        bus2i[b] = int(i)
    for line in subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id",
                                         "--format=csv,noheader"], text=True).strip().splitlines():
        if not line.strip():
            continue
        pid, b = [x.strip() for x in line.split(",")]
        try:
            if b in bus2i and os.stat(f"/proc/{pid}").st_uid == UID:
                st[bus2i[b]]["mine"] = True
        except (FileNotFoundError, OSError):
            pass
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="1,2,5")
    ap.add_argument("--max_gpu_jobs", type=int, default=3)
    ap.add_argument("--poll_s", type=int, default=30)
    ap.add_argument("--extend", action="append", default=[], metavar="FAMILY=L1,L2",
                    help="append lambdas to one grid (tv_T1, tv_FLAIR, jtv, spirit); recorded in the log")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    grids = {k: list(v) for k, v in GRIDS.items()}
    for e in a.extend:
        fam, vals = e.split("=", 1)
        assert fam in grids, f"unknown family {fam}"
        grids[fam] += [v for v in vals.split(",") if v and v not in grids[fam]]
    cards = [int(x) for x in a.cards.split(",")]
    logf = None if a.dry_run else open(N / "logs/run_all_noise.log", "a")

    def say(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        if logf:
            logf.write(line + "\n")
            logf.flush()

    queue, running, waiting, done, retried = jobs(grids), [], [], {}, set()
    say(f"armed: {len(queue)} jobs, cards {cards}, max {a.max_gpu_jobs} GPU jobs, grids {json.dumps(grids)}")
    while True:
        for r in list(running):
            rc = r["proc"].poll()
            if rc is None:
                continue
            running.remove(r)
            j = r["job"]
            ok = Path(j["done"]).exists()
            done[j["name"]] = rc
            release(r["gpu"], say)
            say(f"DONE {j['name']} gpu{r['gpu']} rc={rc} result={'yes' if ok else 'NO'} "
                f"{(time.time() - r['t0']) / 60:.1f} min")
            if rc != 0 and not ok and rc == j.get("requeue_rc"):
                queue.insert(0, j)
                say(f"YIELDED {j['name']} (rc={rc}): queued again to resume")
            elif rc != 0 and not ok:
                if j["name"].startswith("gs_"):
                    say(f"FAILED {j['name']}: not retried (run_noise_mcgs.sh refuses to overwrite; read its log)")
                elif j["name"] not in retried:
                    retried.add(j["name"])
                    queue.insert(0, j)
                    say(f"RETRY {j['name']} once")
        procs = my_procs()
        for w in list(waiting):
            if not any(matches(v, w) for v in procs):
                waiting.remove(w)
                if Path(w["done"]).exists():
                    done[w["name"]] = "external"
                    say(f"EXTERNAL DONE {w['name']} result=yes")
                elif w["name"] not in retried:
                    retried.add(w["name"])
                    queue.insert(0, w)
                    say(f"EXTERNAL DONE {w['name']} result=NO -> re-queued once")
                else:
                    done[w["name"]] = "external"
                    say(f"EXTERNAL DONE {w['name']} result=NO (already re-queued once; giving up)")
        for j in list(queue):
            if Path(j["done"]).exists():
                queue.remove(j)
                done[j["name"]] = "skip"
                say(f"SKIP {j['name']} (result exists)")
            elif any(matches(v, j) for v in procs):
                queue.remove(j)
                waiting.append(j)
                say(f"WAIT {j['name']} (already running outside this queue)")
        if not queue and not running and not waiting:
            break
        try:
            st = gpu_state()
        except Exception as e:
            say(f"nvidia-smi failed: {e}")
            time.sleep(a.poll_s)
            continue
        busy = {r["gpu"] for r in running}
        eligible = sorted((g for g in cards if not st[g]["mine"] and g not in busy), key=lambda g: -st[g]["free"])
        if a.dry_run:
            for j in queue:
                print(f"  queued {j['name']:28s} need {j['need']:>2} GB  -> {j['done']}")
            print(f"  {len(queue)} queued")
            print("  cards:", {g: dict(free_gb=round(st[g]["free"], 1), mine=st[g]["mine"],
                                       locked=lock_path(g).is_dir()) for g in cards},
                  "\n  eligible by process/memory:", eligible)
            return
        n_gpu = len(running)
        for g in eligible:
            if n_gpu >= a.max_gpu_jobs:
                break
            j = next((x for x in queue if x["need"] <= st[g]["free"]), None)
            if j is None:
                continue
            if not acquire(g, say):
                say(f"card {g} is locked by another scheduler; skipping it this round")
                continue
            queue.remove(j)
            Path(j["log"]).parent.mkdir(parents=True, exist_ok=True)
            p = subprocess.Popen(j["cmd"].format(gpu=g), shell=True, cwd=j["cwd"], env=ENV, executable="/bin/bash",
                                 stdout=open(j["log"], "a"), stderr=subprocess.STDOUT)
            try:
                (lock_path(g) / "pid").write_text(str(p.pid))
            except OSError:
                pass
            running.append(dict(job=j, gpu=g, proc=p, t0=time.time()))
            say(f"START {j['name']} gpu{g} (free {st[g]['free']:.0f} GB) -> {j['log']}")
            n_gpu += 1
        json.dump(dict(time=time.strftime("%m-%d %H:%M:%S"),
                       running=[dict(name=r["job"]["name"], gpu=r["gpu"], min=round((time.time() - r["t0"]) / 60, 1))
                                for r in running],
                       waiting=[w["name"] for w in waiting], queued=[j["name"] for j in queue], done=done),
                  open(N / "logs/run_all_noise_status.json", "w"), indent=1)
        time.sleep(a.poll_s)
    fails = {n: rc for n, rc in done.items() if isinstance(rc, int) and rc != 0}
    say(f"QUEUE EMPTY. failures: {fails}. finalize ...")
    rc = subprocess.call(["bash", str(N / "scripts/finalize_noise.sh"), GS_EXPS["joint"], GS_EXPS["singleT1"],
                          GS_EXPS["singleFLAIR"]], cwd=N, env=dict(ENV, GPU=str(cards[0])),
                         stdout=open(N / "logs/finalize_noise.log", "a"), stderr=subprocess.STDOUT)
    say(f"FINALIZE rc={rc}")
    (N / "logs/RUN_ALL_NOISE_DONE").write_text(json.dumps(dict(done=done, finalize_rc=rc, grids=grids), indent=1))


if __name__ == "__main__":
    main()
```

`N/scripts/launch_noise.sh` (mode 755):

```bash
#!/bin/bash
# launch_noise.sh [run_all_noise.py args] -- start the noise-study queue detached (the tool shell is zsh and its
# background tasks are killed by the session's memory watchdog, so this has to be a bash script file).
cd /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou || exit 1
exec env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -u scripts/run_all_noise.py "$@"
```

- [ ] **Step 2: The finalize script**

`N/scripts/finalize_noise.sh` (mode 755):

```bash
#!/bin/bash
# finalize_noise.sh JOINT_EXP SINGLE_T1_EXP SINGLE_FLAIR_EXP -- after run_all_noise.py's queue is empty.
#   1. register the three GS runs under the AF16 phase-3 keys (gs_joint_p3 / gs_single_p3), so every table lines
#      up by key with the zero-noise topic T1FLAIR_AF16_v3_jiangyizhou;
#   2. MC-INR: its score.py (gauge units against the clean AF16 topic), then this topic's score_mcinr_arm.py;
#   3. the render-only diagnostic of the GS arms (logs/gs_render_<c>.json);
#   4. summarize.py, README tables against the zero-noise topic, compare_noise.py for the three head metrics;
#   5. figures: the noise figure and the compact three-plane figures (one render at a time).
# GPU=<card> (default 1) is the card used by the two GPU scoring steps.
set -uo pipefail
N=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou
H=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou
M=/data0/congcong/code/GS/mcinr_niessen2025
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
PY_TCNN=/home/congcongliu/anaconda3/envs/GR_pt1.10/bin/python
GPU=${GPU:-1}
JOINT=$1; ST1=$2; SFL=$3
export PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID
cd "$N" || exit 1

"$PY" scripts/add_gs_arm.py --exp "$JOINT" --key gs_joint_p3 --label "MC-GS" --map T1:plain,FLAIR:contrast || exit 1
"$PY" scripts/add_gs_arm.py --exp "$ST1" --key gs_single_p3 --label "GS-single" --map T1:plain || exit 1
"$PY" - "$N" "$SFL" <<'PYEOF' || exit 1
# the single-FLAIR run registers under the same key as the single-T1 run: merge its FLAIR channel in
import json, sys
from pathlib import Path
import numpy as np
R = Path(sys.argv[1]); exp = sys.argv[2]
p = R / "logs/extra_gs_arms.json"; arms = json.load(open(p)) if p.exists() else []
with np.load(f"/data0/congcong/code/GS/mcgs_noise_wt/results_arc/{exp}_volumes.npz") as z:
    np.save(R / "recon/gs_single_p3_FLAIR_mag.npy", np.asarray(z["recT1"], dtype=np.float32))
    np.save(R / "recon/gs_single_p3_FLAIR_complex.npy", np.asarray(z["recT1_complex"], dtype=np.complex64))
arms = [a for a in arms if not (a["key"] == "gs_single_p3" and "FLAIR" in a["map"])]
arms.append(dict(key="gs_single_p3", label="GS-single", exp=exp, map={"FLAIR": "plain"}))
json.dump(arms, open(p, "w"), indent=1); print("registered single-FLAIR", exp)
PYEOF

if [ -f /data0/congcong/code/GS/Results_20260908/T1FLAIR_MCINR_Niessen2025_v3_jiangyizhou/recon/af16_noise2x/meta.json ]; then
  # the MC-INR repository runs everything in its own env (its CLAUDE.md); score.py is CPU-only
  (cd "$M" && "$PY_TCNN" -u scripts/score.py --af 16 --tag _noise2x) || { echo "MC-INR score.py failed"; exit 1; }
  "$PY" -u scripts/score_mcinr_arm.py --gpu "$GPU" --af 16 --tag _noise2x || exit 1
else
  echo "MC-INR run missing: not scored"
fi

"$PY" -u scripts/score_gs_render.py --gpu "$GPU" --joint "$JOINT" --single_T1 "$ST1" --single_FLAIR "$SFL" || exit 1

"$PY" scripts/summarize.py > logs/post_summarize.log 2>&1 || { tail -5 logs/post_summarize.log; exit 1; }
tail -30 logs/post_summarize.log
"$PY" scripts/readme_tables.py "$N" "$H" > logs/readme_tables.md 2>&1 && "$PY" scripts/refresh_readme_tables.py "$N"
for m in head_psnr head_ssim head_hfen; do
  "$PY" scripts/compare_noise.py --metric $m --out logs/compare_noise_$m.md > /dev/null || exit 1
done

CUDA_VISIBLE_DEVICES= "$PY" scripts/fig_noise_data.py || exit 1
for c in T1 FLAIR; do
  "$PY" scripts/fig_three_planes_compact.py --contrast $c --slices 174 110 47 --mask_error \
      --mcgs_key gs_joint_p3 --single_key gs_single_p3 --tag _p3_mcinr --l1_arm spirit --with_zf --caps | tail -2
  "$PY" scripts/fig_three_planes_compact.py --contrast $c --slices 180 124 76 --inset 48 --inset_corner tr bl bl \
      --mask_error --gray_cbar --cbar_planes 0 --tra_rot 1 --err_vmax 0.1 --pdf \
      --mcgs_key gs_joint_p3 --single_key gs_single_p3 --tag _p3_mcinr --l1_arm spirit --with_zf --caps | tail -2
done
echo "FINALIZE_NOISE_DONE $JOINT $ST1 $SFL"
```

- [ ] **Step 3: The render diagnostic**

`N/scripts/score_gs_render.py`:

```python
#!/usr/bin/env python3
"""score_gs_render.py -- the render-only diagnostic of spec section 6 (noise study, 2026-09-16).

The GS arms are scored, like every topic of the series, on x_dc: the model k-space with the MEASURED samples put back.
With noisy measurements that step returns the noise on the acquired points to the image. The runs were saved with
--save_render, so each npz also holds the render before data consistency. This scores both volumes of each GS run
through this topic's Pair (clean ground truth), so the two numbers differ only by the data-consistency step.
Writes logs/gs_render_<c>.json (rows gs_joint_p3_dc / _render, gs_single_p3_dc / _render); these rows are a
diagnostic and never enter logs/summary.json.
usage: score_gs_render.py --gpu N --joint EXP --single_T1 EXP --single_FLAIR EXP
"""
import argparse
import json
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--joint", required=True)
ap.add_argument("--single_T1", required=True)
ap.add_argument("--single_FLAIR", required=True)
a = ap.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1t2_common import Pair, R  # noqa: E402

ARC = Path("/data0/congcong/code/GS/mcgs_noise_wt/results_arc")
# (key, contrast) -> (experiment, npz channel): cycle.py names the channels recT1 (plain) and recFL (contrast);
# a single-contrast run carries its contrast in both, and its plain channel is used
RUNS = {("gs_joint_p3", "T1"): (a.joint, "recT1"), ("gs_joint_p3", "FLAIR"): (a.joint, "recFL"),
        ("gs_single_p3", "T1"): (a.single_T1, "recT1"), ("gs_single_p3", "FLAIR"): (a.single_FLAIR, "recT1")}
dev = torch.device("cuda:0")
for c in ("T1", "FLAIR"):
    P = Pair(c, dev)
    rows = []
    for (key, cc), (exp, ch) in RUNS.items():
        if cc != c:
            continue
        with np.load(ARC / f"{exp}_volumes.npz") as z:
            vols = {"dc": np.asarray(z[f"{ch}_complex"]), "render": np.asarray(z[f"{ch}_render_complex"])}
        for kind, v in vols.items():
            x = torch.from_numpy(v).to(torch.complex64).to(dev)
            assert tuple(x.shape) == tuple(P.gt.shape), (exp, kind, tuple(x.shape))
            row = P.score(x, f"{key}_{kind}", exp=exp)
            rows.append(row)
            print(f"[{c}] {key:13s} {kind:6s} head PSNR {row['head_psnr']:.3f}  SSIM {row['head_ssim']:.4f}  "
                  f"HFEN {row['head_hfen']:.4f}  sharp {row['sharp']:.3f}", flush=True)
    json.dump({"contrast": c, "scale_p999": P.scale, "rows": rows,
               "note": "diagnostic: each GS volume with (dc) and without (render) hard data consistency, both "
                       "scored through this topic's Pair; not an arm"},
              open(R / f"logs/gs_render_{c}.json", "w"), indent=1)
    del P
    torch.cuda.empty_cache()
```

- [ ] **Step 4: The comparison tables**

`N/scripts/compare_noise.py`:

```python
#!/usr/bin/env python3
"""compare_noise.py -- the noise study's headline tables: every arm without and with 2 sigma0 of added noise.

    no noise   T1FLAIR_AF16_v3_jiangyizhou           the house topic, recorded, not re-run
    2 sigma0   T1FLAIR_AF16_noise2x_v3_jiangyizhou   this topic

Same subject, reference k-space, coil maps, mask, ground truth, scorer and frozen AF16 phase-3 MC-GS recipe; the
classical lambdas are re-selected on the noisy data. Only the measurements differ.
Sections: per-arm table; MC-GS against the strongest comparison arm (GS-single counts); the rank of MC-GS on the three
head metrics; the frozen-lambda view; edge winners of the swept families; the render-only diagnostic.
usage: compare_noise.py [--metric head_psnr] [--out FILE]      (read-only on both topics)
"""
import argparse
import json
import re
from pathlib import Path

RES = Path("/data0/congcong/code/GS/Results_20260908")
CLEAN = RES / "T1FLAIR_AF16_v3_jiangyizhou"
NOISE = RES / "T1FLAIR_AF16_noise2x_v3_jiangyizhou"
# summary.json is keyed by arm KEY; the house topic also carries older gs_joint / gs_joint_ar rows (see compare_masks.py)
ARMS = [("gs_joint_p3", "MC-GS"), ("gs_single_p3", "GS-single"), ("zsssdu", "ZS-SSDU"),
        ("cssense_o200", "CS-SENSE (TV, converged)"), ("jcssense", "Joint SENSE (JTV)"),
        ("cssense", "CS-SENSE (35 outer)"), ("spirit", "L1-SPIRiT"), ("cgsense", "CG-SENSE"),
        ("mcinr", "MC-INR"), ("dip", "3D-DIP"), ("inr", "3D-INR"), ("zf", "ZF")]
LOWER = {"head_hfen", "nmse", "hfen"}
HEAD = ("head_psnr", "head_ssim", "head_hfen")
FAMILY = {"cssense_o200": "classical_{c}_o200_lam*.json", "jcssense": "jcssense_o200_lam*.json",
          "spirit": "spirit_{c}_ks{ks}_lam*.json"}

ap = argparse.ArgumentParser()
ap.add_argument("--metric", default="head_psnr")
ap.add_argument("--out", default=None)
a = ap.parse_args()
M = a.metric
better = min if M in LOWER else max
fmt = (lambda v: "--" if v is None else f"{v:.4f}") if "ssim" in M or "hfen" in M else \
      (lambda v: "--" if v is None else f"{v:.3f}")


def load(T):
    return {(r["contrast"], r["method"]): r for r in json.load(open(T / "logs/summary.json"))}


clean, noise = load(CLEAN), load(NOISE)


def val(rows, c, k, m=M):
    r = rows.get((c, k))
    return None if r is None or r.get(m) is None else float(r[m])


def lam_of(p):
    m = re.search(r"_lam([0-9.eE+-]+)\.json$", p.name)
    return None if m is None else float(m.group(1))


out = [f"# Noise study: every arm without and with 2 sigma0 of added measurement noise ({M})", "",
       "Same subject, reference k-space, coil maps, house vd-Poisson mask, ground truth, scorer and frozen AF16 phase-3",
       "MC-GS recipe. Only the measurements differ; the classical lambdas are re-selected on the noisy data.", ""]
for c in ("T1", "FLAIR"):
    out += [f"**{c}**", "", "| arm | no noise | 2 sigma0 | change |", "|---|---|---|---|"]
    for key, lab in ARMS:
        x, y = val(clean, c, key), val(noise, c, key)
        d = "--" if x is None or y is None else fmt(y - x) if y - x < 0 else "+" + fmt(y - x)
        lam = noise.get((c, key), {}).get("lam")
        out.append(f"| {lab}{'' if lam is None else f' (λ {float(lam):g})'} | {fmt(x)} | {fmt(y)} | {d} |")
    out.append("")

out += ["## MC-GS against the strongest comparison arm (GS-single counts)", "",
        "| contrast | data | MC-GS | strongest comparison arm | margin |", "|---|---|---|---|---|"]
for c in ("T1", "FLAIR"):
    for name, rows in (("no noise", clean), ("2 sigma0", noise)):
        mc = val(rows, c, "gs_joint_p3")
        others = [(val(rows, c, k), lab) for k, lab in ARMS if k != "gs_joint_p3" and val(rows, c, k) is not None]
        if mc is None or not others:
            continue
        v, lab = better(others)
        margin = mc - v if M not in LOWER else v - mc
        out.append(f"| {c} | {name} | {fmt(mc)} | {lab} {fmt(v)} | {margin:+.4f} |")
out.append("")

out += ["## Rank of MC-GS on the three head metrics (1 = best among the listed arms)", "",
        "| contrast | data | head PSNR | head SSIM | head HFEN |", "|---|---|---|---|---|"]
for c in ("T1", "FLAIR"):
    for name, rows in (("no noise", clean), ("2 sigma0", noise)):
        cells = []
        for m in HEAD:
            vs = sorted(((val(rows, c, k, m), k) for k, _ in ARMS if val(rows, c, k, m) is not None),
                        key=lambda t: t[0], reverse=m not in LOWER)
            cells.append(next((f"{i + 1}/{len(vs)}" for i, (_, k) in enumerate(vs) if k == "gs_joint_p3"), "--"))
        out.append(f"| {c} | {name} | " + " | ".join(cells) + " |")
out.append("")

out += ["## Frozen-lambda view: each swept family at its zero-noise lambda, on the noisy data", "",
        "| contrast | family | zero-noise λ | noisy data at that λ | noisy data at the re-selected λ |",
        "|---|---|---|---|---|"]
for c in ("T1", "FLAIR"):
    for key, pat in FAMILY.items():
        r0, r1 = clean.get((c, key)), noise.get((c, key))
        if not r0 or not r1 or r0.get("lam") is None:
            continue
        lam0 = float(r0["lam"])
        hit = None
        for p in sorted((NOISE / "logs").glob(pat.format(c=c, ks=r0.get("ksize", "*")))):
            if lam_of(p) is not None and abs(lam_of(p) - lam0) < 1e-12:
                hit = next((r for r in json.load(open(p)).get("rows", []) if r.get("contrast", c) == c), None)
        v0 = None if hit is None else float(hit[M])
        ks = f" (kernel {r0['ksize']})" if r0.get("ksize") else ""
        out.append(f"| {c} | {key} | {lam0:g}{ks} | {fmt(v0)} | {fmt(val(noise, c, key))} (λ {float(r1['lam']):g}) |")
out.append("")

out += ["## Where the re-selected lambdas sit on their grids (2 sigma0)", ""]
for c in ("T1", "FLAIR"):
    for key in FAMILY:
        r = noise.get((c, key))
        if not r or r.get("lam") is None or not r.get("lams"):
            continue
        lams = sorted(set(float(x) for x in r["lams"]))
        lam = float(r["lam"])
        where = "UPPER EDGE" if lam >= lams[-1] else "lower edge" if lam <= lams[0] else "interior"
        out.append(f"- {c} {key}: λ {lam:g} of [{', '.join(f'{x:g}' for x in lams)}] -> {where}")
out.append("")

rend = [NOISE / f"logs/gs_render_{c}.json" for c in ("T1", "FLAIR")]
if all(p.exists() for p in rend):
    out += ["## Diagnostic: GS arms with and without hard data consistency (both through this topic's Pair)", "",
            "| contrast | arm | with DC (as in the tables) | render only | render − with DC |", "|---|---|---|---|---|"]
    for p in rend:
        d = json.load(open(p))
        by = {r["method"]: r for r in d["rows"]}
        for key, lab in (("gs_joint_p3", "MC-GS"), ("gs_single_p3", "GS-single")):
            dc, rd = by.get(f"{key}_dc"), by.get(f"{key}_render")
            if dc and rd:
                out.append(f"| {d['contrast']} | {lab} | {fmt(float(dc[M]))} | {fmt(float(rd[M]))} | "
                           f"{float(rd[M]) - float(dc[M]):+.4f} |")
    out.append("")

text = "\n".join(out)
print(text)
if a.out:
    Path(a.out).write_text(text + "\n")
    print(f"wrote {a.out}")
```

- [ ] **Step 5: The noise figure**

`N/scripts/fig_noise_data.py`:

```python
#!/usr/bin/env python3
"""fig_noise_data.py -- what 2 sigma0 of added measurement noise looks like (noise study, 2026-09-16).

Rows per contrast: the clean fully sampled image (the ground truth, recon/gt_<c>_complex.npy), the fully sampled image
from the noisy measurements (recon/fullnoisy_<c>_complex.npy, written by make_noisy_kspace.py) and
|noisy - clean| / max(GT) on jet, fixed [0, 0.1]. Columns: transverse 180 / coronal 124 / sagittal 76 (the
figure-of-record slices), axis 0 transposed and turned eyes up as in fig_three_planes_compact.py.
Writes figures/t1flair_v3_af16n_noise_data.png and the same file under /home/congcongliu/figs/t1flair_v3_af16_noise2x/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path(__file__).resolve().parents[1]
FIGS = Path("/home/congcongliu/figs/t1flair_v3_af16_noise2x")
SL = (180, 124, 76)
NAMES = ("Transverse", "Coronal", "Sagittal")
rep = json.load(open(R / "logs/make_noisy_kspace.json"))


def plane(v, axis, idx):
    s = [slice(None)] * 3
    s[axis] = idx
    x = v[tuple(s)]
    return np.rot90(x.T, 1) if axis == 0 else x


fig, axes = plt.subplots(6, 3, figsize=(11, 21), constrained_layout=True)
for ci, c in enumerate(("T1", "FLAIR")):
    info = rep["contrasts"][c]
    gt = np.load(R / f"recon/gt_{c}_complex.npy")
    noisy = np.load(R / f"recon/fullnoisy_{c}_complex.npy")
    g, n = np.abs(gt), np.abs(noisy)
    vmax = float(np.quantile(g, 0.999))
    err = np.abs(noisy - gt) / float(g.max())
    rows = ((g, f"{c}: clean full data (GT)\nair noise {info['sigma0_img']:.4f}", "gray", vmax),
            (n, f"{c}: noisy full data\nair noise {info['air_noise_noisy']:.4f}", "gray", vmax),
            (err, f"{c}: |noisy − clean| / max(GT)", "jet", 0.1))
    for ri, (v, title, cmap, hi) in enumerate(rows):
        r = ci * 3 + ri
        for ai in range(3):
            ax = axes[r, ai]
            im = ax.imshow(plane(v, ai, SL[ai]), cmap=cmap, vmin=0, vmax=hi, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{NAMES[ai]} {SL[ai]}", fontsize=12)
            if ai == 0:
                ax.set_ylabel(title, fontsize=10)
        fig.colorbar(im, ax=list(axes[r, :]), shrink=0.8)
fig.suptitle("jiangyizhou, noise study: complex Gaussian noise added to the measured k-space, per-sample std = "
             f"2 x the data's own (seed {rep['seed']})", fontsize=12)
FIGS.mkdir(parents=True, exist_ok=True)
for d in (R / "figures", FIGS):
    fig.savefig(d / "t1flair_v3_af16n_noise_data.png", dpi=110)
    print("wrote", d / "t1flair_v3_af16n_noise_data.png")
```

- [ ] **Step 6: The compact three-plane figure script**

```bash
cd /tmp && PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python - <<'EOF'
from pathlib import Path
RES = Path("/data0/congcong/code/GS/Results_20260908")
src = RES / "T1FLAIR_AF16_v3_jiangyizhou/scripts/fig_three_planes_compact.py"
dst = RES / "T1FLAIR_AF16_noise2x_v3_jiangyizhou/scripts/fig_three_planes_compact.py"
t = src.read_text()
edits = [
    ('F = Path("/home/congcongliu/figs/t1flair_v3_af16"); F.mkdir(parents=True, exist_ok=True)',
     'F = Path("/home/congcongliu/figs/t1flair_v3_af16_noise2x"); F.mkdir(parents=True, exist_ok=True)   # noise study'),
    ('AF_LABEL = f"nominal AF 16, actual {_ex[\'mask2d_AF_rect\']:.2f}x"',
     'AF_LABEL = f"nominal AF 16, actual {_ex[\'mask2d_AF_rect\']:.2f}x, measurements + 2σ0 noise"'),
    ('f"t1flair_v3_af16_{c}_cube', 'f"t1flair_v3_af16n_{c}_cube'),
    ('f"t1flair_v3_af16_{c}_row3planes_compact_', 'f"t1flair_v3_af16n_{c}_row3planes_compact_'),
    ('Writes t1flair_v3_af16_<contrast>_row3planes_compact_<tra>_<cor>_<sag>.png to the served figs folder and figures/.',
     'Writes t1flair_v3_af16n_<contrast>_row3planes_compact_<tra>_<cor>_<sag>.png to the noise study\'s served figs folder\n'
     '(/home/congcongliu/figs/t1flair_v3_af16_noise2x) and figures/.'),
]
for old, new in edits:
    assert t.count(old) == 1, (old, t.count(old))
    t = t.replace(old, new)
dst.write_text(t)
print("wrote", dst)
EOF
grep -n -E "figs/|t1flair_v3_af16" /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou/scripts/fig_three_planes_compact.py
```
Expected: `wrote ...`; the grep shows only `t1flair_v3_af16_noise2x` (folder) and `t1flair_v3_af16n_` (file names).

- [ ] **Step 7: Dry run of the queue**

```bash
chmod 755 /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou/scripts/{launch_noise,finalize_noise}.sh
bash /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou/scripts/launch_noise.sh --dry_run | tail -8
```
Expected: `57 queued`, the first lines `gs_joint`, `gs_singleFLAIR`, `base_T1`, `base_FLAIR`, `gs_singleT1`; the card dictionary for 1, 2 and 5. Nothing is launched and no log file is written.

- [ ] **Step 8: Syntax checks**

```bash
cd /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou/scripts && \
PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m py_compile \
  run_all_noise.py score_gs_render.py compare_noise.py fig_noise_data.py fig_three_planes_compact.py \
  make_noisy_kspace.py gauge_check_clean.py t1t2_common.py recon_classical.py summarize.py readme_tables.py \
  score_mcinr_arm.py && bash -n finalize_noise.sh && bash -n launch_noise.sh && echo SYNTAX_OK
```
Expected: `SYNTAX_OK`.

---

### Task 9: README skeleton, user go-ahead, launch

**Files:**
- Create: `N/README.md`

**Interfaces:**
- Consumes: `N/logs/make_noisy_kspace.json`, `N/logs/gauge_check_clean.log`.
- Produces: README with the `<!-- TABLES:BEGIN -->` / `<!-- TABLES:END -->` markers that `refresh_readme_tables.py` fills; the running queue.

- [ ] **Step 1: Write the README skeleton with the measured noise numbers**

```bash
cd /tmp && PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python - <<'EOF'
import json
from pathlib import Path
N = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou")
rep = json.load(open(N / "logs/make_noisy_kspace.json"))
rows = []
for c in ("T1", "FLAIR"):
    e = rep["contrasts"][c]
    rows.append(f"| {c} | {e['support_lines']} ({e['f_acq']:.4f}) | {e['sigma0_img']:.5f} | {e['sigma0_k']:.5f} | "
                f"{e['sigma']:.5f} | {e['air_noise_expected']:.5f} | {e['air_noise_noisy']:.5f} | "
                f"{'all pass' if e['ok'] else 'FAILED'} |")
text = f"""# Retrospective nominal AF 16 on the 3D T1 + FLAIR pair, jiangyizhou -- measurements with 2 sigma0 of added noise

Noise-robustness study, 2026-09-16. **Read `../T1FLAIR_AF16_v3_jiangyizhou/README.md` first**: same subject, same CFL
k-space, same ESPIRiT maps, same house vd-Poisson disc mask, same ground truth, same gauge, same arms and the same frozen
AF16 phase-3 MC-GS recipe are documented there and not repeated here. Spec:
`/data0/congcong/code/GS/mcgs_noise_wt/docs/superpowers/specs/2026-09-16-jyz-af16-noise-robustness-design.md`.

## What this topic adds

**Exactly one thing differs from `T1FLAIR_AF16_v3_jiangyizhou`: every arm fits noisier measurements.** That topic is
the zero-noise column of this study and is not re-run. `data/ksp_<c>.h5` (symlinks into the AF12 topic) stay the
clean reference and define only the ground truth and the scale; `data/meas_<c>.h5` are what every arm fits.

### The noise

Complex Gaussian noise, independent over coils and samples, added only on the acquired samples (T1 the whole PE plane,
FLAIR the lines inside the vendor shutter; outside stays exactly zero). Per-sample standard deviation
sigma = 2 sigma0_k, where sigma0_k is the data's own noise measured on the clean ground truth's air
(sigma0_img, gauge units, P99.9 = 1) and divided by sqrt(f_acq). `numpy` PCG64 from `SeedSequence({rep['seed']})`, one
realization shared by every arm. Generator `scripts/make_noisy_kspace.py`, report `logs/make_noisy_kspace.json`.

| contrast | acquired lines (fraction) | sigma0_img | sigma0_k | added sigma | expected air noise | measured air noise | checks |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

Checks: meas minus clean exactly zero outside the support; per-coil mean within 4 SE of 0 (52 tests); per-coil std
within 0.5 %; air noise of the noisy full image within 2 % of sqrt(sigma0_img^2 + sigma^2 f_acq); ground truth within
1e-4 of the AF16 topic's single-precision volume and scale within 1e-5 relative. Gauge check (`logs/gauge_check_clean.log`): with the measurements pointed
at the clean file this topic's code reproduces the AF16 topic's CG-SENSE row on every metric.

## Results

<!-- TABLES:BEGIN -->
<!-- TABLES:END -->

## Known limitations

- One subject, one noise level, one realization; the spread over realizations is not measured.
- The added noise is white and independent across coils; the scanner's own noise is correlated across coils.
- Maps and ground truth are clean; only L1-SPIRiT's kernel sees the noisy calibration block.
- The ground truth carries the scanner's own noise sigma0.
- The classical lambdas are chosen against the ground truth while MC-GS runs a frozen recipe.
- The GS arms are scored with hard data consistency, which puts the noise on the acquired points back into the image;
  the render-only diagnostic (`logs/gs_render_<c>.json`) shows how much that costs.

## Regenerating

```
python scripts/make_noisy_kspace.py                       # refuses to overwrite
bash scripts/launch_noise.sh --dry_run                    # the 57 jobs
setsid nohup bash scripts/launch_noise.sh > /dev/null 2>&1 < /dev/null &
```
"""
(N / "README.md").write_text(text)
print("wrote", N / "README.md")
EOF
```

- [ ] **Step 2: Ask the user for the go-ahead** — report the generator checks, the gauge check, the GS and MC-INR smoke results (NOT_EVIDENCE numbers labelled), the dry-run queue and the card state, then wait for an explicit yes before launching.

- [ ] **Step 3: Launch**

```bash
setsid nohup bash /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou/scripts/launch_noise.sh \
  > /dev/null 2>&1 < /dev/null &
sleep 60; tail -5 /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou/logs/run_all_noise.log
```
Expected: `armed: 57 jobs, cards [1, 2, 5], max 3 GPU jobs`, then `START gs_joint gpu<k>` on free cards. Record the scheduler PID (`pgrep -f run_all_noise.py`) in STATUS.

- [ ] **Step 4: Monitor** — check `logs/run_all_noise_status.json` and `nvidia-smi` about hourly; on `FAILED gs_*` read `mcgs_noise_wt/results_arc/<exp>.log` and report before re-running anything. Expected wall time 12 to 16 hours.

---

### Task 10: After the queue — edges, scores, figures, reading

**Files:**
- Modify (outputs): `N/logs/*`, `N/figures/*`, `N/README.md`

**Interfaces:**
- Consumes: `N/logs/RUN_ALL_NOISE_DONE`, `N/logs/finalize_noise.log`.
- Produces: final tables, figures and the reading of spec section 1.

- [ ] **Step 1: Check the queue ended cleanly**

```bash
N=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou
cat $N/logs/RUN_ALL_NOISE_DONE | head -80; tail -15 $N/logs/finalize_noise.log
```
Expected: every job `rc=0` or `skip`, `finalize_rc: 0`, `FINALIZE_NOISE_DONE`.

- [ ] **Step 2: Edge winners**

```bash
cd /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou && sed -n '/Where the re-selected/,/^$/p' logs/compare_noise_head_psnr.md
```
If any family reads `UPPER EDGE`, extend it by two more steps of x sqrt(2) (e.g. T1 TV at 0.0028 -> `--extend tv_T1=0.004,0.0056`; JTV -> `--extend jtv=0.004,0.0056`; FLAIR TV at 0.002 -> `--extend tv_FLAIR=0.0028,0.004`), relaunch with the same `launch_noise.sh --extend ...` (done jobs are skipped; finalize runs again), and record the extension in the README. At most two extension rounds per family; an edge winner after that is written into the limitations.

- [ ] **Step 3: Sanity of the gauge across jobs**

```bash
cd /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou && PYTHONNOUSERSITE=1 PYTHONPATH= \
/home/congcongliu/anaconda3/envs/GSTMI/bin/python - <<'EOF'
import json
from pathlib import Path
N = Path(".")
H = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou")
ref = {c: json.load(open(H / f"logs/classical_{c}.json"))["scale_p999"] for c in ("T1", "FLAIR")}
bad, n = [], 0
for p in sorted(N.glob("logs/*.json")):
    d = json.load(open(p))
    if not isinstance(d, dict) or "scale_p999" not in d:
        continue
    c = d.get("contrast")
    if c in ref:
        n += 1
        if abs(d["scale_p999"] - ref[c]) / ref[c] > 1e-6:
            bad.append((p.name, d["scale_p999"]))
s = json.load(open(N / "logs/summary.json"))
arms = sorted({(r["contrast"], r["method"]) for r in s})
print(f"{n} per-contrast job files checked; scale mismatches: {bad}")
print(f"summary rows: {len(s)}; arms: {arms}")
EOF
```
Expected: no scale mismatch; `summary.json` has, per contrast, zf, cgsense, cssense, cssense_o200, jcssense, spirit, zsssdu, dip, inr, mcinr, gs_joint_p3, gs_single_p3 (24 rows). (`jcssense_*.json` hold both contrasts and have no top-level `contrast`; they are skipped by this check.)

- [ ] **Step 4: Read the headline tables**

```bash
cd /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou && cat logs/compare_noise_head_psnr.md && \
sed -n '/Rank of MC-GS/,/^$/p;/strongest comparison/,/^$/p;/Diagnostic/,/^$/p' logs/compare_noise_head_ssim.md logs/compare_noise_head_hfen.md
```
Write down, for spec section 1: MC-GS rank per head metric and contrast; the margin against the strongest comparison arm, noisy vs zero noise; each arm's loss; the frozen-lambda view; the render diagnostic.

- [ ] **Step 5: Look at every figure**

```bash
ls -la /home/congcongliu/figs/t1flair_v3_af16_noise2x/
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8765/t1flair_v3_af16_noise2x/t1flair_v3_af16n_noise_data.png
```
Open each PNG with the Read tool (the noise figure and the four compact figures). Check: panels not stretched, labels readable, the inset box inside its panel, error maps on jet [0, 0.1], the MC-GS and GS-single columns present, the title carries the noise label. Fix and re-render before reporting anything.

- [ ] **Step 6: Fill the README** — add under `## Results` (outside the TABLES markers) the headline table (margin and ranks), the frozen-lambda view, the render diagnostic, the edge report and a `## Figures` table with URLs, all copied from `logs/compare_noise_*.md` and the figure file names; add a `## Status` line with the date.

---

### Task 11: Records, mirror, commit, push, tag, memory

**Files:**
- Create: `docs/superpowers/records/<date>-jyz-af16-noise.md`, `docs/<date>-af16-noise-jyz-summary.md`, `docs/handoff/<date>-jyz-af16-noise.md`, `t1flair_af16_noise2x_v3_jiangyizhou/` (mirror)
- Modify: `STATUS.md`, `CLAUDE.md`
(`<date>` = the completion day in `YYYY-MM-DD`.)

**Interfaces:**
- Consumes: everything above.
- Produces: branch `jyz-af16-noise` and tag `handoff/<date>-af16-noise` on origin.

- [ ] **Step 1: Mirror the topic into the repo (no data, no volumes, PNG figures only)**

```bash
rsync -a --delete --prune-empty-dirs \
  --include '*/' --include 'README.md' --include 'scripts/*.py' --include 'scripts/*.sh' \
  --include 'logs/*.json' --include 'logs/*.md' --include 'logs/*.tsv' --include 'logs/*.log' \
  --include 'logs/gauge_check/*.json' --include 'figures/*.png' --exclude '*' \
  /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou/ \
  /data0/congcong/code/GS/mcgs_noise_wt/t1flair_af16_noise2x_v3_jiangyizhou/
du -sh /data0/congcong/code/GS/mcgs_noise_wt/t1flair_af16_noise2x_v3_jiangyizhou
```
Expected: well under 100 MB (the gauss mirror is about 48 MB).

- [ ] **Step 2: Write the record, the Chinese summary, the handoff, STATUS.md (Chinese, five sections: done and verified with re-run commands and raw output; decisions waiting for the user; next steps; pitfalls and do-not-redo; why) and a CLAUDE.md section** that names the branch, the worktree, the topic, the reading order and the recipe. Numbers are copied from `logs/compare_noise_*.md`, never retyped from memory.

- [ ] **Step 3: Commit**

```bash
cd /data0/congcong/code/GS/mcgs_noise_wt && git add docs/superpowers/records docs/handoff docs/*-af16-noise-jyz-summary.md STATUS.md CLAUDE.md && \
git add -f t1flair_af16_noise2x_v3_jiangyizhou && git status --short | head -30 && \
git commit -q -m "noise study: jiangyizhou AF16 with 2 sigma0 of added measurement noise - every arm, tables against the zero-noise topic, render diagnostic, figures, record, summary and handoff"
```

- [ ] **Step 4: Push and tag (after the user has seen the results)**

```bash
cd /data0/congcong/code/GS/mcgs_noise_wt && git push -u origin jyz-af16-noise && \
git tag handoff/<date>-af16-noise && git push origin handoff/<date>-af16-noise && \
git ls-remote origin refs/heads/jyz-af16-noise refs/tags/handoff/<date>-af16-noise
```
Expected: both refs listed with the local HEAD hash.

- [ ] **Step 5: Session boundary of the MC-INR repository** (its CLAUDE.md: commit and tag, never push)

```bash
cd /data0/congcong/code/GS/mcinr_niessen2025 && git status --short | head -5 && git tag handoff/<date>-af16-noise && git log -1 --format='%h %s'
```

- [ ] **Step 6: Update the auto-memory** (`/home/congcongliu/.claude/projects/-home-congcongliu/memory/`): one new file for the noise study (branch, worktree, topic, headline numbers, open decisions) and its line in `MEMORY.md`.
