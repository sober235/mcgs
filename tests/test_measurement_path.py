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
