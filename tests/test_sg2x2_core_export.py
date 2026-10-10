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
