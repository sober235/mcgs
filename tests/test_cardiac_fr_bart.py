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
