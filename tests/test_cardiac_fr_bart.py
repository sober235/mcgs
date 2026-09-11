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
    # 1e-2, not 1e-3: at 30 CG steps the two runs once differed by 7e-3 (2026-09-11) and agreed to < 1e-3 when rerun
    assert _nrmse(bart.read_cfl(d / "x2"), bart.read_cfl(d / "x1")) < 1e-2
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


def test_parse_max_eigenvalue():
    assert bart.parse_max_eigenvalue("x\nMaximum eigenvalue: 5.616503e+03\ny\n") == pytest.approx(5616.503)
    with pytest.raises(ValueError):
        bart.parse_max_eigenvalue("conjugate gradients\n")


def test_max_eigenvalue_grows_with_the_number_of_spokes(phantom, tmp_path):
    d = phantom
    bart.write_cfl(tmp_path / "traj200", bart.read_cfl(d / "traj")[:, :, :200])
    bart.write_cfl(tmp_path / "ksp200", bart.read_cfl(d / "ksp")[:, :, :200])
    l400 = bart.max_eigenvalue(d / "ksp", d / "sens", d / "traj", tmp_path)
    l200 = bart.max_eigenvalue(tmp_path / "ksp200", d / "sens", tmp_path / "traj200", tmp_path)
    assert l400 > 0 and 1.5 < l400 / l200 < 2.5


@pytest.fixture(scope="module")
def grid2x(phantom, tmp_path_factory):
    """Spec 5.6/5.7 correction 5: the same object on a 2x FOV grid (32^3, same voxels). A trajectory whose extent
    matches the 16^3 grid with a 2x oversampled readout, like the data (the `phantom` trajectory reaches |k| 15.5,
    twice the 16 grid's Nyquist); doubled coordinates for the 32 grid; maps Fourier-interpolated from 16^3."""
    d = tmp_path_factory.mktemp("grid2x")
    bart.run(["traj", "-x", 16, "-o", 2, "-y", 400, "-r", "-3", "-G", d / "traj"])
    bart.run(["nufft", d / "traj", phantom / "cimg_true", d / "ksp"])
    bart.coil_maps(d / "traj", d / "ksp", d, d / "sens", grid=16, calib=12)
    bart.write_cfl(d / "traj2x", 2 * bart.read_cfl(d / "traj"))
    bart.interp_maps(d / "sens", d / "sens32_interp", 32, d)          # same FOV, half the voxel: numerics only
    bart.write_cfl(d / "sens32", bart.embed_center(bart.read_cfl(d / "sens"), 32))   # the 2x FOV, same voxels
    return d


def test_dims_reads_the_header(phantom):
    assert bart.dims(phantom / "sens") == (16, 16, 16, 4) + (1,) * 12


def test_crop_and_embed_are_inverse_and_centred():
    a = np.arange(6 * 6 * 6 * 2, dtype=np.complex64).reshape(6, 6, 6, 2)
    c = bart.crop_center(a, 4)
    assert c.shape == (4, 4, 4, 2) and c[0, 0, 0, 0] == a[1, 1, 1, 0]
    e = bart.embed_center(c, 6)
    assert e.shape == a.shape and np.array_equal(bart.crop_center(e, 4), c) and e[0].sum() == 0


def test_doubled_trajectory_sees_the_centre_embedded_image(phantom, grid2x, tmp_path):
    """The k-space of the 16^3 coil image embedded in the centre of a 32^3 grid, sampled at 2 x the trajectory,
    equals the k-space of the 16^3 image at the trajectory: both grids put the image centre at N/2, so the
    central 16^3 of a 32^3 reconstruction is the 16^3 FOV (crop_center)."""
    bart.write_cfl(tmp_path / "cimg32", bart.embed_center(bart.read_cfl(phantom / "cimg_true"), 32))
    bart.run(["nufft", grid2x / "traj2x", tmp_path / "cimg32", tmp_path / "ksp2x"])
    # BART's NUFFT carries a unitary-FFT normalisation: the forward on a grid with 8x the voxels is 8^-1/2 smaller
    # (images reconstructed with -S on the 2x grid are 2.83x brighter in data units; every metric is scale-free)
    assert _nrmse(bart.read_cfl(tmp_path / "ksp2x") * (32 / 16) ** 1.5, bart.read_cfl(grid2x / "ksp")) < 1e-2


def test_interpolated_maps_keep_the_original_nodes(grid2x):
    s16, s32 = bart.read_cfl(grid2x / "sens"), bart.read_cfl(grid2x / "sens32_interp")
    assert s32.shape == (32, 32, 32, 4)
    assert _nrmse(s32[::2, ::2, ::2], s16) < 1e-4


def test_pics_on_the_doubled_grid_cropped_matches_the_single_grid(grid2x, tmp_path):
    """With the 16^3 maps embedded in the 2x FOV grid (zero outside, where the phantom has no signal) the two
    least-squares problems are equivalent, so pics on 32^3 cropped to the centre must reproduce pics on 16^3."""
    d = grid2x
    l16 = bart.max_eigenvalue(d / "ksp", d / "sens", d / "traj", tmp_path)
    l32 = bart.max_eigenvalue(d / "ksp", d / "sens32", d / "traj2x", tmp_path)
    bart.pics(d / "ksp", d / "sens", tmp_path / "x16", d / "traj", reg="l2", lam=1e-3 * l16, iters=50)
    bart.pics(d / "ksp", d / "sens32", tmp_path / "x32", d / "traj2x", reg="l2", lam=1e-3 * l32, iters=50)
    x16 = np.abs(bart.read_cfl(tmp_path / "x16"))
    x32 = np.abs(bart.read_cfl(tmp_path / "x32"))
    inside = bart.crop_center(x32, 16)
    assert np.linalg.norm(x32) ** 2 - np.linalg.norm(inside) ** 2 < 0.01 * np.linalg.norm(inside) ** 2
    assert _nrmse(inside * (16 / 32) ** 1.5, x16) < 0.02       # the NUFFT normalisation, see the test above
