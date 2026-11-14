import numpy as np

import geom
from common import PIVOT_OFFSET_MM


def test_pivot_conversion_matches_direct():
    g = geom.rigid_from_pivot((0.5, 0.5, 1.5), (0.8, -0.5, 0.3))
    q = np.array(PIVOT_OFFSET_MM)
    r = np.array([10.0, -20.0, 30.0])
    R = geom.rotmat(np.deg2rad([0.5, 0.5, 1.5]))
    np.testing.assert_allclose(g.apply(r), R @ (r - q) + q + [0.8, -0.5, 0.3], atol=1e-9)


def test_rotation_sign_is_right_handed_about_axis2():
    R = geom.rotmat(np.deg2rad([0.0, 0.0, 90.0]))               # axis 0 -> axis 1
    np.testing.assert_allclose(R @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(geom.axis_angle(R), np.deg2rad([0.0, 0.0, 90.0]), atol=1e-12)


def test_inverse_and_compose():
    g = geom.rigid_from_pivot((1, -2, 3), (1, 2, 3))
    r = np.random.default_rng(0).normal(size=(5, 3)) * 50
    np.testing.assert_allclose(g.inverse().apply(g.apply(r)), r, atol=1e-9)
    np.testing.assert_allclose(g.compose(g.inverse()).apply(r), r, atol=1e-9)
    h = geom.Rigid(geom.rotmat(np.deg2rad([0.0, 4.0, 0.0])), np.array([0.0, 0.0, 2.0]))
    np.testing.assert_allclose(g.compose(h).apply(r), g.apply(h.apply(r)), atol=1e-9)   # a.compose(b) = b first


def test_e1a_brain_displacement_is_about_2mm():
    g = geom.rigid_from_pivot((0.5, 0.5, 1.5), (0.8, -0.5, 0.3))
    pts = np.array([[-30.0, 0, 0], [0, 0, 0], [20.0, 40.0, 0]])
    d = np.linalg.norm(g.apply(pts) - pts, axis=1)
    assert 1.0 < d.min() and d.max() < 4.5


def test_grid_and_tre():
    r = geom.grid_mm((4, 6, 8))
    assert r.shape == (4, 6, 8, 3)
    np.testing.assert_allclose(r[0, 0, 0], [-1.5, -2.5, -3.5])
    np.testing.assert_allclose(r.reshape(-1, 3).mean(0), 0, atol=1e-12)
    g = geom.Rigid(np.eye(3), np.array([0.3, 0.0, 0.4]))
    mask = np.zeros((4, 6, 8), bool)
    mask[1:3] = True
    out = geom.tre(g.apply(r) - r, np.zeros_like(r), mask)
    assert abs(out["mean_mm"] - 0.5) < 1e-9 and abs(out["max_mm"] - 0.5) < 1e-9
