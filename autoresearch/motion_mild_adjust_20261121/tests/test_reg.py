import json

import numpy as np

import geom
import reg
from mcommon import DATA2, REG2, c1


def test_input_paths_by_condition():
    assert reg.t2_path("il", "C0") == c1.HYW / "ksp_T2.h5"                    # no simulated change: the measured data
    assert reg.t2_path("il", "D2") == c1.DATA / "D2" / "ksp_T2.h5"            # the first version's files
    assert reg.t2_path("il", "E2") == c1.DATA / "E2" / "ksp_T2.h5"
    assert reg.t2_path("kz", "Tc") == DATA2 / "kz" / "Tc" / "ksp_T2.h5"       # the motion-corrupted measurement
    assert reg.t2_path("il", "E2Tb") == DATA2 / "il" / "E2Tb" / "ksp_T2.h5"
    assert reg.t1_path("il", "TbL") == DATA2 / "il" / "TbL" / "ksp_T1.h5"
    assert reg.t1_path("il", "Tb") == c1.HYW / "ksp_T1.h5"
    assert reg.out_path("af12", "il", "D2") == REG2 / "af12_il_D2.json"


def test_record_and_load_pose(tmp_path):
    x = np.array([0.1, -0.2, 0.3, 0.5, 0.1, -0.14])
    rec = reg.record(x, -0.123, 1000)
    assert set(rec) == {"theta_deg", "t_mm", "cost", "brain_in_voxels"}
    assert rec["cost"] == -0.123 and rec["brain_in_voxels"] == 1000
    p = tmp_path / "r.json"
    p.write_text(json.dumps(rec))
    g = reg.load_pose(p)
    np.testing.assert_allclose(g.t, x[3:], atol=1e-12)
    np.testing.assert_allclose(g.R, geom.rotmat(np.deg2rad(x[:3])), atol=1e-12)


def _pair():
    import warp
    from helpers import blob_at, phantom
    shape = (64, 64, 48)
    m = np.abs(phantom(shape, seed=3) * blob_at((32, 32, 24), shape, sigma=14.0))
    g = geom.Rigid(geom.rotmat(np.deg2rad([0.0, 0.0, 0.8])), np.array([0.9, -0.5, 0.3]))
    moved = np.abs(warp.rigid(m.astype(np.complex128), g))                # the object after a lasting shift
    brain = np.abs(blob_at((32, 32, 24), shape, sigma=14.0)) > 0.3
    return m, moved, g, brain


def test_register_identical_images_gives_no_motion():
    """The cost must have its optimum at the identity for two identical images. The first version's probe cost
    (mean of the squared normalised dot product) does not: it also grows when more strong edges of the moving image
    are pulled into the mask, and returns 0.44 deg / 0.18 mm here."""
    m, _, _, brain = _pair()
    x, cost = reg.register(m, m, brain)
    assert np.abs(x).max() < 1e-3, x
    assert abs(cost + 1.0) < 1e-9


def test_register_recovers_a_known_pose():
    m, moved, g, brain = _pair()
    x, cost = reg.register(m, moved, brain)
    assert np.abs(x[3:] - g.t).max() < 0.05, x
    assert np.abs(x[:3] - np.array([0.0, 0.0, 0.8])).max() < 0.05, x
    assert -1.0 <= cost < -0.9
    x_inv, _ = reg.register(moved, m, brain)                               # the other way round: the inverse pose
    inv = g.inverse()
    assert np.abs(x_inv[3:] - inv.t).max() < 0.08, x_inv


def test_register_is_blind_to_scale_and_contrast_inversion():
    m, moved, g, brain = _pair()
    x1, _ = reg.register(m, 3.0 * moved, brain)
    x2, _ = reg.register(m, moved.max() - moved, brain)
    assert np.abs(x1[3:] - g.t).max() < 0.05 and np.abs(x2[3:] - g.t).max() < 0.05


def test_register_states_recovers_a_known_pose():
    m, moved, g, brain = _pair()
    poses = reg.register_states([m, moved, m], brain)
    assert len(poses) == 3
    np.testing.assert_array_equal(poses[0].R, np.eye(3))                  # the reference state is the identity
    assert not poses[0].t.any()
    assert np.abs(poses[1].t - g.t).max() < 0.05, poses[1].t
    assert np.rad2deg(np.linalg.norm(geom.axis_angle(poses[1].R @ g.R.T))) < 0.05
    assert np.abs(poses[2].t).max() < 1e-3                                # an unmoved state comes back as no motion
