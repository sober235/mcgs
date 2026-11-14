import numpy as np

import geom
import schedule


def ellipse_support(shape=(256, 176)):
    ky, kz = np.meshgrid(np.arange(shape[0]) - shape[0] // 2, np.arange(shape[1]) - shape[1] // 2, indexing="ij")
    return (ky / (shape[0] / 2)) ** 2 + (kz / (shape[1] / 2)) ** 2 <= 1.0


def test_counts_and_reference():
    sup = ellipse_support()
    s = schedule.build(sup)
    assert s["n_trains"] == -(-int(sup.sum()) // 320)
    assert (s["train_of"][~sup] == -1).all() and (s["seg_of"][~sup] == -1).all()
    assert s["seg_of"].max() == 7 and s["seg_of"][sup].min() == 0
    assert s["train_of"][128, 88] == s["ref_train"] and s["seg_of"][128, 88] == s["ref_seg"]
    assert np.bincount(s["train_of"][sup])[:-1].tolist() == [320] * (s["n_trains"] - 1)


def test_order_is_kz_outer_ky_inner():
    sup = ellipse_support()
    s = schedule.build(sup)
    t = s["train_of"]
    first = np.argwhere(t == 0)
    assert first[:, 1].max() <= np.argwhere(t == 1)[:, 1].min()          # kz (axis 1 of the PE plane) advances slowest
    col = t[:, 88][sup[:, 88]]
    assert (np.diff(col) >= 0).all()                                    # within one kz, ky runs in order


def test_poses_zero_at_reference_and_drift_endpoint():
    s = schedule.build(ellipse_support())
    p = schedule.poses([dict(kind="drift", theta_deg=(0, 0.3, 1.0), t_mm=(0.8, 0, 0))], s["train_time_frac"], s["ref_train"])
    assert len(p) == s["n_trains"]
    np.testing.assert_allclose(p[s["ref_train"]].t, 0, atol=1e-12)
    np.testing.assert_allclose(p[s["ref_train"]].R, np.eye(3), atol=1e-12)
    total = p[-1].compose(p[0].inverse())
    np.testing.assert_allclose(np.rad2deg(geom.axis_angle(total.R)), (0, 0.3, 1.0), atol=0.02)


def test_step_pose_is_constant_on_each_side():
    s = schedule.build(ellipse_support())
    f = s["train_time_frac"]
    p = schedule.poses([dict(kind="step", at=0.60, theta_deg=(0, 0, 1.0), t_mm=(0.5, 0, 0))], f, s["ref_train"])
    before, after = [i for i in range(len(p)) if f[i] < 0.60], [i for i in range(len(p)) if f[i] >= 0.60]
    assert s["ref_train"] in before
    for i in before:
        np.testing.assert_allclose(p[i].R, np.eye(3), atol=1e-12)
    for i in after:
        np.testing.assert_allclose(np.rad2deg(geom.axis_angle(p[i].R)), (0, 0, 1.0), atol=1e-9)


def test_segment_mean_poses():
    s = schedule.build(ellipse_support())
    p = schedule.poses([dict(kind="drift", theta_deg=(0, 0, 1.6), t_mm=(0, 0, 0))], s["train_time_frac"], s["ref_train"])
    m = schedule.segment_mean_poses(p, s["seg_of_train"], 8)
    ang = [np.rad2deg(geom.axis_angle(x.R))[2] for x in m]
    assert len(m) == 8 and (np.diff(ang) > 0.15).all() and (np.diff(ang) < 0.25).all()
