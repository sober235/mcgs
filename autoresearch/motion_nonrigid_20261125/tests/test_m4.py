import json

import numpy as np
import pytest
import torch

import ncommon
import nadapter
import npatch
import nrun
import nsim
from adapter import MotionAdapter

SHAPE = (24, 20, 16)


def _pair(grid=3, n_seg=3):
    cls = nadapter.field_adapter(grid, 3.0)
    return cls(SHAPE, True, False, n_seg, 0), MotionAdapter(SHAPE, True, False, n_seg, 0)


def _points(n=200, seed=0):
    return torch.tensor(np.random.default_rng(seed).uniform(-0.9, 0.9, (n, 3)), dtype=torch.float32)


def test_zero_field_is_the_parent_transform():
    a, b = _pair()
    with torch.no_grad():
        for ad in (a, b):
            ad.theta.copy_(torch.tensor([0.01, -0.02, 0.005]))
            ad.t_mm.copy_(torch.tensor([0.3, -0.2, 0.1]))
            ad.seg_t_mm[1].copy_(torch.tensor([0.5, 0.0, -0.4]))
    for p_ in a.parameters():
        p_.requires_grad_(False)
    p = _points()
    for seg in (None, 0, 1, 2):
        pa, Aa = a.transform(p, seg)
        pb, Ab = b.transform(p, seg)
        assert torch.equal(pa, pb) and torch.equal(Aa, Ab)
    assert not a.is_identity()


def test_field_moves_only_its_state_and_has_no_mean():
    a, _ = _pair()
    with torch.no_grad():
        a.seg_ctrl[1].normal_(0.0, 0.3)
    p = _points()
    half = a.half_mm
    base = a.transform(p, 0)[0]
    assert torch.equal(a.transform(p, 2)[0], base)                       # state 2: field still zero
    moved = (a.transform(p, 1)[0] - base) * half
    assert float(moved.abs().max()) > 0.05
    assert torch.allclose(moved, a.field(p, 1), atol=1e-5)
    assert float(a.field_ctrl_mm(1).mean(dim=(0, 1, 2)).abs().max()) < 1e-6
    assert float(a.field_ctrl_mm(1).norm(dim=-1).max()) <= 2 * 3.0 * 3 ** 0.5


def test_set_field_reproduces_a_representable_field_and_hands_the_mean_to_the_pose():
    a, _ = _pair(grid=3)
    r = np.stack(np.meshgrid(*[(np.arange(n) + 0.5) / n for n in SHAPE], indexing="ij"), axis=-1)
    u = np.stack([0.8 * r[..., 1] - 0.2, 0.5 * r[..., 0] * 0 + 0.3, -0.6 * r[..., 2]], axis=-1)   # linear: exact on any grid
    peak, mean = a.set_field(1, u)
    assert peak > 0.1 and np.allclose(mean, [0.2, 0.3, -0.3], atol=1e-3)
    dense = nadapter.dense_field(a.field_ctrl_mm(1).detach().numpy(), SHAPE)
    assert np.allclose(dense + np.asarray(mean), u, atol=2e-3)
    ax = [torch.tensor((np.arange(n) + 0.5) / n * 2 - 1, dtype=torch.float32) for n in SHAPE]
    p = torch.stack(torch.meshgrid(*ax, indexing="ij"), dim=-1).reshape(-1, 3)
    with torch.no_grad():
        got = (a.transform(p, 1)[0] - p) * a.half_mm                     # field + the pose's translation = u
    assert np.allclose(got.numpy().reshape(*SHAPE, 3), u, atol=2e-3)


def test_gradient_reaches_the_field_of_the_rendered_state_only():
    a, _ = _pair()
    for p_ in a.parameters():
        p_.requires_grad_(False)
    a.seg_ctrl.requires_grad_(True)
    p = _points()
    a.transform(p, 1)[0].pow(2).sum().backward()
    g = a.seg_ctrl.grad
    assert float(g[1].abs().max()) > 0 and float(g[0].abs().max()) == 0 and float(g[2].abs().max()) == 0
    a.seg_ctrl.grad = None
    out = a.transform(p, 0)[0]
    assert not out.requires_grad                                         # the reference state has no field


def test_regulariser_and_groups():
    a, _ = _pair()
    assert float(a.field_regulariser(1.0, 1.0, [1, 2])) == 0.0
    with torch.no_grad():
        a.seg_ctrl[1].normal_(0.0, 0.3)
    assert float(a.field_regulariser(0.0, 1.0, [1])) > 0 and float(a.field_regulariser(1.0, 0.0, [2])) == 0.0
    m4 = dict(lr=0.003)
    base = [dict(params=[], lr=1.0, name="x")]
    names = [g["name"] for g in npatch.state_groups(m4)(base, a, dict(seg_theta=0.0005, seg_t=0.01))]
    assert names == ["x", "seg_theta", "seg_t", "seg_ctrl"]
    assert npatch.state_groups(m4)(base, None, {}) is base
    st = a.state()["field"]
    assert st["grid"] == 3 and np.asarray(st["ctrl_mm"]).shape == (3, 3, 3, 3, 3)


def test_detached_model_and_freeze_window():
    class M:
        has_private = False

        def __init__(self):
            self.positions = torch.zeros(4, 3, requires_grad=True)

        def get_density(self, ch):
            return self.positions.sum(1) * 2

    d = npatch.DetachedModel(M())
    assert not d.positions.requires_grad and not d.get_density("contrast").requires_grad and d.has_private is False

    class S:
        detected_at, it = None, 5
    m4 = dict(mode="learn", freeze=10)
    assert not npatch.frozen_now(S, m4)
    S.detected_at, S.it = 100, 110
    assert npatch.frozen_now(S, m4)
    S.it = 111
    assert not npatch.frozen_now(S, m4)
    assert not npatch.frozen_now(S, dict(mode="truth", freeze=10))


def test_validate_needs_matching_truth_modes():
    base = dict(exp="e", arm="N4", cond="DNc", order="il", af="af12", seed=42, channel="contrast", out_dir="o",
                schedule_path="s", truth_path="t", brain_path="b", reg=dict(), native_shift_mm=[0, 0, 0],
                m1=dict(reg_path="r", align=None),
                m2=dict(mode="adaptive", z0=6.0, z1=12.0, cap=0.25, warmup=600, every=100, n_shells=24, stat="median"),
                m3=dict(mode="detect", at=1000, min_frac=0.1, cp_t=8.0, max_states=3, init="reg", stat="shift",
                        lrs=dict(seg_theta=0.0005, seg_t=0.01)),
                m4=dict(mode="learn", grid=5, u_max_mm=3.0, lr=0.003, freeze=300, l2=0.001, smooth=0.01))
    npatch.validate(base)
    with pytest.raises(AssertionError):
        npatch.validate(dict(base, m4=dict(base["m4"], mode="truth")))
    with pytest.raises(AssertionError):
        npatch.validate(dict(base, m3=None))
    npatch.validate(dict(base, m4=None))


def test_deformation_states_and_derived_truth():
    frac = np.linspace(0, 1, 11)
    step = [dict(kind="step", at=0.45, theta_deg=(0, 0, 0), t_mm=(0, 0, 0), soft=1.0)]
    st, amps = nsim.deformation_states(step, frac, ref_train=2)
    assert st == [0] * 5 + [1] * 6 and amps == [0.0, 1.0]
    st, amps = nsim.deformation_states(step, frac, ref_train=8)          # the reference lies in the deformed state
    assert st == [1] * 5 + [0] * 6 and amps == [0.0, -1.0]
    pulse = [dict(kind="pulse", at=0.3, dur=0.2, theta_deg=(0, 0, 0), t_mm=(0, 0, 0), soft=1.0)]
    assert nsim.deformation_states(pulse, frac, 0) == ([0] * 11, [0.0])
    assert nsim.deformation_states(None, frac, 0) == ([0] * 11, [0.0])
    truth = dict(nr_state_of_train=[0, 0, 1, 1], nr_state_amp=[0.0, 1.0], soft_amp=[0.0, 1.0, 1.0, 1.0],
                 state_poses=[dict(R=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], t=[0, 0, 0])], state_of_train=[0, 0, 0, 0])
    t = nrun.derived_truth(truth)
    assert t["state_of_train"] == [0, 0, 1, 1] and len(t["state_poses"]) == 2 and t["ref_state"] == 0
    assert t["contaminated_in_state"] == [False, True, False, False]
    assert truth["state_of_train"] == [0, 0, 0, 0]                       # the source is not changed
    assert nrun.derived_truth(dict(a=1)) == dict(a=1)


def test_the_second_version_points_at_this_study():
    import mcommon
    import mrun
    assert set(mcommon.CONDITIONS2) == set(ncommon.CONDITIONS3) and set(mcommon.ARMS2) == set(ncommon.ARMS3)
    assert mrun.DATA2 == ncommon.DATA3 and str(mrun.RES2).startswith(str(ncommon.ROOT3))
    assert str(ncommon.V2_RES).endswith("mcgs_motion_mild_adjust_20261121/trainer/results_arc")
    exp, cmd = nrun.command("af12", "il", "DNc", "N4", "0", 42, False)
    assert exp == "mm_af12_il_DNc_N4" and cmd[2].endswith("motion_nonrigid_20261125/npatch.py")
    for c in ncommon.CONDITIONS3.values():
        for e in c["intra"] or []:
            assert not any(e["theta_deg"]) and not any(e["t_mm"])        # deformation only
    assert len(ncommon.MATRIX3["formal"]) == 16
