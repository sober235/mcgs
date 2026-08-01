#!/usr/bin/env python3
"""TV penalty shape: the two knobs that attack staircasing, and the guarantee they are opt-in.

Anisotropic L1 TV (sum_a mean|g_a|) has a constant sub-gradient of magnitude 1 no matter how
small g_a is, so the cheapest way to pay the penalty is to set g_a to exactly zero over whole
regions — piecewise-constant plateaus with steps between them. Two changes address it:

  eps > 0  Charbonnier / pseudo-Huber, sqrt(g^2 + eps^2) - eps. The derivative near g=0 is
           g/eps rather than sign(g), so a small gradient is no longer worth flattening, while
           |g| >> eps still costs |g| and edges survive. This is the anti-staircase knob.
  vector   penalise ||(gx,gy,gz)||_2 per voxel instead of |gx|+|gy|+|gz|. The per-axis form
           is minimised by aligning structure with the voxel axes, which is why the plateaus
           come out axis-aligned; the vector norm is rotationally symmetric.

Both default OFF and must then reproduce the shipped loss EXACTLY — that is what makes every
previous number still valid.

Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python tests/test_tv_charbonnier.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
import torch
from losses.edge_tv import charbonnier, edge_preserving_tv, tv_magnitude


def _legacy_tv(vol):
    """the shipped JointTrainer._tv_loss, transcribed"""
    m = vol.abs()
    return ((m[1:] - m[:-1]).abs().mean()
            + (m[:, 1:] - m[:, :-1]).abs().mean()
            + (m[:, :, 1:] - m[:, :, :-1]).abs().mean())


def _legacy_edge_tv(vol, kappa, w_floor):
    """the shipped edge_preserving_tv, transcribed"""
    def w(g):
        return torch.clamp(1.0 / (1.0 + (g.detach().abs() / kappa) ** 2), min=w_floor, max=1.0)
    m = vol.abs()
    gx, gy, gz = m[1:] - m[:-1], m[:, 1:] - m[:, :-1], m[:, :, 1:] - m[:, :, :-1]
    return ((w(gx) * gx.abs()).mean() + (w(gy) * gy.abs()).mean() + (w(gz) * gz.abs()).mean())


def _vol(seed=0, n=17):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, n + 2, n + 1, generator=g, dtype=torch.float64)


def test_defaults_are_bit_identical():
    v = _vol()
    assert torch.equal(tv_magnitude(v), _legacy_tv(v)), "plain TV default changed the value"
    for kq, wf in ((0.05, 0.1), (0.2, 0.3)):
        assert torch.equal(edge_preserving_tv(v, kq, wf), _legacy_edge_tv(v, kq, wf)), \
            f"edge TV default changed at kappa={kq}"
    print("PASS test_defaults_are_bit_identical")


def test_charbonnier_matches_its_definition_and_limits():
    g = torch.tensor([-2.0, -1e-4, 0.0, 1e-4, 0.5, 3.0], dtype=torch.float64)
    eps = 0.01
    want = torch.sqrt(g ** 2 + eps ** 2) - eps
    assert torch.allclose(charbonnier(g, eps), want), "charbonnier does not match sqrt(g^2+e^2)-e"
    assert torch.equal(charbonnier(g, 0.0), g.abs()), "eps=0 must be exactly |g|"
    # for |g| >> eps the penalty is |g| - eps, not |g|: the -eps is a per-voxel CONSTANT that
    # keeps penalty(0)=0 and contributes nothing to the gradient, so lambda keeps its meaning
    big = torch.tensor([5.0], dtype=torch.float64)
    # residual is exactly eps^2 / (2|g|) = 1e-5 here, so the tolerance must sit above it
    assert abs(float(charbonnier(big, eps) - (big.abs() - eps))) < 2e-5, \
        "|g|>>eps must cost |g|-eps"
    b = big.clone().requires_grad_(True)
    charbonnier(b, eps).sum().backward()
    assert abs(float(b.grad) - 1.0) < 1e-5, "at |g|>>eps the slope must still be 1, like L1"
    print("PASS test_charbonnier_matches_its_definition_and_limits")


def test_small_gradients_stop_being_worth_flattening():
    """The staircase mechanism, tested directly: with L1 the marginal cost of a tiny gradient
    is 1 regardless of its size, so flattening always pays. Charbonnier makes it g/eps."""
    eps = 0.01
    for g0 in (1e-4, 1e-3):
        g = torch.tensor([g0], dtype=torch.float64, requires_grad=True)
        charbonnier(g, eps).backward()
        slope = float(g.grad)
        exact = g0 / (g0 ** 2 + eps ** 2) ** 0.5      # g/eps only to leading order
        assert abs(slope - exact) < 1e-9, f"expected slope {exact:.6g}, got {slope:.6g}"
        assert slope < 0.15, f"tiny gradient still costs {slope:.3f} per unit — no better than L1"
        g = torch.tensor([g0], dtype=torch.float64, requires_grad=True)
        charbonnier(g, 0.0).backward()
        assert abs(float(g.grad) - 1.0) < 1e-9, "L1 control should have slope exactly 1"
    print("PASS test_small_gradients_stop_being_worth_flattening")


def _smoothed_noise(n=64, passes=4, seed=3):
    """Approximately stationary isotropic Gaussian field: repeated separable 1-2-1 smoothing
    of white noise. For such a field the gradient components are mutually uncorrelated, which
    is the condition the 3/2 factor is derived under. (Plain white noise will NOT do — its
    finite differences share a voxel and correlate at 1/2; nor will a random walk, whose
    gradients are wildly anisotropic by construction.)"""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, n, n, generator=g, dtype=torch.float64)
    for _ in range(passes):
        for ax in range(3):
            v = v.movedim(ax, 0)
            v = torch.cat([v[:1], v, v[-1:]], dim=0)
            v = 0.25 * v[:-2] + 0.5 * v[1:-1] + 0.25 * v[2:]
            v = v.movedim(0, ax)
    return v


def test_vector_norm_is_scale_matched_to_per_axis():
    """The vector form is normalised by 3/2 so that lambda transfers: for an isotropic
    gradient field sum_a E|g_a| = 3*sigma*sqrt(2/pi) while E||g||_2 = 2*sigma*sqrt(2/pi).
    Without this the two arms would be compared at different effective strengths — the same
    trap that once manufactured a fake multi-contrast gain in this project."""
    v = _smoothed_noise()
    a = float(tv_magnitude(v))
    b = float(tv_magnitude(v, vector=True))
    assert abs(b / a - 1.0) < 0.05, f"vector arm is {b / a:.3f}x the per-axis arm, not ~1"
    print(f"PASS test_vector_norm_is_scale_matched_to_per_axis (ratio {b / a:.4f})")


def test_per_axis_tv_is_the_one_with_the_axis_bias():
    """Two ramps of EQUAL gradient magnitude, one along a voxel axis and one along the cube
    diagonal. Per-axis TV charges the diagonal sqrt(3) times more, which is why plateau
    boundaries come out axis-aligned; the vector norm must charge them the same.
    (Matching ||grad||_2 between the two ramps is the whole point — comparing ramps of
    different gradient magnitude measures nothing.)"""
    n = 40
    idx = torch.arange(n, dtype=torch.float64)
    s = 0.01
    axis = idx.view(n, 1, 1).expand(n, n, n).contiguous() * s
    step = s / torch.tensor(3.0, dtype=torch.float64).sqrt()      # ||grad||_2 == s as well
    diag = (idx.view(n, 1, 1) + idx.view(1, n, 1) + idx.view(1, 1, n)) * step
    r_axis = float(tv_magnitude(diag)) / float(tv_magnitude(axis))
    r_vec = float(tv_magnitude(diag, vector=True)) / float(tv_magnitude(axis, vector=True))
    assert abs(r_axis - 3.0 ** 0.5) < 0.02, f"per-axis ratio should be sqrt(3), got {r_axis:.3f}"
    assert abs(r_vec - 1.0) < 0.02, f"vector ratio should be 1, got {r_vec:.3f}"
    print(f"PASS test_per_axis_tv_is_the_one_with_the_axis_bias "
          f"(per-axis charges the diagonal {r_axis:.3f}x, vector {r_vec:.3f}x)")


def test_edge_tv_accepts_the_same_knobs():
    v = _vol(seed=5)
    base = float(edge_preserving_tv(v, 0.1, 0.1))
    sm = float(edge_preserving_tv(v, 0.1, 0.1, eps=0.02))
    vec = float(edge_preserving_tv(v, 0.1, 0.1, vector=True))
    assert sm < base, "charbonnier must not exceed L1 (it is sqrt(g^2+e^2)-e <= |g|)"
    assert vec > 0 and abs(vec / base - 1.0) < 0.6, f"vector edge-TV out of scale: {vec / base:.3f}"
    print("PASS test_edge_tv_accepts_the_same_knobs")


if __name__ == "__main__":
    test_defaults_are_bit_identical()
    test_charbonnier_matches_its_definition_and_limits()
    test_small_gradients_stop_being_worth_flattening()
    test_vector_norm_is_scale_matched_to_per_axis()
    test_per_axis_tv_is_the_one_with_the_axis_bias()
    test_edge_tv_accepts_the_same_knobs()
    print("ALL PASS")
