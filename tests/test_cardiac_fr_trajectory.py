import numpy as np

from cardiac_fr.trajectory import GOLDEN_ANGLE, phyllotaxis3d, radial_positions, spoke_coordinates


def test_small_case_matches_the_matlab_loop_by_hand():
    # phyllotaxis3D.m with 2 shots x 3 segments: kost = pi / (2 sqrt(6 - 2)) = pi / 4.
    # The loop runs segment-outer, shot-inner; index = seg + shot * 3; seg 0 is the SI spoke.
    polar, azim = phyllotaxis3d(n_shot=2, n_seg=3, self_nav=True)
    k = np.pi / 4
    np.testing.assert_allclose(polar, [0, k, k * np.sqrt(3), 0, k * np.sqrt(2), 2 * k])
    tp = 2 * np.pi
    np.testing.assert_allclose(azim, [0, GOLDEN_ANGLE % tp, (3 * GOLDEN_ANGLE) % tp,
                                      0, (2 * GOLDEN_ANGLE) % tp, (4 * GOLDEN_ANGLE) % tp])


def test_protocol_size_si_spokes_and_ranges():
    polar, azim = phyllotaxis3d(1000, 22, True)
    si = np.arange(22000) % 22 == 0
    assert (polar[si] == 0).all() and (azim[si] == 0).all()
    assert (polar[~si] > 0).all() and polar.max() <= np.pi / 2 + 1e-12
    assert (azim >= 0).all() and (azim < 2 * np.pi).all()


def test_radial_positions_center_and_spacing():
    r = radial_positions(224)
    assert r[0] == -0.5 and r[112] == 0.0 and np.isclose(r[-1], 0.5 - 1 / 224)
    np.testing.assert_allclose(np.diff(r), 1 / 224)


def test_coordinates_lie_on_spokes_through_the_center():
    k = spoke_coordinates(n_shot=10, n_seg=22, n_samp=224)
    assert k.shape == (3, 224, 220)
    r = radial_positions(224)
    np.testing.assert_allclose(np.linalg.norm(k, axis=0), np.repeat(np.abs(r)[:, None], 220, 1), atol=1e-12)
    np.testing.assert_allclose(k[:, :, 0], np.stack([0 * r, 0 * r, r]), atol=1e-15)   # SI spoke on z
