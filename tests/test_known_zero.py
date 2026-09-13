"""Known-zero region Z (spec 2026-09-05 §4.1 as amended by appendix A, 2026-09-06): whole PE lines that are
numerically zero outside a hard shutter made of padding cuts and/or a DC-centred ellipse. Estimated from the
ACQUIRED lines only. Plus the readout-end noise covariance (R2), whitening and the fit-side tensors.

Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_known_zero.py -v
"""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
from data.known_zero import (MIN_NOISE_SAMPLES, KnownZeroInfo, build_fit_tensors, estimate_known_zero,  # noqa: E402
                             estimate_noise_cov, known_zero_from_support, load_support_map, whitening_matrix)

C, P0, P1, R = 4, 48, 40, 24


def _synth(cut_lo1=None, ellipse=None, noise=2e-3, seed=0, ro_axis=2, dc_shift=(0, 4), hard=True,
          extra_lines=(), extra_amp=0.0):
    """Textured-object multicoil k-space (C,P0,P1,R) with the DC shifted by `dc_shift` on the PE plane (the real
    data has a +22 px offset), whole PE lines outside the vendor support made numerically zero: array columns
    j < cut_lo1 (padding) and/or lines outside an axis-aligned ellipse (a, b) about the DC. hard=True zeroes
    exactly; hard=False leaves a 1e-30 residue like the real data. vd-random mask (35%) + ACS about the DC,
    broadcast along readout. Returns (kspace, mask, dc, keep) with the readout axis moved to `ro_axis`."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.ogrid[:P0, :P1, :R]
    blob = np.exp(-(((zz - P0 / 2) / (P0 / 4)) ** 2 + ((yy - P1 / 2) / (P1 / 4)) ** 2 + ((xx - R / 2) / (R / 4)) ** 2))
    tex = ndimage.gaussian_filter(rng.normal(size=(P0, P1, R)), 0.7)
    obj = blob * (1.0 + 0.8 * tex / tex.std())
    ks = []
    for _ in range(C):
        cm = ndimage.gaussian_filter(rng.normal(size=(P0, P1, R)), 8.0)
        cm = cm / np.abs(cm).max() + 0.5
        ks.append(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(cm * obj), norm="ortho")))
    k = np.stack(ks).astype(np.complex64)
    k = k + noise * (rng.normal(size=k.shape) + 1j * rng.normal(size=k.shape)).astype(np.complex64) / np.sqrt(2)
    k = np.roll(k, dc_shift, axis=(1, 2))
    dc = (P0 // 2 + dc_shift[0], P1 // 2 + dc_shift[1])
    ii, jj = np.meshgrid(np.arange(P0) - dc[0], np.arange(P1) - dc[1], indexing="ij")
    keep = np.ones((P0, P1), bool)
    if cut_lo1 is not None:
        keep[:, :cut_lo1] = False
    if ellipse is not None:
        a, b = ellipse
        keep &= (ii / a) ** 2 + (jj / b) ** 2 < 1.0
    fac = np.where(keep, 1.0, 0.0 if hard else 1e-30).astype(np.float32)
    k = k * fac[None, :, :, None]
    for (i, j) in extra_lines:                                    # sampled lines outside the support with given content
        k[:, i, j, :] = (extra_amp * (rng.normal(size=(C, R)) + 1j * rng.normal(size=(C, R))) / np.sqrt(2)).astype(np.complex64)
    m2 = rng.random((P0, P1)) < 0.35
    m2[dc[0] - 4:dc[0] + 4, dc[1] - 4:dc[1] + 4] = True
    for (i, j) in extra_lines:
        m2[i, j] = True
    mask = np.repeat(m2[:, :, None], R, axis=2).astype(np.float32)
    k = k * mask[None]
    kt, mt = torch.from_numpy(k), torch.from_numpy(mask)
    return torch.movedim(kt, 3, ro_axis + 1).contiguous(), torch.movedim(mt, 2, ro_axis).contiguous(), dc, keep


def test_padding_cut_is_found_and_broadcast_along_readout():
    k, m, dc, keep = _synth(cut_lo1=12)
    Z, info = estimate_known_zero(k, m, ro_axis=2)
    assert isinstance(info, KnownZeroInfo) and info.ellipse is None
    lo = info.cuts["axis1_lo"]
    assert lo is not None and 12 <= lo <= 15                      # first SAMPLED nonzero column is >= 12
    assert info.cuts["axis0_lo"] is None and info.cuts["axis0_hi"] is None and info.cuts["axis1_hi"] is None
    zn = Z.numpy()
    assert zn.dtype == bool and zn.shape == tuple(m.shape)
    assert zn[:, :12, :].all() and not zn[:, lo:, :].any()
    assert (zn == zn[:, :, :1]).all()                              # broadcast along readout
    assert info.n_nonzero_in_z == 0 and info.n_zero_explained == info.n_zero_lines
    assert info.frac_z == pytest.approx(lo / P1, abs=1e-9)


def test_ellipse_is_found_within_one_pixel():
    k, m, dc, keep = _synth(ellipse=(18.0, 14.0))
    Z, info = estimate_known_zero(k, m, ro_axis=2)
    assert info.ellipse is not None
    a, b, s = info.ellipse
    assert abs(a - 18.0) <= 1.0 and abs(b - 14.0) <= 1.0 and 1.0 <= s <= 1.1
    assert info.dc[:2] == dc
    assert all(v is None for v in info.cuts.values())              # an ellipse tip is not a padding cut
    ii, jj = np.meshgrid(np.arange(P0) - dc[0], np.arange(P1) - dc[1], indexing="ij")
    r = np.sqrt((ii / 18.0) ** 2 + (jj / 14.0) ** 2)
    zn = Z.numpy()[:, :, 0]
    assert zn[r >= 1.12].all() and not zn[r <= 0.98].any()
    assert info.n_nonzero_in_z == 0 and info.n_zero_explained >= 0.9 * info.n_zero_lines


def test_ellipse_plus_padding_cut_with_residue():
    k, m, dc, keep = _synth(cut_lo1=8, ellipse=(20.0, 22.0), hard=False)     # 1e-30 residue like the real data
    Z, info = estimate_known_zero(k, m, ro_axis=2)
    assert info.cuts["axis1_lo"] is not None and info.ellipse is not None
    zn = Z.numpy()[:, :, 0]
    assert zn[:, :8].all()
    sampled = m.numpy()[:, :, 0] > 0.5
    assert not zn[keep & sampled].any()                            # never a sampled nonzero line
    assert info.n_nonzero_in_z == 0 and info.frac_z > 0.3


def test_no_shutter_gives_empty_Z():
    k, m, dc, keep = _synth()
    Z, info = estimate_known_zero(k, m, ro_axis=2)
    assert not bool(Z.any()) and info.ellipse is None and all(v is None for v in info.cuts.values())
    assert info.n_zero_lines == 0 and info.frac_z == 0.0


OUTSIDE = [(24 + 20, 24), (24 - 20, 24), (24, 24 + 15), (24, 24 - 16)]    # just outside the (18, 14) ellipse about dc=(24, 24)
                                                                          # (+j capped at 15: P1=40 leaves only 15px of
                                                                          # headroom past dc1=24 on that side, vs. 23-24px
                                                                          # on the other three cardinal directions)


def test_expansion_ignores_noise_level_outliers():
    k, m, dc, keep = _synth(ellipse=(18.0, 14.0), extra_lines=OUTSIDE, extra_amp=2e-3, seed=1)   # pure noise at the fixture's
                                                                          # noise level (seed=1: seed=0's ~330 nonzero
                                                                          # sampled lines give 1980 readout-end samples,
                                                                          # just under MIN_NOISE_SAMPLES=2000)
    Z, info = estimate_known_zero(k, m, ro_axis=2)
    assert info.noise_level_used is not None and info.n_outliers >= 4 and info.n_significant_outliers == 0
    assert info.ellipse is not None and info.ellipse[2] == 1.0
    zn = Z.numpy()[:, :, 0]
    for (i, j) in OUTSIDE:
        assert not zn[i, j]                                        # a sampled nonzero line is never zeroed
    assert zn[24 + 22, 24] and zn[24 - 22, 24] and zn[24, 24 - 18]  # lines further out are still in Z (+j has no
                                                                    # room left past the offset-15 outlier; -j does)
    assert info.n_nonzero_in_z == 0


def test_expansion_respects_significant_outliers():
    k, m, dc, keep = _synth(ellipse=(18.0, 14.0), extra_lines=OUTSIDE, extra_amp=0.2)    # signal 100x the noise
    Z, info = estimate_known_zero(k, m, ro_axis=2)
    assert info.n_significant_outliers == 4 and info.ellipse is not None
    a, b, s = info.ellipse
    assert s > 1.0
    for (i, j) in OUTSIDE:
        assert ((i - 24) / (a * s)) ** 2 + ((j - 24) / (b * s)) ** 2 < 1.0   # the expanded ellipse covers them
    assert info.n_nonzero_in_z == 0


def test_uses_only_sampled_points():
    k, m, dc, keep = _synth(ellipse=(18.0, 14.0), seed=3)
    Za, a = estimate_known_zero(k, m, ro_axis=2)
    junk = torch.randn_like(k) * (1 - m).unsqueeze(0) * 10.0
    Zb, b = estimate_known_zero(k + junk, m, ro_axis=2)
    assert a.ellipse == b.ellipse and a.cuts == b.cuts and a.frac_z == b.frac_z
    assert torch.equal(Za, Zb) and a.zero_threshold == b.zero_threshold and a.dc == b.dc


def test_readout_axis_position_does_not_matter():
    k2, m2, dc, _ = _synth(cut_lo1=10, ellipse=(20.0, 22.0), seed=5, ro_axis=2)
    k0, m0, _, _ = _synth(cut_lo1=10, ellipse=(20.0, 22.0), seed=5, ro_axis=0)
    Z2, i2 = estimate_known_zero(k2, m2, ro_axis=2)
    Z0, i0 = estimate_known_zero(k0, m0, ro_axis=0)
    assert i2.cuts == i0.cuts and i2.ellipse == i0.ellipse and i2.frac_z == i0.frac_z
    assert torch.equal(torch.movedim(Z0, 0, 2), Z2)


def test_info_is_json_serializable():
    k, m, dc, keep = _synth(cut_lo1=8, ellipse=(20.0, 22.0))
    _, info = estimate_known_zero(k, m, ro_axis=2)
    json.dumps(dataclasses.asdict(info))


def _corr_noise(sigma_true, n, seed=0):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(sigma_true)
    w = (rng.normal(size=(sigma_true.shape[0], n)) + 1j * rng.normal(size=(sigma_true.shape[0], n))) / np.sqrt(2)
    return L @ w


def test_noise_cov_from_readout_ends_recovers_sigma():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(C, C)) + 1j * rng.normal(size=(C, C))
    S_true = (A @ A.conj().T + np.eye(C)) * 1e-4
    noise = _corr_noise(S_true, P0 * P1 * R).reshape(C, P0, P1, R).astype(np.complex64)
    mask = (rng.random((P0, P1)) < 0.6)[:, :, None].repeat(R, axis=2).astype(np.float32)
    k = torch.from_numpy(noise) * torch.from_numpy(mask)
    Z, info = estimate_known_zero(k, torch.from_numpy(mask), ro_axis=2)
    S, winfo = estimate_noise_cov(k, torch.from_numpy(mask), Z, info)
    assert winfo["sigma_source"] == "readout_ends" and winfo["n_noise_samples"] > MIN_NOISE_SAMPLES
    assert winfo["n_noise_samples"] == int(mask[:, :, 0].sum()) * 2 * 3      # 1/8 of R=24 is 3 indices per end
    rel = np.linalg.norm(S.numpy().astype(np.complex128) - S_true) / np.linalg.norm(S_true)
    assert rel < 0.1
    assert winfo["noise_level"] == pytest.approx(float(np.real(np.trace(S_true)) / C), rel=0.1)


def test_noise_cov_uses_only_nonzero_lines():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(C, C)) + 1j * rng.normal(size=(C, C))
    S_true = (A @ A.conj().T + np.eye(C)) * 1e-4
    noise = _corr_noise(S_true, P0 * P1 * R, seed=1).reshape(C, P0, P1, R).astype(np.complex64)
    noise[:, :, :12, :] = 0.0                                      # padding: array columns < 12 are zero lines
    mask = (rng.random((P0, P1)) < 0.6)[:, :, None].repeat(R, axis=2).astype(np.float32)
    k = torch.from_numpy(noise) * torch.from_numpy(mask)
    Z, info = estimate_known_zero(k, torch.from_numpy(mask), ro_axis=2)
    assert info.cuts["axis1_lo"] is not None
    S, winfo = estimate_noise_cov(k, torch.from_numpy(mask), Z, info)
    n_nonzero_lines = int(mask[:, 12:, 0].sum())
    assert winfo["sigma_source"] == "readout_ends" and winfo["n_noise_samples"] == n_nonzero_lines * 2 * 3
    rel = np.linalg.norm(S.numpy().astype(np.complex128) - S_true) / np.linalg.norm(S_true)
    assert rel < 0.1                                               # zero lines did not dilute Sigma


def test_padding_cut_on_the_high_side():
    k, m, dc, keep = _synth(seed=7)
    kn, mn = k.numpy().copy(), m.numpy()
    kn[:, :, 30:, :] = 0.0                                         # array columns >= 30 are zero lines
    k = torch.from_numpy(kn)
    Z, info = estimate_known_zero(k, m, ro_axis=2)
    hi = info.cuts["axis1_hi"]
    assert hi is not None and 26 <= hi <= 29 and info.ellipse is None
    zn = Z.numpy()
    assert zn[:, 30:, :].all() and not zn[:, :hi + 1, :].any() and info.n_nonzero_in_z == 0


def test_noise_cov_identity_fallback_has_uniform_keys():
    k, m, dc, keep = _synth(cut_lo1=8)
    Z, info = estimate_known_zero(k, m, ro_axis=2)
    S, winfo = estimate_noise_cov(k, m, Z, info, min_samples=10 ** 9)
    assert S is None and winfo["sigma_source"] == "identity"
    assert set(winfo) == {"sigma_source", "n_noise_samples", "coil_corr_max_offdiag", "coil_corr_mean_offdiag", "eig_ratio", "noise_level"}


def test_whitening_of_diagonal_equal_variance_is_identity():
    S = torch.eye(C, dtype=torch.complex64) * 0.3
    assert torch.equal(whitening_matrix(S), torch.eye(C, dtype=torch.complex64))


def test_whitening_decorrelates_and_keeps_mean_variance():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(C, C)) + 1j * rng.normal(size=(C, C))
    S_true = A @ A.conj().T + np.eye(C)
    W = whitening_matrix(torch.from_numpy(S_true.astype(np.complex64))).numpy().astype(np.complex128)
    out = W @ S_true @ W.conj().T
    c = np.real(np.trace(S_true)) / C
    assert np.allclose(out, c * np.eye(C), atol=1e-4 * c)


def test_build_fit_tensors_defaults_and_hard_dc_zero_on_Z():
    k, m, dc, keep = _synth(cut_lo1=8, ellipse=(20.0, 22.0), seed=2)
    csm = torch.randn(C, *m.shape, dtype=torch.complex64)
    Z, _ = estimate_known_zero(k, m, ro_axis=2)
    mask_fit, k_dc, k_fit, csm_fit = build_fit_tensors(k, m, csm, Z, None)
    assert k_fit is k_dc and csm_fit is csm
    assert torch.equal(mask_fit, torch.maximum(m, Z.float()))
    assert float(k_dc[:, Z].abs().max()) == 0.0
    pred = torch.randn_like(k)
    hard_dc = pred * (1 - mask_fit).unsqueeze(0) + k_dc
    assert float(hard_dc[:, Z].abs().max()) == 0.0
    assert torch.equal(hard_dc[:, (m > 0.5) & ~Z], k[:, (m > 0.5) & ~Z])
    assert float(mask_fit.sum()) > float(m.sum())


def test_build_fit_tensors_with_identity_W_is_exact():
    k, m, dc, keep = _synth(seed=2)
    csm = torch.randn(C, *m.shape, dtype=torch.complex64)
    Z = torch.zeros_like(m, dtype=torch.bool)
    W = torch.eye(C, dtype=torch.complex64)
    mask_fit, k_dc, k_fit, csm_fit = build_fit_tensors(k, m, csm, Z, W)
    assert torch.equal(mask_fit, m) and torch.equal(k_dc, k)
    assert torch.allclose(k_fit, k) and torch.allclose(csm_fit, csm)


def test_hard_dc_combine_is_zero_on_Z_and_keeps_measured_samples():
    from data.known_zero import hard_dc_combine
    from data.transforms import fft3c
    k, m, dc, keep = _synth(cut_lo1=8, ellipse=(20.0, 22.0), seed=2)
    csm = torch.randn(C, *m.shape, dtype=torch.complex64)
    Z, _ = estimate_known_zero(k, m, ro_axis=2)
    mask_fit, k_dc, _, _ = build_fit_tensors(k, m, csm, Z, None)
    vol = torch.randn(*m.shape, dtype=torch.complex64)
    x = hard_dc_combine(vol, csm, k_dc, mask_fit)
    assert x.shape == tuple(m.shape) and x.dtype == torch.complex64
    k_out = fft3c(vol.unsqueeze(0) * csm) * (1 - mask_fit.unsqueeze(0)) + k_dc      # the k-space the helper combined
    assert float(k_out[:, Z].abs().max()) == 0.0
    assert torch.equal(k_out[:, (m > 0.5) & ~Z], k[:, (m > 0.5) & ~Z])


# ---- explicit support (phase 3, 2026-09-12): the shutter as protocol knowledge --------------------------------

def _inside_only(k, m, keep):
    """Restrict the sampling to lines inside `keep`, so the estimator sees no zero line at all."""
    m2 = (torch.movedim(m, 2, 2)[:, :, 0] > 0.5).numpy() & keep
    mask = torch.from_numpy(np.repeat(m2[:, :, None], m.shape[2], axis=2).astype(np.float32))
    return k * mask[None], mask


def test_explicit_support_gives_Z_where_the_estimator_cannot():
    k, m, dc, keep = _synth(ellipse=(18, 14))
    k, m = _inside_only(k, m, keep)
    Z_est, info_est = estimate_known_zero(k, m, ro_axis=2)
    assert not bool(Z_est.any()) and info_est.n_zero_lines == 0          # nothing to learn the shutter from
    Z, info = known_zero_from_support(keep, k, m, ro_axis=2)
    z_pe = Z[:, :, 0].numpy()
    assert bool((Z == Z[:, :, :1]).all())                                 # broadcast along the readout
    assert np.array_equal(z_pe, ~keep)
    assert info.n_nonzero_in_z == 0 and info.n_outliers == 0 and info.ellipse is None
    assert info.cuts == {"source": "explicit_support"} and abs(info.frac_z - float((~keep).mean())) < 1e-9
    assert info.dc[:2] == dc


def test_explicit_support_never_overrides_a_measured_line():
    k, m, dc, keep = _synth(ellipse=(18, 14), extra_lines=OUTSIDE, extra_amp=0.05)
    Z, info = known_zero_from_support(keep, k, m, ro_axis=2)
    z_pe = Z[:, :, 0].numpy()
    for (i, j) in OUTSIDE:
        assert not z_pe[i, j]
    assert info.n_outliers == len(OUTSIDE) and info.n_nonzero_in_z == 0
    assert int((~keep).sum()) - len(OUTSIDE) == int(z_pe.sum())


def test_explicit_support_readout_axis_position_does_not_matter():
    for ro in (0, 1, 2):
        k, m, dc, keep = _synth(ellipse=(18, 14), ro_axis=ro)
        Z, info = known_zero_from_support(keep, k, m, ro_axis=ro)
        z_pe = torch.movedim(Z, ro, -1)[:, :, 0].numpy()
        assert np.array_equal(z_pe, ~keep)


def test_load_support_map_formats(tmp_path):
    import h5py
    keep = np.zeros((P0, P1), bool); keep[4:-4, 3:-3] = True
    np.save(tmp_path / "s.npy", keep)
    with h5py.File(tmp_path / "s.h5", "w") as h:
        h.create_dataset("support", data=keep.astype(np.uint8))
    with h5py.File(tmp_path / "m.h5", "w") as h:
        h.create_dataset("mask", data=keep.astype(np.float32))
    for f in ("s.npy", "s.h5", "m.h5"):
        assert np.array_equal(load_support_map(tmp_path / f), keep)
    np.save(tmp_path / "bad.npy", np.zeros((2, 3, 4)))
    with pytest.raises(AssertionError):
        load_support_map(str(tmp_path / "bad.npy"))


def test_explicit_support_fit_tensors_match_estimator_path():
    """With the ORIGINAL (rectangle) sampling the estimator finds the ellipse itself; the explicit map of the
    same ellipse must give the same fit-side tensors up to the lines the estimator's safety expansion adds."""
    k, m, dc, keep = _synth(ellipse=(18, 14))
    Z_est, _ = estimate_known_zero(k, m, ro_axis=2)
    Z_exp, _ = known_zero_from_support(keep, k, m, ro_axis=2)
    both = Z_est & Z_exp
    assert float(both.float().sum() / Z_exp.float().sum()) > 0.9          # the estimator recovers >90 % of the true Z
