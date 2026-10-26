"""Unit-style sanity tests for the lesion metrics module.

Synthetic 3D phantom: uniform background with a Gaussian-blob "lesion" at the
centre of a known size. ``pre`` lacks the lesion, ``post`` adds it. We check:

  * lesion ROI is auto-detected at the volume centre,
  * lbc_post > lbc_pre by approximately the synthesized contrast,
  * er_max approximately equals contrast / background,
  * FWHM of a line profile through the blob equals 2.355 * sigma to within 10%.

Run via:
    /home/congcongliu/anaconda3/envs/vediffusion0328/bin/python \
        -m unittest 3DGS/3dgsVC/metrics/test_lesion_metrics.py
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)  # 3dgsVC/
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from metrics.lesion_metrics import (  # noqa: E402
    auto_detect_lesion_roi,
    compute_enhancement_ratio,
    compute_fwhm_line_profile,
    compute_lesion_cnr,
    compute_lesion_metrics_bundle,
    compute_lesion_snr,
    compute_lesion_snr_db,
)


def _make_phantom(
    shape=(64, 64, 64),
    bg_value=1.0,
    bg_noise_std=0.02,
    lesion_amp=2.0,
    lesion_sigma=4.0,
    seed=0,
):
    """Build (pre, post) volumes with a Gaussian blob added in post."""
    rng = np.random.default_rng(seed)
    pre = bg_value + rng.normal(0.0, bg_noise_std, size=shape).astype(np.float64)
    centre = tuple(s // 2 for s in shape)
    grids = np.meshgrid(
        *[np.arange(s, dtype=np.float64) - centre[i] for i, s in enumerate(shape)],
        indexing="ij",
    )
    r2 = sum(g * g for g in grids)
    blob = lesion_amp * np.exp(-r2 / (2.0 * lesion_sigma ** 2))
    post = pre + blob
    return pre.astype(np.float64), post.astype(np.float64), centre, blob


class LesionMetricsTests(unittest.TestCase):
    def setUp(self):
        self.shape = (48, 48, 48)
        self.bg_value = 1.0
        self.bg_noise = 0.02
        self.lesion_amp = 2.0
        self.lesion_sigma = 3.0
        self.pre, self.post, self.centre, self.blob = _make_phantom(
            shape=self.shape,
            bg_value=self.bg_value,
            bg_noise_std=self.bg_noise,
            lesion_amp=self.lesion_amp,
            lesion_sigma=self.lesion_sigma,
        )

    # ------------------------------------------------------------------
    def test_auto_detect_centroid(self):
        """Auto-detected lesion centroid should be within 2 voxels of true centre."""
        det = auto_detect_lesion_roi(
            self.post,
            self.pre,
            diff_quantile=0.97,
            bbox_size=(8, 8, 8),
            smooth_sigma=1.0,
        )
        c = det["centroid"]
        for axis, (got, true_) in enumerate(zip(c, self.centre)):
            self.assertLessEqual(
                abs(got - true_),
                2,
                msg=f"axis {axis}: detected {got}, true {true_}",
            )
        # the largest connected component must contain the true centre
        comp = det["component_mask"]
        self.assertTrue(comp[self.centre], msg="true centre not in component_mask")

    # ------------------------------------------------------------------
    def test_enhancement_ratio_max(self):
        """E_R at the lesion peak should be approximately lesion_amp / bg."""
        er = compute_enhancement_ratio(self.pre, self.post)
        # E_R at exact centre (where blob == lesion_amp)
        er_centre = float(er[self.centre])
        expected = self.lesion_amp / self.bg_value
        # small tolerance for noise floor / quantile floor
        self.assertAlmostEqual(er_centre, expected, delta=0.1)
        # er_max over the whole volume should be roughly the centre value
        self.assertGreater(er.max(), expected * 0.9)
        self.assertLess(er.max(), expected * 1.2)

    # ------------------------------------------------------------------
    def test_fwhm_matches_sigma(self):
        """FWHM of the blob line profile should be 2.355 * sigma within 10%."""
        # draw a long line through the centre along axis 0
        c = self.centre
        half = int(8 * self.lesion_sigma)
        p0 = (max(0, c[0] - half), c[1], c[2])
        p1 = (min(self.shape[0] - 1, c[0] + half), c[1], c[2])
        # use enough samples that step ~ 1 voxel
        n = (p1[0] - p0[0]) + 1
        # Profile measured on the post image will sit on top of bg; subtract bg
        # so half-max corresponds to half of the blob amplitude.
        fwhm_res = compute_fwhm_line_profile(
            self.post - self.bg_value, p0, p1, num_samples=int(n * 4)
        )
        fwhm_world = fwhm_res["fwhm_world"]
        expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * self.lesion_sigma  # 2.355*sigma
        self.assertTrue(np.isfinite(fwhm_world), msg=f"FWHM not finite: {fwhm_res}")
        rel = abs(fwhm_world - expected) / expected
        self.assertLess(
            rel,
            0.10,
            msg=f"FWHM rel error {rel:.3f}: got {fwhm_world:.3f}, expected {expected:.3f}",
        )

    # ------------------------------------------------------------------
    def test_lbc_post_greater_than_pre(self):
        """LBC should jump from ~0 (pre) to a known positive value (post)."""
        bundle = compute_lesion_metrics_bundle(
            self.pre,
            self.post,
            diff_quantile=0.97,
            bbox_size=(6, 6, 6),
            smooth_sigma=1.0,
        )
        self.assertTrue(np.isfinite(bundle["lbc_pre"]))
        self.assertTrue(np.isfinite(bundle["lbc_post"]))
        # post LBC should be positive (lesion brighter); pre LBC near 0
        self.assertGreater(bundle["lbc_post"], 0.2)
        self.assertLess(abs(bundle["lbc_pre"]), 0.05)
        # delta must be positive and non-trivial
        self.assertGreater(bundle["delta_lbc"], 0.2)

    # ------------------------------------------------------------------
    def test_lesion_cnr_with_supplied_masks(self):
        """compute_lesion_cnr with supplied masks gives plausible values."""
        c = self.centre
        # Spherical mask of radius ~ sigma around centre
        grids = np.indices(self.shape)
        r2 = sum((grids[i] - c[i]) ** 2 for i in range(3))
        lesion = r2 <= int(self.lesion_sigma ** 2)
        # background = far corner cube (where blob is negligible)
        bg = np.zeros(self.shape, dtype=bool)
        bg[:8, :8, :8] = True
        cnr_pre = compute_lesion_cnr(self.pre, lesion, bg)
        cnr_post = compute_lesion_cnr(self.post, lesion, bg)
        # post CNR should be much larger than pre (which is essentially noise)
        self.assertGreater(cnr_post, 5 * max(cnr_pre, 0.5))

    # ------------------------------------------------------------------
    def test_bundle_returns_arrays_and_scalars(self):
        bundle = compute_lesion_metrics_bundle(self.pre, self.post)
        for key in (
            "lesion_cnr_pre",
            "lesion_cnr_post",
            "lesion_snr_pre",
            "lesion_snr_post",
            "lesion_snr_pre_db",
            "lesion_snr_post_db",
            "delta_lesion_snr_db",
            "lbc_pre",
            "lbc_post",
            "er_mean",
            "er_max",
            "er_p99",
            "fwhm_main",
            "diff_map",
            "er_map",
            "lesion_mask",
            "bg_mask",
        ):
            self.assertIn(key, bundle, msg=f"missing key {key}")
        self.assertEqual(bundle["diff_map"].shape, self.shape)
        self.assertEqual(bundle["er_map"].shape, self.shape)
        self.assertEqual(bundle["lesion_mask"].dtype, np.bool_)
        self.assertGreater(bundle["n_lesion_voxels"], 0)
        self.assertGreater(bundle["n_bg_voxels"], 0)

    # ------------------------------------------------------------------
    def test_lesion_snr_synthetic(self):
        """Synthetic lesion+noise with amp=2 / bg=1 / noise_sigma=0.1 must
        give SNR_lesion > 5 (sig ~ 3, noise std ~ 0.1, so SNR ~ 30)."""
        shape = (40, 40, 40)
        bg = 1.0
        noise_sigma = 0.1
        lesion_amp = 2.0  # added on top of bg → lesion mean ~ 3.0
        rng = np.random.default_rng(123)
        vol_pre = bg + rng.normal(0.0, noise_sigma, size=shape)
        c = tuple(s // 2 for s in shape)
        grids = np.indices(shape)
        r2 = sum((grids[i] - c[i]) ** 2 for i in range(3))
        # spherical lesion of radius 4
        lesion_mask = r2 <= 4 ** 2
        vol_post = vol_pre.copy()
        vol_post[lesion_mask] = vol_post[lesion_mask] + lesion_amp

        # background = far corner cube (no lesion overlap, contains noise)
        bg_mask = np.zeros(shape, dtype=bool)
        bg_mask[:8, :8, :8] = True
        # ensure no overlap with lesion (centre is far from corner)
        self.assertFalse(np.any(bg_mask & lesion_mask))

        snr = compute_lesion_snr(vol_post, lesion_mask, bg_mask)
        self.assertTrue(np.isfinite(snr), msg=f"SNR not finite: {snr}")
        self.assertGreater(
            snr,
            5.0,
            msg=f"SNR_lesion {snr:.2f} should be > 5 for amp=2/bg=1/sigma=0.1",
        )
        # dB form should also be finite and > 20*log10(5) ≈ 13.98 dB
        snr_db = compute_lesion_snr_db(vol_post, lesion_mask, bg_mask)
        self.assertTrue(np.isfinite(snr_db))
        self.assertGreater(snr_db, 20.0 * np.log10(5.0))


if __name__ == "__main__":
    unittest.main()
