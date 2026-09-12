import numpy as np
import pytest

from cardiac_fr.physio import (dwell_asymmetry, cardiac_phase, expiration_end_by_histogram, per_spoke, spoke_table,
                               third_at_end, timestamp_mismatch)


def test_timestamp_mismatch_zero_and_shape_check():
    ticks = np.array([1000, 1001, 1001, 1003], dtype=np.int64)
    assert timestamp_mismatch(np.array([0, 2.5, 2.5, 7.5]), ticks) == 0.0
    assert timestamp_mismatch(np.array([0, 2.5, 5.0, 7.5]), ticks) == 2.5
    with pytest.raises(ValueError):
        timestamp_mismatch(np.zeros(3), ticks)


def test_per_spoke_mean_over_echoes():
    np.testing.assert_allclose(per_spoke(np.arange(6.0), n_echo=3), [1.0, 4.0])


def test_cardiac_phase():
    ph = cardiac_phase(np.array([50, 100, 150, 300, 400, 500.0]), np.array([100, 200, 400.0]))
    np.testing.assert_allclose(ph, [np.nan, 0.0, 0.5, 0.5, np.nan, np.nan])


def test_expiration_end_by_histogram():
    rng = np.random.default_rng(0)
    plateau_high = np.r_[1 + 0.01 * rng.standard_normal(700), np.linspace(0, 1, 300)]
    assert expiration_end_by_histogram(plateau_high) == "high"
    assert expiration_end_by_histogram(-plateau_high) == "low"


def test_third_at_end_picks_extremes():
    v = np.array([5, 1, 4, 2, 6, 3.0])
    assert third_at_end(v, "high").nonzero()[0].tolist() == [0, 4]
    assert third_at_end(v, "low").nonzero()[0].tolist() == [1, 3]


def test_spoke_table_flags():
    ph = {"TimeStamp_ms": np.repeat(np.arange(6) * 5000.0, 2),
          "respSignal": np.repeat(np.arange(6.0), 2), "cardTriggers": np.array([0.0, 100000.0])}
    t = spoke_table(ph, np.arange(6), n_echo=2, n_seg=3, t_steady_ms=10000)
    assert t["is_si"].tolist() == [True, False, False, True, False, False]
    assert t["steady"].tolist() == [False, False, True, True, True, True]
    assert t["shot"].tolist() == [0, 0, 0, 1, 1, 1]
    np.testing.assert_allclose(t["resp"], np.arange(6.0))
    np.testing.assert_allclose(t["card_phase"], np.arange(6) * 0.05)


def test_dwell_asymmetry_points_at_the_plateau_end():
    t = np.linspace(0, 40, 4000)
    # breathing with a long pause at the top of the trace (end-expiration = high end)
    plateau_high = np.clip(np.sin(2 * np.pi * t / 4.0) * 2.0, -2.0, 0.6)
    assert dwell_asymmetry(plateau_high) >= 2
    assert dwell_asymmetry(-plateau_high) <= 0.5
    assert 0.5 < dwell_asymmetry(np.sin(2 * np.pi * t / 4.0)) < 2
