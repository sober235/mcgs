"""Per-contrast strength for the geometry (TV) term (paired_gs.losses.edge_tv.geo_term)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
from losses.edge_tv import geo_term  # noqa: E402


def test_equal_weights_reproduce_the_single_lambda_term():
    tv_p, tv_c = torch.tensor(3.0), torch.tensor(5.0)
    added, reported = geo_term(tv_p, tv_c, 0.1, 0.1, 0.1)
    assert float(added) == pytest.approx(0.1 * 8.0)
    assert float(reported) == pytest.approx(8.0)


def test_each_channel_carries_its_own_strength():
    tv_p, tv_c = torch.tensor(3.0), torch.tensor(5.0)
    added, _ = geo_term(tv_p, tv_c, 0.02, 0.10, 0.10)
    assert float(added) == pytest.approx(0.02 * 3.0 + 0.10 * 5.0)


def test_the_reported_value_still_means_lambda_times_it():
    """The trainer logs lambda_geo * R_geo, so R_geo must stay that quotient when the weights differ."""
    tv_p, tv_c = torch.tensor(3.0), torch.tensor(5.0)
    added, reported = geo_term(tv_p, tv_c, 0.02, 0.10, 0.10)
    assert float(0.10 * reported) == pytest.approx(float(added))


def test_a_zero_weight_switches_its_channel_off():
    tv_p, tv_c = torch.tensor(3.0), torch.tensor(5.0)
    added, _ = geo_term(tv_p, tv_c, 0.0, 0.10, 0.10)
    assert float(added) == pytest.approx(0.10 * 5.0)
