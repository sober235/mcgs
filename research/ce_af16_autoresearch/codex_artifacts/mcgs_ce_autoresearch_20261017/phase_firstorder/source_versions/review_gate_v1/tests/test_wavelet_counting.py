"""Count actual batched forward transforms without changing prior arithmetic."""
import torch

import surrogates


def test_prior_counter_matches_real_transform_calls_including_construction(monkeypatch):
    actual = surrogates.wavelet_details
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return actual(*args, **kwargs)
    monkeypatch.setattr(surrogates, "wavelet_details", counted)
    torch.manual_seed(4)
    x = torch.randn(2, 6, 6, 6, dtype=torch.complex128)
    prior = surrogates.build_frozen_prior(x, lambda_tv=0., lambda_phase=0.,
                                         lambda_wavelet=(.12, .1), kappa=1.)
    assert prior.wavelet_transform_batches == len(calls) == 3
    prior.merit(x)
    assert prior.wavelet_transform_batches == len(calls) == 4
    prior.quadratic(x)
    assert prior.wavelet_transform_batches == len(calls) == 5


def test_disabled_wavelet_executes_and_reports_no_transform(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled wavelet transformed an image")
    monkeypatch.setattr(surrogates, "wavelet_details", forbidden)
    x = torch.ones(2, 3, 3, 3, dtype=torch.complex64)
    prior = surrogates.build_frozen_prior(x, lambda_tv=.1, lambda_phase=.03,
                                         lambda_wavelet=0., kappa=1.)
    prior.merit(x)
    prior.quadratic(x)
    assert prior.wavelet_transform_batches == 0
