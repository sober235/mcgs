"""CPU contracts for a shared positive field with no source-amplitude shortcut."""

import copy
import json

import pytest
import torch
import torch.nn.functional as F


def source(n=5, requires_grad=False):
    xyz = torch.tensor([[0., 0., 0.], [0., 0., 0.], [.2, -.3, .4],
                        [1.2, .1, -.2], [-.4, .5, .6]], dtype=torch.float64)[:n]
    logs = torch.tensor([[0., 1., 2.], [0., 1., 2.], [2., 3., 2.],
                         [4., 5., 2.], [6., 7., 2.]], dtype=torch.float64)[:n]
    plain = torch.tensor([1, 3j, -5, -7j, 9], dtype=torch.complex128)[:n]
    contrast = torch.tensor([2j, -4, -6j, 8, 10j], dtype=torch.complex128)[:n]
    return tuple(t.requires_grad_(requires_grad) for t in (xyz, logs, plain, contrast))


def make(encoder="mlp", chunk_size=2, **kwargs):
    from full_amplitude import FullAmplitudeField
    torch.manual_seed(49)
    return FullAmplitudeField(*source(), encoder=encoder, chunk_size=chunk_size, **kwargs)


@pytest.mark.parametrize("encoder", ["mlp", "grid"])
def test_initial_moduli_are_global_medians_and_duplicate_queries_remain_equal(encoder):
    # A retained per-point modulus or gain breaks both the medians and duplicates.
    field = make(encoder)
    torch.testing.assert_close(field.magnitudes(), torch.tensor([[5., 6.]] * 5).double())
    optimizer = torch.optim.Adam(field.amplitude_parameters(), lr=.01)
    for _ in range(3):
        optimizer.zero_grad()
        loss = (field.magnitudes() - torch.tensor([[1., 2.], [9., 12.], [3., 5.],
                                                  [7., 4.], [8., 9.]])).square().mean()
        loss.backward()
        optimizer.step()
    predicted = field.magnitudes()
    assert torch.equal(predicted[0], predicted[1])
    assert torch.all(predicted > 0)
    assert not torch.allclose(predicted, torch.tensor([[5., 6.]] * 5).double())
    assert field.phase_delta.requires_grad is False
    assert torch.count_nonzero(field.phase_delta) == 0
    assert all(p is not field.phase_delta for p in field.amplitude_parameters())
    assert {id(p) for p in field.amplitude_parameters()} == {
        id(p) for name, p in field.named_parameters() if name != "phase_delta"
    }


@pytest.mark.parametrize("encoder", ["mlp", "grid"])
def test_density_uses_source_unit_phases_and_independent_phase_gradients(encoder):
    field = make(encoder)
    field.phase_delta.requires_grad_(True)
    with torch.no_grad():
        field.phase_delta[:, 0] = .3
        field.phase_delta[:, 1] = -.2
    expected = torch.tensor([5, 5j, -5, -5j, 5], dtype=torch.complex128)
    expected *= torch.exp(torch.tensor(.3j, dtype=torch.complex128))
    actual = field.get_density("plain")
    torch.testing.assert_close(actual, expected)
    weights = torch.tensor([1+2j, 2-3j, -1+4j, 3+1j, 2j], dtype=torch.complex128)
    (weights.conj() * actual).real.sum().backward()
    torch.testing.assert_close(field.phase_delta.grad[:, 0], -(weights.conj() * expected).imag)
    assert torch.count_nonzero(field.phase_delta.grad[:, 1]) == 0
    assert torch.count_nonzero(field.network[-1].bias.grad[1]) == 0
    field.zero_grad()
    field.get_density("contrast").real.sum().backward()
    assert torch.count_nonzero(field.phase_delta.grad[:, 0]) == 0
    assert torch.count_nonzero(field.phase_delta.grad[:, 1]) > 0


@pytest.mark.parametrize("encoder", ["mlp", "grid"])
def test_inputs_are_private_frozen_copies_without_per_point_modulus_buffers(encoder):
    from full_amplitude import FullAmplitudeField
    inputs = source(requires_grad=True)
    field = FullAmplitudeField(*inputs, encoder=encoder)
    before = field.get_density("plain").detach().clone()
    with torch.no_grad():
        for value in inputs:
            value.add_(10)
    torch.testing.assert_close(field.get_density("plain"), before, rtol=0, atol=0)
    field.magnitudes().sum().backward()
    assert all(t.grad is None for t in inputs)
    assert all(not b.requires_grad for b in field.buffers())
    # All per-source complex state carries unit phase, never original moduli.
    for b in field.buffers():
        if b.is_complex():
            torch.testing.assert_close(b.abs(), torch.ones_like(b.real))
    assert not any(p.shape == (5, 2) for p in field.amplitude_parameters())


@pytest.mark.parametrize("encoder", ["mlp", "grid"])
def test_chunked_selected_outputs_and_all_parameter_gradients_match_full(encoder):
    field = make(encoder, chunk_size=2)
    # A nonzero readout makes encoder gradients observable immediately.
    with torch.no_grad():
        field.network[-1].weight.normal_(0, .1)
    full = make(encoder, chunk_size=100)
    full.load_state_dict(copy.deepcopy(field.state_dict()))
    indices = torch.tensor([4, 0, 4, 2, 1])
    selected = field.magnitudes(indices)
    torch.testing.assert_close(selected, full.magnitudes()[indices], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(field.get_density("contrast", indices), full.get_density("contrast")[indices])
    weights = torch.tensor([[1., 2.], [3., 1.], [2., 4.], [-1., 2.], [.5, 3.]])
    (selected * weights).sum().backward()
    (full.magnitudes(indices) * weights).sum().backward()
    for left, right in zip(field.amplitude_parameters(), full.amplitude_parameters()):
        assert left.grad is not None
        assert torch.isfinite(left.grad).all()
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-10, atol=1e-10)
    assert all(p.grad.abs().sum() > 0 for p in field.amplitude_parameters())
    with torch.no_grad():
        torch.testing.assert_close(field.magnitudes(indices), full.magnitudes(indices))


@pytest.mark.parametrize("encoder", ["mlp", "grid"])
def test_state_roundtrip_and_metadata_describe_complete_source(encoder):
    field = make(encoder)
    with torch.no_grad():
        field.network[-1].bias.add_(.2)
        field.phase_delta[:, 1].add_(.1)
    restored = make(encoder, chunk_size=3)
    restored.load_state_dict(copy.deepcopy(field.state_dict()))
    torch.testing.assert_close(restored.get_density("contrast"), field.get_density("contrast"))
    metadata = json.loads(json.dumps(field.metadata(), allow_nan=False))
    assert metadata["encoder"] == encoder
    assert metadata["num_points"] == 5
    assert metadata["domain_outside_fraction"] == .2
    torch.testing.assert_close(field.magnitudes(torch.tensor([0]))[0], field.magnitudes()[0])


def test_single_source_zero_coefficients_are_finite_positive_with_unit_phase():
    from full_amplitude import FullAmplitudeField
    xyz, logs, plain, contrast = source(n=1)
    field = FullAmplitudeField(xyz, logs, plain * 0, contrast * 0)
    expected = torch.full((1, 2), 1e-12, dtype=torch.float64)
    torch.testing.assert_close(field.magnitudes(), expected, rtol=1e-12, atol=0)
    torch.testing.assert_close(field.get_density("plain"), expected[:, 0].to(torch.complex128))
    field.magnitudes().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in field.amplitude_parameters())


@pytest.mark.parametrize("feature,expected", [(0, .2), (3, .8), (6, 0.), (9, -1.)])
def test_mlp_uses_original_xyz_fourier_and_complete_source_standardized_logscales(feature, expected):
    # A one-feature path exposes wrong coordinate units, frequency, or log stats.
    from full_amplitude import FullAmplitudeField
    xyz = torch.tensor([[.2, 0., 0.], [.4, .5, .6]], dtype=torch.float64)
    logs = torch.tensor([[1., 2., 4.], [3., 6., 4.]], dtype=torch.float64)
    field = FullAmplitudeField(xyz, logs, torch.ones(2, dtype=torch.complex128),
                               torch.ones(2, dtype=torch.complex128), bands=1, width=1)
    with torch.no_grad():
        for p in field.amplitude_parameters():
            p.zero_()
        field.network[0].weight[0, feature] = 1
        field.network[2].weight[0, 0] = 1
        field.network[4].weight[0, 0] = 1
    if feature == 3:
        expected = .5877852522924731  # sin(pi * .2)
    if feature == 6:
        expected = .8090169943749475  # cos(pi * .2)
    want = F.softplus(F.silu(F.silu(torch.tensor(expected, dtype=torch.float64))))
    torch.testing.assert_close(field.magnitudes(torch.tensor([0]))[0, 0], want)


def test_grid_trilinear_coordinates_use_xyz_order_and_declared_domain():
    from full_amplitude import FullAmplitudeField
    xyz = torch.tensor([[-.55, .55, 0.], [.55, -.55, 0.], [2.2, 0., 0.]], dtype=torch.float64)
    field = FullAmplitudeField(xyz, torch.zeros_like(xyz), torch.ones(3, dtype=torch.complex128),
                               torch.ones(3, dtype=torch.complex128), encoder="grid", width=1)
    with torch.no_grad():
        for p in field.amplitude_parameters():
            p.zero_()
        # Only the first grid channel varies from -1 to +1 along x (last axis).
        field.grids[0][0, 0] = torch.linspace(-1, 1, 8).double()[None, None, :]
        field.network[0].weight[0, 6] = 1
        field.network[2].weight[0, 0] = 1
        field.network[4].weight[0, 0] = 1
    want = F.softplus(F.silu(F.silu(torch.tensor([-.5, .5, 1.], dtype=torch.float64))))
    torch.testing.assert_close(field.magnitudes()[:, 0], want)


@pytest.mark.parametrize("change", ["empty", "xyz_shape", "logs_shape", "complex_shape", "integer_xyz",
                                     "integer_logs", "real_coefficient", "mixed_dtype", "nan_xyz",
                                     "inf_logs", "nan_complex"])
def test_invalid_source_is_rejected_before_evaluation(change):
    from full_amplitude import FullAmplitudeField
    args = list(source())
    if change == "empty":
        args = [a[:0] for a in args]
    elif change == "xyz_shape":
        args[0] = args[0][:, :2]
    elif change == "logs_shape":
        args[1] = args[1][:-1]
    elif change == "complex_shape":
        args[2] = args[2][:, None]
    elif change == "integer_xyz":
        args[0] = args[0].long()
    elif change == "integer_logs":
        args[1] = args[1].long()
    elif change == "real_coefficient":
        args[2] = args[2].real
    elif change == "mixed_dtype":
        args[1] = args[1].float()
    else:
        index = {"nan_xyz": 0, "inf_logs": 1, "nan_complex": 2}[change]
        args[index].reshape(-1)[0] = float("inf") if change == "inf_logs" else float("nan")
    with pytest.raises((TypeError, ValueError)):
        FullAmplitudeField(*args)


@pytest.mark.parametrize("options", [{"encoder": "hash"}, {"bands": -1}, {"bands": True},
                                     {"width": 0}, {"chunk_size": 0}, {"chunk_size": 1.5}])
def test_invalid_configuration_is_rejected(options):
    with pytest.raises((TypeError, ValueError)):
        make(**options)


@pytest.mark.parametrize("indices", [torch.tensor([.5]), torch.tensor([[0]]), torch.tensor([-1]),
                                    torch.tensor([5]), torch.tensor([True, False])])
def test_invalid_point_selection_is_rejected(indices):
    with pytest.raises((TypeError, ValueError, IndexError)):
        make().magnitudes(indices)


@pytest.mark.parametrize("encoder", ["mlp", "grid"])
def test_empty_point_selection_has_correct_shapes(encoder):
    field = make(encoder)
    indices = torch.tensor([], dtype=torch.long)
    assert field.magnitudes(indices).shape == (0, 2)
    assert field.get_density("plain", indices).shape == (0,)
    assert field.get_density("plain", indices).dtype == torch.complex128
    with pytest.raises(ValueError):
        field.get_density("post")


@pytest.mark.parametrize("encoder", ["mlp", "grid"])
@pytest.mark.parametrize("channel", [0, 1])
@pytest.mark.parametrize("real_dtype,complex_dtype,component", [
    (torch.float32, torch.complex64, 3e38),
    (torch.float64, torch.complex128, 1.5e308),
])
def test_finite_components_with_unrepresentable_modulus_are_rejected(
    encoder, channel, real_dtype, complex_dtype, component,
):
    # A single overflowing modulus invalidates its unit phase even when the
    # channel's median modulus remains finite; checking only the scale misses it.
    from full_amplitude import FullAmplitudeField
    xyz = torch.zeros(3, 3, dtype=real_dtype)
    coefficients = [torch.ones(3, dtype=complex_dtype), torch.ones(3, dtype=complex_dtype)]
    coefficients[channel][0] = complex(component, component)
    assert torch.isfinite(coefficients[channel]).all()
    with pytest.raises(ValueError):
        FullAmplitudeField(xyz, xyz, *coefficients, encoder=encoder)
