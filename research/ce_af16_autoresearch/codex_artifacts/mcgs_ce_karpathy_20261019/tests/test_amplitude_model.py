"""CPU regression checks for the residual complex amplitude parameterization."""

import copy
import json

import pytest
import torch

from amplitude_model import ResidualAmplitudeField


def inputs(n=11, requires_grad=False):
    generator = torch.Generator().manual_seed(28)
    xyz = torch.randn(n, 3, dtype=torch.float64, generator=generator)
    plain = torch.randn(n, dtype=torch.complex128, generator=generator)
    contrast = torch.randn(n, dtype=torch.complex128, generator=generator)
    return tuple(t.requires_grad_(requires_grad) for t in (xyz, plain, contrast))


def model(mode="shared", n=11, chunk_size=4, requires_grad=False):
    torch.manual_seed(91)
    return ResidualAmplitudeField(
        *inputs(n, requires_grad), mode=mode, bands=2, width=7,
        gain_bound=0.25, chunk_size=chunk_size,
    )


@pytest.mark.parametrize("mode", ["shared", "explicit"])
def test_zero_residual_is_elementwise_exact_identity(mode):
    field = model(mode)
    plain, contrast = inputs()[1:]
    assert torch.equal(field.get_density("plain"), plain)
    assert torch.equal(field.get_density("contrast"), contrast)
    assert torch.count_nonzero(field.phase_delta) == 0
    if mode == "shared":
        assert torch.count_nonzero(field.network[-1].weight) == 0
        assert torch.count_nonzero(field.network[-1].bias) == 0


@pytest.mark.parametrize("mode", ["shared", "explicit"])
def test_nonzero_gain_and_phase_match_analytic_derivatives(mode):
    field = model(mode)
    raw = 0.37
    phase = -0.21
    with torch.no_grad():
        field.phase_delta[:, 0] = phase
        if mode == "shared":
            field.network[-1].bias[0] = raw
        else:
            field.log_gain[:, 0] = raw
    weight = inputs()[2]
    output = field.get_density("plain")
    gain = field.gain_bound * torch.tanh(torch.tensor(raw, dtype=torch.float64))
    expected = field.plain_complex * torch.exp(gain) * torch.exp(
        torch.tensor(phase * 1j, dtype=torch.complex128)
    )
    torch.testing.assert_close(output, expected, rtol=1e-13, atol=1e-13)
    (weight.conj() * output).real.sum().backward()
    product = weight.conj() * expected
    torch.testing.assert_close(field.phase_delta.grad[:, 0], -product.imag)
    derivative = field.gain_bound * (1 - torch.tanh(torch.tensor(raw)) ** 2)
    expected_gain_grad = derivative.double() * product.real
    if mode == "shared":
        torch.testing.assert_close(
            field.network[-1].bias.grad[0], expected_gain_grad.sum(),
            rtol=1e-7, atol=1e-10,
        )
    else:
        torch.testing.assert_close(
            field.log_gain.grad[:, 0], expected_gain_grad, rtol=1e-7, atol=1e-10,
        )


@pytest.mark.parametrize("mode", ["shared", "explicit"])
def test_single_contrast_changes_and_gradients_are_isolated(mode):
    field = model(mode)
    with torch.no_grad():
        field.phase_delta[:, 1] = 0.2
        if mode == "shared":
            field.network[-1].bias[1] = 0.4
        else:
            field.log_gain[:, 1] = 0.4
    assert torch.equal(field.get_density("plain"), field.plain_complex)
    assert not torch.equal(field.get_density("contrast"), field.contrast_complex)
    field.get_density("plain").real.sum().backward()
    assert torch.count_nonzero(field.phase_delta.grad[:, 1]) == 0
    gain_gradient = (
        field.network[-1].bias.grad if mode == "shared" else field.log_gain.grad
    )
    assert torch.count_nonzero(gain_gradient[..., 1]) == 0


@pytest.mark.parametrize("mode", ["shared", "explicit"])
def test_input_coordinates_and_complex_coefficients_are_detached_copies(mode):
    xyz, plain, contrast = inputs(requires_grad=True)
    field = ResidualAmplitudeField(
        xyz, plain, contrast, mode=mode, bands=2, width=7, chunk_size=4,
    )
    saved_xyz = xyz.detach().clone()
    saved_plain = plain.detach().clone()
    with torch.no_grad():
        xyz.add_(20)
        plain.mul_(2)
    assert torch.equal(field.positions, saved_xyz)
    assert torch.equal(field.plain_complex, saved_plain)
    field.get_density("contrast").abs().sum().backward()
    assert xyz.grad is None
    assert plain.grad is None
    assert contrast.grad is None
    assert all(not buffer.requires_grad for buffer in field.buffers())
    assert "positions" not in dict(field.named_parameters())
    assert "plain_complex" not in dict(field.named_parameters())
    assert "contrast_complex" not in dict(field.named_parameters())


@pytest.mark.parametrize("mode", ["shared", "explicit"])
def test_gain_bound_limits_amplitude_ratios(mode):
    field = model(mode)
    with torch.no_grad():
        field.phase_delta.uniform_(-3, 3)
        if mode == "shared":
            field.network[-1].bias.copy_(torch.tensor([100.0, -100.0]))
        else:
            field.log_gain[:, 0] = 100
            field.log_gain[:, 1] = -100
    for channel, sign in [("plain", 1), ("contrast", -1)]:
        base = getattr(field, f"{channel}_complex")
        ratio = field.get_density(channel).abs() / base.abs()
        expected = torch.full_like(ratio, torch.exp(torch.tensor(sign * 0.25)))
        torch.testing.assert_close(ratio, expected, rtol=1e-7, atol=1e-10)


def test_chunked_checkpoint_matches_single_chunk_values_and_all_gradients():
    chunked = model(n=17, chunk_size=3)
    with torch.no_grad():
        chunked.network[-1].weight.normal_(std=0.1)
        chunked.network[-1].bias.uniform_(-0.2, 0.2)
        chunked.phase_delta.uniform_(-0.2, 0.2)
    whole = model(n=17, chunk_size=100)
    whole.load_state_dict(chunked.state_dict())
    weight = inputs(17)[2]
    for field in (chunked, whole):
        loss = sum(
            ((weight.conj() * field.get_density(channel)).real ** 2).sum()
            for channel in ("plain", "contrast")
        )
        loss.backward()
    for channel in ("plain", "contrast"):
        torch.testing.assert_close(chunked.get_density(channel), whole.get_density(channel))
    for (name, actual), (_, expected) in zip(
        chunked.named_parameters(), whole.named_parameters(), strict=True,
    ):
        assert actual.grad is not None, name
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-12, atol=1e-12)
        assert torch.count_nonzero(actual.grad) > 0, name


def test_checkpoint_trains_shared_network_without_any_trainable_input_tensor():
    field = model()
    field.phase_delta.requires_grad_(False)
    field.get_density("plain").abs().sum().backward()
    assert field.network[-1].bias.grad[0] > 0
    assert field.network[-1].bias.grad[1] == 0


def test_checkpoint_does_not_save_full_field_mlp_activations():
    field = model(n=101, chunk_size=4)
    saved_shapes = []

    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = field.get_density("plain")
    # The returned [N] density is required by the renderer, but no saved
    # Fourier/hidden-layer activation should scale with all N points.
    assert saved_shapes
    assert all(not shape or shape[0] <= field.chunk_size for shape in saved_shapes)
    output.real.sum().backward()
    assert field.network[-1].bias.grad is not None


@pytest.mark.parametrize("mode", ["shared", "explicit"])
def test_state_dict_roundtrip_and_named_optimizer_groups(mode):
    field = model(mode)
    optimizer = torch.optim.Adam(field.parameter_groups(lr=0.001))
    sum(field.get_density(c).real.sum() for c in ("plain", "contrast")).backward()
    optimizer.step()
    restored = model(mode)
    restored.load_state_dict(copy.deepcopy(field.state_dict()))
    for channel in ("plain", "contrast"):
        assert torch.equal(field.get_density(channel), restored.get_density(channel))
    metadata = json.loads(json.dumps(field.metadata()))
    assert metadata["mode"] == mode
    assert metadata["num_points"] == 11
    groups = field.parameter_groups(lr=0.002)
    assert [group["name"] for group in groups] == ["residual_amplitude", "phase_delta"]
    assert all(group["lr"] == 0.002 for group in groups)
    optimized = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(optimized) == len(set(optimized))
    assert set(optimized) == {id(parameter) for parameter in field.parameters()}


def test_fourier_features_are_deterministic_and_no_grad_inference_works():
    field = model()
    another = model()
    assert all(torch.equal(a, b) for a, b in zip(field.buffers(), another.buffers()))
    with torch.no_grad():
        assert torch.equal(field.get_density("plain"), field.plain_complex)
    with pytest.raises(ValueError, match="channel"):
        field.get_density("unknown")


@pytest.mark.parametrize("option", [{"mode": "wrong"}, {"bands": -1}, {"width": 0},
                                   {"gain_bound": -0.1}, {"chunk_size": 0}])
def test_invalid_configuration_is_rejected(option):
    with pytest.raises((TypeError, ValueError)):
        ResidualAmplitudeField(*inputs(), **option)
