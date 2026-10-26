"""CPU evidence for the single paired-G2 coefficient conditioning change."""
import copy
import json
import subprocess
import types
from pathlib import Path

import pytest
import torch

from amplitude_model import ResidualAmplitudeField


def inputs(n=11, requires_grad=False):
    generator = torch.Generator().manual_seed(28)
    return tuple(value.requires_grad_(requires_grad) for value in (
        torch.randn(n, 3, dtype=torch.float64, generator=generator),
        torch.randn(n, dtype=torch.complex128, generator=generator),
        torch.randn(n, dtype=torch.complex128, generator=generator)))


def field(feature_mode="paired_anchor", *, n=11, chunk_size=4, values=None):
    torch.manual_seed(91)
    return ResidualAmplitudeField(*(inputs(n) if values is None else values),
        mode="shared", feature_mode=feature_mode, bands=2, width=7,
        gain_bound=.25, chunk_size=chunk_size)


def active_network(value):
    with torch.no_grad():
        value.network[-1].weight.fill_(.13)
        value.network[-1].bias.copy_(torch.tensor([.02, -.03]))


def test_coordinates_keep_frozen_v1_initialization_forward_and_common_gradients():
    # This fixed, pre-candidate revision is the numerical oracle, not a second
    # implementation that can silently acquire the same defect as the candidate.
    root = Path(__file__).parents[1]
    source = subprocess.run(["git", "show", "000c53d:amplitude_model.py"], cwd=root,
        check=True, capture_output=True, text=True).stdout
    legacy = types.ModuleType("frozen_coordinate_v1")
    exec(compile(source, "frozen_coordinate_v1.py", "exec"), legacy.__dict__)
    torch.manual_seed(91)
    original = legacy.ResidualAmplitudeField(*inputs(), bands=2, width=7, chunk_size=4)
    original_rng = torch.random.get_rng_state().clone()
    current = field("coordinates")
    assert torch.equal(torch.random.get_rng_state(), original_rng)
    assert current.state_dict().keys() == original.state_dict().keys()
    for key, value in original.state_dict().items():
        assert torch.equal(current.state_dict()[key], value), key
    for value in (original, current):
        active_network(value)
        sum(value.get_density(c).real.square().sum() for c in ("plain", "contrast")).backward()
    for channel in ("plain", "contrast"):
        assert torch.equal(current.get_density(channel), original.get_density(channel))
    for name, value in original.named_parameters():
        assert torch.equal(dict(current.named_parameters())[name].grad, value.grad), name


def test_zero_projection_keeps_initial_hidden_state_rng_densities_and_common_gradients():
    coordinates = field("coordinates")
    rng = torch.random.get_rng_state().clone()
    paired = field()
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.count_nonzero(paired.anchor_projection.weight) == 0
    assert paired.anchor_projection.weight.shape == (7, 2)
    for key, value in coordinates.state_dict().items():
        assert torch.equal(paired.state_dict()[key], value), key
    hidden = []
    for value in (coordinates, paired):
        activations = []
        hook = value.network[1].register_forward_pre_hook(
            lambda module, args: activations.append(args[0].detach().clone()))
        assert torch.equal(value.get_density("plain"), value.plain_complex)
        hook.remove()
        hidden.append(torch.cat(activations))
        active_network(value)
        sum(value.get_density(c).real.square().sum() for c in ("plain", "contrast")).backward()
    assert torch.equal(*hidden)
    for name, value in coordinates.named_parameters():
        assert torch.equal(dict(paired.named_parameters())[name].grad, value.grad), name


def test_features_use_both_global_coefficient_medians_and_detached_private_buffers():
    xyz = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
    plain = torch.tensor([0j, 3+4j, 12j], dtype=torch.complex128, requires_grad=True)
    contrast = torch.tensor([8j, 1j, 2j], dtype=torch.complex128, requires_grad=True)
    value = field(values=(xyz, plain, contrast), chunk_size=1)
    expected = torch.log1p(torch.tensor([[0., 4.], [1., .5], [2.4, 1.]], dtype=torch.float64))
    torch.testing.assert_close(value.anchor_features, expected)
    assert torch.equal(value.anchor_medians, torch.tensor([5., 2.], dtype=torch.float64))
    saved = value.anchor_features.clone()
    with torch.no_grad():
        plain.mul_(3); contrast.zero_(); xyz.add_(10)
    assert torch.equal(value.anchor_features, saved)
    active_network(value)
    value.get_density("plain").real.sum().backward()
    assert xyz.grad is plain.grad is contrast.grad is None
    assert all(not buffer.requires_grad for buffer in value.buffers())
    metadata = json.loads(json.dumps(value.metadata()))
    assert metadata["schema_version"] == 2
    assert metadata["feature_mode"] == "paired_anchor"
    assert metadata["anchor_features"]["medians"] == [5., 2.]
    assert metadata["anchor_features"]["cached_bytes"] == 3 * 2 * 8


@pytest.mark.parametrize("anchor_channel", [0, 1])
def test_each_anchor_changes_gain_but_phase_stays_channel_independent(anchor_channel):
    value = field()
    active_network(value)
    with torch.no_grad():
        value.anchor_projection.weight[:, anchor_channel] = .2
    before = value.get_density("plain").detach().clone()
    with torch.no_grad():
        value.anchor_features[:, anchor_channel].add_(.7)
    assert not torch.equal(value.get_density("plain"), before)
    value.get_density("plain").real.square().sum().backward()
    assert torch.count_nonzero(value.phase_delta.grad[:, 1]) == 0
    assert torch.count_nonzero(value.anchor_projection.weight.grad[:, anchor_channel]) > 0
    optimized = [p for group in value.parameter_groups() for p in group["params"]]
    assert sum(p is value.anchor_projection.weight for p in optimized) == 1
    assert {id(p) for p in optimized} == {id(p) for p in value.parameters()}
    optimizer = torch.optim.Adam(value.parameter_groups())
    projection = value.anchor_projection.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(value.anchor_projection.weight, projection)


def test_paired_checkpoint_chunks_gradients_and_state_roundtrip():
    chunked = field(n=17, chunk_size=3)
    active_network(chunked)
    with torch.no_grad():
        chunked.anchor_projection.weight.uniform_(-.2, .2)
        chunked.phase_delta.uniform_(-.2, .2)
    whole = field(n=17, chunk_size=100)
    whole.load_state_dict(copy.deepcopy(chunked.state_dict()))
    saved_shapes = []
    with torch.autograd.graph.saved_tensors_hooks(
            lambda tensor: (saved_shapes.append(tuple(tensor.shape)) or tensor), lambda tensor: tensor):
        outputs = [chunked.get_density(c) for c in ("plain", "contrast")]
    assert all(not shape or shape[0] <= 3 for shape in saved_shapes)
    for value, densities in ((chunked, outputs), (whole, [whole.get_density(c) for c in ("plain", "contrast")])):
        sum(d.real.square().sum() for d in densities).backward()
    for channel in ("plain", "contrast"):
        torch.testing.assert_close(chunked.get_density(channel), whole.get_density(channel))
    for name, parameter in chunked.named_parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, dict(whole.named_parameters())[name].grad,
            rtol=1e-12, atol=1e-12)
        assert torch.count_nonzero(parameter.grad) > 0, name


@pytest.mark.parametrize("n", [0, 3])
def test_zero_amplitudes_and_empty_point_cloud_remain_finite(n):
    xyz = torch.zeros(n, 3, dtype=torch.float64)
    base = torch.zeros(n, dtype=torch.complex128)
    value = field(n=n, values=(xyz, base, base))
    assert value.anchor_features.shape == (n, 2)
    assert torch.isfinite(value.anchor_features).all()
    assert value.metadata()["anchor_features"]["epsilon"] == 1e-12
    assert torch.equal(value.get_density("plain"), base)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_paired_anchor_rejects_nonfinite_input(index):
    values = list(inputs())
    values[index].flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        field(values=values)


def test_explicit_cannot_silently_ignore_paired_anchor_mode():
    with pytest.raises(ValueError, match="shared"):
        ResidualAmplitudeField(*inputs(), mode="explicit", feature_mode="paired_anchor")
