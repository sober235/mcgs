"""CPU boundary fixtures; no acquisition, checkpoint, GT, or CUDA is loaded."""
import ast
import copy
from pathlib import Path
import types

import pytest
import torch

from density_operator import DensityLinearOperator


SOURCE = Path(__file__).resolve().parents[2] / "nonlocal_autoresearch/trainer/paired_gs"
NAMES = ("positions", "scales", "rotations", "density_real", "density_imag",
         "density_real_contrast", "density_imag_contrast")


def source_loss_methods():
    """Load verbatim source methods without importing CUDA trainer dependencies."""
    transforms = ast.parse((SOURCE / "data/transforms.py").read_text())
    fft_node = next(node for node in transforms.body if isinstance(node, ast.FunctionDef) and node.name == "fft3c")
    namespace = {"torch": torch, "fft": torch.fft}
    exec(compile(ast.Module(body=[fft_node], type_ignores=[]), str(SOURCE / "data/transforms.py"), "exec"), namespace)
    module = ast.parse((SOURCE / "trainers/joint_trainer.py").read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "JointTrainer")
    selected = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name in ("_kspace_loss", "_build_freq_weight")]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE / "trainers/joint_trainer.py"), "exec"), namespace)
    return namespace


class TinyModel(torch.nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.positions = torch.nn.Parameter(torch.arange(9, dtype=dtype).reshape(3, 3) / 10)
        self.scales = torch.nn.Parameter(torch.zeros(3, 3, dtype=dtype))
        self.rotations = torch.nn.Parameter(torch.tensor([[1., 0, 0, 0]] * 3, dtype=dtype))
        self.density_real = torch.nn.Parameter(torch.tensor([-2., 1., .5], dtype=dtype))
        self.density_imag = torch.nn.Parameter(torch.tensor([1., -.5, 2.], dtype=dtype))
        self.density_real_contrast = torch.nn.Parameter(torch.tensor([1., -3., 2.], dtype=dtype))
        self.density_imag_contrast = torch.nn.Parameter(torch.tensor([-.5, 2., 1.], dtype=dtype))
        self.num_points = 3
        self.has_private = False

    def get_density(self, channel):
        if channel == "plain":
            return torch.complex(self.density_real, self.density_imag)
        return torch.complex(self.density_real_contrast, self.density_imag_contrast)


class TinyTrainer:
    def __init__(self, dtype=torch.float64):
        self.model = TinyModel(dtype)
        self.device = torch.device("cpu")
        self.volume_shape = (2, 2, 2)
        self.config = {"adaptive_control": {"enable": False}, "gaussian": {"num_private_points": 0},
                       "loss": {"kspace_freq_adaptive": True, "kspace_high_weight": 2.8,
                                "kspace_high_weight_plain": 2.2, "kspace_center_weight": .3}}
        # Nonnegative Gaussian-like basis, including one structural zero voxel.
        self.matrix = torch.tensor([[1., .5, 0], [0, 1, .5], [0, 0, 0], [.25, 0, 1],
                                    [.1, .2, .3], [1, 1, 0], [0, .1, 1], [.5, 0, .5]], dtype=dtype)
        ctype = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.fail_channel = None
        self.renderer_calls = 0
        for tag, offset in (("p", 0.), ("c", .5)):
            sensitivity = torch.ones(2, *self.volume_shape, dtype=ctype)
            sensitivity[1] = .5 + .25j
            self.__dict__[f"csm_fit_{tag}"] = sensitivity
            self.__dict__[f"kspace_fit_{tag}"] = torch.complex(
                torch.arange(16, dtype=dtype).reshape(2, *self.volume_shape) / 20 + offset,
                torch.full((2, *self.volume_shape), .2, dtype=dtype))
            # Fit mask has more samples than the original acquisition denominator.
            self.__dict__[f"mask_fit_{tag}"] = torch.tensor([1., 1, 0, 1, 1, 0, 1, 1], dtype=dtype).reshape(self.volume_shape)
            # Keep original acquisition separate from the expanded fit mask.
            self.__dict__[f"mask_{tag}"] = torch.tensor(
                [1., 1, 0, 1, 0, 0, 0, 0] if tag == "p" else [1., 1, 0, 1, 1, 0, 0, 0],
                dtype=dtype).reshape(self.volume_shape)
            self.__dict__[f"_dc_norm_{tag}"] = 3. if tag == "p" else 4.
        for name, function in source_loss_methods().items():
            if name in ("_kspace_loss", "_build_freq_weight"):
                setattr(self, name, types.MethodType(function, self))
        self.optimizer = torch.optim.Adam(self.model.parameters())
        # Populate real optimizer state once, before snapshots and operator calls.
        sum(p.square().sum() for p in self.model.parameters()).backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _render(self, channel):
        self.renderer_calls += 1
        assert all(not getattr(self.model, name).requires_grad for name in NAMES[:3])
        if channel == self.fail_channel:
            raise RuntimeError("deliberate renderer failure")
        return (self.matrix.to(self.model.get_density(channel).dtype)
                @ self.model.get_density(channel)).reshape(self.volume_shape)


def assert_nested_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_nested_equal(left, right)
    else:
        assert actual == expected


@pytest.fixture
def trainer():
    return TinyTrainer()


def test_signed_complex_superposition_external_autograd_and_adjoint(trainer):
    op = DensityLinearOperator(trainer)
    rho = op.initial_density.clone().requires_grad_()
    other = torch.tensor([[1j, 2., -1.], [-2., 3j, .5]], dtype=rho.dtype)
    alpha, beta = 1.25 - .75j, -.3 + .4j
    pair = op.render(rho)
    assert torch.allclose(pair[0].flatten(), trainer.matrix.to(rho.dtype) @ rho[0])
    assert torch.any(pair[0].real < 0)  # Signed density must never be clamped.
    combined = op.render(alpha * rho + beta * other)
    other_pair = op.render(other)
    for got, first, second in zip(combined, pair, other_pair):
        assert torch.allclose(got, alpha * first + beta * second, atol=1e-12)
    gradients = (torch.arange(8, dtype=torch.float64).reshape(trainer.volume_shape) * (.1 + .2j),
                 torch.ones(trainer.volume_shape, dtype=rho.dtype) * (-.5 + .3j))
    actual = op.adjoint(*gradients)
    expected = torch.stack([trainer.matrix.to(rho.dtype).mH @ g.flatten() for g in gradients])
    assert torch.allclose(actual, expected, atol=1e-12)
    lhs = sum(torch.vdot(image.flatten(), g.flatten()) for image, g in zip(pair, gradients))
    assert torch.allclose(lhs, torch.vdot(rho.flatten(), actual.flatten()), atol=1e-12)
    external_gradient, = torch.autograd.grad(sum((image.conj() * g).real.sum() for image, g in zip(pair, gradients)), rho)
    assert torch.allclose(external_gradient, expected, atol=1e-12)
    assert not actual.requires_grad


def test_structural_support_uses_positive_basis_once_and_has_no_graph(trainer):
    op = DensityLinearOperator(trainer)
    # Cancellation at row0 is not structural; row2 has no Gaussian support.
    rho = torch.tensor([[1., -2., 1.], [1., -2., 1.]], dtype=torch.complex128)
    assert op.render(rho)[0].flatten()[0] == 0
    before = trainer.renderer_calls
    first = op.structural_support()
    after = trainer.renderer_calls
    second = op.structural_support()
    assert after - before == 2
    assert trainer.renderer_calls == after
    expected = torch.tensor([True, True, False, True, True, True, True, True]).reshape(trainer.volume_shape)
    for support, repeated in zip(first, second):
        assert torch.equal(support, expected)
        assert torch.equal(support, repeated)
        assert support.dtype == torch.bool and not support.requires_grad


def test_render_adjoint_and_exceptions_preserve_exact_parameters_and_adam(trainer):
    op = DensityLinearOperator(trainer)
    references = dict(trainer.model._parameters)
    values = {name: value.detach().clone() for name, value in references.items()}
    state = copy.deepcopy(trainer.optimizer.state_dict())
    original_density = op.initial_density.clone()
    pre, post = op.render(op.initial_density.requires_grad_())
    op.adjoint(torch.ones_like(pre), torch.ones_like(post))
    op.structural_support()
    trainer.fail_channel = "contrast"
    with pytest.raises(RuntimeError, match="deliberate"):
        op.render(op.initial_density)
    with pytest.raises(RuntimeError, match="deliberate"):
        op.adjoint(torch.ones_like(pre), torch.ones_like(post))
    for name, reference in references.items():
        assert trainer.model._parameters[name] is reference
        assert torch.equal(reference, values[name])
        assert reference.grad is None
    assert torch.equal(op.initial_density.detach(), original_density)
    assert_nested_equal(trainer.optimizer.state_dict(), state)


def explicit_acquisition_matrix(trainer, channel):
    """Dense SENSE/centered FFT from the DFT definition, independent of source FFT."""
    shape = trainer.volume_shape
    indices = torch.cartesian_prod(*(torch.arange(n) for n in shape)).to(torch.float64)
    centered = indices - torch.tensor([n // 2 for n in shape])
    phase = centered @ (centered / torch.tensor(shape)).T
    fourier = torch.exp(-2j * torch.pi * phase) / (8. ** .5)
    csm = getattr(trainer, f"csm_fit_{channel}").reshape(2, -1)
    mask = getattr(trainer, f"mask_fit_{channel}").flatten()
    high = torch.tensor(2.2 if channel == "p" else 2.8, dtype=torch.float32).item()
    # All coordinates of a 2x2x2 grid have radius sqrt(3)>0.5.
    return torch.cat([mask[:, None] * high * fourier * coil[None, :] for coil in csm])


def test_source_weighted_masked_sense_dc_and_normal_keep_original_denominator(trainer):
    op = DensityLinearOperator(trainer)
    rho = op.initial_density.clone().requires_grad_()
    images = op.render(rho)
    expected_loss = rho.real.new_zeros(())
    expected_zero = rho.real.new_zeros(())
    expected_normal = []
    for tag, image in zip(("p", "c"), images):
        acquisition = explicit_acquisition_matrix(trainer, tag)
        norm = getattr(trainer, f"_dc_norm_{tag}")
        mask = getattr(trainer, f"mask_fit_{tag}").flatten()
        high = torch.tensor(2.2 if tag == "p" else 2.8, dtype=torch.float32).item()
        measured = getattr(trainer, f"kspace_fit_{tag}").reshape(2, -1)
        weighted_target = (measured * mask * high).flatten()
        prediction = acquisition @ image.flatten()
        expected_loss = expected_loss + (prediction - weighted_target).abs().square().sum() / norm
        expected_zero = expected_zero + prediction.abs().square().sum() / norm
        normal = trainer.matrix.to(rho.dtype).mH @ acquisition.mH @ acquisition @ image.flatten() / norm
        expected_normal.append(normal)
    actual_loss = op.data_loss(*images)
    actual_zero = op.data_loss(*images, zero_target=True)
    assert torch.allclose(actual_loss, expected_loss, atol=1e-11)
    assert torch.allclose(actual_zero, expected_zero, atol=1e-11)
    # d(DC/2)/d(rho) = A^H A rho under PyTorch's complex gradient convention.
    normal, = torch.autograd.grad(actual_zero / 2, rho)
    assert torch.allclose(normal, torch.stack(expected_normal), atol=1e-11)
    assert getattr(trainer, "_dc_norm_p") == 3.
    assert torch.allclose(op.data_loss(*images), actual_loss)


@pytest.mark.parametrize("flag", ["private", "adaptive", "curriculum", "shape", "extra_parameter"])
def test_unsupported_models_and_dynamic_flags_are_rejected(trainer, flag):
    if flag == "private":
        trainer.model.has_private = True
    elif flag == "adaptive":
        trainer.config["adaptive_control"]["enable"] = True
    elif flag == "curriculum":
        trainer.config["loss"]["kspace_freq_curriculum"] = True
    elif flag == "shape":
        trainer.model.density_real = torch.nn.Parameter(torch.ones(3, 1, dtype=torch.float64))
    else:
        trainer.model.extra = torch.nn.Parameter(torch.ones(1))
    with pytest.raises((ValueError, NotImplementedError)):
        DensityLinearOperator(trainer)


def test_bad_external_density_and_nonfinite_basis_fail_with_restoration(trainer):
    op = DensityLinearOperator(trainer)
    references = dict(trainer.model._parameters)
    for rho in (torch.ones(2, 3), torch.ones(3, 2, dtype=torch.complex128),
                torch.full((2, 3), complex(float("nan"), 0), dtype=torch.complex128)):
        with pytest.raises((TypeError, ValueError)):
            op.render(rho)
    trainer.matrix[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        op.structural_support()
    assert all(trainer.model._parameters[name] is reference for name, reference in references.items())


def test_frozen_weights_norms_and_actual_renderer_vjp_call_accounting(trainer):
    op = DensityLinearOperator(trainer)
    images = op.render(op.initial_density)
    loss = op.data_loss(*images)
    trainer.config["loss"]["kspace_high_weight"] = 99.
    trainer.config["loss"]["kspace_high_weight_plain"] = 101.
    trainer._dc_norm_p = trainer._dc_norm_c = 999.
    assert torch.equal(op.data_loss(*images), loss)
    op.adjoint(*(torch.ones_like(image) for image in images))
    op.structural_support()
    op.structural_support()
    assert op.counters == {"render_pairs": 3, "renderer_calls": 6,
                           "adjoint_calls": 1, "vjp_calls": 1,
                           "data_loss_calls": 2, "kspace_loss_calls": 4}
    assert trainer.renderer_calls == 6


def test_first_order_vjp_failure_restores_parameters_without_requesting_double_backward(trainer):
    class FirstOrderFailure(torch.autograd.Function):
        @staticmethod
        def forward(ctx, image):
            return image

        @staticmethod
        def backward(ctx, gradient):
            assert not torch.is_grad_enabled()  # create_graph must remain False.
            raise RuntimeError("deliberate first-order VJP failure")

    op = DensityLinearOperator(trainer)
    original_render = trainer._render
    references = dict(trainer.model._parameters)
    trainer._render = lambda channel: FirstOrderFailure.apply(original_render(channel))
    with pytest.raises(RuntimeError, match="VJP failure"):
        op.adjoint(torch.ones(trainer.volume_shape, dtype=torch.complex128),
                   torch.ones(trainer.volume_shape, dtype=torch.complex128))
    assert all(trainer.model._parameters[name] is reference for name, reference in references.items())


def test_complex64_operator_matches_dense_adjoint_and_refuses_missing_frequency_weights():
    trainer = TinyTrainer(torch.float32)
    trainer.config["loss"]["kspace_freq_adaptive"] = False
    trainer.config["loss"].pop("kspace_high_weight_plain")
    with pytest.raises(ValueError, match="frequency weight"):
        DensityLinearOperator(trainer)
    trainer.config["loss"]["kspace_freq_adaptive"] = True
    op = DensityLinearOperator(trainer)
    rho = op.initial_density
    images = op.render(rho)
    gradients = tuple(torch.full_like(image, -.3 + .7j) for image in images)
    expected = torch.stack([trainer.matrix.to(rho.dtype).mH @ gradient.flatten() for gradient in gradients])
    assert rho.dtype == torch.complex64
    assert torch.allclose(op.adjoint(*gradients), expected, atol=2e-6)
    loss = op.data_loss(*images)
    expected_loss = sum(trainer._kspace_loss(image, getattr(trainer, f"csm_fit_{tag}"),
                                           getattr(trainer, f"kspace_fit_{tag}"),
                                           getattr(trainer, f"mask_fit_{tag}"), weight,
                                           norm=getattr(trainer, f"_dc_norm_{tag}"))
                        for image, tag, weight in zip(images, ("p", "c"), op._frequency_weights))
    assert torch.equal(loss, expected_loss)
