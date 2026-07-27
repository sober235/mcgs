import numpy as np
import torch
from crisp.io import normalize_p999


def test_normalize_p999_scales_kspace_and_returns_scale():
    g = torch.Generator().manual_seed(3)
    ksp = torch.randn(2, 4, 4, 4, generator=g, dtype=torch.complex64)
    csm = torch.randn(2, 4, 4, 4, generator=g, dtype=torch.complex64)
    csm = csm / (csm.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)
    ksp_n, scale = normalize_p999(ksp, csm)
    assert scale > 0
    assert torch.allclose(ksp_n * scale, ksp, atol=1e-5)


def test_normalize_p999_scale_is_actually_p999_of_coil_combined_magnitude():
    """回环恒等式(上一测试)对任意非零 scale 都成立,不能证明用的是 P99.9 分位数
    (例如误用中位数也会通过)。这里独立算出预期分位数并与返回值比对,才能区分。"""
    g = torch.Generator().manual_seed(11)
    ksp = torch.randn(3, 6, 6, 6, generator=g, dtype=torch.complex64)
    csm = torch.randn(3, 6, 6, 6, generator=g, dtype=torch.complex64)
    csm = csm / (csm.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)

    from crisp.metrics.coil_domain import ifft3c
    gt_mag = (csm.conj() * ifft3c(ksp)).sum(0).abs().numpy()
    expected_scale = float(np.quantile(gt_mag, 0.999))

    _, scale = normalize_p999(ksp, csm)
    assert np.isclose(scale, expected_scale, rtol=1e-6)
    assert not np.isclose(scale, float(np.quantile(gt_mag, 0.5)), rtol=1e-2)
