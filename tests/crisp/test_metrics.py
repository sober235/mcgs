import numpy as np
import torch

from crisp.metrics.coil_domain import e_coil_full, e_unobs, fft3c
from crisp.metrics.image_domain import brain_mask, brain_gauge


def _toy(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(8, 6, 8, generator=g, dtype=torch.complex64)
    csm = torch.randn(3, 8, 6, 8, generator=g, dtype=torch.complex64)
    csm = csm / (csm.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)   # SoS == 1
    y = fft3c(csm * x.unsqueeze(0))
    return x, csm, y


def test_e_coil_full_is_zero_for_exact_and_one_for_zero():
    x, csm, y = _toy()
    assert e_coil_full(x, csm, y) < 1e-5
    assert abs(e_coil_full(torch.zeros_like(x), csm, y) - 1.0) < 1e-5


def test_e_unobs_ignores_sampled_region():
    """必须能在遮罩写反时失败:三条断言分别锁定零点、已采样区不敏感、未采样区敏感。"""
    x, csm, y = _toy()
    mask = torch.zeros(8, 6, 8)
    mask[:4] = 1.0
    e_ref = e_unobs(x, csm, y, mask)
    assert e_ref < 1e-5                                   # 完美重建 -> 未采样区误差为 0
    y_s = y.clone(); y_s[:, :4] *= 3.0                    # 只破坏已采样区的真值
    assert abs(e_unobs(x, csm, y_s, mask) - e_ref) < 1e-6  # e_unobs 不应受其影响
    y_u = y.clone(); y_u[:, 4:] *= 3.0                    # 只破坏未采样区的真值
    assert e_unobs(x, csm, y_u, mask) > 0.1               # 遮罩写反则此条必挂


def test_brain_gauge_perfect_and_zero():
    gt = np.zeros((16, 16, 16), dtype=np.float32)
    gt[4:12, 4:12, 4:12] = 1.0
    m = brain_mask(gt)
    assert m.sum() == 8 ** 3
    perfect = brain_gauge(gt, gt)
    assert perfect["nmse_brain"] < 1e-12
    assert perfect["ssim_brain"] > 0.999
    zero = brain_gauge(np.zeros_like(gt), gt)
    assert abs(zero["nmse_brain"] - 1.0) < 1e-6
