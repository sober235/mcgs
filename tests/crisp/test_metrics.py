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


def test_e_unobs_zero_estimate_is_exactly_one():
    """手算锁定分母公式:F S 0 = 0 对任意线性前向算子恒成立,所以零估计的分子
    恰好等于 ||(1-M) y_full||,与正确分母相除必然精确等于 1.0——与 csm/y/mask 取值
    无关,不需要真的算 FFT。若分母被误改成 ||y_full||(未按未采样区裁剪),比值会
    变成 ||(1-M) y_full|| / ||y_full||,只要采样区里也有能量就会 < 1,从而与这里断言
    的精确值 1.0 产生可测的偏差,而不是像现有阈值测试那样只保证"足够小/足够大"。
    """
    g = torch.Generator().manual_seed(1)
    csm = torch.randn(3, 4, 4, 4, generator=g, dtype=torch.complex64)
    csm = csm / (csm.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)   # SoS == 1
    y = torch.randn(3, 4, 4, 4, generator=g, dtype=torch.complex64)     # 任意"测得" k 空间,采样区非零
    mask = torch.zeros(4, 4, 4)
    mask[:2] = 1.0                                                      # 半采样、半未采样
    x_zero = torch.zeros(4, 4, 4, dtype=torch.complex64)
    assert abs(e_unobs(x_zero, csm, y, mask) - 1.0) < 1e-6


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


def test_brain_gauge_p999_peak_and_mask_frac_boundary():
    """上面这个二值 0/1 体模测不出两处历史 mutation,因为它的脑内 P99.9 恰好等于全局
    max,且 frac 在 (0,1) 内怎么变掩膜都不变。这里换一个'脑内渐变组织 + 脑外单体素
    热点(如血管/伪影)'的体模,专门锁死这两处:
      (a) peak 必须是脑内 P99.9,不能被脑外热点主导的全局 max 取代;
      (b) brain_mask 的 frac 阈值改了,掩膜边界必须跟着动。
    """
    gt = np.zeros((30, 30, 30), dtype=np.float32)
    n = 20 ** 3
    grad = np.linspace(0.5, 1.0, n, dtype=np.float64).astype(np.float32).reshape(20, 20, 20)
    gt[5:25, 5:25, 5:25] = grad                    # 渐变"脑组织",8000 个体素,值域 [0.5, 1.0]
    gt[0, 0, 0] = 3.0                               # 脑外热点,只占 1 个体素,远高于组织值域

    mask_default = brain_mask(gt)                   # frac=0.05 (默认): threshold=0.15,组织+热点都算"脑内"
    assert mask_default.sum() == n + 1
    assert mask_default[0, 0, 0]

    # 手算:8001 个"脑内"值里,8000 个来自 grad(升序,rank 0..7999),1 个热点(3.0,rank 8000,最大)。
    # P99.9 分位数的虚拟下标 = 0.999*(8001-1) = 7992.0,恰好是整数、落在 grad 的 rank 7992 处
    # (< 8000,还没碰到热点那个 rank),所以脑内 P99.9 必然等于 grad 里第 7992 小的值,与热点
    # 无关;linspace(0.5,1.0,8000) 的第 i 个值(升序)就是 0.5+0.5*i/7999。
    expected_peak = 0.5 + 0.5 * 7992 / 7999
    g = brain_gauge(gt, gt, frac=0.05)
    assert abs(g["peak_used"] - expected_peak) < 1e-3   # 必须是脑内 P99.9(手算值),不能是别的
    assert g["peak_used"] < gt.max() - 1.0               # 必须显著小于全局 max(=热点的 3.0)

    mask_strict = brain_mask(gt, frac=0.60)              # threshold=1.8 > 组织最大值 1.0 -> 组织整体出局
    assert mask_strict.sum() == 1                        # 只剩热点体素本身满足更严阈值
    assert mask_default.sum() > mask_strict.sum()         # frac 变了,掩膜边界必须跟着动
