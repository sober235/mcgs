import itertools

import pytest
import torch

from crisp.metrics.coil_domain import fft3c, e_coil_full
from crisp.recon.cs_sense import fista_tv, fista_joint_tv


def _toy(seed=2):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(8, 8, 8, dtype=torch.complex64)
    x[2:6, 2:6, 2:6] = 1.0                       # 分片常数 -> TV 友好
    csm = torch.randn(3, 8, 8, 8, generator=g, dtype=torch.complex64)
    csm = csm / (csm.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)
    y = fft3c(csm * x.unsqueeze(0))
    return x, csm, y


def _toy_alt(seed=3):
    """与 _toy 结构不同的第二个体模(方块位置不同),用于检验跨对比度耦合是否真正生效。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(8, 8, 8, dtype=torch.complex64)
    x[0:3, 1:4, 5:8] = 1.0
    csm = torch.randn(3, 8, 8, 8, generator=g, dtype=torch.complex64)
    csm = csm / (csm.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)
    y = fft3c(csm * x.unsqueeze(0))
    return x, csm, y


def test_tv_fit_improves_over_zero_filled():
    from crisp.metrics.coil_domain import ifft3c
    x, csm, y = _toy()
    mask = torch.zeros(8, 8, 8); mask[::2] = 1.0
    yu = y * mask.unsqueeze(0)
    zf = (ifft3c(yu) * csm.conj()).sum(0)                 # 真正的零填充伴随解
    xh = fista_tv(yu, csm, mask, lam=1e-3, iters=100)
    assert e_coil_full(xh, csm, y) < e_coil_full(zf, csm, y)


def test_joint_channels_agree_when_inputs_identical():
    x, csm, y = _toy()
    mask = torch.zeros(8, 8, 8); mask[::2] = 1.0
    yu = y * mask.unsqueeze(0)
    solo = fista_tv(yu, csm, mask, lam=1e-3, iters=50)
    joint = fista_joint_tv([yu, yu], [csm, csm], [mask, mask], lam=1e-3, iters=50)
    # 两个通道数据相同 -> 联合解的两个通道彼此相同
    assert torch.allclose(joint[0], joint[1], atol=1e-5)
    assert joint[0].shape == solo.shape


def test_joint_reduces_to_independent_at_zero_coupling():
    """lam=0 时两者都退化为纯数据一致性梯度下降,应逐点相等。"""
    x, csm, y = _toy()
    mask = torch.zeros(8, 8, 8); mask[::2] = 1.0
    yu = y * mask.unsqueeze(0)
    solo = fista_tv(yu, csm, mask, lam=0.0, iters=40)
    joint = fista_joint_tv([yu, yu], [csm, csm], [mask, mask], lam=0.0, iters=40)
    assert torch.allclose(joint[0], solo, atol=1e-5)


def test_larger_lambda_gives_smaller_tv():
    x, csm, y = _toy()
    mask = torch.zeros(8, 8, 8); mask[::2] = 1.0
    yu = y * mask.unsqueeze(0)
    tv = lambda v: (v.abs().diff(dim=0).abs().sum() + v.abs().diff(dim=1).abs().sum()
                    + v.abs().diff(dim=2).abs().sum())
    assert tv(fista_tv(yu, csm, mask, lam=1e-2, iters=80)) < tv(fista_tv(yu, csm, mask, lam=1e-4, iters=80))


def test_tv_grad_matches_finite_difference():
    """直接用有限差分核验 _tv_grad 的散度符号与幅度,独立于 FISTA 收敛动态。

    用实值正数体积规避复数 Wirtinger 导数的因子歧义(此时 mag=x, x/mag=1,
    _tv_grad 输出退化为对光滑 TV 泛函的普通实梯度)。这是判定散度符号的最直接证据:
    若 `div[1:] += u[:-1]; div[:-1] -= u[:-1]` 的符号反了,这里会直接不匹配。
    """
    from crisp.recon.cs_sense import _tv_grad, _EPS

    # float64/complex128: 用 float32 时中心差分的舍入噪声(~1e-3 量级)和真实符号误差
    # 混在一起不好区分,双精度下把噪声压到 ~1e-7,任何符号错误都会显著超出该量级。
    g = torch.Generator().manual_seed(0)
    x_real = 0.5 + torch.rand(4, 4, 4, generator=g, dtype=torch.float64)
    x = x_real.to(torch.complex128)

    def tv(v):
        mag = v.abs()
        gx = torch.zeros_like(mag); gy = torch.zeros_like(mag); gz = torch.zeros_like(mag)
        gx[:-1] = mag[1:] - mag[:-1]
        gy[:, :-1] = mag[:, 1:] - mag[:, :-1]
        gz[:, :, :-1] = mag[:, :, 1:] - mag[:, :, :-1]
        return torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2 + _EPS ** 2).sum()

    analytic = _tv_grad(x).real
    h = 1e-5
    numeric = torch.zeros_like(analytic)
    for i, j, k in itertools.product(range(4), range(4), range(4)):
        xp = x.clone(); xp[i, j, k] += h
        xm = x.clone(); xm[i, j, k] -= h
        numeric[i, j, k] = (tv(xp) - tv(xm)) / (2 * h)

    assert torch.allclose(analytic, numeric, atol=1e-6, rtol=1e-6)


def test_joint_coupling_changes_result_vs_independent():
    """两通道结构不同时,联合(vectorial TV)解应与逐通道独立 TV 求解的堆叠结果不同——
    否则说明 `coupled_norm` 被忽略,联合退化为“各自独立”的伪联合实现。

    这条测试专门弥补 test_joint_channels_agree_when_inputs_identical 的盲区:
    该测试用两个完全相同的通道,即便实现悄悄退化成逐通道独立 TV,两通道依旧会互相相等,
    无法暴露耦合是否真的生效。
    """
    x1, csm1, y1 = _toy()
    x2, csm2, y2 = _toy_alt()
    mask = torch.zeros(8, 8, 8); mask[::2] = 1.0
    yu1, yu2 = y1 * mask.unsqueeze(0), y2 * mask.unsqueeze(0)
    lam = 5e-2

    joint = fista_joint_tv([yu1, yu2], [csm1, csm2], [mask, mask], lam=lam, iters=60)
    indep0 = fista_tv(yu1, csm1, mask, lam=lam, iters=60)
    indep1 = fista_tv(yu2, csm2, mask, lam=lam, iters=60)

    diff0 = (joint[0] - indep0).abs().max()
    diff1 = (joint[1] - indep1).abs().max()
    assert diff0 > 1e-4 or diff1 > 1e-4


def test_step_guard_rejects_oversized_csm():
    """sum_c|csm|^2 逐点 = 4(远超 <=1 假设)时,两个入口都必须在迭代前拒绝,而不是
    静默振荡/发散。这不是假设性的:Task 4 在真实线圈图上就测到过和 ~1 严重偏离的
    sum-of-squares(那边偏低,这里构造偏高的情形来触发同一条守卫的上界分支)。
    """
    x, csm, y = _toy()
    mask = torch.zeros(8, 8, 8); mask[::2] = 1.0
    csm_big = csm * 2.0                      # sum_c|csm_big|^2 = 4 * sum_c|csm|^2 = 4 逐点
    y_big = fft3c(csm_big * x.unsqueeze(0))
    yu_big = y_big * mask.unsqueeze(0)

    with pytest.raises(ValueError):
        fista_tv(yu_big, csm_big, mask, lam=1e-3, iters=5)
    with pytest.raises(ValueError):
        fista_joint_tv([yu_big, yu_big], [csm_big, csm_big], [mask, mask], lam=1e-3, iters=5)


def test_sub_unit_csm_is_accepted_and_still_converges():
    """守卫只查上界:sum_c|csm|^2 逐点 = 0.25(明显小于 1)必须被接受,且仍应收敛——
    防止以后有人把这条守卫"顺手"改成两边都查(下界本来就是安全的,不该被拒绝)。
    """
    from crisp.metrics.coil_domain import ifft3c
    x, csm, y = _toy()
    csm_small = csm * 0.5                    # sum_c|csm_small|^2 = 0.25 逐点
    y_small = fft3c(csm_small * x.unsqueeze(0))
    mask = torch.zeros(8, 8, 8); mask[::2] = 1.0
    yu_small = y_small * mask.unsqueeze(0)
    zf = (ifft3c(yu_small) * csm_small.conj()).sum(0)

    xh = fista_tv(yu_small, csm_small, mask, lam=1e-3, iters=100)  # 不应抛异常
    assert e_coil_full(xh, csm_small, y_small) < e_coil_full(zf, csm_small, y_small)
