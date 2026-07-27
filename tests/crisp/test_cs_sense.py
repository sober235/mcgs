import itertools

import numpy as np
import pytest
import torch

from crisp.metrics.coil_domain import fft3c, ifft3c, e_coil_full
from crisp.metrics.image_domain import brain_gauge
from crisp.masks.vdpoisson import vd_poisson_2d
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

    eps 用 _tv_grad 的实际默认值(fix round 2: Huber 平滑半径 1e-3,不再是数值下限
    1e-8)——参考 tv() 里的平滑常数必须和 _tv_grad 内部用的 eps 一致,否则两者根本
    不是同一个平滑泛函的梯度,有限差分自然对不上。
    """
    from crisp.recon.cs_sense import _tv_grad

    EPS = 1e-3  # 必须与下面 _tv_grad(x) 调用所用的默认 eps 一致

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
        return torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2 + EPS ** 2).sum()

    analytic = _tv_grad(x, eps=EPS).real
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


def _brain_phantom(n=64, seed=7):
    """真实尺度合成脑体模:>=64^3、>=8 线圈、非单一常数块的分片结构。

    fix round 2(见模块顶部说明及 task-6-report.md):Task 6 原来的 8^3 单常数块玩具
    体模测不出"次梯度伪装成光滑梯度"这个 bug——真实 12 线圈 256x220x256 数据上每个
    重建都劣于零填充,而 8^3 玩具体模的 4 条测试全过。这里用一个有真实分片结构(外部
    "脑"椭球 + 3 个不同强度的内部团块 + 细条纹)的 64^3 体模、8 个平滑衰减的线圈灵敏度
    图(而不是逐体素独立随机噪声,更接近真实阵列线圈),作为能复现该失效模式的最小
    足尺度回归。
    """
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
    x = torch.zeros(n, n, n)
    brain = (zz ** 2 / 0.85 ** 2 + yy ** 2 / 0.95 ** 2 + xx ** 2 / 0.8 ** 2) < 1.0
    x[brain] = 1.0
    blob1 = ((zz - 0.15) ** 2 + (yy + 0.1) ** 2 + (xx - 0.05) ** 2) < 0.30 ** 2
    x[blob1 & brain] = 0.65
    blob2 = ((zz + 0.25) ** 2 + (yy - 0.2) ** 2 + (xx + 0.15) ** 2) < 0.18 ** 2
    x[blob2 & brain] = 1.35
    blob3 = ((zz - 0.35) ** 2 + (yy + 0.35) ** 2 + xx ** 2) < 0.10 ** 2
    x[blob3 & brain] = 0.3
    stripe = (torch.abs(torch.sin(8 * xx)) < 0.05) & brain
    x[stripe] = 1.1
    x = x.to(torch.complex64)

    n_coils = 8
    g = torch.Generator().manual_seed(seed)
    csm = torch.zeros(n_coils, n, n, n, dtype=torch.complex64)
    for k in range(n_coils):
        ang = 2 * torch.pi * k / n_coils
        cy, cx = 1.3 * torch.cos(torch.tensor(ang)), 1.3 * torch.sin(torch.tensor(ang))
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2 + 0.3 * zz ** 2
        mag = torch.exp(-d2 / (2 * 0.9 ** 2))
        phase = torch.rand((), generator=g) * 2 * torch.pi
        csm[k] = mag * torch.exp(1j * phase)
    csm = csm / (csm.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-6)
    return x, csm


def _af12_mask(n, seed=0, acs=8):
    """相位编码平面 vd-Poisson 掩膜(AF~12)沿读出轴(第 3 维)复制为全采样。"""
    m2d = vd_poisson_2d((n, n), af=12.0, acs=acs, seed=seed)
    return torch.from_numpy(np.repeat(m2d[:, :, None], n, axis=2).astype(np.float32))


def test_real_scale_af12_beats_zero_fill():
    """更正要求的验收判据(task-6-brief.md「Task 6 更正」item 3):8^3 玩具体模测不出
    次梯度伪装成光滑梯度这个 bug,这里换成 >=64^3、>=8 线圈、AF~12、真实会用的 lam、
    300 次迭代的回归——fista_tv 必须打败零填充,否则不能被称为基线。
    lam=6.3e-3 取自本项目 G2 体素域实验实际使用的 lam 网格
    (crisp/experiments/g2_voxel.py 的 --lams 默认值)中间一档。
    """
    n = 64
    x, csm = _brain_phantom(n)
    mask = _af12_mask(n)
    y = fft3c(csm * x.unsqueeze(0))
    yu = y * mask.unsqueeze(0)
    zf = (ifft3c(yu) * csm.conj()).sum(0)
    gt_mag = x.abs().numpy()

    xh = fista_tv(yu, csm, mask, lam=6.3e-3, iters=300)

    psnr_zf = brain_gauge(zf.abs().numpy(), gt_mag)["psnr_brain"]
    psnr_rec = brain_gauge(xh.abs().numpy(), gt_mag)["psnr_brain"]
    assert psnr_rec > psnr_zf, f"recon psnr {psnr_rec:.2f} did not beat zero-fill {psnr_zf:.2f}"
