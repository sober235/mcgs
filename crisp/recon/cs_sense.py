"""CS-SENSE:FISTA + 各向同性 TV(独立)与 vectorial TV(联合多对比度)。

TV 用固定步长的次梯度近端替代(smoothed TV,eps 平滑),便于两种变体共用同一步长规则。
联合版的耦合仅通过 vectorial TV 的跨对比度梯度范数实现:
    TV_joint = sum_voxel sqrt( sum_m sum_axis |grad_axis |x_m| |^2 + eps^2 )
"""
from __future__ import annotations

import torch

from crisp.metrics.coil_domain import fft3c, ifft3c

_EPS = 1e-8

# 固定步长 step=0.25(而非教科书式 1/L=1.0):数据项的算子范数确实 <=1,但光滑 TV 项在
# |grad|~0 附近曲率随 1/eps 增长,step=1.0 在 lam 稍大时会让 TV 随 lam 增大而不是减小
# (数值不稳定,不是散度符号错——用有限差分核验过 _tv_grad 本身是对的)。step=0.25 在本模块
# 覆盖的 lam 范围内经验证是安全的。


def _grads(mag):
    gx = torch.zeros_like(mag); gy = torch.zeros_like(mag); gz = torch.zeros_like(mag)
    gx[:-1] = mag[1:] - mag[:-1]
    gy[:, :-1] = mag[:, 1:] - mag[:, :-1]
    gz[:, :, :-1] = mag[:, :, 1:] - mag[:, :, :-1]
    return gx, gy, gz


def _tv_grad(x, coupled_norm=None):
    """smoothed-TV 对复数 x 的梯度(经由 |x|)。coupled_norm 给出跨对比度归一化项。

    前向差分 g_j = m_{j+1} - m_j(末位补 0)。对 sum_j u_j g_j 关于 m_j 求导得
    d/dm_j = u_{j-1} - u_j,故散度按 `div[1:] += u[:-1]; div[:-1] -= u[:-1]` 累加。
    最后经由 d|x|/dx = x/|x| 传回复数域。
    """
    mag = x.abs().clamp_min(_EPS)
    gx, gy, gz = _grads(mag)
    n = coupled_norm if coupled_norm is not None else torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2 + _EPS ** 2)
    ux, uy, uz = gx / n, gy / n, gz / n
    div = torch.zeros_like(mag)
    div[1:] += ux[:-1];        div[:-1] -= ux[:-1]
    div[:, 1:] += uy[:, :-1];  div[:, :-1] -= uy[:, :-1]
    div[:, :, 1:] += uz[:, :, :-1]; div[:, :, :-1] -= uz[:, :, :-1]
    return div * (x / mag)


def _data_grad(x, y, csm, mask):
    resid = mask.unsqueeze(0) * (fft3c(csm * x.unsqueeze(0)) - y)
    return (ifft3c(resid) * csm.conj()).sum(0)


_CSM_SOS_TOL = 1.01  # 逐点 sum_c|csm_c|^2 允许超过 1 的容差(浮点/归一化误差余量)


def _check_csm_scale(csm):
    """守卫,不是修正:确认 step 默认值的稳定性论证(见模块顶部注释)真的成立。

    数据项梯度算子范数 <=1 依赖 sum_c|csm_c|^2 逐点 <=1。这里只查上界——sum-of-squares
    小于 1 只会让算子范数更小,不会破坏稳定性,不要在这里加下界检查(那会把这条守卫
    悄悄变成两边都查,而下界本来就是安全的)。一次 O(1) reduction,发生在迭代循环之外,
    不影响每步开销。
    """
    peak = float((csm.abs() ** 2).sum(0).max())
    if peak > _CSM_SOS_TOL:
        raise ValueError(
            f"csm 的逐点 sum_c|csm_c|^2 峰值为 {peak:.4f},超过 {_CSM_SOS_TOL}。"
            "step 的默认值是按 sum_c|csm_c|^2<=1 校准的稳定步长(见模块顶部注释);超出该假设会"
            "让数据项算子范数 >1,默认 step 不再安全,失效方式是静默振荡/缓慢发散,不是报错。"
            "请重新归一化线圈灵敏度图(除以 sqrt(sum_c|csm_c|^2)),或显式传入更小的 step。"
        )


def fista_tv(y, csm, mask, lam: float, iters: int = 200, device: str | None = None,
             step: float = 0.25) -> torch.Tensor:
    """FISTA CS-SENSE:最小化 (1/2)||P_mask F S x - y||^2 + lam * TV_smooth(x)。

    `mask` 须为严格 0/1 的欠采样投影。`step` 默认值 0.25 的稳定性论证依赖
    `sum_c|csm_c|^2` 逐点 <=1(见模块顶部注释);违反这条假设(例如线圈图用了别的缩放
    约定,或像 Task 4 实测的真实线圈图那样某些体素灵敏度和显著偏离 1)会被
    `_check_csm_scale` 检测到并抛出 `ValueError`——本函数不会替调用者静默重新缩放数据,
    需要调用方自己重新归一化线圈图,或显式传入更小的 `step`。
    """
    if device is not None:
        y, csm, mask = y.to(device), csm.to(device), mask.to(device)
    _check_csm_scale(csm)
    x = (ifft3c(mask.unsqueeze(0) * y) * csm.conj()).sum(0)
    z, t = x.clone(), 1.0
    for _ in range(iters):
        g = _data_grad(z, y, csm, mask) + lam * _tv_grad(z)
        x_new = z - step * g
        t_new = 0.5 * (1 + (1 + 4 * t * t) ** 0.5)
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return x


def fista_joint_tv(ys, csms, masks, lam: float, iters: int = 200, device: str | None = None,
                   step: float = 0.25) -> list[torch.Tensor]:
    """联合多对比度 FISTA CS-SENSE,通过 vectorial TV 的共享梯度范数耦合各通道。

    每个通道各自的 `mask` 须为严格 0/1 投影。`step` 默认值 0.25 的稳定性论证同
    `fista_tv`,依赖每个通道的 `sum_c|csm_c|^2` 逐点 <=1(见模块顶部注释);任一通道违反
    该假设都会被 `_check_csm_scale` 检测到并抛出 `ValueError`,不会静默重新缩放数据。
    """
    if device is not None:
        ys = [v.to(device) for v in ys]; csms = [v.to(device) for v in csms]
        masks = [v.to(device) for v in masks]
    for c in csms:
        _check_csm_scale(c)
    xs = [(ifft3c(m.unsqueeze(0) * y) * c.conj()).sum(0) for y, c, m in zip(ys, csms, masks)]
    zs, t = [v.clone() for v in xs], 1.0
    for _ in range(iters):
        acc = None
        for z in zs:
            gx, gy, gz = _grads(z.abs().clamp_min(_EPS))
            s = gx ** 2 + gy ** 2 + gz ** 2
            acc = s if acc is None else acc + s
        n = torch.sqrt(acc + _EPS ** 2)                      # 跨对比度共享的归一化
        new_xs = []
        for z, y, c, m in zip(zs, ys, csms, masks):
            g = _data_grad(z, y, c, m) + lam * _tv_grad(z, coupled_norm=n)
            new_xs.append(z - step * g)
        t_new = 0.5 * (1 + (1 + 4 * t * t) ** 0.5)
        zs = [xn + ((t - 1) / t_new) * (xn - xo) for xn, xo in zip(new_xs, xs)]
        xs, t = new_xs, t_new
    return xs
