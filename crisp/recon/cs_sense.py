"""CS-SENSE:FISTA + 各向同性 TV(独立)与 vectorial TV(联合多对比度)。

Huber 型平滑 TV:TV_smooth(x) = sum_voxel sqrt(|grad|x||^2 + eps^2)。`eps` 是一个真正的
光滑化参数(不是数值下限),默认 1e-3,假定输入图像已按 P99.9 归一化到量级 1(参见
`crisp/io.py` 的 `normalize_p999`)。在该假设下 TV_smooth 处处可微,Lipschitz 常数约
`lam/eps`;FISTA 步长按 `step = 1/(1 + lam/eps)` 与之匹配(数据项算子范数 L<=1,依赖
`sum_c|csm_c|^2<=1`,见 `_check_csm_scale`)。
联合版的耦合仅通过 vectorial TV 的跨对比度梯度范数实现:
    TV_joint = sum_voxel sqrt( sum_m sum_axis |grad_axis |x_m| |^2 + eps^2 )

fix round 2(2026-07-27,更正见 docs/superpowers/plans/2026-07-27-crisp-phase0.md
「Task 6 更正」):此前 `eps` 硬编码为数值下限 `1e-8`,使 `u = g/sqrt(g^2+eps^2) ≈ sign(g)`——
TV 项名义上"光滑",实际是非光滑项的次梯度,包在 FISTA 壳里没有收敛保证。真实
12 线圈 256x220x256 数据上,λ 网格里每一个重建都劣于零填充(4-14 dB),且 λ 越大/迭代
越多越坏——未采样零空间里没有任何东西纠正次梯度注入的符号噪声。修复:把 `eps` 提升为
函数参数(Huber 光滑,默认 1e-3),步长按 Lipschitz 常数重新匹配,不再用与 `lam`/`eps`
无关的固定 `0.25`。选择"提升 eps 做 Huber 光滑"而非"换成 Chambolle 对偶 prox-TV"的理由:
本模块的核心结构(FISTA 壳 + `_tv_grad` 经 `coupled_norm` 实现独立/联合共用同一份耦合
逻辑)已经是"光滑梯度"范式而非"近端"范式,Huber 化只需要把一个硬编码常数提升为参数、
并同步步长公式,独立版和联合版的改动逐字对称;换成 prox-TV 需要为 vectorial TV 重新推导
一个联合近端算子(该算子没有闭式解,duality-based Chambolle 迭代通常只对单通道各向同性 TV
有标准形式),破坏"两者改动完全对称"这条约束,且是本任务范围外的更大改动。
"""
from __future__ import annotations

import torch

from crisp.metrics.coil_domain import fft3c, ifft3c

_EPS = 1e-8  # 数值下限,只用于避免 x=0 处 x/|x| 出现 0/0;与下面的 Huber 平滑参数 eps 无关

_STEP_MARGIN = 0.5
# `1/(1+lam/eps)` 里的 `lam/eps` 只是 Huber-TV 项 Lipschitz 常数的近似(brief 原文措辞
# 也是"约"),不是证明过的紧上界:真正的曲率上界是 `||D||^2 * lam/eps`(D 是离散梯度+
# 散度算子,各向同性 3 轴迭加的 ||D||^2 理论上可到 ~12,而不是 1)。这在真实尺度
# (64^3、8 线圈、AF12、300 迭代、lam 网格 1e-3..4e-2)上是可测的,不是假设:
#   1) 把 FISTA 换成不带动量的纯梯度下降,同一个 step=1/(1+lam/eps),目标函数和脑内
#      PSNR 在 50/150/300 次迭代上都严格单调(objective 5.54e6→1.69e6→8.55e5,
#      psnr 21.64→22.61→23.22dB)——说明这个 step 对纯梯度下降是安全的。
#   2) 换回 FISTA(带动量)后,同一个 step 在多个 lam 上目标函数和 PSNR 都变成非单调
#      (例如 lam=6.3e-3 时 objective 4.64e5→6.27e5、psnr 23.40→21.97dB,150→300 次
#      迭代之间回升),但从未跌破零填充——这正是 FISTA 动量本身没有逐步单调保证的
#      经典性质(Beck & Teboulle 2009),不是原始 bug 的"未采样零空间无界随机游走"。
# 加 0.5x 安全系数后,同一批 lam 在 iters=300 时相对零填充的最坏余量从 ~0.26dB
# (margin=1.0, lam=4e-2)提升到 ~1.3dB(margin=0.5),波动幅度也从 ~1.9dB 降到
# ~0.8dB 量级。没有做到逐迭代严格单调(那是动量的固有代价,消掉它需要重新设计成
# 带目标函数回退的 monotone-FISTA,超出本轮修复范围),但幅度和方向都和原始 bug
# 的性质不同——不是"未采样零空间的无界随机游走",而是"围绕一个稳定的正则化不动点
# 有界波动"。见 task-6-report.md fix round 2 的完整验证数据。


def _grads(mag):
    gx = torch.zeros_like(mag); gy = torch.zeros_like(mag); gz = torch.zeros_like(mag)
    gx[:-1] = mag[1:] - mag[:-1]
    gy[:, :-1] = mag[:, 1:] - mag[:, :-1]
    gz[:, :, :-1] = mag[:, :, 1:] - mag[:, :, :-1]
    return gx, gy, gz


def _tv_grad(x, coupled_norm=None, eps: float = 1e-3):
    """Huber 平滑 TV 对复数 x 的梯度(经由 |x|)。coupled_norm 给出跨对比度归一化项。

    前向差分 g_j = m_{j+1} - m_j(末位补 0)。对 sum_j sqrt(g_j^2+eps^2) 关于 m_j 求导得
    u_j = g_j/sqrt(g_j^2+eps^2),d/dm_j = u_{j-1} - u_j,故散度按
    `div[1:] += u[:-1]; div[:-1] -= u[:-1]` 累加。最后经由 d|x|/dx = x/|x| 传回复数域。

    `eps` 是 Huber 平滑半径(不是数值下限):越大越光滑、Lipschitz 常数 ~1/eps 越小;
    越小越接近真 TV(非光滑,收敛性丢失,参见模块顶部 fix round 2 说明)。`coupled_norm`
    不为 None 时(联合版)按调用方已经用同一个 eps 算好的共享范数,本函数不再重算。
    """
    mag = x.abs().clamp_min(_EPS)
    gx, gy, gz = _grads(mag)
    n = coupled_norm if coupled_norm is not None else torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2 + eps ** 2)
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
             eps: float = 1e-3, step: float | None = None) -> torch.Tensor:
    """FISTA CS-SENSE:最小化 (1/2)||P_mask F S x - y||^2 + lam * TV_smooth(x)。

    `mask` 须为严格 0/1 的欠采样投影。`eps`(默认 1e-3)是 TV 的 Huber 平滑半径,假定
    `y`/`csm` 已按 P99.9 归一化到量级 1(见模块顶部说明)。`step` 缺省时按 Lipschitz
    匹配公式 `_STEP_MARGIN/(1+lam/eps)` 自动计算(数据项 L<=1 + TV 项 Huber Lipschitz
    常数 ~lam/eps,`_STEP_MARGIN=0.5` 是从真实尺度测量中得到的安全系数,理由见模块
    顶部 `_STEP_MARGIN` 旁的注释);显式传入 `step` 会覆盖该公式。该公式依赖
    `sum_c|csm_c|^2` 逐点 <=1(见模块顶部注释);违反这条假设(例如线圈图用了别的缩放
    约定,或像 Task 4 实测的真实线圈图那样某些体素灵敏度和显著偏离 1)会被
    `_check_csm_scale` 检测到并抛出 `ValueError`——本函数不会替调用者静默重新缩放数据,
    需要调用方自己重新归一化线圈图,或显式传入更小的 `step`。
    """
    if device is not None:
        y, csm, mask = y.to(device), csm.to(device), mask.to(device)
    _check_csm_scale(csm)
    if step is None:
        step = _STEP_MARGIN / (1.0 + lam / eps)
    x = (ifft3c(mask.unsqueeze(0) * y) * csm.conj()).sum(0)
    z, t = x.clone(), 1.0
    for _ in range(iters):
        g = _data_grad(z, y, csm, mask) + lam * _tv_grad(z, eps=eps)
        x_new = z - step * g
        t_new = 0.5 * (1 + (1 + 4 * t * t) ** 0.5)
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return x


def fista_joint_tv(ys, csms, masks, lam: float, iters: int = 200, device: str | None = None,
                   eps: float = 1e-3, step: float | None = None) -> list[torch.Tensor]:
    """联合多对比度 FISTA CS-SENSE,通过 vectorial TV 的共享梯度范数耦合各通道。

    每个通道各自的 `mask` 须为严格 0/1 投影。`eps`/`step` 语义同 `fista_tv`(同一 Huber
    平滑半径、同一 Lipschitz 匹配步长公式,应用到共享的跨对比度归一化范数上)——独立版和
    联合版必须用完全相同的 eps/step 处理,否则两者的比较会被混淆(参见模块顶部说明)。
    任一通道违反 `sum_c|csm_c|^2` 逐点 <=1 都会被 `_check_csm_scale` 检测到并抛出
    `ValueError`,不会静默重新缩放数据。
    """
    if device is not None:
        ys = [v.to(device) for v in ys]; csms = [v.to(device) for v in csms]
        masks = [v.to(device) for v in masks]
    for c in csms:
        _check_csm_scale(c)
    if step is None:
        step = _STEP_MARGIN / (1.0 + lam / eps)
    xs = [(ifft3c(m.unsqueeze(0) * y) * c.conj()).sum(0) for y, c, m in zip(ys, csms, masks)]
    zs, t = [v.clone() for v in xs], 1.0
    for _ in range(iters):
        acc = None
        for z in zs:
            gx, gy, gz = _grads(z.abs().clamp_min(_EPS))
            s = gx ** 2 + gy ** 2 + gz ** 2
            acc = s if acc is None else acc + s
        n = torch.sqrt(acc + eps ** 2)                       # 跨对比度共享的归一化
        new_xs = []
        for z, y, c, m in zip(zs, ys, csms, masks):
            g = _data_grad(z, y, c, m) + lam * _tv_grad(z, coupled_norm=n)
            new_xs.append(z - step * g)
        t_new = 0.5 * (1 + (1 + 4 * t * t) ** 0.5)
        zs = [xn + ((t - 1) / t_new) * (xn - xo) for xn, xo in zip(new_xs, xs)]
        xs, t = new_xs, t_new
    return xs
