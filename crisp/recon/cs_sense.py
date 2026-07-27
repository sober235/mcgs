"""CS-SENSE:FISTA + 各向同性 TV(独立)与 vectorial TV(联合多对比度)。

Huber 型平滑 TV:TV_smooth(x) = sum_voxel sqrt(|grad|x||^2 + eps^2)。`eps` 是一个真正的
光滑化参数(不是数值下限),默认 1e-3,假定输入图像已按 P99.9 归一化到量级 1(参见
`crisp/io.py` 的 `normalize_p999`)。在该假设下 TV_smooth 处处可微,Hessian 上界为
`_GRAD_OP_SQ * lam/eps`;FISTA 步长取教科书的 `1/L`,即
`step = 1/(1 + _GRAD_OP_SQ*lam/eps)`(推导与实测见下方常量旁的注释;数据项算子范数
L<=1,依赖 `sum_c|csm_c|^2<=1`,见 `_check_csm_scale`)。
联合版的耦合仅通过 vectorial TV 的跨对比度梯度范数实现:
    TV_joint = sum_voxel sqrt( sum_m sum_axis |grad_axis |x_m| |^2 + eps^2 )

本模块最小化的目标函数是(注意 1/(2N) 这个系数,N = 体素数):
    J(x) = (1/(2N))*||M(F S x - y)||^2 + lam * TV_smooth(x)
`_data_grad` 用的是 `ifft3c` = F^{-1} = (1/N)F^H(torch 默认 norm="backward"),所以它是
上式的梯度,**不是** `(1/2)||M(F S x - y)||^2` 的梯度。任何监控/早停逻辑都必须用这个
带 1/(2N) 的 J,否则监控的根本不是本迭代在最小化的函数(fix round 3 就踩了这个坑)。

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

fix round 4(2026-07-27):步长规则里漏了离散梯度算子范数 ||D||^2(见 `_GRAD_OP_SQ`),
已补上。**同时推翻 fix round 3 的"求解器仍在发散"结论**:round 2/3 监控的"目标函数"是
`0.5*||M(FSx-y)||^2 + lam*TV`,少了上面那个 1/(2N),等于把数据项相对 TV 项放大了
N=262144 倍——它跟踪的基本上只是数据残差,而 CS 迭代里数据残差本来就可以在 TV 下降的
同时上升,那不是发散。用正确的 J 逐迭代(不是每 50 步取样)复测:旧步长(1/L 的 5.25 倍)
在 600 次迭代里 0 次上升,新步长同样 0 次上升——从来没有发散过。round 3 观察到的
PSNR 见顶回落是**半收敛(semi-convergence)**,重建质量其实来自早停,改步长不会、也不该
消除这个现象。

⚠️ 真正的缺陷不在求解器,而在目标函数本身(fix round 4 实测,详见 task-6-report.md §6):
TV 只惩罚 |x|,**相位完全不受约束**。x 有 2N 个实自由度,AF12/8 线圈只提供约 0.73N 个
实测量,所以把 |x| 压成常数(去掉 N 个自由度)之后,剩下的 N 个相位自由度仍然够拟合数据。
优化器正是这么做的:64^3 体模上 lam>=1e-3 收敛解的 TV = 262.2,而"幅度全平"的下界是
N*eps = 262.1——幅度图被压成常数(且均值 1.30,比 GT 的 0.99 还大,不是被缩小),结构
全被塞进相位,数据项却拟合到 0.167(零填充是 75.4)。目标值上
J(退化解)=1.82 << J(真值)=85.8,**真值比退化解差 47 倍**:模型本身就偏好这个解,任何
步长/迭代数/停机准则都改不了这个排序。修法是把 TV 改成惩罚复图像(如逐体素
sqrt(|grad Re|^2+|grad Im|^2+eps^2),CS-MRI 标准做法),外加把数据项(逐体素均值)和
TV 项(逐体素求和)的权重统一成与 N 无关——两者都超出本轮范围,未实施。
在此之前,fista_tv/fista_joint_tv 的任何结果都只是早停产物,引用时必须如实标注。
"""
from __future__ import annotations

import torch

from crisp.metrics.coil_domain import fft3c, ifft3c

_EPS = 1e-8  # 数值下限,只用于避免 x=0 处 x/|x| 出现 0/0;与下面的 Huber 平滑参数 eps 无关

_GRAD_OP_SQ = 12.0
# Huber-TV 项 Hessian 上界里的离散梯度算子范数平方 ||D||^2,D = `_grads` 的三轴前向差分。
# 推导:逐体素 phi(g)=sqrt(|g|^2+eps^2)(g 是该体素的 3 分量差分向量)的 Hessian 为
# (1/s)(I - g g^T/s^2),s=sqrt(|g|^2+eps^2),最大特征值 1/s <= 1/eps;整个 TV 项的
# Hessian = D^T blkdiag(Hess phi) D,故 ||Hess|| <= ||D||^2 / eps。单轴前向差分(末位补 0)
# 是 D_1 = S - I(S 为截断平移,||S||<=1),所以 ||D_1||<=2、||D_1||^2<=4;三轴堆叠
# ||D||^2 <= 3*4 = 12。加上数据项的 L<=1,得 L_total <= 1 + 12*lam/eps。
# fix round 4 幂迭代实测(float64,详见 task-6-report.md fix round 4):
#   ||D||^2 = 11.967(n=64),11.965(n=32) —— 上界 12 基本达到,不是保守估计;
#   Huber-TV 幅度域 Hessian 的 lambda_max:在真实工作点 m=|零填充解|(64^3 体模)处
#     11767,在最坏点 m=const 处 11918,上界 12/eps=12000 —— 达到 98%,**紧**;
#   联合 vectorial TV(M=2)同一上界,实测 11943:因为堆叠算子 D_total=blkdiag(D,..,D)
#     的范数就等于 ||D||,所以两个入口共用同一条 step 规则,arm-to-arm 严格可比;
#   数据项 Hessian S^H U^H M U S(U=F/sqrt(N) 酉)实测范数 0.9992 <= 1。
# 因此步长取教科书的 1/L,不再乘任何经验安全系数(round 2/3 的 `_STEP_MARGIN=0.5` 是在
# 漏掉 ||D||^2 的情况下用来"兜底"的经验因子,补上 12 之后它既无理论依据、又白白让收敛
# 慢一倍,已删除)。注意 lam=6.3e-3/eps=1e-3 时旧规则给出的 step 是 1/L 的 5.25 倍——
# 但实测它并没有发散(见模块顶部 fix round 4 说明),只是没有任何收敛保证:上界在起点
# 就已经被打满,旧步长能活下来靠的是轨迹很快进入 |grad m| >> eps 的低曲率区,是运气不是
# 保证。换一份更平坦的图像(|grad m| <~ eps 占比更大)就会真的踩到 12/eps 这个上界。


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

    `eps` 是 Huber 平滑半径(不是数值下限):越大越光滑、Hessian 上界 ||D||^2/eps 越小;
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
    """FISTA CS-SENSE:最小化 (1/(2N))||P_mask F S x - y||^2 + lam * TV_smooth(x),N=体素数。

    1/(2N) 不是笔误:`_data_grad` 走的是 `ifft3c`=F^{-1}=(1/N)F^H,梯度对应的就是这个
    带 1/(2N) 的数据项(等价于把 F 换成酉的 F/sqrt(N) 后的标准最小二乘)。见模块顶部。

    `mask` 须为严格 0/1 的欠采样投影。`eps`(默认 1e-3)是 TV 的 Huber 平滑半径,假定
    `y`/`csm` 已按 P99.9 归一化到量级 1(见模块顶部说明)。`step` 缺省时按 Lipschitz
    匹配公式 `1/(1 + _GRAD_OP_SQ*lam/eps)` 自动计算(数据项 L<=1 + TV 项 Huber Hessian
    上界 `||D||^2*lam/eps`,`_GRAD_OP_SQ=||D||^2=12` 的推导与实测见模块顶部该常量旁的
    注释);显式传入 `step` 会覆盖该公式。该公式依赖
    `sum_c|csm_c|^2` 逐点 <=1(见模块顶部注释);违反这条假设(例如线圈图用了别的缩放
    约定,或像 Task 4 实测的真实线圈图那样某些体素灵敏度和显著偏离 1)会被
    `_check_csm_scale` 检测到并抛出 `ValueError`——本函数不会替调用者静默重新缩放数据,
    需要调用方自己重新归一化线圈图,或显式传入更小的 `step`。
    """
    if device is not None:
        y, csm, mask = y.to(device), csm.to(device), mask.to(device)
    _check_csm_scale(csm)
    if step is None:
        step = 1.0 / (1.0 + _GRAD_OP_SQ * lam / eps)
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
        step = 1.0 / (1.0 + _GRAD_OP_SQ * lam / eps)
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
