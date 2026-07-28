"""CS-SENSE:FISTA + 各向同性**复数** TV(独立)与 vectorial 复数 TV(联合多对比度)。

本模块最小化的目标函数(两个权重都与体积尺寸解耦,见 `_term_weights`):

    J(x) = 1/(2*n_samp) * ||M(U S x - y/sqrt(N))||^2  +  (lam/N) * TV_c(x)
    TV_c(x) = sum_voxel sqrt( sum_a |d_a x|^2 + eps^2 )      # d_a 保持复数,不取模

其中 U = F/sqrt(N) 是酉 FFT、n_samp = mask.sum()(已采样 k 空间点数)、N = 体素数。
`eps` 是 Huber 平滑半径(默认 1e-3,假定图像已按 P99.9 归一化到量级 1,参见
`crisp/io.py` 的 `normalize_p999`)。TV_c 是**凸**的(线性算子 + 范数的复合),所以整个
J 是凸的,FISTA 的收敛理论在此真正适用——这一点是 round 5 相对之前所有版本的实质变化。

联合版的耦合通过 vectorial TV 在**复数**梯度上跨对比度共享归一化实现:
    TV_joint = sum_voxel sqrt( sum_m sum_a |d_a x_m|^2 + eps^2 )
两个入口的 TV 形式、权重、步长规则逐字对称,联合/独立的对比才不会被混淆。

步长取教科书的 1/L(推导与实测见 `_GRAD_OP_SQ` 旁的注释):
    L <= 1/n_samp + _GRAD_OP_SQ * (lam/N) / eps

---- 修复历史(每一轮都推翻了上一轮的判决,按时间倒序) ----

fix round 5(2026-07-28,更正见 plans/2026-07-27-crisp-phase0.md「Task 6 更正之二」):
**TV 从作用于 |x| 改为作用于复数 x,并把两项权重与尺寸解耦。**
(a) 退化性:旧的 TV(|x|) 完全不惩罚相位,而 AF=12 下相位自由度(N 个实自由度)就足以
    独自拟合已采样数据(约 0.73N 个实测量),于是极小点把幅值抹平、把结构藏进相位——
    实测收敛后 TV=262.2 vs 全平下界 N*eps=262.1,平均 |x| 1.30 vs GT 0.99,
    J(退化解)=1.82 << J(真值)=85.8。复数 TV 让相位付出代价,退化解消失。
(b) 权重:旧式 `(1/(2N))||.||^2 + lam*sum(TV)` 的等效正则强度是 N*lam,64^3 上
    lam=6.3e-3 相当于 1651、256x220x256 上相当于 9.1e4——同一个 lam 在不同矩阵尺寸上
    差 55 倍。现在数据项按 n_samp 归一、TV 按体素数 N 归一(即更正之二的字面要求)。
    实测验证:同一连续体模渲染到 32^3 / 64^3 / 96x80x96(N 跨 22 倍,且含一个非立方
    体积,因为真实目标是 256x220x256),三种归一化先在 n=64 上校准到同一个 w_tv,再固定
    lam 改变尺寸,看 TV(recon)/TV(GT) 的跨尺寸 max/min:
        sum(rounds 1-4) 1.141 | /N^(2/3) 1.048 | **/N 1.007**
    本轮一度按"GT 的 TV/N 随 1/n 漂移"改用 N^(2/3),被上面这组实测否掉了——决定极小点的
    是 TV 项与数据项的**梯度平衡**(见 `_term_weights`),不是两者的数值大小。

fix round 4(2026-07-27):补上步长规则里漏掉的 ||D||^2(见 `_GRAD_OP_SQ`),**并推翻
round 3 的"求解器发散"判决**:round 2/3 监控的是 `0.5*||M(FSx-y)||^2+lam*TV`,而当时
`_data_grad` 实为 `(1/(2N))||.||^2+lam*TV` 的梯度(`ifft3c=(1/N)F^H` 被当成了伴随 F^H),
监控量根本不是被最小化的函数。按正确的 J 逐迭代复测:旧步长(1/L 的 5.25 倍)600 次
迭代 0 次上升——从未发散。见到的 PSNR 见顶回落是上面 (a) 的退化在起作用。

fix round 2(2026-07-27):`eps` 曾硬编码为 1e-8,使 `u = g/sqrt(g^2+eps^2) ~ sign(g)`,
TV 项名义光滑、实为非光滑项的次梯度,包在 FISTA 壳里没有收敛保证。改为真正的 Huber
平滑(`eps` 提升为函数参数,默认 1e-3)并同步步长。
"""
from __future__ import annotations

import torch

from crisp.metrics.coil_domain import fft3c, ifft3c

_EPS = 1e-8
# fix round 5 起本模块内部**不再使用** `_EPS`:复数 TV 的梯度不经过 d|x|/dx = x/|x|,
# 也就没有 x=0 处 0/0 的问题。之所以保留这个名字,是因为 `crisp/experiments/g2_voxel.py`
# 有 `from crisp.recon.cs_sense import _EPS, ...`,而该文件不在本任务允许改动的清单里,
# 删掉会让它直接 ImportError。那个脚本本来就已经过时(它自带一份 FISTA 循环的副本),
# 应当改成直接调用 fista_tv/fista_joint_tv,详见 task-6-report.md。

_GRAD_OP_SQ = 12.0
# TV 项 Hessian 上界里的离散梯度算子范数平方 ||D||^2,D = `_grads` 的三轴前向差分。
# 推导:逐体素 phi(g)=sqrt(|g|^2+eps^2) 的 Hessian 为 (1/s)(I - g g^T/s^2),
# s=sqrt(|g|^2+eps^2),最大特征值 1/s <= 1/eps;整个 TV 项的 Hessian =
# D^T blkdiag(Hess phi) D,故 ||Hess_TV|| <= ||D||^2 / eps。单轴前向差分(末位补 0)是
# D_1 = S - I(S 为截断平移,||S||<=1),所以 ||D_1||^2<=4;三轴堆叠 ||D||^2 <= 12。
# 精确地,单轴 D_1^T D_1 的特征值是 4 sin^2(m*pi/(2n)),三轴可交换,故
# ||D||^2 = 12 sin^2((n-1)*pi/(2n)) < 12,且 n -> inf 时趋于 12。
#
# fix round 5:**这个常数在复数 TV 下不变**。d_a x 现在是复数,但 D 仍是同一个实矩阵,
# 只是同时作用在实部和虚部上(R^{2N} 上的 blkdiag(D,D)),范数不变;逐体素的 g 从 R^3
# 变成 C^3 ~ R^6,phi 的 Hessian 上界仍是 1/eps。实测(幂迭代,float64):
#   ||D||^2 = 11.885(n=16,与闭式 12 sin^2(15pi/32)=11.8846 吻合)、11.967(n=64);
#   复数 TV Hessian 的 lambda_max(eps=1e-3,上界 12/eps=12000):最坏点 x=const 处
#     11946.8、|dx|<<eps 的小随机点 11936.7、真实工作点 x=零填充解处 11771.9
#     —— 达到 98-99.6%,**紧**;而 |dx|>>eps 的随机点只有 15.8(曲率随边缘出现而塌缩);
#   联合 vectorial 复数 TV(M=2)同一上界,实测 11949.2 —— 堆叠算子 blkdiag(D,..,D) 的
#     范数就等于 ||D||,所以两个入口共用同一条 step 规则,arm-to-arm 严格可比。
# 与 round 4 的幅度 TV 相比,复数 TV 还额外消除了 x -> TV(|x|) 在 x=0 处的不可微点
# (round 4 在零填充点幂迭代测到 ~2.2e7 的伪曲率,现在同一点是 11771.9 的真实特征值),
# 现在 J 处处可微且**凸**。梯度已用方向有限差分核验到 8 位有效数字(见 fix round 5 报告)。
# 步长取教科书的 1/L,不乘任何经验安全系数。


def _grads(x):
    """三轴前向差分,末位补 0。对实数和复数张量同样适用(round 5 起传入的是复数 x)。"""
    gx = torch.zeros_like(x); gy = torch.zeros_like(x); gz = torch.zeros_like(x)
    gx[:-1] = x[1:] - x[:-1]
    gy[:, :-1] = x[:, 1:] - x[:, :-1]
    gz[:, :, :-1] = x[:, :, 1:] - x[:, :, :-1]
    return gx, gy, gz


def _term_weights(mask, numel: int, lam: float):
    """数据项/TV 项的尺寸无关权重,以及与之匹配的 1/L 步长。

    w_data = 1/n_samp 把数据项变成"每个已采样 k 空间点的均方残差"(在酉 FFT 约定下)。
    w_tv   = lam/N 让 TV 项与数据项同阶:`_data_grad` 化简后是 S^H U^H M(U S x - y/sqrt(N)),
    逐体素量级 O(残差);而 TV 梯度 D^T u 的逐体素量级 O(1)(|u|<=1,至多 6 项)。两边
    要在每个体素上可比,就要求 w_tv 与 w_data 同阶,而 w_data = 1/n_samp 正比于 1/N
    (固定采样率下 n_samp = frac*N)。

    实测(见 task-6-report.md fix round 5:同一连续体模渲染到 32^3 / 64^3 / 96x80x96,
    N 跨 22 倍且含一个非立方体积,三种归一化都先在 n=64 上校准到同一个 w_tv):
        TV(recon)/TV(GT) 跨尺寸的 max/min —— sum(旧): 1.141;/N^(2/3): 1.048;**/N: 1.007**
    /N 明显最好。注意"GT 的 TV/N 会随 1/n 漂移"这个观察**不能**用来选归一化:决定极小点
    的是 TV 项与数据项的**梯度平衡**,不是两者的数值大小,round 5 一度按后者选了 N^(2/3),
    被上面这组实测否掉了。
    """
    w_data = 1.0 / float(mask.sum())
    w_tv = lam / float(numel)
    return w_data, w_tv


def _tv_norm(gx, gy, gz, eps: float):
    """逐体素的复数梯度模 sqrt(sum_a |d_a x|^2 + eps^2)(实数张量)。"""
    return torch.sqrt(gx.abs() ** 2 + gy.abs() ** 2 + gz.abs() ** 2 + eps ** 2)


def _tv_grad(x, coupled_norm=None, eps: float = 1e-3):
    """各向同性**复数** TV 对 x 的梯度。coupled_norm 给出跨对比度共享的归一化项。

    TV_c(x) = sum_j phi_j,phi_j = sqrt(sum_a |g_{a,j}|^2 + eps^2),
    g_{a,j} = x_{j+e_a} - x_j 为**复数**前向差分(末位补 0)。
    沿方向 h 求方向导数:
        dTV/dt = sum_j (1/phi_j) sum_a Re( conj(g_{a,j}) * (h_{j+e_a} - h_j) )
               = sum_a Re< D_a^T u_a , h >,   u_a = g_a / phi
    其中 (D_a^T u)_k = u_{k-1} - u_k,即下面的 `div` 累加(与 round 1-4 的散度写法逐字
    相同,符号未变)。所以梯度就是 div 本身——round 4 之前那个 `* (x/|x|)` 的链式因子
    随幅度 TV 一起消失了,x=0 处的不可微点也随之消失。

    实数输入时(TV 只作用在幅度上)本函数退化为旧行为;但本模块现在一律传入复数 x。

    `eps` 是 Huber 平滑半径:越大越光滑、Hessian 上界 ||D||^2/eps 越小;越小越接近真 TV
    (非光滑,收敛性丢失)。`coupled_norm` 不为 None 时(联合版)用调用方已经算好的共享
    范数,本函数不再重算。
    """
    gx, gy, gz = _grads(x)
    n = coupled_norm if coupled_norm is not None else _tv_norm(gx, gy, gz, eps)
    ux, uy, uz = gx / n, gy / n, gz / n
    div = torch.zeros_like(x)
    div[1:] += ux[:-1];        div[:-1] -= ux[:-1]
    div[:, 1:] += uy[:, :-1];  div[:, :-1] -= uy[:, :-1]
    div[:, :, 1:] += uz[:, :, :-1]; div[:, :, :-1] -= uz[:, :, :-1]
    return div


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
    """FISTA CS-SENSE。最小化(权重与体积尺寸解耦,详见模块顶部):

        J(x) = 1/(2*n_samp) * ||M(U S x - y/sqrt(N))||^2 + (lam/N) * TV_c(x)

    n_samp = mask.sum(),N = 体素数,U = F/sqrt(N) 酉 FFT,TV_c 为各向同性**复数** TV。
    实现上 `w_data * _data_grad(...)` 正好是第一项的梯度:`_data_grad` 走 `ifft3c` =
    F^{-1} = (1/N)F^H,自带一个 1/N,因此它已经是 `(1/(2N))||M(FSx-y)||^2` 的梯度,
    再乘 1/n_samp 就得到上式。任何监控/早停逻辑都必须用这个 J(round 3 曾因为监控了
    别的函数而误判发散)。

    `mask` 须为严格 0/1 的欠采样投影。`eps`(默认 1e-3)是 TV 的 Huber 平滑半径,假定
    `y`/`csm` 已按 P99.9 归一化到量级 1(见模块顶部说明)。`step` 缺省时按 1/L 自动计算,
    `L = w_data + _GRAD_OP_SQ * w_tv / eps`(推导与实测见 `_GRAD_OP_SQ` 旁注释);显式传入
    `step` 会覆盖该公式。该公式依赖 `sum_c|csm_c|^2` 逐点 <=1;违反这条假设(例如线圈图用了
    别的缩放约定,或像 Task 4 实测的真实线圈图那样某些体素灵敏度和显著偏离 1)会被
    `_check_csm_scale` 检测到并抛出 `ValueError`——本函数不会替调用者静默重新缩放数据,
    需要调用方自己重新归一化线圈图,或显式传入更小的 `step`。
    """
    if device is not None:
        y, csm, mask = y.to(device), csm.to(device), mask.to(device)
    _check_csm_scale(csm)
    x = (ifft3c(mask.unsqueeze(0) * y) * csm.conj()).sum(0)
    w_data, w_tv = _term_weights(mask, x.numel(), lam)
    if step is None:
        step = 1.0 / (w_data + _GRAD_OP_SQ * w_tv / eps)
    z, t = x.clone(), 1.0
    for _ in range(iters):
        g = w_data * _data_grad(z, y, csm, mask) + w_tv * _tv_grad(z, eps=eps)
        x_new = z - step * g
        t_new = 0.5 * (1 + (1 + 4 * t * t) ** 0.5)
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return x


def fista_joint_tv(ys, csms, masks, lam: float, iters: int = 200, device: str | None = None,
                   eps: float = 1e-3, step: float | None = None) -> list[torch.Tensor]:
    """联合多对比度 FISTA CS-SENSE,通过 vectorial **复数** TV 的共享梯度范数耦合各通道。

        J(xs) = sum_m 1/(2*n_samp_m) * ||M_m(U S_m x_m - y_m/sqrt(N))||^2
                + (lam/N) * sum_voxel sqrt( sum_m sum_a |d_a x_m|^2 + eps^2 )

    与 `fista_tv` 完全平行:同一个复数 TV、同一套权重、同一条 1/L 步长规则,只是归一化项
    `coupled_norm` 跨对比度共享——独立版和联合版必须逐字对称,否则两者的比较会被混淆
    (参见模块顶部说明)。步长里的数据项曲率取各通道的最大值 `max_m w_data_m`(堆叠后的
    数据 Hessian 是块对角,范数即各块最大者);各通道 mask 相同时与独立版给出同一个 step。

    每个通道各自的 `mask` 须为严格 0/1 投影。任一通道违反 `sum_c|csm_c|^2` 逐点 <=1 都会被
    `_check_csm_scale` 检测到并抛出 `ValueError`,不会静默重新缩放数据。
    """
    if device is not None:
        ys = [v.to(device) for v in ys]; csms = [v.to(device) for v in csms]
        masks = [v.to(device) for v in masks]
    for c in csms:
        _check_csm_scale(c)
    xs = [(ifft3c(m.unsqueeze(0) * y) * c.conj()).sum(0) for y, c, m in zip(ys, csms, masks)]
    ws = [_term_weights(m, xs[0].numel(), lam) for m in masks]
    w_datas = [w[0] for w in ws]
    w_tv = ws[0][1]                                          # 只依赖 lam 与体素数,各通道相同
    if step is None:
        step = 1.0 / (max(w_datas) + _GRAD_OP_SQ * w_tv / eps)
    zs, t = [v.clone() for v in xs], 1.0
    for _ in range(iters):
        acc = None
        for z in zs:
            gx, gy, gz = _grads(z)
            s = gx.abs() ** 2 + gy.abs() ** 2 + gz.abs() ** 2
            acc = s if acc is None else acc + s
        n = torch.sqrt(acc + eps ** 2)                       # 跨对比度共享的归一化
        new_xs = []
        for z, y, c, m, wd in zip(zs, ys, csms, masks, w_datas):
            g = wd * _data_grad(z, y, c, m) + w_tv * _tv_grad(z, coupled_norm=n)
            new_xs.append(z - step * g)
        t_new = 0.5 * (1 + (1 + 4 * t * t) ** 0.5)
        zs = [xn + ((t - 1) / t_new) * (xn - xo) for xn, xo in zip(new_xs, xs)]
        xs, t = new_xs, t_new
    return xs
