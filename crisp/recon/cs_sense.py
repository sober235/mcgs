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

---- 调用方必读:实测的工作点事实(fix round 6) ----

以下都是 64^3 / 8 线圈 / AF~12 合成体模上的**实测值**,不是估计:

* **平台从约 1500 次迭代开始。** lam=6.3e-3 时脑内 PSNR 在 600-2500 之间的极差是
  0.306 dB(600 步那点偏高),从 1500 步起只剩 0.027 dB。要做 arm-to-arm 比较,
  **迭代数取 >=1500**,否则比的是各自处在收敛暂态的哪个位置。300 步恰好是 2500 步内
  PSNR 的全局最大点(暂态过冲),用 300 步会系统性地高估约 1 dB。
* **lam 的最优点远小于 round 5 报告里写的可用区间 [6.3e-3, 3]。** 1500 步实测:
      lam    1e-5     3e-5     1e-4     3e-4     1e-3    6.3e-3
      PSNR  32.77    39.78    38.19    37.64    36.75    34.65   (零填充 20.41)
  最优在 lam~3e-5 附近(+19.4 dB),而不是区间下端的 6.3e-3(+14.2 dB);小 lam 端没有
  任何退化迹象(TV/TV_gt 0.98-1.02)。**调参必须往 1e-5..1e-3 扫,不要照抄那个区间。**
* **复数 TV 会惩罚平滑的背景相位**,而真实数据有背景相位、本体模没有。给体模加上
  1-6 个周期的平滑背景相位后,同一 lam 下相对零填充的余量从 +14.2 dB 掉到 +6.8..+9.6 dB:
      背景相位周期数   0      1      2      4      6
      相对零填充余量 +14.24  +9.61  +8.03  +6.76  +7.51 dB
  **+14.2 dB 这个数字绝不能拿去代表真实数据。** 真实数据上应先做相位解调,或在相位
  校正后的图像上做 TV。

---- 修复历史(每一轮都推翻了上一轮的判决,按时间倒序) ----

fix round 6(2026-07-28):四处更正,都不改变数值行为(除了删除 `_EPS`):
(1) 加了一条真正钉住"质量不随迭代退化"的测试(此前只有报告里的一次性 ablation:
    出厂回归测试跑 300 步,而 300 步恰是 PSNR 全局最大点,把 `_tv_grad` 换回 round 4 的
    幅度 TV 它依然能过)。
(2) **更正 round 5 关于权重的因果说法。** round 5 声称旧权重"同一 lam 在不同尺寸上差
    55 倍"、并用一组 max/min=1.141 的"rounds 1-4"对照支持它——那组对照是**混合体**
    (新的 1/n_samp 数据权重 + 旧的常数 TV 权重),代码里从来没有存在过。真实的
    rounds 1-4 是 `w_data=1, w_tv=lam`。决定极小点的不变量是
        rho = w_tv / w_data
    rounds 1-4:rho = lam(与尺寸、与 AF 都无关);round 5:rho = lam*n_samp/N =
    lam*采样率 ~ lam/AF。**两者都是尺寸可迁移的**;实测在 rho 对齐后两种权重给出几乎
    逐位相同的结果(32^3/64^3/96x80x96 上 TV/TV_gt 分别 0.9398/0.9715/0.9713 与
    0.9402/0.9715/0.9724,跨尺寸 max/min 都是 1.034)。所以 /N 仍是"给定 1/n_samp 数据
    权重后"的正确选择,但它修的不是尺寸迁移 bug。**代价**:round 5 的 rho 里多了一个
    采样率因子,所以 **lam 在固定 AF 下跨矩阵尺寸可迁移,一旦改 AF 或 ACS 就要重标**。
    (另注:round 5 的 rho 还会随 mask 实际采样率的抖动而抖动,上表 frac 0.0823-0.0911,
     旧权重的 rho 则是严格常数。)
(3) 在 `fista_joint_tv` 的 docstring 里写明 vectorial TV 的 1/sqrt(M) 效应,以及
    "两个 arm 不能在同一 lam 下直接比"。
(4) 删除 `_EPS`(复数 TV 后本模块已不用它;它此前只为让 `crisp/experiments/g2_voxel.py`
    的 import 不炸而留着,而那个脚本会**静默**算出幅度 TV 的旧模型——让它 ImportError
    炸掉比让它出错误数字好)。

fix round 5(2026-07-28,更正见 plans/2026-07-27-crisp-phase0.md「Task 6 更正之二」):
**TV 从作用于 |x| 改为作用于复数 x,并把两项权重与尺寸解耦。**
(a) 退化性:旧的 TV(|x|) 完全不惩罚相位,而 AF=12 下相位自由度(N 个实自由度)就足以
    独自拟合已采样数据(约 0.73N 个实测量),于是极小点把幅值抹平、把结构藏进相位——
    实测收敛后 TV=262.2 vs 全平下界 N*eps=262.1,平均 |x| 1.30 vs GT 0.99,
    J(退化解)=1.82 << J(真值)=85.8。复数 TV 让相位付出代价,退化解消失。
(b) 权重:数据项按 n_samp 归一、TV 按体素数 N 归一(更正之二的字面要求,人类裁定)。
    这是**量纲/可读性**上的改进(两项都成为各自索引集上的均值),**不是**在修一个尺寸
    迁移 bug——见下面 fix round 6 (2) 的更正。

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

# fix round 6:`_EPS = 1e-8` 已删除。复数 TV 的梯度不经过 d|x|/dx = x/|x|,本模块不再需要
# 这个数值下限。round 5 曾只为了让 `crisp/experiments/g2_voxel.py` 的
# `from crisp.recon.cs_sense import _EPS, ...` 不炸而保留它——但那个脚本自带一份 FISTA 循环
# 副本,会用 `_grads(x.abs())` **静默**算出 round 4 的幅度 TV(即已被证明退化的旧模型)。
# 让它在 import 处大声 ImportError,好过让它安静地产出错误数字。修法是把它改成直接调用
# fista_tv / fista_joint_tv,详见 task-6-report.md fix round 6。

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

    w_data = 1/n_samp 把数据项变成"每个已采样 k 空间点的均方残差"(在酉 FFT 约定下);
    w_tv   = lam/N 把 TV 项变成"每体素的平均 TV"。两者都是各自索引集上的均值,所以 lam
    是一个量纲干净、可读的数,这是选这组权重的**真正理由**(人类裁定的形式)。

    ⚠️ fix round 6 更正:**这组权重并没有修好一个"尺寸迁移 bug",因为本来就不存在。**
    决定极小点的不变量是两项的梯度平衡比

        rho = w_tv / w_data

    因为 `_data_grad` 化简后是 S^H U^H M(U S x - y/sqrt(N))(逐体素 O(残差)),而
    TV 梯度 D^T u 逐体素 O(1)(|u|<=1,至多 6 项)。代入:
        rounds 1-4 (w_data=1, w_tv=lam) : rho = lam            —— 与尺寸、与 AF 都无关
        round 5+   (本函数)             : rho = lam*n_samp/N   —— = lam*采样率 ~ lam/AF
    **两者都与矩阵尺寸无关。** 实测(rho 对齐后跑同一体模的 32^3/64^3/96x80x96):
        rounds 1-4: TV/TV_gt = 0.9398 / 0.9715 / 0.9713  (max/min 1.034)
        round 5+  : TV/TV_gt = 0.9402 / 0.9715 / 0.9724  (max/min 1.034)
    几乎逐位相同。round 5 报告里那个"rounds 1-4 max/min = 1.141"的对照是**混合体**
    (新的 1/n_samp 数据权重 + 旧的常数 TV 权重),代码里从未存在过;"旧权重等效正则
    强度 N*lam、跨尺寸差 55 倍"的说法同样是错的,已在模块顶部更正。

    **代价(必须知道)**:本函数的 rho 里带了采样率因子,所以
        lam 在**固定 AF/ACS** 下跨矩阵尺寸可迁移;改了 AF 或 ACS 就必须重新标定 lam。
    而且 rho 会随 mask 实际采样率的抖动而抖动(实测 frac 0.0823-0.0911),旧权重的
    rho 则是严格常数。
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

    ============================================================================
    ⚠️ **不要在同一个 lam 下把本函数和 `fista_tv` 直接对比。**(fix round 6)
    ============================================================================
    vectorial TV 的归一化项是 `sqrt(sum_m sum_a |g|^2 + eps^2)`,分母里叠了 M 个通道的
    梯度能量。在各通道内容相近时该分母约为单通道的 sqrt(M) 倍,于是每个通道**实际受到的
    正则强度只有独立版的 1/sqrt(M)**。也就是说

        joint(M=2) @ lam   ~   independent @ lam/sqrt(2)

    实测(32^3、AF12、1500 步、lam=6.3e-3,两个**完全相同**的通道):
        joint@lam            TV/TV_gt = 0.9367
        independent@lam/√2   TV/TV_gt = 0.9368     <-- 与 joint 吻合到 4 位小数
        independent@lam      TV/TV_gt = 0.9334     <-- 不是同一个正则强度
    因此"joint 在同一 lam 下 PSNR 更高"这件事**大部分不是跨对比度信息带来的**:用两个
    完全相同的通道(跨对比度信息为零)重跑,joint 仍然比 independent@同 lam 高
    +0.196 dB;换成两个真正不同的通道也只有 +0.552 dB。前者纯粹是 1/sqrt(M) 的正则强度
    差,后者才含真实耦合收益,而两者混在一起。

    **公平比较的做法**:两个 arm **各自扫 lam**,用各自最优值比(best-against-best);
    或者把 joint 的 lam 乘 sqrt(M) 再比。本函数刻意**不**替调用方做这个缩放——vectorial
    TV 是标准形式,耦合本身就是目的,自动缩放会把"耦合"和"正则强度"这两件事再次搅在一起。
    该关系由 tests/crisp/test_cs_sense.py::test_joint_is_independent_at_lam_over_sqrt_m 钉住。
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
            g = wd * _data_grad(z, y, c, m) + w_tv * _tv_grad(z, coupled_norm=n, eps=eps)
            new_xs.append(z - step * g)
        t_new = 0.5 * (1 + (1 + 4 * t * t) ** 0.5)
        zs = [xn + ((t - 1) / t_new) * (xn - xo) for xn, xo in zip(new_xs, xs)]
        xs, t = new_xs, t_new
    return xs
