import itertools

import numpy as np
import pytest
import torch

from crisp.metrics.coil_domain import fft3c, ifft3c, e_coil_full
from crisp.metrics.image_domain import brain_gauge, brain_mask
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
    """“λ 越大 -> TV 越小”是**极小点**的性质,所以两个 λ 都必须跑到收敛才能比。

    fix round 4:迭代数从 brief 的 80 提到 1000。步长与 λ 有关(见 cs_sense.py 的
    `_term_weights`/`_GRAD_OP_SQ`),λ 越大步长越小,固定小迭代数下两次运行处在完全不同的
    收敛阶段,比较到的是"各自走了多远",不是极小点的 TV 大小。

    fix round 5 重新标定(复数 TV + 新权重后步长完全变了:λ=1e-4/1e-3/1e-2 的步长分别是
    160.0/36.6/4.2)。实测 TV(在 |x| 上、各向异性,即本测试自己的 `tv()` 口径):

        iters:      50      100      200      400     1000     2000     4000
        λ=1e-4  96.4210  96.0366  96.0164  96.0164  96.0164  96.0164  96.0164
        λ=1e-3  99.4805  96.2961  95.9704  95.9721  95.9724  95.9724  95.9724
        λ=1e-2 114.8351 100.2430  95.3130  94.9599  94.9938  94.9948  94.9943
        顺序对?     NO       NO      YES      YES      YES      YES      YES

    iters=200 起顺序就正确,>=400 后三个值都不再变化,故保留 iters=1000。
    判据本身(严格小于)一个字没动。
    """
    x, csm, y = _toy()
    mask = torch.zeros(8, 8, 8); mask[::2] = 1.0
    yu = y * mask.unsqueeze(0)
    tv = lambda v: (v.abs().diff(dim=0).abs().sum() + v.abs().diff(dim=1).abs().sum()
                    + v.abs().diff(dim=2).abs().sum())
    assert tv(fista_tv(yu, csm, mask, lam=1e-2, iters=1000)) < tv(fista_tv(yu, csm, mask, lam=1e-4, iters=1000))


def test_tv_grad_matches_finite_difference_complex():
    """有限差分核验**复数** TV 的梯度,独立于 FISTA 收敛动态。

    fix round 5:TV 现在作用在复数 x 上(TV_c = sum_j sqrt(sum_a |d_a x|^2 + eps^2),
    差分保留复数),`_tv_grad` 直接返回散度 div,不再有 `* (x/|x|)` 的链式因子。
    因此这条测试也必须在**复数**方向上验证,否则(像 round 5 之前那样只取实值正数体积)
    只测到了实轴上的限制,恰好是相位不受惩罚的那个退化子空间——正是它藏了 4 轮 bug。

    判据用方向导数:对实值泛函 TV_c,梯度 g 满足 d/dt TV_c(x + t*h) = Re<g, h>。
    随机取 8 个复数方向 h,逐个比对解析值与中心差分,同时单独比对纯虚方向(相位方向)
    ——若相位真的不受惩罚(round 4 的退化模型),纯虚方向的导数会恒为 0,这里会直接暴露。
    """
    from crisp.recon.cs_sense import _tv_grad, _grads, _tv_norm

    EPS = 1e-3  # 必须与下面 _tv_grad(x) 调用所用的 eps 一致

    # complex128:float32 下中心差分的舍入噪声(~1e-3)会和真实符号误差混在一起。
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(5, 5, 5, generator=g, dtype=torch.float64)
         + 1j * torch.randn(5, 5, 5, generator=g, dtype=torch.float64))

    def tv_c(v):
        gx, gy, gz = _grads(v)
        return _tv_norm(gx, gy, gz, EPS).sum()

    analytic = _tv_grad(x, eps=EPS)
    h = 1e-6
    for k in range(8):
        d = (torch.randn(5, 5, 5, generator=g, dtype=torch.float64)
             + 1j * torch.randn(5, 5, 5, generator=g, dtype=torch.float64))
        d = d / d.norm()
        numeric = float((tv_c(x + h * d) - tv_c(x - h * d)) / (2 * h))
        predicted = float((analytic.conj() * d).real.sum())
        assert abs(numeric - predicted) <= 1e-6 * max(abs(numeric), 1.0), (
            f"方向 {k}: 数值 {numeric:.10f} vs 解析 {predicted:.10f}")

    # 纯虚(相位)方向必须有非零梯度分量——这正是 round 5 修掉的退化
    d_imag = 1j * torch.randn(5, 5, 5, generator=g, dtype=torch.float64)
    d_imag = d_imag / d_imag.norm()
    numeric = float((tv_c(x + h * d_imag) - tv_c(x - h * d_imag)) / (2 * h))
    predicted = float((analytic.conj() * d_imag).real.sum())
    assert abs(numeric - predicted) <= 1e-6 * max(abs(numeric), 1.0)
    assert abs(numeric) > 1e-3, (
        f"纯虚方向的方向导数是 {numeric:.3e},≈0 说明相位不受 TV 惩罚——"
        "那就退回到了 round 4 的退化模型(见 cs_sense.py fix round 5 说明)")


def test_tv_grad_sign_on_real_volume():
    """实值正数体积上的散度符号核验(round 1-4 的原判据,保留)。

    复数 TV 限制在实轴上就等于原来的幅度 TV,所以这条老判据仍然适用:若
    `div[1:] += u[:-1]; div[:-1] -= u[:-1]` 的符号反了,这里会直接不匹配。
    保留它是因为它把**符号**这一件事单独钉死,与上面那条复数测试互为独立证据。
    """
    from crisp.recon.cs_sense import _tv_grad

    EPS = 1e-3

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


def _complex_tv(x, eps=1e-3):
    """TV_c(x) = sum_voxel sqrt(sum_a |d_a x|^2 + eps^2),差分保留复数。

    刻意在测试里独立重写(不从 cs_sense import),这样"退化守卫"和"目标函数单调"两条
    测试用的是独立推导的参考实现,而不是被测代码自己的说法。
    """
    gx = torch.zeros_like(x); gy = torch.zeros_like(x); gz = torch.zeros_like(x)
    gx[:-1] = x[1:] - x[:-1]
    gy[:, :-1] = x[:, 1:] - x[:, :-1]
    gz[:, :, :-1] = x[:, :, 1:] - x[:, :, :-1]
    return float(torch.sqrt(gx.abs() ** 2 + gy.abs() ** 2 + gz.abs() ** 2 + eps ** 2).sum())


def _fista_objective(x, y, csm, mask, lam, eps=1e-3):
    """fista_tv 真正在最小化的目标函数,**独立于 cs_sense.py 重写一遍**:

        J(x) = 1/(2*n_samp) * ||M(U S x - y/sqrt(N))||^2 + (lam/N) * TV_c(x)

    等价地(避免显式构造酉 FFT):第一项 = (1/n_samp) * (1/(2N)) * ||M(F S x - y)||^2,
    因为 ||M(U S x - y/sqrt(N))||^2 = (1/N)||M(F S x - y)||^2。

    这两个归一化常数都不是可有可无的:`_data_grad` 用 `ifft3c` 作伴随,而 torch 默认
    norm="backward" 下 ifft3c = F^{-1} = (1/N)F^H,所以它自带一个 1/N。fix round 2/3
    监控的是 `0.5*||M(FSx-y)||^2 + lam*TV`,少了这些常数,跟踪的基本上只是数据残差,
    于是把"数据残差随 TV 下降而上升"(CS 迭代的正常行为)误判成了发散。差一个常数就会
    让这条测试变成在检查一个本迭代根本没在最小化的函数,所以这里刻意重推一遍。
    """
    N = x.numel()
    n_samp = float(mask.sum())
    resid = mask.unsqueeze(0) * (fft3c(csm * x.unsqueeze(0)) - y)
    data = float((resid.abs() ** 2).sum()) / (2.0 * N * n_samp)
    return data + (lam / N) * _complex_tv(x, eps)


def test_objective_is_monotone_non_increasing():
    """收敛性的验收判据:目标函数逐迭代非增。三轮修复缺的一直是这条证据。

    `fista_tv` 是确定性的、每次都从同一个零填充初值重启,所以 `iters=k` 返回的就是
    第 k 次迭代的迭代点 x_k;逐个 k 调用即可拿到真正**逐迭代**(不是每 50 步抽样)的
    序列——fix round 3 正是因为只看 3 个稀疏 checkpoint 才把半收敛误读成发散。

    用 32^3 体模而不是 8^3 玩具:8^3 单常数块连着放过了前三轮的 bug(见 _brain_phantom
    的说明)。λ 取本项目 G2 网格的中位数,步长走 cs_sense 的默认规则(step=None),
    所以这条测试锁的是**出厂步长**,而不是某个测试里临时选的步长。

    诚实说明这条测试的**分辨力有限**(fix round 4 实测,同一体模,把步长按倍数放大后
    记录目标函数首次上升的迭代):x1/x5.25/x20 在 300 次迭代内都不上升,x40 到第 203 次
    才上升,x80 在第 5 次、x120 在第 3 次上升。也就是说本条测试只能抓到 >~40 倍于出厂
    步长的回归(brief 原稿的 `step=1.0` 是 76 倍,会被抓到;round 1 的 `step=0.25` 是
    19 倍,抓不到)。步长常数本身由 test_grad_operator_norm_pins_step_rule_constant
    直接钉住,那条才是高分辨力的守卫。K 取 30 是成本折中:fista_tv 每次都重跑,总代价
    是 O(K^2)。

    ⚠️ **不要把 K 提上去**(fix round 6)。J 现在是 O(1e-3) 量级(两项都是均值),迭代到
    平台后每步的真实下降量会掉到 float32 的分辨率以下,监控值就开始出现 ~1e-7 相对量级的
    抖动。本体模/lam 下,float32 监控的**第一次上升出现在第 812 步**;K=30 时离它还很远,
    所以这里可以用"零次上升"这种严格判据。要跑更长就必须换成 complex128,或把判据改成
    "上升幅度不超过若干个 float32 eps"——直接抬 K 会得到一条随机翻红的测试。
    (float64 交叉验证:32^3、800 步、complex128 下 0/800 次上升。)
    """
    n, K, lam = 32, 30, 6.3e-3
    x, csm = _brain_phantom(n)
    mask = _af12_mask(n)
    yu = fft3c(csm * x.unsqueeze(0)) * mask.unsqueeze(0)

    js = [_fista_objective(fista_tv(yu, csm, mask, lam=lam, iters=k), yu, csm, mask, lam)
          for k in range(1, K + 1)]
    bad = [(k + 1, js[k - 1], js[k]) for k in range(1, K) if js[k] > js[k - 1]]
    assert not bad, (
        f"lam={lam:g}: 目标函数在 {len(bad)} 处上升(前 3 处 {bad[:3]})——"
        "步长超出 Lipschitz 界时会这样,见 cs_sense.py 的 _GRAD_OP_SQ")


def test_solution_does_not_degenerate_to_flat_magnitude():
    """退化守卫:收敛解的幅度统计不得塌陷。这是本该从第一轮就存在的测试。

    round 1-4 的模型把 TV 只加在 |x| 上,相位完全自由;AF=12 下相位自由度足以独自拟合
    已采样数据,于是极小点把幅度抹平、把结构塞进相位。实测那时:TV_c 收敛到 262.2,而
    "幅度全平"的下界正是 N*eps = 262.1(即 TV/TV_gt = 0.019),脑内平均 |x| 漂到 GT 的
    1.31 倍,脑内 PSNR 12.59 dB(零填充 20.41)。三轮验收全过,没有一条测试看幅度统计。

    判据必须**两侧都卡**,因为"相位自由"这个同一个病根有两种表现,取决于正则强度:
      - 旧权重(round 4,等效正则强 ~55 倍)下,幅度被抹平,TV_c 塌到全平下界
        N*eps(TV/TV_gt = 0.019);
      - 新权重下把 `_tv_grad` 换回幅度 TV(下面的变异测试实测),幅度并不塌,反而是**相位
        噪声爆炸**:TV_c 冲到 GT 的 2.6 倍(300 步)、6.4 倍(600 步)、20.5 倍(1500 步),
        PSNR 掉到 19-22 dB。
    所以只卡下界会漏掉后者(round 5 第一版就是这样,被变异测试当场抓到)。这里卡
    0.5 < TV_c(recon)/TV_c(GT) < 1.5;复数 TV 实测稳定在 0.933(300/600/1500 步分别
    0.936/0.933/0.933),两边都有充裕余量。

    第二条(脑内平均 |x| 在 GT 的 ±15% 内)专门钉 round 4 那个"幅值抹平 + 均值漂到
    1.31 倍"的签名。**诚实说明:这一条单独并不能抓住上面那个新权重下的变异**
    (变异实测均值比 0.959-1.077,仍在 ±15% 内)——有牙齿的是 TV 的双侧界。

    跑到 600 次迭代(远在平台上,见 fix round 5 报告的 2000+ 迭代轨迹)而不是早停,
    因为退化恰恰是**收敛之后**才显现的——早停会把它藏起来。
    """
    n, lam, iters = 32, 6.3e-3, 600
    x, csm = _brain_phantom(n)
    mask = _af12_mask(n)
    yu = fft3c(csm * x.unsqueeze(0)) * mask.unsqueeze(0)
    gt_mag = x.abs().numpy()
    bm = brain_mask(gt_mag)
    gt_mean = float(gt_mag[bm].mean())

    xh = fista_tv(yu, csm, mask, lam=lam, iters=iters)

    flat_floor = n ** 3 * 1e-3                      # 幅度/相位全平时 TV_c 的下界 N*eps
    tv_gt, tv_rec = _complex_tv(x), _complex_tv(xh)
    ratio = tv_rec / tv_gt
    rec_mean = float(xh.abs().numpy()[bm].mean())

    assert ratio > 0.5, (
        f"TV_c(recon)/TV_c(GT) = {ratio:.3f} 掉到 0.5 以下(全平下界 N*eps={flat_floor:.1f},"
        f"即 {flat_floor/tv_gt:.4f} 倍)——解正在向全平塌陷,相位又变自由了")
    assert ratio < 1.5, (
        f"TV_c(recon)/TV_c(GT) = {ratio:.3f} 超过 1.5——相位噪声爆炸,"
        "这是 TV 退回只惩罚 |x| 时在当前权重下的签名(见本测试 docstring)")
    assert 0.85 * gt_mean < rec_mean < 1.15 * gt_mean, (
        f"脑内平均 |x| = {rec_mean:.4f},GT 是 {gt_mean:.4f}(比值 {rec_mean/gt_mean:.3f})"
        "——幅度统计已经漂移,round 4 退化解的签名是这个比值漂到 1.31")


def test_quality_does_not_degrade_with_iterations():
    """更正之二的**核心**验收判据,现在有出厂测试钉住了(fix round 6)。

    此前这条判据只活在报告里的一次性 ablation:出厂回归测试
    `test_real_scale_af12_beats_zero_fill` 跑 300 步,而 300 步恰恰是 2500 步内 PSNR 的
    **全局最大点**(暂态过冲);把 `_tv_grad` 换回 round 4 的幅度 TV,它仍能拿到 24.884 dB
    对 20.406 的零填充而**通过**。也就是说旗舰测试对本轮修掉的那个缺陷是瞎的。

    这里改成看**平台**:跑到 2000 步,要求
      1. 每个晚期 checkpoint 都高于零填充 —— 收敛解本身必须有用,不是靠早停;
      2. 晚期 checkpoint 之间的**极差** < 1 dB —— 质量不随迭代漂移。

    ⚠️ 判据必须用**多点极差**,不能用两点之差。本轮第一版按"|psnr(2000)-psnr(600)|<1dB"
    写,结果**变异测试直接放行**:幅度 TV 的轨迹不是单调下滑而是**来回摆**,32^3 实测
    600/800/1000/1200/1500/1800/2000/2500 步依次是
    19.23 / 17.96 / 18.96 / 20.24 / 21.81 / 20.47 / 19.85 / 17.87 dB
    ——恰好 600 与 2000 只差 0.62 dB,两点判据看不见中间 3.94 dB 的摆动。改成极差后
    立刻分开。这条教训本身就是"两点采样看不出轨迹形状"的又一次复现(round 3 同款)。

    实测(n=24、acs=4、lam=6.3e-3、零填充 15.930,checkpoint 800/1400/2000):
        复数 TV(出厂)  24.683 / 24.706 / 24.678   极差 0.028 dB,最小余量 +8.748 dB
        幅度 TV(变异)  17.479 / 21.890 / 18.587   极差 3.232 dB
    判据阈值 1.0 dB,两侧各有两个数量级的余量。注意变异版的最小余量仍是 +0.93 dB
    (仍在零填充之上),所以**有牙齿的是极差那一条**,不是"打败零填充"那一条。

    体模取 n=24(acs=4,实测采样率 0.0781 ~ 1/12.8)而不是 32^3:本判据检的是轨迹形状,
    与体积大小无关,而 4200 次迭代在 32^3 上要 ~45 s、在 24^3 上只要 ~18 s。分辨力没有
    损失(极差 0.028 vs 3.232 dB)。
    """
    n, lam = 24, 6.3e-3
    x, csm = _brain_phantom(n)
    mask = _af12_mask(n, acs=4)
    yu = fft3c(csm * x.unsqueeze(0)) * mask.unsqueeze(0)
    gt_mag = x.abs().numpy()
    zf = (ifft3c(yu) * csm.conj()).sum(0)
    psnr_zf = brain_gauge(zf.abs().numpy(), gt_mag)["psnr_brain"]

    cps = (800, 1400, 2000)
    ps = [brain_gauge(fista_tv(yu, csm, mask, lam=lam, iters=k).abs().numpy(), gt_mag)["psnr_brain"]
          for k in cps]

    assert min(ps) > psnr_zf, (
        f"晚期 checkpoint {dict(zip(cps, [round(p, 3) for p in ps]))} 中有低于零填充 "
        f"{psnr_zf:.3f} dB 的——重建质量只在早停时成立就不算基线")
    assert max(ps) - min(ps) < 1.0, (
        f"晚期 PSNR 极差 {max(ps)-min(ps):.3f} dB 超过 1 dB "
        f"({dict(zip(cps, [round(p, 3) for p in ps]))})——质量随迭代漂移,"
        "这正是模型退化的签名(见 cs_sense.py fix round 5/6)")


def test_joint_is_independent_at_lam_over_sqrt_m():
    """钉住 vectorial TV 的 1/sqrt(M) 效应,防止它被悄悄改掉或被误读成"耦合收益"。

    联合版的归一化项 sqrt(sum_m sum_a |g|^2 + eps^2) 在各通道内容相近时约为单通道的
    sqrt(M) 倍,所以每个通道实际受到的正则强度只有独立版的 1/sqrt(M):
        joint(M=2) @ lam  ~  independent @ lam/sqrt(2)
    用两个**完全相同**的通道(跨对比度信息为零)最干净地暴露这一点。

    这条测试的意义不只是回归保护:它是"两个 arm 不能在同一 lam 下直接比"这句话的
    可执行证据(详见 fista_joint_tv 的 docstring)。
    """
    n, lam, iters = 24, 6.3e-3, 300
    x, csm = _brain_phantom(n)
    mask = _af12_mask(n, acs=4)
    yu = fft3c(csm * x.unsqueeze(0)) * mask.unsqueeze(0)

    joint = fista_joint_tv([yu, yu], [csm, csm], [mask, mask], lam=lam, iters=iters)
    indep_scaled = fista_tv(yu, csm, mask, lam=lam / (2 ** 0.5), iters=iters)
    indep_same = fista_tv(yu, csm, mask, lam=lam, iters=iters)

    tv_j = _complex_tv(joint[0])
    tv_s = _complex_tv(indep_scaled)
    tv_i = _complex_tv(indep_same)

    # joint@lam 必须比 independent@lam 更接近 independent@lam/sqrt(2)
    assert abs(tv_j - tv_s) < abs(tv_j - tv_i), (
        f"joint@lam 的 TV={tv_j:.2f} 离 independent@lam/√2 的 {tv_s:.2f} 反而比离 "
        f"independent@lam 的 {tv_i:.2f} 更远——1/sqrt(M) 关系不成立了")
    assert abs(tv_j - tv_s) / tv_s < 0.01, (
        f"joint@lam 的 TV={tv_j:.2f} 与 independent@lam/√2 的 {tv_s:.2f} 相差 "
        f"{abs(tv_j-tv_s)/tv_s:.3%},超过 1%")


def test_grad_operator_norm_pins_step_rule_constant():
    """`_GRAD_OP_SQ` 必须真的等于 `_grads` 这个离散梯度算子的 ||D||^2,不能是随手填的数。

    步长规则 `1/(1 + _GRAD_OP_SQ*lam/eps)` 的合法性来自 Hessian 上界 ||D||^2/eps(推导见
    cs_sense.py 该常量旁的注释)。round 1-3 的 bug 正是这个因子被当成 1:上界算小了 12 倍,
    步长就越界了。这里用幂迭代直接测 ||D^T D|| 并两边夹:
      - 上界:实测 ||D||^2 不得超过 _GRAD_OP_SQ,否则步长规则不再是安全步长;
      - 下界:也不得远小于 _GRAD_OP_SQ,否则常量被无谓放大,白白拖慢收敛。
    D^T 按 _tv_grad 里 `div[1:] += u[:-1]; div[:-1] -= u[:-1]` 的同一套边界约定实现。
    """
    from crisp.recon.cs_sense import _GRAD_OP_SQ, _grads

    def _div(ux, uy, uz):
        d = torch.zeros_like(ux)
        d[1:] += ux[:-1];        d[:-1] -= ux[:-1]
        d[:, 1:] += uy[:, :-1];  d[:, :-1] -= uy[:, :-1]
        d[:, :, 1:] += uz[:, :, :-1]; d[:, :, :-1] -= uz[:, :, :-1]
        return d

    n = 16
    g = torch.Generator().manual_seed(0)
    v = torch.randn(n, n, n, generator=g, dtype=torch.float64)
    v = v / v.norm()
    s = 0.0
    for _ in range(400):                      # 幂迭代收敛到 ||D^T D|| = ||D||^2
        w = _div(*_grads(v))
        s = float(w.norm())
        v = w / s

    # 伴随自检:<D m, u> == <m, D^T u>,保证上面测的确实是 _grads 的伴随
    m = torch.randn(n, n, n, generator=g, dtype=torch.float64)
    us = [torch.randn(n, n, n, generator=g, dtype=torch.float64) for _ in range(3)]
    gx, gy, gz = _grads(m)
    lhs = float((gx * us[0] + gy * us[1] + gz * us[2]).sum())
    rhs = float((m * _div(*us)).sum())
    assert abs(lhs - rhs) <= 1e-8 * abs(lhs), f"_div 不是 _grads 的伴随:{lhs} vs {rhs}"

    assert s <= _GRAD_OP_SQ, f"实测 ||D||^2={s:.4f} 超过 _GRAD_OP_SQ={_GRAD_OP_SQ},步长规则不再安全"
    assert s >= 0.95 * _GRAD_OP_SQ, (
        f"实测 ||D||^2={s:.4f} 远小于 _GRAD_OP_SQ={_GRAD_OP_SQ},常量被无谓放大、白白拖慢收敛")


def test_real_scale_af12_beats_zero_fill():
    """更正要求的验收判据(task-6-brief.md「Task 6 更正」item 3):8^3 玩具体模测不出
    次梯度伪装成光滑梯度这个 bug,这里换成 >=64^3、>=8 线圈、AF~12、真实会用的 lam、
    300 次迭代的回归——fista_tv 必须打败零填充,否则不能被称为基线。
    ⚠️ fix round 5:lam 的含义已经变了(数据项按 n_samp 归一、TV 按 N 归一,见
    cs_sense.py),`crisp/experiments/g2_voxel.py` 那个旧 lam 网格**不再适用**,必须重调。
    这里继续用 6.3e-3 只是因为它恰好落在新口径下的可用区间内(实测可用区间约
    [6.3e-3, 3],64^3/AF12 上 2000 次迭代分别高出零填充 +14.2 到 +1.5 dB),不再声称它
    来自任何已调好的网格。该 lam 下 300 次迭代实测 35.63 dB(零填充 20.41),平台值
    34.65 dB —— 判据只断言"优于零填充",余量极大。
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
