import numpy as np
from crisp.masks.vdpoisson import align_to_dc, complementary_pair, vd_poisson_2d


def test_sampling_fraction_and_acs():
    m = vd_poisson_2d((220, 256), af=12.0, acs=16, seed=0)
    frac = m.mean()
    assert 0.9 / 12 < frac < 1.1 / 12
    c0, c1 = 220 // 2, 256 // 2
    assert m[c0 - 8:c0 + 8, c1 - 8:c1 + 8].all()          # ACS 全采


def test_density_decreases_with_radius():
    m = vd_poisson_2d((220, 256), af=12.0, acs=16, seed=0)
    yy, xx = np.mgrid[0:220, 0:256]
    r = np.sqrt(((yy - 110) / 110.0) ** 2 + ((xx - 128) / 128.0) ** 2)
    inner = m[(r > 0.15) & (r < 0.35)].mean()
    outer = m[(r > 0.7) & (r < 0.9)].mean()
    assert inner > outer


def test_density_decreases_with_radius_is_robust_to_noise():
    """单种子/单次内外环对比(上一测试)统计噪声太大:用抑制密度的坏实现(alpha
    被忽略、密度处处均匀)做变异检验,20 个种子里仍有 9 个让 `inner > outer` 凑巧
    成立(单种子通过率约 45%),不能证明密度确实随半径衰减。这里把 20 个种子的
    内/外环密度分别聚合平均以压低噪声,并要求内环密度显著高于外环(3 倍而非仅仅
    "大于"),真实实现的聚合比值约 42 倍,均匀密度的坏实现聚合比值约 1.0,阈值 3
    留了充分余量又不会被噪声碰到。"""
    shape, af, acs = (220, 256), 12.0, 16
    yy, xx = np.mgrid[0:220, 0:256]
    r = np.sqrt(((yy - 110) / 110.0) ** 2 + ((xx - 128) / 128.0) ** 2)
    inner_ring = (r > 0.15) & (r < 0.35)
    outer_ring = (r > 0.7) & (r < 0.9)
    inner_vals, outer_vals = [], []
    for seed in range(20):
        m = vd_poisson_2d(shape, af=af, acs=acs, seed=seed)
        inner_vals.append(m[inner_ring].mean())
        outer_vals.append(m[outer_ring].mean())
    assert np.mean(inner_vals) > 3 * np.mean(outer_vals)


def test_complementary_pair_budget_and_overlap():
    m1, m2, ov = complementary_pair((220, 256), af=12.0, acs=16, seed=0)
    assert abs(m1.mean() - m2.mean()) / m1.mean() < 0.05     # 预算相同
    assert ov <= 0.10                                        # 非 ACS 重叠 <= 10%
    c0, c1 = 220 // 2, 256 // 2
    assert m1[c0 - 8:c0 + 8, c1 - 8:c1 + 8].all() and m2[c0 - 8:c0 + 8, c1 - 8:c1 + 8].all()


def test_seed_reproducibility():
    a = vd_poisson_2d((64, 64), af=6.0, acs=8, seed=7)
    b = vd_poisson_2d((64, 64), af=6.0, acs=8, seed=7)
    assert np.array_equal(a, b)


def test_align_to_dc_moves_acs_onto_energy_peak():
    """任务说明特别点名 align_to_dc:只查返回 shape 的测试,roll 方向反了也测不出来。
    这里自己构造一个 k 空间,能量峰放在偏离中心的已知位置 (p0, p1),验证:
    (a) 返回的 shift 恰好等于 (p0 - d//2, p1 - w//2)(符号对,不是取反);
    (b) roll 之后 ACS 方块真的落在了峰值位置上,而不再覆盖原来的图像中心。
    若 roll 方向反了(shift 取负号)或未生效,(a)(b) 都会失败。
    """
    d, w = 64, 64
    h = 4
    mask = np.zeros((d, w), dtype=bool)
    mask[d // 2 - h:d // 2 + h, w // 2 - h:w // 2 + h] = True

    p0, p1 = 40, 20  # 偏离中心 (32, 32) 的已知峰值位置
    n_coil, readout = 3, 5
    ksp = np.zeros((n_coil, d, w, readout), dtype=np.complex64)
    ksp[:, p0, p1, :] = 10.0  # 沿线圈与读出方向堆能量,其余位置为零

    aligned, shift = align_to_dc(mask, ksp)

    assert shift == (p0 - d // 2, p1 - w // 2)
    assert aligned[p0 - h:p0 + h, p1 - h:p1 + h].all()               # ACS 移到峰值处
    assert not aligned[d // 2 - h:d // 2 + h, w // 2 - h:w // 2 + h].all()  # 不再覆盖原中心
