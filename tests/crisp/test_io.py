import cmath

import h5py
import numpy as np
import pytest
import torch
from crisp.io import load_pair_h5, normalize_p999


def test_normalize_p999_scales_kspace_and_returns_scale():
    g = torch.Generator().manual_seed(3)
    ksp = torch.randn(2, 4, 4, 4, generator=g, dtype=torch.complex64)
    csm = torch.randn(2, 4, 4, 4, generator=g, dtype=torch.complex64)
    csm = csm / (csm.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)
    ksp_n, scale = normalize_p999(ksp, csm)
    assert scale > 0
    assert torch.allclose(ksp_n * scale, ksp, atol=1e-5)


def test_normalize_p999_scale_is_actually_p999_of_coil_combined_magnitude():
    """回环恒等式(上一测试)对任意非零 scale 都成立,不能证明用的是 P99.9 分位数
    (例如误用中位数也会通过)。这里独立算出预期分位数并与返回值比对,才能区分。"""
    g = torch.Generator().manual_seed(11)
    ksp = torch.randn(3, 6, 6, 6, generator=g, dtype=torch.complex64)
    csm = torch.randn(3, 6, 6, 6, generator=g, dtype=torch.complex64)
    csm = csm / (csm.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)

    from crisp.metrics.coil_domain import ifft3c
    gt_mag = (csm.conj() * ifft3c(ksp)).sum(0).abs().numpy()
    expected_scale = float(np.quantile(gt_mag, 0.999))

    _, scale = normalize_p999(ksp, csm)
    assert np.isclose(scale, expected_scale, rtol=1e-6)
    assert not np.isclose(scale, float(np.quantile(gt_mag, 0.5)), rtol=1e-2)


def test_normalize_p999_scale_matches_hand_derived_ground_truth():
    """上一测试的 expected 值用的是与被测代码(crisp/io.py:75)完全相同的表达式
    `(csm.conj() * ifft3c(kspace)).sum(0)`,查不出该表达式本身的 bug(丢 conj、
    sum 轴错、系数/线圈缺漏等——这些 bug 会在测试自己的计算里原样重现,悄悄通过)。

    这里反过来构造:先选定已知图像 x,选两路复线圈灵敏度 c0=0.6*e^{i0.3}、
    c1=0.8*e^{-i0.7}(逐点常数),满足 |c0|^2+|c1|^2 = 0.36+0.64 = 1.0——手算可核
    (3-4-5 勾股数缩放)。用前向模型构造 kspace_c = fft3c(c_coil * x)(forward,
    非被测的 adjoint 表达式)。因为线圈灵敏度平方和恒为 1,解析上
    sum_c conj(c_coil) * ifft3c(kspace_c) == sum_c |c_coil|^2 * x == x,
    即 GT 应精确等于 x 本身——这个 x 是我们自己选的已知量,不经由被测公式算出。
    expected_scale 直接从 x 的幅值算分位数,不调用 csm.conj()/.sum() 半步。

    若被测代码丢了 .conj():结果变成 sum_c c_coil^2 * x(而非 |c_coil|^2 * x),
    c0^2+c1^2 = 0.36*e^{i0.6}+0.64*e^{-i1.4} 是个模长偏离 1 的复数,与 x 不再
    相等,scale 会跟 expected_scale 对不上——见 fix report 里的变异检验。
    若 sum 轴错或漏加一路线圈,同理会得到 x 的某个错误标量倍数,也会对不上。
    """
    D, H, W = 3, 5, 4  # 三个空间维两两不同,线圈数(2)也跟它们都不同,轴错了会露馅
    g = torch.Generator().manual_seed(5)
    x = torch.randn(D, H, W, generator=g, dtype=torch.float32).to(torch.complex64)

    c0 = 0.6 * cmath.exp(1j * 0.3)
    c1 = 0.8 * cmath.exp(-1j * 0.7)
    assert abs(abs(c0) ** 2 + abs(c1) ** 2 - 1.0) < 1e-9  # 手算核对:0.6^2+0.8^2=1

    csm = torch.empty(2, D, H, W, dtype=torch.complex64)
    csm[0] = torch.full((D, H, W), c0, dtype=torch.complex64)
    csm[1] = torch.full((D, H, W), c1, dtype=torch.complex64)

    from crisp.metrics.coil_domain import fft3c
    coil_imgs = csm * x.unsqueeze(0)
    kspace = fft3c(coil_imgs)  # forward transform,不是被测的 adjoint 表达式

    expected_scale = float(np.quantile(x.abs().numpy(), 0.999))  # 直接来自已知量 x

    _, scale = normalize_p999(kspace, csm)
    assert np.isclose(scale, expected_scale, rtol=1e-4)


def _write_h5(path, dataset_name, arr):
    with h5py.File(path, "w") as f:
        f.create_dataset(dataset_name, data=arr)


def _tiny_arrays():
    ksp = np.random.default_rng(0).standard_normal((2, 4, 4, 4)).astype(np.complex64)
    csm = np.random.default_rng(1).standard_normal((2, 4, 4, 4)).astype(np.complex64)
    return ksp, csm


def test_load_pair_h5_with_explicit_csm_path(tmp_path):
    ksp, csm = _tiny_arrays()
    ksp_path = tmp_path / "ksp1.h5"
    csm_path = tmp_path / "whatever_name.h5"
    _write_h5(ksp_path, "kspace", ksp)
    _write_h5(csm_path, "csm", csm)

    ksp_t, csm_t = load_pair_h5(str(ksp_path), str(csm_path))
    assert torch.allclose(ksp_t, torch.from_numpy(ksp))
    assert torch.allclose(csm_t, torch.from_numpy(csm))


def test_load_pair_h5_auto_discovers_unique_sibling(tmp_path):
    ksp, csm = _tiny_arrays()
    _write_h5(tmp_path / "ksp1.h5", "kspace", ksp)
    _write_h5(tmp_path / "csm.h5", "csm", csm)  # irregular naming, like the real CE_Data

    ksp_t, csm_t = load_pair_h5(str(tmp_path / "ksp1.h5"))
    assert torch.allclose(ksp_t, torch.from_numpy(ksp))
    assert torch.allclose(csm_t, torch.from_numpy(csm))


def test_load_pair_h5_raises_when_no_sibling_csm(tmp_path):
    ksp, _ = _tiny_arrays()
    _write_h5(tmp_path / "ksp1.h5", "kspace", ksp)

    with pytest.raises(FileNotFoundError):
        load_pair_h5(str(tmp_path / "ksp1.h5"))


def test_load_pair_h5_raises_when_sibling_csm_ambiguous(tmp_path):
    ksp, csm = _tiny_arrays()
    _write_h5(tmp_path / "ksp1.h5", "kspace", ksp)
    _write_h5(tmp_path / "csm.h5", "csm", csm)
    _write_h5(tmp_path / "csm2.h5", "csm", csm)  # two candidates -> ambiguous

    with pytest.raises(FileNotFoundError):
        load_pair_h5(str(tmp_path / "ksp1.h5"))
