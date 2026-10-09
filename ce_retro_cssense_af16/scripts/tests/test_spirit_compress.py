"""Per-slice SVD coil compression of recon_spirit.py (--vcoils, 2026-10-09 comparison-arm audit)."""
import torch


def _data(nb=3, nc=6, ny=8, nz=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randn(nb, 2, ny, nz, generator=g, dtype=torch.complex64)             # two underlying signals per slice
    mix = torch.randn(nb, nc, 2, generator=g, dtype=torch.complex64)                  # -> rank-2 coil data + noise
    yb = torch.einsum("bcs,bsyz->bcyz", mix, src) + 1e-3 * torch.randn(nb, nc, ny, nz, generator=g, dtype=torch.complex64)
    m2 = torch.rand(ny, nz, generator=g) < 0.5
    return yb, m2


def test_compression_matrix_is_orthonormal_per_slice_and_ranks_energy():
    import recon_spirit as RS
    yb, m2 = _data()
    V, energy = RS.compress_matrix(yb, m2, 3)
    assert V.shape == (3, 6, 3)
    eye = V.conj().transpose(1, 2) @ V
    assert torch.allclose(eye, torch.eye(3, dtype=eye.dtype).expand(3, 3, 3), atol=1e-5)
    assert 0.999 < energy <= 1.0 + 1e-6                        # rank-2 data: 3 virtual coils keep essentially everything
    V1, e1 = RS.compress_matrix(yb, m2, 1)
    assert 0.3 < e1 < energy                                   # one coil keeps less, but the strongest direction first


def test_full_rank_compression_keeps_the_coil_combination():
    """With n = NC the compression is a unitary rotation per slice: sum_v conj(S'_v) x'_v == sum_c conj(S_c) x_c."""
    import recon_spirit as RS
    yb, m2 = _data()
    S = torch.randn_like(yb); S = S / S.abs().pow(2).sum(1, keepdim=True).sqrt()
    V, energy = RS.compress_matrix(yb, m2, 6)
    assert abs(energy - 1.0) < 1e-5
    lhs = (RS.apply_compress(S, V).conj() * RS.apply_compress(yb, V)).sum(1)
    rhs = (S.conj() * yb).sum(1)
    assert torch.allclose(lhs, rhs, atol=1e-4, rtol=1e-4)
