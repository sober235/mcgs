import math

import numpy as np
import torch


def test_pair_loads_tiny_topic_and_scores_truth_perfectly(tiny_topic):
    import t1t2_common as C
    P = C.Pair("pre", torch.device("cpu"))
    assert P.NC == 3 and P.G == (16, 16, 16) and P.contrast == "pre"
    assert abs(P.scale - tiny_topic["scales"]["scale_pre"]) < 1e-6
    assert bool((P.y[:, :, ~torch.from_numpy(tiny_topic["m2"])] == 0).all())
    row = P.score(P.gt_c, "truth")
    assert row["head_psnr"] == math.inf and row["psnr_cplx"] == math.inf
    assert row["head_nmse"] == 0.0 and row["nmse"] < 1e-10 and abs(row["ssim"] - 1) < 1e-6


def test_forward_and_adjoint_agree(tiny_topic):
    import t1t2_common as C
    P = C.Pair("post", torch.device("cpu"))
    g = torch.Generator().manual_seed(1)
    x = torch.randn(P.G, generator=g, dtype=torch.complex64)
    k = torch.randn((P.NC,) + P.G, generator=g, dtype=torch.complex64)
    Ax = C.fft3c(P.S * x) * P.m3
    AHk = P.sense_adj(k * P.m3)
    lhs = torch.vdot(Ax.flatten(), k.flatten()); rhs = torch.vdot(x.flatten(), AHk.flatten())
    assert abs(lhs - rhs) < 1e-3 * abs(lhs)
    assert torch.allclose(P.AHA(x), P.sense_adj(Ax), atol=1e-5)


def test_head_nmse_matches_head_psnr(tiny_topic):
    import t1t2_common as C
    P = C.Pair("pre", torch.device("cpu"))
    x = P.gt_c * 0.9
    row = P.score(x, "scaled")
    head = torch.from_numpy(C.head_mask(P.gt.numpy()))
    mse = float(((x.abs() - P.gt)[head] ** 2).mean())
    assert abs(row["head_psnr"] - (20 * np.log10(row["gmax"]) - 10 * np.log10(mse))) < 1e-4
    assert abs(row["head_nmse"] - mse * head.sum().item() / float((P.gt[head] ** 2).sum())) < 1e-7
