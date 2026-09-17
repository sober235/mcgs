import subprocess, sys, json
from pathlib import Path

import numpy as np
import torch


def test_cg_solves_normal_equations(tiny_topic):
    import t1t2_common as C, recon_classical as RC
    P = C.Pair("pre", torch.device("cpu"))
    b = P.sense_adj(P.y)
    x = RC.cg(lambda v: P.AHA(v) + 0.01 * v, b, 60)
    r = P.AHA(x) + 0.01 * x - b
    assert float(r.abs().norm() / b.abs().norm()) < 1e-3


def test_div3_is_adjoint_of_grad3():
    import recon_classical as RC
    g = torch.Generator().manual_seed(0)
    x = torch.randn(6, 5, 4, generator=g, dtype=torch.complex64)
    k = torch.randn(3, 6, 5, 4, generator=g, dtype=torch.complex64)
    lhs = torch.vdot(RC.grad3(x).flatten(), k.flatten()); rhs = torch.vdot(x.flatten(), RC.div3(k).flatten())
    assert abs(lhs - rhs) < 1e-4 * abs(lhs)


def test_cli_runs_zf_on_tiny_topic_cpu(tiny_topic):
    top = tiny_topic["root"]
    py = sys.executable
    r = subprocess.run([py, str(Path(__file__).resolve().parents[1] / "recon_classical.py"), "--scan", "pre", "--methods", "zf",
                        "--gpu", "-1"], capture_output=True, text=True, env={**dict(__import__("os").environ), "RETRO_TOPIC": str(top)})
    assert r.returncode == 0, r.stderr
    assert (top / "recon/zf_pre_complex.npy").exists() and (top / "recon/gt_pre_complex.npy").exists()
    rows = json.load(open(top / "logs/classical_pre.json"))["rows"]
    assert rows[0]["method"] == "zf" and rows[0]["head_psnr"] > 0
