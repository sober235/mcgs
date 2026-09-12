"""R_nl (losses/nonlocal_red.py): the lagged-target non-local prior of the jyz AF16 phase-3 loop.

    R_nl(x) = mean_B((|x| - x_tilde)^2) / mean_B(x_tilde^2)     B = dilated head bounding box, x_tilde detached
    x_tilde = D(|x_k|) refreshed at nl_start and every nl_refresh iterations (D = BM4D in the trainer)

Everything here runs with fake denoisers (identity, Gaussian blur); the real BM4D is never called. The last test
is the 30-iteration trainer smoke on the AF16 config (GPU, a few minutes; the card comes from CUDA_VISIBLE_DEVICES).

Run: PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES=4 TORCH_EXTENSIONS_DIR=<worktree>.torch_ext \
       /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_nonlocal_red.py -q
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paired_gs"))
from losses.nonlocal_red import NonLocalRED, crop, head_bbox, make_denoiser, nonlocal_red, uncrop  # noqa: E402

PY = sys.executable


def _ball_support(shape=(24, 28, 20), r=0.5):
    d, h, w = shape
    zz, yy, xx = torch.meshgrid(torch.linspace(-1, 1, d), torch.linspace(-1, 1, h), torch.linspace(-1, 1, w),
                                indexing="ij")
    return (zz ** 2 + yy ** 2 + xx ** 2) < r ** 2


def _vol(shape=(24, 28, 20), seed=0, dtype=torch.complex128):
    g = torch.Generator().manual_seed(seed)
    re_, im_ = torch.rand(shape, generator=g, dtype=torch.float64), torch.rand(shape, generator=g, dtype=torch.float64)
    return (re_ + 0.5 + 1j * im_).to(dtype)          # |x| bounded away from 0 so abs() is smooth


# --- (a) lambda 0: the trainer builds nothing and imports nothing --------------------------------------------

def test_trainer_imports_nothing_by_default():
    """Importing the trainer must not import the module; the import sits under the lambda guard, so a run
    with lambda_nl = 0 never builds the term (bit-identical to before)."""
    code = ("import sys; sys.path.insert(0, 'paired_gs'); import trainers.joint_trainer; "
            "print('losses.nonlocal_red' in sys.modules)")
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="")
    out = subprocess.run([PY, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip().endswith("False"), out.stdout
    src = (ROOT / "paired_gs/trainers/joint_trainer.py").read_text()
    assert re.search(r"if nl_lam_p > 0 or nl_lam_c > 0:\n\s+from losses\.nonlocal_red import", src), \
        "the nonlocal_red import must sit directly under the lambda guard"
    assert "from losses.nonlocal_red" not in src.split("def train(self):")[0]


def test_term_is_zero_before_the_first_refresh():
    nl = NonLocalRED(_ball_support(), make_denoiser("fake_identity"), start=10, refresh=10)
    v = _vol().requires_grad_(True)
    r = nl("T1", v)
    assert float(r) == 0.0 and not r.requires_grad
    assert nl.targets == {}


# --- (b) refresh cadence with the fake denoisers ------------------------------------------------------------

@pytest.mark.parametrize("name", ["fake_identity", "fake_blur"])
def test_refresh_cadence(name):
    calls = []
    base = make_denoiser(name)

    def den(z, sigma):
        assert isinstance(z, np.ndarray) and z.dtype == np.float32 and z.ndim == 3
        calls.append(sigma)
        return base(z, sigma)

    nl = NonLocalRED(_ball_support(), den, sigma_rel=0.04, start=10, refresh=10)
    v = _vol()
    for it in range(1, 31):
        expect = it >= 10 and (it - 10) % 10 == 0
        assert nl.due(it) == expect, it
        if it < 10:
            assert float(nl("T1", v)) == 0.0 and calls == []
        if nl.due(it):
            info = nl.refresh("T1", v, it)
            assert info["iter"] == it and info["crop"] == tuple(v[nl.bbox].shape) and info["seconds"] >= 0.0
    assert len(calls) == 3                                   # iterations 10, 20, 30
    head = v.abs()[nl.bbox][nl.head_crop]
    assert calls[0] == pytest.approx(0.04 * float(torch.quantile(head.float(), 0.99)), rel=1e-5)
    r = float(nl("T1", v))
    if name == "fake_identity":
        assert r == pytest.approx(0.0, abs=1e-10)           # target == |x| on the crop
    else:
        assert r > 1e-4                                      # the blur moved the target away from |x|
    assert nl.targets["T1"].dtype == torch.float32 and nl.targets["T1"].device == v.device


def test_refresh_never_before_start_and_target_is_constant_between_refreshes():
    nl = NonLocalRED(_ball_support(), make_denoiser("fake_identity"), start=500, refresh=250)
    assert [it for it in range(1, 2001) if nl.due(it)] == [500, 750, 1000, 1250, 1500, 1750, 2000]
    v = _vol(seed=1)
    nl.refresh("FLAIR", v, 500)
    t = nl.targets["FLAIR"].clone()
    assert float(nl("FLAIR", 2.0 * v)) > 0.0                 # the render moved, the target did not
    assert torch.equal(nl.targets["FLAIR"], t)


# --- (c) the target is detached; the gradient matches finite differences --------------------------------------

def test_target_is_detached_and_gradient_matches_finite_differences():
    sup = _ball_support((8, 9, 10), r=0.6)
    nl = NonLocalRED(sup, make_denoiser("fake_blur"), start=1, refresh=1, dilate=1)
    v0 = _vol((8, 9, 10), seed=2).requires_grad_(True)
    nl.refresh("T1", v0, 1)
    t = nl.targets["T1"]
    assert not t.requires_grad and t.grad_fn is None
    v = (v0.detach() * 1.1 + 0.05j).requires_grad_(True)
    r = nl("T1", v)
    assert r.requires_grad
    r.backward()
    assert v0.grad is None                                   # nothing flows into the render that made the target
    g = v.grad
    assert torch.isfinite(g).all()
    outside = torch.ones_like(v, dtype=torch.bool)
    outside[nl.bbox] = False
    assert g[outside].abs().max() == 0.0                     # only the crop is charged

    def f(x):
        return float(nonlocal_red(x, t, nl.bbox))
    gen = torch.Generator().manual_seed(3)
    idx = torch.nonzero(~outside)
    h = 1e-6
    for k in torch.randperm(idx.shape[0], generator=gen)[:6].tolist():
        i, j, l = idx[k].tolist()
        for part, delta in (("real", h), ("imag", 1j * h)):
            xp, xm = v.detach().clone(), v.detach().clone()
            xp[i, j, l] += delta
            xm[i, j, l] -= delta
            fd = (f(xp) - f(xm)) / (2 * h)
            an = getattr(g[i, j, l], part).item()
            assert an == pytest.approx(fd, rel=1e-5, abs=1e-9), (part, i, j, l)


def test_gradient_finite_at_an_exact_zero_voxel():
    sup = torch.ones(6, 6, 6, dtype=torch.bool)
    v = _vol((6, 6, 6), dtype=torch.complex64)
    v[2, 2, 2] = 0.0
    v.requires_grad_(True)
    t = torch.rand(6, 6, 6)
    nonlocal_red(v, t, head_bbox(sup, 0)).backward()
    assert torch.isfinite(v.grad).all()


# --- (d) bounding box and crop / un-crop --------------------------------------------------------------------

def test_head_bbox_is_dilated_by_8_and_clipped_to_the_volume():
    sup = torch.zeros(32, 40, 24, dtype=torch.bool)
    sup[3:10, 20:33, 0:5] = True
    assert head_bbox(sup, 8) == (slice(0, 18), slice(12, 40), slice(0, 13))
    sup = torch.zeros(40, 40, 40, dtype=torch.bool)
    sup[10:20, 15:25, 8:16] = True
    assert head_bbox(sup, 8) == (slice(2, 28), slice(7, 33), slice(0, 24))
    assert head_bbox(sup, 0) == (slice(10, 20), slice(15, 25), slice(8, 16))
    with pytest.raises(RuntimeError):
        head_bbox(torch.zeros(4, 4, 4, dtype=torch.bool))


def test_bbox_contains_the_whole_support_and_defaults_to_8():
    sup = _ball_support()
    nl = NonLocalRED(sup, make_denoiser("fake_identity"))
    assert nl.bbox == head_bbox(sup, 8)
    assert int(sup[nl.bbox].sum()) == int(sup.sum())
    assert torch.equal(nl.head_crop, sup[nl.bbox])


def test_crop_uncrop_round_trip_is_exact():
    sup = _ball_support()
    bbox = head_bbox(sup, 8)
    x = _vol(dtype=torch.complex64)
    c = crop(x, bbox)
    assert c.shape == tuple(s.stop - s.start for s in bbox)
    y = uncrop(c, bbox, x.shape)
    assert y.shape == x.shape and y.dtype == x.dtype
    assert torch.equal(y[bbox], c) and torch.equal(crop(y, bbox), c)
    outside = torch.ones(x.shape, dtype=torch.bool)
    outside[bbox] = False
    assert (y[outside] == 0).all()
    full = head_bbox(torch.ones_like(sup), 8)
    assert torch.equal(uncrop(crop(x, full), full, x.shape), x)


# --- (e) the normaliser makes R_nl dimensionless -------------------------------------------------------------

def test_scaling_x_and_target_together_leaves_r_nl_unchanged():
    sup = _ball_support()
    bbox = head_bbox(sup, 8)
    x = _vol(seed=4)
    t = torch.rand(crop(x, bbox).shape, dtype=torch.float64) + 0.5
    ref = float(nonlocal_red(x, t, bbox))
    assert ref > 0
    for c in (0.01, 3.0, 250.0):
        assert float(nonlocal_red(c * x, c * t, bbox)) == pytest.approx(ref, rel=1e-6)
    two = torch.full_like(x, 2.0)
    assert float(nonlocal_red(two, torch.ones_like(t), bbox)) == pytest.approx(1.0, rel=1e-9)


def test_denoiser_names():
    z = np.random.default_rng(0).random((9, 8, 7)).astype(np.float32)
    assert np.array_equal(make_denoiser("fake_identity")(z, 0.1), z)
    b = make_denoiser("fake_blur")(z, 0.1)
    assert b.shape == z.shape and b.std() < z.std()
    with pytest.raises(ValueError):
        make_denoiser("nope")
    with pytest.raises(ValueError):
        make_denoiser("bm4d", profile="fast")


# --- (f) 30-iteration trainer smoke on the AF16 config -------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="trainer smoke needs a GPU")
def test_trainer_smoke_af16_with_fake_blur():
    cfg = ROOT / "configs/t1flair_v3_af16_jyz.yaml"
    if not Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/data/ksp_T1.h5").exists():
        pytest.skip("AF16 jyz bundle not available")
    exp = "nl_smoke_pytest"
    for p in (ROOT / "results_arc" / exp, ROOT / "results_arc" / f"{exp}_score.json"):
        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    cmd = [PY, "-u", "arc/cycle.py", "--exp", exp, "--gpu", gpu, "--iters", "30", "--warmup", "5",
           "--lambda_geo", "0.10", "--max_radius", "20", "--kspace_high", "2.2",
           "--edge_tv", "--edge_tv_kappa_q", "0.85", "--edge_tv_wfloor", "0.10", "--tv_eps", "0.04",
           "--set", "gaussian.init_mode=head", "--set", "gaussian.init_scale_mode=3nn",
           "--set", "loss.edge_tv_kappa_source=zf",
           "--set", "loss.lambda_wav=0.07", "--set", "loss.lambda_wav_plain=0.10",
           "--set", "loss.lambda_nl=0.5", "--set", "loss.nl_start=10", "--set", "loss.nl_refresh=10",
           "--set", "loss.nl_denoiser=fake_blur", "--set", "gaussian.initial_num_points=200000",
           "--base_cfg", str(cfg)]
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    assert out.returncode == 0, out.stdout[-4000:] + out.stderr[-4000:]
    log = (ROOT / "results_arc" / exp / exp / "joint_train.log").read_text()
    lines = [l for l in log.splitlines() if l.startswith("[nl] ")]
    assert [re.search(r"iter=(\d+) contrast=(\w+)", l).groups() for l in lines] == \
        [("10", "T1"), ("10", "FLAIR"), ("20", "T1"), ("20", "FLAIR")]
    for l in lines:
        assert re.search(r"sigma=[0-9.e+-]+ crop=\(\d+, \d+, \d+\) bm4d=[0-9.]+s$", l), l
    assert "R_nl=" in log
    assert (ROOT / "results_arc" / f"{exp}_score.json").exists()
