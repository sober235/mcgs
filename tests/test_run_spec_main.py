"""run_spec.main end to end on the CPU with a stub JointTrainer (final review 2026-09-28, recommendation 2).

What only a GPU run exercised before: volumes saved by slot even when both channels carry one name (GS-single),
the score and audit json fields, a key read with two different defaults, the run-name reservation and the
PYTHONNOUSERSITE check. No GPU is touched: the stub trainer holds CPU tensors, run_spec's .to(cuda:0) calls reach
stub objects that stay on the CPU, scoring returns canned numbers, and set_seed / CUDA memory calls are stubbed.
Everything run_spec.main changes outside tmp_path (os.environ, sys.path, sys.modules) is restored by monkeypatch.

Run: env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_run_spec_main.py -q
"""
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "arc"))
import run_spec  # noqa: E402
import spec_io  # noqa: E402
import spec_scoring  # noqa: E402

SHAPE = (4, 4, 4)
# per channel: the render, the ground truth and the zero-filled image, all constants, so every saved volume
# shows where it came from
VALUES = {"plain": (1.0, 2.0, 5.0), "contrast": (3.0, 4.0, 6.0)}


class _StaysOnCpu:
    """A tensor whose .to(device) returns it unchanged: run_spec moves ground truths and masks to cuda:0."""

    def __init__(self, t):
        self.t, self.ndim = t, t.ndim

    def to(self, _device):
        return self.t


class StubVoxelizer:
    backend = "tile"          # what TileVoxelizer.backend reads after a silent fallback to the PyTorch reference


class StubTrainer:
    """Reads a few keys like JointTrainer; voxelizer.type twice with different defaults, as
    voxelizer_config.py:22 ("chunk") and joint_trainer.py:344 ("tile_cuda") do when the spec leaves it out."""

    def __init__(self, cfg, device=None):
        cfg["training"]["seed"]
        vox = cfg.get("voxelizer", {})
        vox.get("type", "chunk")
        vox.get("type", "tile_cuda")
        cfg["loss"].get("lambda_ptv", 0.0)
        cfg["loss"].get("lambda_ptv", 0.0)
        self.model = types.SimpleNamespace(num_points=123, load_state_dict=lambda sd: None)
        self.voxelizer = StubVoxelizer()
        for ch, tag in (("plain", "p"), ("contrast", "c")):
            _, gt, zf = VALUES[ch]
            csm = torch.ones((1, *SHAPE), dtype=torch.complex64)    # one coil, sensitivity 1
            y = spec_scoring.fft3c(torch.full((1, *SHAPE), zf, dtype=torch.complex64))
            gt_t = torch.full(SHAPE, gt, dtype=torch.complex64)
            setattr(self, f"csm_{tag}", csm)
            setattr(self, f"kspace_{tag}", y)
            setattr(self, f"kspace_dc_{tag}", y)
            setattr(self, f"mask_{tag}", _StaysOnCpu(torch.ones(SHAPE)))          # zf = the measured image
            setattr(self, f"mask_fit_{tag}", _StaysOnCpu(torch.zeros(SHAPE)))     # rec = the render
            setattr(self, f"ds_{tag}", types.SimpleNamespace(get_data=lambda g=gt_t: {"ground_truth": _StaysOnCpu(g)}))

    def train(self):
        pass

    def _render(self, ch):
        return torch.full(SHAPE, VALUES[ch][0], dtype=torch.complex64)


def _canned(tr, which, dev, anatomy_rel, with_grain_ratio):
    return dict(psnr=30.0 if which == "plain" else 32.0, ssim=0.9, nmse=0.01, hfen=0.2, head_psnr=31.0,
                head_ssim=0.91, head_hfen=0.21, head_nmse=0.02, grain_ratio=None, sharp=1.0, grain=0.01,
                head_psnr_cplx=29.0, comp=40.0, beat=True)


def _no_gpu(*args, **kwargs):
    raise AssertionError("run_spec touched the GPU in a CPU test")


@pytest.fixture
def stubbed(monkeypatch):
    for var in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):   # main() sets both; restored after the test
        monkeypatch.setenv(var, os.environ.get(var, ""))
    monkeypatch.setattr(sys, "path", list(sys.path))
    trainer_mod = types.ModuleType("trainers.joint_trainer")
    trainer_mod.JointTrainer = StubTrainer
    monkeypatch.setitem(sys.modules, "trainers.joint_trainer", trainer_mod)
    seeds = []
    seed_mod = types.ModuleType("seed")
    seed_mod.set_seed = lambda seed, deterministic=False: seeds.append((seed, deterministic))
    monkeypatch.setitem(sys.modules, "seed", seed_mod)
    monkeypatch.setattr(spec_scoring, "score_channel", _canned)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "_lazy_init", _no_gpu)
    monkeypatch.setattr(spec_io, "check_results_root", lambda *a: None)   # tmp_path is outside the repo
    monkeypatch.setattr(run_spec, "_no_user_site", lambda: True)
    return seeds


def _spec_file(tmp_path, names=("E1", "E2"), drop=()):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    for n in ("ksp.h5", "csm.h5", "mask.h5"):
        (data / n).write_bytes(n.encode())
    files = {"data_path": str(data / "ksp.h5"), "csm_path": str(data / "csm.h5"), "mask_path": str(data / "mask.h5")}
    s = {"spec": {"dataset": "toy", "exp_prefix": "toy", "results_root": str(tmp_path / "res"), "save_volumes": True,
                  "from_ckpt": None},
         "scoring": {"anatomy_rel": 0.05, "grain_ratio": False},
         "trainer": {"data": {"plain": dict(files, name=names[0]), "contrast": dict(files, name=names[1])},
                     "training": {"seed": 7, "deterministic": False, "max_iterations": 10, "shared_warmup_iters": 2},
                     "voxelizer": {"type": "tile_cuda"}, "loss": {"tv_weight": 0.1, "lambda_ptv": 0.0},
                     "output": {"use_tensorboard": False}}}
    for section, key in drop:
        del s["trainer"][section][key]
    f = tmp_path / "spec.yaml"
    f.write_text(yaml.safe_dump(s))
    return f


ENV_FIELDS = ("python", "torch", "torch_cuda", "no_user_site", "voxelizer")


def test_gs_single_names_keep_both_channels_in_the_volumes(tmp_path, stubbed):
    f = _spec_file(tmp_path, names=("E1", "E1"))
    run_spec.main(["--spec", str(f), "--exp", "toy_single", "--gpu", "0"])
    res = tmp_path / "res"
    with np.load(res / "toy_single_volumes.npz") as z:
        vols = {k: z[k] for k in z.files}
    slots = {f"{kind}_{ch}" for kind in ("rec", "gt", "zf") for ch in ("plain", "contrast")}
    assert set(vols) == slots | {s + "_complex" for s in slots} | {"plain_name", "contrast_name"}
    assert vols["plain_name"].shape == () and str(vols["plain_name"]) == "E1" and str(vols["contrast_name"]) == "E1"
    for ch, (rec, gt, zf) in VALUES.items():
        for kind, value in (("rec", rec), ("gt", gt), ("zf", zf)):
            assert vols[f"{kind}_{ch}"].dtype == np.float32 and vols[f"{kind}_{ch}_complex"].dtype == np.complex64
            np.testing.assert_allclose(vols[f"{kind}_{ch}"], value, atol=1e-5, err_msg=f"{kind}_{ch}")
            np.testing.assert_allclose(vols[f"{kind}_{ch}_complex"], value, atol=1e-5, err_msg=f"{kind}_{ch}_complex")
    row = json.loads((res / "toy_single_score.json").read_text())
    for k in ("plain", "contrast", "plain_name", "contrast_name", "git_commit", "git_dirty", "git_dirty_paths",
              *ENV_FIELDS):
        assert k in row, k
    assert row["plain"]["psnr"] == 30.0 and row["contrast"]["psnr"] == 32.0
    assert row["python"] == sys.executable and row["torch"] == str(torch.__version__)
    assert row["torch_cuda"] == torch.version.cuda and row["no_user_site"] is True
    assert row["voxelizer"] == {"class": "StubVoxelizer", "backend": "tile", "type": "tile_cuda", "strict_cuda": None}
    assert row["fallbacks"] == [] and stubbed == [(7, False)]


def test_audit_groups_fallbacks_and_flags_conflicting_defaults(tmp_path, stubbed, capsys):
    f = _spec_file(tmp_path, drop=(("voxelizer", "type"), ("loss", "lambda_ptv")))
    run_spec.main(["--spec", str(f), "--exp", "toy_audit", "--gpu", "0", "--audit", "3"])
    res = tmp_path / "res"
    out = json.loads((res / "toy_audit_audit.json").read_text())
    assert out["fallbacks"] == [
        {"key": "voxelizer.type", "default": "chunk", "conflicting_defaults": ["chunk", "tile_cuda"]},
        {"key": "loss.lambda_ptv", "default": 0.0}]
    assert out["iters"] == 3 and out["exp"] == "toy_audit"
    for k in ENV_FIELDS:
        assert k in out, k
    assert out["voxelizer"]["class"] == "StubVoxelizer" and out["voxelizer"]["type"] is None
    assert "2 fallback key(s), 1 with conflicting defaults" in capsys.readouterr().out
    assert not (res / "toy_audit_score.json").exists()


def test_a_run_name_is_used_once(tmp_path, stubbed):
    argv = ["--spec", str(_spec_file(tmp_path)), "--exp", "toy_a", "--gpu", "0"]
    run_spec.main(argv)
    assert (tmp_path / "res" / "toy_a").is_dir()          # reserved by run_spec (the stub trainer creates nothing)
    score = tmp_path / "res" / "toy_a_score.json"
    before = score.read_bytes()
    with pytest.raises(spec_io.SpecError, match="exists"):
        run_spec.main(argv)
    assert score.read_bytes() == before


def test_an_existing_score_json_alone_blocks_the_name(tmp_path, stubbed):
    f = _spec_file(tmp_path)
    res = tmp_path / "res"
    res.mkdir()
    (res / "toy_b_score.json").write_text("{}")
    with pytest.raises(spec_io.SpecError, match="toy_b_score.json exists"):
        run_spec.main(["--spec", str(f), "--exp", "toy_b", "--gpu", "0"])
    assert not (res / "toy_b").exists() and (res / "toy_b_score.json").read_text() == "{}"


def test_user_site_packages_are_refused(tmp_path, stubbed, monkeypatch):
    monkeypatch.setattr(run_spec, "_no_user_site", lambda: False)
    before = os.environ.get("CUDA_VISIBLE_DEVICES")
    with pytest.raises(SystemExit, match="PYTHONNOUSERSITE=1"):
        run_spec.main(["--spec", str(_spec_file(tmp_path)), "--exp", "toy_c", "--gpu", "0"])
    assert not (tmp_path / "res").exists() and os.environ.get("CUDA_VISIBLE_DEVICES") == before
