"""SingleContrastTrainer = one contrast per process. Pinned here: (1) its objective is the plain half of the
joint objective (a duplicate-slot joint run on the same contrast reads the same dc and twice each regulariser,
and the joint gradient of every shared tensor is one positive multiple of the single run's gradient, twice as
large for the geometry as for a density table), (2) nothing of a second channel exists (contrast
density parameters never receive a gradient, the contrast tensors are aliases), (3) every cross-contrast knob is
refused, (4) the FLAIR shutter (known_zero) builds the same fit tensors as the joint trainer's contrast channel,
(5) arc/cycle.py --trainer single produces a plain-only score json.
Run: PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=<idle> /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_sg2x2_single_trainer.py -q
"""
import contextlib
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paired_gs"))
sys.path.insert(0, str(ROOT / "tests"))
from sg2x2_fixture import base_cfg, make_pair  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="trainer needs the CUDA voxelizer")
PY = "/home/congcongliu/anaconda3/envs/GSTMI/bin/python"


def _dup_joint_cfg(pair, tmp, exp, contrast, reg=True):
    """the historical proxy: the same contrast in both slots of the joint trainer"""
    cfg = base_cfg(pair, tmp, exp, reg=reg)
    blk = cfg["data"]["plain"] if contrast == "T1" else cfg["data"]["contrast"]
    cfg["data"] = {"plain": copy.deepcopy(blk), "contrast": copy.deepcopy(blk)}
    return cfg


def _iter_line(out_dir: Path, exp: str, it: int) -> dict:
    txt = (out_dir / exp / "joint_train.log").read_text()
    line = next(l for l in txt.splitlines() if l.startswith(f"[iter {it:5d}/"))
    return {k: float(v) for k, v in re.findall(r"(\w+)=([0-9.e+-]+)", line)}


# The single trainer re-implements parts of JointTrainer.train() (kappa source and refresh, the geometry-weight shim, the
# phase-TV and wavelet terms) and calls the inherited k-space curriculum. Each variant switches one of them on, identically on
# the single config and on the duplicate-slot joint config, so iteration 1 of the two trainers is compared with it live.
VARIANTS = {
    "base_reg": dict(reg=True),
    "kappa_render": dict(reg=True, loss={"edge_tv_kappa_source": "render"}),
    "refresh1": dict(reg=True, loss={"edge_tv_kappa_refresh_iter": 1}),      # the refresh fires at iteration 1
    "curriculum": dict(reg=True, loss={"kspace_freq_curriculum": True}),     # the other curriculum keys at their defaults
    "tv_weight_key": dict(reg=True, tv_weight_key=True),                     # the geometry weight spelled tv_weight only
    "ptv_dominant": dict(reg=True, loss={"lambda_ptv": 3.0}),                # 100 x the recipe's phase-TV weight: over half of the density_imag gradient
    "no_reg": dict(reg=False),
}


def _apply_variant(cfg: dict, spec: dict) -> dict:
    cfg["loss"].update(spec.get("loss", {}))
    if spec.get("tv_weight_key"):
        cfg["loss"]["tv_weight"] = cfg["loss"].pop("lambda_geo")
    return cfg


@cuda
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_single_objective_is_the_plain_half_of_the_duplicate_slot_joint(tmp_path, variant):
    from trainers.joint_trainer import JointTrainer
    from trainers.single_contrast_trainer import SingleContrastTrainer
    spec = VARIANTS[variant]
    reg = spec["reg"]
    pair = make_pair(tmp_path)
    dev = torch.device("cuda", 0)
    s = SingleContrastTrainer(_apply_variant(base_cfg(pair, tmp_path, "s", reg=reg, single="FLAIR"), spec), device=dev)
    j = JointTrainer(_apply_variant(_dup_joint_cfg(pair, tmp_path, "j", "FLAIR", reg=reg), spec), device=dev)
    # the tv_weight variant makes both trainers warn that loss.tv_weight is deprecated; expect it so the output stays clean
    with (pytest.warns(DeprecationWarning, match="tv_weight") if spec.get("tv_weight_key") else contextlib.nullcontext()):
        s.train(); j.train()
    ls, lj = _iter_line(tmp_path / "out", "s", 1), _iter_line(tmp_path / "out", "j", 1)
    assert ls["dc_p"] == pytest.approx(lj["dc_p"], rel=1e-5) and lj["dc_c"] == pytest.approx(lj["dc_p"], rel=1e-5)
    assert ls["dc_c"] == 0.0
    for k in ("R_geo", "R_wav"):
        assert lj[k] == pytest.approx(2.0 * ls[k], rel=1e-4, abs=1e-12)
    if reg:
        assert ls["R_geo"] > 0 and ls["R_wav"] > 0 and ls["R_ptv"] > 0
    else:
        assert ls["R_geo"] == 0 and ls["R_wav"] == 0 and ls["R_ptv"] == 0
    # the single objective is the plain half of the joint objective, so after the one backward pass (clip_grad_norm_
    # rescales all gradients of a run by one positive constant) each shared tensor's joint gradient is one positive
    # multiple of the single run's gradient, and the geometry multiple is twice the density multiple (both DC halves
    # push the shared geometry, each density table sees one half). Rotations are left out: the 3-NN scales are
    # isotropic, so their gradient is identically zero at this step.
    def ratio(a, b):
        a, b = a.detach().flatten().double(), b.detach().flatten().double()
        r = float((a @ b) / (b @ b))
        return r, float(torch.linalg.norm(a - r * b) / torch.linalg.norm(a))
    r_pos, e_pos = ratio(j.model.positions.grad, s.model.positions.grad)
    r_scl, e_scl = ratio(j.model.scales.grad, s.model.scales.grad)
    r_dr, e_dr = ratio(j.model.density_real.grad, s.model.density_real.grad)
    r_di, e_di = ratio(j.model.density_imag.grad, s.model.density_imag.grad)
    assert max(e_pos, e_scl, e_dr, e_di) < 1e-4
    assert r_scl == pytest.approx(r_pos, rel=1e-4) and r_di == pytest.approx(r_dr, rel=1e-4)
    assert r_pos == pytest.approx(2.0 * r_dr, rel=1e-4)


@cuda
def test_single_has_no_second_channel(tmp_path):
    from trainers.single_contrast_trainer import SingleContrastTrainer
    pair = make_pair(tmp_path)
    s = SingleContrastTrainer(base_cfg(pair, tmp_path, "s", single="T1"), device=torch.device("cuda", 0))
    assert s.ds_c is s.ds_p and s.kspace_c is s.kspace_p and s.csm_c is s.csm_p and s.mask_c is s.mask_p
    assert s.zf_c is s.zf_p and s.kspace_fit_c is s.kspace_fit_p and s.mask_fit_c is s.mask_fit_p
    with pytest.raises(RuntimeError):
        s._render("contrast")
    s.train()
    assert s.model.density_real_contrast.grad is None and s.model.density_imag_contrast.grad is None
    assert s.model.positions.grad is not None


@cuda
@pytest.mark.parametrize("key,val", [
    (("loss", "lambda_edge"), 0.001), (("loss", "lambda_dtv"), 0.01), (("loss", "lambda_vtv"), 0.01),
    (("loss", "lambda_dtv_proj"), 0.01), (("loss", "edge_tv_guide_alpha"), 0.25), (("loss", "ptv_guide_alpha"), 0.25),
    (("loss", "guide_path_plain"), "/tmp/x.npy"), (("loss", "lambda_nl"), 0.1), (("loss", "kspace_high_weight_plain"), 3.0),
    (("adaptive_control", "enable"), True), (("gaussian", "num_private_points"), 10),
    (("gaussian", "warm_start_checkpoint"), "/tmp/x.pth"),
])
def test_single_refuses_every_cross_contrast_or_out_of_scope_knob(tmp_path, key, val):
    from trainers.single_contrast_trainer import SingleContrastTrainer
    pair = make_pair(tmp_path)
    cfg = base_cfg(pair, tmp_path, "r", single="T1")
    cfg.setdefault(key[0], {})[key[1]] = val
    with pytest.raises(ValueError, match=key[1]):
        SingleContrastTrainer(cfg, device=torch.device("cuda", 0))


@cuda
def test_single_refuses_a_contrast_block(tmp_path):
    from trainers.single_contrast_trainer import SingleContrastTrainer
    pair = make_pair(tmp_path)
    cfg = base_cfg(pair, tmp_path, "two")           # joint-style config with data.contrast
    with pytest.raises(ValueError, match="data.contrast"):
        SingleContrastTrainer(cfg, device=torch.device("cuda", 0))


@cuda
def test_single_known_zero_matches_joint_contrast_channel(tmp_path):
    """the FLAIR shutter: the single run's fit tensors equal the joint run's contrast-channel fit tensors"""
    from trainers.joint_trainer import JointTrainer
    from trainers.single_contrast_trainer import SingleContrastTrainer
    pair = make_pair(tmp_path)
    dev = torch.device("cuda", 0)
    cj = base_cfg(pair, tmp_path, "kj"); cj["data"]["contrast"]["known_zero"] = True
    cs = base_cfg(pair, tmp_path, "ks", single="FLAIR"); cs["data"]["plain"]["known_zero"] = True
    j = JointTrainer(cj, device=dev); s = SingleContrastTrainer(cs, device=dev)
    assert torch.equal(j.mask_fit_c, s.mask_fit_p) and torch.equal(j.kspace_fit_c, s.kspace_fit_p)
    assert torch.equal(j.csm_fit_c, s.csm_fit_p) and j._dc_norm_c == s._dc_norm_p
    assert s.physics_info.get("p", {}).get("known_zero") is True and "c" not in s.physics_info


@cuda
def test_cycle_single_writes_a_plain_only_score_json(tmp_path):
    pair = make_pair(tmp_path)
    cfg = base_cfg(pair, tmp_path, "cyc", single="T1", iters=2)
    cfg_path = tmp_path / "single_T1.yaml"; cfg_path.write_text(yaml.safe_dump(cfg))
    exp = f"sg2x2_test_cycle_single_{os.getpid()}"
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": ""}
    r = subprocess.run([PY, "-u", str(ROOT / "arc/cycle.py"), "--exp", exp, "--gpu", gpu, "--iters", "2", "--warmup", "0",
                        "--trainer", "single", "--base_cfg", str(cfg_path), "--lambda_geo", "0.10", "--edge_tv",
                        "--edge_tv_kappa_q", "0.85", "--edge_tv_wfloor", "0.10", "--tv_eps", "0.04", "--kspace_high", "2.2",
                        "--lambda_ptv", "0.03", "--save_volumes"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    row = json.load(open(ROOT / "results_arc" / f"{exp}_score.json"))
    assert row["contrast"] is None and row["contrast_name"] is None and row["plain_name"] == "T1"
    assert row["plain"]["head_psnr"] > 0
    import numpy as np
    z = np.load(ROOT / "results_arc" / f"{exp}_volumes.npz")
    assert {"rec", "gt", "zf", "rec_complex"} <= set(z.files) and str(z["contrast_name"]) == "T1"
