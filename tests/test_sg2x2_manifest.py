"""The regulariser audit of scripts/sg2x2_manifest.py: a reg-off cell must show zero weighted priors in the resolved
config, no prior start-up line, and zero R_* on the first and last logged iteration; a reg-on cell must show the
frozen recipe. Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_sg2x2_manifest.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pytest

from sg2x2_cells import CELLS  # noqa: E402
from sg2x2_manifest import audit, parse_iter_line  # noqa: E402

LINE_ON = ("[iter     1/4000] loss=1.2345e-01 dc_p=1.0000e-01 dc_c=2.0000e-02 R_geo=3.0000e-03 R_dtv=0.0000e+00 "
           "R_edge=0.0000e+00 R_vtv=0.0000e+00 R_dtvp=0.0000e+00 R_wav=4.0000e-03 R_nl=0.0000e+00 balance_violation=False")
LINE_OFF = ("[iter  4000/4000] loss=1.0000e-01 dc_p=1.0000e-01 dc_c=0.0000e+00 R_geo=0.0000e+00 R_dtv=0.0000e+00 "
            "R_edge=0.0000e+00 R_vtv=0.0000e+00 R_dtvp=0.0000e+00 R_wav=0.0000e+00 R_ptv=0.0000e+00 R_nl=0.0000e+00 balance_violation=False")
RES_ON = {"lambda_geo": 0.10, "edge_tv": True, "lambda_wav": 0.10, "lambda_wav_plain": 0.12, "lambda_ptv": 0.03,
          "lambda_edge": 0.0, "lambda_dtv": 0.0, "lambda_vtv": 0.0, "lambda_dtv_proj": 0.0, "edge_tv_guide_alpha": 0.0,
          "ptv_guide_alpha": 0.0, "kspace_freq_adaptive": True, "kspace_high_weight": 2.2}
RES_OFF = {**{k: 0.0 for k in RES_ON}, "edge_tv": False, "kspace_freq_adaptive": True, "kspace_high_weight": 2.2}


ARGS = {"seed": 42, "npts": 3600000, "init": "support_file", "support": "/x/support.npz", "iters": 4000}


def _cfg(res, cell="A", **extra):
    """A complete resolved config as the launcher writes it (the frozen knobs set explicitly)."""
    loss = {"tv_weight": res["lambda_geo"], **{k: v for k, v in res.items() if k != "lambda_geo"}}
    loss.update(kspace_center_fraction=0.05, kspace_center_weight=0.3, kspace_mid_weight=1.0, tv_eps=0.04,
                edge_tv_kappa_source="zf", wav_name="db4", wav_levels=1, edge_tv_kappa_q=0.85, edge_tv_wfloor=0.1)
    loss.update(extra)
    plain = "FLAIR" if cell.endswith("FLAIR") else "T1"
    data = {"plain": {"name": plain, "known_zero": plain == "FLAIR"}}
    if CELLS[cell].trainer == "joint":
        data["contrast"] = {"name": "FLAIR", "known_zero": True}
    return {"loss": loss, "data": data, "adaptive_control": {"enable": False}, "voxelizer": {"max_radius": 20},
            "gaussian": {"init_scale_mode": "3nn", "num_private_points": 0, "initial_num_points": 3600000,
                         "init_mode": "support_file", "init_support_path": "/x/support.npz"},
            "training": {"seed": 42, "max_iterations": 4000}}


def test_parse_iter_line():
    d = parse_iter_line(LINE_ON)
    assert d["it"] == 1 and d["R_geo"] == 3e-3 and d["R_wav"] == 4e-3 and d["dc_c"] == 2e-2


def test_audit_reg_on_cell_passes_with_recipe_lines():
    log = "[edge_tv] kappa_q=0.85 source=zf refresh_iter=0 -> kappa_plain=0.01 kappa_contrast=0.01 w_floor=0.1\n" \
          "[wavelet] lambda=0.1 (plain 0.12, contrast 0.1) wavelet=db4 levels=1 eps=0.0 cycle_spin=True on=complex\n" \
          "[phase-tv] lambda_ptv=0.03 eps=0.04\n" + LINE_ON + "\n" + LINE_ON.replace("    1/", " 4000/") + "\n"
    a = audit("A", _cfg(RES_ON), log)
    assert a["resolved_ok"] and a["startup_ok"] and a["loss_ok"] and a["ok"]


def test_audit_reg_off_cell_rejects_nonzero_logged_prior():
    log = LINE_ON + "\n" + LINE_OFF + "\n"
    a = audit("D_T1", _cfg(RES_OFF, "D_T1"), log)
    assert a["resolved_ok"] and not a["loss_ok"] and not a["ok"]


def test_audit_rejects_regulariser_startup_lines_in_reg_off_cell():
    log = "[wavelet] lambda=0.1 (plain 0.1)\n" + LINE_OFF + "\n" + LINE_OFF + "\n"
    a = audit("D_FLAIR", _cfg(RES_OFF, "D_FLAIR"), log)
    assert not a["startup_ok"] and not a["ok"]


def test_audit_rejects_wrong_resolved_wavelet_weight():
    bad = dict(RES_ON); bad["lambda_wav_plain"] = 0.12         # FLAIR single scored with T1's weight: the old confound
    log = LINE_ON + "\n" + LINE_ON + "\n"
    a = audit("B_FLAIR", _cfg(bad, "B_FLAIR"), log)
    assert not a["resolved_ok"] and not a["ok"]


def test_audit_rejects_nonzero_nl_weight():
    log = LINE_OFF + "\n" + LINE_OFF + "\n"
    a = audit("D_T1", _cfg(RES_OFF, "D_T1", lambda_nl_plain=0.1), log)
    assert not a["resolved_ok"] and not a["ok"]


def test_audit_rejects_non_adaptive_kspace_weighting():
    log = LINE_OFF + "\n" + LINE_OFF + "\n"
    a = audit("D_T1", _cfg(RES_OFF, "D_T1", kspace_freq_adaptive=False), log)
    assert not a["resolved_ok"] and not a["ok"]


ON_LOG = LINE_ON + "\n" + LINE_ON + "\n"
ON_LOG_STARTUP = ("[edge_tv] kappa_q=0.85\n[wavelet] lambda=0.1\n[phase-tv] lambda_ptv=0.03\n" + ON_LOG)


def _set(cfg, path, value):
    d = cfg
    keys = path.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value
    return cfg


def test_audit_accepts_full_config_with_matching_args():
    a = audit("A", _cfg(RES_ON), ON_LOG_STARTUP, ARGS)
    assert a["resolved_ok"] and a["ok"]
    assert audit("D_FLAIR", _cfg(RES_OFF, "D_FLAIR"), LINE_OFF + "\n" + LINE_OFF + "\n", ARGS)["resolved_ok"]


@pytest.mark.parametrize("path,value", [
    ("loss.kspace_high_weight_plain", 0.5), ("loss.kspace_freq_curriculum", True), ("loss.kspace_center_weight", 0.9),
    ("loss.kspace_mid_weight", 0.5), ("loss.kspace_center_fraction", 0.1), ("loss.prewhiten", True),
    ("loss.lambda_private_sp", 1.0), ("loss.wav_name", "db2"), ("loss.wav_levels", 3), ("loss.edge_tv_kappa_q", 0.5),
    ("loss.edge_tv_wfloor", 0.5), ("loss.tv_eps", 0.0), ("loss.edge_tv_kappa_source", "render"),
    ("gaussian.num_private_points", 5), ("adaptive_control.enable", True), ("voxelizer.max_radius", 10),
    ("gaussian.init_scale_mode", "fixed"), ("data.plain.prewhiten", True), ("data.contrast.prewhiten", False),
    ("data.contrast.known_zero", False), ("data.plain.known_zero", True)])
def test_audit_rejects_each_frozen_knob_mutation(path, value):
    a = audit("A", _set(_cfg(RES_ON), path, value), ON_LOG_STARTUP)
    assert not a["resolved_ok"] and not a["ok"]


@pytest.mark.parametrize("path", ["loss.tv_eps", "loss.edge_tv_kappa_source", "gaussian.init_scale_mode",
                                  "loss.wav_name"])
def test_audit_rejects_absent_key_whose_trainer_default_differs(path):
    cfg = _cfg(RES_ON)
    d = cfg
    keys = path.split(".")
    for k in keys[:-1]:
        d = d[k]
    del d[keys[-1]]
    assert not audit("A", cfg, ON_LOG_STARTUP)["resolved_ok"]


def test_audit_known_zero_by_channel_name():
    assert not audit("B_FLAIR", _set(_cfg(RES_ON, "B_FLAIR"), "data.plain.known_zero", False), ON_LOG_STARTUP)["resolved_ok"]
    assert not audit("D_T1", _set(_cfg(RES_OFF, "D_T1"), "data.plain.known_zero", True), LINE_OFF)["resolved_ok"]


@pytest.mark.parametrize("key,value", [("seed", 43), ("npts", 1000), ("init", "head"), ("support", "/y/other.npz"),
                                       ("iters", 2)])
def test_audit_rejects_launcher_argument_mismatch(key, value):
    a = audit("A", _cfg(RES_ON), ON_LOG_STARTUP, {**ARGS, key: value})
    assert not a["resolved_ok"] and not a["ok"]


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
@pytest.mark.parametrize("key", ["dc_p", "R_nl"])
def test_audit_rejects_non_finite_logged_values(bad, key):
    line = LINE_ON.replace(f"{key}=" + ("1.0000e-01" if key == "dc_p" else "0.0000e+00"), f"{key}={bad}")
    assert f"{key}={bad}" in line
    assert not audit("A", _cfg(RES_ON), ON_LOG_STARTUP.replace(LINE_ON, line))["loss_ok"]


def test_audit_rejects_missing_dc_p_and_missing_prior_r_in_reg_off_cell():
    no_dc = LINE_OFF.replace("dc_p=1.0000e-01 ", "")
    assert not audit("D_T1", _cfg(RES_OFF, "D_T1"), no_dc + "\n" + no_dc + "\n")["loss_ok"]
    no_geo = LINE_OFF.replace("R_geo=0.0000e+00 ", "")
    assert not audit("D_T1", _cfg(RES_OFF, "D_T1"), no_geo + "\n" + no_geo + "\n")["loss_ok"]
