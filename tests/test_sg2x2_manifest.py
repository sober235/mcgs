"""The regulariser audit of scripts/sg2x2_manifest.py: a reg-off cell must show zero weighted priors in the resolved
config, no prior start-up line, and zero R_* on the first and last logged iteration; a reg-on cell must show the
frozen recipe. Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_sg2x2_manifest.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sg2x2_manifest import audit, parse_iter_line  # noqa: E402

LINE_ON = ("[iter     1/4000] loss=1.2345e-01 dc_p=1.0000e-01 dc_c=2.0000e-02 R_geo=3.0000e-03 R_dtv=0.0000e+00 "
           "R_edge=0.0000e+00 R_vtv=0.0000e+00 R_dtvp=0.0000e+00 R_wav=4.0000e-03 R_nl=0.0000e+00 balance_violation=False")
LINE_OFF = ("[iter  4000/4000] loss=1.0000e-01 dc_p=1.0000e-01 dc_c=0.0000e+00 R_geo=0.0000e+00 R_dtv=0.0000e+00 "
            "R_edge=0.0000e+00 R_vtv=0.0000e+00 R_dtvp=0.0000e+00 R_wav=0.0000e+00 R_ptv=0.0000e+00 R_nl=0.0000e+00 balance_violation=False")
RES_ON = {"lambda_geo": 0.10, "edge_tv": True, "lambda_wav": 0.10, "lambda_wav_plain": 0.12, "lambda_ptv": 0.03,
          "lambda_edge": 0.0, "lambda_dtv": 0.0, "lambda_vtv": 0.0, "lambda_dtv_proj": 0.0, "edge_tv_guide_alpha": 0.0,
          "ptv_guide_alpha": 0.0, "kspace_freq_adaptive": True, "kspace_high_weight": 2.2}
RES_OFF = {**{k: 0.0 for k in RES_ON}, "edge_tv": False, "kspace_freq_adaptive": True, "kspace_high_weight": 2.2}


def _cfg(res, **extra):
    loss = {"tv_weight": res["lambda_geo"], **{k: v for k, v in res.items() if k != "lambda_geo"}}
    loss.update(extra)
    return {"loss": loss}


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
    a = audit("D_T1", _cfg(RES_OFF), log)
    assert a["resolved_ok"] and not a["loss_ok"] and not a["ok"]


def test_audit_rejects_regulariser_startup_lines_in_reg_off_cell():
    log = "[wavelet] lambda=0.1 (plain 0.1)\n" + LINE_OFF + "\n" + LINE_OFF + "\n"
    a = audit("D_FLAIR", _cfg(RES_OFF), log)
    assert not a["startup_ok"] and not a["ok"]


def test_audit_rejects_wrong_resolved_wavelet_weight():
    bad = dict(RES_ON); bad["lambda_wav_plain"] = 0.12         # FLAIR single scored with T1's weight: the old confound
    log = LINE_ON + "\n" + LINE_ON + "\n"
    a = audit("B_FLAIR", _cfg(bad), log)
    assert not a["resolved_ok"] and not a["ok"]


def test_audit_rejects_nonzero_nl_weight():
    log = LINE_OFF + "\n" + LINE_OFF + "\n"
    a = audit("D_T1", _cfg(RES_OFF, lambda_nl_plain=0.1), log)
    assert not a["resolved_ok"] and not a["ok"]


def test_audit_rejects_non_adaptive_kspace_weighting():
    log = LINE_OFF + "\n" + LINE_OFF + "\n"
    a = audit("D_T1", _cfg(RES_OFF, kspace_freq_adaptive=False), log)
    assert not a["resolved_ok"] and not a["ok"]
