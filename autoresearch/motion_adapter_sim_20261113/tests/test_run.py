import json

import pytest

import run
from common import CONDITIONS, DATA, HYW, K_SEG, MASKS, MATRIX


def test_dry_commands():
    exp, cmd = run.command("af12", "E2", "A2", "3", 42, False)
    assert exp == "ma_af12_E2_A2" and cmd[2].endswith("patch.py") and "loss.edge_tv_guide_alpha=0.5" in cmd
    assert cmd[cmd.index("--gpu") + 1] == "3" and "--save_volumes" in cmd
    exp, cmd = run.command("af12", "E2", "I-T2", "3", 42, False)
    assert cmd[2].endswith("arc/cycle.py") and cmd[cmd.index("--trainer") + 1] == "single"
    assert not any(a.startswith("loss.edge_tv_guide_alpha") for a in cmd)
    assert "loss.lambda_wav=0.05" in cmd and "loss.lambda_wav_plain=0.05" in cmd          # T2's own wavelet weight
    exp, cmd = run.command("af12", "E3", "I-T1", "3", 42, False)
    assert "loss.lambda_wav=0.06" in cmd and "loss.lambda_wav_plain=0.06" in cmd
    exp, cmd = run.command("af12", "C0", "A0", "3", 42, False)
    assert cmd[2].endswith("arc/cycle.py") and "patch.py" not in " ".join(cmd)
    exp, cmd = run.command("af12", "E4b", "A0", "3", 42, False)
    assert cmd[2].endswith("patch.py")                           # intra-scan data need the reference-pose ground truth
    exp, cmd = run.command("af16", "C0", "A2", "1", 7, True)
    assert exp == "smoke_ma_af16_C0_A2_s7" and "training.seed=7" in cmd and "--save_volumes" not in cmd
    assert cmd[cmd.index("--iters") + 1] == "30"


def test_data_config_paths():
    c = run.data_config("af12", "C0", "A0")
    assert c["data"]["contrast"]["data_path"] == str(HYW / "ksp_T2.h5") and "motion_ksp_path" not in c["data"]["contrast"]
    assert c["data"]["plain"]["mask_path"] == str(MASKS["af12"])
    c = run.data_config("af16", "E3", "A2")
    assert c["data"]["contrast"]["data_path"] == str(DATA / "E3/ksp_T2.h5")
    assert c["data"]["plain"]["data_path"] == str(DATA / "E3/ksp_T1.h5")
    assert c["data"]["contrast"]["mask_path"] == str(MASKS["af16"]) and c["data"]["contrast"]["known_zero"] is True
    assert c["data"]["contrast"]["acceleration_factor"] == 16
    c = run.data_config("af12", "E4b", "A3", gt_ksp={"E4b": str(HYW / "ksp_T2.h5")})
    assert c["data"]["contrast"]["data_path"] == str(HYW / "ksp_T2.h5")
    assert c["data"]["contrast"]["motion_ksp_path"] == str(DATA / "E4b/ksp_T2.h5")
    c = run.data_config("af12", "E4c", "I-T2", gt_ksp={"E4c": str(DATA / "E4c/gt_ksp_T2.h5")})
    assert list(c["data"]) == ["plain"] and c["data"]["plain"]["name"] == "T2"
    assert c["data"]["plain"]["data_path"] == str(DATA / "E4c/gt_ksp_T2.h5")
    assert c["data"]["plain"]["motion_ksp_path"] == str(DATA / "E4c/ksp_T2.h5")
    c = run.data_config("af12", "E2", "I-T1")
    assert c["data"]["plain"]["data_path"] == str(HYW / "ksp_T1.h5")


def test_motion_config_per_arm():
    s = run.DEFAULT_SETTINGS
    m = run.motion_config("af12", "E2", "A2", 42, s, ref_seg=3)
    assert (m["rigid"], m["nonrigid"], m["segments"], m["oracle"], m["channel"]) == (True, True, 0, False, "contrast")
    m = run.motion_config("af12", "E2", "O", 42, s, ref_seg=3)
    assert (m["rigid"], m["nonrigid"], m["segments"], m["oracle"]) == (True, True, 0, True)
    assert m["truth_path"].endswith("E2/motion_truth.json") and m["disp_path"].endswith("E2/disp_T2.npy")
    m = run.motion_config("af12", "E4b", "O", 42, s, ref_seg=3)
    assert (m["nonrigid"], m["segments"], m["ref_seg"]) == (False, K_SEG, 3)
    assert m["schedule_path"].endswith("E4b/schedule.npz")
    m = run.motion_config("af12", "C0", "A3", 42, s, ref_seg=3)
    assert m["segments"] == K_SEG and m["schedule_path"].endswith("C0/schedule.npz")
    m = run.motion_config("af12", "E4c", "A23", 42, s, ref_seg=3)
    assert (m["rigid"], m["nonrigid"], m["segments"]) == (True, True, K_SEG)
    m = run.motion_config("af12", "E4b", "I-T2", 42, s, ref_seg=3)
    assert (m["rigid"], m["nonrigid"], m["segments"], m["oracle"], m["channel"]) == (False, False, 0, False, "plain")
    m = run.motion_config("af12", "E4b", "I-T2+S", 42, s, ref_seg=3)
    assert (m["rigid"], m["segments"], m["channel"]) == (False, K_SEG, "plain")
    import patch
    for cond, arm in [x for part in MATRIX.values() for x in part]:
        if run.via_patch(cond, arm):
            patch.validate(run.motion_config("af12", cond, arm, 42, s, ref_seg=3))


def test_ledger_row_has_no_clock_fields():
    half = dict(psnr=30.0, ssim=0.9, nmse=0.01, hfen=0.1, e_unobs=0.3, head_psnr=35.8, head_ssim=0.95,
                head_hfen=0.07, sharp=1.0, grain=0.0, comp=1.0, beat=True, physics={}, zf=(1, 2, 3, 4))
    score = dict(exp="ma_af12_C0_A0", elapsed_s=3000.5, argv=["--exp", "x"], plain=half, contrast=dict(half, head_psnr=36.3),
                 vram_gb=20.0, npts=2400000)
    row = run.ledger_row(score, dict(exp="ma_af12_C0_A0", af="af12", cond="C0", arm="A0", seed=42,
                                     config_sha256="c" * 64, source_digest="d" * 64, motion_file=None))
    text = json.dumps(row)
    assert not any(k in text for k in ("elapsed", "utc", "timestamp", "seconds"))
    assert row["contrast"]["head_psnr"] == 36.3 and row["plain"]["nmse"] == 0.01 and row["vram_gb"] == 20.0
    single = run.ledger_row(dict(score, contrast=None), dict(exp="e", af="af12", cond="C0", arm="I-T2", seed=42,
                                                              config_sha256="c", source_digest="d", motion_file=None))
    assert single["contrast"] is None


def test_output_filter_removes_durations():
    f = run.strip_durations
    assert f("SCORE combined=1.0 both_beat_zf=True N=24 vram_gb=20.1 elapsed_s=3021\n") == \
        "SCORE combined=1.0 both_beat_zf=True N=24 vram_gb=20.1\n"
    assert f("joint training done in 2999.1s\n") == "joint training done\n"
    assert f("[nl] iter=500 contrast=T1 sigma=0.01 crop=(1, 2) bm4d=12.3s\n") == "[nl] iter=500 contrast=T1 sigma=0.01 crop=(1, 2)\n"
    assert f("[iter   50/4000] loss=1.0e-02\n") == "[iter   50/4000] loss=1.0e-02\n"


def test_matrix_rows_all_have_commands():
    n = 0
    for part in ("af12_core", "af12_rest", "af16", "dev"):
        af = "af16" if part == "af16" else "af12"
        for cond, arm in MATRIX[part]:
            exp, cmd = run.command(af, cond, arm, "0", 42, False)
            assert exp.startswith(f"ma_{af}_{cond}_") and "--base_cfg" in cmd
            n += 1
    assert n == 40 + 21 + 3


def test_tag_names_a_development_round():
    exp, cmd = run.command("af12", "D2", "A2", "3", 42, False, tag="r1")
    assert exp == "ma_af12_D2_A2_r1" and cmd[cmd.index("--exp") + 1] == exp
    assert cmd[cmd.index("--motion") + 1].endswith("ma_af12_D2_A2_r1.json")
    cfg = run.motion_config("af12", "D2", "A2", 42, run.DEFAULT_SETTINGS, ref_seg=3, tag="r1")
    assert cfg["exp"] == exp
    row = run.ledger_row(dict(plain=None, contrast=None),
                         dict(exp=exp, af="af12", cond="D2", arm="A2", seed=42, tag="r1", config_sha256="c",
                              source_digest="d", motion_file=None))
    assert row["tag"] == "r1"
    assert run.ledger_row(dict(plain=None, contrast=None),
                          dict(exp="e", af="af12", cond="C0", arm="A0", seed=42, config_sha256="c", source_digest="d",
                               motion_file=None))["tag"] == ""
