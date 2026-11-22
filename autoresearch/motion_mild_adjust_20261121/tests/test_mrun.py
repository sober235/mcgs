import json
import pathlib
import sys

import pytest

import common as c1
import mrun
from mcommon import CFG2, DATA2, E1, RES2

sys.path.insert(0, str(E1 / "tools"))
import queue_study  # noqa: E402

M = pathlib.Path(mrun.__file__).resolve().parent
SETTINGS = json.loads((M / "settings_dev.json").read_text())


def test_command_and_config():
    exp, cmd = mrun.command("af12", "il", "DTa", "Ow", "3", 42, False)
    assert exp == "mm_af12_il_DTa_Ow" and cmd[2].endswith("mpatch.py")
    assert cmd[cmd.index("--motion") + 1] == str(CFG2 / "motion/mm_af12_il_DTa_Ow.json")
    assert "loss.edge_tv_guide_alpha=0.5" in cmd and cmd[cmd.index("--gpu") + 1] == "3" and "--save_volumes" in cmd
    assert cmd[cmd.index("--base_cfg") + 1] == str(CFG2 / "af12_il_DTa_joint.yaml")
    assert cmd[cmd.index("--iters") + 1] == "4000"
    exp, cmd = mrun.command("af12", "il", "Ta", "I-T2", "3", 42, False)
    assert cmd[2].endswith("mpatch.py") and cmd[cmd.index("--trainer") + 1] == "single"
    assert not any(a.startswith("loss.edge_tv_guide_alpha") for a in cmd)
    assert "loss.lambda_wav=0.05" in cmd and "loss.lambda_wav_plain=0.05" in cmd          # T2's own wavelet weight
    assert cmd[cmd.index("--base_cfg") + 1] == str(CFG2 / "af12_il_Ta_single_T2.yaml")
    exp, cmd = mrun.command("af12", "kz", "C0", "B0", "1", 7, True, tag="d1")
    assert exp == "smoke_mm_af12_kz_C0_B0_s7_d1" and "training.seed=7" in cmd
    assert cmd[cmd.index("--iters") + 1] == "30" and "--save_volumes" in cmd


def test_intra_condition_uses_reference_gt():
    gt = str(DATA2 / "il/E2Tb/gt_ksp_T2.h5")
    c = mrun.data_config("af12", "il", "E2Tb", "B0", gt_ksp=gt)
    assert c["data"]["contrast"]["data_path"] == gt
    assert c["data"]["contrast"]["motion_ksp_path"] == str(DATA2 / "il/E2Tb/ksp_T2.h5")
    assert c["data"]["plain"]["data_path"] == str(c1.HYW / "ksp_T1.h5")
    c = mrun.data_config("af12", "kz", "Ta", "I-T2", gt_ksp=str(c1.HYW / "ksp_T2.h5"))
    assert sorted(c["data"]) == ["plain"] and c["data"]["plain"]["data_path"] == str(c1.HYW / "ksp_T2.h5")
    assert c["data"]["plain"]["motion_ksp_path"] == str(DATA2 / "kz/Ta/ksp_T2.h5")


def test_static_lesion_and_first_version_data_paths():
    c = mrun.data_config("af12", "il", "C0", "B0")
    assert c["data"]["contrast"]["data_path"] == str(c1.HYW / "ksp_T2.h5")
    assert "motion_ksp_path" not in c["data"]["contrast"] and c["data"]["plain"]["mask_path"] == str(c1.MASKS["af12"])
    assert c["data"]["contrast"]["acceleration_factor"] == 12
    c = mrun.data_config("af12", "il", "E2", "B0")
    assert c["data"]["contrast"]["data_path"] == str(c1.DATA / "E2/ksp_T2.h5") and "motion_ksp_path" not in c["data"]["contrast"]
    c = mrun.data_config("af12", "il", "TbL", "B0", gt_ksp=str(c1.HYW / "ksp_T2.h5"))
    assert c["data"]["plain"]["data_path"] == str(DATA2 / "il/TbL/ksp_T1.h5")
    base = mrun.data_config("af12", "il", "C0", "B0")
    v1 = __import__("run").data_config("af12", "C0", "A0")
    assert base == v1                                         # C0 is the first version's C0 configuration


def test_motion_config():
    m = mrun.motion_config("af12", "il", "DTa", "Ow", 42, SETTINGS)
    assert m["exp"] == "mm_af12_il_DTa_Ow" and m["channel"] == "contrast" and m["order"] == "il"
    assert m["m2"] == dict(SETTINGS["m2"], mode="truth") and m["m1"] is None and m["m3"] is None
    assert m["schedule_path"] == str(DATA2 / "il/DTa/schedule.npz")
    assert m["truth_path"] == str(DATA2 / "il/DTa/motion_truth.json") and m["out_dir"] == str(RES2)
    __import__("mpatch").validate(m)
    m = mrun.motion_config("af12", "il", "C0", "I-T2", 42, SETTINGS, smoke=True)
    assert m["channel"] == "plain" and m["m2"]["mode"] == "monitor"
    assert m["m2"]["warmup"] == 10 and m["m2"]["every"] == 10 and m["exp"] == "smoke_mm_af12_il_C0_I-T2"


def test_motion_config_m1():
    from mcommon import REG2
    m = mrun.motion_config("af12", "il", "D2", "B1", 42, SETTINGS)
    assert m["m1"] == dict(reg_path=str(REG2 / "af12_il_D2.json"), align=SETTINGS["m1"]["align"])
    assert m["m2"]["mode"] == "monitor" and m["channel"] == "contrast"
    __import__("mpatch").validate(m)
    off = dict(SETTINGS, m1=dict(align=None))                              # the alignment term switched off
    assert mrun.motion_config("af12", "il", "D2", "B1", 42, off)["m1"]["align"] is None
    assert mrun.motion_config("af12", "il", "DTa", "B12", 42, SETTINGS)["m2"]["mode"] == "adaptive"
    # the registration file is one of the files an M1 arm needs; arms without M1 do not ask for it
    assert REG2 / "af12_il_D2.json" in mrun.needed_files("af12", "il", "D2", "B1")
    assert REG2 / "af12_il_D2.json" not in mrun.needed_files("af12", "il", "D2", "B0")


def test_settings_file_has_the_plan_values():
    assert SETTINGS["m2"] == dict(z0=4.0, z1=8.0, cap=0.25, warmup=600, every=100, n_shells=24)
    assert SETTINGS["m3"]["at"] == 1000 and SETTINGS["m3"]["init"] == "reg" and SETTINGS["m1"]["align"]["end"] == 1500


def test_ledger_row_carries_the_order_and_no_clock():
    score = dict(plain=dict(head_psnr=35.0, psnr=30.0, elapsed_s=12.0), contrast=dict(head_psnr=36.0), npts=5,
                 vram_gb=3.0, elapsed_s=99.0, timestamp_utc="x", plain_name="a", contrast_name="b")
    row = mrun.ledger_row(score, dict(exp="e", af="af12", order="il", cond="Ta", arm="B2", seed=42, tag="",
                                      config_sha256="c", source_digest="d", motion_file="m", mm_file="mm"))
    assert row["order"] == "il" and row["mm_file"] == "mm" and row["contrast"] == dict(head_psnr=36.0)
    text = json.dumps(row)
    assert "elapsed" not in text and "timestamp" not in text


def test_queue_default_study_unchanged():
    v1 = "autoresearch/motion_adapter_sim_20261113/run.py"
    assert queue_study.launch_argv("v1", ("af12", "C0", "A0"), 3, 42, None, "") == \
        ["@python", v1, "--af", "af12", "--cond", "C0", "--arm", "A0", "--gpu", "3", "--seed", "42"]
    assert queue_study.launch_argv("v1", ("af12", "D3", "A3"), 0, 42, "settings_r4.json", "r4") == \
        ["@python", v1, "--af", "af12", "--cond", "D3", "--arm", "A3", "--gpu", "0", "--seed", "42",
         "--settings", "settings_r4.json", "--tag", "r4"]
    assert queue_study.exp_name("v1", ("af12", "E4b", "A3"), 7, "r2") == "ma_af12_E4b_A3_s7_r2"
    assert queue_study.parse_row("v1", "af12:C0:A0") == ("af12", "C0", "A0")
    p = queue_study.STUDIES["v1"]
    assert p["rel"] == "autoresearch/motion_adapter_sim_20261113" and p["root"].endswith("mcgs_motion_adapter_20261113")
    assert p["code"] == ("adapter.py", "common.py", "geom.py", "patch.py", "run.py", "schedule.py", "warp.py")


def test_queue_mild_study():
    assert queue_study.parse_row("mild", "af12:il:Ta:B2") == ("af12", "il", "Ta", "B2")
    assert queue_study.exp_name("mild", ("af12", "il", "Ta", "B2"), 42, "d1") == "mm_af12_il_Ta_B2_d1"
    assert queue_study.launch_argv("mild", ("af12", "il", "Ta", "B2"), 5, 42, "settings_dev.json", "d1") == \
        ["@python", "autoresearch/motion_mild_adjust_20261121/mrun.py", "--af", "af12", "--order", "il", "--cond", "Ta",
         "--arm", "B2", "--gpu", "5", "--seed", "42", "--settings", "settings_dev.json", "--tag", "d1"]
    p = queue_study.STUDIES["mild"]
    assert p["root"].endswith("mcgs_motion_mild_adjust_20261121")
    names = [pathlib.Path(f).name for f in p["code"]]
    for need in ("mpatch.py", "mrun.py", "mcommon.py", "weights.py", "adapter.py", "schedule.py", "geom.py"):
        assert need in names, need
    with pytest.raises(AssertionError):
        queue_study.parse_row("mild", "af12:Ta:B2")


def test_motion_config_m3():
    from mcommon import REG2
    m = mrun.motion_config("af12", "il", "DTc", "B123", 42, SETTINGS)
    assert m["m3"] == dict(SETTINGS["m3"], mode="detect") and m["m2"]["mode"] == "adaptive"
    assert m["m1"]["reg_path"] == str(REG2 / "af12_il_DTc.json")
    __import__("mpatch").validate(m)
    m = mrun.motion_config("af12", "il", "DTc", "Op", 42, SETTINGS)
    assert m["m3"]["mode"] == "truth" and m["m2"]["mode"] == "truth"
    __import__("mpatch").validate(m)
    m = mrun.motion_config("af12", "il", "DTc", "B123", 42, SETTINGS, smoke=True)
    assert m["m3"]["at"] == 20 and m["m2"]["warmup"] == 10 and m["m1"]["align"]["end"] == 20
    assert mrun.motion_config("af12", "il", "DTc", "B12", 42, SETTINGS)["m3"] is None
