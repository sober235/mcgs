# tests/test_sg2x2_cells.py
"""The four cells of the shared-geometry 2x2 ablation, as arc/cycle.py arguments (scripts/sg2x2_cells.py).

A/B carry the frozen AF16 phase-3 per-channel recipe and no cross-contrast weight; C/D carry no explicit image
prior at all; a single-contrast cell gets its own wavelet weight on both wavelet keys; every cell shares the
init mode, the support file, the point count and the seed.
Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_sg2x2_cells.py -q
"""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sg2x2_cells import CELLS, cell_args, resolved_switches  # noqa: E402

SUP = "/tmp/sup.npz"


def _flag_value(args, flag):
    i = args.index(flag)
    return args[i + 1]


def _sets(args):
    return {a.split("=", 1)[0]: a.split("=", 1)[1] for a in args if "=" in a and args[args.index(a) - 1] == "--set"}


def test_six_cells_and_their_structure():
    assert set(CELLS) == {"A", "B_T1", "B_FLAIR", "C", "D_T1", "D_FLAIR"}
    assert CELLS["A"].trainer == "joint" and CELLS["A"].reg and CELLS["A"].contrast is None
    assert CELLS["C"].trainer == "joint" and not CELLS["C"].reg
    assert CELLS["B_FLAIR"].trainer == "single" and CELLS["B_FLAIR"].contrast == "FLAIR" and CELLS["B_FLAIR"].reg
    assert CELLS["D_T1"].trainer == "single" and CELLS["D_T1"].contrast == "T1" and not CELLS["D_T1"].reg


def test_reg_on_cells_carry_the_frozen_recipe_and_no_cross_contrast_weight():
    for cell in ("A", "B_T1", "B_FLAIR"):
        a = cell_args(cell, npts=3_600_000, seed=42, support=SUP)
        assert _flag_value(a, "--lambda_geo") == "0.10" and "--edge_tv" in a
        assert _flag_value(a, "--edge_tv_kappa_q") == "0.85" and _flag_value(a, "--edge_tv_wfloor") == "0.10"
        assert _flag_value(a, "--tv_eps") == "0.04" and _flag_value(a, "--lambda_ptv") == "0.03"
        assert _flag_value(a, "--lambda_edge") == "0" and _flag_value(a, "--lambda_dtv") == "0"
        assert _flag_value(a, "--lambda_vtv") == "0"
        s = _sets(a)
        assert s["loss.wav_name"] == "db4" and s["loss.wav_levels"] == "1"
        assert s["loss.lambda_dtv_proj"] == "0" and s["loss.edge_tv_guide_alpha"] == "0" and s["loss.ptv_guide_alpha"] == "0"


def test_wavelet_weights_per_cell():
    assert _sets(cell_args("A", 1, 42, SUP))["loss.lambda_wav_plain"] == "0.12"
    assert _sets(cell_args("A", 1, 42, SUP))["loss.lambda_wav"] == "0.10"
    s_t1 = _sets(cell_args("B_T1", 1, 42, SUP))
    assert s_t1["loss.lambda_wav_plain"] == "0.12" and s_t1["loss.lambda_wav"] == "0.12"
    s_fl = _sets(cell_args("B_FLAIR", 1, 42, SUP))
    assert s_fl["loss.lambda_wav_plain"] == "0.10" and s_fl["loss.lambda_wav"] == "0.10"


def test_reg_off_cells_force_edge_tv_false():
    for cell in ("C", "D_T1", "D_FLAIR"):
        a = cell_args(cell, npts=3_600_000, seed=42, support=SUP)
        assert _flag_value(a, "--lambda_geo") == "0" and "--edge_tv" not in a
        assert _flag_value(a, "--lambda_ptv") == "0"
        s = _sets(a)
        assert s["loss.edge_tv"] == "false" and s["loss.lambda_wav"] == "0" and s["loss.lambda_wav_plain"] == "0"
        assert _flag_value(a, "--lambda_edge") == "0"


def test_shared_knobs_identical_across_cells():
    seen = set()
    for cell in CELLS:
        a = cell_args(cell, npts=1_800_000, seed=43, support=SUP, iters=4000, warmup=500)
        s = _sets(a)
        assert s["gaussian.init_mode"] == "support_file" and s["gaussian.init_support_path"] == SUP
        assert s["gaussian.initial_num_points"] == "1800000" and s["training.seed"] == "43"
        assert s["gaussian.init_scale_mode"] == "3nn" and s["loss.edge_tv_kappa_source"] == "zf"
        assert _flag_value(a, "--kspace_high") == "2.2" and _flag_value(a, "--warmup") == "500"
        assert _flag_value(a, "--max_radius") == "20" and _flag_value(a, "--iters") == "4000"
        seen.add((s["gaussian.init_mode"], s["gaussian.initial_num_points"], s["training.seed"]))
    assert len(seen) == 1


def test_trainer_flag_and_config_per_cell():
    a = cell_args("A", 1, 42, SUP)
    assert _flag_value(a, "--trainer") == "joint" and _flag_value(a, "--base_cfg") == "configs/sg2x2/af16_jyz_joint.yaml"
    b = cell_args("D_FLAIR", 1, 42, SUP)
    assert _flag_value(b, "--trainer") == "single" and _flag_value(b, "--base_cfg") == "configs/sg2x2/af16_jyz_single_FLAIR.yaml"


def test_head_init_variant_drops_the_support_file():
    s = _sets(cell_args("A", 1, 42, SUP, init="head"))
    assert s["gaussian.init_mode"] == "head" and "gaussian.init_support_path" not in s


def test_resolved_switches_table():
    on = resolved_switches("B_FLAIR")
    assert on["lambda_geo"] == 0.10 and on["edge_tv"] is True and on["lambda_wav_plain"] == 0.10 and on["lambda_ptv"] == 0.03
    off = resolved_switches("D_T1")
    assert off["lambda_geo"] == 0 and off["edge_tv"] is False and off["lambda_wav"] == 0 and off["lambda_ptv"] == 0
    for k in ("lambda_edge", "lambda_dtv", "lambda_vtv", "lambda_dtv_proj", "edge_tv_guide_alpha", "ptv_guide_alpha"):
        assert on[k] == 0 and off[k] == 0


def test_cli_prints_one_argument_per_line():
    out = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/sg2x2_cells.py"), "C",
                          "--npts", "100", "--seed", "42", "--support", SUP], capture_output=True, text=True, check=True).stdout
    lines = out.strip().split("\n")
    assert lines == cell_args("C", 100, 42, SUP)


def test_no_subject_is_todays_behaviour():
    from sg2x2_cells import cell_config
    for cell in CELLS:
        assert cell_config(cell) == CELLS[cell].config
        assert cell_args(cell, 3_600_000, 42, SUP) == cell_args(cell, 3_600_000, 42, SUP, subject=None)
        assert _flag_value(cell_args(cell, 3_600_000, 42, SUP), "--base_cfg") == CELLS[cell].config


def test_subject_selects_the_core_config():
    from sg2x2_cells import cell_config
    assert cell_config("A", "huangyuwei_20230705") == "configs/sg2x2/core/huangyuwei_20230705/joint.yaml"
    assert cell_config("C", "huangyuwei_20230705") == "configs/sg2x2/core/huangyuwei_20230705/joint.yaml"
    assert cell_config("B_FLAIR", "litianyu_20231107") == "configs/sg2x2/core/litianyu_20231107/single_FLAIR.yaml"
    assert cell_config("D_T1", "jiangyizhou_20230705") == "configs/sg2x2/core/jiangyizhou_20230705/single_T1.yaml"
    a = cell_args("D_FLAIR", 3_600_000, 42, SUP, subject="yangkun_20231102")
    assert _flag_value(a, "--base_cfg") == "configs/sg2x2/core/yangkun_20231102/single_FLAIR.yaml"
    assert a[:a.index("--base_cfg")] == cell_args("D_FLAIR", 3_600_000, 42, SUP)[:a.index("--base_cfg")]


def test_unknown_subject_raises():
    from sg2x2_cells import cell_config
    with pytest.raises(ValueError):
        cell_config("A", "nobody_20990101")
    with pytest.raises(ValueError):
        cell_args("A", 1, 42, SUP, subject="790_tangbin_20230706")


def test_cli_subject_and_unknown_subject():
    script = str(Path(__file__).resolve().parents[1] / "scripts/sg2x2_cells.py")
    ok = subprocess.run([sys.executable, script, "B_T1", "--npts", "100", "--seed", "42", "--support", SUP,
                         "--subject", "zhoushuo_20231030"], capture_output=True, text=True, check=True).stdout
    assert ok.strip().split("\n") == cell_args("B_T1", 100, 42, SUP, subject="zhoushuo_20231030")
    bad = subprocess.run([sys.executable, script, "B_T1", "--npts", "100", "--seed", "42", "--support", SUP,
                          "--subject", "nobody_20990101"], capture_output=True, text=True)
    assert bad.returncode != 0 and bad.stdout == ""
