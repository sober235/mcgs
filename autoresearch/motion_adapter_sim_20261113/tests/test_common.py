import ast
import pathlib

import common as c


def _const(name):
    src = pathlib.Path(c.__file__).resolve().parents[1] / "ce_af16_ablation_20261110/abl_framework.py"
    for node in ast.parse(src.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name:
            return node.value
    raise KeyError(name)


def test_v32_matches_framework_ablation():
    assert c.V32 == ast.literal_eval(_const("V32"))
    assert c.V32G == [*c.V32, "--set", "loss.edge_tv_guide_alpha=0.5"]
    assert c.SRC_HEAD == ast.literal_eval(_const("SRC_HEAD"))
    assert list(c.JOINT_ONLY) == list(ast.literal_eval(_const("JOINT_ONLY")))
    assert c.CROSS_OFF == ast.literal_eval(_const("CROSS_OFF"))


def test_matrix_counts_and_names():
    assert len(c.MATRIX["af12_core"]) + len(c.MATRIX["af12_rest"]) == 40   # design section 10, AF12 without dev runs
    assert len(c.MATRIX["af16"]) == 21
    for part in c.MATRIX.values():
        assert len(set(part)) == len(part)
        for cond, arm in part:
            assert cond in c.CONDITIONS and arm in c.ARMS
    assert c.exp_name("af12", "E2", "A2") == "ma_af12_E2_A2"
    assert c.exp_name("af16", "C0", "I-T2", seed=7) == "ma_af16_C0_I-T2_s7"
    assert c.CONDITIONS["E1b"]["inter"]["theta_deg"] == (1.0, 1.0, 3.0)
    assert c.CONDITIONS["E1b"]["inter"]["t_sim_mm"] == (1.6, -1.0, 0.6)
    assert c.CONDITIONS["E4c"]["inter"] == c.CONDITIONS["E1a"]["inter"]
    assert c.CONDITIONS["E4c"]["intra"] == c.CONDITIONS["E4b"]["intra"]
