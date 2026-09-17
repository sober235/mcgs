import ast
from pathlib import Path

S = Path(__file__).resolve().parents[1]


def test_figure_scripts_parse_and_have_no_jyz_paths():
    for name in ("fig_three_planes_compact.py", "fig_enhancement.py"):
        src = (S / name).read_text()
        ast.parse(src)
        assert "T1FLAIR_AF16_v3_jiangyizhou" not in src and "mcgs_af16_wt/results_arc" not in src
        assert "RETRO_TOPIC" in src


def test_enhancement_script_reads_truth_and_arms():
    src = (S / "fig_enhancement.py").read_text()
    assert "arm_files(" in src and 'sim/gt.npz' in src and "enhance_rois_cssense.npz" in src
    assert "--gpu" not in src and "WORK" not in src
