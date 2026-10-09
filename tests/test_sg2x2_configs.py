"""The three sg2x2 base configs: same data / gaussian / training / voxelizer as the AF16 topic config, single
configs carry one contrast only, FLAIR keeps its shutter, and the init support file exists with the volume's shape.
Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_sg2x2_configs.py -q
"""
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
TOPIC = yaml.safe_load(open(ROOT / "configs/t1flair_v3_af16_jyz.yaml"))
JOINT = yaml.safe_load(open(ROOT / "configs/sg2x2/af16_jyz_joint.yaml"))
T1 = yaml.safe_load(open(ROOT / "configs/sg2x2/af16_jyz_single_T1.yaml"))
FL = yaml.safe_load(open(ROOT / "configs/sg2x2/af16_jyz_single_FLAIR.yaml"))


def test_joint_base_equals_topic_config_outside_output():
    for k in ("data", "gaussian", "loss", "training", "voxelizer", "mode", "hardware"):
        assert JOINT[k] == TOPIC[k], k


def test_single_configs_carry_one_contrast_and_the_right_block():
    assert "contrast" not in T1["data"] and T1["data"]["plain"] == TOPIC["data"]["plain"]
    assert "contrast" not in FL["data"] and FL["data"]["plain"] == TOPIC["data"]["contrast"]
    assert FL["data"]["plain"]["name"] == "FLAIR" and FL["data"]["plain"].get("known_zero") is True
    assert T1["data"]["plain"]["name"] == "T1" and not T1["data"]["plain"].get("known_zero", False)
    for c in (T1, FL):
        for k in ("gaussian", "loss", "training", "voxelizer"):
            assert c[k] == TOPIC[k], k


def test_init_support_file_matches_the_volume():
    z = np.load(ROOT / "configs/sg2x2/init_support_af16_jyz.npz")
    sup = z["support"]
    assert sup.dtype == bool and sup.shape == (256, 256, 176)
    assert 0.3 < sup.mean() < 0.7       # the dev subject's union-ZF support covers ~43 % of the box; dilation adds a little


def test_launcher_refuses_existing_outputs(tmp_path):
    import subprocess
    exp = "sg2x2_refuse_test"
    (ROOT / "results_arc").mkdir(exist_ok=True)
    marker = ROOT / "results_arc" / f"{exp}_score.json"
    marker.write_text("{}")
    try:
        r = subprocess.run(["bash", str(ROOT / "run_sg2x2.sh"), exp, "0", "A", "1"], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 1 and "refusing to overwrite" in r.stderr
    finally:
        marker.unlink()
