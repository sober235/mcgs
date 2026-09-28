"""StrictConfig: reads that fall back to a code default are recorded (record mode) or refused (strict mode).

Run: env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_strict_config.py -q
"""
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "arc"))
from strict_config import StrictConfig, StrictConfigError  # noqa: E402

CFG = {"loss": {"tv_weight": 0.1, "edge_tv": True}, "training": {"max_iterations": 4000}, "name": "x"}


def test_present_keys_read_like_a_dict():
    s = StrictConfig(CFG)
    assert s.get("name", "d") == "x"
    assert s["loss"].get("tv_weight", 0.0) == 0.1
    assert s.get("training", {}).get("max_iterations", 8000) == 4000
    assert "edge_tv" in s["loss"]
    assert s.fallbacks == []


def test_missing_key_raises_in_strict_mode():
    s = StrictConfig(CFG)
    with pytest.raises(StrictConfigError, match="loss.lambda_ptv"):
        s["loss"].get("lambda_ptv", 0.0)


def test_missing_keys_recorded_in_record_mode_including_children_of_missing_sections():
    s = StrictConfig(CFG, strict=False)
    assert s["loss"].get("lambda_ptv", 0.0) == 0.0
    assert s.get("mode", {}).get("strict_cuda", False) is False
    assert s.fallbacks == [("loss.lambda_ptv", 0.0), ("mode", {}), ("mode.strict_cuda", False)]


def test_record_mode_logs_every_read_of_a_missing_key_with_its_own_default():
    # voxelizer_config.py:22 reads voxelizer.type with default "chunk", joint_trainer.py:344 with "tile_cuda"
    s = StrictConfig(CFG, strict=False)
    v = s.setdefault("voxelizer", {})
    assert v.get("type", "chunk") == "chunk"
    assert s.get("voxelizer", {}).get("type", "tile_cuda") == "tile_cuda"
    assert s.fallbacks == [("voxelizer", {}), ("voxelizer.type", "chunk"), ("voxelizer.type", "tile_cuda")]


def test_setdefault_is_tracked_and_stores_a_wrapped_value():
    s = StrictConfig(CFG, strict=False)
    v = s.setdefault("voxelizer", {})
    v["type"] = "tile_cuda"
    assert s["voxelizer"]["type"] == "tile_cuda"
    assert isinstance(s["voxelizer"], StrictConfig)
    assert ("voxelizer", {}) in s.fallbacks


def test_assignment_wraps_nested_dicts_and_keeps_paths():
    s = StrictConfig(CFG)
    s["output"] = {"output_dir": "/tmp/x"}
    assert isinstance(s["output"], StrictConfig)
    with pytest.raises(StrictConfigError, match="output.exp_name"):
        s["output"].get("exp_name", "d")


def test_dumps_as_plain_mapping():
    s = StrictConfig(CFG)
    assert yaml.safe_load(yaml.dump(s)) == CFG
    assert json.loads(json.dumps(s)) == CFG
    plain = s.to_plain()
    assert plain == CFG and type(plain) is dict and type(plain["loss"]) is dict
