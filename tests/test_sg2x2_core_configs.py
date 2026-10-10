"""Per-subject base configs of the sg2x2 core test (scripts/sg2x2_core_configs.py): copies of the development
configs with only data_path / csm_path changed; the committed files match the generator; nothing is overwritten.
Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_sg2x2_core_configs.py -q
"""
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sg2x2_core_configs import CORE_DIR, TEMPLATES, subject_config, write_subject  # noqa: E402
from sg2x2_core_subjects import SUBJECTS, subject_paths  # noqa: E402


def _strip_paths(cfg):
    for slot in cfg["data"].values():
        slot.pop("data_path")
        slot.pop("csm_path")
    return cfg


@pytest.mark.parametrize("kind", sorted(TEMPLATES))
def test_only_the_two_data_paths_change(kind):
    tmpl = yaml.safe_load(open(ROOT / TEMPLATES[kind]))
    for s in SUBJECTS:
        cfg = subject_config(tmpl, s)
        assert _strip_paths(cfg) == _strip_paths(yaml.safe_load(open(ROOT / TEMPLATES[kind])))
        cfg = subject_config(tmpl, s)
        p = subject_paths(s)
        for slot in cfg["data"].values():
            assert slot["data_path"] == p["ksp"][slot["name"]] and slot["csm_path"] == p["csm"][slot["name"]]
            assert slot["known_zero"] is True if slot["name"] == "FLAIR" else not slot.get("known_zero", False)


def test_committed_configs_match_the_generator():
    for s in SUBJECTS:
        for kind, tpath in TEMPLATES.items():
            got = yaml.safe_load(open(ROOT / CORE_DIR / s / f"{kind}.yaml"))
            assert got == subject_config(yaml.safe_load(open(ROOT / tpath)), s), (s, kind)


def test_write_subject_refuses_to_overwrite(tmp_path):
    for tpath in TEMPLATES.values():
        (tmp_path / tpath).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / tpath, tmp_path / tpath)
    written = write_subject("huangyuwei_20230705", root=tmp_path)
    assert sorted(written) == sorted(f"{CORE_DIR}/huangyuwei_20230705/{k}.yaml" for k in TEMPLATES)
    first = (tmp_path / written[0]).read_text()
    assert first.startswith("# sg2x2 core test, subject huangyuwei_20230705")
    with pytest.raises(FileExistsError):
        write_subject("huangyuwei_20230705", root=tmp_path)
