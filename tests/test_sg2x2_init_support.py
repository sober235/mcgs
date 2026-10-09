"""init_mode=support_file: positions drawn inside a frozen boolean support, seeded, independent of the subject's
data; densities from each contrast's own ZF; shape mismatch refused before any training.
Run: PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=<idle> /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_sg2x2_init_support.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paired_gs"))
sys.path.insert(0, str(ROOT / "tests"))
from gaussian.init_cloud import load_init_support, sample_positions_in_support  # noqa: E402
from sg2x2_fixture import base_cfg, make_pair  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="trainer needs the CUDA voxelizer")


def test_sample_positions_in_support_inside_and_seeded():
    sup = torch.zeros(16, 20, 12, dtype=torch.bool)
    sup[4:12, 5:15, 3:9] = True
    g = torch.Generator().manual_seed(3)
    vox, pos = sample_positions_in_support(sup, 500, generator=g)
    assert vox.shape == (500, 3) and pos.shape == (500, 3)
    assert bool(sup[vox[:, 0], vox[:, 1], vox[:, 2]].all())
    assert pos.min() >= -1.0 and pos.max() <= 1.0
    g2 = torch.Generator().manual_seed(3)
    vox2, pos2 = sample_positions_in_support(sup, 500, generator=g2)
    assert torch.equal(vox, vox2) and torch.equal(pos, pos2)


def test_sample_positions_in_support_empty_raises():
    with pytest.raises(RuntimeError):
        sample_positions_in_support(torch.zeros(4, 4, 4, dtype=torch.bool), 10)


def test_load_init_support_reads_npz_and_checks_shape(tmp_path):
    sup = np.zeros((6, 5, 4), dtype=bool); sup[1:4, 1:3, 1:2] = True
    p = tmp_path / "s.npz"; np.savez_compressed(p, support=sup)
    t = load_init_support(str(p), (6, 5, 4), "cpu")
    assert t.dtype == torch.bool and int(t.sum()) == 6
    with pytest.raises(ValueError, match="shape"):
        load_init_support(str(p), (6, 5, 5), "cpu")


@cuda
def test_support_file_init_is_identical_for_joint_and_single_and_densities_are_own(tmp_path):
    from trainers.joint_trainer import JointTrainer
    from trainers.single_contrast_trainer import SingleContrastTrainer
    pair = make_pair(tmp_path)
    dev = torch.device("cuda", 0)
    joint = JointTrainer(base_cfg(pair, tmp_path, "j"), device=dev)
    single = SingleContrastTrainer(base_cfg(pair, tmp_path, "s", single="FLAIR"), device=dev)
    assert torch.equal(joint.model.positions, single.model.positions)
    assert torch.equal(joint.model.scales, single.model.scales)
    # densities: joint plain slot = T1 ZF, joint contrast slot = FLAIR ZF, single plain slot = FLAIR ZF
    assert torch.equal(joint.model.density_real_contrast, single.model.density_real)
    assert not torch.equal(joint.model.density_real, single.model.density_real)
    # every drawn point sits inside the frozen support
    sup = load_init_support(pair["support"], pair["shape"], dev)
    d, h, w = pair["shape"]
    p = joint.model.positions.detach()
    ix = ((p[:, 0] + 1) * 0.5 * d).clamp(0, d - 1).long(); iy = ((p[:, 1] + 1) * 0.5 * h).clamp(0, h - 1).long()
    iz = ((p[:, 2] + 1) * 0.5 * w).clamp(0, w - 1).long()
    assert bool(sup[ix, iy, iz].all())


@cuda
def test_support_file_positions_do_not_depend_on_the_data(tmp_path):
    from trainers.joint_trainer import JointTrainer
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    a = make_pair(tmp_path / "a", seed=1); b = make_pair(tmp_path / "b", seed=2)
    cfg_b = base_cfg(b, tmp_path / "b", "b"); cfg_b["gaussian"]["init_support_path"] = a["support"]   # same support
    dev = torch.device("cuda", 0)
    ta = JointTrainer(base_cfg(a, tmp_path / "a", "a"), device=dev)
    tb = JointTrainer(cfg_b, device=dev)
    assert torch.equal(ta.model.positions, tb.model.positions)


@cuda
def test_support_file_shape_mismatch_raises(tmp_path):
    from trainers.joint_trainer import JointTrainer
    pair = make_pair(tmp_path)
    bad = tmp_path / "bad.npz"; np.savez_compressed(bad, support=np.ones((8, 8, 8), dtype=bool))
    cfg = base_cfg(pair, tmp_path, "bad"); cfg["gaussian"]["init_support_path"] = str(bad)
    with pytest.raises(ValueError, match="shape"):
        JointTrainer(cfg, device=torch.device("cuda", 0))


@cuda
def test_support_file_logs_own_support_fraction(tmp_path, capsys):
    from trainers.joint_trainer import JointTrainer
    pair = make_pair(tmp_path)
    JointTrainer(base_cfg(pair, tmp_path, "log"), device=torch.device("cuda", 0))
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith("[init_cloud] mode=support_file"))
    assert "in_own_zf_support=" in line and "support_frac=" in line
