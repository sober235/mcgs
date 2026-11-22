import json

import numpy as np

import geom
import reg
from mcommon import DATA2, REG2, c1


def test_input_paths_by_condition():
    assert reg.t2_path("il", "C0") == c1.HYW / "ksp_T2.h5"                    # no simulated change: the measured data
    assert reg.t2_path("il", "D2") == c1.DATA / "D2" / "ksp_T2.h5"            # the first version's files
    assert reg.t2_path("il", "E2") == c1.DATA / "E2" / "ksp_T2.h5"
    assert reg.t2_path("kz", "Tc") == DATA2 / "kz" / "Tc" / "ksp_T2.h5"       # the motion-corrupted measurement
    assert reg.t2_path("il", "E2Tb") == DATA2 / "il" / "E2Tb" / "ksp_T2.h5"
    assert reg.t1_path("il", "TbL") == DATA2 / "il" / "TbL" / "ksp_T1.h5"
    assert reg.t1_path("il", "Tb") == c1.HYW / "ksp_T1.h5"
    assert reg.out_path("af12", "il", "D2") == REG2 / "af12_il_D2.json"


def test_record_and_load_pose(tmp_path):
    x = np.array([0.1, -0.2, 0.3, 0.5, 0.1, -0.14])
    rec = reg.record(x, -0.123, 1000)
    assert set(rec) == {"theta_deg", "t_mm", "cost", "brain_in_voxels"}
    assert rec["cost"] == -0.123 and rec["brain_in_voxels"] == 1000
    p = tmp_path / "r.json"
    p.write_text(json.dumps(rec))
    g = reg.load_pose(p)
    np.testing.assert_allclose(g.t, x[3:], atol=1e-12)
    np.testing.assert_allclose(g.R, geom.rotmat(np.deg2rad(x[:3])), atol=1e-12)
