"""Constants of the third motion study: non-rigid motion inside the T2W scan, T1W without motion (plan:
docs/superpowers/plans/2026-11-25-motion-nonrigid.md). No third-party imports. The first and second versions'
directories are used as libraries and are not changed."""
import sys
from pathlib import Path

E2 = Path(__file__).resolve().parents[1] / "motion_mild_adjust_20261121"
if str(E2) not in sys.path:
    sys.path.insert(0, str(E2))

import mcommon as c2  # noqa: E402  (second version: orders, arms, paths; puts the first version on the path too)

c1 = c2.c1
ROOT3 = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_motion_nonrigid_20261125")
DATA3 = ROOT3 / "data"                     # data/<order>/<cond>/
PROBE3 = ROOT3 / "probe"

# the deformation field of a condition (warp.rbf_field): formal conditions seed 31, development conditions seed 47
FIELDS = {"formal": dict(seed=31, peak_mm=3.0, brain_max_mm=1.5), "dev": dict(seed=47, peak_mm=3.0, brain_max_mm=1.5)}
_ZERO = (0.0, 0.0, 0.0)


def _pulse(at, dur, soft=1.0):
    return dict(kind="pulse", at=at, dur=dur, theta_deg=_ZERO, t_mm=_ZERO, soft=soft)


def _step(at, soft=1.0):
    return dict(kind="step", at=at, theta_deg=_ZERO, t_mm=_ZERO, soft=soft)


def _cond(intra, field="formal", lesions=False):
    return dict(inter=None, nonrigid=None, lesions=lesions, intra=intra, v1_data=None, field=field)


# no rigid part anywhere: train n is deformed by (a_n - a_ref) * field
CONDITIONS3 = {
    "Na": _cond([_pulse(0.40, 0.05)]),                       # one brief deformation
    "Nc": _cond([_step(0.60)]),                              # a lasting deformation
    "Ncs": _cond([_step(0.60, soft=0.5)]),                   # the same at half the amplitude ("Nc-s" in the plan)
    "NcL": _cond([_step(0.60)], lesions=True),
    # development conditions: settings are chosen on these only
    "DNa": _cond([_pulse(0.55, 0.06)], field="dev"),
    "DNc": _cond([_step(0.45)], field="dev"),
}
