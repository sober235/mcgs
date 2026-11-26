"""Constants of the third motion study: non-rigid motion inside the T2W scan, T1W without motion (plan:
docs/superpowers/plans/2026-11-25-motion-nonrigid.md). No third-party imports.

The first and second versions' directories are used as libraries and their files are not changed. Importing this
module FIRST points the second version's constants at this study: its conditions and arms are added to the second
version's tables and its server root, data, config, registration, result and ledger paths are replaced. Every tool
of the second version (msim, reg, mrun, mpatch, mevaluate) imported afterwards then works on this study's root."""
import sys
from pathlib import Path

E2 = Path(__file__).resolve().parents[1] / "motion_mild_adjust_20261121"
if str(E2) not in sys.path:
    sys.path.insert(0, str(E2))
for _m in ("msim", "reg", "mrun", "mpatch", "mevaluate"):
    assert _m not in sys.modules, f"import ncommon before {_m}"

import mcommon as c2  # noqa: E402  (second version: orders, arms, paths; puts the first version on the path too)

c1 = c2.c1
ROOT3 = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_motion_nonrigid_20261125")
DATA3 = ROOT3 / "data"                     # data/<order>/<cond>/
PROBE3 = ROOT3 / "probe"
V2_RES = c2.RES2                           # the second version's results (read-only: the probe's model)

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
    "C0": dict(c2.CONDITIONS2["C0"], field=None),
    "Na": _cond([_pulse(0.40, 0.05)]),                       # one brief deformation
    "Nc": _cond([_step(0.60)]),                              # a lasting deformation
    "Ncs": _cond([_step(0.60, soft=0.5)]),                   # the same at half the amplitude ("Nc-s" in the plan)
    "NcL": _cond([_step(0.60)], lesions=True),
    # development conditions: settings are chosen on these only
    "DNa": _cond([_pulse(0.55, 0.06)], field="dev"),
    "DNc": _cond([_step(0.45)], field="dev"),
}
# N4: M1 + M2 + detected states, every state with a rigid pose (second version) AND a control-grid field (m4);
# On: the truth's states, weights and field (fitted to the control grid), fixed
ARMS3 = {
    **{a: dict(c2.ARMS2[a], m4=None) for a in ("B0", "B2", "B123", "Ow")},
    "N4": dict(c2._arm(m1=True, m2="adaptive", m3="detect"), m4="learn"),
    "On": dict(c2._arm(m1=True, m2="truth", m3="truth"), m4="truth"),
}
MATRIX3 = {
    "dev": [("il", "DNc", "N4"), ("il", "DNc", "B0"), ("il", "DNc", "On"), ("il", "DNa", "N4")],
    "gate": [("il", "C0", "N4")],
    "formal": [("il", c, a) for c in ("Na", "Nc") for a in ("B0", "B2", "B123", "Ow", "N4", "On")]
              + [("il", c, a) for c in ("Ncs", "NcL") for a in ("B0", "N4")],
}
THRESH3 = dict(n1_psnr=-0.05, n2_recover=0.5, min_loss_db=0.1, n3_field_mm=0.3, n3_field_rel=0.5,
               n4_recovery_drop=0.05, n5_dT1=-0.05)

# ---- point the second version at this study (in place for the tables, rebinding for the paths)
for _t in (c2.CONDITIONS2, c2.ARMS2, c2.MATRIX2):
    _t.clear()
c2.CONDITIONS2.update(CONDITIONS3)
c2.ARMS2.update(ARMS3)
c2.MATRIX2.update(MATRIX3)
c2.FORMAL[:] = [("il", "C0", "B0"), *MATRIX3["gate"], *MATRIX3["formal"]]
c2.ROOT2 = ROOT3
c2.TR2 = ROOT3 / "trainer"
c2.RES2 = c2.TR2 / "results_arc"
c2.DATA2 = DATA3
c2.CFG2 = ROOT3 / "configs"
c2.REG2 = ROOT3 / "reg"
c2.LEDGER2 = ROOT3 / "ledger.jsonl"
