"""Constants of the second version of the motion study: three small adjustments of MC-GS for mild motion (design:
docs/superpowers/specs/2026-11-21-motion-mild-adjust-design.md). No third-party imports; loads on the Mac and on the
server. The first version's directory is used as a library."""
import sys
from pathlib import Path

E1 = Path(__file__).resolve().parents[1] / "motion_adapter_sim_20261113"
if str(E1) not in sys.path:
    sys.path.insert(0, str(E1))

import common as c1  # noqa: E402  (first version: recipes, hyw paths, geometry, E1a / E2 definitions)

ROOT2 = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_motion_mild_adjust_20261121")
TR2 = ROOT2 / "trainer"
RES2 = TR2 / "results_arc"
DATA2 = ROOT2 / "data"                     # data/<order>/<cond>/
CFG2 = ROOT2 / "configs"
REG2 = ROOT2 / "reg"
LEDGER2 = ROOT2 / "ledger.jsonl"

ORDERS = ("il", "kz")                      # il: every echo train spans k-space (main); kz: the first version's order
ORDER_NAME = {"il": "interleaved", "kz": "linear"}
CONTAM_MM = 0.2                            # an echo train is contaminated above this mean brain displacement
SOFT = dict(seed=23, peak_mm=2.0, brain_max_mm=0.5)      # soft-tissue field that accompanies a movement


def _pulse(at, dur, theta_deg, t_mm, soft=1.0):
    return dict(kind="pulse", at=at, dur=dur, theta_deg=theta_deg, t_mm=t_mm, soft=soft)


def _step(at, theta_deg, t_mm, soft=1.0):
    return dict(kind="step", at=at, theta_deg=theta_deg, t_mm=t_mm, soft=soft)


# theta_deg = (turn, tilt, nod) about axis 0, 1, 2; t_mm along axis 0, 1, 2 (conventions of the first version)
_TA = [_pulse(0.40, 0.05, (0.0, 0.0, 1.0), (0.5, 0.0, 0.0))]
_TB = [_pulse(0.25, 0.04, (0.8, 0.0, 0.0), (0.0, 0.4, 0.0)),
       _pulse(0.70, 0.04, (0.0, 0.0, -1.0), (-0.5, 0.0, 0.0)),
       _step(0.74, (0.0, 0.0, 0.05), (0.1, 0.0, 0.0), soft=0.1)]
_TC = [_step(0.60, (0.0, 0.0, 1.0), (0.5, 0.0, 0.0))]
_DTA = [_pulse(0.55, 0.06, (0.9, 0.0, 0.0), (0.0, -0.5, 0.0))]
_DTC = [_step(0.45, (0.0, 0.7, -0.6), (0.0, 0.4, 0.3))]


def _cond(inter=None, nonrigid=None, lesions=False, intra=None, v1_data=None):
    return dict(inter=inter, nonrigid=nonrigid, lesions=lesions, intra=intra, v1_data=v1_data)


CONDITIONS2 = {
    "C0": _cond(),
    "E2": _cond(v1_data="E2"),                               # inter-scan only: the first version's files, read-only
    "Ta": _cond(intra=_TA),
    "Tb": _cond(intra=_TB),
    "Tc": _cond(intra=_TC),
    "E2Tb": _cond(inter=c1._E1A, nonrigid=c1._NR_E2, intra=_TB),
    "TbL": _cond(intra=_TB, lesions=True),
    # development conditions: settings are chosen on these only; never in the main tables
    "DTa": _cond(intra=_DTA),
    "DTc": _cond(intra=_DTC),
    "D2": _cond(v1_data="D2"),
}


def _arm(trainer="joint", slot=None, m1=False, m2="monitor", m3=None):
    return dict(trainer=trainer, slot=slot, m1=m1, m2=m2, m3=m3)


ARMS2 = {
    "B0": _arm(),                                            # MC-GS unchanged (the monitor only records)
    "B1": _arm(m1=True),
    "B2": _arm(m2="adaptive"),
    "B12": _arm(m1=True, m2="adaptive"),
    "B123": _arm(m1=True, m2="adaptive", m3="detect"),
    "Ow": _arm(m2="truth"),                                  # truth weights: upper bound of M2
    "Op": _arm(m1=True, m2="truth", m3="truth"),             # truth states, poses and weights: upper bound of M3
    "I-T2": _arm(trainer="single", slot="contrast"),
    "I-T2+w": _arm(trainer="single", slot="contrast", m2="adaptive"),
}


def _rows(spec):
    return [(order, cond, arm) for order, conds, arms in spec for cond in conds for arm in arms]


MATRIX2 = {
    "feas": [("il", "DTa", "B0"), ("il", "DTa", "Ow"), ("il", "C0", "B0")],
    # the D2/B1 row runs twice in the development round: alignment term on (tag d1) and off (tag d1r)
    "dev": [("il", "DTa", "B2"), ("il", "DTc", "B0"), ("il", "DTc", "B123"), ("il", "DTc", "Op"), ("il", "D2", "B1")],
    "gate": _rows([("il", ("C0",), ("B1", "B2", "B12", "B123", "I-T2+w")), ("kz", ("C0",), ("B2", "B123"))]),
    "upper": _rows([("il", ("Ta", "Tb", "Tc", "E2Tb"), ("B0", "Ow")), ("il", ("Tc",), ("Op",)),
                    ("kz", ("Ta", "Tc"), ("B0",)), ("kz", ("Tc",), ("Ow", "Op"))]),
    "rest": _rows([("il", ("C0",), ("I-T2",)), ("il", ("E2",), ("B0", "B1", "B123", "I-T2")),
                   ("il", ("Ta",), ("B2", "B123", "I-T2")), ("il", ("Tb",), ("B2", "B123", "I-T2", "I-T2+w")),
                   ("il", ("Tc",), ("B2", "B123", "I-T2")), ("il", ("E2Tb",), ("B12", "B123")),
                   ("il", ("TbL",), ("B0", "B123", "I-T2")), ("kz", ("Ta", "Tc"), ("B123",))]),
}
FORMAL = [("il", "C0", "B0"), *MATRIX2["gate"], *MATRIX2["upper"], *MATRIX2["rest"]]

THRESH2 = dict(feas_auc=0.95, feas_recover=0.5, min_loss_db=0.1,
               s1_psnr=-0.05, s1_w1_frac=0.99,
               s2_tre_rigid=0.2, s2_tre_nonrigid=0.3, s2_dT2=-0.1, s2_dT1=-0.05,
               s3_recover=0.5, s3_recall=0.9, s3_false=0.02, s3_pose_deg=0.3, s3_pose_mm=0.3,
               s4_recovery_drop=0.05, kz_harm=-0.05)


def exp_name(af, order, cond, arm, seed=42, tag=""):
    assert af in c1.MASKS and order in ORDERS and cond in CONDITIONS2 and arm in ARMS2, (af, order, cond, arm)
    return f"mm_{af}_{order}_{cond}_{arm}" + ("" if seed == 42 else f"_s{seed}") + (f"_{tag}" if tag else "")
