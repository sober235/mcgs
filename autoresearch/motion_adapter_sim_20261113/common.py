"""Constants of the Motion-Aware MC-GS simulation study (design: docs/superpowers/specs/2026-11-13-motion-adapter-
sim-design.md). No third-party imports, so this file loads on the Mac and on the server."""
from pathlib import Path

# ---- frozen trainer and data (read-only sources)
SRC = Path("/data0/congcong/code/GS/mcgs_sg2x2_wt")
SRC_HEAD = "54f8a2a3bef402e785314b748f730a9623ca9d43"
CODE_DIRS = ("arc", "paired_gs", "diag_artefact")
ROOT = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_motion_adapter_20261113")
TR = ROOT / "trainer"
RES = TR / "results_arc"
DATA = ROOT / "data"
CFG = ROOT / "configs"
LEDGER = ROOT / "ledger.jsonl"
HYW = Path("/data0/congcong/code/GS/Results_20260908/T1T2_AF12_retro/data")
BASE_CFG = SRC / "configs/t1t2_af12_hyw.yaml"
MASKS = {"af12": HYW / "mask_af12_3d.h5",
         "af16": Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_retro/data/mask_af16_3d.h5")}

# ---- recipes: copied from autoresearch/ce_af16_ablation_20261110/abl_framework.py (tests/test_common.py compares)
V32 = ["--iters", "4000", "--lambda_geo", "0.07", "--warmup", "500", "--max_radius", "20", "--kspace_high", "2.2",
       "--edge_tv", "--edge_tv_kappa_q", "0.85", "--edge_tv_wfloor", "0.10", "--tv_eps", "0.04",
       "--set", "gaussian.init_mode=head", "--set", "gaussian.init_scale_mode=3nn",
       "--set", "loss.edge_tv_kappa_source=zf", "--set", "gaussian.initial_num_points=2400000",
       "--set", "loss.lambda_wav=0.05", "--set", "loss.lambda_wav_plain=0.06"]
V32G = [*V32, "--set", "loss.edge_tv_guide_alpha=0.5"]
JOINT_ONLY = ("loss.edge_tv_guide_alpha=", "loss.ptv_guide_alpha=")
CROSS_OFF = ["--lambda_edge", "0", "--lambda_dtv", "0", "--lambda_vtv", "0"]

# ---- geometry and protocol
# axis 0 = S-I (readout): index 0 is the INFERIOR end (the neck leaves the FOV there), the vertex is near index 229
# (measured on the topic's T1, 2026-11-14); axis 1 = A-P (ky), index 0 anterior; axis 2 = L-R (kz); 1 mm
N = (256, 256, 176)
PIVOT_OFFSET_MM = (-70.0, 20.0, 0.0)      # simulated rotation pivot relative to the FOV centre: 70 inferior, 20 posterior
# hyw's own canonical(T1) -> T2 displacement, image-based: NGF peak inside the brain mask in the trainer's frame
# (probe_native.py, 2026-11-14; NMI gives (0.23, 0.10, -0.06)). The T2 anatomy sits about 0.5 mm towards the vertex.
# The value used before the axis-0 correction was (0.59, 0.12, -0.13), measured on the neck and lower head.
NATIVE_SHIFT_MM = (0.50, 0.10, -0.14)
TR_S = 2.5
ETL = 320
K_SEG = 8


def _twice(v):
    return tuple(2.0 * x for x in v)


_E1A = dict(theta_deg=(0.5, 0.5, 1.5), t_sim_mm=(0.8, -0.5, 0.3))     # turn, tilt, nod (about axis 0, 1, 2)
_E1B = dict(theta_deg=_twice(_E1A["theta_deg"]), t_sim_mm=_twice(_E1A["t_sim_mm"]))
_NR_E2 = dict(seed=7, peak_mm=1.5, brain_max_mm=0.8)
_E4A = [dict(kind="step", at=0.60, theta_deg=(0.0, 0.0, 1.0), t_mm=(0.5, 0.0, 0.0))]
_E4B = [dict(kind="drift", theta_deg=(0.0, 0.3, 1.0), t_mm=(0.8, 0.0, 0.0))]
_D1 = dict(theta_deg=(-0.8, 0.9, -1.0), t_sim_mm=(-0.6, 0.7, -0.4))


def _cond(inter=None, nonrigid=None, lesions=False, intra=None):
    return dict(inter=inter, nonrigid=nonrigid, lesions=lesions, intra=intra)


CONDITIONS = {
    "C0": _cond(),
    "E1a": _cond(inter=_E1A),
    "E1b": _cond(inter=_E1B),
    "E2": _cond(inter=_E1A, nonrigid=_NR_E2),
    "E3": _cond(inter=_E1A, nonrigid=_NR_E2, lesions=True),
    "E4a": _cond(intra=_E4A),
    "E4b": _cond(intra=_E4B),
    "E4c": _cond(inter=_E1A, intra=_E4B),
    # development conditions: adapter settings are chosen on these only; never in the main tables
    "D1": _cond(inter=_D1),
    "D2": _cond(inter=_D1, nonrigid=dict(seed=101, peak_mm=1.2, brain_max_mm=0.6)),
    "D3": _cond(intra=[dict(kind="drift", theta_deg=(0.0, 0.4, -0.8), t_mm=(0.0, 0.6, 0.0)),
                       dict(kind="step", at=0.35, theta_deg=(0.7, 0.0, 0.0), t_mm=(0.0, 0.0, 0.0))]),
}


def _arm(trainer="joint", slot=None, rigid=False, nonrigid=False, segments=False, oracle=False):
    return dict(trainer=trainer, slot=slot, rigid=rigid, nonrigid=nonrigid, segments=segments, oracle=oracle)


ARMS = {
    "A0": _arm(),
    "A1": _arm(rigid=True),
    "A2": _arm(rigid=True, nonrigid=True),
    "A3": _arm(rigid=True, segments=True),
    "A23": _arm(rigid=True, nonrigid=True, segments=True),
    "O": _arm(oracle=True),                      # transform fixed to the truth; parts follow the condition
    "I-T1": _arm(trainer="single", slot="plain"),
    "I-T2": _arm(trainer="single", slot="contrast"),
    "I-T2+S": _arm(trainer="single", slot="contrast", segments=True),
}


def _rows(spec):
    return [(cond, arm) for conds, arms in spec for cond in conds for arm in arms]


MATRIX = {
    "af12_core": _rows([(("C0",), ("A0", "A1", "A2", "A3", "I-T1", "I-T2")),
                        (("E1a", "E2"), ("A0", "A1", "A2", "O", "I-T2")),
                        (("E3",), ("A0", "A2", "O", "I-T2", "I-T1")),
                        (("E4b",), ("A0", "A1", "A3", "O", "I-T2"))]),
    "af12_rest": _rows([(("E1b",), ("A0", "A1", "A2", "O", "I-T2")),
                        (("E4a",), ("A0", "A1", "A3", "O", "I-T2")),
                        (("E4c",), ("A0", "A23", "O", "I-T2"))]),
    "af16": _rows([(("C0",), ("A0", "A2", "I-T1", "I-T2")),
                   (("E1a", "E2"), ("A0", "A2", "O", "I-T2")),
                   (("E3",), ("A0", "A2", "O", "I-T2", "I-T1")),
                   (("E4b",), ("A0", "A3", "O", "I-T2"))]),
    "dev": [("D1", "A1"), ("D2", "A2"), ("D3", "A3")],
}


def exp_name(af, cond, arm, seed=42):
    assert af in MASKS and cond in CONDITIONS and arm in ARMS
    return f"ma_{af}_{cond}_{arm}" + ("" if seed == 42 else f"_s{seed}")
