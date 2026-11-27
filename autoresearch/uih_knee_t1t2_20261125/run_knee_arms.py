#!/usr/bin/env python3
"""run_knee_arms.py -- the nine comparator methods (eight arms) on the UIH knee T1 + T2 topic, AF12.

Thin wrapper around the comparator pipeline of the T1 + FLAIR test-set study
(autoresearch/t1flair_af_testset_20261121: ts_common.py, make_topics.py, run_arms.py), which is used unchanged:
the same cloned scripts of the jiangyizhou AF12 topic, the same command lines and the same frozen lambdas
(FROZEN[12], selected on the brain tuning subject; NOT tuned on the knee). Only four definitions are replaced
before those modules are imported:
  OUT      -> <knee output>/arms
  SCANS    -> {"knee_sNN"}
  AFS      -> (12,)
  MASK     -> the knee topic's mask_af12_3d.h5
  data_dir -> OUT/data/knee_sNN, links to the knee topic's k-space and coil maps
The cloned scripts only accept the contrast names T1 and FLAIR, so the knee T2 files are linked under the name
FLAIR (ksp_FLAIR.h5 -> ksp_T2.h5, csm_espirit_FLAIR.h5 -> csm_espirit_T2.h5). In every output of this wrapper
"FLAIR" therefore means the knee T2; T2 uses the lambdas registered for FLAIR.
The comparators see the prepared k-space (unacquired lines exactly zero) but, unlike the GS trainer, are not told
which lines are unacquired.

  gpu.py run --cpu   -- @python autoresearch/uih_knee_t1t2_20261125/run_knee_arms.py --subject 6 --make
  gpu.py run --cpu   -- @python autoresearch/uih_knee_t1t2_20261125/run_knee_arms.py --subject 6 --status
  gpu.py run --gpu N -- @python autoresearch/uih_knee_t1t2_20261125/run_knee_arms.py --subject 6 [--smoke] --rows classical ...
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TS = HERE.parent / "t1flair_af_testset_20261121"
sys.path.insert(0, str(TS))

argv = sys.argv[1:]
assert argv[:1] == ["--subject"], __doc__
SI = int(argv[1])
rest = argv[2:]
SCAN = "knee_s%02d" % SI
KOUT = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_uih_knee_t1t2_20261125/arms")
TOPIC = Path("/data0/congcong/code/GS/Results_20260908") / f"KNEE_UIH_T1T2_AF12_S{SI:02d}"
LINKS = {"ksp_T1.h5": "ksp_T1.h5", "ksp_FLAIR.h5": "ksp_T2.h5",
         "csm_espirit_T1.h5": "csm_espirit_T1.h5", "csm_espirit_FLAIR.h5": "csm_espirit_T2.h5"}

import ts_common as tc  # noqa: E402

tc.OUT = KOUT
tc.SCANS = {SCAN: dict(person=SCAN, group="knee", reuse=None)}
tc.AFS = (12,)
tc.MASK = lambda af: TOPIC / "data/mask_af12_3d.h5"
tc.data_dir = lambda scan: KOUT / "data" / scan

import make_topics as mt  # noqa: E402
import run_arms as ra     # noqa: E402

assert ra.OUT == KOUT and mt.OUT == KOUT and list(ra.SCANS) == [SCAN] and ra.MASK(12) == tc.MASK(12)

_plan = ra.arm_plan


def arm_plan(root, row, gpu, smoke=False):
    """run_arms.arm_plan; the MC-INR scoring job re-enters through this wrapper so that it sees the knee mask"""
    jobs, res = _plan(root, row, gpu, smoke)
    for j in jobs:
        if j["name"] == "mcinr_score":
            i = j["cmd"].index(str(Path(ra.__file__).resolve()))
            j["cmd"][i:i + 1] = [str(Path(__file__).resolve()), "--subject", str(SI)]
    return jobs, res


ra.arm_plan = arm_plan
ra.SELF_REL = "autoresearch/uih_knee_t1t2_20261125/run_knee_arms.py"


def make():
    d = tc.data_dir(SCAN)
    d.mkdir(parents=True, exist_ok=True)
    for name, src in LINKS.items():
        print(name, mt.put_link(d / name, TOPIC / "data" / src), flush=True)
    m = mt.make_topic(SCAN, 12)
    import h5py
    with h5py.File(d / "ksp_T1.h5", "r") as f, h5py.File(tc.MASK(12), "r") as g:
        print("kspace", f["kspace"].shape, "mask", g["mask"].shape, g["mask"].dtype, flush=True)
    print("TOPIC_OK", m["topic"], flush=True)


if rest[:1] == ["--make"]:
    make()
else:
    if "--rows" in rest:                                   # short arm names -> rows of this scan
        i = rest.index("--rows") + 1
        j = i
        while j < len(rest) and not rest[j].startswith("--"):
            rest[j] = rest[j] if ":" in rest[j] else f"{SCAN}:12:{rest[j]}"
            j += 1
    sys.argv = [sys.argv[0], *rest]
    ra.main()
