#!/usr/bin/env python3
"""ts_common.py -- shared definitions of the different-mask study (T1 + FLAIR, acceleration factor 16) on the six test scans.

User request 2026-11-22: "different masks (uniform density, radial) are also applied to the test set", run separately
from the acceleration-factor test-set study (autoresearch/t1flair_af_testset_20261121, session mcgs-0f), whose
runner files this folder ports: the "af" key of that study becomes the sampling pattern here ("uniform" = uniform
density, "radial" = the September pseudo-radial pattern the user kept for the radial row on 2026-11-12). The
vd-Poisson AF 16 rows of the other study are the third pattern and are not rerun.

The frozen settings are those registered on the tuning subject (jiangyizhou, topics
T1FLAIR_AF16_{uniform,radial}_v3_jiangyizhou) and are checked against their summary.json by --selftest. Nothing is
tuned on the test scans. The prepared scan data of the other study (TS_OUT/data/<scan>) are read only; everything
this study writes goes under OUT.

    gpu.py run --cpu -- @python autoresearch/diffmask_testset_20261122/ts_common.py --selftest
"""
import hashlib
import json
import sys
from pathlib import Path

OUT = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_diffmask_testset_20261122")
TS_OUT = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_t1flair_af_testset_20261121")   # read only
RES = Path("/data0/congcong/code/GS/Results_20260908")

# file names are the real ones (the spellings differ for vol5)
SCANS = {
    "vol1_huangyuwei": dict(person="huangyuwei", t1="t1_fsp_train_huangyuwei_cc", flair="flair_train_huangyuwei_cc",
                            group="aligned", reuse=RES / "T1FLAIR_AF12_retro/data"),
    "vol2_jiangdian": dict(person="jiangdian", t1="t1_fsp_train_jiangdian_cc", flair="flair_train_jiangdian_cc",
                           group="aligned", reuse=None),
    "vol4_tangbing": dict(person="tangbing", t1="t1_fsp_train_tangbing_cc", flair="flair_train_tangbing_cc",
                          group="aligned", reuse=RES / "T1FLAIR_AF12_v4_tangbing/data"),
    "vol5_sunxinyi": dict(person="sunxinyi", t1="t1_fsp_train_sunxinyin_cc", flair="flair_train_sunyinyi_cc",
                          group="offset", reuse=None),
    "vol6_jiangdian_790": dict(person="jiangdian", t1="t1_fsp_train_jiangdian_790_cc",
                               flair="flair_train_jiangdian_790_cc", group="offset", reuse=None),
    "vol7_tangbin_790": dict(person="tangbing", t1="t1_fsp_train_tangbin_790_cc",
                             flair="flair_train_tangbin_790_cc", group="offset", reuse=None),
}
AFS = ("uniform", "radial")          # sampling patterns; the name AFS is kept so the ported runners read the same
TOPIC = {"uniform": RES / "T1FLAIR_AF16_uniform_v3_jiangyizhou", "radial": RES / "T1FLAIR_AF16_radial_v3_jiangyizhou"}
LABEL = {"uniform": "Uniform density", "radial": "Pseudo-radial"}

# lambdas selected on the tuning subject, per pattern (each topic's logs/summary.json); spirit = (kernel size, lambda)
FROZEN = {
    "uniform": dict(cssense={"T1": 0.0014, "FLAIR": 0.0014}, jcssense={"T1": 0.002, "FLAIR": 0.002},
                    spirit={"T1": (5, 0.001), "FLAIR": (5, 0.001)}),
    "radial": dict(cssense={"T1": 0.0014, "FLAIR": 0.001}, jcssense={"T1": 0.0014, "FLAIR": 0.0014},
                   spirit={"T1": (5, 0.0005), "FLAIR": (5, 0.0005)}),
}


def JYZ_TOPIC(af):
    return TOPIC[af]


def MASK(af):
    return JYZ_TOPIC(af) / "data/mask_af16_3d.h5"


def data_dir(scan):
    return TS_OUT / "data" / scan


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def claim(root, name):
    """Atomic claim: mkdir succeeds for exactly one caller. True if this caller got it."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        (root / name).mkdir()
        return True
    except FileExistsError:
        return False


def selftest():
    assert len(SCANS) == 6 and len({v["person"] for v in SCANS.values()}) == 4
    assert set(FROZEN) == set(AFS) == set(TOPIC)
    for af in AFS:
        rows = json.load(open(JYZ_TOPIC(af) / "logs/summary.json"))
        for c in ("T1", "FLAIR"):
            for m, key in (("cssense_o200", "cssense"), ("jcssense", "jcssense"), ("spirit", "spirit")):
                hit = [r for r in rows if r.get("method") == m and r.get("contrast") == c]
                assert len(hit) == 1, (af, m, c, len(hit))
                r = hit[0]
                want = FROZEN[af][key][c]
                if key == "spirit":
                    assert (r["ksize"], r["lam"]) == want, (af, m, c, r["ksize"], r["lam"], want)
                else:
                    assert r["lam"] == want, (af, m, c, r["lam"], want)
        assert MASK(af).is_file(), MASK(af)
        print(af, "mask sha256", sha256(MASK(af)))
    for scan in SCANS:
        for n in ("ksp_T1.h5", "ksp_FLAIR.h5", "csm_espirit_T1.h5", "csm_espirit_FLAIR.h5"):
            assert (data_dir(scan) / n).is_file(), (scan, n)
    print("SELFTEST_OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
