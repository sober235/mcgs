#!/usr/bin/env python3
"""ts_common.py -- shared definitions of the T1 + FLAIR acceleration study on the six test scans.

Everything here is a constant or a tiny helper; the frozen settings are those registered on the tuning subject
(jiangyizhou, topics T1FLAIR_AF{12,16,20,24}_v3_jiangyizhou) and are checked against their summary.json files by
--selftest. Nothing is tuned on the test scans.

    gpu.py run --cpu -- @python autoresearch/t1flair_af_testset_20261121/ts_common.py --selftest
"""
import hashlib
import json
import sys
from pathlib import Path

OUT = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_t1flair_af_testset_20261121")
RES = Path("/data0/congcong/code/GS/Results_20260908")
SRC = Path("/data1/data1_congcong/data/MC_Jia/Train_SpiritDiffusion_CFL")   # read only

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
AFS = (12, 16, 20, 24)

# lambdas selected on the tuning subject (spec, section "frozen settings"); spirit = (kernel size, lambda)
FROZEN = {
    12: dict(cssense={"T1": 0.0007, "FLAIR": 0.0007}, jcssense={"T1": 0.001, "FLAIR": 0.001},
             spirit={"T1": (5, 0.002), "FLAIR": (5, 0.001)}),
    **{af: dict(cssense={"T1": 0.0007, "FLAIR": 0.0005}, jcssense={"T1": 0.001, "FLAIR": 0.0007},
                spirit={"T1": (5, 0.001), "FLAIR": (5, 0.001)}) for af in (16, 20, 24)},
}


def JYZ_TOPIC(af):
    return RES / f"T1FLAIR_AF{af}_v3_jiangyizhou"


def MASK(af):
    return JYZ_TOPIC(af) / f"data/mask_af{af}_3d.h5"


def data_dir(scan):
    return OUT / "data" / scan


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
    assert len(SCANS) == 6
    assert len({v["person"] for v in SCANS.values()}) == 4
    assert sum(v["group"] == "aligned" for v in SCANS.values()) == 3
    assert sum(v["group"] == "offset" for v in SCANS.values()) == 3
    assert {k for k, v in SCANS.items() if v["reuse"]} == {"vol1_huangyuwei", "vol4_tangbing"}
    assert set(FROZEN) == set(AFS)
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
    print("SELFTEST_OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
