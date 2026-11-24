#!/usr/bin/env python3
# PORT of autoresearch/t1flair_af_testset_20261121/make_topics.py (session mcgs-0f) for the different-mask study on
# the test set (user, 2026-11-22): the "af" of a row is the sampling pattern ("uniform" | "radial") at
# acceleration factor 16; registered runs, row counts and names changed accordingly, the logic is unchanged.
"""make_topics.py -- topic clones of the jiangyizhou comparator scripts for the six test scans.

One clone per (scan, AF): OUT/topics/<scan>_af<af>/{scripts,data,logs,recon}. scripts/ holds copies of the six
comparator scripts of JYZ_TOPIC(af)/scripts (that AF's own topic is the source), in which only two kinds of string
are replaced:
  1. the topic path JYZ_TOPIC(af) -> the clone path (the only functional change: data are read from and results
     written to the clone);
  2. three provenance remarks inside module docstrings ("Transcribed from Results_20260908/CE_pre_post_contrast/...")
     lose the "Results_20260908/" prefix, so that grep Results_20260908 over the clone gives 0 hits. Docstrings only.
Every clone is checked by reversing both replacements, which must give the source file byte for byte.
data/ is symlinks: ksp_{T1,FLAIR}.h5 and csm_espirit_{T1,FLAIR}.h5 -> data_dir(scan), mask_af<af>_3d.h5 -> MASK(af).
Control clone OUT/topics/jyz_af20: the AF20 scripts, data linked to the jiangyizhou AF20 topic (read only).

An existing clone is accepted only if identical (file bytes, link targets); anything different is an error.
Nothing is written outside OUT/topics.

  gpu.py run --cpu -- @python autoresearch/diffmask_testset_20261122/make_topics.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ts_common import AFS, JYZ_TOPIC, MASK, OUT, RES, SCANS, data_dir, sha256   # noqa: E402

SCRIPTS = ("t1t2_common.py", "recon_classical.py", "recon_joint_cssense.py", "recon_spirit.py",
           "recon_zsssdu.py", "recon_dip_inr.py")
DATA = ("ksp_T1.h5", "ksp_FLAIR.h5", "csm_espirit_T1.h5", "csm_espirit_FLAIR.h5")
DOC_OLD = "Results_20260908/CE_pre_post_contrast/"            # provenance remark in three docstrings
DOC_NEW = "CE_pre_post_contrast/"
DOC_FILES = {"recon_classical.py", "recon_zsssdu.py", "recon_dip_inr.py"}
CONTROL = ("jyz", "uniform")
log = lambda *a: print(*a, flush=True)


def topic_dir(scan, af, root=OUT):
    return Path(root) / "topics" / f"{scan}_{af}"


def clone_text(af, name, dst):
    """the source script of AF `af` with the two replacements; checked to be reversible to the source exactly"""
    src = (JYZ_TOPIC(af) / "scripts" / name).read_text()
    topic = str(JYZ_TOPIC(af))
    # t1t2_common.R and recon_classical.R are the only topic paths; the other scripts take R from t1t2_common
    # (recon_joint_cssense.py gets the topic from --topic)
    want = 1 if name in ("t1t2_common.py", "recon_classical.py") else 0
    assert src.count(topic) == want, (af, name, src.count(topic), want)
    assert str(dst) not in src and DOC_NEW not in src.replace(DOC_OLD, ""), (af, name)
    out = src.replace(topic, str(dst))
    if name in DOC_FILES:
        assert out.count(DOC_OLD) == 1, (af, name, out.count(DOC_OLD))
        i = out.index(DOC_OLD)
        assert out.count('"""', 0, i) == 1, (af, name, "provenance remark must sit inside the module docstring")
        out = out.replace(DOC_OLD, DOC_NEW)
    assert "Results_20260908" not in out, (af, name, [l for l in out.splitlines() if "Results_20260908" in l])
    back = out.replace(str(dst), topic)
    if name in DOC_FILES:
        back = back.replace(DOC_NEW, DOC_OLD, 1)
    assert back == src, (af, name, "clone is not the source up to the two replacements")
    return out


def put_file(p, text):
    if p.exists() or p.is_symlink():
        assert not p.is_symlink() and p.read_text() == text, f"{p} exists with different content"
        return "same"
    tmp = p.with_name(f".{p.name}.tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, p)
    return "new"


def put_link(p, target):
    target = Path(target)
    assert target.exists(), f"link target missing: {target}"
    if p.exists() or p.is_symlink():
        assert p.is_symlink() and os.readlink(p) == str(target), f"{p} exists and is not a link to {target}"
        return "same"
    p.symlink_to(target)
    return "new"


def make_topic(scan, af, root=OUT):
    """create or verify one clone; returns its manifest"""
    d = topic_dir(scan, af, root)
    for sub in ("scripts", "data", "logs", "recon"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    src_data = (JYZ_TOPIC(af) / "data") if scan == "jyz" else data_dir(scan)
    man = dict(topic=str(d), scan=scan, af=af, source_scripts=str(JYZ_TOPIC(af) / "scripts"), scripts={}, data={})
    for name in SCRIPTS:
        st = put_file(d / "scripts" / name, clone_text(af, name, d))
        man["scripts"][name] = dict(state=st, sha256=sha256(d / "scripts" / name),
                                    source_sha256=sha256(JYZ_TOPIC(af) / "scripts" / name))
    links = {n: src_data / n for n in DATA}
    links[MASK(af).name] = MASK(af)
    for n, tgt in links.items():
        st = put_link(d / "data" / n, tgt)
        man["data"][n] = dict(state=st, target=str(tgt), resolved=str((d / "data" / n).resolve()))
    check_topic(scan, af, root)
    return man


def check_topic(scan, af, root=OUT):
    """grep Results_20260908 over the cloned scripts = 0 hits; every link resolves; mask sha256 = MASK(af)"""
    d = topic_dir(scan, af, root)
    hits = [(f.name, i) for f in sorted((d / "scripts").glob("*.py"))
            for i, line in enumerate(f.read_text().splitlines(), 1) if "Results_20260908" in line]
    assert not hits, (d, hits)
    assert sorted(f.name for f in (d / "scripts").glob("*.py")) == sorted(SCRIPTS), d
    for n in (*DATA, MASK(af).name):
        p = d / "data" / n
        assert p.is_symlink() and p.resolve().is_file(), (p, "link does not resolve")
    assert sha256(d / "data" / MASK(af).name) == sha256(MASK(af)), (d, "mask sha256")
    if scan != "jyz":
        for n in DATA:                       # a formal clone must never read jiangyizhou data
            assert (d / "data" / n).resolve() == (data_dir(scan) / n).resolve(), (d, n)
            assert not str(os.readlink(d / "data" / n)).startswith(str(RES)), (d, n)
    return True


def main():
    jobs = [(s, af) for s in SCANS for af in AFS] + [CONTROL]
    art = Path(os.environ.get("MCGS_ARTIFACT_DIR", OUT / "artifacts"))
    art.mkdir(parents=True, exist_ok=True)
    mans = []
    for scan, af in jobs:
        for n in DATA:
            src = (JYZ_TOPIC(af) / "data" if scan == "jyz" else data_dir(scan)) / n
            assert src.is_file(), f"missing input {src}"
        m = make_topic(scan, af)
        new = sum(v["state"] == "new" for g in ("scripts", "data") for v in m[g].values())
        log(f"topic {scan}_{af}: {len(SCRIPTS)} scripts, {len(m['data'])} links, {new} new, checks OK")
        mans.append(m)
    (art / "topics_manifest.json").write_text(json.dumps(mans, indent=1))
    log(f"TOPICS_OK {len(mans)}")


if __name__ == "__main__":
    main()
