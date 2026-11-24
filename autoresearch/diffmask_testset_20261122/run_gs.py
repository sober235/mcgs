#!/usr/bin/env python3
# PORT of autoresearch/t1flair_af_testset_20261121/run_gs.py (session mcgs-0f) for the different-mask study on
# the test set (user, 2026-11-22): the "af" of a row is the sampling pattern ("uniform" | "radial") at
# acceleration factor 16; registered runs, row counts and names changed accordingly, the logic is unchanged.
"""run_gs.py -- MC-GS (joint) and GS-single runner of the T1 + FLAIR acceleration study on the six test scans.

Row "<scan>:<af>:<kind>", kind in joint | singleT1 | singleFLAIR, experiment name ts_<scan>_af<af>_<kind>.
Every row replays what the jiangyizhou registered run of that AF did: the argv of the registered run's
<exp>_score.json (without --exp / --gpu / --base_cfg / --save_volumes) and its --base_cfg yaml, where only
data.*.data_path and csm_path are replaced by the prepared files of the scan; mask_path stays MASK(af).
GS-single is the registered version: the same contrast goes into both channels of the joint trainer (the registered
singleT1 / singleFLAIR yamls), not a separate single-contrast trainer.

Control row "jyz:<af>:<kind>": the registered yaml unchanged (jiangyizhou data), outputs under OUT. It is the only
row that reads jiangyizhou data and is not one of the 72 formal rows.

Trainer copy: server worktree mcgs_sg2x2_wt at HEAD 54f8a2a, copied into <root>/trainer and checked file by file.
Nothing is written outside OUT (or OUT/smoke with --smoke); a claimed row is skipped; nothing is overwritten.

  gpu.py run --cpu   -- @python .../run_gs.py --selftest
  gpu.py run --cpu   -- @python .../run_gs.py --status
  gpu.py run --gpu N -- @python .../run_gs.py [--smoke] --rows vol1_huangyuwei:20:joint ...
  gpu.py run --cpu   -- @python .../run_gs.py --release ROW
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ts_common import AFS, MASK, OUT, SCANS, claim, data_dir, sha256   # noqa: E402

GS = Path("/data0/congcong/code/GS")
SRC = GS / "mcgs_sg2x2_wt"
SRC_HEAD = "54f8a2a3bef402e785314b748f730a9623ca9d43"
CODE_DIRS = ("arc", "paired_gs", "diag_artefact")
KINDS = ("joint", "singleT1", "singleFLAIR")
KEYS = ("head_psnr", "head_ssim", "head_hfen", "nmse")

# registered jiangyizhou runs (worktree, experiment names of joint / singleT1 / singleFLAIR); read only
_DM_WT = GS / "mcgs_diffmask_wt"      # the frozen AF16 phase-3 recipe, as registered on each pattern (jiangyizhou)
REGISTERED = {
    "uniform": (_DM_WT, "af16u_p3_joint_full4k", "af16u_p3_singleT1_full4k", "af16u_p3_singleFLAIR_full4k"),
    "radial": (_DM_WT, "af16r_p3_joint_full4k", "af16r_p3_singleT1_full4k", "af16r_p3_singleFLAIR_full4k"),
}
FORMAL_ROWS = [f"{s}:{af}:{k}" for s in SCANS for af in AFS for k in KINDS]
CONTROL_ROWS = [f"jyz:{af}:{k}" for af in AFS for k in KINDS]
log = lambda *a: print(*a, flush=True)


def parse_row(row):
    scan, af, kind = row.split(":")
    assert scan in SCANS or scan == "jyz", row
    assert af in AFS and kind in KINDS, row
    return scan, af, kind


def exp_name(row):
    scan, af, kind = parse_row(row)
    return f"dm_{scan}_{af}_{kind}"


def registered_score(af, kind):
    wt, *names = REGISTERED[af]
    return wt / "results_arc" / f"{names[KINDS.index(kind)]}_score.json"


def registered_recipe(af, kind):
    """(recipe argv, registered yaml path) of a registered run: --exp / --gpu / --base_cfg / --save_volumes removed"""
    argv = json.load(open(registered_score(af, kind)))["argv"]
    recipe, cfg, i = [], None, 0
    while i < len(argv):
        if argv[i] in ("--exp", "--gpu"):
            i += 2
        elif argv[i] == "--base_cfg":
            cfg, i = REGISTERED[af][0] / argv[i + 1], i + 2
        elif argv[i] == "--save_volumes":
            i += 1
        else:
            recipe.append(argv[i])
            i += 1
    assert cfg is not None and cfg.is_file(), (af, kind, cfg)
    return recipe, cfg


def scan_cfg(af, kind, scan):
    """the registered yaml of (af, kind) with data_path / csm_path of every channel pointing at the scan's prepared files"""
    import yaml
    _, path = registered_recipe(af, kind)
    cfg = yaml.safe_load(open(path))
    for ch in cfg["data"].values():
        name = ch["name"]
        assert ch["mask_path"] == str(MASK(af)), (path, ch["mask_path"])
        ch["data_path"] = str(data_dir(scan) / f"ksp_{name}.h5")
        ch["csm_path"] = str(data_dir(scan) / f"csm_espirit_{name}.h5")
    return cfg


def diff_paths(a, b, prefix=()):
    """paths (tuples of keys) at which two nested dicts differ"""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            out += [(*prefix, k)] if k not in a or k not in b else diff_paths(a[k], b[k], (*prefix, k))
        return out
    return [] if a == b else [prefix]


def write_cfg(root, row):
    import yaml
    scan, af, kind = parse_row(row)
    p = root / "configs" / f"{exp_name(row)}.yaml"
    text = yaml.safe_dump(scan_cfg(af, kind, scan), sort_keys=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        assert p.read_text() == text, f"{p} exists with different content"
    else:
        tmp = p.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(text)
        os.replace(tmp, p)
    return p


def row_cmd(root, row, gpu, smoke):
    """(experiment name, trainer command) of a row; --gpu takes the physical id, as the trainer sets the device itself"""
    scan, af, kind = parse_row(row)
    recipe, reg_cfg = registered_recipe(af, kind)
    if smoke:
        recipe[recipe.index("--iters") + 1] = "30"
        recipe[recipe.index("--warmup") + 1] = "10"
    cfg = reg_cfg if scan == "jyz" else write_cfg(root, row)
    exp = exp_name(row)
    tail = [] if smoke else ["--save_volumes"]
    return exp, [sys.executable, "-u", "arc/cycle.py", "--exp", exp, "--gpu", gpu, *recipe, "--base_cfg", str(cfg), *tail]


def row_paths(root, row):
    exp = exp_name(row)
    res = root / "trainer/results_arc"
    return dict(claim=root / "claims" / exp, score=res / f"{exp}_score.json", log=res / f"{exp}.log",
                outs=[res / f"{exp}.log", res / f"{exp}_score.json", res / exp, res / f"{exp}_volumes.npz"])


def row_state(root, row):
    """done = score exists; stale = claimed without a score (a failed job, or one still running); open = unclaimed"""
    p = row_paths(root, row)
    return "done" if p["score"].exists() else "stale" if p["claim"].exists() else "open"


def status(root, rows):
    for row in rows:
        log(f"{row_state(root, row):6s} {row}")
    states = [row_state(root, r) for r in rows]
    log("summary: " + ", ".join(f"{s} {states.count(s)}" for s in ("done", "stale", "open")))


def release(root, row):
    p = row_paths(root, row)
    assert not p["score"].exists(), f"{row} has a score; refusing to release"
    assert p["claim"].is_dir(), f"{row} has no claim"
    p["claim"].rmdir()
    log(f"released {row} (partial outputs, if any, stay and must be removed by hand before a rerun)")


def ensure_trainer(root):
    git = lambda *a: subprocess.run(["git", "-C", str(SRC), *a], check=True, capture_output=True, text=True).stdout
    assert git("rev-parse", "HEAD").strip() == SRC_HEAD, "source worktree moved"
    assert git("status", "--porcelain", "--", *CODE_DIRS).strip() == "", "source worktree has local changes"
    tr = root / "trainer"
    if not tr.exists():
        root.mkdir(parents=True, exist_ok=True)
        tmp = root / f"trainer.tmp{os.getpid()}"
        for d in CODE_DIRS:
            shutil.copytree(SRC / d, tmp / d, ignore=shutil.ignore_patterns("__pycache__", "results_arc"))
        try:
            tmp.rename(tr)
        except OSError:                      # another worker won the race
            shutil.rmtree(tmp)
    tracked = [f for f in git("ls-files", "--", *CODE_DIRS).splitlines() if f]
    bad = [f for f in tracked if not (tr / f).is_file() or sha256(tr / f) != sha256(SRC / f)]
    assert not bad, f"trainer copy differs from the source worktree: {bad[:10]}"
    (tr / "results_arc").mkdir(exist_ok=True)
    return tr, len(tracked)


def selftest():
    import yaml
    assert len(FORMAL_ROWS) == 12 * 3 == 36 and len(set(FORMAL_ROWS)) == 36
    for af in AFS:
        for kind in KINDS:
            recipe, reg = registered_recipe(af, kind)
            assert recipe[recipe.index("--iters") + 1] == "4000" and "--trainer" not in recipe, (af, kind)
            reg_cfg = yaml.safe_load(open(reg))
            for scan in SCANS:
                got = scan_cfg(af, kind, scan)
                assert list(got["data"]) == list(reg_cfg["data"])
                diffs = diff_paths(reg_cfg, got)
                assert diffs and all(len(d) == 3 and d[0] == "data" and d[2] in ("data_path", "csm_path")
                                     for d in diffs), (af, kind, scan, diffs)
                assert len(diffs) == 2 * len(reg_cfg["data"]), (af, kind, scan, diffs)
                for ch in got["data"].values():
                    assert Path(ch["data_path"]).parent == data_dir(scan) and ch["mask_path"] == str(MASK(af))
            # single rows feed the same contrast into both channels, as registered
            if kind != "joint":
                assert {c["name"] for c in reg_cfg["data"].values()} == {kind[len("single"):]}
    tmp = Path(tempfile.mkdtemp(dir=OUT, prefix="selftest_"))
    try:
        row = FORMAL_ROWS[0]
        assert row_state(tmp, row) == "open"
        assert claim(tmp / "claims", exp_name(row)) is True
        assert claim(tmp / "claims", exp_name(row)) is False
        assert row_state(tmp, row) == "stale"
        release(tmp, row)
        assert row_state(tmp, row) == "open"
        claim(tmp / "claims", exp_name(row))
        score = row_paths(tmp, row)["score"]
        score.parent.mkdir(parents=True)
        score.write_text("{}")
        assert row_state(tmp, row) == "done"
        try:
            release(tmp, row)
            raise SystemExit("release of a scored row must be refused")
        except AssertionError:
            pass
    finally:
        shutil.rmtree(tmp)
    print("SELFTEST_OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", choices=FORMAL_ROWS + CONTROL_ROWS)
    ap.add_argument("--smoke", action="store_true", help="30 iterations, no volumes, into OUT/smoke")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--release", metavar="ROW", choices=FORMAL_ROWS + CONTROL_ROWS)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    root = OUT / "smoke" if args.smoke else OUT
    if args.selftest:
        return selftest()
    if args.status:
        return status(root, args.rows or FORMAL_ROWS)
    if args.release:
        return release(root, args.release)
    assert args.rows, "--rows is required"
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert vis.isdigit(), f"needs exactly one GPU from gpu.py --gpu, got CUDA_VISIBLE_DEVICES={vis!r}"
    tr, n = ensure_trainer(root)
    (root / "claims").mkdir(exist_ok=True)
    art = Path(os.environ.get("MCGS_ARTIFACT_DIR", root / "artifacts"))
    art.mkdir(parents=True, exist_ok=True)
    log(f"root {root}; trainer copy verified ({n} tracked files, source {SRC_HEAD[:7]}); physical GPU {vis}; rows {args.rows}")
    record = dict(root=str(root), source=str(SRC), source_head=SRC_HEAD, gpu=vis, smoke=args.smoke, runs=[])
    failed = []
    for row in args.rows:
        scan, af, kind = parse_row(row)
        need = [Path(registered_recipe(af, kind)[1])] if scan == "jyz" else \
            [data_dir(scan) / f"{k}_{c}.h5" for k in ("ksp", "csm_espirit") for c in ("T1", "FLAIR")]
        assert all(p.is_file() for p in need), f"{row}: missing input {[str(p) for p in need if not p.is_file()]}"
        exp, cmd = row_cmd(root, row, vis, args.smoke)
        p = row_paths(root, row)
        if not claim(root / "claims", exp):
            log(f"-- {exp}: already claimed, skipped")
            continue
        assert not any(o.exists() for o in p["outs"]), f"refusing to overwrite partial outputs of {exp}; remove by hand"
        env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID")
        log(f"\n===== {exp}\n{' '.join(cmd)}")             # no server clock or durations in logs (time rule)
        with open(p["log"], "w") as lf:
            rc = subprocess.run(cmd, cwd=tr, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        run = dict(row=row, exp=exp, cmd=cmd, rc=rc)
        if rc == 0 and p["score"].exists():
            s = json.load(open(p["score"]))
            run.update(npts=s["npts"], plain_name=s.get("plain_name"), plain={k: s["plain"][k] for k in KEYS},
                       contrast=s["contrast"] and {k: s["contrast"][k] for k in KEYS})
            shutil.copy2(p["score"], art / p["score"].name)
            c = f" / {s['contrast']['head_psnr']:.3f}" if s["contrast"] else ""
            log(f"===== {exp} done: head PSNR {s['plain']['head_psnr']:.3f}{c}, N={s['npts']}")
        else:
            failed.append(exp)
            log(f"===== {exp} FAILED rc={rc}; see {p['log']} (not echoed: trainer logs may carry clocks)")
        record["runs"].append(run)
        (art / f"record_gpu{vis}{'_smoke' if args.smoke else ''}_{args.rows[0].replace(':', '_')}.json").write_text(
            json.dumps(record, indent=2))
    log(f"DONE; failed rows: {failed or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
