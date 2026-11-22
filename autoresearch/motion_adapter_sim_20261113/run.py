#!/usr/bin/env python3
"""Launcher of the motion study (runs on amax, inside a gpu.py job).

    run.py --af af12 --cond E2 --arm A2 --gpu 3 [--seed 42] [--settings settings_frozen.json] [--smoke] [--tag r1]
    run.py --dry --all            every row of common.MATRIX: experiment name, command, data files present
    run.py --selftest --gpu 3     30-iteration checks of the patched trainer on real data

One training per call. The frozen trainer is copied once to ROOT/trainer and every tracked file is compared with the
source worktree before each run. The child's output is passed through with every program-measured duration removed,
and the ledger row carries no clock value (start and finish are recorded on the Mac, local wall time).
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from common import (ARMS, BASE_CFG, CFG, CODE_DIRS, CONDITIONS, CROSS_OFF, DATA, HYW, JOINT_ONLY, K_SEG, LEDGER,
                    MASKS, MATRIX, NATIVE_SHIFT_MM, RES, ROOT, SRC, SRC_HEAD, TR, V32, V32G, exp_name)

HERE = Path(__file__).resolve().parent
# starting point of Task 8; replaced by settings_frozen.json once the adapter settings are frozen on D1-D3
DEFAULT_SETTINGS = dict(lrs=dict(rigid_theta=1e-3, rigid_t=2e-2, nonrigid=5e-3, seg_theta=5e-4, seg_t=1e-2),
                        reg=dict(ctrl_l2=1e-3, ctrl_smooth=1e-2, seg_l2=1e-3, seg_smooth=1e-2),
                        stage_iters=[500, 1500], n_extra=2, K=K_SEG)
METRICS = ("head_psnr", "head_ssim", "head_hfen", "nmse", "psnr", "ssim", "hfen", "e_unobs")
log = lambda *a: print(*a, flush=True)


def strip_durations(line):
    """Remove program-measured durations from a line of trainer output (time rule)."""
    line = re.sub(r"\s+elapsed_s=\S+", "", line)
    line = re.sub(r"(done) in [0-9.]+ ?s\b", r"\1", line)
    line = re.sub(r"\s+bm4d=[0-9.]+s\b", "", line)
    return line


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _git(*a):
    return subprocess.run(["git", "-C", str(SRC), *a], check=True, capture_output=True, text=True).stdout


def ensure_trainer(tr=None, root=None, res=None):
    """Same rule as abl_framework.ensure_trainer: the source worktree must be clean at SRC_HEAD for the code
    directories; copy them once; every git-tracked file of the copy must equal the source."""
    TR_, ROOT_, RES_ = tr or TR, root or ROOT, res or RES     # another study passes its own private copy
    assert _git("rev-parse", "HEAD").strip() == SRC_HEAD, "source worktree moved"
    assert _git("status", "--porcelain", "--", *CODE_DIRS).strip() == "", "source worktree has local changes"
    if not TR_.exists():
        ROOT_.mkdir(parents=True, exist_ok=True)
        tmp = ROOT_ / f"trainer.tmp{os.getpid()}"
        for d in CODE_DIRS:
            shutil.copytree(SRC / d, tmp / d, ignore=shutil.ignore_patterns("__pycache__", "results_arc"))
        try:
            tmp.rename(TR_)
        except OSError:
            shutil.rmtree(tmp)
    tracked = [f for f in _git("ls-files", "--", *CODE_DIRS).splitlines() if f]
    bad = [f for f in tracked if not (TR_ / f).is_file() or sha(TR_ / f) != sha(SRC / f)]
    assert not bad, f"trainer copy differs from the source worktree: {bad[:10]}"
    RES_.mkdir(exist_ok=True)
    return len(tracked)


# ------------------------------------------------------------------------------------------- configuration
def is_single(arm):
    return ARMS[arm]["trainer"] == "single"


def via_patch(cond, arm):
    a = ARMS[arm]
    return bool(a["rigid"] or a["nonrigid"] or a["segments"] or a["oracle"] or CONDITIONS[cond]["intra"])


def _gt_ksp(cond):
    return json.loads((DATA / cond / "motion_truth.json").read_text())["gt_ksp_path"]


def _ref_seg(cond):
    if CONDITIONS[cond]["intra"]:
        return int(json.loads((DATA / cond / "motion_truth.json").read_text())["ref_seg"])
    return int(json.loads((DATA / "C0/schedule.json").read_text())["ref_seg"])


def data_config(af, cond, arm, gt_ksp=None):
    """The hyw config with this condition's files and this AF's mask; single-contrast arms keep one data block."""
    import yaml
    cfg = yaml.safe_load(open(BASE_CFG))
    spec = CONDITIONS[cond]
    d = cfg["data"]
    for slot in ("plain", "contrast"):
        d[slot]["mask_path"] = str(MASKS[af])
        d[slot]["acceleration_factor"] = int(af[2:])
    if cond != "C0":
        moved = str(DATA / cond / "ksp_T2.h5")
        if spec["intra"]:
            d["contrast"]["data_path"] = (gt_ksp or {}).get(cond) or _gt_ksp(cond)
            d["contrast"]["motion_ksp_path"] = moved
        else:
            d["contrast"]["data_path"] = moved
        if spec["lesions"]:
            d["plain"]["data_path"] = str(DATA / cond / "ksp_T1.h5")
    if is_single(arm):
        cfg["data"] = {"plain": d[ARMS[arm]["slot"]]}
    return cfg


def config_path(af, cond, arm):
    if is_single(arm):
        return CFG / f"{af}_{cond}_single_{'T1' if ARMS[arm]['slot'] == 'plain' else 'T2'}.yaml"
    return CFG / f"{af}_{cond}_joint.yaml"


def write_config(af, cond, arm):
    import yaml
    p = config_path(af, cond, arm)
    text = yaml.safe_dump(data_config(af, cond, arm), sort_keys=False)
    CFG.mkdir(parents=True, exist_ok=True)
    if p.exists():
        assert p.read_text() == text, f"{p} exists with different content"
    else:
        tmp = p.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(text)
        os.replace(tmp, p)
    return p


def run_name(af, cond, arm, seed, smoke=False, tag=""):
    """tag: a development round (Task 8); its runs keep their own names and never collide with the matrix."""
    return ("smoke_" if smoke else "") + exp_name(af, cond, arm, seed) + (f"_{tag}" if tag else "")


def motion_config(af, cond, arm, seed, settings, ref_seg=None, smoke=False, tag=""):
    a, c = ARMS[arm], CONDITIONS[cond]
    if a["oracle"]:
        rigid, nonrigid, segments = True, bool(c["nonrigid"]), bool(c["intra"])
    else:
        rigid, nonrigid, segments = a["rigid"], a["nonrigid"], a["segments"]
    cdir = DATA / cond
    sched = (cdir if c["intra"] else DATA / "C0") / "schedule.npz"
    brain = cdir / "brain.npy" if (c["inter"] or c["nonrigid"] or c["lesions"]) else DATA / "C0/brain.npy"
    return dict(arm=arm, cond=cond, af=af, channel="plain" if is_single(arm) else "contrast",
                rigid=bool(rigid), nonrigid=bool(nonrigid), segments=int(settings["K"]) if segments else 0,
                ref_seg=(ref_seg if ref_seg is not None else _ref_seg(cond)) if segments else None,
                n_extra=int(settings["n_extra"]), schedule_path=str(sched) if segments else None,
                truth_path=str(cdir / "motion_truth.json") if cond != "C0" else None,
                disp_path=str(cdir / "disp_T2.npy") if (c["inter"] or c["nonrigid"]) else None,
                brain_path=str(brain), native_shift_mm=list(NATIVE_SHIFT_MM), oracle=bool(a["oracle"]),
                lrs=settings["lrs"], reg=settings["reg"], stage_iters=list(settings["stage_iters"]),
                adam_eps=settings.get("adam_eps"),
                out_dir=str(RES), exp=run_name(af, cond, arm, seed, smoke, tag), seed=int(seed))


def motion_path(exp):
    return CFG / "motion" / f"{exp}.json"


def command(af, cond, arm, gpu, seed, smoke, tag=""):
    """(experiment name, command). Pure: only names paths, reads nothing."""
    exp = run_name(af, cond, arm, seed, smoke, tag)
    recipe = list(V32 if is_single(arm) else V32G)
    if smoke:
        recipe[recipe.index("--iters") + 1] = "30"
        recipe[recipe.index("--warmup") + 1] = "10"
    tail = [] if smoke else ["--save_volumes"]
    if seed != 42:
        tail += ["--set", f"training.seed={seed}"]
    if is_single(arm):
        slot = ARMS[arm]["slot"]
        key = "loss.lambda_wav_plain=" if slot == "plain" else "loss.lambda_wav="
        w = [x.split("=", 1)[1] for x in recipe if x.startswith(key)][-1]        # this contrast's own weight
        drop = {i for i, x in enumerate(recipe) if x.startswith(JOINT_ONLY)}
        recipe = [x for i, x in enumerate(recipe) if i not in drop and i + 1 not in drop]
        tail += ["--trainer", "single", *CROSS_OFF, "--set", f"loss.lambda_wav={w}", "--set", f"loss.lambda_wav_plain={w}"]
    cyc = ["--exp", exp, "--gpu", str(gpu), *recipe, "--base_cfg", str(config_path(af, cond, arm)), *tail]
    if via_patch(cond, arm):
        return exp, [sys.executable, "-u", str(HERE / "patch.py"), "--motion", str(motion_path(exp)), "--", *cyc]
    return exp, [sys.executable, "-u", "arc/cycle.py", *cyc]


def ledger_row(score, meta):
    """Ledger line from a trainer score file: metrics only, no clock value, no duration."""
    def half(s):
        return None if s is None else {k: s[k] for k in METRICS if k in s}
    return dict(exp=meta["exp"], af=meta["af"], cond=meta["cond"], arm=meta["arm"], seed=meta["seed"],
                tag=meta.get("tag", ""),
                config_sha256=meta["config_sha256"], source_digest=meta["source_digest"],
                plain=half(score.get("plain")), contrast=half(score.get("contrast")),
                plain_name=score.get("plain_name"), contrast_name=score.get("contrast_name"),
                npts=score.get("npts"), vram_gb=score.get("vram_gb"), motion_file=meta["motion_file"],
                argv=meta.get("argv"), settings=meta.get("settings"))


# ------------------------------------------------------------------------------------------- execution
def ensure_c0_side_files():
    """C0 has no simulated files; arms with segments on a condition without intra-scan motion use the schedule of
    E4a (same shutter, so the same schedule), and the evaluation uses E1a's brain mask (computed from the T1)."""
    c0 = DATA / "C0"
    if (c0 / "schedule.json").exists():
        return
    c0.mkdir(parents=True, exist_ok=True)
    truth = json.loads((DATA / "E4a/motion_truth.json").read_text())
    shutil.copy(DATA / "E4a/schedule.npz", c0 / "schedule.npz")
    shutil.copy(DATA / "E1a/brain.npy", c0 / "brain.npy")
    tmp = c0 / f"schedule.tmp{os.getpid()}"
    tmp.write_text(json.dumps(dict(ref_seg=truth["ref_seg"], ref_train=truth["ref_train"], n_trains=truth["n_trains"],
                                   copied_from="E4a/schedule.npz, E1a/brain.npy")))
    os.replace(tmp, c0 / "schedule.json")


def _source_digest():
    p = Path(".mcgs-manifest.json")
    return json.loads(p.read_text()).get("source_digest") if p.exists() else None


def run_one(af, cond, arm, gpu, seed, smoke, settings, tag=""):
    ensure_c0_side_files()
    digest = _source_digest()                                # cwd is still the job's code directory here
    exp, cmd = command(af, cond, arm, gpu, seed, smoke, tag)
    score_file = RES / f"{exp}_score.json"
    if score_file.exists():
        assert smoke, f"{score_file} exists; finished experiments are not overwritten"
        score_file.unlink()
    cfg = write_config(af, cond, arm)
    mfile = None
    if via_patch(cond, arm):
        mfile = motion_path(exp)
        mfile.parent.mkdir(parents=True, exist_ok=True)
        mfile.write_text(json.dumps(motion_config(af, cond, arm, seed, settings, smoke=smoke, tag=tag), indent=1))
    log(f"===== {exp}\n" + " ".join(cmd))
    p = subprocess.Popen(cmd, cwd=str(TR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        sys.stdout.write(strip_durations(line))
        sys.stdout.flush()
    rc = p.wait()
    assert rc == 0 and score_file.exists(), f"{exp} failed (exit {rc})"
    row = ledger_row(json.loads(score_file.read_text()),
                     dict(exp=exp, af=af, cond=cond, arm=arm, seed=seed, tag=tag, config_sha256=sha(cfg), source_digest=digest,
                          settings=settings if via_patch(cond, arm) else None,
                          argv=cmd[2:],
                          motion_file=str(RES / f"{exp}_motion.json") if mfile else None))
    if not smoke:
        with open(LEDGER, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(row) + "\n")
    art = os.environ.get("MCGS_ARTIFACT_DIR")
    if art:
        (Path(art) / f"{exp}_row.json").write_text(json.dumps(row, indent=1))
        if mfile and (RES / f"{exp}_motion.json").exists():
            shutil.copy(RES / f"{exp}_motion.json", Path(art) / f"{exp}_motion.json")
    for tag, s in (("plain", row["plain"]), ("contrast", row["contrast"])):
        if s:
            log(f"RESULT {exp} {tag}: head_psnr={s['head_psnr']:.3f} head_ssim={s['head_ssim']:.4f} nmse={s['nmse']:.5f}")
    return row


def dry_all():
    ensure_c0_side_files()
    bad = 0
    for part in ("af12_core", "af12_rest", "af16", "dev"):
        af = "af16" if part == "af16" else "af12"
        for cond, arm in MATRIX[part]:
            exp, cmd = command(af, cond, arm, "0", 42, False)
            need = [MASKS[af], HYW / "ksp_T1.h5", HYW / "ksp_T2.h5"]
            if cond != "C0":
                need += [DATA / cond / "ksp_T2.h5", DATA / cond / "manifest.json", DATA / cond / "motion_truth.json"]
                if CONDITIONS[cond]["lesions"]:
                    need.append(DATA / cond / "ksp_T1.h5")
            missing = [str(p) for p in need if not Path(p).exists()]
            ok = not missing
            if ok and via_patch(cond, arm):
                import patch
                patch.validate(motion_config(af, cond, arm, 42, DEFAULT_SETTINGS))
            bad += not ok
            log(f"{'ok ' if ok else 'MISSING'} {part:10s} {exp:28s} {'patch' if via_patch(cond, arm) else 'cycle'} {missing or ''}")
    log(f"dry run: {bad} rows with missing files")
    return bad


def selftest(gpu):
    """Short runs of the patched trainer on real data. 1) plain A0; 2) A2 with all learning rates 0 (the adapter
    stays the identity but goes through the transformed-render path): the scores must agree; 3) A23 on E4c
    (segments, non-rigid, reference-pose data set); 4) the oracle on E2."""
    zero = dict(DEFAULT_SETTINGS, lrs={k: 0.0 for k in DEFAULT_SETTINGS["lrs"]},
                reg={k: 0.0 for k in DEFAULT_SETTINGS["reg"]})
    a = run_one("af12", "C0", "A0", gpu, 42, True, DEFAULT_SETTINGS)
    b = run_one("af12", "C0", "A2", gpu, 42, True, zero)
    worst = max(abs(a[t][k] - b[t][k]) for t in ("plain", "contrast") for k in ("psnr", "head_psnr"))
    log(f"selftest: A0 vs identity-A2 max |dPSNR| = {worst:.4f} dB (limit 0.05)")
    assert worst < 0.05, "an identity adapter changes the training"
    c = run_one("af12", "E4c", "A23", gpu, 42, True, DEFAULT_SETTINGS)
    log(f"selftest: A23 on E4c ran, peak memory {c['vram_gb']:.1f} GB with n_extra={DEFAULT_SETTINGS['n_extra']}")
    run_one("af12", "E2", "O", gpu, 42, True, DEFAULT_SETTINGS)
    log("selftest: all four runs finished")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--af", choices=sorted(MASKS))
    ap.add_argument("--cond", choices=sorted(CONDITIONS))
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--gpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--settings", help="JSON file with lrs / reg / stage_iters / n_extra / K [/ adam_eps] (relative to this folder)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="", help="development round; appended to the experiment name")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.dry:
        assert args.all, "--dry needs --all"
        sys.exit(1 if dry_all() else 0)
    assert args.gpu is not None, "--gpu is required"
    n = ensure_trainer()
    log(f"trainer copy verified against {SRC_HEAD[:10]}: {n} files")
    if args.selftest:
        selftest(args.gpu)
        return
    settings = DEFAULT_SETTINGS
    if args.settings:
        settings = json.loads((HERE / args.settings).read_text())
    run_one(args.af, args.cond, args.arm, args.gpu, args.seed, args.smoke, settings, args.tag)


if __name__ == "__main__":
    main()
