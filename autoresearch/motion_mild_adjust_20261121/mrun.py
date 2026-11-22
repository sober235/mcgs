#!/usr/bin/env python3
"""Launcher of the second version (runs on amax, inside a gpu.py job).

    mrun.py --af af12 --order il --cond DTa --arm Ow --gpu 3 [--seed 42] [--settings settings_dev.json] [--smoke] [--tag d1]
    mrun.py --dry --all           every row of mcommon.MATRIX2: experiment name and the data files it needs
    mrun.py --selftest --gpu 3    30-iteration checks of the patched trainer on real data

One training per call; every arm runs through mpatch.py (B0 with the recording-only monitor). The frozen trainer is
copied once to ROOT2/trainer and compared file by file with the source worktree before each run. The child's output
is passed through with every program-measured duration removed, and the ledger row carries no clock value (start
and finish are recorded on the Mac, local wall time).
"""
import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from mcommon import ARMS2, CFG2, CONDITIONS2, DATA2, FORMAL, LEDGER2, MATRIX2, ORDERS, RES2, ROOT2, TR2, c1, exp_name

import run as run1  # noqa: E402  (first version's launcher: trainer-copy check, duration filter, ledger row)

HERE = Path(__file__).resolve().parent
log = run1.log
SMOKE_M2 = dict(warmup=10, every=10)


def is_single(arm):
    return ARMS2[arm]["trainer"] == "single"


def cond_dir(order, cond):
    return DATA2 / order / cond


def _gt_ksp(order, cond):
    return json.loads((cond_dir(order, cond) / "motion_truth.json").read_text())["gt_ksp_path"]


def data_config(af, order, cond, arm, gt_ksp=None):
    """The hyw config with this condition's files and this AF's mask; single-contrast arms keep one data block."""
    import yaml
    cfg = yaml.safe_load(open(c1.BASE_CFG))
    spec = CONDITIONS2[cond]
    d = cfg["data"]
    for slot in ("plain", "contrast"):
        d[slot]["mask_path"] = str(c1.MASKS[af])
        d[slot]["acceleration_factor"] = int(af[2:])
    if spec["v1_data"]:
        d["contrast"]["data_path"] = str(c1.DATA / spec["v1_data"] / "ksp_T2.h5")
    elif spec["intra"]:
        d["contrast"]["data_path"] = gt_ksp or _gt_ksp(order, cond)        # reference pose: ground truth and scale
        d["contrast"]["motion_ksp_path"] = str(cond_dir(order, cond) / "ksp_T2.h5")
    if spec["lesions"]:
        d["plain"]["data_path"] = str(cond_dir(order, cond) / "ksp_T1.h5")
    if is_single(arm):
        cfg["data"] = {"plain": d[ARMS2[arm]["slot"]]}
    return cfg


def config_path(af, order, cond, arm):
    kind = f"single_{'T1' if ARMS2[arm]['slot'] == 'plain' else 'T2'}" if is_single(arm) else "joint"
    return CFG2 / f"{af}_{order}_{cond}_{kind}.yaml"


def write_config(af, order, cond, arm):
    import yaml
    p = config_path(af, order, cond, arm)
    text = yaml.safe_dump(data_config(af, order, cond, arm), sort_keys=False)
    CFG2.mkdir(parents=True, exist_ok=True)
    if p.exists():
        assert p.read_text() == text, f"{p} exists with different content"
    else:
        tmp = p.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(text)
        os.replace(tmp, p)
    return p


def run_name(af, order, cond, arm, seed, smoke=False, tag=""):
    return ("smoke_" if smoke else "") + exp_name(af, order, cond, arm, seed, tag)


def motion_config(af, order, cond, arm, seed, settings, smoke=False, tag=""):
    a = ARMS2[arm]
    assert not a["m1"] and not a["m3"], f"{arm}: M1 / M3 are not implemented yet"
    cdir = cond_dir(order, cond)
    m2 = dict(settings["m2"], mode=a["m2"], **(SMOKE_M2 if smoke else {}))
    return dict(exp=run_name(af, order, cond, arm, seed, smoke, tag), arm=arm, cond=cond, order=order, af=af,
                seed=int(seed), channel="plain" if is_single(arm) else "contrast", out_dir=str(RES2),
                schedule_path=str(cdir / "schedule.npz"), truth_path=str(cdir / "motion_truth.json"),
                brain_path=str(cdir / "brain.npy"), m1=None, m2=m2, m3=None, reg=settings["reg"],
                native_shift_mm=list(c1.NATIVE_SHIFT_MM))


def motion_path(exp):
    return CFG2 / "motion" / f"{exp}.json"


def recipe_tail(arm, seed, smoke):
    """Recipe of the first version's launcher (run.command): V32G for the joint model, V32 without the joint-only
    items and with the contrast's own wavelet weight for a single-contrast model."""
    recipe = list(c1.V32 if is_single(arm) else c1.V32G)
    if smoke:
        recipe[recipe.index("--iters") + 1] = "30"
        recipe[recipe.index("--warmup") + 1] = "10"
    tail = ["--save_volumes"]                                 # also in a smoke run: the self-test scores a saved volume
    if seed != 42:
        tail += ["--set", f"training.seed={seed}"]
    if is_single(arm):
        slot = ARMS2[arm]["slot"]
        key = "loss.lambda_wav_plain=" if slot == "plain" else "loss.lambda_wav="
        w = [x.split("=", 1)[1] for x in recipe if x.startswith(key)][-1]
        drop = {i for i, x in enumerate(recipe) if x.startswith(c1.JOINT_ONLY)}
        recipe = [x for i, x in enumerate(recipe) if i not in drop and i + 1 not in drop]
        tail += ["--trainer", "single", *c1.CROSS_OFF, "--set", f"loss.lambda_wav={w}", "--set", f"loss.lambda_wav_plain={w}"]
    return recipe, tail


def command(af, order, cond, arm, gpu, seed, smoke, tag="", plain=False):
    """(experiment name, command). Pure. plain=True: the unpatched trainer (self-test reference only)."""
    exp = run_name(af, order, cond, arm, seed, smoke, tag) + ("_plain" if plain else "")
    recipe, tail = recipe_tail(arm, seed, smoke)
    cyc = ["--exp", exp, "--gpu", str(gpu), *recipe, "--base_cfg", str(config_path(af, order, cond, arm)), *tail]
    if plain:
        return exp, [sys.executable, "-u", "arc/cycle.py", *cyc]
    return exp, [sys.executable, "-u", str(HERE / "mpatch.py"), "--motion", str(motion_path(exp)), "--", *cyc]


def ledger_row(score, meta):
    """Ledger line: the first version's (metrics only, no clock value, no duration) plus the order and the mm file."""
    row = run1.ledger_row(score, dict(meta, argv=meta.get("argv"), settings=meta.get("settings")))
    row.update(order=meta["order"], mm_file=meta.get("mm_file"))
    return row


def needed_files(af, order, cond):
    spec, cdir = CONDITIONS2[cond], cond_dir(order, cond)
    need = [c1.MASKS[af], c1.HYW / "ksp_T1.h5", c1.HYW / "ksp_T2.h5", cdir / "schedule.npz",
            cdir / "motion_truth.json", cdir / "brain.npy", cdir / "manifest.json"]
    if spec["intra"]:
        need.append(cdir / "ksp_T2.h5")
    if spec["v1_data"]:
        need.append(c1.DATA / spec["v1_data"] / "ksp_T2.h5")
    if spec["lesions"]:
        need.append(cdir / "ksp_T1.h5")
    return need


LOSS = re.compile(r"\[iter\s+\d+/\d+\] loss=\S+")


def run_one(af, order, cond, arm, gpu, seed, smoke, settings, tag="", plain=False):
    missing = [str(p) for p in needed_files(af, order, cond) if not Path(p).exists()]
    assert not missing, f"missing data files: {missing}"
    digest = run1._source_digest()                           # cwd is still the job's code directory here
    exp, cmd = command(af, order, cond, arm, gpu, seed, smoke, tag, plain=plain)
    score_file = RES2 / f"{exp}_score.json"
    if score_file.exists():
        assert smoke, f"{score_file} exists; finished experiments are not overwritten"
        score_file.unlink()
    cfg = write_config(af, order, cond, arm)
    mfile = None
    if not plain:
        mfile = motion_path(exp)
        mfile.parent.mkdir(parents=True, exist_ok=True)
        mfile.write_text(json.dumps(motion_config(af, order, cond, arm, seed, settings, smoke=smoke, tag=tag), indent=1))
    log(f"===== {exp}\n" + " ".join(cmd))
    p = subprocess.Popen(cmd, cwd=str(TR2), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    losses = []
    for line in p.stdout:
        losses += LOSS.findall(line)
        sys.stdout.write(run1.strip_durations(line))
        sys.stdout.flush()
    rc = p.wait()
    assert rc == 0 and score_file.exists(), f"{exp} failed (exit {rc})"
    row = ledger_row(json.loads(score_file.read_text()),
                     dict(exp=exp, af=af, order=order, cond=cond, arm=arm, seed=seed, tag=tag,
                          config_sha256=run1.sha(cfg), source_digest=digest, settings=None if plain else settings,
                          argv=cmd[2:], motion_file=None, mm_file=None if plain else str(RES2 / f"{exp}_mm.json")))
    if not smoke:
        with open(LEDGER2, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(row) + "\n")
    art = os.environ.get("MCGS_ARTIFACT_DIR")
    if art:
        (Path(art) / f"{exp}_row.json").write_text(json.dumps(row, indent=1))
        if not plain and (RES2 / f"{exp}_mm.json").exists():
            shutil.copy(RES2 / f"{exp}_mm.json", Path(art) / f"{exp}_mm.json")
    for name, s in (("plain", row["plain"]), ("contrast", row["contrast"])):
        if s:
            log(f"RESULT {exp} {name}: head_psnr={s['head_psnr']:.3f} head_ssim={s['head_ssim']:.4f} nmse={s['nmse']:.5f}")
    row["_losses"] = losses
    return row


def head_scores(exp, key, single=False):
    """Head metrics (the frozen scorer's score_headonly) of one array of <exp>_canon.npz against the saved GT."""
    import numpy as np
    if str(TR2) not in sys.path:
        sys.path.insert(0, str(TR2))
    from diag_artefact.headonly import score_headonly
    with np.load(RES2 / f"{exp}_volumes.npz") as v:
        gt = v["gt"] if single else v["gtFL"]
    with np.load(RES2 / f"{exp}_canon.npz") as z:
        return score_headonly(np.abs(z[key]), gt)


def dry_all():
    bad = 0
    for part, rows in (*MATRIX2.items(), ("formal", FORMAL)):
        for order, cond, arm in rows:
            exp, _ = command("af12", order, cond, arm, "0", 42, False)
            missing = [str(p) for p in needed_files("af12", order, cond) if not Path(p).exists()]
            done = (RES2 / f"{exp}_score.json").exists()
            bad += bool(missing)
            log(f"{'ok ' if not missing else 'MISSING'} {part:7s} {exp:30s} {'scored' if done else 'not run'} {missing or ''}")
    log(f"dry run: {bad} rows with missing files")
    return bad


def selftest(gpu, settings):
    """30-iteration runs on real data. (a) B0 through the patch (monitor on) prints the same losses as the unpatched
    trainer; (b) truth weights on C0 (all 1): the kept-lines hard-DC image scores like the scorer's own; (c) truth
    weights on DTa run and write both files."""
    a = run_one("af12", "il", "C0", "B0", gpu, 42, True, settings)
    b = run_one("af12", "il", "C0", "B0", gpu, 42, True, settings, plain=True)
    # the printed losses must be equal; the scored PSNR may differ at the level of two GPU runs of the same code
    # (1e-6 dB observed between the patched and the unpatched run), so it gets a tolerance of 1e-4 dB
    dmax = max(abs(a[c]["head_psnr"] - b[c]["head_psnr"]) for c in ("plain", "contrast"))
    same = a["_losses"] == b["_losses"] and len(a["_losses"]) > 0 and dmax < 1e-4
    mm = json.loads((RES2 / f"{a['exp']}_mm.json").read_text())
    log(f"selftest (a): {len(a['_losses'])} loss lines identical to the unpatched trainer, head PSNR within {dmax:.1e} dB: {same}; "
        f"statistic computed {len(mm['m2']['history'])} times")
    assert same and len(mm["m2"]["history"]) >= 2, f"the monitor changed the training: {a['_losses'][-2:]} vs {b['_losses'][-2:]}"
    log("SELFTEST ok (a)")
    c = run_one("af12", "il", "C0", "Ow", gpu, 42, True, settings)
    s = head_scores(c["exp"], "rec_w_dc")
    d = abs(s["head_psnr"] - c["contrast"]["head_psnr"])
    log(f"selftest (b): kept-lines hard-DC head PSNR {s['head_psnr']:.5f} vs scorer {c['contrast']['head_psnr']:.5f}, "
        f"difference {d:.2e} dB (limit 1e-4)")
    assert d < 1e-4, "the kept-lines image does not reproduce the scorer when every weight is 1"
    log("SELFTEST ok (b)")
    e = run_one("af12", "il", "DTa", "Ow", gpu, 42, True, settings)
    mm = json.loads((RES2 / f"{e['exp']}_mm.json").read_text())
    n0 = sum(1 for w in mm["m2"]["w"] if w == 0.0)
    assert (RES2 / f"{e['exp']}_canon.npz").exists() and n0 > 0
    log(f"selftest (c): DTa with truth weights ran; {n0} trains at weight 0; "
        f"T2 head PSNR {e['contrast']['head_psnr']:.3f} (all lines), {head_scores(e['exp'], 'rec_w_dc')['head_psnr']:.3f} (kept lines)")
    log("SELFTEST ok (c)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--af", choices=sorted(c1.MASKS))
    ap.add_argument("--order", choices=ORDERS)
    ap.add_argument("--cond", choices=sorted(CONDITIONS2))
    ap.add_argument("--arm", choices=sorted(ARMS2))
    ap.add_argument("--gpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--settings", default="settings_dev.json", help="JSON file with m1 / m2 / m3 / reg (relative to this folder)")
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
    n = run1.ensure_trainer(tr=TR2, root=ROOT2, res=RES2)
    log(f"trainer copy verified against {c1.SRC_HEAD[:10]}: {n} files")
    settings = json.loads((HERE / args.settings).read_text())
    if args.selftest:
        selftest(args.gpu, settings)
        return
    run_one(args.af, args.order, args.cond, args.arm, args.gpu, args.seed, args.smoke, settings, args.tag)


if __name__ == "__main__":
    main()
