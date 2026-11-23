#!/usr/bin/env python3
"""tune12.py -- one MC-GS training of the AF12 plain-scan / enhanced tuning loop (program.md in this directory).

Question (user, 2026-11-22): can parameter tuning raise the reconstruction quality of MC-GS on the AF12
plain-scan / enhanced pair above the untuned September recipe (the result of record)?

One call = one training of the joint model with the registered recipe P3 plus the overrides given:

  gpu.py run --gpu 2 -- @python autoresearch/ce_af12_tune_20261122/tune12.py \
      --name c1_edge0 "--ov=--lambda_edge 0" "--hyp=common-edge prior off"

  --ov       flag/value pairs of arc/cycle.py, later ones win ("--lambda_edge 0 --set loss.x=1")
  --seed     42 (default)
  --volumes  also save the reconstructed volumes (keeper / finalisation runs)
  --dry      build and check the command only (CPU)

Fixed: data, truth, mask, coil maps and config of the AF12 topic (configs/ce_retro_af12.yaml), the trainer copy of the
framework ablation (source 54f8a2a; it has the cross-contrast guides the September snapshot lacks) and its scorer.
The first run of the loop is the control (no override), which says how far this trainer is from the registered run
of the September snapshot. Metric m = mean of the two head PSNR (plain scan, enhanced) of the trainer's score json;
SSIM, NMSE and HFEN are logged. The truth is used for scoring only, never by the training.

Experiments are named ta12_<name>; an existing name is refused. Each run appends one line to
ROOT/experiments.jsonl (copied to MCGS_ARTIFACT_DIR with the score json). No clock value and no measured duration is
written (the project's time rule).
"""
import argparse
import fcntl
import hashlib
import json, time
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

TR = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_framework_ablation_20261110/trainer")
RES = TR / "results_arc"
T = Path("/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af12")
CFG = T / "configs/ce_retro_af12.yaml"
REG = Path("/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16/trainer/_trainer_e778bf2/results_arc/"
           "ce_retro_af12_joint_score.json")                      # the registered AF12 run (September snapshot)
ROOT = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_af12_tune_20261122")
# the registered recipe (run_af12.py P3 = abl_framework.py P3), checked below against the registered run's argv
P3 = ["--iters", "4000", "--lambda_geo", "0.10", "--warmup", "500", "--max_radius", "20", "--kspace_high", "2.2",
      "--edge_tv", "--edge_tv_kappa_q", "0.85", "--edge_tv_wfloor", "0.10", "--tv_eps", "0.04",
      "--set", "gaussian.init_mode=head", "--set", "gaussian.init_scale_mode=3nn",
      "--set", "loss.edge_tv_kappa_source=zf", "--set", "gaussian.initial_num_points=3600000",
      "--set", "loss.wav_name=db4", "--set", "loss.wav_levels=1",
      "--set", "loss.lambda_wav_plain=0.12", "--set", "loss.lambda_wav=0.10",
      "--lambda_edge", "0.001", "--lambda_ptv", "0.03"]
KEYS = ("head_psnr", "head_ssim", "head_hfen", "nmse")
log = lambda *a: print(*a, flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True, help="experiment is ta12_<name>")
ap.add_argument("--ov", default="")
ap.add_argument("--hyp", default="")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--volumes", action="store_true")
ap.add_argument("--dry", action="store_true")
ap.add_argument("--share_mib", type=int, default=0,
                help="user 2026-11-22: a card with only a lightweight task may be shared; start if it holds at most this much")
args = ap.parse_args()
assert re.fullmatch(r"[a-z0-9_]+", args.name), "--name: lower-case letters, digits, underscore"


def strip(argv, drop=("--exp", "--gpu", "--base_cfg")):
    out, i = [], 0
    while i < len(argv):
        if argv[i] in drop:
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


ov = shlex.split(args.ov)
assert len(ov) % 2 == 0, f"overrides must be flag/value pairs: {ov}"
for flag, val in zip(ov[0::2], ov[1::2]):
    assert flag.startswith("--") and flag not in ("--exp", "--gpu", "--base_cfg", "--from_ckpt", "--trainer"), flag
    assert flag != "--set" or "=" in val, f"--set needs KEY=VALUE: {val}"
reg = json.load(open(REG))
assert strip(reg["argv"]) == P3 + ["--save_volumes"], f"P3 is not the registered AF12 command:\n{reg['argv']}"
assert CFG.is_file() and (TR / "arc/cycle.py").is_file()
vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
assert args.dry or vis.isdigit(), f"needs exactly one GPU from gpu.py --gpu, got CUDA_VISIBLE_DEVICES={vis!r}"
cyc = [*P3, *(["--save_volumes"] if args.volumes else []), *ov]
if args.seed != 42:
    cyc += ["--set", f"training.seed={args.seed}"]
exp = f"ta12_{args.name}"
cmd = [sys.executable, "-u", "arc/cycle.py", "--exp", exp, "--gpu", vis or "0", *cyc, "--base_cfg", str(CFG)]
outs = [RES / f"{exp}.log", RES / f"{exp}_score.json", RES / exp, RES / f"{exp}_volumes.npz"]
if args.dry:
    assert not (ROOT / "claims" / exp).exists() and not any(p.exists() for p in outs), f"{exp} is taken"
    log(f"DRY {exp}: P3 = registered argv; cfg sha256 {sha(CFG)[:12]}; cycle.py sha256 {sha(TR / 'arc/cycle.py')[:12]}; "
        f"registered pre {reg['plain']['head_psnr']:.3f} post {reg['contrast']['head_psnr']:.3f}\n{' '.join(cmd)}")
    sys.exit(0)
def card_pids():
    r = subprocess.run(["nvidia-smi", "-i", vis, "--query-compute-apps=pid", "--format=csv,noheader"],
                       capture_output=True, text=True, check=True)
    return {int(x) for x in r.stdout.split()}
def card_used():
    r = subprocess.run(["nvidia-smi", "-i", vis, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True)
    return int(r.stdout.split()[0])
if args.share_mib:
    if card_used() > args.share_mib:
        log(f"CARD_BUSY {exp}: GPU {vis} holds {card_used()} MiB > {args.share_mib}; nothing started, name not used")
        sys.exit(3)
elif card_pids():                                                 # someone took the card since the local check
    log(f"CARD_BUSY {exp}: GPU {vis} already has a compute process; nothing started, name not used")
    sys.exit(3)
(ROOT / "claims").mkdir(parents=True, exist_ok=True)
(ROOT / "claims" / exp).mkdir()                                   # an existing name stops here
assert not any(p.exists() for p in outs), f"refusing to overwrite outputs of {exp}"
art = Path(os.environ.get("MCGS_ARTIFACT_DIR", ROOT / "artifacts"))
art.mkdir(parents=True, exist_ok=True)
log(f"===== {exp}; physical GPU {vis}; overrides {ov or 'none'}; seed {args.seed}\n{' '.join(cmd)}")
env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID")
with open(outs[0], "w") as lf:                                    # one job per card: leave if another process shows up
    pr = subprocess.Popen(cmd, cwd=TR, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    polls = 0
    while pr.poll() is None:
        time.sleep(15)
        polls += 1
        if polls > 40 or args.share_mib:                                            # only the first ten minutes are a launch race
            continue
        mine = {int(x) for x in subprocess.run(["ps", "-s", str(pr.pid), "-o", "pid="], capture_output=True,
                                               text=True).stdout.split()}
        if card_pids() - mine:
            os.killpg(pr.pid, 15)
            pr.wait()
            for q in outs:
                shutil.rmtree(q) if q.is_dir() else q.unlink(missing_ok=True)
            (ROOT / "claims" / exp).rmdir()
            log(f"CARD_SHARED {exp}: another process appeared on GPU {vis}; training stopped, outputs removed, name free")
            sys.exit(3)
    rc = pr.returncode
text = open(outs[0]).read()
entry = dict(name=args.name, exp=exp, hypothesis=args.hyp, overrides=ov, seed=args.seed, gpu=vis, rc=rc,
             job=os.environ.get("MCGS_JOB_ID"), cfg_sha256=sha(CFG), cycle_sha256=sha(TR / "arc/cycle.py"), ok=False)
ok = rc == 0 and outs[1].exists()
if ok:
    s = json.load(open(outs[1]))
    want_n = [a.split("=", 1)[1] for a in cyc if a.startswith("gaussian.initial_num_points=")][-1]
    checks = dict(npts=s["npts"] == int(float(want_n)), beat_zf=bool(s["both_beat"]),
                  argv_as_sent=strip(s["argv"]) == cyc)
    for key, tag in (("loss.edge_tv_guide_alpha=", "[edge_tv] cross-contrast guide alpha="),
                     ("loss.ptv_guide_alpha=", "[phase-tv] cross-contrast guide alpha=")):
        if any(a.startswith(key) and float(a.split("=")[1]) > 0 for a in ov):
            checks[key.rstrip("=")] = tag in text
    pre, post = {k: s["plain"][k] for k in KEYS}, {k: s["contrast"][k] for k in KEYS}
    d = dict(pre=round(pre["head_psnr"] - reg["plain"]["head_psnr"], 4),
             post=round(post["head_psnr"] - reg["contrast"]["head_psnr"], 4))
    m = (pre["head_psnr"] + post["head_psnr"]) / 2
    ok = all(checks.values())
    entry.update(npts=s["npts"], vram_gb=s["vram_gb"], checks=checks, pre=pre, post=post, m=round(m, 4),
                 delta_registered_psnr=d, ok=ok)
    shutil.copy2(outs[1], art / outs[1].name)
    log(f"===== {exp} {'done' if ok else 'CHECK FAILED ' + str(checks)}: pre {pre['head_psnr']:.3f} ({d['pre']:+.3f} vs "
        f"registered), post {post['head_psnr']:.3f} ({d['post']:+.3f}); m = {m:.4f}; SSIM {pre['head_ssim']:.4f} / "
        f"{post['head_ssim']:.4f}; NMSE {pre['nmse']:.5f} / {post['nmse']:.5f}; N={s['npts']}, {s['vram_gb']:.1f} GB")
else:
    log(f"===== {exp} FAILED rc={rc}; log tail:\n{''.join(text.splitlines(True)[-25:])}")
ledger = ROOT / "experiments.jsonl"
with open(ledger, "a+") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0)
    entry = dict(id=sum(1 for _ in f) + 1, **entry)
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
shutil.copy2(ledger, art / "experiments.jsonl")
log("LEDGER " + json.dumps(entry, ensure_ascii=False))
sys.exit(0 if ok else 1)
