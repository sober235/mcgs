#!/usr/bin/env python3
"""tune.py -- one training of the coupling-tuning loop (program.md in this directory).

Question: with only the parameters that exist in the joint model alone (cross-contrast coupling), can the full
MC-GS be above the independent single-contrast models on both contrasts of a task? Stage 1 therefore accepts
only coupling overrides (ALLOWED below), and the registered independent runs of the framework ablation
(fw_<task>_nojoint_*) stay the comparator, untouched. Within-contrast priors, Gaussian count, iterations, data
term, masks, references and the scorer are never changed here.

One call = one training:

  gpu.py run --gpu 2 -- @python autoresearch/ablation_coupling_tune_20261111/tune.py \
      --task ce --name c1_edge0 "--ov=--lambda_edge 0" "--hyp=common-edge prior off"

  --kind   full (default) | iso | noprior | nojoint_p | nojoint_c   the framework-ablation configurations
  --sym    stage 2 (T1+T2 only, confirmed by the user 2026-11-12): a within-contrast prior changed the same way
           in every configuration that has within-contrast priors (full, iso, both independent models); the
           no-prior row keeps it at zero. A full run with --sym must name its comparators (--ref_p / --ref_c),
           the independent models trained with the same --sym.
  --seed   42 (default); another seed only for the confirmation runs
  --ref_p / --ref_c   experiment names of the independent comparators (default: the registered fw_ runs)
  --volumes           also save the reconstructed volumes (finalisation runs)
  --smoke             30 iterations in the smoke tree

It reuses the verified trainer copy of the framework ablation (ROOT/trainer, source 54f8a2a) and checks every
tracked file of it against the git tree of that commit; it copies nothing and edits nothing. Recipes and the
ablation overrides are read from abl_framework.py (ast, no second copy), and before any override is applied
the command is checked to be, argument for argument, the registered fw_ run of the same task and kind.
Experiments are named tn_<task>_<name>; an existing name is refused. Each run appends one line to
ROOT/tune/experiments.jsonl (copied to MCGS_ARTIFACT_DIR with the score json).

Margins of a joint model: delta_indep / m against the comparators in use (--ref_p / --ref_c, default the
registered runs); delta_indep_registered against the registered independent runs; m_strict = the smaller of the
two per contrast, then the smaller contrast -- a symmetric prior change cannot win by weakening the comparator.

No clock value and no measured duration is written to the ledger or the log (the project's time rule: times
are taken on the launching machine).
"""
import argparse
import ast
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ABL = Path(__file__).resolve().parent.parent / "ce_af16_ablation_20261110" / "abl_framework.py"
SRC = Path("/data0/congcong/code/GS/mcgs_sg2x2_wt")
OUT = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_framework_ablation_20261110")
SEPT = Path("/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16")
TASKS = {
    "ce": dict(cfg=SEPT / "configs/ce_retro_af16.yaml", recipe="P3"),
    "t1flair": dict(cfg=SRC / "configs/sg2x2/af16_jyz_joint.yaml", recipe="P3"),
    "t1t2": dict(cfg=SRC / "configs/t1t2_af12_hyw.yaml", recipe="V32"),
}
KINDS = ("full", "iso", "noprior", "nojoint_p", "nojoint_c")
KEYS = ("head_psnr", "head_ssim", "head_hfen", "nmse")
# stage 1: parameters of the joint model only (no counterpart in an independent single-contrast model)
ALLOWED_FLAGS = ("--lambda_edge", "--lambda_dtv", "--dtv_eta_scale", "--edge_p", "--edge_kappa")
ALLOWED_SETS = ("loss.edge_tv_guide_alpha", "loss.ptv_guide_alpha", "loss.dtv_w_floor")
# stage 2: within-contrast priors, changed symmetrically (full, iso and both independent models alike)
SYM_FLAGS = ("--lambda_ptv",)
SYM_TASKS = ("t1t2",)
log = lambda *a: print(*a, flush=True)


def constants():
    """SRC_HEAD, CODE_DIRS, the two recipes and the ablation overrides, as written in abl_framework.py."""
    text = ABL.read_text()
    want = {"SRC_HEAD", "CODE_DIRS", "P3", "V32", "NOPRIOR", "CROSS_OFF"}
    out = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) in want:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    assert set(out) == want, f"abl_framework.py no longer defines {want - set(out)}"
    for p in (SRC, OUT, SEPT):
        assert f'Path("{p}")' in text, f"{p} is not the path abl_framework.py uses"
    return out


C = constants()

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, choices=list(TASKS))
ap.add_argument("--name", required=True, help="experiment is tn_<task>_<name>")
ap.add_argument("--kind", default="full", choices=KINDS)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--ov", default="", help='coupling overrides, one string: "--ov=--lambda_edge 0 --set loss.x=1"')
ap.add_argument("--sym", default="", help='stage 2, symmetric within-contrast overrides: "--sym=--lambda_ptv 0.03"')
ap.add_argument("--hyp", default="", help="hypothesis, for the ledger")
ap.add_argument("--ref_p", default=None)
ap.add_argument("--ref_c", default=None)
ap.add_argument("--volumes", action="store_true")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
assert re.fullmatch(r"[a-z0-9_]+", args.name), "--name: lower-case letters, digits, underscore"
root = OUT / "smoke" if args.smoke else OUT
TR, RES, TUNE = root / "trainer", root / "trainer/results_arc", root / "tune"


def parse_overrides(text):
    """Split --ov and refuse anything that is not a joint-only coupling parameter."""
    tok = shlex.split(text)
    assert len(tok) % 2 == 0, f"overrides must be flag/value pairs: {tok}"
    for flag, val in zip(tok[0::2], tok[1::2]):
        if flag == "--set":
            key = val.split("=", 1)[0]
            assert "=" in val and key in ALLOWED_SETS, f"stage 1 does not tune {key}; allowed: {ALLOWED_SETS}"
        else:
            assert flag in ALLOWED_FLAGS, f"stage 1 does not tune {flag}; allowed: {ALLOWED_FLAGS}"
            float(val)
    return tok


def parse_sym(text):
    """Split --sym and refuse anything that is not a stage-2 within-contrast prior of a stage-2 task."""
    tok = shlex.split(text)
    assert len(tok) % 2 == 0, f"symmetric overrides must be flag/value pairs: {tok}"
    assert not tok or args.task in SYM_TASKS, f"stage 2 is confirmed for {SYM_TASKS} only"
    for flag, val in zip(tok[0::2], tok[1::2]):
        assert flag in SYM_FLAGS, f"stage 2 does not tune {flag}; allowed: {SYM_FLAGS}"
        assert float(val) > 0, f"{flag} {val}: a symmetric prior is switched on, not off"
    return tok


def last_value(argv, flag):
    """The value argparse ends up with for a flag that may be given more than once."""
    v = [argv[i + 1] for i in range(len(argv) - 1) if argv[i] == flag]
    return float(v[-1]) if v else 0.0


def verify_trainer():
    """Every blob of arc/, paired_gs/, diag_artefact/ in the tree of SRC_HEAD must be in the copy, byte for byte."""
    assert TR.is_dir(), f"{TR} is missing: run the framework ablation's runner first"
    tree = subprocess.run(["git", "-C", str(SRC), "ls-tree", "-r", C["SRC_HEAD"], "--", *C["CODE_DIRS"]],
                          check=True, capture_output=True, text=True).stdout
    n, bad = 0, []
    for line in tree.splitlines():
        meta, path = line.split("\t", 1)
        if meta.split()[1] != "blob":
            continue
        f = TR / path
        data = f.read_bytes() if f.is_file() else None
        if data is None or hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() != meta.split()[2]:
            bad.append(path)
        n += 1
    assert n > 0 and not bad, f"trainer copy differs from {C['SRC_HEAD'][:7]}: {bad[:10]}"
    return n


def names(task):
    import yaml
    d = yaml.safe_load(open(TASKS[task]["cfg"]))["data"]
    return d["plain"]["name"], d["contrast"]["name"]


def registered(task, kind):
    pn, cn = names(task)
    return f"fw_{task}_{kind}".replace("nojoint_p", f"nojoint_{pn}").replace("nojoint_c", f"nojoint_{cn}")


def base_args(task, kind):
    """cycle.py arguments of the registered fw_ run of this task and kind (without --exp / --gpu)."""
    recipe = list(C[TASKS[task]["recipe"]])
    if args.smoke:
        recipe[recipe.index("--iters") + 1] = "30"
        recipe[recipe.index("--warmup") + 1] = "10"
    vol = [] if args.smoke else ["--save_volumes"]
    cfg = str(TASKS[task]["cfg"])
    if kind in ("full", "iso"):
        return [*recipe, "--base_cfg", cfg, *vol]
    if kind == "noprior":
        return [*recipe, "--base_cfg", cfg, *vol, *C["NOPRIOR"]]
    pn, cn = names(task)
    scfg = root / "configs" / f"{task}_single_{pn if kind == 'nojoint_p' else cn}.yaml"
    assert scfg.is_file(), f"{scfg} is missing"
    key = "loss.lambda_wav_plain=" if kind == "nojoint_p" else "loss.lambda_wav="
    w = [a.split("=", 1)[1] for a in recipe if a.startswith(key)][-1]        # this contrast's own wavelet weight
    return [*recipe, "--base_cfg", str(scfg), *vol, "--trainer", "single", *C["CROSS_OFF"],
            "--set", f"loss.lambda_wav={w}", "--set", f"loss.lambda_wav_plain={w}"]


def strip(argv, drop=("--exp", "--gpu")):
    out, i = [], 0
    while i < len(argv):
        if argv[i] in drop:
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def load(exp):
    p = RES / f"{exp}_score.json"
    assert p.is_file(), f"{p} is missing"
    return json.load(open(p))


def metrics(block):
    return {k: block[k] for k in KEYS}


ov = parse_overrides(args.ov)
assert not (ov and args.kind.startswith("nojoint")), "an independent model has no coupling parameter"
sym = parse_sym(args.sym)
sym_applied = [] if args.kind == "noprior" else sym               # the no-prior row has every such prior at zero
if sym and args.kind == "full" and not args.smoke:
    assert args.ref_p and args.ref_c, "a full run with --sym needs --ref_p / --ref_c (independents with the same --sym)"
vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
assert vis.isdigit(), f"needs exactly one GPU from gpu.py --gpu, got CUDA_VISIBLE_DEVICES={vis!r}"
n = verify_trainer()
base = base_args(args.task, args.kind)
reg = load(registered(args.task, args.kind))
assert strip(reg["argv"]) == base, f"not the registered command of {registered(args.task, args.kind)}:\n{strip(reg['argv'])}\n{base}"
cyc = [a for a in base if a != "--save_volumes" or args.volumes] + sym_applied + ov
if args.seed != 42:
    cyc += ["--set", f"training.seed={args.seed}"]
exp = f"tn_{args.task}_{args.name}"
head = ["--exp", exp, "--gpu", vis]
if args.kind == "iso":
    cmd = [sys.executable, "-u", str(ABL), "--child", "iso", "--", *head, *cyc]
else:
    cmd = [sys.executable, "-u", "arc/cycle.py", *head, *cyc]
(TUNE / "claims").mkdir(parents=True, exist_ok=True)
(TUNE / "claims" / exp).mkdir()                                   # an existing name stops here
outs = [RES / f"{exp}.log", RES / f"{exp}_score.json", RES / exp, RES / f"{exp}_volumes.npz"]
assert not any(p.exists() for p in outs), f"refusing to overwrite outputs of {exp}"
art = Path(os.environ.get("MCGS_ARTIFACT_DIR", TUNE / "artifacts"))
art.mkdir(parents=True, exist_ok=True)
log(f"root {root}; trainer copy verified ({n} blobs of {C['SRC_HEAD'][:7]}); physical GPU {vis}; "
    f"registered base {registered(args.task, args.kind)}; symmetric {sym_applied or 'none'}; "
    f"overrides {ov or 'none'}; seed {args.seed}")
log(f"===== {exp}\n{' '.join(cmd)}")
env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID")
with open(outs[0], "w") as lf:
    rc = subprocess.run(cmd, cwd=TR, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
text = open(outs[0]).read()
entry = dict(task=args.task, name=args.name, exp=exp, kind=args.kind, seed=args.seed, smoke=args.smoke,
             stage=2 if sym else 1, hypothesis=args.hyp, sym=sym_applied, overrides=ov, gpu=vis, rc=rc,
             job=os.environ.get("MCGS_JOB_ID"), ok=False)
ok = rc == 0 and outs[1].exists() and (args.kind != "iso" or "[iso] verified" in text)
if ok:
    s = json.load(open(outs[1]))
    pn, cn = names(args.task)
    checks = dict(npts_as_registered=s["npts"] == reg["npts"], beat_zf=bool(s["both_beat"]),
                  argv_as_sent=strip(s["argv"]) == cyc)
    if any(a.startswith("loss.edge_tv_guide_alpha=") and float(a.split("=")[1]) > 0 for a in ov) and args.kind != "noprior":
        checks["edge_guide_on"] = "[edge_tv] cross-contrast guide alpha=" in text
    if any(a.startswith("loss.ptv_guide_alpha=") and float(a.split("=")[1]) > 0 for a in ov) and args.kind != "noprior":
        checks["ptv_guide_on"] = "[phase-tv] cross-contrast guide alpha=" in text
    if last_value(sym_applied, "--lambda_ptv") > 0:
        checks["ptv_on"] = f"[phase-tv] lambda_ptv={last_value(sym_applied, '--lambda_ptv')}" in text
    entry.update(npts=s["npts"], vram_gb=s["vram_gb"], checks=checks)
    if s["contrast"]:                                              # a joint model: margins against the independents
        rp, rc_ = args.ref_p or registered(args.task, "nojoint_p"), args.ref_c or registered(args.task, "nojoint_c")
        ip, ic = load(rp)["plain"], load(rc_)["plain"]
        for ref in (rp, rc_):                                      # comparators carry the same symmetric prior
            for flag in SYM_FLAGS:
                same = last_value(strip(load(ref)["argv"]), flag) == last_value(cyc, flag)
                checks[f"ref_same_{flag.lstrip('-')}"] = checks.get(f"ref_same_{flag.lstrip('-')}", True) and (
                    same or args.kind == "noprior")
        full = load(registered(args.task, "full"))
        d = {pn: s["plain"]["head_psnr"] - ip["head_psnr"], cn: s["contrast"]["head_psnr"] - ic["head_psnr"]}
        gp, gc = load(registered(args.task, "nojoint_p"))["plain"], load(registered(args.task, "nojoint_c"))["plain"]
        dr = {pn: s["plain"]["head_psnr"] - gp["head_psnr"], cn: s["contrast"]["head_psnr"] - gc["head_psnr"]}
        ms = min(min(d[k], dr[k]) for k in d)
        entry.update({pn: metrics(s["plain"]), cn: metrics(s["contrast"]), "ref": {pn: rp, cn: rc_},
                      "delta_indep": {k: round(v, 4) for k, v in d.items()}, "m": round(min(d.values()), 4),
                      "delta_indep_registered": {k: round(v, 4) for k, v in dr.items()}, "m_strict": round(ms, 4),
                      "delta_ssim_indep": {pn: round(s["plain"]["head_ssim"] - ip["head_ssim"], 5),
                                           cn: round(s["contrast"]["head_ssim"] - ic["head_ssim"], 5)},
                      "delta_registered_full": {pn: round(s["plain"]["head_psnr"] - full["plain"]["head_psnr"], 4),
                                                cn: round(s["contrast"]["head_psnr"] - full["contrast"]["head_psnr"], 4)}})
        line = (f"{pn} {s['plain']['head_psnr']:.3f} ({d[pn]:+.3f} vs independent), "
                f"{cn} {s['contrast']['head_psnr']:.3f} ({d[cn]:+.3f}); m = {min(d.values()):+.3f}, m_strict = {ms:+.3f}")
    else:
        me = pn if args.kind == "nojoint_p" else cn
        entry.update({me: metrics(s["plain"]),
                      "delta_registered": {me: round(s["plain"]["head_psnr"] - reg["plain"]["head_psnr"], 4)}})
        line = f"{me} {s['plain']['head_psnr']:.3f}"
    ok = all(checks.values())
    entry["ok"] = ok
    shutil.copy2(outs[1], art / outs[1].name)
    log(f"===== {exp} {'done' if ok else 'CHECK FAILED ' + str(checks)}: {line}; N={s['npts']}, {s['vram_gb']:.1f} GB")
else:
    log(f"===== {exp} FAILED rc={rc}; log tail:\n{''.join(text.splitlines(True)[-25:])}")
ledger = TUNE / "experiments.jsonl"
with open(ledger, "a+") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0)
    entry = dict(id=sum(1 for _ in f) + 1, **entry)
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
shutil.copy2(ledger, art / "experiments.jsonl")
log("LEDGER " + json.dumps(entry, ensure_ascii=False))
sys.exit(0 if ok else 1)
