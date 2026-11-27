#!/usr/bin/env python3
"""run_subject.py -- the reconstruction arms of the AF12 study on one test subject's topic (subject_topic.py).

Nothing is re-implemented. Three existing programs are run with the subject's topic in place of the record's:

  --worker        ../ce_af12_rerun_20261112/run_af12.py, its queue and rules unchanged (ZF, CG-SENSE, GS-single x 2 and
                  MC-GS with the registered recipe in the September trainer snapshot, ZS-SSDU, CS-SENSE and Joint SENSE
                  grids with the grid-edge rule, L1-SPIRiT grid, MC-INR). Text changes (PORT, each must apply exactly
                  as often as stated): topic path, grid and point target, experiment / config names, no AF16 column in
                  the table, --setup refused, and the topic's unsupported 3D-DIP / 3D-INR jobs left out (the figure of
                  record uses the support versions below).
  --support SCAN  ../ce_af12_rerun_20261112/af12_cmp_support.py (3D-DIP and 3D-INR with output support, the versions of
                  the optimal figure), text changes: topic path, output folder, and its record-only loader check.
  --tuned         MC-GS with the tuned final config of ../ce_af12_tune_20261122 (registered recipe P3 + OV, seed 42) in
                  the trainer copy that study used; the finished run is linked into the September snapshot's
                  results_arc under its own name and converted / scored by the topic's extract_gs_volumes.py, i.e. by
                  the same code as the registered MC-GS (stored as gs_joint_tuned).
  --all           --worker, --tuned, --support pre, --support post, one after another on the card gpu.py gave.
  --table         (CPU) run_af12.py's standings for this subject.

  gpu.py run --gpu ID -- @python autoresearch/ce_prepost_testset_20261124/run_subject.py --subject S --all
No truth enters any training; the truth is used by the scorers only. No clock value is printed by this wrapper.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_prepost_testset_20261124")
S16 = Path("/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16")
ARC16 = S16 / "trainer/_trainer_e778bf2/results_arc"
TRA = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_framework_ablation_20261110/trainer")
PY = "/home/congcongliu/anaconda3/envs/GSTMI/bin/python"
OV = ["--lambda_edge", "0", "--set", "loss.edge_tv_guide_alpha=0.75", "--set", "loss.ptv_guide_alpha=0.5",
      "--set", "gaussian.rotation_lr=0.008", "--set", "gaussian.scale_lr=0.004"]
log = lambda *a: print(*a, flush=True)

argv = sys.argv[1:]
if "--subject" in argv:
    i = argv.index("--subject")
    os.environ["TS_SUBJECT"] = argv[i + 1]
    del argv[i:i + 2]
SUBJ = os.environ["TS_SUBJECT"]
T = OUT / "subjects" / SUBJ / "topic_af12"
EXP = f"ce_ts{SUBJ}_af12"
setup = json.load(open(T / "logs/setup.json"))["summary"]
assert setup["exp_prefix"] == EXP
nx, ny, nz = setup["grid"]
os.environ.update(TS_TOPIC=str(T), TS_EXP=EXP, TS_GRID=f"{nx},{ny},{nz},24,{setup['points']}",
                  TS_SUPPORT=str(OUT / "subjects" / SUBJ / "cmp_support"))


def port(path, rules):
    text = path.read_text()
    for old, new, n in rules:
        assert text.count(old) == n, f"{path.name}: expected {n} x\n{old}\nfound {text.count(old)}"
        text = text.replace(old, new)
    return text


def run_ported(path, rules, args):
    sys.argv = [str(Path(__file__).resolve()), *args]
    ns = {"__name__": "__main__", "__file__": str(Path(__file__).resolve())}
    exec(compile(port(path, rules), str(path) + " (ported)", "exec"), ns)


PORT_RUN = [
    ('T = RES / "CE_retro_cssense_af12"', 'T = Path(os.environ["TS_TOPIC"])\nEXPP = os.environ["TS_EXP"]', 1),
    ("NX, NY, NZ, ACS, TARGET = 256, 232, 176, 24, 3403",
     'NX, NY, NZ, ACS, TARGET = [int(v) for v in os.environ["TS_GRID"].split(",")]', 1),
    ('f"ce_retro_af12_{arm}"', '(EXPP + "_" + arm)', 3),
    ('cfg="ce_retro_af12', 'cfg=EXPP + "', 3),
    ('ref = {(r["contrast"], r["method"]): r for r in json.load(open(S16 / "logs/summary.json"))}', "ref = {}", 1),
    ("(AF12 mask, 3403 points; AF16 = the September study)", "(test subject, AF12 mask; no AF16 run exists)", 1),
    ("    if a.setup:\n        setup()", '    if a.setup:\n        raise SystemExit("topics are created by subject_topic.py")', 1),
    ('    assert len({j["name"] for j in q}) == len(q)\n    return q, cs, joint',
     '    assert len({j["name"] for j in q}) == len(q)\n'
     '    q[:] = [j for j in q if not j["name"].startswith(("inr_", "dip_"))]\n    return q, cs, joint', 1),
]
PORT_SUP = [
    ('T12 = RES / "CE_retro_cssense_af12"', 'T12 = Path(os.environ["TS_TOPIC"])', 1),
    ('OUT = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_af12_cmp_20261113")',
     'OUT = Path(os.environ["TS_SUPPORT"])', 1),
    ('    assert (OUT / "logs/check.json").exists(), "run --check first"\n', "", 1),
]
RUN_AF12 = HERE.parent / "ce_af12_rerun_20261112/run_af12.py"
SUPPORT = HERE.parent / "ce_af12_rerun_20261112/af12_cmp_support.py"


def tuned():
    """MC-GS, tuned final config: P3 + --save_volumes + OV (the argv order of the optimal run ta12_x1combo3_vol)."""
    ns = {}
    src = RUN_AF12.read_text()
    exec(src[src.index("P3 = ["):src.index("GS = {")], ns)                 # run_af12.py's P3, not a second copy
    P3 = ns["P3"]
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert vis.isdigit(), "needs exactly one card from gpu.py --gpu"
    exp = f"{EXP}_tuned"
    res = TRA / "results_arc"
    outs = [res / f"{exp}.log", res / f"{exp}_score.json", res / f"{exp}_volumes.npz"]
    cyc = [*P3, "--save_volumes", *OV]
    if not (outs[1].exists() and outs[2].exists()):
        left = [str(p) for p in outs + [res / exp] if p.exists()]
        assert not left, f"outputs of {exp} exist without a finished run: {left}"
        ref = json.load(open(res / "ta12_x1combo3_vol_score.json"))["argv"]
        drop = lambda a: [v for i, v in enumerate(a) if a[i] not in ("--exp", "--gpu", "--base_cfg")
                          and (i == 0 or a[i - 1] not in ("--exp", "--gpu", "--base_cfg"))]
        assert drop(ref) == cyc, f"the tuned command is not the optimal run's argv:\n{ref}"
        cmd = [PY, "-u", "arc/cycle.py", "--exp", exp, "--gpu", vis, *cyc, "--base_cfg", str(T / "configs" / f"{EXP}.yaml")]
        log(" ".join(cmd))
        env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID")
        with open(outs[0], "w") as lf:
            rc = subprocess.run(cmd, cwd=TRA, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc or not (outs[1].exists() and outs[2].exists()):
            log(f"tuned training failed rc={rc}; log tail:\n" + "".join(open(outs[0]).read().splitlines(True)[-25:]))
            sys.exit(rc or 5)
    for p in outs:                                                           # same extractor as the registered MC-GS
        link = ARC16 / p.name
        if not link.is_symlink():
            assert not link.exists(), link
            link.symlink_to(p)
        assert link.resolve() == p.resolve()
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", RETRO_TOPIC=str(T),
               RETRO_TRAINER=str(S16 / "trainer/_trainer_e778bf2"))
    rc = subprocess.run([PY, "-u", str(T / "scripts/extract_gs_volumes.py"), "--exp", exp, "--key", "gs_joint_tuned",
                         "--map", "pre:plain,post:contrast", "--gpu", "0"], cwd=T, env=env).returncode
    art = Path(os.environ.get("MCGS_ARTIFACT_DIR", "."))
    for s in ("pre", "post"):
        p = T / f"logs/gs_joint_tuned_{s}.json"
        if p.exists():
            shutil.copy2(p, art / f"{SUBJ}_{p.name}")
    log(f"TUNED {'OK' if rc == 0 else 'FAILED'} rc={rc}")
    sys.exit(rc)


if argv[:1] == ["--all"]:
    me = [PY, "-u", str(Path(__file__).resolve())]
    rcs = {}
    for name, args in (("worker", ["--worker"]), ("tuned", ["--tuned"]), ("support_pre", ["--support", "pre"]),
                       ("support_post", ["--support", "post"])):
        rcs[name] = subprocess.run(me + args).returncode
        log(f"STAGE {name} rc={rcs[name]}")
    subprocess.run(me + ["--table"], env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    log(f"ALL {SUBJ} {rcs}")
    sys.exit(1 if any(rcs.values()) else 0)
elif argv[:1] == ["--tuned"]:
    tuned()
elif argv[:1] == ["--support"]:
    done = all((Path(os.environ["TS_SUPPORT"]) / f"logs/record_{m}_{argv[1]}.json").exists() for m in ("dip", "inr"))
    if done:
        log(f"support {argv[1]}: both results exist, nothing to do")
        sys.exit(0)
    todo = [m for m in ("dip", "inr") if not (Path(os.environ["TS_SUPPORT"]) / f"logs/record_{m}_{argv[1]}.json").exists()]
    run_ported(SUPPORT, PORT_SUP, ["--scan", argv[1], "--methods", *todo])
else:
    run_ported(RUN_AF12, PORT_RUN, argv)
