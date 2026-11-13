#!/usr/bin/env python3
"""abl_framework.py -- the three framework-level ablations of MC-GS (design: DESIGN.md in this directory).

Four configurations per task, each ablation compared with the same full model (not cumulative):

  full      the task's registered MC-GS recipe: anisotropic Gaussians, shared geometry + per-contrast complex
            densities, the within-contrast priors and the cross-contrast constraint that recipe has
  iso       w/o anisotropy: one learnable log-scale per Gaussian (s_x = s_y = s_z for the whole training), rotations
            frozen at identity (redundant for a sphere). Everything else as full.
  nojoint   w/o joint modelling: each contrast alone in SingleContrastTrainer (its own positions, scales, rotations,
            densities; no second contrast is loaded), the same number of Gaussians per contrast, the same
            within-contrast priors with that contrast's own wavelet weight, every cross-contrast weight 0.
            Two runs per task: nojoint_<plain name>, nojoint_<contrast name>.
  noprior   w/o within-contrast priors: magnitude (edge-weighted) TV, phase TV and wavelet L1 off. Shared geometry,
            the k-space data term with its frequency weighting, hard data consistency at output and the
            cross-contrast common-edge prior (where the full recipe has it) stay.

Tasks (TASKS below): ce = pre/post contrast AF16 (September data), t1flair = T1+FLAIR AF16 (jiangyizhou),
t1t2 = T1+T2 AF12 (huangyuwei). One seed (42), the registered iteration and point counts.

Code base: the trainer of branch exp/shared-geometry-2x2-ablation-20261009 (54f8a2a; e778bf2 plus the genuine
single-contrast trainer), copied from the server worktree SRC into ROOT/trainer and checked file by file against
that worktree's git index. No source file of the copy is edited: the isotropic variant is applied in a child
process (--child iso) that patches JointGaussianModel3D in memory and then runs arc/cycle.py unchanged.
Nothing is written outside ROOT; existing outputs are never overwritten; a row with a claim directory is skipped,
so two workers (one per GPU) can share one row list.

  gpu.py run --gpu 2 -- @python autoresearch/ce_af16_ablation_20261110/abl_framework.py --smoke
  gpu.py run --gpu 2 -- @python autoresearch/ce_af16_ablation_20261110/abl_framework.py
  gpu.py run --cpu   -- @python autoresearch/ce_af16_ablation_20261110/abl_framework.py --table
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path("/data0/congcong/code/GS/mcgs_sg2x2_wt")
SRC_HEAD = "54f8a2a3bef402e785314b748f730a9623ca9d43"
CODE_DIRS = ("arc", "paired_gs", "diag_artefact")
OUT = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_framework_ablation_20261110")
SEPT = Path("/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16")

# The AF16 phase-3 recipe (run_wd_mcgs.sh with lambda_edge 0.001); the September CE run used it unchanged.
P3 = ["--iters", "4000", "--lambda_geo", "0.10", "--warmup", "500", "--max_radius", "20", "--kspace_high", "2.2",
      "--edge_tv", "--edge_tv_kappa_q", "0.85", "--edge_tv_wfloor", "0.10", "--tv_eps", "0.04",
      "--set", "gaussian.init_mode=head", "--set", "gaussian.init_scale_mode=3nn",
      "--set", "loss.edge_tv_kappa_source=zf", "--set", "gaussian.initial_num_points=3600000",
      "--set", "loss.wav_name=db4", "--set", "loss.wav_levels=1",
      "--set", "loss.lambda_wav_plain=0.12", "--set", "loss.lambda_wav=0.10",
      "--lambda_edge", "0.001", "--lambda_ptv", "0.03"]
# T1+T2 AF12 "MC-GS joint v3.2" (T1T2_AF12_retro README, ar19_n24_T1p06_geo007): no cross-contrast loss, no phase TV.
V32 = ["--iters", "4000", "--lambda_geo", "0.07", "--warmup", "500", "--max_radius", "20", "--kspace_high", "2.2",
       "--edge_tv", "--edge_tv_kappa_q", "0.85", "--edge_tv_wfloor", "0.10", "--tv_eps", "0.04",
       "--set", "gaussian.init_mode=head", "--set", "gaussian.init_scale_mode=3nn",
       "--set", "loss.edge_tv_kappa_source=zf", "--set", "gaussian.initial_num_points=2400000",
       "--set", "loss.lambda_wav=0.05", "--set", "loss.lambda_wav_plain=0.06"]
# V32 + the cross-contrast edge-TV guide: the stage-one keeper of ablation_coupling_tune_20261111 (tuned on huangyuwei,
# frozen). The guide is a joint-only parameter (SingleContrastTrainer refuses it), so the nojoint rows drop it.
V32G = [*V32, "--set", "loss.edge_tv_guide_alpha=0.5"]
JOINT_ONLY = ("loss.edge_tv_guide_alpha=", "loss.ptv_guide_alpha=")
TB = Path("/data0/congcong/code/GS/Results_20260908/T1T2_AF12_v4_tangbing")
TASKS = {
    "ce": dict(cfg=SEPT / "configs/ce_retro_af16.yaml", recipe=P3, label="pre/post contrast, AF16"),
    "t1flair": dict(cfg=SRC / "configs/sg2x2/af16_jyz_joint.yaml", recipe=P3, label="T1+FLAIR, AF16"),
    "t1t2": dict(cfg=SRC / "configs/t1t2_af12_hyw.yaml", recipe=V32, label="T1+T2, AF12"),
    # user 2026-11-13: a T1W/T2W pair without a voxel offset -> vol4_tangbing, T2 rolled by one voxel (A-P) when
    # the data set was built (autoresearch/t1t2_aligned_subject_20261113/prep_tangbing.py). Recipes frozen.
    "t1t2tb": dict(cfg=TB / "configs/t1t2_af12_tangbing.yaml", recipe=V32G,
                   label="T1+T2, AF12, tangbing (V32 + edge-TV guide 0.5)"),
    "t1t2tbv32": dict(cfg=TB / "configs/t1t2_af12_tangbing.yaml", recipe=V32, kinds=("full",),
                      label="T1+T2, AF12, tangbing (V32 reference; full row only)"),
}
NOPRIOR = ["--lambda_geo", "0", "--lambda_ptv", "0", "--set", "loss.edge_tv=false",
           "--set", "loss.lambda_wav=0", "--set", "loss.lambda_wav_plain=0"]
CROSS_OFF = ["--lambda_edge", "0", "--lambda_dtv", "0", "--lambda_vtv", "0"]
KINDS = ("full", "iso", "nojoint_p", "nojoint_c", "noprior")
ALL_ROWS = [f"{t}:{k}" for t, v in TASKS.items() for k in v.get("kinds", KINDS)]
KEYS = ("head_psnr", "head_ssim", "head_hfen", "nmse")
log = lambda *a: print(*a, flush=True)


# ---------------------------------------------------------------- child: isotropic Gaussians, then arc/cycle.py
def child(mode, cyc):
    import runpy
    tr_root = Path.cwd()
    sys.argv = [str(tr_root / "arc/cycle.py"), *cyc]
    if mode == "iso":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = cyc[cyc.index("--gpu") + 1]      # what cycle.py sets itself
        sys.path.insert(0, str(tr_root / "paired_gs"))
        import torch
        from gaussian.joint_gaussian_model import JointGaussianModel3D as M
        from trainers.joint_trainer import JointTrainer

        init0, train0 = M.__init__, JointTrainer.train

        def init(self, *a, **k):
            init0(self, *a, **k)
            s = self.scales.data                                    # log-scales, (N, 3)
            spread = float((s.max(1).values - s.min(1).values).max())
            assert spread < 1e-6, f"initial scales are not isotropic (spread {spread}); the iso row needs 3nn/fixed init"
            self.scales = torch.nn.Parameter(s[:, :1].clone().contiguous())   # one log-scale per Gaussian
            self.rotations.requires_grad_(False)                    # identity; no effect on a sphere
            print(f"[iso] scales parameter {tuple(self.scales.shape)}, rotations frozen", flush=True)

        def get_scale_values(self):
            return torch.exp(self.scales).expand(-1, 3).contiguous()

        def train(self, *a, **k):
            out = train0(self, *a, **k)
            m = self.model
            ident = torch.zeros_like(m.rotations); ident[:, 0] = 1.0
            sv = m.get_scale_values()
            assert m.scales.shape[1] == 1 and bool((sv[:, 0] == sv[:, 1]).all()) and bool((sv[:, 0] == sv[:, 2]).all())
            assert bool((m.rotations == ident).all()), "rotations moved in the isotropic run"
            print(f"[iso] verified after training: scales {tuple(m.scales.shape)} (median {float(sv[:, 0].median()):.5f}, "
                  f"min {float(sv.min()):.5f}, max {float(sv.max()):.5f}), rotations identity", flush=True)
            return out

        M.__init__, M.get_scale_values, M.get_scales, JointTrainer.train = init, get_scale_values, get_scale_values, train
    runpy.run_path(sys.argv[0], run_name="__main__")


if len(sys.argv) > 2 and sys.argv[1] == "--child":
    child(sys.argv[2], sys.argv[4:])
    sys.exit(0)

# ---------------------------------------------------------------- parent
ap = argparse.ArgumentParser()
ap.add_argument("--rows", nargs="+", default=ALL_ROWS, choices=ALL_ROWS)
ap.add_argument("--smoke", action="store_true", help="30 iterations, no volumes, into OUT/smoke")
ap.add_argument("--table", action="store_true", help="print the result tables (CPU), run nothing")
args = ap.parse_args()
root = OUT / "smoke" if args.smoke else OUT
TR, RES, CFG = root / "trainer", root / "trainer/results_arc", root / "configs"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", str(SRC), *a], check=True, capture_output=True, text=True).stdout


def ensure_trainer():
    """Copy arc/, paired_gs/ (with its compiled voxelizer) and diag_artefact/ once, then check every git-tracked
    file of the copy against the source worktree, which must be clean at SRC_HEAD for these directories."""
    assert git("rev-parse", "HEAD").strip() == SRC_HEAD, "source worktree moved"
    assert git("status", "--porcelain", "--", *CODE_DIRS).strip() == "", "source worktree has local changes"
    if not TR.exists():
        root.mkdir(parents=True, exist_ok=True)
        tmp = root / f"trainer.tmp{os.getpid()}"
        for d in CODE_DIRS:
            shutil.copytree(SRC / d, tmp / d, ignore=shutil.ignore_patterns("__pycache__", "results_arc"))
        try:
            tmp.rename(TR)
        except OSError:                      # another worker won the race
            shutil.rmtree(tmp)
    tracked = [f for f in git("ls-files", "--", *CODE_DIRS).splitlines() if f]
    bad = [f for f in tracked if not (TR / f).is_file() or sha(TR / f) != sha(SRC / f)]
    assert not bad, f"trainer copy differs from the source worktree: {bad[:10]}"
    RES.mkdir(exist_ok=True)
    return len(tracked)


def names(task):
    import yaml
    d = yaml.safe_load(open(TASKS[task]["cfg"]))["data"]
    return d["plain"]["name"], d["contrast"]["name"]


def single_cfg(task, slot):
    """A data.plain-only config for one contrast: the task's joint config with the other contrast removed."""
    import yaml
    cfg = yaml.safe_load(open(TASKS[task]["cfg"]))
    block = cfg["data"][slot]
    cfg["data"] = {"plain": block}
    CFG.mkdir(parents=True, exist_ok=True)
    p = CFG / f"{task}_single_{block['name']}.yaml"
    text = yaml.safe_dump(cfg, sort_keys=False)
    if p.exists():
        assert p.read_text() == text, f"{p} exists with different content"
    else:
        tmp = p.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(text)
        os.replace(tmp, p)
    return p


def wav(recipe, key):
    v = [a.split("=", 1)[1] for a in recipe if a.startswith(f"loss.{key}=")]
    return v[-1] if v else None


def row_cmd(row, gpu):
    """(experiment name, command) of one row."""
    task, kind = row.split(":")
    t = TASKS[task]
    recipe = list(t["recipe"])
    if args.smoke:
        recipe[recipe.index("--iters") + 1] = "30"
        recipe[recipe.index("--warmup") + 1] = "10"
    vol = [] if args.smoke else ["--save_volumes"]
    pn, cn = names(task)
    exp = f"fw_{task}_{kind}".replace("nojoint_p", f"nojoint_{pn}").replace("nojoint_c", f"nojoint_{cn}")
    head = ["--exp", exp, "--gpu", gpu]
    cyc = [sys.executable, "-u", "arc/cycle.py"]
    if kind == "full":
        return exp, [*cyc, *head, *recipe, "--base_cfg", str(t["cfg"]), *vol]
    if kind == "iso":
        return exp, [sys.executable, "-u", str(Path(__file__).resolve()), "--child", "iso", "--",
                     *head, *recipe, "--base_cfg", str(t["cfg"]), *vol]
    if kind == "noprior":
        return exp, [*cyc, *head, *recipe, "--base_cfg", str(t["cfg"]), *vol, *NOPRIOR]
    slot = "plain" if kind == "nojoint_p" else "contrast"
    w = wav(recipe, "lambda_wav_plain" if slot == "plain" else "lambda_wav")    # this contrast's own weight
    assert w is not None
    drop = {i for i, a in enumerate(recipe) if a.startswith(JOINT_ONLY)}           # joint-only "--set k=v" pairs
    recipe = [a for i, a in enumerate(recipe) if i not in drop and i + 1 not in drop]
    return exp, [*cyc, *head, *recipe, "--base_cfg", str(single_cfg(task, slot)), *vol, "--trainer", "single",
                 *CROSS_OFF, "--set", f"loss.lambda_wav={w}", "--set", f"loss.lambda_wav_plain={w}"]


def check_ce_recipe():
    """The CE full row must replay the registered September run's own argv."""
    argv = json.load(open(SEPT / "trainer/_trainer_e778bf2/results_arc/ce_retro_af16_joint_score.json"))["argv"]
    out, i = [], 0
    while i < len(argv):
        if argv[i] in ("--exp", "--gpu", "--base_cfg"):
            i += 2
            continue
        if argv[i] != "--save_volumes":
            out.append(argv[i])
        i += 1
    assert out == P3, f"registered September argv differs from P3:\n{out}\n{P3}"


def load(p):
    return json.load(open(p)) if Path(p).exists() else None


def table():
    """Per task: the full model and each ablation, per contrast, with the difference from full."""
    lines = []
    for task, t in TASKS.items():
        pn, cn = names(task)
        s = {k: load(RES / f"fw_{task}_{k}_score.json") for k in ("full", "iso", "noprior")}
        sp, sc = load(RES / f"fw_{task}_nojoint_{pn}_score.json"), load(RES / f"fw_{task}_nojoint_{cn}_score.json")
        rows = [("Full MC-GS", s["full"] and (s["full"]["plain"], s["full"]["contrast"]), s["full"]),
                ("w/o anisotropy", s["iso"] and (s["iso"]["plain"], s["iso"]["contrast"]), s["iso"]),
                ("w/o joint modelling", (sp and sc) and (sp["plain"], sc["plain"]), None),
                ("w/o intra-contrast priors", s["noprior"] and (s["noprior"]["plain"], s["noprior"]["contrast"]), s["noprior"])]
        lines += [f"\n### {task}: {t['label']} ({pn} / {cn})\n",
                  f"| configuration | {pn} PSNR | d | SSIM | HFEN | NMSE | {cn} PSNR | d | SSIM | HFEN | NMSE | Gaussians | GB |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        ref = rows[0][1]
        kinds = t.get("kinds", KINDS)
        for (name, pc, meta), kind in zip(rows, ("full", "iso", "nojoint_p", "noprior")):
            if kind not in kinds:
                continue
            if not pc:
                lines.append(f"| {name} | (not finished) |")
                continue
            cells = []
            for i in (0, 1):
                d = f"{pc[i]['head_psnr'] - ref[i]['head_psnr']:+.3f}" if ref and name != "Full MC-GS" else ""
                cells += [f"{pc[i]['head_psnr']:.3f}", d, f"{pc[i]['head_ssim']:.4f}", f"{pc[i]['head_hfen']:.4f}",
                          f"{pc[i]['nmse']:.5f}"]
            if meta:            # no run-time column: durations are measured on the local machine (time rule)
                res = [f"{meta['npts'] / 1e6:.1f}M", f"{meta['vram_gb']:.1f}"]
            else:
                res = [f"2 x {sp['npts'] / 1e6:.1f}M", f"{max(sp['vram_gb'], sc['vram_gb']):.1f}"]
            lines.append("| " + " | ".join([name, *cells, *res]) + " |")
    text = "\n".join(lines)
    log(text)
    return text


if args.table:
    art = Path(os.environ.get("MCGS_ARTIFACT_DIR", root / "artifacts"))
    art.mkdir(parents=True, exist_ok=True)
    (art / "table.md").write_text(table() + "\n")
    for p in sorted(RES.glob("fw_*_score.json")):
        shutil.copy2(p, art / p.name)
    sys.exit(0)

vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
assert vis.isdigit(), f"needs exactly one GPU from gpu.py --gpu, got CUDA_VISIBLE_DEVICES={vis!r}"
check_ce_recipe()
n = ensure_trainer()
(root / "claims").mkdir(exist_ok=True)
art = Path(os.environ.get("MCGS_ARTIFACT_DIR", root / "artifacts"))
art.mkdir(parents=True, exist_ok=True)
log(f"root {root}; trainer copy verified ({n} tracked files, source {SRC_HEAD[:7]}); physical GPU {vis}; rows {args.rows}")
record = dict(root=str(root), source=str(SRC), source_head=SRC_HEAD, gpu=vis, smoke=args.smoke, runs=[])
failed = []
for row in args.rows:
    exp, cmd = row_cmd(row, vis)
    try:
        (root / "claims" / exp).mkdir()
    except FileExistsError:
        log(f"-- {exp}: already claimed, skipped")
        continue
    outs = [RES / f"{exp}.log", RES / f"{exp}_score.json", RES / exp, RES / f"{exp}_volumes.npz"]
    assert not any(p.exists() for p in outs), f"refusing to overwrite outputs of {exp}"
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    log(f"\n===== {exp}\n{' '.join(cmd)}")             # no server clock or durations in logs (time rule)
    with open(outs[0], "w") as lf:
        rc = subprocess.run(cmd, cwd=TR, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    run = dict(row=row, exp=exp, cmd=cmd, rc=rc)
    text = open(outs[0]).read()
    if rc == 0 and outs[1].exists() and ("[iso] verified" in text or ":iso" not in row):
        s = json.load(open(outs[1]))
        run.update(npts=s["npts"], vram_gb=s["vram_gb"], plain={k: s["plain"][k] for k in KEYS},
                   contrast=s["contrast"] and {k: s["contrast"][k] for k in KEYS})
        shutil.copy2(outs[1], art / outs[1].name)
        c = f" / {s['contrast']['head_psnr']:.3f}" if s["contrast"] else ""
        log(f"===== {exp} done: head PSNR {s['plain']['head_psnr']:.3f}{c}, N={s['npts']}, {s['vram_gb']:.1f} GB")
    else:
        failed.append(exp)
        log(f"===== {exp} FAILED rc={rc}; log tail:\n{''.join(text.splitlines(True)[-20:])}")
    record["runs"].append(run)
    (art / f"record_gpu{vis}{'_smoke' if args.smoke else ''}.json").write_text(json.dumps(record, indent=2))
log("\n")
table()
log(f"DONE; failed rows: {failed or 'none'}")
sys.exit(1 if failed else 0)
