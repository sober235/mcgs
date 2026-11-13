#!/usr/bin/env python3
"""run_uniform.py -- row 2 of the redone diff-mask study: every arm on the uniform-density mask (2026-11-11).

The September diff-mask study (Results_20260908/T1FLAIR_AF16_{gauss,radial}_v3_jiangyizhou, worktree
/data0/congcong/code/GS/mcgs_diffmask_wt) changed ONE thing against T1FLAIR_AF16_v3_jiangyizhou: the mask pattern.
This script repeats that study's procedure, arm for arm, on the mask the user approved on 2026-11-11 for row 2
(ESPIRiT-toolbox vdPoisMex with pp = 0: uniform density + the 24 x 16 fully sampled block, 2964 points), in a NEW
topic, T1FLAIR_AF16_uniform_v3_jiangyizhou. Nothing of the September topics or of the worktree's tracked files is
changed; they are read as templates and comparison columns only.

One mode per call, through ../gpu-tools/gpu.py:

  --setup            (CPU) create the topic: data links, the approved mask (byte copy, md5 checked) and its readout
                     broadcast, the per-topic scripts ported from the Gaussian topic, the three trainer configs.
                     The porting rule is proved first: applied with kind = radial it must reproduce the radial
                     topic's own scripts byte for byte. Refuses to touch an existing topic (then it only verifies).
  --worker           (GPU) run the queue: the September arms with the September arguments and grids
                     (run_all_diffmask.py's job list), MC-GS and GS-single on the frozen AF16 phase-3 recipe
                     (run_diffmask_mcgs.sh's argument list, checked against the September runs' recorded argv).
                     Cards come from gpu.py --gpu (CUDA_VISIBLE_DEVICES); every program gets its card as an explicit
                     --gpu. A card is used only while no other compute process is on it, one job per card, at most
                     --max_jobs at once. Jobs are claimed by an atomic mkdir, so several workers can share the queue.
  --finalize         (CPU) the September finalize_diffmask.sh (register the GS arms, summary, README tables,
                     three-plane figures), then the standings.
  --table            (CPU) standings from whatever has finished.
  --check_ext        (CPU) prove that starting the trainer compiles nothing (see CUDA_BIN below).
  --readme FILE      (CPU) install FILE (a path in this snapshot) as the topic README, keeping its TABLES block.
  --study CMD...     (CPU) port the cross-pattern scripts (comparison table, slice / box search, stacked figure)
                     so that they read vd-Poisson / uniform density / pseudo-radial and write here, then run each CMD.

Internal: --mcinr_chain GPU (the three steps of the MC-INR arm, called by the worker).
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import shlex
import subprocess
import sys
import time
from pathlib import Path

RES = Path("/data0/congcong/code/GS/Results_20260908")
KIND, SHORT = "uniform", "af16u"
T = RES / f"T1FLAIR_AF16_{KIND}_v3_jiangyizhou"          # the new topic (the only place this study writes, besides
G = RES / "T1FLAIR_AF16_gauss_v3_jiangyizhou"            # results_arc/af16u_* and the MC-INR topic's af16_uniform)
RAD = RES / "T1FLAIR_AF16_radial_v3_jiangyizhou"         # READ ONLY, proves the porting rule
HOUSE = RES / "T1FLAIR_AF16_v3_jiangyizhou"              # READ ONLY, the vd-Poisson column
F = Path("/data0/congcong/code/GS/mcgs_diffmask_wt")     # the September trainer worktree, run in place
ARC = F / "results_arc"
MCINR = Path("/data0/congcong/code/GS/mcinr_niessen2025")
MCINR_TOPIC = RES / "T1FLAIR_MCINR_Niessen2025_v3_jiangyizhou"
PY = "/home/congcongliu/anaconda3/envs/GSTMI/bin/python"
PY_TCNN = "/home/congcongliu/anaconda3/envs/GR_pt1.10/bin/python"
APPROVED = Path("/home/congcongliu/codex_artifacts/mcgs_diffmask_af16_redo_20261111/"
                "mask_vdpoismex_pp0_256x176_af16_preview.mat")
APPROVED_MD5 = "d0796ac1a198ae2aba8983146b3830fc"
NX, NY, NZ, CY, CZ, TARGET = 256, 256, 176, 128, 88, 2964
N_PLANE = NY * NZ
ART = Path(os.environ.get("MCGS_ARTIFACT_DIR", "."))

# run_diffmask_mcgs.sh's recipe block (the frozen AF16 phase-3 recipe), argument for argument
RECIPE = ["--iters", "4000", "--lambda_geo", "0.10", "--warmup", "500", "--max_radius", "20", "--kspace_high", "2.2",
          "--edge_tv", "--edge_tv_kappa_q", "0.85", "--edge_tv_wfloor", "0.10", "--tv_eps", "0.04",
          "--set", "gaussian.init_mode=head", "--set", "gaussian.init_scale_mode=3nn",
          "--set", "loss.edge_tv_kappa_source=zf", "--set", "gaussian.initial_num_points=3600000",
          "--set", "loss.wav_name=db4", "--set", "loss.wav_levels=1",
          "--set", "loss.lambda_wav_plain=0.12", "--set", "loss.lambda_wav=0.10",
          "--lambda_edge", "0.001", "--lambda_ptv", "0.03"]
GS_ARMS = {"joint": "t1flair_v3_{s}_jyz.yaml", "singleT1": "t1flair_v3_{s}_jyz_singleT1.yaml",
           "singleFLAIR": "t1flair_v3_{s}_jyz_singleFLAIR.yaml"}
# run_all_diffmask.py's grids
TV = {"T1": ["0.00035", "0.0005", "0.0007", "0.001", "0.0014"],
      "FLAIR": ["0.00025", "0.00035", "0.0005", "0.0007", "0.001"]}
JTV = ["0.00035", "0.0005", "0.0007", "0.001", "0.0014"]
SPIRIT_KS = ("3", "5")
SPIRIT_LAM = ("0", "0.0005", "0.001", "0.002", "0.004", "0.008")
NEEDED = {"t1t2_common.py", "recon_classical.py", "recon_joint_cssense.py", "recon_zsssdu.py", "recon_dip_inr.py",
          "recon_spirit.py", "score_mcinr_arm.py", "summarize.py", "add_gs_arm.py", "readme_tables.py",
          "refresh_readme_tables.py", "finalize_diffmask.sh", "fig_three_planes_compact.py", "check_gauge.py"}
# The September sessions had /usr/local/cuda-12.6/bin on PATH, and torch writes the nvcc it finds there into the
# voxelizer's build.ninja. Without it torch falls back to /usr/local/cuda (a link to the same toolkit), the file
# changes and ninja would RECOMPILE the worktree's extension. With it the file is byte-identical and a trainer
# start compiles nothing; --check_ext proves that on a private copy of the build directory before anything runs.
CUDA_BIN = "/usr/local/cuda-12.6/bin"
ENV = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID",
           PATH=CUDA_BIN + ":" + os.environ.get("PATH", ""))
GS_ENV = dict(ENV, TORCH_EXTENSIONS_DIR=str(F) + ".torch_ext")      # as run_diffmask_mcgs.sh sets it
EXT_BUILD = F / "paired_gs/gaussian/cuda_ext/build"
log = lambda *a: print(*a, flush=True)


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def port(text, kind, short):
    """the September derivation of a per-topic script from the Gaussian topic's (checked against the radial topic)"""
    for a, b in (("T1FLAIR_AF16_gauss_v3_jiangyizhou", f"T1FLAIR_AF16_{kind}_v3_jiangyizhou"),
                 ("t1flair_v3_af16_diffmask_gauss", f"t1flair_v3_af16_diffmask_{kind}"),
                 ("mask_gauss_256x176_af16", f"mask_{kind}_256x176_af16"),
                 ("t1flair_v3_af16g_", f"t1flair_v3_{short}_"),
                 ("--tag _gauss", f"--tag _{kind}"),
                 ('default="_gauss"', f'default="_{kind}"')):
        text = text.replace(a, b)
    return text


def exp_name(arm):
    return f"{SHORT}_p3_{arm}_full4k"


# ---------------------------------------------------------------------------------------------- setup
def acs_box(m2):
    """recon_spirit.py's acs_box(), transcribed: the block L1-SPIRiT will calibrate from"""
    ny, nz = m2.shape
    y0, z0, h, w = ny // 2 - 1, nz // 2 - 1, 2, 2
    grew = True
    while grew:
        grew = False
        if y0 > 0 and y0 + h < ny and bool(m2[y0 - 1:y0 + h + 1, z0:z0 + w].all()):
            y0, h, grew = y0 - 1, h + 2, True
        if z0 > 0 and z0 + w < nz and bool(m2[y0:y0 + h, z0 - 1:z0 + w + 1].all()):
            z0, w, grew = z0 - 1, w + 2, True
    return h, w, y0, z0


def mask_report():
    import h5py
    import numpy as np
    import scipy.io as sio
    mat = T / f"data/mask_{KIND}_256x176_af16.mat"
    assert md5(mat) == APPROVED_MD5, f"{mat} is not the approved mask"
    d = sio.loadmat(mat)
    m = d["mask"] != 0
    assert m.shape == (NY, NZ) and int(m.sum()) == TARGET, (m.shape, int(m.sum()))
    with h5py.File(T / "data/mask_af16_3d.h5", "r") as h:
        m3 = h["mask"][:]
    assert m3.shape == (NX, NY, NZ) and m3.dtype == np.uint8 and (m3 == m.astype(np.uint8)[None]).all()
    with h5py.File(T / "data/ksp_FLAIR.h5", "r") as f:
        sup = np.any(np.abs(np.array(f["kspace"][0])) > 0, axis=0)
    assert sup.shape == (NY, NZ) and int(sup.sum()) == 34139, (sup.shape, int(sup.sum()))
    ky, kz = np.mgrid[0:NY, 0:NZ]
    rn = np.sqrt(((ky - CY) / (NY / 2)) ** 2 + ((kz - CZ) / (NZ / 2)) ** 2)
    n, n_fl = int(m.sum()), int((m & sup).sum())
    h, w, y0, z0 = acs_box(m)
    return dict(kind=KIND, pattern="uniform density + fully sampled centre block: Michael Lustig's ESPIRiT-toolbox "
                "vdPoisMex(256,176,256,176, accelx, 4, 24, 0, 0), pp = 0 (the house call uses pp = 1)",
                source=str(APPROVED), mask_mat=str(mat), mask_mat_md5=md5(mat), mask3d=str(T / "data/mask_af16_3d.h5"),
                generator={k: float(d[k].squeeze()) for k in ("accelx", "accely", "calib", "ellipse", "pp", "draw",
                                                               "n_draws", "exact_hit", "accel_star") if k in d},
                points=n, plane_points=N_PLANE, T1_AF=N_PLANE / n, FLAIR_points_with_data=n_fl,
                FLAIR_points_outside_shutter=n - n_fl, FLAIR_AF=N_PLANE / n_fl, FLAIR_AF_vs_own_shutter=34139 / n_fl,
                centre24x24=int(m[CY - 12:CY + 12, CZ - 12:CZ + 12].sum()), acs_box_ky_kz=[h, w], acs_box_origin=[y0, z0],
                median_rn=float(np.median(rn[m])), mean_rn=float(rn[m].mean()))


def setup():
    import h5py
    import numpy as np
    import scipy.io as sio
    import yaml
    if T.exists():
        log(f"{T} exists: nothing is written, verifying only")
        log(json.dumps(mask_report(), indent=1))
        return
    # 1. prove the porting rule on the radial topic, collect the per-topic scripts
    per_topic = []
    for f in sorted((RAD / "scripts").iterdir()):
        g = G / "scripts" / f.name
        if f.is_file() and f.suffix in (".py", ".sh") and g.is_file() and port(g.read_text(), "radial", "af16r") == f.read_text():
            per_topic.append(f.name)
    assert NEEDED <= set(per_topic), f"porting rule does not reproduce the radial topic for {NEEDED - set(per_topic)}"
    assert md5(APPROVED) == APPROVED_MD5, "the approved mask file changed"
    for arm in GS_ARMS:
        for p in (ARC / f"{exp_name(arm)}.log", ARC / f"{exp_name(arm)}_score.json", ARC / exp_name(arm)):
            assert not p.exists(), f"{p} exists"
    assert not (MCINR_TOPIC / f"recon/af16_{KIND}").exists()
    # 2. the topic
    for d in ("data", "scripts", "configs", "logs/spirit", "logs/claims", "recon", "figures"):
        (T / d).mkdir(parents=True)
    for name in ("ksp_T1.h5", "ksp_FLAIR.h5", "csm_espirit_T1.h5", "csm_espirit_FLAIR.h5"):
        src = G / "data" / name
        assert src.is_symlink() and src.exists(), src
        (T / "data" / name).symlink_to(os.readlink(src))
    mat = T / f"data/mask_{KIND}_256x176_af16.mat"
    shutil.copyfile(APPROVED, mat)
    m = sio.loadmat(mat)["mask"] != 0
    assert m.shape == (NY, NZ) and int(m.sum()) == TARGET
    m3 = np.broadcast_to(m[None, :, :], (NX, NY, NZ)).astype(np.uint8)       # make_masks_af16_diffmask.py's layout
    with h5py.File(T / "data/mask_af16_3d.h5", "w") as h:
        h.create_dataset("mask", data=m3, compression="gzip", compression_opts=1)
    for name in per_topic:
        src, dst = G / "scripts" / name, T / "scripts" / name
        dst.write_text(port(src.read_text(), KIND, SHORT))
        shutil.copymode(src, dst)
    # 3. trainer configs: the Gaussian topic's, paths and names ported, nothing else
    rep = mask_report()
    header = (
        "# ============================================================================\n"
        f"# {{name}} -- jiangyizhou T1+FLAIR, diff-mask study redone 2026-11-11, nominal AF 16.\n"
        f"# Cloned from the worktree's {{src}}; the ONLY difference from T1FLAIR_AF16_v3_jiangyizhou is the mask\n"
        "# PATTERN: uniform density + the 24 x 16 fully sampled block (ESPIRiT-toolbox vdPoisMex, pp = 0).\n"
        f"# Mask: data/mask_{KIND}_256x176_af16.mat, md5 {APPROVED_MD5}, a byte copy of the mask the user approved on\n"
        f"#   2026-11-11 ({APPROVED.name}); not regenerable (vdPoisMex seeds from time(0)).\n"
        f"#   {rep['points']} of {N_PLANE} PE points: T1 AF {rep['T1_AF']:.3f}, FLAIR AF {rep['FLAIR_AF']:.3f}\n"
        f"#   ({rep['FLAIR_points_outside_shutter']} mask points fall outside FLAIR's elliptical shutter). Labelled AF 16 (nominal).\n"
        "# Data, geometry, gauge and every recipe value: unchanged from the September diff-mask configs. The recipe's\n"
        "# argument list lives in autoresearch/diffmask_af16_masks_20261111/run_uniform.py (RECIPE), a transcript of\n"
        "# run_diffmask_mcgs.sh's, checked at run time against the September runs' recorded argv.\n"
        "# ============================================================================\n")
    for arm, pat in GS_ARMS.items():
        src = F / "configs" / pat.format(s="af16g")
        body = port(src.read_text(), KIND, SHORT)
        lines = body.splitlines(keepends=True)
        first = next(i for i, ln in enumerate(lines) if not ln.startswith("#"))
        dst = T / "configs" / pat.format(s=SHORT)
        dst.write_text(header.format(name=dst.name, src=f"configs/{src.name}") + "".join(lines[first:]))
        a, b = yaml.safe_load(open(src)), yaml.safe_load(open(dst))
        assert json.loads(port(json.dumps(a), KIND, SHORT)) == b, f"{dst.name} differs from {src.name} beyond the topic name"
        assert all(str(T) in b["data"][ch][k] for ch in ("plain", "contrast") for k in ("data_path", "csm_path", "mask_path"))
    # 4. the read-outs the shared figure and table scripts expect
    json.dump(rep, open(T / "logs/register_mask.json", "w"), indent=1)
    ex = json.load(open(G / "logs/prep_04_export.json"))
    ex["note"] = ("written by run_uniform.py --setup (2026-11-11) in the format the Gaussian topic's prep_04_export.json "
                  "has; the mask is the approved vdPoisMex pp = 0 draw and the k-space h5 files are symlinks into the "
                  "AF12 topic, so nothing was exported here.")
    ex.update(mask2d_samples=rep["points"], mask2d_frac=rep["points"] / N_PLANE, mask2d_AF_rect=rep["T1_AF"],
              mask3d_path=rep["mask3d"])
    for c, n_c, sup_n in (("T1", rep["points"], N_PLANE), ("FLAIR", rep["FLAIR_points_with_data"], 34139)):
        e = ex["contrasts"][c]
        assert abs(e["support_frac"] - sup_n / N_PLANE) < 1e-9, (c, e["support_frac"])
        e.update(sampled_frac_of_plane=n_c / N_PLANE, AF_vs_full_rect=N_PLANE / n_c, AF_vs_own_support=sup_n / n_c,
                 acs24_kept=rep["centre24x24"] / 576, ksp_h5=str(T / f"data/ksp_{c}.h5"))
        assert Path(e["ksp_h5_target"]).resolve() == (T / f"data/ksp_{c}.h5").resolve()
    json.dump(ex, open(T / "logs/prep_04_export.json", "w"), indent=2)
    (T / "README.md").write_text(
        f"# Retrospective nominal AF 16 on the 3D T1 + FLAIR pair, jiangyizhou -- uniform-density mask (diff-mask study, redone 2026-11-11)\n\n"
        "Status: topic created by `run_uniform.py --setup`; reconstructions running. This file is replaced when they finish.\n\n"
        "## Results\n\n<!-- TABLES:BEGIN -->\n<!-- TABLES:END -->\n")
    json.dump(dict(per_topic_scripts=per_topic, worktree_head=git_head(), job=os.environ.get("MCGS_JOB_ID"),
                   utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), open(T / "logs/setup.json", "w"), indent=1)
    log(f"created {T}: {len(per_topic)} per-topic scripts ported (rule reproduces the radial topic byte for byte)")
    log(json.dumps(rep, indent=1))
    shutil.copy2(T / "logs/register_mask.json", ART / "register_mask.json")


def check_ext():
    """Would a trainer start recompile the September voxelizer? Tried on a private copy of its build directory:
    the trainer's own loader runs with build_directory redirected there (sources stay the worktree's)."""
    import tempfile
    snap = lambda d: {q.name: md5(q) for q in sorted(d.iterdir()) if q.is_file() and q.name != "lock"}
    tmp = Path(tempfile.mkdtemp(prefix="extcheck_"))
    subprocess.run(["cp", "-a", str(EXT_BUILD), str(tmp / "build")], check=True)
    real_before, before = snap(EXT_BUILD), snap(tmp / "build")
    assert real_before == before and any(k.endswith(".so") for k in before), "no compiled voxelizer in the worktree"
    code = ("import sys, torch.utils.cpp_extension as ce\n"
            "orig = ce.load\n"
            "def load(*a, **k):\n"
            f"    k['build_directory'] = {str(tmp / 'build')!r}; k['verbose'] = True\n"
            "    return orig(*a, **k)\n"
            "ce.load = load\n"
            f"sys.path.insert(0, {str(F)!r}); sys.path.insert(0, {str(F / 'paired_gs')!r})\n"
            "from gaussian import tile_voxelizer as tv\n"
            "tv._load_cuda_ext()\n")
    r = subprocess.run([PY, "-c", code], cwd=F, env=GS_ENV, capture_output=True, text=True)
    after, real_after = snap(tmp / "build"), snap(EXT_BUILD)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    shutil.rmtree(tmp, ignore_errors=True)
    assert real_before == real_after, "the worktree's build directory changed during the check"
    ok = not changed
    log(f"[check_ext] files changed in the private copy: {changed or 'none'} -> a trainer start "
        f"{'compiles nothing' if ok else 'WOULD RECOMPILE the extension'}")
    if not ok:
        log((r.stdout + r.stderr)[-3000:])
    return ok


def git_head():
    return subprocess.run(["git", "-C", str(F), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------------------------------------- queue
def gs_argv(arm, exp=None, extra=()):
    cfg = T / "configs" / GS_ARMS[arm].format(s=SHORT)
    return [PY, "-u", "arc/cycle.py", "--exp", exp or exp_name(arm), "--gpu", "{gpu}", *RECIPE, "--base_cfg", str(cfg),
            "--save_volumes", *extra]


def tune_name(tag, arm):
    """a run that leaves the frozen recipe: the recipe plus extra arguments (the last occurrence of an option wins)"""
    assert re.fullmatch(r"[a-z0-9]+", tag) and arm in GS_ARMS, (tag, arm)
    return f"{SHORT}_tune_{tag}_{arm}_full4k"


def strip(argv, drop=("--exp", "--gpu", "--base_cfg")):
    out, i = [], 0
    while i < len(argv):
        if argv[i] in drop:
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def check_recipe():
    """the September Gaussian-mask runs' recorded argv must be this script's recipe, argument for argument"""
    for arm in GS_ARMS:
        s = json.load(open(ARC / f"af16g_p3_{arm}_full4k_score.json"))
        assert strip(s["argv"]) == strip(gs_argv(arm)[3:]), f"recipe differs from the September {arm} run:\n{s['argv']}"
        assert s["argv"][s["argv"].index("--base_cfg") + 1] == "configs/" + GS_ARMS[arm].format(s="af16g")
    dirty = subprocess.run(["git", "-C", str(F), "status", "--porcelain", "--", "arc", "paired_gs", "diag_artefact"],
                           capture_output=True, text=True).stdout.strip()
    assert not dirty, f"the trainer worktree has uncommitted changes:\n{dirty}"
    assert check_ext(), "starting the trainer here would recompile the September voxelizer; refusing"
    return git_head()


def build_queue(extra):
    L, q = T / "logs", []

    def add(name, need, cwd, argv, done, logp, env=ENV, **kw):
        q.append(dict(name=name, need=need, cwd=str(cwd), argv=[str(a) for a in argv], done=str(done), log=str(logp), env=env, **kw))

    def classical(c, lam=None):
        if lam is None:
            add(f"base_{c}", 12, T, [PY, "-u", "scripts/recon_classical.py", "--contrast", c, "--gpu", "{gpu}"],
                L / f"classical_{c}.json", L / f"classical_{c}.log")
        else:
            add(f"tv_{c}_{lam}", 12, T, [PY, "-u", "scripts/recon_classical.py", "--contrast", c, "--gpu", "{gpu}", "--methods",
                                         "cssense", "--admm_outer", "200", "--lam", lam, "--suffix", f"_o200_lam{lam}"],
                L / f"classical_{c}_o200_lam{lam}.json", L / f"classical_{c}_o200_lam{lam}.log")

    def jtv(lam):
        add(f"jtv_{lam}", 20, T, [PY, "-u", "scripts/recon_joint_cssense.py", "--topic", T, "--contrasts", "T1,FLAIR", "--gpu",
                                  "{gpu}", "--admm_outer", "200", "--lam", lam, "--suffix", f"_o200_lam{lam}"],
            L / f"jcssense_o200_lam{lam}.json", L / f"jcssense_o200_lam{lam}.log")

    def spirit(c, ks, lam):
        sfx = f"_ks{ks}_lam{lam}"
        add(f"spirit_{c}{sfx}", 24, T, [PY, "-u", "scripts/recon_spirit.py", "--contrast", c, "--gpu", "{gpu}", "--ksize", ks,
                                        "--lam", lam, "--iters", "150", "--chunk", "24", "--suffix", sfx],
            L / f"spirit_{c}{sfx}.json", L / f"spirit/{c}{sfx}.log")

    def gs(arm, tag=None, extra=()):
        e = tune_name(tag, arm) if tag else exp_name(arm)
        add(f"gstune_{tag}_{arm}" if tag else f"gs_{arm}", 46, F, gs_argv(arm, e, extra), ARC / f"{e}_score.json", ARC / f"{e}.log",
            env=GS_ENV, fresh=True, guard=[str(ARC / f"{e}.log"), str(ARC / e), str(ARC / f"{e}_volumes.npz")])

    for c in ("T1", "FLAIR"):                      # cheap, and the first proof that mask, data and gauge load
        classical(c)
    for arm in ("joint", "singleT1", "singleFLAIR"):
        gs(arm)
    for c in ("T1", "FLAIR"):
        add(f"zsssdu_{c}", 18, T, [PY, "-u", "scripts/recon_zsssdu.py", "--epochs", "100", "--contrast", c, "--gpu", "{gpu}"],
            L / f"zsssdu_{c}.json", L / f"zsssdu_{c}.log")
    for c in ("T1", "FLAIR"):
        for lam in TV[c]:
            classical(c, lam)
    for lam in JTV:
        jtv(lam)
    for c in ("T1", "FLAIR"):
        for ks in SPIRIT_KS:
            for lam in SPIRIT_LAM:
                spirit(c, ks, lam)
    add("mcinr", 10, T, [PY, "-u", str(Path(__file__).resolve()), "--mcinr_chain", "{gpu}"], L / "mcinr_FLAIR.json",
        L / "mcinr_chain.log", requeue_rc=3)
    for c in ("T1", "FLAIR"):
        add(f"inr_{c}", 18, T, [PY, "-u", "scripts/recon_dip_inr.py", "--method", "inr", "--iters", "1500", "--eval_every", "50",
                                "--contrast", c, "--gpu", "{gpu}"], L / f"inr_{c}.json", L / f"inr_{c}.log")
    for c in ("T1", "FLAIR"):
        add(f"dip_{c}", 40, T, [PY, "-u", "scripts/recon_dip_inr.py", "--method", "dip", "--iters", "5000", "--lr", "3e-3",
                                "--eval_every", "100", "--contrast", c, "--gpu", "{gpu}"], L / f"dip_{c}.json", L / f"dip_{c}.log")
    for item in extra:                             # grid extensions when a sweep's winner sits on the grid edge
        fam, *rest = item.split(":")
        if fam == "tv":
            classical(rest[0], rest[1])
        elif fam == "jtv":
            jtv(rest[0])
        elif fam == "spirit":
            spirit(rest[0], rest[1], rest[2])
        elif fam == "gstune":                      # NOT the frozen recipe: gstune:<tag>:<arm>:<extra cycle.py arguments>
            tag, arm, more = item.split(":", 3)[1:]
            gs(arm, tag, shlex.split(more))
        else:
            raise SystemExit(f"--add {item}: tv:<c>:<lam> | jtv:<lam> | spirit:<c>:<ks>:<lam> | gstune:<tag>:<arm>:<args>")
    assert len({j["name"] for j in q}) == len(q)
    return q


def mcinr_chain(gpu):
    """the MC-INR arm: the reproduction on this topic's mask, its own CPU scoring, then the topic's re-score"""
    tag = f"_{KIND}"
    mask = str(T / "data/mask_af16_3d.h5")
    env = dict(os.environ, PYTHONNOUSERSITE="1", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    if not (MCINR_TOPIC / f"recon/af16{tag}/meta.json").exists():
        with open(T / f"logs/mcinr{tag}.log", "a") as lf:
            rc = subprocess.run([PY_TCNN, "-u", "scripts/run_volume.py", "--af", "16", "--tag", tag, "--gpu", str(gpu), "--resume",
                                 "--yield_to_others", "--mask_T1", mask, "--mask_FLAIR", mask],
                                cwd=MCINR, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc:
            sys.exit(rc)                           # 3 = yielded the card to another process: the worker re-queues
    if not (MCINR_TOPIC / f"logs/score_af16{tag}.json").exists():
        with open(MCINR_TOPIC / f"logs/score_af16{tag}.log", "w") as lf:
            rc = subprocess.run([PY_TCNN, "-u", "scripts/score.py", "--af", "16", "--tag", tag], cwd=MCINR, env=env,
                                stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc:
            sys.exit(10 + rc)
    with open(T / "logs/score_mcinr_arm.log", "w") as lf:
        rc = subprocess.run([PY, "-u", "scripts/score_mcinr_arm.py", "--gpu", str(gpu), "--tag", tag], cwd=T, env=ENV,
                            stdout=lf, stderr=subprocess.STDOUT).returncode
    log(open(T / "logs/score_mcinr_arm.log").read()[-1500:])
    sys.exit(20 + rc if rc else 0)


# ---------------------------------------------------------------------------------------------- worker
def cards_state():
    """index -> pids of every compute process on the card (any user)"""
    bus = {}
    for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
                                        text=True, timeout=60).strip().splitlines():
        i, b = [x.strip() for x in line.split(",")]
        bus[b] = int(i)
    st = {i: [] for i in bus.values()}
    for line in subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id", "--format=csv,noheader"],
                                        text=True, timeout=60).strip().splitlines():
        if line.strip():
            pid, b = [x.strip() for x in line.split(",")]
            if b in bus:
                st[bus[b]].append(int(pid))
    return st


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def claim(name):
    d = T / "logs/claims" / name
    try:
        d.mkdir()
    except FileExistsError:
        try:
            owner = json.load(open(d / "owner.json"))
        except (OSError, ValueError):
            return False
        if alive(int(owner["pid"])):
            return False
        log(f"reclaiming {name}: its worker (pid {owner['pid']}, job {owner.get('job')}) is gone and it has no result")
    json.dump(dict(pid=os.getpid(), job=os.environ.get("MCGS_JOB_ID"), t=time.strftime("%F %T")), open(d / "owner.json", "w"))
    return True


def result_line(job):
    try:
        d = json.load(open(job["done"]))
    except (OSError, ValueError):
        return ""
    if "plain" in d:
        return "; ".join(f"{d[ch + '_name']} head {d[ch]['head_psnr']:.3f} / {d[ch]['head_ssim']:.4f}"
                         for ch in ("plain", "contrast") if d.get(ch)) + f" (N {d['npts']}, {d['vram_gb']:.1f} GB)"
    return "; ".join(f"{r.get('contrast', '')} {r['method']} head {r['head_psnr']:.3f} / {r['head_ssim']:.4f}"
                     for r in d.get("rows", []) if r.get("head_psnr") is not None)


def worker(a):
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cards = [int(x) for x in vis.split(",") if x.strip().isdigit()]
    assert cards and all(x.strip().isdigit() for x in vis.split(",")), f"needs cards from gpu.py --gpu, got {vis!r}"
    assert T.is_dir(), "run --setup first"
    rep, head = mask_report(), check_recipe()
    queue = [j for j in build_queue(a.add) if (not a.only or any(j["name"].startswith(p) for p in a.only.split(",")))
             and not (a.skip and any(j["name"].startswith(p) for p in a.skip.split(",")))]
    if a.first:                                    # these go to the head of the queue, in the order given
        pre = a.first.split(",")
        rank = lambda j: next((i for i, p in enumerate(pre) if j["name"].startswith(p)), len(pre))  # noqa: E731
        queue.sort(key=rank)                       # stable: the rest keeps its order
    lf = open(T / "logs/run_uniform.log", "a")

    def say(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        log(line)
        lf.write(line + "\n")
        lf.flush()

    say(f"worker pid {os.getpid()} job {os.environ.get('MCGS_JOB_ID')}: {len(queue)} jobs, cards {cards}, max {a.max_jobs} at once; "
        f"mask md5 {rep['mask_mat_md5']} ({rep['points']} points); trainer worktree {head[:7]}, recipe = the September argv")
    running, results, free_since, waited, held = [], {}, {c: None for c in cards}, None, set()
    while True:
        for r in list(running):
            rc = r["proc"].poll()
            if rc is None:
                continue
            running.remove(r)
            j, ok = r["job"], Path(r["job"]["done"]).exists()
            mins = (time.time() - r["t0"]) / 60
            if rc == j.get("requeue_rc") and not ok:
                say(f"YIELDED {j['name']} gpu{r['gpu']} after {mins:.1f} min: queued again")
                shutil.rmtree(T / "logs/claims" / j["name"], ignore_errors=True)
                queue.insert(0, j)
                continue
            results[j["name"]] = rc if not (rc == 0 and not ok) else "no result file"
            say(f"DONE {j['name']} gpu{r['gpu']} rc={rc} result={'yes' if ok else 'NO'} {mins:.1f} min  {result_line(j) if ok else ''}")
            if not ok:
                say(f"FAILED {j['name']}: not retried; log {j['log']}")
        for j in list(queue):
            if Path(j["done"]).exists():
                queue.remove(j)
                results[j["name"]] = "skip"
                say(f"SKIP {j['name']} (result exists)")
        if not queue and not running:
            break
        try:
            st = cards_state()
        except Exception as e:                     # noqa: BLE001
            say(f"nvidia-smi failed: {e}")
            time.sleep(a.poll_s)
            continue
        mine, now = {r["gpu"] for r in running}, time.time()
        for c in cards:
            if c in mine:
                continue
            if st.get(c, [1]):                     # somebody else's process is on it
                free_since[c] = None
                held.discard(c)
            elif free_since[c] is None:
                free_since[c] = now
        for c in sorted(cards, key=lambda c: c not in held):       # stay on the card this worker already uses
            if len(running) >= a.max_jobs or not queue:
                break
            if c in mine or free_since[c] is None:
                continue
            if c not in held and now - free_since[c] < a.settle_s:   # a card someone else just left: let it settle
                continue
            j = next((x for x in queue if claim(x["name"])), None)
            if j is None:
                queue.clear()                      # everything left is claimed by another worker
                break
            queue.remove(j)
            if j.get("fresh") and any(Path(p).exists() for p in j["guard"]):
                results[j["name"]] = "refused"
                say(f"REFUSED {j['name']}: outputs exist without a score ({[p for p in j['guard'] if Path(p).exists()]})")
                continue
            Path(j["log"]).parent.mkdir(parents=True, exist_ok=True)
            p = subprocess.Popen([x.replace("{gpu}", str(c)) for x in j["argv"]], cwd=j["cwd"], env=j["env"],
                                 stdout=open(j["log"], "a"), stderr=subprocess.STDOUT)
            running.append(dict(job=j, gpu=c, proc=p, t0=time.time()))
            held.add(c)
            waited = None
            say(f"START {j['name']} gpu{c} pid {p.pid} -> {j['log']}")
        if queue and len(running) < a.max_jobs and not waited:
            waited = True
            say(f"waiting for a free card among {cards}: " + ", ".join(f"gpu{c} has {len(st.get(c, []))} process(es)" for c in cards if c not in mine))
        json.dump(dict(time=time.strftime("%F %T"), running=[dict(name=r["job"]["name"], gpu=r["gpu"],
                                                                  min=round((time.time() - r["t0"]) / 60, 1)) for r in running],
                       queued=[j["name"] for j in queue], done=results), open(T / "logs/run_uniform_status.json", "w"), indent=1)
        time.sleep(a.poll_s)
    bad = {n: rc for n, rc in results.items() if rc not in (0, "skip")}
    say(f"QUEUE EMPTY. failures: {bad or 'none'}")
    table()
    sys.exit(1 if bad else 0)


# ---------------------------------------------------------------------------------------------- standings
def collect():
    """every finished arm: summarize.py's sources and its rule (a swept arm = its best lambda by head PSNR)"""
    rows, L = [], T / "logs"
    for c in ("T1", "FLAIR"):
        for stem in ("classical", "zsssdu", "dip", "inr", "mcinr"):
            p = L / f"{stem}_{c}.json"
            if p.exists():
                rows += [dict(r, contrast=c) for r in json.load(open(p)).get("rows", [])]
        for key, pat in (("cssense_o200", f"classical_{c}_o200_lam*.json"), ("jcssense", "jcssense_o200_lam*.json"),
                         ("spirit", f"spirit_{c}_ks*_lam*.json")):
            cands = [r for f in sorted(L.glob(pat)) for r in json.load(open(f)).get("rows", [])
                     if r.get("contrast", c) == c and r.get("head_psnr") is not None]
            if cands:
                b = max(cands, key=lambda r: r["head_psnr"])
                lams = sorted({float(r["lam"]) for r in cands if r.get("lam") is not None})
                rows.append(dict(b, contrast=c, method=key, n=len(cands), lams=lams,
                                 edge=bool(lams) and float(b.get("lam", -1)) in (lams[0], lams[-1])))
    for arm, key in (("joint", "gs_joint_p3"), ("singleT1", "gs_single_p3"), ("singleFLAIR", "gs_single_p3")):
        p = ARC / f"{exp_name(arm)}_score.json"
        if p.exists():
            d = json.load(open(p))
            for ch in (("plain", "contrast") if arm == "joint" else ("plain",)):
                rows.append(dict(d[ch], contrast=d[f"{ch}_name"], method=key))
    for p in sorted(ARC.glob(f"{SHORT}_tune_*_full4k_score.json")):          # runs off the frozen recipe, shown apart
        tag, arm = p.name[len(SHORT) + 6:-len("_full4k_score.json")].rsplit("_", 1)
        d = json.load(open(p))
        extra = " ".join(d["argv"][d["argv"].index("--save_volumes") + 1:])
        for ch in (("plain", "contrast") if arm == "joint" else ("plain",)):
            rows.append(dict(d[ch], contrast=d[f"{ch}_name"], method=("gs_joint" if arm == "joint" else "gs_single") + f"_tune_{tag}",
                             extra=extra))
    return rows


LABEL = {"zf": "ZF", "cgsense": "CG-SENSE", "cssense": "CS-SENSE", "zsssdu": "ZS-SSDU", "dip": "3D-DIP", "inr": "3D-INR",
         "mcinr": "MC-INR", "gs_single_p3": "GS-single", "gs_joint_p3": "MC-GS", "cssense_o200": "CS-SENSE (TV, converged)",
         "jcssense": "joint CS-SENSE (JTV)", "spirit": "L1-SPIRiT"}


def lab(m):
    if "_tune_" in m:
        return ("MC-GS" if m.startswith("gs_joint") else "GS-single") + " [" + m.split("_tune_")[1] + "]"
    return LABEL.get(m, m)


def table():
    rows, out, verdict = collect(), [], {}
    for c in ("T1", "FLAIR"):
        sub = sorted((r for r in rows if r["contrast"] == c), key=lambda r: -r["head_psnr"])
        out.append(f"\n=== {c}  (uniform-density mask, nominal AF 16; sorted by head PSNR)")
        out.append(f"{'method':26s} {'headPSNR':>9s} {'headSSIM':>9s} {'headHFEN':>9s} {'PSNR':>7s} {'SSIM':>7s}  note")
        for r in sub:
            note = (f"lam {float(r['lam']):g}" + (f" ks {r['ksize']}" if r.get("ksize") else "") + f" of {r['lams']}"
                    + ("  <- GRID EDGE" if r.get("edge") else "")) if r.get("lams") else ""
            if r.get("extra"):
                note = "NOT the frozen recipe: + " + r["extra"]
            out.append(f"{lab(r['method']):26s} {r['head_psnr']:9.3f} {r['head_ssim']:9.4f} {r['head_hfen']:9.4f} "
                       f"{r['psnr']:7.3f} {r['ssim']:7.4f}  {note}")
        # the comparison arms: everything that is not an MC-GS run; GS-single stays on the frozen recipe (as in September)
        others = [r for r in sub if not r["method"].startswith(("gs_joint", "gs_single_tune"))]
        for mc in (r for r in sub if r["method"].startswith("gs_joint") and others):
            v = {}
            for k, lower in (("head_psnr", False), ("head_ssim", False), ("head_hfen", True)):
                b = (min if lower else max)(others, key=lambda r: r[k])
                d = (b[k] - mc[k]) if lower else (mc[k] - b[k])
                v[k] = dict(margin=d, runner_up=lab(b["method"]), first=d > 0)
                out.append(f"  {lab(mc['method'])} {k}: {mc[k]:.4f}, best other {v[k]['runner_up']} {b[k]:.4f}, margin {d:+.4f} "
                           f"({'FIRST' if d > 0 else 'not first'})")
            same = next((r for r in sub if r["method"] == mc["method"].replace("gs_joint", "gs_single")), None)
            if same:
                out.append(f"  {lab(mc['method'])} minus the single arm on the same recipe: {mc['head_psnr'] - same['head_psnr']:+.4f} dB")
            verdict.setdefault(c, {})[mc["method"]] = v
    text = "\n".join(out)
    log(text)
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "standings.txt").write_text(text + "\n")
    json.dump(dict(rows=[{k: v for k, v in r.items() if isinstance(v, (int, float, str, bool, list))} for r in rows],
                   verdict=verdict), open(ART / "standings.json", "w"), indent=1)
    return verdict


def finalize():
    exps = [exp_name(arm) for arm in ("joint", "singleT1", "singleFLAIR")]
    with open(T / "logs/finalize_diffmask.log", "a") as lf:
        rc = subprocess.run(["bash", "scripts/finalize_diffmask.sh", KIND, *exps], cwd=T, env=ENV, stdout=lf,
                            stderr=subprocess.STDOUT).returncode
    log(open(T / "logs/finalize_diffmask.log").read()[-4000:])
    table()
    for p in (T / "logs/summary.json", T / "logs/summary.tsv", T / "logs/readme_tables.md", T / "README.md",
              T / "logs/register_mask.json"):
        if p.exists():
            shutil.copy2(p, ART / p.name)
    total = 0
    for p in sorted((T / "figures").glob("*.png")):
        if p.stat().st_size < 18e6 and total + p.stat().st_size < 45e6:
            shutil.copy2(p, ART / p.name)
            total += p.stat().st_size
    sys.exit(rc)


# ---------------------------------------------------------------------------------------------- the study level
# The cross-pattern scripts live in the Gaussian topic and name the three topics of the September study. Ported
# here they name the redone rows (vd-Poisson / uniform density / the September pseudo-radial until row 3 is
# redone), write into THIS topic and into a new served folder; the September files stay as they are.
STUDY = ("compare_masks.py", "fig_masks_transverse_stack.py", "find_slice_all_masks.py", "find_slice_nongs_gap.py",
         "find_zoom_box_visible.py", "find_zoom_box_all_masks.py", "eval_zoom_box.py")
SERVED = "/home/congcongliu/figs/t1flair_v3_af16_diffmask_redo"
# per-topic figure scripts (ported at --setup) that --study may run as well. finalize_diffmask.sh's own figure step
# asks for the retired L1-ESPIRiT column and stops (it did in September too); the figures are drawn from here with
# --l1_arm spirit --with_mcinr --with_zf, the September twelve-column layout.
TOPIC_FIGS = ("fig_three_planes_compact.py",)


def port_study(name, text):
    G_, U_ = "T1FLAIR_AF16_gauss_v3_jiangyizhou", T.name
    once = {"compare_masks.py": [
                (f"    Gaussian     {G_}    2-D Gaussian variable-density random",
                 f"    Uniform density  {U_}  vdPoisMex pp = 0: uniform density + 24 x 16 ACS (2026-11-11)"),
                ("| Gaussian | radial | Gaussian − vd-Poisson |", "| Uniform density | radial | Uniform density − vd-Poisson |"),
                ("{f(vals['Gaussian'])}", "{f(vals['Uniform density'])}"),
                ("{d(vals['Gaussian'], vals['vd-Poisson'])}", "{d(vals['Uniform density'], vals['vd-Poisson'])}")],
            "fig_masks_transverse_stack.py": [
                ("of the study), Gaussian, pseudo-radial.", "of the study), uniform density, pseudo-radial."),
                ("writes to this (Gaussian) topic's figures/", "writes to this (uniform-density) topic's figures/"),
                ("figs/t1flair_v3_af16_diffmask/ (served at http://localhost:8765/t1flair_v3_af16_diffmask/)",
                 "figs/t1flair_v3_af16_diffmask_redo/ (served at http://localhost:8765/t1flair_v3_af16_diffmask_redo/)")]}
    for a, b in once.get(name, []):
        assert text.count(a) == 1, f"{name}: expected exactly one {a!r}, found {text.count(a)}"
        text = text.replace(a, b)
    n0 = text.count(G_)
    for a, b in ((f'("Gaussian", RES / "{G_}")', f'("Uniform density", RES / "{U_}")'),
                 (f'("gauss", RES / "{G_}")', f'("uniform", RES / "{U_}")'),
                 (f'RES / "{G_}"', f'RES / "{U_}"'),
                 ('Path("/home/congcongliu/figs/t1flair_v3_af16_diffmask")', f'Path("{SERVED}")')):
        text = text.replace(a, b)
    assert n0 and G_ not in text, f"{name}: the Gaussian topic is still named"
    assert "/figs/t1flair_v3_af16_diffmask\"" not in text, f"{name}: the September served folder is still named"
    return text


def study(cmds):
    for name in STUDY:
        dst, new = T / "scripts" / name, port_study(name, (G / "scripts" / name).read_text())
        if dst.exists():
            assert dst.read_text() == new, f"{dst} exists and differs from the port"
        else:
            dst.write_text(new)
            log(f"ported {name}")
    rc = 0
    for i, cmd in enumerate(cmds):
        argv = shlex.split(cmd)
        assert argv[0] in STUDY + TOPIC_FIGS, f"{argv[0]} is not a study script"
        lp = T / "logs" / f"study_{Path(argv[0]).stem}.log"
        env = dict(ENV, CUDA_VISIBLE_DEVICES="")
        with open(lp, "a") as lf:
            lf.write(f"\n===== {time.strftime('%F %T')} {cmd}\n")
            lf.flush()
            r = subprocess.run([PY, "-u", f"scripts/{argv[0]}", *argv[1:]], cwd=T, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
            lf.write(r.stdout)
        log(f"===== [{i + 1}/{len(cmds)}] {cmd} -> rc {r.returncode}\n{r.stdout[-a_tail:]}")
        rc = rc or r.returncode
    return rc


def install_readme(src):
    new, dst = Path(src).read_text(), T / "README.md"
    old = dst.read_text()
    a, b = "<!-- TABLES:BEGIN -->", "<!-- TABLES:END -->"
    assert a in new and b in new and a in old and b in old
    block = old[old.index(a) + len(a):old.index(b)]
    dst.write_text(new[:new.index(a) + len(a)] + block + new[new.index(b):])
    log(f"installed {src} as {dst} ({len(dst.read_text())} bytes), TABLES block kept")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--setup", action="store_true")
    g.add_argument("--worker", action="store_true")
    g.add_argument("--finalize", action="store_true")
    g.add_argument("--table", action="store_true")
    g.add_argument("--check_ext", action="store_true")
    g.add_argument("--readme")
    g.add_argument("--study", nargs="*", metavar="CMD", help='port the cross-pattern scripts, then run each "script.py args"')
    g.add_argument("--mcinr_chain", type=int)
    ap.add_argument("--max_jobs", type=int, default=1)
    ap.add_argument("--first", help="--worker: comma-separated job-name prefixes that go to the head of the queue")
    ap.add_argument("--tail", type=int, default=6000, help="--study: characters of each command's output to print")
    ap.add_argument("--fetch", action="append", default=[], help="--study: topic-relative files to copy to the artifacts")
    ap.add_argument("--poll_s", type=int, default=10)
    ap.add_argument("--settle_s", type=int, default=45, help="a card another process used must be empty this long")
    ap.add_argument("--only", default="", help="comma-separated job-name prefixes")
    ap.add_argument("--skip", default="", help="comma-separated job-name prefixes")
    ap.add_argument("--add", action="append", default=[],
                    help="extra jobs: tv:<c>:<lam> | jtv:<lam> | spirit:<c>:<ks>:<lam> | gstune:<tag>:<arm>:<extra cycle.py args>")
    a = ap.parse_args()
    if a.setup:
        setup()
    elif a.worker:
        worker(a)
    elif a.finalize:
        finalize()
    elif a.table:
        table()
    elif a.check_ext:
        sys.exit(0 if check_ext() else 1)
    elif a.readme:
        install_readme(a.readme)
    elif a.study is not None:
        a_tail = a.tail
        rc = study(a.study)
        for rel in a.fetch:
            for q in sorted(T.glob(rel)):
                shutil.copy2(q, ART / q.name)
                log(f"artifact {q.name} ({q.stat().st_size} bytes)")
        sys.exit(rc)
    else:
        mcinr_chain(a.mcinr_chain)
