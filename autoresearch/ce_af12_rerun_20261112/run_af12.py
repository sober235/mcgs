#!/usr/bin/env python3
"""run_af12.py -- every reconstruction arm of the plain-scan / enhanced study again, on an AF12 mask (2026-11-12).

User 2026-11-12: "re-perform all the reconstruction experiments for all methods using the new mask with af = 12
based on the plain scan and enhanced data", "simply by replacing the mask". The September study
(Results_20260908/CE_retro_cssense_af16: CS-SENSE truth, real 42-coil maps, noisy k-space of the whole grid, eleven
arms) is repeated arm for arm with ONE change, the sampling mask (make_af12.py: the September generator with the
point target 3403 instead of 2552). Truth, coil maps, noise draw, gauge, solvers, grids, network settings and the
frozen MC-GS / GS-single recipe are the September ones. No arm is tuned for AF12.

Where things go:
  T    Results_20260908/CE_retro_cssense_af12     the new topic: mask, configs, ported scripts, logs, recon, figures
  S16  Results_20260908/CE_retro_cssense_af16     the September topic, read only -- with one exception: MC-GS and
       GS-single run in its trainer snapshot (trainer/_trainer_e778bf2, the compiled voxelizer is used in place and
       not rebuilt) and leave three NEW experiments there, results_arc/ce_retro_af12_{joint,single_pre,single_post}*.
The per-topic scripts are the September ones with the textual changes of PORT below (topic path, mask file name,
experiment / figure names, the AF label); how often each rule applied to each file is written to logs/setup.json.

One mode per call, through ../gpu-tools/gpu.py:

  --setup        (CPU) create the topic. Refuses to touch an existing one (then it only verifies the mask).
  --check_ext    (CPU) prove that starting the trainer in the September snapshot compiles nothing.
  --worker       (GPU) run the queue, one job at a time on the card gpu.py --gpu gave. The September job list
                 (retro_orchestrate.py) with its arguments and lambda grids, the same grid-edge extension rule
                 (x1.4 / /1.4, at most two per family and scan). A job starts only while no other compute process
                 is on the card; with --after PATTERN the first job also waits until no process of this user
                 matches PATTERN and the card has then stayed empty for --first_settle_s.
  --table        (CPU) standings from whatever has finished, next to the September AF16 numbers.
  --finalize     (CPU) summarize.py, the gauge cross-check of the converted arms, then the table.
  --script CMD   (CPU) run one ported topic script, e.g. --script "find_best_slice.py".
  --extract ARM  (GPU) convert + score finished MC-GS / GS-single runs again (the step after training).

Internal: --gs_chain ARM GPU (train, then convert + score; called by the worker).
No clock value and no measured duration is printed by this program (the project's time rule).
"""
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

RES = Path("/data0/congcong/code/GS/Results_20260908")
S16 = RES / "CE_retro_cssense_af16"
T = RES / "CE_retro_cssense_af12"
TR = S16 / "trainer/_trainer_e778bf2"
ARC = TR / "results_arc"
ROOT12 = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_af12_20261112")     # make_af12.py's output
PY = "/home/congcongliu/anaconda3/envs/GSTMI/bin/python"
NX, NY, NZ, ACS, TARGET = 256, 232, 176, 24, 3403
SCANS = ("pre", "post")
ART = Path(os.environ.get("MCGS_ARTIFACT_DIR", "."))
log = lambda *a: print(*a, flush=True)

# textual port of the September per-topic scripts
PORT = [('Results_20260908/CE_retro_cssense_af16"', 'Results_20260908/CE_retro_cssense_af12"'),   # topic default, not the trainer path
        ("mask_af16", "mask_af12"),
        ("ce_retro_af16", "ce_retro_af12"),
        ("t1flair_v3_af16_", "t1flair_v3_af12_"),
        ('"--af", "16"', '"--af", "12"'),
        ("nominal AF 16, actual", "nominal AF 12, actual"),
        ("Poisson-disc AF16)", "Poisson-disc AF12)")]
SCRIPTS = ["t1t2_common.py", "arms.py", "recon_classical.py", "recon_joint_cssense.py", "recon_spirit.py", "recon_zsssdu.py",
           "recon_dip_inr.py", "run_mcinr.py", "extract_gs_volumes.py", "summarize.py", "score_volume.py", "readme_tables.py",
           "enhancement_fidelity.py", "find_best_slice.py", "best_slice_mcgs.py", "fig_three_planes_compact.py",
           "fig_enhancement.py"]
VERBATIM = ["diag_artefact/__init__.py", "diag_artefact/headonly.py", "diag_artefact/volumes.py", "diag_artefact/SOURCE.txt"]

# scripts/run_gs.sh's argument list (the frozen jyz AF16 phase-3 recipe), checked against the September runs' recorded argv
P3 = ["--iters", "4000", "--lambda_geo", "0.10", "--warmup", "500", "--max_radius", "20", "--kspace_high", "2.2",
      "--edge_tv", "--edge_tv_kappa_q", "0.85", "--edge_tv_wfloor", "0.10", "--tv_eps", "0.04",
      "--set", "gaussian.init_mode=head", "--set", "gaussian.init_scale_mode=3nn",
      "--set", "loss.edge_tv_kappa_source=zf", "--set", "gaussian.initial_num_points=3600000",
      "--set", "loss.wav_name=db4", "--set", "loss.wav_levels=1",
      "--set", "loss.lambda_wav_plain=0.12", "--set", "loss.lambda_wav=0.10",
      "--lambda_edge", "0.001", "--lambda_ptv", "0.03"]
GS = {"joint": dict(cfg="ce_retro_af12.yaml", key="gs_joint", map="pre:plain,post:contrast", done="gs_joint_post"),
      "single_pre": dict(cfg="ce_retro_af12_single_pre.yaml", key="gs_single", map="pre:plain", done="gs_single_pre"),
      "single_post": dict(cfg="ce_retro_af12_single_post.yaml", key="gs_single", map="post:plain", done="gs_single_post")}
# retro_orchestrate.py's grids
GRID = {"cs": [2.5e-4, 3.5e-4, 5e-4, 7e-4, 1e-3, 1.4e-3, 2e-3], "joint": [5e-4, 7e-4, 1e-3, 1.4e-3, 2e-3],
        "spirit_lam": [0.0, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3], "spirit_ks": [3, 5]}
MAX_EXT = 2
lam_s = lambda v: f"{v:g}"
# The September build.ninja names /usr/local/cuda-12.6/bin/nvcc; with that directory first on PATH torch writes the
# same file again and ninja has nothing to do (--check_ext proves it on a private copy of the build directory).
CUDA_BIN = "/usr/local/cuda-12.6/bin"
ENV = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID",
           RETRO_TOPIC=str(T), RETRO_TRAINER=str(TR), PATH=CUDA_BIN + ":" + os.environ.get("PATH", ""))
EXT_BUILD = TR / "paired_gs/gaussian/cuda_ext/build"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def port(text):
    counts = []
    for a, b in PORT:
        counts.append(text.count(a))
        text = text.replace(a, b)
    return text, counts


# ---------------------------------------------------------------------------------------------- setup
def mask_report():
    import h5py
    import numpy as np
    info = json.load(open(T / "sim/mask/mask_af12.json"))
    assert sha(T / "sim/mask/mask_af12_pe.npy") == info["mask_pe_sha256"], "mask_af12_pe.npy is not make_af12.py's mask"
    assert sha(T / "sim/mask/mask_af12_3d.h5") == info["mask_3d_sha256"], "mask_af12_3d.h5 is not make_af12.py's mask"
    m = np.load(T / "sim/mask/mask_af12_pe.npy") > 0
    assert m.shape == (NY, NZ) and int(m.sum()) == TARGET
    assert m[NY // 2 - ACS // 2:NY // 2 + ACS // 2, NZ // 2 - ACS // 2:NZ // 2 + ACS // 2].all()
    with h5py.File(T / "sim/mask/mask_af12_3d.h5", "r") as f:
        assert f["mask"].shape == (NX, NY, NZ) and (f["mask"][0] == m.astype(np.uint8)).all()
    with np.load(T / "sim/gt.npz") as g:
        assert (np.asarray(g["mask2d"]) == m).all(), "sim/gt.npz carries another mask"
    return dict(points=TARGET, plane=NY * NZ, af=NY * NZ / TARGET, seed=info["seed"], density=info["acc"],
                trimmed=info["trimmed"], pe_sha256=info["mask_pe_sha256"], h5_sha256=info["mask_3d_sha256"])


def setup():
    import numpy as np
    import yaml
    if T.exists():
        log(f"{T} exists: nothing is written, verifying only")
        log(json.dumps(mask_report(), indent=1))
        return
    src_scripts = (S16 / "scripts").resolve()
    for n in SCRIPTS + VERBATIM:
        assert (src_scripts / n).is_file(), f"{src_scripts / n} is missing"
    for arm in GS:
        e = f"ce_retro_af12_{arm}"
        for p in (ARC / f"{e}.log", ARC / f"{e}_score.json", ARC / e, ARC / f"{e}_volumes.npz"):
            assert not p.exists(), f"{p} exists"
    for d in ("sim/gs", "sim/mask", "configs", "scripts/diag_artefact", "logs/claims", "logs/run", "recon", "figures"):
        (T / d).mkdir(parents=True)
    # 1. data: links to the September k-space and maps (unmasked, noisy), nothing copied
    for s in SCANS:
        for kind in ("kspace", "csm"):
            src = (S16 / f"sim/gs/{s}_{kind}.h5").resolve()
            assert src.is_file(), src
            (T / f"sim/gs/{s}_{kind}.h5").symlink_to(src)
    # 2. mask: byte copies of make_af12.py's files
    for n in ("mask_af12_pe.npy", "mask_af12_3d.h5", "mask_af12.json"):
        shutil.copyfile(ROOT12 / "sim/mask" / n, T / "sim/mask" / n)
    m = np.load(T / "sim/mask/mask_af12_pe.npy") > 0
    # 3. truth: the September gt.npz with its mask2d entry replaced (truth and scales byte for byte)
    with np.load(S16 / "sim/gt.npz") as g:
        keys = list(g.keys())
        assert set(keys) == {"mask2d", "pre", "post", "scale_pre", "scale_post"}, keys
        old = {k: np.asarray(g[k]) for k in keys}
    assert old["mask2d"].shape == m.shape and int(old["mask2d"].sum()) == NY * NZ // 16
    new = dict(old, mask2d=m.astype(old["mask2d"].dtype))
    np.savez(T / "sim/gt.npz", **new)
    with np.load(T / "sim/gt.npz") as g:
        for k in ("pre", "post", "scale_pre", "scale_post"):
            assert np.array_equal(np.asarray(g[k]), old[k]) and np.asarray(g[k]).dtype == old[k].dtype, k
    meta = json.load(open(S16 / "sim/meta.json"))
    assert meta["pe_points"] == NY * NZ // 16
    meta.update(pe_points=TARGET, af=NY * NZ / TARGET,
                note="AF12 rerun (run_af12.py --setup, 2026-11-12): the September meta.json with pe_points / af replaced; "
                     "k-space, maps, truth, noise are the September files (sim/gs links), only the mask differs.")
    json.dump(meta, open(T / "sim/meta.json", "w"), indent=1)
    af = NY * NZ / TARGET
    json.dump(dict(mask2d_shape=[NY, NZ], mask2d_samples=TARGET, mask2d_AF_rect=af,
                   contrasts={s: dict(AF_vs_own_support=af, AF_vs_full_rect=af) for s in SCANS}),
              open(T / "logs/prep_04_export.json", "w"), indent=1)
    # 4. scripts
    ported = {}
    for n in SCRIPTS:
        text, counts = port((src_scripts / n).read_text())
        assert "mask_af16" not in text and "ce_retro_af16" not in text
        (T / "scripts" / n).write_text(text)
        shutil.copymode(src_scripts / n, T / "scripts" / n)
        ported[n] = dict(source_sha256=sha(src_scripts / n), rule_counts=counts,
                         af16_topic_mentions_left=text.count("CE_retro_cssense_af16"))
    for n in VERBATIM:
        shutil.copyfile(src_scripts / n, T / "scripts" / n)
    common = (T / "scripts/t1t2_common.py").read_text()
    assert f'"{T}"' in common and f'"{TR}"' in common and 'sim/mask/mask_af12_3d.h5' in common
    assert ported["run_mcinr.py"]["rule_counts"][4] == 1 and ported["t1t2_common.py"]["rule_counts"][0] == 1
    # 5. trainer configs: the September three with mask path, declared factor and names replaced, nothing else
    old_mask, new_mask = str(S16 / "sim/mask/mask_af16_3d.h5"), str(T / "sim/mask/mask_af12_3d.h5")
    for arm, g in GS.items():
        src = S16 / "configs" / g["cfg"].replace("af12", "af16")
        text = src.read_text()
        assert text.count(old_mask) == 2 and text.count("acceleration_factor: 16") == 2, src
        body = text.replace(old_mask, new_mask).replace("acceleration_factor: 16", "acceleration_factor: 12").replace(
            "ce_retro_af16", "ce_retro_af12")
        head = (f"# {g['cfg']} -- AF12 rerun of the plain-scan / enhanced study (run_af12.py --setup, 2026-11-12).\n"
                f"# The September configs/{src.name} with the mask path, the declared factor and the names replaced;\n"
                "# k-space and maps are still read from the September topic. Mask: 3403 of 40832 points, AF 11.999.\n")
        (T / "configs" / g["cfg"]).write_text(head + body)
        a, b = yaml.safe_load(text), yaml.safe_load(head + body)
        for slot in ("plain", "contrast"):
            assert b["data"][slot]["mask_path"] == new_mask and b["data"][slot]["acceleration_factor"] == 12
            for k in ("mask_path", "acceleration_factor"):
                a["data"][slot].pop(k), b["data"][slot].pop(k)
        assert json.loads(json.dumps(a).replace("ce_retro_af16", "ce_retro_af12")) == b, f"{g['cfg']} differs beyond mask, factor, names"
    rep = mask_report()
    json.dump(dict(port_rules=PORT, scripts=ported, verbatim=VERBATIM, mask=rep, job=os.environ.get("MCGS_JOB_ID"),
                   source_scripts=str(src_scripts)), open(T / "logs/setup.json", "w"), indent=1)
    (T / "README.md").write_text(
        "# CE Retro CS-SENSE AF12\n\nThe September study `CE_retro_cssense_af16` repeated with an AF12 mask (3403 of 40832 "
        "phase-encode points, same generator, seed 0, ACS 24 x 24). Created by `run_af12.py --setup` "
        "(repository `autoresearch/ce_af12_rerun_20261112/`); reconstructions pending. Everything except the mask is the "
        "September setting; see `logs/setup.json` for the script port.\n\n"
        "MC-GS and GS-single run in the September trainer snapshot; their raw outputs are "
        "`../CE_retro_cssense_af16/trainer/_trainer_e778bf2/results_arc/ce_retro_af12_*`.\n")
    for n, d in ported.items():
        log(f"  {n:30s} rules applied {d['rule_counts']}  AF16-topic mentions left {d['af16_topic_mentions_left']}")
    log(f"created {T}")
    log(json.dumps(rep, indent=1))
    shutil.copy2(T / "logs/setup.json", ART / "setup.json")
    log("SETUP OK")


def check_ext():
    """Would a trainer start recompile the September voxelizer? Tried on a private copy of its build directory: the
    trainer's own loader runs with build_directory redirected there (the sources stay the snapshot's)."""
    import tempfile
    snap = lambda d: {q.name: hashlib.md5(q.read_bytes()).hexdigest() for q in sorted(d.iterdir()) if q.is_file() and q.name != "lock"}
    tmp = Path(tempfile.mkdtemp(prefix="extcheck_"))
    subprocess.run(["cp", "-a", str(EXT_BUILD), str(tmp / "build")], check=True)
    real_before, before = snap(EXT_BUILD), snap(tmp / "build")
    assert real_before == before and any(k.endswith(".so") for k in before), "no compiled voxelizer in the snapshot"
    code = ("import sys, torch.utils.cpp_extension as ce\n"
            "orig = ce.load\n"
            "def load(*a, **k):\n"
            f"    k['build_directory'] = {str(tmp / 'build')!r}; k['verbose'] = True\n"
            "    return orig(*a, **k)\n"
            "ce.load = load\n"
            f"sys.path.insert(0, {str(TR)!r}); sys.path.insert(0, {str(TR / 'paired_gs')!r})\n"
            "from gaussian import tile_voxelizer as tv\n"
            "tv._load_cuda_ext()\n")
    env = dict(ENV)
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit():
        # without a visible card torch cannot derive the architecture flags and stops before ninja; give it the flags
        # an A100 yields (the September build.ninja has compute_80 + sm_80), so that the CPU check reaches ninja
        env["TORCH_CUDA_ARCH_LIST"] = "8.0+PTX"
        log("[check_ext] no card visible: TORCH_CUDA_ARCH_LIST=8.0+PTX for this check")
    r = subprocess.run([PY, "-c", code], cwd=TR, env=env, capture_output=True, text=True)
    after, real_after = snap(tmp / "build"), snap(EXT_BUILD)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    shutil.rmtree(tmp, ignore_errors=True)
    assert real_before == real_after, "the snapshot's build directory changed during the check"
    binaries = [k for k in changed if k.endswith((".so", ".o"))]
    reached = "ninja: no work to do" in (r.stdout + r.stderr)
    ok = reached and not binaries
    log(f"[check_ext] files changed in the private copy: {changed or 'none'}; compiled files changed: {binaries or 'none'}; "
        f"ninja said 'no work to do': {reached} -> a trainer start "
        f"{'compiles nothing' if ok else ('WOULD RECOMPILE the extension' if binaries else 'NOT PROVEN (ninja was not reached)')}")
    if not ok:
        log((r.stdout + r.stderr)[-2500:])
    return ok


# ---------------------------------------------------------------------------------------------- queue
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
    """the September runs' recorded argv must be P3, argument for argument"""
    for arm in GS:
        s = json.load(open(ARC / f"ce_retro_af16_{arm}_score.json"))
        assert strip(s["argv"]) == P3 + ["--save_volumes"], f"recipe differs from the September {arm} run:\n{s['argv']}"


def gs_chain(arm, gpu):
    """MC-GS / GS-single: train in the September snapshot, then convert to the topic's gauge and score."""
    g, exp = GS[arm], f"ce_retro_af12_{arm}"
    outs = [ARC / f"{exp}.log", ARC / f"{exp}_score.json", ARC / exp, ARC / f"{exp}_volumes.npz"]
    if not (outs[1].exists() and outs[3].exists()):
        left = [str(p) for p in outs if p.exists()]
        if left:
            log(f"REFUSED: outputs of {exp} exist without a finished run: {left}")
            sys.exit(4)
        cmd = [PY, "-u", "arc/cycle.py", "--exp", exp, "--gpu", str(gpu), *P3, "--base_cfg", str(T / "configs" / g["cfg"]),
               "--save_volumes"]
        log(" ".join(cmd))
        with open(outs[0], "w") as lf:
            rc = subprocess.run(cmd, cwd=TR, env=ENV, stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc or not (outs[1].exists() and outs[3].exists()):
            log(f"training failed rc={rc}; log tail:\n" + "".join(open(outs[0]).read().splitlines(True)[-25:]))
            sys.exit(rc or 5)
        s = json.load(open(outs[1]))
        assert strip(s["argv"]) == P3 + ["--save_volumes"] and s["argv"][s["argv"].index("--base_cfg") + 1] == str(T / "configs" / g["cfg"])
    sys.exit(extract(arm))


def extract(arm):
    """Convert a finished run to the topic's gauge and score it. extract_gs_volumes.py addresses the card as
    cuda:<--gpu> without restricting the visible cards itself; here exactly one card is visible (gpu.py --gpu), so
    the index it must be given is 0. (2026-11-12: the first three runs trained, then failed here with the physical
    index 7, 'invalid device ordinal'; the trainings were kept and only this step was repeated.)"""
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit(), "needs exactly one card from gpu.py --gpu"
    g, exp = GS[arm], f"ce_retro_af12_{arm}"
    cmd = [PY, "-u", str(T / "scripts/extract_gs_volumes.py"), "--exp", exp, "--key", g["key"], "--map", g["map"], "--gpu", "0"]
    log(" ".join(cmd))
    return subprocess.run(cmd, cwd=T, env=ENV).returncode


def build_queue():
    L, S, q = T / "logs", T / "scripts", []

    def add(name, argv, done):
        q.append(dict(name=name, argv=[str(a) for a in argv], done=str(done), log=str(L / "run" / f"{name}.log")))

    def cs(s, lam):
        add(f"cs_{s}_lam{lam_s(lam)}", [PY, "-u", S / "recon_classical.py", "--scan", s, "--methods", "cssense", "--admm_outer", "200",
                                        "--lam", lam_s(lam), "--suffix", f"_o200_lam{lam_s(lam)}", "--gpu", "{gpu}"],
            L / f"classical_{s}_o200_lam{lam_s(lam)}.json")

    def joint(lam):
        add(f"joint_lam{lam_s(lam)}", [PY, "-u", S / "recon_joint_cssense.py", "--topic", T, "--contrasts", "pre,post", "--admm_outer",
                                       "200", "--lam", lam_s(lam), "--suffix", f"_o200_lam{lam_s(lam)}", "--gpu", "{gpu}"],
            L / f"jcssense_o200_lam{lam_s(lam)}.json")

    for s in SCANS:                                 # cheap, and the first proof that mask, data and gauge load
        add(f"cls_{s}", [PY, "-u", S / "recon_classical.py", "--scan", s, "--methods", "zf,cgsense", "--gpu", "{gpu}"],
            L / f"classical_{s}.json")
    for arm, g in GS.items():
        add(f"gs_{arm}", [PY, "-u", Path(__file__).resolve(), "--gs_chain", arm, "{gpu}"], L / f"{g['done']}.json")
    for s in SCANS:
        add(f"zs_{s}", [PY, "-u", S / "recon_zsssdu.py", "--scan", s, "--gpu", "{gpu}", "--epochs", "100"], L / f"zsssdu_{s}.json")
    for s in SCANS:
        for lam in GRID["cs"]:
            cs(s, lam)
    for lam in GRID["joint"]:
        joint(lam)
    for s in SCANS:
        for ks in GRID["spirit_ks"]:
            for lam in GRID["spirit_lam"]:
                sfx = f"_ks{ks}_lam{lam_s(lam)}"
                add(f"spirit_{s}{sfx}", [PY, "-u", S / "recon_spirit.py", "--scan", s, "--gpu", "{gpu}", "--ksize", ks, "--lam",
                                         lam_s(lam), "--suffix", sfx], L / f"spirit_{s}{sfx}.json")
    add("mcinr", [PY, "-u", S / "run_mcinr.py", "--gpu", "{gpu}"], L / "mcinr_post.json")
    for s in SCANS:
        add(f"inr_{s}", [PY, "-u", S / "recon_dip_inr.py", "--method", "inr", "--scan", s, "--gpu", "{gpu}", "--iters", "1500",
                         "--eval_every", "50"], L / f"inr_{s}.json")
    for s in SCANS:
        add(f"dip_{s}", [PY, "-u", S / "recon_dip_inr.py", "--method", "dip", "--scan", s, "--gpu", "{gpu}", "--iters", "5000",
                         "--lr", "3e-3", "--eval_every", "100"], L / f"dip_{s}.json")
    assert len({j["name"] for j in q}) == len(q)
    return q, cs, joint


def family(fam, scan):
    """(best lambda, lambdas seen) of a swept family on one scan, from the finished jsons"""
    pat = f"classical_{scan}_o200_lam*.json" if fam == "cs" else "jcssense_o200_lam*.json"
    cands = [(float(r["lam"]), float(r["head_psnr"])) for f in sorted((T / "logs").glob(pat))
             for r in json.load(open(f)).get("rows", []) if r.get("contrast", scan) == scan and r.get("head_psnr") is not None]
    if not cands:
        return None, []
    return max(cands, key=lambda t: t[1])[0], sorted({l for l, _ in cands})


def plan_extensions(seen):
    """retro_orchestrate.py's rule: when a family's best lambda sits on its grid edge, one more step outwards"""
    new = []
    for fam in ("cs", "joint"):
        base = set(GRID[fam])
        for scan in SCANS:
            best, lams = family(fam, scan)
            if best is None or len(set(lams) - base) >= MAX_EXT * (2 if fam == "joint" else 1):
                continue
            if best == max(lams):
                lam = float(f"{best * 1.4:.2g}")
            elif best == min(lams):
                lam = float(f"{best / 1.4:.2g}")
            else:
                continue
            name = f"cs_{scan}_lam{lam_s(lam)}" if fam == "cs" else f"joint_lam{lam_s(lam)}"
            if name in seen:
                continue
            seen.add(name)
            new.append((fam, scan, lam))
    return new


# ---------------------------------------------------------------------------------------------- worker
def card_pids(card):
    bus = {}
    for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
                                        text=True, timeout=60).strip().splitlines():
        i, b = [x.strip() for x in line.split(",")]
        bus[b] = int(i)
    pids = []
    for line in subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id", "--format=csv,noheader"],
                                        text=True, timeout=60).strip().splitlines():
        if line.strip():
            pid, b = [x.strip() for x in line.split(",")]
            if bus.get(b) == card:
                pids.append(int(pid))
    return pids


def matching(pattern):
    out = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", pattern], capture_output=True, text=True).stdout.split()
    return [int(p) for p in out if int(p) != os.getpid()]


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
    json.dump(dict(pid=os.getpid(), job=os.environ.get("MCGS_JOB_ID")), open(d / "owner.json", "w"))
    return True


def result_line(job):
    try:
        d = json.load(open(job["done"]))
    except (OSError, ValueError):
        return ""
    return "; ".join(f"{r.get('contrast', d.get('contrast', ''))} {r['method']} head {r['head_psnr']:.3f} / {r['head_ssim']:.4f}"
                     for r in d.get("rows", []) if r.get("head_psnr") is not None)


def worker(a):
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert vis.isdigit(), f"needs exactly one card from gpu.py --gpu, got CUDA_VISIBLE_DEVICES={vis!r}"
    card = int(vis)
    assert T.is_dir(), "run --setup first"
    rep = mask_report()
    check_recipe()
    queue = build_queue()[0]
    ext_ok = None                                   # checked right before the first MC-GS / GS-single job, when the card is ours
    if a.only:
        queue = [j for j in queue if any(j["name"].startswith(p) for p in a.only.split(","))]
    seen = {j["name"] for j in queue}
    lf = open(T / "logs/run_af12.log", "a")

    def say(msg):
        log(msg)
        lf.write(msg + "\n")
        lf.flush()

    say(f"worker pid {os.getpid()} job {os.environ.get('MCGS_JOB_ID')}: {len(queue)} jobs, card {card}; mask {rep['points']} points "
        f"(AF {rep['af']:.3f}, sha256 {rep['pe_sha256'][:12]}); recipe = the September argv"
        + (f"; first job waits for {a.after!r} to be gone and {a.first_settle_s} s of empty card" if a.after else ""))
    running, results, free_since, held, gate_open, told = None, {}, None, False, not a.after, None
    while True:
        if running and running["proc"].poll() is not None:
            j, rc = running["job"], running["proc"].returncode
            ok = Path(j["done"]).exists()
            results[j["name"]] = rc if not (rc == 0 and not ok) else "no result file"
            say(f"DONE {j['name']} rc={rc} result={'yes' if ok else 'NO'}  {result_line(j) if ok else ''}")
            if not ok:
                say(f"FAILED {j['name']}: not retried; log {j['log']}")
            running = None
        for j in list(queue):
            if Path(j["done"]).exists():
                queue.remove(j)
                results[j["name"]] = "skip"
                say(f"SKIP {j['name']} (result exists)")
        if not queue and not running:
            new = [] if a.only else plan_extensions(seen)
            if not new:
                break
            q2, cs2, joint2 = build_queue()                    # the job builders, bound to a list that is emptied first
            del q2[:]
            for fam, scan, lam in new:
                (cs2(scan, lam) if fam == "cs" else joint2(lam))
            queue.extend(q2)
            say(f"grid edge: added {[j['name'] for j in q2]}")
            continue
        if not running and queue:
            try:
                pids = card_pids(card)
            except Exception as e:                  # noqa: BLE001
                say(f"nvidia-smi failed: {e}")
                time.sleep(a.poll_s)
                continue
            blockers = matching(a.after) if (a.after and not gate_open) else []
            now = time.time()
            if pids or blockers:
                free_since, held = None, False
                state = f"card {card} has {len(pids)} process(es)" + (f", {len(blockers)} process(es) match {a.after!r}" if blockers else "")
                if told != state:
                    told = state
                    say(f"waiting: {state}")
            else:
                if free_since is None:
                    free_since = now
                need = 0 if held else (a.first_settle_s if not gate_open else a.settle_s)
                if now - free_since >= need:
                    j = next((x for x in queue if claim(x["name"])), None)
                    if j is None:
                        queue.clear()
                        continue
                    queue.remove(j)
                    if j["name"].startswith("gs_"):
                        if ext_ok is None:
                            ext_ok = check_ext()
                        if not ext_ok:              # the other arms do not use the voxelizer
                            results[j["name"]] = "refused"
                            say(f"REFUSED {j['name']}: a trainer start would recompile the September voxelizer")
                            continue
                    Path(j["log"]).parent.mkdir(parents=True, exist_ok=True)
                    p = subprocess.Popen([x.replace("{gpu}", str(card)) for x in j["argv"]], cwd=T, env=ENV,
                                         stdout=open(j["log"], "a"), stderr=subprocess.STDOUT)
                    running, held, gate_open, told = dict(job=j, proc=p), True, True, None
                    say(f"START {j['name']} pid {p.pid} -> {j['log']}")
        json.dump(dict(running=running and running["job"]["name"], queued=[j["name"] for j in queue], done=results,
                       gate_open=gate_open), open(T / "logs/run_af12_status.json", "w"), indent=1)
        time.sleep(a.poll_s)
    bad = {n: rc for n, rc in results.items() if rc not in (0, "skip")}
    say(f"QUEUE EMPTY. failures: {bad or 'none'}")
    table()
    sys.exit(1 if bad else 0)


# ---------------------------------------------------------------------------------------------- standings
ORDER = ["zf", "cgsense", "cssense_o200", "jcssense", "spirit", "zsssdu", "dip", "inr", "mcinr", "gs_single", "gs_joint"]
LABEL = {"zf": "ZF", "cgsense": "CG-SENSE", "cssense_o200": "CS-SENSE", "jcssense": "Joint SENSE", "spirit": "L1-SPIRiT",
         "zsssdu": "ZS-SSDU", "dip": "3D-DIP", "inr": "3D-INR", "mcinr": "MC-INR", "gs_single": "GS-single", "gs_joint": "MC-GS"}


def collect():
    """summarize.py's sources and its rule (a swept arm = its best lambda / kernel by head PSNR)"""
    rows, L = [], T / "logs"
    for c in SCANS:
        for stem in ("classical", "zsssdu", "dip", "inr", "mcinr", "gs_single", "gs_joint"):
            p = L / f"{stem}_{c}.json"
            if p.exists():
                rows += [dict(r, contrast=c) for r in json.load(open(p)).get("rows", []) if r.get("contrast", c) == c]
        for key, pat in (("cssense_o200", f"classical_{c}_o200_lam*.json"), ("jcssense", "jcssense_o200_lam*.json"),
                         ("spirit", f"spirit_{c}_ks*_lam*.json")):
            cands = [r for f in sorted(L.glob(pat)) for r in json.load(open(f)).get("rows", [])
                     if r.get("contrast", c) == c and r.get("head_psnr") is not None]
            if cands:
                b = max(cands, key=lambda r: r["head_psnr"])
                lams = sorted({float(r["lam"]) for r in cands if r.get("lam") is not None})
                rows.append(dict(b, contrast=c, method=key, lams=lams, edge=float(b.get("lam", -1)) in (lams[0], lams[-1])))
    return [r for r in rows if r["method"] in ORDER]


def table():
    rows, out = collect(), []
    ref = {(r["contrast"], r["method"]): r for r in json.load(open(S16 / "logs/summary.json"))}
    for c in SCANS:
        sub = sorted((r for r in rows if r["contrast"] == c), key=lambda r: ORDER.index(r["method"]))
        out += [f"\n### {c} (AF12 mask, 3403 points; AF16 = the September study)\n",
                "| method | head PSNR AF12 | AF16 | AF12 - AF16 | head SSIM AF12 | AF16 | head NMSE AF12 | head HFEN AF12 | note |",
                "|---|---|---|---|---|---|---|---|---|"]
        for r in sub:
            o = ref.get((c, r["method"]))
            note = ""
            if r.get("lams"):
                note = f"lam {float(r['lam']):g}" + (f", kernel {r['ksize']}" if r.get("ksize") else "") + (" (GRID EDGE)" if r.get("edge") else "")
                if o and o.get("lam") is not None:
                    note += f"; AF16 lam {float(o['lam']):g}"
            out.append(f"| {LABEL[r['method']]} | {r['head_psnr']:.3f} | " + (f"{o['head_psnr']:.3f} | {r['head_psnr'] - o['head_psnr']:+.3f}" if o else " | ")
                       + f" | {r['head_ssim']:.4f} | " + (f"{o['head_ssim']:.4f}" if o else "") + f" | {r['head_nmse']:.5f} | {r['head_hfen']:.4f} | {note} |")
        mc = next((r for r in sub if r["method"] == "gs_joint"), None)
        others = [r for r in sub if r["method"] != "gs_joint"]
        if mc and others:
            for k, lower in (("head_psnr", False), ("head_ssim", False), ("head_nmse", True), ("head_hfen", True)):
                b = (min if lower else max)(others, key=lambda r: r[k])
                d = (b[k] - mc[k]) if lower else (mc[k] - b[k])
                out.append(f"\nMC-GS {k}: {mc[k]:.4f}; best other {LABEL[b['method']]} {b[k]:.4f}; margin {d:+.4f} ({'first' if d > 0 else 'NOT first'})")
            gsr = next((r for r in sub if r["method"] == "gs_single"), None)
            if gsr:
                out.append(f"\nMC-GS minus GS-single, head PSNR: {mc['head_psnr'] - gsr['head_psnr']:+.3f} dB")
    missing = [f"{c}:{LABEL[m]}" for c in SCANS for m in ORDER if not any(r["contrast"] == c and r["method"] == m for r in rows)]
    out.append(f"\narms without a result yet: {missing or 'none'}")
    text = "\n".join(out)
    log(text)
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "table_af12.md").write_text(text + "\n")
    st = T / "logs/run_af12_status.json"
    if st.exists():
        shutil.copy2(st, ART / "run_af12_status.json")
    return text


def run_script(cmd, tail=4000):
    tok = shlex.split(cmd)
    assert (T / "scripts" / tok[0]).is_file(), f"{tok[0]} is not a ported topic script"
    r = subprocess.run([PY, "-u", str(T / "scripts" / tok[0]), *tok[1:]], cwd=T, env=dict(ENV, CUDA_VISIBLE_DEVICES=""),
                       capture_output=True, text=True)
    log(f"$ {cmd}  -> rc {r.returncode}\n{(r.stdout + r.stderr)[-tail:]}")
    return r.returncode


def finalize():
    rc = run_script("summarize.py")
    bad = []
    for s in SCANS:
        for k in ("cgsense", "zsssdu", "dip", "inr", "mcinr", "gs_single", "gs_joint"):
            jf = f"logs/classical_{s}.json" if k == "cgsense" else f"logs/{k}_{s}.json"
            r = subprocess.run([PY, "-u", str(T / "scripts/score_volume.py"), "--scan", s, "--file", f"recon/{k}_{s}_complex.npy",
                                "--method", k, "--check", jf, "--gpu", "-1"], cwd=T, env=dict(ENV, CUDA_VISIBLE_DEVICES=""),
                               capture_output=True, text=True)
            last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
            log(f"gauge {s} {k}: {last}")
            if last != "GAUGE OK":
                bad.append(f"{s}:{k}")
    with open(T / "logs/gauge_check.txt", "w") as f:
        f.write(f"gauge cross-check (score_volume.py --check, CPU, double precision): failures {bad or 'none'}\n")
    table()
    log(f"FINALIZE {'OK' if not (rc or bad) else 'FAILED'}: summarize rc {rc}, gauge failures {bad or 'none'}")
    sys.exit(1 if (rc or bad) else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--setup", action="store_true")
    g.add_argument("--check_ext", action="store_true")
    g.add_argument("--worker", action="store_true")
    g.add_argument("--table", action="store_true")
    g.add_argument("--finalize", action="store_true")
    g.add_argument("--script")
    g.add_argument("--gs_chain", nargs=2, metavar=("ARM", "GPU"))
    g.add_argument("--extract", nargs="+", choices=list(GS), help="(GPU) convert + score finished MC-GS / GS-single runs")
    ap.add_argument("--only", default="", help="--worker: comma-separated job-name prefixes (no grid extensions then)")
    ap.add_argument("--after", default="", help="--worker: the first job waits until no process of this user matches this pattern")
    ap.add_argument("--first_settle_s", type=int, default=600, help="with --after: the card must then stay empty this long")
    ap.add_argument("--settle_s", type=int, default=75, help="a card another process used must be empty this long")
    ap.add_argument("--poll_s", type=int, default=15)
    a = ap.parse_args()
    if a.setup:
        setup()
    elif a.check_ext:
        sys.exit(0 if check_ext() else 1)
    elif a.worker:
        worker(a)
    elif a.table:
        table()
    elif a.finalize:
        finalize()
    elif a.script:
        sys.exit(run_script(a.script))
    elif a.extract:
        rcs = {arm: extract(arm) for arm in a.extract}
        log(f"EXTRACT {rcs}")
        sys.exit(1 if any(rcs.values()) else 0)
    else:
        gs_chain(a.gs_chain[0], int(a.gs_chain[1]))
