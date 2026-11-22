#!/usr/bin/env python3
"""net20.py -- the jiangyizhou T1 + FLAIR study at a NET acceleration of exactly 20x (user, 2026-11-21).

Net acceleration (user's definition) = points of the phase-encode plane / sampled points, calibration region included:
45056 / 2253 = 19.998. The existing "AF20" topic is nominal 20 = net 18.56x, so a new mask is drawn and every method
is run again on it. Everything else is the T1FLAIR_AF24_v3_jiangyizhou topic's procedure, unchanged:
  - mask: the house vdPoisMex generator, only accelx differs; a pool of draws, the lowest-index draw with exactly
    2253 points is taken (rule of prep_03_pick_mask_af24.py, fixed before the pool is drawn);
  - k-space and ESPIRiT maps: symlinks into the AF12 topic (mask-independent);
  - arms and arguments: run_all_af24.py (lambda grids of the AF24 topic); L1-ESPIRiT is not run (dropped from the
    comparison by the user on 2026-09-15); MC-INR and the L1-SPIRiT sweep as in its "extra" study;
  - MC-GS / GS-single: the AF16 phase-3 recipe, frozen (as at nominal AF20 and AF24), registered-style single
    (the joint trainer fed one scan in both channels). Trainer = a verified copy of mcgs_sg2x2_wt at 54f8a2a, the
    code base in which the AF12 joint run was reproduced to 0.001 dB on 2026-11-21 (mcgs_af16_wt has since moved
    to another branch).
Nothing outside the new topic J and OUT is written; no existing file is overwritten; one seed (42); neither GT nor
any ROI enters a reconstruction (lambda of the classical arms is chosen against GT on this subject, as in every
topic of the series).

  gpu.py run --cpu   -- @python autoresearch/t1flair_net20_20261121/net20.py --stage setup
  gpu.py run --gpu N -- @python autoresearch/t1flair_net20_20261121/net20.py --stage run --jobs gs_joint ...
  gpu.py run --gpu N -- @python autoresearch/t1flair_net20_20261121/net20.py --stage finalize
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

GS = Path("/data0/congcong/code/GS")
TOP = GS / "Results_20260908"
A24 = TOP / "T1FLAIR_AF24_v3_jiangyizhou"
A12 = TOP / "T1FLAIR_AF12_v3_jiangyizhou"
J = TOP / "T1FLAIR_AF20net_v3_jiangyizhou"
OUT = GS / "3DGS/outputs/codex_artifacts/mcgs_t1flair_net20_20261121"
SRC = GS / "mcgs_sg2x2_wt"
SRC_HEAD = "54f8a2a3bef402e785314b748f730a9623ca9d43"
CODE_DIRS = ("arc", "paired_gs", "diag_artefact")
TR, RES, CFG = OUT / "trainer", OUT / "trainer/results_arc", OUT / "configs"
MCINR = GS / "mcinr_niessen2025"
PY_TCNN = "/home/congcongliu/anaconda3/envs/GR_pt1.10/bin/python"
PY = sys.executable
TARGET = 2253
MASK3D = J / "data/mask_af20n_3d.h5"
ENV = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID")
log = lambda *a: print(*a, flush=True)

SKIP = {"setup_af24_jyz.sh", "run_all_af24.py", "prep_02b_mask_pool_draws_af24.m", "prep_03b_pick_export_draws_af24.py",
        "score_mask_draws_af24.py", "export_af24_bundle.py", "finalize_af24_jyz.sh", "score_mcinr_arm.py"}
P3 = ["--iters", "4000", "--lambda_geo", "0.10", "--warmup", "500", "--max_radius", "20", "--kspace_high", "2.2",
      "--edge_tv", "--edge_tv_kappa_q", "0.85", "--edge_tv_wfloor", "0.10", "--tv_eps", "0.04",
      "--set", "gaussian.init_mode=head", "--set", "gaussian.init_scale_mode=3nn",
      "--set", "loss.edge_tv_kappa_source=zf", "--set", "gaussian.initial_num_points=3600000",
      "--set", "loss.wav_name=db4", "--set", "loss.wav_levels=1", "--set", "loss.lambda_wav_plain=0.12",
      "--set", "loss.lambda_wav=0.10", "--lambda_edge", "0.001", "--lambda_ptv", "0.03"]
GSRUN = {"gs_joint": ("net20_p3_joint_full4k", "t1flair_v3_af24_jyz.yaml"),
         "gs_singleT1": ("net20_p3_singleT1_full4k", "t1flair_v3_af24_jyz_singleT1.yaml"),
         "gs_singleFLAIR": ("net20_p3_singleFLAIR_full4k", "t1flair_v3_af24_jyz_singleFLAIR.yaml")}
TV = {"T1": ["0.00035", "0.0005", "0.0007", "0.001", "0.0014"], "FLAIR": ["0.00025", "0.00035", "0.0005", "0.0007", "0.001"]}
JTV = ["0.00035", "0.0005", "0.0007", "0.001", "0.0014"]
SPIRIT_KS, SPIRIT_LAM = ("3", "5"), ("0", "0.0005", "0.001", "0.002", "0.004", "0.008")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sub(text):
    return (text.replace("T1FLAIR_AF24_v3_jiangyizhou", J.name).replace("af24", "af20n").replace("AF24", "AF20N")
            .replace("AF 24", "net AF 20"))


# ------------------------------------------------------------------------------------------------ setup
def setup():
    for d in ("scripts", "logs", "figures", "data", "recon"):
        (J / d).mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted((A24 / "scripts").iterdir()):
        if f.suffix not in (".py", ".m", ".sh") or f.name in SKIP:
            continue
        t = J / "scripts" / sub(f.name)
        if t.exists():
            continue
        text = sub(f.read_text())
        if f.name in ("add_gs_arm.py", "summarize.py"):
            a = 'Path("/data0/congcong/code/GS/mcgs_af16_wt/results_arc")'
            assert text.count(a) == 1, f.name
            text = text.replace(a, f'Path("{RES}")')
        if f.name in ("prep_02_mask_pool_af24.m", "prep_03_pick_mask_af24.py"):
            assert "1877" in text
            text = text.replace("1877", str(TARGET)).replace("24.004", "19.998").replace("24x", "20x")
            if f.suffix == ".m":
                a = "repelem(6.40:0.10:7.00, 2)"
                assert text.count(a) == 1
                text = text.replace(a, "repelem(5.10:0.10:5.80, 2)")
        t.write_text(text)
        n += 1
    log(f"cloned {n} scripts into {J / 'scripts'}")
    for f in ("ksp_T1.h5", "ksp_FLAIR.h5", "csm_espirit_T1.h5", "csm_espirit_FLAIR.h5"):
        link = J / "data" / f
        if not link.is_symlink():
            link.symlink_to(A12 / "data" / f)
        assert link.resolve() == (A24 / "data" / f).resolve(), f
    mat = J / "data/mask_vdpois_256x176_af20n.mat"
    if not mat.exists():
        pool = J / "data/mask_pool_af20n"
        if not (pool.exists() and any(pool.glob("draw_*.mat"))):
            with open(J / "logs/prep_02_mask_pool_af20n.log", "w") as lf:
                rc = subprocess.run(["/usr/local/bin/matlab", "-batch", f"run('{J}/scripts/prep_02_mask_pool_af20n.m')"],
                                    stdout=lf, stderr=subprocess.STDOUT).returncode
            tail = [l for l in open(J / "logs/prep_02_mask_pool_af20n.log").read().splitlines() if "stage A fit" in l or "POOL_DONE" in l]
            log("mask pool rc", rc, *tail)
            assert rc == 0
        out = subprocess.run([PY, "scripts/prep_03_pick_mask_af20n.py"], cwd=J, env=ENV, capture_output=True, text=True)
        log(out.stdout[-1500:], out.stderr[-1500:])
        assert out.returncode == 0
    if not MASK3D.exists():
        out = subprocess.run([PY, "scripts/prep_04_export.py"], cwd=J, env=ENV, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-2000:]
    rep = json.load(open(J / "logs/prep_04_export.json"))
    pick = json.load(open(J / "logs/prep_03_pick_mask_af20n.json"))
    log(json.dumps(dict(points=rep["mask2d_samples"], net_AF=rep["mask2d_AF_rect"], how=pick["how"], chosen=pick["chosen"],
                        calib24=pick["calib24_points"], flair_points_on_data=pick["FLAIR"]["sampled_points_with_data"],
                        pool={k: v for k, v in pick["pool"].items() if k != "draws_list"}, mask_md5=pick["mask_mat_md5"]), indent=1))
    assert rep["mask2d_samples"] == TARGET
    CFG.mkdir(parents=True, exist_ok=True)
    for name, (exp, src) in GSRUN.items():
        text = (GS / "mcgs_af16_wt/configs" / src).read_text()
        text = (text.replace("T1FLAIR_AF24_v3_jiangyizhou", J.name).replace("mask_af24_3d.h5", MASK3D.name)
                .replace("acceleration_factor: 24", "acceleration_factor: 20"))
        assert "AF24_v3" not in text and text.count(MASK3D.name) >= 2, name
        p = CFG / f"{name}.yaml"
        if p.exists():
            assert p.read_text() == text
        else:
            p.write_text(text)
    log("configs:", sorted(x.name for x in CFG.iterdir()))
    log("SETUP_DONE")


# ------------------------------------------------------------------------------------------------ jobs
def git(*a):
    return subprocess.run(["git", "-C", str(SRC), *a], check=True, capture_output=True, text=True).stdout


def ensure_trainer():
    assert git("rev-parse", "HEAD").strip() == SRC_HEAD, "source worktree moved"
    assert git("status", "--porcelain", "--", *CODE_DIRS).strip() == "", "source worktree has local changes"
    if not TR.exists():
        tmp = OUT / f"trainer.tmp{os.getpid()}"
        for d in CODE_DIRS:
            shutil.copytree(SRC / d, tmp / d, ignore=shutil.ignore_patterns("__pycache__", "results_arc"))
        try:
            tmp.rename(TR)
        except OSError:
            shutil.rmtree(tmp)
    tracked = [f for f in git("ls-files", "--", *CODE_DIRS).splitlines() if f]
    bad = [f for f in tracked if not (TR / f).is_file() or sha(TR / f) != sha(SRC / f)]
    assert not bad, f"trainer copy differs from the source worktree: {bad[:10]}"
    RES.mkdir(exist_ok=True)


def jobs(g):
    L, q = J / "logs", {}

    def add(name, cwd, cmd, done, logp):
        q[name] = dict(cwd=str(cwd), cmd=[str(x) for x in cmd], done=Path(done), log=Path(logp))

    for name, (exp, _) in GSRUN.items():
        add(name, TR, [PY, "-u", "arc/cycle.py", "--exp", exp, "--gpu", g, *P3, "--base_cfg", CFG / f"{name}.yaml", "--save_volumes"],
            RES / f"{exp}_score.json", RES / f"{exp}.log")
    for c in ("T1", "FLAIR"):
        add(f"base_{c}", J, [PY, "-u", "scripts/recon_classical.py", "--contrast", c, "--gpu", g],
            L / f"classical_{c}.json", L / f"classical_{c}.log")
        for lam in TV[c]:
            add(f"tv_{c}_{lam}", J, [PY, "-u", "scripts/recon_classical.py", "--contrast", c, "--gpu", g, "--methods", "cssense",
                                     "--admm_outer", "200", "--lam", lam, "--suffix", f"_o200_lam{lam}"],
                L / f"classical_{c}_o200_lam{lam}.json", L / f"classical_{c}_o200_lam{lam}.log")
        add(f"zsssdu_{c}", J, [PY, "-u", "scripts/recon_zsssdu.py", "--epochs", "100", "--contrast", c, "--gpu", g],
            L / f"zsssdu_{c}.json", L / f"zsssdu_{c}.log")
        add(f"inr_{c}", J, [PY, "-u", "scripts/recon_dip_inr.py", "--method", "inr", "--iters", "1500", "--eval_every", "50",
                            "--contrast", c, "--gpu", g], L / f"inr_{c}.json", L / f"inr_{c}.log")
        add(f"dip_{c}", J, [PY, "-u", "scripts/recon_dip_inr.py", "--method", "dip", "--iters", "5000", "--lr", "3e-3",
                            "--eval_every", "100", "--contrast", c, "--gpu", g], L / f"dip_{c}.json", L / f"dip_{c}.log")
        for ks in SPIRIT_KS:
            for lam in SPIRIT_LAM:
                sfx = f"_ks{ks}_lam{lam}"
                add(f"spirit_{c}{sfx}", J, [PY, "-u", "scripts/recon_spirit.py", "--contrast", c, "--gpu", g, "--ksize", ks,
                                            "--lam", lam, "--iters", "150", "--chunk", "24", "--suffix", sfx],
                    L / f"spirit_{c}{sfx}.json", L / f"spirit/{c}{sfx}.log")
    for lam in JTV:
        add(f"jtv_{lam}", J, [PY, "-u", "scripts/recon_joint_cssense.py", "--topic", J, "--contrasts", "T1,FLAIR", "--gpu", g,
                              "--admm_outer", "200", "--lam", lam, "--suffix", f"_o200_lam{lam}"],
            L / f"jcssense_o200_lam{lam}.json", L / f"jcssense_o200_lam{lam}.log")
    add("mcinr", MCINR, [PY_TCNN, "-u", "scripts/run_volume.py", "--af", "20", "--gpu", g, "--mask_T1", MASK3D,
                         "--mask_FLAIR", MASK3D, "--out", J / "mcinr", "--resume"], J / "mcinr/meta.json", L / "mcinr_run.log")
    return q


MCINR_SCORE = r'''
import json, sys
from pathlib import Path
import numpy as np, torch
R = Path(sys.argv[1]); sys.path.insert(0, str(R / "scripts"))
from t1t2_common import Pair
dev = torch.device("cuda:0")
for c in ("T1", "FLAIR"):
    src = R / f"mcinr/mcinr_{c}_complex.npy"
    v = np.load(src); p = Pair(c, dev)
    assert v.shape == tuple(p.gt.shape), (v.shape, p.gt.shape)
    row = p.score(torch.from_numpy(v).to(torch.complex64).to(dev), "mcinr", seconds=None)
    json.dump({"contrast": c, "rows": [row], "scale_p999": float(p.scale), "coils": int(p.NC), "source": str(src),
               "note": "multi-contrast INR of Niessen et al. 2025, per-slice 2-D, paper recipe (mcinr_niessen2025 "
                       "run_volume.py with this topic's mask), scored through t1t2_common.Pair.score"},
              open(R / f"logs/mcinr_{c}.json", "w"), indent=1)
    link = R / f"recon/mcinr_{c}_complex.npy"
    if not link.exists():
        link.symlink_to(src)
    print(c, "head_psnr", row["head_psnr"]); del p; torch.cuda.empty_cache()
'''


def rows_of(p):
    try:
        d = json.load(open(p))
    except Exception:
        return ""
    if "rows" in d:
        return "; ".join(f"{r.get('contrast', '')} {r['method']} {r['head_psnr']:.3f}" for r in d["rows"] if r.get("head_psnr") is not None)
    if "plain" in d:
        return f"head PSNR {d['plain']['head_psnr']:.3f}" + (f" / {d['contrast']['head_psnr']:.3f}" if d.get("contrast") else "")
    return ""


def run(names):
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert vis.isdigit(), f"needs exactly one GPU from gpu.py --gpu, got CUDA_VISIBLE_DEVICES={vis!r}"
    assert MASK3D.exists(), "run --stage setup first"
    q = jobs(vis)
    unknown = [n for n in names if n not in q]
    assert not unknown, f"unknown jobs {unknown}; known: {sorted(q)}"
    if any(n.startswith("gs_") for n in names):
        ensure_trainer()
    (OUT / "claims").mkdir(parents=True, exist_ok=True)
    failed = []
    for n in names:
        j = q[n]
        if j["done"].exists():
            log(f"-- {n}: result exists, skipped ({rows_of(j['done'])})")
            continue
        try:
            (OUT / "claims" / n).mkdir()
        except FileExistsError:
            log(f"-- {n}: claimed by another worker, skipped")
            continue
        j["log"].parent.mkdir(parents=True, exist_ok=True)
        if n.startswith("gs_"):
            assert not j["log"].exists(), f"refusing to overwrite outputs of {n}"
        rc = None
        for attempt in (1, 2):
            with open(j["log"], "a") as lf:               # no server clock or durations in this job's output (time rule)
                rc = subprocess.run(j["cmd"], cwd=j["cwd"], env=ENV, stdout=lf, stderr=subprocess.STDOUT).returncode
            if (rc == 0 and j["done"].exists()) or n.startswith("gs_"):
                break
            log(f"   {n}: attempt {attempt} rc={rc}")
        if rc == 0 and n == "mcinr":
            out = subprocess.run([PY, "-c", MCINR_SCORE, str(J)], env=ENV, capture_output=True, text=True)
            log(out.stdout.strip(), out.stderr[-800:] if out.returncode else "")
            rc = out.returncode
        ok = rc == 0 and j["done"].exists()
        if not ok:
            failed.append(n)
            (OUT / "claims" / n).rmdir()
            tail = [l for l in open(j["log"], errors="replace").read().splitlines()[-12:]]
            log(f"===== {n} FAILED rc={rc}; log tail:\n" + "\n".join(tail))
        else:
            log(f"===== {n} done: {rows_of(j['done'])}")
    log(f"DONE; failed: {failed or 'none'}")
    sys.exit(1 if failed else 0)


# ------------------------------------------------------------------------------------------------ finalize
def finalize():
    import numpy as np
    sc = lambda *a: subprocess.run([PY, *a], cwd=J, env=ENV, capture_output=True, text=True)
    e = {k: v[0] for k, v in GSRUN.items()}
    for a in (["scripts/add_gs_arm.py", "--exp", e["gs_joint"], "--key", "gs_joint_p3", "--label", "MC-GS", "--map", "T1:plain,FLAIR:contrast"],
              ["scripts/add_gs_arm.py", "--exp", e["gs_singleT1"], "--key", "gs_single_p3", "--label", "GS-single", "--map", "T1:plain"]):
        o = sc(*a)
        log(o.stdout.strip(), o.stderr[-800:])
        assert o.returncode == 0
    exp = e["gs_singleFLAIR"]                     # as finalize_af24_jyz.sh: merge the FLAIR channel under the same key
    s = json.load(open(RES / f"{exp}_score.json"))
    assert s["plain_name"] == "FLAIR"
    with np.load(RES / f"{exp}_volumes.npz") as z:
        np.save(J / "recon/gs_single_p3_FLAIR_mag.npy", np.asarray(z["recT1"], dtype=np.float32))
        np.save(J / "recon/gs_single_p3_FLAIR_complex.npy", np.asarray(z["recT1_complex"], dtype=np.complex64))
    p = J / "logs/extra_gs_arms.json"
    arms = [a for a in json.load(open(p)) if not (a["key"] == "gs_single_p3" and "FLAIR" in a["map"])]
    arms.append(dict(key="gs_single_p3", label="GS-single", exp=exp, map={"FLAIR": "plain"}))
    json.dump(arms, open(p, "w"), indent=1)
    o = sc("scripts/summarize.py")
    assert o.returncode == 0, o.stderr[-1500:]
    for c in ("T1", "FLAIR"):                     # the ground truth must be the AF24 topic's, bit for bit (mask-independent)
        a, b = np.load(J / f"recon/gt_{c}_complex.npy"), np.load(A24 / f"recon/gt_{c}_complex.npy")
        log(f"GT {c} identical to the AF24 topic's: {bool(a.shape == b.shape and (a == b).all())}")
    rows = json.load(open(J / "logs/summary.json"))
    art = Path(os.environ.get("MCGS_ARTIFACT_DIR", OUT / "artifacts"))
    art.mkdir(parents=True, exist_ok=True)
    for r in rows:
        r.pop("seconds", None)                    # no server durations leave the server (time rule)
    (art / "summary_net20.json").write_text(json.dumps(rows, indent=1))
    shutil.copy2(J / "logs/prep_04_export.json", art / "prep_04_export.json")
    for c in ("T1", "FLAIR"):
        log(f"\n=== {c} (net 20.00x), sorted by head PSNR")
        for r in sorted((r for r in rows if r["contrast"] == c and r.get("head_psnr") is not None), key=lambda r: -r["head_psnr"]):
            lam = f" lam {r['lam']}" if r.get("lam") is not None else ""
            log(f"  {r['method']:14s} head PSNR {r['head_psnr']:.3f}  SSIM {r['head_ssim']:.4f}  NMSE {r['nmse']:.5f}{lam}")
    log("FINALIZE_DONE")


ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=["setup", "run", "finalize", "list"])
ap.add_argument("--jobs", nargs="+", default=[])
args = ap.parse_args()
if args.stage == "setup":
    setup()
elif args.stage == "list":
    q = jobs("0")
    log("\n".join(f"{n:28s} {'done' if j['done'].exists() else '-'}  {rows_of(j['done'])}" for n, j in q.items()))
elif args.stage == "run":
    run(args.jobs)
else:
    finalize()
