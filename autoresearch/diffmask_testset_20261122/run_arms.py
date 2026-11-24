#!/usr/bin/env python3
# PORT of autoresearch/t1flair_af_testset_20261121/run_arms.py (session mcgs-0f) for the different-mask study on
# the test set (user, 2026-11-22): the "af" of a row is the sampling pattern ("uniform" | "radial") at
# acceleration factor 16; registered runs, row counts and names changed accordingly, the logic is unchanged.
"""run_arms.py -- comparator runner (nine methods, eight arms) of the T1 + FLAIR acceleration study on the six test scans.

Row "<scan>:<af>:<arm>", arm in classical | cssense | jcssense | spirit | zsssdu | dip | inr | mcinr; 24 x 8 = 192
formal rows. Every arm runs the cloned topic scripts of make_topics.py in OUT/topics/<scan>_af<af>, with the
command lines of that AF's jiangyizhou topic (run_classical_af20.sh, run_spirit_sweep.sh, run_stage2_af20.sh and
their AF12 / AF16 / AF24 counterparts); the lambdas are FROZEN[af]:
  classical  recon_classical.py --contrast C --gpu G                      (ZF, CG-SENSE; also its default CS-SENSE)
  cssense    recon_classical.py ... --methods cssense --admm_outer 200 --lam L --suffix _o200_lamL
  jcssense   recon_joint_cssense.py --topic T --contrasts T1,FLAIR --gpu G --admm_outer 200 --lam L --suffix _o200_lamL
             (one run per distinct lambda; each contrast is taken from the run at its own lambda)
  spirit     recon_spirit.py --contrast C --gpu G --ksize 5 --lam L --iters 150 --chunk 24 --suffix _ks5_lamL
  zsssdu     recon_zsssdu.py --contrast C --gpu G --epochs 100
  dip        recon_dip_inr.py --method dip --contrast C --gpu G --iters 5000 --lr 3e-3 --eval_every 100
  inr        recon_dip_inr.py --method inr --contrast C --gpu G --iters 1500 --eval_every 50
  mcinr      mcinr_niessen2025/scripts/run_volume.py (GR_pt1.10, paper config, 200 steps per slice) on the clone's
             data links, output <clone>/mcinr/, then this file's --score_mcinr: to_gauge_units and the clone's
             Pair.score, as score.py + score_mcinr_arm.py did for jiangyizhou.
Every script sets CUDA_VISIBLE_DEVICES from its own --gpu, so --gpu gets the physical id gpu.py exposes.
Control rows "jyz:20:<arm>" run the same commands in the control clone OUT/topics/jyz_af20 (jiangyizhou AF20 data)
and print the difference from the registered logs/summary.json of the AF20 topic.

A row is done when every expected log json and every expected recon file of the arm exists. A job whose outputs are
all present is skipped (it finished before); a job with only some of them refuses to run (remove by hand); a claimed
row is skipped; nothing is overwritten. --smoke: a few iterations, two MC-INR slices, root OUT/smoke.

  gpu.py run --cpu   -- @python .../run_arms.py --selftest
  gpu.py run --cpu   -- @python .../run_arms.py --status
  gpu.py run --gpu N -- @python .../run_arms.py [--smoke] --rows vol1_huangyuwei:20:classical ...
  gpu.py run --cpu   -- @python .../run_arms.py --release ROW
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import re

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# gpu.py runs this file from a per-job source copy whose directory name carries a server clock value; records and
# log lines name the repo copy by its repo-relative path instead (time rule)
SRC_ROOT = HERE.parents[1]
SELF_REL = "autoresearch/diffmask_testset_20261122/run_arms.py"
JOB_CLOCK = re.compile(r"j-[0-9]{8}T[0-9]{6}")
from ts_common import AFS, FROZEN, JYZ_TOPIC, MASK, OUT, SCANS, claim, sha256   # noqa: E402
from make_topics import check_topic, make_topic, topic_dir                      # noqa: E402

ARMS = ("classical", "cssense", "jcssense", "spirit", "zsssdu", "dip", "inr", "mcinr")
CONTRASTS = ("T1", "FLAIR")
KEYS = ("head_psnr", "head_ssim", "nmse")
FORMAL_ROWS = [f"{s}:{af}:{a}" for s in SCANS for af in AFS for a in ARMS]
CONTROL_ROWS = [f"jyz:uniform:{a}" for a in ARMS]
# settings other than lambda, the same at every AF (read from each topic's launch scripts; checked by --selftest
# against the registered result json of every topic)
SET = dict(admm_outer=200, spirit_ksize=5, spirit_iters=150, spirit_chunk=24, zs_epochs=100,
           dip_iters=5000, dip_lr="3e-3", dip_eval=100, inr_iters=1500, inr_eval=50, mcinr_steps=200, mcinr_lr=0.01)
SMOKE = dict(admm_outer=2, spirit_iters=2, zs_epochs=1, dip_iters=20, dip_eval=10, inr_iters=20, inr_eval=10,
             mcinr_slices="128:130")
MCINR = Path("/data0/congcong/code/GS/mcinr_niessen2025")                   # read only
MCINR_PY = "/home/congcongliu/anaconda3/envs/GR_pt1.10/bin/python"
MCINR_SHA = {   # repo HEAD 25110dc; every change since the registered runs' commits only adds options (defaults kept)
    "scripts/run_volume.py": "363a196346714919fc585b55319fb22c49a15698d053efade7b2618b7516c01d",
    "scripts/common.py": "9e6376367bf0b03f152274d88376a7a2c2cf80de6cbf13abfb1c101a54881010",
    "mcinr/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "mcinr/data.py": "21db3654f5381d46284930c3fb8b894ce427a37ea8915a950845e9d3d16c619f",
    "mcinr/fft.py": "abaf3eeb42455dee93555f2ee5a5a76fb91e97320636a00900f17c0c012ae8b8",
    "mcinr/loss.py": "47881d33123cb302e2774bd00609c4b01e161ee968e59944190adba7d5106f09",
    "mcinr/model.py": "3bc5af9c1274aca15776c68b97784e690e9e8d0a79199093491d87b600e6bba5",
    "mcinr/recon.py": "eb45ce27922f1e228fe64b56acbdbba9d47be47c0e2f5b92c6e79a875f847fdc",
    "mcinr/niessen2025/config_direct_inr.yaml": "338a89b524871e87273ae9eb4f1a866797837be6cf567309e64774af48b53e74",
}
log = lambda *a: print(*a, flush=True)


def parse_row(row):
    scan, af, arm = row.split(":")
    assert scan in SCANS or (scan == "jyz" and af == "uniform"), row
    assert af in AFS and arm in ARMS, row
    return scan, af, arm


def shown(cmd):
    """a command as it may be logged or recorded: paths inside the job's source copy become repo-relative"""
    out = [a.replace(str(SRC_ROOT) + "/", "") for a in cmd]
    assert not any(JOB_CLOCK.search(a) for a in out), out
    return out


def job(name, cmd, outs, partial=()):
    return dict(name=name, cmd=[str(c) for c in cmd], outs=[Path(o) for o in outs], partial=[Path(p) for p in partial])


def arm_plan(root, row, gpu, smoke=False):
    """(jobs, results) of a row. results: one entry per (contrast, method) with the json that holds its row and its
    recon file. Commands are built from FROZEN and SET only; gpu is the physical id."""
    scan, af, arm = parse_row(row)
    T = topic_dir(scan, af, root)
    L, R, S = T / "logs", T / "recon", T / "scripts"
    py = [sys.executable, "-u"]
    s = dict(SET, **(SMOKE if smoke else {}))
    jobs, res = [], []
    if arm == "classical":
        for c in CONTRASTS:
            jobs.append(job(f"classical_{c}", [*py, S / "recon_classical.py", "--contrast", c, "--gpu", gpu],
                            [L / f"classical_{c}.json", *(R / f"{m}_{c}_complex.npy" for m in ("zf", "cgsense", "cssense", "gt"))]))
            res += [dict(contrast=c, method=m, json=L / f"classical_{c}.json", recon=R / f"{m}_{c}_complex.npy", reg=m)
                    for m in ("zf", "cgsense")]
    elif arm == "cssense":
        for c in CONTRASTS:
            lam = FROZEN[af]["cssense"][c]
            sfx = f"_o200_lam{lam}"
            jobs.append(job(f"classical_{c}{sfx}", [*py, S / "recon_classical.py", "--contrast", c, "--gpu", gpu,
                                                    "--methods", "cssense", "--admm_outer", s["admm_outer"],
                                                    "--lam", lam, "--suffix", sfx],
                            [L / f"classical_{c}{sfx}.json", R / f"cssense{sfx}_{c}_complex.npy"]))
            res.append(dict(contrast=c, method=f"cssense{sfx}", json=L / f"classical_{c}{sfx}.json",
                            recon=R / f"cssense{sfx}_{c}_complex.npy", reg="cssense_o200"))
    elif arm == "jcssense":
        lams = FROZEN[af]["jcssense"]
        for lam in sorted(set(lams.values())):
            sfx = f"_o200_lam{lam}"
            jobs.append(job(f"jcssense{sfx}", [*py, S / "recon_joint_cssense.py", "--topic", T, "--contrasts", "T1,FLAIR",
                                               "--gpu", gpu, "--admm_outer", s["admm_outer"], "--lam", lam, "--suffix", sfx],
                            [L / f"jcssense{sfx}.json", *(R / f"jcssense{sfx}_{c}_complex.npy" for c in CONTRASTS)]))
        for c in CONTRASTS:
            sfx = f"_o200_lam{lams[c]}"
            res.append(dict(contrast=c, method=f"jcssense{sfx}", json=L / f"jcssense{sfx}.json",
                            recon=R / f"jcssense{sfx}_{c}_complex.npy", reg="jcssense"))
    elif arm == "spirit":
        for c in CONTRASTS:
            ks, lam = FROZEN[af]["spirit"][c]
            assert ks == s["spirit_ksize"], (af, c, ks)
            sfx = f"_ks{ks}_lam{lam}"
            jobs.append(job(f"spirit_{c}{sfx}", [*py, S / "recon_spirit.py", "--contrast", c, "--gpu", gpu, "--ksize", ks,
                                                 "--lam", lam, "--iters", s["spirit_iters"], "--chunk", s["spirit_chunk"],
                                                 "--suffix", sfx],
                            [L / f"spirit_{c}{sfx}.json", R / f"spirit{sfx}_{c}_complex.npy"]))
            res.append(dict(contrast=c, method=f"spirit{sfx}", json=L / f"spirit_{c}{sfx}.json",
                            recon=R / f"spirit{sfx}_{c}_complex.npy", reg="spirit"))
    elif arm in ("zsssdu", "dip", "inr"):
        for c in CONTRASTS:
            if arm == "zsssdu":
                cmd = [*py, S / "recon_zsssdu.py", "--contrast", c, "--gpu", gpu, "--epochs", s["zs_epochs"]]
            elif arm == "dip":
                cmd = [*py, S / "recon_dip_inr.py", "--method", "dip", "--contrast", c, "--gpu", gpu,
                       "--iters", s["dip_iters"], "--lr", s["dip_lr"], "--eval_every", s["dip_eval"]]
            else:
                cmd = [*py, S / "recon_dip_inr.py", "--method", "inr", "--contrast", c, "--gpu", gpu,
                       "--iters", s["inr_iters"], "--eval_every", s["inr_eval"]]
            jobs.append(job(f"{arm}_{c}", cmd, [L / f"{arm}_{c}.json", R / f"{arm}_{c}_complex.npy"]))
            res.append(dict(contrast=c, method=arm, json=L / f"{arm}_{c}.json", recon=R / f"{arm}_{c}_complex.npy", reg=arm))
    else:   # mcinr
        D, M = T / "data", T / "mcinr"
        cmd = [MCINR_PY, "-u", MCINR / "scripts/run_volume.py", "--af", 16, "--gpu", gpu,
               "--mask_T1", D / MASK(af).name, "--mask_FLAIR", D / MASK(af).name,
               "--ksp_T1", D / "ksp_T1.h5", "--ksp_FLAIR", D / "ksp_FLAIR.h5",
               "--csm_T1", D / "csm_espirit_T1.h5", "--csm_FLAIR", D / "csm_espirit_FLAIR.h5", "--out", M]
        if smoke:
            cmd += ["--slices", s["mcinr_slices"]]
        jobs.append(job("mcinr_volume", cmd,
                        [M / "meta.json", M / "losses.npy", *(M / f"mcinr_norm_{c}_complex.npy" for c in CONTRASTS)],
                        partial=[M / "partial_state.npz", *(M / f"mcinr_norm_{c}_complex.partial.npy" for c in CONTRASTS)]))
        jobs.append(job("mcinr_score", [*py, Path(__file__).resolve(), "--score_mcinr", T, "--af", af, "--gpu", gpu],
                        [*(L / f"mcinr_{c}.json" for c in CONTRASTS), *(R / f"mcinr_{c}_complex.npy" for c in CONTRASTS)]))
        res += [dict(contrast=c, method="mcinr", json=L / f"mcinr_{c}.json", recon=R / f"mcinr_{c}_complex.npy", reg="mcinr")
                for c in CONTRASTS]
    return jobs, res


def claim_name(row):
    scan, af, arm = parse_row(row)
    return f"arm_{scan}_{af}_{arm}"


def row_state(root, row):
    """done = every expected json and recon of the arm exists; stale = claimed, not done; open = unclaimed"""
    jobs, res = arm_plan(root, row, "0")
    need = {o for j in jobs for o in j["outs"]} | {r["json"] for r in res} | {r["recon"] for r in res}
    if all(p.exists() for p in need):
        return "done"
    return "stale" if (Path(root) / "claims" / claim_name(row)).exists() else "open"


def status(root, rows):
    states = [row_state(root, r) for r in rows]
    for row, st in zip(rows, states):
        log(f"{st:6s} {row}")
    log("summary: " + ", ".join(f"{s} {states.count(s)}" for s in ("done", "stale", "open")))


def release(root, row):
    assert row_state(root, row) != "done", f"{row} has all its results; refusing to release"
    c = Path(root) / "claims" / claim_name(row)
    assert c.is_dir(), f"{row} has no claim"
    c.rmdir()
    log(f"released {row} (complete jobs are kept and skipped on a rerun; partial outputs must be removed by hand)")


def result_row(r):
    """the scored row of one result, from the json the script wrote (seconds fields are not read)"""
    d = json.load(open(r["json"]))
    hit = [x for x in d["rows"] if x["method"] == r["method"] and x.get("contrast", d.get("contrast")) == r["contrast"]]
    assert len(hit) == 1, (r["json"], r["method"], r["contrast"], len(hit))
    return {k: hit[0][k] for k in KEYS}


def registered(row):
    """registered rows of the jiangyizhou AF topic of a control row: {(contrast, reg method): {KEYS}}"""
    _, af, _ = parse_row(row)
    rows = json.load(open(JYZ_TOPIC(af) / "logs/summary.json"))
    return {(r["contrast"], r["method"]): {k: r[k] for k in KEYS} for r in rows}


def check_mcinr_repo():
    bad = [f for f, h in MCINR_SHA.items() if sha256(MCINR / f) != h]
    assert not bad, f"MC-INR repository changed: {bad}"
    assert Path(MCINR_PY).is_file(), MCINR_PY


def score_mcinr(topic, af, gpu):
    """score.py's to_gauge_units + score_mcinr_arm.py's Pair.score on the clone; writes recon/mcinr_<c>_complex.npy
    and logs/mcinr_<c>.json in the topic row format. The run's duration fields are not copied (time rule)."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    sys.dont_write_bytecode = True                       # nothing is written into the MC-INR repository
    import numpy as np
    import torch
    T = Path(topic)
    sys.path.insert(0, str(T / "scripts"))
    import t1t2_common as TC
    assert Path(TC.R) == T, f"imported the wrong gauge module: {TC.R}"
    sys.path.insert(0, str(MCINR))
    from mcinr.data import to_gauge_units
    meta = json.load(open(T / "mcinr/meta.json"))
    assert meta["af"] == 16 and meta["hparams"]["n_epochs"] == SET["mcinr_steps"] and meta["hparams"]["lr"] == SET["mcinr_lr"], meta["hparams"]
    for c in CONTRASTS:
        assert Path(meta["masks"][c]) == MASK(af).resolve(), meta["masks"]
        assert Path(meta["ksp"][c]) == (T / f"data/ksp_{c}.h5").resolve(), meta["ksp"]
        assert Path(meta["csm"][c]) == (T / f"data/csm_espirit_{c}.h5").resolve(), meta["csm"]
    dev = torch.device("cuda:0")
    for c in CONTRASTS:
        p = TC.Pair(c, dev)
        xn = torch.from_numpy(np.load(T / f"mcinr/mcinr_norm_{c}_complex.npy"))
        x = to_gauge_units(xn.to(torch.complex128), meta["M_acquired"], p.scale).to(torch.complex64)
        if meta["slices"] == "all":
            assert tuple(x.shape) == tuple(p.gt.shape), (x.shape, p.gt.shape)
            note = "multi-contrast INR of Niessen et al. 2025, per-slice 2-D, paper recipe"
        else:                                            # smoke: the fitted slices in an otherwise zero volume
            full = torch.zeros(p.gt.shape, dtype=torch.complex64)
            full[meta["slices"]] = x
            x, note = full, f"SMOKE: only readout slices {meta['slices']} fitted, the rest is zero; not a result"
        np.save(T / f"recon/mcinr_{c}_complex.npy", x.numpy())
        row = p.score(x.to(dev), "mcinr")
        json.dump({"contrast": c, "rows": [row], "scale_p999": float(p.scale), "coils": int(p.NC),
                   "source": str(T / f"mcinr/mcinr_norm_{c}_complex.npy"), "M_acquired": meta["M_acquired"],
                   "note": note}, open(T / f"logs/mcinr_{c}.json", "w"), indent=1)
        log(f"[{c}] mcinr head_psnr={row['head_psnr']:.3f} head_ssim={row['head_ssim']:.4f} nmse={row['nmse']:.5f}")
        del p
        torch.cuda.empty_cache()


def selftest():
    import yaml
    assert len(FORMAL_ROWS) == 12 * 8 == 96 and len(set(FORMAL_ROWS)) == 96
    num = lambda cmd, flag: cmd[cmd.index(flag) + 1]
    for row in FORMAL_ROWS + CONTROL_ROWS:
        scan, af, arm = parse_row(row)
        jobs, res = arm_plan(OUT, row, "5")
        T = topic_dir(scan, af)
        for j in jobs:
            cmd = j["cmd"]
            assert num(cmd, "--gpu") == "5", (row, cmd)
            if scan != "jyz":
                assert not any("Results_20260908" in a for a in cmd), (row, cmd)
            assert all(str(o).startswith(str(T) + "/") for o in j["outs"]), (row, j["outs"])
            if arm in ("classical", "cssense", "jcssense", "spirit", "zsssdu", "dip", "inr"):
                assert cmd[2].startswith(str(T / "scripts") + "/"), (row, cmd)
        if arm == "classical":
            assert [len(j["cmd"]) for j in jobs] == [7, 7] and not any("--lam" in j["cmd"] for j in jobs), row
        if arm == "cssense":
            for c, j in zip(CONTRASTS, jobs):
                assert float(num(j["cmd"], "--lam")) == FROZEN[af]["cssense"][c], (row, c)
                assert num(j["cmd"], "--admm_outer") == "200" and num(j["cmd"], "--methods") == "cssense", row
        if arm == "jcssense":
            lams = FROZEN[af]["jcssense"]
            assert len(jobs) == (1 if lams["T1"] == lams["FLAIR"] else 2), (row, len(jobs))
            assert {float(num(j["cmd"], "--lam")) for j in jobs} == set(lams.values()), row
            for j in jobs:
                assert num(j["cmd"], "--admm_outer") == "200" and num(j["cmd"], "--contrasts") == "T1,FLAIR", row
            for r in res:                     # each contrast from the run at its own lambda
                assert r["json"].name == f"jcssense_o200_lam{lams[r['contrast']]}.json", (row, r)
        if arm == "spirit":
            for c, j in zip(CONTRASTS, jobs):
                assert (int(num(j["cmd"], "--ksize")), float(num(j["cmd"], "--lam"))) == FROZEN[af]["spirit"][c], (row, c)
                assert num(j["cmd"], "--ksize") == "5" and num(j["cmd"], "--iters") == "150", row
        if arm == "zsssdu":
            assert all(num(j["cmd"], "--epochs") == "100" for j in jobs), row
        if arm == "dip":
            assert all((num(j["cmd"], "--iters"), float(num(j["cmd"], "--lr")), num(j["cmd"], "--eval_every"))
                       == ("5000", 3e-3, "100") for j in jobs), row
        if arm == "inr":
            assert all((num(j["cmd"], "--iters"), num(j["cmd"], "--eval_every")) == ("1500", "50")
                       and "--lr" not in j["cmd"] for j in jobs), row
        if arm == "mcinr":
            cmd = jobs[0]["cmd"]
            assert "--slices" not in cmd and "--n_epochs" not in cmd and "--lr" not in cmd, row
            assert cmd[0] == MCINR_PY and num(cmd, "--out") == str(T / "mcinr"), row
            for flag in ("--mask_T1", "--mask_FLAIR", "--ksp_T1", "--ksp_FLAIR", "--csm_T1", "--csm_FLAIR"):
                assert num(cmd, flag).startswith(str(T / "data") + "/"), (row, flag)
        assert len(res) == 2 * (2 if arm == "classical" else 1), (row, len(res))
    # the non-lambda settings are what every topic registered (lambdas: ts_common --selftest)
    for af in AFS:
        J = JYZ_TOPIC(af)
        for c in CONTRASTS:
            sp = json.load(open(J / f"logs/spirit_{c}_ks5_lam{FROZEN[af]['spirit'][c][1]}.json"))
            assert (sp["ksize"], sp["iters"], sp["solver"], sp["tyk"], sp["levels"]) == (5, 150, "fista", 0.01, 4), (af, c)
            for arm, it, lr in (("dip", 5000, 3e-3), ("inr", 1500, 5e-3)):
                r = json.load(open(J / f"logs/{arm}_{c}.json"))["rows"][0]
                assert (r["iters"], r["lr"]) == (it, lr), (af, c, arm, r["iters"], r["lr"])
            zs = json.load(open(J / f"logs/zsssdu_{c}.json"))["rows"][0]
            assert zs["epochs_run"] <= SET["zs_epochs"] and (zs["unrolls"], zs["cg_iters_net"]) == (10, 6), (af, c)
        src = (J / "scripts/recon_zsssdu.py").read_text()
        assert "def __init__(self, ch=64, nb=5)" in src and "net = ResNet().to(dev)" in src, af   # 5 residual blocks
        assert 'add_argument("--patience", type=int, default=10)' in src, af       # early stop on the Gamma loss
        cl = (J / "scripts/recon_classical.py").read_text()
        assert 'add_argument("--cg_iters", type=int, default=60)' in cl and 'add_argument("--tik", type=float, default=0.01)' in cl
        for m in ("mcinr", "zsssdu", "dip", "inr"):
            for c in CONTRASTS:
                assert (J / f"logs/{m}_{c}.json").is_file(), (af, m, c)
    cfg = yaml.safe_load(open(MCINR / "mcinr/niessen2025/config_direct_inr.yaml"))["training"]
    assert int(cfg["train_iters"]) == SET["mcinr_steps"] and float(cfg["train_lr"]) == SET["mcinr_lr"], cfg
    check_mcinr_repo()
    # what goes into records and log lines carries no job directory and no job clock (time rule)
    raw_clock, recs = False, []
    for row in FORMAL_ROWS + CONTROL_ROWS:
        for j in arm_plan(OUT, row, "5")[0]:
            raw_clock |= any(JOB_CLOCK.search(a) for a in j["cmd"])
            recs.append(dict(job=j["name"], cmd=shown(j["cmd"]), log=str(OUT / "x.log")))
            if j["name"] == "mcinr_score":
                assert shown(j["cmd"])[2] == SELF_REL, shown(j["cmd"])[:3]
    text = json.dumps(dict(root=str(OUT), gpu="5", runs=recs), default=str)
    assert not JOB_CLOCK.search(text) and str(SRC_ROOT) not in text
    try:
        shown(["j-" + "20990101T000000" + "-0000abcd"])     # a synthetic id, split so a grep finds no clock here
        raise SystemExit("shown() must refuse a job clock outside the source copy")
    except AssertionError:
        pass
    print(f"record check: raw commands carried a job clock: {raw_clock}; recorded commands: none")
    # claims, states and release on a throw-away root
    tmp = Path(tempfile.mkdtemp(dir=OUT, prefix="selftest_arms_"))
    try:
        row = FORMAL_ROWS[0]
        make_topic(parse_row(row)[0], parse_row(row)[1], tmp)
        assert row_state(tmp, row) == "open"
        assert claim(tmp / "claims", claim_name(row)) is True
        assert claim(tmp / "claims", claim_name(row)) is False
        assert row_state(tmp, row) == "stale"
        jobs, res = arm_plan(tmp, row, "0")
        for p in jobs[0]["outs"]:                   # half the arm: still stale
            p.write_text("{}")
        assert row_state(tmp, row) == "stale"
        release(tmp, row)
        assert row_state(tmp, row) == "open"
        claim(tmp / "claims", claim_name(row))
        for p in jobs[1]["outs"]:
            p.write_text("{}")
        assert row_state(tmp, row) == "done"
        try:
            release(tmp, row)
            raise SystemExit("release of a done row must be refused")
        except AssertionError:
            pass
    finally:
        shutil.rmtree(tmp)
    print("SELFTEST_OK")


def run_job(j, cwd, env, record):
    have = [o.exists() for o in j["outs"]]
    logf = Path(cwd) / "logs" / f"run_{j['name']}.log"
    if all(have):
        log(f"-- {j['name']}: all outputs present (finished before), skipped")
        record.append(dict(job=j["name"], cmd=shown(j["cmd"]), rc="skipped_complete"))
        return True
    if any(have) or any(p.exists() for p in j["partial"]) or logf.exists():
        log(f"-- {j['name']}: partial outputs or an old log exist; refusing to overwrite, remove by hand")
        record.append(dict(job=j["name"], cmd=shown(j["cmd"]), rc="refused_partial"))
        return False
    log(f"   {' '.join(shown(j['cmd']))}")                       # no server clock or durations in logs (time rule)
    with open(logf, "w") as lf:
        rc = subprocess.run(j["cmd"], cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    ok = rc == 0 and all(o.exists() for o in j["outs"])
    record.append(dict(job=j["name"], cmd=shown(j["cmd"]), rc=rc, log=str(logf)))
    if not ok:
        log(f"   {j['name']} FAILED rc={rc}; see {logf} (not echoed: script logs carry durations)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", choices=FORMAL_ROWS + CONTROL_ROWS)
    ap.add_argument("--smoke", action="store_true", help="few iterations, two MC-INR slices, into OUT/smoke")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--release", metavar="ROW", choices=FORMAL_ROWS + CONTROL_ROWS)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--score_mcinr", metavar="TOPIC", help=argparse.SUPPRESS)
    ap.add_argument("--af", help=argparse.SUPPRESS)
    ap.add_argument("--gpu", help=argparse.SUPPRESS)
    args = ap.parse_args()
    root = OUT / "smoke" if args.smoke else OUT
    if args.score_mcinr:
        return score_mcinr(args.score_mcinr, args.af, args.gpu)
    if args.selftest:
        return selftest()
    if args.status:
        return status(root, args.rows or FORMAL_ROWS)
    if args.release:
        return release(root, args.release)
    assert args.rows, "--rows is required"
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert vis.isdigit(), f"needs exactly one GPU from gpu.py --gpu, got CUDA_VISIBLE_DEVICES={vis!r}"
    (root / "claims").mkdir(parents=True, exist_ok=True)
    art = Path(os.environ.get("MCGS_ARTIFACT_DIR", root / "artifacts"))
    art.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MASK_H5", "T1T2_MASK_"))}   # default masks only
    env.update(PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONDONTWRITEBYTECODE="1",
               GIT_OPTIONAL_LOCKS="0", OMP_NUM_THREADS="24", MKL_NUM_THREADS="24")
    log(f"root {root}; physical GPU {vis}; smoke {args.smoke}; rows {args.rows}")
    record = dict(root=str(root), gpu=vis, smoke=args.smoke, runs=[])
    failed = []
    for row in args.rows:
        scan, af, arm = parse_row(row)
        if args.smoke:
            make_topic(scan, af, root)                         # smoke clones live under OUT/smoke/topics
        assert topic_dir(scan, af, root).is_dir(), f"{row}: clone missing; run make_topics.py first"
        check_topic(scan, af, root)
        if arm == "mcinr":
            check_mcinr_repo()
        if not claim(root / "claims", claim_name(row)):
            log(f"-- {row}: already claimed, skipped")
            continue
        T = topic_dir(scan, af, root)
        jobs, res = arm_plan(root, row, vis, args.smoke)
        log(f"\n===== {row}  ({T})")
        run = dict(row=row, topic=str(T), jobs=[], results=[])
        ok = True
        for j in jobs:
            if not run_job(j, T, env, run["jobs"]):
                ok = False
                break
        if ok and row_state(root, row) == "done":
            reg = registered(row) if scan == "jyz" else {}
            for r in res:
                got = result_row(r)
                e = dict(contrast=r["contrast"], method=r["method"], json=str(r["json"]), recon=str(r["recon"]), **got)
                ref = reg.get((r["contrast"], r["reg"]))
                if ref:
                    e["registered"] = ref
                    e["diff"] = {k: got[k] - ref[k] for k in KEYS}
                run["results"].append(e)
                d = f"  (registered {ref['head_psnr']:.6f}, diff {e['diff']['head_psnr']:+.6f})" if ref else ""
                log(f"   {r['contrast']:5s} {r['method']:24s} head PSNR {got['head_psnr']:.6f} head SSIM "
                    f"{got['head_ssim']:.6f} NMSE {got['nmse']:.6f}{d}")
            log(f"===== {row} done")
        else:
            failed.append(row)
            log(f"===== {row} FAILED")
        record["runs"].append(run)
        (art / f"record_arms_gpu{vis}{'_smoke' if args.smoke else ''}_{args.rows[0].replace(':', '_')}.json").write_text(
            json.dumps(record, indent=2, default=str))
    log(f"DONE; failed rows: {failed or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
