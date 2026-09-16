#!/usr/bin/env python3
"""ce_orchestrate.py -- run the 2026-09-14 comparison arms (converged CS-SENSE, Joint SENSE, L1-ESPIRiT, MC-INR)
on the UIH CE pair: lambda grids on the 95% lines -> held-out selection (with edge extension) -> 100% refits.

GPU lane (TV-SENSE, MC-INR): a card is taken as soon as a poll (every POLL s) finds no compute process of this
user on it and >= NEED_MB free; one job per card; at most MAX_GPU jobs at once. (A two-poll rule starved: the
other sessions' schedulers refill a freed card within a minute.) Other users'
processes are not waited for (house rule, T1FLAIR_MCINR topic 2026-09-14).
CPU lane (BART): at most MAX_CPU jobs, nice 10.
Gate: before any TV-SENSE grid, the ported ADMM must reproduce the archived 35-iteration CS-SENSE (pre).
Restartable: finished work is recognised from its meta files; job pids are kept in logs/ext_20260914/jobs/.

run detached:  setsid nohup python scripts/ce_orchestrate.py > logs/ext_20260914/orchestrate.log 2>&1 &
plan heldout4 (2026-09-15): python scripts/ce_orchestrate.py heldout4 -- 95%-line runs of CG-SENSE, ZS-SSDU,
GS-single and MC-GS (GS on the 3dd24aa trainer snapshot), then ce_score_heldout.py.
plan spirit (2026-09-15): L1-SPIRiT grids (kernel 3 and 5 x lambda) on the 95% lines, held-out selection of
lambda per kernel (with edge extension) and then of the kernel, 100% refit per scan (ce_spirit.py).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ce_common as C  # noqa: E402
from ce_select_lambda import select  # noqa: E402

T = C.TOPIC
SC = T / "scripts"
LOGS = T / "logs" / "ext_20260914"
JOBS = LOGS / "jobs"
JOBS.mkdir(parents=True, exist_ok=True)
PY = "/home/congcongliu/anaconda3/envs/GSTMI/bin/python"
PY_TCNN = "/home/congcongliu/anaconda3/envs/GR_pt1.10/bin/python"
NEED_MB, MAX_GPU, MAX_CPU, POLL = 36000, 2, 2, 15
GRIDS = json.load(open(T / "lambda_grids.json"))
ENV = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID")
UID = os.getuid()
PLAN = sys.argv[1] if len(sys.argv) > 1 else "arms"


def log(*a):
    print(time.strftime("%m-%d %H:%M:%S"), *a, flush=True)


def lamstr(lams):
    return ",".join(f"{v:g}" for v in lams)


def meta_ok(pattern_list):
    return all(Path(p).exists() for p in pattern_list)


def tv_metas(mode, scan_list, mask, lams, outer=200, sub="tv_sense"):
    name = "cssense" if mode == "cs" else "jsense"
    return [T / sub / f"{name}_o{outer}_lam{l:g}_{mask}_{s}_meta.json" for l in lams for s in scan_list]


def tv_job(name, mode, scan, mask, lams, prio, outer=200, sub="tv_sense", deps=()):
    scans = [scan] if mode == "cs" else ["pre", "post"]
    cmd = lambda g: [PY, "-u", str(SC / "ce_tv_sense.py"), "--mode", mode, "--mask", mask, "--lams", lamstr(lams),
                     "--outer", str(outer), "--gpu", str(g), "--out", str(T / sub)] + (["--scan", scan] if mode == "cs" else [])
    return dict(name=name, lane="gpu", prio=prio, cmd=cmd, deps=list(deps),
                done=lambda: meta_ok(tv_metas(mode, scans, mask, lams, outer, sub)))


def l1_metas(scan, mask, lams):
    return [T / "l1espirit" / f"l1espirit_i100_lam{l:g}_{mask}_{scan}_meta.json" for l in lams]


def l1_job(name, scan, mask, lams, prio, cleanup=False):
    cmd = lambda g: ["nice", "-n", "10", PY, "-u", str(SC / "ce_l1espirit.py"), "--scan", scan, "--mask", mask,
                     "--lams", lamstr(lams), "--threads", "24", "--out", str(T / "l1espirit")] + (["--cleanup"] if cleanup else [])
    return dict(name=name, lane="cpu", prio=prio, cmd=cmd, deps=[], done=lambda: meta_ok(l1_metas(scan, mask, lams)))


def mcinr_job(name, mask, prio):
    cmd = lambda g: [PY_TCNN, "-u", str(SC / "ce_mcinr.py"), "--mask", mask, "--gpu", str(g), "--out", str(T / "mcinr")]
    return dict(name=name, lane="gpu", prio=prio, cmd=cmd, deps=[],
                done=lambda: meta_ok([T / "mcinr" / f"mcinr_{mask}_{s}_meta.json" for s in C.SCANS]))


VALIDATION = T / "tv_sense" / "validation" / "cssense_o35_lam0.001_full_pre_meta.json"


def validation_ok():
    if not VALIDATION.exists():
        return None
    m = json.load(open(VALIDATION))
    ref = json.load(open(T / "kres_summary.json"))["pre|CS-SENSE"]
    return abs(m["kres"] - ref) < 2e-4, m["kres"], ref


def alive(pid):
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


jobs = {}


def add(j):
    if j["name"] not in jobs:
        done = j["done"]()                  # a job queued by plan_family after a restart may already be complete
        pf = JOBS / f"{j['name']}.pid"
        live = (not done) and pf.exists() and alive(int(pf.read_text()))   # still running from a previous scheduler
        jobs[j["name"]] = dict(j, state="done" if done else ("running" if live else "pending"), proc=None, gpu=None,
                               tries=0, pid=int(pf.read_text()) if live else None)
        log(f"{'already complete' if done else ('adopted running' if live else 'queued')} {j['name']} ({j['lane']}, prio {j['prio']})")


if PLAN == "arms":
    add(tv_job("validate_cs35_pre", "cs", "pre", "full", [1e-3], prio=0, outer=35, sub="tv_sense/validation"))
    add(mcinr_job("mcinr_full", "full", prio=1))
    add(tv_job("joint_grid", "joint", None, "fit", GRIDS["joint"], prio=2, deps=["validate_cs35_pre"]))
    add(tv_job("cs_pre_grid", "cs", "pre", "fit", GRIDS["cs_pre"], prio=3, deps=["validate_cs35_pre"]))
    add(tv_job("cs_post_grid", "cs", "post", "fit", GRIDS["cs_post"], prio=4, deps=["validate_cs35_pre"]))
    add(mcinr_job("mcinr_fit", "fit", prio=9))
    add(l1_job("l1_pre_grid", "pre", "fit", GRIDS["l1_pre"], prio=2))
    add(l1_job("l1_post_grid", "post", "fit", GRIDS["l1_post"], prio=3))
    FINALS = None                                   # filled after final_names is defined
elif PLAN == "heldout4":
    SNAP = str(SC / "_trainer_3dd24aa")
    HS = T / "heldout_split"
    add(dict(name="h4_make_fit_kspace", lane="cpu", prio=0, deps=[],
             cmd=lambda g: ["nice", "-n", "10", PY, "-u", str(SC / "ce_make_fit_kspace.py")],
             done=lambda: meta_ok([HS / "ksp_fit_check.json"])))
    for s_, pr in (("pre", 1), ("post", 2)):
        add(dict(name=f"h4_cg_{s_}", lane="gpu", prio=pr, need=30000, deps=[],
                 cmd=lambda g, s_=s_: [PY, "-u", str(SC / "ce_classical.py"), "--method", "cgsense", "--scan", s_, "--fit",
                                       "--gpu", str(g), "--out", str(T / "classical_fit")],
                 done=lambda s_=s_: meta_ok([T / "classical_fit" / f"cgsense_{s_}_fit_meta.json"])))
    for m_, pr in (("single_pre", 3), ("single_post", 4), ("joint", 5)):
        tags = {"single_pre": ["gs_single_pre"], "single_post": ["gs_single_post"], "joint": ["gs_joint_pre", "gs_joint_post"]}[m_]
        add(dict(name=f"h4_gs_{m_}", lane="gpu", prio=pr, need=60000, deps=["h4_make_fit_kspace"],
                 cmd=lambda g, m_=m_: [PY, "-u", str(SC / "ce_gs.py"), "--mode", m_, "--fit", "--root", SNAP,
                                       "--gpu", str(g), "--out", str(T / "gs_fit" / m_)],
                 done=lambda m_=m_, tags=tags: meta_ok([T / "gs_fit" / m_ / f"{t}_meta.json" for t in tags])))
    for s_, pr in (("pre", 6), ("post", 7)):
        add(dict(name=f"h4_zs_{s_}", lane="gpu", prio=pr, need=40000, deps=[],
                 cmd=lambda g, s_=s_: [PY, "-u", str(SC / "ce_classical.py"), "--method", "zsssdu", "--scan", s_, "--fit",
                                       "--epochs", "100", "--gpu", str(g), "--out", str(T / "classical_fit")],
                 done=lambda s_=s_: meta_ok([T / "classical_fit" / f"zsssdu_{s_}_fit_meta.json"])))
    h4 = [n for n in jobs]
    add(dict(name="h4_score", lane="cpu", prio=9, deps=h4,
             cmd=lambda g: ["nice", "-n", "10", PY, "-u", str(SC / "ce_score_heldout.py")],
             done=lambda: meta_ok([T / "heldout4_scores.json"])))
    FINALS = list(jobs)
elif PLAN == "spirit":
    SP_FAMS = [f"spirit_ks{k}_{s_}" for s_ in ("pre", "post") for k in (3, 5)]

    def sp_job(name, scan, ks, mask, lams, prio):
        return dict(name=name, lane="gpu", prio=prio, need=40000, deps=[],
                    cmd=lambda g: [PY, "-u", str(SC / "ce_spirit.py"), "--scan", scan, "--mask", mask, "--ksize", str(ks),
                                   "--lams", lamstr(lams), "--gpu", str(g), "--out", str(T / "spirit")],
                    done=lambda: meta_ok([T / "spirit" / f"spirit_ks{ks}_i150_lam{l:g}_{mask}_{scan}_meta.json" for l in lams]))
    for pr, fam in enumerate(SP_FAMS, start=1):
        _, k_, s_ = fam.split("_")
        add(sp_job(f"{fam}_grid", s_, int(k_[2:]), "fit", GRIDS[fam], pr))
    FINALS = ["sp_pre_final", "sp_post_final"]
elif PLAN == "seednull":
    # 2026-09-16: the floor of each randomly initialised arm. Same scan, same data, same code, only the seed
    # differs, so the difference between the two runs is what the method invents on its own. Compare it with the
    # post - pre the arm reports in the non-enhancing lesion (3D-DIP +0.057, 3D-INR +0.029, MC-GS +0.024).
    SNAP = str(SC / "_trainer_3dd24aa")                  # the trainer the archived GS runs used
    SEED = int(os.environ.get("SEED_B", "1"))
    OUTD = T / f"seed_null_s{SEED}"
    add(dict(name="sn_dip_pre", lane="gpu", prio=1, need=40000, deps=[],
             cmd=lambda g: [PY, "-u", str(SC / "ce_dip_inr.py"), "--method", "dip", "--scan", "pre", "--iters", "5000",
                            "--lr", "3e-3", "--eval_every", "100", "--seed", str(SEED), "--gpu", str(g),
                            "--out", str(OUTD)],
             done=lambda: meta_ok([OUTD / "dip_pre_meta.json"])))
    add(dict(name="sn_inr_pre", lane="gpu", prio=2, need=40000, deps=[],
             cmd=lambda g: [PY, "-u", str(SC / "ce_dip_inr.py"), "--method", "inr", "--scan", "pre", "--iters", "1500",
                            "--eval_every", "50", "--seed", str(SEED), "--gpu", str(g), "--out", str(OUTD)],
             done=lambda: meta_ok([OUTD / "inr_pre_meta.json"])))
    add(dict(name="sn_gs_single_pre", lane="gpu", prio=3, need=60000, deps=[],
             cmd=lambda g: [PY, "-u", str(SC / "ce_gs.py"), "--mode", "single_pre", "--root", SNAP, "--seed", str(SEED),
                            "--gpu", str(g), "--out", str(OUTD / "gs_single_pre")],
             done=lambda: meta_ok([OUTD / "gs_single_pre" / "gs_single_pre_meta.json"])))
    add(dict(name="sn_gs_joint", lane="gpu", prio=4, need=60000, deps=[],
             cmd=lambda g: [PY, "-u", str(SC / "ce_gs.py"), "--mode", "joint", "--root", SNAP, "--seed", str(SEED),
                            "--gpu", str(g), "--out", str(OUTD / "gs_joint")],
             done=lambda: meta_ok([OUTD / "gs_joint" / f"gs_joint_{s_}_meta.json" for s_ in C.SCANS])))
    FINALS = list(jobs)
else:
    raise SystemExit(f"unknown plan {PLAN}")


def plan_spirit(scan):
    """per scan: settle lambda for kernel 3 and 5 (extension as in plan_family), then take the (kernel, lambda)
    with the lowest held-out kres and queue the 100% refit."""
    final = f"sp_{scan}_final"
    if final in jobs:
        return
    best = None
    for k in (3, 5):
        fam = f"spirit_ks{k}_{scan}"
        if not family_jobs_idle(fam):
            return
        r = select(fam, GRIDS[fam])
        if not r["complete"]:
            return
        if r["next_lam"] is not None:
            nm = f"{fam}_ext_{r['next_lam']:g}"
            if nm not in jobs:
                log(f"{fam}: held-out minimum at the {r['at_edge']} edge ({r['best']:g}); extending to {r['next_lam']:g}")
                add(sp_job(nm, scan, k, "fit", [r["next_lam"]], 5))
            return
        score = r["rows"][f"{r['best']:g}"]["score"]
        C.dump_json(r, LOGS / f"selection_{fam}.json")
        if best is None or score < best[2]:
            best = (k, r["best"], score)
    k, lam, score = best
    C.dump_json(dict(scan=scan, ksize=k, lam=lam, heldout=score), LOGS / f"selection_spirit_{scan}.json")
    log(f"spirit {scan}: selected kernel {k}, lambda {lam:g} (held-out {score:.5f}); queueing the 100% refit")
    add(sp_job(final, scan, k, "full", [lam], 6))

FAMILY_JOBS = {"cs_pre": "cs_pre", "cs_post": "cs_post", "joint": "joint", "l1_pre": "l1_pre", "l1_post": "l1_post"}
final_names = {"cs_pre": "cs_pre_final", "cs_post": "cs_post_final", "joint": "joint_final",
               "l1_pre": "l1_pre_final", "l1_post": "l1_post_final"}


def family_jobs_idle(fam):
    return all(j["state"] in ("done", "failed") for n, j in jobs.items() if n.startswith(fam + "_") and not n.endswith("final"))


def plan_family(fam):
    """after a family's grid/extension jobs finish: extend or queue the 100% refit."""
    if final_names[fam] in jobs or not family_jobs_idle(fam):
        return
    r = select(fam, GRIDS[fam])
    if not r["complete"]:
        return
    if r["next_lam"] is not None:
        nm = f"{fam}_ext_{r['next_lam']:g}"
        if nm in jobs:
            return
        log(f"{fam}: held-out minimum at the {r['at_edge']} edge ({r['best']:g}); extending to {r['next_lam']:g}")
        if fam.startswith("l1"):
            add(l1_job(nm, fam.split("_")[1], "fit", [r["next_lam"]], prio=5))
        elif fam == "joint":
            add(tv_job(nm, "joint", None, "fit", [r["next_lam"]], prio=5))
        else:
            add(tv_job(nm, "cs", fam.split("_")[1], "fit", [r["next_lam"]], prio=5))
        return
    log(f"{fam}: selected lambda {r['best']:g} (edge flag {r['at_edge']}); queueing the 100% refit")
    C.dump_json(r, LOGS / f"selection_{fam}.json")
    if fam.startswith("l1"):
        add(l1_job(final_names[fam], fam.split("_")[1], "full", [r["best"]], prio=6, cleanup=True))
    elif fam == "joint":
        add(tv_job(final_names[fam], "joint", None, "full", [r["best"]], prio=6))
    else:
        add(tv_job(final_names[fam], "cs", fam.split("_")[1], "full", [r["best"]], prio=6))


def my_busy_gpus():
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout
    uuid2idx, free = {}, {}
    for line in gpus.strip().splitlines():
        i, u, used, tot, util = (p.strip() for p in line.split(","))
        uuid2idx[u] = int(i)
        # a card another user is computing on is not free: a job there ran ~9x slower (2026-09-15, GPU 3)
        free[int(i)] = float(tot) - float(used) if float(util) < 30 else 0.0
    busy = set()
    for line in apps.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[1])
            with open(f"/proc/{pid}/status") as f:
                uid = next(int(l.split()[1]) for l in f if l.startswith("Uid:"))
        except (ValueError, OSError, StopIteration):
            continue
        if uid == UID and parts[0] in uuid2idx:
            busy.add(uuid2idx[parts[0]])
    return busy, free


prev_free = set()
while True:
    # ---- refresh states -----------------------------------------------------------------------------------
    for n, j in jobs.items():
        if j["state"] == "running":
            rc = j["proc"].poll() if j["proc"] is not None else (None if alive(j["pid"]) else 0)
            if rc is None:
                continue
            (JOBS / f"{n}.pid").unlink(missing_ok=True)
            if j["done"]():
                j["state"] = "done"; log(f"done {n} (gpu {j['gpu']})")
            else:
                j["tries"] += 1
                j["state"] = "pending" if j["tries"] < 2 else "failed"
                log(f"{'RETRY' if j['state'] == 'pending' else 'FAILED'} {n} rc={rc}; see {LOGS / (n + '.log')}")
        elif j["state"] == "pending":
            pf = JOBS / f"{n}.pid"
            if j["done"]():
                j["state"] = "done"; log(f"already complete: {n}")
            elif pf.exists() and alive(int(pf.read_text())):
                j.update(state="running", proc=None, pid=int(pf.read_text())); log(f"adopted running {n}")
    v = validation_ok()
    if v is not None and not v[0]:
        log(f"VALIDATION FAILED: ported ADMM kres {v[1]:.6f} vs archived {v[2]:.6f}; TV-SENSE jobs held")
        for n, j in jobs.items():
            if "validate_cs35_pre" in j["deps"] and j["state"] == "pending":
                j["state"] = "failed"
    if PLAN == "arms":
        for fam in final_names:
            plan_family(fam)
    elif PLAN == "spirit":
        for sc_ in ("pre", "post"):
            plan_spirit(sc_)
    finals = (["mcinr_full", "mcinr_fit"] + list(final_names.values())) if PLAN == "arms" else FINALS
    if all(jobs.get(f, {}).get("state") == "done" for f in finals):
        marker = "ALL_DONE" if PLAN == "arms" else f"ALL_DONE_{PLAN}"
        log("ALL DONE"); (LOGS / marker).write_text(time.strftime("%Y-%m-%d %H:%M:%S")); break
    if all(j["state"] in ("done", "failed") for j in jobs.values()) and not all(jobs.get(f, {}).get("state") == "done" for f in finals):
        missing = [f for f in finals if jobs.get(f, {}).get("state") != "done"]
        # selection may still add work on the next pass; stop only if nothing is pending or running twice in a row
        time.sleep(POLL)
        if PLAN == "arms":
            for fam in final_names:
                plan_family(fam)
        elif PLAN == "spirit":
            for sc_ in ("pre", "post"):
                plan_spirit(sc_)
        if all(j["state"] in ("done", "failed") for j in jobs.values()):
            log(f"STOPPED with unfinished finals {missing}"); break

    # ---- launch ---------------------------------------------------------------------------------------------
    ready = sorted((j for j in jobs.values() if j["state"] == "pending"
                    and all(jobs.get(d, {}).get("state") == "done" for d in j["deps"])
                    and (v is None or v[0] or "validate_cs35_pre" not in j["deps"])), key=lambda j: j["prio"])
    running = [j for j in jobs.values() if j["state"] == "running"]
    busy, free = my_busy_gpus()
    mine = {j["gpu"] for j in running if j["lane"] == "gpu"}
    cand = {g for g, f in free.items() if g not in busy and g not in mine and f >= 30000}
    stable = set(cand)
    prev_free = cand
    for j in ready:
        if j["lane"] == "gpu":
            if sum(r["lane"] == "gpu" for r in running) >= MAX_GPU or not stable:
                continue
            fits = sorted(c for c in stable if free[c] >= j.get("need", NEED_MB))
            if not fits:
                continue
            g = fits[0]; stable.discard(g)
        else:
            if sum(r["lane"] == "cpu" for r in running) >= MAX_CPU:
                continue
            g = -1
        lf = open(LOGS / f"{j['name']}.log", "a")
        lf.write(f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')} gpu {g} =====\n"); lf.flush()
        p = subprocess.Popen(j["cmd"](g), stdout=lf, stderr=subprocess.STDOUT, env=ENV, cwd=str(T), start_new_session=True)
        (JOBS / f"{j['name']}.pid").write_text(str(p.pid))
        j.update(state="running", proc=p, pid=p.pid, gpu=g)
        running.append(j)
        log(f"launched {j['name']} on {'gpu ' + str(g) if g >= 0 else 'cpu'} pid {p.pid}")
    time.sleep(POLL)
