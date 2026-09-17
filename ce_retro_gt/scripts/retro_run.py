#!/usr/bin/env python3
"""retro_run.py -- reconstruct the simulated pre/post cases with the same arms as the prospective topic (2026-09-16).

Every arm runs the script of Results_20260908/CE_pre_post_contrast, pointed at a simulated case through the
CE_WORK_PRE / CE_WORK_POST (and CE_GS_PRE / CE_GS_POST) environment variables, so the code path is identical to the
prospective work and only the data differ. One job per card, a card is taken only when no process of this user is on
it and its utilisation is below 30% (the house rule from 2026-09-15), and finished work is recognised from its meta
file, so the runner is restartable.

  setsid nohup python scripts/retro_run.py --cases UCSF-PDGM-0004,... > logs/run.log 2>&1 &
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

T = Path(__file__).resolve().parents[1]
CE = Path("/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/scripts")
PY = "/home/congcongliu/anaconda3/envs/GSTMI/bin/python"
ap = argparse.ArgumentParser()
ap.add_argument("--cases", default="UCSF-PDGM-0004")
ap.add_argument("--arms", default="zf,cgsense,cssense,dip,inr,gs_single,gs_joint")
ap.add_argument("--lam", type=float, default=2e-3, help="CS-SENSE lambda; 2e-3 is the prospective held-out choice")
ap.add_argument("--max_gpu", type=int, default=2)
ap.add_argument("--exclude", default="1,2,5",
                help="cards to leave alone; 1,2,5 belong to the jyz AF16 noise study (peer session, 2026-09-17)")
ap.add_argument("--lockglob", default="", help="glob of lock files like .gpulock.<index>; a card whose lock holds a "
                                               "live pid is skipped as well")
ap.add_argument("--poll", type=int, default=15)
a = ap.parse_args()
CASES = a.cases.split(",")
ARMS = a.arms.split(",")
LOGS = T / "logs"
LOGS.mkdir(exist_ok=True)
JOBS = LOGS / "jobs"
JOBS.mkdir(exist_ok=True)


def live(name):
    f = JOBS / f"{name}.pid"
    try:
        os.kill(int(f.read_text()), 0)
        return True
    except (ValueError, OSError):
        return False
UID = os.getuid()
EXCLUDE = {int(v) for v in a.exclude.split(",") if v.strip()}
log = lambda *m: print(time.strftime("%m-%d %H:%M:%S"), *m, flush=True)


def env_for(case):
    s = T / "sim" / case
    return dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                CE_WORK_PRE=str(s / "work_pre"), CE_WORK_POST=str(s / "work_post"),
                CE_GS_PRE=str(s / "gs_pre"), CE_GS_POST=str(s / "gs_post"))


def jobs_for(case):
    R = T / "recon" / case
    out = []
    for scan in ("pre", "post"):
        for m in ("zf", "cgsense"):
            if m in ARMS:
                out.append((f"{m}_{scan}_{case}", [PY, "-u", str(CE / "ce_classical.py"), "--method", m,
                                                   "--scan", scan, "--gpu", "{gpu}", "--out", str(R)],
                            R / f"{m}_{scan}_meta.json", 30000, case))
        if "cssense" in ARMS:
            out.append((f"cssense_{scan}_{case}",
                        [PY, "-u", str(CE / "ce_tv_sense.py"), "--mode", "cs", "--scan", scan, "--mask", "full",
                         "--lams", f"{a.lam:g}", "--outer", "200", "--gpu", "{gpu}", "--out", str(R / "tv")],
                        R / "tv" / f"cssense_o200_lam{a.lam:g}_full_{scan}_meta.json", 40000, case))
        for m, iters, extra in (("dip", "5000", ["--lr", "3e-3", "--eval_every", "100"]),
                                ("inr", "1500", ["--eval_every", "50"])):
            if m in ARMS:
                out.append((f"{m}_{scan}_{case}",
                            [PY, "-u", str(CE / "ce_dip_inr.py"), "--method", m, "--scan", scan, "--iters", iters,
                             *extra, "--gpu", "{gpu}", "--out", str(R)],
                            R / f"{m}_{scan}_meta.json", 40000, case))
    if "gs_single" in ARMS:
        for mode in ("single_pre", "single_post"):
            out.append((f"gs_{mode}_{case}",
                        [PY, "-u", str(CE / "ce_gs.py"), "--mode", mode, "--gpu", "{gpu}", "--out", str(R / f"gs_{mode}")],
                        R / f"gs_{mode}" / f"gs_{mode}_meta.json", 60000, case))
    if "gs_joint" in ARMS:
        out.append((f"gs_joint_{case}",
                    [PY, "-u", str(CE / "ce_gs.py"), "--mode", "joint", "--gpu", "{gpu}", "--out", str(R / "gs_joint")],
                    R / "gs_joint" / "gs_joint_pre_meta.json", 60000, case))
    return out


def locked_gpus():
    """cards another scheduler has claimed with a lock file holding a live pid."""
    out = set()
    if not a.lockglob:
        return out
    import glob
    for f in glob.glob(a.lockglob):
        q = Path(f)
        try:
            idx = int(q.name.split(".")[-1])
            # the cardiac scheduler's lock is a directory made with mkdir, with the owner pid in <dir>/pid
            # (peer session claude-71, 2026-09-17); a plain file holding the pid also works
            pid = int((q / "pid" if q.is_dir() else q).read_text().split()[0])
            os.kill(pid, 0)
            out.add(idx)
        except (ValueError, OSError, IndexError):
            continue                                    # a stale lock whose owner is gone does not hold the card
    return out


def free_gpus():
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    uuid2idx, free = {}, {}
    for line in gpus.strip().splitlines():
        i, u, used, tot, util = (p.strip() for p in line.split(","))
        uuid2idx[u] = int(i)
        free[int(i)] = float(tot) - float(used) if float(util) < 30 else 0.0   # a busy card runs ~9x slower
    for line in apps.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            with open(f"/proc/{parts[1]}/status") as f:
                uid = next(int(l.split()[1]) for l in f if l.startswith("Uid:"))
        except (ValueError, OSError, StopIteration):
            continue
        if uid == UID and parts[0] in uuid2idx:
            free[uuid2idx[parts[0]]] = 0.0                                     # one job per card
    for g in EXCLUDE | locked_gpus():
        free[g] = 0.0                                                          # reserved for another session
    return free


todo = [j for c in CASES for j in jobs_for(c)]
done = [j for j in todo if j[2].exists()]
todo = [j for j in todo if not j[2].exists()]
adopted = [j for j in todo if live(j[0])]
todo = [j for j in todo if not live(j[0])]
log(f"{len(done)} already done, {len(adopted)} still running from an earlier scheduler, {len(todo)} to run "
    f"over {len(CASES)} case(s); cards off limits: {sorted(EXCLUDE)}")
running = [dict(name=j[0], p=None, pid=int((JOBS / f"{j[0]}.pid").read_text()), gpu=None, meta=j[2], t0=time.time())
           for j in adopted]
while todo or running:
    for r in list(running):
        if (r["p"].poll() if r["p"] is not None else (None if live(r["name"]) else 0)) is not None:
            ok = r["meta"].exists()
            log(f"{'done' if ok else 'FAILED'} {r['name']} (gpu {r['gpu']}, {time.time() - r['t0']:.0f}s)")
            running.remove(r)
    free = free_gpus()
    taken = {r["gpu"] for r in running}
    while todo and len(running) < a.max_gpu:
        name, cmd, meta, need, case = todo[0]
        cand = sorted((g for g, f in free.items() if g not in taken and f >= need), key=lambda g: -free[g])
        if not cand:
            break
        g = cand[0]
        todo.pop(0)
        lf = open(LOGS / f"{name}.log", "a")
        lf.write(f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')} gpu {g} =====\n"); lf.flush()
        p = subprocess.Popen([c.replace("{gpu}", str(g)) for c in cmd], stdout=lf, stderr=subprocess.STDOUT,
                             env=env_for(case), cwd=str(T), start_new_session=True)
        (JOBS / f"{name}.pid").write_text(str(p.pid))
        running.append(dict(name=name, p=p, pid=p.pid, gpu=g, meta=meta, t0=time.time()))
        taken.add(g)
        log(f"launched {name} on gpu {g} pid {p.pid}")
    time.sleep(a.poll)
log("ALL DONE")
(LOGS / "ALL_DONE_retro").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
