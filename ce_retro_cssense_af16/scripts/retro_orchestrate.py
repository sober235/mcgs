#!/usr/bin/env python3
"""retro_orchestrate.py -- run every arm of the retrospective AF16 study, then summarise. Plan name: retro16.
GPU lane: a card is taken when a poll finds no compute process of this user on it, utilisation < 30 % and >= need MB free;
at most MAX_GPU jobs at once, one per card. Restartable: finished work is recognised from its output files; pids in logs/orch/jobs/.
Lambda families extend one step (x1.4 / /1.4, at most 2 times per scan) when the best value sits on a grid edge.
Before launching anything, a job whose command (script + key arguments, GPU index ignored) already matches a running
process of this user is skipped -- 2026-09-17: MC-GS joint, GS-single pre/post, ZS-SSDU pre/post and 3D-DIP pre were
started by hand before this scheduler existed, so their output files do not exist yet but the jobs must not be duplicated.
  cd $TOPIC && setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID python scripts/retro_orchestrate.py > logs/orch/orchestrate.log 2>&1 &
  python scripts/retro_orchestrate.py --dry      # print the job graph and exit"""
import json, os, re, subprocess, sys, time
from pathlib import Path

PY = "/home/congcongliu/anaconda3/envs/GSTMI/bin/python"
GRID = {"cs": [2.5e-4, 3.5e-4, 5e-4, 7e-4, 1e-3, 1.4e-3, 2e-3], "joint": [5e-4, 7e-4, 1e-3, 1.4e-3, 2e-3],
        "spirit_lam": [0.0, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3], "spirit_ks": [3, 5]}
NEED = dict(cls=30000, cs=30000, joint=45000, spirit=30000, zs=40000, dip=40000, inr=40000, mcinr=20000, gs=60000)
MAX_GPU, POLL, MAX_EXT = 3, 20, 2
SCANS = ("pre", "post")


def lam_s(v):
    return f"{v:g}"


def build_jobs(T):
    T = Path(T); S = T / "scripts"
    J = []
    def job(name, kind, cmd, done, deps=(), prio=5):
        J.append(dict(name=name, lane="gpu", prio=prio, need=NEED[kind], deps=list(deps), cmd=cmd, done=done))
    for s in SCANS:
        job(f"cls_{s}", "cls", lambda g, s=s: [PY, "-u", str(S / "recon_classical.py"), "--scan", s, "--methods", "zf,cgsense", "--gpu", str(g)],
            lambda s=s: (T / f"logs/classical_{s}.json").exists(), prio=0)
        for lam in GRID["cs"]:
            job(f"cs_{s}_lam{lam_s(lam)}", "cs", lambda g, s=s, lam=lam: [PY, "-u", str(S / "recon_classical.py"), "--scan", s, "--methods", "cssense",
                "--admm_outer", "200", "--lam", lam_s(lam), "--suffix", f"_o200_lam{lam_s(lam)}", "--gpu", str(g)],
                lambda s=s, lam=lam: (T / f"logs/classical_{s}_o200_lam{lam_s(lam)}.json").exists(), prio=1)
        for ks in GRID["spirit_ks"]:
            for lam in GRID["spirit_lam"]:
                job(f"spirit_{s}_ks{ks}_lam{lam_s(lam)}", "spirit", lambda g, s=s, ks=ks, lam=lam: [PY, "-u", str(S / "recon_spirit.py"), "--scan", s, "--gpu", str(g),
                    "--ksize", str(ks), "--lam", lam_s(lam), "--suffix", f"_ks{ks}_lam{lam_s(lam)}"],
                    lambda s=s, ks=ks, lam=lam: (T / f"logs/spirit_{s}_ks{ks}_lam{lam_s(lam)}.json").exists(), prio=2)
        job(f"zs_{s}", "zs", lambda g, s=s: [PY, "-u", str(S / "recon_zsssdu.py"), "--scan", s, "--gpu", str(g), "--epochs", "100"],
            lambda s=s: (T / f"logs/zsssdu_{s}.json").exists(), prio=3)
        job(f"dip_{s}", "dip", lambda g, s=s: [PY, "-u", str(S / "recon_dip_inr.py"), "--method", "dip", "--scan", s, "--gpu", str(g), "--iters", "5000", "--lr", "3e-3", "--eval_every", "100"],
            lambda s=s: (T / f"logs/dip_{s}.json").exists(), prio=3)
        job(f"inr_{s}", "inr", lambda g, s=s: [PY, "-u", str(S / "recon_dip_inr.py"), "--method", "inr", "--scan", s, "--gpu", str(g), "--iters", "1500", "--eval_every", "50"],
            lambda s=s: (T / f"logs/inr_{s}.json").exists(), prio=3)
        job(f"gs_single_{s}", "gs", lambda g, s=s: ["env", f"SINGLE={s}", "bash", str(S / "run_gs.sh"), f"ce_retro_af16_single_{s}", str(g), "4000"],
            lambda s=s: (T / f"trainer/_trainer_e778bf2/results_arc/ce_retro_af16_single_{s}_volumes.npz").exists(), prio=4)
        job(f"extract_single_{s}", "cls", lambda g, s=s: [PY, "-u", str(S / "extract_gs_volumes.py"), "--exp", f"ce_retro_af16_single_{s}", "--key", "gs_single", "--map", f"{s}:plain", "--gpu", str(g)],
            lambda s=s: (T / f"logs/gs_single_{s}.json").exists(), deps=[f"gs_single_{s}"], prio=4)
    for lam in GRID["joint"]:
        job(f"joint_lam{lam_s(lam)}", "joint", lambda g, lam=lam: [PY, "-u", str(S / "recon_joint_cssense.py"), "--topic", str(T), "--contrasts", "pre,post",
            "--admm_outer", "200", "--lam", lam_s(lam), "--suffix", f"_o200_lam{lam_s(lam)}", "--gpu", str(g)],
            lambda lam=lam: (T / f"logs/jcssense_o200_lam{lam_s(lam)}.json").exists(), prio=1)
    job("mcinr", "mcinr", lambda g: [PY, "-u", str(S / "run_mcinr.py"), "--gpu", str(g)],
        lambda: all((T / f"logs/mcinr_{s}.json").exists() for s in SCANS), prio=3)
    job("gs_joint", "gs", lambda g: ["bash", str(S / "run_gs.sh"), "ce_retro_af16_joint", str(g), "4000"],
        lambda: (T / "trainer/_trainer_e778bf2/results_arc/ce_retro_af16_joint_volumes.npz").exists(), prio=4)
    job("extract_joint", "cls", lambda g: [PY, "-u", str(S / "extract_gs_volumes.py"), "--exp", "ce_retro_af16_joint", "--key", "gs_joint", "--map", "pre:plain,post:contrast", "--gpu", str(g)],
        lambda: all((T / f"logs/gs_joint_{s}.json").exists() for s in SCANS), deps=["gs_joint"], prio=4)
    arms = [j["name"] for j in J]
    J.append(dict(name="summarize", lane="cpu", prio=9, need=0, deps=arms, cmd=lambda g: [PY, "-u", str(S / "summarize.py")],
                  done=lambda: (T / "logs/orch/summarized").exists()))
    return J


def _best_lam(T, fam, scan):
    """(best lambda, sorted lambdas seen) of a swept family from its jsons; None if nothing finished."""
    pat = {"cs": f"logs/classical_{scan}_o200_lam*.json", "joint": "logs/jcssense_o200_lam*.json"}[fam]
    cands = []
    for f in sorted(Path(T).glob(pat)):
        for r in json.load(open(f)).get("rows", []):
            if r.get("contrast", scan) == scan and r.get("head_psnr") is not None:
                cands.append((float(r["lam"]), float(r["head_psnr"])))
    if not cands:
        return None, []
    lams = sorted({l for l, _ in cands})
    return max(cands, key=lambda t: t[1])[0], lams


def plan_extensions(T, jobs):
    """new jobs for lambda families whose best sits on the grid edge (x1.4 up / /1.4 down, MAX_EXT per family+scan)."""
    T = Path(T); S = T / "scripts"; new = []
    for fam, scans in (("cs", SCANS), ("joint", ("pre", "post"))):
        base = set(GRID[fam])
        for scan in scans:
            prefix = f"cs_{scan}_lam" if fam == "cs" else "joint_lam"
            pending = [n for n, j in jobs.items() if n.startswith(prefix) and j["state"] != "done"]
            if pending:
                continue
            best, lams = _best_lam(T, fam, scan)
            if best is None or len(set(lams) - base) >= MAX_EXT * (2 if fam == "joint" else 1):
                continue
            if best == max(lams):
                lam = float(f"{best * 1.4:.2g}")
            elif best == min(lams):
                lam = float(f"{best / 1.4:.2g}")
            else:
                continue
            name = f"{prefix}{lam_s(lam)}"
            if name in jobs or any(j["name"] == name for j in new):
                continue
            if fam == "cs":
                cmd = lambda g, s=scan, lam=lam: [PY, "-u", str(S / "recon_classical.py"), "--scan", s, "--methods", "cssense", "--admm_outer", "200",
                                                  "--lam", lam_s(lam), "--suffix", f"_o200_lam{lam_s(lam)}", "--gpu", str(g)]
                done = lambda s=scan, lam=lam: (T / f"logs/classical_{s}_o200_lam{lam_s(lam)}.json").exists()
            else:
                cmd = lambda g, lam=lam: [PY, "-u", str(S / "recon_joint_cssense.py"), "--topic", str(T), "--contrasts", "pre,post", "--admm_outer", "200",
                                          "--lam", lam_s(lam), "--suffix", f"_o200_lam{lam_s(lam)}", "--gpu", str(g)]
                done = lambda lam=lam: (T / f"logs/jcssense_o200_lam{lam_s(lam)}.json").exists()
            new.append(dict(name=name, lane="gpu", prio=1, need=NEED[fam], deps=[], cmd=cmd, done=done))
    return new


def log(*a):
    print(time.strftime("%m-%d %H:%M:%S"), *a, flush=True)


def alive(pid):
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


jobs = {}


def add(j):
    if j["name"] not in jobs:
        done = j["done"]()                  # a job queued by plan_extensions after a restart may already be complete
        pf = JOBS / f"{j['name']}.pid"
        live = (not done) and pf.exists() and alive(int(pf.read_text()))   # still running from a previous scheduler
        jobs[j["name"]] = dict(j, state="done" if done else ("running" if live else "pending"), proc=None, gpu=None,
                               tries=0, pid=int(pf.read_text()) if live else None)
        log(f"{'already complete' if done else ('adopted running' if live else 'queued')} {j['name']} ({j['lane']}, prio {j['prio']})")


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


def _job_pattern(cmd):
    """The command's script + arguments as a pgrep -f regex: basename'd (a hand launch may use a relative script
    path where this builds an absolute one) starting at the script itself (drop the interpreter/wrapper before it,
    which a hand launch may not share), with whichever token selects the GPU wildcarded (a hand launch may have
    used a different card)."""
    toks = [str(t) for t in cmd]
    i = next((k for k, t in enumerate(toks) if "/" in t and (t.endswith(".py") or t.endswith(".sh"))), 0)
    toks = toks[i:]
    toks[0] = Path(toks[0]).name
    if "--gpu" in toks:
        toks[toks.index("--gpu") + 1] = None
    elif toks[0] == "run_gs.sh":
        toks[2] = None                                  # bash run_gs.sh EXP GPU ITERS: GPU is positional
    return " ".join(r"[0-9]+" if t is None else re.escape(t) for t in toks)


def already_running(cmd):
    """True if a process of this user matching this job's command is already alive, whichever GPU it is on: the
    production arms of 2026-09-17 were started by hand outside this scheduler, before any output file exists, so
    `done()` alone would not stop a duplicate launch."""
    pattern = _job_pattern(cmd)
    if not pattern:
        return False
    out = subprocess.run(["pgrep", "-u", str(os.getuid()), "-af", pattern], capture_output=True, text=True).stdout
    me = os.getpid()
    return any(tok.isdigit() and int(tok) != me for tok in (line.split(None, 1)[0] for line in out.strip().splitlines()))


if __name__ == "__main__":
    T = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
    SC = T / "scripts"
    LOGS = T / "logs" / "orch"
    JOBS = LOGS / "jobs"

    if "--dry" in sys.argv:
        for j in build_jobs(T):
            print(j["name"], j["lane"], j["need"], j["deps"], " ".join(j["cmd"](0)))
        sys.exit(0)

    JOBS.mkdir(parents=True, exist_ok=True)
    ENV = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    UID = os.getuid()

    for j in build_jobs(T):
        add(j)

    while True:
        # ---- refresh states -----------------------------------------------------------------------------------
        for n, j in jobs.items():
            if j["state"] == "running":
                rc = j["proc"].poll() if j["proc"] is not None else (None if alive(j["pid"]) else 0)
                if rc is None:
                    continue
                (JOBS / f"{n}.pid").unlink(missing_ok=True)
                if n == "summarize" and rc == 0 and (T / "logs/summary.json").exists():
                    (LOGS / "summarized").touch()
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

        for j in plan_extensions(T, jobs):
            add(j)
        jobs["summarize"]["deps"] = [n for n in jobs if n != "summarize"]

        if jobs["summarize"]["state"] == "done":
            log("ALL DONE"); (LOGS / "ALL_DONE_retro16").write_text(time.strftime("%Y-%m-%d %H:%M:%S")); break

        if all(j["state"] in ("done", "failed") for j in jobs.values()) and jobs["summarize"]["state"] != "done":
            time.sleep(POLL)
            for j in plan_extensions(T, jobs):
                add(j)
            jobs["summarize"]["deps"] = [n for n in jobs if n != "summarize"]
            if all(j["state"] in ("done", "failed") for j in jobs.values()) and jobs["summarize"]["state"] != "done":
                failed = [n for n, j in jobs.items() if j["state"] == "failed"]
                log(f"STOPPED with unfinished work; failed: {failed}"); break

        # ---- launch ---------------------------------------------------------------------------------------------
        ready = sorted((j for j in jobs.values() if j["state"] == "pending"
                        and all(jobs.get(d, {}).get("state") == "done" for d in j["deps"])), key=lambda j: j["prio"])
        running = [j for j in jobs.values() if j["state"] == "running"]
        busy, free = my_busy_gpus()
        mine = {j["gpu"] for j in running if j["lane"] == "gpu"}
        stable = {g for g, f in free.items() if g not in busy and g not in mine}
        for j in ready:
            if already_running(j["cmd"](0)):
                log(f"skip {j['name']}: a matching process is already running outside the scheduler")
                continue
            if j["lane"] == "gpu":
                if sum(r["lane"] == "gpu" for r in running) >= MAX_GPU or not stable:
                    continue
                fits = sorted(c for c in stable if free[c] >= j["need"])
                if not fits:
                    continue
                g = fits[0]; stable.discard(g)
            else:
                g = -1
            lf = open(LOGS / f"{j['name']}.log", "a")
            lf.write(f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')} gpu {g} =====\n"); lf.flush()
            p = subprocess.Popen(j["cmd"](g), stdout=lf, stderr=subprocess.STDOUT, env=ENV, start_new_session=True)
            (JOBS / f"{j['name']}.pid").write_text(str(p.pid))
            j.update(state="running", proc=p, pid=p.pid, gpu=g)
            running.append(j)
            log(f"launched {j['name']} on {'gpu ' + str(g) if g >= 0 else 'cpu'} pid {p.pid}")
        time.sleep(POLL)
