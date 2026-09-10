#!/usr/bin/env python3
"""sched_af16.py -- run every arm of the AF16 study on a fixed set of GPUs, in strict priority order.

Why a scheduler: only GPUs 0 and 1 are available (2-7 belong to a concurrent autoresearch loop),
and another user's ~33 GB job sits on both, so the fifteen runs per subject share about 47 GB per
card. Each job declares the VRAM it needs. A card takes the FIRST queued job that fits now; if the
first job does not fit now but would fit on the card once my own jobs have drained, the card
waits for it (no lighter job jumps ahead -- that starved the MC-GS runs in the first version);
only a job that could not fit even on an empty card is skipped. Max 2 of my jobs per card.

The scheduler is restartable: a job whose result file exists is skipped, a job whose process is
already running (matched by its command signature) is adopted and only waited for.

Usage:  sched_af16.py [--subjects hyw,jyz] [--gpus 0,1]
Status: <first subject topic>/logs/sched_status.json, sched.log, DONE_SCHED when the queue is empty.
"""
import argparse, json, os, subprocess, time
from pathlib import Path

PY = "/home/congcongliu/anaconda3/envs/GSTMI/bin/python"
F = "/data0/congcong/code/GS/mcgs_af16_wt"
ENV = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID",
           TORCH_EXTENSIONS_DIR="/data0/congcong/code/GS/mcgs_af16_wt.torch_ext")
UID = os.getuid()
SUBJ = {
    "hyw": dict(W="/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_retro",
                cfg="t1flair_af16_hyw", exp="t1flair_af16"),
    "jyz": dict(W="/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou",
                cfg="t1flair_v3_af16_jyz", exp="t1flair_v3_af16"),
}
GS_ARGS = ("--iters 4000 --lambda_geo 0.10 --warmup 500 --max_radius 20 --kspace_high 2.2 "
           "--edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 --save_volumes")
# measured 2026-09-09 on this data: classical ~9.3 GB, ZS-SSDU ~15.8, INR ~16.4, MC-GS 30-35; DIP unmeasured
# wall-time estimates (min) from the AF12 runs of the same arms; used only to fill a draining card
EST = dict(classical=10, gs=30, zsssdu=240, dip=60, inr=75)   # uncontended card (the other user left at 01:29)
# another user's training jobs restart periodically on GPUs 0/1 and reach ~33 GB; a transient low
# reading (17 GB between restarts) once let two MC-GS runs onto one card. Never assume less than this.
OTHERS_FLOOR = 10.0   # lowered 04:05 after 2.5 h without the other user; a failed job is retried once


def est_min(name):
    return EST[name.split(":")[1].split("_")[0]]


def jobs_for(s):
    W, cfg, exp = SUBJ[s]["W"], SUBJ[s]["cfg"], SUBJ[s]["exp"]
    S = f"{W}/scripts"
    J = []   # dict(name, need, cwd, cmd, sig, done)
    for c in ("T1", "FLAIR"):
        J.append(dict(name=f"{s}:classical_{c}", need=12, cwd=W,
                      cmd=f"{PY} -u {S}/recon_classical.py --contrast {c} --gpu {{gpu}}",
                      sig=f"{S}/recon_classical.py --contrast {c} ", done=f"{W}/logs/classical_{c}.json"))
    for arm, cfgsuf in (("joint", ""), ("single_T1", "_singleT1"), ("single_FLAIR", "_singleFLAIR")):
        J.append(dict(name=f"{s}:gs_{arm}", need=38, cwd=F,
                      cmd=f"{PY} -u arc/cycle.py --exp {exp}_{arm} --gpu {{gpu}} {GS_ARGS} --base_cfg configs/{cfg}{cfgsuf}.yaml",
                      sig=f"--exp {exp}_{arm} ", done=f"{F}/results_arc/{exp}_{arm}_score.json"))
    for c in ("T1", "FLAIR"):
        J.append(dict(name=f"{s}:zsssdu_{c}", need=18, cwd=W,
                      cmd=f"{PY} -u {S}/recon_zsssdu.py --contrast {c} --gpu {{gpu}} --epochs 100",
                      sig=f"{S}/recon_zsssdu.py --contrast {c} ", done=f"{W}/logs/zsssdu_{c}.json"))
    for c in ("T1", "FLAIR"):
        J.append(dict(name=f"{s}:dip_{c}", need=45, cwd=W,
                      cmd=f"{PY} -u {S}/recon_dip_inr.py --method dip --contrast {c} --gpu {{gpu}} --iters 5000 --lr 3e-3 --eval_every 100",
                      sig=f"{S}/recon_dip_inr.py --method dip --contrast {c} ", done=f"{W}/logs/dip_{c}.json"))
    for c in ("T1", "FLAIR"):
        J.append(dict(name=f"{s}:inr_{c}", need=18, cwd=W,
                      cmd=f"{PY} -u {S}/recon_dip_inr.py --method inr --contrast {c} --gpu {{gpu}} --iters 1500 --eval_every 50",
                      sig=f"{S}/recon_dip_inr.py --method inr --contrast {c} ", done=f"{W}/logs/inr_{c}.json"))
    return J


def gpu_state():
    """per GPU index: used/total (GB), memory held by OTHER users, my CUDA processes [(pid, gb, args)]."""
    bus2idx, st = {}, {}
    for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,pci.bus_id,memory.used,memory.total",
                                         "--format=csv,noheader,nounits"], text=True).strip().splitlines():
        i, b, u, t = [x.strip() for x in line.split(",")]
        bus2idx[b] = int(i); st[int(i)] = dict(used=float(u) / 1024, total=float(t) / 1024, others=0.0, mine=[])
    out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id,used_memory",
                                   "--format=csv,noheader,nounits"], text=True)
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid, bus, mem = [x.strip() for x in line.split(",")]
        g = bus2idx.get(bus)
        if g is None:
            continue
        try:
            owner = os.stat(f"/proc/{pid}").st_uid
            args = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except FileNotFoundError:
            continue
        if owner == UID:
            st[g]["mine"].append((int(pid), float(mem) / 1024, args))
        else:
            st[g]["others"] += float(mem) / 1024
    return st


def my_procs():
    """[(pid, elapsed_s, args)] for my processes (the python ones, not their sh wrappers)."""
    out = subprocess.check_output(["ps", "-u", str(UID), "-o", "pid=,etimes=,args="], text=True)
    res = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and not parts[2].startswith(("/bin/sh", "sh ", "/bin/zsh", "bash ")):
            res.append((int(parts[0]), float(parts[1]), parts[2]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="hyw,jyz")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--max_per_gpu", type=int, default=2)
    ap.add_argument("--settle_s", type=int, default=240)   # a fresh launch is assumed to hold its need_gb (and a slot) for this long
    ap.add_argument("--poll_s", type=int, default=45)
    ap.add_argument("--core_first", type=int, default=1)
    ap.add_argument("--others_floor", type=float, default=OTHERS_FLOOR)   # 1: classical/MC-GS/ZS-SSDU of every subject before any DIP/INR
    a = ap.parse_args()
    gpus = [int(x) for x in a.gpus.split(",")]
    queue = [j for s in a.subjects.split(",") for j in jobs_for(s)]
    if a.core_first:
        # under the current contention DIP / INR (3+ dB behind at AF12, never competitive) would otherwise
        # sit between the two subjects' informative arms; run every subject's core first
        core = [j for j in queue if not j["name"].split(":")[1].startswith(("dip", "inr"))]
        rest = [j for j in queue if j not in core]
        queue = core + rest
    status_dir = Path(SUBJ[a.subjects.split(",")[0]]["W"]) / "logs"
    slog = open(status_dir / "sched.log", "a")

    def say(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True); slog.write(line + "\n"); slog.flush()

    def log_path(name):
        s, arm = name.split(":")
        return Path(SUBJ[s]["W"]) / "logs" / f"{arm}.log"

    launched = []       # my launches: dict(name, gpu, need, proc, t0)
    retried = set()
    adopted = {}        # name -> gpu (running before this scheduler started; waited for via its done file / signature)
    done = {}           # name -> rc ('adopted' for jobs this scheduler did not start)
    last_wait = {}
    say(f"v2 queue: {[j['name'] for j in queue]}  gpus={gpus}")
    while True:
        # reap my launches
        for r in list(launched):
            rc = r["proc"].poll()
            if rc is not None:
                launched.remove(r); done[r["name"]] = rc
                say(f"DONE {r['name']} gpu{r['gpu']} rc={rc} {(time.time()-r['t0'])/60:.1f} min")
                j = next(x for x in ALL if x["name"] == r["name"])
                if rc != 0 and not Path(j["done"]).exists() and r["name"] not in retried:
                    retried.add(r["name"]); queue.insert(0, j)
                    say(f"RETRY {r['name']} queued once more (rc={rc}, no result file)")
        procs = my_procs()
        # adopted jobs: finished when their signature disappears
        for name, info in list(adopted.items()):
            j = next(x for x in ALL if x["name"] == name)
            if not any(j["sig"] in c for _, _, c in procs):
                del adopted[name]; done[name] = "adopted"
                say(f"DONE {name} (adopted run finished after ~{(time.time()-info['t0'])/60:.0f} min; "
                    f"result {'present' if Path(j['done']).exists() else 'MISSING'})")
        # drop finished / adopt running from the queue
        for j in list(queue):
            if Path(j["done"]).exists():
                queue.remove(j); done[j["name"]] = "skip:done"; say(f"SKIP {j['name']} (result exists)")
            else:
                hit = next(((pid, el) for pid, el, c in procs if j["sig"] in c), None)
                if hit:
                    queue.remove(j); adopted[j["name"]] = dict(pid=hit[0], t0=time.time() - hit[1])
                    say(f"ADOPT {j['name']} (already running {hit[1]/60:.0f} min, pid {hit[0]})")
        if not queue and not launched and not adopted:
            break
        try:
            st = gpu_state()
        except Exception as e:
            say(f"nvidia-smi failed: {e}"); time.sleep(a.poll_s); continue
        for g in gpus:
            s = st[g]
            others_eff = max(s["others"], a.others_floor)
            my_vis = {p: (m, args) for p, m, args in s["mine"]}          # my CUDA processes on this card
            fresh = [r for r in launched if r["gpu"] == g and time.time() - r["t0"] < a.settle_s]
            # a fresh launch is matched to its CUDA process by command signature (Popen's pid is the sh wrapper);
            # until it holds what it asked for, the difference stays reserved
            reserve, n_invisible = 0.0, 0
            for r in fresh:
                hit = next((m for m, args in my_vis.values() if r["sig"] in args), None)
                if hit is None:
                    reserve += r["need"]; n_invisible += 1
                else:
                    reserve += max(0.0, r["need"] - hit)
            n_slots = len(my_vis) + n_invisible
            if n_slots >= a.max_per_gpu:
                continue
            used_eff = max(s["used"], others_eff + sum(m for m, _ in my_vis.values()))
            free_now = s["total"] - used_eff - reserve
            free_if_empty = s["total"] - others_eff
            # remaining minutes of my jobs on this card (launched: known t0; adopted: pid -> CUDA process on g)
            remaining = [max(0.0, est_min(r["name"]) - (time.time() - r["t0"]) / 60) for r in launched if r["gpu"] == g]
            for name, info in adopted.items():
                if info["pid"] in my_vis:
                    remaining.append(max(0.0, est_min(name) - (time.time() - info["t0"]) / 60))
            drain_eta = max(remaining) if remaining else 0.0

            def launch(j, tag):
                queue.remove(j)
                lp = log_path(j["name"]); lp.parent.mkdir(parents=True, exist_ok=True)
                proc = subprocess.Popen(j["cmd"].format(gpu=g), shell=True, cwd=j["cwd"], env=ENV,
                                        stdout=open(lp, "a"), stderr=subprocess.STDOUT)
                launched.append(dict(name=j["name"], gpu=g, need=j["need"], proc=proc, t0=time.time(), sig=j["sig"]))
                say(f"{tag} {j['name']} on gpu{g} (free {free_now:.0f} GB, others {s['others']:.0f}->{others_eff:.0f}, drain eta {drain_eta:.0f} min) -> {lp}")
                last_wait.pop(g, None)

            for j in queue:
                if j["need"] <= free_now:
                    launch(j, "START"); break
                if j["need"] <= free_if_empty:
                    # head of the line must wait for this card to drain; only a job that fits now AND
                    # finishes before the drain may use the idle slot meanwhile
                    filler = next((k for k in queue if k is not j and k["need"] <= free_now
                                   and (drain_eta == 0.0 or est_min(k["name"]) + 3 <= drain_eta)), None)
                    if filler:
                        launch(filler, f"FILL(while draining for {j['name']})")
                    else:
                        msg = (f"WAIT gpu{g} drains for {j['name']} (need {j['need']}, free now {free_now:.0f}, "
                               f"free if empty {free_if_empty:.0f}, drain eta {drain_eta:.0f} min)")
                        if last_wait.get(g) != msg:
                            say(msg); last_wait[g] = msg
                    break
                # cannot fit even on an empty card while the other user is there: try the next job
        json.dump(dict(time=time.strftime("%m-%d %H:%M:%S"),
                       running=[dict(name=r["name"], gpu=r["gpu"], min=round((time.time()-r["t0"])/60, 1)) for r in launched],
                       adopted=list(adopted), queued=[j["name"] for j in queue], done=done,
                       gpus={g: dict(used=round(st[g]["used"], 1), others=round(st[g]["others"], 1), mine=len(st[g]["mine"])) for g in gpus}),
                  open(status_dir / "sched_status.json", "w"), indent=1)
        time.sleep(a.poll_s)
    fails = [n for n, rc in done.items() if isinstance(rc, int) and rc != 0]
    say(f"ALL DONE. failures: {fails}")
    (status_dir / "DONE_SCHED").write_text(json.dumps(done, indent=1))


ALL = [j for s in SUBJ for j in jobs_for(s)]

if __name__ == "__main__":
    main()
