#!/usr/bin/env python3
"""run_all_diffmask.py -- every arm of the jiangyizhou diff-mask study, both topics, queued onto free cards (2026-09-15).

The study varies ONE thing against T1FLAIR_AF16_v3_jiangyizhou (the house vd-Poisson mask, already reconstructed and
scored -- it is the comparison column, nothing is re-run there): the mask PATTERN. Two new topics, both at the same
2964-point budget (T1 15.20x) and the same 24 x 16 fully sampled block, both labelled AF 16 (nominal):

    gauss   T1FLAIR_AF16_gauss_v3_jiangyizhou    2-D Gaussian variable-density random
    radial  T1FLAIR_AF16_radial_v3_jiangyizhou   Cartesian pseudo-radial spokes

Arms, arguments and lambda grids are the AF16 topic's, extended where its grid had only two points (JTV), so the
winner can land in the interior. MC-GS and GS-single run the AF16 phase-3 recipe frozen (run_diffmask_mcgs.sh).
L1-ESPIRiT is NOT queued by default: the user dropped it from the comparison on 2026-09-15 (its data is kept in the
older topics, it is simply no longer listed). --with_l1espirit puts it back.

Scheduling rules, inherited from run_all_af24.py with one addition:
- One GPU job of this user per card. A card is eligible when no compute process of this uid is on it (other sessions
  count -- two trainers on one card hung the machine twice on 2026-09-11), no job of this queue is on it, and it has
  the job's memory free.
- ADDITION: the card's lock under Cardiac_FreeRunning_MCGS/work/.gpulock.<idx> must be acquirable (atomic mkdir).
  That is the lock the cardiac line's launch_gs_when_free.sh uses, so the two schedulers cannot pick the same card in
  the seconds before a new process appears in nvidia-smi. A lock whose recorded owner pid is dead protects nothing and
  is reaped here regardless of age (launch_gs_when_free.sh reaps the same locks, but only after 24 h, which left three
  cards fenced off by dead owners on 2026-09-15). The lock is released when the job is reaped.
- At most --max_gpu_jobs GPU jobs of this queue at once; L1-ESPIRiT (BART on the CPU) runs in --max_cpu_jobs CPU slots
  and takes no GPU slot.
- Priority interleaves the two topics so neither finishes hours after the other: the long MC-GS runs first, then the
  cheap classical arms, then the sweeps, then the learned arms, then L1-SPIRiT, then MC-INR.
- A job whose result file exists is skipped; a job already running outside this queue is waited for, not relaunched; a
  non-MC-GS job that exits non-zero without its result is retried once; a failed MC-GS run is only logged
  (run_diffmask_mcgs.sh refuses to overwrite its outputs, so it needs a look first).
- The "already running outside this queue" test compares the live process's WORKING DIRECTORY as well as its argv.
  run_all_af24.py compared argv alone, which is unique when a launcher drives one topic; driving two topics at once it
  is not -- gauss and radial run the same script with the same --contrast / --suffix, so on the first run (2026-09-15)
  46 radial jobs were classified as duplicates of the gauss job then running, moved to `waiting`, and marked
  EXTERNAL DONE result=NO without ever being launched. The cwd is what separates them (each topic job runs in its own
  topic directory; MC-GS runs in the worktree but its --exp carries the topic, and MC-INR's --tag does).
- A job that comes back from `waiting` with no result is re-queued once rather than being written off, so a
  mis-identification costs a re-run instead of a silently missing arm.
- When nothing is queued or running: finalize_diffmask.sh per topic.

Launch detached from a bash script file:
    setsid nohup PY -u scripts/run_all_diffmask.py > /dev/null 2>&1 < /dev/null &
Status: logs/run_all_diffmask.log, logs/run_all_diffmask_status.json, logs/RUN_ALL_DIFFMASK_DONE (in the gauss topic).
--dry_run prints the queue, the skips and the card eligibility once, and launches nothing.
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

RES = Path("/data0/congcong/code/GS/Results_20260908")
F = Path("/data0/congcong/code/GS/mcgs_diffmask_wt")
LOCK_ROOT = RES / "Cardiac_FreeRunning_MCGS/work"
PY = "/home/congcongliu/anaconda3/envs/GSTMI/bin/python"
PY_TCNN = "/home/congcongliu/anaconda3/envs/GR_pt1.10/bin/python"
MCINR = Path("/data0/congcong/code/GS/mcinr_niessen2025")
MCINR_TOPIC = RES / "T1FLAIR_MCINR_Niessen2025_v3_jiangyizhou"
ENV = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID")
UID = os.getuid()

KINDS = {"gauss": "af16g", "radial": "af16r"}
# the AF16 topic's own grids; JTV widened from its two points to the AF24 five so the winner can be interior
TV = {"T1": ["0.00035", "0.0005", "0.0007", "0.001", "0.0014"],
      "FLAIR": ["0.00025", "0.00035", "0.0005", "0.0007", "0.001"]}
JTV = ["0.00035", "0.0005", "0.0007", "0.001", "0.0014"]
L1 = {"T1": ["0.001", "0.0015", "0.0025"], "FLAIR": ["0.002", "0.003", "0.005"]}
SPIRIT_KS = ("3", "5")
SPIRIT_LAM = ("0", "0.0005", "0.001", "0.002", "0.004", "0.008")
CLASSICAL = ("base_", "tv_", "jtv_")


def topic(kind):
    return RES / f"T1FLAIR_AF16_{kind}_v3_jiangyizhou"


def jobs_for(kind, with_l1):
    """(priority tier, job) pairs; the caller interleaves the two topics by tier"""
    T, short, L, q = topic(kind), KINDS[kind], topic(kind) / "logs", []

    def add(tier, name, kind_, need, cwd, cmd, done, log, script, args, absent=(), **extra):
        q.append((tier, dict(name=f"{kind}/{name}", kind=kind_, need=need, cwd=str(cwd), cmd=cmd, done=str(done),
                             log=str(log), script=script, args=args, absent=list(absent), **extra)))

    for tier, arm, single in ((0, "joint", ""), (1, "singleFLAIR", "FLAIR"), (3, "singleT1", "T1")):
        exp = f"{short}_p3_{arm}_full4k"
        add(tier, f"gs_{arm}", "gpu", 38, F, f"SINGLE={single} SAVE_VOLUMES=1 ./run_diffmask_mcgs.sh {exp} {{gpu}} 4000",
            F / f"results_arc/{exp}_score.json", L / f"sched_gs_{arm}.log", "arc/cycle.py", {"--exp": exp})
    for c in ("T1", "FLAIR"):
        add(2, f"base_{c}", "gpu", 12, T, f"{PY} -u scripts/recon_classical.py --contrast {c} --gpu {{gpu}}",
            L / f"classical_{c}.json", L / f"classical_{c}.log", "recon_classical.py", {"--contrast": c},
            absent=["--suffix"])
    for c in ("T1", "FLAIR"):
        for lam in TV[c]:
            add(4, f"tv_{c}_{lam}", "gpu", 12, T,
                f"{PY} -u scripts/recon_classical.py --contrast {c} --gpu {{gpu}} --methods cssense --admm_outer 200 "
                f"--lam {lam} --suffix _o200_lam{lam}",
                L / f"classical_{c}_o200_lam{lam}.json", L / f"classical_{c}_o200_lam{lam}.log",
                "recon_classical.py", {"--contrast": c, "--suffix": f"_o200_lam{lam}"})
    for lam in JTV:
        add(5, f"jtv_{lam}", "gpu", 20, T,
            f"{PY} -u scripts/recon_joint_cssense.py --topic {T} --contrasts T1,FLAIR --gpu {{gpu}} --admm_outer 200 "
            f"--lam {lam} --suffix _o200_lam{lam}",
            L / f"jcssense_o200_lam{lam}.json", L / f"jcssense_o200_lam{lam}.log",
            "recon_joint_cssense.py", {"--suffix": f"_o200_lam{lam}"})
    for c in ("T1", "FLAIR"):
        add(6, f"zsssdu_{c}", "gpu", 18, T, f"{PY} -u scripts/recon_zsssdu.py --epochs 100 --contrast {c} --gpu {{gpu}}",
            L / f"zsssdu_{c}.json", L / f"zsssdu_{c}.log", "recon_zsssdu.py", {"--contrast": c})
    for c in ("T1", "FLAIR"):
        add(7, f"inr_{c}", "gpu", 18, T,
            f"{PY} -u scripts/recon_dip_inr.py --method inr --iters 1500 --eval_every 50 --contrast {c} --gpu {{gpu}}",
            L / f"inr_{c}.json", L / f"inr_{c}.log", "recon_dip_inr.py", {"--method": "inr", "--contrast": c})
    for c in ("T1", "FLAIR"):
        add(8, f"dip_{c}", "gpu", 40, T,
            f"{PY} -u scripts/recon_dip_inr.py --method dip --iters 5000 --lr 3e-3 --eval_every 100 --contrast {c} "
            f"--gpu {{gpu}}",
            L / f"dip_{c}.json", L / f"dip_{c}.log", "recon_dip_inr.py", {"--method": "dip", "--contrast": c})
    for c in ("T1", "FLAIR"):
        for ks in SPIRIT_KS:
            for lam in SPIRIT_LAM:
                sfx = f"_ks{ks}_lam{lam}"
                add(9, f"spirit_{c}{sfx}", "gpu", 24, T,
                    f"{PY} -u scripts/recon_spirit.py --contrast {c} --gpu {{gpu}} --ksize {ks} --lam {lam} "
                    f"--iters 150 --chunk 24 --suffix {sfx}",
                    L / f"spirit_{c}{sfx}.json", L / f"spirit/{c}{sfx}.log",
                    "recon_spirit.py", {"--contrast": c, "--suffix": sfx})
    add(10, "mcinr", "gpu", 10, MCINR,
        f"{PY_TCNN} -u scripts/run_volume.py --af 16 --tag _{kind} --gpu {{gpu}} --resume --yield_to_others "
        f"--mask_T1 {T}/data/mask_af16_3d.h5 --mask_FLAIR {T}/data/mask_af16_3d.h5",
        MCINR_TOPIC / f"recon/af16_{kind}/meta.json", L / f"mcinr_{kind}.log",
        "run_volume.py", {"--tag": f"_{kind}"}, requeue_rc=3)
    if with_l1:
        for c in ("T1", "FLAIR"):
            for lam in L1[c]:
                add(11, f"l1_{c}_{lam}", "cpu", 12, T,
                    f"{PY} -u scripts/recon_l1espirit.py --topic {T} --contrast {c} --gpu {{gpu}} --lam {lam} "
                    f"--iters 100 --suffix _lam{lam} --threads 12",
                    L / f"l1espirit_{c}_lam{lam}.json", L / f"l1espirit_{c}_lam{lam}.log",
                    "recon_l1espirit.py", {"--contrast": c, "--suffix": f"_lam{lam}"})
    return q


def build_queue(kinds, with_l1):
    tiered = []
    for k in kinds:
        tiered += jobs_for(k, with_l1)
    tiered.sort(key=lambda tj: tj[0])                      # stable: within a tier, topic order is preserved
    return [j for _, j in tiered]


# ---- the shared per-card lock (see the module docstring) --------------------------------------
def lock_path(g):
    return LOCK_ROOT / f".gpulock.{g}"


def owner_alive(d):
    try:
        pid = int((d / "pid").read_text().strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                                        # it exists, it just belongs to someone else
    except OSError:
        return False
    return True


def acquire(g, say):
    d = lock_path(g)
    if d.is_dir() and not owner_alive(d):
        say(f"reaping stale lock {d} (recorded owner not running)")
        shutil.rmtree(d, ignore_errors=True)
    try:
        d.mkdir()
    except (FileExistsError, OSError):
        return False
    return True


def release(g, say):
    d = lock_path(g)
    if d.is_dir():
        shutil.rmtree(d, ignore_errors=True)
        say(f"released lock {d}")


def my_procs():
    """(argv, resolved cwd) of every live process of this uid"""
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != UID:
                continue
            toks = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
            cwd = os.path.realpath(os.readlink(f"/proc/{pid}/cwd"))
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            continue
        out.append(([t.decode(errors="replace") for t in toks if t], cwd))
    return out


def matches(proc, j):
    argv, cwd = proc
    if cwd != os.path.realpath(j["cwd"]):       # the two topics share script names and flags; the cwd separates them
        return False
    if not any(t.endswith(j["script"]) for t in argv):
        return False
    for k, v in j["args"].items():
        if not any(argv[i] == k and argv[i + 1] == v for i in range(len(argv) - 1)):
            return False
    return not any(k in argv for k in j["absent"])


def gpu_state():
    st, bus2i = {}, {}
    for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,pci.bus_id,memory.free",
                                         "--format=csv,noheader,nounits"], text=True).strip().splitlines():
        i, b, fr = [x.strip() for x in line.split(",")]
        st[int(i)] = dict(free=float(fr) / 1024, mine=False)
        bus2i[b] = int(i)
    for line in subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id",
                                         "--format=csv,noheader"], text=True).strip().splitlines():
        if not line.strip():
            continue
        pid, b = [x.strip() for x in line.split(",")]
        try:
            if b in bus2i and os.stat(f"/proc/{pid}").st_uid == UID:
                st[bus2i[b]]["mine"] = True
        except (FileNotFoundError, OSError):
            pass
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", default="gauss,radial")
    ap.add_argument("--cards", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--max_gpu_jobs", type=int, default=4)
    ap.add_argument("--max_cpu_jobs", type=int, default=3)
    ap.add_argument("--poll_s", type=int, default=30)
    ap.add_argument("--with_l1espirit", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    kinds = [k.strip() for k in a.topics.split(",") if k.strip()]
    assert all(k in KINDS for k in kinds), kinds
    cards = [int(x) for x in a.cards.split(",")]
    G = topic("gauss")
    logf = None if a.dry_run else open(G / "logs/run_all_diffmask.log", "a")

    def say(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        if logf:
            logf.write(line + "\n")
            logf.flush()

    queue, running, waiting, done, retried = build_queue(kinds, a.with_l1espirit), [], [], {}, set()
    say(f"armed: {len(queue)} jobs over {kinds}, cards {cards}, max {a.max_gpu_jobs} GPU / {a.max_cpu_jobs} CPU jobs"
        + ("" if a.with_l1espirit else "; L1-ESPIRiT not queued (dropped from the comparison 2026-09-15)"))
    while True:
        for r in list(running):                                    # reap
            rc = r["proc"].poll()
            if rc is None:
                continue
            running.remove(r)
            j = r["job"]
            ok = Path(j["done"]).exists()
            done[j["name"]] = rc
            release(r["gpu"], say)
            say(f"DONE {j['name']} gpu{r['gpu']} rc={rc} result={'yes' if ok else 'NO'} "
                f"{(time.time() - r['t0']) / 60:.1f} min")
            if rc != 0 and not ok and rc == j.get("requeue_rc"):
                queue.insert(0, j)
                say(f"YIELDED {j['name']} (rc={rc}): queued again to resume")
            elif rc != 0 and not ok:
                if "/gs_" in j["name"]:
                    say(f"FAILED {j['name']}: not retried (run_diffmask_mcgs.sh refuses to overwrite; read its log)")
                elif j["name"] not in retried:
                    retried.add(j["name"])
                    queue.insert(0, j)
                    say(f"RETRY {j['name']} once")
        procs = my_procs()
        for w in list(waiting):
            if not any(matches(v, w) for v in procs):
                waiting.remove(w)
                if Path(w["done"]).exists():
                    done[w["name"]] = "external"
                    say(f"EXTERNAL DONE {w['name']} result=yes")
                elif w["name"] not in retried:
                    retried.add(w["name"])
                    queue.insert(0, w)
                    say(f"EXTERNAL DONE {w['name']} result=NO -> re-queued once (it may have been mis-identified)")
                else:
                    done[w["name"]] = "external"
                    say(f"EXTERNAL DONE {w['name']} result=NO (already re-queued once; giving up)")
        for j in list(queue):
            if Path(j["done"]).exists():
                queue.remove(j)
                done[j["name"]] = "skip"
                say(f"SKIP {j['name']} (result exists)")
            elif any(matches(v, j) for v in procs):
                queue.remove(j)
                waiting.append(j)
                say(f"WAIT {j['name']} (already running outside this queue)")
        if not queue and not running and not waiting:
            break
        try:
            st = gpu_state()
        except Exception as e:
            say(f"nvidia-smi failed: {e}")
            time.sleep(a.poll_s)
            continue
        busy = {r["gpu"] for r in running if r["job"]["kind"] == "gpu"}
        n_gpu = sum(1 for r in running if r["job"]["kind"] == "gpu")
        eligible = sorted((g for g in cards if not st[g]["mine"] and g not in busy), key=lambda g: -st[g]["free"])
        if a.dry_run:
            for j in queue:
                print(f"  queued {j['name']:32s} {j['kind']} need {j['need']:>2} GB  -> {j['done']}")
            print("  cards:", {g: dict(free_gb=round(st[g]["free"], 1), mine=st[g]["mine"],
                                       locked=lock_path(g).is_dir()) for g in cards},
                  "\n  eligible by process/memory:", eligible)
            return

        def launch(j, g, locked):
            queue.remove(j)
            Path(j["log"]).parent.mkdir(parents=True, exist_ok=True)
            p = subprocess.Popen(j["cmd"].format(gpu=g), shell=True, cwd=j["cwd"], env=ENV, executable="/bin/bash",
                                 stdout=open(j["log"], "a"), stderr=subprocess.STDOUT)
            if locked:
                try:
                    (lock_path(g) / "pid").write_text(str(p.pid))
                except OSError:
                    pass
            running.append(dict(job=j, gpu=g, proc=p, t0=time.time()))
            say(f"START {j['name']} gpu{g} (free {st[g]['free']:.0f} GB) -> {j['log']}")

        for g in eligible:
            if n_gpu >= a.max_gpu_jobs:
                break
            j = next((x for x in queue if x["kind"] == "gpu" and x["need"] <= st[g]["free"]), None)
            if j is None:
                continue
            if not acquire(g, say):
                say(f"card {g} is locked by another scheduler; skipping it this round")
                continue
            launch(j, g, True)
            n_gpu += 1
        n_cpu = sum(1 for r in running if r["job"]["kind"] == "cpu")
        for j in [x for x in queue if x["kind"] == "cpu"]:
            if n_cpu >= a.max_cpu_jobs:
                break
            host = [r["gpu"] for r in running
                    if r["job"]["name"].split("/")[-1].startswith(CLASSICAL) and st[r["gpu"]]["free"] >= j["need"]]
            spare = sorted((g for g in cards if st[g]["free"] >= j["need"]), key=lambda g: -st[g]["free"])
            g = host[0] if host else (spare[0] if spare else None)
            if g is None:
                break
            launch(j, g, False)                                    # CPU job: no card claimed, no lock
            n_cpu += 1
        json.dump(dict(time=time.strftime("%m-%d %H:%M:%S"),
                       running=[dict(name=r["job"]["name"], gpu=r["gpu"], min=round((time.time() - r["t0"]) / 60, 1))
                                for r in running],
                       waiting=[w["name"] for w in waiting], queued=[j["name"] for j in queue], done=done),
                  open(G / "logs/run_all_diffmask_status.json", "w"), indent=1)
        time.sleep(a.poll_s)
    fails = {n: rc for n, rc in done.items() if isinstance(rc, int) and rc != 0}
    say(f"QUEUE EMPTY. failures: {fails}. finalize ...")
    rcs = {}
    for k in kinds:
        short = KINDS[k]
        exps = [f"{short}_p3_{arm}_full4k" for arm in ("joint", "singleT1", "singleFLAIR")]
        rcs[k] = subprocess.call(["bash", str(topic(k) / "scripts/finalize_diffmask.sh"), k, *exps],
                                 cwd=topic(k), env=ENV,
                                 stdout=open(topic(k) / "logs/finalize_diffmask.log", "a"),
                                 stderr=subprocess.STDOUT)
    say(f"FINALIZE {rcs}")
    (G / "logs/RUN_ALL_DIFFMASK_DONE").write_text(json.dumps(dict(done=done, finalize_rc=rcs), indent=1))


if __name__ == "__main__":
    main()
