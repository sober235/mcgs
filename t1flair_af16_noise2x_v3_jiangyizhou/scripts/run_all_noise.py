#!/usr/bin/env python3
"""run_all_noise.py -- every arm of the jiangyizhou AF16 noise study, queued onto free cards (2026-09-16).

The study varies ONE thing against T1FLAIR_AF16_v3_jiangyizhou (the zero-noise column, not re-run): the measurements
carry 2 sigma0 of added complex Gaussian noise (data/meas_<c>.h5); ground truth and scale stay on the clean k-space.
Spec: /data0/congcong/code/GS/mcgs_noise_wt/docs/superpowers/specs/2026-09-16-jyz-af16-noise-robustness-design.md.

Arms and grids are the AF16 topic's (as in run_all_diffmask.py), with the TV and JTV grids extended two steps upward
because noise moves the regularisation optimum up; --extend adds more steps to one family if a winner still sits on
the upper edge. MC-GS / GS-single run the frozen AF16 phase-3 recipe (run_noise_mcgs.sh) with --save_render.
L1-ESPIRiT is retired (2026-09-16) and not queued.

Scheduling rules, as in run_all_diffmask.py: one GPU job of this user per card; a card is eligible when no compute
process of this uid is on it, no job of this queue is on it, it has the job's memory free, and its
Cardiac_FreeRunning_MCGS/work/.gpulock.<idx> lock can be taken (a lock whose recorded owner is dead is reaped). A job
whose result exists is skipped; a job already running outside the queue (same argv AND working directory) is waited
for; a failed non-GS job is retried once; a failed GS run is only logged; MC-INR yields with rc 3 and is re-queued.
When nothing is queued or running: finalize_noise.sh.

Launch detached through scripts/launch_noise.sh. --dry_run prints the queue and the card eligibility and exits.
usage: run_all_noise.py [--cards 1,2,5] [--max_gpu_jobs 3] [--extend tv_T1=0.004,0.0056] [--dry_run]
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

RES = Path("/data0/congcong/code/GS/Results_20260908")
N = RES / "T1FLAIR_AF16_noise2x_v3_jiangyizhou"
F = Path("/data0/congcong/code/GS/mcgs_noise_wt")
LOCK_ROOT = RES / "Cardiac_FreeRunning_MCGS/work"
PY = "/home/congcongliu/anaconda3/envs/GSTMI/bin/python"
PY_TCNN = "/home/congcongliu/anaconda3/envs/GR_pt1.10/bin/python"
MCINR = Path("/data0/congcong/code/GS/mcinr_niessen2025")
MCINR_TOPIC = RES / "T1FLAIR_MCINR_Niessen2025_v3_jiangyizhou"
ENV = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID")
UID = os.getuid()
SHORT, TAG = "af16n", "_noise2x"
GS_EXPS = {arm: f"{SHORT}_p3_{arm}_full4k" for arm in ("joint", "singleT1", "singleFLAIR")}
# A placeholder CUDA context on the card for HOLD_S seconds after each launch (added 2026-09-17 01:25): the jobs load
# their data for 1-2 min before they create a context, and schedulers of other lines that ignore .gpulock (CE retro_run.py,
# peds p5_sched.sh) read the card as free in that window -- at 00:30:52 a CE job landed on card 1 six seconds after
# inr_T1 and both ran ~8x slower. The holder stays well under MC-INR's 1500 MiB yield threshold, and it is stopped
# as soon as its job is reaped, so a short job does not leave the card looking busy.
HOLD_S = 240
# Jobs that must not run on a card shared with another user's busy job (--light_only_cards; added 2026-09-17 03:30):
# on card 5, next to another user's job at 100 % utilisation, inr_FLAIR ran 17 s/it against 2.3 s/it on a free card.
HEAVY = ("gs_", "inr_", "dip_", "zsssdu_", "mcinr")
HOLD_CODE = "import sys, time, torch; torch.zeros(1, device='cuda'); time.sleep(float(sys.argv[1]))"

# the AF16 grids plus two upward steps (spec section 5)
GRIDS = {"tv_T1": ["0.00035", "0.0005", "0.0007", "0.001", "0.0014", "0.002", "0.0028"],
         "tv_FLAIR": ["0.00025", "0.00035", "0.0005", "0.0007", "0.001", "0.0014", "0.002"],
         "jtv": ["0.00035", "0.0005", "0.0007", "0.001", "0.0014", "0.002", "0.0028"],
         "spirit": ["0", "0.0005", "0.001", "0.002", "0.004", "0.008"]}
SPIRIT_KS = ("3", "5")


def jobs(grids):
    L, q = N / "logs", []

    def add(tier, name, need, cwd, cmd, done, log, script, args, absent=(), **extra):
        q.append((tier, dict(name=name, need=need, cwd=str(cwd), cmd=cmd, done=str(done), log=str(log),
                             script=script, args=args, absent=list(absent), **extra)))

    for tier, arm, single in ((0, "joint", ""), (1, "singleFLAIR", "FLAIR"), (3, "singleT1", "T1")):
        exp = GS_EXPS[arm]
        add(tier, f"gs_{arm}", 38, F,
            f"SINGLE={single} SAVE_VOLUMES=1 ./run_noise_mcgs.sh {exp} {{gpu}} 4000 --save_render",
            F / f"results_arc/{exp}_score.json", L / f"sched_gs_{arm}.log", "arc/cycle.py", {"--exp": exp})
    for c in ("T1", "FLAIR"):
        add(2, f"base_{c}", 12, N, f"{PY} -u scripts/recon_classical.py --contrast {c} --gpu {{gpu}}",
            L / f"classical_{c}.json", L / f"classical_{c}.log", "recon_classical.py", {"--contrast": c},
            absent=["--suffix"])
    for c in ("T1", "FLAIR"):
        for lam in grids[f"tv_{c}"]:
            add(4, f"tv_{c}_{lam}", 12, N,
                f"{PY} -u scripts/recon_classical.py --contrast {c} --gpu {{gpu}} --methods cssense --admm_outer 200 "
                f"--lam {lam} --suffix _o200_lam{lam}",
                L / f"classical_{c}_o200_lam{lam}.json", L / f"classical_{c}_o200_lam{lam}.log",
                "recon_classical.py", {"--contrast": c, "--suffix": f"_o200_lam{lam}"})
    for lam in grids["jtv"]:
        add(5, f"jtv_{lam}", 20, N,
            f"{PY} -u scripts/recon_joint_cssense.py --topic {N} --contrasts T1,FLAIR --gpu {{gpu}} --admm_outer 200 "
            f"--lam {lam} --suffix _o200_lam{lam}",
            L / f"jcssense_o200_lam{lam}.json", L / f"jcssense_o200_lam{lam}.log",
            "recon_joint_cssense.py", {"--suffix": f"_o200_lam{lam}"})
    for c in ("T1", "FLAIR"):
        add(6, f"zsssdu_{c}", 18, N, f"{PY} -u scripts/recon_zsssdu.py --epochs 100 --contrast {c} --gpu {{gpu}}",
            L / f"zsssdu_{c}.json", L / f"zsssdu_{c}.log", "recon_zsssdu.py", {"--contrast": c})
    for c in ("T1", "FLAIR"):
        add(7, f"inr_{c}", 18, N,
            f"{PY} -u scripts/recon_dip_inr.py --method inr --iters 1500 --eval_every 50 --contrast {c} --gpu {{gpu}}",
            L / f"inr_{c}.json", L / f"inr_{c}.log", "recon_dip_inr.py", {"--method": "inr", "--contrast": c})
    for c in ("T1", "FLAIR"):
        add(8, f"dip_{c}", 40, N,
            f"{PY} -u scripts/recon_dip_inr.py --method dip --iters 5000 --lr 3e-3 --eval_every 100 --contrast {c} "
            f"--gpu {{gpu}}",
            L / f"dip_{c}.json", L / f"dip_{c}.log", "recon_dip_inr.py", {"--method": "dip", "--contrast": c})
    for c in ("T1", "FLAIR"):
        for ks in SPIRIT_KS:
            for lam in grids["spirit"]:
                sfx = f"_ks{ks}_lam{lam}"
                add(9, f"spirit_{c}{sfx}", 24, N,
                    f"{PY} -u scripts/recon_spirit.py --contrast {c} --gpu {{gpu}} --ksize {ks} --lam {lam} "
                    f"--iters 150 --chunk 24 --suffix {sfx}",
                    L / f"spirit_{c}{sfx}.json", L / f"spirit/{c}{sfx}.log",
                    "recon_spirit.py", {"--contrast": c, "--suffix": sfx})
    add(10, "mcinr", 10, MCINR,
        f"{PY_TCNN} -u scripts/run_volume.py --af 16 --tag {TAG} --gpu {{gpu}} --resume --yield_to_others "
        f"--mask_T1 {N}/data/mask_af16_3d.h5 --mask_FLAIR {N}/data/mask_af16_3d.h5 "
        f"--ksp_T1 {N}/data/meas_T1.h5 --ksp_FLAIR {N}/data/meas_FLAIR.h5",
        MCINR_TOPIC / f"recon/af16{TAG}/meta.json", L / "mcinr_noise2x.log",
        "run_volume.py", {"--tag": TAG}, requeue_rc=3)
    q.sort(key=lambda tj: tj[0])
    return [j for _, j in q]


# ---- the shared per-card lock (unchanged from run_all_diffmask.py) -----------------------------
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
        return True
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
    if cwd != os.path.realpath(j["cwd"]):
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
    ap.add_argument("--cards", default="1,2,5")
    ap.add_argument("--max_gpu_jobs", type=int, default=3)
    ap.add_argument("--poll_s", type=int, default=30)
    ap.add_argument("--extend", action="append", default=[], metavar="FAMILY=L1,L2",
                    help="append lambdas to one grid (tv_T1, tv_FLAIR, jtv, spirit); recorded in the log")
    ap.add_argument("--light_only_cards", default="",
                    help="cards shared with another user's busy job: only light jobs (base, TV, JTV, L1-SPIRiT) go there")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    grids = {k: list(v) for k, v in GRIDS.items()}
    for e in a.extend:
        fam, vals = e.split("=", 1)
        assert fam in grids, f"unknown family {fam}"
        grids[fam] += [v for v in vals.split(",") if v and v not in grids[fam]]
    cards = [int(x) for x in a.cards.split(",")]
    light_only = {int(x) for x in a.light_only_cards.split(",") if x.strip()}
    logf = None if a.dry_run else open(N / "logs/run_all_noise.log", "a")

    def say(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        if logf:
            logf.write(line + "\n")
            logf.flush()

    queue, running, waiting, done, retried = jobs(grids), [], [], {}, set()
    say(f"armed: {len(queue)} jobs, cards {cards}, light-only cards {sorted(light_only)}, max {a.max_gpu_jobs} GPU jobs, "
        f"grids {json.dumps(grids)}")
    while True:
        for r in list(running):
            rc = r["proc"].poll()
            if rc is None:
                continue
            running.remove(r)
            h = r.get("holder")
            if h is not None and h.poll() is None:                 # the placeholder of a short job
                h.terminate()
                h.wait(timeout=30)
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
                if j["name"].startswith("gs_"):
                    say(f"FAILED {j['name']}: not retried (run_noise_mcgs.sh refuses to overwrite; read its log)")
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
                    say(f"EXTERNAL DONE {w['name']} result=NO -> re-queued once")
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
        busy = {r["gpu"] for r in running}
        eligible = sorted((g for g in cards if not st[g]["mine"] and g not in busy), key=lambda g: -st[g]["free"])
        if a.dry_run:
            for j in queue:
                print(f"  queued {j['name']:28s} need {j['need']:>2} GB  -> {j['done']}")
            print(f"  {len(queue)} queued")
            print("  cards:", {g: dict(free_gb=round(st[g]["free"], 1), mine=st[g]["mine"],
                                       locked=lock_path(g).is_dir()) for g in cards},
                  "\n  eligible by process/memory:", eligible)
            return
        n_gpu = len(running)
        for g in eligible:
            if n_gpu >= a.max_gpu_jobs:
                break
            j = next((x for x in queue if x["need"] <= st[g]["free"]
                      and not (g in light_only and x["name"].startswith(HEAVY))), None)
            if j is None:
                continue
            if not acquire(g, say):
                say(f"card {g} is locked by another scheduler; skipping it this round")
                continue
            queue.remove(j)
            Path(j["log"]).parent.mkdir(parents=True, exist_ok=True)
            p = subprocess.Popen(j["cmd"].format(gpu=g), shell=True, cwd=j["cwd"], env=ENV, executable="/bin/bash",
                                 stdout=open(j["log"], "a"), stderr=subprocess.STDOUT)
            try:
                (lock_path(g) / "pid").write_text(str(p.pid))
            except OSError:
                pass
            h = subprocess.Popen([PY, "-c", HOLD_CODE, str(HOLD_S)], env=dict(ENV, CUDA_VISIBLE_DEVICES=str(g)),
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            running.append(dict(job=j, gpu=g, proc=p, t0=time.time(), holder=h))
            say(f"START {j['name']} gpu{g} (free {st[g]['free']:.0f} GB) -> {j['log']}")
            n_gpu += 1
        json.dump(dict(time=time.strftime("%m-%d %H:%M:%S"),
                       running=[dict(name=r["job"]["name"], gpu=r["gpu"], min=round((time.time() - r["t0"]) / 60, 1))
                                for r in running],
                       waiting=[w["name"] for w in waiting], queued=[j["name"] for j in queue], done=done),
                  open(N / "logs/run_all_noise_status.json", "w"), indent=1)
        time.sleep(a.poll_s)
    fails = {n: rc for n, rc in done.items() if isinstance(rc, int) and rc != 0}
    say(f"QUEUE EMPTY. failures: {fails}. finalize ...")
    rc = subprocess.call(["bash", str(N / "scripts/finalize_noise.sh"), GS_EXPS["joint"], GS_EXPS["singleT1"],
                          GS_EXPS["singleFLAIR"]], cwd=N, env=dict(ENV, GPU=str(cards[0])),
                         stdout=open(N / "logs/finalize_noise.log", "a"), stderr=subprocess.STDOUT)
    say(f"FINALIZE rc={rc}")
    (N / "logs/RUN_ALL_NOISE_DONE").write_text(json.dumps(dict(done=done, finalize_rc=rc, grids=grids), indent=1))


if __name__ == "__main__":
    main()
