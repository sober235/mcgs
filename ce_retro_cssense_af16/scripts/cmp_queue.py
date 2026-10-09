#!/usr/bin/env python3
"""cmp_queue.py -- run the comparison-arm improvement jobs (2026-10-09 audit) on fixed GPU lanes, one job per lane.

Jobs file: one job per line, tab-separated   NAME <TAB> DONE_FILE <TAB> COMMAND   ('#' comments and blank lines ignored).
DONE_FILE is relative to the topic root; a job whose DONE_FILE exists is skipped (restartable). COMMAND is run with
`bash -c` from the topic root with RETRO_TOPIC / RETRO_TRAINER / PYTHONNOUSERSITE / PYTHONPATH / CUDA_DEVICE_ORDER set;
the token {gpu} in COMMAND is replaced by the lane's GPU index. stdout+stderr -> logs/cmp/<NAME>.log, the queue's own
log -> logs/cmp/queue.log. Unlike retro_orchestrate.py it never picks cards itself: the lanes are given, and the user
decides which cards this loop may take (the sg2x2 queue and the AnatoBind chain take every empty card among their own lists).
  python scripts/cmp_queue.py --jobs cmp_jobs_round1.tsv --lanes 2,4
"""
import argparse, os, subprocess, time
from pathlib import Path

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois"))
TR = Path(os.environ.get("RETRO_TRAINER", str(R / "trainer/_trainer_e778bf2")))
ENV = dict(os.environ, RETRO_TOPIC=str(R), RETRO_TRAINER=str(TR), PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID")


def log(msg):
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(R / "logs/cmp/queue.log", "a") as f:
        f.write(line + "\n")


def read_jobs(path):
    jobs = []
    for raw in open(path):
        line = raw.split("#", 1)[0].rstrip("\n")
        if not line.strip():
            continue
        name, done, cmd = (s.strip() for s in line.split("\t", 2))
        jobs.append(dict(name=name, done=done, cmd=cmd))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--lanes", required=True, help="comma-separated GPU indices")
    ap.add_argument("--poll", type=int, default=60)
    a = ap.parse_args()
    (R / "logs/cmp").mkdir(parents=True, exist_ok=True)
    lanes = [int(g) for g in a.lanes.split(",")]
    pending = [j for j in read_jobs(a.jobs) if not (R / j["done"]).exists()]
    for j in read_jobs(a.jobs):
        if (R / j["done"]).exists():
            log(f"already done {j['name']} ({j['done']})")
    log(f"queue {a.jobs}: {len(pending)} jobs, lanes {lanes}")
    running = {}
    while pending or running:
        for g, (j, p, t) in list(running.items()):
            if p.poll() is None:
                continue
            del running[g]
            ok = (R / j["done"]).exists()
            log(f"{'finished' if ok else 'FAILED'} {j['name']} on gpu {g} rc={p.returncode} ({time.time() - t:.0f} s); see logs/cmp/{j['name']}.log")
        for g in lanes:
            if g in running or not pending:
                continue
            j = pending.pop(0)
            cmd = j["cmd"].replace("{gpu}", str(g))
            lf = open(R / "logs/cmp" / f"{j['name']}.log", "a")
            lf.write(f"\n===== {time.strftime('%F %T')} gpu {g}: {cmd}\n"); lf.flush()
            p = subprocess.Popen(["bash", "-c", cmd], cwd=str(R), stdout=lf, stderr=subprocess.STDOUT, env=ENV, start_new_session=True)
            running[g] = (j, p, time.time())
            log(f"launched {j['name']} on gpu {g} pid {p.pid}: {cmd}")
        time.sleep(a.poll)
    log("QUEUE DONE")


if __name__ == "__main__":
    main()
