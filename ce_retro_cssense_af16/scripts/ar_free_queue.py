#!/usr/bin/env python3
"""ar_free_queue.py -- ar_queue.py's jobs, but on whichever cards are EMPTY rather than on fixed lanes (2026-10-10).

Same jobs file (EXP <TAB> SINGLE <TAB> extra cycle.py args), same launch (`SINGLE=<single> bash run_gs.sh EXP GPU ITERS
extra`), same scoring into the topic gauge (extract_gs_volumes.py --key ar_EXP) and the same restart rule (a job
whose score json exists is skipped; one whose npz exists is only scored).

A card is used only when it has been empty -- memory.used < 1500 MiB and no compute process -- on two consecutive
polls (another session's job that is still loading its data shows an empty card for a minute; 2026-10-10 two of our
launches landed on such cards). At most --max_run jobs at once, one job per card, one launch per poll.
  python scripts/ar_free_queue.py --jobs ar_jobs_phase6a.tsv --cards 0,1,2,3,4,5,6,7 --max_run 3
"""
import argparse, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ar_queue as Q                     # noqa: E402  (R, TR, ENV, S, read_jobs, extract, key_of, npz, log)


def empty_cards(cards):
    """cards with memory.used < 1500 MiB and no compute process, per nvidia-smi."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout
    busy_uuid = {line.strip() for line in apps.splitlines() if line.strip()}
    free = set()
    for line in out.splitlines():
        i, uuid, used = (s.strip() for s in line.split(","))
        if int(i) in cards and float(used) < 1500 and uuid not in busy_uuid:
            free.add(int(i))
    return free


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--cards", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--max_run", type=int, default=3)
    ap.add_argument("--poll", type=int, default=60)
    a = ap.parse_args()
    (Q.R / "logs/ar").mkdir(parents=True, exist_ok=True)
    cards = {int(g) for g in a.cards.split(",")}
    pending = Q.read_jobs(a.jobs); running = {}
    for j in list(pending):
        done_scan = "pre" if j["single"] != "post" else "post"
        if (Q.R / "logs" / f"{Q.key_of(j['exp'])}_{done_scan}.json").exists():
            Q.log(f"already scored {j['exp']}"); pending.remove(j)
        elif Q.npz(j["exp"]).exists():
            Q.log(f"npz present, scoring only: {j['exp']}"); Q.extract(j); pending.remove(j)
    Q.log(f"free-card queue {a.jobs}: {len(pending)} jobs, cards {sorted(cards)}, max_run {a.max_run}")
    seen_free = {}                                  # card -> consecutive polls seen empty
    while pending or running:
        for g, (j, p) in list(running.items()):
            if p.poll() is None:
                continue
            del running[g]
            if Q.npz(j["exp"]).exists():
                Q.log(f"finished {j['exp']} on gpu {g} rc={p.returncode}"); Q.extract(j)
            else:
                Q.log(f"FAILED {j['exp']} on gpu {g} rc={p.returncode}; see {Q.TR / 'results_arc' / (j['exp'] + '.log')}")
        free = empty_cards(cards) - set(running)
        seen_free = {g: seen_free.get(g, 0) + 1 for g in free}
        ready = sorted(g for g, n in seen_free.items() if n >= 2)
        if pending and ready and len(running) < a.max_run:
            g = ready[0]; j = pending.pop(0); seen_free.pop(g, None)
            cmd = ["bash", str(Q.S / "run_gs.sh"), j["exp"], str(g), j["iters"]] + j["extra"]
            lf = open(Q.R / "logs/ar" / f"{j['exp']}_launch.log", "a")
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=dict(Q.ENV, SINGLE=j["single"]),
                                 start_new_session=True)
            running[g] = (j, p)
            Q.log(f"launched {j['exp']} on gpu {g} pid {p.pid}: SINGLE={j['single']} iters={j['iters']} extra={' '.join(j['extra'])}")
        time.sleep(a.poll)
    Q.log("FREE QUEUE DONE")


if __name__ == "__main__":
    main()
