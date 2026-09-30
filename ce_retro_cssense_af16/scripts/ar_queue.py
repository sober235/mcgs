#!/usr/bin/env python3
"""ar_queue.py -- run a list of MC-GS / GS-single training jobs on fixed GPU lanes, one job per lane, and score each
one into the topic gauge as soon as it finishes (autoresearch loop, vdPoisMex line, 2026-09-30).

Jobs file: one job per line, tab-separated  EXP <TAB> SINGLE <TAB> extra cycle.py args   (SINGLE = joint | pre | post;
'#' comments and blank lines ignored). Each job is `env SINGLE=<single> bash run_gs.sh EXP GPU ITERS <extra args>`;
run_gs.sh appends the extra args after the frozen recipe, so a repeated flag overrides it (argparse keeps the last
occurrence; --set entries are applied in order, later ones win). ITERS is taken from an `--iters N` token in the extra
args if present (removed from them), else 4000.

When a job's `<EXP>_volumes.npz` appears, `extract_gs_volumes.py --exp EXP --key ar_EXP` is run (CPU) so
logs/ar_EXP_<scan>.json carries the head-only gauge; a job whose npz never appears is logged FAILED and the lane moves on.
Restartable: a job whose score json already exists is skipped; a job whose npz exists but not its json is only re-scored.
  python scripts/ar_queue.py --jobs ar_jobs_phase1.tsv --lanes 5,0        # blocks until every job is done; log in logs/ar/queue.log
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois"))
TR = Path(os.environ.get("RETRO_TRAINER", str(R / "trainer/_trainer_e778bf2")))
S = Path(__file__).resolve().parent
PY = "/home/congcongliu/anaconda3/envs/GSTMI/bin/python"
ENV = dict(os.environ, RETRO_TOPIC=str(R), RETRO_TRAINER=str(TR), PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID")


def log(msg):
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(R / "logs/ar/queue.log", "a") as f:
        f.write(line + "\n")


def read_jobs(path):
    jobs = []
    for raw in open(path):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split("\t")
        exp, single = parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "joint")
        extra = parts[2].split() if len(parts) > 2 else []
        iters = "4000"
        if "--iters" in extra:
            i = extra.index("--iters"); iters = extra[i + 1]; del extra[i:i + 2]
        jobs.append(dict(exp=exp, single=single, iters=iters, extra=extra))
    return jobs


def npz(exp):   return TR / "results_arc" / f"{exp}_volumes.npz"
def score(exp): return R / "logs" / f"ar_{exp}_pre.json"


def extract(job):
    exp, single = job["exp"], job["single"]
    m = {"joint": "pre:plain,post:contrast", "pre": "pre:plain", "post": "post:plain"}[single]
    cmd = [PY, str(S / "extract_gs_volumes.py"), "--exp", exp, "--key", f"ar_{exp}", "--map", m]
    with open(R / "logs/ar" / f"{exp}_extract.log", "a") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=ENV).returncode
    key_scan = "pre" if single != "post" else "post"
    ok = rc == 0 and (R / "logs" / f"ar_{exp}_{key_scan}.json").exists()
    log(f"{'scored' if ok else 'EXTRACT FAILED'} {exp} (rc={rc})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--lanes", required=True, help="comma-separated GPU indices")
    ap.add_argument("--poll", type=int, default=60)
    a = ap.parse_args()
    (R / "logs/ar").mkdir(parents=True, exist_ok=True)
    lanes = [int(g) for g in a.lanes.split(",")]
    pending = read_jobs(a.jobs); running = {}   # gpu -> (job, Popen)
    for j in list(pending):
        done_scan = "pre" if j["single"] != "post" else "post"
        if (R / "logs" / f"ar_{j['exp']}_{done_scan}.json").exists():
            log(f"already scored {j['exp']}"); pending.remove(j)
        elif npz(j["exp"]).exists():
            log(f"npz present, scoring only: {j['exp']}"); extract(j); pending.remove(j)
    log(f"queue: {len(pending)} jobs, lanes {lanes}")
    while pending or running:
        for g, (j, p) in list(running.items()):
            if p.poll() is None:
                continue
            del running[g]
            if npz(j["exp"]).exists():
                log(f"finished {j['exp']} on gpu {g} rc={p.returncode}"); extract(j)
            else:
                log(f"FAILED {j['exp']} on gpu {g} rc={p.returncode}; see {TR / 'results_arc' / (j['exp'] + '.log')}")
        for g in lanes:
            if g in running or not pending:
                continue
            j = pending.pop(0)
            cmd = ["bash", str(S / "run_gs.sh"), j["exp"], str(g), j["iters"]] + j["extra"]
            env = dict(ENV, SINGLE=j["single"])
            lf = open(R / "logs/ar" / f"{j['exp']}_launch.log", "a")
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            running[g] = (j, p)
            log(f"launched {j['exp']} on gpu {g} pid {p.pid}: SINGLE={j['single']} iters={j['iters']} extra={' '.join(j['extra'])}")
        time.sleep(a.poll)
    log("QUEUE DONE")


if __name__ == "__main__":
    main()
