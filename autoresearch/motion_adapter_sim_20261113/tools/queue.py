#!/usr/bin/env python3
"""Local queue (this Mac) for the trainings of the motion study.

    tools/queue.py --rows af12:C0:A0 af12:D1:A1 [--settings settings_frozen.json] [--seed 42] [--max-cards 8]
    tools/queue.py --part af12_core --settings settings_frozen.json

Rules it implements (project CLAUDE.md, GPU use):
  - a card is used only if it has no compute process of anyone and no held memory on TWO checks at least 70 s apart,
    the second one right before the launch; one job per card; nothing of anybody else is touched;
  - every launch goes through gpu.py (tools/rrun.py's client, the worktree's private config) with an explicit card;
  - all times are this Mac's clock: start = when the launch was issued, end = when a poll here saw the job finished.
    Nothing about time is read from the server; a finished job contributes its ledger row (no clock fields) only.

State: .claude/gpu/queue_state_<name>.json per queue (running jobs, so a restarted queue re-attaches instead of
launching twice) and .claude/gpu/claims.json shared by all queues (card -> experiment, under a file lock), because a
card still looks empty for a minute after a launch.
Results: experiments.jsonl (one line per finished training: ledger row + local_start_pst, local_end_pst, job, gpu,
code_sha256). A job id embeds the server's clock, so only its random suffix (job) is printed or recorded; the full id
stays in the local state file. The training code is hashed when the queue starts and before every launch; a change stops the queue.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve()
E = HERE.parents[1]
WORKTREE = HERE.parents[3]
WORKSPACE = WORKTREE.parents[3]
REL = "autoresearch/motion_adapter_sim_20261113"
ROOT = "/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_motion_adapter_20261113"
CLAIMS = WORKTREE / ".claude/gpu/claims.json"
AVOID = WORKTREE / ".claude/gpu/avoid.json"      # [card, ...] promised to another session; re-read at every check
RESULTS = E / "experiments.jsonl"
CODE = ("adapter.py", "common.py", "geom.py", "patch.py", "run.py", "schedule.py", "warp.py")
SSH = ["ssh", "-p", "2222", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=25", "alias"]
FREE = ('nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits | tr -d " " > /tmp/.maq_g.$$; '
        'nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | tr -d " " > /tmp/.maq_a.$$; '
        'while IFS=, read i u m; do if ! grep -q "$u" /tmp/.maq_a.$$ && [ "$m" -lt 200 ]; then printf "%s " $i; fi; '
        'done < /tmp/.maq_g.$$; rm -f /tmp/.maq_g.$$ /tmp/.maq_a.$$; echo')

spec = importlib.util.spec_from_file_location("gpu", WORKSPACE / "gpu-tools/gpu.py")
gpu = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(WORKSPACE / "gpu-tools"))
spec.loader.exec_module(gpu)


def tag(job_id):
    return job_id.rsplit("-", 1)[-1]


def claim(card, exp):
    """exp=None releases the card. Returns False if another queue holds it."""
    with open(str(CLAIMS) + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        c = json.loads(CLAIMS.read_text()) if CLAIMS.exists() else {}
        if exp is None:
            c.pop(str(card), None)
        elif str(card) in c:
            return False
        else:
            c[str(card)] = exp
        CLAIMS.write_text(json.dumps(c))
        return True


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def say(*a):
    print(time.strftime("%H:%M"), *a, flush=True)


def code_sha(settings):
    h = hashlib.sha256()
    for name in (*CODE, *([settings] if settings else [])):
        h.update(name.encode() + b"\0" + (E / name).read_bytes() + b"\0")
    return h.hexdigest()


def exp_name(af, cond, arm, seed, tag=""):
    return f"ma_{af}_{cond}_{arm}" + ("" if seed == 42 else f"_s{seed}") + (f"_{tag}" if tag else "")


def ssh(cmd, timeout=90):
    return subprocess.run([*SSH, cmd], capture_output=True, text=True, timeout=timeout, check=True).stdout


def free_cards():
    try:
        return {int(x) for x in ssh(FREE).split()}
    except Exception as e:                                   # a failed check frees nothing
        say(f"gpu check failed ({type(e).__name__})")
        return set()


def job_state(client, job_id):
    """(state, exit_code) of a job; only these two fields of the job record are read."""
    j = client.rpc("job", job_id=job_id)
    return j["state"], j.get("exit_code")


def ledger_row(exp):
    out = ssh(f"grep -F '\"exp\": \"{exp}\"' {ROOT}/ledger.jsonl | tail -1")
    return json.loads(out) if out.strip() else None


def log_tail(client, job_id, n=25):
    import base64
    r = client.rpc("logs", job_id=job_id, offset=0)
    text = base64.b64decode(r["data"]).decode("utf-8", "replace")
    return "\n".join(text.splitlines()[-n:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="*", default=[], help="af:cond:arm")
    ap.add_argument("--part", help="a key of common.MATRIX (af12_core, af12_rest, af16, dev)")
    ap.add_argument("--settings")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="", help="development round (run.py --tag)")
    ap.add_argument("--max-cards", type=int, default=8)
    ap.add_argument("--poll", type=int, default=75)
    ap.add_argument("--name", help="queue name (state file); default: tag, part or 'main'")
    args = ap.parse_args()
    rows = [tuple(r.split(":")) for r in args.rows]
    if args.part:
        sys.path.insert(0, str(E))
        sys.dont_write_bytecode = True
        import common
        af = "af16" if args.part == "af16" else "af12"
        rows += [(af, c, a) for c, a in common.MATRIX[args.part]]
    assert rows, "no rows"
    STATE = WORKTREE / f".claude/gpu/queue_state_{args.name or args.tag or args.part or 'main'}.json"
    client = gpu.Client(WORKTREE / ".claude/gpu/config.json")
    assert client.repo == WORKTREE
    sha0 = code_sha(args.settings)
    done = set()
    if RESULTS.exists():
        done = {json.loads(l)["exp"] for l in RESULTS.read_text().splitlines() if l.strip()}
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    todo = [(af, c, a) for af, c, a in rows if exp_name(af, c, a, args.seed, args.tag) not in done]
    pending = [r for r in todo if exp_name(*r, args.seed, args.tag) not in state]
    say(f"queue: {len(rows)} rows, {len(rows) - len(todo)} already finished, {len(todo) - len(pending)} running, "
        f"{len(pending)} to launch; code {sha0[:12]}")
    seen, failed = {}, []                                    # card -> local time of the last check that found it free
    while pending or state:
        for exp, j in list(state.items()):
            try:
                st, rc = job_state(client, j["job_id"])
            except Exception as e:
                say(f"state check of {exp} failed ({type(e).__name__})")
                continue
            if st in ("queued", "running"):
                continue
            row = ledger_row(exp) if (st == "completed" and rc == 0) else None
            if row is None:
                failed.append(exp)
                say(f"FAILED {exp} job {tag(j['job_id'])} state={st} exit={rc}")
                try:
                    (WORKTREE / ".claude/gpu" / f"failed_{exp}.log").write_text(log_tail(client, j["job_id"]))
                except Exception:
                    pass
            else:
                row.update(local_start_pst=j["local_start"], local_end_pst=now(), job=tag(j["job_id"]), gpu=j["gpu"],
                           code_sha256=j["code_sha256"])
                with open(RESULTS, "a") as f:
                    f.write(json.dumps(row) + "\n")
                p, c = row.get("plain") or {}, row.get("contrast") or {}
                say(f"DONE {exp} gpu {j['gpu']} plain head_psnr={p.get('head_psnr')} "
                    f"contrast head_psnr={c.get('head_psnr')}")
            claim(j["gpu"], None)
            del state[exp]
            STATE.write_text(json.dumps(state, indent=1))
        if pending and len(state) < args.max_cards:
            free = free_cards() - {j["gpu"] for j in state.values()}
            free -= set(json.loads(AVOID.read_text())) if AVOID.exists() else set()
            t = time.time()
            ready = sorted(c for c in free if c in seen and t - seen[c] >= 70)
            seen = {c: seen.get(c, t) for c in free}         # a card that was busy in between starts over
            for card in ready:
                if not pending or len(state) >= args.max_cards:
                    break
                assert code_sha(args.settings) == sha0, "training code changed while the queue runs"
                af, cond, arm = pending[0]
                exp = exp_name(af, cond, arm, args.seed, args.tag)
                if not claim(card, exp):
                    continue
                pending.pop(0)
                argv = ["@python", f"{REL}/run.py", "--af", af, "--cond", cond, "--arm", arm, "--gpu", str(card),
                        "--seed", str(args.seed)] + (["--settings", args.settings] if args.settings else []) \
                    + (["--tag", args.tag] if args.tag else [])
                start = now()
                try:
                    job = client.run(argv, cpu=False, gpu_ids=[card])
                except Exception as e:
                    say(f"launch of {exp} failed ({type(e).__name__}: {e}); back in the queue")
                    pending.insert(0, (af, cond, arm))
                    claim(card, None)
                    break
                state[exp] = dict(job_id=job["id"], gpu=card, local_start=start, code_sha256=sha0)
                STATE.write_text(json.dumps(state, indent=1))
                seen.pop(card, None)
                say(f"LAUNCH {exp} gpu {card} job {tag(job['id'])}")
        if pending or state:
            time.sleep(args.poll)
    say(f"queue finished; failed: {failed or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
