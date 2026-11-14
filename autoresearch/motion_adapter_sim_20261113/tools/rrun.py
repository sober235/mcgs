#!/usr/bin/env python3
"""Local helper (this Mac): snapshot THIS worktree, run one command on amax through gpu.py with the worktree's
private config, follow its log and exit with the job's exit code. Prints the log text only (no job metadata).

    tools/rrun.py --cpu -- @python autoresearch/motion_adapter_sim_20261113/tests/run_tests.py
    tools/rrun.py --gpu 3 --detach -- @python autoresearch/motion_adapter_sim_20261113/run.py ...
"""
import argparse
import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
WORKTREE = HERE.parents[3]
WORKSPACE = WORKTREE.parents[3]                      # mcgs-workspace (worktree = mcgs/.claude/worktrees/<name>)
spec = importlib.util.spec_from_file_location("gpu", WORKSPACE / "gpu-tools/gpu.py")
gpu = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(WORKSPACE / "gpu-tools"))
spec.loader.exec_module(gpu)

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--cpu", action="store_true")
g.add_argument("--gpu", type=int)
ap.add_argument("--detach", action="store_true", help="print the job's suffix and return without following the log")
ap.add_argument("argv", nargs=argparse.REMAINDER)
a = ap.parse_args()
argv = a.argv[1:] if a.argv[:1] == ["--"] else a.argv
client = gpu.Client(WORKTREE / ".claude/gpu/config.json")
assert client.repo == WORKTREE, f"private config points at {client.repo}"
job = client.run(argv, cpu=a.cpu, gpu_ids=None if a.cpu else [a.gpu])
# the job id embeds the server's clock, so only its random suffix is shown; the full id stays in a local file
tag = job["id"].rsplit("-", 1)[-1]
print(f"[rrun] job {tag} source {job.get('source_digest', '')[:12]}", file=sys.stderr, flush=True)
if a.detach:
    (WORKTREE / ".claude/gpu" / f"job_{tag}").write_text(job["id"])
    print(tag)
    sys.exit(0)
out = client.logs(job["id"], follow=True)
print(f"[rrun] state={out['state']} exit_code={out['exit_code']}", file=sys.stderr, flush=True)
sys.exit(0 if (out["state"] == "completed" and out["exit_code"] == 0) else 1)
