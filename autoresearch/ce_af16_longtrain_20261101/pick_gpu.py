#!/usr/bin/env python3
"""Pick GPUs for a new run under the shared-server rule.

Rule (in priority order):
  1. Prefer empty GPUs (no compute process, < EMPTY_MIB used).
  2. Then lightly loaded busy GPUs (mean utilization <= --max-util and at least --need-gb
     free), lowest utilization first, then lowest memory used.
  3. Always leave at least one empty GPU untouched for other users.

Prints the chosen physical (nvidia-smi / PCI_BUS_ID) indices, space separated.
Exits 1 with the reason on stderr when fewer than --count GPUs qualify.

Usage: pick_gpu.py [--count N] [--need-gb G] [--max-util U] [--exclude 3,5] [--samples S]
"""
import argparse
import subprocess
import sys
import time

EMPTY_MIB = 1024


def smi(query, kind):
    out = subprocess.run(["nvidia-smi", f"--query-{kind}={query}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def snapshot(samples):
    util = {}
    for i in range(samples):
        for idx, u in smi("index,utilization.gpu", "gpu"):
            util.setdefault(int(idx), []).append(float(u))
        if i + 1 < samples:
            time.sleep(1.0)
    busy_uuids = {row[0] for row in smi("gpu_uuid,pid", "compute-apps")}
    gpus = []
    for idx, uuid, used, total in smi("index,uuid,memory.used,memory.total", "gpu"):
        idx = int(idx)
        gpus.append({"index": idx, "used": float(used), "total": float(total),
                     "util": sum(util[idx]) / len(util[idx]),
                     "empty": uuid not in busy_uuids and float(used) < EMPTY_MIB})
    return gpus


def pick(gpus, count, need_mib, max_util, exclude):
    empty = sorted((g for g in gpus if g["empty"]), key=lambda g: g["index"])
    usable_empty = [g for g in empty if g["index"] not in exclude][: max(0, len(empty) - 1)]
    light = sorted((g for g in gpus if not g["empty"] and g["index"] not in exclude
                    and g["util"] <= max_util and g["total"] - g["used"] >= need_mib),
                   key=lambda g: (g["util"], g["used"]))
    return [g["index"] for g in usable_empty + light][:count]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--need-gb", type=float, default=55.0)
    ap.add_argument("--max-util", type=float, default=60.0)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--samples", type=int, default=5)
    args = ap.parse_args()
    exclude = {int(x) for x in args.exclude.split(",") if x.strip()}
    gpus = snapshot(args.samples)
    chosen = pick(gpus, args.count, args.need_gb * 1024, args.max_util, exclude)
    for g in gpus:
        state = "empty" if g["empty"] else f"busy util={g['util']:.0f}%"
        print(f"  gpu{g['index']}: {state} used={g['used'] / 1024:.1f}/{g['total'] / 1024:.0f} GB", file=sys.stderr)
    if len(chosen) < args.count:
        print(f"only {len(chosen)} of {args.count} GPU(s) qualify (busy GPUs need util <= {args.max_util}% "
              f"and {args.need_gb} GB free; one empty GPU is always kept free): {chosen}", file=sys.stderr)
        sys.exit(1)
    print(" ".join(str(i) for i in chosen))


if __name__ == "__main__":
    main()
