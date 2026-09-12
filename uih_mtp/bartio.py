"""BART cfl/hdr file I/O (complex64, Fortran order) and a runner for the house BART binary."""
from __future__ import annotations

import os
import subprocess

import numpy as np

BART = "/home/congcongliu/src/bart/bart"


def write_cfl(base: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.complex64)
    with open(base + ".hdr", "w") as f:
        f.write("# Dimensions\n" + " ".join(str(d) for d in arr.shape) + "\n")
    arr.flatten(order="F").tofile(base + ".cfl")


def read_cfl(base: str) -> np.ndarray:
    with open(base + ".hdr") as f:
        dims = [int(x) for x in f.read().splitlines()[1].split()]
    data = np.fromfile(base + ".cfl", dtype=np.complex64)
    return data.reshape(dims, order="F")


def run_bart(args: list[str], threads: int = 8, nice: int = 10, log=None) -> str:
    """Run one BART command with a thread cap and reduced priority; returns its stdout+stderr."""
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    cmd = ["nice", "-n", str(nice), BART] + [str(a) for a in args]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    out = res.stdout + res.stderr
    if log is not None:
        log.write(" ".join(cmd) + "\n" + out + "\n"); log.flush()
    if res.returncode != 0:
        raise RuntimeError(f"bart failed ({res.returncode}): {' '.join(cmd)}\n{out}")
    return out
