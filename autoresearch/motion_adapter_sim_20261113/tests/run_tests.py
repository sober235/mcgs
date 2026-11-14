#!/usr/bin/env python3
"""Run pytest on the server and print its output with every measured duration removed (time rule: no server
clock value or duration leaves the server). Usage: run_tests.py [pytest args]; default = all tests; GPU-marked tests only with --gpu."""
import pathlib
import re
import subprocess
import sys

here = pathlib.Path(__file__).resolve().parent
args = sys.argv[1:]
gpu = "--gpu" in args                                   # run_tests.py --gpu [files]: only the GPU-marked tests
args = [a for a in args if a != "--gpu"] or [str(here)]
args += ["-m", "gpu" if gpu else "not gpu", "--tb=short"]
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--color=no", *args],
                   capture_output=True, text=True, cwd=str(here.parent))
out = r.stdout + r.stderr
out = re.sub(r"=+ slowest.*?(?=\n=|\Z)", "", out, flags=re.S)          # the durations table, if any
out = re.sub(r"\s+in \d+(\.\d+)?s( \(\d+:\d+:\d+\))?", "", out)          # "5 passed in 1.23s"
out = re.sub(r"\b\d+(\.\d+)?s (call|setup|teardown)\b.*", "", out)
print(out.strip())
sys.exit(r.returncode)
