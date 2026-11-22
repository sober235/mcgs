#!/usr/bin/env python3
"""Run this directory's tests through the first version's runner (pytest on the server, every measured duration
removed). Usage: run_tests.py [--gpu] [test files or pytest arguments]; default = all tests of this directory."""
import pathlib
import runpy
import sys

here = pathlib.Path(__file__).resolve().parent
v1 = here.parents[1] / "motion_adapter_sim_20261113/tests/run_tests.py"
given = sys.argv[1:]
rest = [str(pathlib.Path(a).resolve()) if pathlib.Path(a).exists() else a for a in given if a != "--gpu"]
sys.argv = [str(v1), *(["--gpu"] if "--gpu" in given else []), *(rest or [str(here)])]
runpy.run_path(str(v1), run_name="__main__")
