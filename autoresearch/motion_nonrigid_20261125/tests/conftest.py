import pathlib
import sys

N = pathlib.Path(__file__).resolve().parents[1]
E1 = N.parent / "motion_adapter_sim_20261113"
for p in (E1 / "tests", E1, N):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ncommon  # noqa: E402,F401  (before any module of the second version)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device (run in the GPU smoke)")
