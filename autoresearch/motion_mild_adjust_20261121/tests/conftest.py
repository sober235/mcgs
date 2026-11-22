import pathlib
import sys

M = pathlib.Path(__file__).resolve().parents[1]
E1 = M.parent / "motion_adapter_sim_20261113"
for p in (E1 / "tests", E1, M):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device (run in the GPU smoke)")
