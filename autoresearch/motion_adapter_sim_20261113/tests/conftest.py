import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device (run in the GPU smoke)")
