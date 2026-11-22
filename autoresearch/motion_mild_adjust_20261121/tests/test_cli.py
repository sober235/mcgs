import pathlib
import subprocess
import sys

import pytest

M = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", ["msim.py", "mrun.py", "mevaluate.py", "reg.py", "mfigures.py"])
def test_script_starts_on_its_own(script, tmp_path):
    """Every entry point must find the first version's modules by itself (the tests' conftest puts them on the
    path, a job on the server does not)."""
    r = subprocess.run([sys.executable, str(M / script), "--help"], cwd=tmp_path, capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr[-600:]
    assert "usage" in r.stdout.lower()
