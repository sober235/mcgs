"""Covers the running-process guard added on top of the task-13 brief: the production arms of 2026-09-17 (MC-GS
joint, GS-single pre/post, ZS-SSDU pre/post, 3D-DIP pre) were started by hand before this scheduler existed, so
their done() output files do not exist yet and the scheduler must not launch a duplicate. already_running() matches
on the script + key arguments, ignoring the GPU index (a hand launch may use a different card) and the interpreter
path (a hand launch may use a relative script path where build_jobs uses an absolute one)."""
import subprocess
import sys
import time


def _spawn(tmp_path, marker, extra_args):
    script = tmp_path / f"{marker}.py"
    script.write_text("import time\ntime.sleep(20)\n")
    cmd = [sys.executable, "-u", str(script)] + extra_args
    return subprocess.Popen(cmd), cmd


def test_already_running_ignores_gpu_index_but_not_other_args(tmp_path):
    import retro_orchestrate as O
    marker = f"orch_guard_{__import__('os').getpid()}_{time.time_ns()}"
    proc, real_cmd = _spawn(tmp_path, marker, ["--scan", "pre", "--gpu", "0"])
    try:
        time.sleep(0.5)
        same_job_other_gpu = real_cmd[:-1] + ["5"]                 # same script + args, different GPU
        different_job = [t if t != "pre" else "post" for t in real_cmd]   # same script, different key argument
        assert O.already_running(same_job_other_gpu)
        assert not O.already_running(different_job)
        assert not O.already_running([sys.executable, "-u", str(tmp_path / "does_not_exist.py"), "--gpu", "0"])
    finally:
        proc.terminate()
        proc.wait(timeout=5)
