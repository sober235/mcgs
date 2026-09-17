"""Covers the running-process guard added on top of the task-13 brief: the production arms of 2026-09-17 (MC-GS
joint, GS-single pre/post, ZS-SSDU pre/post, 3D-DIP pre) were started by hand before this scheduler existed, so
their done() output files do not exist yet and the scheduler must not launch a duplicate. already_running() matches
on the script + key arguments, ignoring the GPU index (a hand launch may use a different card) and the interpreter
path (a hand launch may use a relative script path where build_jobs uses an absolute one)."""
import os
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
    marker = f"orch_guard_{os.getpid()}_{time.time_ns()}"
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


def test_guard_is_gpu_lane_only(tmp_path):
    """Review finding (2026-09-17): a cpu-lane job (summarize is the only one) must never be skipped by the
    guard, even if its command happens to match a running process -- a false-positive skip is never counted as a
    try, so it would stay "pending" forever and the run would never reach ALL_DONE_retro16. The controller's
    ruling was specifically about six hand-launched GPU jobs."""
    import retro_orchestrate as O
    marker = f"orch_guard_cpu_{os.getpid()}_{time.time_ns()}"
    proc, real_cmd = _spawn(tmp_path, marker, ["--scan", "pre", "--gpu", "0"])
    try:
        time.sleep(0.5)
        cpu_job = dict(lane="cpu", cmd=lambda g, c=real_cmd: c)
        gpu_job = dict(lane="gpu", cmd=lambda g, c=real_cmd: c)
        assert O.already_running(real_cmd)                 # sanity: the underlying match still fires
        assert not O.should_skip_as_running(cpu_job)        # ... but a cpu-lane job is never skipped for it
        assert O.should_skip_as_running(gpu_job)            # while a gpu-lane job with the same command is
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_bare_script_only_pattern_is_rejected_not_queried():
    """Review finding (2026-09-17): a command with neither a --gpu flag nor run_gs.sh's positional GPU argument
    (in practice: summarize's [python, -u, summarize.py]) must not fall through to a bare script-name pattern --
    that would match any same-user process merely containing the name, and this machine routinely runs 8+
    sessions of this user."""
    import retro_orchestrate as O
    cmd = [sys.executable, "-u", "/does/not/matter/summarize.py"]
    assert O._job_pattern(cmd) is None
    assert O.already_running(cmd) is False
