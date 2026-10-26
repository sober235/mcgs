# Launch review, round 2

Actionable findings: **1 unresolved**, restricted to the remaining claim/cancellation interaction. The ordinary successful-handshake and ordinary SIGTERM paths from round 1 are repaired. No GPU work was run and no production file was modified by the reviewer.

Reviewed `run_gpu.py` SHA256: `b9e5fe5deb6ea8762443f0a4056697fc0e493251dcec818a80cc9f0074b62302`. The other four production files retain their round-1 contents.

## Finding

1. **P1 — The first stop signal can interrupt timeout cleanup and release the claim while the child remains alive.** `run_gpu.py:413–421`: the inner `terminate_owned(worker)` is still interruptible by the newly installed SIGTERM/SIGINT handler. If the child has timed out and cleanup is waiting for it to exit, a first SIGTERM raises `SupervisorCancelled` from inside that cleanup. Python then unwinds `with claim_lock()` before reaching the outer `finally`, which calls `terminate_owned()` again. Until that second cleanup completes, another cooperating supervisor can acquire the common claim even though the first child is still running, violating the retained-claim invariant.

   This was verified with the existing `cpu_supervisor_process()` harness using the real production `execute()`, Popen/session handling, lock and signal handler. Only the CPU fixture child was changed to ignore TERM and write a `term_seen` file, and the numerical timeout was shortened to one second. After observing the TERM sent by timeout cleanup, the reviewer sent SIGTERM to the supervisor from outside that process. The following were true simultaneously:

   ```text
   claim_acquired_during_cleanup=true
   supervisor_alive=true
   child_alive=true
   ```

   About five seconds later, the supervisor exited 143 with a `cancelled` record and reaped the child. There was no exception injected into the supervisor's code in this reproduction. The impact is a temporary duplicate GPU reservation during cancellation of an unresponsive/slow-to-exit child. Make owned-child cleanup/reap noninterruptible by stop requests, retaining the claim until exit; add this combined timeout-plus-first-signal regression. A handler that records a pending cancellation, with exceptions raised only at controlled points outside cleanup, is a suitable focused approach.

## Verified resolutions and checks

- The common lock now encloses the successful context handshake, complete numerical wait and ordinary exceptional cleanup. The normal low-memory-context race is closed, with the deliberately serialized `common_lock_until_child_exit_serialized` policy recorded.
- Ordinary SIGTERM is converted to cancellation, the owned process group is reaped, an unrelated process remains alive, the cancellation record is persisted with `scientific_rejection=false`, exit code is 143, and the lock is released after the child is reaped.
- NaN refusal evidence is persisted as `refused_setup` with a string request value. The explicit external numerical adapter hashes are present in `snapshot_invariants()`; its corresponding regression passed.
- Independent command: `env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest -q -p no:cacheprovider tests/test_run_gpu.py` — **37 passed in 0.99s**. The delegated independent resource reviewer also ran the four directly relevant regressions successfully before testing the remaining combined failure path.
- The round-1 report was left unchanged. Numerical/scoring code was not re-reviewed because its contents and architecture did not change. Actual CUDA zero-step reproduction, gradient smoke and resource requirements remain for the authorized numerical stages.

Do not create a clean review gate for this hash while the finding remains unresolved. A final incremental review should cover the cancellation lifecycle correction and its CPU evidence; another full numerical review is unnecessary unless scope changes.
