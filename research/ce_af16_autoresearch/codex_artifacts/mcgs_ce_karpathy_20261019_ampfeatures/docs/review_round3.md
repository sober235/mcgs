# Launch review, round 3

Actionable findings: **0**. Unresolved findings from rounds 1–2: **0**. This is the final incremental review of the supervision/cancellation correction; no GPU work was run and no production file was modified by the reviewer.

## Verified correction

`run_gpu.py:50–76` now records the first requested stop signal without asynchronously raising. Cancellation is checked after the Popen handle is assigned (`run_gpu.py:436`), during bounded context/normal-wait polling, and at other safe boundaries. `terminate_owned()` does not check or raise pending cancellation: it waits for its own process group to exit, escalating to SIGKILL after the grace period, while the common claim remains held. Pending cancellation is reported only after that lock-protected cleanup/reap completes. Lock acquisition is itself cancellable before a child exists.

The ordinary low-memory-context and SIGTERM regressions still pass. The added timeout-plus-first-SIGTERM regression verifies that an ignoring/slow-to-exit child retains the common claim during the grace period, then is reaped. The added real-spawn fixture sends SIGTERM after the actual child is created but before the wrapper returns its handle: the assigned-handle cancellation check subsequently cleans up that child. These tests close the round-2 resource lifecycle finding and the related spawn ownership boundary without changing numerical behavior.

The delegated independent resource reviewer also reran the prior timeout fixture while repeatedly sending TERM/INT: **489 lock checks** found no interval with a live child and an acquirable claim. The owned child was reaped, an unrelated process remained alive, the first SIGTERM was retained, exit code was 143, the audit record was `cancelled` with `scientific_rejection=false`, and no completion gate was emitted.

## Observed validation

- Independent command: `env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest -q -p no:cacheprovider tests/test_run_gpu.py` — **39 passed in 7.72s**.
- The resource reviewer separately ran the four cancellation/claim regressions — **4 passed** — and the repeated-signal fixture described above.
- The `run_gpu.py` SHA256 was checked before and after the tests and remained identical. The other four production files are unchanged from the complete round-1 review. Existing round-1 and round-2 reports were left unchanged.

## Final reviewed production binding

All paths below are relative to `/home/congcongliu/codex_artifacts/mcgs_ce_karpathy_20261019`.

| File | SHA256 |
| --- | --- |
| `amplitude_model.py` | `8d4faf7d936539ea3be551a421f1e9eeb680df2d8490939949014dd7c0bfb769` |
| `runtime.py` | `a42fb02828910c4ba3526439444b8170b4a4a2767ebae2c6a932c09ea26efd10` |
| `runner.py` | `fb5d2e3097e37c6bec84a3d5a5bae6e0452d1f37bb7835acc32be86f7ad91af4` |
| `evaluate.py` | `9a2c38d1942d6d6ad2331f3364dfad5be86a03cb316fec96f27deb7c5e40a875` |
| `run_gpu.py` | `50b79bb3c74e9465cbbeb0cc365310a802147929870f0fa4888625b3a2278610` |

There is no remaining review blocker for the already authorized zero-step and smoke stages using this binding. Their review/recipe gates must still bind fresh verified external invariants and the exact selected parameters through the implemented gate checks. This report does not claim that numerical zero-step reproduction or CUDA backward smoke has already passed: those actual GPU stages, memory measurements and later scientific comparisons remain to be executed and recorded.
