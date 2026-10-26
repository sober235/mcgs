3 actionable findings: one P1 and two P2. All three are unresolved in the frozen revision reviewed here. This is not a clean numerical-launch review.

Independent review of the complete uncommitted change against
`000c53d9743183c7d5f365690bc4f83d9cf34273` in this isolated proxDC worktree.
The frozen increment contains optional protocol hooks in `run_gpu.py`, new
`run_proxdc.py`/`proxdc_runner.py`, three new test files and
`docs/proxdc_candidate.md`. Read the applicable AGENTS/design, old proximal
AGENTS/PLAN, exact old physics/PCG/export helpers and original supervisor
lifecycle. Old H1/four-metric gates were treated as obsolete reference material,
not evidence for the new G2/three-metric route. A second read-only reviewer
independently examined the supervisor/protocol and lifecycle. Reviewers modified
no implementation, inputs, gates, recipes, runtime state or scores, and used no
GPU or actual clinical GT. This reviewer authored only this report.

1. **P1 — Actual canonical output cannot pass its parent validator.**
   Location: `proxdc_runner.py:94`, consumed by `run_proxdc.py:157`.
   Trigger: a successful real `reconstruct(..., stage='canonical')` with both
   parity checks passing. `dict(operator.work)` contains the twelve zero solver
   counters but no `roles` key. The parent requires `work.get('roles') == {}`,
   so it rejects that valid output with
   `ValueError: canonical control has nonzero solver work`. This prevents a
   completed canonical gate and therefore blocks the declared prox sequence.
   An independent CPU probe used the actual frozen export helper and
   `reconstruct` on synthetic source tensors, then inserted its actual returned
   result into the route's otherwise valid child-artifact fixture and called
   the real `validate_child_artifacts`: the error reproduced. All counters were
   zero. Existing supervisor tests manually construct canonical work from a
   tiny solve and insert `roles={}`, concealing the producer/consumer mismatch.
   Fix the canonical work schema and add a regression that passes the actual
   canonical reconstruction result to the real artifact validator.

2. **P2 — A fast successful tiny child can exit before the context handshake is observed.**
   Location: `run_gpu.py:373`/`:381`, newly exposed by
   `proxdc_runner.py:221` through its tiny path.
   Trigger: after publishing `childcontext_ready.json`, the small child finishes
   within the parent's 250ms polling interval. `wait_context` checks
   `worker.poll()` before inspecting readiness/NVML and rejects an exited child,
   even when it exited successfully after producing its result. The new child
   has no acknowledgment wait. The alternate-protocol test includes an
   artificial `.5s` sleep that avoids this race. A separate real CPU
   injected-protocol probe reused that test and changed only the generated
   temporary child's `time.sleep(.5)` to `time.sleep(0)`; in 0.270s the actual
   lifecycle raised `child exited before its CUDA context became visible`,
   phase `context`, despite child returncode0 and ready/result files both
   present. The owned PID had been reaped and the claim lock was recoverable,
   so cleanup works while the valid fast control is spuriously rejected.
   This is inherited lifecycle logic newly exposed by the tiny route, not a
   regression in the original slower default numerical path. Use a real
   acknowledgment: after parent observation of the ready PID/UUID in NVML,
   atomically publish a matching child-PID/parent-PID/GPU-UUID acknowledgment;
   the prox child should wait with a bound and parent-liveness check before
   numeric work. Bind this acknowledgment in the route artifacts. Retain the
   owned-child/lock/cancellation behavior and add a genuine immediate-child
   regression instead of depending on a delay.

3. **P2 — Completed route gates do not validate independent tiny identities or complete work evidence.**
   Location: `run_proxdc.py:150` through `:167`.
   Trigger: metrics and provenance agree with each other but omit or contradict
   `tiny_identity` or `work_total`. The tiny branch accepts
   `tiny_checks_passed=True` and validates only each solver row; it never reads
   the independent identity values/work. No stage checks the total construction,
   canonical, identity and solver counters against their component evidence.
   Six independent temporary-fixture probes using real CPU tiny solver results
   were all accepted by the actual artifact validator after synchronously
   changing metrics/provenance: tiny without `tiny_identity`; tiny with
   `tiny_identity.pre.forward_adjoint_relative=1.0`; tiny without `work_total`;
   tiny with `work_total.total_fft_batches=0`; prox without `work_total`; and
   prox with `work_total={total_fft_batches:0, gaussian_renders:0}`. Such artifact
   sets can subsequently be wrapped by the normal observed-gate path and used
   as prior-stage evidence. The claimed independent CUDA identity check and
   disclosed complete compute accounting are therefore not enforced by the
   parent contract. Add explicit finite identity/curvature/tolerance checks
   matching the exact old helper, require all work fields, and recompute the
   construction/canonical/identity/solver totals and roles. Positive tests must
   consume real route results, with negative tests for these corruptions.

The remaining inspected numerical path follows the declared intervention:
G2 checkpoint hash/iteration/frequency recipe are checked before the existing
strict renderer construction; only baseline model tensors are attached; the
acquired-only wrapper and exact acquired/fit mask/CSM/data checks are retained.
The same canonical export call supplies the raw Gaussian anchor. Both scans'
canonical parity checks run before the first solve, including in the prox stage.
Physics uses the exact old dtype-consistent SENSE/PCG source, float64 coil-power
L, lambda0.1L rounded to the operand real dtype, max32 CG matvecs including the
nonzero initial residual, with RHS1/terminal-normal1/objectives2 separately
counted. The candidate is the verified solver image in reporting gauge, with
no subsequent hard DC. Before/after export state snapshots and restored temporary
forbidden-step hooks protect model/optimizer/scheduler/gradient/config state.
No additional verified defect was found in these paths during this pass.

The old exact helper performs its own tiny identity checks internally, and the
current child normally computes complete work totals. Finding3 concerns the
new parent evidence contract; it does not claim the observed CPU arithmetic
failed or that the existing helper produced the six corrupted records.

Observed independent checks:

```sh
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest -q -p no:cacheprovider
# 145 passed in 12.56s

git diff --check 000c53d
# exit 0
```

The second reviewer's scoped run of `tests/test_run_gpu.py`,
`tests/test_proxdc_supervisor.py` and `tests/test_proxdc_adapter.py` passed
**70 tests in 11.70s**. These green suite results are distinct from the
additional uncovered-path probes above. All probes used temporary files and
synthetic CPU data; no real passing gates were created.

Frozen SHA256 values reviewed before any fixes:

| File | SHA256 |
| --- | --- |
| run_gpu.py | 0f3d119204fca6c2899c49e24bf4902aa227d542be39ded8870c66629a4c7859 |
| run_proxdc.py | 42d5c32de03b1a6ca00b05d45040854c45193b18b4f95fae928ea591f7b38113 |
| proxdc_runner.py | 750ee3eed6920baf501820b657fbbbdc4f512c682469437133944be823c3c73f |
| tests/test_proxdc_adapter.py | 8d73ea97b9738ca9ad20ccef22de4c9bae8078cecd49e1e182f5ed40b671b5ac |
| tests/test_proxdc_runner.py | ff37db0a66f7a942b3638d5c1920fad4c7fcca781624abe1aceca2c1aff0bf23 |
| tests/test_proxdc_supervisor.py | 1c51c4b4aee00c9034e2b1abe4ea9f61dc6e2d5a45ea105bfff235f02db298ef |
| docs/proxdc_candidate.md | 2c48ebfb7c9eb7bbf475bc17ba705bd7ef0a14451009b53babfd34f621938131 |
| runtime.py | a42fb02828910c4ba3526439444b8170b4a4a2767ebae2c6a932c09ea26efd10 |
| evaluate.py | 9a2c38d1942d6d6ad2331f3364dfad5be86a03cb316fec96f27deb7c5e40a875 |
| runner.py | fb5d2e3097e37c6bec84a3d5a5bae6e0452d1f37bb7835acc32be86f7ad91af4 |
| amplitude_model.py | 8d4faf7d936539ea3be551a421f1e9eeb680df2d8490939949014dd7c0bfb769 |
| autoresearch_loop.py | d3a992826032b794feb34fbc1cfe2eb7ebe7ad48089a1d541fee63221f415bf2 |

The final five files in the table are byte-identical to the base revision.
The external physics, PCG and export helpers are separately fixed by the new
route's explicit SHA256 mapping; OLD files were read only. The `run_gpu.py`
increment retains the default protocol and existing critical cleanup path,
subject to the newly exposed fast-child issue above.

Fix the three findings, freeze the resulting increment and perform the next
independent review before a real route launch. No GPU parity, full-case
convergence, runtime/memory or reconstruction-quality result is established by
this review; none was run. Failed/superseded evidence should remain preserved.
