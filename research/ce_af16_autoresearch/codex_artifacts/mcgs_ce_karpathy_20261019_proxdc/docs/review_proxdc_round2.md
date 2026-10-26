0 actionable findings. All three substantive round1 findings are resolved in the frozen numeric revision reviewed here; no unresolved numeric-review finding remains.

Independent round2 review of the exact incremental patch
`/tmp/proxdc_round2_increment_lj5q0ehy.patch`, SHA256
`32caf456e5668dc83c33b6be4f6bc698f20afd7a0cdd4d6b2b1298d5d7859fef`,
against `/tmp/proxdc_round1_before_lj5q0ehy/`. A generated unified diff of all
seven frozen changed files matched that supplied patch exactly. The affected
producer/validator, tiny identities/work aggregation and complete context-handshake
lifecycle were reopened for review. The previous mathematical/export review
remains applicable because the exact old physics/PCG, G2 constructor, acquired-only
runtime, evaluator and amplitude module are unchanged.

This review covers the numerical route and its shared supervisor increment.
The separately authored `proxdc_loop.py`, `tests/test_proxdc_loop.py` and
`docs/proxdc_queue.md` are outside this review. The reviewer edited only this
new report; the round1 report and all implementation, gates, recipes, numerical
outputs, state and journals were preserved. No GPU or actual clinical GT was
used, and no real passed-stage evidence was created.

| Round1 finding | Resolution and independent evidence |
| --- | --- |
| P1, canonical producer/consumer schema mismatch | The actual canonical producer now includes its real empty `roles` map. The parent requires the complete exact zero-counter schema. Positive artifact fixtures call the actual `reconstruct(canonical)` with 42 synthetic coils instead of manufacturing canonical work from a tiny solve. A separate CPU probe passed the actual result through the real parent validator successfully. |
| P2, fast tiny exit before context observation | The new route explicitly opts into `CONTEXT_ACK`; the default route does not. After the existing ready-file and NVML PID/UUID checks succeed, the parent atomically publishes `childcontext_ack.json`. The child retains its reserved context while waiting, checks its actual parent identity/liveness, and requires exact child PID, parent PID, GPU UUID and observation flag. The wait is bounded to at most90 seconds. The acknowledgment is required and hashed in the route artifact set. Immediate-child tests have no artificial completion delay; never-observed context receives no ACK and its child is reaped before the claim lock is reusable. |
| P2, unchecked independent identities and complete totals | The parent now requires both tiny identity records; finite, nonnegative identity errors at the original2e-5 limits; positive finite curvature; the fixed tiny coil/chunk recipe and exact independent FFT roles/counters. It requires the full work schema, validates constructor counters, checks exact canonical export counts, recomputes identity/solver aggregates and all scalar/coil/FFT totals, and rejects undeclared work. The original six bad-metadata cases are all refused by independent probes. |

The ACK change retains the existing ownership and cleanup boundaries: the shared
claim lock spans context observation, ACK publication, numerical work and owned
child termination. Deferred signal handling, Popen-handle ownership, bounded
waits and process-group cleanup are unchanged. A separate read-only lifecycle
reviewer found zero actionable findings after inspecting the affected default
and injected paths. Beyond the suite, that reviewer exercised actual CPU parent
death: the waiting child rejected parent loss and exited within0.150s, with
subreaper cleanup. An additional injected-path cancellation before ACK returned
signal15/context failure, child returncode-15, ready present, ACK/result absent,
no surviving owned PID, and a recoverable claim lock. These are synthetic CPU
process checks, not observations of a CUDA device.

The work validator continues to respect the original separation of work:
32 CG normal applications including any initial nonzero-anchor residual;
terminal normal1, RHS adjoint1 and objectives2 separately counted; construction,
canonical exports and tiny identity work are distinct components. The solver
and its precision, lambda, support, endpoint/J and descent checks were not
changed. Direct solver-image export with no final hard DC and both canonical
parity checks before any solve remain in place.

Observed independent CPU command, explicitly selecting all baseline and frozen
numerical-route test files so the parallel queue work is excluded:

```sh
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_amplitude_model.py tests/test_evaluate.py tests/test_loop.py \
  tests/test_run_gpu.py tests/test_runner.py tests/test_runtime.py \
  tests/test_proxdc_adapter.py tests/test_proxdc_runner.py tests/test_proxdc_supervisor.py
# 164 passed in 14.01s

git diff --check 000c53d
# exit 0
```

The second reviewer's narrower supervisor/ACK check passed **89 tests in12.53s**
(`test_run_gpu`, `test_proxdc_adapter`, `test_proxdc_supervisor`). Tests cover
missing/wrong child/parent/device/observation ACK, bounded timeout and parent
loss, context never observed, no-delay success, original default cancellation
and cleanup, real canonical/prox producer outputs, and corrupt identity/work
evidence. Temporary locks and temporary synthetic child/artifact directories
were used throughout.

An additional standalone CPU probe exited0 after checking all three actual
stage producers (`tiny_stage`, `reconstruct(canonical)`, `reconstruct(prox)`)
through the real artifact validator and confirming ACK membership in each
returned artifact set. It then reproduced the exact six round1 corruptions:

| Corruption applied consistently to fixture metrics/provenance | Actual refusal |
| --- | --- |
| Tiny: delete `tiny_identity` | `missing tiny identities` |
| Tiny: set pre forward-adjoint error to1.0 | `tiny forward_adjoint_relative failed` |
| Tiny: delete `work_total` | `complete work total is missing` |
| Tiny: set total FFT batches to0 | `complete actual work total differs` |
| Prox: delete `work_total` | `complete work total is missing` |
| Prox: replace totals with only zero FFT/render fields | `complete work total is missing` |

Final frozen SHA256 values independently checked:

| File | SHA256 |
| --- | --- |
| run_gpu.py | 983688ada3c2c3e37e4dcd0fe7cb64dbbc4dff587f302ed16fd99bad58136ac3 |
| run_proxdc.py | 894f43a120029b16a66f65900b1593959d675b85e6ef7003cdd790046aab0106 |
| proxdc_runner.py | 68a1b36c36ca29f62aec3b64371a0933e1b3f66edaae77471ddce1a055618124 |
| tests/test_proxdc_adapter.py | 74090cde6aeeb434c74660cb1c38821a999bf63f7b949d797c695259f2aaca90 |
| tests/test_proxdc_runner.py | e93972a0d3e40ca5448d7aa77566ccdbe6afe6ba40d7b40a93879554f72a9cc2 |
| tests/test_proxdc_supervisor.py | bd210563a0194734536dcf3b105771904b4c61beaa852404c87a5ed476ac220b |
| docs/proxdc_candidate.md | 57cff5e1f2508b9f234888e3a1101e200559c24d49ced7ec3f9993467964a644 |
| amplitude_model.py | 8d4faf7d936539ea3be551a421f1e9eeb680df2d8490939949014dd7c0bfb769 |
| runtime.py | a42fb02828910c4ba3526439444b8170b4a4a2767ebae2c6a932c09ea26efd10 |
| runner.py | fb5d2e3097e37c6bec84a3d5a5bae6e0452d1f37bb7835acc32be86f7ad91af4 |
| evaluate.py | 9a2c38d1942d6d6ad2331f3364dfad5be86a03cb316fec96f27deb7c5e40a875 |
| autoresearch_loop.py | d3a992826032b794feb34fbc1cfe2eb7ebe7ad48089a1d541fee63221f415bf2 |
| OLD/proximal_dc/physics.py | 7d85e86bce70ce2498b233acf7312dbcc9ed527038393a724f42cebb2f8fd59e |
| OLD/proximal_dc/runner.py | da270d1ccddd2406bf5130723e2bfbcae0f1ea7c71b6bf0c3b147ac599c05502 |
| OLD/phase_firstorder/pcg.py | 224a1552873dd0486f3a820d986aee3e76e0376d8c2006077588788666800897 |

`OLD` is the sibling `mcgs_ce_autoresearch_20261017` directory. The original
amplitude/runtime/G2-runner/evaluator/old-loop files were also compared
byte-for-byte with revision `000c53d` and remain identical. The supplied patch
and frozen source hashes remained stable throughout this review.

This is a clean CPU code/interface re-review of the numerical route. Actual
CUDA identity/parity stages, full-case acquisition/gauge consistency, peak
memory/runtime, convergence and every quality metric still require their own
genuine observed evidence. No scientific success or passed GPU stage is claimed.
