# G2 raw-render proximal-DC adapter

This branch is CPU preparation for one separately labeled **MC-GS+proxDC** route,
`g2_raw_proxdc_v1`, based on revision `000c53d9743183c7d5f365690bc4f83d9cf34273`.
Its starting diff was empty. No GPU job, clinical reconstruction score, selected
recipe, real passing gate, or main-loop change has been produced here. The
original main and paired-anchor worktrees and all OLD sources/data/results remain
unchanged. Do not run this branch's unchanged `autoresearch_loop.py`; this route
uses the separate `run_proxdc.py` entry point.

## Model, data and exact numerical sources

The source model is frozen G2 `adamflat_g2_v1`, checkpoint iteration4500,
SHA256 `282e0303c46a6c18775d6224c032611072bd465b4ef975ba7a7e2ea636230fd0`, with
3.6M Gaussian points, frequencyt1/geometryfactor2. Existing `runner.py` restores
the seven G2 model tensors using the original strict prebuilt CUDA voxelizer.
Only a zero-step baseline construction is used; no residual field is attached.
Optimizer/scheduler state is newly constructed and never stepped; this is an
export-only mapping, not original Cartesian Adam continuation. Model values and
identities, gradients, optimizer state/group identities, scheduler state and
configuration are snapshotted before/after export, and temporary forbidden-step
hooks block train/optimizer.step/scheduler.step and are restored correctly.

The numerical adapter uses the existing acquired-only dataset wrapper in
`runtime.py`. It retains the frozen reference-based normalization constants
already used by G2; it does not claim a new strict GT-free benchmark. It checks
42 coils, no constructed GT image, no nonzero unacquired measurements, unchanged
normalization, and exact acquired==fit masks, CSM and data. The child reads no
comparator or ROI and calls no scorer. The parent still hashes the existing
source/input freezes and three-metric contract without changing them.

The adapter imports these exact, read-only OLD files:

| Source | SHA256 |
|---|---|
| `proximal_dc/physics.py` | `7d85e86bce70ce2498b233acf7312dbcc9ed527038393a724f42cebb2f8fd59e` |
| `phase_firstorder/pcg.py` | `224a1552873dd0486f3a820d986aee3e76e0376d8c2006077588788666800897` |
| `proximal_dc/runner.py` (capture/state/work/export helpers only) | `da270d1ccddd2406bf5130723e2bfbcae0f1ea7c71b6bf0c3b147ac599c05502` |

`code_hashes()` rejects changes to these files and binds them with the new route,
the shared lifecycle, G2 constructor, runtime, evaluator and amplitude module.
The original trainer, transforms, prebuilt binary, frequency helpers and all
frozen input definitions are covered by the inherited fresh invariant snapshot.
The old H1 restore, model definition, selected recipes, four-metric gates and
scorer are never invoked. The evaluator is byte-identical to the starting revision.

## Raw anchor, fixed proximal rule and gauge

The old `canonical_control` helper wraps `trainer._render` while calling the
verbatim original `export_trainer_volume` for each scan. It captures that same
call's raw Gaussian volume and restores the temporary render/FFT hooks. No second
Gaussian render and no inversion of a hard-DC image is used. Before **any** solve,
both newly exported canonical arrays, in the unchanged reporting gauge, must
reproduce the saved G2 complex64 arrays with relative complex L2≤1e-5. A failed
scan blocks both solves; the check is repeated in the prox stage, even after a
separately passed canonical stage. Saved G2 hard-DC candidates are comparison
controls only and never become the anchor.

For each scan, the exact old operator computes in float64
`L=max_voxel sum_coils |CSM|²` and requests `lambda=0.1*L`. The effective lambda is
the same scalar rounded to the operand real dtype (float32 in this route), with
no floor. On nonzero-CSM support, solve
`(AᴴA+lambda I)x=Aᴴy+lambda(raw*support)` from `x0=raw*support`, where `A=MFS`
uses the original centered FFT convention and acquired mask. There is no lambda
sweep, external prior, learned model, GT/ROI fit, additional phase/amplitude fit,
or additional hard DC after solving. The candidate is the solver's verified image.

The fixed solver recipe is `max_matvecs=32, rtol=1e-5`. The 32 **CG** normal
applications include a nonzero-anchor initial residual evaluation. A separate
true-endpoint verification normal adds1; therefore total normal work is≤33.
Also count RHS adjoint1 and objective evaluations2 separately. With C coils,
B=ceil(C/chunk_size) and N actual CG matvecs, solver FFT batches=B*(N+3), IFFT
batches=B*(N+2), coil FFTs=C*(N+3), coil IFFTs=C*(N+2). Source construction,
canonical export and synthetic identity work are recorded separately and totaled.
No Gaussian render/VJP occurs inside the solver. Failed/budget-exhausted runs are
not retried or accepted, and partial work survives, including nonfinite failure
records serialized as strings in `progress.json` rather than passing evidence.

The old endpoint policy is retained: only converged/stationary outcomes; true
complex64 relative residual≤2e-5; finite nonincreasing J within its fixed
absolute1e-10/relative2e-6 tolerance; exact zero outside support; and negative
directional derivative for nonstationary updates. The stdlib route rechecks the
actual numeric values, stopping history, lambda rounding and complete work roles.

Raw, canonical and solved arrays use the unchanged legacy export order:
convert to complex64, multiply once by float32 `factor_scale`, convert to
complex64 and save. The factor must agree with
`scale_trainer/scale_gauge` within relative1e-15. Physics stays in trainer gauge.

## Independent stage contract and future commands

`run_gpu.execute(args, protocol=...)` now accepts explicit route hooks for code
hashes, invariants, parameter binding, gates, artifact validation and child path.
Its default remains the original amplitude route. Resource selection, visibility,
claim locking, deferred cancellation, timeout and owned-child cleanup remain
shared and retain their original regression coverage. A separate real CPU child
test exercises the alternate protocol through that same lifecycle. The new prox
child retains its CUDA context while waiting up to90 seconds for a matching
`childcontext_ack.json`. The parent atomically publishes that acknowledgment only
after it observes the ready child PID on the reserved GPU UUID in NVML. The child
checks its own PID, actual parent PID/liveness, UUID and observation flag before
numeric work; the parent binds the acknowledgment as a required artifact. The
immediate-child regression uses no artificial completion delay and only a
temporary claim lock; an unobserved context gets no acknowledgment and its owned
child is reaped before the temporary lock can be reacquired.

The public route parameters are exactly `route`, `stage`, `cg_max_matvecs=32`,
`cg_rtol=1e-5`, and `lambda_ratio=0.1`. Review JSON must explicitly name this
route, include the clean independent report descriptor, and bind the actual
current full source hashes/invariants. A selected recipe must name this route,
contain exactly the parsed CLI parameter object, and bind those same hashes and
invariants. A descriptor is verified before its JSON contents are used.

The complete `validate_stage_gates` rejects old H1/four-metric or amplitude gates.
It requires actual child artifacts, route schema, numeric/state/acquisition
evidence, unchanged before/after source/input proofs and unchanged controls.
The parent checks all independent tiny identity values against the original
2e-5 limits and positive finite curvature, their actual FFT roles, the complete
constructor/canonical/identity/solver counters, and every recomputed aggregate.
Canonical-only rows include their actual empty `roles` dictionary and strictly
zero solver counters. Positive artifact tests consume actual `tiny_stage` and
42-coil synthetic `reconstruct(canonical/prox)` results without fabricating
canonical work fields.
`canonical` requires a fresh observed route `tiny` gate. `prox` requires both
fresh route `tiny` and `canonical` gates. All gates must match the current complete
source/input snapshot. Any source change invalidates prior gates.

After a clean independent review and a newly selected recipe for **each** exact
stage, root may use the following real entry points from this worktree. These
commands were not executed. Replace the reviewed versioned paths with actual
fresh evidence; do not create placeholders claiming a pass.

```sh
/home/congcongliu/anaconda3/envs/nvgen/bin/python -B -u run_proxdc.py \
  --stage tiny --exp g2_proxdc_tiny_v1 --preferred-gpu 0 \
  --review-gate verification/proxdc_review_v1.json \
  --recipe recipes/g2_proxdc_tiny_v1.json

/home/congcongliu/anaconda3/envs/nvgen/bin/python -B -u run_proxdc.py \
  --stage canonical --exp g2_proxdc_canonical_v1 --preferred-gpu 0 \
  --review-gate verification/proxdc_review_v1.json \
  --recipe recipes/g2_proxdc_canonical_v1.json \
  --tiny-gate outputs/g2_proxdc_tiny_v1/completed.json

/home/congcongliu/anaconda3/envs/nvgen/bin/python -B -u run_proxdc.py \
  --stage prox --exp g2_proxdc_candidate_v1 --preferred-gpu 0 \
  --review-gate verification/proxdc_review_v1.json \
  --recipe recipes/g2_proxdc_candidate_v1.json \
  --tiny-gate outputs/g2_proxdc_tiny_v1/completed.json \
  --canonical-gate outputs/g2_proxdc_canonical_v1/completed.json
```

Before each command root must actually recompute `run_proxdc.code_hashes()` and
`run_proxdc.snapshot_invariants()`, dereference any handoff descriptor, parse the
exact public arguments and call the full `validate_stage_gates`. The stdlib
supervisor repeats these bindings before launch and afterward and owns its one
GSTMI child. GPU selection remains dynamic; it preserves device visibility,
requires65GiB free and retains the shared lock until the owned child exits.

The tiny stage checks synthetic CUDA SENSE identities/solver only; it makes no
G2 restore or renderer-parity claim. The canonical stage separately proves the
actual G2 strict renderer/export. Quality assessment follows immutable arrays
using the unchanged three-metric evaluator and all ten complete frozen opponents,
all59 planes and six ROI/background reports, outside this numerical adapter. A
common output correction alone is not evidence for a joint-model advantage;
provisional success still requires repeat and matched improved GS-single.

## CPU evidence and remaining uncertainty

The original baseline suite passed104 tests in11.04s. New tests were observed
failing before implementation for protocol injection, route/CLI, raw capture,
failure work persistence, scalar checks and export-only state checks. Actual old
physics/PCG was exercised on complex64/128 synthetic inputs against a direct
small-system solve. Tests cover same-call raw capture and hook restoration,
exact export gauge, no second hard DC, both canonical checks before any solve,
no retries/failed exports, acquisition mismatch, partial failed work, finite
endpoint/lambda/J checks, old-route/stale-input/missing-proof refusals, complete
stage chains, and actual alternate-protocol child lifecycle.

The first preparation suite passed145 tests but missed the three integration
defects recorded in `review_proxdc_round1.md`. That report remains unchanged.
The fixes add actual producer-to-validator regressions, a bounded context
acknowledgment and complete identity/work-evidence checks. The six corruptions
from that review and related role/counter/acknowledgment failures are rejected.
These fixes still require the second independent review before a numerical stage.

Final complete suite after these fixes:164 passed in13.82s; affected suites
passed99 tests in13.00s. The command was
`env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest -q -p no:cacheprovider`;
`git diff --check` exited0. No original tests were removed or modified.
GPU arithmetic, actual G2 canonical parity, acquisition-scale consistency on the
full case, peak memory, runtime, convergence and every quality metric remain
unverified for this route. No recovery changes external state: failed run
directories are preserved, and a new unique version is required for another run.
