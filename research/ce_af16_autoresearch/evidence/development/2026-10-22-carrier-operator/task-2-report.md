# Task 2 source/CPU implementation report

Status: DONE for the assigned source/CPU portion; compilation and GPU numerical
acceptance intentionally remain pending independent source review. No build,
GPU execution, MRI reads, dependency installation, network operation, staging,
commit, historical-source edit, or optimizer integration was performed.

Owned deliverables: `carrier_operator.py`, `carrier_cuda/carrier.cpp`,
`carrier_cuda/carrier_kernel.cu`, `tests/test_carrier_cuda.py`,
`docs/carrier_operator_commands.md`, this report and `task-2-logs/`.

Baseline was HEAD `db841cb` (full revision recorded in baseline.json), with only
untracked root-owned `carrier_state.json`. Parent-owned spec/plan edits appeared
during implementation and were preserved. Task 1 reference and its tests have
the same SHA256 before and after this work. No `carrier_builds/` was created.

## Behavior and mathematical boundary

The public selected-row CUDA API validates shapes, common device, dtypes,
finiteness and strict frequency bounds, makes differentiable contiguous copies,
and chains centers and covariance back to positions, log scales and wxyz
quaternions. Its custom autograd boundary returns center, symmetric-M, complex
density and frequency gradients. First derivatives only; second derivatives are
explicitly excluded with `once_differentiable`.

The CUDA kernel processes stable detached tile pairs, with one block per tile
and disjoint voxel ownership per thread. It loops through only that tile's
selected atoms, has no block barriers, applies exact `mahal<=9`, and uses stable
`cos(theta)-1=-2*sin(theta/2)^2`. Backward includes the phase-center term and
atomic accumulations. Zero frequencies are never short-circuited. Empty K goes
through the custom boundary and returns connected zero gradients. Current CUDA
stream, device guard and launch checks are explicit. Raw binding validation
checks rank/dtype/device/contiguity/finite values, symmetric M, frequency bounds,
range partition, atom bounds, and duplicate/unsorted per-tile atom IDs before
kernel pointer access. Atom pointer offsets use int64 even though tile IDs and
pair counts use checked int32 limits.

Per controller rulings, support bounds derive from actual M rather than the
historical approximate covariance bbox. Reliable finite positive definite M
uses inverse-M axis radii; finite singular/poorly conditioned M receives all
volume tiles followed by exact kernel support. The historical floor/ceil/clamp
and stable atom ordering remain intact. Derived nonfinite geometry is rejected.
No historical bbox helper was edited. This can cost more for pathological
geometry and is not a claim of bitwise nonzero-support equality to the old
renderer on arbitrary inputs.

Self-review found actual float32 matrix-product asymmetry (up to 2.38e-7 in a
bounded fixture), which would violate the local symmetric-M center derivative
contract and raw symmetry check. A focused failing regression test was captured
before adding differentiable symmetrization. Averaging M with its transpose
retains its quadratic form and the original autograd chain.

## Loader and future gate

Import has no cpp_extension import or CUDA initialization. Production forward
refuses before a successful explicit `load_carrier_extension`. The loader
requires CUDA_HOME=/usr/local/cuda-11.8 before cpp_extension import and an explicit
architecture list. Build paths must be strictly under component carrier_builds;
foreign, nonempty unowned, partial and mismatched artifacts are refused.
Content identity binds new sources, wrapper, Torch/CUDA configuration/ABI,
compiler/nvcc paths/hashes/versions, flags and architectures. The returned
provenance and manifest add the loaded binary location/hash. There is no old
loader import, implicit forward compilation, or fallback backend.

Documented future GPU command creates a unique report/build directory, records
environment, exact command, source hashes before/after, separate stdout/stderr,
and preserves public failure status. GPU opt-in without availability fails.
The 13 numerical fixtures plus invalid-input test must all run; missing fixture
records make final report fail. Reports include output maxabs/relative errors,
VJP errors, three-run atomic variation, toolchain/binary identity and old binary
SHA before/after. Invalid raw bindings include out-of-range IDs, corrupt ranges,
uncovered/duplicate pairs, dtype and shape errors.

Twelve ordinary bounded fixtures compare all five public-input VJPs against
CPU double at unchanged rtol=2e-4/atol=2e-5. The additional near-singular tiny-q
fixture checks public output/support and local center/M/density/frequency VJPs
at those same tolerances using identical rounded float32 geometry converted to
double. It explicitly excludes ill-conditioned quaternion-chain relative
accuracy; it is not counted as an all-public-input double VJP claim. This
distinction was accepted by the controller before finalization. The CPU support
fixture proves a distant voxel remains supported and the degenerate atom reaches
all 12 tiles. Zero-frequency and tiny-phase tests include exact-zero old-gradient,
nonzero frequency-gradient and retained quadratic real-part assertions.

## Observed verification

The assigned test-driven CPU RED run failed because `carrier_operator` was
absent. Full focused run initially observed 21 failed and 13 explicit skips;
the captured `-x` reproduction records 1 failed/exit1, with exact command and
test source hash in `task-2-logs/red.json`, stdout/stderr alongside it. A later
focused GREEN observed 21 passed/13 skips, and expanded focused checks observed
24 passed/14 skips before the symmetry regression.

The symmetry regression specifically failed before its fix (exit1), as recorded
in `symmetry-red.{json,stdout,stderr}`. Its affected focused check after the fix
observed 1 passed/38 deselected, exit0. No native compilation was used to test
that behavior.

Final full component command was executed once after all source changes:

```sh
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest -q \
  -p no:cacheprovider tests/test_carrier_cuda.py tests/test_carrier_reference.py
```

Observed: **118 passed, 14 explicitly skipped, exit0**, 3.85 seconds pytest
runtime, empty stderr. The skipped tests are the 13 declared CUDA fixture cases
and invalid-input CUDA test; no CUDA numerical case was executed. The 118 passes
comprise 25 CPU operator-contract checks and 93 accepted reference checks.
`final-cpu.json` records exact argv/cwd/environment and all component source
hashes before/after; hashes were unchanged during the run. No unrelated legacy
606-test suite was run, per task scope. Whitespace checks for all five owned
deliverable files produced no findings (see whitespace.json). Exact new-source
diff is preserved in source.patch; final-status.txt captures concurrent state.

## Review and remaining uncertainty

Self-review of all new Python/C++/CUDA/test/doc source found no remaining known
actionable source issue after the symmetry and pointer-offset fixes. This is
not independent review. C++/CUDA compiler/ABI compatibility, raw native binding
behavior, CUDA outputs/VJPs, device/stream behavior, ragged execution, actual
atomic variation, binary provenance/reuse and historical binary invariance
during a build remain unverified until the independent reviewed build/microgate.
Full-scale runtime and all MRI integration/reconstruction/metric claims remain
outside this component phase. The controller should commit the exact diff and
obtain fresh task plus whole-component source review before authorizing a build.

## Round 1 review fixes (source/CPU only)

All four Important findings in `final-source-review-round1.md` were read and
addressed on source revision `198db2d` (full baseline revision/status/hashes in
`task-2-logs/round1-baseline.json`). This is the first fix round of the three-round
cap. Source approval, compilation, ABI checks and all GPU behavior remain pending
controller commit and scoped independent re-review. No native build, CUDA
execution, MRI reads, network, dependency installation, staging or commit was
performed. Task 1 reference and tests remain unchanged and were not rerun.

1. Finite-input intermediate overflow: `_geometry` now checks voxel centers,
   exponential scales and quaternion norms before normalization/inversion or
   entry to the custom autograd boundary. Finite log scales of 100 are rejected
   rather than allowing exp=inf to be concealed by finite zero M and later
   returning NaN through a zero local VJP. The analogous norm overflow and
   position-to-center overflow are also rejected. Finite singular-M support
   remains allowed; it is distinct from nonfinite conversion intermediates.
   Focused RED: three missing-refusal failures and one existing zero-VJP pass.
   GREEN: four passes, including finite, exactly zero original-input gradients
   from a zero local center/M VJP. Evidence: `round1-fix1-{red,green}.*`.

2. Effective compiler identity: the loader now requires the explicit command
   CXX=/usr/bin/c++ and the reviewed CUDA/bin:/usr/bin:/bin PATH. The CUDA host
   compiler is pinned via `-ccbin /usr/bin/gcc`; nvcc remains explicitly under
   CUDA_HOME=/usr/local/cuda-11.8. It refuses inherited CC/PYTORCH_NVCC and related
   compiler selectors, nvcc prepend/append flags, Torch CUDA/include/CUDNN/ROCM
   and ABI/dependency overrides, and implicit compiler/include/library search or
   flag overrides before tool invocation or cache reuse. Installed Torch JIT
   source was inspected read-only: CC and PYTORCH_NVCC, CUDA_INC_PATH, cached
   CUDNN/ROCM paths, ABI/dependency selectors and PATH-resolved compiler/Ninja
   behavior were the concrete build controls motivating this policy. Generic
   package or environment management was not introduced.

   Identity now records effective environment, explicit host compiler
   path/hash/version, builder source, Python ABI context, nvcc profile, Ninja
   executable/hash/version and flags. Provenance includes generated build.ninja
   text/hash, with its hash checked on artifact reuse. Cached cpp_extension
   CUDNN/ROCM overrides are refused alongside mismatched CUDA_HOME. The documented
   launcher explicitly removes refused overrides only in the child process and
   records their names; it sets the exact child compiler/PATH and asynchronous
   stream mode. It does not mutate the parent process environment.

   Focused policy RED: 33 failures for the absent effective-environment boundary.
   GREEN: 33 passes. Final suite additionally exercises all invalid-selector and
   unpinned-executable refusals through the public loader itself, with a guarded
   external-tool boundary preventing an accidental compiler invocation if refusal
   regresses. Architecture changes alter the recorded effective identity. Actual
   build/reuse identity remains unverified until the GPU build stage. Evidence:
   `round1-fix2-{red,green}.*` and `round1-final-cpu.*`.

3. Required nondefault-stream gate: a new mandatory numerical fixture uses a
   producer event, operator forward/all-public-input backward on a separate
   stream, and a consumer event that captures output and all gradient snapshots
   into pinned CPU memory. Linalg/autograd/allocator warmup completes before the
   tested sequence. The default stream is deliberately delayed; its completion
   event must still be incomplete when consumer capture finishes. Therefore a
   kernel accidentally queued on default cannot be rescued by a later global
   synchronization. Only the consumer event is synchronized before CPU snapshot
   comparison. Global synchronization afterward is cleanup. Expired delay or
   CUDA_LAUNCH_BLOCKING=1 causes explicit refusal/failure, not a stream pass.

   The fixture compares output, downstream consumer output and all five public
   input VJPs against CPU double at unchanged rtol=2e-4/atol=2e-5, repeats backward
   three times, and records variation, ordering and default-delay state. Report
   completeness now requires all fourteen numerical fixtures, including this
   stream case, plus the invalid-input test and historical binary invariance.
   CPU report-completeness RED: one missing-boundary failure. GREEN: one pass,
   specifically rejecting an otherwise complete report that omits the stream
   fixture. The actual GPU stream fixture was not executed; its runtime behavior
   remains explicitly unverified. Evidence: `round1-fix3-{red,green}.*`.

4. Tile dimension arithmetic: positive ceil-division is consistently
   `1+(n-1)/8` in Python/C++/CUDA instead of adding seven to int32 dimensions.
   CUDA tile counts, tile coordinates, padded voxel coordinates and flattened
   voxel offsets use int64. Immediate products were audited: each tile count is
   at most its positive axis size, so their product is at most the already
   checked voxel product <=INT32_MAX; each valid flattened voxel offset is below
   that product; padded coordinates can equal INT32_MAX and are rejected by the
   axis-bound check before pointer access. The existing int64 atom/range offsets
   remain intact. A tiny CPU test exercises the production validation/tile-layout
   helper with INT32_MAX on each axis separately and literal expected tile count
   268435456, plus a ragged small shape, without allocating volumes or huge tile
   arrays. RED: four missing-helper failures. GREEN: four passes. This checks
   production Python arithmetic and validates the shared formula; native code
   execution remains unverified because compilation is deferred. Evidence:
   `round1-fix4-{red,green}.*`.

After all source/doc changes, the affected component suite ran exactly once:

```sh
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest -q \
  -p no:cacheprovider tests/test_carrier_cuda.py
```

Observed **67 passed, 15 explicitly skipped, exit0**, 3.47 seconds pytest runtime,
empty stderr. Fifteen skips are fourteen CUDA numerical fixtures and the CUDA
invalid-input test, all still unexecuted. No reference suite repeat or unrelated
legacy suite was run. `round1-final-cpu.json` records exact command/cwd/environment
and source hashes before/after. Final freeze confirms all seven component files
still match that run, the two Task 1 files match the round baseline, and no
carrier_builds directory exists (`round1-final-freeze.json`). Incremental source
diff and status are in `round1-fix-source.patch` and `round1-final-status.txt`.

`git diff --check` passed. The documented Python launcher was parsed with
`ast.parse` without execution. The first syntax probe itself failed with
IndexError because its delimiter matched a literal backslash-n instead of a
newline; the corrected splitlines extraction parsed successfully. Both the
failed-probe cause and successful syntax/whitespace results are recorded in
`round1-docs-check.json`; this was a verification-probe error, not a launcher or
native-build execution failure.

Separate source self-review of the complete incremental changes found no known
remaining actionable finding among these four requests. This does not substitute
for independent re-review. The substantive remaining uncertainty is unchanged:
actual compiler/API/ABI compatibility, effective artifact build/reuse, CUDA
numerics, stream/device execution and measured atomics require the reviewed
isolated GPU gate. No tolerance, numerical scope, historical source, MRI input,
optimizer, score or benchmark protocol was changed.

Additional controller-requested F1 source inspection: finite positions around
1e30 with finite log scales around 80 can produce finite centers and underflowed
zero M. At zero frequency this does not recreate the reported NaN path: the
kernel evaluates `-.5f*weight*Q*d[x]*d[y]` with Q=0 before either finite large
displacement multiplication, rather than materializing overflowing d*d; Md=0
and phase=0 as well. The scale chain has finite exp(80), so its underflowed zero
derivative does not multiply infinity. No further test was run for this source
inspection, as requested. The controller accepted that resolution and explicitly
declined a broader arbitrary numeric-range policy. Bounded-fixture accuracy
limitations remain unchanged; no guarantee of representable float32 gradients
for every arbitrarily large finite nonzero-frequency input is claimed.

## First GPU gate: noncontiguous fixture correction (test-only)

The controller's first isolated GPU gate built the extension successfully but
returned public exit1: 1 failed, 15 passed, 66 deselected, and one teardown error.
The complete stdout/stderr and command/report at
`carrier_verification/20261023T011403Z-bc86c383e901/` were read before editing.
The sole case failure was the noncontiguous-input assertion before the operator
call. The report had all thirteen other numerical fixture records marked passed,
including the required nondefault-stream case, and `invalid_inputs=true`. The
teardown error was report completeness correctly refusing the absent
noncontiguous case, not a separate invalid-input failure. That failed gate and
all its source-bound evidence are preserved; it remains a failed overall gate.

A CPU-only probe confirmed the concrete layout cause: for density shape[1],
the view `torch.stack((density,density),-1)[...,0]` has stride[2] but PyTorch still
reports contiguous because the only axis has length one. With K=2, shape[2] and
the same stride[2] are genuinely noncontiguous. Values are preserved in both
cases, and CUDA remained uninitialized. Probe results are recorded in
`task-2-logs/noncontiguous-layout-probe.json`; starting revision, component
hashes and hashes of all four failed-gate files are in
`noncontiguous-fix-baseline.json`.

Only `tests/test_carrier_cuda.py` changed. The real GPU fixture now shares its
row-count and strided-view constructors with a focused CPU regression. The
noncontiguous case alone changes from K=1 to K=2; other fixture row counts stay
the same. Its required `all(not x.is_contiguous())` assertion remains in the
actual GPU test. Oracle comparisons, all VJPs, numeric tolerances and report
completeness are unchanged. The regression exercises those same constructors,
checks all five inputs are noncontiguous with values/shapes preserved, and
checks their leaf gradient connections through contiguous copies, including
complex density. It does not call a production CPU backend or claim CUDA parity.

The focused CPU regression was added against the original K=1 construction and
observed RED at the same noncontiguous assertion: **1 failed, 82 deselected,
exit1**. After the single K=2 correction, the identical command observed GREEN:
**1 passed, 82 deselected, exit0**, 1.35 seconds pytest runtime, empty stderr.

```sh
env CUDA_VISIBLE_DEVICES= CARRIER_RUN_GPU=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest -q \
  -p no:cacheprovider tests/test_carrier_cuda.py \
  -k noncontiguous_fixture_preserves
```

Exact command/cwd/environment, stdout/stderr, exit status and source hashes
before/after are captured in `noncontiguous-fix-{red,green}.*`. No other suite,
GPU execution, compilation, subagent, MRI read, staging or commit was performed
in this correction dispatch. `git diff --check` passed. Incremental test diff
is `noncontiguous-fix-source.patch`. Final test SHA256 is
`4a936e9b6642e65b8abe008d640cbfceda4f8b316dc3acf4ee6b1951e7f9a0e8`.
The authoritative current hash and full source/evidence freeze are recorded in
`noncontiguous-fix-final-freeze.json`; all production/reference files and all
four failed-gate evidence files match the starting hashes. The changed test
hash matches its focused GREEN run before and after.

No production defect was identified in this failure. Production sources and
the loader's source/toolchain identity are unchanged, permitting the controller
to request verified reuse of its existing dedicated artifact after reviewing
this test-only correction. The corrected noncontiguous case has not yet run on
GPU. A fresh uniquely reported GPU gate remains required before declaring the
overall component gate passed.
