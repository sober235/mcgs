# Complex Carrier Operator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a separately built, differentiable fixed-subset carrier correction with CPU derivative and CUDA parity evidence before any MRI integration.

**Architecture:** Original full Gaussian renderer remains unchanged. A new selected-row correction implements G*rho*(exp(i*theta)-1); a dense float64 CPU oracle checks a sparse tile-based float32 CUDA operator. Original geometry conventions and hard support remain identical.

**Tech Stack:** Existing GSTMI Python/Torch2.7.1+cu118, C++/CUDA extension, pytest. No dependency upgrade.

**Spec:** docs/superpowers/specs/2026-10-22-carrier-operator-design.md

## Global Constraints

- Do not modify any historical source, CUDA binary, data, mask, checkpoint or score. New files only; all build products have a new dedicated directory and module identity. Retain negative evidence.
- Tensor axis order D,H,W. h=shape/2, C=(positions+1)*h-0.5; normalized displacement d=(v-C)/h; s=exp(log_scales); quaternion wxyz q/(norm(q)+1e-8); M=R diag((s+1e-8)^(-2)) R^T.
- Envelope exp(-d^T M d/2) inside mahal<=9, zero outside; support/tile decisions detached. Frequency cycles per voxel strictly (-0.5,0.5). Stable cos(theta)-1=-2*sin(theta/2)^2.
- CPU algebraic/autograd rtol=atol=1e-10; finite-difference rtol1e-5/atol1e-7 away from support crossings. CUDA bounded-fixture rtol2e-4/atol2e-5; zero correction and old-input correction gradients exactly zero at zero frequency; constructed frequency derivative nonzero.
- No MRI data or benchmark fitting in this phase, no GPU until source review and resource inspection. No fallback backend or implicit production compile. No claim of MRI gain from microgates.
- Use original seven-state rendering/training sources only as read-only reference. No integration scaffold, optimizer or selected MRI subset in this phase.
- Each worker owns only listed files/report, preserves others, does not commit or spawn reviewers. Controller stages/commits and dispatches fresh review. User review/fix cap3; preserve all evidence. Run focused checks, not unrelated legacy606 tests.

## Review Focus

- A zero-frequency fast path can erase nonzero frequency gradients: Task1/2 use asymmetric imaginary upstream example.
- Center-dependent phase has an additional gradient term: Task1 finite differences and Task2 CPU-VJP comparisons use nonzero carriers and anisotropic shape.
- Ragged tiles or malformed launch indices can cause invalid memory access: Task2 uses partial tiles plus exposed bad-index refusal and launch checks.
- CPU oracle silently downcasts or shares the same derivative mistake: Task1 double gradients versus independent direct Gabor formula and central differences; Task2 oracle remains untouched.
- Build/device selection can alter historical artifacts or use a wrong stream/backend: Task2 provenance/binary hash checks, nondefault stream and same-device checks; explicit CPU-empty visibility for CPU task.

### Task 1: Double precision reference and derivative acceptance

**Files:** Create carrier_reference.py and tests/test_carrier_reference.py. Report owned task-1-report.md in this plan's SDD workspace.

**Interfaces:** Produces `carrier_correction_reference(positions, log_scales, rotations, density, frequencies, volume_shape)` CPU float64/complex128 autograd output; selected K rows in shapes[K,3],[K,3],[K,4],[K],[K,3]. `volume_shape` three positive ints. Reject invalid shapes/dtypes/devices/nonfinite/frequencies outside strict Nyquist. EmptyK supported with connected zero gradients. Geometry/internal helpers may be private. No CUDA import/build side effect.

- [ ] Write failing tests for direct-Gabor-minus-Gaussian algebra, all-five-input derivative correctness, zero-frequency old-input gradient zeros/nonzero frequency example, anisotropic quarter-cycle units, emptyK/overlap/outside support, noncontiguous inputs and validation refusal. Use exact acceptance tolerances and central differences avoiding support boundary crossings. Independent direct formula must not call implementation geometry helpers; analytic VJP checks include center phase term.
- [ ] Run RED with `env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest -q -p no:cacheprovider tests/test_carrier_reference.py`; record actual failure and exit.
- [ ] Implement minimal reference in carrier_reference.py using native Torch float64 autograd, stable correction and original geometry convention. No hardcoded test output or production fallback.
- [ ] Run same focused command GREEN once; self-review calculations/input error paths; fixes rerun affected tests only. Freeze source/test hashes and output/exit evidence. Confirm historical224 source closure unchanged.
- [ ] Controller inspects report/current diff, commits only owned two files, generates exact BASE..HEAD package and dispatches fresh task spec/quality review. Resolve substantive findings within3 rounds before Task2. No independent suite rerun absent a named doubt.

### Task 2: Sparse CUDA correction, isolated build and microgate

**Files:** Create carrier_operator.py, carrier_cuda/carrier.cpp, carrier_cuda/carrier_kernel.cu, tests/test_carrier_cuda.py and docs/carrier_operator_commands.md. Task1 oracle unchanged. Report owned task-2-report.md in this plan workspace.

**Interfaces:** Consumes Task1 oracle signature. Produces `carrier_operator.load_carrier_extension(build_directory)` -> (module, provenance_dict), and `carrier_operator.carrier_correction(positions, log_scales, rotations, density, frequencies, volume_shape)` CUDAfloat32/complex64 tensor. API exactly matches reference selected-row inputs; wrapper chains normalized centers/inverse covariance to original differentiable tensors. Loader explicitly owns dedicated build dir and content-specific module, validates Torch/CUDA/compiler/source/binary identity; repeated identical load can reuse its own verified artifact. Forward without explicit successful load refuses, except argument validation may run first; no old loader fallback.

- [ ] Write CPU-runnable tests for import without CUDA initialization, explicit-load requirement, dtype/shape/frequency rejection where possible, source/build destination isolation; GPU tests for all fixture and VJP cases in spec. Missing GPU is explicitly skipped, not reported as numerical pass. Write actual microgate command to run GPU tests only after review; persist per-fixture maxabs/relative differences and repeatedVJP variation in unique verification output (JSON), with binary/source provenance.
- [ ] Run focused CPU RED, implement wrapper and independent C++/CUDA kernel. Reuse original detached tile membership semantics read-only; do not copy its unsafe thread-dependent barrier pattern. Explicit device guard/current stream/rankdtypecontiguity/index checks and launch checks. New bounds use actual inverse-M conservative radii; finite singular/ill-conditioned atoms without reliable inverse receive explicit full-volume bounds, then exactmahal test. Reject nonfinite derived geometry, clamp before integer conversion; add near-boundary non-unit quaternion and degenerate-support cases. Historical bbox bytes remain untouched. Keep selected subset only, no all-volume-by-all-atoms dense production loop. EmptyK is differentiable zeros; zero frequency still enters derivative-capable path. No unsupported autograd claims beyond first derivatives.
- [ ] Run focused CPU checks and Task1 reference checks because its interface is now consumed. Capture source-bound evidence. Do not compile or run GPU before independent source review. Controller commits new files and requests fresh task+whole-component review against complete component diff; actual compilation/microgate remains explicitly unverified.
- [ ] Controller or assigned worker under controller authorization inspects realGPU/utilization/memory/PIDs/visibility, selects idle compatible device, builds only dedicated new extension under unique carrier build directory with existing GSTMI/toolchain, then executes documented GPU microgate. Preserve compilation stdout/stderr and failures. Record actual command/environment/source hashes/compiler/ABI/binary hash plus unchanged historical binary3973daeef5f55ebfed931c38538a37d83921a9287b96b69a431da238057194b5.
- [ ] If build/numerical failure, investigate root cause and fix only newcomponent; focused RED/GREEN + scoped independent review for changed code, max3 rounds. Never relax frozen numeric tolerance after failures. Actual passing gate requires all declared cases exercised, no unexpected skips; deterministic report and publicexit0.
- [ ] Independent scoped evidence review confirms actual microgate results and limitations without rebuilding/repeating passed tests. Controller commits final docs/state. If acceptance passes, proceed to a separate MRI integration design; if not, record refusal and choose next supported route. No formalMRI count increment for any component checks.

## Self-review

Coverage: spec math/interfaces inTask1+2; independent CPU oracle is Task1 output consumed byTask2; all five ReviewFocus risks pinned above. Build comes after source review; new numerical changes after build failures return through scopedreview. No task changes historical source or input. CPU/GPU checks are distinct and skippedGPU is never a pass. Small fixtures constrain tolerances but do not promise fullscale stability/performance; next integration phase is explicit. Builder check returned nothing important left. Existing autonomous execution/SDD authorization applies; no redundant user confirmation.
