# Full Amplitude Representation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Measure true shared amplitude representation of G2 without MRI data or evaluation leakage.
**Architecture:** A pure positive amplitude field and the original raw Gaussian renderer feed a bounded G2-only teacher fit. The inherited reviewed supervisor owns all GPU processes; a separate stdlib protocol validates actual evidence.
**Tech Stack:** Existing GSTMI/PyTorch, NumPy, original strict CUDA binary; no installs.
**Spec:** docs/superpowers/specs/2026-10-22-full-amplitude-representation-design.md

## Global Constraints

- Preserve original/completed sources/data; edit only the new worktree and assigned files.
- No kspace/CSM/reference/ROI/scorer loading, external data/weights, new packages or renderer compilation/fallback.
- G2 checkpoint SHA282e0303c46a6c18775d6224c032611072bd465b4ef975ba7a7e2ea636230fd0; binary SHA3973daeef5f55ebfed931c38538a37d83921a9287b96b69a431da238057194b5.
- Frozen3.6M geometry/source phases; positive shared amplitudes with no per-point amplitude residual.
- MLP/grid64 decoder, coefficient2000*32768 updates then raw128 updates; phase frozen; seed42; LR.001 Adam, clip1.0.
- Fresh original raw parity<=1e-5; representation admission both raw errors<=0.01. Admission is not MRI quality.
- Only PSNR/SSIM/NMSE decide later MRI quality; these diagnostics add zero formal MRI trials.
- Parent stdlib and unchanged owned-child lifecycle/ACK; actual code/input/review/recipe binding before GPU.

## Review Focus

1. A function accidentally retains per-point moduli or updates phases, making a nominal full model a residual shortcut.
2. Grid axes, coordinate units, nonfinite/zero inputs and chunk rematerialization corrupt amplitudes or gradients.
3. A different kernel, implicit JIT/fallback, log-scale roundtrip or reversed gauge defeats faithful raw calibration.
4. Self-reported admission, incomplete optimizer/render counters or unbound artifacts masquerade as passed evidence.
5. Fast-child ACK, cancellation and isolated output lifecycle must remain the inherited reviewed behavior.

---

### Task 1: Full positive amplitude field

**Files:** full_amplitude.py; tests/test_full_amplitude.py.
**Interfaces:** consumes frozen positions/logscales/coefficient vectors. Produces
FullAmplitudeField constructor/magnitudes/get_density/amplitude_parameters/metadata
with the exact signatures and shapes in the spec. phase_delta is accessible.

- [x] Write tests for duplicate-query moduli despite differing source amplitudes,
  positivity, both encoding modes, independent phase gradients, frozen private
  buffers, no per-point amplitude parameter, point selection, chunk/full outputs
  and parameter gradients, state roundtrip and invalid/zero inputs.
- [x] Observe targeted RED for the missing class or behavior.
- [x] Implement full_amplitude.py with fixed encoders/initialization and
  nonreentrant chunks. Do not add training, renderer or input loading.
- [x] Run `env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest tests/test_full_amplitude.py -q -p no:cacheprovider`; expect PASS.
- [x] Commit only owned files and record actual RED/GREEN evidence.

### Task 2: Faithful standalone raw renderer and frozen inputs

**Files:** raw_g2.py; tests/test_raw_g2.py.
**Interfaces:** `source_definition()->dict` returns exact absolute input paths,
expected hashes, volume/gauges/source names without importing Torch. Child-only
`construct_source(device)->(model,voxelizer)` validates bytes and restores original
model; `render_density(model,voxelizer,density)->complex volume` keeps density VJP;
`render_source(model,voxelizer,channel)`; `report_array(image,scan)->complex64 numpy`;
`source_hashes(model)->dict`; `verify_raw_anchor(image,scan)->dict` records actual
equality and relative L2. Torch imports must be lazy so parent can use definitions.

- [x] Write tests for exact seven-key/shape restoration, exp(logscale) constructor
  followed by strict load, correct original voxelizer call, connected density VJP,
  fixed gauge arithmetic, reversed/wrong anchor rejection, changed source/binary
  refusal and compile/fallback/source-identity refusal. Real CPU small tensor
  adapters may stand in for CUDA; do not claim they validate the GPU kernel.
- [x] Observe RED for missing functions/incorrect restoration.
- [x] Implement only original-source binding, definitions and raw calls. Never
  instantiate JointTrainer or load MRI data.
- [x] Run `env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest tests/test_raw_g2.py -q -p no:cacheprovider`; expect PASS.
- [x] Commit only owned files and record RED/GREEN evidence.

### Task 3: Bounded fitting child and stdlib evidence protocol

**Files:** representation_runner.py; run_representation.py;
tests/test_representation_runner.py; tests/test_representation_protocol.py;
docs/representation_commands.md; program.md (new-route binding only).
**Interfaces:** consumes Task1/2 APIs, inherited run_gpu.execute(protocol=...),
proxdc_runner.wait_context_ack and the fixed source/fit specification. Produces
public `run_representation.py --stage render_control|fit --encoder none|mlp|grid
--exp NAME --review-gate PATH --recipe PATH [--render-gate PATH] --preferred-gpu N`
with fixed budget/thresholds in the exact parameters object. Own child launch
is representation_runner.py, CONTEXT_ACK=True. The inherited files stay unchanged.

- [x] Write tests of the real raw teacher/coefficient fit on small CPU adapters:
  exact update counts, Adam reset between phases, only amplitude parameters update,
  fixed phase/source, actual two-channel relative-error arithmetic, gradient
  finiteness, final checkpoint and no dependence on MRI files. Parameterize the
  same architecture contracts as Task1.
- [x] Write protocol negative tests: fake passing/admission boolean; error above
  .01 claimed admitted; missing/mismatched ACK, source/hash/recipe/artifact;
  incomplete counts/checkpoint; foreign route, baseline or encoder gate; a changed
  raw anchor; attempts to import Torch in supervisor. A finite high-error final
  fit must complete with admission=false, not become an infrastructure failure.
- [x] Observe RED for absent runner/protocol behavior.
- [x] Implement fixed loops and evidence. Coefficient fitting uses point indices;
  raw fitting uses fresh verified trainer-gauge teacher, both full complex renders
  and original VJP. Phase must be frozen before optimizer creation. No actual MRI
  trainer, scorer, DC or references are called. Every source/raw/model/phase
  invariant is checked before and after.
- [x] Reuse inherited supervised execute and ACK, validate actual numeric/counter
  fields and all bound files. Never create a passing real gate from CPU fixtures.
- [x] Run targeted runner/protocol suites, then the complete CPU suite; expect PASS.
- [x] Commit only owned files and record RED/GREEN evidence.

### Integration and experiments

- [x] Self-check all three public interfaces and declared input/source whitelist;
  run the final complete CPU suite and inspect the full diff.
- [x] Obtain fresh independent whole-change review, resolve all substantive
  findings within the user-requested maximum three rounds, retain report and
  meaningful regression evidence. No GPU before a clean review.
- [x] Bind the actual reviewed source, actual G2/raw/binary inputs and exact
  selected public CLI recipe. Passing review is not a numerical-stage pass.
- [x] Inspect GPU utilization/memory/compute PIDs and visibility through the
  inherited selector; prefer an idle compatible device; launch genuine raw control.
- [x] After successful raw control, run fixed MLP fit and record actual results;
  then the predeclared grid alternative under the same code/budget and real gate.
  Preserve failed/negative outputs and distinguish admission from software pass.
- [x] If an arm meets the declared fidelity bound, prepare its separately reviewed
  MRI continuation; otherwise choose the next approved hypothesis from evidence.
  Never promote based on teacher-fit errors or silently lower admission thresholds.

## Initial self-review and rulings

The field and standalone renderer are independent and may be implemented in
parallel; the fitting protocol consumes both exact APIs after their tests pass.
No dependencies or inherited numerical source edits are planned. Approval for
the existing full-amplitude direction and autonomous continuation is already in
the session; additional routine plan/compute approval is not requested. The .01
bound is a predeclared researcher choice; if unsuitable, this experiment may fail
admission and require a new documented hypothesis rather than rewriting results.
