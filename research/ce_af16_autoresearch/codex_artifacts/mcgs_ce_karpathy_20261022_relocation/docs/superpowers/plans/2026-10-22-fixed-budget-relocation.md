# Fixed-Budget Relocation Implementation Plan

> Use superpowers:subagent-driven-development; preserve the plan workspace and
> scientific evidence. Existing autonomous/GPU authorization applies.

**Goal:** Test one fixed-count redistribution of explicit G2 Gaussian capacity
against a matched original-G2500-update continuation, under the frozen benchmark.
**Spec:** docs/superpowers/specs/2026-10-22-fixed-budget-relocation-design.md.
**Architecture:** Pure in-place relocation helper plus strict seven-group resume
utility feed a new child/protocol. Inherited acquired-only data, original
training/renderer/export/evaluator and supervised GPU lifecycle remain unchanged.
**Environment:** Existing GSTMI Torch2.7.1+cu118; no dependencies or compilation.

## Global constraints and actual starting point

New sibling worktree from d33666c, branch autoresearch/ce-af16-relocation-20261022,
starting diff empty. The same inherited183-test source baseline already passed;
no original/inherited numerical changes are permitted. CodeGraph absent. Fixed
data/AF16/42coils/mask/CSM/normalization/ten opponents; no training references/ROI
or external priors. Original exact CUDA binary only. Three automatic review/fix
rounds maximum; no unresolved blocker before GPU. Preserve all attempts.

Both preceding full-function representation arms are completed negative
diagnostics, not admitted MRI candidates. Formal MRI training count28; this
matched family adds two only when its500-update arms actually complete.

## Task 1: Pure fixed-count selection and transaction

**Files owned:** relocation.py, tests/test_relocation.py.
**Consumes:** original model seven named parameters, real Adam/scheduler, scores
and original quaternion-to-rotation function. No datasets, trainer, file loading
or renderer invocation.
**Produces:** `select_relocations(model,scores,requested_k)` returns donors,
victims and JSON-safe selection record. `relocate_rows(model,optimizer,scheduler,
donors,victims,rotation_matrix_fn)` performs the exact spec transaction and returns
JSON-safe actual before/after evidence. Interfaces have no public backend seam.

- [ ] Read the spec's selection and transaction clauses; write meaningful CPU
  tests using tagged seven-tensor rows and populated real seven-group Adam state.
- [ ] Observe RED for absent helper/functions or original failure behavior.
- [ ] Implement stable index ties, float64 mass proxy, disjoint selection,
  fewer-positive/zero evidence, explicit validation and precomputed children.
- [ ] Implement in-place two-channel long-axis split with original rotation
  function, new-row-only clamping, correct log scales and analytic integral/phase.
- [ ] Keep point count/Parameter/moment identities, untouched rows/gradients/
  moments, all options/LRs/scalar steps/scheduler; reset changed-row moments and
  gradients only. Reject invalid/nonfinite requests before mutation.
- [ ] Verify exact no-op, known quaternion orientation, both complex channels,
  untouched hashes/state IDs, real next Adam step and failure atomicity.
- [ ] Run focused tests, self-review and report frozen hashes/actual RED/GREEN.
  Do not stage/commit or use GPU; root serializes commits.

## Task 2: Strict full-G2 Adam continuation utility

**Files owned:** relocation_runtime.py, tests/test_relocation_runtime.py.
**Independent of Task1.** Reuse unchanged original resume_protocol, not the
4000-based entrypoint or geometryfactor intervention.
**Interfaces:** `restore_training_state(trainer,checkpoint)->JSON-safe record`
unfreezes the original seven parameters, constructs canonical doubled base-LR
groups and exact Adam/ExponentialLR, and restores4500 without an update.
`snapshot_training_state(trainer)->JSON-safe dict` records actual seven tensor/
gradient/moment hashes/identities, optimizer membership/options/LRs/steps and
scheduler/config state for zero and boundary evidence. Imports Torch only in
this child-side utility; parent must not import it.

- [ ] Write CPU tests with an exact seven-tensor tiny model and real populated
  Adam checkpoint: canonical names/order, step4500/LRs/clock restore, preserved
  Parameter identities and strict wrong-source/options/schema rejection.
- [ ] Observe RED before implementing missing functions/behavior.
- [ ] Implement lazy original-source load/binding and strict G2 recipe/schema,
  fixed optimizer options, base LRs and original restore call expected4500.
- [ ] Verify zero restore does not take a dummy step, all seven trainable,
  no network attachment, unchanged before/after snapshots without work, and two
  real optimizer/scheduler steps reach4502/epoch4502/count4503.
- [ ] Test missing/incorrect group/moment/step/LR/recipe cases; inspect all
  current public interfaces. Use no actual MRI input or GPU.
- [ ] Run focused tests/self-review; report actual commands/hashes and source
  dependencies. Do not stage/commit. Preserve other worker edits.

## Task 3: Original MRI child, stdlib protocol and exact evidence

**Files owned:** relocation_runner.py, run_relocation.py,
tests/test_relocation_runner.py, tests/test_relocation_protocol.py,
docs/relocation_commands.md, program.md new-route binding only.
**Consumes:** approved Task1/2 APIs and actual record schemas, inherited
runner.setup_trainer baseline branch, runtime's4500 observer, original training,
evidence/rotation/export helpers, immutable evaluator and supervised lifecycle.
Read their actual return signatures; setup_trainer returns
(trainer,field,frequency,before). Invoke with baseline launch to avoid networks,
then restore_training_state on the trainer before any training/export control.

- [ ] Write focused runner/protocol tests, including a real tiny CPU
  restore→training→event→artifact producer-to-parent-validator integration.
  Cover clocks/counts/frequency/gradients/event state and finite actual exports;
  CPU fixtures may not create real reviewed/numerical gates.
- [ ] Observe RED for missing production behavior.
- [ ] Implement new route zero/control, matched two-step smokes and500-update
  trials with exact recipe, seed/source/clock, no original source edits. Restore
  source first, install event hook only for relocate and execute once after its
  first ordinary Adam/scheduler update. Disable inherited adaptive control.
- [ ] Use original acquired-only evidence and original quaternion function;
  collect actual event/transformation/state evidence and counters. Separate
  training/event/export/initialization render roles and observe actual VJPs.
- [ ] Keep all seven original parameters in clipping/Adam; verify no residual
  networks, no point growth, finite gradients/parameters, restored4500 clocks,
  observed steps and final4500+steps clocks/checkpoint. No old zero source-
  unchanged assertion is applied to actual training.
- [ ] Reuse unchanged generic run_gpu.execute protocol, shared lock, device
  visibility/selection, owned-child cleanup and opt-in observed ACK. Parent
  stdlib only; child exact GSTMI/backend. Bound entire executing source closure
  and real frozen MRI inputs/G2/normalization/contract/opponent/ROI definitions.
- [ ] Validate exact selected recipes, actual restore/source/clock/evidence,
  real zero/matched-smoke gates, actual ACK/artifact bytes, numeric counters,
  event indices/counts/state hashes/identities and before/after input proofs.
  Test false flags, foreign/stale gates, incomplete states/counters, changed
  inputs/recipes/checkpoints, accidental second event/point growth and Torch
  parent import. Zero preserves restored model/moments/clock and reproduces G2.
- [ ] Export canonical arrays through original hard-DC path/gauges; record
  raw/control exports separately from scoring. Final checkpoint/trace/events/
  config/data/source/runtime/precision evidence must be bound to completion.
- [ ] Run focused tests, then complete CPU suite once frozen with actual stdout
  and source snapshots retained. Self-review full diff and original failure
  cases. No GPU/real gates/selected recipe or commit by implementer.

## Integration, experiments and decisions

- [ ] Root verifies reported source hashes/stdout/diff and serializes task
  commits. Obtain independent spec/quality reviews for Tasks1/2; final Task3
  review covers complete branch on a fresh most-capable reviewer. Resolve all
  substantive findings within three rounds, preserve all reports.
- [ ] Bind clean actual review, complete fresh source/input snapshot and exact
  selected CLI recipes. Inspect current utilization/memory/compute PIDs and
  visibility, then launch new-route zero and both matching smokes on idle GPU.
- [ ] After genuine controls, run matched control500 then relocation500.
  Record actual compute/memory/state/event counts and retained failures.
- [ ] Score each actual candidate with unchanged evaluate.py into unique CPU
  score directories. Check all six whole-head metrics,59planes and ROI reports;
  append actual values/decisions to TSV/JSONL. Do not count smokes as formal runs.
- [ ] Advance only a measured keeper improvement>.0001. Provisional overall win
  triggers repeat and matched GS-single, not an immediate superiority claim.
  Otherwise retain negative evidence and choose the next bounded hypothesis.

## Preflight rulings

Independent Task1 and Task2 own disjoint files and can work in parallel; Task3
waits for both approved interfaces/record schemas. Root serializes commits and
runs full-suite verification only after both freeze. This follows proactive
delegation without mixed-index commits or redundant tests during temporary RED.
Existing user authorization covers this already researched candidate direction
and routine GPU use. Researcher choices remain explicitly labeled. Preserve
worktrees and plan evidence; discard means scientific decision, not deletion.
