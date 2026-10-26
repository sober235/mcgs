# Fixed-budget explicit Gaussian relocation

## Intent and authority

Continue the user's authorized Karpathy autoresearch campaign to improve MC-GS
plain/enhanced AF16 quality. The two full-function warm starts completed but
failed their predeclared fidelity bound (MLP32.47%/35.93%, grid27.35%/29.74%).
They are preserved in sibling mcgs_ce_karpathy_20261022_fullamp and add no MRI
training experiment. Current count remains28 formal MRI training experiments,
65 historical rescorings and one separate discarded proximal correction.

Fixed-point-budget relocation was already selected as a candidate in the user's
research design. This document makes its first bounded experiment concrete.
Schedule, selection proxy, relocation amount and optimizer policy below are
researcher choices under existing autonomous authorization. Do not pause for
routine design/GPU permission. A negative result is retained, not hidden.

## Frozen benchmark and source

Keep existing data, AF16,42 coils, mask, CSM, original Gaussian renderer/binary,
losses/normalization/export and ten complete frozen comparators. Retain shared3D
geometry and two independent complex coefficient channels. No external prior,
new dependencies, references/ROI/oracle parameters in training, kernel rebuild,
point-count growth or scorer changes. Original shared reference-derived scalar
normalization remains disclosed; this is not a newly strict GT-free benchmark.

Only PSNR/SSIM/NMSE decide quality. Whole-head success requires all six raw
inequalities against all ten opponents. Report all59 planes and six enhancement
ROI/background measurements. Keeper lowreg_f075_solve_v1 has minimum-six margin
-0.11118128532897047; advancement requires an improvement>.0001, not merely
beating G2. Provisional overall success still requires repeat and matched
GS-single evidence. One-case adaptive development is not independent testing.

Start from exact G2 iteration4500 checkpoint SHA256
282e0303c46a6c18775d6224c032611072bd465b4ef975ba7a7e2ea636230fd0,
original seven tensor states,3.6M points. Reuse original strict prebuilt binary
SHA3973daeef5f55ebfed931c38538a37d83921a9287b96b69a431da238057194b5.
Preserve all inherited sources/completed outputs/metadata; new worktree starts
at d33666c with a clean diff. The same183-test inherited source baseline already
passed before the immediately preceding isolated work; no dependency setup.
There is no CodeGraph index; do not create one.

## Matched experiment

Run a pure explicit Cartesian/geometry continuation control and one relocation
candidate, each500 actual MRI updates from the same G2 model/Adam/scheduler/RNG
state. Do not attach residual or full-amplitude networks. All seven original
tensors train under the original objective. The sole difference is one
fixed-shape relocation event in the candidate after its first optimizer and
scheduler update, before the second update. The control performs no event or
evidence-score rendering. Record this compute difference.

Requested k=18000 donor/victim pairs (.5% of3.6M); replace at most36000 rows
(1%) while keeping N exactly3600000. This is a researcher-chosen first scale,
not a tuned optimum. No additional event, sweep or extra MRI update is allowed
inside the selected trial. Rejected configurations use fresh versions.

## Faithful optimizer continuation

Reuse acquired-only setup from runner.py baseline branch and frozen G2 config,
but explicitly unfreeze the seven original parameters. Construct original named
groups in canonical order with base LRs
(.0006,.004,.0006,.015,.015,.015,.015), Adam betas(.9,.999), epsilon1e-8,
zero decay, non-AMSGrad, and ExponentialLR gamma.9995. Call the unchanged original
resume_protocol.restore_checkpoint(..., expected_iteration=4500, gamma=.9995).
Never apply geometryfactor2 again: G2 already includes it in its optimizer LR
policy; it is never a render-scale multiplier.

Initial model, all moment tensors, seven scalar Adam steps4500, actual saved
current LRs, scheduler epoch4500/count4501 and configuration must match G2.
After500 updates all seven scalar steps and scheduler epoch are5000/count5001.
Replay the original wavelet generator to offset4500; same initial seed42 and
unchanged training iterator/curriculum. Reuse runtime.py's4500-aware frequency
observer and original flat t1 weights. Do not call the older4001-only observer,
the original4000-based entrypoint or geometry LR intervention.

## Selection from acquired data only

At the candidate event use original JointTrainer._evidence_energy_score after
the first normal update. It sums both scans' acquired-mask multicoil adjoint
residual energy and trilinearly samples at existing Gaussian centers, using the
original coordinate permutation/border/align_corners behavior. It is the
unweighted acquired residual score, not a new frequency-weighted oracle.
Bind the original executing Python source. Count its extra renders/physics
calls separately from ordinary paired training/export. No held-out claim: G2
has already used all acquired samples.

Victim proxy is max(abs(z_plain),abs(z_contrast))*prod(exp(log_scales)), computed
in float64. It approximates analytic coefficient mass, not actual rendering
contribution; phase cancellation/truncation are not measured by this proxy.
Select lowest k proxies, stable ties by original row index. Select highest
strictly-positive finite evidence among other rows, stable index ties. If fewer
eligible donors exist, use k_eff=min(k,eligible count), truncating the victim
list to k_eff. Requested victims excluded during donor ranking remain excluded
even if k_eff is smaller. k_eff=0 is an exact no-op event, recorded honestly.
Reject nonfinite/negative evidence and nonfinite proxy calculations.

## Fixed-shape transaction

Pure helper APIs:
`select_relocations(model, scores, requested_k)->(donors,victims,selection_record)`;
`relocate_rows(model, optimizer, scheduler, donors, victims, rotation_matrix_fn)
->transaction_record`. Use canonical seven parameter names. The public child
passes the exact original quaternion-to-rotation function; tests may inject a
small CPU function. No public CLI permits alternative models/physics providers.

Validate shapes, device/dtypes, unique/disjoint/in-range indices, finite donor
values, all relevant Adam state shapes/options and derived children before any
mutation. Snapshot donors before writes. Both rows of each pair become the two
donor children. On the donor's longest local scale axis, shrink by.5; other
axes shrink by.85. Local shift on the long axis is child_scale[axis] (original
offset factor1), rotated by the original normalized-wxyz function. Use symmetric
parent_position +/- shift. Clamp only the new children to[-1,1]. Retain the
donor quaternion and store child log scales, not linear scales.

For each contrast independently multiply the donor complex coefficient by
.5/(.5*.85*.85) for each child, preserving phase and total analytic coefficient
times scale product. This is not exact sampled/truncated/clamped image parity;
the event's effect on reconstruction must be measured in the MRI trial.

Write affected rows in place under no_grad. Preserve all seven Parameter
objects, optimizer group membership/options/LRs, moment tensor objects, scalar
steps and scheduler state. Zero exp_avg/exp_avg_sq only for the changed2*k_eff
rows across all seven tensors; zero any existing gradients on changed rows.
Untouched parameter/gradient/moment rows remain bitwise unchanged across this
transaction. Scalar Adam age is retained; there is no per-row age reset. Do not
call _update_params, densify/prune/clone, _rebuild_optimizer or recreate scheduler.

Record actual indices/hash, k_eff/counts, before/after parameter and state object
identities, untouched-row tensor hashes, changed-row-zero checks, group options,
seven scalar steps and full scheduler state. Parent checks actual numeric/hash
fields and identities, not a passed boolean alone. An invalid request must fail
before writes. A no-op must preserve everything exactly.

## Route, controls and evidence

New files only: relocation.py, relocation_runner.py, run_relocation.py, focused
tests and route documentation/program binding. Inherited lifecycle, data,
frequency helpers, renderer and evaluate.py stay byte-identical. Parent stdlib
only, unchanged run_gpu.execute(protocol=...), common claim lock, idle/visibility
selection, observed-context ACK from proxdc_runner.wait_context_ack and owned
child cleanup. Maximum owned-child time3600 seconds, no installs/fallbacks.

Public CLI: `run_relocation.py --stage zero|smoke|trial --mode control|relocate
--steps 0|2|500 --exp NAME --review-gate PATH --recipe PATH [--zero-gate PATH]
[--smoke-gate PATH] --preferred-gpu N`. Exact recipe binds source/input snapshots,
mode/budget, k18000, eventstep1, transform, selection, source4500 and restoration
policy. New route/schema only; old completed gates do not authorize it.

Zero is control mode only, strict full G2 model/optimizer/scheduler restore,
zero updates/events, complete state before/after proof and original canonical
export parity<=1e-5 (record elementwise equality separately). Both smoke modes
execute exactly2 real paired MRI updates from G2; relocation smoke must execute
the same step1 event policy and verify the transaction/state invariants. Each
trial requires genuine zero and matched-mode smoke gates. Controls are never
counted as new formal MRI experiments.

Count observed optimizer/scheduler steps, paired frequency/loss calls, actual
Gaussian forwards/VJPs and event calls separately. Bind complete actual source
closure including original restore/evidence/quaternion functions, all executing
legacy helpers, CUDA binary, fixed MRI acquisitions/mask/CSM/G2/normalization,
contract/opponents/ROI freeze. Reference/ROI arrays enter only separate scoring.
Check reviewed sources/inputs/controls before and after; unique output folders,
actual final iteration5000 checkpoint/candidates/trace/event evidence, immutable
completed descriptors and separate CPU scoring directories.

Run control500 then relocate500 under matched budgets. Use unchanged evaluate.py
for actual whole-head scores,59-plane and ROI reports, and append every attempt
and keep/discard/crash outcome to results.tsv/autoresearch.jsonl. Preserve all
negative runs and scientific revisions. Two completed500 arms add two formal
training experiments; zero/smokes/helper tests add none. Source review/controls
never imply MRI quality or a winner.

## Verification and recovery

Meaningful CPU tests: exact two-channel split formula/orientation/log units;
stable disjoint ranking, insufficient/zero evidence; fixed count and untouched
rows; parameter/moment identities, clock/group/scheduler preservation across a
real populated seven-group Adam step; only changed moments/gradients zero;
invalid/overlapping/nonfinite requests fail before mutation; next ordinary Adam
update works. Route integration uses real tiny tensors/optimizer/restore/event
producer-to-validator evidence and negative metadata cases. Synthetic CPU
fixtures are not real GPU controls or recipe gates.

Fresh independent whole-change review before any GPU, at most three automatic
fix/re-review rounds; no unresolved blocker may launch. Observe actual idle GPU
resources and retain other workloads. Restore/schema failures are setup or
numerical failures, not scientific rejection. Preserve evidence and use a new
version after a failed stage; never overwrite completed metadata or relax checks.
