# Conditional implementation notes, October22

These are read-only source-mapping findings, not experimental results. No full
field has yet passed raw-render admission. Current work remains the bounded
representation diagnostic; no MRI or relocation implementation is implied.

## MRI route after actual admission

The original JointTrainer._render calls model.get_density(channel) alongside
shared positions, exp(log scales) and rotations. FullAmplitudeField supplies the
compatible complex-vector method. The minimal later child can restore G2 using
the inherited baseline setup, strictly restore the admitted final field_state,
register the field as a model submodule, and bind get_density. Registration is
needed because original clipping visits model.parameters().

Geometry must remain frozen in this minimal continuation: the field privately
clones query positions/log scales. Changing only the renderer geometry would
leave its amplitude queries stale. A trainable-coordinate design is separate.
Enable independent phase_delta only after verifying its representation-stage
zero/frozen invariant. Use fresh disjoint amplitude and phase Adam groups, with
explicitly declared LRs/scheduler; no representation or Cartesian moments carry
over automatically.

The acquired-only dataset adapter, fixed G2 likelihood/regularizers/frequency
observer, canonical hard-DC export and evaluate.py can be reused unchanged.
The existing500-update clock is4500→5000. Existing amplitude route gates expect
residual metadata and cannot authorize this new model. A new stdlib protocol
and child must verify a real admitted descriptor, numeric errors, actual counts,
final checkpoint and hashes, then implement distinct frozen-G2 canonical and
approximate warm-field controls. The.01 raw fidelity rule is not G2 equality.
Actual MRI scoring stays separate and reports six whole-head inequalities,
all59 planes and the enhancement measurements against the fixed ten opponents.

Relevant sources: this worktree full_amplitude.py:153/164, runner.py:43/85/209,
runtime.py:38/60, evaluate.py:103/239/346; original paired_gs
trainers/joint_trainer.py:478/504. Exact new-route source closure and MRI-input
snapshots require a fresh review before launching any MRI stage.

## Fixed-budget relocation if another hypothesis is needed

Original adaptive control is not a fixed-count mechanism. In paired_gs
trainers/joint_trainer.py:671, N>=max_num_points returns before pruning or split;
setting the cap to the current3.6M therefore disables it. Existing split removes
k parents and appends2k children, net+k. Separate prune/split correspondence
metadata describes only one transaction, so chaining them before one optimizer
rebuild loses row provenance.

A separate fixed-shape transaction could replace disjoint donor/victim pairs
with two donor children in place, preserving all seven parameter objects, point
count, untouched rows, group options, scalar Adam steps and scheduler clock.
Reset only changed rows of exp_avg/exp_avg_sq across all seven tensors. This
does not reset a per-row age: existing Adam has scalar per-parameter steps.

Existing long-axis child corrections preserve each channel's phase and analytic
coefficient*prod(scale) integral, not exact sampled/clamped/truncated rendering.
The acquired-data evidence score sums both contrasts' adjoint residual energy
and samples at existing centers; it cannot directly propose an absent center.
Victim selection cannot call coefficient magnitude a measured rendering
contribution. This fallback must remain a separate explicit-coefficient route;
relocation would invalidate the current field's fixed query buffers.

Relevant original sources: gaussian/gaussian_model.py:55/164/282,
gaussian/joint_gaussian_model.py:109/191/198, trainers/trainer.py:649 and
trainers/joint_trainer.py:388/647/671/740/1198. A later design needs a fixed event
schedule, deterministic disjoint ranking, state policy and CPU checks for
unchanged rows/moments/identities, two-channel analytic integral/phase, invalid
transaction atomicity, coordinate permutation and zero-residual/no-op behavior.
