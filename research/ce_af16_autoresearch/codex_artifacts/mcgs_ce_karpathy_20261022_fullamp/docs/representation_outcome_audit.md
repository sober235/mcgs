0 actionable findings. Actual-output integrity audit: PASS.

The genuine raw control and both representation fits have consistent completion
evidence. Both fits are correctly rejected under the unchanged .01 admission
rule. No MRI promotion is supported or claimed.

## Scope

Audited only `outputs/g2_fullamp_raw_control_v1`,
`outputs/g2_fullamp_mlp_rep_v1`, `outputs/g2_fullamp_grid_rep_v1`, the new outcome
report, representation state, results/ledger records and their source/control
bindings. Scientific HEAD remains
`131bb0d078b5028eecffba868b0088332773cec7`. This is an incremental actual-output
audit following the clean source review, not another whole-branch review.

No GPU access, new fitting, complete test-suite rerun, output modification,
production edit, metadata edit, index edit or subagent was used. Only this report
was written.

## Independent validation performed

In a fresh stdlib-only Python process, called `run_representation.code_hashes()`
and `snapshot_invariants()`, then independently called `validate_observed_gate`
on the actual control with stage `render_control` and both actual fits with stage
`fit`. All three passed against **31 current code files and 29 verified input
files**. Torch and NumPy remained absent from `sys.modules` throughout this
process. Reconstructed each exact stage request and also called
`validate_stage_gates`, including the real control gate for each fit; all passed.

This revalidated the selected recipe descriptors, reviewed source/input bindings,
actual artifact bytes, context ACK, parity/error values, complete update traces,
per-step finite-gradient and VJP counts, source/teacher/field invariant hashes,
Adam reset evidence, final progress, final checkpoint descriptor and parent
before/after proofs. Additionally checked that each completion return code is
zero and that `worker_pid`, `child_started.json` and `childcontext_ready.json`
identify the same child. All recorded supervisor durations are below 1800 seconds.

Independently compared every completed ledger summary field with actual metrics,
verified all ledger launch/recipe/gate/metrics descriptors, and compared results
rows with actual final errors, peak memory and keep/discard decisions. All outcome
table endpoints, numerical/supervisor durations, peak-memory values, gate hashes
and final-checkpoint hashes match the corresponding artifacts. Representation
state correctly records one control, two completed fits, no active run, no admitted
architecture and zero added formal MRI experiments. The bound public review
report is byte-identical to the original independent final review report.

In a separate process using existing GSTMI Python with `CUDA_VISIBLE_DEVICES=''`,
one OMP/MKL thread and `-B`, loaded both actual `final_field.pth` files with
`torch.load(..., map_location='cpu', weights_only=True)`. Both passed these checks:

- Final fixed-budget checkpoint kind, exact fixed configuration, field metadata,
  source hashes and completed 2000/128 update counts agree with actual metrics.
- Phase deltas have shape `[3600000, 2]`, contain exactly zero nonzero entries,
  and metadata declares phase gradients disabled.
- Every stored tensor is finite; all immutable field hashes match the before/after
  evidence. Position and log-scale tensor bytes match the original source-state
  hashes. Stored complex phases have unit modulus within float32 tolerance.
- No density-named per-point modulus state is present. Actual network/grid tensor
  parameter totals are 7810 for MLP and 604290 for grid.
- CUDA remained uninitialized in the checkpoint audit process.

All three audit commands completed with exit status 0.

## Verified outcomes

| Stage | Coefficient/raw updates | Forwards/VJPs | Final plain error | Final enhanced error | Admission |
|---|---:|---:|---:|---:|---|
| Raw control | 0/0 | 2/0 | 0 anchor error | 0 anchor error | Not applicable |
| MLP | 2000/128 | 264/256 | 0.3247297186805714 | 0.3593065686077883 | False |
| Grid | 2000/128 | 264/256 | 0.273462019725601 | 0.2974255657616767 | False |

Raw reporting-gauge parity records show elementwise equality and zero relative
complex L2 for both channels. Both final fit errors exceed .01 by large margins.
The report's claim that the grid fits better at this fixed budget is supported by
both final channel errors; it correctly avoids a general capacity-limit claim.

All three retained launch snapshots select GPU 2 with zero utilization, no active
compute PIDs, 2 MiB used and 81918 MiB free. Recorded context ACK identities agree.
Precision records consistently show float32, autocast disabled, CUDA matmul TF32
false and cuDNN TF32 true. Launch logs identify the original `tile_cuda` backend.

| Actual completed gate | SHA256 |
|---|---|
| g2_fullamp_raw_control_v1 | 303b96d63383b54534dc79861aa7c41f7e834f02ece13199264aa65763fff48a |
| g2_fullamp_mlp_rep_v1 | 17471d4f8d10612b285beccd084b3e1e22ed8f29b375d81eaa8612b87be8e1cd |
| g2_fullamp_grid_rep_v1 | 49217c504a17ce64ea2fd0882f67b63e106c6878fe8ec163dc91f3c8deb42006 |

MLP final checkpoint SHA256:
`1d5798e06c65f0d9f020964976c588d2ca5f367a8a83deafb96e9e080afbd07a`.
Grid final checkpoint SHA256:
`99843b569cf734ca0d06d0b95ee4e498a54ba3e21bb81cf2b5accca71db71469`.

Audited report/state/ledger SHA256 values:

- `docs/representation_outcome.md`: `6c2f446e6ac14a4b2e061792eb38bd6e56d65947758924146879dea04fd0f231`.
- `representation_state.json`: `6c8e6e4190e3fea344406f9121671c2f4784318313a002870e70529f6917597b`.
- `results.tsv`: `d6969e40fceef22361282f747c079a30670a2508adf523db0c8e8b97b84548e8`.
- `autoresearch.jsonl`: `f652344ebdd97ecb8989ee7f3226fac086dff463f95d55f7d06b69d76e3bfbe2`.

## Limits and disposition

This audit verifies the retained genuine-run evidence and actual final checkpoint
contents; it does not independently rerender the fields or rerun fitting. It does
not reproduce historical MRI metrics or validate the future relocation design.
Historical campaign totals are context; the new route's zero MRI increment and
absence of admission are consistent across the audited artifacts.

No new mismatch or regression was found, and no finding was declined. Retain all
three directories and both negative checkpoints. Neither fit may serve as an
admitted full-field MRI initialization. The separate prospective fallback needs
its own implementation, review and numerical controls.
