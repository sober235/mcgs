# Case-internal magnitude prior route

`run_caseprior.py` is the stdlib parent for route
`g2_case_internal_magnitude_prior_v1`. It uses the unchanged
`run_gpu.execute(protocol=...)` supervisor: common GPU claim held through child
exit/reaping, actual context ACK, fixed GSTMI interpreter/renderer, cancellation,
maximum 3600 seconds, fresh source/input snapshots and before/after proof.
These are command templates, not records of executed MRI stages.

The only public budgets are zero0/control, smoke2/control or learned, and
trial500/control or learned. Production retains 3.6M Gaussian points, 42 coils,
the historical AF16 mask/CSM/scales and original seven-group G2 Adam/scheduler
at4500. Geometryfactor2 is never reapplied; no relocation or optimizer rebuilding
occurs. Only learned mode adds original NonLocalRED coefficients0.1 for both
scans, sigma_rel0.015, start4501, refresh100 and `case_internal_cnn_v1`.

An independent clean review gate and selected recipe must bind the current
`run_caseprior.code_hashes()` and `snapshot_invariants()`. The recipe has
`schema_version:1`, `status:selected`, `route`, the exact `parameters` (route,
stage, mode, steps), the complete `POLICY`, `code_sha256` and `invariants`.
The source closure includes both new adapters, Task1 and all original reused
trainer/runtime/export/supervisor/validator sources and the frozen CUDA binary.
Old-route reviews or controls cannot serve as new-route gates.

```bash
python -B run_caseprior.py --stage zero --mode control --steps 0 \
  --exp caseprior_zero_v1 --review-gate REVIEW.json --recipe ZERO_RECIPE.json

python -B run_caseprior.py --stage smoke --mode control --steps 2 \
  --exp caseprior_control_smoke_v1 --review-gate REVIEW.json \
  --recipe CONTROL_SMOKE_RECIPE.json --zero-gate ZERO_COMPLETED.json

python -B run_caseprior.py --stage smoke --mode learned --steps 2 \
  --exp caseprior_learned_smoke_v1 --review-gate REVIEW.json \
  --recipe LEARNED_SMOKE_RECIPE.json --zero-gate ZERO_COMPLETED.json

python -B run_caseprior.py --stage trial --mode control --steps 500 \
  --exp caseprior_control500_v1 --review-gate REVIEW.json --recipe CONTROL500_RECIPE.json \
  --zero-gate ZERO_COMPLETED.json --control-smoke-gate CONTROL_SMOKE_COMPLETED.json \
  --learned-smoke-gate LEARNED_SMOKE_COMPLETED.json

python -B run_caseprior.py --stage trial --mode learned --steps 500 \
  --exp caseprior_learned500_v1 --review-gate REVIEW.json --recipe LEARNED500_RECIPE.json \
  --zero-gate ZERO_COMPLETED.json --control-smoke-gate CONTROL_SMOKE_COMPLETED.json \
  --learned-smoke-gate LEARNED_SMOKE_COMPLETED.json --control-gate CONTROL500_COMPLETED.json
```

`--preferred-gpu` defaults to0 and `--timeout` to3600 (cannot exceed3600).
No public network, noise, loss, update, point-count, backend or test-provider
override exists. Both new smokes and actual new zero are required before either
trial. Choose fresh experiment and scoring names for every attempt.

To reuse the archived native500, replace `--control-gate` with
`--native-control-receipt NATIVE_RECEIPT.json` for the learned trial. Exactly one
is required. A receipt has `schema:native_control_receipt_v1`,
`route:g2_case_internal_magnitude_prior_v1`, `status:verified`, the new
`code_sha256`/`invariants`, the actual old `native_code_sha256`/`native_invariants`,
and path/SHA256 descriptors `native_completion`, `native_smoke`, `ordinary_smoke`,
`objective`. These point to actual old control500 completion, old ordinary
smoke completion, new ordinary smoke completion, and the already scored
native objective produced by unchanged `evaluate.py`.

The validator uses unmodified `run_relocation.code_hashes()` and old completion
validation, preserving the old route/policy. Common frozen inputs come from
the fresh new snapshot with only route/policy restored to their old values;
it takes no second MRI input snapshot for this comparison. Every old artifact,
ACK, checkpoint/clock, control binding and postproof must verify. The new
ordinary smoke must have the same restored model/Adam bytes, options/scheduler,
actual Gaussian/FFT work and final clocks, and canonical complex relativeL2
<=1e-5 against the old ordinary smoke. Scored candidate path/hash/shape/dtype,
six metrics, objective identity,59-plane count and evaluator hash must match the
actual old control. Transitive evidence is retained in new launch bindings.
If any criterion fails, preserve the refusal and run a versioned new control500.
No existing completion is relabeled as a new-route run.

The child uses original `runner.build_config/setup_trainer(mode=baseline)`,
strict full-state restoration, and `relocation_runner.run_continuation(mode=control)`
for every Gaussian update and canonical save. Small fixed private `FunctionType`
bindings change only new route/policy/parameter dispatch, prior configuration,
continuation and validation hooks. They never mutate original modules/bytecode
or copy the training loop. The only scoped factory replacement recognizes the
new denoiser name and delegates other names, restoring the binding on exceptions.
Control constructs no CNN. Zero skips original `train()` and truthfully has
`nonlocal_summary:null`; ordinary control has the actual disabled summary.

Each learned refresh invokes fresh Task1 fitting under original `no_grad`:
five3x3x3 convolutions1->32->32->32->32->1, four ReLUs, zero last convolution,
float32 residual noise, fresh Adam0.001/128 updates, batch4/32-cubed patches.
Normalize crop/noise by max(P99,1e-6), infer32-cubed interiors with halo5, clamp0,
seed call j with42+1009*j. Targets remain detached and independent per scan.
Production RED history names are `pre`/`post`; Gaussian render roles remain
`plain`/`contrast`. Frequency loss-call scan labels remain `pre`/`post`.

`nonlocal_summary` is the actual original trainer summary and `denoiser_calls`
contains unmodified ordered Task1 records. Each record binds its unique
`denoiser/call_0000/` target, manifest and checkpoint. Parent validation checks
exact settings/updates, positive sigma/nonzero target energy, finite nonzero
training gradients, changed parameter hashes, refresh/history linkage, target
raw bytes and file descriptors. It follows original active-prior semantics:
all losses finite/nonnegative, positive per-scan maxima, actual nonzero weighted
work and at least one changed target per scan. An occasional zero step is valid.

Separate `denoiser_rng` records contain schema/call_index, CPU before/after SHA256
and preservation flag, and one such observation per visible CUDA device.
Separate `denoiser_checkpoint_checks` bind call_index/checkpoint/final digest,
fixed tensor shapes, `metadata_equal` and `finite_float32_cpu`. The child performs
actual `torch.load(weights_only=True,map_location=cpu)` readback of each trained
checkpoint and recomputes Task1's final parameter digest without another model,
fit or RNG draw. Both observations are checked by the parent. This trusts the
reviewed child semantic readback and exact file descriptors, as for the Gaussian
checkpoint; it is not cryptographic attestation against fully rewritten artifacts.
CPU fixtures honestly report an empty CUDA RNG list and cannot pass public
N=3.6M/CUDA/real frozen-input validation.

Zero/smoke/trial Gaussian forward/VJP counts are2/0,6/4,1002/1000. No events occur;
seven Adam clocks finish4500+steps and scheduler count4501+steps. Learned smoke
has two fresh fits/256 CNN updates; learned500 has ten/1280, refreshing each scan
at4501,4601,4701,4801,4901. CNN work is separate from500 Gaussian updates.
Actual training/stage time and peak allocated/reserved GPU memory are recorded.
Outputs retain config, scheduler, trace, original final checkpoint, frequency
evidence, raw/canonical/candidate arrays, per-call artifacts, matching
metrics/provenance, supervisor completion and before/after invariant proof.

After genuine completion use the unchanged separate CPU evaluator in a unique
directory, for example:

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B evaluate.py \
  --run-dir outputs/caseprior_learned500_v1 --out-dir scores/caseprior_learned500_v1 \
  --name caseprior_learned500_v1
```

Report six whole-head PSNR/SSIM/NMSE inequalities over ten opponents,59 planes and
six ROI/background enhancement measurements. Only improvement>keeper+0.0001
advances; success still needs all six strict inequalities and later independent
repeat, matched GS-single and visual checks. No external data/weights/reference,
ROI or opponent enters fitting. Existing reference-derived normalization and
historical vdPoisMex AF16 exception remain disclosed. This is same-case adaptive
development, not independent validation or a reproduction of ReSiDe's algorithm.
Fitting/loss decrease/smoke success establishes no MRI quality or equal-time claim.

For recovery preserve failed/partial directories, descriptors and logs; choose a
new version from unchanged G2. Never overwrite a denoiser call or archived result,
delete a discarded candidate, relax checks, or replace only a mask illustration.
Use the full CPU validation command in `docs/relocation_commands.md`; focused
tests are `tests/test_caseprior_route.py tests/test_caseprior_protocol.py`.
