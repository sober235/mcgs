# Case-internal learned magnitude prior for MC-GS

## Intent and evidence

Improve the fixed plain/enhanced AF16 reconstruction while retaining the shared
3D Gaussian geometry, independent complex contrast coefficients and original
physical renderer. The target remains all six whole-head PSNR/SSIM/NMSE
inequalities over ten complete opponents. Current keeper is
`lowreg_f075_solve_v1`, minimum-six margin -0.11118128532897047. A new observed
keeper requires improvement greater than 0.0001; an overall win still requires
all six inequalities, repeat, matched improved GS-single and image inspection.

Thirty formal MRI trials are complete. The latest matched 500-step native and
relocation arms score -0.11718558579803065 and -0.11718772593014594, with paired
plane success 0/59. Their complete evidence is in `docs/relocation_outcome.md`.
Do not sweep small relocation variants to disguise this negative result.

Independent history inspection found active BM4D and patch-low-rank trials,
all rejected, but no executed current-case learned denoiser. They were older
i1 experiments, so they are not matched G2 denoiser controls. ReSiDe-S motivates
learning a patch denoiser from the intermediate image plus synthetic noise.
Its brain study was R4; it supplies no measured R16 gain here. This proposal is
a magnitude-prior adaptation, not a reproduction of its complex PDS algorithm.
Sources: [paper](https://arxiv.org/html/2304.12890v4),
[official code](https://github.com/OSU-MR/reside/tree/dc487b0c2600de40110ee193f1e21338febe94ae).

## Scope and authorization

The user authorized continued autonomous official Karpathy autoresearch, with
existing data/mask/CSM/ten opponents and no external large-data learned prior.
The choices below are researcher choices under that authorization, not claimed
user-selected hyperparameters. No additional permission question is needed for
this bounded case-only experiment. Preserve every earlier source and result.

Reuse the existing isolated worktree; create the new research branch
`autoresearch/ce-af16-caseprior-20261022` from the archived relocation result.
Do not create another worktree when already isolated. Numerical relocation
files and original external sources remain byte-identical. New files implement
this route and its tests. The inherited 401-test baseline already passed in
this continuous session; do not rerun it without new code or risk.

No external images, model weights, package installs, renderer rebuilds, scoring
reference, ROI, opponent output, reference-fitted parameter or new mask enters
training. The existing reference-derived normalization restriction remains
disclosed. This is same-case adaptive development, not independent validation.
The fixed historical vdPoisMex mask is the already disclosed task-specific
exception. Newly generated masks must use BART Poisson with full 24x24 ACS and
a new benchmark version; this route generates no mask or sampling remapping.

## One candidate

Start each run from original G2 iteration4500 and restore all seven trainable
Parameter tensors, Adam moments/steps and ExponentialLR state using the reviewed
`relocation_runtime.py`. Never reapply the historical geometry LR factor2.
Keep 3.6M points; perform no relocation, prune/split or optimizer rebuilding.
Retain original losses, flat frequency t1, clipping, source wavelet sequence and
canonical hard-DC export. The added prior is the only numerical intervention.

Use the existing `losses.nonlocal_red.NonLocalRED` implementation unchanged.
It constructs the ZF-derived support and bounding box, selects sigma from the
current rendered magnitude, stores detached per-scan targets and applies
`mean((abs(x)-target)^2)/(mean(target^2)+1e-12)` in that box. Its support is not
a scoring ROI. Set both `lambda_nl` and per-scan coefficients to0.1, sigma_rel
to0.015, start4501 and refresh100. Actual refreshes must be4501,4601,4701,4801,
4901 for each scan. Set the denoiser name to `case_internal_cnn_v1`.

Scope a factory adapter around the existing `make_denoiser` binding, restoring
it on exit. It recognizes only this named callable and delegates other names
to the unchanged factory. Never alter the original source file or globally
replace unrelated helpers. With coefficient0 the original trainer must not
construct, train or call the network at all.

Each callable invocation trains a fresh independent magnitude CNN solely on
the supplied current crop, so plain and enhanced signals are never forced to
match. Gaussian phases remain independent and retain their original losses.
The CNN is not registered as a Gaussian-model child or added to its optimizer.
Its result is only a detached lagged prior target; the final reconstruction
must be the canonical export of the trained original Gaussian model.

## Denoiser behavior and interface

Create `case_denoiser.py` with
`CaseMagnitudeDenoiser(output_dir, *, device, seed=42, updates=128, patch_size=32,
batch_size=4, width=32, lr=1e-3, tile_size=32)`. It is callable as
`(z: numpy.ndarray, sigma: float) -> numpy.ndarray` and exposes `last_stats`.
Production uses exactly these defaults; smaller private CPU-test instances are
allowed. Validate before mutation: input is finite nonnegative real 3D data,
sigma is finite/nonnegative, dimensions fit the selected patch and dimensions,
counts, learning rate and device are valid. Preserve input bytes and never
overwrite an existing invocation artifact. Zero input or zero sigma returns
an independent exact identity result in canonical float32 with zero updates, no RNG change; active
production validation requires positive sigma and actual learning work. Production crops are float32. Other real dtypes may be converted, with original dtype/bytes recorded; finite values that overflow float32 must be refused before artifacts or RNG consumption.

Normalize by `max(quantile(z,0.99),1e-6)` and divide sigma by the same value.
Use float32, no mixed precision and no network batch normalization. The CNN
has five 3x3x3 stride1 padding1 convolutions:1->32->32->32->32->1, ReLU after
each of the first four, and predicts residual noise so output=input-residual.
Initialize the final convolution to zero, retain ordinary Torch initialization
for earlier convolutions, and use fresh Adam(lr=1e-3, betas=(0.9,0.999),eps=1e-8)
for exactly128 updates. No weight decay, schedule or warm-trained network.

Each update samples four uniform32-cubed crops with replacement from this
crop and adds fresh independent real Gaussian noise at the supplied normalized
sigma. The target is the original sampled magnitude patch. Minimize mean
squared error of denoised versus target patches. Synthetic noise strength is
a declared training choice, not an estimate of independent white image noise.

Use a local Python crop RNG and local Torch noise generator. Seed invocation
j (zero-based) with42+1009*j. Restore CPU and visible CUDA RNG state after
initialization/training; do not change global backend settings. Training must
explicitly enable gradients inside the original `@torch.no_grad()` refresh;
otherwise the real original caller would silently prevent fitting. No CNN
gradient may enter the Gaussian graph, and no retained Gaussian graph may be
backpropagated during network training.

Inference uses eval/no_grad and tiled32-cubed output interiors with a five-voxel
halo, preserving full-volume zero-padding at global boundaries. The result
must agree with full inference for a tested small nonsymmetric volume. There
are no global normalization layers that change tile results. Rescale to input
units and clamp negative inferred magnitudes to zero; record clipped count.

Each fresh invocation writes its unique checkpoint, target and manifest under
the run directory. Record seed, source crop hash, sigma, normalization, actual
128 updates, batch/patch sizes, first/final training losses, finite/nonzero
gradient observations, changed parameter hashes, initial/final model hashes,
time, output hash and checkpoint descriptor. `last_stats` binds those artifacts
to the original NonLocalRED per-scan refresh history. Zero/no-op invocations
must be distinguishable. A formal active run needs ten fresh invocations and
1280 total CNN updates; the learned smoke needs two and256. These do not count
as additional formal MRI reconstruction trials.

## Numerical route, controls and evaluation

Add `run_caseprior.py` (stdlib parent protocol) and `caseprior_runner.py` (GSTMI
child). New route `g2_case_internal_magnitude_prior_v1` supports only stages
zero0, smoke2, trial500 and modes control/learned; zero is control only. Exact
recipes freeze all policy values; no public numeric override or fallback.
Reuse the inherited common supervisor, cancellation, GPU claim, actual-context
ACK, child reaping, source/input hashing and before/after invariant proof.
Bind all executing source files including the unchanged reused continuation
and runtime helpers plus the new files; old review/control gates cannot
authorize this new route.

Reuse the original trainer through `runner.setup_trainer(...,mode=baseline)`,
strict restoration, `relocation_runner.run_continuation(...,mode=control)` and
canonical export helpers where they fit. It observes real render/VJP/physics
counts and returns actual snapshots/traces/checkpoint evidence. Do not copy
or replace the original training loop. Keep the scheduler instance state pure.
If an interface cannot support this route, resolve that specific boundary in
the plan instead of silently changing a reviewed source or simplifying gates.

First run a genuine zero reproduction, then ordinary and learned two-step
smokes. Zero must preserve full restored model/Adam/scheduler and reproduce G2
arrays at relativeL2<=1e-5. The ordinary smoke must match the existing ordinary
relocation smoke start/clock/work and canonical output within that tolerance.
Learned smoke must show real CNN fitting under the original no_grad caller,
positive detached targets/prior loss, original seven finite Gaussian gradients,
two Adam/scheduler updates and original save/export path. All real gates must
be dereferenced, source/input-bound and publicly validated before trial.

The existing completed native500 control may be reused only through a strict
verified receipt: dereference its actual public completion/proofs/artifacts,
verify the unchanged old source closure and common frozen inputs, require
control mode/500 updates, and bind its already scored array hashes/objective.
Also require the new ordinary smoke parity described above. This is reuse of
an observed equivalent control, not a fabricated new-route completion. If
equivalence cannot be established, run a newly versioned native500 control.
No repeated MRI data rehash merely to build that receipt: derive its common
input binding from the freshly taken new-stage snapshot and preserve the old
route/policy in its comparison. Never pass the new route hash map as the old
route's different source closure.

For every control/learned run, Gaussian counts remain2*steps+2 forwards and
2*steps VJPs, no event, with all seven final clocks4500+steps and scheduler
count4501+steps. CNN work is separate and fully counted. A learned500 trial has
500 actual Gaussian updates, ten target refreshes and1280 CNN updates; a3600s
supervisor timeout and observed wall/memory costs apply. No equal-time claim.

Run the unchanged separate CPU evaluator after genuine completion, using a
unique output directory. Report all six whole-head metrics/inequalities, all
59 planes and six enhancement regions/background. Advance only on the fixed
keeper rule; preserve evidence on discard or failure. A smoke, loss decrease,
network fitting result or target change cannot establish an MRI win. If this
bounded candidate fails, record the failure and choose a new hypothesis rather
than silently changing its noise/weight/refresh budget.

## Verification and review

Test meaningful behavior: fitting under no_grad, positive finite detached
output, unchanged input/RNG, learning and saved-state linkage, tile/full parity
including boundaries, malformed-input refusal, zero-prior no-network behavior,
actual original trainer callback/RED refresh and source-frequency semantics,
real save-to-validator agreement, counters/clocks, and rejection of stale,
missing, mismatched or forged new/old-control evidence. Use actual producers
for protocol fixtures; do not guess the schema. Follow focused RED/GREEN,
then one complete frozen suite, then independent complete-change review.
Resolve every substantive finding within the user's three-round limit before
GPU. Real zero and smokes remain required after CPU/review passes.

The remaining implementation details are the builder's call within the exact
behavior above. Do not add unrelated refactors, dependencies or evaluation
changes. Keep archived relocation results and its audit intact.
