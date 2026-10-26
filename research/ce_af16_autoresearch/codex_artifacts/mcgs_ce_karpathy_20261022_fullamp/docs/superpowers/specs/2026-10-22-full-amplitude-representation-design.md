# G2-only full amplitude representation experiment

## Intent and authority

Continue the user's approved CE plain/enhanced AF16 Karpathy autoresearch design:
replace independent Gaussian coefficient moduli with a case-internal shared
function while retaining shared 3D Gaussian geometry and independent complex
phases. The October19 residual modulation trials did not implement this full
replacement. Three500-update residual trials and the fixed proximal output
correction were discarded. Whole-head keeper remains lowreg_f075_solve_v1,
minimum-six margin -0.11118128532897047; all candidates still had0/59 paired wins.

This phase is a representation diagnostic, **not an MRI-quality trial**. Existing
user authorization to execute the described design and the October22 continue
request covers these implementation choices. Encoder details, fit budgets and
the admission threshold below are the researcher's calls, not additional claims
of user-selected hyperparameters. Do not pause for routine permission.

## Frozen boundaries

- No original source/data/completed output edits, new packages, renderer rebuild,
  external training data/weights, scoring references, ROI masks or oracle fits.
- GSTMI Python, original Gaussian Python and exact existing CUDA binary only.
- No kspace, CSM, acquisition masks or reference images are loaded by this
  diagnostic, and no scoring Pair/evaluator is constructed or MRI scoring called.
  The unchanged resource selector imports legacy stdlib objective definitions;
  those definitions must not evaluate metrics or load MRI/reference arrays.
  Preserve AF16/42coils/mask/CSM/ten opponents for
  a later separately reviewed MRI trial; no such trial is implemented here.
- MRI quality continues to mean PSNR/SSIM/NMSE only. Representation error below
  is an initialization check and never substitutes for these metrics.
- Preserve all inherited numerical files. New code is confined to this worktree.
- Source geometry and source phases are frozen throughout representation fitting.
  No per-point amplitude buffer/residual/gain or trainable phase may bypass the
  shared function. Store source unit phases and two global amplitude scales only.

## Sources and faithful raw renderer

G2 checkpoint: sibling mcgs_ce_autoresearch_20261017/frequency_adam/outputs/
adamflat_g2_v1/training/ar_i1_k30p90/checkpoints/latest.pth, SHA256
282e0303c46a6c18775d6224c032611072bd465b4ef975ba7a7e2ea636230fd0.
Seven strict state tensors: positions, log scales, wxyz rotations, plain real/
imaginary densities, contrast real/imaginary densities. N=3600000; volume is
(256,232,176). Source iteration4500/frequencyt1/geometryfactor2 is verified;
geometryfactor2 describes historical optimizer LRs, not a render multiplier.

Original code is sibling mcgs_ce_autoresearch_20261017/nonlocal_autoresearch/
trainer/paired_gs/gaussian. Use JointGaussianModel3D, strict full state load and
TileVoxelizer(volume,tile_size8,max_radius20,use_cuda=True,strict_cuda=True).
Prebind the exact extension before construction: tile_voxelize_cuda_0141c525_
7aedd3b0.so, SHA3973daeef5f55ebfed931c38538a37d83921a9287b96b69a431da238057194b5.
Reject a different imported source/module/backend, compilation or fallback.

Raw call is voxelizer(model.positions, model.get_scale_values(), model.rotations,
density), then torch.complex(real,imag). The original voxelizer owns all coordinate,
quaternion, covariance, truncation and CUDA VJP transforms. Never replace them.
All seven original tensor bytes remain unchanged. No JointTrainer is constructed.

Checksum anchors come from completed sibling mcgs_ce_karpathy_20261019_proxdc/
outputs/g2_proxdc_canonical_v1. Verify completed descriptor and raw array bytes:
completed.json SHA9ab09a15fe422df7c188ac6a4951985281cf917d071ff5b7dc2cf95aa663b0ee;
raw_pre_complex.npy SHA202985d20e2a6554e369e5573668169d30ae73096f985753acd0caa172da92ab;
raw_post_complex.npy SHA9830d5bcb22a70d5af5cd56c2d4c01d248e694911d81888295cc796f253dee75.
They are G2-derived raw volumes, not references or hard-DC candidates. Compare a
fresh original raw render after the exact CPU complex64/float32 gauge arithmetic:
pre factor0.9997583823636317; post1.0003600786949414. Both relative complex L2 must
be <=1e-5. Report array equality separately. Keep the fresh trainer-gauge teacher
on GPU for fitting; do not reverse a rounded saved gauge multiplication.

## Full amplitude function

FullAmplitudeField(positions, log_scales, plain_complex, contrast_complex,
encoder='mlp'|'grid', bands=8, width=64, chunk_size=131072).
It privately clones/detaches inputs, rejects bad shapes/dtypes/nonfinite values,
and uses complete-source log-scale mean/std (floor1e-6). Two amplitude scales are
complete-source median moduli, floored1e-12. Unit phase is z/abs(z), with unit1 at
zero. A separate phase_delta[N,2] starts at zero and is frozen for this phase.

Both encoders feed two hidden64 SiLU layers and a two-output layer. Output is
amplitude_scale[c]*softplus(raw[c]); final weights zero, final bias
inverse_softplus(1), giving the same initial constant moduli in both arms.

1. MLP: original-unit xyz, sin/cos(pi*2**k*xyz), k0..7, plus standardized three
   frozen log scales. This follows the earlier coefficient diagnostic.
2. Grid: xyz plus standardized log scales and learned trilinear features from
   resolutions8,16,32,64 with two channels each, initialized N(0,1e-4).
   PyTorch grid_sample only; xyz/1.1, align_corners=True, padding_mode='border'.
   Report the fraction outside the declared domain. No hash-grid dependency.

Each complex density is predicted modulus times source unit phase times
exp(i*phase_delta), independently indexed by contrast. There is no original
per-point modulus in forward. Equal positions/log scales yield equal predicted
moduli even if original moduli differ. Chunked nonreentrant checkpoint preserves
network/grid gradients; optional point indices support minibatch teacher fitting.
Public methods: magnitudes(indices=None)->[n,2], get_density(channel,indices=None)
->[n] complex; amplitude_parameters()->list; metadata()->JSON-safe dictionary.

## Fixed fitting and diagnostic admission

Each arm starts fresh from sourceG2 and seed42, with phase_delta frozen at zero.
Stage1:2000 Adam updates, batch32768 sampled with replacement from all source
points; LR.001, betas(.9,.999), epsilon1e-8, no decay, no scheduler. Loss is mean
per-contrast squared log-modulus error standardized by source log-modulus std
(floor1e-6; log floor1e-12). No held-out MRI claim.

Stage2: reset Adam moments, same LR/betas/epsilon;128 updates on the full fresh
G2 raw complex teacher. Loss is half the sum over contrasts of
sum(abs(render-teacher)^2)/sum(abs(teacher)^2), accumulated in float64. Only
amplitude parameters update; all gradients must be finite; norm clip1.0.
No phase fitting, per-point amplitudes, geometry fitting, DC or extra rendering
postprocessor. The original density VJP must remain connected.

Record errors initially, after coefficient fit and finally, actual update traces,
time/peak memory, parameter counts, immutable-source hashes, source/backend,
phase/geometry invariance, final checkpoint and actual Gaussian render counts.
Use the final checkpoint, not a score-selected intermediate.
The fixed fit executes264 Gaussian forwards:2 teacher,2 initial,2 after coefficient
fit,256 image-fitting and2 final. Its256 Gaussian VJPs must actually execute.
Render-control executes2 forwards and no VJP. Counters observe calls/hooks;
they are not manufactured from the requested loop length.

Researcher admission rule chosen before experiments: both final raw complex
relative L2 errors <=0.01. This is a pragmatic warm-start fidelity bound, not a
claim of exact G2 identity or MRI improvement. Failed fidelity is a completed
negative diagnostic; nonfinite arithmetic/timeout/incomplete traces are numerical
failures. Do not weaken the bound to admit a result. A low-capacity fit failure
does not prove a general neural-function capacity limit.

Run MLP first, then grid as the second predeclared encoder alternative. Both have
the same fit budget/decoder/source and report differing parameter/compute costs.
Neither run increments the28 formal MRI trials or the one output-correction count.
Only an admitted architecture may move to a separately reviewed500-update MRI
likelihood experiment with the fixed existing evaluator, all59planes and ROIs.

## Supervision and evidence

Reuse the unchanged inherited run_gpu.execute(protocol=...) lifecycle, common
claim lock, idle-device choice, owned-child timeout/reap and observed-context ACK.
Parent imports stdlib only. New run_representation.py protocol and child are
separate from inherited amplitude/prox routes; old numerical gates cannot pass
this new route. Code hashes include executing new files, lifecycle/ACK helpers,
selector sources, Gaussian Python sources and binary. Input snapshot includes
actual verified G2 checkpoint/metadata, raw anchor descriptors/arrays and gauges;
it does not pretend to validate unread MRI acquisitions. Fresh CPU review and
exact selected public-CLI recipe must bind current source/input snapshots.

Stage render_control has encoder='none', zero updates, faithful source rendering.
Stage fit has encoder='mlp'|'grid', the fixed2000/128 budget and a real matching
render-control gate. Check actual result values/counters/artifact hashes and ACK,
not a self-reported passed boolean. Software completion and representation
admission are separate fields. Preserve unique output directories and failures.
Maximum owned-child wall time1800s. Preserve other GPU workloads and visibility.

## Calls and recovery

Use a sibling worktree so completed gated sources retain their bytes. The native
task worktree API has no repository argument and this task's home is not a Git
repo; manual sibling placement follows existing campaign layout/relative paths.
No dependency setup is needed. Baseline inherited suite183 passed in15.54s.
Builder decisions above are within the already approved amplitude direction.
Failure keeps all evidence and requires a new version, never overwriting results.
