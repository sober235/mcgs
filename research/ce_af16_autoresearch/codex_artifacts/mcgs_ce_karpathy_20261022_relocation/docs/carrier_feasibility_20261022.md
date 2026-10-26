# Fixed-subset complex carriers: source feasibility

Status: read-only feasibility study, 2026-10-22. No source changes, compilation, tests, GPU work, MRI input rehashing, or new reconstruction were performed. The active caseprior512 experiment is unaffected. All proposed gates below remain unexecuted.

The carrier is feasible without changing the historical Gaussian source or binary. The smallest integration is an unchanged full Gaussian render plus a new differentiable correction over a fixed subset:

\[
x_c(v)=x^{old}_c(v)+\sum_{i\in S} G_i(v)\rho_{ic}\left[\exp(i2\pi f_{ic}\cdot(v-C_i))-1\right].
\]

This is algebraically replacement of selected Gaussian atoms by Gabor atoms, with the original point count and original complex densities. It is not an independent image residual or a postprocessing network. It costs an additional subset render per contrast. Most importantly, at zero carrier the correction and its gradients with respect to the original seven tensors are zero, while its frequency gradient can be nonzero. A zero-frequency shortcut which returns an unattached zero tensor would destroy that last property.

## Inspection provenance and concrete interfaces

`/home/congcongliu/.codegraph` exists, so `codegraph explore "paired_gs gaussian voxelize tile_voxelize_cuda carrier"` was attempted first. CodeGraph reported no usable index and instructed ordinary tools; none was created. The inspected campaign roots had no `.codegraph` directory. Subsequent reads used `rg` and `sed`.

Here `P` denotes `/home/congcongliu/codex_artifacts/mcgs_ce_autoresearch_20261017/nonlocal_autoresearch/trainer/paired_gs`, and `S` denotes this report's campaign root.

| Existing interface | Observed contract and required new adapter |
|---|---|
| `P/gaussian/tile_voxelizer.py:918`, `TileVoxelizer.forward` | Accepts normalized positions, positive scales, wxyz rotations and one complex density vector. Converts to voxel centers, inverse normalized covariance and tile pairs. Keep this full render unchanged; attach a new subset correction before physics/losses/export. |
| Same file:578,588,607 | `_quaternion_to_rotation_matrix`, `_bbox_radius_vox`, `build_tile_pairs` provide existing geometry and detached stable tile membership. Reuse their exact conventions for the subset. Carrier magnitude is one, so support does not grow. |
| Same file:686 | `_TileVoxelizeCUDA` is a custom autograd wrapper with four differentiable inputs and four returned gradients. A new correction wrapper needs frequency input and a fifth gradient; it must not call or replace the old ABI. |
| `P/gaussian/cuda_ext/tile_voxelize_cuda.cpp` | Binds `forward`, `backward`, and no-grad `forward_with_mask`. None accepts frequencies. The masked API cannot implement training carriers. New source files and a distinct extension module are required. |
| `P/gaussian/cuda_ext/tile_voxelize_cuda_kernel.cu:48,255` | Forward sums real/imaginary density times the Gaussian envelope; backward atomically accumulates center, inverse-covariance and density gradients. These are the mathematical patterns for the new correction kernel, not files to edit. |
| `P/gaussian/joint_gaussian_model.py:57,211` | `get_density(channel)` returns independent complex signals; `get_optimizable_params` returns the seven original groups. Keep those tensors and group identities intact. Store carrier state separately. |
| `P/trainers/joint_trainer.py:478` | `_render(channel)` combines the voxelizer outputs. A new scoped subclass/adapter can add the subset correction for both training and final export. Both paths must use the same carrier state. |
| Same file:975,1184,1198,1203 | Callback events are `start`, `pre_step`, `end`; original Adam zeroing, original seven-parameter global clipping, Adam step and scheduler step already occur. A separate carrier optimizer can use these callbacks without copying the loop. |
| `P/trainers/resume_protocol.py:33,54` | Restore requires exactly seven parameter names and seven Adam groups, and restores scheduler clocks from iteration4500. Registering an eighth model parameter before restore fails. |
| `S/relocation_runtime.py:60,69,222` | `_named_model`, `snapshot_training_state`, `restore_training_state` also require exactly seven state tensors, including no extra buffers. Reuse this proof for the Gaussian component, and add separate, explicitly versioned carrier state/proofs. |
| `S/relocation_runner.py:121,170,227` | Checkpoint readback, seven Adam clocks and render/VJP counters are specific to the old route. A carrier route needs new component-specific evidence and counts; it cannot label old evidence as proving the additional operator. |

The CPU reference should be a new small dense float64 implementation. The existing `_tile_forward_pytorch` forcibly constructs float32 grids and complex64 outputs, so it is not a double-precision gradient-check oracle as written.

## Units and exact local derivatives

Axis order is stored tensor `(D,H,W)`, not physical scanner x/y/z. For this case `n=(256,232,176)` and `h=n/2=(128,116,88)`:

\[
C=(p+1)h-\tfrac12,\quad u=v-C,\quad d=u/h,\quad
M=R\operatorname{diag}((s+10^{-8})^{-2})R^T.
\]

The stored scale parameter is logarithmic, so `s=exp(scales)`. Rotation normalization is `q/(norm(q)+1e-8)`, wxyz. The envelope is `w=exp(-dᵀMd/2)` inside `dᵀMd<=9`, zero outside. Support and tile membership are treated as nondifferentiable, matching the old renderer. Finite differences must avoid support and tile-boundary crossings.

Define frequency `f` in **cycles per voxel** and phase `theta=2π f·u`. Each component must stay strictly inside `(-0.5,0.5)`; a proposed smooth parameterization is `f=f_max*tanh(eta)` with a declared `0<f_max<0.5`, initialized `eta=0`. Its zero derivative is `f_max`, not zero. Bound and learning rate are experiment choices, not settled by this inspection. A carrier written as `exp(i2π omega·d)` instead uses cycles per normalized coordinate, `omega=h*f`; its axiswise Nyquist magnitudes are `(64,58,44)`. Mixing these units causes factors128/116/88 errors. There is no millimeter conversion without voxel-spacing metadata. Frequencies need not be integer DFT bins because atoms are spatially localized.

For a selected atom let `rho=a+ib`, `c=cos(theta)`, `s_theta=sin(theta)`. Write the full rotated coefficient `t_r=a*c-b*s_theta`, `t_i=a*s_theta+b*c`; the correction coefficient is `z_r=a*(c-1)-b*s_theta`, `z_i=a*s_theta+b*(c-1)`. For upstream real-valued loss gradients `(g_r,g_i)`, define

\[
Q=g_r z_r+g_i z_i,\qquad T=-g_r t_i+g_i t_r.
\]

The correction's per-voxel derivatives, summed over contributing voxels, are

\[
\begin{aligned}
\partial_a L &= w[(c-1)g_r+s_\theta g_i],\\
\partial_b L &= w[-s_\theta g_r+(c-1)g_i],\\
\partial_M L &= -\tfrac12 wQ\,dd^T,\\
\partial_C L &= wQ\,(Md)/h-2\pi wT f,\\
\partial_f L &= 2\pi wT u.
\end{aligned}
\]

The center formula assumes symmetric `M`, which the geometry construction guarantees; a general unsymmetric matrix oracle must use `(M+Mᵀ)d/2`. PyTorch then propagates center gradients to normalized positions by multiplying by `h`, and inverse-covariance gradients through exp scales and normalized quaternions. The `-2π wT f` center term is essential: stopping gradients through the phase center produces a wrong geometry gradient at nonzero frequency.

For a direct full Gabor kernel replace `Q` by `g_r t_r+g_i t_i`, and replace `c-1` by `c` in the density derivatives. The same frequency derivative applies. The correction approach avoids recompiling/replacing the ordinary full renderer.

At `f=0`, `z=Q=0`, so all original-parameter correction gradients vanish. But `T=-b*g_r+a*g_i` need not vanish. A falsifying example uses a single off-center supported voxel with `u=(1,0,0)`, `a=1`, `b=0`, and loss equal to output imaginary part: `dL/df_x=2πw`, which is nonzero. A magnitude-squared self-loss or symmetric upstream pattern may accidentally yield zero frequency gradient; neither is an adequate learnability test.

Compute `cos(theta)-1` stably as `-2*sin(theta/2)^2`, retaining its derivative. Direct subtraction loses its quadratic term for small float32 phases. Do not detach the frequency, center, covariance, or selected densities, and do not multiply a zero-initialized frequency by a second zero-initialized gate.

## Small subset and contrast policy

Use a fixed unique sorted index array chosen without reference images/ROIs, frozen and hashed before optimization. A seeded, RNG-isolated sample of Gaussian row IDs is a simple reproducible first choice. Existing checkpoint magnitude could support a different predeclared selection rule, but that changes the hypothesis and needs its own rationale. Selection must not be refreshed from metric outcomes. Leave adaptive densification and relocation disabled, so indices retain their identity and total N stays3.6M.

Carrier sharing is a design decision still to freeze. A shared `f[K,3]` adds3K scalars and accumulates gradients from both contrasts, but imposes a shared local phase slope. Independent `f_plain[K,3]` and `f_contrast[K,3]` add6K scalars and better preserve independent contrast signals. Either retains shared Gaussian geometry. For a first implementation, independent carriers are the more faithful extension of the current independent complex contrast signal; do not silently substitute one policy for the other. The same selected row IDs can be shared.

For illustration only, K=18000 (0.5%) gives108000 trainable scalars for independent carriers. Float32 parameters, gradients and two Adam moments require about1.65MiB, excluding mapping, gathered geometry, temporary volumes and tile pairs. Independent carriers on all3.6M points would require about329.6MiB for those four arrays alone. These are arithmetic storage counts, not measured GPU peaks. Runtime is governed by supported voxel/atom pairs, not just K; a few very broad atoms can dominate. Sparse correction adds a full-volume output buffer and extra launches/accumulation; all-atom replacement could be faster for large K but changes the ordinary execution path and has a larger parity burden.

The Gaussian is already capable of high-frequency content; this inspection proves neither a capacity bottleneck nor expected metric gain. Bounded carrier frequency does not make the Gaussian product band-limited: the envelope has spectral width, and the hard support cutoff has high-frequency tails. Nyquist bounds prevent an avoidable carrier alias equivalence, not all sampling artifacts.

## Optimizer, checkpoint and isolation recommendation

Keep the old model and Adam exactly seven-tensor/seven-group, restore them using the original strict restoration, and keep carrier `nn.Module` and Adam on a separate route-owned object. Do not attach that module under the original model, since its `state_dict()` and `parameters()` would change. Gather selected original tensors without detaching so correction geometry/density gradients accumulate into the original model.

In the original loop's `start` callback, clear carrier gradients. At `pre_step`, require finite carrier gradients and apply a separately declared carrier clipping bound. At `end`, step the carrier optimizer once using the same iteration's already-computed gradients; there is no second forward/backward. The original optimizer and scheduler have already stepped by `end`, so record this ordering. The source loop saves its Gaussian checkpoint after `end`; save a matching carrier state with iteration, selected IDs, units, transform, optimizer state, scheduler policy, architecture and source hashes. Export and resume must fail on missing/mismatched carrier components rather than silently reverting to zero. Separate clipping preserves the original seven-tensor clipping operation; global clipping across both blocks would change old Gaussian updates even at zero carrier if its gradient is nonzero. Either policy must be explicit, never described as identical ordinary optimization.

Keep the exact old binary loaded by the existing verified path. Build only new correction files in a unique carrier-owned directory under a content/version-named module, never the historical `cuda_ext/build`. Bind the new source digests, binary digest, compiler, flags, Torch/CUDA ABI and architecture target into the new route. The existing loader can fall back to CuPy/PyTorch; the new formal carrier route must instead require the declared new binary and refuse fallback. No new build was attempted here, so compiler/toolkit availability and successful ABI loading remain unverified.

The new CUDA launcher must validate rank/dtype/device/contiguity, subset indices and shapes before pointers are used; use the current PyTorch stream, a device guard, and launch checks. Preserve deterministic per-tile forward ordering and disclose atomic backward roundoff. Avoid copying the old forward kernel's thread-dependent voxel loop with block barriers into a general ragged-grid implementation: ensure all threads participate in every barrier, including tiles with fewer than256 valid voxels. Production dimensions are multiples of8, but tiny reference fixtures and boundary tiles must also work. Test this obligation; it is a source-inspection concern, not a reproduced old failure.

## Minimal falsifiable acceptance gate

1. **CPU float64 correctness, no MRI:** tiny anisotropic3D grids, non-unit wxyz quaternion, nonzero real/imaginary coefficients, multiple overlapping atoms, partial subset and both contrasts. Check correction plus ordinary output against direct full Gabor summation. Analytic VJPs must match autograd and central differences for normalized positions, log scales, quaternion components, real/imaginary densities and frequencies at zero and nonzero frequencies. Proposed CPU thresholds: `rtol=1e-5, atol=1e-7` for finite differences away from support crossings; tighten algebraic/autograd comparison to `1e-10`. Include the explicit nonzero zero-frequency derivative example above, axiswise quarter-cycle phase and normalized/voxel frequency conversion.
2. **Isolated CUDA microgate:** compare the new float32 correction and VJPs with the float64 CPU reference rounded appropriately, including ragged tiles, empty subset, K=1, non-contiguous input handling, duplicate/out-of-range index refusal, far-outside-support atoms, mixed signs and near-boundary frequencies. Proposed thresholds on bounded small fixtures: `rtol=2e-4, atol=2e-5`, declared before observing results. At zero carrier require exactly zero correction and exactly zero old-parameter correction gradients, with finite nonzero carrier gradient in the constructed example. Compare repeated atomic VJPs and report variation. If thresholds fail, fix or stop; do not silently relax them after seeing errors.
3. **Continuation/control gate:** source-bound genuine G2 zero stage must preserve its two raw volumes and canonical hard-DC outputs. A carrier-disabled two-update control must reproduce the old seven-tensor path within its established tolerance, with original Adam/scheduler clocks and no carrier updates. An enabled two-update smoke must demonstrate actual nonzero carrier gradients and changed carrier parameters, both contrasts reaching physics/output, original N/mask/CSM unchanged, separate optimizer clocks and a safe CPU checkpoint readback reproducing the saved state. Count old full and new subset renders/VJPs separately. Native-control reuse is conditional on these new proofs, not automatic.
4. **Only then one predeclared MRI trial:** fixed subset/bound/LR/clipping, same500 Gaussian updates and final hard-DC protocol; new formal count and actual additional wall time/memory. Evaluate the unchanged PSNR/SSIM/NMSE contract, all59 planes and enhancement regions, compare against keeper and all ten opponents. A training-loss decrease or sharper appearance alone is failure to establish quality progress. Promotion/repeat/matched GS-single/visual review retain the campaign's existing rules.

Failure of the small derivative/zero-carrier gates is sufficient to stop this route before costly MRI work. Passing them establishes implementation consistency only, not metric superiority.
