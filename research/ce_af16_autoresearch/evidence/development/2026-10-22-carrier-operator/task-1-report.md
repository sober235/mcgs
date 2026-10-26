# Task 1 report: CPU carrier reference

Status: DONE for controller inspection and independent review. No commit or staging performed.

## Implementation and intended behavior

Created only `carrier_reference.py` and `tests/test_carrier_reference.py`, plus Task 1 evidence in this workspace. The public signature is exactly `carrier_correction_reference(positions, log_scales, rotations, density, frequencies, volume_shape)`. It returns a CPU complex128 volume from CPU float64 geometry/frequencies and complex128 density, with native Torch autograd through all five inputs.

The implementation preserves D,H,W axis order, `h=shape/2`, `C=(positions+1)*h-0.5`, normalized displacement, exp log scales, wxyz normalization by `norm+1e-8`, and `R diag((exp(log_scales)+1e-8)^-2) R.T`. The exact detached `mahal<=9` support is used. The real correction coefficient is evaluated as `-2*sin(theta/2)^2`; the imaginary coefficient is `sin(theta)`. Centers remain in the phase graph, and zero frequency has no shortcut. The private geometry evaluation function provides a narrow local derivative test boundary. Normal strided inputs need no copy; the same operations naturally retain connected, correctly shaped gradients for K=0.

The public boundary rejects non-tensors, non-CPU tensors, incorrect dtypes/ranks/widths/row counts, NaN/Inf including imaginary density, invalid three-dimensional positive-integer volume shapes, and frequency components outside strict Nyquist. It imports no historical renderer, CUDA extension loader, or build utility.

## Baseline and TDD evidence

The initial pre-edit tool inspection showed an empty tracked diff and only the controller-owned untracked `carrier_state.json`. `task-1-starting-diff.txt` is the controller's pre-edit tracked-diff record. `task-1-baseline.json` additionally records the base revision, empty tracked diff and working status after writing the test but before creating the implementation. Other workers' state and documentation were preserved.

Both runs executed the exact required command:

```sh
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest -q -p no:cacheprovider tests/test_carrier_reference.py
```

| Run | Actual subprocess exit | Pytest result | Pytest seconds | Subprocess wall seconds |
| --- | --- | --- | --- | --- |
| RED, before implementation existed | 1 | 93 failed, missing `carrier_reference` module | 3.32 | 4.084274 |
| First and only GREEN | 0 | 93 passed | 2.05 | 3.059251 |

Captured outputs and actual exits are in `task-1-red.stdout.txt`, `task-1-red.stderr.txt`, `task-1-red.json`, `task-1-green.stdout.txt`, `task-1-green.stderr.txt`, and `task-1-green.json`. Both stderr captures are empty. RED consisted of test failures, not collection errors. The test SHA stayed identical from RED to GREEN. No test or implementation changed after GREEN; there was no repeat pass or unrelated legacy-suite run.

## Acceptance coverage

- Independent Gaussian and direct full Gabor sums, using a separate skew-matrix quaternion construction and per-atom loop, compare outputs and all-five-input autograd VJPs. Two independent complex density/frequency fixtures exercise both contrast coefficient patterns, overlap, signed real/imaginary coefficients, non-unit quaternions, and zero/nonzero frequency.
- Documented local analytic VJPs for centers, inverse covariance, complex density and frequency match implementation autograd. The center formula explicitly includes `-2*pi*w*T*f`. Chaining the independently constructed center/covariance geometry back to positions, log scales and quaternions checks all five public-input VJPs.
- Central differences check every scalar of positions, log scales, rotations, both real and imaginary density components, and frequencies at zero and nonzero frequency. Every plus/minus perturbation is explicitly checked against the original hard-support pattern, avoiding support crossings.
- Frozen tolerances: algebraic/autograd `rtol=atol=1e-10`; finite difference `rtol=1e-5`, `atol=1e-7`, step `1e-6`. No tolerance was changed. The tiny-phase real coefficient additionally passes the stricter zero-absolute-tolerance check.
- Zero frequency gives exactly zero correction and exactly zero gradients for positions, log scales, rotations and density. Frequency gradients are finite/nonzero under asymmetric upstream values, and the isolated real-density, imaginary-loss voxel example matches `2*pi*G` directly.
- Axiswise quarter cycles and normalized/voxel frequency equivalence on anisotropic (3,4,5) dimensions catch unit errors. Tiny phase retains the quadratic real correction.
- K=0 connected gradients, K=1, overlapping additivity, far-outside support output/VJPs, exact `mahal=9` inclusion and exclusion beyond it, and noncontiguous inputs with output/VJP parity are covered.
- All five argument positions are exercised for shape, rank, dtype, non-tensor, non-CPU (meta device, requiring no GPU), and nonfinite refusal; coordinate width, imaginary density nonfiniteness, every axis's Nyquist endpoints/outside values, near-Nyquist valid frequencies, and invalid volume shapes are covered.

## Source freeze and preservation

Hashes were frozen before and checked after the final focused test (`task-1-green-before.json` and `task-1-green.json`):

| File | SHA256 |
| --- | --- |
| `carrier_reference.py` | `4d7e0ec7ca32d50462dde583a1b573485b6971c82641145ef6a3726e9f376c78` |
| `tests/test_carrier_reference.py` | `637082ab365cc18ded133f1e6a46cb4a6ed060cc4a30b95f0cdba5a82b4345ca` |

All 224 paths in `historical-source-before.json` were rehashed directly, with zero differences (`task-1-historical-source-after.json`, `task-1-historical-comparison.json`). That closure includes the historical renderer binary. No MRI data was accessed or rehashed, no original outputs or inputs were changed, no GPU was selected/used, no dependency was installed, and no CUDA build was invoked.

## Separate self-review and limitations

0 actionable findings. Inspected the complete owned source/test additions after GREEN and checked quaternion/matrix orientation, scale epsilon placement, voxel-center phase gradients, complex-density conventions, detached support, validation before numerics, K=0 graph connectivity, noncontiguous graph preservation, and unchanged exact acceptance thresholds. The controller's fresh independent review is still required; this is a self-review only.

This dense CPU oracle allocates O(K*D*H*W) intermediates and is intended for tiny bounded numerical fixtures. No CUDA numerical or launch safety claim, MRI integration proof, production-scale performance claim, optimizer trajectory claim, or scientific quality gain follows from these tests. No historical 606-test suite was run because the authorized work is new isolated component code, and the task explicitly requires only the focused CPU acceptance command. Recovery requires dropping these new files only; there are no persistent-data changes.

The controller owns staging/commit, exact BASE..HEAD review packaging, fresh spec/quality review, substantive review resolutions within three rounds, and authorization of Task 2.
