0 actionable findings. No verified blocking defect was found in the reviewed CPU preparation change.

Independent review of the complete working-tree change against
`000c53d9743183c7d5f365690bc4f83d9cf34273`, including the untracked candidate
document and `tests/test_anchor_features.py`. Read `AGENTS.md`, `docs/design.md`
and `docs/anchor_features_candidate.md` before reviewing the implementation.
The starting worktree already contained the author's modifications to
`amplitude_model.py`, `runner.py`, `run_gpu.py`, three existing test files, and
the two new candidate/test files. The reviewer changed only this report.

The intended intervention is a shared bounded gain field conditioned on both
frozen G2 coefficient moduli, with independent per-point phase. It preserves the
coordinate Fourier features, geometry, coefficient buffers, priors, bounds,
learning rate and declared update budget. It is not a fully implicit replacement
of G2. No external weights or scoring inputs enter the added feature path.

Reviewed evidence:

- Coordinate mode preserves the original module construction, random draws,
  tensor state keys, forward values and common parameter gradients. The tests
  execute `000c53d:amplitude_model.py` as the fixed numerical oracle. The paired
  projection consumes no random draws and adds zero to the initial first-layer
  preactivation. The common initial weights and RNG state match coordinate mode.
- Each feature column uses the complete corresponding frozen coefficient column
  before chunking. Positions and coefficients are privately cloned detached
  buffers; derived feature and median buffers are also detached and serialized.
  Nonfinite inputs/moduli/features are rejected for paired mode. Zero and empty
  coefficient cases are covered. The declared lower-middle median and epsilon
  match the implementation.
- Both plain and contrast feature columns affect the shared gain, while phase
  columns remain independent. The added projection belongs exactly once to the
  actual residual-amplitude optimizer group. A separate CPU probe starting from
  the unmodified all-zero final layer/projection used three real Adam steps.
  Projection gradient norms were `[0, 0.022303, 0.045451]` in float32 and
  `[0, 0.010399, 0.021271]` in float64; its maximum weight changes were zero on
  step one and positive on steps two and three. This verifies effective
  optimization without manually activating the last layer.
- Each non-reentrant checkpoint receives only its feature slice. Output and
  gradient equivalence between chunked and whole evaluation, saved activation
  shapes, and strict paired state roundtrip are covered. The full `[N,2]` cache
  remains an explicit fixed memory cost; real CUDA peak memory is not inferred
  from these CPU checks.
- The required public `feature_mode` passes through CLI validation, PARAMETERS,
  exact recipe equality, launch fields, completed gate parameters, metrics and
  provenance, field metadata and checkpoint. Baseline/explicit reject paired
  mode. Paired smoke/trial require matching shared+paired zero evidence; paired
  trial also requires the matching two-step smoke. Missing or inconsistent child
  mode/features/schema are refused. A second read-only reviewer independently
  checked this interface and reported zero actionable findings.
- Checkpoint/model schema is explicitly `g2_residual_amplitude_v2`, with paired
  architecture and normalization information in field metadata and cached
  features/medians/projection in the state. No legacy-v1 resume compatibility is
  claimed. The old queue is explicitly excluded in the candidate document.
  `runtime.py`, `evaluate.py` and `autoresearch_loop.py` were confirmed byte
  identical to the base revision. Resource cancellation/root-freeze behavior
  was not changed by this diff; only its new parameter binding was in scope.

Observed validation, with GPU visibility disabled:

```sh
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest -q -p no:cacheprovider
```

Result: **134 passed in 11.34s**. The independent interface reviewer also ran
`tests/test_run_gpu.py tests/test_runner.py tests/test_loop.py`: **77 passed in
10.38s**. `git diff --check 000c53d` exited 0. The additional synthetic CPU probe
above exited 0 and confirmed the reviewed source hashes below and unchanged
runtime/evaluator/queue bytes. No actual clinical GT was loaded, and no GPU or
numerical stage was run by either reviewer.

Reviewed production SHA256 values:

| File | SHA256 |
| --- | --- |
| amplitude_model.py | 1d1601992e2e4610d16de75ab4612d7f191f6fc3af96a9ebe7f8d15bf14ba4d3 |
| runner.py | 2d35289121fc50296af271a18e301550d2099cdcad4d1636f42f1c7df03e424e |
| run_gpu.py | 2928f89d3f129a4672f08fb4faafed5402232fc8b4784c09d1dccfa601b2b873 |
| runtime.py | a42fb02828910c4ba3526439444b8170b4a4a2767ebae2c6a932c09ea26efd10 |
| evaluate.py | 9a2c38d1942d6d6ad2331f3364dfad5be86a03cb316fec96f27deb7c5e40a875 |

Remaining coverage is material: actual paired AF16 zero-step export parity,
CUDA arithmetic/peak memory, a real two-step smoke, convergence and all frozen
PSNR/SSIM/NMSE/clinical/ROI scores remain unverified. This report is a CPU code
review, not an observed zero/smoke/trial gate, recipe selection, quality claim or
permission to bypass the existing source/input/recipe binding. Any later
numerical sequence must obtain its own genuine stage evidence under the
reviewed hashes and fixed benchmark.
