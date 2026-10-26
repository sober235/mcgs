# G2-only amplitude representation diagnostic

This route implements the frozen [representation design](superpowers/specs/2026-10-22-full-amplitude-representation-design.md).
It reads the G2 model and metadata, original Gaussian Python/CUDA binary, and
checksum-bound G2 raw teachers. It does not load acquisitions, CSM, masks,
references or ROI arrays and contributes zero formal MRI experiments.
[Current progress](representation_progress.md) is separate from historical campaign reports.

The public launcher is `run_representation.py`. Its supervisor imports only
stdlib modules and reuses `run_gpu.execute(protocol=...)`, the common GPU claim,
visibility-aware selector, owned-child reap/timeout and observed-context ACK.
The child reuses `proxdc_runner.wait_context_ack` unchanged, restores only the
original seven-tensor model, prebinds the existing binary, and uses its original
raw Gaussian forward and density VJP. No renderer rebuild or fallback is allowed.

Before launching, obtain a real clean independent CPU review and checksum-bound
review report, snapshot `code_hashes()` and `snapshot_invariants()`, and select a
recipe whose `parameters` exactly equal `parameters(parser().parse_args(CLI))`.
The review and selected recipe must bind those exact snapshots and this route
(`g2_full_amplitude_representation_v1`). Inherited amplitude/proximal gates
cannot authorize this route. CPU fixtures cannot establish actual CUDA parity,
performance, a real render gate or representation admission for G2.

Use unique names and real reviewed descriptors in place of these placeholders:

```sh
/home/congcongliu/anaconda3/envs/GSTMI/bin/python -B run_representation.py \
  --stage render_control --encoder none --exp g2_raw_control_UNIQUE \
  --review-gate verification/REAL_REVIEW.json \
  --recipe verification/REAL_RENDER_RECIPE.json --preferred-gpu 0

/home/congcongliu/anaconda3/envs/GSTMI/bin/python -B run_representation.py \
  --stage fit --encoder mlp --exp g2_fullamp_mlp_UNIQUE \
  --review-gate verification/REAL_REVIEW.json \
  --recipe verification/REAL_MLP_RECIPE.json \
  --render-gate outputs/g2_raw_control_UNIQUE/completed.json --preferred-gpu 0

/home/congcongliu/anaconda3/envs/GSTMI/bin/python -B run_representation.py \
  --stage fit --encoder grid --exp g2_fullamp_grid_UNIQUE \
  --review-gate verification/REAL_REVIEW.json \
  --recipe verification/REAL_GRID_RECIPE.json \
  --render-gate outputs/g2_raw_control_UNIQUE/completed.json --preferred-gpu 0
```

Run MLP first, then the predeclared grid alternative. The fixed parameters object
binds seed 42, bands 8, width 64, chunk size 131072, 2000 coefficient Adam updates
(batch 32768 sampled with replacement), and 128 full raw updates. Both phases
use LR .001, betas (.9,.999), epsilon 1e-8, no decay/scheduler and gradient norm
clip 1.0. Raw fitting starts with a new Adam optimizer. Geometry, source unit
phases and zero phase deltas remain frozen; only shared amplitude parameters
update. Raw errors use float64 reductions and the fresh trainer-gauge teacher.
The original TF32 flags/default dtype are preserved; autocast is disabled.

`--timeout` may lower the default 1800-second owned-child maximum. Budgets,
architecture and admission thresholds cannot be altered by CLI flags. Control
runs perform two observed Gaussian forwards and no VJP. A completed fit must
have 264 observed forwards, 256 density VJPs and complete 2000/128 update traces.
Missing or nonfinite arithmetic, counts, ACKs, checkpoints and changed input,
recipe or artifact bytes cause refusal/failure with retained evidence.

`completed.json` means successful software completion. Read
`metrics.json`/`provenance.json` for `representation_admitted`: both final raw
complex relative L2 errors must be at most .01. A finite fit above this bound
completes normally with admission false. Use the final fixed-budget checkpoint
(`final_field.pth`), never a score-selected intermediate. Its descriptor and all
traces/progress/context artifacts are hashed in the completion gate. Failed and
negative outputs remain in their unique directories; reruns require new names.

Representation admission is a warm-start check. It is not PSNR, SSIM or NMSE,
not MRI quality, and not evidence of a general capacity limit when an arm fails.
Only a separate reviewed continuation may introduce the frozen MRI likelihood,
evaluator, 59-plane/ROI checks and a new MRI training experiment.

CPU validation uses the existing GSTMI environment:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest -q -p no:cacheprovider
```
