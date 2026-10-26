# Fixed-count G2 relocation route

`run_relocation.py` reuses the unchanged `run_gpu.execute(protocol=...)` lifecycle:
stdlib parent, common claim lock held through owned-child exit, visibility-aware
GPU selection, observed-context ACK, GSTMI child, maximum 3600 seconds, immutable
completed descriptor. No dependency install, renderer rebuild, provider override,
point-count growth or evaluator change is permitted.

The route is `g2_fixed_count_relocation_v1`. Its exact selected recipe must contain
`schema_version: 1`, `status: selected`, `route`, the exact `parameters`, the full
`run_relocation.POLICY`, current `code_sha256`, and current `invariants`. Parameters
include route, stage, mode, steps, requested_k=18000, event_step=1. These last two
values are fixed defaults, with no public override. The separate review gate must
bind the same source and input snapshots and cover this route. Existing residual,
full-amplitude and proximal-DC gates do not authorize it. CPU fixtures are internal
integration tests and cannot pass the public N=3.6M/CUDA/frozen-source validator.

A reviewed operator creates actual selected recipes and genuine review/control
gates after source freeze. The following are command templates, not evidence that
any numerical stage has run. Use new experiment names and evidence paths for each
attempt, including a failed setup or failed numerical stage.

```bash
python -B run_relocation.py --stage zero --mode control --steps 0 \
  --exp relocation_zero_v1 --review-gate REVIEW.json --recipe ZERO_RECIPE.json \
  --preferred-gpu 0

python -B run_relocation.py --stage smoke --mode control --steps 2 \
  --exp relocation_control_smoke_v1 --review-gate REVIEW.json \
  --recipe CONTROL_SMOKE_RECIPE.json --zero-gate ZERO_COMPLETED.json --preferred-gpu 0

python -B run_relocation.py --stage smoke --mode relocate --steps 2 \
  --exp relocation_event_smoke_v1 --review-gate REVIEW.json \
  --recipe EVENT_SMOKE_RECIPE.json --zero-gate ZERO_COMPLETED.json --preferred-gpu 0

python -B run_relocation.py --stage trial --mode control --steps 500 \
  --exp relocation_control500_v1 --review-gate REVIEW.json --recipe CONTROL500_RECIPE.json \
  --zero-gate ZERO_COMPLETED.json --smoke-gate CONTROL_SMOKE_COMPLETED.json --preferred-gpu 0

python -B run_relocation.py --stage trial --mode relocate --steps 500 \
  --exp relocation_event500_v1 --review-gate REVIEW.json --recipe EVENT500_RECIPE.json \
  --zero-gate ZERO_COMPLETED.json --smoke-gate EVENT_SMOKE_COMPLETED.json --preferred-gpu 0
```

Each arm restores the original seven tensors, all Adam moments/scalar steps,
current LRs, scheduler epoch4500/count4501 and seed42 policy from the exact G2
checkpoint. Geometryfactor2 is already in its base rates and is never reapplied.
The unchanged original training loop replays the wavelet generator to4500 and
clips all seven parameters. Adaptive control stays disabled. The relocation event
runs once in the existing end callback at4501 after the first normal Adam and
scheduler update. It fresh-renders acquired residual energy, uses the original
quaternion conversion and the approved fixed-shape transaction. A zero-eligible-
donor event is a measured no-op, still counted as one event and two extra renders.

Gaussian forward/VJP totals are zero2/0, control-smoke6/4, relocation-smoke8/4,
control-trial1002/1000 and relocation-trial1004/1000. Density-input hooks observe
VJPs after Gaussian backward; ordinary paired loss calls, optimizer and scheduler
steps and FFT work have separate counters. Initialization uses frozen zf kappas
and has zero Gaussian renders. Logging/checkpoints add zero renders. No call to
`trainer.export_results` adds an undeclared second export.

Every completion binds the actual original final `checkpoints/latest.pth`
(iter4500/4502/5000), a separately saved full scheduler, config YAML/JSON, trace,
events, original frequency arrays/manifest, measured states/identities/hashes,
precision/runtime/data evidence and raw/canonical/candidate exports. Candidate
arrays equal the original hard-DC canonical exports in the stored reporting
gauge. Raw exports remain separate. Zero requires complete state preservation
and canonical G2 relative parity<=1e-5; elementwise equality is recorded separately.
Training and total stage wall times and peak allocated/reserved memory are
recorded separately. The parent rechecks actual artifact bytes, finite complex64
volumes, source/input snapshots and prerequisite control bytes.

After genuine trial completion, score each candidate in a unique CPU directory
through the unchanged evaluator:

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B evaluate.py \
  --run-dir outputs/relocation_control500_v1 --out-dir scores/relocation_control500_v1 \
  --name relocation_control500_v1
```

Repeat for the relocation arm. References and ROI arrays enter only this separate
scoring process; the numerical child uses acquired data, G2 parity anchors for
zero only, and previously stored shared reference-derived scalar normalizations.
This remains one-case adaptive development. Only PSNR/SSIM/NMSE decide quality;
report six whole-head comparisons against all ten opponents, all59 planes and six
ROI/background measurements. Append every actual result/failure and decision to
the ledger; zero/smokes/CPU tests add no formal MRI experiment. Two completed500
arms add two. Advance only an observed keeper-margin improvement>0.0001. A
provisional overall win still requires repeat and matched GS-single evidence.

For recovery, retain the failed directory, failure descriptor and logs. Restore
from unchanged G2 through a new version/name and newly bound recipe/gates when
required. Never overwrite completed metadata or relax a numerical/source check.

The CPU validation command is:

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest -q -p no:cacheprovider
```
