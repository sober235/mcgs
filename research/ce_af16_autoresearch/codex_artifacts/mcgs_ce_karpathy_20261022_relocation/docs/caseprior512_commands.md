# Fixed512 case-internal magnitude prior

`run_caseprior512.py` supervises route
`g2_case_internal_magnitude_prior_fit512_v1`. The fixed adapters execute the
unchanged reviewed `run_caseprior.py` and `caseprior_runner.py` in isolated
registered modules. `caseprior_budget_variant.load_protocol()` returns
`_caseprior_fit512_protocol_v1`; `load_child()` returns
`_caseprior_fit512_child_v1`. Both accessors cache one module per process.
Original public protocol functions and parser are available on the returned
protocol. The entry point exposes `parser()` and `main(argv=None)`; the child
entry exposes `execute(path)` and its `--launch` command. Original `main()`
resolves its own registered private module through `sys.modules[__name__]`.

Only CNN updates per fit change128 to512. Fresh initialization and Adam per
call, seed42+1009*j, Adam learning rate0.001, width32, batch4, patch32,
tile32/halo5, five3x3x3 convolutions and original NonLocalRED0.1,
sigma_rel0.015/start4501/refresh100 stay fixed. Learned smoke has two fits,
1024 total CNN updates; learned500 has ten fits,5120 total. Gaussian work
remains500 updates from original G2 at4500,3.6M points, all seven original
Adam states and gamma0.9995. Zero/control constructs no CNN. CNN wall time and
peak GPU memory remain measured separately; this is no equal-time claim.

The private child factory supplies the fixed settings explicitly to the real
unchanged `CaseMagnitudeDenoiser` constructor, whose inherited default remains128.
Its existing private tiny-fixture hook remains private. There is no public
budget, backend, provider, point-count, seed, network or loss override.

All221 inherited executing sources and the three new adapter/entry files are
bound by `load_protocol().code_hashes()`. Reviews and selected recipes must
use that fresh224-file closure, the new route/policy and one fresh input
snapshot. The original native218-file source closure is kept unchanged.
No archived128 completion or review authorizes this route. Before trial,
obtain an independent clean review and produce genuine new zero and both
smokes. These are command templates, not execution receipts:

```bash
python -B run_caseprior512.py --stage zero --mode control --steps 0 \
  --exp caseprior512_zero_v1 --review-gate REVIEW512.json --recipe ZERO512_RECIPE.json

python -B run_caseprior512.py --stage smoke --mode control --steps 2 \
  --exp caseprior512_control_smoke_v1 --review-gate REVIEW512.json \
  --recipe CONTROL_SMOKE512_RECIPE.json --zero-gate ZERO512_COMPLETED.json

python -B run_caseprior512.py --stage smoke --mode learned --steps 2 \
  --exp caseprior512_learned_smoke_v1 --review-gate REVIEW512.json \
  --recipe LEARNED_SMOKE512_RECIPE.json --zero-gate ZERO512_COMPLETED.json

python -B run_caseprior512.py --stage trial --mode control --steps 500 \
  --exp caseprior512_control500_v1 --review-gate REVIEW512.json \
  --recipe CONTROL500512_RECIPE.json --zero-gate ZERO512_COMPLETED.json \
  --control-smoke-gate CONTROL_SMOKE512_COMPLETED.json \
  --learned-smoke-gate LEARNED_SMOKE512_COMPLETED.json

python -B run_caseprior512.py --stage trial --mode learned --steps 500 \
  --exp caseprior512_learned500_v1 --review-gate REVIEW512.json \
  --recipe LEARNED500512_RECIPE.json --zero-gate ZERO512_COMPLETED.json \
  --control-smoke-gate CONTROL_SMOKE512_COMPLETED.json \
  --learned-smoke-gate LEARNED_SMOKE512_COMPLETED.json \
  --native-control-receipt NATIVE512_RECEIPT.json
```

Learned trial requires exactly one `--native-control-receipt` or `--control-gate`.
For fallback, replace the receipt flag with
`--control-gate CONTROL500512_COMPLETED.json`, obtained by the new control500
command above. The native receipt reuses schema `native_control_receipt_v1`,
with `route:g2_case_internal_magnitude_prior_fit512_v1`, `status:verified`,
new `code_sha256`/`invariants`, actual native `native_code_sha256`/
`native_invariants`, and path/SHA256 descriptors `native_completion`,
`native_smoke`, `ordinary_smoke`, `objective`. These must reference the actual
old native500, old ordinary smoke, genuine new512 ordinary smoke and scored
native objective. All old completion/artifact/ACK/restore/clock/postproof,
canonical relativeL2<=1e-5 smoke parity and evaluator/array bindings remain
checked by the inherited validators. Common old inputs derive from the one
new snapshot with old route/policy restored; no second MRI snapshot is needed.
Any refusal is preserved and followed by a new versioned control500.

The inherited supervisor retains owned GPU claim/reaping/context ACK,
fixed GSTMI interpreter/renderer, cancellation, maximum3600 seconds, immutable
artifacts and before/after proofs. Checkpoint semantic readback and all-visible
caller RNG observations still use the original child producer and schemas.
CPU fixtures neither create nor authorize scientific gates.

All data/42 coils, historical AF16 vdPoisMex mask (explicit legacy exception,
not BART), CSM, shared reference-derived normalization, original renderer,
canonical export and ten opponents stay fixed. No scoring reference, ROI,
opponent, external data or pretrained weight enters fitting. Use the unchanged
separate evaluator with a unique output directory after genuine completion:

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B evaluate.py \
  --run-dir outputs/caseprior512_learned500_v1 \
  --out-dir scores/caseprior512_learned500_v1 --name caseprior512_learned500_v1
```

Only PSNR/SSIM/NMSE decide quality. Report all six whole-head comparisons,
59 planes and six ROI/background enhancement measurements. Keep the observed128
candidate as an additional comparison; never relabel its evidence. Promotion
requires keeper+0.0001; overall success still requires all six strict inequalities,
later independent repeat, matched improved GS-single and visual inspection.
This is adaptive same-case development. Loss decrease establishes no MRI gain.

Preserve failed/partial artifacts and choose fresh experiment/scoring names for
recovery from unchanged G2; never overwrite old calls or numerical results.
Focused CPU verification is `tests/test_caseprior512.py`; the complete command is:

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest -q -p no:cacheprovider
```
