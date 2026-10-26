# Paired frozen-G2 coefficient conditioning candidate

This isolated branch prepares one additional shared-gain arm for the fixed AF16
case. It has CPU acceptance evidence only. No numerical GPU stage, reconstruction
score, selected recipe, passing stage gate, or quality claim has been produced for
this candidate. The original coordinate-only shared500 and explicit500 campaign
continues separately. Its results determine whether this candidate warrants a
new, reviewed GPU sequence. Do not run this branch's unchanged
`autoresearch_loop.py`: that queue belongs to the original coordinate-only family
and has not been adapted to the new required feature parameter.

## Single change and frozen boundaries

The shared gain MLP may use `feature_mode="paired_anchor"` in addition to the
existing `"coordinates"` option. Its two extra inputs are the moduli of the
complete frozen G2 Gaussian coefficient columns, plain and contrast. They are
not voxel intensities. For contrast c and Gaussian i, the cached feature is
`log1p(abs(z0[i,c]) / max(median(abs(z0[:,c])), 1e-12))`. Each median comes from
the entire coefficient column before chunking; no GT, ROI, rendered image,
external weights, learned normalization, or external prior enters this feature.
The even-sized-column median follows `torch.median` (the lower middle value).
Both the `[N,2]` features and two medians are detached registered buffers and are
serialized in the field state. Empty clouds retain empty features and zero
medians; zero medians use the fixed epsilon. Paired mode explicitly rejects
nonfinite coordinates, coefficients, coefficient moduli, or normalized features.
The coordinate-only constructor retains its original input behavior.

The original coordinate network is constructed with the same layers, order,
random draws, and zero final layer. A bias-free `Linear(2,width)` projection adds
to its first preactivation. Its weights are initialized directly to zero without
consuming random numbers. With width64, this is exactly128 new trainable weights.
All original shared network weights and RNG state match coordinate-only at the
same seed. At zero projection, the hidden values, initial densities and common
parameter gradients match. Both anchor columns condition the shared gain field;
the two per-point phase columns remain independent.

The projection belongs to the actual residual-amplitude optimizer group. It can
initially have zero gradients because the final MLP layer starts at zero; it
receives effective gradients once that final layer becomes nonzero. Gain bound
0.25, Fourier bands4, width64, LR0.001, Adam/scheduler, phase, geometry, original
G2 coefficient buffers, priors, physics, normalization, training measurements and
500-update formal budget are unchanged. `runtime.py` and `evaluate.py` are byte
identical to revision `000c53d`. Neither external source nor data is modified.

The cached features cost N*2*element_size bytes: at3.6M points in float32,
28,800,000 bytes (27.47MiB). Metadata reports the actual size and medians. The
constructor also temporarily computes moduli and median workspace; no GPU peak
memory has been measured yet. Each checkpoint call receives only its matching
feature chunk, so saved MLP/Fourier activations remain proportional to
`chunk_size`, in addition to the fixed full feature cache and required output.

## Public parameter and schema contract

`run_gpu.py` requires `--feature-mode coordinates|paired_anchor`. The exact value
is included in `PARAMETERS`, recipe equality, launch fields, metrics and
provenance top-level fields, completed-gate parameters, smoke matching, and field
metadata validation. Missing/unknown flags, missing recipe keys, and mismatched
child metadata are refused. Baseline and explicit modes require `coordinates`;
they cannot silently accept a paired flag. A paired smoke/trial requires a real
shared+paired zero gate with matching architecture/hyperparameters; a trial also
requires a matching shared+paired two-step smoke. Old parameterless gates cannot
stand in for the new parameterization.

The numerical checkpoint schema and reported `model_schema` are
`g2_residual_amplitude_v2`. Field metadata uses `schema_version=2`, explicitly
records `feature_mode`, and includes anchor normalization, ordered columns,
medians, epsilon, frozen status, cache shape/bytes, and projection semantics.
Coordinate mode keeps the old tensor state keys; paired mode adds
`anchor_features`, `anchor_medians`, and `anchor_projection.weight`. A paired
state must be loaded into a paired constructor; it is not an old-v1 checkpoint.

A future approved sequence would invoke the public supervisor with
`--stage zero --mode shared --feature-mode paired_anchor --steps 0`, followed by
`--stage smoke --mode shared --feature-mode paired_anchor --steps 2`, then a
500-update trial. Each stage still needs freshly bound reviewed source hashes,
frozen-input hashes and an exact selected recipe. These are command shapes,
not executable authorization or evidence that any stage passed.

## CPU acceptance record

Starting revision: `000c53d9743183c7d5f365690bc4f83d9cf34273`; starting diff empty.
Interpreter and complete-suite command:

```sh
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest -q -p no:cacheprovider
```

The unmodified baseline produced104 passed in11.21s. New model tests were first
observed failing because `feature_mode` was absent (12 failures), and runner/CLI
tests likewise failed before their implementations. The unchanged frozen
coordinate module at revision `000c53d` is executed as a numerical oracle for
initial state, forward values and common gradients. Other cases cover zero
projection hidden identity and RNG parity, both global medians, effective
conditioning from both contrasts, independent phase, detached private buffers,
projection gradient/optimizer updates, chunk memory and gradient equivalence,
state roundtrip, empty/zero/nonfinite input, public flags, missing/wrong recipe
and child fields, and incompatible zero/smoke evidence. The one original queue
command-construction fixture now supplies `feature_mode="coordinates"`; the
queue script is unchanged and has not been run.

Final complete-suite result:134 passed in11.18s; `git diff --check` exited0.
`runtime.py`, `evaluate.py` and `autoresearch_loop.py` retain their original bytes.
GPU arithmetic,
real AF16 export parity, numerical stability, convergence, whole-head metrics and
clinical/ROI metrics remain unverified for this candidate. CPU success is not a
quality improvement or a passed numerical stage.
