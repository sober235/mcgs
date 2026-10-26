0 actionable findings. No verified unresolved defect was found in the new one-candidate queue or its reviewed helper/interface wiring.

Independent review of the complete new `proxdc_loop.py`,
`tests/test_proxdc_loop.py` and `docs/proxdc_queue.md`, with the reused journal,
process ownership/cancellation, scoring, historical-prior and public numerical
route interfaces. Numerical code remains the clean reviewed revision `f1a16f5`;
the mathematical implementation was not changed or reopened unnecessarily.
A second read-only reviewer independently checked process ownership,
interruption recovery, foreign-state preservation and completed-stage reuse.
Reviewers authored no implementation, gate, recipe, state, score or journal.
This reviewer added only this report. The manually running real tiny control
was not operated or used as synthetic evidence.

The queue follows exactly one declared sequence: route `g2_raw_proxdc_v1`,
tiny control, canonical G2 export control, fixed raw-anchor prox correction,
then the actual frozen scorer. It does not call the old amplitude queue's
`execute` or `main`. Commands use the real public parser and bind max32 CG
matvecs, rtol1e-5, lambda ratio0.1 and timeout3600. The full route stage validator
runs before either launching or reusing each stage. Reuse additionally validates
the observed completed artifact set and exact parameters, directory and current
control bindings. An in-progress manual tiny directory is refused until it has
a genuine fully validated completion; the queue does not invent its completion
or launch a duplicate into that directory.

The queue and reused helper import without Torch. The numerical supervisor
continues to select/reserve the GPU and own its numerical child; the queue owns
that supervisor and the separate CPU scorer. Queue source hash, unchanged helper
hash, complete numerical source/input snapshot, review descriptor, all three
selected recipe descriptors, root/family, historical prior and baseline context
are pinned in state/journal. Source/input/control bindings are checked before
each stage and after scoring. Completed evidence remains subject to fresh full
validation. Foreign/stale state or journal bindings and failed/cancelled saved
states are refused without relabeling them. Incomplete numerical or score
directories remain preserved and stop the queue without an automatic retry.

The quality anchor is the stronger historical `lowreg_f075_solve_v1`, not G2:
margin `-0.11118128532897047`, fixed minimum keep improvement `0.0001`, source
metrics SHA256
`8ceeeee1f0d44aff6c080f572714be9382fc7f4789c6f052510bc1f32bd26805`.
`load_prior` verifies the source descriptor, and equality to the immutable queue
prior rejects a weakened saved value. An additional read-only metadata check
verified this descriptor and recomputed the margin from its six saved `head_*`
values against the existing frozen best opponents; it matched exactly.
No clinical array or full-volume score was loaded/recomputed for that check.

The actual main G2 objective descriptor is separately verified at SHA256
`55fcbfc170e27d8d1be6c6b0013743037228f237baf25ccc6032da42edbcc0c2`.
Its score `-0.11778703065863405` is reused only as baseline context, with zero new
stages/training. It is not used as the promotion anchor. The existing measured
minimum-six PSNR/SSIM/NMSE rules are reused without weakening the comparisons or
reintroducing HFEN. Numerical control completion cannot create a quality win.

Only the final prox candidate is scored. The reused `score_run` checks the
objective's run/name, completed candidate-array hashes, complete score-input
manifest and reviewed evaluator bytes, then hashes all six mandatory score
reports. Resume rechecks those same artifacts and rejects changed scientific
evidence. Journal IDs and prefix-only TSV projection prevent duplicate results.
The prox `stage_complete` row counts one formal output correction; the scored
row links that correction ID and contributes zero additional corrections.
All optimizer-update counts remain zero. Per-scan solver work and aggregate
cost are copied from completion-validated metrics, and elapsed numerical work
is labeled `numerical_seconds` despite the inherited file's `training_seconds`
field name. A measured `keep` remains distinct from strict success; a strict
win leaves `success=false` pending independent repeat and matched GS-single.

Observed independent CPU checks:

```sh
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest -q -p no:cacheprovider tests/test_proxdc_loop.py
# 19 passed in 1.83s

env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest -q -p no:cacheprovider
# 183 passed in 15.87s

git diff --check f1a16f5
# exit 0
```

The second reviewer's queue/helper subset passed **36 tests in1.95s**. An
additional actual CPU process probe, operating only on a synthetic temporary
queue, sent SIGTERM while its owned supervisor managed a separate numerical
stand-in: queue exit143/status cancelled, cleanup confirmed without forced
termination, both owned processes reaped, unrelated process still running,
and failure marked `scientific_rejection=false`. Two synthetic abrupt-stop
recovery probes also passed: after the tiny ledger commit but before its state
publication, only canonical/prox/scorer remained to run; after the scored ledger
commit but before final-state publication, no process reran. Both ended with
the same five unique journal records. None of these probes ran the actual queue
against its production controls or touched the real tiny process.

Tests execute the actual public parser, complete review/recipe stage validator,
durable journal, decision formulas and candidate-hash checks. Numerical process
execution, clinical scoring output and observed-child artifacts are replaced by
explicit synthetic fixtures confined to temporary directories. Real numerical
artifact/ACK/work validation retains its separately reviewed test coverage.

Final reviewed SHA256 values:

| File | SHA256 |
| --- | --- |
| proxdc_loop.py | 2554821310308add59ef7552322785fd7190f5b2e255d8849ee2f421d1fe476c |
| tests/test_proxdc_loop.py | 52e9741b6e5e21c93c5f10bbc7c209c707f22c06367d44345eb477ed65b43f17 |
| docs/proxdc_queue.md | 906eceeaeb7b2cf298f8c92e0c4506ee747ad4e4d2f04802e16416a35329064e |
| autoresearch_loop.py | d3a992826032b794feb34fbc1cfe2eb7ebe7ad48089a1d541fee63221f415bf2 |
| run_gpu.py | 983688ada3c2c3e37e4dcd0fe7cb64dbbc4dff587f302ed16fd99bad58136ac3 |
| run_proxdc.py | 894f43a120029b16a66f65900b1593959d675b85e6ef7003cdd790046aab0106 |
| proxdc_runner.py | 68a1b36c36ca29f62aec3b64371a0933e1b3f66edaae77471ddce1a055618124 |
| runtime.py | a42fb02828910c4ba3526439444b8170b4a4a2767ebae2c6a932c09ea26efd10 |
| evaluate.py | 9a2c38d1942d6d6ad2331f3364dfad5be86a03cb316fec96f27deb7c5e40a875 |
| runner.py | fb5d2e3097e37c6bec84a3d5a5bae6e0452d1f37bb7835acc32be86f7ad91af4 |
| amplitude_model.py | 8d4faf7d936539ea3be551a421f1e9eeb680df2d8490939949014dd7c0bfb769 |

The numerical/helper files were compared byte-for-byte with `f1a16f5` and are
unchanged. The three new file hashes match the author's frozen handoff.

This is a clean CPU queue review. No actual queue, GPU reconstruction, clinical
scoring operation or passed scientific stage was performed by the reviewers.
The real tiny/canonical/prox stages and final PSNR/SSIM/NMSE, all59-plane and
ROI/background outcomes still require their own genuine completed evidence.
