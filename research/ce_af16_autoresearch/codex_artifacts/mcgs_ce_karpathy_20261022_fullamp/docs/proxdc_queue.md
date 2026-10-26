# One-candidate proxDC queue

`proxdc_loop.py` executes exactly the declared `g2_raw_proxdc_v1` route: tiny identity control, real G2 canonical-export control, one fixed raw-anchor proximal correction, then the unchanged frozen PSNR/SSIM/NMSE scorer. It uses the shared `run_proxdc.py` supervisor for GPU choice, claim ownership, observed context, timeout and numerical-child cleanup. The queue itself uses only the Python standard library and never imports Torch or chooses a device.

After independent review and selection of the actual three recipes, launch from this worktree:

```sh
python3 -B -u proxdc_loop.py --gpu 0 \
  --review-gate verification/proxdc_review_v1.json \
  --tiny-recipe recipes/g2_proxdc_tiny_v1.json \
  --canonical-recipe recipes/g2_proxdc_canonical_v1.json \
  --prox-recipe recipes/g2_proxdc_candidate_v1.json
```

The paths shown are also the defaults. The queue changes its working directory to its own worktree, holds `.proxdc_queue.lock`, and waits for its owned supervisors/scorer. It has no daemon, scheduler, automatic retries, parameter search or indefinite idea loop. A user or root can run this command through an owned background session. Terminating the queue forwards cancellation through the reused owned-process helper; that helper lets the real numerical supervisor clean up its own child. It does not signal unrelated jobs.

The fixed experiment names are `g2_proxdc_tiny_v1`, `g2_proxdc_canonical_v1` and `g2_proxdc_candidate_v1`. Each exact numerical command is constructed with route `g2_raw_proxdc_v1`, CG budget32, rtol1e-5, lambda ratio0.1 and timeout3600. The queue parses that command with the real public `run_proxdc.parser()` and calls the complete `validate_stage_gates()` before launching **or reusing** a stage. Completed reuse also invokes `validate_observed_gate()` and requires matching parameters, directory and exact current control descriptors. Old amplitude/H1 gates are not substituted.

`loop_state.json` and each journal record bind the queue hash, reused `autoresearch_loop.py` helper hash, actual `run_proxdc.code_hashes()`, fresh frozen-input snapshot, actual review descriptor and all selected recipe descriptors. The complete binding is checked again before every stage and after scoring. A different family/root, changed source/control binding, altered completed evidence or foreign journal is refused. Incomplete run/score folders and stopped failure/cancellation states are retained and refused. No failed numerical execution is automatically repeated.

The queue reuses the existing helper implementations for atomic state, canonical JSONL/TSV journal, owned subprocess cancellation, scoring, scored-candidate hash verification, historical-source verification and measured keep/discard. It never calls the old amplitude loop's `execute()` or `main()`. The unchanged helper's root is this same worktree. `autoresearch.jsonl` is authoritative; `results.tsv` remains its five-column projection: commit, whole-head margin, memory GiB, status, description.

The scientific anchor is immutable: `lowreg_f075_solve_v1`, score−0.11118128532897047, metrics source SHA256 `8ceeeee1f0d44aff6c080f572714be9382fc7f4789c6f052510bc1f32bd26805`; minimum keep improvement is0.0001. The source descriptor is reverified through the existing `load_prior()` helper, and a saved weaker prior is refused. Mutable `campaign_state.json` status cannot lower this anchor. The existing main G2 baseline objective is reused only as labeled context, SHA256 `55fcbfc170e27d8d1be6c6b0013743037228f237baf25ccc6032da42edbcc0c2`. A baseline-context TSV row does not represent a newly run baseline stage.

Only the actual prox candidate is scored. Its score is checked against completed candidate-array hashes and the full frozen score input manifest by the existing helper. Tiny/canonical are controls, not new training or quality experiments. The prox stage is **one formal output correction with zero optimizer updates**. Its `stage_complete` record contributes `formal_output_corrections: 1`; the later `scored` record contributes0 and names that same correction ID, so summing the counter does not double-count it. Actual per-scan solver records and aggregate `work_total` are copied from completion-validated numerical metrics, including CG work and separate endpoint/objective/RHS/construction/export work. Numerical elapsed time is labeled `numerical_seconds`, even though the inherited metrics file calls it `training_seconds`.

A valid measured score may produce `discard`, `keep`, or `provisional_win`. Keep means improvement over the stronger historical anchor by more than the fixed threshold. It does not mean all comparisons were won. Whole-head success requires both scans and all three metrics against all ten complete opponents. Even a measured strict win leaves `success: false` and requests independent repeat plus matched improved GS-single before attribution. Numerical controls never establish that target. Otherwise the queue stops at `candidate_complete_goal_unmet`, retaining the result for root's next decision.

Setup refusals, process/scoring failures and cancellation are recorded separately from scientific rejection. They preserve the outputs/logs and require root diagnosis before another unique candidate/version is prepared. A foreign or stale saved queue is refused without relabeling its state/journal. A completed queue can be rechecked and reused idempotently; it does not duplicate records, rerun the candidate, or silently launch another idea.

CPU tests use explicitly synthetic temporary fixtures. They run the actual public parser, review/recipe stage validator, journal, objective formula and candidate-hash checks; only numerical processes, clinical scoring output and observed-child artifacts are replaced. They never create real passing clinical gates, launch GPUs, write selected production recipes, or claim MRI-quality observations. The production stage validation and complete numerical lifecycle remain covered by the inherited route tests. No actual queue or GPU launch has been performed during implementation.
