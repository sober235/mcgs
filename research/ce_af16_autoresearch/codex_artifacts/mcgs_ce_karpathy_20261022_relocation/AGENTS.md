# MC-GS Karpathy autoresearch campaign

This is an isolated project adaptation of https://github.com/karpathy/autoresearch at 228791fb499afffb54b46200aca536f79142f117. The user explicitly authorized execution on 2026-10-19.

Preserve all original source trees, datasets, CSM, masks, references, completed experiments and their metadata. Write new code and outputs only here. Workers are not alone: preserve each other's edits and honor assigned file ownership.

Use the existing GSTMI Python and prebuilt CUDA backend. Never rebuild or silently substitute the renderer. Before selecting a GPU inspect current utilization, memory and compute processes; respect scheduler/device visibility, prefer idle compatible devices and retain a claim until the owned child has exited. Leave other workloads running.

Keep the AF16 acquisition and all ten complete comparators fixed. Only PSNR, SSIM and NMSE decide quality; exclude HFEN. Training may use existing acquired measurements and G2 initialization, never scoring references, ROI masks or oracle-fitted parameters. Retain and disclose the original shared reference-based normalization constants; this campaign does not create a new strict GT-free benchmark.

Before launching a numerical stage, bind reviewed source hashes, fresh frozen-input hashes and the exact selected recipe through validate_stage_gates(). A path/hash handoff descriptor must be verified and dereferenced before use. Refused setup and infrastructure failures are distinct from scientific rejections. Keep versioned evidence; do not fabricate passed gates.

Primary success: both scans' whole-head PSNR/SSIM/NMSE strictly exceed all ten complete frozen opponents. Also report all 59 lesion-containing planes under the unchanged six-condition rule and six ROI/background enhancement measurements. A provisional win requires an independent repeat and a matched GS-single improvement before attributing an advantage to joint modeling.

Keep a baseline-first results.tsv and detailed JSONL ledger of every attempted candidate, including failures. Keep/discard applies to this campaign only; never reset external repositories or remove existing work. Preserve runnable candidate revisions and measured compute cost. Do not call one-case adaptive development an independent test-set result.

October22 relocation: keep exactly3.6M rows and all seven Parameter/moment objects. Restore G2 Adam/scheduler at4500 without reapplying geometryfactor2. Resolve the original unnamed optimizer groups by Parameter identity and canonical order; do not require or inject a name option. Relocate in place after a normal step; preserve untouched rows, group options, scalar clocks and scheduler, and reset only changed-row moments/gradients. Never chain the old prune/split/rebuild helpers for this fixed-count route. Scalar Adam age is not per-row age. Read the fixed-budget relocation spec; old gates cannot authorize a new route. Preserve all evidence and honor the user's three-round review/fix cap.

Scheduler observation must preserve its actual serializable state: ExponentialLR.state_dict includes instance attributes, so do not store wrapped step methods or other observer callables on the scheduler instance. Observe actual counters or use scoped observation that leaves the instance state unchanged; retain a regression exercising snapshots during a real update/event.

New MRI masks default to BART bart poisson with fully sampled central24x24 ACS and recorded command/version/seed/support/AF/checksum. This historical AF16 campaign explicitly keeps the task-locked existing mask SHA93a50c2a4bb299e110b3938f15761a0420434354b29f4894476f215f7abdd5f0: provenance is vdPoisMex, not BART, with historical75-point ACS completion and measuredAF16.000. Disclose that legacy exception; never relabel it BART or silently regenerate it. Gaussian-row relocation does not change k-space sampling points. Any new mask requires a new benchmark/result version with matching reconstruction and illustration inputs.

October22 case-only magnitude prior: `run_caseprior.py` is a separate source-bound
route; see docs/caseprior_commands.md. Preserve reviewed Task1 and all inherited
numerical/evaluation sources. Restore original G2 full seven-state continuation,
retain3.6M points and add no relocation. Control constructs no network; learned
mode alone uses original NonLocalRED0.1/sigma0.015/start4501/refresh100 with fresh
case-crop/noise CNN fits. Exact new-route review/recipe, actual zero and both
smokes precede trial500. Native500 reuse requires its unchanged old closure and
actual completion/scoring receipt plus new ordinary-smoke equivalence; otherwise
run a new versioned control500. Keep Task1 records unchanged and separate observed
caller-RNG/checkpoint-readback evidence. CPU fixtures authorize no scientific gate.
