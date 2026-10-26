Actionable finding count: **0**. There are no actionable findings or unresolved blockers in this scoped formal-outcome audit.

**Integrity verdict: PASS for completion evidence, scoring/reporting and the discard decision.** The actual learned500 run completed its declared Gaussian and CNN work. Its minimum-six whole-head margin is below both the matched native500 and the retained keeper. This conclusion is specific to this observed candidate; it establishes neither a general failure of case-internal priors nor a successful MRI method.

Scope: `outputs/caseprior_learned500_v1`, `scores/caseprior_learned500_v1`, `docs/caseprior_outcome.md`, their prerequisite bindings and the new JSONL/TSV/state/progress records. The earlier source review, 577-pass CPU suite, actual controls audit and native-reuse audit were already complete and were not repeated. The proposed512-update-per-fit candidate is outside this audit and has no implementation/run approval from this report. Only this report was written; concurrent files and all numerical artifacts were preserved.

| Observed item | Result |
| --- | --- |
| Gaussian updates | 500 |
| Gaussian forwards / VJPs | 1002 / 1000 |
| Events / point count | 0 / 3,600,000 |
| Final seven Adam clocks | 5000 |
| Scheduler epoch / count | 5000 / 5001 |
| Fresh CNN calls / total CNN updates | 10 / 1280 |
| Refresh iterations per scan | 4501, 4601, 4701, 4801, 4901 |
| Training / child-stage seconds | 591.790155 / 779.362691 |
| Supervisor wall seconds | 1036.414173 |
| Peak allocated / reserved GiB | 36.015682 / 42.615234 |
| Whole-head minimum-six margin | -0.12526398563890848 |
| Difference from native / keeper | -0.008078399840877826 / -0.01408270030993801 |
| Jointly passing planes | 0/59 |
| Pre-only / post-only passing planes | 0/59 / 2/59 |

I independently verified the actual completion SHA256 `592a3b531ebd430a41a2281884970306f8e6f9e33ec7a486289aa04a88a73153` and metrics SHA256 `576a33062b5f2677aa46e18ee97548f058769b04e276af515ac47ebd56c4c5c9`. Launch, completion, metrics and identical provenance bind the same reviewed221-source closure and frozen inputs. The exact selected trial/learned/500 recipe and prior policy agree. The verified post-run proof has equal before/after source-input maps and177 equal transitive prerequisite bindings. The actual zero, two smokes and native receipt hashes agree with the previously audited gates. Started/ready/ACK/completion records agree on child1035401, parent and device; completion records supervised exit0. The strict original CUDA runtime and no-autocast evidence are present.

The actual restore record equals the continuation's initial snapshot. All seven Parameter/storage identities and Adam state identities persist through the500 updates, retaining3.6M rows. Final Adam clocks, scheduler state and saved config agree. The original Gaussian checkpoint remains at `training/fixed_count_relocation/checkpoints/latest.pth`; its bound SHA256 is `eb8ae930fc481410d3919e4f118cb54b48b63141ca458d13709a67379dbb3838`. Child readback evidence binds that file to iteration5000, every final model/Adam hash and all optimizer groups. I checked this descriptor linkage and the saved state metadata, without reloading the large Gaussian checkpoint.

All500 actual trace rows were checked for consecutive iterations4501–5000, all seven finite gradient observations, exact Adam/scheduler clocks and paired flat-t1 frequency calls. Ordered render records account for1000 training forwards/VJPs and two export forwards with no event. The42-coil physics counters and final frequency observations agree with the declared work. The original RED is active on all500 steps; weighted/per-scan losses and cumulative refresh clocks agree with the final summary. The minimum and maximum weighted prior losses are `1.2588387221512676e-08` and `6.982866761973128e-05`. The actual final training log reaches5000 and reports completion.

All ten CNN records have their ordered index and seed42+1009*j, trained status, exact width32/patch32/batch4/tile32/lr0.001 settings and128 updates. Each reports128 finite nonzero-gradient steps, all ten parameter tensors changed and differing initial/final hashes. Initial hashes are distinct across the ten fresh fits. Each record matches the original pre/post history at its actual refresh iteration, including sigma, crop and target-change observations. The actual checkpoint/manifest/target file hashes and child semantic-readback descriptors match. Streamed target payload hashes agree with record output hashes and NPY shape/dtype metadata. Every observed per-call CPU and visible CUDA0 RNG before/after hash is equal, with preserved flags. No new uncertainty justified repeating the previous independent checkpoint-loading exercise: this audit checked the new artifacts and their reviewed-child readback evidence without additional CNN training, inference or Torch loading.

I independently streamed and hashed all six actual raw/canonical/candidate reconstruction files. Their descriptors match completion artifacts; candidate bytes equal canonical bytes, and raw hashes remain distinct. Exact scored candidate hashes are:

- Pre: `eacaa1cb742c3e34356b9c86f09af19c1a941cb4841c4f6f53ff39baec8819a2`.
- Post: `58ec39ad27ddaa0cd31591b89b1a66722894138a1d2d3cc5501ef6f88ac7ce1b`.

Those same descriptors appear in the objective, scoring input manifest and enhancement inputs. The actual objective SHA256 is `3d3479acf1b87e0b873afe3c1c30d2da612f243a7a44b3047323c95a82faff5a`; enhancement SHA256 is `699339e11f6566a2ed5fcbd4f1ad34d5c6b581b568c11f3e1de95fa28f0fd652`. CPU scoring's saved runtime and all-inputs-unchanged observation are present. Its whole-head/slice contracts and every noncandidate scoring-input hash equal the already audited native score's frozen bindings. The evaluator-source hash agrees with the reviewed closure. Root reported the scoring process's observed exit0; the resulting completed objective and all mandatory reports are present.

I recomputed all six saved whole-head margins and strict comparisons from their scalar metrics and all ten opponent values. They reproduce the reported score; all six best values belong to ZS-SSDU and no candidate condition wins. The report's candidate/native whole-head numbers and whole_head.csv agree with the actual objective to the displayed precision. Against native500, all three pre metrics worsen; post PSNR/NMSE improve slightly while post SSIM worsens. Neither result approaches the keeper+0.0001 advancement threshold. The unchanged keeper remains `lowreg_f075_solve_v1`, score `-0.11118128532897047`.

All59 saved plane records were checked for matching frozen axis/index/core counts, six condition comparisons, ten opponents, margins and strict flags. Their aggregates reproduce0 jointly passing planes, pre0/post2, and best-plane minimum-six score `-0.03773608292644295`. Objective plane summaries agree with `lesion_planes.json`. This checks saved arithmetic and completeness; it does not recompute image metrics from volumes.

The enhancement report includes lesion core,0–3mm rim, white matter, cortex, vessels, plexus and background. Formula, ROI and reference descriptors match scoring inputs; region/background definitions, voxel counts and all reference/opponent rows are identical to the prior native report. Every saved region error equals its candidate-minus-reference value. The report's seven displayed candidate/reference/native values agree to their printed precision. It correctly states independent per-scan white-matter normalization for enhancement, post-magnitude background units and scoring-only use of reference/ROI. Enhancement is not folded into the primary keeper decision.

All ten reported fit-diagnostic rows agree with the actual call records/history. The report explicitly says first/last losses use different random patches and fresh noise and cannot demonstrate strict convergence or MRI quality. Several small target changes, including the final post call's `4.599451494868845e-05`, are honestly reported. The final post loss is slightly higher than the first; the table preserves this. The report describes insufficient fitting only as a hypothesis worth checking and does not claim it caused pre-scan degradation. Its512-update proposal is clearly a future, separate bounded hypothesis requiring new bindings and controls, with no promised gain or equal-compute claim.

The reported training/stage/memory costs match metrics, and supervisor time matches completion. The report correctly explains that supervisor timing starts after the initial input snapshot/gate preflight and is not total command elapsed time. CNN work is included in training time and counted separately from Gaussian updates.

Accounting is consistent: exactly one `training_completed` entry adds one formal MRI experiment, raising30 to31; its later scoring entry adds zero and records discard. JSONL descriptors agree with actual files. The TSV contains one discarded formal candidate row at the correct rounded score. State retains the same keeper, strict success false, no active GPU/scoring, formalcount31, historical rescorings65 and the actual latest completion/scoring records. The earlier progress tail initially still described a running trial; root appended the actual final completion/scoring/count31 entry during this audit, resolving that bookkeeping observation before handoff. Historical entries remain preserved.

Both focused audit scripts exited0. The run-evidence script independently hashed59 selected files and checked all500 trace rows, ten call artifacts and six reconstruction outputs. The scoring/report script checked all six whole-head conditions, all59 plane records, enhancement arithmetic, displayed values, diagnostics and accounting. No source review, test suite, public validator, GPU work, acquisition/reference input rehash, MRI volume loading, rerender or numerical scoring was repeated. Existing source/input proofs and reviewed-child semantic evidence remain the trust boundary for runtime observations and large checkpoint contents. I did not independently reproduce CNN outputs, live RNG snapshots, image-level metrics, ROI calculations or process reaping. No repeat/cross-case/matched-improved-GS-single experiment was performed or claimed.

The discard decision is supported and all negative evidence is preserved. The completed formal candidate can be archived with this scoped audit; any next hypothesis requires its own separately reviewed and gated evidence.
