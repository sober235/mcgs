Actionable findings: **0**. There are no actionable findings or unresolved blockers in this scoped audit of the two completed and scored formal relocation arms.

**Verdict: PASS for outcome integrity, reporting and both discard decisions.** Both arms completed their actual 500-update budgets. Neither exceeds the retained whole-head keeper. This is an incremental actual-output audit, not a repeat of the source review or a claim that the broader MRI quality goal was achieved.

Scope: `outputs/g2_relocation_control500_v1`, `outputs/g2_relocation_event500_v1`, their corresponding `scores` directories, `docs/relocation_outcome.md`, `results.tsv`, `autoresearch.jsonl` and `relocation_state.json`. I recorded the existing working changes before inspection and left them untouched. All 218 executing source hashes still match the reviewed `80de16d` implementation and `verification/relocation_source_snapshot_pre_review_v1.json`.

| Observed result | Ordinary continuation | Relocation continuation |
| --- | ---: | ---: |
| Actual updates | 500 | 500 |
| Events | 0 | 1 at absolute iteration 4501 |
| Gaussian forwards / VJPs | 1002 / 1000 | 1004 / 1000 |
| Final seven Adam steps | 5000 | 5000 |
| Scheduler epoch / count | 5000 / 5001 | 5000 / 5001 |
| Whole-head minimum-six margin | -0.11718558579803065 | -0.11718772593014594 |
| Difference from keeper | -0.006004300469060184 | -0.006006440601175472 |
| Jointly passing planes | 0/59 | 0/59 |
| Plain-only / enhanced-only passing planes | 0/59 / 1/59 | 0/59 / 2/59 |
| Training seconds | 560.4427822856233 | 571.8787029827945 |
| Supervisor wall seconds | 918.0391070842743 | 932.1443426609039 |
| Peak allocated GiB | 35.8495397567749 | 35.84984493255615 |

For both arms, launch, completion, metrics and provenance bind the same reviewed sources and frozen input snapshot. Each exact selected recipe agrees with the completed route/mode/budget. The post-run proof is verified, its before/after source-input snapshots are equal, and prerequisite-control bindings remain equal and hash-valid. The actual child-started, context-ready, observed ACK and completion records agree on the owned PID, parent and device, and completion records supervised exit 0. I independently checked the bound hashes of launch, metrics/provenance, ACK/start records, trace/events, scheduler, configuration and progress files.

Each trace contains exactly 500 consecutive steps at iterations 4501–5000. Every trace row has all seven finite gradient observations, the expected Adam and scheduler clocks, and the exact two-scan frequency call sequence. The full ordered renderer call records, FFT role counters, 1000 paired loss calls, optimizer steps and scheduler steps match the declared work. Initial parameter and full Adam tensor hashes agree between arms, and their first losses match exactly. This does not assert bitwise-identical CUDA gradients or trajectories.

The relocation event record validates against the reviewed event schema and numeric checks: 18,000 disjoint pairs, 36,000 changed rows, 3,564,000 untouched rows and fixed N=3,600,000. Indices and index digests, live model/optimizer/scheduler/parameter/moment identities, untouched-row hashes, changed-row zero hashes, and unchanged event clocks/state are consistent. The event occurs after the first ordinary update at 4501; its two extra forwards are acquired-residual evidence renders. There is no extra event or point growth.

Both actual final checkpoint paths exist at the trainer's `training/fixed_count_relocation/checkpoints/latest.pth`. Their file descriptors agree across final-checkpoint evidence, artifact manifests and completed artifact sets: control SHA256 `ccf26e778ed52fcc1b82c21aebb1293a491979bc222c47c5d4c8e26feadd9c06`; event SHA256 `747f056b502a98857e3885e0797378f26de0cd52e5f9388ff0fd182b512cad03`. The existing producer's live-versus-saved evidence binds iteration 5000, every model/Adam tensor hash and all group options to the final state; the separately saved scheduler agrees with that state. No named uncertainty justified loading these large checkpoints again, so this audit checked their linkage rather than repeating the prior producer's tensor comparison.

I independently hashed the four actual scored candidate arrays and matched them against completion artifacts, child manifests, canonical-export hashes, objective candidate hashes, scoring input manifests and enhancement inputs:

| Array | SHA256 |
| --- | --- |
| Control pre | `045a6e45ed0deef899fb3f6dee6d84a807cb6ccde330dbdcee6d206929cb4e19` |
| Control post | `b6948fdb7860578d3a51174521110fee14ac6e8ff1bc0046aaa40009cff45695` |
| Event pre | `77ff31db3112dacd052f3b80d1b963b81fde3478d237794aa9a97d86c2781d3a` |
| Event post | `824af18c991c5ea6551c6907ca475a733eb5f60bfde230333bfedd35e6e93be5` |

Both objectives identify CPU scoring and unchanged scoring inputs. I independently recomputed the six whole-head margins from saved numeric metrics, checked each best value against all ten opponents, and checked strict inequality decisions. All six best opponent values are from ZS-SSDU; neither candidate wins a whole-head condition. The minimum remains pre NMSE. All 59 saved plane records were checked for six conditions, ten opponents, strict comparisons and the aggregate paired/per-scan pass counts. The objective's plane summary agrees with `lesion_planes.json`.

The unchanged keeper is `lowreg_f075_solve_v1`, margin `-0.11118128532897047`, with required advancement greater than 0.0001. Both candidates improve G2's margin slightly, by 0.0006014448606034006 and 0.000599304728488112, but remain below that keeper. Relocation minus ordinary continuation is only `-0.000002140132115288629`; corresponding plain/enhanced PSNR differences are `-0.00000929317469910984` and `-0.0006119389896994676` dB. The report correctly avoids interpreting these single-run differences as statistical degradation, and the enhanced-only 1-to-2-plane change is not promoted into a joint or whole-head win.

All displayed candidate whole-head and enhancement numbers in `docs/relocation_outcome.md` agree with the saved results to their printed precision. Enhancement includes every declared region—lesion core, 0–3 mm rim, white matter, cortex, vessels and plexus—plus background. Candidate image inputs match the exact scored hashes. Reference/comparator enhancement rows are identical across arms; recorded per-region errors equal candidate-minus-reference values. Formula, ROI and reference hashes agree with the scoring input manifests. The report correctly states independent per-scan white-matter normalization for enhancement, post-magnitude background units, scoring-only reference/ROI use, the CS-SENSE reference limitation and the disclosed legacy mask exception. These auxiliary measurements are not mixed into the primary six-margin decision.

Ledger and state accounting are consistent. Each formal arm has exactly one `training_completed` record adding one MRI experiment, then one `scored` record adding zero. Controls also add zero: this phase contributes two, taking the prior 28 to **30** completed formal MRI training experiments. Historical rescorings remain 65; two representation diagnostics and the earlier output correction remain separate categories. The TSV contains the baseline plus the two discarded candidates; its rounded margins agree with the canonical JSON values. Objective/enhancement and completion/metrics descriptors in JSONL match actual files. State names no active trial or scoring, retains the same keeper, records strict success false and reproduces the last scored event-arm record.

Verification consisted of focused stdlib metadata, hash and arithmetic checks, all exiting 0. No GPU work, rerender, source rereview, full suite, full public validator repetition, raw MRI input rehash or new scoring run was performed. Image-level PSNR/SSIM/NMSE and ROI calculations were not independently rerun from the large volumes; their exact input linkage and saved arithmetic/reporting were checked. Final checkpoint contents rely on the bound existing live-versus-saved producer evidence as described above. This audit does not establish repeatability, statistical significance, cross-case generalization, or failure of every possible relocation strategy. It does not approve the next hypothesis's design or reuse these gates for another route.

Only this report was written. All numerical sources, completed outputs, ledgers, user reports, state files, index and concurrent work were preserved.
