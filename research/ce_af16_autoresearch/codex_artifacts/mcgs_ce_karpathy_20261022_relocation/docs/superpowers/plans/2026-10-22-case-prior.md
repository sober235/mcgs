# Case-internal magnitude prior implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test a case-only learned magnitude prior inside the original MC-GS Gaussian reconstruction, without changing acquisition, renderer or evaluation.

**Architecture:** A fresh patch CNN implements the existing NumPy denoiser callable. A scoped factory adapter supplies it to the unchanged NonLocalRED target/gradient implementation. A new parent/child route reuses strict G2 state restoration, the original training loop, real work observation and canonical export, with route-specific evidence and genuine controls.

**Tech Stack:** Existing GSTMI Torch2.7.1/cu118, NumPy, original CUDA binary; stdlib-only supervisor. No installation.

**Spec:** `docs/superpowers/specs/2026-10-22-case-prior-design.md`.

## Global constraints

- Existing data,42coils, AF16, historical mask/CSM, scales, ten opponents and three-metric evaluator are fixed; preserve all original and completed artifacts.
- Only case-rendered magnitude patches and synthetic noise train the CNN; no reference, ROI, opponent, external images or weights.
- Original G2 iteration4500,3.6M points, seven Parameter/Adam objects and complete state, gamma0.9995; no relocation, geometry-factor reapplication or optimizer reconstruction.
- Original losses/export/frequency/clipping/wavelet remain fixed; only learned mode adds NonLocalRED lambda0.1 to both scans, sigma_rel0.015,start4501,refresh100.
- Fresh CNN per invocation: five3x3x3 convolutions1->32->32->32->32->1, four ReLUs, residual-noise output, zero final convolution, float32, no BN/AMP, fresh Adam(lr0.001,betas0.9/0.999,eps1e-8),128 updates,batch4,32-cubed patches.
- Seed invocation j with42+1009*j, local crop/noise RNG, restore caller CPU/visible-CUDA RNG and do not change backend flags; detached independent scan targets.
- Normalize crop and sigma by max(P99(crop),1e-6); inference32-cubed interiors plus halo5 must match full zero-padded inference; clamp output magnitude at0.
- Only new route zero0/control,smoke2/control or learned,trial500/control or learned;3600s timeout, actual supervised ACK/claim/reaping and fresh input/source proofs.
- Actual new zero and both smokes precede trial. Reuse old native500 only via exact source/common-input/artifact/scoring receipt and new ordinary smoke equivalence; otherwise rerun a versioned control500.
- A formal learned trial has500 Gaussian updates,1002 Gaussian forwards,1000 VJPs,no events,ten fresh CNNs/1280 updates. Report actual time/memory, all six metrics,59 planes and enhancement/background.
- Advance only score>keeper+0.0001; overall success requires all six strict inequalities and later repeat/matched GS-single/visual checks. Never delete a discarded candidate.
- Preserve reviewed inherited numerical source files; new implementation files and focused tests only. Resolve substantive findings within three fix/review rounds before GPU.

## Review focus

- Original NonLocalRED calls under no_grad: genuine CNN optimization must reenable gradients, while source input/Gaussian graph remain detached and untouched; Task1 tests it directly and Task2 uses the real caller.
- Tile edges and odd shapes: tiled inference must agree with a full nonidentity network including global boundaries; Task1 tests nonsymmetric, multi-tile volumes.
- Global RNG and shared state: initialization/training must restore caller RNG; CNN parameters never join the seven-group Gaussian optimizer; both tasks test these boundaries.
- Evidence of a different route or missing training: parent must reject stale gates, forged refresh/work counts, no-op active networks and unbound checkpoint/target/manifest data; Task2 uses actual Task1 producer output in fixtures.
- Reused control mismatches: old route/source closure and policy remain distinct, common frozen data agree, and scored arrays dereference exactly; Task2 tests source/input/array/clock mismatches and proves no fabricated new-route control completion.

### Task 1: Current-case magnitude denoiser

**Files:** Create `case_denoiser.py`, `tests/test_case_denoiser.py`, and a concise interface report in `.superpowers/sdd/2026-10-22-case-prior/task-1-report.md`.

**Interfaces:** Produces `MagnitudeNoiseCNN(width=32)` and `CaseMagnitudeDenoiser(output_dir, *, device, seed=42, updates=128, patch_size=32, batch_size=4, width=32, lr=1e-3, tile_size=32)`, callable `(z,sigma)->float32 ndarray`, with `last_stats` and ordered `records:list[dict]`. Constructor validates settings; call validates input before mutation. Invocation j owns a unique `call_0000`-style directory and refuses an existing destination. A trained record uses schema `case_magnitude_denoiser_call_v1`, call_index,seed,status,input shape/dtype/hash,sigma,normalization,normalized_sigma,updates,patch/batch/width/lr,loss/gradient/change/time observations, initial/final parameter hashes, output hash/clipped count and checkpoint/target descriptors. The per-call `manifest.json` stores the record excluding its own descriptor; returned record adds `manifest:{path,sha256}`. Identity records are explicit and do not fabricate a trained checkpoint. Actual exact field schema is documented in the report and producer tests; Task2 consumes that frozen schema, not a guessed copy.

- [ ] Write failing tests for exact initial identity; active fitting under an outer `torch.no_grad()` on a simple synthetic signal; finite/nonzero parameter gradients and changed weights; separate fresh invocation models/seeds; immutable input and caller RNG; finite nonnegative detached output and truthful saved-state/target linkage.
- [ ] Write failing tests for full/tiled inference equality with nonzero weights, odd shapes, multiple interiors and global boundaries; test invalid/nonfinite/negative inputs and settings, existing-path refusal, zero sigma/input identity and no training/RNG changes.
- [ ] Run the focused tests with `env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest tests/test_case_denoiser.py -q`; preserve observed RED output.
- [ ] Implement only the specified model, patch fitting, RNG scope, tiled inference and artifact protocol. No MRI imports, GPU launch, new dependencies, reference loading or changes to inherited files. CPU fixtures may use smaller widths/patches/update counts; public production defaults remain fixed.
- [ ] Run the same focused suite to GREEN; inspect input/RNG/state and actual saved artifacts, not just return flags. Document exact schema, validation, source hashes and coverage limits. Root serializes commit and requests one focused independent review before Task2.

### Task 2: Original-trainer integration and public protocol

**Files:** Create `caseprior_runner.py`, `run_caseprior.py`, `tests/test_caseprior_route.py`, `tests/test_caseprior_protocol.py`, `docs/caseprior_commands.md`; modify only `AGENTS.md` and `program.md` for the new route binding. No edits to inherited numerical sources or evaluator. Report to `.superpowers/sdd/2026-10-22-case-prior/task-2-report.md`.

**Interfaces:** Consume the frozen Task1 callable/records schema; `runner.build_config/setup_trainer`, `relocation_runtime.restore_training_state/snapshot_training_state`, `relocation_runner.run_continuation/save_exports`, original NonLocalRED factory/history, and the existing `run_gpu` protocol interface. Produce route `g2_case_internal_magnitude_prior_v1` with exact stage/mode/steps policy. Child result includes the existing continuation evidence plus `nonlocal_summary` and `denoiser_calls`. Parent remains stdlib-only. Reuse old route validators' state/frequency/physics/NPY primitives and current shared supervisor where valid; do not weaken old validators or copy the training loop.

- [ ] Read actual Task1 producer records and the referenced original trainer/RED/save/supervisor interfaces. Confirm consumed names and file schemas in the report before tests; resolve a genuine interface mismatch rather than fabricating fixture fields.
- [ ] Write failing original-Torch-trainer tests for zero coefficient constructing no CNN, scoped factory restoration on success/failure, original no_grad refresh invoking the real Task1 callable, detached positive per-scan targets, nonzero original Gaussian gradients/prior loss, refresh order and actual saved checkpoint/counters/clocks. Use a tiny original trainer and small private CNN fixture, not a copied training loop.
- [ ] Write failing protocol tests from actual produced records for new zero and smokes, full500 clock/count validation, exact review/recipe/source/input/ACK/postproof bindings and artifact refusal. Test each listed review-focus mismatch, including forged learning work and stale/different-route controls.
- [ ] Specify and test a strict `native_control_receipt`: actual old native500 completion and source closure/common frozen inputs, before/after proof, artifact/checkpoint/clock evidence and scored candidate/objective descriptors; require new ordinary-smoke parity. Do not call the new route hash map an old-route closure or rehash the complete MRI inputs twice to compare policies.
- [ ] Run targeted tests with the existing CPU-only GSTMI command and preserve RED evidence; implement the adapter, child and route using the actual existing interfaces. Keep extra source closure coverage, original seven-group state/seed/frequency and all artifact/counter validation. Learned mode's summary must prove exact expected refreshes, two/ten fresh CNN calls and256/1280 actual CNN updates.
- [ ] Run focused tests to GREEN. Once files freeze, run exactly one complete CPU suite using the project's existing test command; preserve stdout/exit/source hashes before and after. Do not mint actual review, recipe, zero or smoke pass files from CPU fixtures.
- [ ] Document exact public commands, output roles, control-reuse criterion, scientific limitations and recovery by preserving a failed version. Root serializes commit, then dispatches one fresh independent whole-change review at the exact frozen diff, including Task1 and real producer-to-validator evidence. Fix substantive findings and rerun only affected checks plus broader checks justified by new changes.

## Numerical execution after review

- [ ] Root freezes fresh source/input hashes, clean review descriptor and exact selected recipes. Actual numerical gates must belong to this route and use the public CLI.
- [ ] Inspect available GPU memory/utilization/compute PIDs and use an idle compatible device, respecting visibility/allocations and foreign work.
- [ ] Run actual zero, ordinary smoke and learned smoke. Verify actual manifests/gradients/learning/clocks/arrays and control equivalence through the public parent. A CPU fixture cannot pass these stages.
- [ ] Establish verified native500 receipt or execute the fallback new control; run the single declared learned500 trial from fresh original G2, not smoke/control output.
- [ ] Use unchanged CPU `evaluate.py` in unique scoring directories, log all real attempts/results/costs, compare keeper and all ten, retain or discard and continue according to the scientific outcome. A provisional overall win triggers repeat and matched GS-single.

## Self-review and rulings

The spec's image, optimizer, data, learned-target, lifecycle, control, scoring and evidence requirements map to the two tasks and numerical steps above. Task2 waits for Task1's reviewed exact producer schema. New files have exclusive owners; root alone stages/commits/updates ledgers, and workers preserve concurrent changes.

Existing autonomous authorization selects execution without an approval pause; implementation follows task-by-task subagent development. Builder check returned “nothing important left.” Research parameters are declared choices, and the bounded measured trial can reject them. Work remains in the already isolated linked worktree on the new caseprior branch; no redundant worktree is created. The inherited401-pass result remains a prior baseline, not evidence that new code passes. No source/evaluator/data changes occur merely to improve a number.
