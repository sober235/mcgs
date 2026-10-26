# Fixed-subset Carrier MRI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute one fixed carrier500 MC-GS candidate with truthful controls, separate carrier state and unchanged benchmark evaluation.

**Architecture:** New runtime composes the original continuation loop and render; original Gaussian states remain exactly seven. A pure artifact context loads the accepted binary without build tools. New stdlib protocol/child adapters reuse the existing supervisor and evaluator. Numeric choices and artifact descriptors live in controller-owned carrier_mri_contract.json.

**Tech Stack:** Existing GSTMI Torch2.7.1+cu118,accepted sm80 carrier binary,original tile_cuda,pytest; no dependency changes.

**Spec:** docs/superpowers/specs/2026-10-26-carrier-mri-design.md

## Global Constraints

- New files only; do not modify historical sources/evaluator/data/mask/CSM/checkpoints/results,or accepted carrier_operator/reference/CUDA sources/binary.
- Fixed originalG2 at4500,N3600000,seven original Adam groups/scheduler gamma0.9995,original losses/frequency/clip/export. No CNN/relocation/reference/ROI/external prior in training.
- K18000,private CPUrandperm seed42023,sorted unique rows; independent eta[K,3] per scan,f=.25*tanh(eta); Adam lr.001,betas(.9,.999),eps1e-8,decay0,AMSGradFalse,constantLR,carrier clipnorm1.0.
- Fixed zero0/smoke2/trial500. Zero enables two zero-frequency export calls; controlsmoke/trial disables correction/updates; carrier mode enables training. Old counters/schema untouched, separate carrier_continuation_v1 evidence.
- Guard conservative upper pair total<=2000000 before pair allocation; never drop rows/support. Binary b8d08c0aeead3c4c6d2e9f0fd803929c4ea5252bdd6de692053da6ced4c4c95d,sm80 and runtime contract exact; pure load before MRI, no compiler/Ninja/cpp_extension.
- Only PSNR/SSIM/NMSE decide; fixed ten opponents/59planes/ROI;32 completed MRI/65historical rescoring at start. Keep rule keeper+0.0001, currentkeeper-.11118128532897047. No quality inference from control/software pass.
- User3-round review/fix cap; independent source approval before actual GPU; preserve all failed evidence/workspaces. Workers do not stage/commit/spawn agents; root owns Git/ledger/recipes/state/launches. Use focused checks only, no old606 or passed operator-suite repetition.

## Review Focus

- Removing render adapter before canonical export loses carrier: Task1 actual tiny-trainer export and before/after state checks.
- Reused density node can corrupt old Gaussian VJP counts: Task1 tests independent native/correction work and shared leaf gradient accumulation.
- Carrier steps/clip or ambient default device can perturb old restore/RNG: Task1 real restore,callback order,identity,clock and explicitCPU selection tests.
- Missing component or incompatible binary can silently zero carrier: Task1 paired readback/artifact refusal,Task2 child-before-data ordering tests.
- Forged source/control/partial evidence can authorize wrong route: Task2 protocol corruption fixtures and genuine stage prerequisites/native reuse tests.

### Task 1: Pure artifact and original-loop carrier runtime

**Files:** Create carrier_artifact.py,carrier_runtime.py,tests/test_carrier_artifact.py,tests/test_carrier_runtime.py. Read controller-owned carrier_mri_contract.json; never edit it without a ruling.

**Interfaces:**
- `carrier_artifact.verified_carrier_extension(device)` context manager -> provenance dict. Stdlib-only module import; Torch import is lazy. Check contract descriptors/identity/sources/runtime/capability, import exact native module, install accepted operator._extension/_provenance under its lock, restore in finally. Refuse conflicting state; no build tools or subprocess. It is entered by Task2 before MRI data load.
- `carrier_runtime.run_carrier_continuation(trainer, frequency, directory, *, mode, steps, rotation_matrix_fn, requested_k=18000)` -> original run_continuation result dict plus `carrier` evidence. Mode control/carrier,zero control special meaning. Uses active verified artifact. Private helper may accept explicit tinyCPU correction/K injection for tests only, never automatic publicCPU fallback.
- `carrier_runtime.validate_carrier_checkpoint(path, *, expected_iteration, expected_policy, expected_selection_sha256, expected_binary_sha256)` -> validated CPU state/readback dict; lazy Torch safeweights_only. Reject schema/iteration/policy/selection/binary/shape/clock corruption. Runtime actual readback additionally compares every tensor/moment/option with live state.
- Separate carrier evidence sections exactnames: schema,enabled,policy,selection,initial_state,final_state,trace,work,extension,bounds,checkpoint,readback,paired_checkpoint_manifest,export_state. Worker documents precise field types in report for Task2 validator.

- [ ] Write failing focused tests covering pure-load/no-tool refusal and compatibility; real original tiny trainer restoration/loop/physics/export with explicit privateCPU correction; zero/control/carrier paths,shared geometry gradients,independent carrier updates,both Adam blocks,callback order,RNG/IDs,work counts,resource refusal,adapter unwind,export persistence and paired checkpoint corruption. Configure tiny fixture before real restore; never rewrite its evidence.
- [ ] Run actual RED with `env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest -q -p no:cacheprovider tests/test_carrier_artifact.py tests/test_carrier_runtime.py`.
- [ ] Implement only the two modules. Compose original continuation/control helper with instance-scoped render/train and private WorkObserver binding; preserve exact old function logic. Proxy native forward/backward after successful return for separate counters, retaining observer role/channel/iteration. Artifact context remains external; runtime proxy restores in finally. Resource upper-bound guard computes counts from actual geometry/eigenvalue sphere bounds before operator pair allocation. Carrier state stays external to original model/optimizer.
- [ ] Run focused GREEN once at final state,save actualexit/stdout/stderr and sourcebefore/after hashes; self-review; original source/binary closure remains unchanged. No GPU/MRI/operator rerun.
- [ ] Root commits owned files and requests fresh task review with exact BASE..HEAD package. Resolve substantive findings within3 rounds, then freeze producer evidence schema for Task2.

### Task 2: Source-bound protocol and clinical child

**Files:** Create run_carrier.py,carrier_runner.py,tests/test_carrier_protocol.py,tests/test_carrier_runner.py,docs/carrier_mri_commands.md. Task1 files unchanged unless a reviewed interface defect is ruled. Root owns contract/spec/state.

**Interfaces:**
- `run_carrier.py` exposes standard supervisor protocol: ROOT,OLD,ROUTE,CHILD_RUNNER,CONTEXT_ACK,POLICY,code_hashes(),snapshot_invariants(),parameters(args),validate_parameters(args),validate_stage_gates(args,hashes,invariants),validate_child_artifacts(args,directory,hashes,invariants),validate_observed_gate(...),validate_native_control_receipt(...),parser(),main(argv=None). Preserve inherited run_gpu call contract; import/help stdlib-only.
- `carrier_runner.py` exposes validate_launch(path),clinical_stage(launch,directory),publish_result(...),execute(path),main(argv=None) with only --launch. Enter Task1 verified artifact before original clinical loader, keep context through continuation/export. Fixed FunctionType private adapters preserve defaults and globals; no shared-module mutation/bytecode/source rewrite.

- [ ] Write focused RED for parent stdlib import/help/CLI binding,exact closure/contract/artifact hashes,strict carrier evidence consumers,all stage gates/native receipt,corrupt/missing components,ACK/restore/source/array checks,child artifact-before-data ordering,real private tiny-trainer producer->validator. Any test-only data proof is explicitly refused by public production validators.
- [ ] Implement minimal protocol/child using original supervisor/common validators and unchanged clinical stage. Extend rather than weaken actual evidence. Native receipt reuses unchanged old218 closure and fresh actual ordinary smoke,never an enabled smoke. Code closure includes contract,all executed new sources,reused receipt source,accepted operator/CUDA sources and frozen .so. New policy stays in route/policy invariant fields for strict old-common-input projection.
- [ ] Run focused exact command `env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest -q -p no:cacheprovider tests/test_carrier_protocol.py tests/test_carrier_runner.py`; then one final entire new-route suite including Task1 because interfaces are consumed. Freeze hashes/capture actualexit; no old suites or GPU.
- [ ] Root commits and requests combined Task2+whole-route fresh review; source/CPU review is not an MRI pass. Address blockers within3 rounds.
- [ ] Root snapshots actual invariants and selected recipes,checks available GPU/visibility and sm80 compatibility,then genuine zero/control2/carrier2 in unique directories through unchanged serialized supervisor. Record fresh resource snapshot and all completion hashes. Carrier2 must show both frequency tensors changed,finite gradients,actual separate forward/backward/optimizer clocks,resource and checkpoint proofs.
- [ ] Validate actual new native receipt or run fallbackcontrol500. Only then one carrier500 freshG2. On genuine completion add1 formal count; run unchanged CPU evaluator in unique directory,add0 for scoring. Compare keeper/native/ten opponents/all59/ROI/cost,write actual keep/discard and independent outcome audit. Continue research under standing authorization; do not stop at queued training or mere software acceptance.

## Self-review

Spec math/choices are frozen in contract; Task1 provides exactruntime APIs/evidence schema to Task2. Loader executes before MRI in child, not inside late continuation. Original seven-state proof remains intact; carrier evidence external. All five ReviewFocus risks have explicit owner/checks. CPU private injection cannot authorize publicMRI gates. Build tool invocation is excluded from clinical path. New clinical source review and actualcontrols remain mandatory despite accepted operator. Existing isolated worktree/new branch and standing autonomous authorization cover execution; no routine approval question is needed. Fresh builder check returned nothing important left.
