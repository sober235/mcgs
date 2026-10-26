# Fixed-subset carrier MC-GS continuation

## Intent and scope

Continue the user-authorized official Karpathy autoresearch MRI adaptation. The bounded operator is independently reviewed and actually GPU-accepted (docs/carrier_operator_outcome_20261026.md). Now test one carrier continuation from original G2 against the existing full native control and keeper. This is adaptive single-case development, not an independent validation cohort. All technical numbers below are researcher choices under the existing autonomous authorization.

Retain existing data, AF16, historical legacy vdPoisMex mask,42 coils,CSM,original normalization and ten complete opponents. No new mask or sample remapping. Reference remains CS-SENSE AF5.33; reference-derived shared normalization is disclosed. No reference/ROI/opponent or external prior enters training/selection. PSNR/SSIM/NMSE only; all six whole-head inequalities,59 planes and enhancement regions remain reported. Current formalMRI32/historyrescore65; keeper lowreg score-0.11118128532897047. Advance only keeper+0.0001. An overall win still requires repeat, matched improved GS-single and visual inspection.

No historical source, old renderer binary, checkpoint, output, evaluator or prior numerical component is edited. New files only. Existing isolated worktree, new carrier-MRI branch. Retain all negative artifacts/workspaces; no merge/push/delete. User three-round review/fix cap applies, with no unresolved substantive blocker permitted into GPU stages.

## Frozen experiment

Route `g2_fixed_subset_carrier_v1`. Original shared3D Gaussian geometry and independent complex densities remain trainable through their exact original seven-group Adam restoration at4500. Original regularizers, clipping, flat-t1 frequency weights, exponential scheduler gamma0.9995, voxelizer and canonical hard-DC output remain intact; no CNN or relocation.

Select K=18000 unique Gaussian row IDs (0.5% of3.6M), using `torch.randperm(N,generator=privateCPUgenerator(seed42023),device='cpu')[:K].sort()`. Save actual int64 IDs and hash; preserve caller RNG and do not refresh selection. Tiny private CPU fixtures use a predeclared smallerK without a public override.

Two independent float32 eta[K,3] Parameters start exactly zero, held outside trainer.model. Frequency f=.25*tanh(eta), cycles/voxel,DHW; bound .25 is strictly below Nyquist. Carrier Adam has one group containing both Parameters, lr.001,betas(.9,.999),eps1e-8,weight_decay0,amsgradFalse, no schedule. Separate total carrier-gradient clip norm1.0; original seven-tensor clipping stays untouched. Step carrier once after original Adam/scheduler from the same backward; no extra loss or data forward.

Each enabled render is the unchanged original render plus accepted selected-row carrier_correction. Gather live geometry and freshly constructed complex density without detach. Separate density nodes keep original WorkObserver backward hooks attributable to the original voxelizer. Context persists through final canonical_control exports. Actual route is old render + new-M correction; conservative inverse-M support/symmetrization can differ from historical approximate bounds at pathological inputs. Disclose this; do not call it exact nonzero-frequency replacement of every historical atom.

Before each correction, conservatively bound pair work without allocating pair arrays. Derive M with accepted geometry, use lambda_min spherical radius3/sqrt(lambda_min)*shape/2 as an upper bound when the operator's reliability condition holds; otherwise all volume tiles. Apply identical floor/ceil/clamp and tile8 counting. Refuse upper total>2000000 before calling the operator. Record upper counts, fallback counts and maxima. Never drop atoms, shrink support or silently changeK. Allocation limits and observed full-scale cost are execution questions, not assumed from the microgate. GPU peak/time measured.

Budget: zero0, smoke2, formal500 original Gaussian updates from original G2. Enabled smoke6 correction forwards/4 actual backward returns; trial1002/1000. Original Gaussian counts remain6/4 or1002/1000; own Adam clocks2 or500, old clocks4502/5000. Export adds2 correction forwards. No initialization render/probe added to MRI work.

## Fixed artifact without compilation

Use existing artifact identity7e4074ab699211d0b99e28f0766c6c6b2f12ad9f3057105fe6993faa6d95e719, binarySHA b8d08c0aeead3c4c6d2e9f0fd803929c4ea5252bdd6de692053da6ced4c4c95d, modulecarrier_cuda_7e4074ab699211d0b99e28f0, from carrier_builds/20261023T011403Z-bc86c383e901. Its provenance and accepted microgate are immutable descriptors. OldbinarySHA3973daeef5f55ebfed931c38538a37d83921a9287b96b69a431da238057194b5 remains bound.

New carrier_artifact.py performs pure existing-file load using importlib extension loader, never cpp_extension, Ninja, compiler or subprocess. Before MRI load it verifies frozen manifest/binary/source/identity digests and active GSTMI Torch/CUDA/Python/CXX11 ABI compatibility. Actual visible device capability must exactly8.0; fail before reading MRI on incompatible allocation. It loads the exact module path/name, validates forward/backward entry points, and installs operator._extension/_provenance under its existing lock in a scoped context, refusing conflicting active state. Restore prior state in finally. Do not edit carrier_operator.py. Runtime observation may temporarily delegate native methods through a narrow proxy recording successful forward/backward returns; preserve native provenance. No backend fallback. Build-tool environment is irrelevant to a pure binary load; active runtime ABI/device are required and recorded.

## Reuse boundaries

Follow docs/carrier_integration_map_20261023.md. New carrier_runtime.py composes the real original run_continuation/control helper, scoped render/train adapters and observer role. Keep original restoration return unchanged. Original loop's start/pre_step/end callbacks retain all original calls; carrier zeroing, separate clipping and post-original-step update fit those boundaries. No training-loop copy or shared-module global mutation.

New run_carrier.py is stdlib-only, reuses existing run_gpu supervisor and old common artifact validation through private fixed function bindings. Existing lock/ACK/reaping/timeout/source-input proofs stay mandatory. New carrier_runner.py privately adapts old clinical stage/execute/publication, loads exact artifact before data, adds actual carrier files and metadata. Do not rewrite saved restoration/snapshot evidence. Distinguish external carrier mode from the internally ordinary Gaussian helper.

For common protocol compatibility keep fixed requested_k18000/event_step1 fields where inherited helpers require them; event_step is unused legacy transport and no relocation event executes. Public CLI modes control/carrier only, no numeric/backend/source overrides. Route policy is the sole new invariant identity; common invariant fields remain identical so old native receipt can project route/policy only. Code closure includes unchanged original218 sources, any executed caseprior receipt source, all new runtime/runner/protocol/artifact sources, accepted operator/CUDA sources and exact new binary. No mutable directory recursion.

## State, evidence and export

Carrier state/Adam are never registered under original trainer.model. Separate schema carrier_continuation_v1 includes selection,policy,initial/final state,trace,work,extension,bounds,checkpoint/readback,paired manifest,export state. Per-step trace records actual finite pre/post clip gradient norms, Adam clocks and frequency bounds. Original counters and seven-state snapshots keep their exact schemas. Correction counters use actual native forward/backward returns with inherited observer role/iteration/channel; output hooks alone do not claim a completed backward. GPU asynchronous execution remains subject to real finite-gradient observations.

Save separate carrier checkpoint containing eta,IDs,ownAdam/options/clocks,policy and source/binary identity. Safe CPU weights_only readback compares every saved tensor/moment/option to actual state. Paired manifest binds original Gaussian checkpoint and carrier at same absolute iteration; incomplete pair cannot publish success. Export snapshots require unchanged carrier state and actual two correction export calls. Raw/canonical/candidate descriptors remain original protocol, candidate==canonical for both scans. No public arbitrary resume path; every experiment starts from fixedG2, and checkpoint consumers refuse missing/mismatched component/iteration/policy.

Stage meanings: zero modecontrol executes two zero-frequency carrier export calls and no updates, proving actual raw and canonical G2 parity (exact equality where deterministic, relativeL2<=1e-5 recorded). Ordinary control smoke/trial disables correction and carrier updates entirely; it provides new native parity proof. Enabled carrier smoke/trial constructs fresh selected state and trains it. All start freshG2; never warm from smoke.

## Acceptance sequence

1. New CPU fixtures exercise real original tiny trainer/restore/loop/physics/export with a private CPU operator injection (accepted double oracle cast back, no public backend switch). Assert callback order, live five-input gradients, independent eta updates, original seven-state restoration/identities, separate counters, unchanged RNG/IDs, resource refusal before allocation, export persistence, paired readback and corrupted/missing evidence refusal. Include pure artifact load refusal/no-build and runtime compatibility checks. No old606 suite repeat.
2. Independent task and whole-route reviews, source-bound focused checks, actual invariant snapshot and selected recipes. Source/CPU approval is not an MRI pass.
3. Genuine source-bound G2 zero, disabled ordinary2, enabled carrier2. Enabled smoke must show finite nonzero carrier gradients/changed eta in both scans, exact separate clocks, accepted resource bounds and actual safe checkpoints. If any gate fails preserve evidence and fix/review before formal.
4. New native receipt from actual old native500 + actual new ordinary smoke, enforcing existing1e-5 parity/closure/score/evaluator bindings; otherwise new control500. Reuse is not a new experiment.
5. One carrier500 from originalG2, immutable CPU scorer after true completion. Training adds1 only at real completion, scoring adds0. Record all six values/59planes/ROI/cost and keep/discard; preserve negative outputs. Further optimization choices require their own recorded hypothesis.

## File scope

Task1 creates carrier_artifact.py,carrier_runtime.py,tests/test_carrier_runtime.py,tests/test_carrier_artifact.py. Task2 creates run_carrier.py,carrier_runner.py,tests/test_carrier_protocol.py,tests/test_carrier_runner.py and docs/carrier_mri_commands.md. Controller owns design/plan/reviews/state/recipes/ledgers. Old files unchanged. Internal helper names remain builder choices; freeze public cross-task interfaces in the plan. No deployment, new dependency or GPU provisioning required.

Implementation organization: controller-owned carrier_mri_contract.json freezes policy and artifact descriptors once; all new modules read it rather than duplicating numbers. It is included in the source closure. Task1 artifact context is entered by Task2 child before the old clinical data loader; runtime assumes that validated context is active. Test-only CPU injection remains private.
