# Task 1 implementation report

Status: implementation complete for controller commit and fresh independent review. No GPU/MRI acceptance is claimed. No files were staged or committed. No original source, operator/reference/CUDA file, native binary, historical output, MRI data, or checkpoint was edited. The only numerical execution used synthetic five-row CPU fixtures through the real original restoration, trainer loop, FFT physics, and canonical export.

Owned production files: `carrier_artifact.py`, `carrier_runtime.py`. Owned tests: `tests/test_carrier_artifact.py`, `tests/test_carrier_runtime.py`.

## Verification and source freeze

The exact focused command was:

```
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest -q -p no:cacheprovider tests/test_carrier_artifact.py tests/test_carrier_runtime.py
```

Every invocation used this command with separate redirected stdout/stderr files. `task1-command-evidence.json` records each exact redirected command, observed exit, stdout/stderr descriptors and final pytest summary. All stderr files are empty. Original RED was actual execution before either production module existed: exit 1, 26 failed in 1.43 s. Final GREEN: exit 0, **30 passed in 11.52 s**. Intermediate failing/passing development runs are retained, including fixture binding errors, descriptor mismatch diagnosis, and later deliberate new-test REDs; none are overwritten.

`task1-final-before.json` and `task1-final-after.json` are identical. The frozen controller map `preserved-source-before.json` contains 230 explicit source/binary paths; all 230 hashes were recomputed before and after final verification and all match. This is a cheap source/binary check, without reading or rehashing MRI data. The four owned files and controller contract hashes are unchanged across final GREEN. All four owned files parsed with `ast.parse` and passed trailing-whitespace checks. `git diff --check` exited 0, but the owned files are still untracked, so the separate owned-file syntax/whitespace check is the relevant additional check.

Focused coverage includes stdlib-only loader import, immutable file/hash refusal before runtime, every frozen runtime/ABI/sm80 field, installed context/conflict handling/unwind, explicit private CPU injection, public CPU refusal, exact private sorted selection with preserved caller RNG, zero/control/carrier modes, actual original seven-state restoration and IDs, original 2- and 500-update clocks, original and separate native work counts, finite shared-geometry gradients, independent asymmetric per-scan updates and Adam moments, original-Adam/scheduler-before-carrier ordering, no second loss/backward, allocation refusal before correction pair construction, native backward failure unwind, export persistence and saved-state correction reconstruction, paired checkpoint readback and schema/iteration/policy/selection/binary/shape/clock/nonfinite/missing-component corruption.

The tiny fixture's data naming and asymmetric synthetic checkpoint densities are configured **before** the actual original restore. The actual returned restore record is captured and compared unchanged; no evidence is rewritten. All original old tests/operator suites were left alone and were not rerun.

## Implementation and self-review

`carrier_artifact.verified_carrier_extension(device)` performs descriptor/source/identity/build-manifest verification using stdlib code, then lazy Torch runtime checks, and imports the exact extension with `importlib` without subprocesses, compiler, Ninja or cpp_extension calls. It requires the frozen active Torch/CUDA/Python/CXX11 ABI/config and capability 8.0. It rejects conflicting installed state under the accepted operator lock, validates native path/name/forward/backward, installs `_extension/_provenance`, and restores previous state in `finally`. `active_provenance()` is the runtime's explicit proof of an active verified context. Task2 must enter the context before the original clinical data loader. Private loader seams exist only for CPU tests; no public backend/build override exists.

`carrier_runtime.run_carrier_continuation(...)` requires the active verified CUDA context, the fixed public requested K and full G2 row count. The private `_run_carrier_continuation(...,_cpu_extension=...,_test_k=...)` accepts only CPU fixtures with N<=32 and explicit positive K. There is no automatic CPU fallback. Carrier eta parameters remain outside the original model, seven-group optimizer and scheduler. Selection uses the contract's private CPU generator and seed, with actual saved int64 IDs and raw-byte hash. Independent eta starts exactly zero, with one separate two-parameter Adam group and constant schedule.

A private `FunctionType` binding substitutes a derived `WorkObserver` only into the original continuation function, called in control mode. Instance-scoped render/train adapters preserve whether each binding was already instance-owned and unwind on success/failure. The original loop is not copied. Original start/pre_step/end callbacks retain their order; carrier zeroing follows original start, separate clipping precedes original pre_step observation, and carrier Adam follows actual original Adam plus scheduler and precedes original end callback. The original seven-tensor clipping and all original work/state/counter schemas remain untouched.

Each corrected render calls the original render and uses a **new** complex density node for the correction. Live selected positions/log-scales/quaternions/density/frequency remain attached to autograd. The resource check computes accepted actual-M geometry/eigenvalues and conservative spherical bounds with identical floor/ceil/clamp/tile counting before accepted operator pair allocation. Finite unreliable matrices use full-volume bounds. Excess total pairs fail instead of altering support/K. The route remains old render plus actual-M correction, with its documented support difference from the historical approximate bounding box.

The narrow native delegate increments separate forward/backward counters only after the native method successfully returns. Each backward is attributed to its actual saved forward row and original observer role/channel/iteration. The CPU oracle exercises this same custom-autograd/native delegate boundary with the accepted double geometry oracle cast back to native dtypes. An output hook is not used as backward evidence.

The render adapter spans the original helper's final canonical export. Separate state is snapshotted before the first export and after both exports, checked equal. The zero stage records actual raw parity from original raw versus raw plus native zero-frequency correction, without an extra render. The existing clinical canonical-to-G2 parity remains Task2's preserved old check. New checkpoint and pair files use exclusive creation; safe CPU weights-only load checks complete structure, both states/moments/clocks/full Adam options, frozen policy/source/binary identity and the complete same-iteration Gaussian pair. Actual readback is recursively compared with the complete live-derived saved state, including every tensor and option.

Self-review found and resolved: instance-owned frequency render functions do not necessarily have `__func__` (test capture corrected); historical descriptors contain exactly path/sha256 (new descriptors aligned); public checkpoint consumers must reject tiny-N/K state (private fixture validator separated). No unresolved substantive finding was identified in this self-review. Controller owns the required fresh-context independent review; this report does not substitute for it.

## Frozen cross-task producer API and evidence schema

Public signatures are exactly the brief:

```
verified_carrier_extension(device) -> context manager yielding provenance dict
run_carrier_continuation(trainer, frequency, directory, *, mode, steps,
                         rotation_matrix_fn, requested_k=18000) -> original result + carrier
validate_carrier_checkpoint(path, *, expected_iteration, expected_policy,
                            expected_selection_sha256, expected_binary_sha256) -> CPU state dict
```

Public checkpoint validation requires full frozen N/K. Tests exclusively use `_validate_carrier_checkpoint(...,_test_geometry=(5,2))`; this private setting is not an accepted clinical/public override. The public reader requires the adjacent `paired_carrier_checkpoint.json`, hashes both components, and safe-loads the Gaussian absolute iteration; missing pairs/components fail.

All file descriptors below have **exactly** `{path: str (absolute), sha256: str}`. A tensor record `T` has `{shape: list[int], dtype: str, sha256: str, finite: bool, zero: bool}`; sha256 covers contiguous CPU raw tensor bytes. State record `S` has:

- `eta`: pre/post mapping to `T` plus `tensor_id:int`, `device:str`.
- `adam_steps`: pre/post mapping to actual integer local clocks (0 with uninitialized Adam).
- `moments`: pre/post mapping to empty dict initially, or step/exp_avg/exp_avg_sq mapping to `T`.
- `frequency`: pre/post mapping to `{min:float,max:float,max_abs:float}`.
- `independent_parameters:bool` (true), `optimizer_options:list[dict]` (actual PyTorch param groups; Python betas tuple becomes a JSON array).

The separate `result['carrier']` sections are:

| Section | Produced value/types and validation meaning |
|---|---|
| `schema` | String `carrier_continuation_v1`. |
| `enabled` | Bool: true for zero stage and carrier training, false for nonzero control. |
| `policy` | Exact `contract['policy']['carrier']` dict, including full frozen K regardless of private fixture size. |
| `selection` | `k,n,seed,min,max:int`; `method,dtype,sha256,rng_before_sha256,rng_after_sha256:str`; `sorted_unique:bool`; `file:descriptor`. IDs use torch.int64; sha256 is the raw int64 byte hash, distinct from the .pt file hash. RNG hashes cover actual CPU generator state bytes before/after selection. Actual selection source iteration comes from the common contract; Task2 should associate its already verified source-checkpoint descriptor without rereading/rehashing the source checkpoint. |
| `initial_state`, `final_state` | `S` above. |
| `trace` | List of actual update rows: `iteration,step:int`; `gradient_norms:{pre:float,post:float}`; `pre_clip_norm,post_clip_norm:float`; `finite:bool`; `before,after:S`; `changed_parameters:{pre:bool,post:bool}`. Empty for zero/disabled control. |
| `work` | `forwards,backwards,optimizer_steps,scheduler_steps:int`; `roles:{initialization,training,event,export:{forwards:int,backwards:int}}`; `calls:list[{role:str,iteration:int,channel:str ('plain'/'contrast'),backwards:int}]`. Native-return counts, not inferred counts. Scheduler steps always 0. Carrier2 has 6/4 native returns and local clock2; carrier500 has1002/1000 and clock500; zero has2/0 and clock0; disabled control has0/0 and clock0. |
| `extension` | Public actual accepted provenance dict, including original `module,identity_sha256,identity,binary,binary_sha256,build_ninja_sha256,effective_build_ninja`; plus `runtime` equal to frozen runtime, `load_kind:'existing_binary_only'`, `manifest:descriptor`, `accepted_microgate:descriptor`. Private CPU fixture has explicit `test_only:true` and `module:'private_cpu_test_oracle'`, and must never pass a clinical validator. |
| `bounds` | `policy:str ('actual_M_lambda_min_sphere_or_full_volume')`; `limit,tile_size,max_upper_pairs,fallback_atoms_total:int`; `calls:list[{role:str,iteration:int,channel:str,upper_pairs:int,max_pairs_per_atom:int,reliable_atoms:int,fallback_atoms:int}]`. One actual row per successful preallocation guard, including two export rows when enabled. |
| `checkpoint` | Descriptor of `carrier_checkpoint.pt`. |
| `readback` | `weights_only:true,map_location:'cpu',live_state_equal:true,iteration:int,trace_file:descriptor` for `carrier_trace.jsonl`. Every saved tensor/moment/option was compared. |
| `paired_checkpoint_manifest` | Descriptor of `paired_carrier_checkpoint.json`. Its JSON is `{schema:'carrier_checkpoint_pair_v1',iteration:int,gaussian:descriptor,carrier:descriptor,policy:dict,selection_sha256:str,binary_sha256:str}`. Gaussian descriptor is exactly the original `final_checkpoint_evidence.file`. |
| `export_state` | `before:S,after:S,unchanged:true,calls:int,zero_raw_parity:dict`. Enabled calls exactly2; disabled calls0. Zero parity is pre/post `{exact_equal:bool,relative_l2:float}` only at steps0; otherwise `{}`. Both parity entries derive from the same original export render and actual correction. Existing canonical/candidate array closure remains Task2's original export writer and validator. |

Additional produced files to include in Task2 artifact closure: `carrier_selection.pt`, `carrier_trace.jsonl`, `carrier_checkpoint.pt`, `paired_carrier_checkpoint.json`. Original files remain unchanged.

Carrier checkpoint safe-loaded dict has exact keys: `schema,iteration,policy,n,ids,eta,optimizer,local_updates,enabled,binary_sha256,identity_sha256,gaussian,selection_sha256,schedule`. `ids` is CPU int64[K]; eta maps pre/post to CPU float32[K,3]; optimizer is the actual CPU-copy PyTorch state dict; schedule string is `constant`; identity SHA binds the accepted source/build identity. Local updates equal absolute iteration minus4500 when enabled, otherwise0. Public expected absolute iterations are4500,4502,5000. Each initialized Adam state has float32 scalar step and float32[K,3] exp_avg/exp_avg_sq; both clocks must match actual local updates. Untrained state must remain zero with empty Adam state. A separate pair validator must not treat file presence as completion: Task2 still requires successful original child/publication evidence.

## Remaining coverage boundaries

No actual native .so import, CUDA launch, clinical data/checkpoint load, reference/ROI/opponent read, reconstruction score, GPU cost, or MRI parity was attempted. The loader compatibility/context tests intentionally inject fake runtime and native import seams while hashing the real immutable artifact/source descriptors. CPU proof does not establish numerical native acceptance, resource behavior during clinical updates, or quality improvement. Existing accepted microgate and subsequent independent source review/G2 gates remain controller-owned prerequisites.
