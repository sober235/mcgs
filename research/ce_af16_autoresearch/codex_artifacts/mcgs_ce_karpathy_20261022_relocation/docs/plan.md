# CE AF16 MC-GS Implementation Plan

**Goal:** Execute a measured Karpathy-style experiment loop until the proposed MC-GS improves toward whole-head superiority on the fixed pre/post benchmark.

**Architecture:** A fresh isolated runner restores G2 model tensors, preserves source losses and CUDA renderer, and adds zero-initialized residual amplitude and phase parameters. A separate scorer reads saved outputs using the frozen evaluation. A supervisor records reviewed source/input/recipe evidence and owns GPU selection and the candidate ledger.

**Tech Stack:** Existing Python GSTMI, PyTorch2.7.1+cu118, NumPy/SciPy/skimage and the unchanged prebuilt CUDA voxelizer.

**Spec:** docs/design.md

## Global constraints

Preserve all external source/data/completed artifacts. AF16, masks, CSM, original normalization and ten complete opponents are fixed. Only PSNR/SSIM/NMSE decide quality. No GT/ROI/oracle supervision and no external pretrained prior. Retain all failures and actual resource costs.

## Review focus

Zero-step drift; accidental GT/ROI use during construction/optimization; phase and amplitude derivatives; stale or changed source/input/recipe evidence; GPU loading collisions and incomplete-run acceptance.

## Tasks

- [ ] Add residual-amplitude module and meaningful CPU tests: identity, gradients, channel/coordinate isolation, bounded gains, buffers, checkpoint round-trip.
- [ ] Add frozen three-metric scorer and tests. Consume fresh candidate_pre_complex.npy/candidate_post_complex.npy. Produce whole-head six margins, all59-plane S6, six ROI/background enhancement metrics and input hashes.
- [ ] Add numerical runner using existing source constructor/renderer/loss/export. Restore G2 explicitly, use frozen normalization constants, install the selected parameterization and save a new checkpoint schema. Produce provenance, trace, reconstructed arrays and runtime measurements.
- [ ] Add supervisor snapshot_invariants(), validate_stage_gates() and GPU claim/resource selection. Bind reviewed sources, baseline evidence and exact recipe. Record before/after proofs and all failures.
- [ ] Run CPU suites and fresh-context review, resolve findings, then execute actual zero-step and two-step controls.
- [ ] Record G2 baseline; execute matched candidate/control pilot and formal runs, score each, log keep/discard/crash, select the next hypothesis from observed evidence.
- [ ] On provisional overall success, repeat and run matched GS-single; deliver locally clickable reconstruction/zoom/error figures and the complete ledger.
