# Karpathy autoresearch adapted to CE AF16 MC-GS

Upstream workflow: https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/program.md.

The user authorized starting this campaign on2026-10-19. Use branch autoresearch/ce-af16-20261019 and the existing GSTMI environment. Preserve all prior experiments and external repositories. Follow docs/design.md, docs/plan.md and AGENTS.md.

## Objective and immutable benchmark

Optimize whole_head_min_six_normalized_margin (higher is better). All six whole-head PSNR/SSIM/NMSE conditions must strictly exceed every frozen opponent for an overall win. Also report the unchanged59-plane six-condition evaluation and sixROI/background enhancement; neither substitutes for whole-head success. HFEN is excluded.

Freeze the existing AF16 acquired data, CSM, mask, normalization, references, ten complete opponents and metric helpers. The canonical G2 source is adamflat_g2_v1 at iteration4500. Reconstruct and score it before any new scientific trial. References and ROI enter only the separate CPU scoring process, never training.

## Editable scope and first hypothesis

Editable sources are amplitude_model.py, runtime.py, runner.py, run_gpu.py, evaluate.py and their tests in this fresh campaign. Evaluator changes are limited to faithfully routing the existing frozen formulas; changing evaluation to improve a result is forbidden.

First compare a residual shared amplitude field plus independent point phases with a matched explicit gain/phase control. Base coefficients and shared geometry remain fixed for this first family. Both start exactly at G2, use fresh declared optimizer state and the original losses, flat frequency weights, CUDA renderer and hard-DC export. No external pretrained prior is used.

## Execution and ledger

Run CPU checks and independent review, then actual zero-step reproduction and two-step CUDA smokes. Each matched formal candidate has500 updates; a200-update pilot is explicitly labeled when used. Record wall time and peak memory. MRI's declared update budget is a project adaptation; do not claim the upstream five-minute budget or equal compute across algorithms.

run_gpu.py is the public numerical launcher. Its exact selected recipe, code hashes and fresh invariant snapshot must pass validate_stage_gates before a child starts. The child uses the original prebuilt renderer; no dependency installation or fallback compilation. Each numerical stage has a3600-second timeout and owns only its child process group.

For every completed candidate: run the separate CPU evaluate.py, record the whole-head score and all metrics, append results.tsv and autoresearch.jsonl, then keep an improvement>0.0001 or record discard. Keep code revisions and failed outputs; never reset another project. Setup refusals and crashes are recorded separately from scientific rejection. All candidate results remain visible.

Continue generating the next bounded hypothesis from measured failures while preserving the benchmark. Do not stop to ask whether to continue. A provisional overall win triggers a repeat and a matched GS-single variant, followed by reconstructed-image/lesion/error visual checks before claiming superiority. A result here is one-case adaptive development, not independent population validation.

## October 22 full-amplitude representation route

Continue under the [frozen representation design](docs/superpowers/specs/2026-10-22-full-amplitude-representation-design.md)
and [public commands/evidence protocol](docs/representation_commands.md).
`run_representation.py` is the separate public entry point for G2 raw control,
then fixed 2000/128-update MLP and grid amplitude fitting. It preserves inherited
numerical sources and gates, uses only the G2 model/original raw renderer and
checksum-bound G2 raw teachers, and adds zero formal MRI experiments. A completed
finite negative fit remains recorded with admission false. Representation
admission requires both final relative complex L2 errors <= .01; MRI likelihood
training and PSNR/SSIM/NMSE evaluation require a separately reviewed continuation.
Use [representation progress](docs/representation_progress.md) for current status.
