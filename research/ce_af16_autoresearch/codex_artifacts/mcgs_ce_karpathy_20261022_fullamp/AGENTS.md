# MC-GS Karpathy autoresearch campaign

This is an isolated project adaptation of https://github.com/karpathy/autoresearch at 228791fb499afffb54b46200aca536f79142f117. The user explicitly authorized execution on 2026-10-19.

Preserve all original source trees, datasets, CSM, masks, references, completed experiments and their metadata. Write new code and outputs only here. Workers are not alone: preserve each other's edits and honor assigned file ownership.

Use the existing GSTMI Python and prebuilt CUDA backend. Never rebuild or silently substitute the renderer. Before selecting a GPU inspect current utilization, memory and compute processes; respect scheduler/device visibility, prefer idle compatible devices and retain a claim until the owned child has exited. Leave other workloads running.

Keep the AF16 acquisition and all ten complete comparators fixed. Only PSNR, SSIM and NMSE decide quality; exclude HFEN. Training may use existing acquired measurements and G2 initialization, never scoring references, ROI masks or oracle-fitted parameters. Retain and disclose the original shared reference-based normalization constants; this campaign does not create a new strict GT-free benchmark.

Before launching a numerical stage, bind reviewed source hashes, fresh frozen-input hashes and the exact selected recipe through validate_stage_gates(). A path/hash handoff descriptor must be verified and dereferenced before use. Refused setup and infrastructure failures are distinct from scientific rejections. Keep versioned evidence; do not fabricate passed gates.

Primary success: both scans' whole-head PSNR/SSIM/NMSE strictly exceed all ten complete frozen opponents. Also report all 59 lesion-containing planes under the unchanged six-condition rule and six ROI/background enhancement measurements. A provisional win requires an independent repeat and a matched GS-single improvement before attributing an advantage to joint modeling.

Keep a baseline-first results.tsv and detailed JSONL ledger of every attempted candidate, including failures. Keep/discard applies to this campaign only; never reset external repositories or remove existing work. Preserve runnable candidate revisions and measured compute cost. Do not call one-case adaptive development an independent test-set result.

October22 full-amplitude representation phase: read the new spec in docs/superpowers/specs. This phase may load only the frozen G2 model, original Gaussian code/binary and checksum-bound G2 raw teachers; it does not load acquisitions, CSM, scoring references or ROI arrays. Representation admission is not MRI quality and contributes zero formal MRI experiments. A full amplitude function must not retain a per-point modulus residual or train phases to bypass the function during warm fitting. Preserve all inherited numerical sources and actual previous gates; they are not gates for this new route. The user caps automatic review/fix cycles at three; never ship an unresolved blocker. Keep diagnostic evidence and plan ledgers for continuation.
