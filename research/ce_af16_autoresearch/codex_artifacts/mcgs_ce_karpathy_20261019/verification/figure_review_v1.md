# Independent review of the CPU comparison adapter

Actionable findings: **0**. No substantive correctness, compatibility, provenance, or requested-behavior defect was verified in this revision.

Reviewed files:

- `render_comparison.py`, SHA256 `4f862f2e0f5a2b3662d350426e41252fd101ff1971c1aad3756850b0f9a769b7`.
- `docs/render_comparison.md`, SHA256 `0ef61fedbdd7022247468a8c6b1b4b9e757cb3890225988b6c22e32bb6a8c5c2`.

The initial working tree contained only the untracked adapter, its documentation, and `figures/`. This review wrote only this report. Numerical sources, training runs, completed artifacts, scores, and existing figures were not modified. No GPU work was performed.

## Scope and source comparison

Read the complete adapter and documentation, the campaign `AGENTS.md`, the relevant complete `evaluate.py` scoring path, the pinned original renderer, the original orientation helpers, and the actual explicit trial's completion/objective/input-contract/plane/provenance records. There is no `.codegraph/` directory in this repository.

The adapter preserves the original twelve-column order, plane order and orientation, fixed lesion boxes, reference-derived common display windows, image/error halves, error arithmetic and range, typography, color map, and panel geometry. The hard-coded boxes and windows agree with the original renderer's saved provenance. The original orientation helper functions are extracted from the hash-verified source rather than importing its main program.

Only PSNR, SSIM, and NMSE are calculated or displayed. Displayed slice arithmetic matches the frozen scorer. Candidate naming is tied to the actual objective. Historical coronal129 remains separately labeled as a G2-selected example. The footer reports the actual saved target decision and validated counts, without an inherited claim of 59/59 wins, no new training, global superiority, or generalization. The reference is described as CS-SENSE AF5.33 rather than fully sampled ground truth.

## Independent checks

Executed the real `prepare()` path using GSTMI Python with `CUDA_VISIBLE_DEVICES=` and `OMP_NUM_THREADS=1`. It validated **42 input hashes**, including both candidate arrays, all twenty frozen comparator volumes, both references, source/manifest bindings, and the completion-bound launch and numerical metrics. The actual trial had **500 observed updates**. Candidate array hashes agree across the scoring objective, candidate descriptor, score file manifest, and completed artifact map.

Recomputed the raw strict comparisons and aggregate counts for all 59 frozen plane identities. Observed counts are:

- Both scans passing all three metrics: **0/59**.
- Plain scan passing all three metrics: **0/59**.
- Enhanced scan passing all three metrics: **1/59**.
- Per-condition counts: plain PSNR/SSIM/NMSE **0/0/0**, enhanced **3/1/3**.

Separately checked all six whole-head raw inequalities from the actual objective metrics and its opponent table. All six are false, consistent with the saved `strict_success: false` and the figure's **unmet** status. The observed whole-head metrics are plain **34.81066745868763 / 0.9415454268455505 / 0.009927871293368992**, enhanced **37.1603895396045 / 0.9541727304458618 / 0.009595347758660406** in PSNR/SSIM/NMSE order.

Ten isolated negative metadata checks passed without changing files:

1. Altered paired passing count was rejected.
2. Altered strict-win flag was rejected.
3. Altered frozen opponent plane score was rejected.
4. Altered frozen plane identity was rejected.
5. Missing opponent was rejected.
6. Candidate hash differing in the completed artifact map was rejected.
7. Objective run-directory mismatch was rejected.
8. Altered comparator volume descriptor was rejected.
9. A metrics record claiming 499 observed updates was rejected.
10. Altered scored candidate descriptor hash was rejected.

The last five checks mocked metadata reads and cached hashes only in the reviewing process; the separate real preparation check above hashed the actual files.

## Existing render evidence and limits

The existing render at `figures/family1_explicit_trial_v1_20261019_v1` records **264 displayed-plane metric comparisons with maximum absolute difference 0.0**, and identical before/after input manifests. Its recorded generator hash matches the reviewed revision. The source implements this parity check before creating the fresh output directory, and rechecks all bound inputs after writing figures. The public CLI resolves the output path, requires a previously nonexistent directory below this campaign's `figures/`, uses exclusive JSON creation, and loads NumPy arrays with `allow_pickle=False`.

This review did not rerender figures, recompute whole-volume MRI metrics, or personally perform visual QA of the rendered images. Read the parent's separate `verification/figure_adapter_checks_v1.json` and `verification/figure_visual_qa_v1.json`: they record nine passing metadata regression checks, an actual existing-output refusal with exit code 1, successful PNG overview/native-resolution crop/historical-overview inspection, and matching hashes for both PNGs and PDFs. PDF pages were not independently rasterized. These are parent-observed checks, separate from the independent checks above.

The adapter's whole-head label intentionally uses the completed scoring objective rather than recomputing whole-head metrics. Its actual current label was independently checked against the six raw comparisons as described above. These figures remain internal review evidence for one adaptively developed case and seed, with unequal comparator budgets; they do not establish superiority or generalization.
