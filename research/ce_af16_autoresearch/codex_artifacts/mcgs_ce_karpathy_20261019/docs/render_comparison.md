# Render a completed CE AF16 candidate

`render_comparison.py` creates a fresh CPU-only figure directory for one actual completed and scored trial. It does not train or rescore the whole volume. The twelve columns are Reference, the ten frozen complete comparators, and **MC-GS candidate**. The exact scored candidate name appears in the title. The reference is the existing CS-SENSE AF5.33 reconstruction, not a fully sampled ground truth.

```sh
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -u render_comparison.py \
  --run-dir outputs/family1_explicit_trial_v1 \
  --score-dir scores/family1_explicit_trial_v1 \
  --out-dir figures/family1_explicit_trial_v1_20261019_v1 \
  --name family1_explicit_trial_v1 \
  --include-historical-coronal129
```

The output directory must not exist and must be below this campaign's `figures/`. Run and score directories can point to another campaign worktree. `--name` must exactly match `objective.json`. Inputs and completed artifacts remain unchanged. Omit the optional historical flag to produce only the original transverse204, coronal118 and sagittal44 example planes. The optional coronal129 figure is labeled as a historical G2-selected example; it is not selected using the new candidate.

Before rendering, the adapter verifies the candidate array hashes against three independent saved bindings: the scoring objective, the score input manifest, and the completed numerical trial's artifact map. It verifies the run path, the completion-bound launch/metrics records and actual update count, the frozen three-metric contract, and all twenty pre/post comparator-volume hashes against the frozen input manifest and saved whole-head contract. The reference arrays are bound to the frozen report manifest. It verifies all 59 plane identities and comparator scores against the frozen baseline, recomputes strict raw comparisons and their aggregate counts, and checks agreement with the objective summary. On displayed planes, all PSNR, SSIM and NMSE values—including candidate and comparator values—are independently recomputed from the loaded arrays with tolerance1e-10, then the actual saved values are displayed. No HFEN is computed or displayed.

The rendering layout, muted error colormap, typography and panel spacing follow the pinned original `qualitative_g2_three_metrics_20261019/v2/render_source.py` (SHA256 `50ab25b2ed21907671cf1c96c12b644bb456307ff58ab41930e2bcf98d19bd9e`). The two orientation helpers are extracted verbatim from the pinned `original_layout_reference.py`; its main function is never imported or run. Each plane shows pre full slice and lesion zoom, then post full slice and lesion zoom. Zooms retain the original image/error halves and sampling-mask display in the reference column. Bold values indicate the best displayed reconstruction on that metric and plane; this is not a whole-head or generalization claim.

Display settings are fixed across methods:

- Pre magnitude range: `[0, 0.5555555292632828]`; post: `[0, 0.5555555590324888]`. These are the original reference-derived P99.9/1.8 display windows, checked against the loaded references. No output scaling is fitted.
- Original boxes `(y0,y1,x0,x1)`: transverse204 `(81,133,21,73)`, coronal118 `(180,222,24,66)`, sagittal44 `(182,222,106,146)`, historical coronal129 `(184,224,27,67)`.
- Error: `abs(abs(rec)-abs(reference))/max(abs(reference))`, masked by the original reference head mask and displayed in `[0,0.1]`, with `origin='lower'` and the original orientation.
- The PE sampling-mask display must reproduce the frozen 3-D acquired mask broadcast along readout.

The footer uses actual validated counts and the saved whole-head success decision. It does not carry over the old figure's candidate name, old win count, or statement that no training occurred. There are no global-best claims. A failed target remains visibly labeled as unmet.

Each output directory contains PNG/PDF figures, the exact displayed scores, before/after input hashes, and `provenance.json` binding the figure to the objective, completion proof, candidate arrays and generator. The initial status is `rendered_pending_visual_QA`. A separate QA record can document visual inspection without modifying completed scores, training evidence or generated provenance. Creating figures adds zero formal MRI experiments.
