# CRISP Phase 0 — carried-forward follow-ups

Adjudicated non-blocking at the end of Phase 0 (2026-07-29). None of these affect the
correctness of the shipped numbers in `README.md`; each is a trap for whoever touches
these artifacts next. Ordered by when they bite.

## 1. Before regenerating any results JSON

`results_crisp/*.json` carry fields that were **hand-patched**, with no code path that
writes them. Re-running the generating script silently drops them:

| file | hand-patched fields | generator that must be wired |
|---|---|---|
| `g2_voxel_huangyuwei.json` | `verdict`, `caveats`, `csm_source`, `cross_experiment_csm_warning` | `crisp/experiments/g2_voxel.py::finalize` |
| `g0_ceiling_huangyuwei_{T1,FLAIR}.json` | `achieved_npts`, `regularizer_deviation_caveat`, `cross_experiment_csm_warning` | `crisp/experiments/g0_ceiling.py` |
| `ce_check.json` | `verdict_status` | `crisp/experiments/ce_check.py::main` |

The plan required a `verdict` field on the G2 artifact; it exists only as a hand-patch.
Wire these into the generators before any regeneration, or the next run ships artifacts
that state conclusions without their caveats.

## 2. Before the next λ sweep

`crisp/experiments/g2_voxel.py:292` — the single-point exemption in `finalize()`'s
λ-grid check keys off `len(recovered_keys)` (what was found on disk) rather than
`len(expected_lam_keys)` (what was intended). A catastrophic loss leaving 1 shard of an
intended 10-point sweep skips the check entirely and silently reports a "best λ" drawn
from 10% of the grid.

Cannot affect the shipped result — all 10 shards were verified present for all four
(condition, arm) combinations, and `interior_optimum` is hardcoded `False` at a single
point. Fix by keying the exemption off the intended grid.

Related, minor: `gate()` now runs before the shard is written, so a cell that trips the
PSNR abort no longer leaves its shard on disk for post-mortem inspection.

## 3. Before comparing G2 and G0 numbers in the same sentence

They use **different CSMs for the same subject**: G2 uses the trainer `.mat`'s `map`
field, G0 uses the ESPIRiT maps, and the two differ by **111% relative Frobenius**. Their
errors are therefore computed under different forward operators and are not on a common
scale. Computing a "gap to the ceiling" from G2's `E_unobs ≈ 0.65` and G0's
`E_coil_full ≈ 0.28` is invalid.

To make them comparable, re-run one arm under the other's CSM. Until then, cite each
result only within its own experiment.

## 4. Lower priority, but real

- `crisp/metrics/coil_domain.py` — `e_unobs` has no guard for an all-ones sampling mask
  (zero denominator → silent NaN). That is exactly the G0 configuration; nothing calls it
  there today, but Phase 1 might. Make it an explicit error.
- `crisp/metrics/coil_domain.py` — nothing pins the relation `ifft3c == (1/N)·Fᴴ`, on
  which the entire step-size argument in `cs_sense.py` rests. Switching to `norm="ortho"`
  would rescale `w_data` by N with no test naming it. A three-line adjoint/round-trip test
  closes it.
- `crisp/masks/vdpoisson.py` — `acs > min(shape)` silently yields a *full* mask via
  negative-start slicing instead of raising. One `if`.
- `configs/g0_mask_full_ones.mat` is 55 MB, gitignored, and has no generator script, so
  G0 is not reproducible from a clean clone. A three-line generator under `scripts/`
  matches the repo's existing convention.
- The Global Constraint mandating a GT / zero-fill / recon / error panel is written
  unconditionally, but under full sampling the zero-fill column equals GT by construction
  (visible in `g0_ceiling_huangyuwei_*_panel.png`). Narrow the constraint to undersampled
  experiments.

## 5. Blocked on the human, not on code

- **CE pair (`ce_check.json`)** — parked INCONCLUSIVE. Resolving it needs the external
  DICOM `SeriesDescription` / `ContrastBolusAgent` for StudyInstanceUID
  `1.2.392.200036.9125.2.138612190166.20230707002323`, protocols `…0010` (12:43:10) and
  `…0060` (12:50:11). Local metadata is exhausted — `scanparam.xml`, `.prot` and every
  `.raw` header carry an empty `SeriesDescription`. This decides whether the clinical arm
  exists.
- **Multi-subject work** — blocked pending confirmation that the 16 paired T1+FLAIR
  subjects' original ethics approval covers this secondary analysis and publication of
  aggregate results. Nothing touching those subjects should start before that.
