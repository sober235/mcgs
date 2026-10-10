# Task 3 fix round 1 (dispatched 2026-10-10 ~11:00 to the Task 3 implementer; paused at 12:43 before its commit)

This is the exact fix specification the controller sent. A resumed or fresh implementer applies it to
`scripts/sg2x2_core_table.py` and `tests/test_sg2x2_core_table.py` (FIX_BASE 3887190). If the worktree already holds
part of it (uncommitted edits from the paused round), check `git diff` against this text and finish what is missing.

## Findings (verbatim from the task reviewer)

- Important 1 (plan-mandated): "Unequal seed sets are averaged silently, so a 'paired' difference can compare different
  seeds. scripts/sg2x2_core_table.py:44 takes whichever seeds exist per cell. :47 records n_seeds, but nothing uses or
  prints it. :52-55 differences the per-cell means. Probe: A has seeds 42 (35.0) and 43 (36.0), B has only 42 (35.0).
  The output row is | A−B | T1 | PSNR | 1 | +0.500 | ... The seed-matched difference is 0.0. Spec §4 asks for paired
  seeds across groups."
- Important 2 (plan-mandated, test-only): "tests/test_sg2x2_core_table.py:88 asserts N for just two rows. A mutant that
  blanks a subject from all rows of a contrast when any cell of that contrast is missing passes all 4 tests. Fix: in the
  same test assert '| 4 |' on the '| C−D | T1 | PSNR |' row, '| 3 |' on '| (A−B)−(C−D) | T1 | PSNR |', '| 4 |' on
  '| (A−B)−(C−D) | FLAIR | PSNR |', and '+0.200' in t1."

## Controller ruling

Spec §4 ("所有组同病例/采样掩膜使用配对 seed") and §5.3 ("先合并同受试者各 seed 的估计,再分析配对 PSNR 差") are binding:
each difference row of a subject = the mean, over the seeds present in every cell that row needs, of that seed's
difference. The cell-means table keeps per-cell seed means. With complete seed sets every value is unchanged.

## Changes in scripts/sg2x2_core_table.py

1. Move the body of `subject_means` into a helper and keep `subject_means` as a wrapper:

```python
def _means(got: dict, seeds: list[int]) -> dict:
    """(cell, contrast) -> {metric: mean over the seeds present, "n_seeds": k}"""
    out = {}
    for cell in CELLS:
        for c in CONTRASTS:
            rows = [got[(cell, c, s)] for s in seeds if (cell, c, s) in got]
            if rows:
                out[(cell, c)] = {k: _mean([r[k] for r in rows]) for k, _, _ in METRICS}
                out[(cell, c)]["n_seeds"] = len(rows)
    return out


def subject_means(arc: Path, prefix: str, subject: str, seeds: list[int]) -> dict:
    """(cell, contrast) -> {metric: mean over the seeds present, "n_seeds": k}"""
    return _means(collect(Path(arc), f"{prefix}_{subject}", seeds), seeds)
```

2. Directly after `differences()`, add:

```python
ROW_CELLS = {"A−B": ("A", "B"), "C−D": ("C", "D"), "(A−B)−(C−D)": ("A", "B", "C", "D")}


def paired_differences(got: dict, contrast: str, metric: str, seeds: list[int]) -> dict:
    """row -> mean, over the seeds present in every cell the row needs, of that seed's difference (NaN when no seed
    qualifies): seeds are paired across the cells first (spec §4), then averaged within the subject (spec §5.3)."""
    out = {}
    for row, cells in ROW_CELLS.items():
        vals = []
        for s in seeds:
            if all((cell, contrast, s) in got for cell in cells):
                one = {(cell, contrast): got[(cell, contrast, s)] for cell in cells}
                d = differences(one, contrast, metric)[row]
                if not math.isnan(d):
                    vals.append(d)
        out[row] = _mean(vals)
    return out
```

3. In `table()`: replace `means = {s: subject_means(arc, prefix, s, seeds) for s in tests + dev}` by

```python
    gots = {s: collect(Path(arc), f"{prefix}_{s}", seeds) for s in tests + dev}
    means = {s: _means(gots[s], seeds) for s in tests + dev}
```

   replace `vals = [differences(means[s], c, key)[row] for s in tests]` by
   `vals = [paired_differences(gots[s], c, key, seeds)[row] for s in tests]`, and in the per-subject listing replace
   `dd = {c: differences(means[s], c, "head_psnr") for c in CONTRASTS}` by
   `dd = {c: paired_differences(gots[s], c, "head_psnr", seeds) for c in CONTRASTS}`.

4. Wording: in the header string of `table()` change "Per subject the seeds are averaged first;" to "Per subject each
   difference is paired seed by seed, then averaged;"; in the module docstring change "Per subject and contrast the
   seeds are averaged first;" to "Per subject each difference is paired seed by seed over the seeds present in every
   cell it needs, then averaged (spec §4, §5.3); the cell means average each cell's own seeds;".

## Changes in tests/test_sg2x2_core_table.py

a. New test (write it first; on the unfixed code it fails with +0.500):

```python
def test_unequal_seed_sets_are_paired_seed_by_seed(tmp_path):
    s = TESTS[0]
    _subject(tmp_path, s, 42, a=(35.0, 34.0), b=(35.0, 34.0), c=(24.0, 25.0), d=(23.0, 24.0))
    _write(tmp_path, f"p_{s}_s43_A", True, 36.0, 34.0)          # seed 43: A only
    md = table(tmp_path, "p", [42, 43])
    row = next(l for l in md.splitlines() if l.startswith("| A−B | T1 | PSNR |"))
    assert "| 1 |" in row and "+0.000" in row                   # seed 42 only; per-cell means would give +0.500
```

b. At the end of `test_missing_cell_drops_the_subject_from_that_row_only` add:

```python
    cd_t1 = next(l for l in md.splitlines() if l.startswith("| C−D | T1 | PSNR |"))
    ia_t1 = next(l for l in md.splitlines() if l.startswith("| (A−B)−(C−D) | T1 | PSNR |"))
    ia_fl = next(l for l in md.splitlines() if l.startswith("| (A−B)−(C−D) | FLAIR | PSNR |"))
    assert "| 4 |" in cd_t1 and "| 3 |" in ia_t1 and "| 4 |" in ia_fl
    assert "+0.200" in t1
```

## Covering tests and commit

```
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_sg2x2_core_table.py tests/test_sg2x2_table.py -q -p no:cacheprovider
```

Expect 9 passed, no warnings. Commit only the two files with plain
`git commit -m "sg2x2 core test: pair seeds per difference row and pin the row-wise drop in the table tests"` (no
trailer), append a "## Fix round 1" section to `task-3-report.md`, then run the scoped re-review
(`re-review-prompt.md` of the subagent-driven-development skill) on 3887190..<fix commit>.
