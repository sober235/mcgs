"""diag_artefact.provenance's matching primitives are fully unit-tested in test_metrics.py
(extract_panel / match_panel / check_calibration / match_error_panel / check_error_calibration).
What was never wired into a committed, re-runnable driver is the script that actually calls them
against LOOK_T1_axial.png and every results_arc/*_volumes.npz candidate to write
m0_provenance.json -- that JSON was produced by an uncommitted heredoc (task-9-brief.md point 2).
`scripts/run_m0_provenance.py` closes that gap; these tests cover ITS logic (the control-exclusion
filter, and the end-to-end wiring on synthetic data), the same way test_check_nonvacuity.py covers
scripts/check_nonvacuity.py rather than re-testing the gate it calls.

`scripts/` has no `__init__.py`, so the module is loaded by path -- same reason
test_check_nonvacuity.py does this.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_m0_provenance.py"

spec = importlib.util.spec_from_file_location("run_m0_provenance", SCRIPT_PATH)
run_m0_provenance = importlib.util.module_from_spec(spec)
sys.modules["run_m0_provenance"] = run_m0_provenance
spec.loader.exec_module(run_m0_provenance)


def test_candidates_excludes_control_runs(tmp_path):
    # _candidates only globs paths -- it never opens them -- so empty files are enough.
    names = ["tvs4k_e040_volumes.npz", "rc_lam000_volumes.npz",
             "ctrlA_lam010_volumes.npz", "ctrlA_af2_lam010_volumes.npz",
             "ctrlB_seed1_volumes.npz"]
    for n in names:
        (tmp_path / n).write_bytes(b"")
    got = run_m0_provenance._candidates(str(tmp_path / "*_volumes.npz"))
    got_names = {Path(p).name for p in got}
    assert got_names == {"tvs4k_e040_volumes.npz", "rc_lam000_volumes.npz"}, got_names
    print("PASS test_candidates_excludes_control_runs")


def test_candidates_excludes_anything_postdating_the_cutoff_regardless_of_name(tmp_path):
    """Fix round 1 (M3): a concurrently-running, unrelated session added dozens of `af10_*`/
    `ar10_*`/`af12_ar_*` volumes to results_arc/ after LOOK_T1_axial.png was drawn -- none of them
    named ctrlA_*/ctrlB_*, so the OLD prefix-only filter would have silently started treating them
    as M0 candidates. `cutoff_mtime` must exclude them by DATE, independent of name.
    """
    import os
    import time

    old = tmp_path / "old_experiment_volumes.npz"
    old.write_bytes(b"")
    cutoff = old.stat().st_mtime + 0.05
    time.sleep(0.1)
    new = tmp_path / "new_unrelated_experiment_volumes.npz"  # does NOT match ctrlA_/ctrlB_
    new.write_bytes(b"")
    assert new.stat().st_mtime > cutoff, "fixture sanity: new must postdate the cutoff"

    got = run_m0_provenance._candidates(str(tmp_path / "*_volumes.npz"), cutoff_mtime=cutoff)
    got_names = {Path(p).name for p in got}
    assert got_names == {"old_experiment_volumes.npz"}, got_names

    # with no cutoff (the old default behaviour), the postdating file WOULD be included --
    # proving the cutoff is doing real work, not merely restating the prefix filter.
    got_no_cutoff = run_m0_provenance._candidates(str(tmp_path / "*_volumes.npz"))
    assert "new_unrelated_experiment_volumes.npz" in {Path(p).name for p in got_no_cutoff}
    print("PASS test_candidates_excludes_anything_postdating_the_cutoff_regardless_of_name")


def _mri_like(rng, shape=(48, 48)):
    """Bright block. test_metrics.py's `_mri_like_panel` uses a narrow row-span (30-70%) and a wide
    column-span (10-90%) so that, unrotated, every column inside the block has the same darkness
    fraction. This test renders `np.rot90(...)` the way a real MRI figure does, and rot90 swaps
    which axis is "row" and which is "column" -- so the spans are swapped here (wide on axis 0,
    narrow on axis 1) to land in the same place post-rotation: verified empirically that this gives
    a single contiguous dark column-run per panel after rot90, not the two-sided split a naive
    symmetric (or un-swapped) fill produces.
    """
    arr = np.zeros(shape, dtype=np.float32)
    r0, r1 = int(shape[0] * 0.10), int(shape[0] * 0.90)
    c0, c1 = int(shape[1] * 0.30), int(shape[1] * 0.70)
    arr[r0:r1, c0:c1] = 0.5 + rng.random((r1 - r0, c1 - c0)).astype(np.float32) * 0.4
    return np.clip(arr, 0.0, 1.0)


def test_rank_and_gate_applies_the_pre_registered_uniqueness_rule():
    # top > 0.99 AND gap > 0.01, both required -- exercised directly since the low-resolution
    # rendering pipeline can't reliably clear 0.99 in an end-to-end test (see the next test's
    # docstring: even a real, self-identical match is capped below 0.99 by PNG quantisation).
    assert run_m0_provenance._rank_and_gate(0.995, 0.02) is True
    assert run_m0_provenance._rank_and_gate(0.98, 0.02) is False, "must fail on top alone"
    assert run_m0_provenance._rank_and_gate(0.995, 0.005) is False, "must fail on gap alone"
    assert run_m0_provenance._rank_and_gate(0.99, 0.01) is False, "boundary values are exclusive"
    print("PASS test_rank_and_gate_applies_the_pre_registered_uniqueness_rule")


def test_run_m0_ranks_the_true_source_first_and_never_considers_control_runs(tmp_path):
    """A synthetic figure rendered from a known 'true' candidate must rank that candidate on top
    of both readings; a control-run npz that is made to match EXACTLY must still never be picked,
    because it is excluded from candidacy before ranking even starts.

    This does NOT assert `resolved` is set: even an EXACT self-match on this small a synthetic
    panel tops out around r=0.978 (8-bit PNG quantisation + a handful of distinct grey levels cap
    it, the same mechanism `note_uniqueness_ceiling` documents on the real 256x220 figure, just
    more severe at this resolution) -- below the pre-registered top>0.99 bar, so `resolved` is
    correctly left null here. `_rank_and_gate` above covers the resolution decision directly.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from diag_artefact.provenance import _render_error_panel

    shape2d = (48, 48)
    vol_shape = (48, 48, 8)
    slice_index = 4
    rng = np.random.default_rng(11)

    gt2d = _mri_like(rng, shape2d)
    zf2d = gt2d + rng.normal(0, 0.02, shape2d).astype(np.float32)
    true_rec2d = gt2d.copy()  # exact match -- the figure IS drawn from this candidate
    decoy_rec2d = gt2d + rng.normal(0, 0.2, shape2d).astype(np.float32)  # clearly worse match

    def _vol(slice2d):
        v = np.zeros(vol_shape, dtype=np.float32)
        v[:, :, slice_index] = slice2d
        return v

    gt_vol, zf_vol = _vol(gt2d), _vol(zf2d)
    true_rec_vol, decoy_rec_vol = _vol(true_rec2d), _vol(decoy_rec2d)

    gt_peak = float(np.abs(gt_vol).max())
    err_rgb = _render_error_panel(true_rec2d, gt2d, gt_volume_max=gt_peak)

    # Candidates must exist BEFORE the figure (fix round 1, M3: run_m0 now excludes anything
    # postdating the PNG's own mtime by default) -- write them first, matching real chronology.
    np.savez(tmp_path / "true_volumes.npz", recT1=true_rec_vol, gtT1=gt_vol, zfT1=zf_vol,
             recFL=true_rec_vol, gtFL=gt_vol, zfFL=zf_vol)
    np.savez(tmp_path / "decoy_volumes.npz", recT1=decoy_rec_vol, gtT1=gt_vol, zfT1=zf_vol,
             recFL=decoy_rec_vol, gtFL=gt_vol, zfFL=zf_vol)
    # An exact-match "control" -- must be excluded from candidacy no matter how well it would score.
    np.savez(tmp_path / "ctrlA_lam010_volumes.npz", recT1=gt_vol, gtT1=gt_vol, zfT1=gt_vol,
             recFL=gt_vol, gtFL=gt_vol, zfFL=gt_vol)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(np.rot90(gt2d), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
    axes[0].set_title("ground truth")
    axes[1].imshow(np.rot90(zf2d), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
    axes[1].set_title("zero-fill")
    axes[2].imshow(np.rot90(true_rec2d), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
    axes[2].set_title("reconstruction")
    axes[3].imshow(err_rgb, interpolation="nearest", aspect="auto")
    axes[3].set_title("|recon - GT|  jet [0, 0.1]")
    for a in axes:
        a.axis("off")
    png_path = tmp_path / "LOOK_T1_axial.png"
    fig.savefig(png_path, dpi=100)
    plt.close(fig)

    report = run_m0_provenance.run_m0(
        png_path=str(png_path), glob_pat=str(tmp_path / "*_volumes.npz"),
        slice_index=slice_index, negative_slice_index=1,
    )

    grey = report["readings"]["greyscale"]
    err = report["readings"]["error_panel"]
    assert grey["ranked_top5"][0][0] == str(tmp_path / "true_volumes.npz"), grey["ranked_top5"]
    assert err["ranked_top5"][0][0] == str(tmp_path / "true_volumes.npz"), err["ranked_top5"]
    # the control run must never appear as the top candidate of either reading, despite being a
    # bit-identical match to gt -- it must have been excluded before ranking even started.
    grey_labels = [label for label, _ in grey["ranked_top5"]]
    err_labels = [label for label, _ in err["ranked_top5"]]
    assert not any("ctrlA_" in label for label in grey_labels), grey_labels
    assert not any("ctrlA_" in label for label in err_labels), err_labels
    print("PASS test_run_m0_ranks_the_true_source_first_and_never_considers_control_runs")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_candidates_excludes_control_runs(tmp)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_candidates_excludes_anything_postdating_the_cutoff_regardless_of_name(tmp)
    test_rank_and_gate_applies_the_pre_registered_uniqueness_rule()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_run_m0_ranks_the_true_source_first_and_never_considers_control_runs(tmp)
    print("ALL PASS")
