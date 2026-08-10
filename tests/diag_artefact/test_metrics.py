"""Every metric is checked against an input whose answer is known before it is trusted on data.

This is the §14b discipline applied to the tooling itself: the S index ordered runs the way the eye
does while measuring global smoothing rather than staircasing, so "it ranks correctly" is not
evidence that a metric measures the thing it is named after.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from diag_artefact.provenance import match_panel


def test_match_panel_ranks_the_true_source_first():
    rng = np.random.default_rng(0)
    truth = rng.random((64, 64)).astype(np.float32)
    decoy = rng.random((64, 64)).astype(np.float32)
    # the panel is the truth after the render pipeline: normalise then quantise to 8-bit
    panel = np.round(truth / truth.max() * 255).astype(np.float32) / 255.0
    ranked = match_panel(panel, {"truth": truth, "decoy": decoy})
    assert ranked[0][0] == "truth", f"expected truth first, got {ranked}"
    assert ranked[0][1] > 0.99, f"true source correlation too low: {ranked[0][1]:.4f}"
    assert ranked[1][1] < 0.5, f"decoy correlation implausibly high: {ranked[1][1]:.4f}"
    print("PASS test_match_panel_ranks_the_true_source_first")


if __name__ == "__main__":
    test_match_panel_ranks_the_true_source_first()
    print("ALL PASS")
