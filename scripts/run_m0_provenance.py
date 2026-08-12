#!/usr/bin/env python3
"""run_m0_provenance.py — regenerate results_arc/diag_artefact/m0_provenance.json.

M0 identifies which existing results_arc/*_volumes.npz produced results_arc/LOOK_T1_axial.png
(figure dated 2026-08-03 16:42; no *_volumes.npz was written after 2026-08-01 17:05, so the figure
was re-drawn from an EXISTING volume by a script that was never committed to the repo). This
driver was itself, until now, an uncommitted heredoc — the exact failure mode this project has
been bitten by repeatedly (results produced by scripts never written to disk, including this very
figure, and once a metric whose lost definition silently reported the OPPOSITE of a published
result). This script closes that gap for M0: it is committed, so `m0_provenance.json` can be
regenerated on demand rather than trusted as an artefact of an unrepeatable run.

Candidates = every results_arc/*_volumes.npz that existed BEFORE the figure could have been drawn
from it, i.e. every experiment run EXCLUDING this diagnosis's own control runs (ctrlA_*/ctrlB_*,
all created 2026-08-10 or later -- categorically after the 2026-08-03 figure, so they cannot be a
source and are excluded outright rather than merely disadvantaged in the ranking).

Method, entirely reusing diag_artefact.provenance's documented, already-tested API (read-only use;
not modified here) -- this script is orchestration, not new matching logic:
  - the figure has 4 dark-background panels: ground truth / zero-fill / reconstruction /
    |recon-GT| jet[0,0.1] (panel_index 0/1/2/3).
  - "greyscale" reading: calibrate panel0 against every candidate's gtT1 and panel1 against every
    candidate's zfT1 (check_calibration -- raises if the best candidate reads below r=0.95), then
    RANK every candidate's recT1 slice against panel2 (match_panel) -- the actual provenance signal.
  - "error_panel" reading: rank every candidate's rendered |rec-gt| jet[0,0.1] error map against
    the REAL panel3 (match_error_panel), gated by a positive/negative-slice calibration stake
    (check_error_calibration) on the best candidate (claimed slice 160 vs a clearly-wrong slice 60).
  - uniqueness criterion (pre-registered, spec section 4 / plan Task 2): top_correlation > 0.99 AND
    gap (top - 2nd) > 0.01, required independently on BOTH readings for `resolved` to bind.
  - `note_uniqueness_ceiling`: the greyscale GT self-calibration (r_top from check_calibration on
    panel0) sits below 0.99 even for a true match, because 8-bit PNG quantisation + bilinear
    resampling cap the achievable correlation -- recorded, not used to relax the criterion.

Usage:
    PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python \\
        scripts/run_m0_provenance.py \\
        [--png results_arc/LOOK_T1_axial.png] [--glob "results_arc/*_volumes.npz"] \\
        [--json-out results_arc/diag_artefact/m0_provenance.json]
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from diag_artefact.jsonio import write_json
from diag_artefact.provenance import (
    check_calibration,
    check_error_calibration,
    extract_panel,
    match_error_panel,
    match_panel,
)

DEFAULT_PNG = "results_arc/LOOK_T1_axial.png"
DEFAULT_GLOB = "results_arc/*_volumes.npz"
DEFAULT_JSON_OUT = "results_arc/diag_artefact/m0_provenance.json"
SLICE_INDEX = 160
NEGATIVE_SLICE_INDEX = 60
N_PANELS = 4

ERROR_PANEL_DESCRIPTION = (
    "Error |rec-gt| / max(gt_volume) through jet [0,0.1]; fixed range, never autoscaled"
)
ERROR_PANEL_CALIBRATION_DESCRIPTION = (
    "Positive/negative control stake (replaces the fix-round-2 self-comparison tautology, which "
    "called _render_error_panel twice with identical arguments and so was r=1.0 by construction). "
    "positive = best candidate's error map rendered at the claimed true slice vs the REAL extracted "
    "panel 3; negative = same candidate's error map rendered at a clearly different slice (control), "
    "vs the same real panel. Passes iff positive beats negative -- no absolute threshold; the "
    "negative control IS the reference."
)


def _candidates(glob_pat: str = DEFAULT_GLOB, cutoff_mtime: float | None = None) -> list:
    """Every *_volumes.npz matching `glob_pat`, EXCLUDING (a) anything with an mtime AFTER
    `cutoff_mtime` and (b) this diagnosis's own control runs (`ctrlA_*`/`ctrlB_*`) by name.

    Fix round 1 (M3): name-prefix filtering ALONE is fragile — it only protects against files
    that happen to be named `ctrlA_*`/`ctrlB_*`. A concurrently-running, unrelated session has
    since added several dozen `af10_*`/`ar10_*`/`af12_ar_*` volumes to `results_arc/` (none of
    them controls, none of them prefixed `ctrlA_`/`ctrlB_`, all created 2026-08-11+) — every one
    of them postdates `LOOK_T1_axial.png` (2026-08-03) by more than a week and so is categorically
    ineligible regardless of name, and a prefix-only filter would have silently started
    considering them the next time this script runs. `cutoff_mtime` (default: the target PNG's own
    mtime, set by the caller) is the robust, name-independent gate: a file that did not exist yet
    cannot have produced the figure. Name-prefix exclusion is KEPT as a second, explicit, auditable
    gate (task-9-brief.md point 3: this diagnosis's own controls must never be cited, stated
    directly rather than relying solely on their timestamps happening to be late).
    """
    paths = sorted(globmod.glob(glob_pat))
    if cutoff_mtime is not None:
        paths = [p for p in paths if Path(p).stat().st_mtime <= cutoff_mtime]
    return [p for p in paths
            if not (Path(p).name.startswith("ctrlA_") or Path(p).name.startswith("ctrlB_"))]


def _load_key(paths: list, key: str) -> dict:
    """Load array `key` from every path that has it. A handful of older *_volumes.npz predate the
    zf* keys (e.g. b1A_vol_volumes.npz has only rec*/gt*) -- skip those for THIS key rather than
    aborting the whole run; they remain candidates for whichever readings they do have keys for.
    """
    out = {}
    skipped = []
    for p in paths:
        d = np.load(p)
        if key in d.files:
            out[p] = d[key]
        else:
            skipped.append(p)
    if skipped:
        print(f"  ({key} missing, excluded from this reading only: {skipped})")
    return out


def _rank_and_gate(top_corr: float, gap: float) -> bool:
    """The pre-registered uniqueness rule: top_correlation > 0.99 AND gap (top - 2nd) > 0.01."""
    return bool(top_corr > 0.99 and gap > 0.01)


def run_m0(png_path: str = DEFAULT_PNG, glob_pat: str = DEFAULT_GLOB,
          slice_index: int = SLICE_INDEX, negative_slice_index: int = NEGATIVE_SLICE_INDEX,
          n_panels: int = N_PANELS, cutoff_mtime: float | None = None) -> dict:
    if cutoff_mtime is None:
        cutoff_mtime = Path(png_path).stat().st_mtime  # fix round 1, M3: default = the figure's own mtime
    paths = _candidates(glob_pat, cutoff_mtime=cutoff_mtime)
    if len(paths) < 2:
        raise ValueError(f"need >=2 candidates, found {len(paths)} for glob {glob_pat!r}")

    panel0 = extract_panel(png_path, n_panels=n_panels, panel_index=0)
    panel1 = extract_panel(png_path, n_panels=n_panels, panel_index=1)
    panel2 = extract_panel(png_path, n_panels=n_panels, panel_index=2)
    panel3_rgb = extract_panel(png_path, n_panels=n_panels, panel_index=3, keep_rgb=True)

    gt_all = _load_key(paths, "gtT1")
    r_gt_top, r_gt_2nd = check_calibration(panel0, gt_all, slice_index=slice_index)

    zf_all = _load_key(paths, "zfT1")
    r_zf_top, r_zf_2nd = check_calibration(panel1, zf_all, slice_index=slice_index)

    rec_all = _load_key(paths, "recT1")
    ranked_grey = match_panel(panel2, rec_all, slice_index=slice_index)
    top_grey, top2_grey = ranked_grey[0][1], ranked_grey[1][1]
    gap_grey = top_grey - top2_grey
    unique_grey = _rank_and_gate(top_grey, gap_grey)

    cand_data = {p: {"rec": rec_all[p], "gt": gt_all[p]} for p in paths if p in rec_all and p in gt_all}
    ranked_err = match_error_panel(panel3_rgb, cand_data, slice_index=slice_index)
    best_err_path = ranked_err[0][0]
    stake = check_error_calibration(panel3_rgb, cand_data[best_err_path],
                                    slice_index=slice_index, negative_slice_index=negative_slice_index)
    stake = dict(stake)
    stake["description"] = ERROR_PANEL_CALIBRATION_DESCRIPTION
    stake["best_candidate"] = best_err_path
    stake["claimed_slice_index"] = slice_index
    stake["negative_slice_index"] = negative_slice_index

    top_err, top2_err = ranked_err[0][1], ranked_err[1][1]
    gap_err = top_err - top2_err
    unique_err = _rank_and_gate(top_err, gap_err)

    unique_match = bool(unique_grey and unique_err)
    resolved = ranked_grey[0][0] if unique_match else None

    readings = {
        "greyscale": {
            "description": ("Greyscale panels; 8-bit + render resampling limit r to ~0.99, equal "
                           "to mutual brain correlation"),
            "calibration_gt": {"panel_index": 0, "r_top": r_gt_top, "r_2nd": r_gt_2nd},
            "calibration_zf": {"panel_index": 1, "r_top": r_zf_top, "r_2nd": r_zf_2nd},
            "ranked_top5": ranked_grey[:5],
            "top_correlation": top_grey,
            "top_2nd_correlation": top2_grey,
            "gap": gap_grey,
            "unique_match": unique_grey,
        },
        "error_panel": {
            "description": ERROR_PANEL_DESCRIPTION,
            "calibration_stake": stake,
            "ranked_top5": ranked_err[:5],
            "top_correlation": top_err,
            "top_2nd_correlation": top2_err,
            "gap": gap_err,
            "unique_match": unique_err,
        },
    }
    return {
        "readings": readings,
        "unique_match": unique_match,
        "resolved": resolved,
        "note_uniqueness_ceiling": (
            f"Fix round 1 (M2): two DIFFERENT things block `resolved`, not one uniform ceiling "
            f"issue as an earlier version of this note implied. The greyscale reading's "
            f"top_correlation={top_grey:.4f} actually CLEARS the top>0.99 bar; it fails only on "
            f"gap={gap_grey:.4f} (<0.01 -- the top-5 candidates are near-identical tv_eps "
            f"neighbours). The error_panel reading's top_correlation={top_err:.4f} is the one "
            f"that misses the top>0.99 bar, and it sits below this pipeline's own measured "
            f"extraction ceiling: the greyscale GT self-calibration reads r_top={r_gt_top:.4f} "
            f"(< 0.99) even when the reconstruction shown IS the true source, because 8-bit PNG "
            f"quantisation + bilinear resampling cap the achievable correlation. Both are "
            f"documented limitations of the reading (rules registered pre-data are not adjusted "
            f"post-hoc); neither changes the criterion or the unique_match/resolved verdict."
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--png", default=DEFAULT_PNG)
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    ap.add_argument("--slice-index", type=int, default=SLICE_INDEX)
    ap.add_argument("--negative-slice-index", type=int, default=NEGATIVE_SLICE_INDEX)
    ap.add_argument("--cutoff-mtime", type=float, default=None,
                    help="override the candidate-eligibility cutoff (default: the PNG's own mtime)")
    args = ap.parse_args(argv)

    report = run_m0(args.png, args.glob, args.slice_index, args.negative_slice_index,
                    cutoff_mtime=args.cutoff_mtime)

    out_path = Path(args.json_out)
    write_json(out_path, report, generated_by="scripts/run_m0_provenance.py")
    print(f"wrote {out_path}")
    print("greyscale: top=%.4f gap=%.4f unique=%s" % (
        report["readings"]["greyscale"]["top_correlation"],
        report["readings"]["greyscale"]["gap"],
        report["readings"]["greyscale"]["unique_match"]))
    print("error_panel: top=%.4f gap=%.4f unique=%s" % (
        report["readings"]["error_panel"]["top_correlation"],
        report["readings"]["error_panel"]["gap"],
        report["readings"]["error_panel"]["unique_match"]))
    print("unique_match:", report["unique_match"], "resolved:", report["resolved"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
