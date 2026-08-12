#!/usr/bin/env python3
"""check_nonvacuity.py — run the non-vacuity gate over one or more control-run volumes and write
a committed JSON record of the readings.

Companion to `scripts/build_ctrl_mask.py`'s pattern: that script's mask gate
(`assert_target_af_mask`) is wired into a committed driver whose result lands in
`results_arc/diag_artefact/ctrlA_af2_mask_report.json`. `diag_artefact.controls.
assert_control_is_not_vacuous` (added alongside it, same commit) was NOT similarly wired — it was
only ever exercised against synthetic arrays in unit tests, so no run of it against the real
`*_volumes.npz` files this task actually produced left a trace in the repository. This script
closes that gap: it is the thing that must run, and be re-run, every time a control's volumes are
scored, so a future vacuous control (the exact failure this whole task exists to catch) cannot
again go unnoticed by process.

Loads each contrast via `diag_artefact.volumes.load_volumes` (read-only use of that module; not
modified here) and calls `diag_artefact.controls.assert_control_is_not_vacuous` on `(rec, gt)`.
Unlike the mask gate, a FAILING run here is not a bug to abort on -- it is exactly the reading the
retired all-ones control is supposed to produce, so failures are CAUGHT and recorded in the JSON
report rather than raised. The gate still has teeth: `main()` exits non-zero if ANY (npz, contrast)
entry failed, so a caller (human or otherwise) that only checks the exit code still catches it.

Usage:
    PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python \\
        scripts/check_nonvacuity.py results_arc/ctrlA_af2_lam010_volumes.npz \\
        results_arc/ctrlA_af2_lam060_volumes.npz results_arc/ctrlB_seed1_volumes.npz \\
        results_arc/ctrlA_lam010_volumes.npz \\
        [--json-out results_arc/diag_artefact/nonvacuity_report.json] [--contrasts T1,FLAIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diag_artefact.controls import assert_control_is_not_vacuous
from diag_artefact.jsonio import write_json
from diag_artefact.volumes import load_volumes

DEFAULT_CONTRASTS = ("T1", "FLAIR")
DEFAULT_JSON_OUT = "results_arc/diag_artefact/nonvacuity_report.json"


def check_one(npz_path: str, contrasts=DEFAULT_CONTRASTS) -> dict:
    """Run the non-vacuity gate on every requested contrast of one *_volumes.npz. Never raises --
    an AssertionError from the gate is caught and recorded as a FAIL entry, not propagated, so one
    vacuous input in a batch does not stop the rest from being checked and recorded."""
    out = {}
    for contrast in contrasts:
        v = load_volumes(npz_path, contrast)
        try:
            rep = assert_control_is_not_vacuous(v["rec"], v["gt"])
            out[contrast] = {"status": "PASS", **rep}
        except AssertionError as e:
            out[contrast] = {"status": "FAIL", "error": str(e)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="+", help="one or more *_volumes.npz control-run files")
    ap.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    ap.add_argument("--contrasts", default=",".join(DEFAULT_CONTRASTS),
                     help="comma-separated contrast list (default: T1,FLAIR)")
    args = ap.parse_args(argv)
    contrasts = tuple(args.contrasts.split(","))

    report = {}
    any_fail = False
    for npz_path in args.npz:
        exp = Path(npz_path).name
        if exp.endswith("_volumes.npz"):
            exp = exp[: -len("_volumes.npz")]
        entry = check_one(npz_path, contrasts)
        report[exp] = {"npz": npz_path, "contrasts": entry}
        for contrast, rep in entry.items():
            if rep["status"] == "FAIL":
                any_fail = True
                print(f"{exp} {contrast}: FAIL ({rep['error']})")
            else:
                print(f"{exp} {contrast}: PASS max_abs_diff={rep['max_abs_diff']:.3e} "
                      f"rel_max_abs_diff={rep['rel_max_abs_diff']:.3e}")

    # MERGE, don't regenerate (final review, minors): this script is the committed record of which
    # control runs are vacuous, and it is normally invoked on whichever npz files just finished. A
    # wholesale rewrite means a partial invocation silently NARROWS the committed record -- the
    # runs it did not name simply vanish from the file, with nothing to indicate they were ever
    # checked. Existing entries for runs named in THIS invocation are replaced (that is a re-check
    # and the new reading wins); entries for runs not named are carried through untouched.
    out_path = Path(args.json_out)
    merged, carried = dict(report), []
    if out_path.exists():
        try:
            previous = json.loads(out_path.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{out_path} exists but is not valid JSON ({exc}); refusing to "
                             f"overwrite a record that cannot be merged") from exc
        for name, entry in (previous.get("runs") or {}).items():
            if name not in merged:
                merged[name] = entry
                carried.append(name)
    any_fail_merged = any(c.get("status") == "FAIL"
                          for run in merged.values()
                          for c in (run.get("contrasts") or {}).values())
    if carried:
        print(f"carried forward from the existing report (not re-checked): {sorted(carried)}")

    write_json(out_path, {"contrasts": list(contrasts), "runs": merged,
                          "checked_this_invocation": sorted(report),
                          "carried_forward_unchecked": sorted(carried),
                          "any_fail": any_fail_merged},
               generated_by="scripts/check_nonvacuity.py")
    print(f"wrote {out_path}")
    print("RESULT:", "FAIL (at least one run is vacuous)" if any_fail else "PASS (all runs non-vacuous)")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
