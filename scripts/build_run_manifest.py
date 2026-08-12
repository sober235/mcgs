#!/usr/bin/env python3
"""build_run_manifest.py — regenerate results_arc/diag_artefact/run_manifest.json from the runs
themselves.

WHY THIS EXISTS (final whole-branch review, C2a). `run_manifest.json` had no generator anywhere in
the repo: `grep run_manifest` found only the two lines in `diag_artefact/report.py` that READ it.
It is not a passive record either -- `_controls_from_manifest` uses its `status` and `exists`
fields as the gate deciding which runs may be cited at all, including the rule that the vacuous
all-ones `ctrlA_lam010` may never be cited for anything. Those fields were hand-typed, nothing
verified them against reality, and its `measured_from_volumes` numbers disagreed with
`nonvacuity_report.json` in the last digits -- two independent, uncommitted computations of the
same quantity. This is the sixth recurrence in this project of "a result produced by a script that
was never written to disk", and the first one inside the citation gate itself.

WHAT IS DERIVED vs WHAT IS DECLARED. Everything a machine can check is measured here:

    exists                   <- the npz is on disk
    keys                     <- read from the npz
    status                   <- from the non-vacuity gate, never typed:
                                  npz absent                     -> not_run
                                  every contrast passes the gate -> completed_genuine
                                  any contrast fails it          -> completed_vacuous
    measured_from_volumes    <- max|rec-gt| and its ratio to the GT peak come from
                                `check_nonvacuity.check_one`, i.e. from
                                `diag_artefact.controls.assert_control_is_not_vacuous`, the SAME
                                code path that writes nonvacuity_report.json -- one computation of
                                that number in the repository, not two. max|zf-gt| and
                                rec_identical_to_zf are measured here (the gate does not read zf).
    final_metrics_from_log   <- parsed from the run's own log line ("T1    psnr=... ssim=...")

Only prose stays declared: WHY a run was skipped or must not be cited. Those are human decisions,
not measurements, and they are carried in REGISTRY below where they are reviewable.

Usage:
    PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python \\
        scripts/build_run_manifest.py \\
        [--json-out results_arc/diag_artefact/run_manifest.json] [--contrasts T1,FLAIR]
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from diag_artefact.jsonio import write_json
from diag_artefact.volumes import load_volumes

# scripts/ is not a package; load the sibling driver by path so `check_one` -- and through it the
# non-vacuity gate -- is literally the same function object both records are built from.
_spec = importlib.util.spec_from_file_location(
    "check_nonvacuity", REPO_ROOT / "scripts" / "check_nonvacuity.py")
check_nonvacuity = importlib.util.module_from_spec(_spec)
sys.modules["check_nonvacuity"] = check_nonvacuity
_spec.loader.exec_module(check_nonvacuity)

DEFAULT_JSON_OUT = "results_arc/diag_artefact/run_manifest.json"
DEFAULT_CONTRASTS = ("T1", "FLAIR")

DESCRIPTION = (
    "Manifest of the control runs for the T1-artefact diagnosis plan (Task 4). status values: "
    "completed_genuine (scored a real reconstruction, safe to cite), completed_vacuous (ran but "
    "the scoring path discarded the model output; do NOT cite), not_run (deliberately skipped, "
    "reason given). Every status here is DERIVED by scripts/build_run_manifest.py from the "
    "non-vacuity gate against the volumes on disk -- none is hand-typed."
)
TV_EPS_USED = 0.04
TV_EPS_SOURCE = (
    "results_arc/diag_artefact/m0_provenance.json: unique_match=false, resolved=null. Per the "
    "task-4 brief's standing rule (binds every conclusion to tvs4k_e040 when M0 is non-unique), "
    "defaulted to 0.04 (tvs4k_e040's eps)."
)

# name -> everything that is a human decision rather than a measurement.
REGISTRY = {
    "ctrlA_lam010": {
        "npz": "results_arc/ctrlA_lam010_volumes.npz",
        "log": "results_arc/ctrlA_lam010.log",
        "config": "configs/ctrlA_fullsample_edgetv_lam010.yaml",
        "reason": (
            "mask = all-ones (attempted full-sample control). Scoring applies hard data-consistency "
            "k_dc = m*y + (1-m)*k_model (arc/cycle.py); with m all-ones, (1-m)=0, so the trained "
            "model's output is multiplied by zero and never reaches the scored volume. rec is "
            "identical to zf and to gt to float32 precision (measured below), i.e. the run measured "
            "nothing about the reconstruction or the regularizer. MUST NOT be cited as evidence for "
            "or against any patchiness/artefact hypothesis."),
    },
    "ctrlA_lam120": {
        "npz": "results_arc/ctrlA_lam120_volumes.npz",
        "log": "results_arc/ctrlA_lam120.log",
        "config": "configs/ctrlA_fullsample_edgetv_lam120.yaml",
        "reason": (
            "Deliberately skipped. ctrlA_lam010 (lambda_geo=0.10, all-ones mask) was found "
            "completed_vacuous (see that entry): under hard-DC scoring, an all-ones mask makes "
            "(1-m)=0 for every lambda, so a second arm at lambda_geo=1.20 under the same all-ones "
            "mask would be equally vacuous by the same construction -- it would reproduce another "
            "exact copy of the ground truth at the cost of ~35 min of GPU time. Superseded by the "
            "AF=2 repair (ctrlA_af2_lam010 / ctrlA_af2_lam060 below)."),
    },
    "ctrlB_seed1": {
        "npz": "results_arc/ctrlB_seed1_volumes.npz",
        "log": "results_arc/ctrlB_seed1.log",
        "config": "configs/ctrlB_seed1_aligned.yaml",
        "mask": ("ctrlB_indep_af12_seed1_ALIGNED.mat (independent AF=12 draw, seed1, DC-aligned; "
                 "see ctrlB_mask_report.json)"),
        "mask_report": "results_arc/diag_artefact/ctrlB_mask_report.json",
    },
    "ctrlA_af2_lam010": {
        "npz": "results_arc/ctrlA_af2_lam010_volumes.npz",
        "log": "results_arc/ctrlA_af2_lam010.log",
        "config": "configs/ctrlA_af2_lam010.yaml",
        "mask_report": "results_arc/diag_artefact/ctrlA_af2_mask_report.json",
        "note": (
            "Repair of control A after ctrlA_lam010 (all-ones mask) was found vacuous; replaces the "
            "full-sample design with a genuine mild-acceleration AF=2 mask (commit 1dfd1ed), which "
            "is non-vacuous under hard-DC scoring since (1-m) != 0 over a real fraction of k-space. "
            "lambda_geo=0.10 arm."),
    },
    "ctrlA_af2_lam060": {
        "npz": "results_arc/ctrlA_af2_lam060_volumes.npz",
        "log": "results_arc/ctrlA_af2_lam060.log",
        "config": "configs/ctrlA_af2_lam060.yaml",
        "mask_report": "results_arc/diag_artefact/ctrlA_af2_mask_report.json",
        "note": (
            "Repair of control A after ctrlA_lam010 (all-ones mask) was found vacuous; replaces the "
            "full-sample design with a genuine mild-acceleration AF=2 mask (commit 1dfd1ed), which "
            "is non-vacuous under hard-DC scoring since (1-m) != 0 over a real fraction of k-space. "
            "lambda_geo=0.60 arm (measurement-count-scaled second lambda for the AF=2 mask)."),
    },
}

_METRIC_LINE = re.compile(r"^(T1|FLAIR)\s+((?:\w+=[^\s]+\s*)+)$")


def parse_final_metrics(log_path: Path) -> dict | None:
    """The last per-contrast metric line of a run's log, as {contrast: {metric: value}}.

    Returns None when the log is absent or holds no such line (a run that never reached scoring),
    rather than inventing numbers for it.
    """
    if not log_path.exists():
        return None
    out = {}
    for line in log_path.read_text(errors="replace").splitlines():
        m = _METRIC_LINE.match(line.strip())
        if not m:
            continue
        row = {}
        for token in m.group(2).split():
            key, _, value = token.partition("=")
            if key == "beat":
                continue
            try:
                row[key] = float(value)
            except ValueError:
                row[key] = value
        out[m.group(1)] = row
    return out or None


def measure(npz_path: str, contrasts=DEFAULT_CONTRASTS) -> tuple:
    """(gate readings, measured_from_volumes, keys) for one run.

    The gate readings come from `check_nonvacuity.check_one` -- the same function that writes
    nonvacuity_report.json -- so max|rec-gt| is computed once in this repository, not twice.
    """
    gate = check_nonvacuity.check_one(npz_path, contrasts)
    measured, keys = {}, None
    for contrast in contrasts:
        v = load_volumes(npz_path, contrast)
        rec, gt, zf = (v["rec"].astype(np.float64), v["gt"].astype(np.float64),
                       v["zf"].astype(np.float64))
        reading = gate[contrast]
        max_rec_gt = reading.get("max_abs_diff")
        rel = reading.get("rel_max_abs_diff")
        if max_rec_gt is None:  # the gate FAILED, so it recorded the message rather than numbers
            max_rec_gt = float(np.abs(rec - gt).max())
            rel = max_rec_gt / float(np.abs(gt).max())
        measured[contrast] = {
            "max_rec_gt": max_rec_gt,
            "max_rec_gt_rel_gt_peak": rel,
            "max_zf_gt": float(np.abs(zf - gt).max()),
            "rec_identical_to_zf": bool(np.array_equal(v["rec"], v["zf"])),
        }
    with np.load(npz_path) as d:
        keys = list(d.files)
    return gate, measured, keys


def build(contrasts=DEFAULT_CONTRASTS, registry=REGISTRY, repo_root=REPO_ROOT) -> dict:
    runs = {}
    for name, meta in registry.items():
        npz_path = meta["npz"]
        entry = {k: v for k, v in meta.items() if k != "reason"}
        exists = (repo_root / npz_path).exists()
        entry["exists"] = exists
        if not exists:
            entry["keys"] = None
            entry["status"] = "not_run"
            if not (repo_root / meta["log"]).exists():
                entry["log"] = f"{meta['log']} (never created; run never launched)"
        else:
            gate, measured, keys = measure(str(repo_root / npz_path), contrasts)
            statuses = {c: gate[c]["status"] for c in contrasts}
            entry["keys"] = keys
            entry["status"] = ("completed_genuine" if all(s == "PASS" for s in statuses.values())
                               else "completed_vacuous")
            entry["nonvacuity_gate"] = statuses
            entry["final_metrics_from_log"] = parse_final_metrics(repo_root / meta["log"])
            entry["measured_from_volumes"] = measured
        if "reason" in meta:
            entry["reason"] = meta["reason"]
        runs[name] = entry
    return {
        "description": DESCRIPTION,
        "tv_eps_used": TV_EPS_USED,
        "tv_eps_source": TV_EPS_SOURCE,
        "status_derivation": (
            "not_run = the npz is absent; completed_genuine = "
            "diag_artefact.controls.assert_control_is_not_vacuous passes on every contrast; "
            "completed_vacuous = it fails on any contrast. Run "
            "scripts/build_run_manifest.py to re-derive."),
        "contrasts": list(contrasts),
        "runs": runs,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    ap.add_argument("--contrasts", default=",".join(DEFAULT_CONTRASTS))
    args = ap.parse_args(argv)

    manifest = build(tuple(args.contrasts.split(",")))
    for name, entry in manifest["runs"].items():
        print(f"{name}: exists={entry['exists']} status={entry['status']}")
    path = write_json(args.json_out, manifest, generated_by="scripts/build_run_manifest.py")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
