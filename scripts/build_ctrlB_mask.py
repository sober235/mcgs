#!/usr/bin/env python3
"""build_ctrlB_mask.py — reproducible end-to-end driver for control B's mask.

Control B (spec docs/superpowers/specs/2026-08-07-t1-artefact-diagnosis-design.md §5) needs an
AF=12 vd-Poisson mask that is a genuinely DIFFERENT instance from the shipped
`mask3d_vdpois_af12_transverse_ALIGNED.mat` — same generator, same parameters, only the RNG draw
changed — rolled onto the data's true k-space DC exactly like the shipped mask was (§8's
registration bug: an unrolled fresh mask captures ~27% of central k-space energy instead of the
rolled mask's ~74-90%).

CRITICAL FINDING (discovered building this driver, corrects an earlier scouting note that claimed
determinism): `rng(seed)` in MATLAB has NO observed effect on `vdPoisMex`'s output within a single
MATLAB process — five repeated `rng(1)` + `vdPoisMex(...)` calls in one `matlab -batch` session
returned the identical draw every time. The actual entropy comes from something process-level
inside the compiled mex (plausibly a time- or address-based seed not exposed to MATLAB), not from
`rng()`. Separate process launches DO draw differently: across 14 launches during the exploration
that produced this driver, `net_AF` ranged roughly 11.70-11.83. `--seed` is accepted and passed to
`rng()` for provenance/documentation only — do not expect it to reproduce a specific draw, and do
not "seed search" inside one process expecting different results; you must relaunch MATLAB itself.

SELECTION DISCIPLINE — the whole point of this script's structure, not an implementation detail:
the retry loop below decides accept/reject using `net_AF` ONLY. Matching net_AF to the shipped
mask's sampling rate is exactly spec §5's own gate, and matching a RATE does not touch the sampled
ARRANGEMENT — which is the one thing control B exists to vary. The other two checks
(`mask@DC == 1`, and this task's added `assert_differs_from_shipped`) are evaluated exactly ONCE,
after the loop, on whichever draw first clears net_AF — as hard assertions that ABORT the whole
run. They are never read inside the loop and can never trigger a redraw. This is deliberate: a
loop that redrew because the arrangement "didn't look different enough from the shipped mask" (or
because DC happened to land favourably) would be cherry-picking a flattering control B, and the
point is to make that structurally impossible, not merely avoided by convention.

This script does NOT touch the mask already shipped in the repo
(`.../mask/ctrlB_indep_af12_seed1_ALIGNED.mat`) by default — it refuses to overwrite an existing
`--out` or `--json-out` unless `--overwrite` is passed. Re-running this script is expected to draw
a DIFFERENT (but equally valid, equally gate-passing) mask instance than the one already shipped,
because of the process-level randomness above — that is fine for reproducibility/audit purposes,
but it must never silently replace the artifact Task 4+ already bound to.

Usage:
    PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python \\
        scripts/build_ctrlB_mask.py [--out PATH] [--json-out PATH] [--overwrite] \\
        [--max-attempts N] [--seed N]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diag_artefact.controls import (
    align_like_shipped,
    assert_control_mask,
    assert_differs_from_shipped,
    derive_roll,
)

MATLAB = "/usr/local/bin/matlab"
ESPIRIT_UTILS = "/data0/congcong/code/ESPIRiT/utils"
MDIR = "/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/mask"
UNALIGNED_MASK = f"{MDIR}/mask3d_vdpois_af12_transverse.mat"
ALIGNED_MASK = f"{MDIR}/mask3d_vdpois_af12_transverse_ALIGNED.mat"
DATA_PATH = ("/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/"
             "Train_3dBrain_huangyuwei_20230705-160927-1451_161015_T1_trainer.mat")
DEFAULT_OUT = f"{MDIR}/ctrlB_indep_af12_seed1_ALIGNED.mat"
DEFAULT_JSON = "results_arc/diag_artefact/ctrlB_mask_report.json"
SHIPPED_NET_AF = 11.832
AF_ABS_TOL = 0.05          # spec sec5, absolute; keep in sync with diag_artefact.controls.AF_ABS_TOL
MAX_ATTEMPTS_DEFAULT = 20  # generous: observed pass rate ~2/8-3/14 draws during exploration


def draw_plane(tmp_mat: Path, seed: int) -> tuple[np.ndarray, float]:
    """Launch ONE fresh MATLAB process and draw one AF12 vd-Poisson PE-plane.

    A fresh process is required for a new draw (see module docstring: rng() does not affect
    vdPoisMex's output within a process). `seed` is passed through to rng() for provenance only.
    """
    script = (
        f"addpath('{ESPIRIT_UTILS}');\n"
        f"rng({seed});\n"
        "mask = single(vdPoisMex(220,256,220,256,3,4,24,0,1));\n"
        "net = numel(mask)/nnz(mask);\n"
        f"save('{tmp_mat}', 'mask', 'net', '-v7');\n"
        "fprintf('DRAW nnz=%d net_AF=%.6f\\n', nnz(mask), net);\n"
    )
    script_path = tmp_mat.with_suffix(".m")
    script_path.write_text(script)
    proc = subprocess.run(
        [MATLAB, "-batch", f"run('{script_path}')"],
        capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0 or not tmp_mat.exists():
        raise RuntimeError(
            f"MATLAB draw failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    mask2d = sio.loadmat(str(tmp_mat))["mask"].astype(np.float32)
    net_af = float(mask2d.size / mask2d.sum())
    return mask2d, net_af


def find_data_dc() -> tuple:
    """Locate the data's true k-space DC, mirroring dataset.py's _load_h5_kspace /
    _normalize_kspace_layout. This .mat is MATLAB v7.3 (HDF5) -- scipy.io.loadmat cannot read it,
    unlike the mask files below, which are plain v5/v7 and load fine via scipy.io."""
    with h5py.File(DATA_PATH, "r") as f:
        raw = f["ksp"][:]
    if raw.dtype.names is not None and "real" in raw.dtype.names:
        ksp = raw["real"].astype(np.float64) + 1j * raw["imag"].astype(np.float64)
    else:
        ksp = raw.astype(np.complex128)
    energy = (np.abs(ksp) ** 2).sum(axis=-1)  # coil axis is last for this file
    return np.unravel_index(int(np.argmax(energy)), energy.shape)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT, help="3-D mask .mat output path")
    ap.add_argument("--json-out", default=DEFAULT_JSON, help="gate/attempt-log report path")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing --out/--json-out")
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_DEFAULT)
    ap.add_argument("--seed", type=int, default=1,
                     help="passed to MATLAB rng(); has NO observed effect on the draw (see module docstring)")
    args = ap.parse_args()

    out_path = Path(args.out)
    json_path = Path(args.json_out)
    if not args.overwrite:
        existing = [p for p in (out_path, json_path) if p.exists()]
        if existing:
            print(f"REFUSING to overwrite existing output(s) without --overwrite: "
                  f"{[str(p) for p in existing]}", file=sys.stderr)
            print("The shipped control-B mask already passed its gates and Task 4+ bind to it "
                  "by path; pass --overwrite only if you mean to replace it.", file=sys.stderr)
            sys.exit(1)

    unal = sio.loadmat(UNALIGNED_MASK)["mask"].astype(np.float32)
    alig = sio.loadmat(ALIGNED_MASK)["mask"].astype(np.float32)
    roll = derive_roll(unal, alig)
    print(f"derived roll: {roll}  (memory sec8 records (2,24,-6); the axis-2 component is a "
          f"provable no-op here -- both shipped masks are exactly uniform along axis 2, the "
          f"readout axis, so any axis-2 shift changes nothing)")

    dc = find_data_dc()
    print(f"data k-space DC at {dc}  (matrix center "
          f"{tuple(s // 2 for s in (256, 220, 256))})")

    attempts = []
    accepted_2d = None
    with tempfile.TemporaryDirectory() as td:
        for i in range(1, args.max_attempts + 1):
            tmp_mat = Path(td) / f"draw_{i}.mat"
            mask2d, net_af = draw_plane(tmp_mat, seed=args.seed)
            abs_delta = abs(net_af - SHIPPED_NET_AF)
            nnz = int(mask2d.sum())
            # Selection on net_AF ONLY -- see module docstring "SELECTION DISCIPLINE". Do not add
            # any other condition to this `accept` expression.
            accept = abs_delta < AF_ABS_TOL
            attempts.append({"attempt": i, "nnz": nnz, "net_af": net_af,
                              "abs_delta_af": abs_delta, "accepted": accept})
            print(f"attempt {i}: nnz={nnz} net_AF={net_af:.6f} abs_delta={abs_delta:.6f} "
                  f"{'ACCEPTED' if accept else 'rejected'}")
            if accept:
                accepted_2d = mask2d
                break

    if accepted_2d is None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps({"attempts": attempts, "status": "FAILED_NO_DRAW_WITHIN_TOLERANCE"}, indent=2))
        raise RuntimeError(
            f"No draw landed within |net_AF-{SHIPPED_NET_AF}|<{AF_ABS_TOL} in {args.max_attempts} "
            f"attempts. Full attempt log written to {json_path}. Refusing to fall through with a "
            f"rejected draw -- rerun with a larger --max-attempts if this recurs."
        )

    m3 = align_like_shipped(accepted_2d, (256, 220, 256), roll)

    # HARD ASSERTIONS -- not retry conditions (see module docstring). Either of these raising
    # aborts the whole run right here; there is no `except` around them and no way back into the
    # draw loop above.
    gate_report = assert_control_mask(m3, dc_index=(dc[0], dc[1]), shipped_net_af=SHIPPED_NET_AF)
    diff_report = assert_differs_from_shipped(m3, alig)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(out_path), {"mask": m3, "net_af": np.float32(gate_report["net_af"])})

    report = {
        **gate_report, **diff_report,
        "roll": [int(x) for x in roll],
        "dc": [int(x) for x in dc],
        "attempts": attempts,
        "n_attempts": len(attempts),
        "out": str(out_path),
        "status": "OK",
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")
    print(f"wrote {json_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
